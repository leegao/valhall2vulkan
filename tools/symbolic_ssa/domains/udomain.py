#!/usr/bin/env python3
"""
Unified Lattice-Typed Value Domains (`ConcreteUDomain` and `SymbolicZ3UDomain`).

Every SSA value is represented as a `UVal` tagged with a `UValKind`:
  - `UNKNOWN`: bottom / uninitialized value; promotes on first assignment/use.
  - `BVEC64`:  64-bit bitvector container covering bool/i16/i32/i64/u16/u32/u64,
               delegating to `ConcreteBitvecDomain` or `SymbolicZ3BitvecDomain`.
  - `F16X4`:   4-lane float16 vector (4 Python floats or 4 Z3 Reals).
  - `F32X2`:   2-lane float32 vector (2 Python floats or 2 Z3 Reals; scalar f32 in lane 0).
  - `F64`:     scalar float64 (Python float or Z3 Real).
  - `PTR`:     pointer-like value (`PtrVal`: base tag, 64-bit byte offset,
               alignment mask, and `readonly` flag for uniform/UBO/push-constant
               pointers treated as read-only registers).
  - `SYM`:     non-pointer opaque symbolic value (handles, tokens, samplers).

Key invariants:
  1. Lazy / deferred bitcast: `cast_value("Bitcast", val, target_type)` does NOT
     eagerly bitcast; it returns `val` untouched, deferring conversion until a
     use site actually demands a different `UValKind`. Thus Valhall SSA's
     ubiquitous `Bitcast(F32->U32) -> Bitcast(U32->F32)` pairs are zero-cost.
  2. Direct typed execution: when operand kind matches operation kind (e.g.
     `FAdd`/`FMul`/`Fma` on `F32X2`), execution stays in Z3 `Real` arithmetic
     (avoiding IEEE-754 bit-blasting) or native Python float.
  3. On-demand bitcast on mismatch: when an `F32X2`/`F16X4`/`F64` value is
     consumed by a bitwise/integer op (e.g. `BitXor(f, 0x80000000)`), it is
     promoted to `BVEC64` at the use site, preserving exact bit-hacking semantics.
"""

from __future__ import annotations

import enum
import math
import struct
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from tools.symbolic_ssa.domains.base import MASK32, MASK64, SsaDomain
from tools.symbolic_ssa.domains.z3_base import Z3SymbolicDomain
from tools.symbolic_ssa.domains.concrete_bitvec import ConcreteBitvecDomain
from tools.symbolic_ssa.domains.symbolic_z3_bitvec import (
    SymbolicZ3BitvecDomain,
    _real_cos,
    _real_exp2,
    _real_log2,
    _real_sin,
    _real_sqrt,
)

try:
    import z3
except ImportError:
    z3 = None


class UValKind(enum.Enum):
    UNKNOWN = "Unknown"
    BOOL = "Bool"
    BVEC64 = "BVec64"
    F16X4 = "F16x4"
    F32X2 = "F32x2"
    F64 = "F64"
    PTR = "Ptr"
    SYM = "Sym"


@dataclass(frozen=True, slots=True)
class PtrVal:
    """
    Structured pointer value supporting arithmetic, alignment masking, and
    read-only uniform/UBO/push-constant register modeling.
    """
    base: str
    offset: Any  # concrete int (64-bit) or z3.BitVecRef(64)
    align_mask: Optional[int] = None
    readonly: bool = False


@dataclass(frozen=True, slots=True)
class SymVal:
    """Non-pointer opaque symbolic value (e.g. texture handle, sampler, token)."""
    tag: str
    payload: Any = None


@dataclass(frozen=True, slots=True)
class UVal:
    """
    Tagged value in the unified lattice domain.

    `raw_bv` optionally caches the original 64-bit bitvector when a value was
    promoted from `BVEC64` to an FP kind, enabling exact lossless round-trips
    if the value is later re-interpreted as bits.
    """
    kind: UValKind
    data: Any = None
    raw_bv: Any = None

    @staticmethod
    def unknown() -> "UVal":
        return UVal(UValKind.UNKNOWN, None)

    @staticmethod
    def bool_val(b: Any) -> "UVal":
        return UVal(UValKind.BOOL, b)

    @staticmethod
    def bvec64(bv: Any) -> "UVal":
        return UVal(UValKind.BVEC64, bv, raw_bv=bv)

    @staticmethod
    def f16x4(lanes: Tuple[Any, Any, Any, Any], raw_bv: Any = None) -> "UVal":
        return UVal(UValKind.F16X4, lanes, raw_bv=raw_bv)

    @staticmethod
    def f32x2(lo: Any, hi: Any, raw_bv: Any = None) -> "UVal":
        return UVal(UValKind.F32X2, (lo, hi), raw_bv=raw_bv)

    @staticmethod
    def f64(val: Any, raw_bv: Any = None) -> "UVal":
        return UVal(UValKind.F64, val, raw_bv=raw_bv)

    @staticmethod
    def ptr(base: str, offset: Any, *, align_mask: Optional[int] = None, readonly: bool = False) -> "UVal":
        return UVal(UValKind.PTR, PtrVal(base=base, offset=offset, align_mask=align_mask, readonly=readonly))

    @staticmethod
    def sym(tag: str, payload: Any = None) -> "UVal":
        return UVal(UValKind.SYM, SymVal(tag=tag, payload=payload))

    @property
    def is_unknown(self) -> bool:
        return self.kind == UValKind.UNKNOWN

    @property
    def is_bool(self) -> bool:
        return self.kind == UValKind.BOOL

    @property
    def is_bvec(self) -> bool:
        return self.kind == UValKind.BVEC64

    @property
    def is_fp(self) -> bool:
        return self.kind in (UValKind.F16X4, UValKind.F32X2, UValKind.F64)

    @property
    def is_ptr(self) -> bool:
        return self.kind == UValKind.PTR

    @property
    def is_readonly_ptr(self) -> bool:
        return self.kind == UValKind.PTR and isinstance(self.data, PtrVal) and self.data.readonly


def _u32_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def _f32_to_u32(val: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", float(val)))[0]
    except OverflowError:
        return 0xFF800000 if val < 0 else 0x7F800000


def _u16_to_f16(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits & 0xFFFF))[0]


def _f16_to_u16(val: float) -> int:
    try:
        return struct.unpack("<H", struct.pack("<e", float(val)))[0]
    except OverflowError:
        return 0xFC00 if val < 0 else 0x7C00


def _u64_to_f64(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def _f64_to_u64(val: float) -> int:
    try:
        return struct.unpack("<Q", struct.pack("<d", float(val)))[0]
    except OverflowError:
        return 0xFFF0000000000000 if val < 0 else 0x7FF0000000000000


def _is_readonly_base(base: str) -> bool:
    """Return True if a sideband/resource tag represents a read-only uniform/constant region."""
    b = base.lower()
    return b.startswith(("ubo:", "push_const", "uniform:", "ro:", "fau:"))


class _BaseUDomain(SsaDomain):
    """
    Shared lattice-typed UDomain semantics parameterized over a bitvector
    backend (`ConcreteBitvecDomain` or `SymbolicZ3BitvecDomain`).
    """

    def __init__(self, bv_domain: SsaDomain):
        self.bv = bv_domain
        self._ro_reg_cache: Dict[Tuple[str, Any], UVal] = {}

    def _ensure_uval(self, val: Any) -> UVal:
        if isinstance(val, UVal):
            return val
        if isinstance(val, tuple):
            return self._ensure_uval(val[0]) if val else UVal.unknown()
        if val is None:
            return UVal.unknown()
        if isinstance(val, bool):
            return self.const_bool(val)
        if isinstance(val, int):
            return UVal.bvec64(self.bv.const_u64(val & MASK64))
        if isinstance(val, float):
            return self.const_f32(val)
        if z3 is not None and isinstance(val, z3.BoolRef):
            return UVal.bool_val(val)
        return UVal.bvec64(self.bv.as_u64(val))

    def assign(self, current_slot: Any, new_val: Any) -> UVal:
        """Assignment replaces the slot's type with the new value's type."""
        return self._ensure_uval(new_val)

    def bottom(self) -> UVal:
        return UVal.unknown()

    def default_value(self) -> UVal:
        return UVal.unknown()

    def const_u32(self, imm: int) -> UVal:
        return UVal.bvec64(self.bv.const_u64(imm & MASK32))

    def const_u64(self, imm: int) -> UVal:
        return UVal.bvec64(self.bv.const_u64(imm & MASK64))

    def const_bool(self, val: bool) -> UVal:
        return UVal.bool_val(self.bv.const_bool(bool(val)))

    def make_ptr(
        self,
        base: str,
        offset: Any = 0,
        *,
        align_mask: Optional[int] = None,
        readonly: Optional[bool] = None,
    ) -> UVal:
        ro = _is_readonly_base(base) if readonly is None else bool(readonly)
        off_bv = self.bv.const_u64(offset & MASK64) if isinstance(offset, int) else self.bv.as_u64(offset)
        return UVal.ptr(base, off_bv, align_mask=align_mask, readonly=ro)

    def make_sym(self, tag: str, payload: Any = None) -> UVal:
        return UVal.sym(tag, payload)

    def const_f32(self, val: float, hi: float = 0.0) -> UVal:
        raise NotImplementedError

    def const_f16x4(self, v0: float, v1: float = 0.0, v2: float = 0.0, v3: float = 0.0) -> UVal:
        raise NotImplementedError

    def const_f64(self, val: float) -> UVal:
        raise NotImplementedError

    def to_bvec64(self, val: Any) -> Any:
        """Convert any `UVal` to a 64-bit bitvector in `self.bv` (promoting on mismatch)."""
        raise NotImplementedError

    def to_f32x2(self, val: Any) -> Tuple[Any, Any]:
        """Convert any `UVal` to `(f32_lo, f32_hi)` in this domain's FP representation."""
        raise NotImplementedError

    def to_f16x4(self, val: Any) -> Tuple[Any, Any, Any, Any]:
        """Convert any `UVal` to 4 f16 lanes in this domain's FP representation."""
        raise NotImplementedError

    def to_f64(self, val: Any) -> Any:
        """Convert any `UVal` to a scalar f64 in this domain's FP representation."""
        raise NotImplementedError

    def _to_bv32(self, val: Any) -> Any:
        return self.bv.as_u32(self.to_bvec64(val))

    def _to_bv64(self, val: Any) -> Any:
        return self.bv.as_u64(self.to_bvec64(val))

    def _from_bv32(self, bv32: Any) -> UVal:
        return UVal.bvec64(self.bv.as_u64(self.bv.as_u32(bv32)))

    def _from_bv64(self, bv64: Any) -> UVal:
        return UVal.bvec64(self.bv.as_u64(bv64))

    def _from_bool(self, b: Any) -> UVal:
        return UVal.bool_val(self.bv.as_bool(b))

    def _to_bool(self, val: Any) -> Any:
        uv = self._ensure_uval(val)
        if uv.is_unknown:
            return self.bv.const_bool(False)
        if uv.is_bool:
            return uv.data
        return self.bv.bv_ne(self._to_bv32(uv), self.bv.const_u32(0))

    def const_int_val(self, val: Any) -> Optional[int]:
        uv = self._ensure_uval(val)
        if uv.is_bvec:
            return self.bv.const_int_val(uv.data)
        return None

    def as_u32(self, val: Any) -> UVal:
        uv = self._ensure_uval(val)
        if uv.is_bvec:
            return self._from_bv32(uv.data)
        return self._from_bv32(self.to_bvec64(uv))

    def as_u64(self, val: Any) -> UVal:
        uv = self._ensure_uval(val)
        if uv.is_bvec or uv.is_ptr:
            return uv
        return self._from_bv64(self.to_bvec64(uv))

    def as_bool(self, val: Any) -> UVal:
        return self._from_bool(self._to_bool(val))

    def cast_value(self, op: str, val: Any, val_type: str) -> UVal:
        """
        Lazy / deferred bitcast: `Bitcast` preserves the underlying `UVal` and
        its kind without eagerly converting or bit-blasting. Conversion is
        deferred until a downstream operation consumes the value with a
        different kind.
        """
        uv = self._ensure_uval(val)
        if op == "Bitcast":
            return uv
        if val_type in ("U64", "S64"):
            return self.as_u64(uv)
        if val_type == "Bool":
            return self.as_bool(uv)
        if val_type in ("U32", "S32"):
            return self.as_u32(uv)
        if val_type == "F32":
            lo, hi = self.to_f32x2(uv)
            return UVal.f32x2(lo, hi, raw_bv=uv.raw_bv if uv.is_bvec else None)
        if val_type == "F64":
            return UVal.f64(self.to_f64(uv), raw_bv=uv.raw_bv if uv.is_bvec else None)
        return uv

    # --- Bitvector & integer operations (always promote operands to BVEC64) ---

    def int32_unary(self, op: str, x: Any) -> UVal:
        return self._from_bv32(self.bv.int32_unary(op, self._to_bv32(x)))

    def int32_binary(self, op: str, a: Any, b: Any) -> UVal:
        return self._from_bv32(self.bv.int32_binary(op, self._to_bv32(a), self._to_bv32(b)))

    def bfe(self, signed: bool, x: Any, off: Any, width: Any) -> UVal:
        return self._from_bv32(self.bv.bfe(signed, self._to_bv32(x), self._to_bv32(off), self._to_bv32(width)))

    def bfi(self, base: Any, insert: Any, off: Any, width: Any) -> UVal:
        return self._from_bv32(
            self.bv.bfi(self._to_bv32(base), self._to_bv32(insert), self._to_bv32(off), self._to_bv32(width))
        )

    def idot4(self, op: str, a: Any, b: Any) -> UVal:
        return self._from_bv32(self.bv.idot4(op, self._to_bv32(a), self._to_bv32(b)))

    def icmp(self, op: str, a: Any, b: Any) -> UVal:
        return self._from_bool(self.bv.icmp(op, self._to_bv32(a), self._to_bv32(b)))

    def logical_and(self, a: Any, b: Any) -> UVal:
        return self._from_bool(self.bv.logical_and(self._to_bool(a), self._to_bool(b)))

    def logical_or(self, a: Any, b: Any) -> UVal:
        return self._from_bool(self.bv.logical_or(self._to_bool(a), self._to_bool(b)))

    def logical_not(self, a: Any) -> UVal:
        return self._from_bool(self.bv.logical_not(self._to_bool(a)))

    def pack64(self, lo: Any, hi: Any) -> UVal:
        ulo, uhi = self._ensure_uval(lo), self._ensure_uval(hi)
        if ulo.kind == UValKind.F32X2 and uhi.kind == UValKind.F32X2:
            return UVal.f32x2(ulo.data[0], uhi.data[0])
        return self._from_bv64(self.bv.pack64(self._to_bv32(ulo), self._to_bv32(uhi)))

    def split64(self, val64: Any) -> Tuple[UVal, UVal]:
        uv = self._ensure_uval(val64)
        if uv.kind == UValKind.F32X2:
            z0 = self._zero_fp()
            return (UVal.f32x2(uv.data[0], z0), UVal.f32x2(uv.data[1], z0))
        lo, hi = self.bv.split64(self._to_bv64(uv))
        return (self._from_bv32(lo), self._from_bv32(hi))

    def int64_unary(self, op: str, x: Any) -> UVal:
        return self._from_bv64(self.bv.int64_unary(op, self._to_bv64(x)))

    def int64_binary(self, op: str, a: Any, b: Any) -> UVal:
        """
        64-bit binary arithmetic with first-class `PTR` support:
          - `IAdd64(ptr, int)` / `IAdd64(int, ptr)` -> `PTR(base, offset + int)`
          - `ISub64(ptr, int)` -> `PTR(base, offset - int)`
          - `ISub64(ptr_a, ptr_b)` (same base) -> `BVEC64(offset_a - offset_b)`
          - `BitAnd64(ptr, mask)` -> `PTR(base, offset & mask, align_mask=mask)`
            for alignment masks (e.g. `~3`, `~15`), preserving provenance.
        """
        ua, ub = self._ensure_uval(a), self._ensure_uval(b)
        if op == "IAdd64":
            if ua.is_ptr and not ub.is_ptr:
                pa: PtrVal = ua.data
                return UVal.ptr(
                    pa.base,
                    self.bv.bv_add(pa.offset, self._to_bv64(ub), 64),
                    align_mask=pa.align_mask,
                    readonly=pa.readonly,
                )
            if ub.is_ptr and not ua.is_ptr:
                pb: PtrVal = ub.data
                return UVal.ptr(
                    pb.base,
                    self.bv.bv_add(pb.offset, self._to_bv64(ua), 64),
                    align_mask=pb.align_mask,
                    readonly=pb.readonly,
                )
        elif op == "ISub64":
            if ua.is_ptr and not ub.is_ptr:
                pa = ua.data
                return UVal.ptr(
                    pa.base,
                    self.bv.bv_sub(pa.offset, self._to_bv64(ub), 64),
                    align_mask=pa.align_mask,
                    readonly=pa.readonly,
                )
            if ua.is_ptr and ub.is_ptr and ua.data.base == ub.data.base:
                return self._from_bv64(self.bv.bv_sub(ua.data.offset, ub.data.offset, 64))
        elif op == "BitAnd64":
            if ua.is_ptr and not ub.is_ptr:
                pa = ua.data
                mask_c = self.const_int_val(ub)
                if mask_c is not None and (mask_c & 0xFFFFFFFFFFFF0000) == 0xFFFFFFFFFFFF0000:
                    new_mask = mask_c if pa.align_mask is None else (pa.align_mask & mask_c)
                    return UVal.ptr(
                        pa.base,
                        self.bv.bv_and(pa.offset, self.bv.const_u64(mask_c), 64),
                        align_mask=new_mask,
                        readonly=pa.readonly,
                    )
            if ub.is_ptr and not ua.is_ptr:
                pb = ub.data
                mask_c = self.const_int_val(ua)
                if mask_c is not None and (mask_c & 0xFFFFFFFFFFFF0000) == 0xFFFFFFFFFFFF0000:
                    new_mask = mask_c if pb.align_mask is None else (pb.align_mask & mask_c)
                    return UVal.ptr(
                        pb.base,
                        self.bv.bv_and(pb.offset, self.bv.const_u64(mask_c), 64),
                        align_mask=new_mask,
                        readonly=pb.readonly,
                    )
        return self._from_bv64(self.bv.int64_binary(op, self._to_bv64(ua), self._to_bv64(ub)))

    def deref_ptr(self, ptr_val: Any, mem_lookup: Any = None) -> UVal:
        """
        Dereference a `PTR` value. Read-only uniform/UBO/push-constant pointers
        (`readonly=True`) are modeled as immutable read-only registers cached by
        `(base, offset)`, bypassing mutable memory aliasing.
        """
        up = self._ensure_uval(ptr_val)
        if not up.is_ptr:
            if mem_lookup is not None:
                return self._ensure_uval(mem_lookup(self._to_bv64(up)))
            return UVal.unknown()
        pv: PtrVal = up.data
        if pv.readonly:
            key = (pv.base, pv.offset)
            if key not in self._ro_reg_cache:
                if mem_lookup is not None:
                    self._ro_reg_cache[key] = self._ensure_uval(mem_lookup(pv))
                else:
                    self._ro_reg_cache[key] = self._make_ro_register_val(pv.base, pv.offset)
            return self._ro_reg_cache[key]
        if mem_lookup is not None:
            return self._ensure_uval(mem_lookup(pv))
        return UVal.unknown()

    def _make_ro_register_val(self, base: str, offset: Any) -> UVal:
        return UVal.sym(f"ro_reg:{base}", offset)

    def _zero_fp(self) -> Any:
        raise NotImplementedError

    def _select_fp(self, cond_b: Any, a: Any, b: Any) -> Any:
        raise NotImplementedError

    def select(self, cond: Any, t_val: Any, f_val: Any) -> UVal:
        ut, uf = self._ensure_uval(t_val), self._ensure_uval(f_val)
        if ut.is_unknown:
            return uf
        if uf.is_unknown:
            return ut
        cb = self._to_bool(cond)
        # Fast-path concrete bool condition
        if isinstance(cb, bool):
            return ut if cb else uf
        if (ut.kind == UValKind.F32X2 and uf.kind == UValKind.BVEC64) or (
            uf.kind == UValKind.F32X2 and ut.kind == UValKind.BVEC64
        ):
            ft = self.to_f32x2(ut)
            ff = self.to_f32x2(uf)
            return UVal.f32x2(
                self._select_fp(cb, ft[0], ff[0]),
                self._select_fp(cb, ft[1], ff[1]),
            )
        if ut.kind == uf.kind:
            if ut.kind == UValKind.BOOL:
                return UVal.bool_val(self.bv.select(cb, ut.data, uf.data))
            if ut.kind == UValKind.BVEC64:
                return self._from_bv64(self.bv.select(cb, ut.data, uf.data))
            if ut.kind == UValKind.F32X2:
                return UVal.f32x2(
                    self._select_fp(cb, ut.data[0], uf.data[0]),
                    self._select_fp(cb, ut.data[1], uf.data[1]),
                )
            if ut.kind == UValKind.F16X4:
                return UVal.f16x4(
                    tuple(self._select_fp(cb, ut.data[i], uf.data[i]) for i in range(4))
                )
            if ut.kind == UValKind.F64:
                return UVal.f64(self._select_fp(cb, ut.data, uf.data))
            if ut.kind == UValKind.PTR and ut.data.base == uf.data.base:
                return UVal.ptr(
                    ut.data.base,
                    self.bv.select(cb, ut.data.offset, uf.data.offset),
                    align_mask=ut.data.align_mask if ut.data.align_mask == uf.data.align_mask else None,
                    readonly=ut.data.readonly and uf.data.readonly,
                )
            if ut.kind == UValKind.SYM and ut.data == uf.data:
                return ut
        # Mismatched kinds: promote both to BVEC64 at the join/select point
        return self._from_bv64(self.bv.select(cb, self._to_bv64(ut), self._to_bv64(uf)))

    def join(self, a: Any, b: Any) -> UVal:
        """
        Lattice join:
          - `join(UNKNOWN, x) = x`, `join(x, UNKNOWN) = x`
          - Same kind & equal -> `a`
          - Different kinds -> promote to `BVEC64` (top of scalar bit-level join)
            or `SYM` if either is an opaque symbolic handle.
        """
        ua, ub = self._ensure_uval(a), self._ensure_uval(b)
        if ua.is_unknown:
            return ub
        if ub.is_unknown:
            return ua
        if ua == ub:
            return ua
        if ua.kind == UValKind.SYM or ub.kind == UValKind.SYM:
            return UVal.sym("joined_sym", (ua, ub))
        return self._from_bv64(self.bv.join(self._to_bv64(ua), self._to_bv64(ub)))

    def leq(self, a: Any, b: Any) -> bool:
        ua, ub = self._ensure_uval(a), self._ensure_uval(b)
        if ua.is_unknown or ua == ub:
            return True
        if ub.is_unknown:
            return False
        if ua.kind == ub.kind and ua.kind == UValKind.BVEC64:
            return self.bv.leq(ua.data, ub.data)
        return False


class ConcreteUDomain(_BaseUDomain):
    """
    Concrete unified lattice-typed domain:
      - `BVEC64` delegates to `ConcreteBitvecDomain` (bit-exact Python ints).
      - `F16X4`, `F32X2`, `F64` carry native Python floats with exact IEEE-754
        bitcast promotion on type mismatch.
      - `PTR` tracks `(base, offset, align_mask, readonly)` concretely.
    """

    def __init__(self):
        super().__init__(ConcreteBitvecDomain())

    def _zero_fp(self) -> float:
        return 0.0

    def _select_fp(self, cond_b: Any, a: Any, b: Any) -> float:
        return float(a) if bool(cond_b) else float(b)

    def const_f32(self, val: float, hi: float = 0.0) -> UVal:
        return UVal.f32x2(float(val), float(hi))

    def const_f16x4(self, v0: float, v1: float = 0.0, v2: float = 0.0, v3: float = 0.0) -> UVal:
        return UVal.f16x4((float(v0), float(v1), float(v2), float(v3)))

    def const_f64(self, val: float) -> UVal:
        return UVal.f64(float(val))

    def to_bvec64(self, val: Any) -> int:
        uv = self._ensure_uval(val)
        if uv.is_unknown:
            return 0
        if uv.is_bool:
            return 1 if bool(uv.data) else 0
        if uv.is_bvec:
            return int(uv.data) & MASK64
        if uv.raw_bv is not None:
            return int(uv.raw_bv) & MASK64
        if uv.kind == UValKind.F32X2:
            lo_u = _f32_to_u32(uv.data[0])
            hi_u = _f32_to_u32(uv.data[1])
            return ((hi_u << 32) | lo_u) & MASK64
        if uv.kind == UValKind.F16X4:
            out = 0
            for i, lane in enumerate(uv.data):
                out |= (_f16_to_u16(lane) << (16 * i))
            return out & MASK64
        if uv.kind == UValKind.F64:
            return _f64_to_u64(uv.data)
        if uv.kind == UValKind.PTR:
            return int(uv.data.offset) & MASK64
        return 0

    def to_f32x2(self, val: Any) -> Tuple[float, float]:
        uv = self._ensure_uval(val)
        if uv.kind == UValKind.F32X2:
            return (float(uv.data[0]), float(uv.data[1]))
        if uv.is_unknown:
            return (0.0, 0.0)
        b64 = self.to_bvec64(uv)
        return (_u32_to_f32(b64 & MASK32), _u32_to_f32((b64 >> 32) & MASK32))

    def to_f16x4(self, val: Any) -> Tuple[float, float, float, float]:
        uv = self._ensure_uval(val)
        if uv.kind == UValKind.F16X4:
            return tuple(float(x) for x in uv.data)  # type: ignore[return-value]
        if uv.is_unknown:
            return (0.0, 0.0, 0.0, 0.0)
        b64 = self.to_bvec64(uv)
        return tuple(_u16_to_f16((b64 >> (16 * i)) & 0xFFFF) for i in range(4))  # type: ignore[return-value]

    def to_f64(self, val: Any) -> float:
        uv = self._ensure_uval(val)
        if uv.kind == UValKind.F64:
            return float(uv.data)
        if uv.is_unknown:
            return 0.0
        return _u64_to_f64(self.to_bvec64(uv))

    def unpack_half2x16(self, u: Any) -> Tuple[UVal, UVal]:
        uv = self._ensure_uval(u)
        if uv.kind == UValKind.F16X4:
            return (self.const_f32(uv.data[0]), self.const_f32(uv.data[1]))
        b = self.to_bvec64(uv) & MASK32
        return (self.const_f32(_u16_to_f16(b & 0xFFFF)), self.const_f32(_u16_to_f16((b >> 16) & 0xFFFF)))

    def pack_half2x16(self, v2: Any) -> UVal:
        a, b = v2 if isinstance(v2, tuple) and len(v2) == 2 else (v2, self.const_f32(0.0))
        fa = self.to_f32x2(a)[0]
        fb = self.to_f32x2(b)[0]
        return self.const_f16x4(fa, fb, 0.0, 0.0)

    def fp_unary(self, op: str, x: Any) -> UVal:
        fx = self.to_f32x2(x)[0]
        if op == "FAbs": return self.const_f32(abs(fx))
        if op == "FNeg": return self.const_f32(-fx)
        if op == "Floor": return self.const_f32(math.floor(fx) if math.isfinite(fx) else fx)
        if op == "Ceil": return self.const_f32(math.ceil(fx) if math.isfinite(fx) else fx)
        if op == "Trunc": return self.const_f32(math.trunc(fx) if math.isfinite(fx) else fx)
        if op == "RoundEven": return self.const_f32(round(fx) if math.isfinite(fx) else fx)
        if op == "Fract": return self.const_f32(fx - math.floor(fx) if math.isfinite(fx) else float("nan"))
        if op == "Sqrt": return self.const_f32(math.sqrt(fx) if fx >= 0.0 else float("nan"))
        if op == "InverseSqrt":
            if fx > 0.0:
                return self.const_f32(1.0 / math.sqrt(fx))
            return self.const_f32(float("inf") if fx == 0.0 else float("nan"))
        if op == "Exp2": return self.const_f32(math.pow(2.0, fx) if math.isfinite(fx) and fx < 128.0 else (float("inf") if fx >= 128.0 else 0.0))
        if op == "Log2": return self.const_f32(math.log2(fx) if fx > 0.0 else (float("-inf") if fx == 0.0 else float("nan")))
        if op == "Sin": return self.const_f32(math.sin(fx) if math.isfinite(fx) else float("nan"))
        if op == "Cos": return self.const_f32(math.cos(fx) if math.isfinite(fx) else float("nan"))
        return self.const_f32(0.0)

    def fp_binary(self, op: str, a: Any, b: Any) -> UVal:
        fa, fb = self.to_f32x2(a)[0], self.to_f32x2(b)[0]
        if op == "FAdd": return self.const_f32(fa + fb)
        if op == "FSub": return self.const_f32(fa - fb)
        if op == "FMul": return self.const_f32(fa * fb)
        if op == "FDiv":
            if fb == 0.0:
                if fa == 0.0: return self.const_f32(float("nan"))
                return self.const_f32(float("inf") if (fa > 0.0) else float("-inf"))
            return self.const_f32(fa / fb)
        if op in ("FMin", "NMin"):
            if math.isnan(fa): return self.const_f32(fb)
            if math.isnan(fb): return self.const_f32(fa)
            return self.const_f32(min(fa, fb))
        if op in ("FMax", "NMax"):
            if math.isnan(fa): return self.const_f32(fb)
            if math.isnan(fb): return self.const_f32(fa)
            return self.const_f32(max(fa, fb))
        return self.const_f32(0.0)

    def fp_fma(self, a: Any, b: Any, c: Any) -> UVal:
        fa, fb, fc = self.to_f32x2(a)[0], self.to_f32x2(b)[0], self.to_f32x2(c)[0]
        return self.const_f32(fa * fb + fc)

    def fp_cmp(self, op: str, a: Any, b: Any) -> UVal:
        fa, fb = self.to_f32x2(a)[0], self.to_f32x2(b)[0]
        unord = math.isnan(fa) or math.isnan(fb)
        if op == "FCmpOrdEq": return self.const_bool(not unord and fa == fb)
        if op == "FCmpOrdNe": return self.const_bool(not unord and fa != fb)
        if op == "FCmpOrdLt": return self.const_bool(not unord and fa < fb)
        if op == "FCmpOrdLe": return self.const_bool(not unord and fa <= fb)
        if op == "FCmpOrdGt": return self.const_bool(not unord and fa > fb)
        if op == "FCmpOrdGe": return self.const_bool(not unord and fa >= fb)
        if op == "FCmpUnordEq": return self.const_bool(unord or fa == fb)
        if op == "FCmpUnordNe": return self.const_bool(unord or fa != fb)
        if op == "FCmpUnordLt": return self.const_bool(unord or fa < fb)
        if op == "FCmpUnordLe": return self.const_bool(unord or fa <= fb)
        if op == "FCmpUnordGt": return self.const_bool(unord or fa > fb)
        if op == "FCmpUnordGe": return self.const_bool(unord or fa >= fb)
        return self.const_bool(False)

    def fp_is_nan(self, x: Any) -> UVal:
        return self.const_bool(math.isnan(self.to_f32x2(x)[0]))

    def fp_is_inf(self, x: Any) -> UVal:
        return self.const_bool(math.isinf(self.to_f32x2(x)[0]))

    def convert(self, op: str, x: Any) -> UVal:
        if op == "ConvertSToF":
            u = self.to_bvec64(x) & MASK32
            s = u - 0x100000000 if (u & 0x80000000) else u
            return self.const_f32(float(s))
        if op == "ConvertUToF":
            return self.const_f32(float(self.to_bvec64(x) & MASK32))
        if op == "ConvertFToS":
            fx = self.to_f32x2(x)[0]
            if not math.isfinite(fx):
                return self.const_u32(0)
            return self.const_u32(max(-0x80000000, min(0x7FFFFFFF, int(math.trunc(fx)))) & MASK32)
        if op == "ConvertFToU":
            fx = self.to_f32x2(x)[0]
            if not math.isfinite(fx) or fx <= 0.0:
                return self.const_u32(0)
            return self.const_u32(min(MASK32, int(math.trunc(fx))) & MASK32)
        if op == "UConvert32To64":
            return self.const_u64(self.to_bvec64(x) & MASK32)
        if op == "SConvert32To64":
            u = self.to_bvec64(x) & MASK32
            return self.const_u64((u | 0xFFFFFFFF00000000) if (u & 0x80000000) else u)
        if op == "Trunc64To32":
            return self.const_u32(self.to_bvec64(x) & MASK32)
        if op == "FConvert32To64":
            return self.const_f64(self.to_f32x2(x)[0])
        if op == "FConvert64To32":
            return self.const_f32(self.to_f64(x))
        return self._ensure_uval(x)

    def f16x4_binary(self, op: str, a: Any, b: Any) -> UVal:
        la, lb = self.to_f16x4(a), self.to_f16x4(b)
        fn = {
            "FAdd": lambda x, y: x + y,
            "FSub": lambda x, y: x - y,
            "FMul": lambda x, y: x * y,
        }.get(op, lambda x, y: 0.0)
        return self.const_f16x4(*(fn(la[i], lb[i]) for i in range(4)))

    def f64_binary(self, op: str, a: Any, b: Any) -> UVal:
        da, db = self.to_f64(a), self.to_f64(b)
        if op == "FAdd": return self.const_f64(da + db)
        if op == "FSub": return self.const_f64(da - db)
        if op == "FMul": return self.const_f64(da * db)
        if op == "FDiv": return self.const_f64(da / db if db != 0.0 else float("nan"))
        return self.const_f64(0.0)


class SymbolicZ3UDomain(_BaseUDomain, Z3SymbolicDomain):
    """
    Symbolic Z3 unified lattice-typed domain:
      - `BVEC64` delegates to `SymbolicZ3BitvecDomain`.
      - `F16X4`, `F32X2`, `F64` are modeled as Z3 `Real`s (avoiding IEEE-754
        bit-blasting for well-formed FP shaders).
      - Type-mismatch promotion (`F32X2 -> BVEC64` or `BVEC64 -> F32X2`) uses:
          * exact IEEE-754 bit packing/unpacking on concrete constants;
          * zero-cost `raw_bv` recovery when round-tripping;
          * uninterpreted bridge functions (`u_f32_to_bv32` / `u_bv32_to_f32`,
            etc.) by default, or exact Z3 `fpToIEEEBV`/`fpBVToFP` when
            `exact_bitcast=True`.
      - `PTR` tracks `(base, z3.BitVec64 offset, align_mask, readonly)`.
    """

    def __init__(self, *, exact_bitcast: bool = False):
        if z3 is None:
            raise RuntimeError("z3-solver is required for SymbolicZ3UDomain")
        super().__init__(SymbolicZ3BitvecDomain(fp_mode="real"))
        self.exact_bitcast = bool(exact_bitcast)
        self._zero_real = z3.RealVal(0)
        self._u_f32_to_bv32 = z3.Function("u_f32_to_bv32", z3.RealSort(), z3.BitVecSort(32))
        self._u_bv32_to_f32 = z3.Function("u_bv32_to_f32", z3.BitVecSort(32), z3.RealSort())
        self._u_f16_to_bv16 = z3.Function("u_f16_to_bv16", z3.RealSort(), z3.BitVecSort(16))
        self._u_bv16_to_f16 = z3.Function("u_bv16_to_f16", z3.BitVecSort(16), z3.RealSort())
        self._u_f64_to_bv64 = z3.Function("u_f64_to_bv64", z3.RealSort(), z3.BitVecSort(64))
        self._u_bv64_to_f64 = z3.Function("u_bv64_to_f64", z3.BitVecSort(64), z3.RealSort())
        self._u_s32_to_f32 = z3.Function("u_s32_to_f32", z3.BitVecSort(32), z3.RealSort())
        self._u_u32_to_f32 = z3.Function("u_u32_to_f32", z3.BitVecSort(32), z3.RealSort())
        self._u_f32_to_s32 = z3.Function("u_f32_to_s32", z3.RealSort(), z3.BitVecSort(32))
        self._u_f32_to_u32 = z3.Function("u_f32_to_u32", z3.RealSort(), z3.BitVecSort(32))
        self._fp_ufs: Dict[Tuple[str, int], Any] = {}
        self.bitcast_materializations: int = 0

    def _get_fp_uf(self, op: str, arity: int) -> Any:
        key = (op, arity)
        if key not in self._fp_ufs:
            sorts = [z3.RealSort()] * (arity + 1)
            self._fp_ufs[key] = z3.Function(f"u_fp_{op}", *sorts)
        return self._fp_ufs[key]

    def _zero_fp(self) -> Any:
        return self._zero_real

    def _select_fp(self, cond_b: Any, a: Any, b: Any) -> Any:
        return z3.If(cond_b, a, b)

    def _make_ro_register_val(self, base: str, offset: Any) -> UVal:
        # Read-only uniform/UBO/push-constant loads act as pure symbolic registers
        off_c = self.bv.const_int_val(offset)
        suffix = f"0x{off_c:x}" if off_c is not None else str(offset)
        return UVal.bvec64(z3.BitVec(f"ro_reg_{base}_{suffix}", 64))

    def fresh_bvec64(self, name: str) -> UVal:
        return UVal.bvec64(z3.BitVec(name, 64))

    def fresh_f32(self, name: str) -> UVal:
        return UVal.f32x2(z3.Real(name), self._zero_real)

    def fresh_f32x2(self, name: str) -> UVal:
        return UVal.f32x2(z3.Real(f"{name}_lo"), z3.Real(f"{name}_hi"))

    def fresh_f16x4(self, name: str) -> UVal:
        return UVal.f16x4(tuple(z3.Real(f"{name}_{i}") for i in range(4)))

    def fresh_f64(self, name: str) -> UVal:
        return UVal.f64(z3.Real(name))

    def const_f32(self, val: float, hi: float = 0.0) -> UVal:
        return UVal.f32x2(self._float_to_real(val), self._float_to_real(hi))

    def const_f16x4(self, v0: float, v1: float = 0.0, v2: float = 0.0, v3: float = 0.0) -> UVal:
        return UVal.f16x4(tuple(self._float_to_real(v) for v in (v0, v1, v2, v3)))

    def const_f64(self, val: float) -> UVal:
        return UVal.f64(self._float_to_real(val))

    @staticmethod
    def _float_to_real(val: float) -> Any:
        if math.isfinite(val):
            return z3.RealVal(repr(float(val)))
        return z3.Real(f"nonfinite_{val}")

    @staticmethod
    def _try_real_to_float(r: Any) -> Optional[float]:
        if z3.is_rational_value(r):
            return float(r.numerator_as_long()) / float(r.denominator_as_long())
        return None

    def _real_to_bv32(self, r: Any) -> Any:
        f = self._try_real_to_float(r)
        if f is not None:
            return z3.BitVecVal(_f32_to_u32(f), 32)
        if z3.is_app(r) and r.decl().name() == "u_bv32_to_f32" and r.num_args() == 1:
            return r.arg(0)
        self.bitcast_materializations += 1
        if self.exact_bitcast:
            return z3.fpToIEEEBV(z3.fpRealToFP(z3.RNE(), r, z3.Float32()))
        return self._u_f32_to_bv32(r)

    def _bv32_to_real(self, bv32: Any) -> Any:
        c = self.bv.const_int_val(bv32)
        if c is not None:
            return self._float_to_real(_u32_to_f32(c))
        if z3.is_app(bv32) and bv32.decl().name() == "u_f32_to_bv32" and bv32.num_args() == 1:
            return bv32.arg(0)
        self.bitcast_materializations += 1
        if self.exact_bitcast:
            return z3.fpToReal(z3.fpBVToFP(bv32, z3.Float32()))
        return self._u_bv32_to_f32(bv32)

    def _real_to_bv16(self, r: Any) -> Any:
        f = self._try_real_to_float(r)
        if f is not None:
            return z3.BitVecVal(_f16_to_u16(f), 16)
        self.bitcast_materializations += 1
        if self.exact_bitcast:
            return z3.fpToIEEEBV(z3.fpRealToFP(z3.RNE(), r, z3.Float16()))
        return self._u_f16_to_bv16(r)

    def _bv16_to_real(self, bv16: Any) -> Any:
        c = self.bv.const_int_val(bv16)
        if c is not None:
            return self._float_to_real(_u16_to_f16(c))
        self.bitcast_materializations += 1
        if self.exact_bitcast:
            return z3.fpToReal(z3.fpBVToFP(bv16, z3.Float16()))
        return self._u_bv16_to_f16(bv16)

    def _real_to_bv64(self, r: Any) -> Any:
        f = self._try_real_to_float(r)
        if f is not None:
            return z3.BitVecVal(_f64_to_u64(f), 64)
        self.bitcast_materializations += 1
        if self.exact_bitcast:
            return z3.fpToIEEEBV(z3.fpRealToFP(z3.RNE(), r, z3.Float64()))
        return self._u_f64_to_bv64(r)

    def _bv64_to_real(self, bv64: Any) -> Any:
        c = self.bv.const_int_val(bv64)
        if c is not None:
            return self._float_to_real(_u64_to_f64(c))
        self.bitcast_materializations += 1
        if self.exact_bitcast:
            return z3.fpToReal(z3.fpBVToFP(bv64, z3.Float64()))
        return self._u_bv64_to_f64(bv64)

    def _to_bv32(self, val: Any) -> Any:
        uv = self._ensure_uval(val)
        if uv.is_unknown:
            return self.bv.const_u32(0)
        if uv.is_bool:
            if z3.is_true(uv.data):
                return self.bv.const_u32(1)
            if z3.is_false(uv.data):
                return self.bv.const_u32(0)
            return z3.If(uv.data, self.bv.const_u32(1), self.bv.const_u32(0))
        if uv.kind == UValKind.F32X2 and uv.raw_bv is None:
            return self._real_to_bv32(uv.data[0])
        b64 = self.to_bvec64(uv)
        if isinstance(b64, z3.BitVecNumRef):
            return z3.BitVecVal(b64.as_long() & MASK32, 32)
        if isinstance(b64, z3.BitVecRef) and b64.size() == 32:
            return b64
        if z3.is_app(b64):
            k = b64.decl().kind()
            if k == z3.Z3_OP_ZERO_EXT:
                inner = b64.arg(0)
                if isinstance(inner, z3.BitVecRef) and inner.size() == 32:
                    return inner
            elif k == z3.Z3_OP_CONCAT:
                lsb = b64.arg(b64.num_args() - 1)
                if isinstance(lsb, z3.BitVecRef) and lsb.size() == 32:
                    return lsb
        return z3.Extract(31, 0, b64)

    def to_bvec64(self, val: Any) -> Any:
        uv = self._ensure_uval(val)
        if uv.is_unknown:
            return self.bv.const_u64(0)
        if uv.is_bool:
            if z3.is_true(uv.data):
                return self.bv.const_u64(1)
            if z3.is_false(uv.data):
                return self.bv.const_u64(0)
            return z3.If(uv.data, self.bv.const_u64(1), self.bv.const_u64(0))
        if uv.is_bvec:
            return self.bv.as_u64(uv.data)
        if uv.raw_bv is not None:
            return self.bv.as_u64(uv.raw_bv)
        if uv.kind == UValKind.F32X2:
            lo = self._real_to_bv32(uv.data[0])
            hi = self._real_to_bv32(uv.data[1])
            return z3.Concat(hi, lo)
        if uv.kind == UValKind.F16X4:
            lanes = [self._real_to_bv16(uv.data[i]) for i in range(4)]
            return z3.Concat(lanes[3], lanes[2], lanes[1], lanes[0])
        if uv.kind == UValKind.F64:
            return self._real_to_bv64(uv.data)
        if uv.kind == UValKind.PTR:
            return self.bv.as_u64(uv.data.offset)
        return z3.BitVec(f"sym_bv64_{uv.data.tag}", 64)

    def to_f32x2(self, val: Any) -> Tuple[Any, Any]:
        uv = self._ensure_uval(val)
        if uv.kind == UValKind.F32X2:
            return (uv.data[0], uv.data[1])
        if uv.is_unknown:
            return (self._zero_real, self._zero_real)
        b64 = self.to_bvec64(uv)
        lo = z3.simplify(z3.Extract(31, 0, b64))
        hi = z3.simplify(z3.Extract(63, 32, b64))
        return (self._bv32_to_real(lo), self._bv32_to_real(hi))

    def to_f16x4(self, val: Any) -> Tuple[Any, Any, Any, Any]:
        uv = self._ensure_uval(val)
        if uv.kind == UValKind.F16X4:
            return uv.data
        if uv.is_unknown:
            return (self._zero_real, self._zero_real, self._zero_real, self._zero_real)
        b64 = self.to_bvec64(uv)
        return tuple(
            self._bv16_to_real(z3.simplify(z3.Extract(16 * i + 15, 16 * i, b64)))
            for i in range(4)
        )  # type: ignore[return-value]

    def to_f64(self, val: Any) -> Any:
        uv = self._ensure_uval(val)
        if uv.kind == UValKind.F64:
            return uv.data
        if uv.is_unknown:
            return self._zero_real
        return self._bv64_to_real(self.to_bvec64(uv))

    def unpack_half2x16(self, u: Any) -> Tuple[UVal, UVal]:
        uv = self._ensure_uval(u)
        if uv.kind == UValKind.F16X4:
            return (UVal.f32x2(uv.data[0], self._zero_real), UVal.f32x2(uv.data[1], self._zero_real))
        lanes = self.to_f16x4(uv)
        return (UVal.f32x2(lanes[0], self._zero_real), UVal.f32x2(lanes[1], self._zero_real))

    def pack_half2x16(self, v2: Any) -> UVal:
        a, b = v2 if isinstance(v2, tuple) and len(v2) == 2 else (v2, UVal.f32x2(self._zero_real, self._zero_real))
        fa = self.to_f32x2(a)[0]
        fb = self.to_f32x2(b)[0]
        return UVal.f16x4((fa, fb, self._zero_real, self._zero_real))

    def fp_unary(self, op: str, x: Any) -> UVal:
        rx = self.to_f32x2(x)[0]
        if op == "FAbs":
            return UVal.f32x2(z3.If(rx < 0, -rx, rx), self._zero_real)
        if op == "FNeg":
            return UVal.f32x2(-rx, self._zero_real)
        if op == "Sin":
            return UVal.f32x2(_real_sin(rx), self._zero_real)
        if op == "Cos":
            return UVal.f32x2(_real_cos(rx), self._zero_real)
        if op == "Sqrt":
            return UVal.f32x2(_real_sqrt(rx), self._zero_real)
        if op == "InverseSqrt":
            return UVal.f32x2(z3.RealVal(1) / _real_sqrt(rx), self._zero_real)
        if op == "Exp2":
            return UVal.f32x2(_real_exp2(rx), self._zero_real)
        if op == "Exp":
            # Match Valhall lowering Exp(x) = Exp2(x * log2(e)), where log2(e) as f32 is 0x3FB8AA3B
            return UVal.f32x2(_real_exp2(rx * self._float_to_real(_u32_to_f32(0x3FB8AA3B))), self._zero_real)
        if op == "Log2":
            return UVal.f32x2(_real_log2(rx), self._zero_real)
        if op == "Log":
            return UVal.f32x2(_real_log2(rx) * self._float_to_real(_u32_to_f32(0x3F317218)), self._zero_real)
        if op in ("Floor", "Fract", "Trunc", "Ceil"):
            fx = self._try_real_to_float(rx)
            if fx is not None and math.isfinite(fx):
                if op == "Floor": return UVal.f32x2(self._float_to_real(math.floor(fx)), self._zero_real)
                if op == "Fract": return UVal.f32x2(self._float_to_real(fx - math.floor(fx)), self._zero_real)
                if op == "Trunc": return UVal.f32x2(self._float_to_real(math.trunc(fx)), self._zero_real)
                if op == "Ceil": return UVal.f32x2(self._float_to_real(math.ceil(fx)), self._zero_real)
            if not self.exact_bitcast:
                if op == "Fract":
                    return UVal.f32x2(rx - self._get_fp_uf("Floor", 1)(rx), self._zero_real)
                return UVal.f32x2(self._get_fp_uf(op, 1)(rx), self._zero_real)
            fl = z3.ToReal(z3.ToInt(rx))
            if op == "Floor": return UVal.f32x2(fl, self._zero_real)
            if op == "Fract": return UVal.f32x2(rx - fl, self._zero_real)
            if op == "Trunc": return UVal.f32x2(z3.If(rx >= 0, fl, z3.If(rx == fl, fl, fl + z3.RealVal(1))), self._zero_real)
            return UVal.f32x2(z3.If(rx == fl, fl, fl + z3.RealVal(1)), self._zero_real)
        return UVal.f32x2(self._get_fp_uf(op, 1)(rx), self._zero_real)

    def fp_binary(self, op: str, a: Any, b: Any) -> UVal:
        ra, rb = self.to_f32x2(a)[0], self.to_f32x2(b)[0]
        if op == "FAdd": return UVal.f32x2(ra + rb, self._zero_real)
        if op == "FSub": return UVal.f32x2(ra - rb, self._zero_real)
        if op == "FMul": return UVal.f32x2(ra * rb, self._zero_real)
        if op == "FDiv": return UVal.f32x2(ra / rb, self._zero_real)
        if op in ("FMin", "NMin"): return UVal.f32x2(z3.If(ra <= rb, ra, rb), self._zero_real)
        if op in ("FMax", "NMax"): return UVal.f32x2(z3.If(ra >= rb, ra, rb), self._zero_real)
        if op == "Pow":
            fb = self._try_real_to_float(rb)
            if fb is not None:
                if fb == 0.0:
                    return UVal.f32x2(z3.RealVal(1), self._zero_real)
                if fb == 1.0:
                    return UVal.f32x2(ra, self._zero_real)
                if fb == 2.0:
                    return UVal.f32x2(ra * ra, self._zero_real)
                if fb == 0.5:
                    return UVal.f32x2(_real_sqrt(ra), self._zero_real)
            return UVal.f32x2(_real_exp2(rb * _real_log2(ra)), self._zero_real)
        return UVal.f32x2(self._get_fp_uf(op, 2)(ra, rb), self._zero_real)

    def fp_fma(self, a: Any, b: Any, c: Any) -> UVal:
        ra, rb, rc = self.to_f32x2(a)[0], self.to_f32x2(b)[0], self.to_f32x2(c)[0]
        return UVal.f32x2(ra * rb + rc, self._zero_real)

    def fp_cmp(self, op: str, a: Any, b: Any) -> UVal:
        ra, rb = self.to_f32x2(a)[0], self.to_f32x2(b)[0]
        if op in ("FCmpOrdEq", "FCmpUnordEq"): return self._from_bool(ra == rb)
        if op in ("FCmpOrdNe", "FCmpUnordNe"): return self._from_bool(ra != rb)
        if op in ("FCmpOrdLt", "FCmpUnordLt"): return self._from_bool(ra < rb)
        if op in ("FCmpOrdLe", "FCmpUnordLe"): return self._from_bool(ra <= rb)
        if op in ("FCmpOrdGt", "FCmpUnordGt"): return self._from_bool(ra > rb)
        if op in ("FCmpOrdGe", "FCmpUnordGe"): return self._from_bool(ra >= rb)
        return self.const_bool(False)

    def convert(self, op: str, x: Any) -> UVal:
        if op == "ConvertSToF":
            bx = self._to_bv32(x)
            c = self.bv.const_int_val(bx)
            if c is not None:
                sv = c - 0x100000000 if (c & 0x80000000) else c
                return UVal.f32x2(self._float_to_real(float(sv)), self._zero_real)
            if not self.exact_bitcast:
                return UVal.f32x2(self._u_s32_to_f32(bx), self._zero_real)
            return UVal.f32x2(z3.ToReal(z3.BV2Int(bx, is_signed=True)), self._zero_real)
        if op == "ConvertUToF":
            bx = self._to_bv32(x)
            c = self.bv.const_int_val(bx)
            if c is not None:
                return UVal.f32x2(self._float_to_real(float(c & 0xFFFFFFFF)), self._zero_real)
            if not self.exact_bitcast:
                return UVal.f32x2(self._u_u32_to_f32(bx), self._zero_real)
            return UVal.f32x2(z3.ToReal(z3.BV2Int(bx, is_signed=False)), self._zero_real)
        if op in ("ConvertFToS", "ConvertFToU"):
            rx = self.to_f32x2(x)[0]
            fx = self._try_real_to_float(rx)
            if fx is not None and math.isfinite(fx):
                iv = int(math.trunc(fx))
                return self._from_bv32(z3.BitVecVal(iv & 0xFFFFFFFF, 32))
            if not self.exact_bitcast:
                fn = self._u_f32_to_s32 if op == "ConvertFToS" else self._u_f32_to_u32
                return self._from_bv32(fn(rx))
            return self._from_bv32(z3.Int2BV(z3.ToInt(rx), 32))
        if op == "UConvert32To64":
            return self._from_bv64(z3.ZeroExt(32, self._to_bv32(x)))
        if op == "SConvert32To64":
            return self._from_bv64(z3.SignExt(32, self._to_bv32(x)))
        if op == "Trunc64To32":
            return self._from_bv32(z3.Extract(31, 0, self._to_bv64(x)))
        if op == "FConvert32To64":
            return UVal.f64(self.to_f32x2(x)[0])
        if op == "FConvert64To32":
            return UVal.f32x2(self.to_f64(x), self._zero_real)
        return self._ensure_uval(x)

    def f16x4_binary(self, op: str, a: Any, b: Any) -> UVal:
        la, lb = self.to_f16x4(a), self.to_f16x4(b)
        if op == "FAdd": return UVal.f16x4(tuple(la[i] + lb[i] for i in range(4)))
        if op == "FSub": return UVal.f16x4(tuple(la[i] - lb[i] for i in range(4)))
        if op == "FMul": return UVal.f16x4(tuple(la[i] * lb[i] for i in range(4)))
        return UVal.f16x4(tuple(self._get_fp_uf(f"f16_{op}", 2)(la[i], lb[i]) for i in range(4)))

    def f64_binary(self, op: str, a: Any, b: Any) -> UVal:
        da, db = self.to_f64(a), self.to_f64(b)
        if op == "FAdd": return UVal.f64(da + db)
        if op == "FSub": return UVal.f64(da - db)
        if op == "FMul": return UVal.f64(da * db)
        if op == "FDiv": return UVal.f64(da / db)
        return UVal.f64(self._get_fp_uf(f"f64_{op}", 2)(da, db))

    def hw_uf(self, env: Any, name: str, *args: Any, out_bits: int = 32, fallback_tag: Optional[str] = None) -> UVal:
        bv_args = [self._to_bv32(a) for a in args]
        res = self.bv.hw_uf(env, name, *bv_args, out_bits=out_bits, fallback_tag=fallback_tag)
        return self._from_bv64(res) if out_bits == 64 else self._from_bv32(res)

    def unknown_sideband_u64(self, env: Any, imm_u32: int) -> UVal:
        return self._from_bv64(self.bv.unknown_sideband_u64(env, imm_u32))

    def sideband_u32(self, env: Any, imm_u32: int, imm_u32_2: int) -> UVal:
        return self._from_bv32(self.bv.sideband_u32(env, imm_u32, imm_u32_2))

    def subgroup_ballot(self, env: Any, x: Any) -> UVal:
        return self._from_bv32(self.bv.subgroup_ballot(env, self._to_bv32(x)))

    def subgroup_shuffle(self, env: Any, x: Any, lane: Any) -> UVal:
        return self._from_bv32(self.bv.subgroup_shuffle(env, self._to_bv32(x), self._to_bv32(lane)))

    def atomic_compute(self, env: Any, op: str, op_tag: str, old_val: Any, arg: Any, cmp_val: Any) -> UVal:
        res = self.bv.atomic_compute(env, op, op_tag, self._to_bv32(old_val), self._to_bv32(arg), self._to_bv32(cmp_val))
        return self._from_bv32(res)

    # Z3SymbolicDomain boundary contract
    def lift_z3_bv32(self, raw: Any) -> UVal:
        return self._from_bv32(raw)

    def lift_z3_bv64(self, raw: Any) -> UVal:
        return self._from_bv64(raw)

    def lift_z3_bool(self, raw: Any) -> UVal:
        return self._from_bool(raw)

    def lower_z3_bv32(self, val: Any) -> Any:
        return self._to_bv32(val)

    def lower_z3_bv64(self, val: Any) -> Any:
        return self._to_bv64(val)

    def lower_z3_bool(self, val: Any) -> Any:
        return self._to_bool(val)

    def lower_z3_val(self, val: Any) -> Any:
        if isinstance(val, tuple):
            return tuple(self.lower_z3_val(v) for v in val)
        return self._to_bv32(val)

    def merge_z3(self, cond: Any, t_val: Any, f_val: Any) -> UVal:
        return self.select(self.lift_z3_bool(cond), t_val, f_val)

    def to_z3_real_f32(self, val: Any) -> Any:
        return self.to_f32x2(val)[0]

"""
Abstract Base Value Domain (`SsaDomain`) for ValhallSsaModule Interpretation.

Isolates shared denotational semantics (`int32_unary`, `int32_binary`,
`int64_unary`, `int64_binary`, `icmp`, `bfe`, `bfi`, `idot4`, `pack64`,
`split64`, `fp_unary`, `fp_binary`, `fp_clamp`, `fp_fma`, `atomic_compute`,
`cast_value`) once inside `SsaDomain`, delegating only minimal domain
primitives to concrete, symbolic, and abstract-interpretation subclasses.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, Sequence, Tuple, TypeVar

V = TypeVar("V")
S = TypeVar("S")


@dataclass(frozen=True, slots=True)
class LaneVal(Generic[S]):
    """Fixed lane-aware value container for strictly-typed `ValueDomain[LaneVal[S]]`."""

    lanes: Tuple[S, ...]
    is_tuple: bool = False

    @classmethod
    def scalar(cls, val: S) -> "LaneVal[S]":
        return cls((val,), is_tuple=False)

    @classmethod
    def vector(cls, lanes: Sequence[S]) -> "LaneVal[S]":
        return cls(tuple(lanes), is_tuple=False)

    @classmethod
    def pack_tuple(cls, items: Sequence[S]) -> "LaneVal[S]":
        return cls(tuple(items), is_tuple=True)

    @property
    def num_lanes(self) -> int:
        return len(self.lanes)

    def as_scalar(self) -> S:
        return self.lanes[0]

    def __len__(self) -> int:
        return len(self.lanes)

    def __getitem__(self, idx: int) -> S:
        return self.lanes[idx]

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

_F32_ONE_BITS = 0x3F800000
_F32_LOG2_E_BITS = 0x3FB8AA3B
_F32_LN_2_BITS = 0x3F317218


class SsaDomain(Generic[V], abc.ABC):
    """
    Abstract value domain consumed by `SsaInterpreter`.

    Subclasses implement minimal bitvector/boolean/FP/conversion primitives;
    compound operations and lattice defaults are provided here so concrete,
    symbolic, and abstract-interpretation domains cannot drift in semantics.
    """

    @abc.abstractmethod
    def const_u32(self, imm: int) -> Any: ...

    @abc.abstractmethod
    def const_u64(self, imm: int) -> Any: ...

    @abc.abstractmethod
    def const_bool(self, val: bool) -> Any: ...

    def default_value(self) -> Any:
        return self.const_u32(0)

    def pack_lanes(self, lanes: Sequence[V]) -> V:
        """Pack multiple scalar lanes into a single domain value `V`."""
        return tuple(lanes)  # type: ignore[return-value]

    def unpack_lanes(self, val: V) -> Tuple[V, ...]:
        """Return the scalar lanes of `val` as a tuple (1-tuple for scalars)."""
        if isinstance(val, tuple):
            return val  # type: ignore[return-value]
        return (val,)

    def num_lanes(self, val: V) -> int:
        """Return lane count of `val` (1 for scalar, N for N-lane vector)."""
        return len(val) if isinstance(val, tuple) else 1

    def extract_lane(self, val: V, index: int) -> V:
        """Extract lane `index` from a vector value `V` (or coerce scalar)."""
        if isinstance(val, tuple) and index < len(val):
            return val[index]  # type: ignore[return-value]
        return self.cast_value("Copy", val, "U32")

    def shuffle_lanes(self, a: V, b: V, indices: Sequence[int]) -> V:
        """Construct a swizzled/shuffled vector from lanes of `a` and `b`."""
        comb = list(self.unpack_lanes(a)) + list(self.unpack_lanes(b))
        return self.pack_lanes([comb[i] for i in indices])

    def pack_tuple(self, items: Sequence[V]) -> V:
        """Pack heterogeneous `(value, token)` or `(lo, hi)` pair into `V`."""
        return tuple(items)  # type: ignore[return-value]

    def extract_tuple(self, val: V, index: int) -> V:
        """Extract element `index` from a packed tuple value `V`."""
        if isinstance(val, tuple) and index < len(val):
            return val[index]  # type: ignore[return-value]
        return val

    def map_lanes(self, fn: Callable[..., V], *xs: V) -> V:
        """Apply a scalar domain operation lane-wise across scalar/vector arguments `xs`."""
        lens = [self.num_lanes(x) for x in xs if self.num_lanes(x) > 1]
        if not lens:
            return fn(*xs)
        n = min(lens)
        unpacked = [self.unpack_lanes(x) for x in xs]
        return self.pack_lanes([
            fn(*(u[i] if len(u) > 1 else u[0] for u in unpacked))
            for i in range(n)
        ])


    def bottom(self) -> Any:
        """Lattice bottom element (override in abstract lattice domains)."""
        return self.default_value()

    def top(self) -> Any:
        """Lattice top element (override in abstract lattice domains)."""
        return self.default_value()

    def join(self, a: Any, b: Any) -> Any:
        """Lattice join (least upper bound)."""
        return a if a == b else self.top()

    def meet(self, a: Any, b: Any) -> Any:
        """Lattice meet (greatest lower bound)."""
        return a if a == b else self.bottom()

    def leq(self, a: Any, b: Any) -> bool:
        """Lattice partial order (`a <= b`)."""
        return a == b or b == self.top() or a == self.bottom()

    def const_int_val(self, val: Any) -> Optional[int]:
        """Return concrete Python int if `val` is a known constant in this domain, else None."""
        return None

    def mask32(self, val: Any) -> Any:
        return val

    def mask64(self, val: Any) -> Any:
        return val

    def as_u32(self, val: Any) -> Any:
        return val

    def as_s32(self, val: Any) -> Any:
        return self.as_u32(val)

    def as_u64(self, val: Any) -> Any:
        return val

    def as_s64(self, val: Any) -> Any:
        return self.as_u64(val)

    def as_f32(self, val: Any) -> Any:
        return self.as_u32(val)

    def as_bool(self, val: Any) -> Any:
        return val

    def cast_value(self, op: str, val: Any, val_type: str) -> Any:
        if val_type == "U64":
            return self.as_u64(val)
        if val_type == "S64":
            return self.as_s64(val)
        if val_type == "Bool":
            return self.as_bool(val)
        if val_type == "F32" and not isinstance(val, tuple):
            return self.as_u32(val)
        if val_type == "S32" and not isinstance(val, tuple):
            return self.as_s32(val)
        if val_type == "U32" and not isinstance(val, tuple):
            return self.as_u32(val)
        return val

    def bv_add(self, a: Any, b: Any, bits: int = 32) -> Any: return self.default_value()
    def bv_sub(self, a: Any, b: Any, bits: int = 32) -> Any: return self.default_value()
    def bv_mul(self, a: Any, b: Any, bits: int = 32) -> Any: return self.default_value()
    def bv_and(self, a: Any, b: Any, bits: int = 32) -> Any: return self.default_value()
    def bv_or(self, a: Any, b: Any, bits: int = 32) -> Any: return self.default_value()
    def bv_xor(self, a: Any, b: Any, bits: int = 32) -> Any: return self.default_value()
    def bv_not(self, a: Any, bits: int = 32) -> Any: return self.default_value()
    def bv_neg(self, a: Any, bits: int = 32) -> Any: return self.default_value()
    def bv_shl(self, a: Any, sh: Any, bits: int = 32) -> Any: return self.default_value()
    def bv_lshr(self, a: Any, sh: Any, bits: int = 32) -> Any: return self.default_value()
    def bv_ashr(self, a: Any, sh: Any, bits: int = 32) -> Any: return self.default_value()
    def bv_rotl(self, a: Any, sh: Any) -> Any: return self.default_value()
    def bv_rotr(self, a: Any, sh: Any) -> Any: return self.default_value()
    def bv_udiv(self, a: Any, b: Any) -> Any: return self.default_value()
    def bv_sdiv(self, a: Any, b: Any) -> Any: return self.default_value()
    def bv_urem(self, a: Any, b: Any) -> Any: return self.default_value()
    def bv_srem(self, a: Any, b: Any) -> Any: return self.default_value()
    def bv_eq(self, a: Any, b: Any) -> Any: return self.const_bool(False)
    def bv_ne(self, a: Any, b: Any) -> Any: return self.const_bool(False)
    def bv_ult(self, a: Any, b: Any) -> Any: return self.const_bool(False)
    def bv_ule(self, a: Any, b: Any) -> Any: return self.const_bool(False)
    def bv_ugt(self, a: Any, b: Any) -> Any: return self.const_bool(False)
    def bv_uge(self, a: Any, b: Any) -> Any: return self.const_bool(False)
    def bv_slt(self, a: Any, b: Any) -> Any: return self.const_bool(False)
    def bv_sle(self, a: Any, b: Any) -> Any: return self.const_bool(False)
    def bv_sgt(self, a: Any, b: Any) -> Any: return self.const_bool(False)
    def bv_sge(self, a: Any, b: Any) -> Any: return self.const_bool(False)
    def bv_extract(self, hi: int, lo: int, x: Any) -> Any: return self.default_value()
    def bv_zext(self, ext_bits: int, x: Any, src_bits: int) -> Any: return self.default_value()
    def bv_sext(self, ext_bits: int, x: Any, src_bits: int) -> Any: return self.default_value()
    def bv_concat(self, hi: Any, lo: Any) -> Any: return self.default_value()
    def bv_clz(self, x: Any) -> Any: return self.default_value()
    def bv_popcount(self, x: Any) -> Any: return self.default_value()
    def bv_bitrev(self, x: Any) -> Any: return self.default_value()

    def logical_and(self, a: Any, b: Any) -> Any: return self.const_bool(False)
    def logical_or(self, a: Any, b: Any) -> Any: return self.const_bool(False)
    def logical_not(self, a: Any) -> Any: return self.const_bool(False)
    def select(self, cond: Any, t_val: Any, f_val: Any) -> Any: return t_val

    def hw_uf(self, env: Any, name: str, *args: Any, out_bits: int = 32, fallback_tag: Optional[str] = None) -> Any:
        """Evaluate a domain-specific hardware uninterpreted function."""
        return self.default_value()

    def pack64(self, lo: Any, hi: Any) -> Any:
        blo, bhi = self.as_u32(lo), self.as_u32(hi)
        clo, chi = self.const_int_val(blo), self.const_int_val(bhi)
        if clo is not None and chi is not None:
            return self.const_u64(((chi & MASK32) << 32) | (clo & MASK32))
        return self.bv_concat(bhi, blo)

    def split64(self, val64: Any) -> Tuple[Any, Any]:
        v64 = self.as_u64(val64)
        c64 = self.const_int_val(v64)
        if c64 is not None:
            return (self.const_u32(c64 & MASK32), self.const_u32((c64 >> 32) & MASK32))
        return (self.bv_extract(31, 0, v64), self.bv_extract(63, 32, v64))

    def unpack_half2x16(self, u: Any) -> Tuple[Any, Any]:
        return (self.default_value(), self.default_value())

    def pack_half2x16(self, v2: Any) -> Any:
        return self.default_value()

    def int32_unary(self, op: str, x: Any) -> Any:
        bx = self.as_u32(x)
        if op == "BitNot": return self.bv_not(bx, 32)
        if op == "SNeg": return self.bv_neg(bx, 32)
        if op == "SAbs": return self.select(self.bv_slt(bx, self.const_u32(0)), self.bv_neg(bx, 32), bx)
        if op == "Clz": return self.bv_clz(bx)
        if op == "BitCount": return self.bv_popcount(bx)
        if op == "BitReverse": return self.bv_bitrev(bx)
        return self.default_value()

    def int32_binary(self, op: str, a: Any, b: Any) -> Any:
        ba, bb = self.as_u32(a), self.as_u32(b)
        if op == "IAdd": return self.bv_add(ba, bb, 32)
        if op == "ISub": return self.bv_sub(ba, bb, 32)
        if op == "IMul": return self.bv_mul(ba, bb, 32)
        if op == "BitAnd": return self.bv_and(ba, bb, 32)
        if op == "BitOr": return self.bv_or(ba, bb, 32)
        if op == "BitXor": return self.bv_xor(ba, bb, 32)
        if op == "Shl": return self.bv_shl(ba, self.bv_and(bb, self.const_u32(31), 32), 32)
        if op == "ShrU": return self.bv_lshr(ba, self.bv_and(bb, self.const_u32(31), 32), 32)
        if op == "ShrS": return self.bv_ashr(ba, self.bv_and(bb, self.const_u32(31), 32), 32)
        if op == "RotL": return self.bv_rotl(ba, self.bv_and(bb, self.const_u32(31), 32))
        if op == "RotR": return self.bv_rotr(ba, self.bv_and(bb, self.const_u32(31), 32))
        if op == "UDiv": return self.bv_udiv(ba, bb)
        if op == "SDiv": return self.bv_sdiv(ba, bb)
        if op == "UMod": return self.bv_urem(ba, bb)
        if op == "SRem": return self.bv_srem(ba, bb)
        if op == "SMod":
            rem = self.bv_srem(ba, bb)
            zero = self.const_u32(0)
            diff_sign = self.bv_slt(self.bv_xor(rem, bb, 32), zero)
            return self.select(self.logical_and(self.bv_ne(rem, zero), diff_sign), self.bv_add(rem, bb, 32), rem)
        return self.default_value()

    def bfe(self, signed: bool, x: Any, off: Any, width: Any) -> Any:
        """
        SPIR-V OpBitFieldUExtract / OpBitFieldSExtract semantics:
        extract `width` bits from `x` starting at `off & 31`, zero- or
        sign-extending to 32 bits (width == 0 -> 0; width >= 32 -> x).
        """
        bx, boff, bw = self.as_u32(x), self.as_u32(off), self.as_u32(width)
        co, cw = self.const_int_val(boff), self.const_int_val(bw)
        if co is not None and cw is not None:
            o_i = co & 31
            w_i = cw
            if w_i == 0:
                return self.const_u32(0)
            if w_i >= 32 and o_i == 0:
                return bx
            hi_i = min(31, o_i + w_i - 1)
            sl_bits = hi_i - o_i + 1
            sl = self.bv_extract(hi_i, o_i, bx)
            ext = 32 - sl_bits
            if ext == 0:
                return sl
            return self.bv_sext(ext, sl, sl_bits) if signed else self.bv_zext(ext, sl, sl_bits)

        zero = self.const_u32(0)
        c31 = self.const_u32(31)
        c32 = self.const_u32(32)
        if signed:
            sh = self.bv_sub(c32, bw, 32)
            extracted = self.bv_ashr(self.bv_shl(bx, self.bv_sub(sh, boff, 32), 32), sh, 32)
        else:
            mask = self.bv_sub(self.bv_shl(self.const_u32(1), self.bv_and(bw, c31, 32), 32), self.const_u32(1), 32)
            extracted = self.bv_and(self.bv_lshr(bx, self.bv_and(boff, c31, 32), 32), mask, 32)
        return self.select(self.bv_eq(bw, zero), zero, self.select(self.bv_uge(bw, c32), bx, extracted))

    def bfi(self, base: Any, insert: Any, off: Any, width: Any) -> Any:
        """
        SPIR-V OpBitFieldInsert semantics:
        replace `width` bits in `base` starting at `off & 31` with the low
        `width` bits of `insert` (width == 0 -> base; width >= 32 -> insert).
        """
        bb, bi, boff, bw = self.as_u32(base), self.as_u32(insert), self.as_u32(off), self.as_u32(width)
        co, cw = self.const_int_val(boff), self.const_int_val(bw)
        if co is not None and cw is not None:
            o_i = co & 31
            w_i = cw
            if w_i == 0:
                return bb
            if w_i >= 32:
                return bi
            mask_v = (((1 << w_i) - 1) << o_i) & MASK32
            mask_c = self.const_u32(mask_v)
            inv_c = self.const_u32((~mask_v) & MASK32)
            return self.bv_or(self.bv_and(bb, inv_c, 32), self.bv_and(self.bv_shl(bi, self.const_u32(o_i), 32), mask_c, 32), 32)

        zero = self.const_u32(0)
        c31 = self.const_u32(31)
        c32 = self.const_u32(32)
        sh_off = self.bv_and(boff, c31, 32)
        mask = self.bv_shl(self.bv_sub(self.bv_shl(self.const_u32(1), self.bv_and(bw, c31, 32), 32), self.const_u32(1), 32), sh_off, 32)
        res = self.bv_or(self.bv_and(bb, self.bv_not(mask, 32), 32), self.bv_and(self.bv_shl(bi, sh_off, 32), mask, 32), 32)
        return self.select(self.bv_eq(bw, zero), bb, self.select(self.bv_uge(bw, c32), bi, res))

    def idot4(self, op: str, a: Any, b: Any) -> Any:
        """4x8-bit packed integer dot product (IDot4S, IDot4SU, IDot4U)."""
        a32, b32 = self.as_u32(a), self.as_u32(b)
        acc = self.const_u32(0)
        for i in range(4):
            ba = self.bv_extract(8 * i + 7, 8 * i, a32)
            bb = self.bv_extract(8 * i + 7, 8 * i, b32)
            ea = self.bv_sext(24, ba, 8) if op in ("IDot4S", "IDot4SU") else self.bv_zext(24, ba, 8)
            eb = self.bv_sext(24, bb, 8) if op == "IDot4S" else self.bv_zext(24, bb, 8)
            acc = self.bv_add(acc, self.bv_mul(ea, eb, 32), 32)
        return acc

    def int64_unary(self, op: str, x: Any) -> Any:
        return self.bv_not(self.as_u64(x), 64)

    def int64_binary(self, op: str, a: Any, b: Any) -> Any:
        a64, b64 = self.as_u64(a), self.as_u64(b)
        if op == "IAdd64": return self.bv_add(a64, b64, 64)
        if op == "ISub64": return self.bv_sub(a64, b64, 64)
        if op == "IMul64": return self.bv_mul(a64, b64, 64)
        if op == "BitAnd64": return self.bv_and(a64, b64, 64)
        if op == "BitOr64": return self.bv_or(a64, b64, 64)
        if op == "BitXor64": return self.bv_xor(a64, b64, 64)
        if op == "Shl64": return self.bv_shl(a64, self.bv_and(b64, self.const_u64(63), 64), 64)
        if op == "ShrU64": return self.bv_lshr(a64, self.bv_and(b64, self.const_u64(63), 64), 64)
        if op == "ShrS64": return self.bv_ashr(a64, self.bv_and(b64, self.const_u64(63), 64), 64)
        return self.const_u64(0)

    def icmp(self, op: str, a: Any, b: Any) -> Any:
        ba, bb = self.as_u32(a), self.as_u32(b)
        if op == "ICmpEq": return self.bv_eq(ba, bb)
        if op == "ICmpNe": return self.bv_ne(ba, bb)
        if op == "ICmpSLt": return self.bv_slt(ba, bb)
        if op == "ICmpSLe": return self.bv_sle(ba, bb)
        if op == "ICmpSGt": return self.bv_sgt(ba, bb)
        if op == "ICmpSGe": return self.bv_sge(ba, bb)
        if op == "ICmpULt": return self.bv_ult(ba, bb)
        if op == "ICmpULe": return self.bv_ule(ba, bb)
        if op == "ICmpUGt": return self.bv_ugt(ba, bb)
        if op == "ICmpUGe": return self.bv_uge(ba, bb)
        return self.const_bool(False)

    def atomic_compute(self, env: Any, op: str, op_tag: str, old_val: Any, arg: Any, cmp_val: Any) -> Any:
        """Compute post-state value for AtomicExchange / AtomicCmpXchg / AtomicRmw."""
        if op == "AtomicExchange": return arg
        if op == "AtomicCmpXchg": return self.select(self.bv_eq(old_val, cmp_val), arg, old_val)
        if op_tag in ("aadd", "add"): return self.bv_add(old_val, arg, 32)
        if op_tag in ("asub", "sub"): return self.bv_sub(old_val, arg, 32)
        if op_tag in ("aand", "and"): return self.bv_and(old_val, arg, 32)
        if op_tag in ("aor", "or"): return self.bv_or(old_val, arg, 32)
        if op_tag in ("axor", "xor"): return self.bv_xor(old_val, arg, 32)
        if op_tag in ("asmin", "smin"): return self.select(self.bv_slt(old_val, arg), old_val, arg)
        if op_tag in ("aumin", "umin"): return self.select(self.bv_ult(old_val, arg), old_val, arg)
        if op_tag in ("asmax", "smax"): return self.select(self.bv_sgt(old_val, arg), old_val, arg)
        if op_tag in ("aumax", "umax"): return self.select(self.bv_ugt(old_val, arg), old_val, arg)
        return self.hw_uf(env, f"atomic_post_{op_tag}", old_val, arg)

    def fp_unary(self, op: str, x: Any) -> Any:
        """
        Default FP unary semantics expressed in terms of bitvector ops, `convert`,
        `fp_cmp`, `fp_binary`, and `select`. Subclasses override for IEEE-754
        edge-case precision (NaN/Inf/signed-zero/out-of-i32-range) or solver modes.
        """
        if op == "FAbs":
            return self.bv_and(self.as_u32(x), self.const_u32(0x7FFFFFFF), 32)
        if op == "FNeg":
            return self.bv_xor(self.as_u32(x), self.const_u32(0x80000000), 32)
        if op == "Trunc":
            return self.convert("ConvertSToF", self.convert("ConvertFToS", x))
        if op == "Floor":
            t = self.convert("ConvertSToF", self.convert("ConvertFToS", x))
            return self.select(self.fp_cmp("FCmpOrdLt", x, t), self.fp_binary("FSub", t, self.const_u32(_F32_ONE_BITS)), t)
        if op == "Ceil":
            t = self.convert("ConvertSToF", self.convert("ConvertFToS", x))
            return self.select(self.fp_cmp("FCmpOrdGt", x, t), self.fp_binary("FAdd", t, self.const_u32(_F32_ONE_BITS)), t)
        if op == "InverseSqrt":
            return self.fp_binary("FDiv", self.const_u32(_F32_ONE_BITS), self.fp_unary("Sqrt", x))
        if op == "Exp":
            return self.fp_unary("Exp2", self.fp_binary("FMul", x, self.const_u32(_F32_LOG2_E_BITS)))
        if op == "Log":
            return self.fp_binary("FMul", self.fp_unary("Log2", x), self.const_u32(_F32_LN_2_BITS))
        return self.hw_uf(None, f"fp_{op}", x)

    def fp_binary(self, op: str, a: Any, b: Any) -> Any:
        """
        Default FP binary semantics for derived ops (`FSub`, `FMin`/`NMin`,
        `FMax`/`NMax`, `Pow`) in terms of `fp_unary`, `fp_cmp`, `select`, and
        primitive `FAdd`/`FMul`/`FDiv`/`Exp2`/`Log2`.
        """
        if op == "FSub":
            return self.fp_binary("FAdd", a, self.fp_unary("FNeg", b))
        if op in ("FMin", "NMin"):
            return self.select(self.fp_cmp("FCmpOrdLe", a, b), a, b)
        if op in ("FMax", "NMax"):
            return self.select(self.fp_cmp("FCmpOrdGe", a, b), a, b)
        if op == "Pow":
            return self.fp_unary("Exp2", self.fp_binary("FMul", b, self.fp_unary("Log2", a)))
        return self.hw_uf(None, f"fp_{op}", a, b)

    def fp_clamp(self, x: Any, lo: Any, hi: Any) -> Any:
        return self.fp_binary("FMin", self.fp_binary("FMax", x, lo), hi)

    def fp_fma(self, a: Any, b: Any, c: Any) -> Any:
        return self.fp_binary("FAdd", self.fp_binary("FMul", a, b), c)

    def fp_is_nan(self, x: Any) -> Any: return self.const_bool(False)
    def fp_is_inf(self, x: Any) -> Any: return self.const_bool(False)
    def fp_cmp(self, op: str, a: Any, b: Any) -> Any: return self.const_bool(False)
    def convert(self, op: str, x: Any) -> Any: return self.default_value()
    def unknown_sideband_u64(self, env: Any, imm_u32: int) -> Any: return self.default_value()
    def sideband_u32(self, env: Any, imm_u32: int, imm_u32_2: int) -> Any: return self.default_value()
    def subgroup_ballot(self, env: Any, x: Any) -> Any: return self.default_value()
    def subgroup_shuffle(self, env: Any, x: Any, lane: Any) -> Any: return self.default_value()


ValueDomain = SsaDomain

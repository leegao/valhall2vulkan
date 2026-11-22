#!/usr/bin/env python3
"""
Unified Value-Domain Hierarchy for ValhallSsaModule Interpretation.

Isolates all shared denotational semantics (`int32_unary`, `int32_binary`,
`int64_unary`, `int64_binary`, `icmp`, `bfe`, `bfi`, `idot4`, `pack64`,
`split64`, `fp_clamp`, `atomic_compute`, `cast_value`) ONCE inside `SsaDomain`,
delegating only minimal domain primitives (`mask32`, `bv_*`, `ite`, `hw_uf`,
FP/conversion ops) to concrete, symbolic, and abstract-interpretation subclasses.
"""

from __future__ import annotations

import abc
import math
import struct
from typing import Any, Optional, Tuple

import z3

from tools.symbolic_ssa.env import (
    BV32,
    BV64,
    FP32_SORT,
    MASK32,
    MASK64,
    REAL_SORT,
    Z3Val,
    _as_bool,
    _as_bv32,
    _as_bv64,
    _c_bool,
    _c_u32,
    _c_u64,
    _det_hash_u32,
    _f16_to_f32_u32,
    _f32_to_f16_u16,
    _f32_to_u32,
    _s32,
    _s64,
    _u32_to_f32,
    _z3_and,
    _z3_if,
    _z3_not,
    _z3_or,
)
from tools.verify.panfrost_z3_model import (
    _bitrev_bv32,
    _clz_bv,
    _fp_to_int_sat,
    _fp_total_le_bv,
    _hw_uf,
    _popcount_bv32,
    bv32,
    bv32_to_fp32,
    bv64,
    f16_bv_to_f32_bv,
    f32_bv_to_f16_bv,
    fp32_to_bv32,
)


def _f32_bits_to_real_const(u32: int) -> Optional[z3.ArithRef]:
    try:
        f = struct.unpack("<f", struct.pack("<I", u32 & MASK32))[0]
    except Exception:
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    if f == 0.0:
        return z3.RealVal(0)
    num, den = f.as_integer_ratio()
    return z3.RealVal(num) / z3.RealVal(den) if den != 1 else z3.RealVal(num)


_REAL_ZERO = z3.RealVal(0)
_REAL_ONE = z3.RealVal(1)
_LOG2_E_REAL = _f32_bits_to_real_const(0x3FB8AA3B)
_LN_2_REAL = _f32_bits_to_real_const(0x3F317218)


def _is_real2bv(x: z3.ExprRef) -> bool:
    return z3.is_app(x) and x.decl() == _hw_uf("real2bv", REAL_SORT, BV32) and len(x.children()) == 1


def _bv32_to_real(x: z3.BitVecRef) -> z3.ArithRef:
    if _is_real2bv(x):
        return x.children()[0]
    if isinstance(x, z3.BitVecNumRef):
        rc = _f32_bits_to_real_const(x.as_long())
        if rc is not None:
            return rc
    return _hw_uf("bv2real", BV32, REAL_SORT)(x)


def _real_to_bv32(rx: z3.ArithRef) -> z3.BitVecRef:
    if z3.is_app(rx) and rx.decl() == _hw_uf("bv2real", BV32, REAL_SORT) and len(rx.children()) == 1:
        return rx.children()[0]
    return _hw_uf("real2bv", REAL_SORT, BV32)(rx)


def _real_sqrt(rx: z3.ArithRef) -> z3.ArithRef:
    if isinstance(rx, z3.RatNumRef) and rx.numerator_as_long() == 0:
        return _REAL_ZERO
    if isinstance(rx, z3.RatNumRef) and rx.numerator_as_long() == rx.denominator_as_long():
        return _REAL_ONE
    return _hw_uf("real_sqrt", REAL_SORT, REAL_SORT)(rx)


def _real_exp2(rx: z3.ArithRef) -> z3.ArithRef:
    if isinstance(rx, z3.RatNumRef) and rx.numerator_as_long() == 0:
        return _REAL_ONE
    if z3.is_app(rx) and rx.decl() == _hw_uf("real_log2", REAL_SORT, REAL_SORT) and len(rx.children()) == 1:
        return rx.children()[0]
    return _hw_uf("real_exp2", REAL_SORT, REAL_SORT)(rx)


def _real_log2(rx: z3.ArithRef) -> z3.ArithRef:
    if isinstance(rx, z3.RatNumRef) and rx.numerator_as_long() == rx.denominator_as_long():
        return _REAL_ZERO
    if z3.is_app(rx) and rx.decl() == _hw_uf("real_exp2", REAL_SORT, REAL_SORT) and len(rx.children()) == 1:
        return rx.children()[0]
    return _hw_uf("real_log2", REAL_SORT, REAL_SORT)(rx)


def _real_sin(rx: z3.ArithRef) -> z3.ArithRef:
    if isinstance(rx, z3.RatNumRef) and rx.numerator_as_long() == 0:
        return _REAL_ZERO
    return _hw_uf("real_sin", REAL_SORT, REAL_SORT)(rx)


def _real_cos(rx: z3.ArithRef) -> z3.ArithRef:
    if isinstance(rx, z3.RatNumRef) and rx.numerator_as_long() == 0:
        return _REAL_ONE
    return _hw_uf("real_cos", REAL_SORT, REAL_SORT)(rx)


class SsaDomain(abc.ABC):
    """
    Abstract value domain consumed by SsaInterpreter.

    Shared semantic definitions (`int32_unary`, `int32_binary`, `int64_unary`,
    `int64_binary`, `icmp`, `bfe`, `bfi`, `idot4`, `pack64`, `split64`,
    `fp_clamp`, `atomic_compute`, `cast_value`) are implemented once here using
    minimal domain primitives so concrete, symbolic, and abstract-interpretation
    domains cannot drift in semantics.
    """

    @abc.abstractmethod
    def const_u32(self, imm: int) -> Any: ...

    @abc.abstractmethod
    def const_u64(self, imm: int) -> Any: ...

    @abc.abstractmethod
    def const_bool(self, val: bool) -> Any: ...

    def default_value(self) -> Any:
        return self.const_u32(0)

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

    # Minimal bitvector/boolean primitives overridden by concrete & symbolic domains
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

    # Shared denotational semantics built from the domain primitives above
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

    def fp_unary(self, op: str, x: Any) -> Any: return self.default_value()
    def fp_binary(self, op: str, a: Any, b: Any) -> Any: return self.default_value()
    def fp_clamp(self, x: Any, lo: Any, hi: Any) -> Any: return self.fp_binary("FMin", self.fp_binary("FMax", x, lo), hi)
    def fp_fma(self, a: Any, b: Any, c: Any) -> Any: return self.default_value()
    def fp_is_nan(self, x: Any) -> Any: return self.const_bool(False)
    def fp_is_inf(self, x: Any) -> Any: return self.const_bool(False)
    def fp_cmp(self, op: str, a: Any, b: Any) -> Any: return self.const_bool(False)
    def convert(self, op: str, x: Any) -> Any: return self.default_value()
    def unknown_sideband_u64(self, env: Any, imm_u32: int) -> Any: return self.default_value()
    def sideband_u32(self, env: Any, imm_u32: int, imm_u32_2: int) -> Any: return self.default_value()
    def subgroup_ballot(self, env: Any, x: Any) -> Any: return self.default_value()
    def subgroup_shuffle(self, env: Any, x: Any, lane: Any) -> Any: return self.default_value()


class ConcreteDomain(SsaDomain):
    """Bit-exact Python concrete value domain (u32/u64/bool/f32 bits)."""

    def const_u32(self, imm: int) -> int: return imm & MASK32
    def const_u64(self, imm: int) -> int: return imm & MASK64
    def const_bool(self, val: bool) -> bool: return bool(val)
    def const_int_val(self, val: Any) -> Optional[int]:
        return val if isinstance(val, int) and not isinstance(val, bool) else None
    def mask32(self, val: Any) -> int: return int(val) & MASK32
    def mask64(self, val: Any) -> int: return int(val) & MASK64
    def as_u32(self, val: Any) -> int: return _c_u32(val)
    def as_u64(self, val: Any) -> int: return _c_u64(val)
    def as_f32(self, val: Any) -> float: return _u32_to_f32(_c_u32(val))
    def as_bool(self, val: Any) -> bool: return _c_bool(val)

    def _m(self, bits: int) -> int: return MASK64 if bits == 64 else MASK32
    def bv_add(self, a: Any, b: Any, bits: int = 32) -> int: return (a + b) & self._m(bits)
    def bv_sub(self, a: Any, b: Any, bits: int = 32) -> int: return (a - b) & self._m(bits)
    def bv_mul(self, a: Any, b: Any, bits: int = 32) -> int: return (a * b) & self._m(bits)
    def bv_and(self, a: Any, b: Any, bits: int = 32) -> int: return (a & b) & self._m(bits)
    def bv_or(self, a: Any, b: Any, bits: int = 32) -> int: return (a | b) & self._m(bits)
    def bv_xor(self, a: Any, b: Any, bits: int = 32) -> int: return (a ^ b) & self._m(bits)
    def bv_not(self, a: Any, bits: int = 32) -> int: return (~a) & self._m(bits)
    def bv_neg(self, a: Any, bits: int = 32) -> int: return (-a) & self._m(bits)
    def bv_shl(self, a: Any, sh: Any, bits: int = 32) -> int: return (a << sh) & self._m(bits)
    def bv_lshr(self, a: Any, sh: Any, bits: int = 32) -> int: return (a & self._m(bits)) >> sh
    def bv_ashr(self, a: Any, sh: Any, bits: int = 32) -> int:
        return ((_s64(a) if bits == 64 else _s32(a)) >> sh) & self._m(bits)
    def bv_rotl(self, a: Any, sh: Any) -> int: return ((a << sh) | (a >> ((32 - sh) & 31))) & MASK32
    def bv_rotr(self, a: Any, sh: Any) -> int: return ((a >> sh) | (a << ((32 - sh) & 31))) & MASK32
    def bv_udiv(self, a: Any, b: Any) -> int: return MASK32 if b == 0 else (a // b) & MASK32
    def bv_sdiv(self, a: Any, b: Any) -> int:
        sa, sb = _s32(a), _s32(b)
        if sb == 0: return MASK32
        if sa == -0x80000000 and sb == -1: return 0x80000000
        return int(sa / sb) & MASK32
    def bv_urem(self, a: Any, b: Any) -> int: return a if b == 0 else (a % b) & MASK32
    def bv_srem(self, a: Any, b: Any) -> int:
        sa, sb = _s32(a), _s32(b)
        if sb == 0: return sa & MASK32
        if sa == -0x80000000 and sb == -1: return 0
        return (sa - int(sa / sb) * sb) & MASK32
    def bv_eq(self, a: Any, b: Any) -> bool: return a == b
    def bv_ne(self, a: Any, b: Any) -> bool: return a != b
    def bv_ult(self, a: Any, b: Any) -> bool: return a < b
    def bv_ule(self, a: Any, b: Any) -> bool: return a <= b
    def bv_ugt(self, a: Any, b: Any) -> bool: return a > b
    def bv_uge(self, a: Any, b: Any) -> bool: return a >= b
    def bv_slt(self, a: Any, b: Any) -> bool: return _s32(a) < _s32(b)
    def bv_sle(self, a: Any, b: Any) -> bool: return _s32(a) <= _s32(b)
    def bv_sgt(self, a: Any, b: Any) -> bool: return _s32(a) > _s32(b)
    def bv_sge(self, a: Any, b: Any) -> bool: return _s32(a) >= _s32(b)
    def bv_extract(self, hi: int, lo: int, x: Any) -> int:
        return (int(x) >> lo) & ((1 << (hi - lo + 1)) - 1)
    def bv_zext(self, ext_bits: int, x: Any, src_bits: int) -> int:
        return int(x) & ((1 << src_bits) - 1)
    def bv_sext(self, ext_bits: int, x: Any, src_bits: int) -> int:
        mask = (1 << src_bits) - 1
        v = int(x) & mask
        if v & (1 << (src_bits - 1)):
            v |= (~mask) & ((1 << (src_bits + ext_bits)) - 1)
        return v
    def bv_concat(self, hi: Any, lo: Any) -> int:
        return ((self.as_u32(hi) << 32) | self.as_u32(lo)) & MASK64
    def bv_clz(self, x: Any) -> int: return 32 if x == 0 else (32 - x.bit_length())
    def bv_popcount(self, x: Any) -> int: return x.bit_count()
    def bv_bitrev(self, x: Any) -> int: return int(f"{x:032b}"[::-1], 2)

    def logical_and(self, a: Any, b: Any) -> bool: return self.as_bool(a) and self.as_bool(b)
    def logical_or(self, a: Any, b: Any) -> bool: return self.as_bool(a) or self.as_bool(b)
    def logical_not(self, a: Any) -> bool: return not self.as_bool(a)
    def select(self, cond: Any, t_val: Any, f_val: Any) -> Any: return t_val if self.as_bool(cond) else f_val

    def hw_uf(self, env: Any, name: str, *args: Any, out_bits: int = 32, fallback_tag: Optional[str] = None) -> int:
        cargs = [self.as_u32(a) for a in args]
        if env is not None and hasattr(env, "eval_hw_uf"):
            return env.eval_hw_uf(f"hw_{name}", *cargs, fallback_tag=fallback_tag or name)
        return _det_hash_u32(name, *cargs)

    def unpack_half2x16(self, u: Any) -> Tuple[int, int]:
        cu = self.as_u32(u)
        return (_f16_to_f32_u32(cu & 0xFFFF), _f16_to_f32_u32((cu >> 16) & 0xFFFF))

    def pack_half2x16(self, v2: Any) -> int:
        if isinstance(v2, tuple) and len(v2) >= 2:
            return ((_f32_to_f16_u16(self.as_u32(v2[1])) << 16) | _f32_to_f16_u16(self.as_u32(v2[0]))) & MASK32
        h = _f32_to_f16_u16(self.as_u32(v2))
        return ((h << 16) | h) & MASK32

    def fp_unary(self, op: str, x: Any) -> int:
        u = self.as_u32(x)
        if op == "FAbs": return u & 0x7FFFFFFF
        if op == "FNeg": return u ^ 0x80000000
        f = self.as_f32(u)
        if math.isnan(f): return 0x7FC00000
        if op == "Floor": return _f32_to_u32(math.floor(f)) if math.isfinite(f) else u
        if op == "Ceil": return _f32_to_u32(math.ceil(f)) if math.isfinite(f) else u
        if op == "Trunc": return _f32_to_u32(math.trunc(f)) if math.isfinite(f) else u
        if op == "RoundEven": return _f32_to_u32(float(round(f))) if math.isfinite(f) else u
        if op == "Sqrt": return 0x7FC00000 if f < 0.0 else _f32_to_u32(math.sqrt(f))
        if op == "InverseSqrt":
            if f < 0.0: return 0x7FC00000
            if f == 0.0: return 0x7F800000
            return _f32_to_u32(1.0 / math.sqrt(f))
        if op == "Sin": return _f32_to_u32(math.sin(f)) if math.isfinite(f) else 0x7FC00000
        if op == "Cos": return _f32_to_u32(math.cos(f)) if math.isfinite(f) else 0x7FC00000
        if op == "Exp":
            try: return _f32_to_u32(math.exp(f))
            except OverflowError: return 0x7F800000
        if op == "Exp2":
            try: return _f32_to_u32(math.pow(2.0, f))
            except OverflowError: return 0x7F800000
        if op == "Log":
            if f < 0.0: return 0x7FC00000
            if f == 0.0: return 0xFF800000
            return _f32_to_u32(math.log(f))
        if op == "Log2":
            if f < 0.0: return 0x7FC00000
            if f == 0.0: return 0xFF800000
            return _f32_to_u32(math.log2(f))
        return _det_hash_u32(f"fp_{op}", u)

    def fp_binary(self, op: str, a: Any, b: Any) -> int:
        ua, ub = self.as_u32(a), self.as_u32(b)
        fa, fb = self.as_f32(ua), self.as_f32(ub)
        if op == "FAdd": return _f32_to_u32(fa + fb)
        if op == "FSub": return _f32_to_u32(fa - fb)
        if op == "FMul": return _f32_to_u32(fa * fb)
        if op == "FDiv":
            if math.isnan(fa) or math.isnan(fb): return 0x7FC00000
            if fb == 0.0:
                if fa == 0.0: return 0x7FC00000
                return ((ua ^ ub) & 0x80000000) | 0x7F800000
            return _f32_to_u32(fa / fb)
        if op in ("FMin", "NMin"):
            if math.isnan(fa): return ub
            if math.isnan(fb): return ua
            return ua if fa <= fb else ub
        if op in ("FMax", "NMax"):
            if math.isnan(fa): return ub
            if math.isnan(fb): return ua
            return ua if fa >= fb else ub
        if op == "Pow":
            try: return _f32_to_u32(math.pow(fa, fb))
            except Exception: return 0x7FC00000
        return _det_hash_u32(f"fp_{op}", ua, ub)

    def fp_fma(self, a: Any, b: Any, c: Any) -> int:
        return _f32_to_u32(self.as_f32(a) * self.as_f32(b) + self.as_f32(c))

    def fp_is_nan(self, x: Any) -> bool: return math.isnan(self.as_f32(x))
    def fp_is_inf(self, x: Any) -> bool: return math.isinf(self.as_f32(x))

    def fp_cmp(self, op: str, a: Any, b: Any) -> bool:
        ua, ub = self.as_u32(a), self.as_u32(b)
        fa, fb = self.as_f32(ua), self.as_f32(ub)
        unord = math.isnan(fa) or math.isnan(fb)
        if op == "FCmpOrdEq": return (not unord) and (fa == fb)
        if op == "FCmpOrdGt": return (not unord) and (fa > fb)
        if op == "FCmpOrdGe": return (not unord) and (fa >= fb)
        if op == "FCmpOrdLt": return (not unord) and (fa < fb)
        if op == "FCmpOrdLe": return (not unord) and (fa <= fb)
        if op == "FCmpOrdNe": return (not unord) and (fa != fb)
        if op == "FCmpUnordNe": return unord or (fa != fb)
        if op == "FCmpTotal":
            def _key(u: int) -> int:
                return (0x80000000 - (u & 0x7FFFFFFF)) if (u & 0x80000000) else (0x80000000 + u)
            return _key(ua) <= _key(ub)
        return False

    def convert(self, op: str, x: Any) -> int:
        if op == "ConvertFToS":
            f = self.as_f32(x)
            if math.isnan(f): return 0
            if f >= 2147483647.0: return 0x7FFFFFFF
            if f <= -2147483648.0: return 0x80000000
            return int(math.trunc(f)) & MASK32
        if op == "ConvertFToU":
            f = self.as_f32(x)
            if math.isnan(f) or f <= 0.0: return 0
            if f >= 4294967295.0: return MASK32
            return int(math.trunc(f)) & MASK32
        if op == "ConvertSToF": return _f32_to_u32(float(_s32(self.as_u32(x))))
        if op == "ConvertUToF": return _f32_to_u32(float(self.as_u32(x)))
        if op == "SConvert32To64": return _s32(self.as_u32(x)) & MASK64
        if op == "UConvert32To64": return self.as_u32(x) & MASK64
        if op == "UConvert64To32": return self.as_u64(x) & MASK32
        return 0

    def unknown_sideband_u64(self, env: Any, imm_u32: int) -> int:
        return _det_hash_u32(f"sb64_{imm_u32}", env.seed, env.vec_idx)

    def sideband_u32(self, env: Any, imm_u32: int, imm_u32_2: int) -> int:
        return self.hw_uf(env, f"sideband_u32_{imm_u32}", imm_u32_2, fallback_tag=f"sb32_{imm_u32}_{imm_u32_2}")

    def subgroup_ballot(self, env: Any, x: Any) -> int:
        return _det_hash_u32("SubgroupBallot", env.seed, env.vec_idx, self.as_u32(x), env.lane_id)

    def subgroup_shuffle(self, env: Any, x: Any, lane: Any) -> int:
        return _det_hash_u32("SubgroupShuffle", env.seed, env.vec_idx, self.as_u32(x), self.as_u32(lane), env.lane_id)


class SymbolicDomain(SsaDomain):
    """Z3 symbolic value domain supporting `exact` (QF_FP), `real` (QF_NRA), and `uf` FP modes."""

    def __init__(self, fp_mode: str = "exact"):
        self.fp_mode = fp_mode

    def const_u32(self, imm: int) -> z3.BitVecRef: return bv32(imm)
    def const_u64(self, imm: int) -> z3.BitVecRef: return bv64(imm)
    def const_bool(self, val: bool) -> z3.BoolRef: return z3.BoolVal(val)
    def const_int_val(self, val: Any) -> Optional[int]:
        return val.as_long() if isinstance(val, z3.BitVecNumRef) else None
    def as_u32(self, val: Any) -> z3.BitVecRef: return _as_bv32(val)
    def as_u64(self, val: Any) -> z3.BitVecRef: return _as_bv64(val)
    def as_f32(self, val: Any) -> z3.FPRef: return bv32_to_fp32(_as_bv32(val))
    def as_bool(self, val: Any) -> z3.BoolRef: return _as_bool(val)

    def bv_add(self, a: Any, b: Any, bits: int = 32) -> z3.BitVecRef:
        if bits == 64 and isinstance(a, z3.BitVecNumRef) and isinstance(b, z3.BitVecNumRef):
            return bv64((a.as_long() + b.as_long()) & MASK64)
        return a + b
    def bv_sub(self, a: Any, b: Any, bits: int = 32) -> z3.BitVecRef:
        if bits == 64 and isinstance(a, z3.BitVecNumRef) and isinstance(b, z3.BitVecNumRef):
            return bv64((a.as_long() - b.as_long()) & MASK64)
        return a - b
    def bv_mul(self, a: Any, b: Any, bits: int = 32) -> z3.BitVecRef:
        if bits == 64 and isinstance(a, z3.BitVecNumRef) and isinstance(b, z3.BitVecNumRef):
            return bv64((a.as_long() * b.as_long()) & MASK64)
        return a * b
    def bv_and(self, a: Any, b: Any, bits: int = 32) -> z3.BitVecRef: return a & b
    def bv_or(self, a: Any, b: Any, bits: int = 32) -> z3.BitVecRef: return a | b
    def bv_xor(self, a: Any, b: Any, bits: int = 32) -> z3.BitVecRef: return a ^ b
    def bv_not(self, a: Any, bits: int = 32) -> z3.BitVecRef: return ~a
    def bv_neg(self, a: Any, bits: int = 32) -> z3.BitVecRef: return -a
    def bv_shl(self, a: Any, sh: Any, bits: int = 32) -> z3.BitVecRef: return a << sh
    def bv_lshr(self, a: Any, sh: Any, bits: int = 32) -> z3.BitVecRef: return z3.LShR(a, sh)
    def bv_ashr(self, a: Any, sh: Any, bits: int = 32) -> z3.BitVecRef: return a >> sh
    def bv_rotl(self, a: Any, sh: Any) -> z3.BitVecRef: return z3.RotateLeft(a, sh)
    def bv_rotr(self, a: Any, sh: Any) -> z3.BitVecRef: return z3.RotateRight(a, sh)
    def bv_udiv(self, a: Any, b: Any) -> z3.BitVecRef: return z3.UDiv(a, b)
    def bv_sdiv(self, a: Any, b: Any) -> z3.BitVecRef: return a / b
    def bv_urem(self, a: Any, b: Any) -> z3.BitVecRef: return z3.URem(a, b)
    def bv_srem(self, a: Any, b: Any) -> z3.BitVecRef: return z3.SRem(a, b)
    def bv_eq(self, a: Any, b: Any) -> z3.BoolRef: return a == b
    def bv_ne(self, a: Any, b: Any) -> z3.BoolRef: return a != b
    def bv_ult(self, a: Any, b: Any) -> z3.BoolRef: return z3.ULT(a, b)
    def bv_ule(self, a: Any, b: Any) -> z3.BoolRef: return z3.ULE(a, b)
    def bv_ugt(self, a: Any, b: Any) -> z3.BoolRef: return z3.UGT(a, b)
    def bv_uge(self, a: Any, b: Any) -> z3.BoolRef: return z3.UGE(a, b)
    def bv_slt(self, a: Any, b: Any) -> z3.BoolRef: return a < b
    def bv_sle(self, a: Any, b: Any) -> z3.BoolRef: return a <= b
    def bv_sgt(self, a: Any, b: Any) -> z3.BoolRef: return a > b
    def bv_sge(self, a: Any, b: Any) -> z3.BoolRef: return a >= b
    def bv_extract(self, hi: int, lo: int, x: Any) -> z3.BitVecRef: return z3.Extract(hi, lo, x)
    def bv_zext(self, ext_bits: int, x: Any, src_bits: int) -> z3.BitVecRef: return z3.ZeroExt(ext_bits, x)
    def bv_sext(self, ext_bits: int, x: Any, src_bits: int) -> z3.BitVecRef: return z3.SignExt(ext_bits, x)
    def bv_concat(self, hi: Any, lo: Any) -> z3.BitVecRef: return z3.Concat(hi, lo)
    def bv_clz(self, x: Any) -> z3.BitVecRef: return _clz_bv(32, x, mask_zero=False)
    def bv_popcount(self, x: Any) -> z3.BitVecRef: return _popcount_bv32(x)
    def bv_bitrev(self, x: Any) -> z3.BitVecRef: return _bitrev_bv32(x)

    def logical_and(self, a: Any, b: Any) -> z3.BoolRef: return _z3_and(self.as_bool(a), self.as_bool(b))
    def logical_or(self, a: Any, b: Any) -> z3.BoolRef: return _z3_or(self.as_bool(a), self.as_bool(b))
    def logical_not(self, a: Any) -> z3.BoolRef: return _z3_not(self.as_bool(a))
    def select(self, cond: Any, t_val: Any, f_val: Any) -> Z3Val: return _z3_if(self.as_bool(cond), t_val, f_val)

    def hw_uf(self, env: Any, name: str, *args: Any, out_bits: int = 32, fallback_tag: Optional[str] = None) -> z3.BitVecRef:
        bargs = [self.as_u32(a) for a in args]
        out_sort = BV64 if out_bits == 64 else BV32
        sorts = [BV32] * len(bargs) + [out_sort]
        return _hw_uf(name, *sorts)(*bargs)

    def unpack_half2x16(self, u: Any) -> Tuple[z3.BitVecRef, z3.BitVecRef]:
        bu = self.as_u32(u)
        return (f16_bv_to_f32_bv(z3.Extract(15, 0, bu)), f16_bv_to_f32_bv(z3.Extract(31, 16, bu)))

    def pack_half2x16(self, v2: Any) -> z3.BitVecRef:
        if isinstance(v2, tuple) and len(v2) >= 2:
            f0 = f32_bv_to_f16_bv(self.as_u32(v2[0]))
            f1 = f32_bv_to_f16_bv(self.as_u32(v2[1]))
            return z3.Concat(f1, f0)
        s = f32_bv_to_f16_bv(self.as_u32(v2))
        return z3.Concat(s, s)

    def fp_unary(self, op: str, x: Any) -> z3.BitVecRef:
        bx = self.as_u32(x)
        if self.fp_mode == "real":
            rx = _bv32_to_real(bx)
            if op == "FAbs": return _real_to_bv32(z3.If(rx < _REAL_ZERO, -rx, rx))
            if op == "FNeg": return _real_to_bv32(-rx)
            if op == "Sqrt": return _real_to_bv32(_real_sqrt(rx))
            if op == "InverseSqrt": return _real_to_bv32(_REAL_ONE / _real_sqrt(rx))
            if op == "Exp2": return _real_to_bv32(_real_exp2(rx))
            if op == "Exp": return _real_to_bv32(_real_exp2(rx * _LOG2_E_REAL))
            if op == "Log2": return _real_to_bv32(_real_log2(rx))
            if op == "Log": return _real_to_bv32(_real_log2(rx) * _LN_2_REAL)
            if op == "Sin": return _real_to_bv32(_real_sin(rx))
            if op == "Cos": return _real_to_bv32(_real_cos(rx))
            return _real_to_bv32(_hw_uf(f"real_{op}", REAL_SORT, REAL_SORT)(rx))
        if op == "FAbs": return bx & bv32(0x7FFFFFFF)
        if op == "FNeg": return bx ^ bv32(0x80000000)
        if self.fp_mode == "uf" or op in (
            "InverseSqrt", "Sin", "Cos", "Exp", "Exp2", "Log", "Log2",
            "Atan", "Asin", "Acos", "Sinh", "Cosh", "Tanh",
            "DPdx", "DPdy", "Fwidth", "FrexpExp", "FrexpSignificand",
        ):
            return _hw_uf(f"fp_{op}", BV32, BV32)(bx)
        f = self.as_f32(bx)
        if op == "Floor": return fp32_to_bv32(z3.fpRoundToIntegral(z3.RTN(), f))
        if op == "Ceil": return fp32_to_bv32(z3.fpRoundToIntegral(z3.RTP(), f))
        if op == "Trunc": return fp32_to_bv32(z3.fpRoundToIntegral(z3.RTZ(), f))
        if op == "RoundEven": return fp32_to_bv32(z3.fpRoundToIntegral(z3.RNE(), f))
        if op == "Sqrt": return fp32_to_bv32(z3.fpSqrt(z3.RNE(), f))
        return _hw_uf(f"fp_{op}", BV32, BV32)(bx)

    def fp_binary(self, op: str, a: Any, b: Any) -> z3.BitVecRef:
        ba, bb = self.as_u32(a), self.as_u32(b)
        if self.fp_mode == "real":
            ra, rb = _bv32_to_real(ba), _bv32_to_real(bb)
            if op == "FAdd": return _real_to_bv32(ra + rb)
            if op == "FSub": return _real_to_bv32(ra - rb)
            if op == "FMul": return _real_to_bv32(ra * rb)
            if op == "FDiv": return _real_to_bv32(ra / rb)
            if op in ("FMin", "NMin"): return _real_to_bv32(z3.If(ra <= rb, ra, rb))
            if op in ("FMax", "NMax"): return _real_to_bv32(z3.If(ra >= rb, ra, rb))
            if op == "Pow":
                if isinstance(rb, z3.RatNumRef):
                    num, den = rb.numerator_as_long(), rb.denominator_as_long()
                    if num == 0: return _real_to_bv32(_REAL_ONE)
                    if num == den: return _real_to_bv32(ra)
                    if num == 2 * den: return _real_to_bv32(ra * ra)
                    if 2 * num == den: return _real_to_bv32(_real_sqrt(ra))
                    if 2 * num == -den: return _real_to_bv32(_REAL_ONE / _real_sqrt(ra))
                    if num == -den: return _real_to_bv32(_REAL_ONE / ra)
                return _real_to_bv32(_real_exp2(rb * _real_log2(ra)))
            return _real_to_bv32(_hw_uf(f"real_{op}", REAL_SORT, REAL_SORT, REAL_SORT)(ra, rb))
        if self.fp_mode == "uf" or op in ("Pow", "Ldexp", "Atan2"):
            return _hw_uf(f"fp_{op}", BV32, BV32, BV32)(ba, bb)
        fa, fb = self.as_f32(ba), self.as_f32(bb)
        if op == "FAdd": return fp32_to_bv32(z3.fpAdd(z3.RNE(), fa, fb))
        if op == "FSub": return fp32_to_bv32(z3.fpSub(z3.RNE(), fa, fb))
        if op == "FMul": return fp32_to_bv32(z3.fpMul(z3.RNE(), fa, fb))
        if op == "FDiv": return fp32_to_bv32(z3.fpDiv(z3.RNE(), fa, fb))
        if op in ("FMin", "NMin"): return fp32_to_bv32(z3.fpMin(fa, fb))
        if op in ("FMax", "NMax"): return fp32_to_bv32(z3.fpMax(fa, fb))
        return _hw_uf(f"fp_{op}", BV32, BV32, BV32)(ba, bb)

    def fp_clamp(self, x: Any, lo: Any, hi: Any) -> z3.BitVecRef:
        if self.fp_mode == "uf":
            return _hw_uf("fp_FClamp", BV32, BV32, BV32, BV32)(self.as_u32(x), self.as_u32(lo), self.as_u32(hi))
        return super().fp_clamp(x, lo, hi)

    def fp_fma(self, a: Any, b: Any, c: Any) -> z3.BitVecRef:
        ba, bb, bc = self.as_u32(a), self.as_u32(b), self.as_u32(c)
        if self.fp_mode == "real":
            return _real_to_bv32(_bv32_to_real(ba) * _bv32_to_real(bb) + _bv32_to_real(bc))
        if self.fp_mode == "uf":
            return _hw_uf("fp_Fma", BV32, BV32, BV32, BV32)(ba, bb, bc)
        bc_is_pos_zero = isinstance(bc, z3.BitVecNumRef) and bc.as_long() == 0
        if bc_is_pos_zero:
            if isinstance(bb, z3.BitVecNumRef) and bb.as_long() == 0x3F800000:
                return z3.If(
                    z3.fpIsNaN(self.as_f32(ba)), fp32_to_bv32(z3.fpNaN(FP32_SORT)),
                    z3.If(ba == bv32(0x80000000), bv32(0), ba),
                )
            if isinstance(ba, z3.BitVecNumRef) and ba.as_long() == 0x3F800000:
                return z3.If(
                    z3.fpIsNaN(self.as_f32(bb)), fp32_to_bv32(z3.fpNaN(FP32_SORT)),
                    z3.If(bb == bv32(0x80000000), bv32(0), bb),
                )
        return fp32_to_bv32(z3.fpFMA(z3.RNE(), self.as_f32(ba), self.as_f32(bb), self.as_f32(bc)))

    def fp_is_nan(self, x: Any) -> z3.BoolRef:
        bx = self.as_u32(x)
        return z3.fpIsNaN(self.as_f32(bx)) if self.fp_mode == "exact" else _hw_uf("fp_IsNan", BV32, z3.BoolSort())(bx)

    def fp_is_inf(self, x: Any) -> z3.BoolRef:
        bx = self.as_u32(x)
        return z3.fpIsInf(self.as_f32(bx)) if self.fp_mode == "exact" else _hw_uf("fp_IsInf", BV32, z3.BoolSort())(bx)

    def fp_cmp(self, op: str, a: Any, b: Any) -> z3.BoolRef:
        ba, bb = self.as_u32(a), self.as_u32(b)
        if self.fp_mode == "real":
            ra, rb = _bv32_to_real(ba), _bv32_to_real(bb)
            if op == "FCmpOrdEq": return ra == rb
            if op == "FCmpOrdGt": return ra > rb
            if op == "FCmpOrdGe": return ra >= rb
            if op == "FCmpOrdLt": return ra < rb
            if op == "FCmpOrdLe": return ra <= rb
            if op in ("FCmpOrdNe", "FCmpUnordNe"): return ra != rb
            if op == "FCmpTotal": return ra <= rb
            return _hw_uf(f"realcmp_{op}", REAL_SORT, REAL_SORT, z3.BoolSort())(ra, rb)
        if self.fp_mode == "uf":
            return _hw_uf(f"fpcmp_{op}", BV32, BV32, z3.BoolSort())(ba, bb)
        fa, fb = self.as_f32(ba), self.as_f32(bb)
        if op == "FCmpOrdEq": return z3.fpEQ(fa, fb)
        if op == "FCmpOrdGt": return z3.fpGT(fa, fb)
        if op == "FCmpOrdGe": return z3.fpGEQ(fa, fb)
        if op == "FCmpOrdLt": return z3.fpLT(fa, fb)
        if op == "FCmpOrdLe": return z3.fpLEQ(fa, fb)
        if op == "FCmpOrdNe": return z3.Or(z3.fpLT(fa, fb), z3.fpGT(fa, fb))
        if op == "FCmpUnordNe": return z3.Not(z3.fpEQ(fa, fb))
        if op == "FCmpTotal": return _fp_total_le_bv(32, ba, bb)
        return _hw_uf(f"fpcmp_{op}", BV32, BV32, z3.BoolSort())(ba, bb)

    def convert(self, op: str, x: Any) -> z3.BitVecRef:
        if op == "ConvertFToS":
            bx = self.as_u32(x)
            if self.fp_mode == "real":
                return _hw_uf("real_FToS", REAL_SORT, BV32)(_bv32_to_real(bx))
            return _fp_to_int_sat(self.as_f32(bx), 32, True, z3.RTZ()) if self.fp_mode == "exact" else _hw_uf("fp_FToS", BV32, BV32)(bx)
        if op == "ConvertFToU":
            bx = self.as_u32(x)
            if self.fp_mode == "real":
                return _hw_uf("real_FToU", REAL_SORT, BV32)(_bv32_to_real(bx))
            return _fp_to_int_sat(self.as_f32(bx), 32, False, z3.RTZ()) if self.fp_mode == "exact" else _hw_uf("fp_FToU", BV32, BV32)(bx)
        if op == "ConvertSToF":
            bx = self.as_u32(x)
            if self.fp_mode == "real":
                if isinstance(bx, z3.BitVecNumRef):
                    return _real_to_bv32(z3.RealVal(bx.as_signed_long()))
                return _real_to_bv32(_hw_uf("real_SToF", BV32, REAL_SORT)(bx))
            return fp32_to_bv32(z3.fpSignedToFP(z3.RNE(), bx, FP32_SORT)) if self.fp_mode == "exact" else _hw_uf("fp_SToF", BV32, BV32)(bx)
        if op == "ConvertUToF":
            bx = self.as_u32(x)
            if self.fp_mode == "real":
                if isinstance(bx, z3.BitVecNumRef):
                    return _real_to_bv32(z3.RealVal(bx.as_long()))
                return _real_to_bv32(_hw_uf("real_UToF", BV32, REAL_SORT)(bx))
            return fp32_to_bv32(z3.fpUnsignedToFP(z3.RNE(), bx, FP32_SORT)) if self.fp_mode == "exact" else _hw_uf("fp_UToF", BV32, BV32)(bx)
        if op == "SConvert32To64": return z3.SignExt(32, self.as_u32(x))
        if op == "UConvert32To64": return z3.ZeroExt(32, self.as_u32(x))
        if op == "UConvert64To32": return z3.Extract(31, 0, self.as_u64(x))
        return bv32(0)

    def unknown_sideband_u64(self, env: Any, imm_u32: int) -> z3.BitVecRef:
        return self.hw_uf(env, f"sideband_u64_{imm_u32}", bv32(imm_u32), out_bits=64)

    def sideband_u32(self, env: Any, imm_u32: int, imm_u32_2: int) -> z3.BitVecRef:
        return self.hw_uf(env, f"sideband_u32_{imm_u32}", bv32(imm_u32_2))

    def subgroup_ballot(self, env: Any, x: Any) -> z3.BitVecRef:
        return _hw_uf("SubgroupBallot", BV32, z3.BitVecSort(5), BV32)(self.as_u32(x), env.lane_id)

    def subgroup_shuffle(self, env: Any, x: Any, lane: Any) -> z3.BitVecRef:
        return _hw_uf("SubgroupShuffle", BV32, BV32, z3.BitVecSort(5), BV32)(
            self.as_u32(x), self.as_u32(lane), env.lane_id
        )


class AbstractTopDomain(SsaDomain):
    """Lattice-Top domain returning `None` for all value operations."""

    def const_u32(self, imm: int) -> Any: return None
    def const_u64(self, imm: int) -> Any: return None
    def const_bool(self, val: bool) -> Any: return None
    def default_value(self) -> Any: return None
    def cast_value(self, op: str, val: Any, val_type: str) -> Any: return None
    def select(self, cond: Any, t_val: Any, f_val: Any) -> Any: return None
    def idot4(self, op: str, a: Any, b: Any) -> Any: return None
    def bfe(self, signed: bool, x: Any, off: Any, width: Any) -> Any: return None
    def bfi(self, base: Any, insert: Any, off: Any, width: Any) -> Any: return None

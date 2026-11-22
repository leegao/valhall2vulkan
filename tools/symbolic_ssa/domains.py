#!/usr/bin/env python3
"""
Value-Domain Abstractions for ValhallSsaModule Interpretation.

Provides:
  - SsaDomain: abstract value-domain interface defining type coercions
    (`as_u32`, `as_s32`, `as_u64`, `as_s64`, `as_f32`, `as_bool`), default
    `cast_value` and `fp_clamp`, and all pure scalar/vector arithmetic hooks.
  - ConcreteDomain(SsaDomain): bit-exact Python int/bool IEEE-754 + BV32/BV64
    domain with automatic u32 <-> float32 casting.
  - SymbolicDomain(SsaDomain): Z3 QF_BV + QF_FP / QF_NRA / UF domain
    parameterized by `fp_mode` (`exact`, `real`, `uf`), designed so relaxed or
    tagged-union symbolic domains can be swapped in without touching the
    interpreter.
  - AbstractTopDomain(SsaDomain): lattice-Top (`None`) domain for lightweight
    abstract-interpretation canonicalization passes.
"""

from __future__ import annotations

import abc
import math
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
    _LN_2_REAL,
    _LOG2_E_REAL,
    _REAL_ONE,
    _REAL_ZERO,
    _as_bool,
    _as_bv32,
    _as_bv64,
    _bv32_to_real,
    _c_bool,
    _c_u32,
    _c_u64,
    _det_hash_u32,
    _f16_to_f32_u32,
    _f32_to_f16_u16,
    _f32_to_u32,
    _real_cos,
    _real_exp2,
    _real_log2,
    _real_sin,
    _real_sqrt,
    _real_to_bv32,
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


class SsaDomain(abc.ABC):
    """Abstract value domain consumed by SsaInterpreter."""

    @abc.abstractmethod
    def const_u32(self, imm: int) -> Any: ...

    @abc.abstractmethod
    def const_u64(self, imm: int) -> Any: ...

    @abc.abstractmethod
    def const_bool(self, val: bool) -> Any: ...

    def default_value(self) -> Any:
        return self.const_u32(0)

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
        """Default type coercion shared across concrete and symbolic domains."""
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

    def pack64(self, lo: Any, hi: Any) -> Any:
        return self.default_value()

    def split64(self, val64: Any) -> Tuple[Any, Any]:
        return (self.default_value(), self.default_value())

    def unpack_half2x16(self, u: Any) -> Tuple[Any, Any]:
        return (self.default_value(), self.default_value())

    def pack_half2x16(self, v2: Any) -> Any:
        return self.default_value()

    def int32_unary(self, op: str, x: Any) -> Any:
        return self.default_value()

    def int32_binary(self, op: str, a: Any, b: Any) -> Any:
        return self.default_value()

    def bfe(self, signed: bool, x: Any, off: Any, width: Any) -> Any:
        """
        SPIR-V OpBitFieldUExtract / OpBitFieldSExtract semantics:
        extract `width` bits from `x` starting at bit `off & 31`, zero- or
        sign-extending to 32 bits (width == 0 yields 0; width >= 32 yields x).
        """
        return self.default_value()

    def bfi(self, base: Any, insert: Any, off: Any, width: Any) -> Any:
        """
        SPIR-V OpBitFieldInsert semantics:
        replace `width` bits in `base` starting at bit `off & 31` with the low
        `width` bits of `insert` (width == 0 yields base; width >= 32 yields insert).
        """
        return self.default_value()

    def idot4(self, op: str, a: Any, b: Any) -> Any:
        return self.default_value()

    def int64_unary(self, op: str, x: Any) -> Any:
        return self.default_value()

    def int64_binary(self, op: str, a: Any, b: Any) -> Any:
        return self.default_value()

    def icmp(self, op: str, a: Any, b: Any) -> Any:
        return self.const_bool(False)

    def logical_and(self, a: Any, b: Any) -> Any:
        return self.const_bool(False)

    def logical_or(self, a: Any, b: Any) -> Any:
        return self.const_bool(False)

    def logical_not(self, a: Any) -> Any:
        return self.const_bool(False)

    def select(self, cond: Any, t_val: Any, f_val: Any) -> Any:
        return t_val

    def fp_unary(self, op: str, x: Any) -> Any:
        return self.default_value()

    def fp_binary(self, op: str, a: Any, b: Any) -> Any:
        return self.default_value()

    def fp_clamp(self, x: Any, lo: Any, hi: Any) -> Any:
        """Default clamp defined via FMin(FMax(x, lo), hi)."""
        return self.fp_binary("FMin", self.fp_binary("FMax", x, lo), hi)

    def fp_fma(self, a: Any, b: Any, c: Any) -> Any:
        return self.default_value()

    def fp_is_nan(self, x: Any) -> Any:
        return self.const_bool(False)

    def fp_is_inf(self, x: Any) -> Any:
        return self.const_bool(False)

    def fp_cmp(self, op: str, a: Any, b: Any) -> Any:
        return self.const_bool(False)

    def convert(self, op: str, x: Any) -> Any:
        return self.default_value()

    def unknown_sideband_u64(self, env: Any, imm_u32: int) -> Any:
        return self.default_value()

    def sideband_u32(self, env: Any, imm_u32: int, imm_u32_2: int) -> Any:
        return self.default_value()

    def subgroup_ballot(self, env: Any, x: Any) -> Any:
        return self.default_value()

    def subgroup_shuffle(self, env: Any, x: Any, lane: Any) -> Any:
        return self.default_value()


class ConcreteDomain(SsaDomain):
    """Bit-exact Python concrete value domain (u32/u64/bool/f32 bits)."""

    def const_u32(self, imm: int) -> int:
        return imm & MASK32

    def const_u64(self, imm: int) -> int:
        return imm & MASK64

    def const_bool(self, val: bool) -> bool:
        return bool(val)

    def as_u32(self, val: Any) -> int:
        return _c_u32(val)

    def as_u64(self, val: Any) -> int:
        return _c_u64(val)

    def as_f32(self, val: Any) -> float:
        return _u32_to_f32(_c_u32(val))

    def as_bool(self, val: Any) -> bool:
        return _c_bool(val)

    def pack64(self, lo: Any, hi: Any) -> int:
        return ((self.as_u32(hi) << 32) | self.as_u32(lo)) & MASK64

    def split64(self, val64: Any) -> Tuple[int, int]:
        v64 = self.as_u64(val64)
        return (v64 & MASK32, (v64 >> 32) & MASK32)

    def unpack_half2x16(self, u: Any) -> Tuple[int, int]:
        cu = self.as_u32(u)
        return (_f16_to_f32_u32(cu & 0xFFFF), _f16_to_f32_u32((cu >> 16) & 0xFFFF))

    def pack_half2x16(self, v2: Any) -> int:
        if isinstance(v2, tuple) and len(v2) >= 2:
            return ((_f32_to_f16_u16(self.as_u32(v2[1])) << 16) | _f32_to_f16_u16(self.as_u32(v2[0]))) & MASK32
        h = _f32_to_f16_u16(self.as_u32(v2))
        return ((h << 16) | h) & MASK32

    def int32_unary(self, op: str, x: Any) -> int:
        cx = self.as_u32(x)
        if op == "BitNot": return (~cx) & MASK32
        if op == "SNeg": return (-cx) & MASK32
        if op == "SAbs":
            sx = _s32(cx)
            return (-sx if sx < 0 else sx) & MASK32
        if op == "Clz": return 32 if cx == 0 else (32 - cx.bit_length())
        if op == "BitCount": return cx.bit_count()
        if op == "BitReverse": return int(f"{cx:032b}"[::-1], 2)
        return 0

    def int32_binary(self, op: str, a: Any, b: Any) -> int:
        ca, cb = self.as_u32(a), self.as_u32(b)
        if op == "IAdd": return (ca + cb) & MASK32
        if op == "ISub": return (ca - cb) & MASK32
        if op == "IMul": return (ca * cb) & MASK32
        if op == "BitAnd": return (ca & cb) & MASK32
        if op == "BitOr": return (ca | cb) & MASK32
        if op == "BitXor": return (ca ^ cb) & MASK32
        if op == "Shl": return (ca << (cb & 31)) & MASK32
        if op == "ShrU": return (ca >> (cb & 31)) & MASK32
        if op == "ShrS": return (_s32(ca) >> (cb & 31)) & MASK32
        if op == "RotL":
            sh = cb & 31
            return ((ca << sh) | (ca >> ((32 - sh) & 31))) & MASK32
        if op == "RotR":
            sh = cb & 31
            return ((ca >> sh) | (ca << ((32 - sh) & 31))) & MASK32
        if op == "UDiv": return MASK32 if cb == 0 else (ca // cb) & MASK32
        if op == "SDiv":
            sa, sb = _s32(ca), _s32(cb)
            if sb == 0: return MASK32
            if sa == -0x80000000 and sb == -1: return 0x80000000
            return int(sa / sb) & MASK32
        if op == "UMod": return ca if cb == 0 else (ca % cb) & MASK32
        if op == "SRem":
            sa, sb = _s32(ca), _s32(cb)
            if sb == 0: return sa & MASK32
            if sa == -0x80000000 and sb == -1: return 0
            return (sa - int(sa / sb) * sb) & MASK32
        if op == "SMod":
            sa, sb = _s32(ca), _s32(cb)
            if sb == 0: return sa & MASK32
            if sa == -0x80000000 and sb == -1: return 0
            r = sa - int(sa / sb) * sb
            if r != 0 and (r < 0) != (sb < 0):
                r += sb
            return r & MASK32
        return 0

    def bfe(self, signed: bool, x: Any, off: Any, width: Any) -> int:
        cx, coff, cw = self.as_u32(x), self.as_u32(off) & 31, self.as_u32(width)
        if cw == 0: return 0
        if cw >= 32: return cx
        mask = (1 << cw) - 1
        ex = (cx >> coff) & mask
        if signed and (ex & (1 << (cw - 1))):
            ex |= (~mask) & MASK32
        return ex & MASK32

    def bfi(self, base: Any, insert: Any, off: Any, width: Any) -> int:
        cb, ci, coff, cw = self.as_u32(base), self.as_u32(insert), self.as_u32(off) & 31, self.as_u32(width)
        if cw == 0: return cb
        if cw >= 32: return ci
        mask = (((1 << cw) - 1) << coff) & MASK32
        return ((cb & ~mask) | ((ci << coff) & mask)) & MASK32

    def idot4(self, op: str, a: Any, b: Any) -> int:
        a32, b32 = self.as_u32(a), self.as_u32(b)
        acc = 0
        for i in range(4):
            ba = (a32 >> (8 * i)) & 0xFF
            bb = (b32 >> (8 * i)) & 0xFF
            if op in ("IDot4S", "IDot4SU") and (ba & 0x80):
                ba -= 256
            if op == "IDot4S" and (bb & 0x80):
                bb -= 256
            acc += ba * bb
        return acc & MASK32

    def int64_unary(self, op: str, x: Any) -> int:
        return (~self.as_u64(x)) & MASK64

    def int64_binary(self, op: str, a: Any, b: Any) -> int:
        ca, cb = self.as_u64(a), self.as_u64(b)
        if op == "IAdd64": return (ca + cb) & MASK64
        if op == "ISub64": return (ca - cb) & MASK64
        if op == "IMul64": return (ca * cb) & MASK64
        if op == "BitAnd64": return (ca & cb) & MASK64
        if op == "BitOr64": return (ca | cb) & MASK64
        if op == "BitXor64": return (ca ^ cb) & MASK64
        if op == "Shl64": return (ca << (cb & 63)) & MASK64
        if op == "ShrU64": return (ca >> (cb & 63)) & MASK64
        if op == "ShrS64": return (_s64(ca) >> (cb & 63)) & MASK64
        return 0

    def icmp(self, op: str, a: Any, b: Any) -> bool:
        ca, cb = self.as_u32(a), self.as_u32(b)
        if op == "ICmpEq": return ca == cb
        if op == "ICmpNe": return ca != cb
        if op == "ICmpSLt": return _s32(ca) < _s32(cb)
        if op == "ICmpSLe": return _s32(ca) <= _s32(cb)
        if op == "ICmpSGt": return _s32(ca) > _s32(cb)
        if op == "ICmpSGe": return _s32(ca) >= _s32(cb)
        if op == "ICmpULt": return ca < cb
        if op == "ICmpULe": return ca <= cb
        if op == "ICmpUGt": return ca > cb
        if op == "ICmpUGe": return ca >= cb
        return False

    def logical_and(self, a: Any, b: Any) -> bool:
        return self.as_bool(a) and self.as_bool(b)

    def logical_or(self, a: Any, b: Any) -> bool:
        return self.as_bool(a) or self.as_bool(b)

    def logical_not(self, a: Any) -> bool:
        return not self.as_bool(a)

    def select(self, cond: Any, t_val: Any, f_val: Any) -> Any:
        return t_val if self.as_bool(cond) else f_val

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

    def fp_is_nan(self, x: Any) -> bool:
        return math.isnan(self.as_f32(x))

    def fp_is_inf(self, x: Any) -> bool:
        return math.isinf(self.as_f32(x))

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
        return env.eval_hw_uf(f"hw_sideband_u32_{imm_u32}", imm_u32_2, fallback_tag=f"sb32_{imm_u32}_{imm_u32_2}")

    def subgroup_ballot(self, env: Any, x: Any) -> int:
        return _det_hash_u32("SubgroupBallot", env.seed, env.vec_idx, self.as_u32(x), env.lane_id)

    def subgroup_shuffle(self, env: Any, x: Any, lane: Any) -> int:
        return _det_hash_u32("SubgroupShuffle", env.seed, env.vec_idx, self.as_u32(x), self.as_u32(lane), env.lane_id)


class SymbolicDomain(SsaDomain):
    """Z3 symbolic value domain supporting `exact` (QF_FP), `real` (QF_NRA), and `uf` FP modes."""

    def __init__(self, fp_mode: str = "exact"):
        self.fp_mode = fp_mode

    def const_u32(self, imm: int) -> z3.BitVecRef:
        return bv32(imm)

    def const_u64(self, imm: int) -> z3.BitVecRef:
        return bv64(imm)

    def const_bool(self, val: bool) -> z3.BoolRef:
        return z3.BoolVal(val)

    def as_u32(self, val: Any) -> z3.BitVecRef:
        return _as_bv32(val)

    def as_u64(self, val: Any) -> z3.BitVecRef:
        return _as_bv64(val)

    def as_f32(self, val: Any) -> z3.FPRef:
        return bv32_to_fp32(_as_bv32(val))

    def as_bool(self, val: Any) -> z3.BoolRef:
        return _as_bool(val)

    def pack64(self, lo: Any, hi: Any) -> z3.BitVecRef:
        blo, bhi = self.as_u32(lo), self.as_u32(hi)
        if isinstance(blo, z3.BitVecNumRef) and isinstance(bhi, z3.BitVecNumRef):
            return bv64((bhi.as_long() << 32) | blo.as_long())
        return z3.Concat(bhi, blo)

    def split64(self, val64: Any) -> Tuple[z3.BitVecRef, z3.BitVecRef]:
        v64 = self.as_u64(val64)
        if isinstance(v64, z3.BitVecNumRef):
            n = v64.as_long()
            return (bv32(n & MASK32), bv32((n >> 32) & MASK32))
        return (z3.Extract(31, 0, v64), z3.Extract(63, 32, v64))

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

    def int32_unary(self, op: str, x: Any) -> z3.BitVecRef:
        bx = self.as_u32(x)
        if op == "BitNot": return ~bx
        if op == "SNeg": return -bx
        if op == "SAbs": return z3.If(bx < bv32(0), -bx, bx)
        if op == "Clz": return _clz_bv(32, bx, mask_zero=False)
        if op == "BitCount": return _popcount_bv32(bx)
        if op == "BitReverse": return _bitrev_bv32(bx)
        return bv32(0)

    def int32_binary(self, op: str, a: Any, b: Any) -> z3.BitVecRef:
        ba, bb = self.as_u32(a), self.as_u32(b)
        if op == "IAdd": return ba + bb
        if op == "ISub": return ba - bb
        if op == "IMul": return ba * bb
        if op == "BitAnd": return ba & bb
        if op == "BitOr": return ba | bb
        if op == "BitXor": return ba ^ bb
        if op == "Shl": return ba << (bb & bv32(31))
        if op == "ShrU": return z3.LShR(ba, bb & bv32(31))
        if op == "ShrS": return ba >> (bb & bv32(31))
        if op == "RotL": return z3.RotateLeft(ba, bb & bv32(31))
        if op == "RotR": return z3.RotateRight(ba, bb & bv32(31))
        if op == "UDiv": return z3.UDiv(ba, bb)
        if op == "SDiv": return ba / bb
        if op == "UMod": return z3.URem(ba, bb)
        if op == "SRem": return z3.SRem(ba, bb)
        if op == "SMod":
            rem = z3.SRem(ba, bb)
            return z3.If(z3.And(rem != bv32(0), (rem ^ bb) < bv32(0)), rem + bb, rem)
        return bv32(0)

    def bfe(self, signed: bool, x: Any, off: Any, width: Any) -> z3.BitVecRef:
        bx, boff, bw = self.as_u32(x), self.as_u32(off), self.as_u32(width)
        if isinstance(boff, z3.BitVecNumRef) and isinstance(bw, z3.BitVecNumRef):
            o_i, w_i = boff.as_long(), bw.as_long()
            if w_i == 0:
                return bv32(0)
            hi_i = min(31, o_i + w_i - 1)
            sl = z3.Extract(hi_i, o_i, bx)
            ext = 32 - sl.size()
            if ext == 0:
                return sl
            return z3.SignExt(ext, sl) if signed else z3.ZeroExt(ext, sl)
        if signed:
            sh = bv32(32) - bw
            extracted = (bx << (sh - boff)) >> sh
            return z3.If(bw == bv32(0), bv32(0), z3.If(z3.UGE(bw, bv32(32)), bx, extracted))
        mask = (bv32(1) << (bw & bv32(31))) - bv32(1)
        extracted = z3.LShR(bx, boff & bv32(31)) & mask
        return z3.If(bw == bv32(0), bv32(0), z3.If(z3.UGE(bw, bv32(32)), bx, extracted))

    def bfi(self, base: Any, insert: Any, off: Any, width: Any) -> z3.BitVecRef:
        bb, bi, boff, bw = self.as_u32(base), self.as_u32(insert), self.as_u32(off), self.as_u32(width)
        if isinstance(boff, z3.BitVecNumRef) and isinstance(bw, z3.BitVecNumRef):
            o_i, w_i = boff.as_long(), bw.as_long()
            if w_i >= 32: return bi
            if w_i == 0: return bb
            mask_v = ((1 << w_i) - 1) << o_i
            return (bb & bv32(~mask_v)) | ((bi << o_i) & bv32(mask_v))
        mask = ((bv32(1) << (bw & bv32(31))) - bv32(1)) << (boff & bv32(31))
        res_bfi = (bb & ~mask) | ((bi << (boff & bv32(31))) & mask)
        return z3.If(bw == bv32(0), bb, z3.If(z3.UGE(bw, bv32(32)), bi, res_bfi))

    def idot4(self, op: str, a: Any, b: Any) -> z3.BitVecRef:
        a32, b32 = self.as_u32(a), self.as_u32(b)
        acc = bv32(0)
        for i in range(4):
            ba = z3.Extract(8 * i + 7, 8 * i, a32)
            bb = z3.Extract(8 * i + 7, 8 * i, b32)
            ea = z3.SignExt(24, ba) if op in ("IDot4S", "IDot4SU") else z3.ZeroExt(24, ba)
            eb = z3.SignExt(24, bb) if op == "IDot4S" else z3.ZeroExt(24, bb)
            acc = acc + (ea * eb)
        return acc

    def int64_unary(self, op: str, x: Any) -> z3.BitVecRef:
        return ~self.as_u64(x)

    def int64_binary(self, op: str, a: Any, b: Any) -> z3.BitVecRef:
        a64, b64 = self.as_u64(a), self.as_u64(b)
        if isinstance(a64, z3.BitVecNumRef) and isinstance(b64, z3.BitVecNumRef):
            av, bv_ = a64.as_long(), b64.as_long()
            if op == "IAdd64": return bv64((av + bv_) & MASK64)
            if op == "ISub64": return bv64((av - bv_) & MASK64)
            if op == "IMul64": return bv64((av * bv_) & MASK64)
        if op == "IAdd64": return a64 + b64
        if op == "ISub64": return a64 - b64
        if op == "IMul64": return a64 * b64
        if op == "BitAnd64": return a64 & b64
        if op == "BitOr64": return a64 | b64
        if op == "BitXor64": return a64 ^ b64
        if op == "Shl64": return a64 << (b64 & bv64(63))
        if op == "ShrU64": return z3.LShR(a64, b64 & bv64(63))
        if op == "ShrS64": return a64 >> (b64 & bv64(63))
        return bv64(0)

    def icmp(self, op: str, a: Any, b: Any) -> z3.BoolRef:
        ba, bb = self.as_u32(a), self.as_u32(b)
        if op == "ICmpEq": return ba == bb
        if op == "ICmpNe": return ba != bb
        if op == "ICmpSLt": return ba < bb
        if op == "ICmpSLe": return ba <= bb
        if op == "ICmpSGt": return ba > bb
        if op == "ICmpSGe": return ba >= bb
        if op == "ICmpULt": return z3.ULT(ba, bb)
        if op == "ICmpULe": return z3.ULE(ba, bb)
        if op == "ICmpUGt": return z3.UGT(ba, bb)
        if op == "ICmpUGe": return z3.UGE(ba, bb)
        return z3.BoolVal(False)

    def logical_and(self, a: Any, b: Any) -> z3.BoolRef:
        return _z3_and(self.as_bool(a), self.as_bool(b))

    def logical_or(self, a: Any, b: Any) -> z3.BoolRef:
        return _z3_or(self.as_bool(a), self.as_bool(b))

    def logical_not(self, a: Any) -> z3.BoolRef:
        return _z3_not(self.as_bool(a))

    def select(self, cond: Any, t_val: Any, f_val: Any) -> Z3Val:
        return _z3_if(self.as_bool(cond), t_val, f_val)

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
        if op == "FAbs":
            return bx & bv32(0x7FFFFFFF)
        if op == "FNeg":
            return bx ^ bv32(0x80000000)
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
        if op == "SConvert32To64":
            return z3.SignExt(32, self.as_u32(x))
        if op == "UConvert32To64":
            return z3.ZeroExt(32, self.as_u32(x))
        if op == "UConvert64To32":
            return z3.Extract(31, 0, self.as_u64(x))
        return bv32(0)

    def unknown_sideband_u64(self, env: Any, imm_u32: int) -> z3.BitVecRef:
        return _hw_uf(f"sideband_u64_{imm_u32}", BV32, BV64)(bv32(imm_u32))

    def sideband_u32(self, env: Any, imm_u32: int, imm_u32_2: int) -> z3.BitVecRef:
        return _hw_uf(f"sideband_u32_{imm_u32}", BV32, BV32)(bv32(imm_u32_2))

    def subgroup_ballot(self, env: Any, x: Any) -> z3.BitVecRef:
        return _hw_uf("SubgroupBallot", BV32, z3.BitVecSort(5), BV32)(self.as_u32(x), env.lane_id)

    def subgroup_shuffle(self, env: Any, x: Any, lane: Any) -> z3.BitVecRef:
        return _hw_uf("SubgroupShuffle", BV32, BV32, z3.BitVecSort(5), BV32)(
            self.as_u32(x), self.as_u32(lane), env.lane_id
        )


class AbstractTopDomain(SsaDomain):
    """Lattice-Top domain returning `None` for all value operations."""

    def const_u32(self, imm: int) -> Any:
        return None

    def const_u64(self, imm: int) -> Any:
        return None

    def const_bool(self, val: bool) -> Any:
        return None

    def default_value(self) -> Any:
        return None

    def cast_value(self, op: str, val: Any, val_type: str) -> Any:
        return None

    def select(self, cond: Any, t_val: Any, f_val: Any) -> Any:
        return None

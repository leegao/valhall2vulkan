#!/usr/bin/env python3
"""Bit-exact Python concrete value domain (`ConcreteBitvecDomain`)."""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

from tools.symbolic_ssa.domains.base import SsaDomain
from tools.symbolic_ssa.env import (
    MASK32,
    MASK64,
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
)


class ConcreteBitvecDomain(SsaDomain):
    """Bit-exact Python concrete bitvector value domain (u32/u64/bool/f32 bits)."""

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

"""Z3 symbolic bitvector value domain (`SymbolicZ3BitvecDomain`)."""

from __future__ import annotations

import math
import struct
from typing import Any, TypeGuard

from tools.symbolic_ssa.domains.z3_base import Z3SymbolicDomain
from tools.symbolic_ssa.z3utils import dsl
from tools.symbolic_ssa.z3utils.helpers import (
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
    _dsl_bv32,
    _dsl_bv64,
    _dsl_hw_uf,
    _z3_and,
    _z3_if,
    _z3_not,
    _z3_or,
    bv32_to_fp32,
    f16_bv_to_f32_bv,
    f32_bv_to_f16_bv,
    fp32_to_bv32,
)


def _dsl_clz_bv32(x: Any) -> dsl.BitVecRef:
    res: dsl.BitVecRef = _dsl_bv32(32)
    one1 = dsl.BitVecVal(1, 1)
    for idx in range(32):
        bit = dsl.Extract(idx, idx, x)
        res = dsl.If(bit == one1, _dsl_bv32(31 - idx), res)
    return res


def _dsl_popcount_bv32(x: Any) -> dsl.BitVecRef:
    acc: dsl.BitVecRef = _dsl_bv32(0)
    for i in range(32):
        acc = acc + dsl.ZeroExt(31, dsl.Extract(i, i, x))
    return acc


def _dsl_bitrev_bv32(x: Any) -> dsl.BitVecRef:
    return dsl.Concat(*[dsl.Extract(i, i, x) for i in range(32)])


def _dsl_fp_total_le_bv(bits: int, a: Any, b: Any) -> dsl.BoolRef:
    mask = dsl.BitVecVal((1 << (bits - 1)) - 1, bits)
    zero = dsl.BitVecVal(0, bits)
    ka = dsl.If(a < zero, a ^ mask, a)
    kb = dsl.If(b < zero, b ^ mask, b)
    return ka <= kb


def _dsl_fp_to_int_sat(f: Any, bits: int, signed: bool, rm: Any) -> dsl.BitVecRef:
    zero_bv = dsl.BitVecVal(0, bits)
    if signed:
        min_i = -(1 << (bits - 1))
        max_i = (1 << (bits - 1)) - 1
        min_bv = dsl.BitVecVal(min_i, bits)
        max_bv = dsl.BitVecVal(max_i, bits)
        f_min = dsl.FPVal(float(min_i), FP32_SORT)
        f_max = dsl.FPVal(float(max_i), FP32_SORT)
        conv = dsl.fpToSBV(rm, f, dsl.BitVecSort(bits))
        return dsl.If(
            dsl.fpIsNaN(f),
            zero_bv,
            dsl.If(dsl.fpLEQ(f, f_min), min_bv, dsl.If(dsl.fpGEQ(f, f_max), max_bv, conv)),
        )
    max_u = (1 << bits) - 1
    max_bv = dsl.BitVecVal(max_u, bits)
    f_zero = dsl.FPVal(0.0, FP32_SORT)
    f_max = dsl.FPVal(float(max_u), FP32_SORT)
    conv = dsl.fpToUBV(rm, f, dsl.BitVecSort(bits))
    return dsl.If(
        dsl.Or(dsl.fpIsNaN(f), dsl.fpLEQ(f, f_zero)),
        zero_bv,
        dsl.If(dsl.fpGEQ(f, f_max), max_bv, conv),
    )


def _f32_bits_to_real_const(u32: int) -> dsl.ArithRef | None:
    try:
        f = struct.unpack("<f", struct.pack("<I", u32 & MASK32))[0]
    except struct.error:
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    if f == 0.0:
        return dsl.RealVal(0)
    num, den = f.as_integer_ratio()
    return dsl.Q(num, den) if den != 1 else dsl.RealVal(num)


_REAL_ZERO = dsl.RealVal(0)
_REAL_ONE = dsl.RealVal(1)
_LOG2_E_REAL = _f32_bits_to_real_const(0x3FB8AA3B)
_LN_2_REAL = _f32_bits_to_real_const(0x3F317218)


def _is_real2bv(x: Any) -> TypeGuard[dsl.Z3OpUFApp]:
    return isinstance(x, dsl.Z3OpUFApp) and x.func.fn_name == "hw_real2bv" and len(x.args) == 1


def _bv32_to_real(x: dsl.BitVecRef) -> dsl.ArithRef:
    if _is_real2bv(x):
        return x.args[0]
    if dsl.is_bv_value(x):
        rc = _f32_bits_to_real_const(dsl.as_long(x))
        if rc is not None:
            return rc
    return _dsl_hw_uf("bv2real", BV32, REAL_SORT)(x)


def _real_to_bv32(rx: dsl.ArithRef) -> dsl.BitVecRef:
    if isinstance(rx, dsl.Z3OpUFApp) and rx.func.fn_name == "hw_bv2real" and len(rx.args) == 1:
        return rx.args[0]
    return _dsl_hw_uf("real2bv", REAL_SORT, BV32)(rx)


def _real_sqrt(rx: dsl.ArithRef) -> dsl.ArithRef:
    if dsl.is_rational_value(rx) and dsl.numerator_as_long(rx) == 0:
        return _REAL_ZERO
    if dsl.is_rational_value(rx) and dsl.numerator_as_long(rx) == dsl.denominator_as_long(rx):
        return _REAL_ONE
    return _dsl_hw_uf("real_sqrt", REAL_SORT, REAL_SORT)(rx)


def _real_exp2(rx: dsl.ArithRef) -> dsl.ArithRef:
    if dsl.is_rational_value(rx) and dsl.numerator_as_long(rx) == 0:
        return _REAL_ONE
    if isinstance(rx, dsl.Z3OpUFApp) and rx.func.fn_name == "hw_real_log2" and len(rx.args) == 1:
        return rx.args[0]
    return _dsl_hw_uf("real_exp2", REAL_SORT, REAL_SORT)(rx)


def _real_log2(rx: dsl.ArithRef) -> dsl.ArithRef:
    if dsl.is_rational_value(rx) and dsl.numerator_as_long(rx) == dsl.denominator_as_long(rx):
        return _REAL_ZERO
    if isinstance(rx, dsl.Z3OpUFApp) and rx.func.fn_name == "hw_real_exp2" and len(rx.args) == 1:
        return rx.args[0]
    return _dsl_hw_uf("real_log2", REAL_SORT, REAL_SORT)(rx)


def _real_sin(rx: dsl.ArithRef) -> dsl.ArithRef:
    if dsl.is_rational_value(rx) and dsl.numerator_as_long(rx) == 0:
        return _REAL_ZERO
    return _dsl_hw_uf("real_sin", REAL_SORT, REAL_SORT)(rx)


def _real_cos(rx: dsl.ArithRef) -> dsl.ArithRef:
    if dsl.is_rational_value(rx) and dsl.numerator_as_long(rx) == 0:
        return _REAL_ONE
    return _dsl_hw_uf("real_cos", REAL_SORT, REAL_SORT)(rx)


class SymbolicZ3BitvecDomain(Z3SymbolicDomain):
    """Z3 symbolic bitvector value domain supporting `exact` (QF_FP), `real` (QF_NRA), and `uf` FP modes."""

    def __init__(self, fp_mode: str = "exact"):
        self.fp_mode = fp_mode

    def const_u32(self, imm: int) -> Any: return _dsl_bv32(imm)
    def const_u64(self, imm: int) -> Any: return _dsl_bv64(imm)
    def const_bool(self, val: bool) -> Any: return dsl.BoolVal(val)
    def const_int_val(self, val: Any) -> int | None:
        if dsl.is_bv_value(val):
            return dsl.as_long(val)
        if dsl.is_bv(val):
            simp = dsl.simplify(val)
            if dsl.is_bv_value(simp):
                return dsl.as_long(simp)
        return None
    def as_u32(self, val: Any) -> dsl.BitVecRef: return _as_bv32(val)
    def as_u64(self, val: Any) -> dsl.BitVecRef: return _as_bv64(val)
    def as_f32(self, val: Any) -> dsl.FPRef: return bv32_to_fp32(_as_bv32(val))
    def as_bool(self, val: Any) -> dsl.BoolRef: return _as_bool(val)

    def bv_add(self, a: Any, b: Any, bits: int = 32) -> dsl.BitVecRef:
        if bits == 64 and dsl.is_bv_value(a) and dsl.is_bv_value(b):
            return _dsl_bv64((dsl.as_long(a) + dsl.as_long(b)) & MASK64)
        return a + b
    def bv_sub(self, a: Any, b: Any, bits: int = 32) -> dsl.BitVecRef:
        if bits == 64 and dsl.is_bv_value(a) and dsl.is_bv_value(b):
            return _dsl_bv64((dsl.as_long(a) - dsl.as_long(b)) & MASK64)
        return a - b
    def bv_mul(self, a: Any, b: Any, bits: int = 32) -> dsl.BitVecRef:
        if bits == 64 and dsl.is_bv_value(a) and dsl.is_bv_value(b):
            return _dsl_bv64((dsl.as_long(a) * dsl.as_long(b)) & MASK64)
        return a * b
    def bv_and(self, a: Any, b: Any, bits: int = 32) -> dsl.BitVecRef: return a & b
    def bv_or(self, a: Any, b: Any, bits: int = 32) -> dsl.BitVecRef: return a | b
    def bv_xor(self, a: Any, b: Any, bits: int = 32) -> dsl.BitVecRef: return a ^ b
    def bv_not(self, a: Any, bits: int = 32) -> dsl.BitVecRef: return ~a
    def bv_neg(self, a: Any, bits: int = 32) -> dsl.BitVecRef: return -a
    def bv_shl(self, a: Any, sh: Any, bits: int = 32) -> dsl.BitVecRef: return a << sh
    def bv_lshr(self, a: Any, sh: Any, bits: int = 32) -> dsl.BitVecRef: return dsl.LShR(a, sh)
    def bv_ashr(self, a: Any, sh: Any, bits: int = 32) -> dsl.BitVecRef: return a >> sh
    def bv_rotl(self, a: Any, sh: Any) -> dsl.BitVecRef: return dsl.RotateLeft(a, sh)
    def bv_rotr(self, a: Any, sh: Any) -> dsl.BitVecRef: return dsl.RotateRight(a, sh)
    def bv_udiv(self, a: Any, b: Any) -> dsl.BitVecRef: return dsl.UDiv(a, b)
    def bv_sdiv(self, a: Any, b: Any) -> dsl.BitVecRef: return a / b
    def bv_urem(self, a: Any, b: Any) -> dsl.BitVecRef: return dsl.URem(a, b)
    def bv_srem(self, a: Any, b: Any) -> dsl.BitVecRef: return dsl.SRem(a, b)
    def bv_eq(self, a: Any, b: Any) -> dsl.BoolRef: return a == b
    def bv_ne(self, a: Any, b: Any) -> dsl.BoolRef: return a != b
    def bv_ult(self, a: Any, b: Any) -> dsl.BoolRef: return dsl.ULT(a, b)
    def bv_ule(self, a: Any, b: Any) -> dsl.BoolRef: return dsl.ULE(a, b)
    def bv_ugt(self, a: Any, b: Any) -> dsl.BoolRef: return dsl.UGT(a, b)
    def bv_uge(self, a: Any, b: Any) -> dsl.BoolRef: return dsl.UGE(a, b)
    def bv_slt(self, a: Any, b: Any) -> dsl.BoolRef: return a < b
    def bv_sle(self, a: Any, b: Any) -> dsl.BoolRef: return a <= b
    def bv_sgt(self, a: Any, b: Any) -> dsl.BoolRef: return a > b
    def bv_sge(self, a: Any, b: Any) -> dsl.BoolRef: return a >= b
    def bv_extract(self, hi: int, lo: int, x: Any) -> dsl.BitVecRef: return dsl.Extract(hi, lo, x)
    def bv_zext(self, ext_bits: int, x: Any, src_bits: int) -> dsl.BitVecRef: return dsl.ZeroExt(ext_bits, x)
    def bv_sext(self, ext_bits: int, x: Any, src_bits: int) -> dsl.BitVecRef: return dsl.SignExt(ext_bits, x)
    def bv_concat(self, hi: Any, lo: Any) -> dsl.BitVecRef: return dsl.Concat(hi, lo)
    def bv_clz(self, x: Any) -> dsl.BitVecRef: return _dsl_clz_bv32(x)
    def bv_popcount(self, x: Any) -> dsl.BitVecRef: return _dsl_popcount_bv32(x)
    def bv_bitrev(self, x: Any) -> dsl.BitVecRef: return _dsl_bitrev_bv32(x)

    def logical_and(self, a: Any, b: Any) -> dsl.BoolRef: return _z3_and(self.as_bool(a), self.as_bool(b))
    def logical_or(self, a: Any, b: Any) -> dsl.BoolRef: return _z3_or(self.as_bool(a), self.as_bool(b))
    def logical_not(self, a: Any) -> dsl.BoolRef: return _z3_not(self.as_bool(a))
    def select(self, cond: Any, t_val: Any, f_val: Any) -> Z3Val: return _z3_if(self.as_bool(cond), t_val, f_val)

    def hw_uf(self, env: Any, name: str, *args: Any, out_bits: int = 32, fallback_tag: str | None = None) -> dsl.BitVecRef:
        bargs = [self.as_u32(a) for a in args]
        out_sort = BV64 if out_bits == 64 else BV32
        sorts = [BV32] * len(bargs) + [out_sort]
        return _dsl_hw_uf(name, *sorts)(*bargs)

    def unpack_half2x16(self, u: Any) -> tuple[dsl.BitVecRef, dsl.BitVecRef]:
        bu = self.as_u32(u)
        return (f16_bv_to_f32_bv(dsl.Extract(15, 0, bu)), f16_bv_to_f32_bv(dsl.Extract(31, 16, bu)))

    def pack_half2x16(self, v2: Any) -> dsl.BitVecRef:
        if isinstance(v2, tuple) and len(v2) >= 2:
            f0 = f32_bv_to_f16_bv(self.as_u32(v2[0]))
            f1 = f32_bv_to_f16_bv(self.as_u32(v2[1]))
            return dsl.Concat(f1, f0)
        s = f32_bv_to_f16_bv(self.as_u32(v2))
        return dsl.Concat(s, s)

    def fp_unary(self, op: str, x: Any) -> dsl.BitVecRef:
        bx = self.as_u32(x)
        if self.fp_mode == "real":
            rx = _bv32_to_real(bx)
            if op == "FAbs": return _real_to_bv32(dsl.If(rx < _REAL_ZERO, -rx, rx))
            if op == "FNeg": return _real_to_bv32(-rx)
            if op == "Sqrt": return _real_to_bv32(_real_sqrt(rx))
            if op == "InverseSqrt": return _real_to_bv32(_REAL_ONE / _real_sqrt(rx))
            if op == "Exp2": return _real_to_bv32(_real_exp2(rx))
            if op == "Exp": return _real_to_bv32(_real_exp2(rx * _LOG2_E_REAL))
            if op == "Log2": return _real_to_bv32(_real_log2(rx))
            if op == "Log": return _real_to_bv32(_real_log2(rx) * _LN_2_REAL)
            if op == "Sin": return _real_to_bv32(_real_sin(rx))
            if op == "Cos": return _real_to_bv32(_real_cos(rx))
            return _real_to_bv32(_dsl_hw_uf(f"real_{op}", REAL_SORT, REAL_SORT)(rx))
        if op == "FAbs": return bx & _dsl_bv32(0x7FFFFFFF)
        if op == "FNeg": return bx ^ _dsl_bv32(0x80000000)
        if self.fp_mode == "uf" or op in (
            "InverseSqrt", "Sin", "Cos", "Exp", "Exp2", "Log", "Log2",
            "Atan", "Asin", "Acos", "Sinh", "Cosh", "Tanh",
            "DPdx", "DPdy", "Fwidth", "FrexpExp", "FrexpSignificand",
        ):
            return _dsl_hw_uf(f"fp_{op}", BV32, BV32)(bx)
        f = self.as_f32(bx)
        if op == "Floor": return fp32_to_bv32(dsl.fpRoundToIntegral(dsl.RTN(), f))
        if op == "Ceil": return fp32_to_bv32(dsl.fpRoundToIntegral(dsl.RTP(), f))
        if op == "Trunc": return fp32_to_bv32(dsl.fpRoundToIntegral(dsl.RTZ(), f))
        if op == "RoundEven": return fp32_to_bv32(dsl.fpRoundToIntegral(dsl.RNE(), f))
        if op == "Sqrt": return fp32_to_bv32(dsl.fpSqrt(dsl.RNE(), f))
        return _dsl_hw_uf(f"fp_{op}", BV32, BV32)(bx)

    def fp_binary(self, op: str, a: Any, b: Any) -> dsl.BitVecRef:
        ba, bb = self.as_u32(a), self.as_u32(b)
        if self.fp_mode == "real":
            ra, rb = _bv32_to_real(ba), _bv32_to_real(bb)
            if op == "FAdd": return _real_to_bv32(ra + rb)
            if op == "FSub": return _real_to_bv32(ra - rb)
            if op == "FMul": return _real_to_bv32(ra * rb)
            if op == "FDiv": return _real_to_bv32(ra / rb)
            if op in ("FMin", "NMin"): return _real_to_bv32(dsl.If(ra <= rb, ra, rb))
            if op in ("FMax", "NMax"): return _real_to_bv32(dsl.If(ra >= rb, ra, rb))
            if op == "Pow":
                if dsl.is_rational_value(rb):
                    num, den = dsl.numerator_as_long(rb), dsl.denominator_as_long(rb)
                    if num == 0: return _real_to_bv32(_REAL_ONE)
                    if num == den: return _real_to_bv32(ra)
                    if num == 2 * den: return _real_to_bv32(ra * ra)
                    if 2 * num == den: return _real_to_bv32(_real_sqrt(ra))
                    if 2 * num == -den: return _real_to_bv32(_REAL_ONE / _real_sqrt(ra))
                    if num == -den: return _real_to_bv32(_REAL_ONE / ra)
                return _real_to_bv32(_real_exp2(rb * _real_log2(ra)))
            return _real_to_bv32(_dsl_hw_uf(f"real_{op}", REAL_SORT, REAL_SORT, REAL_SORT)(ra, rb))
        if self.fp_mode == "uf" or op in ("Pow", "Ldexp", "Atan2"):
            return _dsl_hw_uf(f"fp_{op}", BV32, BV32, BV32)(ba, bb)
        fa, fb = self.as_f32(ba), self.as_f32(bb)
        if op == "FAdd": return fp32_to_bv32(dsl.fpAdd(dsl.RNE(), fa, fb))
        if op == "FSub": return fp32_to_bv32(dsl.fpSub(dsl.RNE(), fa, fb))
        if op == "FMul": return fp32_to_bv32(dsl.fpMul(dsl.RNE(), fa, fb))
        if op == "FDiv": return fp32_to_bv32(dsl.fpDiv(dsl.RNE(), fa, fb))
        if op in ("FMin", "NMin"): return fp32_to_bv32(dsl.fpMin(fa, fb))
        if op in ("FMax", "NMax"): return fp32_to_bv32(dsl.fpMax(fa, fb))
        return _dsl_hw_uf(f"fp_{op}", BV32, BV32, BV32)(ba, bb)

    def fp_clamp(self, x: Any, lo: Any, hi: Any) -> dsl.BitVecRef:
        if self.fp_mode == "uf":
            return _dsl_hw_uf("fp_FClamp", BV32, BV32, BV32, BV32)(self.as_u32(x), self.as_u32(lo), self.as_u32(hi))
        return super().fp_clamp(x, lo, hi)

    def fp_fma(self, a: Any, b: Any, c: Any) -> dsl.BitVecRef:
        ba, bb, bc = self.as_u32(a), self.as_u32(b), self.as_u32(c)
        if self.fp_mode == "real":
            return _real_to_bv32(_bv32_to_real(ba) * _bv32_to_real(bb) + _bv32_to_real(bc))
        if self.fp_mode == "uf":
            return _dsl_hw_uf("fp_Fma", BV32, BV32, BV32, BV32)(ba, bb, bc)
        bc_is_pos_zero = dsl.is_bv_value(bc) and dsl.as_long(bc) == 0
        if bc_is_pos_zero:
            if dsl.is_bv_value(bb) and dsl.as_long(bb) == 0x3F800000:
                return dsl.If(
                    dsl.fpIsNaN(self.as_f32(ba)), fp32_to_bv32(dsl.fpNaN(FP32_SORT)),
                    dsl.If(ba == _dsl_bv32(0x80000000), _dsl_bv32(0), ba),
                )
            if dsl.is_bv_value(ba) and dsl.as_long(ba) == 0x3F800000:
                return dsl.If(
                    dsl.fpIsNaN(self.as_f32(bb)), fp32_to_bv32(dsl.fpNaN(FP32_SORT)),
                    dsl.If(bb == _dsl_bv32(0x80000000), _dsl_bv32(0), bb),
                )
        return fp32_to_bv32(dsl.fpFMA(dsl.RNE(), self.as_f32(ba), self.as_f32(bb), self.as_f32(bc)))

    def fp_is_nan(self, x: Any) -> dsl.BoolRef:
        bx = self.as_u32(x)
        return dsl.fpIsNaN(self.as_f32(bx)) if self.fp_mode == "exact" else _dsl_hw_uf("fp_IsNan", BV32, dsl.BoolSort())(bx)

    def fp_is_inf(self, x: Any) -> dsl.BoolRef:
        bx = self.as_u32(x)
        return dsl.fpIsInf(self.as_f32(bx)) if self.fp_mode == "exact" else _dsl_hw_uf("fp_IsInf", BV32, dsl.BoolSort())(bx)

    def fp_cmp(self, op: str, a: Any, b: Any) -> dsl.BoolRef:
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
            return _dsl_hw_uf(f"realcmp_{op}", REAL_SORT, REAL_SORT, dsl.BoolSort())(ra, rb)
        if self.fp_mode == "uf":
            return _dsl_hw_uf(f"fpcmp_{op}", BV32, BV32, dsl.BoolSort())(ba, bb)
        fa, fb = self.as_f32(ba), self.as_f32(bb)
        if op == "FCmpOrdEq": return dsl.fpEQ(fa, fb)
        if op == "FCmpOrdGt": return dsl.fpGT(fa, fb)
        if op == "FCmpOrdGe": return dsl.fpGEQ(fa, fb)
        if op == "FCmpOrdLt": return dsl.fpLT(fa, fb)
        if op == "FCmpOrdLe": return dsl.fpLEQ(fa, fb)
        if op == "FCmpOrdNe": return dsl.Or(dsl.fpLT(fa, fb), dsl.fpGT(fa, fb))
        if op == "FCmpUnordNe": return dsl.Not(dsl.fpEQ(fa, fb))
        if op == "FCmpTotal": return _dsl_fp_total_le_bv(32, ba, bb)
        return _dsl_hw_uf(f"fpcmp_{op}", BV32, BV32, dsl.BoolSort())(ba, bb)

    def convert(self, op: str, x: Any) -> dsl.BitVecRef:
        if op == "ConvertFToS":
            bx = self.as_u32(x)
            if self.fp_mode == "real":
                return _dsl_hw_uf("real_FToS", REAL_SORT, BV32)(_bv32_to_real(bx))
            return _dsl_fp_to_int_sat(self.as_f32(bx), 32, True, dsl.RTZ()) if self.fp_mode == "exact" else _dsl_hw_uf("fp_FToS", BV32, BV32)(bx)
        if op == "ConvertFToU":
            bx = self.as_u32(x)
            if self.fp_mode == "real":
                return _dsl_hw_uf("real_FToU", REAL_SORT, BV32)(_bv32_to_real(bx))
            return _dsl_fp_to_int_sat(self.as_f32(bx), 32, False, dsl.RTZ()) if self.fp_mode == "exact" else _dsl_hw_uf("fp_FToU", BV32, BV32)(bx)
        if op == "ConvertSToF":
            bx = self.as_u32(x)
            if self.fp_mode == "real":
                if dsl.is_bv_value(bx):
                    return _real_to_bv32(dsl.RealVal(dsl.as_signed_long(bx)))
                return _real_to_bv32(_dsl_hw_uf("real_SToF", BV32, REAL_SORT)(bx))
            return fp32_to_bv32(dsl.fpSignedToFP(dsl.RNE(), bx, FP32_SORT)) if self.fp_mode == "exact" else _dsl_hw_uf("fp_SToF", BV32, BV32)(bx)
        if op == "ConvertUToF":
            bx = self.as_u32(x)
            if self.fp_mode == "real":
                if dsl.is_bv_value(bx):
                    return _real_to_bv32(dsl.RealVal(dsl.as_long(bx)))
                return _real_to_bv32(_dsl_hw_uf("real_UToF", BV32, REAL_SORT)(bx))
            return fp32_to_bv32(dsl.fpUnsignedToFP(dsl.RNE(), bx, FP32_SORT)) if self.fp_mode == "exact" else _dsl_hw_uf("fp_UToF", BV32, BV32)(bx)
        if op == "SConvert32To64": return dsl.SignExt(32, self.as_u32(x))
        if op == "UConvert32To64": return dsl.ZeroExt(32, self.as_u32(x))
        if op == "UConvert64To32": return dsl.Extract(31, 0, self.as_u64(x))
        return _dsl_bv32(0)

    def unknown_sideband_u64(self, env: Any, imm_u32: int) -> dsl.BitVecRef:
        return self.hw_uf(env, f"sideband_u64_{imm_u32}", _dsl_bv32(imm_u32), out_bits=64)

    def sideband_u32(self, env: Any, imm_u32: int, imm_u32_2: int) -> dsl.BitVecRef:
        return self.hw_uf(env, f"sideband_u32_{imm_u32}", _dsl_bv32(imm_u32_2))

    def subgroup_ballot(self, env: Any, x: Any) -> dsl.BitVecRef:
        return _dsl_hw_uf("SubgroupBallot", BV32, dsl.BitVecSort(5), BV32)(self.as_u32(x), env.lane_id)

    def subgroup_shuffle(self, env: Any, x: Any, lane: Any) -> dsl.BitVecRef:
        return _dsl_hw_uf("SubgroupShuffle", BV32, BV32, dsl.BitVecSort(5), BV32)(
            self.as_u32(x), self.as_u32(lane), env.lane_id
        )

    # Z3SymbolicDomain boundary contract (identity / pass-through for raw Z3 bitvecs)
    def lift_z3_bv32(self, raw: Any) -> dsl.BitVecRef:
        return _as_bv32(raw)

    def lift_z3_bv64(self, raw: Any) -> dsl.BitVecRef:
        return _as_bv64(raw)

    def lift_z3_bool(self, raw: Any) -> dsl.BoolRef:
        return _as_bool(raw)

    def lower_z3_bv32(self, val: Any) -> dsl.BitVecRef:
        return _as_bv32(val)

    def lower_z3_bv64(self, val: Any) -> dsl.BitVecRef:
        return _as_bv64(val)

    def lower_z3_bool(self, val: Any) -> dsl.BoolRef:
        return _as_bool(val)

    def lower_z3_val(self, val: Any) -> Any:
        return val

    def merge_z3(self, cond: Any, t_val: Any, f_val: Any) -> Any:
        return _z3_if(_as_bool(cond), t_val, f_val)

    def to_z3_real_f32(self, val: Any) -> dsl.ArithRef:
        return _bv32_to_real(_as_bv32(val))

#!/usr/bin/env python3
"""
Z3 / DSL Expression Helpers for Valhall Symbolic SSA.

Pure expression-level constants, bitvector/boolean coercions, Real/IEEE-754
bridge helpers, and DSL hardware uninterpreted function (`hw_*`) caches.
Separated from `ShaderEnv` so Z3 term construction has zero dependency on
shader bundle metadata or concrete execution environments.
"""

from __future__ import annotations

import fractions
import math
import struct
from typing import Any, Dict, Optional, Tuple, Union

from tools.symbolic_ssa.z3._raw_z3 import raw_z3 as _raw_z3

from tools.symbolic_ssa.z3 import dsl
from tools.verify.panfrost_z3_model import (
    BV8,
    BV16,
    BV32,
    BV64,
    FP32_SORT,
    _hw_uf,
    bv16_to_fp16,
    bv32_to_fp32,
    fp16_to_bv16,
    fp32_to_bv32,
)

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

Z3Val = Union[Any, Tuple[Any, ...]]

REAL_SORT = _raw_z3.RealSort()
BOOL_SORT = _raw_z3.BoolSort()
RNE = _raw_z3.RNE()
RTZ = _raw_z3.RTZ()

_REAL_ZERO = dsl.RealVal(0)
_REAL_ONE = dsl.RealVal(1)
_REAL_TWO = dsl.RealVal(2)
_REAL_HALF = dsl.Q(1, 2)
_REAL_NEG_HALF = dsl.Q(-1, 2)
_REAL_NEG_ONE = dsl.RealVal(-1)

_DSL_HW_UF_CACHE: Dict[Tuple[str, Tuple[_raw_z3.SortRef, ...]], dsl.Z3FuncDecl] = {}


def _dsl_hw_uf(name: str, *sorts: _raw_z3.SortRef) -> dsl.Z3FuncDecl:
    """Return a cached lazy `Z3FuncDecl` named `hw_<name>` (matching `panfrost_z3_model._hw_uf`)."""
    key = (name, sorts)
    fn = _DSL_HW_UF_CACHE.get(key)
    if fn is None:
        fn = dsl.Function(f"hw_{name}", *sorts)
        _DSL_HW_UF_CACHE[key] = fn
    return fn


def _dsl_bv32(v: int) -> dsl.Z3BvConst:
    return dsl.BitVecVal(v & MASK32, 32)


def _dsl_bv64(v: int) -> dsl.Z3BvConst:
    return dsl.BitVecVal(v & MASK64, 64)


def _f32_bits_to_real_const(u32: int) -> Optional[dsl.Z3ArithExpr]:
    f = struct.unpack("<f", struct.pack("<I", u32 & MASK32))[0]
    if not math.isfinite(f):
        return None
    if f == 0.0:
        return _REAL_ZERO
    frac = fractions.Fraction(f)
    return dsl.Q(frac.numerator, frac.denominator)


_LOG2_E_REAL = _f32_bits_to_real_const(0x3FB8AA3B)
_LN_2_REAL = _f32_bits_to_real_const(0x3F317218)


def _is_real2bv(x: Any) -> bool:
    if isinstance(x, dsl.Z3OpUFApp) and x.func.fn_name == "hw_real2bv" and len(x.args) == 1:
        return True
    return (
        isinstance(x, _raw_z3.ExprRef)
        and _raw_z3.is_app(x)
        and x.decl() == _hw_uf("real2bv", REAL_SORT, BV32)
        and len(x.children()) == 1
    )


def _bv32_to_real(x: Any) -> Any:
    """Decode BV32 to exact DSL Real, cancelling `hw_real2bv` wrappers and pushing through ITEs."""
    if _is_real2bv(x):
        return x.children()[0]
    if isinstance(x, dsl.Z3OpIte):
        return dsl.If(x.cond, _bv32_to_real(x.then_expr), _bv32_to_real(x.else_expr))
    if (
        isinstance(x, _raw_z3.ExprRef)
        and _raw_z3.is_app(x)
        and x.decl().kind() == _raw_z3.Z3_OP_ITE
        and len(x.children()) == 3
    ):
        c, t, e = x.children()
        return dsl.If(c, _bv32_to_real(t), _bv32_to_real(e))
    if dsl.is_bv_value(x):
        rc = _f32_bits_to_real_const(dsl.as_long(x))
        if rc is not None:
            return rc
    return _dsl_hw_uf("bv2real", BV32, REAL_SORT)(x)


def _real_to_bv32(r: Any) -> Any:
    """Wrap a DSL Real expression back into BV32 via the canonical `hw_real2bv` bridge."""
    return _dsl_hw_uf("real2bv", REAL_SORT, BV32)(r)


def _real_sqrt(rx: Any) -> Any:
    if dsl.is_rational_value(rx):
        num, den = rx.numerator_as_long(), rx.denominator_as_long()
        if num == 0:
            return _REAL_ZERO
        if num == den:
            return _REAL_ONE
    return _dsl_hw_uf("real_sqrt", REAL_SORT, REAL_SORT)(rx)


def _real_exp2(rx: Any) -> Any:
    if dsl.is_rational_value(rx):
        num, den = rx.numerator_as_long(), rx.denominator_as_long()
        if num == 0:
            return _REAL_ONE
        if num == den:
            return _REAL_TWO
    return _dsl_hw_uf("real_exp2", REAL_SORT, REAL_SORT)(rx)


def _real_log2(rx: Any) -> Any:
    if dsl.is_rational_value(rx):
        num, den = rx.numerator_as_long(), rx.denominator_as_long()
        if num == den:
            return _REAL_ZERO
        if num == 2 * den:
            return _REAL_ONE
    return _dsl_hw_uf("real_log2", REAL_SORT, REAL_SORT)(rx)


def _real_sin(rx: Any) -> Any:
    if dsl.is_rational_value(rx) and rx.numerator_as_long() == 0:
        return _REAL_ZERO
    return _dsl_hw_uf("real_sin", REAL_SORT, REAL_SORT)(rx)


def _real_cos(rx: Any) -> Any:
    if dsl.is_rational_value(rx) and rx.numerator_as_long() == 0:
        return _REAL_ONE
    return _dsl_hw_uf("real_cos", REAL_SORT, REAL_SORT)(rx)


def _as_bv32(v: Z3Val) -> Any:
    if isinstance(v, tuple):
        return _as_bv32(v[0]) if v else _dsl_bv32(0)
    if isinstance(v, _raw_z3.ExprRef):
        v = dsl.lift(v, depth=1)
    if dsl.is_bool(v):
        return dsl.If(v, _dsl_bv32(1), _dsl_bv32(0))
    if dsl.is_fp(v):
        return dsl.lift(fp32_to_bv32(dsl.lower(v)), depth=1)
    assert dsl.is_bv(v), f"Expected BitVec, got {type(v)}"
    if v.size() == 32:
        return v
    if v.size() < 32:
        return dsl.ZeroExt(32 - v.size(), v)
    return dsl.Extract(31, 0, v)


def _as_bv64(v: Z3Val) -> Any:
    if isinstance(v, tuple):
        return _as_bv64(v[0]) if v else _dsl_bv64(0)
    if isinstance(v, _raw_z3.ExprRef):
        v = dsl.lift(v, depth=1)
    if dsl.is_bv(v):
        if v.size() == 64:
            return v
        if v.size() < 64:
            return dsl.ZeroExt(64 - v.size(), v)
        return dsl.Extract(63, 0, v)
    return dsl.ZeroExt(32, _as_bv32(v))


def _as_bool(v: Z3Val) -> Any:
    if isinstance(v, tuple):
        return _as_bool(v[0]) if v else dsl.BoolVal(False)
    if isinstance(v, _raw_z3.ExprRef):
        v = dsl.lift(v, depth=1)
    if dsl.is_bool(v):
        return v
    if dsl.is_bv(v):
        if dsl.is_bv_value(v):
            return dsl.BoolVal(dsl.as_long(v) != 0)
        return v != dsl.BitVecVal(0, v.size())
    return dsl.BoolVal(False)


def _z3_and(a: Any, b: Any) -> Any:
    if dsl.is_false(a) or dsl.is_false(b):
        return dsl.BoolVal(False)
    if dsl.is_true(a):
        return b
    if dsl.is_true(b):
        return a
    if dsl.expr_eq(a, b):
        return a
    return dsl.And(a, b)


def _z3_or(a: Any, b: Any) -> Any:
    if dsl.is_true(a) or dsl.is_true(b):
        return dsl.BoolVal(True)
    if dsl.is_false(a):
        return b
    if dsl.is_false(b):
        return a
    if dsl.expr_eq(a, b):
        return a
    return dsl.Or(a, b)


def _z3_not(a: Any) -> Any:
    if dsl.is_true(a):
        return dsl.BoolVal(False)
    if dsl.is_false(a):
        return dsl.BoolVal(True)
    return dsl.Not(a)


def _z3_if(cond: Any, t_val: Z3Val, f_val: Z3Val) -> Z3Val:
    if dsl.is_true(cond):
        return t_val
    if dsl.is_false(cond):
        return f_val
    if isinstance(t_val, tuple) and isinstance(f_val, tuple) and len(t_val) == len(f_val):
        return tuple(_z3_if(cond, a, b) for a, b in zip(t_val, f_val))
    if isinstance(t_val, (dsl.Z3Expr, _raw_z3.ExprRef)) and isinstance(f_val, (dsl.Z3Expr, _raw_z3.ExprRef)):
        if isinstance(t_val, _raw_z3.ExprRef):
            t_val = dsl.lift(t_val, depth=1)
        if isinstance(f_val, _raw_z3.ExprRef):
            f_val = dsl.lift(f_val, depth=1)
        if t_val.eq(f_val):
            return t_val
        if t_val.sort() != f_val.sort():
            if dsl.is_bv(t_val) and dsl.is_bv(f_val):
                w = max(t_val.size(), f_val.size())
                if t_val.size() < w:
                    t_val = dsl.ZeroExt(w - t_val.size(), t_val)
                if f_val.size() < w:
                    f_val = dsl.ZeroExt(w - f_val.size(), f_val)
            else:
                t_val, f_val = _as_bv32(t_val), _as_bv32(f_val)
        return dsl.If(cond, t_val, f_val)
    return t_val


def _try_const_u64(e: Any) -> Optional[int]:
    if dsl.is_bv_value(e):
        return dsl.as_long(e)
    if dsl.is_bv(e):
        s = _raw_z3.simplify(dsl.lower(e))
        if isinstance(s, _raw_z3.BitVecNumRef):
            return s.as_long()
    return None

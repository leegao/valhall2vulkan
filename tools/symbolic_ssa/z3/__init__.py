#!/usr/bin/env python3
"""
Drop-in Z3 replacement facade (`tools.symbolic_ssa.z3`).

Exposes the full Z3 Python API while constructing lazy, pattern-matchable
`dsl.Z3Expr` dataclasses by default and transparently lowering `dsl.Z3Expr`
(and re-lifting where appropriate) across Z3 C-API boundaries (`simplify`,
`substitute`, `ForAll`, `Exists`, `Solver`, `Optimize`, `Goal`, `ModelRef`).

Callers can switch from `import z3` to `from tools.symbolic_ssa import z3`
(or `import tools.symbolic_ssa.z3 as z3`) without rewriting call sites.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence, Tuple, Union

from tools.symbolic_ssa.z3._raw_z3 import raw_z3 as _raw_z3

from tools.symbolic_ssa.z3 import dsl
from tools.symbolic_ssa.z3.dsl import (  # noqa: F401
    And,
    ArithCmpOp,
    Array,
    BitVec,
    BitVecVal,
    BitVecs,
    Bool,
    BoolVal,
    Bools,
    BridgeOp,
    BvCmpOp,
    Concat,
    Const,
    Consts,
    Distinct,
    Extract,
    FP,
    FPVal,
    Function,
    If,
    Implies,
    Int,
    Int2BV,
    IntVal,
    Ints,
    K,
    LShR,
    Not,
    Or,
    Q,
    Real,
    RealVal,
    Reals,
    Select,
    SignExt,
    Store,
    ToInt,
    ToReal,
    UDiv,
    UGE,
    UGT,
    ULE,
    ULT,
    URem,
    Xor,
    Z3ArithExpr,
    Z3ArrayExpr,
    Z3ArrayVar,
    Z3BitVecExpr,
    Z3BoolConst,
    Z3BoolExpr,
    Z3BoolVar,
    Z3BvConst,
    Z3BvVar,
    Z3ConstArray,
    Z3Expr,
    Z3FPConst,
    Z3FPExpr,
    Z3FPVar,
    Z3FuncDecl,
    Z3GenericVar,
    Z3IntConst,
    Z3IntVar,
    Z3Node,
    Z3OpAnd,
    Z3OpArithCmp,
    Z3OpBridge,
    Z3OpBv2Int,
    Z3OpBvAShr,
    Z3OpBvAdd,
    Z3OpBvAnd,
    Z3OpBvCmp,
    Z3OpBvLShr,
    Z3OpBvMul,
    Z3OpBvNeg,
    Z3OpBvNot,
    Z3OpBvOr,
    Z3OpBvSDiv,
    Z3OpBvSMod,
    Z3OpBvSRem,
    Z3OpBvShl,
    Z3OpBvSub,
    Z3OpBvUDiv,
    Z3OpBvURem,
    Z3OpBvXor,
    Z3OpConcat,
    Z3OpDistinct,
    Z3OpEq,
    Z3OpExtract,
    Z3OpImplies,
    Z3OpInt2Bv,
    Z3OpIntAdd,
    Z3OpIntDiv,
    Z3OpIntMod,
    Z3OpIntMul,
    Z3OpIntNeg,
    Z3OpIntSub,
    Z3OpIte,
    Z3OpKind,
    Z3OpNot,
    Z3OpOr,
    Z3OpRealAdd,
    Z3OpRealDiv,
    Z3OpRealMul,
    Z3OpRealNeg,
    Z3OpRealSub,
    Z3OpSelect,
    Z3OpSignExt,
    Z3OpStore,
    Z3OpToInt,
    Z3OpToReal,
    Z3OpUFApp,
    Z3OpXor,
    Z3OpZeroExt,
    Z3Other,
    Z3PolyExpr,
    Z3RealConst,
    Z3RealVar,
    Z3SortKind,
    ZeroExt,
    as_long,
    expr_eq,
    is_app,
    is_array,
    is_bool,
    is_bv,
    is_bv_value,
    is_const,
    is_expr,
    is_false,
    is_fp,
    is_int,
    is_int_value,
    is_rational_value,
    is_real,
    is_true,
    lift,
    lower,
    view,
)


def _lower_arg(x: Any) -> Any:
    if isinstance(x, Z3Expr):
        return x.to_z3
    if isinstance(x, Z3FuncDecl):
        return x.to_z3
    if isinstance(x, (list, tuple)):
        return type(x)(_lower_arg(v) for v in x)
    return x


def _lower_args(args: Iterable[Any]) -> Tuple[Any, ...]:
    return tuple(_lower_arg(a) for a in args)


def simplify(expr: Any, *args: Any, relift: bool = False, **kwargs: Any) -> Any:
    """Run `z3.simplify` on `expr` (lowering `dsl.Z3Expr` first); optionally re-lift to `dsl.Z3Expr`."""
    res = _raw_z3.simplify(lower(expr), *args, **kwargs)
    if relift and isinstance(expr, Z3Expr):
        return lift(res, depth=2)
    return res


def substitute(expr: Any, *m: Any) -> Any:
    """Run `z3.substitute` with automatic lowering of `expr` and substitution pairs."""
    was_dsl = isinstance(expr, Z3Expr)
    if len(m) == 1 and isinstance(m[0], (list, tuple)) and m[0] and isinstance(m[0][0], tuple):
        pairs = [(lower(a), lower(b)) for a, b in m[0]]
        res = _raw_z3.substitute(lower(expr), pairs)
    else:
        pairs = [(lower(a), lower(b)) for a, b in m]
        res = _raw_z3.substitute(lower(expr), *pairs)
    return lift(res, depth=2) if was_dsl else res


def ForAll(vs: Any, body: Any, *args: Any, **kwargs: Any) -> _raw_z3.QuantifierRef:
    """Drop-in `z3.ForAll` lowering DSL variables, body, and trigger patterns."""
    vs_l = _lower_arg(vs)
    body_l = lower(body)
    if "patterns" in kwargs and kwargs["patterns"] is not None:
        kwargs = dict(kwargs)
        kwargs["patterns"] = [_lower_arg(p) for p in kwargs["patterns"]]
    return _raw_z3.ForAll(vs_l, body_l, *args, **kwargs)


def Exists(vs: Any, body: Any, *args: Any, **kwargs: Any) -> _raw_z3.QuantifierRef:
    """Drop-in `z3.Exists` lowering DSL variables, body, and trigger patterns."""
    vs_l = _lower_arg(vs)
    body_l = lower(body)
    if "patterns" in kwargs and kwargs["patterns"] is not None:
        kwargs = dict(kwargs)
        kwargs["patterns"] = [_lower_arg(p) for p in kwargs["patterns"]]
    return _raw_z3.Exists(vs_l, body_l, *args, **kwargs)


def is_ast(x: Any) -> bool:
    return isinstance(x, Z3Expr) or _raw_z3.is_ast(x)


def is_var(x: Any) -> bool:
    return _raw_z3.is_var(lower(x)) if isinstance(x, (Z3Expr, _raw_z3.AstRef)) else False


def is_quantifier(x: Any) -> bool:
    return _raw_z3.is_quantifier(lower(x)) if isinstance(x, (Z3Expr, _raw_z3.AstRef)) else False


def BV2Int(a: Any, is_signed: bool = False) -> Any:
    if isinstance(a, Z3Expr) and not is_signed:
        return dsl.BV2Int(a)
    return _raw_z3.BV2Int(lower(a), is_signed=is_signed)


def fpAdd(rm: Any, a: Any, b: Any) -> _raw_z3.FPRef:
    return _raw_z3.fpAdd(rm, lower(a), lower(b))


def fpSub(rm: Any, a: Any, b: Any) -> _raw_z3.FPRef:
    return _raw_z3.fpSub(rm, lower(a), lower(b))


def fpMul(rm: Any, a: Any, b: Any) -> _raw_z3.FPRef:
    return _raw_z3.fpMul(rm, lower(a), lower(b))


def fpDiv(rm: Any, a: Any, b: Any) -> _raw_z3.FPRef:
    return _raw_z3.fpDiv(rm, lower(a), lower(b))


def fpFMA(rm: Any, a: Any, b: Any, c: Any) -> _raw_z3.FPRef:
    return _raw_z3.fpFMA(rm, lower(a), lower(b), lower(c))


def fpNeg(a: Any) -> _raw_z3.FPRef:
    return _raw_z3.fpNeg(lower(a))


def fpAbs(a: Any) -> _raw_z3.FPRef:
    return _raw_z3.fpAbs(lower(a))


def fpSqrt(rm: Any, a: Any) -> _raw_z3.FPRef:
    return _raw_z3.fpSqrt(rm, lower(a))


def fpToIEEEBV(a: Any) -> _raw_z3.BitVecRef:
    return _raw_z3.fpToIEEEBV(lower(a))


def fpBVToFP(a: Any, s: _raw_z3.FPSortRef) -> _raw_z3.FPRef:
    return _raw_z3.fpBVToFP(lower(a), s)


class ModelRef(_raw_z3.ModelRef):
    """Drop-in `z3.ModelRef` subclass that lowers `dsl.Z3Expr`/`dsl.Z3FuncDecl` automatically."""

    def eval(self, t: Any, model_completion: bool = False) -> Any:
        return super().eval(_lower_arg(t), model_completion=model_completion)

    def get_interp(self, decl: Any) -> Any:
        return super().get_interp(_lower_arg(decl))

    def __getitem__(self, idx: Any) -> Any:
        return super().__getitem__(_lower_arg(idx))


def _wrap_model(m: _raw_z3.ModelRef) -> ModelRef:
    return ModelRef(m.model, m.ctx)


class Solver(_raw_z3.Solver):
    """Drop-in `z3.Solver` subclass that lowers `dsl.Z3Expr` arguments automatically."""

    def add(self, *args: Any) -> None:
        super().add(*_lower_args(args))

    def assert_and_track(self, p: Any, q: Any) -> None:
        super().assert_and_track(_lower_arg(p), _lower_arg(q))

    def check(self, *assumptions: Any) -> _raw_z3.CheckSatResult:
        return super().check(*_lower_args(assumptions))

    def model(self) -> ModelRef:
        return _wrap_model(super().model())


class Optimize(_raw_z3.Optimize):
    """Drop-in `z3.Optimize` subclass that lowers `dsl.Z3Expr` arguments automatically."""

    def add(self, *args: Any) -> None:
        super().add(*_lower_args(args))

    def add_soft(self, expr: Any, weight: Any = "1", id: Any = None) -> Any:
        return super().add_soft(_lower_arg(expr), weight=weight, id=id)

    def check(self, *assumptions: Any) -> _raw_z3.CheckSatResult:
        return super().check(*_lower_args(assumptions))

    def model(self) -> ModelRef:
        return _wrap_model(super().model())


class Goal(_raw_z3.Goal):
    """Drop-in `z3.Goal` subclass that lowers `dsl.Z3Expr` arguments automatically."""

    def add(self, *args: Any) -> None:
        super().add(*_lower_args(args))


def SolverFor(logic: str, ctx: Any = None) -> Solver:
    s = _raw_z3.SolverFor(logic, ctx=ctx)
    return Solver(solver=s.solver, ctx=s.ctx)


def __getattr__(name: str) -> Any:
    return getattr(_raw_z3, name)


def __dir__() -> Sequence[str]:
    return sorted(set(globals().keys()) | set(dir(_raw_z3)))

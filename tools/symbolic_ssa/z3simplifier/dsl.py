#!/usr/bin/env python3
"""
Typed Python `match`/`case` AST pattern-matching DSL for Z3 expressions.

Wraps Z3's C-API `ExprRef` kinds (`Z3_OP_*`) into lightweight, immutable
`@dataclass(frozen=True, slots=True)` view nodes so peephole and algebraic
rewrite rules can be written declaratively with Python 3.10+ structural
pattern matching instead of ad-hoc `decl().kind()` / `arg(i)` cascades:

    match view(node, depth=2):
        case Z3OpExtract(hi, 0, Z3OpZeroExt(_, inner)) if inner.size() == hi + 1:
            return inner
        case Z3OpExtract(hi, lo, Z3BvConst(val, _)):
            return z3.BitVecVal((val >> lo) & ((1 << (hi - lo + 1)) - 1), hi - lo + 1)
        case Z3OpBridge("u_f32_to_bv32", Z3OpBridge("u_bv32_to_f32", inner)):
            return inner

Design invariants:
  - Every wrapper preserves `.raw: z3.ExprRef` for O(1) lossless reconstruction.
  - `view(expr, depth=1)` is strictly O(1)-bounded (no unbounded recursion).
  - Zero external dependencies beyond `z3` and stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple, Union

import z3


@dataclass(frozen=True, slots=True)
class Z3BvConst:
    """Concrete BitVec literal `BitVecVal(val, width)`."""

    val: int
    width: int
    raw: z3.BitVecNumRef


@dataclass(frozen=True, slots=True)
class Z3RealConst:
    """Concrete Real rational literal `RealVal(num/den)`."""

    num: int
    den: int
    raw: z3.RatNumRef

    @property
    def is_zero(self) -> bool:
        return self.num == 0

    @property
    def is_one(self) -> bool:
        return self.num == self.den


@dataclass(frozen=True, slots=True)
class Z3IntConst:
    """Concrete Int literal `IntVal(val)`."""

    val: int
    raw: z3.IntNumRef


@dataclass(frozen=True, slots=True)
class Z3BoolConst:
    """Concrete Bool literal (`True` or `False`)."""

    val: bool
    raw: z3.BoolRef


@dataclass(frozen=True, slots=True)
class Z3OpExtract:
    """`Extract(hi, lo, src)`."""

    hi: int
    lo: int
    src: Any  # Z3Node (when depth > 1) or z3.ExprRef
    raw: z3.ExprRef

    @property
    def width(self) -> int:
        return self.hi - self.lo + 1


@dataclass(frozen=True, slots=True)
class Z3OpZeroExt:
    """`ZeroExt(k, src)`."""

    k: int
    src: Any
    raw: z3.ExprRef


@dataclass(frozen=True, slots=True)
class Z3OpSignExt:
    """`SignExt(k, src)`."""

    k: int
    src: Any
    raw: z3.ExprRef


@dataclass(frozen=True, slots=True)
class Z3OpConcat:
    """`Concat(*args)` (MSB at index 0, LSB at index -1)."""

    args: Tuple[z3.ExprRef, ...]
    raw: z3.ExprRef


@dataclass(frozen=True, slots=True)
class Z3OpIte:
    """`If(cond, then_expr, else_expr)`."""

    cond: Any
    then_expr: Any
    else_expr: Any
    raw: z3.ExprRef


@dataclass(frozen=True, slots=True)
class Z3OpEq:
    """`lhs == rhs`."""

    lhs: Any
    rhs: Any
    raw: z3.ExprRef


@dataclass(frozen=True, slots=True)
class Z3OpDistinct:
    """`lhs != rhs` (binary distinct)."""

    lhs: Any
    rhs: Any
    raw: z3.ExprRef


@dataclass(frozen=True, slots=True)
class Z3OpNot:
    """`Not(arg)`."""

    arg: Any
    raw: z3.ExprRef


@dataclass(frozen=True, slots=True)
class Z3OpAnd:
    """`And(*args)`."""

    args: Tuple[z3.ExprRef, ...]
    raw: z3.ExprRef


@dataclass(frozen=True, slots=True)
class Z3OpOr:
    """`Or(*args)`."""

    args: Tuple[z3.ExprRef, ...]
    raw: z3.ExprRef


@dataclass(frozen=True, slots=True)
class Z3OpBvMul:
    """Binary BitVec multiply `lhs * rhs`."""

    lhs: Any
    rhs: Any
    width: int
    raw: z3.ExprRef


@dataclass(frozen=True, slots=True)
class Z3OpBvAdd:
    """Binary BitVec add `lhs + rhs`."""

    lhs: Any
    rhs: Any
    width: int
    raw: z3.ExprRef


@dataclass(frozen=True, slots=True)
class Z3OpBvSub:
    """Binary BitVec subtract `lhs - rhs`."""

    lhs: Any
    rhs: Any
    width: int
    raw: z3.ExprRef


@dataclass(frozen=True, slots=True)
class Z3OpBvShl:
    """Binary BitVec shift-left `lhs << rhs`."""

    lhs: Any
    rhs: Any
    width: int
    raw: z3.ExprRef


@dataclass(frozen=True, slots=True)
class Z3OpBvLShr:
    """Binary BitVec logical shift-right `LShR(lhs, rhs)`."""

    lhs: Any
    rhs: Any
    width: int
    raw: z3.ExprRef


@dataclass(frozen=True, slots=True)
class Z3OpBvAShr:
    """Binary BitVec arithmetic shift-right `lhs >> rhs`."""

    lhs: Any
    rhs: Any
    width: int
    raw: z3.ExprRef


@dataclass(frozen=True, slots=True)
class Z3OpBvAnd:
    """Binary BitVec bitwise AND `lhs & rhs`."""

    lhs: Any
    rhs: Any
    width: int
    raw: z3.ExprRef


@dataclass(frozen=True, slots=True)
class Z3OpBvOr:
    """Binary BitVec bitwise OR `lhs | rhs`."""

    lhs: Any
    rhs: Any
    width: int
    raw: z3.ExprRef


@dataclass(frozen=True, slots=True)
class Z3OpBvXor:
    """Binary BitVec bitwise XOR `lhs ^ rhs`."""

    lhs: Any
    rhs: Any
    width: int
    raw: z3.ExprRef


@dataclass(frozen=True, slots=True)
class Z3OpBvCmp:
    """Binary BitVec signed/unsigned comparison (`slt`, `sleq`, `sgt`, `sgeq`, `ult`, `uleq`, `ugt`, `ugeq`)."""

    op: str
    lhs: Any
    rhs: Any
    raw: z3.ExprRef


@dataclass(frozen=True, slots=True)
class Z3OpBv2Int:
    """`BV2Int(src)`."""

    src: Any
    raw: z3.ExprRef


@dataclass(frozen=True, slots=True)
class Z3OpInt2Bv:
    """`Int2BV(src, width)`."""

    src: Any
    width: int
    raw: z3.ExprRef


@dataclass(frozen=True, slots=True)
class Z3OpToReal:
    """`ToReal(src)`."""

    src: Any
    raw: z3.ExprRef


@dataclass(frozen=True, slots=True)
class Z3OpToInt:
    """`ToInt(src)`."""

    src: Any
    raw: z3.ExprRef


@dataclass(frozen=True, slots=True)
class Z3OpRealAdd:
    """N-ary Real addition `+(args...)`."""

    args: Tuple[z3.ExprRef, ...]
    raw: z3.ExprRef


@dataclass(frozen=True, slots=True)
class Z3OpRealMul:
    """N-ary Real multiplication `*(args...)`."""

    args: Tuple[z3.ExprRef, ...]
    raw: z3.ExprRef


@dataclass(frozen=True, slots=True)
class Z3OpRealSub:
    """Binary Real subtraction `lhs - rhs`."""

    lhs: Any
    rhs: Any
    raw: z3.ExprRef


@dataclass(frozen=True, slots=True)
class Z3OpRealDiv:
    """Binary Real division `lhs / rhs`."""

    lhs: Any
    rhs: Any
    raw: z3.ExprRef


@dataclass(frozen=True, slots=True)
class Z3OpRealNeg:
    """Unary Real negation `-arg`."""

    arg: Any
    raw: z3.ExprRef


@dataclass(frozen=True, slots=True)
class Z3OpBridge:
    """Unary UF domain bridge (`u_f32_to_bv32`, `u_bv32_to_f32`, `u_f16_to_bv16`, `u_bv16_to_f16`, etc.)."""

    name: str
    arg: Any
    raw: z3.ExprRef


@dataclass(frozen=True, slots=True)
class Z3Other:
    """Unmatched Z3 expression (symbolic constant, Select, Store, or general UF)."""

    raw: z3.ExprRef


Z3Node = Union[
    Z3BvConst,
    Z3RealConst,
    Z3IntConst,
    Z3BoolConst,
    Z3OpExtract,
    Z3OpZeroExt,
    Z3OpSignExt,
    Z3OpConcat,
    Z3OpIte,
    Z3OpEq,
    Z3OpDistinct,
    Z3OpNot,
    Z3OpAnd,
    Z3OpOr,
    Z3OpBvMul,
    Z3OpBvAdd,
    Z3OpBvSub,
    Z3OpBvShl,
    Z3OpBvLShr,
    Z3OpBvAShr,
    Z3OpBvAnd,
    Z3OpBvOr,
    Z3OpBvXor,
    Z3OpBvCmp,
    Z3OpBv2Int,
    Z3OpInt2Bv,
    Z3OpToReal,
    Z3OpToInt,
    Z3OpRealAdd,
    Z3OpRealMul,
    Z3OpRealSub,
    Z3OpRealDiv,
    Z3OpRealNeg,
    Z3OpBridge,
    Z3Other,
]

BRIDGE_NAMES = frozenset(
    {
        "u_f32_to_bv32",
        "u_bv32_to_f32",
        "u_f16_to_bv16",
        "u_bv16_to_f16",
        "u_f64_to_bv64",
        "u_bv64_to_f64",
        "real2bv32",
        "bv32_to_real",
    }
)

_BV_CMP_KIND_TO_OP = {
    z3.Z3_OP_SLT: "slt",
    z3.Z3_OP_SLEQ: "sleq",
    z3.Z3_OP_SGT: "sgt",
    z3.Z3_OP_SGEQ: "sgeq",
    z3.Z3_OP_ULT: "ult",
    z3.Z3_OP_ULEQ: "uleq",
    z3.Z3_OP_UGT: "ugt",
    z3.Z3_OP_UGEQ: "ugeq",
}


def unwrap(node: Any) -> z3.ExprRef:
    """Return the underlying `z3.ExprRef` from either a `Z3Node` view or raw `ExprRef`."""
    if isinstance(node, z3.ExprRef):
        return node
    return node.raw


def view(expr: z3.ExprRef, depth: int = 2) -> Z3Node:
    """
    Wrap `expr` into a typed `Z3Node` dataclass for Python `match`/`case` rules.

    When `depth > 1`, unary/binary/ternary child operands are recursively wrapped
    up to `depth - 1` so nested patterns such as
    `Z3OpExtract(hi, 0, Z3OpZeroExt(_, inner))` or
    `Z3OpBridge("u_f32_to_bv32", Z3OpBridge("u_bv32_to_f32", x))`
    match directly in a single `case` clause while guaranteeing O(1) depth.
    """
    if isinstance(expr, z3.BitVecNumRef):
        return Z3BvConst(int(expr.as_long()), expr.size(), expr)
    if isinstance(expr, z3.RatNumRef):
        return Z3RealConst(expr.numerator_as_long(), expr.denominator_as_long(), expr)
    if isinstance(expr, z3.IntNumRef):
        return Z3IntConst(expr.as_long(), expr)
    if z3.is_true(expr):
        return Z3BoolConst(True, expr)
    if z3.is_false(expr):
        return Z3BoolConst(False, expr)

    if not z3.is_app(expr):
        return Z3Other(expr)

    decl = expr.decl()
    kind = decl.kind()
    nargs = expr.num_args()

    def sub(e: z3.ExprRef) -> Any:
        if depth > 1:
            return view(e, depth - 1)
        if isinstance(e, z3.BitVecNumRef):
            return Z3BvConst(int(e.as_long()), e.size(), e)
        if isinstance(e, z3.RatNumRef):
            return Z3RealConst(e.numerator_as_long(), e.denominator_as_long(), e)
        if isinstance(e, z3.IntNumRef):
            return Z3IntConst(e.as_long(), e)
        if z3.is_true(e):
            return Z3BoolConst(True, e)
        if z3.is_false(e):
            return Z3BoolConst(False, e)
        return e

    if kind == z3.Z3_OP_EXTRACT and nargs == 1:
        hi, lo = expr.params()
        return Z3OpExtract(int(hi), int(lo), sub(expr.arg(0)), expr)
    if kind == z3.Z3_OP_ZERO_EXT and nargs == 1:
        return Z3OpZeroExt(int(expr.params()[0]), sub(expr.arg(0)), expr)
    if kind == z3.Z3_OP_SIGN_EXT and nargs == 1:
        return Z3OpSignExt(int(expr.params()[0]), sub(expr.arg(0)), expr)
    if kind == z3.Z3_OP_CONCAT and nargs >= 2:
        return Z3OpConcat(tuple(expr.arg(i) for i in range(nargs)), expr)

    if kind == z3.Z3_OP_ITE and nargs == 3:
        return Z3OpIte(sub(expr.arg(0)), sub(expr.arg(1)), sub(expr.arg(2)), expr)
    if kind == z3.Z3_OP_EQ and nargs == 2:
        return Z3OpEq(sub(expr.arg(0)), sub(expr.arg(1)), expr)
    if kind == z3.Z3_OP_DISTINCT and nargs == 2:
        return Z3OpDistinct(sub(expr.arg(0)), sub(expr.arg(1)), expr)
    if kind == z3.Z3_OP_NOT and nargs == 1:
        return Z3OpNot(sub(expr.arg(0)), expr)
    if kind == z3.Z3_OP_AND:
        return Z3OpAnd(tuple(expr.arg(i) for i in range(nargs)), expr)
    if kind == z3.Z3_OP_OR:
        return Z3OpOr(tuple(expr.arg(i) for i in range(nargs)), expr)

    if kind in _BV_CMP_KIND_TO_OP and nargs == 2:
        return Z3OpBvCmp(_BV_CMP_KIND_TO_OP[kind], sub(expr.arg(0)), sub(expr.arg(1)), expr)

    if nargs == 2 and isinstance(expr, z3.BitVecRef):
        w = expr.size()
        if kind == z3.Z3_OP_BMUL:
            return Z3OpBvMul(sub(expr.arg(0)), sub(expr.arg(1)), w, expr)
        if kind == z3.Z3_OP_BADD:
            return Z3OpBvAdd(sub(expr.arg(0)), sub(expr.arg(1)), w, expr)
        if kind == z3.Z3_OP_BSUB:
            return Z3OpBvSub(sub(expr.arg(0)), sub(expr.arg(1)), w, expr)
        if kind == z3.Z3_OP_BSHL:
            return Z3OpBvShl(sub(expr.arg(0)), sub(expr.arg(1)), w, expr)
        if kind == z3.Z3_OP_BLSHR:
            return Z3OpBvLShr(sub(expr.arg(0)), sub(expr.arg(1)), w, expr)
        if kind == z3.Z3_OP_BASHR:
            return Z3OpBvAShr(sub(expr.arg(0)), sub(expr.arg(1)), w, expr)
        if kind == z3.Z3_OP_BAND:
            return Z3OpBvAnd(sub(expr.arg(0)), sub(expr.arg(1)), w, expr)
        if kind == z3.Z3_OP_BOR:
            return Z3OpBvOr(sub(expr.arg(0)), sub(expr.arg(1)), w, expr)
        if kind == z3.Z3_OP_BXOR:
            return Z3OpBvXor(sub(expr.arg(0)), sub(expr.arg(1)), w, expr)

    if kind == z3.Z3_OP_BV2INT and nargs == 1:
        return Z3OpBv2Int(sub(expr.arg(0)), expr)
    if kind == z3.Z3_OP_INT2BV and nargs == 1 and isinstance(expr, z3.BitVecRef):
        return Z3OpInt2Bv(sub(expr.arg(0)), expr.size(), expr)
    if kind == z3.Z3_OP_TO_REAL and nargs == 1:
        return Z3OpToReal(sub(expr.arg(0)), expr)
    if kind == z3.Z3_OP_TO_INT and nargs == 1:
        return Z3OpToInt(sub(expr.arg(0)), expr)

    if isinstance(expr, z3.ArithRef) and expr.is_real():
        if kind == z3.Z3_OP_ADD:
            return Z3OpRealAdd(tuple(expr.arg(i) for i in range(nargs)), expr)
        if kind == z3.Z3_OP_MUL:
            return Z3OpRealMul(tuple(expr.arg(i) for i in range(nargs)), expr)
        if kind == z3.Z3_OP_SUB and nargs == 2:
            return Z3OpRealSub(sub(expr.arg(0)), sub(expr.arg(1)), expr)
        if kind == z3.Z3_OP_DIV and nargs == 2:
            return Z3OpRealDiv(sub(expr.arg(0)), sub(expr.arg(1)), expr)
        if kind == z3.Z3_OP_UMINUS and nargs == 1:
            return Z3OpRealNeg(sub(expr.arg(0)), expr)

    if kind == z3.Z3_OP_UNINTERPRETED and nargs == 1:
        name = decl.name()
        if name in BRIDGE_NAMES:
            return Z3OpBridge(name, sub(expr.arg(0)), expr)

    return Z3Other(expr)

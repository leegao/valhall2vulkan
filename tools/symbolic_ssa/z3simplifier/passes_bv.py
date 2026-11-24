#!/usr/bin/env python3
"""
BitVector and Boolean local peephole passes written in the `z3.dsl`
`match`/`case` pattern-matching DSL.

Responsibilities (strictly local single-node folding & algebraic rewrites;
global constant/equality propagation and known-bits lattice analysis live in
`dataflow.py`):
  - `BvWrapperPeepholePass`: collapses `Extract`/`ZeroExt`/`SignExt`/`Concat`
    and `If(b, 1, 0) != 0` boolean-wrapper chains, plus constant-folds BV
    comparisons and `BV2Int`/`Int2BV`/`ToReal`/`ToInt` literals.
  - `BvAlgebraicPass`: folds BV arithmetic/bitwise identities and strength-
    reduces power-of-two `BVMul` to `Shl` (`x * 2^k -> x << k`).
"""

from __future__ import annotations

from typing import Optional, Tuple

from tools.symbolic_ssa import z3

from tools.symbolic_ssa.z3simplifier.ast_utils import rewrite_dag
from tools.symbolic_ssa.z3simplifier.context import SimplificationContext
from tools.symbolic_ssa.z3.dsl import (
    BvCmpOp,
    Z3BoolConst,
    Z3BvConst,
    Z3IntConst,
    Z3OpAnd,
    Z3OpBv2Int,
    Z3OpBvAdd,
    Z3OpBvAnd,
    Z3OpBvAShr,
    Z3OpBvCmp,
    Z3OpBvLShr,
    Z3OpBvMul,
    Z3OpBvOr,
    Z3OpBvShl,
    Z3OpBvSub,
    Z3OpBvXor,
    Z3OpConcat,
    Z3OpDistinct,
    Z3OpEq,
    Z3OpExtract,
    Z3OpInt2Bv,
    Z3OpIte,
    Z3OpNot,
    Z3OpOr,
    Z3RealConst,
    Z3OpSignExt,
    Z3OpToInt,
    Z3OpToReal,
    Z3OpZeroExt,
    unwrap,
    view,
)
from tools.symbolic_ssa.z3simplifier.pass_base import Z3TermPass


def _is_power_of_two(n: int) -> Optional[int]:
    if n > 0 and (n & (n - 1)) == 0:
        return n.bit_length() - 1
    return None


_BV_WRAP_KINDS = frozenset(
    {
        z3.Z3_OP_EXTRACT, z3.Z3_OP_ZERO_EXT, z3.Z3_OP_SIGN_EXT, z3.Z3_OP_CONCAT,
        z3.Z3_OP_BV2INT, z3.Z3_OP_INT2BV, z3.Z3_OP_TO_REAL, z3.Z3_OP_TO_INT,
        z3.Z3_OP_SLT, z3.Z3_OP_SLEQ, z3.Z3_OP_SGT, z3.Z3_OP_SGEQ,
        z3.Z3_OP_ULT, z3.Z3_OP_ULEQ, z3.Z3_OP_UGT, z3.Z3_OP_UGEQ,
        z3.Z3_OP_ITE, z3.Z3_OP_EQ, z3.Z3_OP_DISTINCT,
        z3.Z3_OP_NOT, z3.Z3_OP_AND, z3.Z3_OP_OR,
    }
)


def _bv_wrapper_rule(node: z3.ExprRef) -> Optional[z3.ExprRef]:
    """Declarative `match`/`case` local peephole rules for BV/Bool wrappers and literal conversions."""
    if not z3.is_app(node) or node.decl().kind() not in _BV_WRAP_KINDS:
        return None
    match view(node, depth=2):
        # Extract identity, constant folding, and ZeroExt/SignExt/Concat/ITE unwrapping
        case Z3OpExtract(hi, 0, src) if isinstance(unwrap(src), z3.BitVecRef) and unwrap(src).size() == hi + 1:
            return unwrap(src)
        case Z3OpExtract(hi, lo, Z3BvConst(val, _)):
            w = hi - lo + 1
            return z3.BitVecVal((val >> lo) & ((1 << w) - 1), w)
        case Z3OpExtract(hi, 0, Z3OpZeroExt(_, inner)):
            inner_e = unwrap(inner)
            w = hi + 1
            if isinstance(inner_e, z3.BitVecRef):
                if inner_e.size() == w:
                    return inner_e
                if inner_e.size() > w:
                    return z3.Extract(hi, 0, inner_e)
        case Z3OpExtract(hi, 0, Z3OpSignExt(_, inner)):
            inner_e = unwrap(inner)
            if isinstance(inner_e, z3.BitVecRef) and inner_e.size() == hi + 1:
                return inner_e
        case Z3OpExtract(hi, 0, Z3OpConcat(args)):
            lsb = args[-1]
            w = hi + 1
            if isinstance(lsb, z3.BitVecRef):
                if lsb.size() == w:
                    return lsb
                if lsb.size() > w:
                    return z3.Extract(hi, 0, lsb)
        case Z3OpExtract(hi, lo, Z3OpIte(cond, Z3BvConst(tc, _), Z3BvConst(fc, _))):
            w = hi - lo + 1
            mask = (1 << w) - 1
            nt = z3.BitVecVal((tc >> lo) & mask, w)
            nf = z3.BitVecVal((fc >> lo) & mask, w)
            return nt if nt.eq(nf) else z3.If(unwrap(cond), nt, nf)

        # Constant-fold ZeroExt, SignExt, Concat
        case Z3OpZeroExt(k, Z3BvConst(val, w)):
            return z3.BitVecVal(val, w + k)
        case Z3OpSignExt(k, Z3BvConst(_, w) as c):
            nw = w + k
            return z3.BitVecVal(c.as_signed_long() & ((1 << nw) - 1), nw)
        case Z3OpConcat(args) if all(isinstance(a, z3.BitVecNumRef) for a in args):
            acc, tot_w = 0, 0
            for a in args:
                aw = a.size()
                acc = (acc << aw) | a.as_long()
                tot_w += aw
            return z3.BitVecVal(acc, tot_w)

        # Constant-fold BV/Int/Real conversions
        case Z3OpBv2Int(Z3BvConst(val, _)):
            return z3.IntVal(val)
        case Z3OpInt2Bv(Z3IntConst(val), w):
            return z3.BitVecVal(val & ((1 << w) - 1), w)
        case Z3OpToReal(Z3IntConst(val)):
            return z3.RealVal(val)
        case Z3OpToInt(Z3RealConst(num, den)):
            return z3.IntVal(num // den)

        # Constant-fold BV comparisons
        case Z3OpBvCmp(op, Z3BvConst(va, _) as ca, Z3BvConst(vb, _) as cb):
            if op.is_signed:
                sa, sb = ca.as_signed_long(), cb.as_signed_long()
                if op is BvCmpOp.SLT: return z3.BoolVal(sa < sb)
                if op is BvCmpOp.SLEQ: return z3.BoolVal(sa <= sb)
                if op is BvCmpOp.SGT: return z3.BoolVal(sa > sb)
                return z3.BoolVal(sa >= sb)
            if op is BvCmpOp.ULT: return z3.BoolVal(va < vb)
            if op is BvCmpOp.ULEQ: return z3.BoolVal(va <= vb)
            if op is BvCmpOp.UGT: return z3.BoolVal(va > vb)
            return z3.BoolVal(va >= vb)

        # ITE simplification
        case Z3OpIte(Z3BoolConst(True), t_arm, _):
            return unwrap(t_arm)
        case Z3OpIte(Z3BoolConst(False), _, f_arm):
            return unwrap(f_arm)
        case Z3OpIte(_, t_arm, f_arm) if z3.expr_eq(unwrap(t_arm), unwrap(f_arm)):
            return unwrap(t_arm)
        case Z3OpIte(cond, Z3BoolConst(True), Z3BoolConst(False)):
            return unwrap(cond)
        case Z3OpIte(cond, Z3BoolConst(False), Z3BoolConst(True)):
            return z3.Not(unwrap(cond))

        # Eq / Distinct constant folding and `If(c, 1, 0) != 0` unwrapping
        case Z3OpEq(Z3BvConst(ca, _), Z3BvConst(cb, _)):
            return z3.BoolVal(ca == cb)
        case Z3OpDistinct(Z3BvConst(ca, _), Z3BvConst(cb, _)):
            return z3.BoolVal(ca != cb)
        case Z3OpEq(a, b) if z3.expr_eq(unwrap(a), unwrap(b)):
            return z3.BoolVal(True)
        case Z3OpDistinct(a, b) if z3.expr_eq(unwrap(a), unwrap(b)):
            return z3.BoolVal(False)
        case (
            Z3OpEq(Z3BvConst(0, _), Z3OpIte(oc, Z3BvConst(tc, _), Z3BvConst(fc, _)))
            | Z3OpEq(Z3OpIte(oc, Z3BvConst(tc, _), Z3BvConst(fc, _)), Z3BvConst(0, _))
        ):
            t_nz, f_nz = tc != 0, fc != 0
            if t_nz and not f_nz: return z3.Not(unwrap(oc))
            if not t_nz and f_nz: return unwrap(oc)
            return z3.BoolVal(not (t_nz and f_nz))
        case (
            Z3OpDistinct(Z3BvConst(0, _), Z3OpIte(oc, Z3BvConst(tc, _), Z3BvConst(fc, _)))
            | Z3OpDistinct(Z3OpIte(oc, Z3BvConst(tc, _), Z3BvConst(fc, _)), Z3BvConst(0, _))
        ):
            t_nz, f_nz = tc != 0, fc != 0
            if t_nz and not f_nz: return unwrap(oc)
            if not t_nz and f_nz: return z3.Not(unwrap(oc))
            return z3.BoolVal(t_nz and f_nz)

        # Boolean Not / And / Or
        case Z3OpNot(Z3BoolConst(v)):
            return z3.BoolVal(not v)
        case Z3OpNot(Z3OpNot(inner)):
            return unwrap(inner)
        case Z3OpAnd(args):
            kept = []
            changed = False
            for ch in args:
                if z3.is_false(ch):
                    return z3.BoolVal(False)
                if z3.is_true(ch):
                    changed = True
                    continue
                kept.append(ch)
            if changed:
                if not kept:
                    return z3.BoolVal(True)
                return kept[0] if len(kept) == 1 else z3.And(*kept)
        case Z3OpOr(args):
            kept = []
            changed = False
            for ch in args:
                if z3.is_true(ch):
                    return z3.BoolVal(True)
                if z3.is_false(ch):
                    changed = True
                    continue
                kept.append(ch)
            if changed:
                if not kept:
                    return z3.BoolVal(False)
                return kept[0] if len(kept) == 1 else z3.Or(*kept)

    return None


_BV_ALG_KINDS = frozenset(
    {
        z3.Z3_OP_BMUL, z3.Z3_OP_BADD, z3.Z3_OP_BSUB,
        z3.Z3_OP_BSHL, z3.Z3_OP_BLSHR, z3.Z3_OP_BASHR,
        z3.Z3_OP_BAND, z3.Z3_OP_BOR, z3.Z3_OP_BXOR,
    }
)


def _bv_algebraic_rule(node: z3.ExprRef) -> Optional[z3.ExprRef]:
    """Declarative `match`/`case` algebraic & strength-reduction rules for BitVec arithmetic."""
    if not isinstance(node, z3.BitVecRef) or not z3.is_app(node) or node.decl().kind() not in _BV_ALG_KINDS:
        return None
    w = node.size()
    mask = (1 << w) - 1

    match view(node, depth=2):
        # BVMul: const fold, 0/1 identities, and power-of-two -> Shl
        case Z3OpBvMul(Z3BvConst(ca, _), Z3BvConst(cb, _)):
            return z3.BitVecVal((ca * cb) & mask, w)
        case Z3OpBvMul(Z3BvConst(0, _), _) | Z3OpBvMul(_, Z3BvConst(0, _)):
            return z3.BitVecVal(0, w)
        case Z3OpBvMul(Z3BvConst(1, _), rhs):
            return unwrap(rhs)
        case Z3OpBvMul(lhs, Z3BvConst(1, _)):
            return unwrap(lhs)
        case Z3OpBvMul(lhs, Z3BvConst(cb, _)) if (sh := _is_power_of_two(cb)) is not None and sh < w:
            return z3.Z3OpBvShl(unwrap(lhs), z3.BitVecVal(sh, w), w)
        case Z3OpBvMul(Z3BvConst(ca, _), rhs) if (sh := _is_power_of_two(ca)) is not None and sh < w:
            return z3.Z3OpBvShl(unwrap(rhs), z3.BitVecVal(sh, w), w)

        # BVAdd: const fold, 0 identity, and (x + c1) + c2 -> x + (c1 + c2)
        case Z3OpBvAdd(Z3BvConst(ca, _), Z3BvConst(cb, _)):
            return z3.BitVecVal((ca + cb) & mask, w)
        case Z3OpBvAdd(Z3BvConst(0, _), rhs):
            return unwrap(rhs)
        case Z3OpBvAdd(lhs, Z3BvConst(0, _)):
            return unwrap(lhs)
        case Z3OpBvAdd(Z3OpBvAdd(a0, Z3BvConst(c1, _)), Z3BvConst(c2, _)):
            csum = (c1 + c2) & mask
            if csum == 0:
                return unwrap(a0)
            base0 = unwrap(a0)
            rhs_c = z3.BitVecVal(csum, w)
            return (base0 + rhs_c) if isinstance(base0, z3.Z3Expr) else (base0 + z3.lower(rhs_c))

        # BVSub
        case Z3OpBvSub(Z3BvConst(ca, _), Z3BvConst(cb, _)):
            return z3.BitVecVal((ca - cb) & mask, w)
        case Z3OpBvSub(lhs, Z3BvConst(0, _)):
            return unwrap(lhs)
        case Z3OpBvSub(lhs, rhs) if z3.expr_eq(unwrap(lhs), unwrap(rhs)):
            return z3.BitVecVal(0, w)

        # Shifts: Shl, LShR, AShR
        case Z3OpBvShl(Z3BvConst(ca, _), Z3BvConst(cb, _)):
            return z3.BitVecVal(0 if cb >= w else ((ca << cb) & mask), w)
        case Z3OpBvShl(lhs, Z3BvConst(0, _)):
            return unwrap(lhs)
        case Z3OpBvShl(_, Z3BvConst(cb, _)) if cb >= w:
            return z3.BitVecVal(0, w)
        case Z3OpBvShl(Z3OpBvShl(x, Z3BvConst(k1, _)), Z3BvConst(k2, _)):
            tot = k1 + k2
            return z3.BitVecVal(0, w) if tot >= w else (unwrap(x) << z3.BitVecVal(tot, w))

        case Z3OpBvLShr(Z3BvConst(ca, _), Z3BvConst(cb, _)):
            return z3.BitVecVal(0 if cb >= w else (ca >> cb), w)
        case Z3OpBvLShr(lhs, Z3BvConst(0, _)):
            return unwrap(lhs)
        case Z3OpBvLShr(_, Z3BvConst(cb, _)) if cb >= w:
            return z3.BitVecVal(0, w)

        case Z3OpBvAShr(lhs, Z3BvConst(0, _)):
            return unwrap(lhs)

        # Bitwise And / Or / Xor
        case Z3OpBvAnd(Z3BvConst(ca, _), Z3BvConst(cb, _)):
            return z3.BitVecVal(ca & cb, w)
        case Z3OpBvAnd(Z3BvConst(0, _), _) | Z3OpBvAnd(_, Z3BvConst(0, _)):
            return z3.BitVecVal(0, w)
        case Z3OpBvAnd(Z3BvConst(ca, _), rhs) if ca == mask:
            return unwrap(rhs)
        case Z3OpBvAnd(lhs, Z3BvConst(cb, _)) if cb == mask:
            return unwrap(lhs)
        case Z3OpBvAnd(lhs, rhs) if z3.expr_eq(unwrap(lhs), unwrap(rhs)):
            return unwrap(lhs)

        case Z3OpBvOr(Z3BvConst(ca, _), Z3BvConst(cb, _)):
            return z3.BitVecVal(ca | cb, w)
        case Z3OpBvOr(Z3BvConst(0, _), rhs):
            return unwrap(rhs)
        case Z3OpBvOr(lhs, Z3BvConst(0, _)):
            return unwrap(lhs)
        case Z3OpBvOr(Z3BvConst(ca, _), _) | Z3OpBvOr(_, Z3BvConst(ca, _)) if ca == mask:
            return z3.BitVecVal(mask, w)
        case Z3OpBvOr(lhs, rhs) if z3.expr_eq(unwrap(lhs), unwrap(rhs)):
            return unwrap(lhs)

        case Z3OpBvXor(Z3BvConst(ca, _), Z3BvConst(cb, _)):
            return z3.BitVecVal(ca ^ cb, w)
        case Z3OpBvXor(Z3BvConst(0, _), rhs):
            return unwrap(rhs)
        case Z3OpBvXor(lhs, Z3BvConst(0, _)):
            return unwrap(lhs)
        case Z3OpBvXor(lhs, rhs) if z3.expr_eq(unwrap(lhs), unwrap(rhs)):
            return z3.BitVecVal(0, w)

    return None


class BvWrapperPeepholePass(Z3TermPass):
    """Collapses nested `Extract`/`ZeroExt`/`Concat`/`If(b, 1, 0) != 0` boolean wrappers."""

    name = "bv_wrapper_peephole"
    is_equivalence_preserving = True

    def rewrite_term(
        self,
        expr: z3.ExprRef,
        ctx: SimplificationContext,
    ) -> Tuple[z3.ExprRef, int]:
        return rewrite_dag(expr, _bv_wrapper_rule)


class BvAlgebraicPass(Z3TermPass):
    """Normalizes integer `BitVec` arithmetic and eliminates power-of-two `BVMul`s."""

    name = "bv_algebraic"
    is_equivalence_preserving = True

    def rewrite_term(
        self,
        expr: z3.ExprRef,
        ctx: SimplificationContext,
    ) -> Tuple[z3.ExprRef, int]:
        return rewrite_dag(expr, _bv_algebraic_rule)

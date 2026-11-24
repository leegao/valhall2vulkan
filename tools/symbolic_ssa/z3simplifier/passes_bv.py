"""
BitVector and Boolean local peephole passes written in the `dsl.dsl`
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

from tools.symbolic_ssa.z3simplifier.ast_utils import rewrite_dag
from tools.symbolic_ssa.z3simplifier.context import SimplificationContext
from tools.symbolic_ssa.z3simplifier.pass_base import Z3TermPass
from tools.symbolic_ssa.z3utils import dsl
from tools.symbolic_ssa.z3utils.dsl import (
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
    Z3OpSignExt,
    Z3OpToInt,
    Z3OpToReal,
    Z3OpZeroExt,
    Z3RealConst,
)


def _unwrap_f32_to_bv32(expr: dsl.ExprRef) -> tuple[dsl.ExprRef, dsl.ExprRef] | None:
    """If `expr` is `u_f32_to_bv32(r)`, return `(expr, r)`."""
    if dsl.is_app(expr) and expr.num_args() == 1 and expr.decl().name() == "u_f32_to_bv32":
        return expr, expr.arg(0)
    return None


def _rebuild_f32_to_bv32(bridge_node: dsl.ExprRef, inner: dsl.ExprRef) -> dsl.ExprRef:
    if isinstance(bridge_node, dsl.Z3OpBridge):
        return dsl.Z3OpBridge(bridge_node.op, inner)
    return bridge_node.decl()(inner)


def _is_power_of_two(n: int) -> int | None:
    if n > 0 and (n & (n - 1)) == 0:
        return n.bit_length() - 1
    return None


_BV_WRAP_KINDS = frozenset(
    {
        dsl.Z3OpKind.EXTRACT, dsl.Z3OpKind.ZERO_EXT, dsl.Z3OpKind.SIGN_EXT, dsl.Z3OpKind.CONCAT,
        dsl.Z3OpKind.BV2INT, dsl.Z3OpKind.INT2BV, dsl.Z3OpKind.TO_REAL, dsl.Z3OpKind.TO_INT,
        dsl.Z3OpKind.SLT, dsl.Z3OpKind.SLEQ, dsl.Z3OpKind.SGT, dsl.Z3OpKind.SGEQ,
        dsl.Z3OpKind.ULT, dsl.Z3OpKind.ULEQ, dsl.Z3OpKind.UGT, dsl.Z3OpKind.UGEQ,
        dsl.Z3OpKind.ITE, dsl.Z3OpKind.EQ, dsl.Z3OpKind.DISTINCT,
        dsl.Z3OpKind.NOT, dsl.Z3OpKind.AND, dsl.Z3OpKind.OR,
    }
)


def _bv_wrapper_rule(node: dsl.ExprRef) -> dsl.ExprRef | None:
    """Declarative `match`/`case` local peephole rules for BV/Bool wrappers and literal conversions."""
    if not isinstance(node, dsl.Z3Expr):
        node = dsl.view(node)
    if not dsl.is_app(node) or node.decl().kind() not in _BV_WRAP_KINDS:
        return None
    match node:
        # Extract identity, constant folding, and ZeroExt/SignExt/Concat/ITE unwrapping
        case Z3OpExtract(hi, 0, src) if dsl.is_bv(src) and src.size() == hi + 1:
            return src
        case Z3OpExtract(hi, lo, Z3BvConst(val, _)):
            w = hi - lo + 1
            return dsl.BitVecVal((val >> lo) & ((1 << w) - 1), w)
        case Z3OpExtract(hi, 0, Z3OpZeroExt(_, inner)):
            w = hi + 1
            if dsl.is_bv(inner):
                if inner.size() == w:
                    return inner
                if inner.size() > w:
                    return dsl.Extract(hi, 0, inner)
        case Z3OpExtract(hi, 0, Z3OpSignExt(_, inner)):
            if dsl.is_bv(inner) and inner.size() == hi + 1:
                return inner
        case Z3OpExtract(hi, 0, Z3OpConcat(args)):
            lsb = args[-1]
            w = hi + 1
            if dsl.is_bv(lsb):
                if lsb.size() == w:
                    return lsb
                if lsb.size() > w:
                    return dsl.Extract(hi, 0, lsb)
        case Z3OpExtract(hi, lo, Z3OpIte(cond, Z3BvConst(tc, _), Z3BvConst(fc, _))):
            w = hi - lo + 1
            mask = (1 << w) - 1
            nt = dsl.BitVecVal((tc >> lo) & mask, w)
            nf = dsl.BitVecVal((fc >> lo) & mask, w)
            return nt if nt.eq(nf) else dsl.If(cond, nt, nf)

        # Constant-fold ZeroExt, SignExt, Concat
        case Z3OpZeroExt(k, Z3BvConst(val, w)):
            return dsl.BitVecVal(val, w + k)
        case Z3OpSignExt(k, Z3BvConst(_, w) as c):
            nw = w + k
            return dsl.BitVecVal(c.as_signed_long() & ((1 << nw) - 1), nw)
        case Z3OpConcat(args) if all(isinstance(a, dsl.Z3BvConst) for a in args):
            acc, tot_w = 0, 0
            for a in args:
                aw = a.size()
                acc = (acc << aw) | a.as_long()
                tot_w += aw
            return dsl.BitVecVal(acc, tot_w)

        # Constant-fold BV/Int/Real conversions
        case Z3OpBv2Int(Z3BvConst(val, _)):
            return dsl.IntVal(val)
        case Z3OpInt2Bv(Z3IntConst(val), w):
            return dsl.BitVecVal(val & ((1 << w) - 1), w)
        case Z3OpToReal(Z3IntConst(val)):
            return dsl.RealVal(val)
        case Z3OpToInt(Z3RealConst(num, den)):
            return dsl.IntVal(num // den)

        # Constant-fold BV comparisons
        case Z3OpBvCmp(op, Z3BvConst(va, _) as ca, Z3BvConst(vb, _) as cb):
            if op.is_signed:
                sa, sb = ca.as_signed_long(), cb.as_signed_long()
                if op is BvCmpOp.SLT: return dsl.BoolVal(sa < sb)
                if op is BvCmpOp.SLEQ: return dsl.BoolVal(sa <= sb)
                if op is BvCmpOp.SGT: return dsl.BoolVal(sa > sb)
                return dsl.BoolVal(sa >= sb)
            if op is BvCmpOp.ULT: return dsl.BoolVal(va < vb)
            if op is BvCmpOp.ULEQ: return dsl.BoolVal(va <= vb)
            if op is BvCmpOp.UGT: return dsl.BoolVal(va > vb)
            return dsl.BoolVal(va >= vb)

        # ITE simplification
        case Z3OpIte(Z3BoolConst(True), t_arm, _):
            return t_arm
        case Z3OpIte(Z3BoolConst(False), _, f_arm):
            return f_arm
        case Z3OpIte(_, t_arm, f_arm) if dsl.expr_eq(t_arm, f_arm):
            return t_arm
        case Z3OpIte(cond, Z3BoolConst(True), Z3BoolConst(False)):
            return cond
        case Z3OpIte(cond, Z3BoolConst(False), Z3BoolConst(True)):
            return dsl.Not(cond)

        # Eq / Distinct constant folding and `If(c, 1, 0) != 0` unwrapping
        case Z3OpEq(Z3BvConst(ca, _), Z3BvConst(cb, _)):
            return dsl.BoolVal(ca == cb)
        case Z3OpDistinct(Z3BvConst(ca, _), Z3BvConst(cb, _)):
            return dsl.BoolVal(ca != cb)
        case Z3OpEq(a, b) if dsl.expr_eq(a, b):
            return dsl.BoolVal(True)
        case Z3OpDistinct(a, b) if dsl.expr_eq(a, b):
            return dsl.BoolVal(False)
        case (
            Z3OpEq(Z3BvConst(0, _), Z3OpIte(oc, Z3BvConst(tc, _), Z3BvConst(fc, _)))
            | Z3OpEq(Z3OpIte(oc, Z3BvConst(tc, _), Z3BvConst(fc, _)), Z3BvConst(0, _))
        ):
            t_nz, f_nz = tc != 0, fc != 0
            if t_nz and not f_nz: return dsl.Not(oc)
            if not t_nz and f_nz: return oc
            return dsl.BoolVal(not (t_nz and f_nz))
        case (
            Z3OpDistinct(Z3BvConst(0, _), Z3OpIte(oc, Z3BvConst(tc, _), Z3BvConst(fc, _)))
            | Z3OpDistinct(Z3OpIte(oc, Z3BvConst(tc, _), Z3BvConst(fc, _)), Z3BvConst(0, _))
        ):
            t_nz, f_nz = tc != 0, fc != 0
            if t_nz and not f_nz: return oc
            if not t_nz and f_nz: return dsl.Not(oc)
            return dsl.BoolVal(t_nz and f_nz)

        # Boolean Not / And / Or
        case Z3OpNot(Z3BoolConst(v)):
            return dsl.BoolVal(not v)
        case Z3OpNot(Z3OpNot(inner)):
            return inner
        case Z3OpAnd(args):
            kept = []
            changed = False
            for ch in args:
                if dsl.is_false(ch):
                    return dsl.BoolVal(False)
                if dsl.is_true(ch):
                    changed = True
                    continue
                kept.append(ch)
            if changed:
                if not kept:
                    return dsl.BoolVal(True)
                return kept[0] if len(kept) == 1 else dsl.And(*kept)
        case Z3OpOr(args):
            kept = []
            changed = False
            for ch in args:
                if dsl.is_true(ch):
                    return dsl.BoolVal(True)
                if dsl.is_false(ch):
                    changed = True
                    continue
                kept.append(ch)
            if changed:
                if not kept:
                    return dsl.BoolVal(False)
                return kept[0] if len(kept) == 1 else dsl.Or(*kept)

    return None


_BV_ALG_KINDS = frozenset(
    {
        dsl.Z3OpKind.BMUL, dsl.Z3OpKind.BADD, dsl.Z3OpKind.BSUB,
        dsl.Z3OpKind.BSHL, dsl.Z3OpKind.BLSHR, dsl.Z3OpKind.BASHR,
        dsl.Z3OpKind.BAND, dsl.Z3OpKind.BOR, dsl.Z3OpKind.BXOR,
    }
)


def _bv_algebraic_rule(node: dsl.ExprRef) -> dsl.ExprRef | None:
    """Declarative `match`/`case` algebraic & strength-reduction rules for BitVec arithmetic."""
    if not isinstance(node, dsl.Z3Expr):
        node = dsl.view(node)
    if not dsl.is_bv(node) or not dsl.is_app(node) or node.decl().kind() not in _BV_ALG_KINDS:
        return None
    w = node.size()
    mask = (1 << w) - 1

    match node:
        # BVMul: const fold, 0/1 identities, and power-of-two -> Shl
        case Z3OpBvMul(Z3BvConst(ca, _), Z3BvConst(cb, _)):
            return dsl.BitVecVal((ca * cb) & mask, w)
        case Z3OpBvMul(Z3BvConst(0, _), _) | Z3OpBvMul(_, Z3BvConst(0, _)):
            return dsl.BitVecVal(0, w)
        case Z3OpBvMul(Z3BvConst(1, _), rhs):
            return rhs
        case Z3OpBvMul(lhs, Z3BvConst(1, _)):
            return lhs
        case Z3OpBvMul(lhs, Z3BvConst(cb, _)) if (sh := _is_power_of_two(cb)) is not None and sh < w:
            return dsl.Z3OpBvShl(lhs, dsl.BitVecVal(sh, w), w)
        case Z3OpBvMul(Z3BvConst(ca, _), rhs) if (sh := _is_power_of_two(ca)) is not None and sh < w:
            return dsl.Z3OpBvShl(rhs, dsl.BitVecVal(sh, w), w)

        # BVAdd: const fold, 0 identity, and (x + c1) + c2 -> x + (c1 + c2)
        case Z3OpBvAdd(Z3BvConst(ca, _), Z3BvConst(cb, _)):
            return dsl.BitVecVal((ca + cb) & mask, w)
        case Z3OpBvAdd(Z3BvConst(0, _), rhs):
            return rhs
        case Z3OpBvAdd(lhs, Z3BvConst(0, _)):
            return lhs
        case Z3OpBvAdd(Z3OpBvAdd(a0, Z3BvConst(c1, _)), Z3BvConst(c2, _)):
            csum = (c1 + c2) & mask
            if csum == 0:
                return a0
            return dsl.Z3OpBvAdd(a0, dsl.BitVecVal(csum, w), w)

        # BVSub
        case Z3OpBvSub(Z3BvConst(ca, _), Z3BvConst(cb, _)):
            return dsl.BitVecVal((ca - cb) & mask, w)
        case Z3OpBvSub(lhs, Z3BvConst(0, _)):
            return lhs
        case Z3OpBvSub(lhs, rhs) if dsl.expr_eq(lhs, rhs):
            return dsl.BitVecVal(0, w)

        # Shifts: Shl, LShR, AShR
        case Z3OpBvShl(Z3BvConst(ca, _), Z3BvConst(cb, _)):
            return dsl.BitVecVal(0 if cb >= w else ((ca << cb) & mask), w)
        case Z3OpBvShl(lhs, Z3BvConst(0, _)):
            return lhs
        case Z3OpBvShl(_, Z3BvConst(cb, _)) if cb >= w:
            return dsl.BitVecVal(0, w)
        case Z3OpBvShl(Z3OpBvShl(x, Z3BvConst(k1, _)), Z3BvConst(k2, _)):
            tot = k1 + k2
            return dsl.BitVecVal(0, w) if tot >= w else (x << dsl.BitVecVal(tot, w))

        case Z3OpBvLShr(Z3BvConst(ca, _), Z3BvConst(cb, _)):
            return dsl.BitVecVal(0 if cb >= w else (ca >> cb), w)
        case Z3OpBvLShr(lhs, Z3BvConst(0, _)):
            return lhs
        case Z3OpBvLShr(_, Z3BvConst(cb, _)) if cb >= w:
            return dsl.BitVecVal(0, w)

        case Z3OpBvAShr(lhs, Z3BvConst(0, _)):
            return lhs

        # Bitwise And / Or / Xor
        case Z3OpBvAnd(Z3BvConst(ca, _), Z3BvConst(cb, _)):
            return dsl.BitVecVal(ca & cb, w)
        case Z3OpBvAnd(Z3BvConst(0, _), _) | Z3OpBvAnd(_, Z3BvConst(0, _)):
            return dsl.BitVecVal(0, w)
        case Z3OpBvAnd(Z3BvConst(ca, _), rhs) if ca == mask:
            return rhs
        case Z3OpBvAnd(lhs, Z3BvConst(cb, _)) if cb == mask:
            return lhs
        case Z3OpBvAnd(lhs, rhs) if dsl.expr_eq(lhs, rhs):
            return lhs
        case Z3OpBvAnd(lhs, Z3BvConst(0x7FFFFFFF, 32)) if (ub := _unwrap_f32_to_bv32(lhs)) is not None:
            br, r = ub
            return _rebuild_f32_to_bv32(br, dsl.If(r < dsl.RealVal(0), -r, r))
        case Z3OpBvAnd(Z3BvConst(0x7FFFFFFF, 32), rhs) if (ub := _unwrap_f32_to_bv32(rhs)) is not None:
            br, r = ub
            return _rebuild_f32_to_bv32(br, dsl.If(r < dsl.RealVal(0), -r, r))

        case Z3OpBvOr(Z3BvConst(ca, _), Z3BvConst(cb, _)):
            return dsl.BitVecVal(ca | cb, w)
        case Z3OpBvOr(Z3BvConst(0, _), rhs):
            return rhs
        case Z3OpBvOr(lhs, Z3BvConst(0, _)):
            return lhs
        case Z3OpBvOr(Z3BvConst(ca, _), _) | Z3OpBvOr(_, Z3BvConst(ca, _)) if ca == mask:
            return dsl.BitVecVal(mask, w)
        case Z3OpBvOr(lhs, rhs) if dsl.expr_eq(lhs, rhs):
            return lhs

        case Z3OpBvXor(Z3BvConst(ca, _), Z3BvConst(cb, _)):
            return dsl.BitVecVal(ca ^ cb, w)
        case Z3OpBvXor(Z3BvConst(0, _), rhs):
            return rhs
        case Z3OpBvXor(lhs, Z3BvConst(0, _)):
            return lhs
        case Z3OpBvXor(lhs, rhs) if dsl.expr_eq(lhs, rhs):
            return dsl.BitVecVal(0, w)
        case Z3OpBvXor(lhs, Z3BvConst(0x80000000, 32)) if (ub := _unwrap_f32_to_bv32(lhs)) is not None:
            br, r = ub
            return _rebuild_f32_to_bv32(br, -r)
        case Z3OpBvXor(Z3BvConst(0x80000000, 32), rhs) if (ub := _unwrap_f32_to_bv32(rhs)) is not None:
            br, r = ub
            return _rebuild_f32_to_bv32(br, -r)

    return None


class BvWrapperPeepholePass(Z3TermPass):
    """Collapses nested `Extract`/`ZeroExt`/`Concat`/`If(b, 1, 0) != 0` boolean wrappers."""

    name = "bv_wrapper_peephole"
    is_equivalence_preserving = True

    def rewrite_term(
        self,
        expr: dsl.ExprRef,
        ctx: SimplificationContext,
    ) -> tuple[dsl.ExprRef, int]:
        return rewrite_dag(expr, _bv_wrapper_rule)


class BvAlgebraicPass(Z3TermPass):
    """Normalizes integer `BitVec` arithmetic and eliminates power-of-two `BVMul`s."""

    name = "bv_algebraic"
    is_equivalence_preserving = True

    def rewrite_term(
        self,
        expr: dsl.ExprRef,
        ctx: SimplificationContext,
    ) -> tuple[dsl.ExprRef, int]:
        return rewrite_dag(expr, _bv_algebraic_rule)

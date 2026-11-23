#!/usr/bin/env python3
"""
Real-arithmetic, UF-bridge, and polynomial canonicalization passes.

Addresses:
  - Issue 3 (Outer `u_f32_to_bv32` wrapper blocking polynomial canonicalization):
      * `BridgeCancelPass`: cancels round-trip UF bridges anywhere in a term
        (`u_f32_to_bv32(u_bv32_to_f32(x)) -> x`, `u_bv32_to_f32(u_f32_to_bv32(r)) -> r`,
        and the 16-bit / 64-bit / legacy `real2bv32` counterparts).
      * `InjectiveBridgeUnwrapPass`: strips matching deterministic outer
        encoding functions (`u_f32_to_bv32(r1)` vs `u_f32_to_bv32(r2)`, or
        `Concat(hi, u_f32_to_bv32(r1))` after BV-wrapper stripping) so that
        `u_f32_to_bv32(r1) != u_f32_to_bv32(r2)` reduces to `r1 != r2`.
        Sound for UNSAT because `r1 == r2 => f(r1) == f(r2)` for any function `f`.
      * `PolynomialNormalizePass`: when both sides are Z3 `Real` expressions
        whose DAG size is within `ctx.max_som_nodes`, computes
        `z3.simplify(lhs - rhs, som=True, arith_lhs=True)`. Any algebraically
        equivalent FMA / associativity / distributivity lowering collapses to
        `0 == 0` (trivial `UNSAT`) in < 1 ms without invoking the SMT solver.
      * `RealAlgebraicPass`: applies terminating Real arithmetic, reciprocal,
        comparison, and ITE canonicalization identities.
"""

from __future__ import annotations

from fractions import Fraction
import math
import struct
import time
from typing import Dict, List, Optional, Tuple

from tools.symbolic_ssa import z3

from tools.symbolic_ssa.z3simplifier.ast_utils import count_ast_nodes, rewrite_dag
from tools.symbolic_ssa.z3simplifier.context import (
    PassStats,
    SimplificationContext,
    SimplifyQuery,
)
from tools.symbolic_ssa.z3simplifier.pass_base import Z3QueryPass, Z3TermPass


_BRIDGE_INVERSES: Dict[str, str] = {
    "u_f32_to_bv32": "u_bv32_to_f32",
    "u_bv32_to_f32": "u_f32_to_bv32",
    "u_f16_to_bv16": "u_bv16_to_f16",
    "u_bv16_to_f16": "u_f16_to_bv16",
    "u_f64_to_bv64": "u_bv64_to_f64",
    "u_bv64_to_f64": "u_f64_to_bv64",
    "real2bv32": "bv32_to_real",
    "bv32_to_real": "real2bv32",
}

_REAL_TO_BV_BRIDGES = {"u_f32_to_bv32", "u_f16_to_bv16", "u_f64_to_bv64", "real2bv32"}


def _bridge_cancel_rule(node: z3.ExprRef) -> Optional[z3.ExprRef]:
    if not z3.is_app(node):
        return None
    if node.decl().kind() == z3.Z3_OP_ITE and node.num_args() == 3:
        c, t_arm, f_arm = node.arg(0), node.arg(1), node.arg(2)
        t_br = z3.is_app(t_arm) and t_arm.num_args() == 1 and t_arm.decl().name() in _REAL_TO_BV_BRIDGES
        f_br = z3.is_app(f_arm) and f_arm.num_args() == 1 and f_arm.decl().name() in _REAL_TO_BV_BRIDGES
        def _apply_bridge(br_node: z3.ExprRef, inner: z3.ExprRef) -> z3.ExprRef:
            return br_node.decl()(z3.lower(inner))

        if t_br and f_br and t_arm.decl().name() == f_arm.decl().name():
            return _apply_bridge(t_arm, z3.If(c, t_arm.arg(0), f_arm.arg(0)))
        if t_br and t_arm.decl().name() == "u_f32_to_bv32" and z3.is_bv_value(f_arm):
            fv = struct.unpack("<f", struct.pack("<I", f_arm.as_long() & 0xFFFFFFFF))[0]
            if math.isfinite(fv):
                return _apply_bridge(t_arm, z3.If(c, t_arm.arg(0), z3.RealVal(repr(float(fv)))))
        if f_br and f_arm.decl().name() == "u_f32_to_bv32" and z3.is_bv_value(t_arm):
            tv = struct.unpack("<f", struct.pack("<I", t_arm.as_long() & 0xFFFFFFFF))[0]
            if math.isfinite(tv):
                return _apply_bridge(f_arm, z3.If(c, z3.RealVal(repr(float(tv))), f_arm.arg(0)))
        return None
    if node.num_args() != 1:
        return None
    outer_name = node.decl().name()
    inv_name = _BRIDGE_INVERSES.get(outer_name)
    if inv_name is None:
        return None
    arg0 = node.arg(0)
    if (
        outer_name == "u_bv32_to_f32"
        and z3.is_app(arg0)
        and arg0.decl().kind() == z3.Z3_OP_ITE
        and arg0.num_args() == 3
    ):
        c, t_arm, f_arm = arg0.arg(0), arg0.arg(1), arg0.arg(2)
        t_is_br = z3.is_app(t_arm) and t_arm.num_args() == 1 and t_arm.decl().name() == "u_f32_to_bv32"
        f_is_br = z3.is_app(f_arm) and f_arm.num_args() == 1 and f_arm.decl().name() == "u_f32_to_bv32"
        if t_is_br or f_is_br or z3.is_bv_value(t_arm) or z3.is_bv_value(f_arm):
            fn = node.decl()
            nt = t_arm.arg(0) if t_is_br else fn(z3.lower(t_arm))
            nf = f_arm.arg(0) if f_is_br else fn(z3.lower(f_arm))
            return z3.If(c, nt, nf)
    if z3.is_app(arg0) and arg0.num_args() == 1 and arg0.decl().name() == inv_name:
        return arg0.arg(0)
    if outer_name in ("u_bv32_to_f32", "bv32_to_real") and z3.is_bv_value(arg0):
        fval = struct.unpack("<f", struct.pack("<I", arg0.as_long() & 0xFFFFFFFF))[0]
        if math.isfinite(fval):
            return z3.RealVal(repr(float(fval)))
    if outer_name in ("u_f32_to_bv32", "real2bv32") and z3.is_rational_value(arg0):
        fval = float(arg0.numerator_as_long()) / float(arg0.denominator_as_long())
        try:
            u32 = struct.unpack("<I", struct.pack("<f", fval))[0]
        except OverflowError:
            u32 = 0xFF800000 if fval < 0 else 0x7F800000
        return z3.BitVecVal(u32, 32)
    return None


class PreconditionConstSubPass(Z3QueryPass):
    """
    Propagates constant-equality preconditions (`sym == Const` or `Select(arr, k) == Const`)
    into `(lhs, rhs, guard)` before local simplification and bridge folding.

    For example, Vertex `gl_mali_ViewportTransform` preconditions assert
    `fau_sym[0] == 0x3F800000` (1.0f) and `fau_sym[4] == 0x0` (0.0f). Substituting
    these into `u_bv32_to_f32(fau_sym[0])` lets `BridgeCancelPass` fold it to
    `RealVal(1)` and collapse `x * 1.0 + 0.0 -> x` in < 1 ms.
    """

    name = "precondition_const_sub"
    is_equivalence_preserving = True

    @staticmethod
    def _is_const_literal(e: z3.ExprRef) -> bool:
        return (
            z3.is_bv_value(e)
            or z3.is_rational_value(e)
            or z3.is_int_value(e)
            or z3.is_true(e)
            or z3.is_false(e)
        )

    @staticmethod
    def _is_substitutable_lhs(e: z3.ExprRef) -> bool:
        if z3.is_const(e) and e.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            return True
        if z3.is_app(e) and e.decl().kind() == z3.Z3_OP_SELECT and e.num_args() == 2:
            return z3.is_bv_value(e.arg(1)) or z3.is_int_value(e.arg(1))
        return False

    def run(
        self,
        query: SimplifyQuery,
        ctx: SimplificationContext,
    ) -> Tuple[SimplifyQuery, PassStats]:
        t0 = time.perf_counter()
        if query.is_trivially_unsat() or not query.preconditions:
            return query, PassStats(self.name, False, 0, 0.0)

        subs: List[Tuple[z3.ExprRef, z3.ExprRef]] = []
        seen_ids = set()
        work: List[z3.ExprRef] = list(query.preconditions)
        if query.guard is not None:
            work.append(query.guard)
        atoms: List[z3.ExprRef] = []
        while work:
            cur = work.pop()
            if z3.is_app(cur) and cur.decl().kind() == z3.Z3_OP_AND:
                for i in range(cur.num_args()):
                    work.append(cur.arg(i))
            else:
                atoms.append(cur)

        for atom in atoms:
            if not (z3.is_app(atom) and atom.decl().kind() == z3.Z3_OP_EQ and atom.num_args() == 2):
                continue
            a, b = atom.arg(0), atom.arg(1)
            if self._is_substitutable_lhs(a) and self._is_const_literal(b):
                if a.get_id() not in seen_ids:
                    seen_ids.add(a.get_id())
                    subs.append((a, b))
            elif self._is_substitutable_lhs(b) and self._is_const_literal(a):
                if b.get_id() not in seen_ids:
                    seen_ids.add(b.get_id())
                    subs.append((b, a))

        if not subs:
            return query, PassStats(self.name, False, 0, (time.perf_counter() - t0) * 1000.0)

        new_lhs = z3.substitute(query.lhs, *subs)
        new_rhs = z3.substitute(query.rhs, *subs)
        changed = not new_lhs.eq(query.lhs) or not new_rhs.eq(query.rhs)
        out_q = query.with_terms(lhs=new_lhs, rhs=new_rhs) if changed else query
        dt = (time.perf_counter() - t0) * 1000.0
        return out_q, PassStats(self.name, changed, len(subs) if changed else 0, dt, f"subs={len(subs)}")


class BridgeCancelPass(Z3TermPass):
    """Cancels inverse UF bridge pairs (`u_f32_to_bv32(u_bv32_to_f32(x)) -> x`, etc.)."""

    name = "bridge_cancel"
    is_equivalence_preserving = True

    def rewrite_term(
        self,
        expr: z3.ExprRef,
        ctx: SimplificationContext,
    ) -> Tuple[z3.ExprRef, int]:
        return rewrite_dag(expr, _bridge_cancel_rule)


class InjectiveBridgeUnwrapPass(Z3QueryPass):
    """
    Unwraps matching outer deterministic real-to-bv bridge calls on `(lhs, rhs)`.

    For any deterministic function `f` (e.g., `u_f32_to_bv32`), `r1 == r2`
    implies `f(r1) == f(r2)`. Therefore proving `r1 != r2` UNSAT soundly proves
    `f(r1) != f(r2)` UNSAT. Because `f` is uninterpreted (not guaranteed
    injective in Z3's model unless axiomatized), this pass marks `is_overapprox=True`
    so any `SAT` model is double-checked on the exact query.
    """

    name = "injective_bridge_unwrap"
    is_equivalence_preserving = False

    def run(
        self,
        query: SimplifyQuery,
        ctx: SimplificationContext,
    ) -> Tuple[SimplifyQuery, PassStats]:
        t0 = time.perf_counter()
        if query.is_trivially_unsat():
            return query, PassStats(self.name, False, 0, 0.0, "trivial_unsat")

        lhs, rhs = query.lhs, query.rhs
        if (
            z3.is_app(lhs)
            and z3.is_app(rhs)
            and lhs.num_args() == 1
            and rhs.num_args() == 1
            and lhs.decl().name() == rhs.decl().name()
            and lhs.decl().name() in _REAL_TO_BV_BRIDGES
        ):
            bridge_name = lhs.decl().name()
            new_l, new_r = lhs.arg(0), rhs.arg(0)
            out_q = query.with_terms(
                lhs=new_l,
                rhs=new_r,
                is_overapprox=True,
                notes=query.notes + (f"unwrapped outer {bridge_name}",),
            )
            dt = (time.perf_counter() - t0) * 1000.0
            return out_q, PassStats(self.name, True, 1, dt, f"bridge={bridge_name}")

        dt = (time.perf_counter() - t0) * 1000.0
        return query, PassStats(self.name, False, 0, dt)


class PolynomialNormalizePass(Z3QueryPass):
    """
    Bounded Sum-of-Monomials (`som=True`) polynomial normalization for `Real` miters.

    When `lhs` and `rhs` are Z3 `Real` expressions within `ctx.max_som_nodes`,
    simplifies `lhs - rhs` with `som=True, arith_lhs=True`. If the difference
    collapses to `0`, discharges the query immediately to `predicate = False`
    (`UNSAT`) in < 1 ms without invoking the SMT solver.
    """

    name = "polynomial_normalize"
    is_equivalence_preserving = True

    def run(
        self,
        query: SimplifyQuery,
        ctx: SimplificationContext,
    ) -> Tuple[SimplifyQuery, PassStats]:
        t0 = time.perf_counter()
        if query.is_trivially_unsat():
            return query, PassStats(self.name, False, 0, 0.0, "trivial_unsat")

        if query.lhs.sort().kind() != z3.Z3_REAL_SORT or query.rhs.sort().kind() != z3.Z3_REAL_SORT:
            return query, PassStats(self.name, False, 0, (time.perf_counter() - t0) * 1000.0)

        sz = count_ast_nodes([query.lhs, query.rhs], limit=ctx.max_som_nodes + 1)
        if sz > ctx.max_som_nodes:
            dt = (time.perf_counter() - t0) * 1000.0
            return query, PassStats(self.name, False, 0, dt, f"skipped_size={sz}>{ctx.max_som_nodes}")

        diff = z3.simplify(z3.lower(query.lhs) - z3.lower(query.rhs), som=True, arith_lhs=True)
        if z3.is_rational_value(diff) and diff.numerator_as_long() == 0:
            out_q = query.with_terms(
                lhs=z3.RealVal(0),
                rhs=z3.RealVal(0),
                predicate=z3.BoolVal(False),
                notes=query.notes + ("polynomial_diff_zero",),
            )
            dt = (time.perf_counter() - t0) * 1000.0
            return out_q, PassStats(self.name, True, 1, dt, "discharged_unsat")

        diff_sz = count_ast_nodes([diff], limit=sz + 1)
        if diff_sz < sz:
            out_q = query.with_terms(
                lhs=diff,
                rhs=z3.RealVal(0),
                notes=query.notes + (f"polynomial_reduced:{sz}->{diff_sz}",),
            )
            dt = (time.perf_counter() - t0) * 1000.0
            return out_q, PassStats(self.name, True, 1, dt, f"reduced={sz}->{diff_sz}")

        dt = (time.perf_counter() - t0) * 1000.0
        return query, PassStats(self.name, False, 0, dt)


def _rat_frac(e: z3.ExprRef) -> Optional[Fraction]:
    if z3.is_rational_value(e):
        return Fraction(int(e.numerator_as_long()), int(e.denominator_as_long()))
    return None


def _frac_to_real(f: Fraction) -> z3.ExprRef:
    if f.denominator == 1:
        return z3.lower(z3.RealVal(f.numerator))
    return z3.lower(z3.RealVal(f"{f.numerator}/{f.denominator}"))


def _is_uminus(e: z3.ExprRef) -> bool:
    return z3.is_app(e) and e.decl().kind() == z3.Z3_OP_UMINUS and e.num_args() == 1


def _strip_uminus(e: z3.ExprRef) -> Tuple[z3.ExprRef, bool]:
    if _is_uminus(e):
        return e.arg(0), True
    return e, False


def _real_cmp_rule(node: z3.ExprRef, kind: int) -> Optional[z3.ExprRef]:
    if node.num_args() != 2:
        return None
    a, b = node.arg(0), node.arg(1)
    if a.sort().kind() != z3.Z3_REAL_SORT or b.sort().kind() != z3.Z3_REAL_SORT:
        return None
    fa, fb = _rat_frac(a), _rat_frac(b)
    if fa is not None and fb is not None:
        if kind == z3.Z3_OP_LT:
            return z3.lower(z3.BoolVal(fa < fb))
        if kind == z3.Z3_OP_LE:
            return z3.lower(z3.BoolVal(fa <= fb))
        if kind == z3.Z3_OP_GT:
            return z3.lower(z3.BoolVal(fa > fb))
        if kind == z3.Z3_OP_GE:
            return z3.lower(z3.BoolVal(fa >= fb))
    if a.eq(b):
        return z3.lower(z3.BoolVal(kind in (z3.Z3_OP_LE, z3.Z3_OP_GE)))
    if _is_uminus(a) and _is_uminus(b):
        na, nb = a.arg(0), b.arg(0)
        if kind == z3.Z3_OP_LT:
            return nb < na
        if kind == z3.Z3_OP_LE:
            return nb <= na
        if kind == z3.Z3_OP_GT:
            return na < nb
        if kind == z3.Z3_OP_GE:
            return na <= nb
    # Canonicalize `>` to `<` and `>=` to `<=` so flipped clamps match structurally.
    if kind == z3.Z3_OP_GT:
        return b < a
    if kind == z3.Z3_OP_GE:
        return b <= a
    return None


def _real_add_rule(node: z3.ExprRef) -> Optional[z3.ExprRef]:
    nargs = node.num_args()
    const_sum = Fraction(0)
    const_count = 0
    non_const: List[z3.ExprRef] = []
    for i in range(nargs):
        ch = node.arg(i)
        fc = _rat_frac(ch)
        if fc is not None:
            const_sum += fc
            const_count += 1
        else:
            non_const.append(ch)

    # Cancel exact additive inverse pairs `(t, -t)` among non-constant summands.
    cancelled = False
    if len(non_const) >= 2:
        pos_idx: Dict[int, List[int]] = {}
        neg_idx: Dict[int, List[int]] = {}
        for idx, term in enumerate(non_const):
            base, is_neg = _strip_uminus(term)
            bid = base.get_id()
            (neg_idx if is_neg else pos_idx).setdefault(bid, []).append(idx)
        removed: set[int] = set()
        for bid, pids in pos_idx.items():
            nids = neg_idx.get(bid)
            if nids:
                pairs = min(len(pids), len(nids))
                for k in range(pairs):
                    removed.add(pids[k])
                    removed.add(nids[k])
                cancelled = True
        if cancelled:
            non_const = [t for idx, t in enumerate(non_const) if idx not in removed]

    needs_rewrite = (
        cancelled
        or const_count >= 2
        or (const_count == 1 and const_sum == 0)
    )
    if not needs_rewrite:
        return None
    terms = list(non_const)
    if const_sum != 0:
        terms.append(_frac_to_real(const_sum))
    if not terms:
        return z3.lower(z3.RealVal(0))
    acc = terms[0]
    for t in terms[1:]:
        acc = acc + t
    return acc


def _real_mul_rule(node: z3.ExprRef) -> Optional[z3.ExprRef]:
    nargs = node.num_args()
    const_prod = Fraction(1)
    const_count = 0
    neg_count = 0
    raw_non_const: List[z3.ExprRef] = []
    stripped_non_const: List[z3.ExprRef] = []
    for i in range(nargs):
        ch = node.arg(i)
        fc = _rat_frac(ch)
        if fc is not None:
            if fc == 0:
                return z3.lower(z3.RealVal(0))
            const_prod *= fc
            const_count += 1
        else:
            raw_non_const.append(ch)
            base, is_neg = _strip_uminus(ch)
            if is_neg:
                neg_count += 1
            stripped_non_const.append(base)

    # Strip unary minus only when it cancels in pairs (`(-a)*(-b) -> a*b`) or
    # absorbs into a rational constant (`(-a)*c -> (-c)*a`); pulling a lone `-`
    # out of `(-a)*b` breaks CSE on shared negated terms like `-frag_var`.
    strip_negs = neg_count >= 2 or (neg_count == 1 and const_count > 0)
    non_const = stripped_non_const if strip_negs else raw_non_const
    odd_neg = (neg_count % 2) == 1 if strip_negs else False
    if odd_neg and const_count > 0:
        const_prod = -const_prod
        odd_neg = False

    needs_rewrite = (
        strip_negs
        or const_count >= 2
        or (const_count == 1 and const_prod in (Fraction(1), Fraction(-1)))
    )
    if not needs_rewrite:
        return None

    if not non_const:
        return _frac_to_real(-const_prod if odd_neg else const_prod)

    if const_prod == Fraction(-1):
        odd_neg = not odd_neg
        const_prod = Fraction(1)

    factors: List[z3.ExprRef] = []
    if const_prod != Fraction(1):
        factors.append(_frac_to_real(const_prod))
    factors.extend(non_const)

    acc = factors[0]
    for f in factors[1:]:
        acc = acc * f
    return -acc if odd_neg else acc


def _real_div_rule(node: z3.ExprRef) -> Optional[z3.ExprRef]:
    a, b = node.arg(0), node.arg(1)
    fa, fb = _rat_frac(a), _rat_frac(b)
    if fb is not None and fb != 0:
        if fa is not None:
            return _frac_to_real(fa / fb)
        if fb == Fraction(1):
            return a
        if fb == Fraction(-1):
            return -a
        return a * _frac_to_real(Fraction(1) / fb)
    # Double reciprocal: `1 / (1 / x) -> x`, `(-1) / (1 / x) -> -x`, `a / (1 / x) -> a * x`.
    # Matches IEEE-754 finite reciprocal semantics (Valhall lowers `sqrt(x)` as `FRCP(FRSQ(x))`).
    if z3.is_app(b) and b.decl().kind() == z3.Z3_OP_DIV and b.num_args() == 2:
        b_num = _rat_frac(b.arg(0))
        if b_num is not None and b_num != 0:
            b_den = b.arg(1)
            if fa is not None:
                scale = fa / b_num
                if scale == Fraction(1):
                    return b_den
                if scale == Fraction(-1):
                    return -b_den
                return _frac_to_real(scale) * b_den
            if b_num == Fraction(1):
                return a * b_den
            if b_num == Fraction(-1):
                return -(a * b_den)
            return (a * _frac_to_real(Fraction(1) / b_num)) * b_den
    return None


def _real_uminus_rule(node: z3.ExprRef) -> Optional[z3.ExprRef]:
    a = node.arg(0)
    fa = _rat_frac(a)
    if fa is not None:
        return _frac_to_real(-fa)
    if _is_uminus(a):
        return a.arg(0)
    if z3.is_app(a):
        ak = a.decl().kind()
        if ak == z3.Z3_OP_SUB and a.num_args() == 2:
            return a.arg(1) - a.arg(0)
        if ak == z3.Z3_OP_ADD and a.num_args() >= 2:
            acc = -a.arg(0)
            for i in range(1, a.num_args()):
                acc = acc + (-a.arg(i))
            return acc
        if ak == z3.Z3_OP_MUL and a.num_args() >= 2:
            c0 = _rat_frac(a.arg(0))
            if c0 is not None:
                acc = _frac_to_real(-c0)
                for i in range(1, a.num_args()):
                    acc = acc * a.arg(i)
                return acc
    return None


def _real_algebraic_rule(node: z3.ExprRef) -> Optional[z3.ExprRef]:
    if not z3.is_app(node):
        return None
    kind = node.decl().kind()
    nargs = node.num_args()
    sk = node.sort().kind()

    if sk == z3.Z3_BOOL_SORT and kind in (z3.Z3_OP_LT, z3.Z3_OP_LE, z3.Z3_OP_GT, z3.Z3_OP_GE):
        return _real_cmp_rule(node, kind)

    if sk != z3.Z3_REAL_SORT:
        return None

    if kind == z3.Z3_OP_ITE and nargs == 3:
        c, t_arm, f_arm = node.arg(0), node.arg(1), node.arg(2)
        if z3.is_true(c):
            return t_arm
        if z3.is_false(c):
            return f_arm
        if t_arm.eq(f_arm):
            return t_arm
        if z3.is_app(c) and c.decl().kind() == z3.Z3_OP_NOT and c.num_args() == 1:
            return z3.If(c.arg(0), f_arm, t_arm)
        return None

    if kind == z3.Z3_OP_ADD and nargs >= 2:
        return _real_add_rule(node)

    if kind == z3.Z3_OP_SUB and nargs == 2:
        a, b = node.arg(0), node.arg(1)
        fa, fb = _rat_frac(a), _rat_frac(b)
        if fa is not None and fb is not None:
            return _frac_to_real(fa - fb)
        if fb is not None and fb == 0:
            return a
        if fa is not None and fa == 0:
            return -b
        if a.eq(b):
            return z3.lower(z3.RealVal(0))
        if _is_uminus(b):
            if _is_uminus(a):
                return b.arg(0) - a.arg(0)
            return a + b.arg(0)
        if fb is not None and fb < 0:
            return a + _frac_to_real(-fb)
        return None

    if kind == z3.Z3_OP_MUL and nargs >= 2:
        return _real_mul_rule(node)

    if kind == z3.Z3_OP_DIV and nargs == 2:
        return _real_div_rule(node)

    if kind == z3.Z3_OP_UMINUS and nargs == 1:
        return _real_uminus_rule(node)

    return None


class RealAlgebraicPass(Z3TermPass):
    """
    Applies terminating Real arithmetic, reciprocal, comparison, and ITE identities:
      - Exact rational constant folding across `+`, `-`, `*`, `/`, unary `-`, and `<`/`<=`/`>`/`>=`
      - Additive identity, inverse cancellation (`x + (-x) -> 0`), and `-(a + b) -> (-a) + (-b)`
      - Multiplicative identity, zero annihilation, sign normalization (`(-a)*(-b) -> a*b`, `(-1)*x -> -x`)
      - Constant divisor reciprocal folding (`x / c -> x * (1/c)`) and double-reciprocal (`1 / (1 / x) -> x`)
      - Comparison orientation canonicalization (`a >= b -> b <= a`, `a > b -> b < a`, `-a <= -b -> b <= a`)
      - Real `If` simplification (`If(Not(c), t, f) -> If(c, f, t)`, `If(c, x, x) -> x`)
    """

    name = "real_algebraic"
    is_equivalence_preserving = True

    def rewrite_term(
        self,
        expr: z3.ExprRef,
        ctx: SimplificationContext,
    ) -> Tuple[z3.ExprRef, int]:
        return rewrite_dag(expr, _real_algebraic_rule)

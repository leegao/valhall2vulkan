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
"""

from __future__ import annotations

import math
import struct
import time
from typing import Dict, List, Optional, Tuple

import z3

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
    if not z3.is_app(node) or node.num_args() != 1:
        return None
    outer_name = node.decl().name()
    inv_name = _BRIDGE_INVERSES.get(outer_name)
    if inv_name is None:
        return None
    arg0 = node.arg(0)
    if z3.is_app(arg0) and arg0.num_args() == 1 and arg0.decl().name() == inv_name:
        return arg0.arg(0)
    # Fold constant arguments across bridge boundaries (e.g. after PreconditionConstSubPass)
    if outer_name in ("u_bv32_to_f32", "bv32_to_real") and isinstance(arg0, z3.BitVecNumRef):
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
            isinstance(e, z3.BitVecNumRef)
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
            return isinstance(e.arg(1), (z3.BitVecNumRef, z3.IntNumRef))
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
        # Flatten top-level Ands in preconditions
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

        if not (isinstance(query.lhs, z3.ArithRef) and isinstance(query.rhs, z3.ArithRef)):
            return query, PassStats(self.name, False, 0, (time.perf_counter() - t0) * 1000.0)
        if not (query.lhs.is_real() and query.rhs.is_real()):
            return query, PassStats(self.name, False, 0, (time.perf_counter() - t0) * 1000.0)

        sz = count_ast_nodes([query.lhs, query.rhs], limit=ctx.max_som_nodes + 1)
        if sz > ctx.max_som_nodes:
            dt = (time.perf_counter() - t0) * 1000.0
            return query, PassStats(self.name, False, 0, dt, f"skipped_size={sz}>{ctx.max_som_nodes}")

        diff = z3.simplify(query.lhs - query.rhs, som=True, arith_lhs=True)
        if z3.is_rational_value(diff) and diff.numerator_as_long() == 0:
            out_q = query.with_terms(
                lhs=z3.RealVal(0),
                rhs=z3.RealVal(0),
                predicate=z3.BoolVal(False),
                notes=query.notes + ("polynomial_diff_zero",),
            )
            dt = (time.perf_counter() - t0) * 1000.0
            return out_q, PassStats(self.name, True, 1, dt, "discharged_unsat")

        # Even if non-zero, rewriting `lhs != rhs` to `diff != 0` can shrink the term
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


def _rat_val(e: z3.ExprRef) -> Optional[Tuple[int, int]]:
    if z3.is_rational_value(e):
        return (int(e.numerator_as_long()), int(e.denominator_as_long()))
    return None


def _real_algebraic_rule(node: z3.ExprRef) -> Optional[z3.ExprRef]:
    if not z3.is_app(node) or not (isinstance(node, z3.ArithRef) and node.is_real()):
        return None
    kind = node.decl().kind()
    nargs = node.num_args()

    if kind == z3.Z3_OP_ADD and nargs >= 2:
        kept = []
        changed = False
        for i in range(nargs):
            ch = node.arg(i)
            rv = _rat_val(ch)
            if rv is not None and rv[0] == 0:
                changed = True
                continue
            kept.append(ch)
        if changed:
            if not kept:
                return z3.RealVal(0)
            acc = kept[0]
            for k in kept[1:]:
                acc = acc + k
            return acc

    elif kind == z3.Z3_OP_SUB and nargs == 2:
        a, b = node.arg(0), node.arg(1)
        rb = _rat_val(b)
        if rb is not None and rb[0] == 0:
            return a
        if a.eq(b):
            return z3.RealVal(0)

    elif kind == z3.Z3_OP_MUL and nargs >= 2:
        kept = []
        changed = False
        for i in range(nargs):
            ch = node.arg(i)
            rv = _rat_val(ch)
            if rv is not None:
                if rv[0] == 0:
                    return z3.RealVal(0)
                if rv[0] == rv[1]:
                    changed = True
                    continue
            kept.append(ch)
        if changed:
            if not kept:
                return z3.RealVal(1)
            acc = kept[0]
            for k in kept[1:]:
                acc = acc * k
            return acc

    elif kind == z3.Z3_OP_DIV and nargs == 2:
        a, b = node.arg(0), node.arg(1)
        rb = _rat_val(b)
        if rb is not None and rb[0] == rb[1] and rb[0] != 0:
            return a
        ra = _rat_val(a)
        if ra is not None and ra[0] == 0 and rb is not None and rb[0] != 0:
            return z3.RealVal(0)

    elif kind == z3.Z3_OP_UMINUS and nargs == 1:
        a = node.arg(0)
        ra = _rat_val(a)
        if ra is not None and ra[0] == 0:
            return z3.RealVal(0)
        if z3.is_app(a) and a.decl().kind() == z3.Z3_OP_UMINUS and a.num_args() == 1:
            return a.arg(0)

    return None


class RealAlgebraicPass(Z3TermPass):
    """Applies equivalence-preserving Real arithmetic identities (`x + 0 -> x`, `x * 1 -> x`, `x * 0 -> 0`, `x / 1 -> x`, `--x -> x`)."""

    name = "real_algebraic"
    is_equivalence_preserving = True

    def rewrite_term(
        self,
        expr: z3.ExprRef,
        ctx: SimplificationContext,
    ) -> Tuple[z3.ExprRef, int]:
        return rewrite_dag(expr, _real_algebraic_rule)

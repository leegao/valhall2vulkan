"""
Abstract pass interfaces and composable pipeline combinators for `z3simplifier`.

Separation of concerns:
  - `Z3TermPass`: rewrites a single `z3.ExprRef` in isolation (e.g., BV wrapper
    peepholes, algebraic identities, bridge round-trip cancellation).
  - `Z3QueryPass`: transforms a whole `SimplifyQuery` (e.g., read-only array
    scalarization across `(lhs, rhs, guard, preconditions)`, injective outer
    bridge unwrapping, polynomial difference normalization, theory purification,
    and cross-side CSE/conditioned unification).
  - `TermPassAdapter`: lifts any `Z3TermPass` into a `Z3QueryPass` applying it
    uniformly to `lhs`, `rhs`, `guard`, and `preconditions`.
  - `FixedPointPass`: iterates a sub-pass or sub-pipeline until convergence or
    `max_rounds`.
  - `PassPipeline`: executes an ordered sequence of passes, tracking both the
    latest equivalence-preserving query (`exact_query`) and the final (possibly
    over-approximated) query so callers can always fall back soundly.
"""

from __future__ import annotations

import abc
import time
from typing import List, Optional, Sequence, Tuple

from tools.symbolic_ssa.z3utils import dsl

from tools.symbolic_ssa.z3simplifier.ast_utils import count_ast_nodes
from tools.symbolic_ssa.z3simplifier.context import (
    PassStats,
    RewriteStats,
    SimplificationContext,
    SimplifyQuery,
    SimplifyResult,
)


class Z3TermPass(abc.ABC):
    """Abstract base class for single-term Z3 AST rewrite passes."""

    name: str = "unnamed_term_pass"
    is_equivalence_preserving: bool = True

    @abc.abstractmethod
    def rewrite_term(
        self,
        expr: dsl.ExprRef,
        ctx: SimplificationContext,
    ) -> Tuple[dsl.ExprRef, int]:
        """
        Rewrite `expr` and return `(new_expr, rewrite_count)`.

        Implementations should use iterative memoized DAG traversal (`rewrite_dag`)
        so large shader DAGs run in O(|V|) time without recursion overflow.
        """


class Z3QueryPass(abc.ABC):
    """Abstract base class for whole-query Z3 simplification passes."""

    name: str = "unnamed_query_pass"
    is_equivalence_preserving: bool = True

    @abc.abstractmethod
    def run(
        self,
        query: SimplifyQuery,
        ctx: SimplificationContext,
    ) -> Tuple[SimplifyQuery, PassStats]:
        """Transform `query` and return `(new_query, pass_stats)`."""


class TermPassAdapter(Z3QueryPass):
    """Lifts a `Z3TermPass` into a `Z3QueryPass` over all terms in a `SimplifyQuery`."""

    def __init__(self, term_pass: Z3TermPass):
        self.term_pass = term_pass
        self.name = term_pass.name
        self.is_equivalence_preserving = term_pass.is_equivalence_preserving

    def run(
        self,
        query: SimplifyQuery,
        ctx: SimplificationContext,
    ) -> Tuple[SimplifyQuery, PassStats]:
        t0 = time.perf_counter()
        if query.is_trivially_unsat():
            return query, PassStats(self.name, False, 0, 0.0, "trivial_unsat")

        new_lhs, r1 = self.term_pass.rewrite_term(query.lhs, ctx)
        new_rhs, r2 = self.term_pass.rewrite_term(query.rhs, ctx)
        total_rw = r1 + r2

        new_guard = query.guard
        if query.guard is not None:
            new_guard, rg = self.term_pass.rewrite_term(query.guard, ctx)
            total_rw += rg

        new_pres: List[dsl.BoolRef] = []
        pres_changed = False
        for pre in query.preconditions:
            np, rp = self.term_pass.rewrite_term(pre, ctx)
            new_pres.append(np)
            if rp > 0 or not np.eq(pre):
                pres_changed = True
                total_rw += rp

        changed = (
            total_rw > 0
            or not new_lhs.eq(query.lhs)
            or not new_rhs.eq(query.rhs)
            or (query.guard is not None and not new_guard.eq(query.guard))
            or pres_changed
        )
        out_q = (
            query.with_terms(
                lhs=new_lhs,
                rhs=new_rhs,
                guard=new_guard,
                preconditions=tuple(new_pres),
            )
            if changed
            else query
        )
        dt = (time.perf_counter() - t0) * 1000.0
        return out_q, PassStats(self.name, changed, total_rw, dt)


class FixedPointPass(Z3QueryPass):
    """Iterates a list of passes until no pass makes progress or `max_rounds` is hit."""

    def __init__(
        self,
        passes: Sequence[Z3QueryPass],
        name: str = "fixed_point",
        max_rounds: int = 4,
    ):
        self.passes: List[Z3QueryPass] = list(passes)
        self.name = name
        self.max_rounds = max(1, int(max_rounds))
        self.is_equivalence_preserving = all(p.is_equivalence_preserving for p in self.passes)

    def run(
        self,
        query: SimplifyQuery,
        ctx: SimplificationContext,
    ) -> Tuple[SimplifyQuery, PassStats]:
        t0 = time.perf_counter()
        cur = query
        total_rw = 0
        any_changed = False
        rounds_run = 0
        for rnd in range(1, self.max_rounds + 1):
            rounds_run = rnd
            round_changed = False
            for p in self.passes:
                if cur.is_trivially_unsat():
                    break
                cur, st = p.run(cur, ctx)
                if st.changed:
                    round_changed = True
                    any_changed = True
                    total_rw += st.rewrites
            if not round_changed or cur.is_trivially_unsat():
                break
        dt = (time.perf_counter() - t0) * 1000.0
        return cur, PassStats(self.name, any_changed, total_rw, dt, f"rounds={rounds_run}")


class PassPipeline:
    """
    Ordered simplification pipeline with exact-query checkpointing.

    Maintains `exact_query` as the latest state before any over-approximating
    pass fires, allowing downstream solvers to fall back cleanly if an
    over-approximated query produces a potentially spurious `SAT`.
    """

    def __init__(self, passes: Sequence[Z3QueryPass]):
        self.passes: List[Z3QueryPass] = list(passes)

    def run(
        self,
        query: SimplifyQuery,
        ctx: Optional[SimplificationContext] = None,
    ) -> SimplifyResult:
        eff_ctx = ctx if ctx is not None else SimplificationContext()
        t0 = time.perf_counter()
        stats = RewriteStats()

        roots = [query.lhs, query.rhs]
        if query.guard is not None:
            roots.append(query.guard)
        stats.initial_ast_nodes = count_ast_nodes(roots, limit=eff_ctx.max_dag_nodes)

        if query.is_trivially_unsat():
            stats.final_ast_nodes = stats.initial_ast_nodes
            stats.total_ms = (time.perf_counter() - t0) * 1000.0
            return SimplifyResult(query=query, exact_query=query, stats=stats)

        cur = query
        exact_q = cur
        for p in self.passes:
            if cur.is_trivially_unsat():
                break
            if not p.is_equivalence_preserving and not eff_ctx.enable_overapprox:
                continue
            nxt, pst = p.run(cur, eff_ctx)
            stats.pass_stats.append(pst)
            stats.rounds += 1
            if eff_ctx.verbose:
                eff_ctx.trace(
                    f"[z3simplifier] pass={pst.pass_name} changed={pst.changed} "
                    f"rewrites={pst.rewrites} elapsed={pst.elapsed_ms:.2f}ms {pst.detail}"
                )
            cur = nxt
            if not cur.is_overapprox and p.is_equivalence_preserving:
                exact_q = cur

        end_roots = [cur.lhs, cur.rhs]
        if cur.guard is not None:
            end_roots.append(cur.guard)
        stats.final_ast_nodes = count_ast_nodes(end_roots, limit=eff_ctx.max_dag_nodes)
        stats.total_ms = (time.perf_counter() - t0) * 1000.0
        return SimplifyResult(query=cur, exact_query=exact_q, stats=stats)

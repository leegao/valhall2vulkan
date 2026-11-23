#!/usr/bin/env python3
"""
Simplification-aware SMT miter solver driver (`solve_simplified_miter`).

Integrates `PassPipeline` with Z3 solver dispatch:
  1. Runs the configured `PassPipeline` (preserving `exact_query` as the latest
     equivalence-preserving checkpoint before any over-approximating pass).
  2. Discharges syntactically/polynomial-trivial `UNSAT` queries in < 1 ms
     without creating a `z3.Solver`.
  3. Routes pure `QF_NRA` queries (`pure_theory == "QF_NRA"`) to Z3's `nlsat`
     tactic (`z3.Then("simplify", "qfnra").solver()`).
  4. Enforces **zero false-SAT soundness**: if an over-approximated query
     returns `SAT`, re-checks `exact_query` before confirming `SAT`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from tools.symbolic_ssa import z3

from tools.symbolic_ssa.z3simplifier.context import (
    RewriteStats,
    SimplificationContext,
    SimplifyQuery,
)
from tools.symbolic_ssa.z3simplifier.pass_base import PassPipeline


@dataclass
class MiterSolveResult:
    """Outcome of simplifying and solving a miter equivalence query."""

    status: str  # "UNSAT", "SAT", "UNKNOWN"
    model: Optional[z3.ModelRef]
    simplified_query: SimplifyQuery
    exact_query: SimplifyQuery
    stats: RewriteStats
    solver_ms: float
    reason_unknown: str = ""
    used_nlsat: bool = False
    refined_from_overapprox: bool = False


def _make_solver(timeout_ms: int, rlimit: int, pure_theory: Optional[str]) -> Tuple[z3.Solver, bool]:
    """Create a Z3 solver, selecting `qfnra` (`nlsat`) when the query is pure `QF_NRA`."""
    use_nlsat = pure_theory == "QF_NRA"
    if use_nlsat:
        try:
            s = z3.Then("simplify", "qfnra").solver()
        except Exception:
            s = z3.Solver()
            use_nlsat = False
    else:
        s = z3.Solver()
    s.set("timeout", max(1, int(timeout_ms)))
    if rlimit > 0:
        s.set("rlimit", int(rlimit))
    return s, use_nlsat


def _check_query_with_z3(
    q: SimplifyQuery,
    timeout_ms: int,
    rlimit: int,
    smt2_out: Optional[List[str]] = None,
    check_name: str = "",
) -> Tuple[str, Optional[z3.ModelRef], float, str, bool]:
    """Run Z3 on a single `SimplifyQuery`."""
    if q.is_trivially_unsat():
        if smt2_out is not None:
            smt2_out.append(f"; --- {check_name}: simplified to UNSAT ---\n")
        return "UNSAT", None, 0.0, "", False

    pred = q.to_refutation_predicate()
    simp_pred = z3.simplify(pred)
    if z3.is_false(simp_pred):
        if smt2_out is not None:
            smt2_out.append(f"; --- {check_name}: predicate simplified to false (UNSAT) ---\n")
        return "UNSAT", None, 0.0, "", False

    s, used_nlsat = _make_solver(timeout_ms, rlimit, q.pure_theory)
    for pre in q.preconditions:
        s.add(pre)
    s.add(simp_pred)

    if smt2_out is not None:
        smt2_out.append(
            f"; === Output Miter Check: {check_name} (theory={q.pure_theory or 'ALL'}) ===\n"
            f"(push 1)\n{s.to_smt2()}(check-sat)\n(pop 1)\n"
        )

    t0 = time.perf_counter()
    rc = s.check()
    dt = (time.perf_counter() - t0) * 1000.0
    if rc == z3.unsat:
        return "UNSAT", None, dt, "", used_nlsat
    if rc == z3.sat:
        return "SAT", s.model(), dt, "", used_nlsat
    return "UNKNOWN", None, dt, s.reason_unknown(), used_nlsat


def solve_simplified_miter(
    query: SimplifyQuery,
    pipeline: PassPipeline,
    *,
    timeout_ms: int = 1000,
    rlimit: Optional[int] = None,
    ctx: Optional[SimplificationContext] = None,
    smt2_out: Optional[List[str]] = None,
    check_name: str = "",
) -> MiterSolveResult:
    """
    Simplify `query` through `pipeline` and discharge via Z3 with sound
    over-approximation refinement.
    """
    eff_ctx = ctx if ctx is not None else SimplificationContext()
    if rlimit is None:
        eff_rlimit = 0 if timeout_ms >= 5000 else max(2_000_000, int(timeout_ms) * 20_000)
    else:
        eff_rlimit = int(rlimit)
    sim = pipeline.run(query, eff_ctx)
    q = sim.query
    exact_q = sim.exact_query

    if q.is_trivially_unsat():
        if smt2_out is not None:
            smt2_out.append(f"; --- {check_name}: discharged by z3simplifier ({sim.stats.summary_line()}) ---\n")
        return MiterSolveResult(
            status="UNSAT",
            model=None,
            simplified_query=q,
            exact_query=exact_q,
            stats=sim.stats,
            solver_ms=0.0,
        )

    status, model, dt, reason, used_nlsat = _check_query_with_z3(
        q,
        timeout_ms=timeout_ms,
        rlimit=eff_rlimit,
        smt2_out=smt2_out if not q.is_overapprox else None,
        check_name=check_name,
    )

    if status == "UNSAT":
        if smt2_out is not None and q.is_overapprox:
            smt2_out.append(
                f"; --- {check_name}: proved UNSAT on over-approximated query "
                f"({sim.stats.summary_line()}) ---\n"
            )
        return MiterSolveResult(
            status="UNSAT",
            model=None,
            simplified_query=q,
            exact_query=exact_q,
            stats=sim.stats,
            solver_ms=dt,
            used_nlsat=used_nlsat,
        )

    # Soundness guard: an over-approximated SAT (or UNKNOWN when over-approx
    # introduced fresh variables) must be confirmed on `exact_query` before
    # claiming SAT.
    if q.is_overapprox and status == "SAT":
        st_ex, m_ex, dt_ex, reason_ex, nlsat_ex = _check_query_with_z3(
            exact_q,
            timeout_ms=timeout_ms,
            rlimit=eff_rlimit,
            smt2_out=smt2_out,
            check_name=f"{check_name}[exact_confirm]",
        )
        return MiterSolveResult(
            status=st_ex,
            model=m_ex,
            simplified_query=q,
            exact_query=exact_q,
            stats=sim.stats,
            solver_ms=dt + dt_ex,
            reason_unknown=reason_ex,
            used_nlsat=used_nlsat or nlsat_ex,
            refined_from_overapprox=True,
        )

    return MiterSolveResult(
        status=status,
        model=model,
        simplified_query=q,
        exact_query=exact_q,
        stats=sim.stats,
        solver_ms=dt,
        reason_unknown=reason,
        used_nlsat=used_nlsat,
    )

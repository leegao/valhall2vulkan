#!/usr/bin/env python3
"""
Z3 E-Matching Extensional Axiom & Unification Pass (`EmatchingUnifyPass`).

Leverages Z3's built-in E-matching engine (`smt.ematching=True`, `smt.mbqi=False`)
to discharge or unify queries under quantified extensional identities and
domain-bridge round-trip axioms:

  1. Bridge Round-Trip Axioms (with explicit `:pattern` triggers so E-matching
     terminates without matching loops):
       ForAll([x], u_f32_to_bv32(u_bv32_to_f32(x)) == x,
              patterns=[u_f32_to_bv32(u_bv32_to_f32(x))])
       ForAll([r], u_bv32_to_f32(u_f32_to_bv32(r)) == r,
              patterns=[u_bv32_to_f32(u_f32_to_bv32(r))])

  2. Higher-Order Function Extensionality via Skolem Witness:
       Given an abstract sort `S`, `apply: S x S -> S`, and Skolem witness
       `ext_witness: S x S -> S`:
         Implies(apply(fn_f, ext_witness(fn_f, fn_g)) ==
                 apply(fn_g, ext_witness(fn_f, fn_g)),
                 fn_f == fn_g)
       Allows proving `fn_f == fn_g` from pointwise equality
       `ForAll([x], apply(fn_f, x) == apply(fn_g, x))` in Z3's first-order core.

  3. Custom User/Domain Axioms:
     Callers may supply additional `ForAll` axioms (e.g., `ForAll([x], f(x) == g(x))`
     or algebraic identities over abstracted UFs) to discharge equivalences as an
     intermediate pass before lowering abstracted UFs back to concrete conversions.

Soundness:
  - When `extra_axioms` are empty and only exact bridge bijections (matching
    `udomain` semantics) are asserted, proving `UNSAT` is sound under `udomain`.
  - Marked `is_equivalence_preserving = False` whenever `extra_axioms` or
    bridge bijections are active so any `SAT` still respects `exact_query`.
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

from tools.symbolic_ssa import z3utils as z3

from tools.symbolic_ssa.z3simplifier.context import (
    PassStats,
    SimplificationContext,
    SimplifyQuery,
)
from tools.symbolic_ssa.z3simplifier.pass_base import Z3QueryPass


def build_bridge_ematch_axioms() -> List[z3.BoolRef]:
    """Build trigger-guarded `ForAll` round-trip axioms for `udomain` f32/bv32 bridges."""
    u_f2b = z3.Function("u_f32_to_bv32", z3.RealSort(), z3.BitVecSort(32))
    u_b2f = z3.Function("u_bv32_to_f32", z3.BitVecSort(32), z3.RealSort())
    x = z3.BitVec("_em_x", 32)
    r = z3.Real("_em_r")
    t_b = u_f2b(u_b2f(x))
    t_r = u_b2f(u_f2b(r))
    return [
        z3.ForAll([x], t_b == x, patterns=[t_b]),
        z3.ForAll([r], t_r == r, patterns=[t_r]),
    ]


def build_extensionality_witness_axiom(
    sort: z3.SortRef,
    apply_fn: z3.FuncDeclRef,
    fn_f: z3.ExprRef,
    fn_g: z3.ExprRef,
    witness_name: str = "ext_witness",
) -> z3.BoolRef:
    """
    Build the Skolem-witness extensionality axiom for first-order function objects:
      `apply(fn_f, ext_witness(fn_f, fn_g)) == apply(fn_g, ext_witness(fn_f, fn_g)) => fn_f == fn_g`.
    """
    ext_w = z3.Function(witness_name, sort, sort, sort)
    w = ext_w(fn_f, fn_g)
    return z3.Implies(apply_fn(z3.lower(fn_f), z3.lower(w)) == apply_fn(z3.lower(fn_g), z3.lower(w)), z3.lower(fn_f) == z3.lower(fn_g))


class EmatchingUnifyPass(Z3QueryPass):
    """
    Intermediate Z3 E-matching pass that discharges `(preconditions & guard) => (lhs == rhs)`
    under quantified bridge round-trip and extensionality axioms.

    If Z3's E-matching solver proves `UNSAT` within `timeout_ms`, collapses the
    query to `predicate = BoolVal(False)` (`UNSAT`). Otherwise leaves the query
    untouched for downstream passes and full SMT solving.
    """

    name = "ematching_unify"
    is_equivalence_preserving = False

    def __init__(
        self,
        *,
        extra_axioms: Sequence[z3.BoolRef] = (),
        include_bridge_axioms: bool = True,
        timeout_ms: int = 80,
        rlimit: int = 300_000,
    ) -> None:
        self.extra_axioms = tuple(extra_axioms)
        self.include_bridge_axioms = include_bridge_axioms
        self.timeout_ms = max(10, int(timeout_ms))
        self.rlimit = max(0, int(rlimit))

    def run(
        self,
        query: SimplifyQuery,
        ctx: SimplificationContext,
    ) -> Tuple[SimplifyQuery, PassStats]:
        t0 = time.perf_counter()
        if query.is_trivially_unsat():
            return query, PassStats(self.name, False, 0, 0.0, "trivial_unsat")

        axioms: List[z3.BoolRef] = []
        if self.include_bridge_axioms:
            axioms.extend(build_bridge_ematch_axioms())
        axioms.extend(self.extra_axioms)
        if not axioms:
            return query, PassStats(self.name, False, 0, (time.perf_counter() - t0) * 1000.0)

        s = z3.Solver()
        s.set("timeout", self.timeout_ms)
        if self.rlimit > 0:
            s.set("rlimit", self.rlimit)
        s.set("smt.ematching", True)
        s.set("smt.mbqi", False)

        for ax in axioms:
            s.add(ax)
        for pre in query.preconditions:
            s.add(pre)
        s.add(query.to_refutation_predicate())

        rc = s.check()
        dt = (time.perf_counter() - t0) * 1000.0
        if rc == z3.unsat:
            out_q = query.with_terms(
                lhs=query.lhs,
                rhs=query.rhs,
                predicate=z3.BoolVal(False),
                is_overapprox=True,
                notes=query.notes + (f"ematch_discharged(axioms={len(axioms)})",),
            )
            return out_q, PassStats(self.name, True, 1, dt, f"discharged axioms={len(axioms)}")

        return query, PassStats(self.name, False, 0, dt, f"rc={rc}")

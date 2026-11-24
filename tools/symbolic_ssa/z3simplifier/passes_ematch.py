"""
Z3 E-Matching Extensional Axiom & Unification Pass (`EmatchingUnifyPass`).

Leverages Z3's built-in E-matching engine (`smt.ematching=True`, `smt.mbqi=False`)
to discharge or unify queries under quantified extensional identities and
domain-bridge axioms:

  1. Bridge Round-Trip & IEEE-754 Sign-Bit Axioms (with explicit `:pattern`
     triggers so E-matching terminates without matching loops):
       ForAll([x], u_f32_to_bv32(u_bv32_to_f32(x)) == x,
              patterns=[u_f32_to_bv32(u_bv32_to_f32(x))])
       ForAll([r], u_bv32_to_f32(u_f32_to_bv32(r)) == r,
              patterns=[u_bv32_to_f32(u_f32_to_bv32(r))])
       ForAll([r], (u_f32_to_bv32(r) ^ 0x80000000) == u_f32_to_bv32(-r),
              patterns=[u_f32_to_bv32(r) ^ 0x80000000])
       ForAll([r], (u_f32_to_bv32(r) & 0x7FFFFFFF) == u_f32_to_bv32(If(r < 0, -r, r)),
              patterns=[u_f32_to_bv32(r) & 0x7FFFFFFF])

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
  - When `extra_axioms` are empty and only exact bridge bijections / finite
    sign-bit identities (matching `udomain` Real semantics) are asserted,
    proving `UNSAT` is sound under `udomain`.
  - Marked `is_equivalence_preserving = False` whenever `extra_axioms` or
    bridge axioms are active so any `SAT` still respects `exact_query`.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import z3

from tools.symbolic_ssa import z3utils
from tools.symbolic_ssa.z3simplifier.context import (
    PassStats,
    SimplificationContext,
    SimplifyQuery,
)
from tools.symbolic_ssa.z3simplifier.pass_base import Z3QueryPass
from tools.symbolic_ssa.z3utils import dsl


def build_bridge_ematch_axioms(*, include_sign_bit_axioms: bool = True) -> list[z3.BoolRef]:
    """
    Build trigger-guarded `ForAll` round-trip and IEEE-754 sign-bit axioms
    for `udomain` f32/bv32 bridges.
    """
    u_f2b = dsl.Function("u_f32_to_bv32", dsl.RealSort(), dsl.BitVecSort(32))
    u_b2f = dsl.Function("u_bv32_to_f32", dsl.BitVecSort(32), dsl.RealSort())
    x = dsl.BitVec("_em_x", 32)
    r = dsl.Real("_em_r")
    t_b = u_f2b(u_b2f(x))
    t_r = u_b2f(u_f2b(r))
    axioms: list[z3.BoolRef] = [
        dsl.ForAll([x], t_b == x, patterns=[t_b]),
        dsl.ForAll([r], t_r == r, patterns=[t_r]),
    ]
    if include_sign_bit_axioms:
        sign_mask = dsl.BitVecVal(0x80000000, 32)
        abs_mask = dsl.BitVecVal(0x7FFFFFFF, 32)
        t_neg = u_f2b(r) ^ sign_mask
        t_neg_comm = sign_mask ^ u_f2b(r)
        t_abs = u_f2b(r) & abs_mask
        t_abs_comm = abs_mask & u_f2b(r)
        abs_r = dsl.If(r < dsl.RealVal(0), -r, r)
        axioms.extend([
            dsl.ForAll([r], t_neg == u_f2b(-r), patterns=[t_neg]),
            dsl.ForAll([r], t_neg_comm == u_f2b(-r), patterns=[t_neg_comm]),
            dsl.ForAll([r], t_abs == u_f2b(abs_r), patterns=[t_abs]),
            dsl.ForAll([r], t_abs_comm == u_f2b(abs_r), patterns=[t_abs_comm]),
        ])
    return axioms


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
    return z3.Implies(
        apply_fn(dsl.lower(fn_f), dsl.lower(w)) == apply_fn(dsl.lower(fn_g), dsl.lower(w)),
        dsl.lower(fn_f) == dsl.lower(fn_g),
    )


class EmatchingUnifyPass(Z3QueryPass):
    """
    Intermediate Z3 E-matching pass that discharges `(preconditions & guard) => (lhs == rhs)`
    under quantified bridge round-trip, sign-bit, and extensionality axioms.

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
    ) -> tuple[SimplifyQuery, PassStats]:
        t0 = time.perf_counter()
        if query.is_trivially_unsat():
            return query, PassStats(self.name, False, 0, 0.0, "trivial_unsat")

        axioms: list[z3.BoolRef] = []
        if self.include_bridge_axioms:
            axioms.extend(build_bridge_ematch_axioms())
        axioms.extend(self.extra_axioms)
        if not axioms:
            return query, PassStats(self.name, False, 0, (time.perf_counter() - t0) * 1000.0)

        s = z3utils.Solver()
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
                predicate=dsl.BoolVal(False),
                is_overapprox=True,
                notes=query.notes + (f"ematch_discharged(axioms={len(axioms)})",),
            )
            return out_q, PassStats(self.name, True, 1, dt, f"discharged axioms={len(axioms)}")

        return query, PassStats(self.name, False, 0, dt, f"rc={rc}")

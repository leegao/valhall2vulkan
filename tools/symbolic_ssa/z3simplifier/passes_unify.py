#!/usr/bin/env python3
"""
Cross-Side Common-Subexpression Unification & Conditioned Unification Passes.

1. `ExactCseUnificationPass` (UNSAT-sound over-approximation):
   Finds maximal non-trivial subexpressions `e` that appear identically in BOTH
   `lhs` and `rhs` (by Z3 AST identity `get_id()`, leveraging Z3's internal
   hash-consing), and replaces every occurrence of `e` with a fresh typed
   symbolic constant `cse_k`.
   - Why it helps: If both `direct_ssa` and `valhall_ssa` compute an identical
     nonlinear lighting sub-term, `BVMul` stride calculation, or `Select`
     chain, lifting it to `cse_k` slashes DAG size and removes `BVMul` /
     nonlinear terms before Z3 ever sees them!
   - Soundness: Replacing a shared subexpression `e` with an unconstrained
     fresh variable `cse_k` strictly weakens the domain of `e` (it can take
     any value, including the actual value of `e`). Therefore:
       * Abstract `UNSAT` => Original `UNSAT` (100% sound).
       * Abstract `SAT` may be spurious (e.g. if `cse_k` correlates with
         another input variable), so `is_overapprox=True` triggers fallback or
         CEGAR refinement in `solve_simplified_miter`.

2. `ConditionedUnificationPass` (100% equivalence-preserving when discharged!):
   Goes beyond exact structural identity: finds *near-identical* subexpression
   pairs `(e_l, e_r)` across `lhs` and `rhs` that share the same sort,
   non-trivial depth, and matching leaf-variable footprint (or matching root
   operator family, e.g., `Add`/`Mul` FMA reassociation or `BVMul` vs `Shl`
   address expressions).
   For each candidate pair `(e_l, e_r)`, generates a Weakest Liberal
   Precondition (WLP) / congruence side-obligation:
       `wlp_k := (preconditions & guard) => (e_l == e_r)`
   and attempts to discharge it locally (first via `PolynomialNormalizePass` /
   `z3.simplify(..., som=True)` in <1ms, then via a bounded `wlp_timeout_ms`
   Z3 check).
   - If `wlp_k` is proved `UNSAT` (valid): substituting `e_l -> cu_k` in `lhs`
     and `e_r -> cu_k` in `rhs` is **100% sound** (by congruence of equality!)
     and preserves `UNSAT` without introducing spurious `SAT` from mismatching
     subterms (though `cu_k` itself is fresh, so we still track `abstraction_map`
     and `is_overapprox=True` in case `cu_k` correlates with outer inputs).
   - If `wlp_k` is NOT proved (`SAT` or `UNKNOWN`): the pair is skipped and
     left untouched!
"""

from __future__ import annotations

import time
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from tools.symbolic_ssa import z3

from tools.symbolic_ssa.z3simplifier.ast_utils import count_ast_nodes, rewrite_dag
from tools.symbolic_ssa.z3simplifier.context import (
    PassStats,
    SideObligation,
    SimplificationContext,
    SimplifyQuery,
)
from tools.symbolic_ssa.z3simplifier.pass_base import Z3QueryPass


def _collect_subexprs_with_depth(
    root: z3.ExprRef,
    limit: int = 50000,
) -> Tuple[Dict[int, z3.ExprRef], Dict[int, int]]:
    """Iteratively collect all reachable subexpressions and their AST depths."""
    nodes: Dict[int, z3.ExprRef] = {}
    depths: Dict[int, int] = {}
    visited: Set[int] = set()
    stack: List[Tuple[z3.ExprRef, bool]] = [(root, False)]

    while stack and len(visited) < limit:
        cur, expanded = stack.pop()
        cid = cur.get_id()
        if expanded:
            if cid not in depths:
                if z3.is_app(cur) and cur.num_args() > 0:
                    depths[cid] = 1 + max(depths.get(cur.arg(i).get_id(), 0) for i in range(cur.num_args()))
                else:
                    depths[cid] = 0
                nodes[cid] = cur
            continue
        if cid in visited:
            continue
        visited.add(cid)
        stack.append((cur, True))
        if z3.is_app(cur):
            for i in range(cur.num_args() - 1, -1, -1):
                ch = cur.arg(i)
                if ch.get_id() not in visited:
                    stack.append((ch, False))
    return nodes, depths


def _collect_leaf_signature(
    root: z3.ExprRef,
    limit: int = 2000,
) -> FrozenSet[int]:
    """Collect the frozenset of free symbolic constant IDs referenced by `root`."""
    leaves: Set[int] = set()
    visited: Set[int] = set()
    stack: List[z3.ExprRef] = [root]
    while stack and len(visited) < limit:
        cur = stack.pop()
        cid = cur.get_id()
        if cid in visited:
            continue
        visited.add(cid)
        if z3.is_const(cur):
            if cur.decl().kind() == z3.Z3_OP_UNINTERPRETED:
                leaves.add(cid)
            continue
        if z3.is_app(cur):
            for i in range(cur.num_args()):
                ch = cur.arg(i)
                if ch.get_id() not in visited:
                    stack.append(ch)
    return frozenset(leaves)


def _is_liftable_subexpr(node: z3.ExprRef, depth: int, min_depth: int) -> bool:
    """Return True if `node` is a non-trivial scalar subexpression worth lifting."""
    if depth < min_depth:
        return False
    if not z3.is_app(node) or node.num_args() == 0:
        return False
    sk = node.sort().kind()
    return sk in (z3.Z3_BV_SORT, z3.Z3_REAL_SORT, z3.Z3_INT_SORT, z3.Z3_BOOL_SORT)


class ExactCseUnificationPass(Z3QueryPass):
    """
    Lifts maximal identical subexpressions shared between `lhs` and `rhs` into
    fresh typed symbolic constants `cse_k`.
    """

    name = "exact_cse_unification"
    is_equivalence_preserving = False

    def __init__(self, min_depth: int = 2):
        self.min_depth = max(1, int(min_depth))

    def run(
        self,
        query: SimplifyQuery,
        ctx: SimplificationContext,
    ) -> Tuple[SimplifyQuery, PassStats]:
        t0 = time.perf_counter()
        if query.is_trivially_unsat():
            return query, PassStats(self.name, False, 0, 0.0, "trivial_unsat")

        l_nodes, l_depths = _collect_subexprs_with_depth(query.lhs, limit=ctx.max_dag_nodes)
        r_nodes, _ = _collect_subexprs_with_depth(query.rhs, limit=ctx.max_dag_nodes)

        shared_ids = (set(l_nodes.keys()) & set(r_nodes.keys())) - {query.lhs.get_id(), query.rhs.get_id()}
        if not shared_ids:
            return query, PassStats(self.name, False, 0, (time.perf_counter() - t0) * 1000.0)

        # Pick maximal shared subexpressions (deepest first; skip descendants of already-selected nodes)
        candidates = [
            (l_depths[nid], nid, l_nodes[nid])
            for nid in shared_ids
            if _is_liftable_subexpr(l_nodes[nid], l_depths[nid], self.min_depth)
        ]
        if not candidates:
            return query, PassStats(self.name, False, 0, (time.perf_counter() - t0) * 1000.0)

        candidates.sort(key=lambda x: (-x[0], x[1]))

        covered: Set[int] = set()
        repl_map: Dict[int, z3.ExprRef] = {}
        abs_pairs: List[Tuple[z3.ExprRef, z3.ExprRef]] = []

        for _, nid, expr in candidates:
            if len(repl_map) >= ctx.max_unify_candidates:
                break
            if nid in covered:
                continue
            fv = ctx.fresh_const("cse", expr.sort())
            repl_map[nid] = fv
            abs_pairs.append((fv, expr))
            # Mark all descendants of `expr` as covered so we lift maximal cones
            sub_nodes, _ = _collect_subexprs_with_depth(expr, limit=2000)
            covered.update(sub_nodes.keys())

        if not repl_map:
            return query, PassStats(self.name, False, 0, (time.perf_counter() - t0) * 1000.0)

        # Top-down substitution using `z3.substitute` (fast C++ implementation in Z3)
        sub_pairs = [(expr, fv) for fv, expr in abs_pairs]
        new_lhs = z3.substitute(query.lhs, *sub_pairs)
        new_rhs = z3.substitute(query.rhs, *sub_pairs)
        new_guard = z3.substitute(query.guard, *sub_pairs) if query.guard is not None else None
        new_pres = tuple(z3.substitute(p, *sub_pairs) for p in query.preconditions)

        out_q = query.with_terms(
            lhs=new_lhs,
            rhs=new_rhs,
            guard=new_guard,
            preconditions=new_pres,
            is_overapprox=True,
            abstraction_map=query.abstraction_map + tuple(abs_pairs),
            notes=query.notes + (f"cse_lifted={len(abs_pairs)}",),
        )
        dt = (time.perf_counter() - t0) * 1000.0
        return out_q, PassStats(self.name, True, len(abs_pairs), dt, f"lifted={len(abs_pairs)}")


def _discharge_wlp_local(
    lhs: z3.ExprRef,
    rhs: z3.ExprRef,
    guard: Optional[z3.BoolRef],
    preconditions: Tuple[z3.BoolRef, ...],
    timeout_ms: int,
) -> Tuple[bool, float]:
    """
    Attempt to prove `(preconditions & guard) => (lhs == rhs)` locally.

    Returns `(is_valid, elapsed_ms)`. Uses a fast polynomial check for Reals
    before falling back to a bounded Z3 solver check.
    """
    t0 = time.perf_counter()
    if lhs.eq(rhs):
        return True, (time.perf_counter() - t0) * 1000.0

    # Fast polynomial check for Real subexpressions (e.g. FMA reassociation)
    if isinstance(lhs, z3.ArithRef) and isinstance(rhs, z3.ArithRef) and lhs.is_real() and rhs.is_real():
        if count_ast_nodes([lhs, rhs], limit=129) <= 128:
            diff = z3.simplify(lhs - rhs, som=True, arith_lhs=True)
            if z3.is_rational_value(diff) and diff.numerator_as_long() == 0:
                return True, (time.perf_counter() - t0) * 1000.0

    # Fast syntactic check
    simp_neq = z3.simplify(lhs != rhs)
    if z3.is_false(simp_neq):
        return True, (time.perf_counter() - t0) * 1000.0

    # Bounded SMT discharge
    s = z3.Solver()
    s.set("timeout", max(10, int(timeout_ms)))
    s.set("rlimit", 200_000)
    for pre in preconditions:
        s.add(pre)
    if guard is not None:
        s.add(guard)
    s.add(simp_neq)
    rc = s.check()
    dt = (time.perf_counter() - t0) * 1000.0
    return (rc == z3.unsat), dt


class ConditionedUnificationPass(Z3QueryPass):
    """
    Discovers structurally similar (non-identical) subexpression pairs
    `(e_l, e_r)` across `lhs` and `rhs` that share the same sort and leaf
    variable footprint, generates a WLP/congruence side-obligation
    `(preconditions & guard) => (e_l == e_r)`, and lifts proved pairs into a
    shared fresh symbol `cu_k`.

    Example:
      - `lhs` contains `(a * b + c) + d * e` (direct FMA-unfused dot product)
      - `rhs` contains `d * e + (a * b + c)` or `a * b + (c + d * e)` (Valhall FMA chain)
      Both have the exact same `(RealSort, {a, b, c, d, e})` leaf footprint!
      The local WLP check discharges `e_l - e_r == 0` in < 0.5 ms via `som=True`,
      and both subtrees are unified to `cu_0`, collapsing downstream nonlinear
      expressions (e.g., `1 / sqrt(cu_0)`) to syntactic identity!
    """

    name = "conditioned_unification"
    is_equivalence_preserving = False

    def __init__(self, min_depth: int = 2, max_depth: int = 12, max_candidates: int = 16):
        self.min_depth = max(1, int(min_depth))
        self.max_depth = max(self.min_depth, int(max_depth))
        self.max_candidates = max(1, int(max_candidates))

    def run(
        self,
        query: SimplifyQuery,
        ctx: SimplificationContext,
    ) -> Tuple[SimplifyQuery, PassStats]:
        t0 = time.perf_counter()
        if query.is_trivially_unsat():
            return query, PassStats(self.name, False, 0, 0.0, "trivial_unsat")

        l_nodes, l_depths = _collect_subexprs_with_depth(query.lhs, limit=ctx.max_dag_nodes)
        r_nodes, r_depths = _collect_subexprs_with_depth(query.rhs, limit=ctx.max_dag_nodes)

        # Index right-hand-side candidate subexpressions by `(sort_kind, leaf_footprint)`
        r_buckets: Dict[Tuple[int, FrozenSet[int]], List[Tuple[int, z3.ExprRef]]] = {}
        for rid, rexpr in r_nodes.items():
            if rid == query.rhs.get_id() or rid in l_nodes:
                continue
            rd = r_depths[rid]
            if rd < self.min_depth or rd > self.max_depth:
                continue
            if not _is_liftable_subexpr(rexpr, rd, self.min_depth):
                continue
            fp = _collect_leaf_signature(rexpr)
            if not fp:
                continue
            key = (rexpr.sort().kind(), fp)
            r_buckets.setdefault(key, []).append((rd, rexpr))

        if not r_buckets:
            return query, PassStats(self.name, False, 0, (time.perf_counter() - t0) * 1000.0)

        # Sort each bucket deepest-first
        for b in r_buckets.values():
            b.sort(key=lambda x: -x[0])

        # Match left-hand-side subexpressions deepest-first
        l_cands: List[Tuple[int, int, z3.ExprRef]] = []
        for lid, lexpr in l_nodes.items():
            if lid == query.lhs.get_id() or lid in r_nodes:
                continue
            ld = l_depths[lid]
            if ld < self.min_depth or ld > self.max_depth:
                continue
            if not _is_liftable_subexpr(lexpr, ld, self.min_depth):
                continue
            l_cands.append((ld, lid, lexpr))
        l_cands.sort(key=lambda x: (-x[0], x[1]))

        obligations: List[SideObligation] = list(query.side_obligations)
        l_subs: List[Tuple[z3.ExprRef, z3.ExprRef]] = []
        r_subs: List[Tuple[z3.ExprRef, z3.ExprRef]] = []
        abs_pairs: List[Tuple[z3.ExprRef, z3.ExprRef]] = []
        covered_l: Set[int] = set()
        covered_r: Set[int] = set()
        attempted = 0
        discharged_cnt = 0

        for ld, lid, lexpr in l_cands:
            if attempted >= self.max_candidates:
                break
            if lid in covered_l:
                continue
            fp = _collect_leaf_signature(lexpr)
            if not fp:
                continue
            bucket = r_buckets.get((lexpr.sort().kind(), fp))
            if not bucket:
                continue
            for rd, rexpr in bucket:
                rid = rexpr.get_id()
                if rid in covered_r:
                    continue
                if lexpr.sort() != rexpr.sort():
                    continue
                if abs(ld - rd) > 3:
                    continue
                attempted += 1
                ok, dt_wlp = _discharge_wlp_local(
                    lexpr, rexpr, query.guard, query.preconditions, ctx.wlp_timeout_ms
                )
                if ok:
                    fv = ctx.fresh_const("cu", lexpr.sort())
                    ob = SideObligation(
                        obligation_id=f"cu_{discharged_cnt}",
                        fresh_var=fv,
                        lhs=lexpr,
                        rhs=rexpr,
                        guard=query.guard,
                        reason=f"leaf_footprint_match(leaves={len(fp)},depths={ld}/{rd})",
                        discharged=True,
                        elapsed_ms=dt_wlp,
                    )
                    obligations.append(ob)
                    l_subs.append((lexpr, fv))
                    r_subs.append((rexpr, fv))
                    abs_pairs.append((fv, lexpr))
                    sub_l, _ = _collect_subexprs_with_depth(lexpr, limit=2000)
                    sub_r, _ = _collect_subexprs_with_depth(rexpr, limit=2000)
                    covered_l.update(sub_l.keys())
                    covered_r.update(sub_r.keys())
                    discharged_cnt += 1
                    break
                if attempted >= self.max_candidates:
                    break

        if discharged_cnt == 0:
            dt = (time.perf_counter() - t0) * 1000.0
            return query, PassStats(self.name, False, 0, dt, f"attempted={attempted} discharged=0")

        new_lhs = z3.substitute(query.lhs, *l_subs)
        new_rhs = z3.substitute(query.rhs, *r_subs)
        out_q = query.with_terms(
            lhs=new_lhs,
            rhs=new_rhs,
            side_obligations=tuple(obligations),
            is_overapprox=True,
            abstraction_map=query.abstraction_map + tuple(abs_pairs),
            notes=query.notes + (f"cond_unified={discharged_cnt}/{attempted}",),
        )
        dt = (time.perf_counter() - t0) * 1000.0
        return out_q, PassStats(self.name, True, discharged_cnt, dt, f"discharged={discharged_cnt}/{attempted}")

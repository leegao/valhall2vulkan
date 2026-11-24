#!/usr/bin/env python3
"""
Array scalarization and theory purification passes for `z3simplifier`.

Addresses Issue 2 (Mixed-theory `QF_AUFBVNRA` blocking Z3's `nlsat` engine):
  1. `ReadOnlyArrayScalarizePass` (100% equivalence-preserving):
     Scans all expressions in `SimplifyQuery` (`lhs`, `rhs`, `guard`,
     `preconditions`). For every uninterpreted `Array` constant `A` that:
       - never appears as the target of a `Store`,
       - never appears bare (outside `Select(A, idx)`), and
       - is ONLY indexed by concrete constant indices (`BitVecNumRef` / `IntNumRef`),
     replaces every `Select(A, k)` with a unique cached scalar constant
     `ro_arr_<A>_<k>` of the array's range sort.
     By the theory of arrays (extensionality over distinct constant indices),
     this is 100% equi-satisfiable and completely strips `Array` theory from
     UBO, push-constant, and read-only SSBO queries.

  2. `TheoryPurifyRealPass` (UNSAT-sound over-approximation):
     When `lhs` and `rhs` are Z3 `Real` expressions — or `Bool` expressions
     containing `Real`-sorted comparisons (such as fragment `discarded` checks) —
     replaces maximal non-Real-theory subterms of `Real` sort (such as
     `u_bv32_to_f32(bv_expr)`, `ToReal(BV2Int(...))`, or transcendental UFs
     `real_sin(...)`) and foreign `Bool`-sorted atoms (such as BV comparisons)
     with shared fresh constants. Identical subterms across `lhs`, `rhs`, and
     `guard` map to the SAME fresh constant!
     The resulting query is pure `QF_NRA`, allowing Z3 to dispatch directly
     to `nlsat` (`Tactic("qfnra")`).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Set, Tuple

import tools.symbolic_ssa.z3utils.dsl as dsl

from tools.symbolic_ssa.z3simplifier.ast_utils import rewrite_dag
from tools.symbolic_ssa.z3simplifier.context import (
    PassStats,
    SimplificationContext,
    SimplifyQuery,
)
from tools.symbolic_ssa.z3simplifier.pass_base import Z3QueryPass


def _is_concrete_index(idx: dsl.ExprRef) -> Optional[str]:
    if dsl.is_bv_value(idx):
        return f"bv{idx.size()}_{idx.as_long():x}"
    if dsl.is_int_value(idx):
        return f"int_{idx.as_long()}"
    return None


def _is_uninterpreted_array_const(expr: dsl.ExprRef) -> bool:
    return (
        expr.sort().kind() == dsl.Z3_ARRAY_SORT
        and dsl.is_const(expr)
        and expr.decl().kind() == dsl.Z3_OP_UNINTERPRETED
    )


def _find_scalarizable_arrays(roots: List[dsl.ExprRef]) -> Set[int]:
    """
    Return the set of Z3 AST IDs of uninterpreted Array constants that are
    purely read-only with strictly concrete indices across all `roots`.
    """
    seen_arrays: Set[int] = set()
    disqualified: Set[int] = set()
    visited: Set[int] = set()
    stack: List[dsl.ExprRef] = [r for r in roots if r is not None]

    while stack:
        cur = stack.pop()
        cid = cur.get_id()
        if cid in visited:
            continue
        visited.add(cid)
        if not dsl.is_app(cur):
            continue
        kind = cur.decl().kind()
        if kind == dsl.Z3_OP_SELECT and cur.num_args() == 2:
            arr, idx = cur.arg(0), cur.arg(1)
            if _is_uninterpreted_array_const(arr):
                aid = arr.get_id()
                seen_arrays.add(aid)
                if _is_concrete_index(idx) is None:
                    disqualified.add(aid)
                if idx.get_id() not in visited:
                    stack.append(idx)
                continue
        if kind == dsl.Z3_OP_STORE and cur.num_args() == 3:
            arr = cur.arg(0)
            if _is_uninterpreted_array_const(arr):
                disqualified.add(arr.get_id())
        if _is_uninterpreted_array_const(cur):
            disqualified.add(cid)
            continue
        for i in range(cur.num_args()):
            ch = cur.arg(i)
            if ch.get_id() not in visited:
                stack.append(ch)

    return seen_arrays - disqualified


class ReadOnlyArrayScalarizePass(Z3QueryPass):
    """
    Soundly scalarizes `Select(A, const_k)` for read-only arrays `A` accessed
    solely at concrete indices across the entire query.
    """

    name = "readonly_array_scalarize"
    is_equivalence_preserving = True

    def run(
        self,
        query: SimplifyQuery,
        ctx: SimplificationContext,
    ) -> Tuple[SimplifyQuery, PassStats]:
        t0 = time.perf_counter()
        if query.is_trivially_unsat():
            return query, PassStats(self.name, False, 0, 0.0, "trivial_unsat")

        roots: List[dsl.ExprRef] = [query.lhs, query.rhs]
        if query.guard is not None:
            roots.append(query.guard)
        roots.extend(query.preconditions)

        scalarizable = _find_scalarizable_arrays(roots)
        if not scalarizable:
            return query, PassStats(self.name, False, 0, (time.perf_counter() - t0) * 1000.0)

        def _rule(node: dsl.ExprRef) -> Optional[dsl.ExprRef]:
            if dsl.is_app(node) and node.decl().kind() == dsl.Z3_OP_SELECT and node.num_args() == 2:
                arr, idx = node.arg(0), node.arg(1)
                if arr.get_id() in scalarizable:
                    idx_tag = _is_concrete_index(idx)
                    if idx_tag is not None:
                        arr_name = arr.decl().name()
                        return ctx.intern_const(
                            ("ro_arr", arr_name, idx_tag),
                            f"ro_{arr_name}_{idx_tag}",
                            node.sort(),
                        )
            return None

        memo: Dict[int, dsl.ExprRef] = {}
        new_lhs, r1 = rewrite_dag(query.lhs, _rule, memo=memo)
        new_rhs, r2 = rewrite_dag(query.rhs, _rule, memo=memo)
        total_rw = r1 + r2

        new_guard = query.guard
        if query.guard is not None:
            new_guard, rg = rewrite_dag(query.guard, _rule, memo=memo)
            total_rw += rg

        new_pres: List[dsl.BoolRef] = []
        for pre in query.preconditions:
            np, rp = rewrite_dag(pre, _rule, memo=memo)
            new_pres.append(np)
            total_rw += rp

        changed = total_rw > 0
        out_q = (
            query.with_terms(
                lhs=new_lhs,
                rhs=new_rhs,
                guard=new_guard,
                preconditions=tuple(new_pres),
                notes=query.notes + (f"scalarized_arrays={len(scalarizable)}",),
            )
            if changed
            else query
        )
        dt = (time.perf_counter() - t0) * 1000.0
        return out_q, PassStats(self.name, changed, total_rw, dt, f"arrays={len(scalarizable)}")


_PURE_REAL_ARITH_KINDS: Set[int] = {
    dsl.Z3_OP_ADD,
    dsl.Z3_OP_SUB,
    dsl.Z3_OP_UMINUS,
    dsl.Z3_OP_MUL,
    dsl.Z3_OP_DIV,
    dsl.Z3_OP_POWER,
    dsl.Z3_OP_ANUM,
    dsl.Z3_OP_EQ,
    dsl.Z3_OP_DISTINCT,
    dsl.Z3_OP_LT,
    dsl.Z3_OP_LE,
    dsl.Z3_OP_GT,
    dsl.Z3_OP_GE,
    dsl.Z3_OP_AND,
    dsl.Z3_OP_OR,
    dsl.Z3_OP_NOT,
    dsl.Z3_OP_IMPLIES,
    dsl.Z3_OP_XOR,
    dsl.Z3_OP_TRUE,
    dsl.Z3_OP_FALSE,
    dsl.Z3_OP_ITE,
}


def _contains_real_sort(roots: List[dsl.ExprRef], limit: int = 50000) -> bool:
    """Return True if any reachable subterm in `roots` has `Real` sort."""
    visited: Set[int] = set()
    stack: List[dsl.ExprRef] = [r for r in roots if r is not None]
    while stack and len(visited) < limit:
        cur = stack.pop()
        cid = cur.get_id()
        if cid in visited:
            continue
        visited.add(cid)
        if cur.sort().kind() == dsl.Z3_REAL_SORT:
            return True
        if dsl.is_app(cur):
            for i in range(cur.num_args()):
                ch = cur.arg(i)
                if ch.get_id() not in visited:
                    stack.append(ch)
    return False


def _is_pure_qfnra_expr(expr: dsl.ExprRef) -> bool:
    """Return True if every node in `expr` belongs strictly to `QF_NRA`."""
    visited: Set[int] = set()
    stack: List[dsl.ExprRef] = [expr]
    while stack:
        cur = stack.pop()
        cid = cur.get_id()
        if cid in visited:
            continue
        visited.add(cid)
        sk = cur.sort().kind()
        if sk not in (dsl.Z3_REAL_SORT, dsl.Z3_BOOL_SORT):
            return False
        if dsl.is_const(cur):
            continue
        if not dsl.is_app(cur):
            return False
        if cur.decl().kind() not in _PURE_REAL_ARITH_KINDS:
            return False
        for i in range(cur.num_args()):
            stack.append(cur.arg(i))
    return True


class TheoryPurifyRealPass(Z3QueryPass):
    """
    Purifies a `Real`-sorted or `Bool`-over-`Real` miter into pure `QF_NRA` by
    abstracting maximal foreign-theory `Real`-sorted subterms (e.g.,
    `u_bv32_to_f32(bv_expr)`, `ToReal(...)`, transcendental UFs) and foreign
    `Bool`-sorted atoms (e.g., BV comparisons) into shared fresh constants.

    Identical foreign subterms across `lhs`, `rhs`, and `guard` map to the
    same fresh constant (preserving congruence!). If the resulting formula is
    pure `QF_NRA`, sets `pure_theory="QF_NRA"` so the solver can invoke Z3's
    complete `nlsat` engine.

    Marked `is_equivalence_preserving = False` (sound for UNSAT; any SAT must
    be validated against `exact_query`).
    """

    name = "theory_purify_real"
    is_equivalence_preserving = False

    def run(
        self,
        query: SimplifyQuery,
        ctx: SimplificationContext,
    ) -> Tuple[SimplifyQuery, PassStats]:
        t0 = time.perf_counter()
        if query.is_trivially_unsat():
            return query, PassStats(self.name, False, 0, 0.0, "trivial_unsat")

        l_sk = query.lhs.sort().kind()
        r_sk = query.rhs.sort().kind()
        if l_sk != r_sk:
            return query, PassStats(self.name, False, 0, (time.perf_counter() - t0) * 1000.0)
        if l_sk == dsl.Z3_BOOL_SORT:
            if not _contains_real_sort([query.lhs, query.rhs], limit=ctx.max_dag_nodes):
                return query, PassStats(self.name, False, 0, (time.perf_counter() - t0) * 1000.0)
        elif l_sk != dsl.Z3_REAL_SORT:
            return query, PassStats(self.name, False, 0, (time.perf_counter() - t0) * 1000.0)

        abs_pairs: List[Tuple[dsl.ExprRef, dsl.ExprRef]] = []
        foreign_map: Dict[int, dsl.ExprRef] = {}

        def _abstract_node(node: dsl.ExprRef, prefix: str, sort: dsl.SortRef) -> dsl.ExprRef:
            nid = node.get_id()
            existing = foreign_map.get(nid)
            if existing is not None:
                return existing
            fv = ctx.fresh_const(prefix, sort)
            foreign_map[nid] = fv
            abs_pairs.append((fv, node))
            return fv

        def _purify_rule(node: dsl.ExprRef) -> Optional[dsl.ExprRef]:
            if not dsl.is_app(node) or dsl.is_const(node):
                return None
            sk = node.sort().kind()
            kind = node.decl().kind()
            if sk == dsl.Z3_REAL_SORT:
                if kind in _PURE_REAL_ARITH_KINDS:
                    if kind == dsl.Z3_OP_ITE and not _is_pure_qfnra_expr(node.arg(0)):
                        return _abstract_node(node, "purify_ite", dsl.RealSort())
                    return None
                return _abstract_node(node, "purify_r", dsl.RealSort())
            if sk == dsl.Z3_BOOL_SORT:
                if kind not in _PURE_REAL_ARITH_KINDS:
                    return _abstract_node(node, "purify_b", dsl.BoolSort())
                if kind in (dsl.Z3_OP_EQ, dsl.Z3_OP_DISTINCT):
                    for i in range(node.num_args()):
                        if node.arg(i).sort().kind() not in (dsl.Z3_REAL_SORT, dsl.Z3_BOOL_SORT):
                            return _abstract_node(node, "purify_b", dsl.BoolSort())
                return None
            return None

        memo: Dict[int, dsl.ExprRef] = {}
        new_lhs, r1 = rewrite_dag(query.lhs, _purify_rule, memo=memo)
        new_rhs, r2 = rewrite_dag(query.rhs, _purify_rule, memo=memo)
        total_rw = r1 + r2

        new_guard = query.guard
        if query.guard is not None:
            new_guard, rg = rewrite_dag(query.guard, _purify_rule, memo=memo)
            total_rw += rg

        pure_pres: List[dsl.BoolRef] = []
        dropped_pres = 0
        for pre in query.preconditions:
            np, _ = rewrite_dag(pre, _purify_rule, memo=memo)
            if _is_pure_qfnra_expr(np):
                pure_pres.append(np)
            else:
                dropped_pres += 1

        eff_guard = new_guard
        if eff_guard is not None and not _is_pure_qfnra_expr(eff_guard):
            eff_guard = None
            dropped_pres += 1

        is_pure = _is_pure_qfnra_expr(new_lhs) and _is_pure_qfnra_expr(new_rhs) and (
            eff_guard is None or _is_pure_qfnra_expr(eff_guard)
        )
        changed = total_rw > 0 or dropped_pres > 0 or (is_pure and query.pure_theory != "QF_NRA")
        if not changed:
            return query, PassStats(self.name, False, 0, (time.perf_counter() - t0) * 1000.0)

        out_q = query.with_terms(
            lhs=new_lhs,
            rhs=new_rhs,
            guard=eff_guard,
            preconditions=tuple(pure_pres),
            is_overapprox=True if (total_rw > 0 or dropped_pres > 0) else query.is_overapprox,
            pure_theory="QF_NRA" if is_pure else query.pure_theory,
            abstraction_map=query.abstraction_map + tuple(abs_pairs),
            notes=query.notes + (f"purified_leaves={len(abs_pairs)},pure={is_pure}",),
        )
        dt = (time.perf_counter() - t0) * 1000.0
        return out_q, PassStats(self.name, True, total_rw + dropped_pres, dt, f"leaves={len(abs_pairs)} pure={is_pure}")

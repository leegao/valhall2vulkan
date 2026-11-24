#!/usr/bin/env python3
"""
Cross-Side Common-Subexpression Unification & Conditioned Unification Passes.

1. `ExactCseUnificationPass` (UNSAT-sound over-approximation):
   Finds maximal non-trivial subexpressions `e` that appear identically in BOTH
   `lhs` and `rhs` (by Z3 AST identity `get_id()`, leveraging Z3's internal
   hash-consing), and replaces every occurrence of `e` with a fresh typed
   symbolic constant `cse_k`.
   - Soundness: Replacing a shared subexpression `e` with an unconstrained
     fresh variable `cse_k` strictly weakens the domain of `e`. Therefore:
       * Abstract `UNSAT` => Original `UNSAT` (100% sound).
       * Abstract `SAT` may be spurious, so `is_overapprox=True` triggers
         exact refinement in `solve_simplified_miter`.

2. `ConditionedUnificationPass` (congruence-based unification with WLP discharge):
   Combines two complementary strategies:
   - **Leaf-footprint subtree matching**: pairs non-identical subtrees
     `(e_l, e_r)` across `lhs` and `rhs` that share the same sort and free
     variable footprint (e.g., FMA reassociation under `hw_real_sqrt`),
     discharges `(preconditions & guard) => (e_l == e_r)` locally, and
     replaces proved pairs with a shared fresh symbol `cu_k`.
   - **Top-down structural anti-unification** (`enable_anti_unify=True`):
     aligns structurally similar `lhs` and `rhs` (modulo commutative ops and
     `If` branch orientation) to extract the minimal set of mismatch holes
     `{(a_1, b_1), ..., (a_k, b_k)}` whose conjunction implies `lhs == rhs`
     by congruence. Attempts local WLP discharge on each hole (including
     bounded expansion through `query.abstraction_map` for purified leaves):
       * Globally discharged holes are unified to `cu_k` in `(lhs, rhs)` and
         recorded in `query.side_obligations` (`discharged=True`).
       * If every hole is discharged (globally or under its `If` path guard),
         the entire miter is proved `UNSAT` (`predicate = BoolVal(False)`).
       * Any undischarged holes are recorded in `query.residual_obligations`
         (`discharged=False`) for diagnostics and downstream inspection.
"""

from __future__ import annotations

import time

import z3

from tools.symbolic_ssa import z3utils
from tools.symbolic_ssa.z3simplifier.ast_utils import count_ast_nodes
from tools.symbolic_ssa.z3simplifier.context import (
    PassStats,
    SideObligation,
    SimplificationContext,
    SimplifyQuery,
)
from tools.symbolic_ssa.z3simplifier.pass_base import Z3QueryPass
from tools.symbolic_ssa.z3utils import dsl

_COMMUTATIVE_KINDS: frozenset[int] = frozenset(
    {
        dsl.Z3_OP_ADD,
        dsl.Z3_OP_MUL,
        dsl.Z3_OP_AND,
        dsl.Z3_OP_OR,
        dsl.Z3_OP_EQ,
        dsl.Z3_OP_BADD,
        dsl.Z3_OP_BMUL,
        dsl.Z3_OP_BAND,
        dsl.Z3_OP_BOR,
        dsl.Z3_OP_BXOR,
    }
)


def _collect_subexprs_with_depth(
    root: dsl.ExprRef,
    limit: int = 50000,
) -> tuple[dict[int, dsl.ExprRef], dict[int, int]]:
    """Iteratively collect all reachable subexpressions and their AST depths."""
    nodes: dict[int, dsl.ExprRef] = {}
    depths: dict[int, int] = {}
    visited: set[int] = set()
    stack: list[tuple[dsl.ExprRef, bool]] = [(root, False)]

    while stack and len(visited) < limit:
        cur, expanded = stack.pop()
        cid = cur.get_id()
        if expanded:
            if cid not in depths:
                if dsl.is_app(cur) and cur.num_args() > 0:
                    depths[cid] = 1 + max(depths.get(cur.arg(i).get_id(), 0) for i in range(cur.num_args()))
                else:
                    depths[cid] = 0
                nodes[cid] = cur
            continue
        if cid in visited:
            continue
        visited.add(cid)
        stack.append((cur, True))
        if dsl.is_app(cur):
            for i in range(cur.num_args() - 1, -1, -1):
                ch = cur.arg(i)
                if ch.get_id() not in visited:
                    stack.append((ch, False))
    return nodes, depths


def _collect_all_leaf_signatures(
    root: dsl.ExprRef,
    limit: int = 50000,
) -> dict[int, frozenset[int]]:
    """Compute free-variable leaf signatures for all reachable subexpressions in one O(N) DAG pass."""
    sigs: dict[int, frozenset[int]] = {}
    visited: set[int] = set()
    stack: list[tuple[dsl.ExprRef, bool]] = [(root, False)]

    while stack and len(visited) < limit:
        cur, expanded = stack.pop()
        cid = cur.get_id()
        if expanded:
            if cid not in sigs:
                if dsl.is_const(cur):
                    sigs[cid] = frozenset((cid,)) if cur.decl().kind() == dsl.Z3_OP_UNINTERPRETED else frozenset()
                elif dsl.is_app(cur) and cur.num_args() > 0:
                    acc: set[int] = set()
                    for i in range(cur.num_args()):
                        ch_sig = sigs.get(cur.arg(i).get_id())
                        if ch_sig:
                            acc.update(ch_sig)
                    sigs[cid] = frozenset(acc)
                else:
                    sigs[cid] = frozenset()
            continue
        if cid in visited:
            continue
        visited.add(cid)
        stack.append((cur, True))
        if dsl.is_app(cur) and not dsl.is_const(cur):
            for i in range(cur.num_args() - 1, -1, -1):
                ch = cur.arg(i)
                if ch.get_id() not in visited:
                    stack.append((ch, False))
    return sigs


def _collect_leaf_signature(
    root: dsl.ExprRef,
    limit: int = 2000,
) -> frozenset[int]:
    """Collect the frozenset of free symbolic constant IDs referenced by `root`."""
    leaves: set[int] = set()
    visited: set[int] = set()
    stack: list[dsl.ExprRef] = [root]
    while stack and len(visited) < limit:
        cur = stack.pop()
        cid = cur.get_id()
        if cid in visited:
            continue
        visited.add(cid)
        if dsl.is_const(cur):
            if cur.decl().kind() == dsl.Z3_OP_UNINTERPRETED:
                leaves.add(cid)
            continue
        if dsl.is_app(cur):
            for i in range(cur.num_args()):
                ch = cur.arg(i)
                if ch.get_id() not in visited:
                    stack.append(ch)
    return frozenset(leaves)


def _is_liftable_subexpr(node: dsl.ExprRef, depth: int, min_depth: int) -> bool:
    """Return True if `node` is a non-trivial scalar subexpression worth lifting."""
    if depth < min_depth:
        return False
    if not dsl.is_app(node) or node.num_args() == 0:
        return False
    sk = node.sort().kind()
    return sk in (dsl.Z3_BV_SORT, dsl.Z3_REAL_SORT, dsl.Z3_INT_SORT, dsl.Z3_BOOL_SORT)


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
    ) -> tuple[SimplifyQuery, PassStats]:
        t0 = time.perf_counter()
        if query.is_trivially_unsat():
            return query, PassStats(self.name, False, 0, 0.0, "trivial_unsat")

        l_nodes, l_depths = _collect_subexprs_with_depth(query.lhs, limit=ctx.max_dag_nodes)
        r_nodes, _ = _collect_subexprs_with_depth(query.rhs, limit=ctx.max_dag_nodes)

        shared_ids = (set(l_nodes.keys()) & set(r_nodes.keys())) - {query.lhs.get_id(), query.rhs.get_id()}
        if not shared_ids:
            return query, PassStats(self.name, False, 0, (time.perf_counter() - t0) * 1000.0)

        candidates = [
            (l_depths[nid], nid, l_nodes[nid])
            for nid in shared_ids
            if _is_liftable_subexpr(l_nodes[nid], l_depths[nid], self.min_depth)
        ]
        if not candidates:
            return query, PassStats(self.name, False, 0, (time.perf_counter() - t0) * 1000.0)

        candidates.sort(key=lambda x: (-x[0], x[1]))

        covered: set[int] = set()
        repl_map: dict[int, dsl.ExprRef] = {}
        abs_pairs: list[tuple[dsl.ExprRef, dsl.ExprRef]] = []

        for _, nid, expr in candidates:
            if len(repl_map) >= ctx.max_unify_candidates:
                break
            if nid in covered:
                continue
            fv = ctx.fresh_const("cse", expr.sort())
            repl_map[nid] = fv
            abs_pairs.append((fv, expr))
            sub_nodes, _ = _collect_subexprs_with_depth(expr, limit=2000)
            covered.update(sub_nodes.keys())

        if not repl_map:
            return query, PassStats(self.name, False, 0, (time.perf_counter() - t0) * 1000.0)

        sub_pairs = [(expr, fv) for fv, expr in abs_pairs]
        new_lhs = dsl.substitute(query.lhs, *sub_pairs)
        new_rhs = dsl.substitute(query.rhs, *sub_pairs)
        new_guard = dsl.substitute(query.guard, *sub_pairs) if query.guard is not None else None
        new_pres = tuple(dsl.substitute(p, *sub_pairs) for p in query.preconditions)

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


def _contains_ite(roots: list[dsl.ExprRef], limit: int = 64) -> bool:
    """Return True if any reachable node in `roots` is an `If` (`Z3_OP_ITE`)."""
    visited: set[int] = set()
    stack: list[dsl.ExprRef] = list(roots)
    while stack and len(visited) < limit:
        cur = stack.pop()
        cid = cur.get_id()
        if cid in visited:
            continue
        visited.add(cid)
        if dsl.is_app(cur):
            if cur.decl().kind() == dsl.Z3_OP_ITE:
                return True
            for i in range(cur.num_args()):
                ch = cur.arg(i)
                if ch.get_id() not in visited:
                    stack.append(ch)
    return False


class _IncrementalWlpSolver:
    """
    Reusable incremental Z3 solver for local WLP obligations in a single pass run.

    Avoids re-asserting shared preconditions and guards into a fresh `z3utils.Solver`
    for every candidate pair, and skips asserting oversized guards (>256 nodes)
    where `Z3_solver_assert` itself can take >1 s ignoring solver timeouts
    (dropping a guard when proving `guard => (lhs == rhs)` is 100% sound).
    """

    def __init__(
        self,
        guard: dsl.BoolRef | None,
        preconditions: tuple[dsl.BoolRef, ...],
        timeout_ms: int,
        max_guard_nodes: int = 256,
    ):
        self.timeout_ms = max(10, int(timeout_ms))
        self.small_pres = tuple(
            p for p in preconditions if count_ast_nodes([p], limit=max_guard_nodes + 1) <= max_guard_nodes
        )
        self.usable_guard = (
            guard
            if (guard is not None and count_ast_nodes([guard], limit=max_guard_nodes + 1) <= max_guard_nodes)
            else None
        )
        self._uncond_solver: z3utils.Solver | None = None
        self._guarded_solver: z3utils.Solver | None = None

    def _get_uncond_solver(self) -> z3utils.Solver:
        if self._uncond_solver is None:
            s = z3utils.Solver()
            s.set("timeout", self.timeout_ms)
            s.set("rlimit", 200_000)
            for pre in self.small_pres:
                s.add(dsl.lower(pre))
            self._uncond_solver = s
        return self._uncond_solver

    def _get_guarded_solver(self) -> z3utils.Solver | None:
        if self.usable_guard is None:
            return None
        if self._guarded_solver is None:
            s = z3utils.Solver()
            s.set("timeout", self.timeout_ms)
            s.set("rlimit", 200_000)
            for pre in self.small_pres:
                s.add(dsl.lower(pre))
            s.add(dsl.lower(self.usable_guard))
            self._guarded_solver = s
        return self._guarded_solver

    def check_eq(self, lhs: dsl.ExprRef, rhs: dsl.ExprRef, extra_guard: dsl.BoolRef | None = None) -> tuple[bool, float]:
        t0 = time.perf_counter()
        if lhs.eq(rhs):
            return True, (time.perf_counter() - t0) * 1000.0
        if lhs.sort() != rhs.sort():
            return False, (time.perf_counter() - t0) * 1000.0

        if lhs.sort().kind() == dsl.Z3_REAL_SORT and rhs.sort().kind() == dsl.Z3_REAL_SORT:
            if count_ast_nodes([lhs, rhs], limit=49) <= 48 and not _contains_ite([lhs, rhs]):
                diff = dsl.simplify(lhs - rhs, som=True, arith_lhs=True)
                if dsl.is_rational_value(diff) and diff.numerator_as_long() == 0:
                    return True, (time.perf_counter() - t0) * 1000.0

        neq = dsl.lower(lhs != rhs)
        # Fast unconditional check first (avoids asserting guard when equality holds unconditionally)
        s0 = self._get_uncond_solver()
        s0.push()
        s0.add(neq)
        rc = s0.check()
        s0.pop()
        if rc == z3.unsat:
            return True, (time.perf_counter() - t0) * 1000.0

        # Guarded check (if usable_guard or extra_guard is present)
        if self.usable_guard is not None or extra_guard is not None:
            sg = self._get_guarded_solver() if self.usable_guard is not None else s0
            sg.push()
            if extra_guard is not None and count_ast_nodes([extra_guard], limit=257) <= 256:
                sg.add(dsl.lower(extra_guard))
            sg.add(neq)
            rc = sg.check()
            sg.pop()
            if rc == z3.unsat:
                return True, (time.perf_counter() - t0) * 1000.0

        return False, (time.perf_counter() - t0) * 1000.0


def _discharge_wlp_local(
    lhs: dsl.ExprRef,
    rhs: dsl.ExprRef,
    guard: dsl.BoolRef | None,
    preconditions: tuple[dsl.BoolRef, ...],
    timeout_ms: int,
    *,
    wlp_solver: _IncrementalWlpSolver | None = None,
    extra_guard: dsl.BoolRef | None = None,
) -> tuple[bool, float]:
    """
    Attempt to prove `(preconditions & guard) => (lhs == rhs)` locally.

    Returns `(is_valid, elapsed_ms)`. Uses a fast polynomial check for Reals
    before falling back to a bounded incremental Z3 solver check.
    """
    if wlp_solver is not None:
        return wlp_solver.check_eq(lhs, rhs, extra_guard=extra_guard)
    eff_guard = _combine_guard(guard, extra_guard) if extra_guard is not None else guard
    return _IncrementalWlpSolver(eff_guard, preconditions, timeout_ms).check_eq(lhs, rhs)


def _same_operator_head(lhs: dsl.ExprRef, rhs: dsl.ExprRef) -> bool:
    """Return True if `lhs` and `rhs` are non-nullary applications of the same operator."""
    if not (dsl.is_app(lhs) and dsl.is_app(rhs)):
        return False
    if lhs.num_args() == 0 or lhs.num_args() != rhs.num_args():
        return False
    ld, rd = lhs.decl(), rhs.decl()
    lk, rk = ld.kind(), rd.kind()
    if lk != rk:
        return False
    if lk == dsl.Z3_OP_UNINTERPRETED:
        return ld.name() == rd.name()
    if ld.params() != rd.params():
        return False
    return True


def _pair_similarity_score(a: dsl.ExprRef, b: dsl.ExprRef) -> int:
    """Fast O(1) structural similarity score for aligning commutative children."""
    if a.eq(b):
        return 4
    if _same_operator_head(a, b):
        return 2
    if a.sort() == b.sort():
        return 1
    return 0


def _align_children(
    l_args: list[dsl.ExprRef],
    r_args: list[dsl.ExprRef],
    commutative: bool,
) -> list[tuple[dsl.ExprRef, dsl.ExprRef]]:
    """Align argument lists of two same-operator nodes, handling commutativity."""
    n = len(l_args)
    if not commutative or n <= 1:
        return list(zip(l_args, r_args))
    if n == 2:
        direct = _pair_similarity_score(l_args[0], r_args[0]) + _pair_similarity_score(l_args[1], r_args[1])
        swapped = _pair_similarity_score(l_args[0], r_args[1]) + _pair_similarity_score(l_args[1], r_args[0])
        if swapped > direct:
            return [(l_args[0], r_args[1]), (l_args[1], r_args[0])]
        return [(l_args[0], r_args[0]), (l_args[1], r_args[1])]

    used_r: set[int] = set()
    aligned: list[tuple[dsl.ExprRef, dsl.ExprRef]] = []
    unmatched_l: list[dsl.ExprRef] = []
    for la in l_args:
        found = -1
        for j, ra in enumerate(r_args):
            if j not in used_r and la.eq(ra):
                found = j
                break
        if found >= 0:
            used_r.add(found)
            aligned.append((la, r_args[found]))
        else:
            unmatched_l.append(la)
    for la in unmatched_l:
        best_j = -1
        best_score = -1
        for j, ra in enumerate(r_args):
            if j in used_r:
                continue
            sc = _pair_similarity_score(la, ra)
            if sc > best_score:
                best_score = sc
                best_j = j
        if best_j >= 0:
            used_r.add(best_j)
            aligned.append((la, r_args[best_j]))
    return aligned


def _are_negated_bools(a: dsl.ExprRef, b: dsl.ExprRef) -> bool:
    """Return True if `a` and `b` are syntactic boolean negations (including Real cmp complements)."""
    if dsl.is_app(b) and b.decl().kind() == dsl.Z3_OP_NOT and b.num_args() == 1 and a.eq(b.arg(0)):
        return True
    if dsl.is_app(a) and a.decl().kind() == dsl.Z3_OP_NOT and a.num_args() == 1 and b.eq(a.arg(0)):
        return True
    if dsl.is_app(a) and dsl.is_app(b) and a.num_args() == 2 and b.num_args() == 2:
        ak, bk = a.decl().kind(), b.decl().kind()
        a0, a1 = a.arg(0), a.arg(1)
        b0, b1 = b.arg(0), b.arg(1)
        # (x < y) <-> !(x >= y) <-> !(y <= x); (x <= y) <-> !(x > y) <-> !(y < x)
        if a0.eq(b0) and a1.eq(b1):
            if (ak, bk) in (
                (dsl.Z3_OP_LT, dsl.Z3_OP_GE),
                (dsl.Z3_OP_GE, dsl.Z3_OP_LT),
                (dsl.Z3_OP_LE, dsl.Z3_OP_GT),
                (dsl.Z3_OP_GT, dsl.Z3_OP_LE),
            ):
                return True
        if a0.eq(b1) and a1.eq(b0):
            if (ak, bk) in (
                (dsl.Z3_OP_LT, dsl.Z3_OP_LE),
                (dsl.Z3_OP_LE, dsl.Z3_OP_LT),
                (dsl.Z3_OP_GT, dsl.Z3_OP_GE),
                (dsl.Z3_OP_GE, dsl.Z3_OP_GT),
            ):
                return True
    return False


def _anti_unify_pair(
    lhs: dsl.ExprRef,
    rhs: dsl.ExprRef,
    max_holes: int = 32,
) -> tuple[int, list[tuple[dsl.ExprRef, dsl.ExprRef, dsl.BoolRef | None]], bool]:
    """
    Compute top-down DAG-memoized structural anti-unification of `(lhs, rhs)`.

    Returns `(matched_ops, holes, overflowed)`:
      - `matched_ops`: count of shared spine operator nodes matched.
      - `holes`: deduplicated list of `(a_i, b_i, path_cond)` mismatch pairs
        whose conjunction implies `lhs == rhs` by congruence. If a pair is
        reached under multiple distinct path conditions, `path_cond` widens
        to `None` (unconditional).
      - `overflowed`: True if more than `max_holes` unique holes were found.
    """
    matched_ops = 0
    hole_index: dict[tuple[int, int], int] = {}
    holes: list[tuple[dsl.ExprRef, dsl.ExprRef, dsl.BoolRef | None]] = []
    visited: set[tuple[int, int, int]] = set()
    overflowed = False

    def _record_hole(a: dsl.ExprRef, b: dsl.ExprRef, pcond: dsl.BoolRef | None) -> None:
        nonlocal overflowed
        key = (a.get_id(), b.get_id())
        idx = hole_index.get(key)
        if idx is not None:
            prev_a, prev_b, prev_p = holes[idx]
            if prev_p is not None and (pcond is None or not prev_p.eq(pcond)):
                holes[idx] = (prev_a, prev_b, None)
            return
        if len(holes) >= max_holes:
            overflowed = True
            return
        hole_index[key] = len(holes)
        holes.append((a, b, pcond))

    def _combine_cond(base: dsl.BoolRef | None, extra: dsl.BoolRef) -> dsl.BoolRef:
        return extra if base is None else dsl.And(base, extra)

    stack: list[tuple[dsl.ExprRef, dsl.ExprRef, dsl.BoolRef | None]] = [(lhs, rhs, None)]
    while stack and not overflowed:
        a, b, pcond = stack.pop()
        if a.eq(b):
            continue
        p_id = pcond.get_id() if pcond is not None else 0
        vkey = (a.get_id(), b.get_id(), p_id)
        if vkey in visited:
            continue
        visited.add(vkey)

        if not _same_operator_head(a, b):
            _record_hole(a, b, pcond)
            continue

        matched_ops += 1
        kind = a.decl().kind()
        if kind == dsl.Z3_OP_ITE and a.num_args() == 3:
            ca, ta, fa = a.arg(0), a.arg(1), a.arg(2)
            cb, tb, fb = b.arg(0), b.arg(1), b.arg(2)
            if ca.eq(cb):
                stack.append((fa, fb, _combine_cond(pcond, dsl.Not(ca))))
                stack.append((ta, tb, _combine_cond(pcond, ca)))
            elif _are_negated_bools(ca, cb):
                stack.append((fa, tb, _combine_cond(pcond, dsl.Not(ca))))
                stack.append((ta, fb, _combine_cond(pcond, ca)))
            else:
                stack.append((fa, fb, pcond))
                stack.append((ta, tb, pcond))
                stack.append((ca, cb, pcond))
            continue

        l_args = [a.arg(i) for i in range(a.num_args())]
        r_args = [b.arg(i) for i in range(b.num_args())]
        aligned = _align_children(l_args, r_args, kind in _COMMUTATIVE_KINDS)
        for la, ra in reversed(aligned):
            stack.append((la, ra, pcond))

    return matched_ops, holes, overflowed


def _expand_abstractions(
    expr: dsl.ExprRef,
    abs_subs: list[tuple[dsl.ExprRef, dsl.ExprRef]],
    abs_ids: set[int],
    max_nodes: int = 256,
) -> dsl.ExprRef:
    """Expand abstracted symbols (`purify_*`, `cse_*`, `cu_*`) in `expr` up to a node budget."""
    if not abs_subs:
        return expr
    leaves = _collect_leaf_signature(expr, limit=max_nodes)
    relevant = [(fv, orig) for fv, orig in abs_subs if fv.get_id() in leaves]
    if not relevant:
        return expr
    expanded = dsl.substitute(expr, *relevant)
    if count_ast_nodes([expanded], limit=max_nodes + 1) > max_nodes:
        return expr
    return expanded


def _combine_guard(base: dsl.BoolRef | None, extra: dsl.BoolRef | None) -> dsl.BoolRef | None:
    if base is None:
        return extra
    if extra is None:
        return base
    return dsl.And(base, extra)


class ConditionedUnificationPass(Z3QueryPass):
    """
    Discovers structurally similar (non-identical) subexpression pairs
    `(e_l, e_r)` across `lhs` and `rhs` via:
      1. Leaf-footprint matching (e.g., FMA reassociation under transcendentals)
      2. Top-down structural anti-unification (`enable_anti_unify=True`), which
         aligns structurally similar `lhs`/`rhs` skeletons even across disjoint
         purified leaves, discharges every provable congruence hole (including
         via bounded `abstraction_map` expansion), unifies discharged holes to
         `cu_k`, and records any undischarged holes in `query.residual_obligations`.
    """

    name = "conditioned_unification"
    is_equivalence_preserving = False

    def __init__(
        self,
        min_depth: int = 2,
        max_depth: int = 12,
        max_candidates: int = 16,
        enable_anti_unify: bool = True,
        max_anti_unify_holes: int = 32,
    ):
        self.min_depth = max(1, int(min_depth))
        self.max_depth = max(self.min_depth, int(max_depth))
        self.max_candidates = max(1, int(max_candidates))
        self.enable_anti_unify = bool(enable_anti_unify)
        self.max_anti_unify_holes = max(1, int(max_anti_unify_holes))

    def _run_leaf_footprint(
        self,
        query: SimplifyQuery,
        ctx: SimplificationContext,
    ) -> tuple[SimplifyQuery, int, int]:
        l_nodes, l_depths = _collect_subexprs_with_depth(query.lhs, limit=ctx.max_dag_nodes)
        r_nodes, r_depths = _collect_subexprs_with_depth(query.rhs, limit=ctx.max_dag_nodes)
        l_sigs = _collect_all_leaf_signatures(query.lhs, limit=ctx.max_dag_nodes)
        r_sigs = _collect_all_leaf_signatures(query.rhs, limit=ctx.max_dag_nodes)

        r_buckets: dict[tuple[int, frozenset[int]], list[tuple[int, dsl.ExprRef]]] = {}
        for rid, rexpr in r_nodes.items():
            if rid == query.rhs.get_id() or rid in l_nodes:
                continue
            rd = r_depths[rid]
            if rd < self.min_depth or rd > self.max_depth:
                continue
            if not _is_liftable_subexpr(rexpr, rd, self.min_depth):
                continue
            fp = r_sigs.get(rid, frozenset())
            if not fp:
                continue
            r_buckets.setdefault((rexpr.sort().kind(), fp), []).append((rd, rexpr))

        if not r_buckets:
            return query, 0, 0

        for b in r_buckets.values():
            b.sort(key=lambda x: -x[0])

        l_cands: list[tuple[int, int, dsl.ExprRef]] = []
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

        obligations: list[SideObligation] = list(query.side_obligations)
        l_subs: list[tuple[dsl.ExprRef, dsl.ExprRef]] = []
        r_subs: list[tuple[dsl.ExprRef, dsl.ExprRef]] = []
        abs_pairs: list[tuple[dsl.ExprRef, dsl.ExprRef]] = []
        covered_l: set[int] = set()
        covered_r: set[int] = set()
        attempted = 0
        discharged_cnt = 0

        wlp_solver = _IncrementalWlpSolver(query.guard, query.preconditions, ctx.wlp_timeout_ms)
        for ld, lid, lexpr in l_cands:
            if attempted >= self.max_candidates:
                break
            if lid in covered_l:
                continue
            fp = l_sigs.get(lid, frozenset())
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
                    lexpr, rexpr, query.guard, query.preconditions, ctx.wlp_timeout_ms,
                    wlp_solver=wlp_solver,
                )
                if ok:
                    fv = ctx.fresh_const("cu", lexpr.sort())
                    ob = SideObligation(
                        obligation_id=f"cu_{len(obligations)}",
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
            return query, attempted, 0

        new_lhs = dsl.substitute(query.lhs, *l_subs)
        new_rhs = dsl.substitute(query.rhs, *r_subs)
        out_q = query.with_terms(
            lhs=new_lhs,
            rhs=new_rhs,
            side_obligations=tuple(obligations),
            is_overapprox=True,
            abstraction_map=query.abstraction_map + tuple(abs_pairs),
            notes=query.notes + (f"cond_unified={discharged_cnt}/{attempted}",),
        )
        return out_q, attempted, discharged_cnt

    def _run_structural_anti_unify(
        self,
        query: SimplifyQuery,
        ctx: SimplificationContext,
    ) -> tuple[SimplifyQuery, int, int, int]:
        """
        Structurally anti-unify `(query.lhs, query.rhs)` and attempt WLP discharge
        on every extracted mismatch hole `(a_i, b_i)`.

        Returns `(updated_query, total_holes, discharged_cnt, residual_cnt)`.
        """
        if query.is_trivially_unsat():
            return query, 0, 0, 0

        matched_ops, holes, overflowed = _anti_unify_pair(
            query.lhs, query.rhs, max_holes=self.max_anti_unify_holes
        )
        if matched_ops < 1 or overflowed or not holes:
            return query, 0, 0, 0

        abs_subs = list(query.abstraction_map)
        abs_ids = {fv.get_id() for fv, _ in abs_subs}

        obligations: list[SideObligation] = list(query.side_obligations)
        residuals: list[SideObligation] = []
        l_subs: list[tuple[dsl.ExprRef, dsl.ExprRef]] = []
        r_subs: list[tuple[dsl.ExprRef, dsl.ExprRef]] = []
        abs_pairs: list[tuple[dsl.ExprRef, dsl.ExprRef]] = []
        discharged_cnt = 0
        wlp_solver = _IncrementalWlpSolver(query.guard, query.preconditions, ctx.wlp_timeout_ms)

        for idx, (a_i, b_i, pcond) in enumerate(holes):
            if a_i.sort() != b_i.sort():
                fv = ctx.fresh_const("au_res", a_i.sort())
                residuals.append(
                    SideObligation(
                        obligation_id=f"au_res_{idx}",
                        fresh_var=fv,
                        lhs=a_i,
                        rhs=b_i,
                        guard=_combine_guard(query.guard, pcond),
                        reason="anti_unify_sort_mismatch",
                        discharged=False,
                        elapsed_ms=0.0,
                    )
                )
                continue

            # Try global discharge first so `a_i -> cu_k` is sound anywhere in the AST.
            ok_global, dt_wlp = _discharge_wlp_local(
                a_i, b_i, query.guard, query.preconditions, ctx.wlp_timeout_ms,
                wlp_solver=wlp_solver,
            )
            via_expand = False
            if not ok_global and abs_subs:
                exp_a = _expand_abstractions(a_i, abs_subs, abs_ids)
                exp_b = _expand_abstractions(b_i, abs_subs, abs_ids)
                if not exp_a.eq(a_i) or not exp_b.eq(b_i):
                    ok_exp, dt_exp = _discharge_wlp_local(
                        exp_a, exp_b, query.guard, query.preconditions, ctx.wlp_timeout_ms,
                        wlp_solver=wlp_solver,
                    )
                    dt_wlp += dt_exp
                    if ok_exp:
                        ok_global = True
                        via_expand = True

            ok_path = False
            eff_guard = _combine_guard(query.guard, pcond)
            if not ok_global and pcond is not None:
                ok_path, dt_p = _discharge_wlp_local(
                    a_i, b_i, query.guard, query.preconditions, ctx.wlp_timeout_ms,
                    wlp_solver=wlp_solver, extra_guard=pcond,
                )
                dt_wlp += dt_p
                if not ok_path and abs_subs:
                    exp_a = _expand_abstractions(a_i, abs_subs, abs_ids)
                    exp_b = _expand_abstractions(b_i, abs_subs, abs_ids)
                    if not exp_a.eq(a_i) or not exp_b.eq(b_i):
                        ok_ep, dt_ep = _discharge_wlp_local(
                            exp_a, exp_b, query.guard, query.preconditions, ctx.wlp_timeout_ms,
                            wlp_solver=wlp_solver, extra_guard=pcond,
                        )
                        dt_wlp += dt_ep
                        if ok_ep:
                            ok_path = True
                            via_expand = True

            if ok_global or ok_path:
                fv = ctx.fresh_const("cu", a_i.sort())
                tag = "anti_unify_expanded" if via_expand else "anti_unify_discharged"
                if ok_path and not ok_global:
                    tag += "_path"
                obligations.append(
                    SideObligation(
                        obligation_id=f"au_{len(obligations)}",
                        fresh_var=fv,
                        lhs=a_i,
                        rhs=b_i,
                        guard=query.guard if ok_global else eff_guard,
                        reason=tag,
                        discharged=True,
                        elapsed_ms=dt_wlp,
                    )
                )
                if ok_global:
                    l_subs.append((a_i, fv))
                    r_subs.append((b_i, fv))
                    abs_pairs.append((fv, a_i))
                discharged_cnt += 1
            else:
                fv = ctx.fresh_const("au_res", a_i.sort())
                residuals.append(
                    SideObligation(
                        obligation_id=f"au_res_{len(residuals)}",
                        fresh_var=fv,
                        lhs=a_i,
                        rhs=b_i,
                        guard=eff_guard,
                        reason="anti_unify_residual",
                        discharged=False,
                        elapsed_ms=dt_wlp,
                    )
                )

        all_discharged = len(residuals) == 0 and discharged_cnt == len(holes)
        new_lhs = dsl.substitute(query.lhs, *l_subs) if l_subs else query.lhs
        new_rhs = dsl.substitute(query.rhs, *r_subs) if r_subs else query.rhs

        if all_discharged:
            out_q = query.with_terms(
                lhs=new_lhs,
                rhs=new_rhs,
                predicate=dsl.BoolVal(False),
                side_obligations=tuple(obligations),
                residual_obligations=(),
                is_overapprox=True if abs_pairs else query.is_overapprox,
                abstraction_map=query.abstraction_map + tuple(abs_pairs),
                notes=query.notes + (f"anti_unify_discharged_all={discharged_cnt}/{len(holes)}",),
            )
            return out_q, len(holes), discharged_cnt, 0

        if discharged_cnt == 0 and not residuals:
            return query, len(holes), 0, 0

        out_q = query.with_terms(
            lhs=new_lhs,
            rhs=new_rhs,
            side_obligations=tuple(obligations),
            residual_obligations=tuple(residuals),
            is_overapprox=True if abs_pairs else query.is_overapprox,
            abstraction_map=query.abstraction_map + tuple(abs_pairs),
            notes=query.notes + (f"anti_unify={discharged_cnt}/{len(holes)},residual={len(residuals)}",),
        )
        return out_q, len(holes), discharged_cnt, len(residuals)

    def run(
        self,
        query: SimplifyQuery,
        ctx: SimplificationContext,
    ) -> tuple[SimplifyQuery, PassStats]:
        t0 = time.perf_counter()
        if query.is_trivially_unsat():
            return query, PassStats(self.name, False, 0, 0.0, "trivial_unsat")

        query = query.realize_z3()
        cur_q, fp_att, fp_dis = self._run_leaf_footprint(query, ctx)

        au_total = 0
        au_dis = 0
        au_res = 0
        if self.enable_anti_unify and not cur_q.is_trivially_unsat():
            cur_q, au_total, au_dis, au_res = self._run_structural_anti_unify(cur_q, ctx)

        total_dis = fp_dis + au_dis
        changed = total_dis > 0 or au_res > 0
        dt = (time.perf_counter() - t0) * 1000.0
        detail = f"fp={fp_dis}/{fp_att} au={au_dis}/{au_total} residual={au_res}"
        return cur_q, PassStats(self.name, changed, total_dis, dt, detail)

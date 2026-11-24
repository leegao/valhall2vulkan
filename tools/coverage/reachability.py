"""
Maximal Reachability & Disjoint Path Partition Solver for ValhallSsaModule.

Enumerates a strictly pairwise-disjoint set of satisfiable path reachability
conditions {PC_1(X), ..., PC_m(X)} covering all feasible execution paths via
an incremental Z3 Solver (prioritizing unvisited blocks via push/pop) with
Unique-Call UF Pruning, per-iteration loop-unroll decision tracking, and
blocking clauses ~PC_i(X). When the blocked solver returns UNSAT, the union of
reported path conditions is formally certified to partition 100% of reachable paths.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import z3

from tools.symbolic_ssa.z3utils import dsl

from tools.coverage.synthesizer import extract_concrete_overrides_from_model
from tools.coverage.uf_pruning import lift_uf_proxies_into_overrides, prune_unique_hw_ufs
from tools.symbolic_ssa.env import (
    ConcreteShaderEnv,
    SymbolicShaderEnv,
    _as_bool,
    _z3_not,
)
from tools.symbolic_ssa import (
    ConcreteSsaInterpreter,
    SymbolicSsaInterpreter,
)


@dataclass
class DisjointPathRecord:
    """One feasible execution path with its disjoint symbolic reachability condition."""

    path_id: int
    visited_blocks: List[int]
    decisions: List[Tuple[int, int, bool]]  # (blk_id, branch_cond_nid, took_branch)
    condition_smt: str
    witness_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MaximalReachabilityResult:
    """Result of disjoint maximal-reachability path enumeration."""

    total_blocks: int
    reached_blocks: Set[int]
    complete: bool
    stop_reason: str
    elapsed_ms: float
    paths: List[DisjointPathRecord] = field(default_factory=list)


def _trace_straight_line_blocks(
    ssa_json: Dict[str, Any],
    env: ConcreteShaderEnv,
    is_valhall_side: bool,
    max_steps: int = 256,
) -> List[int]:
    """Trace visited blocks on a 0-branch straight-line shader."""
    interp = ConcreteSsaInterpreter(ssa_json, is_valhall_side=is_valhall_side, max_block_steps=max_steps)
    interp.execute_one(env)
    blocks = interp.blocks_by_id
    visited: List[int] = []
    cur = 0
    steps = 0
    while cur in blocks and steps < max_steps:
        steps += 1
        visited.append(cur)
        blk = blocks[cur]
        if blk.get("is_end") or blk.get("is_discard"):
            break
        bt = int(blk.get("branch_target", -1))
        ft = int(blk.get("fallthrough", -1))
        if bt >= 0:
            cur = bt
        elif ft >= 0:
            cur = ft
        else:
            break
    return visited


def solve_maximal_reachability(
    bundle: Dict[str, Any],
    ssa_json: Dict[str, Any],
    is_valhall_side: bool = False,
    fp_mode: str = "exact",
    max_paths: int = 32,
    timeout_ms: int = 500,
    max_unroll: int = 1,
    total_timeout_ms: Optional[int] = None,
    domain: str = "udomain",
) -> MaximalReachabilityResult:
    """
    Compute a pairwise-disjoint set of path reachability conditions covering
    all feasible paths (up to max_paths), prioritizing unvisited blocks first.

    Supports multi-iteration loop unrolling (`max_unroll > 1`) via per-iteration
    branch decisions recorded by SymbolicSsaInterpreter.
    """
    t0 = time.perf_counter()
    deadline = t0 + ((total_timeout_ms if total_timeout_ms is not None else max(timeout_ms * 4, 2000)) / 1000.0)
    blocks = ssa_json.get("blocks", [])
    all_bids = [int(b["id"]) for b in blocks]

    env = SymbolicShaderEnv(bundle, domain=domain, fp_mode=fp_mode)
    env.shared_sym_bkeys = set()
    ev = SymbolicSsaInterpreter(ssa_json, env, is_valhall_side=is_valhall_side, fp_mode=fp_mode, max_unroll=max_unroll, domain=domain)
    ev.evaluate()
    preconditions = env.build_preconditions()
    raw_blk_reach: Dict[int, z3.BoolRef] = dict(getattr(ev, "block_reach", {}))
    raw_iter_decs: List[Tuple[int, int, int, z3.BoolRef, z3.BoolRef]] = list(
        getattr(ev, "iter_branch_decisions", [])
    )

    if not raw_iter_decs:
        ov: Dict[str, Any] = {}
        cenv = ConcreteShaderEnv(bundle, 0, overrides=ov)
        vis = _trace_straight_line_blocks(ssa_json, cenv, is_valhall_side=is_valhall_side)
        rec = DisjointPathRecord(
            path_id=0,
            visited_blocks=vis,
            decisions=[],
            condition_smt="true",
            witness_overrides=ov,
        )
        return MaximalReachabilityResult(
            total_blocks=len(all_bids),
            reached_blocks=set(vis),
            complete=True,
            stop_reason="EXHAUSTED_ALL_PATHS",
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            paths=[rec],
        )

    # Apply Unique-Call UF Pruning jointly across all block-reach and branch-decision ASTs.
    reach_bids = sorted(raw_blk_reach.keys())
    flat_exprs: List[z3.ExprRef] = [raw_blk_reach[b] for b in reach_bids]
    for _, _, _, r_expr, c_expr in raw_iter_decs:
        flat_exprs.append(r_expr)
        flat_exprs.append(c_expr)

    pruned_exprs, uf_proxies = prune_unique_hw_ufs(flat_exprs, preconditions)

    idx = 0
    blk_reach: Dict[int, z3.BoolRef] = {}
    for b in reach_bids:
        blk_reach[b] = pruned_exprs[idx]
        idx += 1

    iter_decs: List[Tuple[int, int, int, z3.BoolRef, z3.BoolRef]] = []
    for iter_idx, bid, cid, _, _ in raw_iter_decs:
        pr_r = pruned_exprs[idx]
        pr_c = pruned_exprs[idx + 1]
        idx += 2
        iter_decs.append((iter_idx, bid, cid, pr_r, pr_c))

    solver = z3.Solver()
    for p in preconditions:
        solver.add(dsl.lower(p))

    paths: List[DisjointPathRecord] = []
    reached_blocks: Set[int] = set()
    unreachable_blocks: Set[int] = set()
    complete = False
    stop_reason = "MAX_PATHS_REACHED"

    for pidx in range(max_paths):
        rem_ms = int((deadline - time.perf_counter()) * 1000.0)
        if rem_ms <= 0:
            stop_reason = "SOLVER_TIMEOUT"
            break
        step_timeout = min(timeout_ms, rem_ms)

        unvisited = [
            bid for bid in all_bids
            if bid not in reached_blocks and bid not in unreachable_blocks and bid in blk_reach
        ]
        unvisited_exprs = [
            blk_reach[bid] for bid in unvisited
            if not dsl.is_true(blk_reach[bid]) and not dsl.is_false(blk_reach[bid])
        ]

        model: Optional[z3.ModelRef] = None
        if unvisited_exprs:
            solver.push()
            solver.set("timeout", step_timeout)
            solver.add(z3.Or(*[dsl.lower(e) for e in unvisited_exprs]))
            rc = solver.check()
            if rc == z3.sat:
                model = solver.model()
                solver.pop()
            elif rc == z3.unsat:
                # None of the remaining unvisited blocks can be reached on any remaining path.
                unreachable_blocks.update(unvisited)
                solver.pop()
            else:
                solver.pop()

        if model is None:
            rem_ms = int((deadline - time.perf_counter()) * 1000.0)
            if rem_ms <= 0:
                stop_reason = "SOLVER_TIMEOUT"
                break
            solver.set("timeout", min(timeout_ms, rem_ms))
            rc = solver.check()
            if rc == z3.unsat:
                complete = True
                stop_reason = "EXHAUSTED_ALL_PATHS"
                break
            if rc != z3.sat:
                stop_reason = "SOLVER_TIMEOUT"
                break
            model = solver.model()

        ov = extract_concrete_overrides_from_model(model, env)
        lift_uf_proxies_into_overrides(model, uf_proxies, ov)

        # Build the exact symbolic path condition PC_i(X) over per-iteration decisions
        # so that ~PC_i(X) is guaranteed to block this model (even across loop iterations).
        pc_lits: List[z3.BoolRef] = []
        sym_decs: List[Tuple[int, int, bool]] = []
        for _iter_idx, bid, cid, r_expr, c_expr in iter_decs:
            if z3.is_true(model.eval(dsl.lower(r_expr), model_completion=True)):
                bexpr = _as_bool(c_expr)
                took = z3.is_true(model.eval(dsl.lower(bexpr), model_completion=True))
                pc_lits.append(bexpr if took else _z3_not(bexpr))
                sym_decs.append((bid, cid, took))

        vis = [
            bid for bid in all_bids
            if bid in blk_reach and z3.is_true(model.eval(dsl.lower(blk_reach[bid]), model_completion=True))
        ]

        if pc_lits:
            pc_expr = pc_lits[0] if len(pc_lits) == 1 else dsl.And(*pc_lits)
        else:
            pc_expr = dsl.BoolVal(True)

        solver.add(dsl.lower(_z3_not(pc_expr)))
        reached_blocks.update(vis)

        paths.append(
            DisjointPathRecord(
                path_id=pidx,
                visited_blocks=vis,
                decisions=sym_decs,
                condition_smt=pc_expr.sexpr() if hasattr(pc_expr, "sexpr") else str(pc_expr),
                witness_overrides=ov,
            )
        )
        if not pc_lits:
            complete = True
            stop_reason = "EXHAUSTED_ALL_PATHS"
            break

    return MaximalReachabilityResult(
        total_blocks=len(all_bids),
        reached_blocks=reached_blocks,
        complete=complete,
        stop_reason=stop_reason,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        paths=paths,
    )

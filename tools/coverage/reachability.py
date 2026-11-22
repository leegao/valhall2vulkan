#!/usr/bin/env python3
"""
Maximal Reachability & Disjoint Path Partition Solver for ValhallSsaModule.

Enumerates a strictly pairwise-disjoint set of satisfiable path reachability
conditions {PC_1(X), ..., PC_m(X)} covering all feasible execution paths via
MaxSMT (maximizing newly reached blocks/edges per step) + blocking clauses
~PC_i(X). When the blocked solver returns UNSAT, the union of reported path
conditions is formally certified to partition 100% of reachable paths.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import z3

from tools.coverage.evaluator import (
    ConcreteShaderEnv,
    ConcreteSsaInterpreter,
    DagSsaZ3Evaluator,
    SharedShaderEnv,
    _as_bool,
    _z3_not,
)
from tools.coverage.synthesizer import extract_concrete_overrides_from_model


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


def _trace_concrete_path(
    ssa_json: Dict[str, Any],
    env: ConcreteShaderEnv,
    is_valhall_side: bool,
    max_steps: int = 256,
) -> Tuple[List[int], List[Tuple[int, int, bool]]]:
    """Run ConcreteSsaInterpreter and record visited block sequence + branch decisions."""
    interp = ConcreteSsaInterpreter(ssa_json, is_valhall_side=is_valhall_side, max_block_steps=max_steps)
    vals, _ = interp.execute_one(env)
    blocks = interp.blocks_by_id
    visited: List[int] = []
    decisions: List[Tuple[int, int, bool]] = []
    if 0 not in blocks:
        return visited, decisions

    cur = 0
    steps = 0
    seen_decisions: Set[int] = set()
    while cur in blocks and steps < max_steps:
        steps += 1
        visited.append(cur)
        blk = blocks[cur]
        if blk.get("is_end") or blk.get("is_discard"):
            break
        bt = int(blk.get("branch_target", -1))
        ft = int(blk.get("fallthrough", -1))
        if blk.get("is_cond") and bt >= 0 and ft >= 0:
            cid = int(blk.get("branch_cond", 0))
            took = bool(vals.get(cid, False))
            if cur not in seen_decisions:
                decisions.append((cur, cid, took))
                seen_decisions.add(cur)
            cur = bt if took else ft
        elif bt >= 0:
            cur = bt
        elif ft >= 0:
            cur = ft
        else:
            break
    return visited, decisions


def solve_maximal_reachability(
    bundle: Dict[str, Any],
    ssa_json: Dict[str, Any],
    is_valhall_side: bool = False,
    fp_mode: str = "exact",
    max_paths: int = 32,
    timeout_ms: int = 500,
    max_unroll: int = 1,
) -> MaximalReachabilityResult:
    """
    Compute a pairwise-disjoint set of path reachability conditions covering
    all feasible paths (up to max_paths), prioritizing unvisited blocks first.
    """
    t0 = time.perf_counter()
    blocks = ssa_json.get("blocks", [])
    all_bids = [int(b["id"]) for b in blocks]

    env = SharedShaderEnv(bundle)
    env.shared_sym_bkeys = set()
    ev = DagSsaZ3Evaluator(ssa_json, env, is_valhall_side=is_valhall_side, fp_mode=fp_mode, max_unroll=max_unroll)
    ev.evaluate()
    preconditions = env.build_preconditions()
    blk_reach: Dict[int, z3.BoolRef] = getattr(ev, "block_reach", {})

    cond_blocks = [
        (int(b["id"]), int(b.get("branch_cond", 0)))
        for b in blocks
        if b.get("is_cond") and int(b.get("branch_cond", 0)) > 0 and int(b.get("branch_cond", 0)) in ev.values
    ]

    # Straight-line 0-branch shader: single trivially complete path PC = True
    if not cond_blocks:
        ov: Dict[str, Any] = {}
        cenv = ConcreteShaderEnv(bundle, 0, overrides=ov)
        vis, decs = _trace_concrete_path(ssa_json, cenv, is_valhall_side=is_valhall_side)
        rec = DisjointPathRecord(
            path_id=0,
            visited_blocks=vis,
            decisions=decs,
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

    blocking_clauses: List[z3.BoolRef] = []
    paths: List[DisjointPathRecord] = []
    reached_blocks: Set[int] = set()
    complete = False
    stop_reason = "MAX_PATHS_REACHED"

    for pidx in range(max_paths):
        unvisited = [bid for bid in all_bids if bid not in reached_blocks and bid in blk_reach]

        opt = z3.Optimize()
        opt.set("timeout", timeout_ms)
        for p in preconditions:
            opt.add(p)
        for bc in blocking_clauses:
            opt.add(bc)
        # Encourage covering maximum unvisited blocks on each new disjoint path
        for bid in unvisited[:24]:
            r = blk_reach[bid]
            if not z3.is_true(r) and not z3.is_false(r):
                opt.add_soft(r, weight=1)

        rc = opt.check()
        if rc == z3.unknown:
            # Fallback to plain Solver if MaxSMT optimization timed out
            s = z3.Solver()
            s.set("timeout", timeout_ms)
            for p in preconditions:
                s.add(p)
            for bc in blocking_clauses:
                s.add(bc)
            rc = s.check()
            if rc != z3.sat:
                complete = (rc == z3.unsat)
                stop_reason = "EXHAUSTED_ALL_PATHS" if complete else "SOLVER_TIMEOUT"
                break
            model = s.model()
        elif rc == z3.unsat:
            complete = True
            stop_reason = "EXHAUSTED_ALL_PATHS"
            break
        else:
            model = opt.model()

        ov = extract_concrete_overrides_from_model(model, env)
        cenv = ConcreteShaderEnv(bundle, pidx, overrides=ov)
        vis, decs = _trace_concrete_path(ssa_json, cenv, is_valhall_side=is_valhall_side)

        # Build the exact symbolic path condition PC_i(X) directly from the Z3 model
        # so that ~PC_i(X) is mathematically guaranteed to block this model on the next round.
        pc_lits: List[z3.BoolRef] = []
        sym_decs: List[Tuple[int, int, bool]] = []
        for bid, cid in cond_blocks:
            r = blk_reach.get(bid, z3.BoolVal(True))
            if z3.is_true(model.eval(r, model_completion=True)):
                bexpr = _as_bool(ev.values[cid])
                took = z3.is_true(model.eval(bexpr, model_completion=True))
                pc_lits.append(bexpr if took else _z3_not(bexpr))
                sym_decs.append((bid, cid, took))
        sym_reached = [
            bid for bid in all_bids
            if bid in blk_reach and z3.is_true(model.eval(blk_reach[bid], model_completion=True))
        ]
        if sym_reached:
            vis = sym_reached

        pc_expr = z3.simplify(z3.And(*pc_lits)) if pc_lits else z3.BoolVal(True)
        # Block this path condition so every subsequent path is strictly disjoint
        blocking_clauses.append(_z3_not(pc_expr))
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

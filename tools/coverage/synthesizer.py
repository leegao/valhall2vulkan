#!/usr/bin/env python3
"""
Coverage-Guided Concolic Input Synthesizer for ValhallSsaModule.

Given a ValhallSsaModule and an initial set of K concrete execution traces,
identifies uncovered branch/Select/Kill predicate polarities and uses an
incremental Z3 Solver with Unique-Call UF Pruning, AST-size ordering, and
greedy target packing to synthesize concrete input vectors that flip multiple
uncovered predicates per vector—and formally certifies unreachable (dead)
predicate polarities when a target formula is UNSAT.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import z3

from tools.coverage.uf_pruning import (
    ast_node_count,
    lift_uf_proxies_into_overrides,
    prune_unique_hw_ufs,
)
from tools.symbolic_ssa.env import (
    MASK32,
    ConcreteShaderEnv,
    SharedShaderEnv,
    _as_bool,
    _z3_not,
)
from tools.symbolic_ssa import (
    ConcreteSsaInterpreter,
    SymbolicSsaInterpreter,
)


@dataclass
class CoverageFuzzResult:
    """Summary of coverage-guided concolic input synthesis."""

    total_predicates: int
    total_targets: int
    initial_covered: int
    final_covered: int
    dead_proved: int
    vectors_synthesized: int
    maxsmt_rounds: int
    fallback_probes: int
    elapsed_ms: float
    synthesized_overrides: List[Dict[str, Any]] = field(default_factory=list)
    covered_targets: Set[Tuple[int, bool]] = field(default_factory=set)
    dead_targets: Set[Tuple[int, bool]] = field(default_factory=set)

    @property
    def initial_pct(self) -> float:
        return 100.0 * self.initial_covered / max(1, self.total_targets)

    @property
    def final_pct(self) -> float:
        return 100.0 * self.final_covered / max(1, self.total_targets)


def collect_predicate_nodes(ssa_json: Dict[str, Any]) -> List[Tuple[int, int]]:
    """
    Collect unique (predicate_node_id, enclosing_block_id) pairs for all
    conditional branches, Select conditions, and predicated Kill nodes.
    """
    nodes_by_id = {int(n["id"]): n for n in ssa_json.get("nodes", [])}
    seen: Set[int] = set()
    preds: List[Tuple[int, int]] = []

    def add_pred(pid: int, blk_id: int) -> None:
        if pid > 0 and pid not in seen and pid in nodes_by_id:
            pn = nodes_by_id[pid]
            if pn["op"] not in ("ConstBool", "ConstU32", "ConstS32"):
                seen.add(pid)
                preds.append((pid, blk_id))

    for blk in ssa_json.get("blocks", []):
        bid = int(blk["id"])
        if blk.get("is_cond") and int(blk.get("branch_cond", 0)) > 0:
            add_pred(int(blk["branch_cond"]), bid)
        for nid in blk.get("nodes", []):
            n = nodes_by_id.get(nid)
            if not n:
                continue
            op = n["op"]
            args = n.get("args", [])
            if op == "Select" and len(args) >= 3:
                add_pred(int(args[0]), bid)
            elif op == "Kill" and n.get("aux_bool", False) and len(args) >= 1:
                add_pred(int(args[0]), bid)
    return preds


def measure_predicate_coverage(
    preds: List[Tuple[int, int]],
    node_traces: Dict[int, List[Any]],
) -> Set[Tuple[int, bool]]:
    """Return the set of (pred_nid, polarity_bool) targets covered by traces."""
    covered: Set[Tuple[int, bool]] = set()
    for pid, _ in preds:
        tr = node_traces.get(pid, [])
        for v in tr:
            if v is None:
                continue
            covered.add((pid, bool(v)))
    return covered


def _extract_array_entries(
    model: z3.ModelRef,
    arr_expr: z3.ExprRef,
    idx_mask: int = MASK32,
) -> Tuple[Dict[int, int], Optional[int]]:
    """Decode Z3 Array model (Store chains, K(v), or as-array FuncInterp)."""
    entries: Dict[int, int] = {}
    default_val: Optional[int] = None

    def walk(e: z3.ExprRef) -> None:
        nonlocal default_val
        if not z3.is_app(e):
            return
        decl = e.decl()
        kind = decl.kind()
        if kind == z3.Z3_OP_STORE and e.num_args() == 3:
            base, idx, val = e.arg(0), e.arg(1), e.arg(2)
            walk(base)
            if z3.is_bv_value(idx) and z3.is_bv_value(val):
                entries[int(idx.as_long()) & idx_mask] = int(val.as_long()) & MASK32
        elif kind == z3.Z3_OP_CONST_ARRAY and e.num_args() == 1:
            v = e.arg(0)
            if z3.is_bv_value(v):
                default_val = int(v.as_long()) & MASK32
        elif kind == z3.Z3_OP_AS_ARRAY:
            fdecl = z3.get_as_array_func(e)
            interp = model.get_interp(fdecl)
            if isinstance(interp, z3.FuncInterp):
                ev = interp.else_value()
                if ev is not None and z3.is_bv_value(ev):
                    default_val = int(ev.as_long()) & MASK32
                for entry in interp.as_list()[:-1]:
                    if len(entry) == 2 and z3.is_bv_value(entry[0]) and z3.is_bv_value(entry[1]):
                        entries[int(entry[0].as_long()) & idx_mask] = int(entry[1].as_long()) & MASK32

    ev = model.eval(arr_expr, model_completion=True)
    walk(ev)
    return entries, default_val


def extract_concrete_overrides_from_model(
    model: z3.ModelRef,
    env: SharedShaderEnv,
) -> Dict[str, Any]:
    """
    Extract concrete BitVec, Bool, FAU-array, buffer-array, image-array, and
    hardware UF assignments from a Z3 model into a ConcreteShaderEnv.overrides dict.
    """
    overrides: Dict[str, Any] = {}

    for decl in model.decls():
        if decl.arity() != 0:
            continue
        name = decl.name()
        if name.startswith("__ufproxy__"):
            continue
        val = model[decl]
        if z3.is_bv_value(val):
            overrides[name] = int(val.as_long()) & MASK32
        elif z3.is_true(val):
            overrides[name] = True
        elif z3.is_false(val):
            overrides[name] = False

    fau_entries, fau_def = _extract_array_entries(model, env.fau_sym)
    for w, v in fau_entries.items():
        overrides[f"fau_sym_{w}"] = v
    if fau_def is not None:
        overrides["fau_sym_default"] = fau_def

    for bkey, arr in env.init_buf_mem.items():
        mem_entries, mem_def = _extract_array_entries(model, arr)
        for off, v in mem_entries.items():
            overrides[f"mem:{bkey}:{off}"] = v
        if mem_def is not None:
            overrides[f"mem_default:{bkey}"] = mem_def

    for sym_tag, fdecl in getattr(env, "sym_buf_ufs", {}).items():
        interp = model.get_interp(fdecl)
        if isinstance(interp, z3.FuncInterp):
            ev = interp.else_value()
            if ev is not None and z3.is_bv_value(ev):
                overrides[f"sym_slot_default:{sym_tag}"] = int(ev.as_long()) & MASK32
            for entry in interp.as_list()[:-1]:
                if len(entry) == 2 and z3.is_bv_value(entry[0]) and z3.is_bv_value(entry[1]):
                    off = int(entry[0].as_long()) & MASK32
                    overrides[f"sym_slot:{sym_tag}:{off}"] = int(entry[1].as_long()) & MASK32

    mask64 = (1 << 64) - 1
    for ikey, arrs in getattr(env, "init_img_mem", {}).items():
        for c, arr in enumerate(arrs):
            img_entries, img_def = _extract_array_entries(model, arr, idx_mask=mask64)
            for xy, v in img_entries.items():
                overrides[f"img:{ikey}:c{c}:{xy}"] = v
            if img_def is not None:
                overrides[f"img_default:{ikey}:c{c}"] = img_def

    if hasattr(env, "fallback_mem"):
        fb_entries, fb_def = _extract_array_entries(model, env.fallback_mem, idx_mask=mask64)
        for addr, v in fb_entries.items():
            overrides[f"mem_fallback:{addr}"] = v
        if fb_def is not None:
            overrides["mem_fallback_default"] = fb_def

    for decl in model.decls():
        if decl.arity() < 1:
            continue
        dname = decl.name()
        if not dname.startswith("hw_"):
            continue
        interp = model.get_interp(decl)
        if not isinstance(interp, z3.FuncInterp):
            continue
        tbl: Dict[str, int] = {}
        ev = interp.else_value()
        def_val: Optional[int] = None
        if ev is not None:
            if z3.is_bv_value(ev):
                def_val = int(ev.as_long()) & MASK32
            elif z3.is_true(ev):
                def_val = 1
            elif z3.is_false(ev):
                def_val = 0
        for entry in interp.as_list()[:-1]:
            if len(entry) == decl.arity() + 1:
                arg_vals = entry[:-1]
                res_v = entry[-1]
                if all(z3.is_bv_value(a) for a in arg_vals):
                    k = ",".join(str(int(a.as_long()) & MASK32) for a in arg_vals)
                    if z3.is_bv_value(res_v):
                        tbl[k] = int(res_v.as_long()) & MASK32
                    elif z3.is_true(res_v):
                        tbl[k] = 1
                    elif z3.is_false(res_v):
                        tbl[k] = 0
        overrides[f"uf:{dname}"] = {"table": tbl, "default": def_val}

    return overrides


def _build_target_formulas(
    ssa_json: Dict[str, Any],
    env: SharedShaderEnv,
    is_valhall_side: bool,
    preds: List[Tuple[int, int]],
    fp_mode: str,
    max_unroll: int = 1,
    preconditions: Sequence[z3.BoolRef] = (),
    domain: str = "bitvec",
) -> Tuple[Dict[Tuple[int, bool], z3.BoolRef], Dict[str, z3.ExprRef], bool]:
    """Symbolically evaluate the SSA module and build pruned (pid, polarity) formulas."""
    evaluator = SymbolicSsaInterpreter(
        ssa_json, env, is_valhall_side=is_valhall_side, fp_mode=fp_mode, max_unroll=max_unroll, domain=domain
    )
    evaluator.evaluate()
    block_reach = getattr(evaluator, "block_reach", {})

    raw_keys: List[Tuple[int, bool]] = []
    raw_exprs: List[z3.ExprRef] = []
    for pid, bid in preds:
        if pid not in evaluator.values:
            continue
        bval = evaluator.lower_bool_value(pid)
        reach = block_reach.get(bid)
        t_expr = z3.And(reach, bval) if reach is not None and not z3.is_true(reach) else bval
        f_expr = z3.And(reach, _z3_not(bval)) if reach is not None and not z3.is_true(reach) else _z3_not(bval)
        raw_keys.append((pid, True))
        raw_exprs.append(t_expr)
        raw_keys.append((pid, False))
        raw_exprs.append(f_expr)

    pruned_exprs, uf_proxies = prune_unique_hw_ufs(raw_exprs, preconditions)
    formulas: Dict[Tuple[int, bool], z3.BoolRef] = {
        k: pruned_exprs[i] for i, k in enumerate(raw_keys)
    }
    return formulas, uf_proxies, evaluator.has_loops


def synthesize_coverage_vectors(
    bundle: Dict[str, Any],
    ssa_json: Dict[str, Any],
    initial_traces: Dict[int, List[Any]],
    is_valhall_side: bool = False,
    fp_mode: str = "exact",
    max_rounds: int = 12,
    max_soft_per_round: int = 16,
    round_timeout_ms: int = 400,
    probe_timeout_ms: int = 200,
    max_fallback_probes: int = 16,
    base_vec_idx: int = 32,
    max_unroll: int = 1,
    adaptive_unroll: bool = True,
    max_adaptive_unroll: int = 3,
    time_budget_ms: int = 5000,
    domain: str = "bitvec",
) -> CoverageFuzzResult:
    """
    Use incremental Z3 Solver + Unique-Call UF Pruning + AST-size ordering +
    greedy multi-target packing to synthesize concrete input override dicts
    maximizing branch/Select/Kill coverage within `time_budget_ms`.
    """
    t0 = time.perf_counter()
    deadline = t0 + (time_budget_ms / 1000.0)
    preds = collect_predicate_nodes(ssa_json)
    all_targets: Set[Tuple[int, bool]] = {(pid, pol) for pid, _ in preds for pol in (True, False)}
    covered = measure_predicate_coverage(preds, initial_traces)
    initial_covered = len(covered)

    if not preds or covered >= all_targets:
        return CoverageFuzzResult(
            total_predicates=len(preds),
            total_targets=len(all_targets),
            initial_covered=initial_covered,
            final_covered=len(covered),
            dead_proved=0,
            vectors_synthesized=0,
            maxsmt_rounds=0,
            fallback_probes=0,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            covered_targets=covered,
        )

    env = SharedShaderEnv(bundle)
    env.shared_sym_bkeys = set()
    cur_unroll = max(1, int(max_unroll))
    preconditions = env.build_preconditions()
    formulas, uf_proxies, has_loops = _build_target_formulas(
        ssa_json,
        env,
        is_valhall_side,
        preds,
        fp_mode=fp_mode,
        max_unroll=cur_unroll,
        domain=domain,
        preconditions=preconditions,
    )
    preconditions = env.build_preconditions()
    target_sizes: Dict[Tuple[int, bool], int] = {
        t: ast_node_count(expr) for t, expr in formulas.items()
    }
    interp = ConcreteSsaInterpreter(ssa_json, is_valhall_side=is_valhall_side)

    solver = z3.Solver()
    for p in preconditions:
        solver.add(p)

    synthesized: List[Dict[str, Any]] = []
    dead_targets: Set[Tuple[int, bool]] = set()
    attempted_failed: Set[Tuple[int, bool]] = set()
    maxsmt_rounds = 0
    fallback_probes = 0

    def record_model(model: z3.ModelRef, round_idx: int) -> int:
        ov = extract_concrete_overrides_from_model(model, env)
        lift_uf_proxies_into_overrides(model, uf_proxies, ov)
        cenv = ConcreteShaderEnv(bundle, base_vec_idx + round_idx, overrides=ov)
        vals, _ = interp.execute_one(cenv)
        newly_hit = 0
        for pid, _ in preds:
            if pid in vals and vals[pid] is not None:
                tgt = (pid, bool(vals[pid]))
                if tgt not in covered:
                    covered.add(tgt)
                    newly_hit += 1
        synthesized.append(ov)
        return newly_hit

    for _ in range(max_rounds):
        if time.perf_counter() >= deadline:
            break

        uncovered = [
            t for t in (all_targets - covered - dead_targets - attempted_failed)
            if t in formulas
        ]
        if not uncovered:
            if adaptive_unroll and has_loops and cur_unroll < max_adaptive_unroll and (all_targets - covered):
                cur_unroll += 1
                dead_targets.clear()
                attempted_failed.clear()
                formulas, uf_proxies, has_loops = _build_target_formulas(
                    ssa_json,
                    env,
                    is_valhall_side,
                    preds,
                    fp_mode=fp_mode,
                    max_unroll=cur_unroll,
                    domain=domain,
                    preconditions=preconditions,
                )
                preconditions = env.build_preconditions()
                target_sizes = {t: ast_node_count(expr) for t, expr in formulas.items()}
                solver = z3.Solver()
                for p in preconditions:
                    solver.add(p)
                uncovered = [t for t in (all_targets - covered) if t in formulas]
                if not uncovered:
                    break
            else:
                break

        # Sort uncovered targets by AST size ascending so easy cones are solved first.
        uncovered.sort(key=lambda t: (target_sizes.get(t, 10000), t[0], t[1]))

        maxsmt_rounds += 1
        seed_model: Optional[z3.ModelRef] = None
        seed_tgt: Optional[Tuple[int, bool]] = None
        pushed_depth = 0

        for tgt in uncovered[:max_fallback_probes]:
            rem_ms = int((deadline - time.perf_counter()) * 1000.0)
            if rem_ms <= 0:
                break
            fallback_probes += 1
            solver.push()
            solver.set("timeout", min(probe_timeout_ms, rem_ms))
            solver.add(formulas[tgt])
            rc = solver.check()
            if rc == z3.sat:
                seed_model = solver.model()
                seed_tgt = tgt
                pushed_depth = 1
                break
            solver.pop()
            if rc == z3.unsat:
                dead_targets.add(tgt)
            else:
                attempted_failed.add(tgt)

        if seed_model is None:
            continue

        # Greedily pack additional uncovered targets onto the same incremental solver stack.
        pack_budget = min(max_soft_per_round - 1, len(uncovered) - 1)
        pack_timeout = min(60, probe_timeout_ms)
        packed = 0
        for cand in uncovered:
            if packed >= pack_budget or time.perf_counter() >= deadline:
                break
            if cand == seed_tgt or cand in dead_targets or cand in attempted_failed:
                continue
            # Skip candidates already satisfied by the current model.
            if z3.is_true(seed_model.eval(formulas[cand], model_completion=True)):
                continue
            rem_ms = int((deadline - time.perf_counter()) * 1000.0)
            if rem_ms <= 0:
                break
            solver.push()
            solver.set("timeout", min(pack_timeout, rem_ms))
            solver.add(formulas[cand])
            prc = solver.check()
            if prc == z3.sat:
                seed_model = solver.model()
                pushed_depth += 1
                packed += 1
            else:
                solver.pop()
                if prc == z3.unsat and pushed_depth == 0:
                    dead_targets.add(cand)

        for _p in range(pushed_depth):
            solver.pop()

        newly = record_model(seed_model, len(synthesized))
        if newly == 0 and seed_tgt is not None:
            attempted_failed.add(seed_tgt)

    return CoverageFuzzResult(
        total_predicates=len(preds),
        total_targets=len(all_targets),
        initial_covered=initial_covered,
        final_covered=len(covered),
        dead_proved=len(dead_targets),
        vectors_synthesized=len(synthesized),
        maxsmt_rounds=maxsmt_rounds,
        fallback_probes=fallback_probes,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        synthesized_overrides=synthesized,
        covered_targets=covered,
        dead_targets=dead_targets,
    )

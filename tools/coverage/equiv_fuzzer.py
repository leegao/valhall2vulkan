"""
Concolic Equivalence Fuzzer for Direct SPIR-V SSA vs. Lowered Valhall SSA.

Uses either the `direct` or `valhall` SSA side to seed concrete input vectors
that maximize block reachability and predicate coverage within a time budget,
then concretely executes both `direct_ssa` and `valhall_ssa` on every seeded
vector to test full output and side-effect equivalence under `DiffPolicy`,
with optional automatic counterexample shrinking and Valhall PC frontier
localization.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from tools.coverage.reachability import solve_maximal_reachability
from tools.coverage.shrinker import (
    DiffPolicy,
    DivergenceReport,
    align_vertex_gl_position,
    classify_value_diff,
    localize_and_shrink_divergence,
)
from tools.coverage.synthesizer import (
    collect_predicate_nodes,
    measure_predicate_coverage,
    synthesize_coverage_vectors,
)
from tools.symbolic_ssa.env import ConcreteShaderEnv
from tools.symbolic_ssa import ConcreteSsaInterpreter


@dataclass
class ConcolicEquivFuzzResult:
    """Summary of concolic equivalence testing between direct_ssa and valhall_ssa."""

    status: str  # EQUIV_EXACT, EQUIV_WITHIN_ULP, DIVERGED, SKIP
    seed_side: str
    vectors_tested: int
    init_vectors: int
    reachability_vectors: int
    coverage_vectors: int
    blocks_reached: int
    total_blocks: int
    predicates_initial: int
    predicates_final: int
    predicates_total: int
    checked_outputs: int
    benign_fp_diffs: int
    diverging_vectors: int
    elapsed_ms: float
    first_divergence: Optional[DivergenceReport] = None
    tested_overrides: List[Dict[str, Any]] = field(default_factory=list)


def _compare_pair_on_env(
    d_interp: ConcreteSsaInterpreter,
    v_interp: ConcreteSsaInterpreter,
    cenv: ConcreteShaderEnv,
    policy: Optional[DiffPolicy],
) -> Tuple[Dict[int, Any], Dict[int, Any], int, int, bool]:
    """
    Execute both interpreters on `cenv` and compare all outputs and side effects.
    Returns `(d_vals, v_vals, checked_count, benign_count, has_real_divergence)`.
    """
    d_vals, d_out = d_interp.execute_one(cenv)
    v_vals, v_out = v_interp.execute_one(cenv)
    d_out = align_vertex_gl_position(d_out, v_out, cenv, policy=policy)

    all_keys = sorted(set(d_out.keys()) | set(v_out.keys()))
    checked = 0
    benign = 0
    real_div = False
    for k in all_keys:
        checked += 1
        if k not in d_out or k not in v_out:
            real_div = True
            continue
        kind, _ = classify_value_diff(d_out[k], v_out[k], k, policy=policy)
        if kind in ("BENIGN_FP_ROUNDING", "NAN_PAYLOAD_ONLY"):
            benign += 1
        elif kind == "REAL_DIVERGENCE":
            real_div = True
    return d_vals, v_vals, checked, benign, real_div


def run_concolic_equiv_fuzz(
    bundle: Dict[str, Any],
    direct_ssa: Dict[str, Any],
    valhall_ssa: Dict[str, Any],
    seed_side: str = "direct",
    k_init: int = 32,
    time_budget_ms: int = 5000,
    fp_mode: str = "exact",
    max_unroll: int = 1,
    max_paths: int = 32,
    policy: Optional[DiffPolicy] = None,
    auto_shrink: bool = True,
    seed: int = 0x5A17,
    domain: str = "udomain",
) -> ConcolicEquivFuzzResult:
    """
    Concolically seed input vectors from `seed_side` (`direct` or `valhall`) to
    maximize reachability and predicate coverage, then concretely test both
    `direct_ssa` and `valhall_ssa` for output and side-effect equivalence.
    """
    t0 = time.perf_counter()
    deadline = t0 + (time_budget_ms / 1000.0)
    is_valhall_seed = seed_side == "valhall"
    seed_ssa = valhall_ssa if is_valhall_seed else direct_ssa
    eff_policy = policy if policy is not None else DiffPolicy()

    d_interp = ConcreteSsaInterpreter(direct_ssa, is_valhall_side=False)
    v_interp = ConcreteSsaInterpreter(valhall_ssa, is_valhall_side=True)

    preds = collect_predicate_nodes(seed_ssa)
    seed_traces: Dict[int, List[Any]] = {}
    tested_overrides: List[Dict[str, Any]] = []
    max_checked_outputs = 0
    total_benign = 0
    diverging_vecs = 0
    first_div_vec: Optional[Tuple[int, Dict[str, Any]]] = None

    # Phase 1: Initial concrete differential seed vectors (corner cases + PRNG).
    for i in range(k_init):
        cenv = ConcreteShaderEnv(bundle, vec_idx=i, seed=seed)
        d_vals, v_vals, chk, ben, is_div = _compare_pair_on_env(d_interp, v_interp, cenv, eff_policy)
        max_checked_outputs = max(max_checked_outputs, chk)
        total_benign += ben
        if is_div:
            diverging_vecs += 1
            if first_div_vec is None:
                first_div_vec = (i, {})
        svals = v_vals if is_valhall_seed else d_vals
        for nid, val in svals.items():
            seed_traces.setdefault(nid, []).append(val)

    init_cov = len(measure_predicate_coverage(preds, seed_traces))
    reach_vecs = 0
    cov_vecs = 0
    blocks_reached = 0
    total_blocks = len(seed_ssa.get("blocks", []))

    # Phase 2: Maximal-reachability path seeding on `seed_side`.
    rem_ms = int((deadline - time.perf_counter()) * 1000.0)
    if rem_ms > 50 and first_div_vec is None:
        reach_budget = max(200, rem_ms // 2)
        mres = solve_maximal_reachability(
            bundle,
            seed_ssa,
            is_valhall_side=is_valhall_seed,
            fp_mode=fp_mode,
            max_paths=max_paths,
            timeout_ms=min(500, reach_budget),
            max_unroll=max_unroll,
            domain=domain,
            total_timeout_ms=reach_budget,
        )
        blocks_reached = len(mres.reached_blocks)
        for p in mres.paths:
            vec_idx = k_init + reach_vecs
            reach_vecs += 1
            ov = dict(p.witness_overrides)
            tested_overrides.append(ov)
            cenv = ConcreteShaderEnv(bundle, vec_idx=vec_idx, seed=seed, overrides=ov)
            d_vals, v_vals, chk, ben, is_div = _compare_pair_on_env(d_interp, v_interp, cenv, eff_policy)
            max_checked_outputs = max(max_checked_outputs, chk)
            total_benign += ben
            if is_div:
                diverging_vecs += 1
                if first_div_vec is None:
                    first_div_vec = (vec_idx, ov)
            svals = v_vals if is_valhall_seed else d_vals
            for nid, val in svals.items():
                seed_traces.setdefault(nid, []).append(val)

    # Phase 3: Predicate-coverage synthesis on `seed_side` within remaining budget.
    rem_ms = int((deadline - time.perf_counter()) * 1000.0)
    if rem_ms > 50 and first_div_vec is None:
        cres = synthesize_coverage_vectors(
            bundle,
            seed_ssa,
            seed_traces,
            is_valhall_side=is_valhall_seed,
            fp_mode=fp_mode,
            domain=domain,
            base_vec_idx=k_init + reach_vecs,
            max_unroll=max_unroll,
            time_budget_ms=rem_ms,
        )
        for ov in cres.synthesized_overrides:
            vec_idx = k_init + reach_vecs + cov_vecs
            cov_vecs += 1
            tested_overrides.append(dict(ov))
            cenv = ConcreteShaderEnv(bundle, vec_idx=vec_idx, seed=seed, overrides=ov)
            d_vals, v_vals, chk, ben, is_div = _compare_pair_on_env(d_interp, v_interp, cenv, eff_policy)
            max_checked_outputs = max(max_checked_outputs, chk)
            total_benign += ben
            if is_div:
                diverging_vecs += 1
                if first_div_vec is None:
                    first_div_vec = (vec_idx, dict(ov))
            svals = v_vals if is_valhall_seed else d_vals
            for nid, val in svals.items():
                seed_traces.setdefault(nid, []).append(val)

    final_cov = len(measure_predicate_coverage(preds, seed_traces))
    first_report: Optional[DivergenceReport] = None
    if first_div_vec is not None and auto_shrink:
        div_idx, div_ov = first_div_vec
        first_report = localize_and_shrink_divergence(
            bundle,
            direct_ssa,
            valhall_ssa,
            overrides=div_ov,
            vec_idx=div_idx,
            policy=eff_policy,
        )

    if max_checked_outputs == 0:
        status = "SKIP"
    elif diverging_vecs > 0:
        status = "DIVERGED"
    elif total_benign > 0:
        status = "EQUIV_WITHIN_ULP"
    else:
        status = "EQUIV_EXACT"

    return ConcolicEquivFuzzResult(
        status=status,
        seed_side=seed_side,
        vectors_tested=k_init + reach_vecs + cov_vecs,
        init_vectors=k_init,
        reachability_vectors=reach_vecs,
        coverage_vectors=cov_vecs,
        blocks_reached=blocks_reached,
        total_blocks=total_blocks,
        predicates_initial=init_cov,
        predicates_final=final_cov,
        predicates_total=len(preds) * 2,
        checked_outputs=max_checked_outputs,
        benign_fp_diffs=total_benign,
        diverging_vectors=diverging_vecs,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        first_divergence=first_report,
        tested_overrides=tested_overrides,
    )

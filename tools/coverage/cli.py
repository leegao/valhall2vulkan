#!/usr/bin/env python3
"""CLI entrypoint for tools.coverage concolic synthesis, reachability, shrinking, and equivalence fuzzing."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

from tools.coverage.equiv_fuzzer import run_concolic_equiv_fuzz
from tools.coverage.reachability import solve_maximal_reachability
from tools.coverage.shrinker import DiffPolicy, localize_and_shrink_divergence
from tools.coverage.synthesizer import synthesize_coverage_vectors
from tools.symbolic_ssa.passes import canonicalize_ssa_module
from tools.verify.cec import run_concrete_diff


def _load_bundle(spv_or_json: str, bin_path: str) -> dict:
    if spv_or_json.endswith(".json"):
        with open(spv_or_json, "r", encoding="utf-8") as fh:
            return json.load(fh)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tmp_json = tf.name
    try:
        proc = subprocess.run(
            [bin_path, spv_or_json, "--quiet", "--dump-equiv-json", tmp_json],
            capture_output=True, text=True, timeout=15,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip()[:200])
        with open(tmp_json, "r", encoding="utf-8") as fh:
            return json.load(fh)
    finally:
        if os.path.exists(tmp_json):
            os.remove(tmp_json)


def _print_divergence_report(rep) -> None:
    print(
        f"  [SHRINK] output={rep.output_name} direct={rep.direct_val} valhall={rep.valhall_val} "
        f"kind={rep.diff_kind} ulp={rep.ulp_diff} "
        f"overrides={rep.original_override_count}->{rep.shrunk_override_count} "
        f"cone=({rep.cone_direct_size},{rep.cone_valhall_size})"
    )
    if rep.frontier_pcs:
        pcs_str = ", ".join(f"0x{pc:x}({dis})" if dis else f"0x{pc:x}" for pc, dis in rep.frontier_pcs[:6])
        print(f"    frontier_pcs: {pcs_str}")
    if rep.shrunk_overrides:
        print(f"    shrunk_overrides: {json.dumps(rep.shrunk_overrides, sort_keys=True)[:200]}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Concolic MaxSMT, Maximal-Reachability & Equivalence Fuzzer for ValhallSSA"
    )
    ap.add_argument("spv_or_json", help="Path to .spv shader or equivalence bundle .json")
    ap.add_argument("--bin", default="./build/spirv2valhall", help="Path to spirv2valhall binary")
    ap.add_argument("--side", choices=["direct", "valhall"], default="direct", help="SSA side to target/seed from")
    ap.add_argument("--k-init", type=int, default=32, help="Initial concrete seed vectors")
    ap.add_argument("--max-rounds", type=int, default=12, help="Maximum synthesis rounds")
    ap.add_argument("--fp-mode", choices=["exact", "real", "uf"], default="exact", help="Z3 FP mode")
    ap.add_argument("--stratum1", action="store_true", help="Canonicalize SSA via Stratum 1 before synthesis")
    ap.add_argument(
        "--maximal-reachability",
        action="store_true",
        help="Solve and report disjoint reachability conditions exercising all paths",
    )
    ap.add_argument(
        "--equiv-fuzz",
        action="store_true",
        help="Seed concrete vectors from --side and concretely test direct vs valhall equivalence",
    )
    ap.add_argument(
        "--shrink",
        action="store_true",
        help="Auto-shrink and localize any concrete divergence discovered during fuzzing/coverage",
    )
    ap.add_argument("--max-ulp", type=int, default=2, help="Max benign FP32 ULP tolerance for DiffPolicy")
    ap.add_argument("--max-paths", type=int, default=32, help="Maximum disjoint paths to enumerate")
    ap.add_argument("--timeout-ms", type=int, default=500, help="Per-solve Z3 timeout in ms")
    ap.add_argument("--time-budget-ms", type=int, default=5000, help="Total time budget in ms for synthesis/fuzzing")
    ap.add_argument("--max-unroll", type=int, default=1, help="Loop unroll depth (adaptive up to 3 in coverage mode)")
    ap.add_argument("--dump-vectors", default=None, help="Write synthesized override vectors or disjoint paths to JSON")
    args = ap.parse_args()

    try:
        bundle = _load_bundle(args.spv_or_json, args.bin)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    d_ssa = bundle["direct_ssa"]
    v_ssa = bundle["valhall_ssa"]
    if args.stratum1:
        d_ssa, _ = canonicalize_ssa_module(d_ssa)
        v_ssa, _ = canonicalize_ssa_module(v_ssa)

    is_valhall = args.side == "valhall"
    target_ssa = v_ssa if is_valhall else d_ssa
    policy = DiffPolicy(max_ulp=args.max_ulp)

    if args.equiv_fuzz:
        ef = run_concolic_equiv_fuzz(
            bundle,
            d_ssa,
            v_ssa,
            seed_side=args.side,
            k_init=args.k_init,
            time_budget_ms=args.time_budget_ms,
            fp_mode=args.fp_mode,
            max_unroll=args.max_unroll,
            max_paths=args.max_paths,
            policy=policy,
            auto_shrink=True,
        )
        print(
            f"[EQUIV-FUZZ] {os.path.basename(args.spv_or_json)} status={ef.status} "
            f"seed={ef.seed_side} vecs={ef.vectors_tested}(init={ef.init_vectors},reach={ef.reachability_vectors},cov={ef.coverage_vectors}) "
            f"blocks={ef.blocks_reached}/{ef.total_blocks} "
            f"preds={ef.predicates_initial}->{ef.predicates_final}/{ef.predicates_total} "
            f"benign_fp={ef.benign_fp_diffs} divs={ef.diverging_vectors} elapsed={ef.elapsed_ms:.1f}ms"
        )
        if ef.first_divergence is not None and ef.first_divergence.diverged:
            _print_divergence_report(ef.first_divergence)
        if args.dump_vectors:
            with open(args.dump_vectors, "w", encoding="utf-8") as fh:
                json.dump(ef.tested_overrides, fh, indent=2)
        return 1 if ef.status == "DIVERGED" else 0

    if args.maximal_reachability:
        mres = solve_maximal_reachability(
            bundle,
            target_ssa,
            is_valhall_side=is_valhall,
            fp_mode=args.fp_mode,
            max_paths=args.max_paths,
            timeout_ms=args.timeout_ms,
            max_unroll=args.max_unroll,
            total_timeout_ms=max(args.time_budget_ms, args.timeout_ms),
        )
        print(
            f"[REACHABILITY] {os.path.basename(args.spv_or_json)} side={args.side} "
            f"disjoint_paths={len(mres.paths)} blocks={len(mres.reached_blocks)}/{mres.total_blocks} "
            f"complete={mres.complete} ({mres.stop_reason}) elapsed={mres.elapsed_ms:.1f}ms"
        )
        for p in mres.paths:
            decs_str = ", ".join(f"bb{b}:n{c}={int(t)}" for b, c, t in p.decisions) or "unconditional"
            cond_preview = p.condition_smt.replace("\n", " ")
            if len(cond_preview) > 120:
                cond_preview = cond_preview[:117] + "..."
            print(f"  path[{p.path_id}] blocks={p.visited_blocks} decisions=[{decs_str}]")
            print(f"    PC: {cond_preview}")
        if args.dump_vectors:
            payload = [
                {
                    "path_id": p.path_id,
                    "visited_blocks": p.visited_blocks,
                    "decisions": p.decisions,
                    "condition_smt": p.condition_smt,
                    "witness_overrides": p.witness_overrides,
                }
                for p in mres.paths
            ]
            with open(args.dump_vectors, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        return 0

    cd = run_concrete_diff(bundle, d_ssa, v_ssa, k=args.k_init)
    traces = cd.v_node_traces if is_valhall else cd.d_node_traces
    res = synthesize_coverage_vectors(
        bundle,
        target_ssa,
        traces,
        is_valhall_side=is_valhall,
        fp_mode=args.fp_mode,
        max_rounds=args.max_rounds,
        probe_timeout_ms=min(args.timeout_ms, 250),
        base_vec_idx=args.k_init,
        max_unroll=args.max_unroll,
        time_budget_ms=args.time_budget_ms,
    )
    print(
        f"[COVERAGE] {os.path.basename(args.spv_or_json)} side={args.side} "
        f"covered={res.initial_covered}->{res.final_covered}/{res.total_targets} "
        f"({res.initial_pct:.1f}%->{res.final_pct:.1f}%) dead={res.dead_proved} "
        f"vecs={res.vectors_synthesized} elapsed={res.elapsed_ms:.1f}ms"
    )
    if args.shrink:
        checked_any = False
        if cd.status == "MISMATCH":
            for i in range(args.k_init):
                rep = localize_and_shrink_divergence(bundle, d_ssa, v_ssa, overrides={}, vec_idx=i, policy=policy)
                if rep.diverged:
                    _print_divergence_report(rep)
                    checked_any = True
                    break
        if not checked_any:
            for idx, ov in enumerate(res.synthesized_overrides):
                rep = localize_and_shrink_divergence(
                    bundle, d_ssa, v_ssa, overrides=ov, vec_idx=args.k_init + idx, policy=policy
                )
                if rep.diverged:
                    _print_divergence_report(rep)
                    break
    if args.dump_vectors:
        with open(args.dump_vectors, "w", encoding="utf-8") as fh:
            json.dump(res.synthesized_overrides, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

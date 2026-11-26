#!/usr/bin/env python3
"""CLI entrypoint for tools.coverage concolic synthesis, reachability, shrinking, and equivalence fuzzing."""

from __future__ import annotations

import argparse
import json
import os
import sys

from tools.coverage.equiv_fuzzer import run_concolic_equiv_fuzz
from tools.coverage.reachability import solve_maximal_reachability
from tools.coverage.recorder import record_repro_spvasm
from tools.coverage.shrinker import (
    DiffPolicy,
    DivergenceReport,
    localize_and_shrink_divergence,
)
from tools.coverage.synthesizer import synthesize_coverage_vectors
from tools.coverage.slice_diff import format_slice_diff
from tools.coverage.triage import format_divergence_and_triage, triage_divergence
from tools.symbolic_ssa.passes import canonicalize_ssa_module
from tools.equivalence.cec import run_concrete_diff
from tools.equivalence.spvasm_io import load_equiv_bundle


def _load_bundle(spv_or_json: str, bin_path: str, *, defau: bool = True, opt_ssa: bool = True) -> dict:
    return load_equiv_bundle(spv_or_json, bin_path, timeout=15, defau=defau, opt_ssa=opt_ssa)


def _handle_divergence(
    args: argparse.Namespace,
    bundle: dict,
    d_ssa: dict,
    v_ssa: dict,
    raw_d_ssa: dict,
    raw_v_ssa: dict,
    rep: DivergenceReport,
    policy: DiffPolicy,
) -> None:
    triage = triage_divergence(
        bundle,
        d_ssa,
        v_ssa,
        rep,
        raw_direct_ssa=raw_d_ssa if args.canonicalize else None,
        raw_valhall_ssa=raw_v_ssa if args.canonicalize else None,
        policy=policy,
        fp_range=args.fp_range,
    )
    for line in format_divergence_and_triage(rep, triage, trace_diff=args.trace_diff):
        print(line)
    if getattr(args, 'slice_diff', 0) > 0:
        for line in format_slice_diff(bundle, d_ssa, v_ssa, rep, depth=args.slice_diff, fp_range=args.fp_range):
            print(line)
    if args.shrink and not args.record and not args.spv_or_json.endswith('.json'):
        import tempfile as _tf
        _out = os.path.join(_tf.mkdtemp(prefix='valhall_shrink_'), 'shrunk.spvasm')
        rec = record_repro_spvasm(
            args.spv_or_json, _out, args.bin, bundle, rep, triage,
            canonicalize=args.canonicalize, policy=policy,
            reduce_timeout_s=args.record_timeout_s, reduce_steps=args.record_steps,
            no_reduce=args.no_reduce,
        )
        print(
            f'  [SHRINK/spirv-reduce] path={rec.out_path} bytes={rec.original_bytes}->{rec.recorded_bytes} '
            f'method={rec.reduce_method} verified={rec.verified_reproduces}'
        )
    if args.record:
        rec = record_repro_spvasm(
            args.spv_or_json,
            args.record,
            args.bin,
            bundle,
            rep,
            triage,
            canonicalize=args.canonicalize,
            policy=policy,
            reduce_timeout_s=args.record_timeout_s,
            reduce_steps=args.record_steps,
            no_reduce=args.no_reduce,
        )
        print(
            f"  [RECORD] path={rec.out_path} bytes={rec.original_bytes}->{rec.recorded_bytes} "
            f"method={rec.reduce_method} verified={rec.verified_reproduces}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Concolic MaxSMT, Maximal-Reachability & Equivalence Fuzzer for ValhallSSA"
    )
    ap.add_argument("spv_or_json", help="Path to .spv/.spvasm shader or equivalence bundle .json")
    ap.add_argument("--bin", default="./build/spirv2valhall", help="Path to spirv2valhall binary")
    ap.add_argument("--side", choices=["direct", "valhall"], default="direct", help="SSA side to target/seed from")
    ap.add_argument("--k-init", type=int, default=32, help="Initial concrete seed vectors")
    ap.add_argument("--max-rounds", type=int, default=12, help="Maximum synthesis rounds")
    ap.add_argument("--domain", choices=["bitvec", "udomain", "udomain_exact"], default="udomain", help="Symbolic value domain")
    ap.add_argument("--fp-mode", choices=["exact", "real", "uf"], default="exact", help="Z3 FP mode")
    ap.add_argument("--fp-range", choices=["unconstrained", "vulkan"], default="unconstrained", help="Concrete input FP range")
    ap.add_argument("--canonicalize", action="store_true", default=True, help="Canonicalize SSA before synthesis (default: on)")
    ap.add_argument("--no-canonicalize", dest="canonicalize", action="store_false", help="Disable SSA canonicalization")
    ap.add_argument("--defau", action="store_true", default=True, help="Run De-FAU devirtualization (default: on)")
    ap.add_argument("--no-defau", dest="defau", action="store_false", help="Disable De-FAU devirtualization")
    ap.add_argument("--opt-ssa", action="store_true", default=True, help="Run C++ ssa-opt canonicalization passes (default: on)")
    ap.add_argument("--no-opt-ssa", dest="opt_ssa", action="store_false", help="Disable C++ ssa-opt passes")
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
        help="Shrink diverging SPIR-V inputs via spirv-reduce (or override-shrink JSON bundles)",
    )
    ap.add_argument(
        "--slice-diff",
        type=int,
        default=0,
        metavar="DEPTH",
        help="Print N-hop side-by-side concrete backward-slice diff (Direct vs Valhall) on divergence",
    )
    ap.add_argument(
        "--trace-diff",
        action="store_true",
        help="Print concrete Valhall frontier node values and disassembly for deeper triage",
    )
    ap.add_argument(
        "--record",
        default=None,
        help="Record a shrunken .spvasm divergence reproduction to this file path or directory (e.g. bugs/)",
    )
    ap.add_argument(
        "--no-reduce",
        action="store_true",
        help="Skip spirv-reduce shrinking when using --record (disassemble original shader directly)",
    )
    ap.add_argument(
        "--record-timeout-s",
        type=float,
        default=15.0,
        help="Max wall-clock seconds for spirv-reduce during --record (default: 15.0)",
    )
    ap.add_argument(
        "--record-steps",
        type=int,
        default=120,
        help="Max spirv-reduce step limit during --record (default: 120)",
    )
    ap.add_argument("--max-ulp", type=int, default=2, help="Max benign FP32 ULP tolerance for DiffPolicy")
    ap.add_argument("--max-paths", type=int, default=32, help="Maximum disjoint paths to enumerate")
    ap.add_argument("--timeout-ms", type=int, default=500, help="Per-solve Z3 timeout in ms")
    ap.add_argument("--time-budget-ms", type=int, default=5000, help="Total time budget in ms for synthesis/fuzzing")
    ap.add_argument("--max-unroll", type=int, default=1, help="Loop unroll depth (adaptive up to 3 in coverage mode)")
    ap.add_argument("--dump-vectors", default=None, help="Write synthesized override vectors or disjoint paths to JSON")
    args = ap.parse_args()

    try:
        bundle = _load_bundle(args.spv_or_json, args.bin, defau=args.defau, opt_ssa=args.opt_ssa)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    raw_d_ssa = bundle["direct_ssa"]
    raw_v_ssa = bundle["valhall_ssa"]
    d_ssa = raw_d_ssa
    v_ssa = raw_v_ssa
    if args.canonicalize:
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
            domain=args.domain,
            max_unroll=args.max_unroll,
            max_paths=args.max_paths,
            policy=policy,
            auto_shrink=True,
            fp_range=args.fp_range,
        )
        print(
            f"[EQUIV-FUZZ] {os.path.basename(args.spv_or_json)} status={ef.status} "
            f"seed={ef.seed_side} vecs={ef.vectors_tested}(init={ef.init_vectors},reach={ef.reachability_vectors},cov={ef.coverage_vectors}) "
            f"blocks={ef.blocks_reached}/{ef.total_blocks} "
            f"preds={ef.predicates_initial}->{ef.predicates_final}/{ef.predicates_total} "
            f"benign_fp={ef.benign_fp_diffs} divs={ef.diverging_vectors} elapsed={ef.elapsed_ms:.1f}ms"
        )
        if ef.first_divergence is not None and ef.first_divergence.diverged:
            _handle_divergence(args, bundle, d_ssa, v_ssa, raw_d_ssa, raw_v_ssa, ef.first_divergence, policy)
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
            domain=args.domain,
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
        domain=args.domain,
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
    if args.shrink or args.record:
        checked_any = False
        if cd.status == "MISMATCH":
            for i in range(args.k_init):
                rep = localize_and_shrink_divergence(bundle, d_ssa, v_ssa, overrides={}, vec_idx=i, policy=policy)
                if rep.diverged:
                    _handle_divergence(args, bundle, d_ssa, v_ssa, raw_d_ssa, raw_v_ssa, rep, policy)
                    checked_any = True
                    break
        if not checked_any:
            for idx, ov in enumerate(res.synthesized_overrides):
                rep = localize_and_shrink_divergence(
                    bundle, d_ssa, v_ssa, overrides=ov, vec_idx=args.k_init + idx, policy=policy
                )
                if rep.diverged:
                    _handle_divergence(args, bundle, d_ssa, v_ssa, raw_d_ssa, raw_v_ssa, rep, policy)
                    break
    if args.dump_vectors:
        with open(args.dump_vectors, "w", encoding="utf-8") as fh:
            json.dump(res.synthesized_overrides, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

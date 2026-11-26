#!/usr/bin/env python3
"""CLI entrypoint for tools.corpus_synth greedy and targeted Valhall-ISA / SPIR-V synthesizer."""

from __future__ import annotations

import argparse
import json
import os
import sys

from tools.corpus_synth.greedy_driver import synthesize_corpus
from tools.corpus_synth.isa_target_synth import (
    build_coverage_report,
    list_all_mnemonics,
    list_mnemonics_by_tier,
    list_synthesizable_mnemonics,
    run_full_raw_sweep,
    synthesize_for_isa,
)
from tools.corpus_synth.spv_fuzzer import fuzz_spv_with_spirv_fuzz


def _resolve_target_mnemonics(spec: str, mode: str) -> list[str]:
    upper = spec.upper()
    if upper == "ALL":
        return list_synthesizable_mnemonics() if mode == "spirv" else list_all_mnemonics()
    if upper in ("COMPUTE", "GRAPHICS", "INTERNAL"):
        return list_mnemonics_by_tier(upper)
    return [m.strip().upper() for m in spec.split(",") if m.strip()]


def _dump_shader_text(label: str, spvasm: str, out_dir: str | None, to_stdout: bool) -> None:
    if not spvasm:
        return
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{label}.spvasm")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(spvasm)
        print(f"  [SHADER-DUMP] {path}")
    if to_stdout:
        print(f"  --- BEGIN SHADER ({label}) ---")
        print(spvasm.rstrip())
        print(f"  --- END SHADER ({label}) ---")


def _run_target_isa_mode(args: argparse.Namespace) -> int:
    # Fast batched path when sweeping ALL in pure raw mode
    if args.target_isa.upper() == "ALL" and args.mode == "raw":
        results = run_full_raw_sweep(
            all_variants=args.variants,
            include_fragments=args.mutate_fields,
            include_branch_points=args.mutate_fields,
            bin_path=args.bin,
        )
        dec_ok = sum(1 for r in results if r.decoded_ok)
        p1_ok = sum(1 for r in results if r.pass1_ok)
        bp_cnt = sum(1 for r in results if r.is_branch_point)
        print(
            f"[RAW-ISA-SWEEP] cases={len(results)} decoded={dec_ok}/{len(results)} "
            f"pass1_ok={p1_ok}/{len(results)} branch_points={bp_cnt}"
        )
        for r in results:
            if not r.decoded_ok or not r.pass1_ok:
                print(f"  [FAIL] {r.case_id} word={r.word_hex} dec={r.decoded_ok} p1={r.pass1_ok} err={r.error}")
        return 0 if dec_ok == len(results) and p1_ok == len(results) else 1

    eff_out_dir = args.output_shaders or args.out_dir
    targets = _resolve_target_mnemonics(args.target_isa, args.mode)
    hit = 0
    equiv_ok = 0
    raw_total = 0
    raw_pass = 0
    total = len(targets)
    for mn in targets:
        res = synthesize_for_isa(
            mn,
            complexity=args.complexity,
            mode=args.mode,
            mutate_fields=args.mutate_fields,
            all_variants=args.variants,
            out_dir=eff_out_dir,
            bin_path=args.bin,
            verify_equiv=args.verify_equiv,
            k_init=args.k_init,
            fp_range=args.fp_range,
            seed=args.seed,
            materialize_intermediates=args.materialize_intermediates,
            max_coverage_seeds=args.max_coverage_seeds,
            slice_diff_depth=args.slice_diff,
        )
        if res.target_emitted:
            hit += 1
        if res.equiv_status in ("EQUIV_EXACT", "EQUIV_WITHIN_ULP"):
            equiv_ok += 1
        if res.raw_results:
            raw_total += len(res.raw_results)
            raw_pass += res.raw_passed
        emit_s = "HIT" if res.target_emitted else ("ERR" if not res.compiled_ok else "MISS")
        eq_s = f" equiv={res.equiv_status}(divs={res.diverging_vectors})" if res.equiv_status else ""
        raw_s = f" raw={res.raw_passed}/{len(res.raw_results)}" if res.raw_results else ""
        print(
            f"[ISA-SYNTH] {mn:18s} mode={res.mode_used:5s} stage={res.stage:8s} "
            f"cplx={res.complexity:7s} emit={emit_s:4s}{eq_s}{raw_s} elapsed={res.elapsed_ms:.0f}ms"
        )
        if res.error:
            print(f"  [ERROR] {res.error[:200]}")
        elif not res.target_emitted:
            print(f"  [EMITTED] {', '.join(sorted(res.emitted_mnemonics))}")

        rep = res.divergence_report
        diverged = rep is not None and rep.diverged
        if diverged:
            print(
                f"  [MISMATCH] vec={rep.vec_idx} out={rep.output_name} "
                f"direct={rep.direct_val} valhall={rep.valhall_val} kind={rep.diff_kind}"
            )
            if rep.frontier_pcs:
                pcs_str = ", ".join(f"0x{pc:x}({dis})" if dis else f"0x{pc:x}" for pc, dis in rep.frontier_pcs[:4])
                print(f"    frontier_pcs: {pcs_str}")
            for line in res.slice_diff_lines:
                print(line)

        # Dump shaders when explicitly requested OR automatically on --verify-equiv mismatch/failure
        if args.dump_shaders or (args.verify_equiv and (diverged or not res.compiled_ok)):
            _dump_shader_text(
                f"isa_{mn.lower()}_{res.complexity}",
                res.spvasm,
                eff_out_dir,
                to_stdout=(args.dump_shaders or not eff_out_dir),
            )

        # Optional spirv-fuzz mutation pass on synthesized SPIR-V
        if args.fuzz and res.spv_path and os.path.isfile(res.spv_path):
            fuzzed = fuzz_spv_with_spirv_fuzz(
                res.spv_path,
                num_variants=args.fuzz_variants,
                base_seed=args.seed,
                out_dir=eff_out_dir,
                bin_path=args.bin,
                verify_equiv=args.verify_equiv,
                k_init=args.k_init,
                fp_range=args.fp_range,
            )
            for idx_f, fr in enumerate(fuzzed):
                feq = f" equiv={fr.equiv_status}(divs={fr.diverging_vectors})" if fr.equiv_status else ""
                ferr = f" err={fr.error[:120]}" if fr.error else ""
                print(
                    f"  [SPIRV-FUZZ #{idx_f}] seed={fr.fuzz_seed} fuzz_ok={fr.fuzz_ok} "
                    f"valid={fr.valid_spv} compiled={fr.compiled_ok}{feq} elapsed={fr.elapsed_ms:.0f}ms{ferr}"
                )

    if total > 1:
        eq_summary = f" equiv={equiv_ok}/{total}" if args.verify_equiv else ""
        raw_summary = f" raw={raw_pass}/{raw_total}" if raw_total else ""
        print(f"[ISA-SYNTH SUMMARY] emitted={hit}/{total}{eq_summary}{raw_summary}")
    return 0 if hit == total else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Greedy & Targeted Valhall-ISA / SPIR-V Corpus Synthesizer")
    ap.add_argument("--target", choices=["valhall-isa", "spirv-ops"], default="valhall-isa")
    ap.add_argument(
        "--target-isa",
        default=None,
        help="Target specific Valhall mnemonic(s) (comma-separated, 'ALL', 'COMPUTE', 'GRAPHICS', or 'INTERNAL')",
    )
    ap.add_argument(
        "--mode",
        choices=["auto", "spirv", "raw", "both"],
        default="auto",
        help="Synthesis backend for --target-isa: spirv (libMali), raw (64b encoder), auto, or both",
    )
    ap.add_argument(
        "--mutate-fields",
        action="store_true",
        help="Attach declarative field fragments & branch-point mutations in --target-isa mode",
    )
    ap.add_argument(
        "--variants",
        action="store_true",
        help="Cover all XML (mnemonic, variant) specs instead of only the primary variant",
    )
    ap.add_argument(
        "--coverage-audit",
        action="store_true",
        help="Print JSON summary of 201-mnemonic / 627-variant / 17-fragment coverage and exit",
    )
    ap.add_argument(
        "--complexity",
        choices=["minimal", "medium", "branchy"],
        default="minimal",
        help="Shader complexity tier for --target-isa mode",
    )
    ap.add_argument("--num-shaders", type=int, default=6, help="Number of shaders to synthesize")
    ap.add_argument("--blocks", type=int, default=1, help="Target CFG blocks per shader (default: 1)")
    ap.add_argument("--insns", type=int, default=12, help="Catalog ops per shader")
    ap.add_argument("--candidates", type=int, default=4, help="Greedy candidates evaluated per shader")
    ap.add_argument("--out-dir", default=None, help="Directory to write synthesized .spv files")
    ap.add_argument("--output-shaders", default=None, help="Directory to write synthesized .spvasm/.spv shaders")
    ap.add_argument("--dump-shaders", action="store_true", help="Print generated .spvasm shader(s) to stdout")
    ap.add_argument(
        "--materialize-intermediates",
        action="store_true",
        help="Pin intermediate scalar SSA IDs in outbuf so optimizations cannot DCE them and equiv_fuzzer verifies them",
    )
    ap.add_argument(
        "--max-coverage-seeds",
        action="store_true",
        help="Use maximal-reachability / coverage-maximizing concolic seeds during --verify-equiv",
    )
    ap.add_argument(
        "--fuzz",
        action="store_true",
        help="Generate semantically equivalent fuzzed alternate shaders via spirv-fuzz",
    )
    ap.add_argument(
        "--fuzz-variants",
        type=int,
        default=2,
        help="Number of spirv-fuzz mutants to generate per seed shader when --fuzz is set",
    )
    ap.add_argument(
        "--slice-diff",
        type=int,
        default=0,
        metavar="DEPTH",
        help="Print N-hop side-by-side concrete backward-slice diff (Direct vs Valhall) on divergence",
    )
    ap.add_argument("--bin", default="./build/spirv2valhall", help="Path to spirv2valhall binary")
    ap.add_argument("--seed", type=int, default=0xC0DE, help="PRNG seed")
    ap.add_argument("--k-init", type=int, default=32, help="Initial vectors for --verify-equiv in --target-isa mode")
    ap.add_argument("--fp-range", choices=["unconstrained", "vulkan"], default="unconstrained")
    ap.add_argument("--domain", choices=["bitvec", "udomain", "udomain_exact"], default="udomain", help="Symbolic value domain")
    ap.add_argument("--verify-equiv", action="store_true", help="Run concolic equiv-fuzz on each synthesized shader")
    args = ap.parse_args()

    if args.coverage_audit:
        print(json.dumps(build_coverage_report(), indent=2, sort_keys=True))
        return 0

    if args.target_isa:
        return _run_target_isa_mode(args)

    eff_out_dir = args.output_shaders or args.out_dir
    res = synthesize_corpus(
        target=args.target,
        num_shaders=args.num_shaders,
        num_blocks=args.blocks,
        num_insns=args.insns,
        out_dir=eff_out_dir,
        bin_path=args.bin,
        seed=args.seed,
        candidates_per_shader=args.candidates,
        verify_equiv=args.verify_equiv,
        domain=args.domain,
    )
    print(
        f"[CORPUS-SYNTH] target={res.target} shaders={len(res.shaders)} "
        f"unique_features={len(res.total_unique_features)} "
        f"min_cover={len(res.minimal_cover_indices)}/{len(res.shaders)} "
        f"elapsed={res.elapsed_ms:.1f}ms"
    )
    for r in res.shaders:
        eq_s = f" equiv={r.equiv_status}" if r.equiv_status else ""
        print(
            f"  synth[{r.shader_id}] blocks={r.num_blocks} +{len(r.new_features)} new "
            f"(total={len(r.all_features)}){eq_s} {r.spv_path}"
        )
        rep = r.divergence_report
        if rep is not None and rep.diverged:
            print(
                f"    [MISMATCH] output={rep.output_name} direct={rep.direct_val} valhall={rep.valhall_val} "
                f"kind={rep.diff_kind} ulp={rep.ulp_diff} "
                f"overrides={rep.original_override_count}->{rep.shrunk_override_count} "
                f"cone=({rep.cone_direct_size},{rep.cone_valhall_size})"
            )
            if rep.frontier_pcs:
                pcs_str = ", ".join(f"0x{pc:x}({dis})" if dis else f"0x{pc:x}" for pc, dis in rep.frontier_pcs[:6])
                print(f"      frontier_pcs: {pcs_str}")
            if rep.shrunk_overrides:
                print(f"      shrunk_overrides: {json.dumps(rep.shrunk_overrides, sort_keys=True)[:200]}")
        if args.fuzz and r.spv_path and os.path.isfile(r.spv_path):
            for idx_f, fr in enumerate(
                fuzz_spv_with_spirv_fuzz(
                    r.spv_path,
                    num_variants=args.fuzz_variants,
                    base_seed=args.seed + r.shader_id,
                    out_dir=eff_out_dir,
                    bin_path=args.bin,
                    verify_equiv=args.verify_equiv,
                    k_init=args.k_init,
                    fp_range=args.fp_range,
                )
            ):
                feq = f" equiv={fr.equiv_status}(divs={fr.diverging_vectors})" if fr.equiv_status else ""
                print(f"    [SPIRV-FUZZ #{idx_f}] fuzz_ok={fr.fuzz_ok} valid={fr.valid_spv} compiled={fr.compiled_ok}{feq}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

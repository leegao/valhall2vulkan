#!/usr/bin/env python3
"""CLI entrypoint for tools.corpus_synth greedy and targeted Valhall-ISA / SPIR-V synthesizer."""

from __future__ import annotations

import argparse
import json
import sys

from tools.corpus_synth.greedy_driver import synthesize_corpus
from tools.corpus_synth.isa_target_synth import (
    list_synthesizable_mnemonics,
    synthesize_for_isa,
)


def _run_target_isa_mode(args: argparse.Namespace) -> int:
    targets = (
        list_synthesizable_mnemonics()
        if args.target_isa.upper() == "ALL"
        else [m.strip().upper() for m in args.target_isa.split(",") if m.strip()]
    )
    hit = 0
    equiv_ok = 0
    total = len(targets)
    for mn in targets:
        res = synthesize_for_isa(
            mn,
            complexity=args.complexity,
            out_dir=args.out_dir,
            bin_path=args.bin,
            verify_equiv=args.verify_equiv,
            k_init=args.k_init,
            fp_range=args.fp_range,
            seed=args.seed,
        )
        if res.target_emitted:
            hit += 1
        if res.equiv_status in ("EQUIV_EXACT", "EQUIV_WITHIN_ULP"):
            equiv_ok += 1
        emit_s = "HIT" if res.target_emitted else ("ERR" if not res.compiled_ok else "MISS")
        eq_s = f" equiv={res.equiv_status}(divs={res.diverging_vectors})" if res.equiv_status else ""
        print(
            f"[ISA-SYNTH] {mn:16s} stage={res.stage:8s} cplx={res.complexity:7s} "
            f"emit={emit_s:4s}{eq_s} elapsed={res.elapsed_ms:.0f}ms"
        )
        if res.error:
            print(f"  [ERROR] {res.error[:200]}")
        elif not res.target_emitted:
            print(f"  [EMITTED] {', '.join(sorted(res.emitted_mnemonics))}")
        rep = res.divergence_report
        if rep is not None and rep.diverged:
            print(
                f"  [MISMATCH] vec={rep.vec_idx} out={rep.output_name} "
                f"direct={rep.direct_val} valhall={rep.valhall_val} kind={rep.diff_kind}"
            )
            if rep.frontier_pcs:
                pcs_str = ", ".join(f"0x{pc:x}({dis})" if dis else f"0x{pc:x}" for pc, dis in rep.frontier_pcs[:4])
                print(f"    frontier_pcs: {pcs_str}")
    if total > 1:
        eq_summary = f" equiv={equiv_ok}/{total}" if args.verify_equiv else ""
        print(f"[ISA-SYNTH SUMMARY] emitted={hit}/{total}{eq_summary}")
    return 0 if hit == total else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Greedy & Targeted Valhall-ISA / SPIR-V Corpus Synthesizer")
    ap.add_argument("--target", choices=["valhall-isa", "spirv-ops"], default="valhall-isa")
    ap.add_argument(
        "--target-isa",
        default=None,
        help="Target specific Valhall mnemonic(s) (comma-separated or 'ALL') using valhall_isel_catalog",
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
    ap.add_argument("--bin", default="./build/spirv2valhall", help="Path to spirv2valhall binary")
    ap.add_argument("--seed", type=int, default=0xC0DE, help="PRNG seed")
    ap.add_argument("--k-init", type=int, default=32, help="Initial vectors for --verify-equiv in --target-isa mode")
    ap.add_argument("--fp-range", choices=["unconstrained", "vulkan"], default="unconstrained")
    ap.add_argument("--domain", choices=["bitvec", "udomain", "udomain_exact"], default="udomain", help="Symbolic value domain")
    ap.add_argument("--verify-equiv", action="store_true", help="Run concolic equiv-fuzz on each synthesized shader")
    args = ap.parse_args()

    if args.target_isa:
        return _run_target_isa_mode(args)

    res = synthesize_corpus(
        target=args.target,
        num_shaders=args.num_shaders,
        num_blocks=args.blocks,
        num_insns=args.insns,
        out_dir=args.out_dir,
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
    return 0


if __name__ == "__main__":
    sys.exit(main())

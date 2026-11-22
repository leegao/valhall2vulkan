#!/usr/bin/env python3
"""CLI entrypoint for tools.corpus_synth greedy Valhall/SPIR-V corpus synthesizer."""

from __future__ import annotations

import argparse
import sys

from tools.corpus_synth.greedy_driver import synthesize_corpus


def main() -> int:
    ap = argparse.ArgumentParser(description="Greedy Valhall-ISA / SPIR-V Op Corpus Synthesizer")
    ap.add_argument("--target", choices=["valhall-isa", "spirv-ops"], default="valhall-isa")
    ap.add_argument("--num-shaders", type=int, default=6, help="Number of shaders to synthesize")
    ap.add_argument("--blocks", type=int, default=1, help="Target CFG blocks per shader (default: 1)")
    ap.add_argument("--insns", type=int, default=12, help="Catalog ops per shader")
    ap.add_argument("--candidates", type=int, default=4, help="Greedy candidates evaluated per shader")
    ap.add_argument("--out-dir", default=None, help="Directory to write synthesized .spv files")
    ap.add_argument("--bin", default="./build/spirv2valhall", help="Path to spirv2valhall binary")
    ap.add_argument("--seed", type=int, default=0xC0DE, help="PRNG seed")
    ap.add_argument("--verify-equiv", action="store_true", help="Run concolic equiv-fuzz on each synthesized shader")
    args = ap.parse_args()

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
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Targeted Single-Instruction Valhall ISA Synthesizer & Field-Mutation Orchestrator.

Unites two complementary synthesis backends behind a single clean interface:
  1. **SPIR-V Synthesis (`mode="spirv"`)**: Uses declarative Vulkan 1.3 SPIR-V
     recipes (`isa_recipes.py` + `isa_spv_harness.py`, 128 empirically verified
     mnemonics + variant recipes) compiled via `libMali-Gxx` (`spirv2valhall`)
     and optionally validated end-to-end with `equiv_fuzzer`.
  2. **Raw Valhall Synthesis & Field Mutation (`mode="raw"`)**: Uses Kraid XML
     metadata (`isa_spec_index.py`), 17 reusable cross-instruction field
     fragments & branch-point rules (`isa_field_fragments.py`), and the 64-bit
     Valhall binary encoder (`valhall_raw_synth.py`) to cover 100% of all 201
     mnemonics, all 627 `(mnemonic, variant)` XML specs, and all hardware
     modifier/operand regimes directly through `decode_valhall_shader` +
     `valhall_pass1`.

In `mode="auto"` (default for single/full sweeps), SPIR-V synthesis is used
whenever an executable SPIR-V recipe exists, and raw Valhall synthesis covers
the remaining mnemonics (and all field/branch-point mutations when
`mutate_fields=True`).
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
import time
from typing import Any

from tools.corpus_synth.isa_field_fragments import (
    FRAGMENT_REGISTRY,
    applicable_fragments,
    generate_field_cases_for_spec,
)
from tools.corpus_synth.isa_recipes import RECIPES, VARIANT_RECIPES
from tools.spec.isa_spec_index import IsaSpecIndex, load_isa_spec_index
from tools.corpus_synth.isa_spv_harness import IsaRecipe, build_spvasm_from_recipe
from tools.corpus_synth.oracle import _parse_valhall_mnemonic
from tools.corpus_synth.valhall_isel_catalog import (
    VALHALL_ISEL_CATALOG,
    get_rule_for_mnemonic,
)
from tools.corpus_synth.valhall_raw_synth import (
    RawVerifyResult,
    encode_mutation_case,
    synthesize_raw_for_mnemonic,
    verify_raw_valhall_batch,
)
from tools.coverage.equiv_fuzzer import run_concolic_equiv_fuzz
from tools.coverage.shrinker import DiffPolicy, DivergenceReport
from tools.coverage.slice_diff import format_slice_diff
from tools.symbolic_ssa.passes import canonicalize_ssa_module
from tools.equivalence.spvasm_io import assemble_spvasm, load_equiv_bundle


@dataclasses.dataclass
class IsaTargetResult:
    """Outcome of synthesizing and optionally fuzz/Pass-1-validating a target Valhall mnemonic."""

    mnemonic: str
    complexity: str
    stage: str
    spvasm: str
    spv_path: str
    compiled_ok: bool
    target_emitted: bool
    emitted_mnemonics: set[str]
    emitted_variants: list[str]
    mode_used: str = "spirv"
    equiv_status: str = ""
    diverging_vectors: int = 0
    benign_fp_diffs: int = 0
    divergence_report: DivergenceReport | None = None
    raw_results: list[RawVerifyResult] = dataclasses.field(default_factory=list)
    slice_diff_lines: list[str] = dataclasses.field(default_factory=list)
    elapsed_ms: float = 0.0
    error: str = ""

    @property
    def raw_passed(self) -> int:
        return sum(1 for r in self.raw_results if r.decoded_ok and r.pass1_ok)


def list_synthesizable_mnemonics() -> list[str]:
    """Return all Valhall mnemonics that have an executable Vulkan SPIR-V recipe."""
    return sorted(RECIPES.keys())


def list_all_mnemonics(index: IsaSpecIndex | None = None) -> list[str]:
    """Return all 201 Valhall mnemonics defined in `isa-v9-v14.xml` (SPIR-V + raw)."""
    idx = index or load_isa_spec_index()
    return idx.all_mnemonics()


def list_mnemonics_by_tier(tier: str) -> list[str]:
    """Return all catalog mnemonics matching `tier` (`compute`, `graphics`, `internal`)."""
    t = tier.lower()
    return sorted(r.mnemonic for r in VALHALL_ISEL_CATALOG if r.tier == t)


def build_spvasm_for_isa(
    mnemonic: str,
    complexity: str = "minimal",
    *,
    variant: str | None = None,
    materialize_intermediates: bool = False,
) -> tuple[str, str]:
    """Return `(stage, spvasm_text)` for `mnemonic` (and optional `variant`) at `complexity`."""
    mn = mnemonic.upper()
    recipe: IsaRecipe | None = None
    if variant is not None:
        recipe = VARIANT_RECIPES.get((mn, variant))
    if recipe is None:
        recipe = RECIPES.get(mn)
    if recipe is None:
        rule = get_rule_for_mnemonic(mn)
        hint = f" (catalog tier={rule.tier}: {rule.spirv_pattern})" if rule else ""
        raise ValueError(f"No executable SPIR-V recipe for Valhall mnemonic '{mn}'{hint}")
    return recipe.stage, build_spvasm_from_recipe(recipe, complexity=complexity, materialize_intermediates=materialize_intermediates)


def synthesize_for_isa(
    mnemonic: str,
    *,
    complexity: str = "minimal",
    variant: str | None = None,
    mode: str = "auto",
    mutate_fields: bool = False,
    all_variants: bool = False,
    max_per_fragment: int = 4,
    out_dir: str | None = None,
    bin_path: str = "./build/spirv2valhall",
    verify_equiv: bool = False,
    k_init: int = 32,
    fp_range: str = "unconstrained",
    seed: int = 42,
    materialize_intermediates: bool = False,
    max_coverage_seeds: bool = False,
    slice_diff_depth: int = 0,
) -> IsaTargetResult:
    """
    Synthesize and verify a target Valhall mnemonic.

    Modes:
      - `"spirv"`: Compile SPIR-V recipe through `spirv2valhall` (requires recipe in `RECIPES`).
      - `"raw"`: Encode raw 64-bit Valhall stream(s) from `isa-v9-v14.xml` and verify decode + Pass-1.
      - `"auto"`: Use `"spirv"` if a SPIR-V recipe exists, else `"raw"`.
      - `"both"`: Run SPIR-V synthesis (if recipe exists) AND raw Valhall synthesis/mutation.
    """
    t0 = time.perf_counter()
    mn = mnemonic.upper()
    has_spv = mn in RECIPES or (variant is not None and (mn, variant) in VARIANT_RECIPES)
    eff_mode = mode
    if eff_mode == "auto":
        eff_mode = "spirv" if has_spv else "raw"

    # Pure raw synthesis mode (covers all 201 mnemonics including internal & non-emitted ops)
    if eff_mode == "raw":
        raw_res = synthesize_raw_for_mnemonic(
            mn,
            all_variants=all_variants,
            include_fragments=mutate_fields,
            include_branch_points=mutate_fields,
            max_per_fragment=max_per_fragment,
            bin_path=bin_path,
        )
        ok_all = bool(raw_res) and all(r.decoded_ok and r.pass1_ok for r in raw_res)
        emitted_mns = {r.decoded_mnemonic for r in raw_res if r.decoded_ok} | {
            r.decoded_canonical_mnemonic for r in raw_res if r.decoded_ok
        }
        emitted_vars = sorted({
            f"{r.decoded_mnemonic}.{r.decoded_variant}" if r.decoded_variant else r.decoded_mnemonic
            for r in raw_res
            if r.decoded_ok
        })
        first_err = next((r.error for r in raw_res if r.error), "")
        elapsed = (time.perf_counter() - t0) * 1000.0
        return IsaTargetResult(
            mnemonic=mn,
            complexity=complexity,
            stage="raw",
            spvasm="",
            spv_path="",
            compiled_ok=ok_all,
            target_emitted=ok_all and all(r.mnemonic_matched for r in raw_res),
            emitted_mnemonics=emitted_mns,
            emitted_variants=emitted_vars,
            mode_used="raw",
            raw_results=raw_res,
            elapsed_ms=elapsed,
            error=first_err,
        )

    # SPIR-V synthesis path (`mode == "spirv"` or `"both"` with SPIR-V recipe)
    stage, spvasm = build_spvasm_for_isa(mn, complexity=complexity, variant=variant, materialize_intermediates=materialize_intermediates)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        vtag = f"_{variant}" if variant else ""
        asm_path = os.path.join(out_dir, f"isa_{mn.lower()}{vtag}_{complexity}.spvasm")
        spv_path = os.path.join(out_dir, f"isa_{mn.lower()}{vtag}_{complexity}.spv")
        cleanup = False
    else:
        td = tempfile.mkdtemp(prefix=f"isa_{mn.lower()}_")
        asm_path = os.path.join(td, f"isa_{mn.lower()}_{complexity}.spvasm")
        spv_path = os.path.join(td, f"isa_{mn.lower()}_{complexity}.spv")
        cleanup = True

    with open(asm_path, "w", encoding="utf-8") as f:
        f.write(spvasm)

    try:
        assemble_spvasm(asm_path, spv_path, target_env="vulkan1.3")
        bundle = load_equiv_bundle(spv_path, bin_path, defau=True, opt_ssa=True)
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000.0
        return IsaTargetResult(
            mnemonic=mn,
            complexity=complexity,
            stage=stage,
            spvasm=spvasm,
            spv_path=spv_path if not cleanup else "",
            compiled_ok=False,
            target_emitted=False,
            emitted_mnemonics=set(),
            emitted_variants=[],
            mode_used=eff_mode,
            elapsed_ms=elapsed,
            error=str(exc),
        )

    pc_disasm: dict[str, str] = bundle.get("mbs2_context", {}).get("pc_disasm", {})
    emitted_variants: list[str] = []
    emitted_mnemonics: set[str] = set()
    for line in pc_disasm.values():
        tok = _parse_valhall_mnemonic(line)
        if tok:
            emitted_variants.append(tok)
            emitted_mnemonics.add(tok.split(".")[0])

    target_emitted = mn in emitted_mnemonics
    equiv_status = ""
    divs = 0
    benign = 0
    div_rep: DivergenceReport | None = None
    slice_lines: list[str] = []

    if verify_equiv and bundle.get("direct_ok") and bundle.get("valhall_ok"):
        d_ssa, _ = canonicalize_ssa_module(bundle["direct_ssa"])
        v_ssa, _ = canonicalize_ssa_module(bundle["valhall_ssa"])
        ef = run_concolic_equiv_fuzz(
            bundle,
            d_ssa,
            v_ssa,
            k_init=k_init,
            fp_range=fp_range,
            seed=seed,
            max_paths=32 if max_coverage_seeds else 8,
            time_budget_ms=5000 if max_coverage_seeds else 2000,
            policy=DiffPolicy(),
        )
        equiv_status = ef.status
        divs = ef.diverging_vectors
        benign = ef.benign_fp_diffs
        div_rep = ef.first_divergence
        if slice_diff_depth > 0 and div_rep is not None and div_rep.diverged:
            slice_lines = format_slice_diff(bundle, d_ssa, v_ssa, div_rep, depth=slice_diff_depth, fp_range=fp_range)

    raw_res: list[RawVerifyResult] = []
    if eff_mode == "both" or mutate_fields:
        raw_res = synthesize_raw_for_mnemonic(
            mn,
            all_variants=all_variants,
            include_fragments=True,
            include_branch_points=True,
            max_per_fragment=max_per_fragment,
            bin_path=bin_path,
        )

    elapsed = (time.perf_counter() - t0) * 1000.0
    return IsaTargetResult(
        mnemonic=mn,
        complexity=complexity,
        stage=stage,
        spvasm=spvasm,
        spv_path=spv_path,
        compiled_ok=bool(bundle.get("direct_ok") and bundle.get("valhall_ok")),
        target_emitted=target_emitted,
        emitted_mnemonics=emitted_mnemonics,
        emitted_variants=sorted(set(emitted_variants)),
        mode_used="both" if raw_res else "spirv",
        equiv_status=equiv_status,
        diverging_vectors=divs,
        benign_fp_diffs=benign,
        divergence_report=div_rep,
        raw_results=raw_res,
        slice_diff_lines=slice_lines,
        elapsed_ms=elapsed,
    )


def build_coverage_report(index: IsaSpecIndex | None = None) -> dict[str, Any]:
    """Compute exact mnemonic, variant, fragment, and branch-point coverage metrics."""
    idx = index or load_isa_spec_index()
    all_mns = set(idx.all_mnemonics())
    spv_mns = set(RECIPES.keys())
    raw_only_mns = sorted(all_mns - spv_mns)

    tier_counts: dict[str, dict[str, int]] = {}
    for tier in ("compute", "graphics", "internal"):
        tier_mns = set(list_mnemonics_by_tier(tier))
        tier_counts[tier] = {
            "total": len(tier_mns),
            "spirv_recipes": len(tier_mns & spv_mns),
            "raw_covered": len(tier_mns & all_mns),
        }

    fragment_mnemonic_counts: dict[str, int] = {fid: 0 for fid in FRAGMENT_REGISTRY}
    total_mutation_cases = 0
    total_branch_points = 0
    for mn in idx.all_mnemonics():
        sp = idx.primary_variant(mn)
        if sp is None:
            continue
        for frag in applicable_fragments(sp):
            fragment_mnemonic_counts[frag.fragment_id] += 1
        cases = generate_field_cases_for_spec(sp, idx)
        total_mutation_cases += len(cases)
        total_branch_points += sum(1 for c in cases if c.is_branch_point)

    return {
        "xml_total_specs": len(idx.variants),
        "xml_unique_mnemonics": len(all_mns),
        "xml_unique_mnemonic_variants": len(idx.unique_variant_keys()),
        "xml_enums": len(idx.enums),
        "spirv_recipe_mnemonics": len(spv_mns),
        "spirv_variant_recipes": len(VARIANT_RECIPES),
        "raw_only_mnemonics": len(raw_only_mns),
        "total_covered_mnemonics": len(all_mns),
        "tiers": tier_counts,
        "fragments_count": len(FRAGMENT_REGISTRY),
        "fragment_mnemonic_counts": fragment_mnemonic_counts,
        "primary_field_mutation_cases": total_mutation_cases,
        "primary_branch_point_cases": total_branch_points,
    }


def run_full_raw_sweep(
    *,
    all_variants: bool = False,
    include_fragments: bool = True,
    include_branch_points: bool = True,
    max_per_fragment: int = 4,
    bin_path: str = "./build/spirv2valhall",
) -> list[RawVerifyResult]:
    """Run a single batched raw Valhall decode + Pass-1 sweep across all 201 mnemonics (or 627 specs)."""
    idx = load_isa_spec_index()
    specs = list(idx.variants) if all_variants else [idx.primary_variant(m) for m in idx.all_mnemonics()]
    encoded = []
    for sp in specs:
        if sp is None:
            continue
        cases = generate_field_cases_for_spec(
            sp,
            idx,
            include_baseline=True,
            include_fragments=include_fragments,
            include_branch_points=include_branch_points,
            max_per_fragment=max_per_fragment,
        )
        for c in cases:
            encoded.append(encode_mutation_case(c, sp, idx))
    return verify_raw_valhall_batch(encoded, bin_path=bin_path)

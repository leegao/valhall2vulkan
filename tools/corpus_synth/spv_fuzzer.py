"""
SPIR-V Equivalence Fuzzer Integration (`spirv-fuzz`) for `tools.corpus_synth`.

Generates semantically equivalent fuzzed alternate SPIR-V shaders from a seed
`.spv` using `third_party/prebuilts/spirv-tools/spirv-fuzz` (when available),
validates each mutant via `spirv-val`, compiles through `spirv2valhall`, and
optionally validates equivalence via `equiv_fuzzer`.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import tempfile
import time

from tools.coverage.equiv_fuzzer import run_concolic_equiv_fuzz
from tools.coverage.shrinker import DiffPolicy, DivergenceReport
from tools.symbolic_ssa.passes import canonicalize_ssa_module
from tools.equivalence.spvasm_io import assemble_spvasm, disassemble_spv, find_spirv_tool, load_equiv_bundle, validate_spv


@dataclasses.dataclass
class FuzzedShaderResult:
    """Outcome of fuzzing a seed SPIR-V binary with `spirv-fuzz`."""

    seed_spv: str
    fuzzed_spv: str
    fuzz_seed: int
    fuzz_ok: bool
    valid_spv: bool
    compiled_ok: bool
    equiv_status: str = ""
    diverging_vectors: int = 0
    divergence_report: DivergenceReport | None = None
    elapsed_ms: float = 0.0
    error: str = ""


def fuzz_spv_with_spirv_fuzz(
    seed_spv: str,
    *,
    num_variants: int = 2,
    base_seed: int = 0xC0DE,
    out_dir: str | None = None,
    bin_path: str = "./build/spirv2valhall",
    verify_equiv: bool = False,
    k_init: int = 16,
    fp_range: str = "vulkan",
    timeout_s: float = 15.0,
) -> list[FuzzedShaderResult]:
    """Run `spirv-fuzz` `num_variants` times on `seed_spv` and compile/verify each mutant."""
    fuzz_bin = find_spirv_tool("spirv-fuzz")
    if not fuzz_bin:
        return [
            FuzzedShaderResult(
                seed_spv=seed_spv,
                fuzzed_spv="",
                fuzz_seed=base_seed,
                fuzz_ok=False,
                valid_spv=False,
                compiled_ok=False,
                error="spirv-fuzz not found in third_party/prebuilts/spirv-tools or PATH",
            )
        ]

    work_dir = out_dir or tempfile.mkdtemp(prefix="valhall_spvfuzz_")
    os.makedirs(work_dir, exist_ok=True)
    donors_path = os.path.join(work_dir, "empty_donors.txt")
    with open(donors_path, "w", encoding="utf-8") as fh:
        fh.write("")

    # spirv-fuzz validates initial binary against SPIR-V 1.3; re-assemble seed as spv1.3 if needed.
    eff_seed_spv = os.path.abspath(seed_spv)
    try:
        with open(eff_seed_spv, "rb") as sf:
            hdr = sf.read(8)
        if len(hdr) == 8 and int.from_bytes(hdr[4:8], "little") > 0x00010300:
            asm_txt = disassemble_spv(eff_seed_spv)
            tmp_asm = os.path.join(work_dir, "seed_spv13.spvasm")
            tmp_spv = os.path.join(work_dir, "seed_spv13.spv")
            with open(tmp_asm, "w", encoding="utf-8") as af:
                af.write(asm_txt)
            assemble_spvasm(tmp_asm, tmp_spv, target_env="spv1.3")
            eff_seed_spv = tmp_spv
    except (RuntimeError, OSError):
        pass

    stem = os.path.splitext(os.path.basename(seed_spv))[0]
    results: list[FuzzedShaderResult] = []

    for i in range(num_variants):
        t0 = time.perf_counter()
        fseed = base_seed + i * 7919
        out_spv = os.path.join(work_dir, f"{stem}_fuzz_{i}.spv")
        cmd = [
            fuzz_bin,
            eff_seed_spv,
            "-o",
            os.path.abspath(out_spv),
            f"--donors={donors_path}",
            f"--seed={fseed}",
        ]
        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            results.append(
                FuzzedShaderResult(
                    seed_spv=seed_spv,
                    fuzzed_spv=out_spv,
                    fuzz_seed=fseed,
                    fuzz_ok=False,
                    valid_spv=False,
                    compiled_ok=False,
                    elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                    error=f"spirv-fuzz timed out after {timeout_s:.1f}s",
                )
            )
            continue

        if proc.returncode != 0 or not os.path.isfile(out_spv) or os.path.getsize(out_spv) == 0:
            results.append(
                FuzzedShaderResult(
                    seed_spv=seed_spv,
                    fuzzed_spv=out_spv,
                    fuzz_seed=fseed,
                    fuzz_ok=False,
                    valid_spv=False,
                    compiled_ok=False,
                    elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                    error=(proc.stderr or proc.stdout or f"rc={proc.returncode}").strip()[:200],
                )
            )
            continue

        is_valid = validate_spv(out_spv)
        compiled_ok = False
        eq_status = ""
        divs = 0
        div_rep: DivergenceReport | None = None
        err = ""

        if is_valid and os.path.exists(bin_path):
            try:
                bundle = load_equiv_bundle(out_spv, bin_path, defau=True, opt_ssa=True)
                compiled_ok = bool(bundle.get("direct_ok") and bundle.get("valhall_ok"))
                if verify_equiv and compiled_ok:
                    d_ssa, _ = canonicalize_ssa_module(bundle["direct_ssa"])
                    v_ssa, _ = canonicalize_ssa_module(bundle["valhall_ssa"])
                    ef = run_concolic_equiv_fuzz(
                        bundle,
                        d_ssa,
                        v_ssa,
                        k_init=k_init,
                        fp_range=fp_range,
                        seed=fseed,
                        policy=DiffPolicy(),
                    )
                    eq_status = ef.status
                    divs = ef.diverging_vectors
                    div_rep = ef.first_divergence
            except (RuntimeError, OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
                err = str(exc)[:200]

        results.append(
            FuzzedShaderResult(
                seed_spv=seed_spv,
                fuzzed_spv=out_spv,
                fuzz_seed=fseed,
                fuzz_ok=True,
                valid_spv=is_valid,
                compiled_ok=compiled_ok,
                equiv_status=eq_status,
                diverging_vectors=divs,
                divergence_report=div_rep,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                error=err,
            )
        )
    return results

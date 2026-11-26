"""Cross-compiler equivalence fuzzer: Mesa Panfrost Valhall ISA -> SSA vs. Mali DDK Valhall ISA -> SSA.

Compiles the same Vulkan GLCompute SPIR-V shader through both:
  1. Mesa Panfrost (`spirv2valhall --panfrost --dump-equiv-json`) -> Valhall ISA -> SSA
  2. Arm Mali DDK (`spirv2valhall --dump-equiv-json`) -> Valhall ISA -> SSA
and runs concolic equivalence fuzzing (`run_concolic_equiv_fuzz`) directly between the two
ISA-lifted SSA modules in a shared ShaderEnv, plus 3-way triangle attribution against Direct
SPIR-V -> SSA whenever a divergence is detected.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
import os
import subprocess
import sys
import tempfile
from typing import Any

from tools.coverage.equiv_fuzzer import run_concolic_equiv_fuzz
from tools.symbolic_ssa.passes import canonicalize_ssa_module


@dataclass
class PanfrostVsDdkResult:
    path: str
    status: str  # EQUIV_EXACT, EQUIV_WITHIN_ULP, DIVERGED, SKIP, PANFROST_UNSUPPORTED, DDK_FAILED
    error: str = ""
    divs: int = 0
    vecs: int = 0
    benign_fp: int = 0
    triangle: str = ""  # MATCH, PANFROST_DIFF, DDK_DIFF, BOTH_DIFF
    direct_vs_panfrost: str = ""
    direct_vs_ddk: str = ""
    panfrost_insts: int = 0
    ddk_insts: int = 0
    panfrost_opcodes: dict[str, int] = field(default_factory=dict)
    ddk_opcodes: dict[str, int] = field(default_factory=dict)
    first_witness: dict[str, Any] | None = None
    panfrost_disasm: list[str] = field(default_factory=list)
    ddk_disasm: list[str] = field(default_factory=list)


def _extract_opcode_hist(pc_disasm: dict[str, str]) -> tuple[int, dict[str, int], list[str]]:
    hist: Counter[str] = Counter()
    lines: list[str] = []
    for pc_s in sorted(pc_disasm.keys(), key=lambda x: int(x)):
        dis = pc_disasm[pc_s].strip()
        lines.append(f"0x{int(pc_s):04x}: {dis}")
        if dis:
            op = dis.split()[0].split(".")[0]
            hist[op] += 1
    return len(lines), dict(hist), lines


def build_panfrost_vs_ddk_bundle(
    spv_path: str,
    spirv2valhall_bin: str = "build/spirv2valhall",
    arch: int = 10,
    timeout_s: float = 20.0,
) -> tuple[dict[str, Any] | None, str | None, str]:
    """Compile `spv_path` with both Mali DDK and Mesa Panfrost and return a unified bundle.

    Returns (bundle_or_none, error_message_or_none, failure_kind).
    """
    with tempfile.TemporaryDirectory(prefix="pan_vs_ddk_") as td:
        ddk_json = os.path.join(td, "ddk.json")
        pan_json = os.path.join(td, "pan.json")

        try:
            p_ddk = subprocess.run(
                [spirv2valhall_bin, "--quiet", "--dump-equiv-json", ddk_json, spv_path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return None, "DDK compilation timed out", "DDK_FAILED"
        if p_ddk.returncode != 0 or not os.path.exists(ddk_json):
            err = (p_ddk.stderr or p_ddk.stdout).strip().splitlines()[:2]
            return None, "DDK compile failed: " + " | ".join(err), "DDK_FAILED"

        try:
            p_pan = subprocess.run(
                [
                    spirv2valhall_bin,
                    "--panfrost",
                    "--arch",
                    str(arch),
                    "--quiet",
                    "--dump-equiv-json",
                    pan_json,
                    spv_path,
                ],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return None, "Panfrost compilation timed out", "PANFROST_UNSUPPORTED"
        if p_pan.returncode != 0 or not os.path.exists(pan_json):
            err = (p_pan.stderr or p_pan.stdout).strip().splitlines()[:2]
            return None, "Panfrost compile failed: " + " | ".join(err), "PANFROST_UNSUPPORTED"

        with open(ddk_json) as f:
            ddk = json.load(f)
        with open(pan_json) as f:
            pan = json.load(f)

        if not ddk.get("valhall_ok") or not pan.get("valhall_ok"):
            return None, "ISA->SSA lowering failed on one side", "LOWERING_FAILED"

        bundle: dict[str, Any] = {
            "stage": ddk.get("stage", "Compute"),
            "arch": ddk.get("arch", arch),
            "local_size": ddk.get("local_size", pan.get("local_size", [1, 1, 1])),
            "direct_ssa": pan["valhall_ssa"],
            "valhall_ssa": ddk["valhall_ssa"],
            "orig_direct_ssa": ddk["direct_ssa"],
            "mbs2_context": ddk.get("mbs2_context", {}),
            "panfrost_pc_disasm": pan.get("pc_disasm", {}),
            "panfrost_arg_bindings": pan.get("panfrost_arg_bindings", []),
        }
        return bundle, None, ""


def run_panfrost_vs_ddk_equiv(
    spv_path: str,
    k_init: int = 64,
    fp_range: str = "vulkan",
    canonicalize: bool = True,
    arch: int = 10,
    spirv2valhall_bin: str = "build/spirv2valhall",
    time_budget_ms: int = 3000,
    compute_triangle: bool = True,
) -> PanfrostVsDdkResult:
    """Run equivalence fuzzing comparing Panfrost ISA->SSA against Mali DDK ISA->SSA."""
    bundle, err, fail_kind = build_panfrost_vs_ddk_bundle(
        spv_path, spirv2valhall_bin=spirv2valhall_bin, arch=arch
    )
    if bundle is None:
        return PanfrostVsDdkResult(path=spv_path, status=fail_kind or "ERROR", error=err or "")

    pan_insts, pan_ops, pan_lines = _extract_opcode_hist(bundle.get("panfrost_pc_disasm", {}))
    ddk_pc = bundle.get("mbs2_context", {}).get("pc_disasm", {})
    ddk_insts, ddk_ops, ddk_lines = _extract_opcode_hist(ddk_pc)

    pan_ssa = bundle["direct_ssa"]
    ddk_ssa = bundle["valhall_ssa"]
    orig_direct = bundle["orig_direct_ssa"]
    if canonicalize:
        pan_ssa, _ = canonicalize_ssa_module(pan_ssa)
        ddk_ssa, _ = canonicalize_ssa_module(ddk_ssa)
        orig_direct, _ = canonicalize_ssa_module(orig_direct)

    res = run_concolic_equiv_fuzz(
        bundle,
        pan_ssa,
        ddk_ssa,
        seed_side="direct",
        k_init=k_init,
        time_budget_ms=time_budget_ms,
        fp_mode="real",
        fp_range=fp_range,
        auto_shrink=False,
    )

    first_wit = None
    if res.first_divergence is not None:
        d = res.first_divergence
        first_wit = {
            "vec_idx": getattr(d, "vec_idx", None),
            "output": getattr(d, "output_key", getattr(d, "channel", None)),
            "panfrost_val": getattr(d, "direct_val", None),
            "ddk_val": getattr(d, "valhall_val", None),
            "ulp": getattr(d, "ulp", None),
            "kind": getattr(d, "kind", None),
        }

    tri = "MATCH"
    d_vs_p = ""
    d_vs_d = ""
    if res.status == "DIVERGED" and compute_triangle:
        rp = run_concolic_equiv_fuzz(
            bundle, orig_direct, pan_ssa, k_init=min(k_init, 32), time_budget_ms=1500,
            fp_mode="real", fp_range=fp_range, auto_shrink=False,
        )
        rd = run_concolic_equiv_fuzz(
            bundle, orig_direct, ddk_ssa, k_init=min(k_init, 32), time_budget_ms=1500,
            fp_mode="real", fp_range=fp_range, auto_shrink=False,
        )
        d_vs_p = rp.status
        d_vs_d = rd.status
        p_ok = d_vs_p in ("EQUIV_EXACT", "EQUIV_WITHIN_ULP")
        d_ok = d_vs_d in ("EQUIV_EXACT", "EQUIV_WITHIN_ULP")
        if d_ok and not p_ok:
            tri = "PANFROST_DIFF"
        elif p_ok and not d_ok:
            tri = "DDK_DIFF"
        elif not p_ok and not d_ok:
            tri = "BOTH_DIFF"
        else:
            tri = "ULP_BOUNDARY"

    return PanfrostVsDdkResult(
        path=spv_path,
        status=res.status,
        divs=res.diverging_vectors,
        vecs=res.vectors_tested,
        benign_fp=res.benign_fp_diffs,
        triangle=tri,
        direct_vs_panfrost=d_vs_p,
        direct_vs_ddk=d_vs_d,
        panfrost_insts=pan_insts,
        ddk_insts=ddk_insts,
        panfrost_opcodes=pan_ops,
        ddk_opcodes=ddk_ops,
        first_witness=first_wit,
        panfrost_disasm=pan_lines[:40],
        ddk_disasm=ddk_lines[:40],
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Equivalence fuzzing between Mesa Panfrost Valhall ISA->SSA and Mali DDK Valhall ISA->SSA."
    )
    ap.add_argument("shaders", nargs="+", help="One or more Vulkan GLCompute .spv paths")
    ap.add_argument("--bin", default="build/spirv2valhall", help="Path to spirv2valhall binary")
    ap.add_argument("--arch", type=int, default=10, help="Valhall architecture version (default: 10)")
    ap.add_argument("--k-init", type=int, default=64, help="Concrete input vectors per shader")
    ap.add_argument("--fp-range", choices=("vulkan", "unconstrained"), default="vulkan")
    ap.add_argument("--no-canonicalize", action="store_true")
    ap.add_argument("--json-out", default="", help="Optional path to write full JSON results")
    args = ap.parse_args(argv)

    all_res = []
    any_div = False
    for spv in args.shaders:
        r = run_panfrost_vs_ddk_equiv(
            spv,
            k_init=args.k_init,
            fp_range=args.fp_range,
            canonicalize=not args.no_canonicalize,
            arch=args.arch,
            spirv2valhall_bin=args.bin,
        )
        all_res.append(r.__dict__)
        tag = os.path.basename(spv)
        print(
            f"[PANFROST-VS-DDK] {tag:36s} status={r.status:20s} "
            f"vecs={r.vecs:3d} divs={r.divs:3d} tri={r.triangle:13s} "
            f"insts=(pan={r.panfrost_insts},ddk={r.ddk_insts})"
            + (f" err={r.error[:60]}" if r.error else "")
        )
        if r.status == "DIVERGED":
            any_div = True

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(all_res, f, indent=2)
    return 1 if any_div else 0


if __name__ == "__main__":
    sys.exit(main())

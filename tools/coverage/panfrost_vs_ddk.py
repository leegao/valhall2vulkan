"""Cross-compiler equivalence fuzzer: Mesa Panfrost Valhall ISA -> SSA vs. Mali DDK Valhall ISA -> SSA.

Compiles the same Vulkan SPIR-V shader (Vertex, Fragment, or Compute) through both:
  1. Mesa Panfrost (`spirv2valhall --panfrost --dump-equiv-json`) -> Valhall ISA -> SSA
     (using standalone `panfrost_vk_compile` with Rust Kraid or C Bifrost backend, or legacy `--panfrost-cl`)
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
from tools.coverage.reachability import solve_maximal_reachability
from tools.coverage.recorder import record_repro_spvasm
from tools.coverage.shrinker import DiffPolicy, DivergenceReport
from tools.coverage.slice_diff import format_slice_diff
from tools.coverage.triage import format_divergence_and_triage, triage_divergence
from tools.equivalence.spvasm_io import ensure_spv_path
from tools.symbolic_ssa.passes import canonicalize_ssa_module


@dataclass
class PanfrostVsDdkResult:
    path: str
    status: str
    stage: str = "Compute"  # EQUIV_EXACT, EQUIV_WITHIN_ULP, DIVERGED, SKIP, PANFROST_UNSUPPORTED, DDK_FAILED
    panfrost_backend: str = ""  # "kraid", "bifrost", or "opencl"
    panfrost_idvs: bool = True
    panfrost_merge_workgroups: bool = False
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
    arch: int = 11,
    timeout_s: float = 20.0,
    panfrost_flag: str = "--panfrost",
    panfrost_backend: str = "auto",
    panfrost_robust: bool = False,
    panfrost_no_idvs: bool = False,
    defau: bool = True,
    opt_ssa: bool = True,
) -> tuple[dict[str, Any] | None, str | None, str]:
    """Compile `spv_path` with both Mali DDK and Mesa Panfrost and return a unified bundle.

    Returns (bundle_or_none, error_message_or_none, failure_kind).
    """
    with ensure_spv_path(spv_path) as resolved_spv, tempfile.TemporaryDirectory(prefix="pan_vs_ddk_") as td:
        ddk_json = os.path.join(td, "ddk.json")
        pan_json = os.path.join(td, "pan.json")

        ddk_cmd = [spirv2valhall_bin, "--quiet", "--dump-equiv-json", ddk_json]
        if defau:
            ddk_cmd.append("--defau")
        if opt_ssa:
            ddk_cmd.append("--opt-ssa")
        ddk_cmd.append(resolved_spv)

        try:
            p_ddk = subprocess.run(
                ddk_cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return None, "DDK compilation timed out", "DDK_FAILED"
        if p_ddk.returncode != 0 or not os.path.exists(ddk_json):
            err = (p_ddk.stderr or p_ddk.stdout).strip().splitlines()[:2]
            return None, "DDK compile failed: " + " | ".join(err), "DDK_FAILED"

        pan_cmd = [
            spirv2valhall_bin,
            panfrost_flag,
            "--arch",
            str(arch),
            "--quiet",
            "--dump-equiv-json",
            pan_json,
        ]
        if defau:
            pan_cmd.append("--defau")
        if opt_ssa:
            pan_cmd.append("--opt-ssa")
        if panfrost_flag == "--panfrost":
            if panfrost_backend and panfrost_backend != "auto":
                pan_cmd.extend(["--panfrost-backend", panfrost_backend])
            if panfrost_robust:
                pan_cmd.append("--panfrost-robust")
            if panfrost_no_idvs:
                pan_cmd.append("--panfrost-no-idvs")
        pan_cmd.append(resolved_spv)

        try:
            p_pan = subprocess.run(
                pan_cmd,
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

        pan_ssa_raw = pan["valhall_ssa"]
        pan_rodata = pan.get("mbs2_context", {}).get("ebin_rodata")
        if pan_rodata:
            for node in pan_ssa_raw.get("nodes", []):
                if node.get("op") == "LoadSidebandU64" and int(node.get("imm_u32", 0)) == 1:
                    node["aux_str"] = "panfrost_ebin_rodata"

        mbs2_merged = {
            **ddk.get("mbs2_context", {}),
            "srt_pka_buf_names": {
                **pan.get("mbs2_context", {}).get("srt_pka_buf_names", {}),
                **ddk.get("mbs2_context", {}).get("srt_pka_buf_names", {}),
            },
        }
        if pan_rodata:
            mbs2_merged["panfrost_ebin_rodata"] = pan_rodata

        stage_str = ddk.get("stage", "Compute")
        bundle: dict[str, Any] = {
            "stage": stage_str,
            "arch": ddk.get("arch", arch),
            "panfrost_backend": pan.get("panfrost_backend", ""),
            "panfrost_idvs": bool(pan.get("panfrost_idvs", True)),
            "panfrost_merge_workgroups": bool(pan.get("panfrost_merge_workgroups", False)),
            "panfrost_single_sample_frag": (stage_str == "Fragment"),
            "local_size": ddk.get("local_size", pan.get("local_size", [1, 1, 1])),
            "direct_ssa": pan_ssa_raw,
            "valhall_ssa": ddk["valhall_ssa"],
            "orig_direct_ssa": ddk["direct_ssa"],
            "mbs2_context": mbs2_merged,
            "panfrost_pc_disasm": pan.get("pc_disasm", {}),
            "panfrost_arg_bindings": pan.get("panfrost_arg_bindings", []),
        }
        return bundle, None, ""


def _run_panfrost_vs_ddk_equiv_ctx(
    spv_path: str,
    k_init: int = 64,
    fp_range: str = "vulkan",
    canonicalize: bool = True,
    arch: int = 11,
    spirv2valhall_bin: str = "build/spirv2valhall",
    time_budget_ms: int = 3000,
    compute_triangle: bool = True,
    panfrost_flag: str = "--panfrost",
    panfrost_backend: str = "auto",
    panfrost_robust: bool = False,
    panfrost_no_idvs: bool = False,
    *,
    defau: bool = True,
    opt_ssa: bool = True,
    domain: str = "udomain_backfill",
    fp_mode: str = "real",
    max_paths: int = 32,
    max_unroll: int = 1,
    policy: DiffPolicy | None = None,
    auto_shrink: bool = False,
    maximal_reachability_seed: bool = False,
    seed_side: str = "direct",
) -> tuple[
    PanfrostVsDdkResult,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    DivergenceReport | None,
    list[dict[str, Any]],
]:
    """Run equivalence fuzzing comparing Panfrost ISA->SSA against Mali DDK ISA->SSA with full context."""
    bundle, err, fail_kind = build_panfrost_vs_ddk_bundle(
        spv_path,
        spirv2valhall_bin=spirv2valhall_bin,
        arch=arch,
        panfrost_flag=panfrost_flag,
        panfrost_backend=panfrost_backend,
        panfrost_robust=panfrost_robust,
        panfrost_no_idvs=panfrost_no_idvs,
        defau=defau,
        opt_ssa=opt_ssa,
    )
    if bundle is None:
        return (
            PanfrostVsDdkResult(
                path=spv_path,
                status=fail_kind or "ERROR",
                panfrost_backend=panfrost_backend if panfrost_flag == "--panfrost" else "opencl",
                error=err or "",
            ),
            None,
            None,
            None,
            None,
            None,
            None,
            [],
        )

    pan_insts, pan_ops, pan_lines = _extract_opcode_hist(bundle.get("panfrost_pc_disasm", {}))
    ddk_pc = bundle.get("mbs2_context", {}).get("pc_disasm", {})
    ddk_insts, ddk_ops, ddk_lines = _extract_opcode_hist(ddk_pc)

    raw_pan_ssa = bundle["direct_ssa"]
    raw_ddk_ssa = bundle["valhall_ssa"]
    pan_ssa = raw_pan_ssa
    ddk_ssa = raw_ddk_ssa
    orig_direct = bundle["orig_direct_ssa"]
    if canonicalize:
        pan_ssa, _ = canonicalize_ssa_module(pan_ssa)
        ddk_ssa, _ = canonicalize_ssa_module(ddk_ssa)
        orig_direct, _ = canonicalize_ssa_module(orig_direct)

    res = run_concolic_equiv_fuzz(
        bundle,
        pan_ssa,
        ddk_ssa,
        seed_side=seed_side,
        k_init=k_init,
        time_budget_ms=time_budget_ms,
        fp_mode=fp_mode,
        max_unroll=max_unroll,
        max_paths=max_paths,
        policy=policy,
        fp_range=fp_range,
        auto_shrink=auto_shrink,
        domain=domain,
        maximal_reachability_seed=maximal_reachability_seed,
    )

    first_wit = None
    if res.first_divergence is not None:
        d = res.first_divergence
        first_wit = {
            "vec_idx": getattr(d, "vec_idx", None),
            "output": getattr(d, "output_key", getattr(d, "output_name", getattr(d, "channel", None))),
            "panfrost_val": getattr(d, "direct_val", None),
            "ddk_val": getattr(d, "valhall_val", None),
            "ulp": getattr(d, "ulp", getattr(d, "ulp_diff", None)),
            "kind": getattr(d, "kind", getattr(d, "diff_kind", None)),
        }

    tri = "MATCH"
    d_vs_p = ""
    d_vs_d = ""
    if res.status == "DIVERGED" and compute_triangle:
        rp = run_concolic_equiv_fuzz(
            bundle, orig_direct, pan_ssa, k_init=min(k_init, 32), time_budget_ms=1500,
            fp_mode=fp_mode, fp_range=fp_range, policy=policy, auto_shrink=False, domain=domain,
        )
        rd = run_concolic_equiv_fuzz(
            bundle, orig_direct, ddk_ssa, k_init=min(k_init, 32), time_budget_ms=1500,
            fp_mode=fp_mode, fp_range=fp_range, policy=policy, auto_shrink=False, domain=domain,
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

    result = PanfrostVsDdkResult(
        path=spv_path,
        status=res.status,
        stage=bundle.get("stage", "Compute"),
        panfrost_backend=bundle.get("panfrost_backend", ""),
        panfrost_idvs=bundle.get("panfrost_idvs", True),
        panfrost_merge_workgroups=bundle.get("panfrost_merge_workgroups", False),
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
    return result, bundle, pan_ssa, ddk_ssa, raw_pan_ssa, raw_ddk_ssa, res.first_divergence, res.tested_overrides


def run_panfrost_vs_ddk_equiv(
    spv_path: str,
    k_init: int = 64,
    fp_range: str = "vulkan",
    canonicalize: bool = True,
    arch: int = 11,
    spirv2valhall_bin: str = "build/spirv2valhall",
    time_budget_ms: int = 3000,
    compute_triangle: bool = True,
    panfrost_flag: str = "--panfrost",
    panfrost_backend: str = "auto",
    panfrost_robust: bool = False,
    panfrost_no_idvs: bool = False,
    *,
    defau: bool = True,
    opt_ssa: bool = True,
    domain: str = "udomain_backfill",
    fp_mode: str = "real",
    max_paths: int = 32,
    max_unroll: int = 1,
    policy: DiffPolicy | None = None,
    auto_shrink: bool = False,
    maximal_reachability_seed: bool = False,
    seed_side: str = "direct",
) -> PanfrostVsDdkResult:
    """Run equivalence fuzzing comparing Panfrost ISA->SSA against Mali DDK ISA->SSA."""
    result, *_ = _run_panfrost_vs_ddk_equiv_ctx(
        spv_path,
        k_init=k_init,
        fp_range=fp_range,
        canonicalize=canonicalize,
        arch=arch,
        spirv2valhall_bin=spirv2valhall_bin,
        time_budget_ms=time_budget_ms,
        compute_triangle=compute_triangle,
        panfrost_flag=panfrost_flag,
        panfrost_backend=panfrost_backend,
        panfrost_robust=panfrost_robust,
        panfrost_no_idvs=panfrost_no_idvs,
        defau=defau,
        opt_ssa=opt_ssa,
        domain=domain,
        fp_mode=fp_mode,
        max_paths=max_paths,
        max_unroll=max_unroll,
        policy=policy,
        auto_shrink=auto_shrink,
        maximal_reachability_seed=maximal_reachability_seed,
        seed_side=seed_side,
    )
    return result


def _handle_divergence(
    args: argparse.Namespace,
    spv_path: str,
    bundle: dict[str, Any],
    pan_ssa: dict[str, Any],
    ddk_ssa: dict[str, Any],
    raw_pan_ssa: dict[str, Any],
    raw_ddk_ssa: dict[str, Any],
    rep: DivergenceReport,
    policy: DiffPolicy,
    panfrost_cfg: dict[str, Any],
) -> None:
    triage = triage_divergence(
        bundle,
        pan_ssa,
        ddk_ssa,
        rep,
        raw_direct_ssa=raw_pan_ssa if args.canonicalize else None,
        raw_valhall_ssa=raw_ddk_ssa if args.canonicalize else None,
        policy=policy,
        fp_range=args.fp_range,
    )
    for line in format_divergence_and_triage(rep, triage, trace_diff=args.trace_diff):
        print(line)
    if getattr(args, "slice_diff", 0) > 0:
        for line in format_slice_diff(bundle, pan_ssa, ddk_ssa, rep, depth=args.slice_diff, fp_range=args.fp_range):
            print(line)
    if args.shrink and not args.record:
        _out = os.path.join(tempfile.mkdtemp(prefix="pan_ddk_shrink_"), "shrunk.spvasm")
        rec = record_repro_spvasm(
            spv_path,
            _out,
            args.bin,
            bundle,
            rep,
            triage,
            canonicalize=args.canonicalize,
            policy=policy,
            reduce_timeout_s=args.record_timeout_s,
            reduce_steps=args.record_steps,
            no_reduce=args.no_reduce,
            defau=args.defau,
            opt_ssa=args.opt_ssa,
            fp_range=args.fp_range,
            panfrost_cfg=panfrost_cfg,
        )
        print(
            f"  [SHRINK/spirv-reduce] path={rec.out_path} bytes={rec.original_bytes}->{rec.recorded_bytes} "
            f"method={rec.reduce_method} verified={rec.verified_reproduces}"
        )
    if args.record:
        rec = record_repro_spvasm(
            spv_path,
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
            defau=args.defau,
            opt_ssa=args.opt_ssa,
            fp_range=args.fp_range,
            panfrost_cfg=panfrost_cfg,
        )
        print(
            f"  [RECORD] path={rec.out_path} bytes={rec.original_bytes}->{rec.recorded_bytes} "
            f"method={rec.reduce_method} verified={rec.verified_reproduces}"
        )


def _normalize_side(side: str) -> str:
    return "valhall" if side in ("valhall", "ddk") else "direct"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Equivalence fuzzing between Mesa Panfrost Valhall ISA->SSA and Mali DDK Valhall ISA->SSA."
    )
    ap.add_argument("shaders", nargs="+", help="One or more Vulkan .spv/.spvasm paths (VS/FS/CS)")
    ap.add_argument("--bin", default="build/spirv2valhall", help="Path to spirv2valhall binary")
    ap.add_argument("--arch", type=int, default=11, help="Valhall architecture version (default: 11)")
    ap.add_argument("--side", choices=("direct", "valhall", "panfrost", "ddk"), default="direct",
                    help="SSA side to target/seed from (direct/panfrost vs valhall/ddk)")
    ap.add_argument("--k-init", type=int, default=64, help="Concrete input vectors per shader")
    ap.add_argument("--max-rounds", type=int, default=12, help="Maximum synthesis rounds")
    ap.add_argument("--domain", choices=("bitvec", "udomain", "udomain_backfill", "udomain_exact"),
                    default="udomain_backfill", help="Symbolic value domain (default: udomain_backfill)")
    ap.add_argument("--fp-mode", choices=("exact", "real", "uf"), default="real", help="Z3 FP mode")
    ap.add_argument("--fp-range", choices=("vulkan", "unconstrained"), default="vulkan")
    ap.add_argument("--canonicalize", action="store_true", default=True,
                    help="Canonicalize SSA before synthesis (default: on)")
    ap.add_argument("--no-canonicalize", dest="canonicalize", action="store_false",
                    help="Disable SSA canonicalization")
    ap.add_argument("--defau", action="store_true", default=True,
                    help="Run De-FAU devirtualization (default: on)")
    ap.add_argument("--no-defau", dest="defau", action="store_false",
                    help="Disable De-FAU devirtualization")
    ap.add_argument("--opt-ssa", action="store_true", default=True,
                    help="Run C++ ssa-opt canonicalization passes (default: on)")
    ap.add_argument("--no-opt-ssa", dest="opt_ssa", action="store_false",
                    help="Disable C++ ssa-opt passes")
    ap.add_argument("--maximal-reachability", action="store_true",
                    help="Solve and report disjoint reachability conditions exercising all paths")
    ap.add_argument("--maximal-reachability-seed", action="store_true",
                    help="Use maximal reachability to find maximal path vector seeds, then run maximal coverage")
    ap.add_argument("--equiv-fuzz", action="store_true",
                    help="Seed concrete vectors from --side and concretely test Panfrost vs DDK equivalence (default)")
    ap.add_argument("--shrink", action="store_true",
                    help="Shrink diverging SPIR-V inputs via spirv-reduce")
    ap.add_argument("--slice-diff", type=int, default=0, metavar="DEPTH",
                    help="Print N-hop side-by-side concrete backward-slice diff (Panfrost vs DDK) on divergence")
    ap.add_argument("--trace-diff", action="store_true",
                    help="Print concrete DDK frontier node values and disassembly for deeper triage")
    ap.add_argument("--record", default=None,
                    help="Record a shrunken .spvasm divergence reproduction to this file path or directory (e.g. bugs/)")
    ap.add_argument("--no-reduce", action="store_true",
                    help="Skip spirv-reduce shrinking when using --record (disassemble original shader directly)")
    ap.add_argument("--record-timeout-s", type=float, default=15.0,
                    help="Max wall-clock seconds for spirv-reduce during --record (default: 15.0)")
    ap.add_argument("--record-steps", type=int, default=120,
                    help="Max spirv-reduce step limit during --record (default: 120)")
    ap.add_argument("--max-ulp", type=int, default=2, help="Max benign FP32 ULP tolerance for DiffPolicy")
    ap.add_argument("--max-paths", type=int, default=32, help="Maximum disjoint paths to enumerate")
    ap.add_argument("--timeout-ms", type=int, default=500, help="Per-solve Z3 timeout in ms")
    ap.add_argument("--time-budget-ms", type=int, default=3000, help="Total time budget in ms for synthesis/fuzzing")
    ap.add_argument("--max-unroll", type=int, default=1, help="Loop unroll depth (adaptive up to 3 in coverage mode)")
    ap.add_argument("--dump-vectors", default=None, help="Write synthesized override vectors or disjoint paths to JSON")
    ap.add_argument("--panfrost-cl", action="store_true", help="Use OpenCL --panfrost-cl backend instead of native Vulkan --panfrost")
    ap.add_argument("--panfrost-backend", choices=("auto", "kraid", "bifrost"), default="auto",
                    help="Backend for panfrost_vk_compile (auto, kraid, or bifrost)")
    ap.add_argument("--panfrost-robust", action="store_true", help="Enable --robust in panfrost_vk_compile")
    ap.add_argument("--panfrost-no-idvs", action="store_true", help="Pass --no-idvs to panfrost_vk_compile")
    ap.add_argument("--no-triangle", action="store_true", help="Skip 3-way Direct attribution on divergence")
    ap.add_argument("--json-out", default="", help="Optional path to write full JSON results")
    args = ap.parse_args(argv)

    panfrost_flag = "--panfrost-cl" if args.panfrost_cl else "--panfrost"
    panfrost_cfg = {
        "arch": args.arch,
        "panfrost_flag": panfrost_flag,
        "panfrost_backend": args.panfrost_backend,
        "panfrost_robust": args.panfrost_robust,
        "panfrost_no_idvs": args.panfrost_no_idvs,
    }
    seed_side = _normalize_side(args.side)
    is_valhall_side = seed_side == "valhall"
    policy = DiffPolicy(max_ulp=args.max_ulp)

    if args.maximal_reachability:
        all_paths_payload = []
        for spv in args.shaders:
            bundle, err, fail_kind = build_panfrost_vs_ddk_bundle(
                spv,
                spirv2valhall_bin=args.bin,
                arch=args.arch,
                panfrost_flag=panfrost_flag,
                panfrost_backend=args.panfrost_backend,
                panfrost_robust=args.panfrost_robust,
                panfrost_no_idvs=args.panfrost_no_idvs,
                defau=args.defau,
                opt_ssa=args.opt_ssa,
            )
            if bundle is None:
                print(f"[ERROR] {spv}: {fail_kind}: {err}", file=sys.stderr)
                return 1
            target_ssa = bundle["valhall_ssa"] if is_valhall_side else bundle["direct_ssa"]
            if args.canonicalize:
                target_ssa, _ = canonicalize_ssa_module(target_ssa)
            mres = solve_maximal_reachability(
                bundle,
                target_ssa,
                is_valhall_side=is_valhall_side,
                fp_mode=args.fp_mode,
                domain=args.domain,
                max_paths=args.max_paths,
                timeout_ms=args.timeout_ms,
                max_unroll=args.max_unroll,
                total_timeout_ms=max(args.time_budget_ms, args.timeout_ms),
            )
            print(
                f"[REACHABILITY] {os.path.basename(spv)} side={args.side} "
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
            all_paths_payload.extend(
                {
                    "shader": spv,
                    "path_id": p.path_id,
                    "visited_blocks": p.visited_blocks,
                    "decisions": p.decisions,
                    "condition_smt": p.condition_smt,
                    "witness_overrides": p.witness_overrides,
                }
                for p in mres.paths
            )
        if args.dump_vectors:
            with open(args.dump_vectors, "w", encoding="utf-8") as fh:
                json.dump(all_paths_payload, fh, indent=2)
        return 0

    need_triage = bool(args.trace_diff or args.slice_diff > 0 or args.shrink or args.record)
    all_res = []
    all_vectors: list[dict[str, Any]] = []
    any_div = False
    for spv in args.shaders:
        r, bundle, pan_ssa, ddk_ssa, raw_pan_ssa, raw_ddk_ssa, first_div, tested_ovs = (
            _run_panfrost_vs_ddk_equiv_ctx(
                spv,
                k_init=args.k_init,
                fp_range=args.fp_range,
                canonicalize=args.canonicalize,
                arch=args.arch,
                spirv2valhall_bin=args.bin,
                time_budget_ms=args.time_budget_ms,
                compute_triangle=not args.no_triangle,
                panfrost_flag=panfrost_flag,
                panfrost_backend=args.panfrost_backend,
                panfrost_robust=args.panfrost_robust,
                panfrost_no_idvs=args.panfrost_no_idvs,
                defau=args.defau,
                opt_ssa=args.opt_ssa,
                domain=args.domain,
                fp_mode=args.fp_mode,
                max_paths=args.max_paths,
                max_unroll=args.max_unroll,
                policy=policy,
                auto_shrink=need_triage,
                maximal_reachability_seed=args.maximal_reachability_seed,
                seed_side=seed_side,
            )
        )
        all_res.append(r.__dict__)
        all_vectors.extend(tested_ovs)
        tag = os.path.basename(spv)
        print(
            f"[PANFROST-VS-DDK] {tag:36s} ({r.panfrost_backend or '?':7s}) status={r.status:20s} "
            f"vecs={r.vecs:3d} divs={r.divs:3d} tri={r.triangle:13s} "
            f"insts=(pan={r.panfrost_insts},ddk={r.ddk_insts})"
            + (f" err={r.error[:60]}" if r.error else "")
        )
        if r.status == "DIVERGED":
            any_div = True
            if (
                need_triage
                and first_div is not None
                and first_div.diverged
                and bundle is not None
                and pan_ssa is not None
                and ddk_ssa is not None
                and raw_pan_ssa is not None
                and raw_ddk_ssa is not None
            ):
                _handle_divergence(
                    args,
                    spv,
                    bundle,
                    pan_ssa,
                    ddk_ssa,
                    raw_pan_ssa,
                    raw_ddk_ssa,
                    first_div,
                    policy,
                    panfrost_cfg,
                )

    if args.dump_vectors:
        with open(args.dump_vectors, "w", encoding="utf-8") as fh:
            json.dump(all_vectors, fh, indent=2)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(all_res, f, indent=2)
    return 1 if any_div else 0


if __name__ == "__main__":
    sys.exit(main())

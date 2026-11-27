"""Shrunken .spvasm reproduction recorder for tools.coverage (--record)."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass

from tools.coverage.shrinker import (
    DiffPolicy,
    DivergenceReport,
    compare_concrete_outputs,
)
from tools.coverage.triage import TriageReport
from tools.symbolic_ssa.passes import canonicalize_ssa_module
from tools.equivalence.spvasm_io import (
    disassemble_spv,
    ensure_spv_path,
    find_spirv_tool,
    load_equiv_bundle,
    validate_spv,
)


@dataclass
class RecordResult:
    """Outcome of recording (and optionally shrinking) a .spvasm bug reproduction."""

    out_path: str
    original_bytes: int
    recorded_bytes: int
    reduced: bool
    reduce_method: str
    verified_reproduces: bool


def _sanitize_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return slug[:40] if slug else "div"


def _resolve_output_path(dest: str, src_path: str, output_name: str) -> str:
    if dest.endswith(os.sep) or os.path.isdir(dest) or not dest.lower().endswith(".spvasm"):
        os.makedirs(dest, exist_ok=True)
        stem = os.path.splitext(os.path.basename(src_path))[0]
        if len(stem) > 24 and re.fullmatch(r"[0-9a-f]{32,}", stem):
            stem = stem[:12]
        out_slug = _sanitize_slug(output_name.split("+")[0])
        return os.path.join(dest, f"{stem}_{out_slug}.spvasm")
    parent = os.path.dirname(os.path.abspath(dest))
    if parent:
        os.makedirs(parent, exist_ok=True)
    return dest


def _load_bundle_for_record(
    spv_path: str,
    bin_path: str,
    *,
    timeout: int = 15,
    defau: bool = True,
    opt_ssa: bool = True,
    panfrost_cfg: dict | None = None,
) -> dict:
    if panfrost_cfg is not None:
        from tools.coverage.panfrost_vs_ddk import build_panfrost_vs_ddk_bundle

        bundle, err, _ = build_panfrost_vs_ddk_bundle(
            spv_path,
            spirv2valhall_bin=bin_path,
            timeout_s=float(timeout),
            defau=defau,
            opt_ssa=opt_ssa,
            **panfrost_cfg,
        )
        if bundle is None:
            raise RuntimeError(err or "build_panfrost_vs_ddk_bundle failed")
        return bundle
    return load_equiv_bundle(spv_path, bin_path, timeout=timeout, defau=defau, opt_ssa=opt_ssa)


def _check_spv_reproduces(
    spv_path: str,
    bin_path: str,
    rep: DivergenceReport,
    canonicalize: bool,
    policy: DiffPolicy,
    *,
    defau: bool = True,
    opt_ssa: bool = True,
    fp_range: str = "unconstrained",
    panfrost_cfg: dict | None = None,
) -> bool:
    try:
        bundle = _load_bundle_for_record(
            spv_path,
            bin_path,
            timeout=15,
            defau=defau,
            opt_ssa=opt_ssa,
            panfrost_cfg=panfrost_cfg,
        )
        d_ssa = bundle["direct_ssa"]
        v_ssa = bundle["valhall_ssa"]
        if canonicalize:
            d_ssa, _ = canonicalize_ssa_module(d_ssa)
            v_ssa, _ = canonicalize_ssa_module(v_ssa)
        div, _, _, _, _, _ = compare_concrete_outputs(
            bundle,
            d_ssa,
            v_ssa,
            overrides=rep.shrunk_overrides,
            vec_idx=rep.vec_idx,
            policy=policy,
            fp_range=fp_range,
        )
        return bool(div)
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.SubprocessError):
        return False


def _try_spirv_reduce(
    src_spv: str,
    out_spv: str,
    bin_path: str,
    rep: DivergenceReport,
    canonicalize: bool,
    policy: DiffPolicy,
    timeout_s: float,
    step_limit: int,
    *,
    defau: bool = True,
    opt_ssa: bool = True,
    fp_range: str = "unconstrained",
    panfrost_cfg: dict | None = None,
) -> bool:
    reduce_bin = find_spirv_tool("spirv-reduce")
    if not reduce_bin:
        return False

    repo_root = os.getcwd()
    py_exe = sys.executable
    py_path = os.environ.get("PYTHONPATH", "")
    if repo_root not in py_path.split(os.pathsep):
        py_path = f"{repo_root}{os.pathsep}{py_path}" if py_path else repo_root

    cfg = {
        "bin_path": os.path.abspath(bin_path),
        "vec_idx": rep.vec_idx,
        "overrides": rep.shrunk_overrides,
        "canonicalize": canonicalize,
        "max_ulp": policy.max_ulp,
        "target_output": rep.output_name,
        "require_both_present": (rep.direct_val != "None" and rep.valhall_val != "None"),
        "defau": defau,
        "opt_ssa": opt_ssa,
        "fp_range": fp_range,
        "panfrost_cfg": panfrost_cfg,
    }

    with tempfile.TemporaryDirectory(prefix="valhall_reduce_") as td:
        cfg_path = os.path.join(td, "cfg.json")
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh)

        best_spv = os.path.join(td, "best.spv")
        shutil.copyfile(src_spv, best_spv)
        pred_path = os.path.join(td, "is_interesting.py")
        with open(pred_path, "w", encoding="utf-8") as fh:
            fh.write(
                f"#!{py_exe}\n"
                "import json, os, shutil, sys\n"
                "from tools.coverage.recorder import _load_bundle_for_record\n"
                "from tools.symbolic_ssa.passes import canonicalize_ssa_module\n"
                "from tools.coverage.shrinker import DiffPolicy, compare_concrete_outputs\n"
                f"with open({cfg_path!r}, 'r', encoding='utf-8') as f: cfg = json.load(f)\n"
                f"best_path = {best_spv!r}\n"
                "cand = sys.argv[1]\n"
                "try:\n"
                "    b = _load_bundle_for_record(\n"
                "        cand, cfg['bin_path'], timeout=10,\n"
                "        defau=cfg.get('defau', True), opt_ssa=cfg.get('opt_ssa', True),\n"
                "        panfrost_cfg=cfg.get('panfrost_cfg'),\n"
                "    )\n"
                "    d, v = b['direct_ssa'], b['valhall_ssa']\n"
                "    if cfg['canonicalize']:\n"
                "        d, _ = canonicalize_ssa_module(d)\n"
                "        v, _ = canonicalize_ssa_module(v)\n"
                "    div, out_name, d_str, v_str, _, _ = compare_concrete_outputs(\n"
                "        b, d, v, overrides=cfg['overrides'], vec_idx=cfg['vec_idx'],\n"
                "        policy=DiffPolicy(max_ulp=cfg['max_ulp']),\n"
                "        fp_range=cfg.get('fp_range', 'unconstrained'),\n"
                "    )\n"
                "    if div and (not cfg.get('require_both_present') or (d_str != 'None' and v_str != 'None')):\n"
                "        if os.path.getsize(cand) < os.path.getsize(best_path):\n"
                "            shutil.copyfile(cand, best_path)\n"
                "        sys.exit(0)\n"
                "    sys.exit(1)\n"
                "except Exception:\n"
                "    sys.exit(1)\n"
            )
        os.chmod(pred_path, os.stat(pred_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        env = os.environ.copy()
        env["PYTHONPATH"] = py_path
        cmd = [
            reduce_bin,
            f"--step-limit={step_limit}",
            os.path.abspath(src_spv),
            "-o",
            os.path.abspath(out_spv),
            "--",
            py_exe,
            pred_path,
        ]
        try:
            subprocess.run(cmd, check=False, cwd=td, env=env, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            # spirv-reduce writes out_spv incrementally when progress is made; verify below.
            pass

        if (not os.path.isfile(out_spv) or os.path.getsize(out_spv) == 0) and os.path.isfile(best_spv) or os.path.isfile(best_spv) and os.path.getsize(best_spv) < os.path.getsize(out_spv):
            shutil.copyfile(best_spv, out_spv)
        if not os.path.isfile(out_spv) or os.path.getsize(out_spv) == 0:
            return False
        if os.path.getsize(out_spv) >= os.path.getsize(src_spv):
            return False
        if not validate_spv(out_spv):
            return False
        return _check_spv_reproduces(
            out_spv,
            bin_path,
            rep,
            canonicalize,
            policy,
            defau=defau,
            opt_ssa=opt_ssa,
            fp_range=fp_range,
            panfrost_cfg=panfrost_cfg,
        )


def record_repro_spvasm(
    src_path: str,
    dest: str,
    bin_path: str,
    bundle: dict,
    rep: DivergenceReport,
    triage: TriageReport | None = None,
    *,
    canonicalize: bool = True,
    policy: DiffPolicy | None = None,
    reduce_timeout_s: float = 15.0,
    reduce_steps: int = 120,
    no_reduce: bool = False,
    defau: bool = True,
    opt_ssa: bool = True,
    fp_range: str = "unconstrained",
    panfrost_cfg: dict | None = None,
) -> RecordResult:
    """
    Record a self-contained, optionally spirv-reduce-shrunken .spvasm reproduction
    file with structured metadata header comments.
    """
    if src_path.endswith(".json"):
        raise RuntimeError("--record requires a .spv or .spvasm input shader (not a .json bundle)")

    eff_policy = policy or DiffPolicy()
    out_path = _resolve_output_path(dest, src_path, rep.output_name)

    with ensure_spv_path(src_path) as base_spv:
        orig_bytes = os.path.getsize(base_spv)
        reduced = False
        reduce_method = "original"

        fd, tmp_reduced = tempfile.mkstemp(suffix=".spv")
        os.close(fd)
        try:
            chosen_spv = base_spv
            if not no_reduce and find_spirv_tool("spirv-reduce"):
                ok = _try_spirv_reduce(
                    base_spv,
                    tmp_reduced,
                    bin_path,
                    rep,
                    canonicalize,
                    eff_policy,
                    timeout_s=reduce_timeout_s,
                    step_limit=reduce_steps,
                    defau=defau,
                    opt_ssa=opt_ssa,
                    fp_range=fp_range,
                    panfrost_cfg=panfrost_cfg,
                )
                if ok:
                    chosen_spv = tmp_reduced
                    reduced = True
                    reduce_method = "spirv-reduce"

            rec_bytes = os.path.getsize(chosen_spv)
            asm_body = disassemble_spv(chosen_spv)
            verified = _check_spv_reproduces(
                chosen_spv,
                bin_path,
                rep,
                canonicalize,
                eff_policy,
                defau=defau,
                opt_ssa=opt_ssa,
                fp_range=fp_range,
                panfrost_cfg=panfrost_cfg,
            )
        finally:
            if os.path.exists(tmp_reduced):
                os.remove(tmp_reduced)

    # Build a valid SPIR-V assembly comment header with full triage/repro context.
    rel_out = os.path.relpath(out_path, os.getcwd()) if os.path.isabs(out_path) else out_path
    canon_flag = " --canonicalize" if canonicalize else ""
    header_lines = [
        "; ValhallSSA divergence reproduction recorded by tools.coverage --record",
        f"; source: {os.path.basename(src_path)}",
        f"; stage: {bundle.get('stage', 'Unknown')} arch: v{bundle.get('arch', '?')}",
        f"; size_bytes: {orig_bytes} -> {rec_bytes} (method={reduce_method})",
        f"; vec_idx: {rep.vec_idx}",
        f"; output: {rep.output_name} direct={rep.direct_val} valhall={rep.valhall_val} kind={rep.diff_kind} ulp={rep.ulp_diff}",
        f"; shrunk_overrides: {json.dumps(rep.shrunk_overrides, sort_keys=True)}",
    ]
    if triage is not None:
        header_lines.append(f"; triage_suspect: {triage.suspect_category}")
        if triage.anomalies:
            header_lines.append(f"; triage_signals: {'; '.join(triage.anomalies)}")
    if rep.frontier_pcs:
        pcs = ", ".join(f"0x{pc:x}({dis})" if dis else f"0x{pc:x}" for pc, dis in rep.frontier_pcs[:6])
        header_lines.append(f"; frontier_pcs: {pcs}")
    header_lines.append(
        f"; repro_cmd: python3 -m tools.coverage{canon_flag} --k-init 64 --domain udomain --fp-mode real "
        f"--equiv-fuzz --shrink --trace-diff {rel_out}"
    )
    header_lines.append("")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(header_lines))
        fh.write(asm_body)

    return RecordResult(
        out_path=out_path,
        original_bytes=orig_bytes,
        recorded_bytes=rec_bytes,
        reduced=reduced,
        reduce_method=reduce_method,
        verified_reproduces=verified,
    )

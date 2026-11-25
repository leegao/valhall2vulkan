"""Evidence-based divergence triage and human-readable trace diagnostics for tools.coverage."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from tools.coverage.shrinker import (
    DiffPolicy,
    DivergenceReport,
    _backward_cone,
    compare_concrete_outputs,
)
from tools.symbolic_ssa import ConcreteSsaInterpreter
from tools.symbolic_ssa.env import ConcreteShaderEnv, _u32_to_f32

_SINK_OPS = {
    "StoreFragColor",
    "StoreBuiltinOutput",
    "StoreVertVarying",
    "PsbStore",
    "StoreTexel",
    "Kill",
    "AtomicRmw",
    "AtomicExchange",
    "AtomicCmpXchg",
}

_FP_OUTPUT_PREFIXES = ("frag_color[", "builtin_out[0]", "vert_var[")

_RODATA_BASE = 0x50000000000
_RODATA_END = 0x60000000000


@dataclass
class TriageReport:
    """Structured evidence-based triage summary for a concrete divergence."""

    vec_idx: int
    suspect_category: str
    canonicalizer_introduced: bool = False
    canonicalizer_side: str = ""
    direct_only_ops: list[str] = field(default_factory=list)
    valhall_only_ops: list[str] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)
    frontier_trace: list[str] = field(default_factory=list)


def _fmt_hex_with_fp(val_str: str, output_name: str) -> str:
    """Annotate 32-bit hex values with decoded FP32 when the output is float-like."""
    if not val_str.startswith("0x") or len(val_str) != 10:
        return val_str
    if not any(output_name.startswith(p) for p in _FP_OUTPUT_PREFIXES):
        return val_str
    try:
        u = int(val_str, 16)
        f = _u32_to_f32(u)
        if math.isnan(f):
            return f"{val_str}(nan)"
        if math.isinf(f):
            return f"{val_str}({'+inf' if f > 0 else '-inf'})"
        if u != 0 and (u & 0x7F800000) == 0:
            return f"{val_str}(subnormal={f:.3e})"
        return f"{val_str}({f:.6g})"
    except Exception:
        return val_str


def _is_nan_or_inf_u32(val: Any) -> str | None:
    if not isinstance(val, int) or isinstance(val, bool):
        return None
    u = val & 0xFFFFFFFF
    if (u & 0x7F800000) == 0x7F800000:
        return "NaN" if (u & 0x007FFFFF) != 0 else ("-Inf" if (u >> 31) else "+Inf")
    return None


def triage_divergence(
    bundle: dict[str, Any],
    direct_ssa: dict[str, Any],
    valhall_ssa: dict[str, Any],
    rep: DivergenceReport,
    *,
    raw_direct_ssa: dict[str, Any] | None = None,
    raw_valhall_ssa: dict[str, Any] | None = None,
    policy: DiffPolicy | None = None,
) -> TriageReport:
    """
    Analyze a localized concrete divergence and produce evidence-based triage hints
    separating canonicalizer bugs, modeling/ABI mismatches, spirv2ssa asymmetries,
    and Valhall ISA->SSA lowering bugs.
    """
    vec_idx = rep.vec_idx
    ov = rep.shrunk_overrides
    anomalies: list[str] = []

    # Check if canonicalization itself introduced the divergence.
    canon_introduced = False
    canon_side = ""
    if raw_direct_ssa is not None and raw_valhall_ssa is not None:
        raw_div, _, _, _, _, _ = compare_concrete_outputs(
            bundle, raw_direct_ssa, raw_valhall_ssa, overrides=ov, vec_idx=vec_idx, policy=policy
        )
        if not raw_div:
            canon_introduced = True
            d_raw_vs_v_can, _, _, _, _, _ = compare_concrete_outputs(
                bundle, raw_direct_ssa, valhall_ssa, overrides=ov, vec_idx=vec_idx, policy=policy
            )
            d_can_vs_v_raw, _, _, _, _, _ = compare_concrete_outputs(
                bundle, direct_ssa, raw_valhall_ssa, overrides=ov, vec_idx=vec_idx, policy=policy
            )
            if d_can_vs_v_raw and not d_raw_vs_v_can:
                canon_side = "direct"
            elif d_raw_vs_v_can and not d_can_vs_v_raw:
                canon_side = "valhall"
            else:
                canon_side = "both"
            anomalies.append(
                f"canonicalizer_introduced(side={canon_side}): raw SSA agrees on vec={vec_idx}"
            )

    # Re-execute once on the minimal witness to inspect concrete traces.
    cenv = ConcreteShaderEnv(bundle, vec_idx, overrides=ov)
    _d_vals, _ = ConcreteSsaInterpreter(direct_ssa, is_valhall_side=False).execute_one(cenv)
    v_vals, _ = ConcreteSsaInterpreter(valhall_ssa, is_valhall_side=True).execute_one(cenv)

    d_cone = _backward_cone(direct_ssa, _SINK_OPS)
    v_cone = _backward_cone(valhall_ssa, _SINK_OPS)
    d_nodes = {int(n["id"]): n for n in direct_ssa.get("nodes", [])}
    v_nodes = {int(n["id"]): n for n in valhall_ssa.get("nodes", [])}

    d_ops: set[str] = {d_nodes[i]["op"] for i in d_cone if i in d_nodes}
    v_ops: set[str] = {v_nodes[i]["op"] for i in v_cone if i in v_nodes}
    d_only = sorted(d_ops - v_ops)
    v_only = sorted(v_ops - d_ops)

    # Detect out-of-bounds ebin_rodata loads (classic 64-bit address/swizzle lowering bug).
    rodata_raw = bundle.get("mbs2_context", {}).get("ebin_rodata")
    if isinstance(rodata_raw, dict):
        rodata_entries = [rodata_raw]
    elif isinstance(rodata_raw, list):
        rodata_entries = [e for e in rodata_raw if isinstance(e, dict)]
    else:
        rodata_entries = []
    max_rodata_end = 0
    for ent in rodata_entries:
        off = int(ent.get("offset", 0))
        sz = int(ent.get("size", len(ent.get("hex", "")) // 2))
        max_rodata_end = max(max_rodata_end, off + sz)

    oob_rodata: list[str] = []
    for vid in sorted(v_cone):
        vn = v_nodes.get(vid)
        if not vn or vn["op"] != "PsbLoad" or vid not in v_vals:
            continue
        args = vn.get("args", [])
        if not args:
            continue
        addr = v_vals.get(int(args[0]))
        if isinstance(addr, int) and _RODATA_BASE <= addr < _RODATA_END:
            rel = addr - _RODATA_BASE
            if max_rodata_end > 0 and rel >= max_rodata_end:
                pc = int(vn.get("orig_pc", 0))
                oob_rodata.append(f"0x{pc:x}(rodata+0x{rel:x})")
    if oob_rodata:
        anomalies.append(f"oob_rodata_load: {', '.join(oob_rodata[:3])}")

    # Detect known modeling/ABI signatures in Valhall disassembly within the cone.
    pc_disasm: dict[str, str] = bundle.get("mbs2_context", {}).get("pc_disasm", {})
    cone_pcs = {int(v_nodes[i]["orig_pc"]) for i in v_cone if i in v_nodes and "orig_pc" in v_nodes[i]}
    cone_dis_lines = [(pc, pc_disasm.get(str(pc), "")) for pc in sorted(cone_pcs)]
    has_tex_gradient = any("TEX_GRADIENT" in d for _, d in cone_dis_lines)
    has_shadd_sample = any("SHADD.u64" in d and "sample" in d for _, d in cone_dis_lines)
    has_cubeface = any("CUBEFACE" in d for _, d in cone_dis_lines)
    has_tex_gather = any("TEX_GATHER" in d for _, d in cone_dis_lines)
    has_ld_pka = any("LD_PKA" in d for _, d in cone_dis_lines)
    has_fexp = any("FEXP.f32" in d or "FMUL_RSCALE" in d for _, d in cone_dis_lines)

    if has_tex_gradient:
        anomalies.append("cone_has_TEX_GRADIENT(implicit->explicit LOD emulation)")
    if has_shadd_sample:
        anomalies.append("cone_has_SHADD_sample(gl_SamplePosition/ImageQuerySize pilot)")
    if has_cubeface:
        anomalies.append("cone_has_CUBEFACE(cube-face projection vs 3D TexSample)")
    if has_tex_gather:
        anomalies.append("cone_has_TEX_GATHER(gather arity/FTZ check)")
    if has_ld_pka:
        anomalies.append("cone_has_LD_PKA(packed descriptor load)")
    if has_fexp:
        anomalies.append("cone_has_FEXP/FMUL_RSCALE(transcendental range)")

    # Check channel presence asymmetry (direct=None or valhall=None).
    if rep.direct_val == "None" or rep.valhall_val == "None":
        anomalies.append(f"asymmetric_output_presence(direct={rep.direct_val}, valhall={rep.valhall_val})")

    # Check subnormal FTZ difference on float outputs.
    if rep.direct_val.startswith("0x") and rep.valhall_val.startswith("0x"):
        try:
            du = int(rep.direct_val, 16)
            vu = int(rep.valhall_val, 16)
            d_sub = (du & 0x7F800000) == 0 and (du & 0x007FFFFF) != 0
            v_zero = (vu & 0x7FFFFFFF) == 0
            v_sub = (vu & 0x7F800000) == 0 and (vu & 0x007FFFFF) != 0
            d_zero = (du & 0x7FFFFFFF) == 0
            if (d_sub and v_zero) or (v_sub and d_zero):
                anomalies.append("subnormal_ftz_mismatch")
        except ValueError:
            pass

    # Find earliest non-constant Valhall frontier nodes with concrete arg -> result values.
    const_ops = {"ConstU32", "ConstS32", "ConstF32", "ConstU64", "ConstBool"}
    ordered_fnids = [
        vid for vid in rep.frontier_valhall_nids
        if vid in v_nodes and v_nodes[vid]["op"] not in const_ops
    ]
    if not ordered_fnids:
        ordered_fnids = list(rep.frontier_valhall_nids)
    frontier_trace: list[str] = []
    for vid in ordered_fnids[:12]:
        vn = v_nodes.get(vid)
        if not vn:
            continue
        pc = int(vn.get("orig_pc", 0))
        dis = pc_disasm.get(str(pc), "")
        arg_strs = []
        for aid in vn.get("args", []):
            av = v_vals.get(int(aid))
            if isinstance(av, int) and not isinstance(av, bool):
                arg_strs.append(f"v{aid}=0x{av & 0xFFFFFFFF:08x}")
            else:
                arg_strs.append(f"v{aid}={av}")
        rv = v_vals.get(vid)
        rv_s = f"0x{rv & 0xFFFFFFFF:08x}" if isinstance(rv, int) and not isinstance(rv, bool) else str(rv)
        special = _is_nan_or_inf_u32(rv)
        tag = f" [{special}]" if special else ""
        frontier_trace.append(
            f"0x{pc:x} v{vid} {vn['op']}({', '.join(arg_strs)}) -> {rv_s}{tag}"
            + (f"  ; {dis}" if dis else "")
        )

    # Classify suspect category based on concrete evidence.
    if canon_introduced:
        suspect = "MODELING_CANONICALIZER"
    elif oob_rodata:
        suspect = "LOWERING_ADDR_OR_SWIZZLE"
    elif has_tex_gradient or has_shadd_sample or has_cubeface or any("subnormal_ftz" in a for a in anomalies):
        suspect = "MODELING_ENV_OR_ABI"
    elif has_tex_gather and ("TexGather" in d_ops or "TexGather" in v_ops):
        suspect = "SPIRV2SSA_OR_MODELING_TEX_GATHER"
    else:
        suspect = "LOWERING_OR_SEMANTICS"

    return TriageReport(
        vec_idx=vec_idx,
        suspect_category=suspect,
        canonicalizer_introduced=canon_introduced,
        canonicalizer_side=canon_side,
        direct_only_ops=d_only,
        valhall_only_ops=v_only,
        anomalies=anomalies,
        frontier_trace=frontier_trace,
    )


def format_divergence_and_triage(
    rep: DivergenceReport,
    triage: TriageReport | None = None,
    *,
    trace_diff: bool = False,
) -> list[str]:
    """Format [SHRINK] and [TRIAGE] lines for human operators."""
    d_fmt = _fmt_hex_with_fp(rep.direct_val, rep.output_name)
    v_fmt = _fmt_hex_with_fp(rep.valhall_val, rep.output_name)
    lines = [
        (
            f"  [SHRINK] vec={rep.vec_idx} output={rep.output_name} direct={d_fmt} valhall={v_fmt} "
            f"kind={rep.diff_kind} ulp={rep.ulp_diff} "
            f"overrides={rep.original_override_count}->{rep.shrunk_override_count} "
            f"cone=({rep.cone_direct_size},{rep.cone_valhall_size})"
        )
    ]
    if rep.frontier_pcs:
        pcs_str = ", ".join(f"0x{pc:x}({dis})" if dis else f"0x{pc:x}" for pc, dis in rep.frontier_pcs[:6])
        lines.append(f"    frontier_pcs: {pcs_str}")
    if rep.shrunk_overrides:
        import json
        lines.append(f"    shrunk_overrides: {json.dumps(rep.shrunk_overrides, sort_keys=True)[:200]}")
    if triage is not None:
        parts = [f"suspect={triage.suspect_category}"]
        if triage.anomalies:
            parts.append(f"signals=[{'; '.join(triage.anomalies[:4])}]")
        lines.append(f"  [TRIAGE] {' '.join(parts)}")
        if triage.direct_only_ops or triage.valhall_only_ops:
            d_preview = ",".join(triage.direct_only_ops[:8]) or "-"
            v_preview = ",".join(triage.valhall_only_ops[:8]) or "-"
            lines.append(f"    cone_ops_diff: direct_only=[{d_preview}] valhall_only=[{v_preview}]")
        if trace_diff and triage.frontier_trace:
            lines.append("    frontier_trace:")
            for ft in triage.frontier_trace[:10]:
                lines.append(f"      {ft}")
    return lines

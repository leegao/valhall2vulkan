"""
Concolic Counterexample Auto-Shrinker, FP/ULP Triage Oracle & Frontier Localizer.

Given a dual-SSA bundle and a concrete counterexample (from ConcreteDiff or a
Z3 SAT model), this module:
  - Classifies per-output differences under a configurable DiffPolicy
    (EXACT, BENIGN_FP_ROUNDING within <= max_ulp, NAN_PAYLOAD_ONLY, or
    REAL_DIVERGENCE) so benign FMA/FRCP rounding and NaN payload bits are
    separated from genuine lowering bugs.
  - Minimizes the input override dictionary to a 1-minimal witness via
    delta-shrinking while preserving the real divergence.
  - Slices the backward SSA cone of the diverging output and identifies the
    earliest Valhall SSA nodes (and their exact Valhall PCs) where Valhall
    semantics diverge from Direct SPIR-V semantics on that minimal witness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from tools.symbolic_ssa.env import (
    ConcreteShaderEnv,
    _c_u32,
    _f32_to_u32,
    _f32_ulp_diff,
    _u32_to_f32,
)
from tools.symbolic_ssa import ConcreteSsaInterpreter


@dataclass
class DiffPolicy:
    """Precision and rounding policy for separating real bugs from benign FP noise."""

    max_ulp: int = 2
    ignore_nan_payload: bool = True
    treat_neg_zero_eq: bool = True
    fp_output_prefixes: Tuple[str, ...] = ("frag_color[", "builtin_out[0]", "vert_var[")


@dataclass
class DivergenceReport:
    """Result of auto-shrinking, FP triage, and localizing a concrete divergence."""

    diverged: bool
    output_name: str
    direct_val: str
    valhall_val: str
    original_override_count: int
    shrunk_override_count: int
    diff_kind: str = "EXACT"
    ulp_diff: Optional[int] = None
    benign_fp_diffs: int = 0
    shrunk_overrides: Dict[str, Any] = field(default_factory=dict)
    frontier_valhall_nids: List[int] = field(default_factory=list)
    frontier_pcs: List[Tuple[int, str]] = field(default_factory=list)
    cone_direct_size: int = 0
    cone_valhall_size: int = 0
    vec_idx: int = 0


def _fmt_val(v: Any) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"0x{_c_u32(v):08x}"
    return str(v)


def _is_fp_output(output_name: str, policy: DiffPolicy) -> bool:
    return any(output_name.startswith(pref) for pref in policy.fp_output_prefixes)


def classify_value_diff(
    d_val: Any,
    v_val: Any,
    output_name: str,
    policy: Optional[DiffPolicy] = None,
) -> Tuple[str, Optional[int]]:
    """
    Classify the difference between two concrete output values as one of:
      EXACT | BENIGN_FP_ROUNDING | NAN_PAYLOAD_ONLY | REAL_DIVERGENCE.
    """
    if d_val == v_val:
        return "EXACT", 0
    if policy is None:
        return "REAL_DIVERGENCE", None
    if isinstance(d_val, bool) or isinstance(v_val, bool):
        return "REAL_DIVERGENCE", None
    if not (isinstance(d_val, int) and isinstance(v_val, int)):
        return "REAL_DIVERGENCE", None
    if not _is_fp_output(output_name, policy):
        return "REAL_DIVERGENCE", None

    du = _c_u32(d_val)
    vu = _c_u32(v_val)
    df = _u32_to_f32(du)
    vf = _u32_to_f32(vu)

    if math.isnan(df) and math.isnan(vf):
        return ("NAN_PAYLOAD_ONLY" if policy.ignore_nan_payload else "REAL_DIVERGENCE"), None
    if math.isnan(df) != math.isnan(vf):
        return "REAL_DIVERGENCE", None
    if df == 0.0 and vf == 0.0:
        return ("BENIGN_FP_ROUNDING" if policy.treat_neg_zero_eq else "REAL_DIVERGENCE"), 0
    ud = _f32_ulp_diff(du, vu)
    if ud is not None and ud <= policy.max_ulp:
        return "BENIGN_FP_ROUNDING", ud
    return "REAL_DIVERGENCE", ud


def align_vertex_gl_position(
    d_out: Dict[str, Any],
    v_out: Dict[str, Any],
    env: ConcreteShaderEnv,
    policy: Optional[DiffPolicy] = None,
) -> Dict[str, Any]:
    """
    Align Direct SPIR-V clip-space `gl_Position` (`builtin_out[0][0..3]`) with
    Mali's Position FIFO viewport-transformed epilogue `(x/w, y/w, z_vp, 1/w)`
    whenever `gl_mali_ViewportTransform` is present in `env.viewport_fau_words`.

    Returns a shallow copy of `d_out` with aligned `builtin_out[0][0..3]` when
    transformation applies, leaving `d_out` untouched if clip-space already
    matches or `gl_mali_ViewportTransform` is absent.
    """
    if env.stage != "Vertex" or not env.viewport_fau_words:
        return d_out
    keys = [f"builtin_out[0][{c}]" for c in range(4)]
    if not all(k in d_out and k in v_out for k in keys):
        return d_out

    # If raw clip-space already matches v_out (e.g. ST_CHECKED.istream already inverted it), keep as-is.
    if all(
        classify_value_diff(d_out[k], v_out[k], k, policy=policy)[0] != "REAL_DIVERGENCE"
        for k in keys
    ):
        return d_out

    cx, cy, cz, cw = (_u32_to_f32(_c_u32(d_out[k])) for k in keys)
    # Exact Mali Position FIFO epilogue (`gl_mali_ViewportTransform`):
    #   r0 = (|x| + |y| + |z|) * 0x272428e0
    #   r1 = FMAX.nan_propagate(|w|, 0.0f)
    #   denom = FMAX.nan_propagate(r0, r1)
    #   inv_w = (w < 0.0f) ? -(1.0f / denom) : (1.0f / denom)
    #   z_out = (|z| > |w|) ? (z * inv_w) : clamp_m1_1(z * inv_w)
    # Note: because r0 includes |x|+|y|+|z| via FMAX.nan_propagate, a NaN in ANY
    # clip-space coordinate propagates into inv_w and all four output lanes.
    eps = _u32_to_f32(0x272428E0)
    r0 = abs(cx) * eps + abs(cy) * eps + abs(cz) * eps
    aw = abs(cw)
    if math.isnan(r0) or math.isnan(aw):
        inv_w = float("nan")
    else:
        # Mali Position FIFO epilogue clamps denom against k0fr2f = 0x38000000 (2^-15 = 1/32768).
        denom = max(r0, aw, _u32_to_f32(0x38000000))
        if denom == 0.0:
            inv_w_abs = float("inf")
        else:
            inv_w_abs = 1.0 / denom
        inv_w = -inv_w_abs if (cw < 0.0) else inv_w_abs

    # Identity viewport from ShaderEnvBase: scale=(1,1,1), bias=(0,0,0)
    x_vp = cx * inv_w
    y_vp = cy * inv_w
    z_ndc = cz * inv_w
    if math.isnan(z_ndc):
        # Valhall FADD.f32.clamp_m1_1 (lowered to FClamp) is NaN-suppressing and clamps NaN to -1.0.
        z_clamped = -1.0
    else:
        z_clamped = max(-1.0, min(1.0, z_ndc))

    # Mali selects unclamped z_ndc when |z| > |w|, else clamp_m1_1(z_ndc).
    if not math.isnan(cz) and not math.isnan(cw) and abs(cz) > abs(cw):
        z_pref = z_ndc
    else:
        z_pref = z_clamped
    u_z_pref = _f32_to_u32(z_pref)
    u_z_alt = _f32_to_u32(z_ndc if z_pref is z_clamped else z_clamped)
    vz = v_out[keys[2]]
    if classify_value_diff(u_z_pref, vz, keys[2], policy=policy)[0] != "REAL_DIVERGENCE":
        u_z_out = u_z_pref
    elif classify_value_diff(u_z_alt, vz, keys[2], policy=policy)[0] != "REAL_DIVERGENCE":
        u_z_out = u_z_alt
    else:
        u_z_out = u_z_pref

    out = dict(d_out)
    out[keys[0]] = _f32_to_u32(x_vp)
    out[keys[1]] = _f32_to_u32(y_vp)
    out[keys[2]] = u_z_out
    out[keys[3]] = _f32_to_u32(inv_w)
    return out


def _extract_comparable_outputs(
    d_out: Dict[str, Any],
    v_out: Dict[str, Any],
) -> List[Tuple[str, Any, Any, Optional[int]]]:
    """Return list of (name, d_val, v_val, valhall_sink_nid) from flat ConcreteSsaInterpreter outputs."""
    items: List[Tuple[str, Any, Any, Optional[int]]] = []
    for k in sorted(set(d_out.keys()) | set(v_out.keys())):
        items.append((k, d_out.get(k), v_out.get(k), None))
    return items


def compare_concrete_outputs(
    bundle: Dict[str, Any],
    direct_ssa: Dict[str, Any],
    valhall_ssa: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
    vec_idx: int = 0,
    policy: Optional[DiffPolicy] = None,
    fp_range: str = "unconstrained",
) -> Tuple[bool, Optional[str], str, str, Dict[int, Any], Dict[int, Any]]:
    """Execute both SSA modules under a concrete env and compare outputs under DiffPolicy."""
    d_interp = ConcreteSsaInterpreter(direct_ssa, is_valhall_side=False)
    v_interp = ConcreteSsaInterpreter(valhall_ssa, is_valhall_side=True)
    cenv = ConcreteShaderEnv(bundle, vec_idx, overrides=overrides, fp_range=fp_range)
    d_vals, d_out = d_interp.execute_one(cenv)
    v_vals, v_out = v_interp.execute_one(cenv)
    d_out = align_vertex_gl_position(d_out, v_out, cenv, policy=policy)

    for name, dv, vv, _ in _extract_comparable_outputs(d_out, v_out):
        kind, _ = classify_value_diff(dv, vv, name, policy=policy)
        if kind == "REAL_DIVERGENCE":
            return True, name, _fmt_val(dv), _fmt_val(vv), d_vals, v_vals
    return False, None, "", "", d_vals, v_vals


def shrink_counterexample_overrides(
    bundle: Dict[str, Any],
    direct_ssa: Dict[str, Any],
    valhall_ssa: Dict[str, Any],
    overrides: Dict[str, Any],
    vec_idx: int = 0,
    policy: Optional[DiffPolicy] = None,
    fp_range: str = "unconstrained",
) -> Dict[str, Any]:
    """
    1-minimal delta-shrinker over concrete input overrides: drops or zeros
    irrelevant entries while preserving the REAL_DIVERGENCE.
    """
    cur = dict(overrides)
    div, target_name, *_ = compare_concrete_outputs(
        bundle, direct_ssa, valhall_ssa, cur, vec_idx=vec_idx, policy=policy, fp_range=fp_range
    )
    if not div:
        return cur

    def still_diverges(cand: Dict[str, Any]) -> bool:
        d_interp = ConcreteSsaInterpreter(direct_ssa, is_valhall_side=False)
        v_interp = ConcreteSsaInterpreter(valhall_ssa, is_valhall_side=True)
        _, d_out = d_interp.execute_one(ConcreteShaderEnv(bundle, vec_idx, overrides=cand, fp_range=fp_range))
        _, v_out = v_interp.execute_one(ConcreteShaderEnv(bundle, vec_idx, overrides=cand, fp_range=fp_range))
        for name, dv, vv, _ in _extract_comparable_outputs(d_out, v_out):
            if name == target_name:
                kind, _ = classify_value_diff(dv, vv, name, policy=policy)
                return kind == "REAL_DIVERGENCE"
        return False

    for k in sorted(list(cur.keys())):
        orig_v = cur[k]
        trial = dict(cur)
        del trial[k]
        if still_diverges(trial):
            cur = trial
            continue
        if isinstance(orig_v, (int, bool)):
            zero_v: Any = False if isinstance(orig_v, bool) else 0
            if orig_v != zero_v:
                trial = dict(cur)
                trial[k] = zero_v
                if still_diverges(trial):
                    cur = trial
    return cur


def _backward_cone(ssa: Dict[str, Any], sink_ops: Set[str]) -> Set[int]:
    nodes = {int(n["id"]): n for n in ssa.get("nodes", [])}
    cone: Set[int] = set()
    stack = [int(n["id"]) for n in ssa.get("nodes", []) if n["op"] in sink_ops]
    while stack:
        nid = stack.pop()
        if nid in cone or nid not in nodes:
            continue
        cone.add(nid)
        for aid in nodes[nid].get("args", []):
            stack.append(int(aid))
        for _, vid in nodes[nid].get("phi", []):
            stack.append(int(vid))
    return cone


def localize_and_shrink_divergence(
    bundle: Dict[str, Any],
    direct_ssa: Dict[str, Any],
    valhall_ssa: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
    vec_idx: int = 0,
    policy: Optional[DiffPolicy] = None,
    fp_range: str = "unconstrained",
) -> DivergenceReport:
    """
    Shrink a concrete counterexample to a 1-minimal witness under DiffPolicy
    and localize the earliest diverging Valhall SSA nodes and PCs.
    """
    orig_ov = dict(overrides) if overrides is not None else {}
    shrunk = shrink_counterexample_overrides(
        bundle, direct_ssa, valhall_ssa, orig_ov, vec_idx=vec_idx, policy=policy, fp_range=fp_range
    )
    d_interp = ConcreteSsaInterpreter(direct_ssa, is_valhall_side=False)
    v_interp = ConcreteSsaInterpreter(valhall_ssa, is_valhall_side=True)
    cenv = ConcreteShaderEnv(bundle, vec_idx, overrides=shrunk, fp_range=fp_range)
    d_vals, d_out = d_interp.execute_one(cenv)
    v_vals, v_out = v_interp.execute_one(cenv)
    d_out = align_vertex_gl_position(d_out, v_out, cenv, policy=policy)

    benign_cnt = 0
    first_real: Optional[Tuple[str, Any, Any, Optional[int]]] = None
    for name, dv, vv, _ in _extract_comparable_outputs(d_out, v_out):
        kind, ud = classify_value_diff(dv, vv, name, policy=policy)
        if kind in ("BENIGN_FP_ROUNDING", "NAN_PAYLOAD_ONLY"):
            benign_cnt += 1
        elif kind == "REAL_DIVERGENCE" and first_real is None:
            first_real = (name, dv, vv, ud)

    if first_real is None:
        return DivergenceReport(
            diverged=False,
            output_name="",
            direct_val="",
            valhall_val="",
            original_override_count=len(orig_ov),
            shrunk_override_count=len(shrunk),
            diff_kind="BENIGN_FP_ROUNDING" if benign_cnt > 0 else "EXACT",
            benign_fp_diffs=benign_cnt,
            shrunk_overrides=shrunk,
            vec_idx=vec_idx,
        )

    out_name, dv0, vv0, ud0 = first_real
    sink_ops = {
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
    d_cone = _backward_cone(direct_ssa, sink_ops)
    v_cone = _backward_cone(valhall_ssa, sink_ops)

    d_cone_vals: Set[Any] = {
        d_vals[nid] for nid in d_cone if nid in d_vals and not isinstance(d_vals[nid], tuple)
    }

    v_nodes = {int(n["id"]): n for n in valhall_ssa.get("nodes", [])}
    skip_ops = {"Nop", "Copy", "Bitcast", "VecConstruct2", "VecConstruct3", "VecConstruct4", "VecExtract", "TupleExtract"}

    frontier_nids: List[int] = []
    pc_disasm: Dict[str, str] = bundle.get("mbs2_context", {}).get("pc_disasm", {})
    frontier_pcs: List[Tuple[int, str]] = []
    seen_pcs: Set[int] = set()

    for vid in sorted(v_cone):
        vn = v_nodes.get(vid)
        if not vn or vn["op"] in skip_ops or vn["op"] in sink_ops:
            continue
        vv = v_vals.get(vid)
        if vv is None or isinstance(vv, tuple):
            continue
        if vv in d_cone_vals:
            continue
        args_match = True
        for aid in vn.get("args", []):
            ia = int(aid)
            an = v_nodes.get(ia)
            if an and an["op"] == "Nop":
                continue
            av = v_vals.get(ia)
            if av is not None and not isinstance(av, tuple) and av not in d_cone_vals:
                args_match = False
                break
        if args_match:
            frontier_nids.append(vid)
            if "orig_pc" in vn:
                pc = int(vn["orig_pc"])
                if pc not in seen_pcs:
                    seen_pcs.add(pc)
                    frontier_pcs.append((pc, pc_disasm.get(str(pc), "")))

    return DivergenceReport(
        diverged=True,
        output_name=out_name,
        direct_val=_fmt_val(dv0),
        valhall_val=_fmt_val(vv0),
        original_override_count=len(orig_ov),
        shrunk_override_count=len(shrunk),
        diff_kind="REAL_DIVERGENCE",
        ulp_diff=ud0,
        benign_fp_diffs=benign_cnt,
        shrunk_overrides=shrunk,
        frontier_valhall_nids=frontier_nids,
        frontier_pcs=frontier_pcs,
        cone_direct_size=len(d_cone),
        cone_valhall_size=len(v_cone),
        vec_idx=vec_idx,
    )

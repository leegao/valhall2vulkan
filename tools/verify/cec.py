#!/usr/bin/env python3
"""
Combinational Equivalence Checking (CEC) via Simulation-Guided Cutpoint Sweeping.

Three composable stages operating on Stratum-1 canonicalized ValhallSsaModule pairs:
  - Concrete differential simulation (`run_concrete_diff`) over K edge-case and
    seeded random input vectors, catching real lowering bugs in <5ms and recording
    per-node K-vector signatures.
  - Candidate cutpoint discovery (`discover_cutpoint_candidates`) matching internal
    node signatures across direct_ssa and valhall_ssa (EXACT, NEGATED_BOOL, FNEG).
  - Incremental bottom-up CEC cutpoint sweeping (`verify_with_cec`) using persistent
    `SymbolicSsaInterpreter` instances with `evaluate_single_block_cone` and
    `invalidate_cone_from` for amortized O(N) DAG traversal instead of O(M * N)
    full re-evaluation, followed by CEGAR output refinement.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import z3

from tools.symbolic_ssa.z3utils import dsl

from tools.symbolic_ssa.env import (
    MASK32,
    ConcreteShaderEnv,
    SymbolicShaderEnv,
    _as_bool,
    _as_bv32,
    _as_bv64,
    _f32_ulp_diff,
    _z3_and,
    _z3_not,
    _z3_or,
)
from tools.symbolic_ssa import (
    BlockEffectState,
    ConcreteSsaInterpreter,
    SymbolicSsaInterpreter,
)
from tools.symbolic_ssa.passes import SIDE_EFFECT_OPS, canonicalize_ssa_module
from tools.verify.panfrost_z3_model import bv32, bv32_to_fp32
from tools.verify.shader_equiv_verifier import _solve_miter_eq


@dataclasses.dataclass
class ConcreteDiffResult:
    """Result of K-vector concrete differential simulation between direct and valhall SSA."""

    status: str  # EXACT_MATCH, APPROX_ULP1, MISMATCH, SKIP
    k_vectors: int
    checked_outputs: int
    exact_outputs: int
    approx_ulp1_outputs: int
    mismatched_outputs: int
    first_mismatch: Optional[str]
    elapsed_ms: float
    d_node_traces: Dict[int, List[Any]]
    v_node_traces: Dict[int, List[Any]]
    mismatches: List[str] = dataclasses.field(default_factory=list)
    direct_only_channels: List[str] = dataclasses.field(default_factory=list)
    valhall_only_channels: List[str] = dataclasses.field(default_factory=list)


def run_concrete_diff(
    bundle: Dict[str, Any],
    direct_ssa: Dict[str, Any],
    valhall_ssa: Dict[str, Any],
    k: int = 32,
    seed: int = 0x5A17,
) -> ConcreteDiffResult:
    """Run K-vector concrete differential simulation and collect per-node value traces."""
    t0 = time.perf_counter()
    d_interp = ConcreteSsaInterpreter(direct_ssa, is_valhall_side=False)
    v_interp = ConcreteSsaInterpreter(valhall_ssa, is_valhall_side=True)

    d_node_traces: Dict[int, List[Any]] = {}
    v_node_traces: Dict[int, List[Any]] = {}
    d_out_traces: Dict[str, List[Any]] = {}
    v_out_traces: Dict[str, List[Any]] = {}
    d_written_prefixes: Set[str] = set()
    v_written_prefixes: Set[str] = set()
    vec_input_summaries: List[str] = []

    for i in range(k):
        cenv = ConcreteShaderEnv(bundle, vec_idx=i, seed=seed)
        d_vals, d_outs = d_interp.execute_one(cenv)
        v_vals, v_outs = v_interp.execute_one(cenv)
        for ok in d_outs:
            d_written_prefixes.add(ok.split("+")[0].split("[")[0])
        for ok in v_outs:
            v_written_prefixes.add(ok.split("+")[0].split("[")[0])
        bi_items = []
        for bkey, bval in sorted(getattr(cenv, "builtins", {}).items())[:4]:
            if isinstance(bval, (list, tuple)):
                bi_items.append(f"bi_{bkey}=(" + ",".join(f"0x{int(x)&0xffffffff:x}" for x in bval[:3]) + ")")
            elif isinstance(bval, int):
                bi_items.append(f"bi_{bkey}=0x{bval&0xffffffff:x}")
        vec_input_summaries.append(", ".join(bi_items) if bi_items else f"seed=0x{seed:x},vec={i}")
        for nid, val in d_vals.items():
            d_node_traces.setdefault(nid, [None] * k)[i] = val
        for nid, val in v_vals.items():
            v_node_traces.setdefault(nid, [None] * k)[i] = val
        all_okeys = set(d_outs.keys()) | set(v_outs.keys())
        for ok in all_okeys:
            if ok not in d_out_traces:
                d_out_traces[ok] = [None] * k
                v_out_traces[ok] = [None] * k
            d_out_traces[ok][i] = d_outs.get(ok)
            v_out_traces[ok][i] = v_outs.get(ok)

    all_keys = sorted(d_out_traces.keys())
    if not all_keys:
        return ConcreteDiffResult(
            "SKIP",
            k,
            0,
            0,
            0,
            0,
            None,
            (time.perf_counter() - t0) * 1000.0,
            d_node_traces,
            v_node_traces,
        )

    exact_cnt = 0
    approx_cnt = 0
    mismatch_cnt = 0
    first_mm: Optional[str] = None
    mm_list: List[str] = []
    for ok in all_keys:
        dl, vl = d_out_traces[ok], v_out_traces[ok]
        if dl == vl:
            exact_cnt += 1
            continue
        max_ulp = 0
        is_approx = True
        first_idx = -1
        for idx, (dv, vv) in enumerate(zip(dl, vl)):
            if dv == vv:
                continue
            if first_idx < 0:
                first_idx = idx
            if isinstance(dv, int) and isinstance(vv, int) and not isinstance(dv, bool) and not isinstance(vv, bool):
                ud = _f32_ulp_diff(dv, vv)
                if ud is None or ud > 1:
                    is_approx = False
                else:
                    max_ulp = max(max_ulp, ud)
            else:
                is_approx = False
        if is_approx and max_ulp <= 1:
            approx_cnt += 1
        else:
            mismatch_cnt += 1
            dv0, vv0 = dl[first_idx], vl[first_idx]
            d_str = f"0x{dv0:08x}" if isinstance(dv0, int) and not isinstance(dv0, bool) else str(dv0)
            v_str = f"0x{vv0:08x}" if isinstance(vv0, int) and not isinstance(vv0, bool) else str(vv0)
            inp_s = vec_input_summaries[first_idx] if first_idx < len(vec_input_summaries) else ""
            mm_entry = f"{ok}@vec[{first_idx}]: direct={d_str} valhall={v_str}" + (f" ({inp_s})" if inp_s else "")
            if first_mm is None:
                first_mm = mm_entry
            if len(mm_list) < 10:
                mm_list.append(mm_entry)

    if mismatch_cnt > 0:
        status = "MISMATCH"
    elif approx_cnt > 0:
        status = "APPROX_ULP1"
    else:
        status = "EXACT_MATCH"
    return ConcreteDiffResult(
        status=status,
        k_vectors=k,
        checked_outputs=len(all_keys),
        exact_outputs=exact_cnt,
        approx_ulp1_outputs=approx_cnt,
        mismatched_outputs=mismatch_cnt,
        first_mismatch=first_mm,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        d_node_traces=d_node_traces,
        v_node_traces=v_node_traces,
        mismatches=mm_list,
        direct_only_channels=sorted(d_written_prefixes - v_written_prefixes),
        valhall_only_channels=sorted(v_written_prefixes - d_written_prefixes),
    )


_SKIP_CUTPOINT_OPS: Set[str] = SIDE_EFFECT_OPS | {
    "Nop",
    "Copy",
    "Bitcast",
    "VecExtract",
    "Phi",
    "ConstU32",
    "ConstS32",
    "ConstF32",
    "ConstU64",
    "ConstBool",
    "LoadBuiltinInput",
    "LoadVertAttr",
    "LoadFragVarying",
    "LoadSidebandU64",
    "LoadSidebandU32",
    "LoadFau",
    "Branch",
    "BranchCond",
    "Return",
}


def _compute_node_depths(ssa: Dict[str, Any]) -> Dict[int, int]:
    nodes = {int(n["id"]): n for n in ssa.get("nodes", [])}
    depth: Dict[int, int] = {}

    def get_d(nid: int, visiting: Set[int]) -> int:
        if nid in depth:
            return depth[nid]
        if nid not in nodes or nid in visiting:
            return 0
        visiting.add(nid)
        n = nodes[nid]
        arg_ds = [get_d(int(a), visiting) for a in n.get("args", [])]
        d = (1 + max(arg_ds)) if arg_ds else 0
        visiting.remove(nid)
        depth[nid] = d
        return d

    for nid in nodes:
        get_d(nid, set())
    return depth


def _build_users_map(ssa: Dict[str, Any]) -> Dict[int, List[int]]:
    users: Dict[int, List[int]] = {}
    for n in ssa.get("nodes", []):
        uid = int(n["id"])
        for a in n.get("args", []):
            users.setdefault(int(a), []).append(uid)
    return users


def discover_cutpoint_candidates(
    direct_ssa: Dict[str, Any],
    valhall_ssa: Dict[str, Any],
    d_traces: Dict[int, List[Any]],
    v_traces: Dict[int, List[Any]],
) -> List[Tuple[int, int, str]]:
    """
    Match internal scalar/bool SSA nodes across direct_ssa and valhall_ssa by
    K-vector concrete signature (EXACT, NEGATED_BOOL, FNEG).
    Returns candidates sorted topologically by direct node depth.
    """
    d_nodes = {int(n["id"]): n for n in direct_ssa.get("nodes", [])}
    v_nodes = {int(n["id"]): n for n in valhall_ssa.get("nodes", [])}
    d_depths = _compute_node_depths(direct_ssa)

    v_by_sig: Dict[Tuple[str, Tuple[Any, ...]], List[int]] = {}
    for vid, vn in v_nodes.items():
        if vn["op"] in _SKIP_CUTPOINT_OPS:
            continue
        vt = vn.get("type", "U32")
        if vt not in ("U32", "S32", "F32", "Bool", "U64", "S64"):
            continue
        tr = v_traces.get(vid)
        if not tr or any(x is None for x in tr):
            continue
        if len(set(tr)) <= 1:
            continue
        tcls = "Bool" if vt == "Bool" else ("64" if vt in ("U64", "S64") else "32")
        v_by_sig.setdefault((tcls, tuple(tr)), []).append(vid)

    candidates: List[Tuple[int, int, int, str]] = []
    used_v: Set[int] = set()
    for did in sorted(d_nodes.keys(), key=lambda x: (d_depths.get(x, 0), x)):
        dn = d_nodes[did]
        if dn["op"] in _SKIP_CUTPOINT_OPS:
            continue
        vt = dn.get("type", "U32")
        if vt not in ("U32", "S32", "F32", "Bool", "U64", "S64"):
            continue
        tr = d_traces.get(did)
        if not tr or any(x is None for x in tr) or len(set(tr)) <= 1:
            continue
        tcls = "Bool" if vt == "Bool" else ("64" if vt in ("U64", "S64") else "32")
        sig = tuple(tr)

        matched: Optional[Tuple[int, str]] = None
        for vid in v_by_sig.get((tcls, sig), []):
            if vid not in used_v:
                matched = (vid, "EXACT")
                break
        if matched is None and tcls == "Bool":
            neg_sig = tuple(not x for x in tr)
            for vid in v_by_sig.get(("Bool", neg_sig), []):
                if vid not in used_v:
                    matched = (vid, "NEGATED_BOOL")
                    break
        if matched is None and vt == "F32":
            fneg_sig = tuple(
                0x7FC00000 if ((x & 0x7FFFFFFF) > 0x7F800000) else ((x ^ 0x80000000) & MASK32)
                for x in tr
            )
            for vid in v_by_sig.get(("32", fneg_sig), []):
                if vid not in used_v:
                    matched = (vid, "FNEG")
                    break
        if matched is not None:
            vid, kind = matched
            used_v.add(vid)
            candidates.append((d_depths.get(did, 0), did, vid, kind))

    candidates.sort(key=lambda x: (x[0], x[1]))
    return [(did, vid, kind) for _, did, vid, kind in candidates]


def _extract_cutpoint_vars(expr: z3.ExprRef) -> Set[str]:
    """Collect names of cut_* symbolic variables appearing in a Z3 expression."""
    found: Set[str] = set()
    visited: Set[int] = set()
    stack: List[z3.ExprRef] = [expr]
    while stack:
        cur = stack.pop()
        cid = cur.get_id()
        if cid in visited:
            continue
        visited.add(cid)
        if z3.is_const(cur) and cur.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            nm = cur.decl().name()
            if nm.startswith("cut_"):
                found.add(nm)
        elif z3.is_app(cur):
            for ch in cur.children():
                stack.append(ch)
    return found


def _empty_effect_state(env: SymbolicShaderEnv) -> BlockEffectState:
    return BlockEffectState(
        discarded=dsl.BoolVal(False),
        terminated=dsl.BoolVal(False),
        demoted=dsl.BoolVal(False),
        frag_colors={},
        builtin_outputs={},
        vert_varyings={},
        const_stores={},
        sym_buf_stores={},
        buf_mems={},
        image_mems={},
        fallback_mem=env.fallback_mem,
        atomic_events={},
        output_pcs={},
    )


@dataclasses.dataclass
class CecResult:
    """Summary result of concrete differential simulation + CEC cutpoint sweeping."""

    status: str  # UNSAT, SAT_CONCRETE, SAT_SMT, UNKNOWN, SKIP
    elapsed_ms: float
    concrete_ms: float
    sweep_ms: float
    final_ms: float
    concrete_status: str
    candidates_found: int
    cutpoints_proved: int
    cutpoints_failed: int
    cegar_refinements: int
    checked_outputs: int
    unsat_outputs: int
    unknown_outputs: int
    sat_outputs: int
    first_mismatch: Optional[str] = None
    sweep_evals: int = 0
    concrete_mismatches: List[str] = dataclasses.field(default_factory=list)
    direct_only_channels: List[str] = dataclasses.field(default_factory=list)
    valhall_only_channels: List[str] = dataclasses.field(default_factory=list)


Stratum2Result = CecResult


def verify_with_cec(
    bundle: Dict[str, Any],
    canonicalize: bool = True,
    k_sim: int = 32,
    cone_timeout_ms: int = 250,
    output_timeout_ms: int = 1000,
    max_cutpoints: int = 128,
    max_cegar_rounds: int = 3,
    skip_smt_on_concrete_mismatch: bool = True,
    domain: str = "udomain",
    fp_mode: str = "real",
) -> CecResult:
    """
    Run Stratum 1 canonicalization (optional) + concrete differential simulation +
    incremental bottom-up CEC cutpoint sweeping + CEGAR output verification.
    """
    t0 = time.perf_counter()
    direct_ssa = bundle["direct_ssa"]
    valhall_ssa = bundle["valhall_ssa"]
    if canonicalize:
        direct_ssa, _ = canonicalize_ssa_module(direct_ssa)
        valhall_ssa, _ = canonicalize_ssa_module(valhall_ssa)

    cdiff = run_concrete_diff(bundle, direct_ssa, valhall_ssa, k=k_sim)
    cands = discover_cutpoint_candidates(direct_ssa, valhall_ssa, cdiff.d_node_traces, cdiff.v_node_traces)

    if cdiff.status == "MISMATCH" and skip_smt_on_concrete_mismatch:
        return CecResult(
            status="SAT_CONCRETE",
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            concrete_ms=cdiff.elapsed_ms,
            sweep_ms=0.0,
            final_ms=0.0,
            concrete_status=cdiff.status,
            candidates_found=len(cands),
            cutpoints_proved=0,
            cutpoints_failed=0,
            cegar_refinements=0,
            checked_outputs=cdiff.checked_outputs,
            unsat_outputs=0,
            unknown_outputs=0,
            sat_outputs=cdiff.mismatched_outputs,
            first_mismatch=cdiff.first_mismatch,
            concrete_mismatches=cdiff.mismatches,
            direct_only_channels=cdiff.direct_only_channels,
            valhall_only_channels=cdiff.valhall_only_channels,
        )

    t_sw = time.perf_counter()
    env = SymbolicShaderEnv(bundle)
    env.shared_sym_bkeys = set()
    preconditions = env.build_preconditions()

    d_cut_map: Dict[int, z3.ExprRef] = {}
    v_cut_map: Dict[int, z3.ExprRef] = {}
    cut_name_to_pair: Dict[str, Tuple[int, int]] = {}
    proved = 0
    failed = 0
    sweep_evals = 0

    d_nodes_by_id = {int(n["id"]): n for n in direct_ssa.get("nodes", [])}

    # Incremental single-block cutpoint sweeping: evaluate only the uncached backward
    # cone between the current cutpoint frontier and each candidate (amortized O(N)).
    can_sweep = len(direct_ssa.get("blocks", [])) == 1 and len(valhall_ssa.get("blocks", [])) == 1
    if can_sweep and cands:
        d_users = _build_users_map(direct_ssa)
        v_users = _build_users_map(valhall_ssa)
        de = SymbolicSsaInterpreter(direct_ssa, env, False, fp_mode, 1, cutpoints=d_cut_map, domain=domain)
        ve = SymbolicSsaInterpreter(valhall_ssa, env, True, fp_mode, 1, cutpoints=v_cut_map, domain=domain)
        d_st = _empty_effect_state(env)
        v_st = _empty_effect_state(env)

        for d_nid, v_nid, kind in cands[:max_cutpoints]:
            prev_d_cnt = len(de.values)
            prev_v_cnt = len(ve.values)
            d_val = de.evaluate_single_block_cone(d_nid, d_st)
            v_val = ve.evaluate_single_block_cone(v_nid, v_st)
            sweep_evals += max(0, len(de.values) - prev_d_cnt) + max(0, len(ve.values) - prev_v_cnt)

            vt = d_nodes_by_id[d_nid].get("type", "U32")
            if vt == "Bool":
                lhs = de.domain.lower_z3_bool(d_val)
                rv_b = ve.domain.lower_z3_bool(v_val)
                rhs = _z3_not(rv_b) if kind == "NEGATED_BOOL" else rv_b
            elif vt in ("U64", "S64"):
                lhs = de.domain.lower_z3_bv64(d_val)
                rhs = ve.domain.lower_z3_bv64(v_val)
            elif vt == "F32" and kind == "FNEG" and (domain == "udomain" or fp_mode == "real"):
                lhs = de.domain.to_z3_real_f32(d_val)
                rhs = -ve.domain.to_z3_real_f32(v_val)
            else:
                lhs = de.domain.lower_z3_bv32(d_val)
                rv_32 = ve.domain.lower_z3_bv32(v_val)
                rhs = (rv_32 ^ bv32(0x80000000)) if kind == "FNEG" else rv_32

            rc, _ = _solve_miter_eq(lhs, rhs, None, cone_timeout_ms, env, preconditions=preconditions)
            if rc == "UNSAT":
                cname = f"cut_{proved}"
                if vt == "Bool":
                    cvar: z3.ExprRef = dsl.Bool(cname)
                    v_cvar = _z3_not(cvar) if kind == "NEGATED_BOOL" else cvar
                    d_cut_map[d_nid] = de.domain.lift_z3_bool(cvar)
                    v_cut_map[v_nid] = ve.domain.lift_z3_bool(v_cvar)
                elif vt in ("U64", "S64"):
                    cvar = dsl.BitVec(cname, 64)
                    d_cut_map[d_nid] = de.domain.lift_z3_bv64(cvar)
                    v_cut_map[v_nid] = ve.domain.lift_z3_bv64(cvar)
                elif vt == "F32" and kind == "FNEG" and (domain == "udomain" or fp_mode == "real"):
                    rvar = dsl.Real(cname)
                    # Lift via fresh F32 Real if domain supports it, else bv32
                    if hasattr(de.domain, "fresh_f32") and hasattr(ve.domain, "fp_unary"):
                        du = de.domain.fresh_f32(cname)
                        d_cut_map[d_nid] = du
                        v_cut_map[v_nid] = ve.domain.fp_unary("FNeg", du)
                    else:
                        cvar = dsl.BitVec(cname, 32)
                        d_cut_map[d_nid] = de.domain.lift_z3_bv32(cvar)
                        v_cut_map[v_nid] = ve.domain.lift_z3_bv32(cvar ^ bv32(0x80000000))
                else:
                    cvar = dsl.BitVec(cname, 32)
                    v_cvar = (cvar ^ bv32(0x80000000)) if kind == "FNEG" else cvar
                    d_cut_map[d_nid] = de.domain.lift_z3_bv32(cvar)
                    v_cut_map[v_nid] = ve.domain.lift_z3_bv32(v_cvar)
                de.invalidate_cone_from(d_nid, d_users)
                ve.invalidate_cone_from(v_nid, v_users)
                cut_name_to_pair[cname] = (d_nid, v_nid)
                proved += 1
            else:
                failed += 1
    sweep_ms = (time.perf_counter() - t_sw) * 1000.0

    t_fin = time.perf_counter()
    stage = bundle.get("stage", "Unknown")

    def _build_output_checks(cur_d_cuts: Dict[int, z3.ExprRef], cur_v_cuts: Dict[int, z3.ExprRef]):
        de = SymbolicSsaInterpreter(direct_ssa, env, False, fp_mode, 1, cutpoints=cur_d_cuts, domain=domain)
        ve = SymbolicSsaInterpreter(valhall_ssa, env, True, fp_mode, 1, cutpoints=cur_v_cuts, domain=domain)
        d_st_out = de.evaluate()
        v_st_out = ve.evaluate()
        checks: List[Tuple[str, z3.ExprRef, z3.ExprRef, Optional[z3.BoolRef]]] = []
        if stage == "Fragment" or not dsl.is_false(d_st_out.discarded) or not dsl.is_false(v_st_out.discarded):
            checks.append(("discarded", d_st_out.discarded, v_st_out.discarded, None))
        not_disc = _z3_not(d_st_out.discarded)
        for rt in sorted(set(d_st_out.frag_colors) | set(v_st_out.frag_colors)):
            dc = d_st_out.frag_colors.get(rt, (bv32(0),) * 4)
            vc = v_st_out.frag_colors.get(rt, (bv32(0),) * 4)
            for ch, cname in enumerate(("r", "g", "b", "a")):
                checks.append((f"frag_color[{rt}].{cname}", dc[ch], vc[ch], not_disc))
        for bi in sorted(set(d_st_out.builtin_outputs) | set(v_st_out.builtin_outputs)):
            do = d_st_out.builtin_outputs.get(bi)
            vo = v_st_out.builtin_outputs.get(bi)
            if do is None or vo is None:
                checks.append((f"builtin_out[{bi}].present", dsl.BoolVal(do is not None), dsl.BoolVal(vo is not None), not_disc))
                continue
            bi_guard = not_disc
            if stage == "Vertex" and bi == 0 and env.viewport_fau_words and isinstance(do, tuple) and len(do) >= 4:
                if fp_mode == "real" or domain == "udomain":
                    vp_conds = [de.domain.to_z3_real_f32(do[3]) == dsl.RealVal(1)]
                else:
                    vp_conds = [_as_bv32(do[3]) == bv32(0x3F800000)]
                    for c in range(3):
                        dbv = _as_bv32(do[c])
                        vp_conds.append(dsl.Not(dsl.fpIsNaN(bv32_to_fp32(dsl.lower(dbv)))))
                        vp_conds.append(dbv != bv32(0x80000000))
                bi_guard = _z3_and(bi_guard, dsl.And(*vp_conds))
            if isinstance(do, tuple) and isinstance(vo, tuple):
                for c in range(min(len(do), len(vo))):
                    checks.append((f"builtin_out[{bi}][{c}]", _as_bv32(do[c]), _as_bv32(vo[c]), bi_guard))
            else:
                checks.append((f"builtin_out[{bi}]", _as_bv32(do), _as_bv32(vo), bi_guard))
        for loc in sorted(set(d_st_out.vert_varyings) | set(v_st_out.vert_varyings)):
            dv = d_st_out.vert_varyings.get(loc, (bv32(0),) * 4)
            vv = v_st_out.vert_varyings.get(loc, (bv32(0),) * 4)
            for c in range(4):
                checks.append((f"vert_var[{loc}][{c}]", dv[c], vv[c], None))
        for bkey, off in sorted(set(d_st_out.const_stores) | set(v_st_out.const_stores)):
            if bkey in ("tls_private", "wls_shared") or bkey.startswith("workgroup:"):
                continue
            dw, dv = d_st_out.const_stores.get((bkey, off), (dsl.BoolVal(False), dsl.Select(env.get_buf_mem(bkey), bv32(off))))
            vw, vv = v_st_out.const_stores.get((bkey, off), (dsl.BoolVal(False), dsl.Select(env.get_buf_mem(bkey), bv32(off))))
            tag = f"{bkey}+0x{off:x}"
            checks.append((f"{tag}.written", dw, vw, None))
            checks.append((f"{tag}.value", dv, vv, _z3_or(dw, vw)))
        return checks

    checks = _build_output_checks(d_cut_map, v_cut_map)
    if not checks:
        return CecResult(
            "SKIP",
            (time.perf_counter() - t0) * 1000.0,
            cdiff.elapsed_ms,
            sweep_ms,
            (time.perf_counter() - t_fin) * 1000.0,
            cdiff.status,
            len(cands),
            proved,
            failed,
            0,
            0,
            0,
            0,
            0,
            sweep_evals=sweep_evals,
        )

    unsat_o = 0
    unk_o = 0
    sat_o = 0
    cegar_refs = 0
    first_mm: Optional[str] = None

    for name, lhs, rhs, guard in checks:
        rc, cex = _solve_miter_eq(lhs, rhs, guard, output_timeout_ms, env, preconditions=preconditions)
        if rc == "SAT" and d_cut_map:
            local_d = dict(d_cut_map)
            local_v = dict(v_cut_map)
            cur_l, cur_r, cur_g = lhs, rhs, guard
            for _ in range(max_cegar_rounds):
                used_cuts = _extract_cutpoint_vars(cur_l) | _extract_cutpoint_vars(cur_r)
                if cur_g is not None:
                    used_cuts |= _extract_cutpoint_vars(cur_g)
                if not used_cuts:
                    break
                cegar_refs += 1
                for cn in used_cuts:
                    pair = cut_name_to_pair.get(cn)
                    if pair:
                        local_d.pop(pair[0], None)
                        local_v.pop(pair[1], None)
                ref_checks = dict((n, (l, r, g)) for n, l, r, g in _build_output_checks(local_d, local_v))
                if name not in ref_checks:
                    break
                cur_l, cur_r, cur_g = ref_checks[name]
                rc, cex = _solve_miter_eq(cur_l, cur_r, cur_g, output_timeout_ms, env, preconditions=preconditions)
                if rc != "SAT":
                    break
        if rc == "UNSAT":
            unsat_o += 1
        elif rc == "SAT":
            sat_o += 1
            if first_mm is None and cex is not None:
                first_mm = f"{name}: direct={cex[0]} valhall={cex[1]}"
        else:
            unk_o += 1

    final_ms = (time.perf_counter() - t_fin) * 1000.0
    if sat_o > 0:
        status = "SAT_SMT"
    elif unk_o > 0:
        status = "UNKNOWN"
    else:
        status = "UNSAT"

    return CecResult(
        status=status,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        concrete_ms=cdiff.elapsed_ms,
        sweep_ms=sweep_ms,
        final_ms=final_ms,
        concrete_status=cdiff.status,
        candidates_found=len(cands),
        cutpoints_proved=proved,
        cutpoints_failed=failed,
        cegar_refinements=cegar_refs,
        checked_outputs=len(checks),
        unsat_outputs=unsat_o,
        unknown_outputs=unk_o,
        sat_outputs=sat_o,
        first_mismatch=first_mm,
        sweep_evals=sweep_evals,
    )


verify_with_stratum2_cec = verify_with_cec

#!/usr/bin/env python3
"""
All-Path Z3 Formal Equivalence Verifier:
  Direct SPIR-V -> ValhallSSA (src/spirv2ssa/)
    vs.
  SPIR-V -> libMali-Gxx -> Valhall ISA -> valhall_pass1 -> ValhallSSA

Defeats combinatorial state explosion via:
  - Whole-CFG topological DAG path-merging (O(|V|+|E|) instead of O(2^branches))
  - Per-buffer disjoint address partitioning + structured constant-offset store maps
  - Eager constant folding & syntactic simplification before Z3 emission
  - Per-output miter decomposition with strict Z3 solver timeouts
  - Theory stratification: exact QF_BV/QF_FP with sound UF-FP fallback on large FP graphs
"""

from __future__ import annotations

import argparse
import dataclasses
import fractions
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import z3

from tools.verify.fp_proof_discharger import (
    FpObligation,
    FpProofDischargerRegistry,
)
from tools.verify.panfrost_z3_model import (
    BV8,
    BV32,
    BV64,
    FP32_SORT,
    _bitrev_bv32,
    _clz_bv,
    _fp_to_int_sat,
    _fp_total_le_bv,
    _hw_uf,
    _popcount_bv32,
    bv32,
    bv64,
    bv32_to_fp32,
    f16_bv_to_f32_bv,
    f32_bv_to_f16_bv,
    fp32_to_bv32,
)

from tools.coverage.evaluator import (
    BV8,
    BV32,
    BV64,
    FP32_SORT,
    REAL_SORT,
    BlockEffectState,
    DagSsaZ3Evaluator,
    SharedShaderEnv,
    Z3Val,
    _as_bool,
    _as_bv32,
    _as_bv64,
    _bv32_to_real,
    _clone_state,
    _eval_elementwise,
    _f32_bits_to_real_const,
    _hw_uf,
    _is_real2bv,
    _merge_states,
    _real_cos,
    _real_exp2,
    _real_log2,
    _real_sin,
    _real_sqrt,
    _real_to_bv32,
    _try_const_u64,
    _vec_len,
    _z3_and,
    _z3_if,
    _z3_not,
    _z3_or,
    bv32,
    bv32_to_fp32,
    bv64,
    f16_bv_to_f32_bv,
    f32_bv_to_f16_bv,
    fp32_to_bv32,
)

@dataclasses.dataclass
class OutputMismatch:
    output_name: str
    direct_hex: str
    valhall_hex: str
    model_summary: str
    suspect_pcs: List[Tuple[int, str]]


@dataclasses.dataclass
class ShaderEquivResult:
    spv_path: str
    stage: str
    status: str  # UNSAT, BOUNDED_UNSAT, SAT, UNKNOWN, SKIP, ERROR
    fp_mode: str
    elapsed_ms: float
    checked_outputs: List[str] = dataclasses.field(default_factory=list)
    mismatches: List[OutputMismatch] = dataclasses.field(default_factory=list)
    notes: List[str] = dataclasses.field(default_factory=list)
    direct_nodes: int = 0
    valhall_nodes: int = 0
    direct_blocks: int = 0
    valhall_blocks: int = 0
    has_loops: bool = False
    max_unroll_depth: int = 0
    fp_obligations: int = 0
    asymmetric_outputs: List[str] = dataclasses.field(default_factory=list)


def _solve_miter_eq(
    lhs: z3.ExprRef,
    rhs: z3.ExprRef,
    guard: Optional[z3.BoolRef],
    timeout_ms: int,
    env: SharedShaderEnv,
    preconditions: Optional[List[z3.BoolRef]] = None,
    smt2_out: Optional[List[str]] = None,
    check_name: str = "",
    verbose: bool = False,
) -> Tuple[str, Optional[Tuple[str, str, str]]]:
    """Check whether `(preconditions & guard) => (lhs == rhs)` holds for all inputs."""
    if _is_real2bv(lhs) or _is_real2bv(rhs):
        lhs = _bv32_to_real(lhs) if isinstance(lhs, z3.BitVecRef) else lhs
        rhs = _bv32_to_real(rhs) if isinstance(rhs, z3.BitVecRef) else rhs
    if lhs.eq(rhs):
        if smt2_out is not None:
            smt2_out.append(f"; --- {check_name}: syntactically identical (UNSAT) ---\n")
        return "UNSAT", None

    # Fast constant-simplification check before invoking SMT
    diff = z3.simplify(lhs != rhs)
    if z3.is_false(diff):
        if smt2_out is not None:
            smt2_out.append(f"; --- {check_name}: simplified to false (UNSAT) ---\n")
        return "UNSAT", None
    if guard is not None:
        query = z3.simplify(_z3_and(guard, diff))
        if z3.is_false(query):
            if smt2_out is not None:
                smt2_out.append(f"; --- {check_name}: guarded query simplified to false (UNSAT) ---\n")
            return "UNSAT", None
    else:
        query = diff

    s = z3.Solver()
    s.set("timeout", timeout_ms)
    s.set("rlimit", 2_000_000)
    if preconditions:
        for pre in preconditions:
            s.add(pre)
    s.add(query)
    if smt2_out is not None:
        smt2_out.append(f"; === Output Miter Check: {check_name} ===\n(push 1)\n{s.to_smt2()}(check-sat)\n(pop 1)\n")
    t_chk = time.perf_counter()
    rc = s.check()
    dt_chk = (time.perf_counter() - t_chk) * 1000.0
    if rc == z3.unsat:
        if verbose:
            print(f"[equiv][tier1] check={check_name} result=UNSAT elapsed={dt_chk:.1f}ms", file=sys.stderr)
        return "UNSAT", None
    if rc == z3.sat:
        if verbose:
            print(f"[equiv][tier1] check={check_name} result=SAT elapsed={dt_chk:.1f}ms", file=sys.stderr)
        m = s.model()
        lv = m.eval(lhs, model_completion=True)
        rv = m.eval(rhs, model_completion=True)
        lhex = f"0x{lv.as_long():08x}" if isinstance(lv, z3.BitVecNumRef) else str(lv)
        rhex = f"0x{rv.as_long():08x}" if isinstance(rv, z3.BitVecNumRef) else str(rv)
        in_parts = []
        for k, v in list(env.inputs.items())[:8]:
            ev = m.eval(v, model_completion=True)
            if isinstance(ev, z3.BitVecNumRef):
                in_parts.append(f"{k}=0x{ev.as_long():x}")
            else:
                in_parts.append(f"{k}={ev}")
        return "SAT", (lhex, rhex, ", ".join(in_parts))
    if verbose:
        reason = s.reason_unknown()
        lhs_kind = lhs.sort().kind() if hasattr(lhs, "sort") else "unknown"
        rhs_kind = rhs.sort().kind() if hasattr(rhs, "sort") else "unknown"
        print(
            f"[equiv][tier1] check={check_name} result=UNKNOWN reason='{reason}' "
            f"elapsed={dt_chk:.1f}ms timeout_ms={timeout_ms} rlimit=2000000 "
            f"lhs_sort={lhs_kind} rhs_sort={rhs_kind}",
            file=sys.stderr,
        )
    return "UNKNOWN", None


def verify_equivalence_bundle(
    bundle: Dict[str, Any],
    timeout_ms: int = 3000,
    fp_mode: str = "exact",
    allow_uf_fallback: bool = True,
    max_unroll: int = 1,
    dump_smt2_path: Optional[str] = None,
    fp_discharger: Optional[FpProofDischargerRegistry] = None,
    verbose: bool = False,
    use_stratum1: bool = False,
) -> ShaderEquivResult:
    """Verify all-path formal equivalence of direct_ssa and valhall_ssa in a JSON bundle."""
    t0 = time.perf_counter()
    spv_path = bundle.get("spv_path", "<memory>")
    stage = bundle.get("stage", "Unknown")
    direct_ssa = bundle.get("direct_ssa", {})
    valhall_ssa = bundle.get("valhall_ssa", {})
    if use_stratum1 and bundle.get("direct_ok", False) and bundle.get("valhall_ok", False):
        from tools.verify.stratum1_canon import canonicalize_ssa_module
        direct_ssa, _ = canonicalize_ssa_module(direct_ssa)
        valhall_ssa, _ = canonicalize_ssa_module(valhall_ssa)

    res = ShaderEquivResult(
        spv_path=spv_path,
        stage=stage,
        status="UNKNOWN",
        fp_mode=fp_mode,
        elapsed_ms=0.0,
        direct_nodes=len(direct_ssa.get("nodes", [])),
        valhall_nodes=len(valhall_ssa.get("nodes", [])),
        direct_blocks=len(direct_ssa.get("blocks", [])),
        valhall_blocks=len(valhall_ssa.get("blocks", [])),
    )

    if not bundle.get("direct_ok", False) or not bundle.get("valhall_ok", False):
        res.status = "ERROR"
        res.notes.append(f"Lowering failed: direct={bundle.get('direct_error', '')}, valhall={bundle.get('valhall_error', '')}")
        res.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return res

    env = SharedShaderEnv(bundle)
    def _collect_sym_bkeys(ssa_d: Dict[str, Any]) -> Set[str]:
        ks: Set[str] = set()
        for n in ssa_d.get("nodes", []):
            a = n.get("aux_str", "")
            if a.startswith("sym:"):
                ks.add(a[4:].split("[", 1)[0])
        return ks
    env.shared_sym_bkeys = _collect_sym_bkeys(bundle.get("direct_ssa", {})) & _collect_sym_bkeys(bundle.get("valhall_ssa", {}))
    d_eval = DagSsaZ3Evaluator(direct_ssa, env, is_valhall_side=False, fp_mode=fp_mode, max_unroll=max_unroll)
    v_eval = DagSsaZ3Evaluator(valhall_ssa, env, is_valhall_side=True, fp_mode=fp_mode, max_unroll=max_unroll)

    d_st = d_eval.evaluate()
    v_st = v_eval.evaluate()
    res.has_loops = d_eval.has_loops or v_eval.has_loops
    res.max_unroll_depth = max(d_eval.max_unroll_depth, v_eval.max_unroll_depth)
    preconditions = env.build_preconditions()
    pc_disasm: Dict[str, str] = bundle.get("mbs2_context", {}).get("pc_disasm", {})

    def lookup_pcs(key: str) -> List[Tuple[int, str]]:
        nids = v_st.output_pcs.get(key, set())
        pcs = v_eval.collect_pcs_for_nodes(nids) if nids else []
        return [(pc, pc_disasm.get(str(pc), "")) for pc in pcs[-16:]]

    checks: List[Tuple[str, z3.ExprRef, z3.ExprRef, Optional[z3.BoolRef], str]] = []

    # Discard equivalence (Fragment)
    if stage == "Fragment" or not z3.is_false(d_st.discarded) or not z3.is_false(v_st.discarded):
        checks.append(("discarded", d_st.discarded, v_st.discarded, None, "discarded"))

    not_disc = _z3_not(d_st.discarded)

    # Fragment color outputs (guarded by ~discarded)
    all_rts = sorted(set(d_st.frag_colors.keys()) | set(v_st.frag_colors.keys()))
    for rt in all_rts:
        dc = d_st.frag_colors.get(rt, (bv32(0),) * 4)
        vc = v_st.frag_colors.get(rt, (bv32(0),) * 4)
        for ch, cname in enumerate(("r", "g", "b", "a")):
            checks.append((f"frag_color[{rt}].{cname}", dc[ch], vc[ch], not_disc, f"frag_color[{rt}]"))

    # Builtin outputs (gl_Position, gl_FragDepth, etc.) — fail-closed on asymmetric presence
    all_bis = sorted(set(d_st.builtin_outputs.keys()) | set(v_st.builtin_outputs.keys()))
    for bi in all_bis:
        do = d_st.builtin_outputs.get(bi)
        vo = v_st.builtin_outputs.get(bi)
        if do is None or vo is None:
            side = "direct" if do is not None else "valhall"
            msg = f"Asymmetric builtin_out[{bi}]: emitted only by {side}"
            res.asymmetric_outputs.append(f"builtin_out[{bi}]")
            res.notes.append(msg)
            # Fail-closed check: require missing side to match (unless always discarded)
            checks.append((f"builtin_out[{bi}].present", z3.BoolVal(do is not None), z3.BoolVal(vo is not None), not_disc, f"builtin_out[{bi}]"))
            continue
        bi_guard = not_disc
        # TODO(leegao@): Replace identity viewport precondition with symbolic gl_mali_ViewportTransform epilogue on direct_ssa.
        # Under Valhall's folded gl_mali_ViewportTransform + perspective divide,
        # constrain clip.w == 1.0f alongside identity viewport (scale=1, bias=0) for gl_Position.
        if stage == "Vertex" and bi == 0 and env.viewport_fau_words and isinstance(do, tuple) and len(do) >= 4:
            if fp_mode == "real":
                vp_conds = [_bv32_to_real(_as_bv32(do[3])) == _REAL_ONE]
            else:
                vp_conds = [_as_bv32(do[3]) == bv32(0x3F800000)]
                for c in range(3):
                    dbv = _as_bv32(do[c])
                    # FMA(x, 1.0, +0.0) == x bit-for-bit iff x is not NaN and x != -0.0
                    vp_conds.append(z3.Not(z3.fpIsNaN(bv32_to_fp32(dbv))))
                    vp_conds.append(dbv != bv32(0x80000000))
            bi_guard = _z3_and(bi_guard, z3.And(*vp_conds))
        if isinstance(do, tuple) and isinstance(vo, tuple):
            for c in range(min(len(do), len(vo))):
                checks.append((f"builtin_out[{bi}][{c}]", _as_bv32(do[c]), _as_bv32(vo[c]), bi_guard, f"builtin_out[{bi}]"))
        else:
            checks.append((f"builtin_out[{bi}]", _as_bv32(do), _as_bv32(vo), bi_guard, f"builtin_out[{bi}]"))

    # Vertex varying outputs
    all_vv = sorted(set(d_st.vert_varyings.keys()) | set(v_st.vert_varyings.keys()))
    for loc in all_vv:
        if (loc in d_st.vert_varyings) != (loc in v_st.vert_varyings):
            side = "direct" if loc in d_st.vert_varyings else "valhall"
            res.asymmetric_outputs.append(f"vert_var[{loc}]")
            res.notes.append(f"Asymmetric vert_var[{loc}]: emitted only by {side}")
        dv = d_st.vert_varyings.get(loc, (bv32(0),) * 4)
        vv = v_st.vert_varyings.get(loc, (bv32(0),) * 4)
        for c in range(4):
            checks.append((f"vert_var[{loc}][{c}]", dv[c], vv[c], None, f"vert_var[{loc}]"))

    def _is_scratch_region(bk: str) -> bool:
        return bk in ("tls_private", "wls_shared") or bk.startswith("workgroup:")

    # Symbolized fixed-stride buffer stores (scalar/UF reasoning without Z3 Array theory)
    all_sym = sorted(set(d_st.sym_buf_stores.keys()) | set(v_st.sym_buf_stores.keys()))
    for stag in all_sym:
        dw, dv, doff = d_st.sym_buf_stores.get(stag, (z3.BoolVal(False), bv32(0), bv32(0)))
        vw, vv, voff = v_st.sym_buf_stores.get(stag, (z3.BoolVal(False), bv32(0), bv32(0)))
        checks.append((f"{stag}.written", dw, vw, None, stag))
        checks.append((f"{stag}.off", doff, voff, _z3_or(dw, vw), stag))
        checks.append((f"{stag}.value", dv, vv, _z3_or(dw, vw), stag))

    # Constant-offset buffer stores (SSBO writes; skip thread/workgroup-local scratch spills)
    all_cs = sorted(set(d_st.const_stores.keys()) | set(v_st.const_stores.keys()))
    for bkey, off in all_cs:
        if _is_scratch_region(bkey):
            continue
        dw, dv = d_st.const_stores.get((bkey, off), (z3.BoolVal(False), z3.Select(env.get_buf_mem(bkey), bv32(off))))
        vw, vv = v_st.const_stores.get((bkey, off), (z3.BoolVal(False), z3.Select(env.get_buf_mem(bkey), bv32(off))))
        tag = f"{bkey}+0x{off:x}"
        checks.append((f"{tag}.written", dw, vw, None, tag))
        checks.append((f"{tag}.value", dv, vv, _z3_or(dw, vw), tag))

    # Symbolic/dynamic-offset buffer stores (SSBO writes via Z3 Array theory)
    const_store_bkeys = {bk for bk, _ in all_cs}
    all_bm_keys = sorted(set(d_st.buf_mems.keys()) | set(v_st.buf_mems.keys()))
    for bkey in all_bm_keys:
        if _is_scratch_region(bkey):
            continue
        has_dyn = (f"{bkey}:dyn" in d_st.output_pcs) or (f"{bkey}:dyn" in v_st.output_pcs)
        if not has_dyn and bkey in const_store_bkeys:
            continue
        d_mem = d_st.buf_mems.get(bkey, env.get_buf_mem(bkey))
        v_mem = v_st.buf_mems.get(bkey, env.get_buf_mem(bkey))
        probe_name = f"probe_{bkey.replace(':', '_')}_off"
        probe_off = z3.BitVec(probe_name, 32)
        env.inputs[probe_name] = probe_off
        checks.append((f"{bkey}[probe_off]", z3.Select(d_mem, probe_off), z3.Select(v_mem, probe_off), None, f"{bkey}:dyn"))

    # Storage image writes (OpImageWrite / StoreTexel)
    all_im_keys = sorted(set(d_st.image_mems.keys()) | set(v_st.image_mems.keys()))
    for ikey in all_im_keys:
        d_im = d_st.image_mems.get(ikey, env.get_img_mem(ikey))
        v_im = v_st.image_mems.get(ikey, env.get_img_mem(ikey))
        probe_xy = z3.BitVec(f"probe_{ikey.replace(':', '_')}_xy", 64)
        env.inputs[f"probe_{ikey.replace(':', '_')}_xy"] = probe_xy
        for c, cname in enumerate(("r", "g", "b", "a")):
            checks.append((f"img:{ikey}[xy].{cname}", z3.Select(d_im[c], probe_xy), z3.Select(v_im[c], probe_xy), None, f"img:{ikey}"))

    # Ordered per-buffer atomic RMW event logs (prevents SKIP on atomic-only compute shaders)
    all_ae_keys = sorted(set(d_st.atomic_events.keys()) | set(v_st.atomic_events.keys()))
    for akey in all_ae_keys:
        if _is_scratch_region(akey):
            continue
        d_evs = d_st.atomic_events.get(akey, [])
        v_evs = v_st.atomic_events.get(akey, [])
        if len(d_evs) != len(v_evs):
            d_cnt = bv32(0)
            for de in d_evs:
                d_cnt = d_cnt + z3.If(de[0], bv32(1), bv32(0))
            v_cnt = bv32(0)
            for ve in v_evs:
                v_cnt = v_cnt + z3.If(ve[0], bv32(1), bv32(0))
            checks.append((f"atomic:{akey}.dyn_count", d_cnt, v_cnt, None, f"atomic:{akey}"))
        else:
            for idx in range(len(d_evs)):
                de, ve = d_evs[idx], v_evs[idx]
                checks.append((f"atomic:{akey}[{idx}].active", de[0], ve[0], None, f"atomic:{akey}"))
                ev_guard = _z3_or(de[0], ve[0])
                checks.append((f"atomic:{akey}[{idx}].addr", _as_bv64(de[2]), _as_bv64(ve[2]), ev_guard, f"atomic:{akey}"))
                checks.append((f"atomic:{akey}[{idx}].arg", de[3], ve[3], ev_guard, f"atomic:{akey}"))

    # Unresolved-address fallback memory writes (never silently skip)
    if not d_st.fallback_mem.eq(env.fallback_mem) or not v_st.fallback_mem.eq(env.fallback_mem):
        probe_addr = z3.BitVec("probe_fallback_addr", 64)
        env.inputs["probe_fallback_addr"] = probe_addr
        checks.append(("fallback_mem[probe_addr]", z3.Select(d_st.fallback_mem, probe_addr), z3.Select(v_st.fallback_mem, probe_addr), None, "fallback_mem"))

    if not checks:
        res.status = "SKIP"
        res.notes.append("No comparable observable outputs extracted")
        res.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return res

    smt2_chunks: Optional[List[str]] = [] if dump_smt2_path else None
    if smt2_chunks is not None:
        smt2_chunks.append(
            f"; Shader Equivalence Verification SMT-LIB2 Model\n"
            f"; Shader: {os.path.basename(spv_path)} | Stage: {stage} | FP Mode: {fp_mode}\n"
            f"; Direct SSA: {res.direct_blocks}B/{res.direct_nodes}N | Valhall SSA: {res.valhall_blocks}B/{res.valhall_nodes}N\n"
            f"; LocalSize: {env.local_size} | MaxUnroll: {max_unroll} | Preconditions: {len(preconditions)}\n"
            f"(set-logic ALL)\n\n"
        )

    if verbose:
        print(
            f"[equiv] start shader={os.path.basename(spv_path)} stage={stage} fp_mode={fp_mode} "
            f"direct=({res.direct_blocks}B/{res.direct_nodes}N) valhall=({res.valhall_blocks}B/{res.valhall_nodes}N) "
            f"checks={len(checks)} preconditions={len(preconditions)} sym_slots={len(getattr(env, 'sym_buf_slot_fns', {}))} "
            f"buf_arrays={len(getattr(env, 'buf_mems', {}))}",
            file=sys.stderr,
        )
    discharger = fp_discharger if fp_discharger is not None else FpProofDischargerRegistry()
    any_unknown = False
    for idx_chk, (name, lhs, rhs, guard, pc_key) in enumerate(checks):
        res.checked_outputs.append(name)
        st_rc, cex = _solve_miter_eq(
            lhs,
            rhs,
            guard,
            timeout_ms,
            env,
            preconditions=preconditions,
            smt2_out=smt2_chunks,
            check_name=name,
            verbose=verbose,
        )
        if st_rc == "SAT" and cex is not None:
            res.mismatches.append(
                OutputMismatch(
                    output_name=name,
                    direct_hex=cex[0],
                    valhall_hex=cex[1],
                    model_summary=cex[2],
                    suspect_pcs=lookup_pcs(pc_key),
                )
            )
        elif st_rc == "UNKNOWN":
            # Submit to Layer-2 pluggable FP proof-obligation discharger
            ob_pre = list(preconditions)
            if guard is not None:
                ob_pre.append(guard)
            ob = FpObligation(
                obligation_id=f"{os.path.basename(spv_path)}#{idx_chk}:{name}",
                output_name=name,
                kind="bit_exact_fp" if fp_mode == "exact" else "uf_refinement",
                lhs=lhs,
                rhs=rhs,
                preconditions=ob_pre,
                ulp_bound=0,
                source_hint=pc_key,
            )
            d_res = discharger.submit(ob, timeout_ms=min(timeout_ms, 1000), verbose=verbose)
            res.fp_obligations += 1
            if d_res.status == "DISCHARGED":
                res.notes.append(f"Layer-2 discharged {name} via {d_res.backend_name}")
            else:
                any_unknown = True

    if smt2_chunks is not None and dump_smt2_path:
        os.makedirs(os.path.dirname(os.path.abspath(dump_smt2_path)), exist_ok=True)
        with open(dump_smt2_path, "w", encoding="utf-8") as sf:
            sf.write("".join(smt2_chunks))

    if res.mismatches:
        res.status = "SAT"
    elif any_unknown:
        if fp_mode == "exact" and allow_uf_fallback:
            if verbose:
                print(
                    f"[equiv][fallback] shader={os.path.basename(spv_path)} exact FP left UNKNOWN; "
                    f"escalating to Tier-3 fp_mode=real",
                    file=sys.stderr,
                )
            real_res = verify_equivalence_bundle(
                bundle,
                timeout_ms=timeout_ms,
                fp_mode="real",
                allow_uf_fallback=False,
                max_unroll=max_unroll,
                dump_smt2_path=dump_smt2_path,
                fp_discharger=discharger,
                verbose=verbose,
            )
            if real_res.status in ("UNSAT", "BOUNDED_UNSAT"):
                real_res.fp_obligations = res.fp_obligations
                return real_res
            if verbose:
                print(
                    f"[equiv][fallback] Tier-3 fp_mode=real returned {real_res.status}; "
                    f"escalating to Tier-4 fp_mode=uf",
                    file=sys.stderr,
                )
            uf_res = verify_equivalence_bundle(
                bundle,
                timeout_ms=timeout_ms,
                fp_mode="uf",
                allow_uf_fallback=False,
                max_unroll=max_unroll,
                dump_smt2_path=dump_smt2_path,
                fp_discharger=discharger,
                verbose=verbose,
            )
            # UF is an over-approximation: UNSAT/BOUNDED_UNSAT is sound, but UF SAT may be spurious.
            if uf_res.status in ("UNSAT", "BOUNDED_UNSAT"):
                uf_res.fp_obligations = res.fp_obligations
                return uf_res
            res.notes.append(f"Exact FP timed out; real ({real_res.status}) and UF ({uf_res.status}) fallbacks could not prove equivalence")
        res.status = "UNKNOWN"
    else:
        res.status = "BOUNDED_UNSAT" if res.has_loops else "UNSAT"

    res.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return res


def verify_spv_file(
    spv_path: str,
    spirv2valhall_bin: str = "./build/spirv2valhall",
    timeout_ms: int = 3000,
    fp_mode: str = "exact",
    max_unroll: int = 1,
    dump_smt2_path: Optional[str] = None,
    verbose: bool = False,
    use_stratum1: bool = False,
) -> ShaderEquivResult:
    """Compile a .spv through both pipelines and run the Z3 equivalence verifier."""
    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tmp_json = tf.name
    try:
        proc = subprocess.run(
            [spirv2valhall_bin, spv_path, "--quiet", "--dump-equiv-json", tmp_json],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode != 0:
            return ShaderEquivResult(
                spv_path=spv_path,
                stage="Unknown",
                status="ERROR",
                fp_mode=fp_mode,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                notes=[proc.stderr.strip()[:200]],
            )
        with open(tmp_json, "r", encoding="utf-8") as fh:
            bundle = json.load(fh)
        return verify_equivalence_bundle(bundle, timeout_ms=timeout_ms, fp_mode=fp_mode, max_unroll=max_unroll, dump_smt2_path=dump_smt2_path, verbose=verbose, use_stratum1=use_stratum1)
    finally:
        if os.path.exists(tmp_json):
            os.remove(tmp_json)


def main() -> int:
    ap = argparse.ArgumentParser(description="Z3 All-Path Formal Equivalence Verifier for ValhallSSA")
    ap.add_argument("spv_or_json", help="Path to .spv shader or pre-dumped equivalence bundle .json")
    ap.add_argument("--bin", default="./build/spirv2valhall", help="Path to spirv2valhall binary")
    ap.add_argument("--timeout-ms", type=int, default=3000, help="Per-output Z3 timeout in ms")
    ap.add_argument("--fp-mode", choices=["exact", "real", "uf"], default="exact")
    ap.add_argument("--max-unroll", type=int, default=1, help="Bounded loop unroll depth (default: 1)")
    ap.add_argument("--dump-smt2", default=None, help="Write Z3 SMT-LIB2 miter queries to this file")
    ap.add_argument("--stratum1", action="store_true", help="Apply Stratum 1 pre-SMT canonicalization before Z3")
    ap.add_argument("--stratum2", action="store_true", help="Run Stratum 2 concrete diff + CEC cutpoint sweeping")
    ap.add_argument("-v", "--verbose", action="store_true", help="Emit detailed tier-by-tier Z3/Layer-2 diagnostics to stderr")
    args = ap.parse_args()

    if args.stratum2:
        from tools.verify.stratum2_cec import verify_with_stratum2_cec
        if args.spv_or_json.endswith(".json"):
            with open(args.spv_or_json, "r", encoding="utf-8") as fh:
                bundle = json.load(fh)
        else:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
                tmp_json = tf.name
            try:
                proc = subprocess.run(
                    [args.bin, args.spv_or_json, "--quiet", "--dump-equiv-json", tmp_json],
                    capture_output=True, text=True, timeout=15,
                )
                if proc.returncode != 0:
                    print(f"[ERROR] {os.path.basename(args.spv_or_json)}: {proc.stderr.strip()[:200]}")
                    return 1
                with open(tmp_json, "r", encoding="utf-8") as fh:
                    bundle = json.load(fh)
            finally:
                if os.path.exists(tmp_json):
                    os.remove(tmp_json)
        s2r = verify_with_stratum2_cec(
            bundle,
            use_stratum1=True,
            output_timeout_ms=args.timeout_ms,
        )
        print(
            f"[{s2r.status}] {os.path.basename(bundle.get('spv_path', args.spv_or_json))} "
            f"stage={bundle.get('stage', 'Unknown')} "
            f"concrete={s2r.concrete_status}({s2r.concrete_ms:.1f}ms) "
            f"cutpoints={s2r.cutpoints_proved}/{s2r.candidates_found} "
            f"cegar={s2r.cegar_refinements} elapsed={s2r.elapsed_ms:.1f}ms"
        )
        return 0 if s2r.status in ("UNSAT", "BOUNDED_UNSAT") else 1

    if args.spv_or_json.endswith(".json"):
        with open(args.spv_or_json, "r", encoding="utf-8") as fh:
            bundle = json.load(fh)
        r = verify_equivalence_bundle(
            bundle,
            timeout_ms=args.timeout_ms,
            fp_mode=args.fp_mode,
            max_unroll=args.max_unroll,
            dump_smt2_path=args.dump_smt2,
            verbose=args.verbose,
            use_stratum1=args.stratum1,
        )
    else:
        r = verify_spv_file(
            args.spv_or_json,
            spirv2valhall_bin=args.bin,
            timeout_ms=args.timeout_ms,
            fp_mode=args.fp_mode,
            max_unroll=args.max_unroll,
            dump_smt2_path=args.dump_smt2,
            verbose=args.verbose,
            use_stratum1=args.stratum1,
        )

    print(f"[{r.status}] {os.path.basename(r.spv_path)} stage={r.stage} fp={r.fp_mode} "
          f"outputs={len(r.checked_outputs)} unroll={r.max_unroll_depth} "
          f"direct=({r.direct_blocks}B/{r.direct_nodes}N) "
          f"valhall=({r.valhall_blocks}B/{r.valhall_nodes}N) elapsed={r.elapsed_ms:.1f}ms")
    for mm in r.mismatches[:5]:
        print(f"  MISMATCH {mm.output_name}: direct={mm.direct_hex} valhall={mm.valhall_hex} ({mm.model_summary})")
        for pc, dis in mm.suspect_pcs[:5]:
            print(f"    @0x{pc:04x}: {dis}")
    return 0 if r.status in ("UNSAT", "BOUNDED_UNSAT") else 1


if __name__ == "__main__":
    sys.exit(main())

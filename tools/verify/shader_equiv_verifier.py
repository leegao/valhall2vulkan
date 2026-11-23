#!/usr/bin/env python3
"""
All-Path Z3 Formal Equivalence Verifier:
  Direct SPIR-V -> ValhallSSA (src/spirv2ssa/)
    vs.
  SPIR-V -> libMali-Gxx -> Valhall ISA -> valhall_pass1 -> ValhallSSA

Defeats combinatorial state explosion via:
  - Whole-CFG topological DAG path-merging (O(|V|+|E|) instead of O(2^branches))
  - Per-buffer disjoint address partitioning + structured constant-offset store maps
  - Unified lattice-typed symbolic domain (`udomain` default) avoiding FP bit-blasting
  - Modular Z3 term/predicate simplification (`tools.symbolic_ssa.z3simplifier`)
  - Two-phase iterative-deepening per-channel solver budgeting
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import z3

from tools.symbolic_ssa.z3 import dsl

from tools.verify.panfrost_z3_model import (
    BV8,
    BV32,
    BV64,
    FP32_SORT,
    bv32,
    bv32_to_fp32,
    bv64,
)

from tools.symbolic_ssa import (
    BlockEffectState,
    DagSsaZ3Evaluator,
    SharedShaderEnv,
    _as_bool,
    _as_bv32,
    _as_bv64,
    _bv32_to_real,
    _is_real2bv,
    _z3_and,
    _z3_not,
    _z3_or,
)
from tools.symbolic_ssa.z3simplifier import (
    PassPipeline,
    SimplificationContext,
    SimplifyQuery,
    build_default_pipeline,
    census_ast,
    format_z3_ast,
    solve_simplified_miter,
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
    asymmetric_outputs: List[str] = dataclasses.field(default_factory=list)


def _format_model_cex(
    m: z3.ModelRef,
    lhs: z3.ExprRef,
    rhs: z3.ExprRef,
    env: SharedShaderEnv,
) -> Tuple[str, str, str]:
    lv = m.eval(dsl.lower(lhs), model_completion=True)
    rv = m.eval(dsl.lower(rhs), model_completion=True)
    lhex = f"0x{lv.as_long():08x}" if isinstance(lv, z3.BitVecNumRef) else str(lv)
    rhex = f"0x{rv.as_long():08x}" if isinstance(rv, z3.BitVecNumRef) else str(rv)
    in_parts = []
    for k, v in list(env.inputs.items())[:8]:
        ev = m.eval(dsl.lower(v), model_completion=True)
        if isinstance(ev, z3.BitVecNumRef):
            in_parts.append(f"{k}=0x{ev.as_long():x}")
        else:
            in_parts.append(f"{k}={ev}")
    return lhex, rhex, ", ".join(in_parts)


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
    pipeline: Optional[PassPipeline] = None,
    enable_conditioned_unify: bool = False,
) -> Tuple[str, Optional[Tuple[str, str, str]]]:
    """Check whether `(preconditions & guard) => (lhs == rhs)` holds for all inputs."""
    if _is_real2bv(lhs) or _is_real2bv(rhs):
        lhs = _bv32_to_real(lhs) if dsl.is_bv(lhs) else lhs
        rhs = _bv32_to_real(rhs) if dsl.is_bv(rhs) else rhs

    q = SimplifyQuery(
        lhs=lhs,
        rhs=rhs,
        guard=guard,
        preconditions=tuple(preconditions) if preconditions else (),
    )
    pipe = (
        pipeline
        if pipeline is not None
        else build_default_pipeline(enable_conditioned_unify=enable_conditioned_unify)
    )
    ctx = SimplificationContext(
        verbose=verbose,
        trace_fn=(lambda msg: print(msg, file=sys.stderr)) if verbose else None,
    )
    res = solve_simplified_miter(
        q,
        pipe,
        timeout_ms=timeout_ms,
        rlimit=None,
        ctx=ctx,
        smt2_out=smt2_out,
        check_name=check_name,
    )

    if res.status == "UNSAT":
        if verbose:
            print(
                f"[equiv] check={check_name} result=UNSAT elapsed={res.solver_ms:.1f}ms "
                f"nlsat={res.used_nlsat} simp=({res.stats.summary_line()})",
                file=sys.stderr,
            )
        return "UNSAT", None

    if res.status == "SAT" and res.model is not None:
        if verbose:
            print(
                f"[equiv] check={check_name} result=SAT elapsed={res.solver_ms:.1f}ms "
                f"refined={res.refined_from_overapprox} simp=({res.stats.summary_line()})",
                file=sys.stderr,
            )
        return "SAT", _format_model_cex(res.model, lhs, rhs, env)

    if verbose:
        lhs_kind = lhs.sort().kind() if hasattr(lhs, "sort") else "unknown"
        rhs_kind = rhs.sort().kind() if hasattr(rhs, "sort") else "unknown"
        print(
            f"[equiv] check={check_name} result=UNKNOWN reason='{res.reason_unknown}' "
            f"elapsed={res.solver_ms:.1f}ms timeout_ms={timeout_ms} "
            f"nlsat={res.used_nlsat} lhs_sort={lhs_kind} rhs_sort={rhs_kind} "
            f"simp=({res.stats.summary_line()})",
            file=sys.stderr,
        )
    return "UNKNOWN", None


def _dump_check_asts(
    checks: List[Tuple[str, z3.ExprRef, z3.ExprRef, Optional[z3.BoolRef], str]],
    preconditions: List[z3.BoolRef],
    pipeline: PassPipeline,
    stream: Any,
) -> None:
    """Pretty-print the pre- and post-simplification Z3 ASTs for every output check."""
    print(f"=== AST Dump ({len(checks)} checks, {len(preconditions)} preconditions) ===", file=stream)
    for idx, pre in enumerate(preconditions):
        print(f"  [precondition {idx}] {format_z3_ast(pre, max_depth=8, max_lines=20)}", file=stream)
    for idx, (name, lhs, rhs, guard, _pc_key) in enumerate(checks):
        q = SimplifyQuery(lhs=lhs, rhs=rhs, guard=guard, preconditions=tuple(preconditions))
        raw_c = census_ast([lhs, rhs] + ([guard] if guard is not None else []))
        sim = pipeline.run(q, SimplificationContext())
        sq = sim.query
        simp_c = census_ast([sq.lhs, sq.rhs] + ([sq.guard] if sq.guard is not None else []))
        print(f"\n--- Check #{idx}: {name} ---", file=stream)
        print(
            f"  Raw census:  nodes={raw_c.total_nodes} depth={raw_c.max_depth} "
            f"FP={raw_c.fp_ops} NLinR={raw_c.nlin_real_ops} BVMul={raw_c.bv_mul_shift_ops} "
            f"ArrSel={raw_c.array_select_ops} ArrSt={raw_c.array_store_ops} "
            f"Bridge={raw_c.bridge_ops} BoolWrap={raw_c.bool_wrapper_ops} IntBV={raw_c.int_bv_ops}",
            file=stream,
        )
        print(
            f"  Simp census: nodes={simp_c.total_nodes} depth={simp_c.max_depth} "
            f"FP={simp_c.fp_ops} NLinR={simp_c.nlin_real_ops} BVMul={simp_c.bv_mul_shift_ops} "
            f"ArrSel={simp_c.array_select_ops} Bridge={simp_c.bridge_ops} IntBV={simp_c.int_bv_ops} "
            f"theory={sq.pure_theory or 'ALL'} trivial_unsat={sq.is_trivially_unsat()}",
            file=stream,
        )
        print(f"  Simplifier:  {sim.stats.summary_line()}", file=stream)
        if guard is not None:
            print("  Guard (raw):", file=stream)
            print("    " + format_z3_ast(guard, max_depth=8, max_lines=24).replace("\n", "\n    "), file=stream)
        print(f"  LHS (direct, sort={lhs.sort()}):", file=stream)
        print("    " + format_z3_ast(lhs, max_depth=10, max_lines=40).replace("\n", "\n    "), file=stream)
        print(f"  RHS (valhall, sort={rhs.sort()}):", file=stream)
        print("    " + format_z3_ast(rhs, max_depth=10, max_lines=40).replace("\n", "\n    "), file=stream)
        if not dsl.expr_eq(sq.lhs, lhs) or not dsl.expr_eq(sq.rhs, rhs):
            print(f"  LHS (simplified, sort={sq.lhs.sort()}):", file=stream)
            print("    " + format_z3_ast(sq.lhs, max_depth=10, max_lines=30).replace("\n", "\n    "), file=stream)
            print(f"  RHS (simplified, sort={sq.rhs.sort()}):", file=stream)
            print("    " + format_z3_ast(sq.rhs, max_depth=10, max_lines=30).replace("\n", "\n    "), file=stream)


def verify_equivalence_bundle(
    bundle: Dict[str, Any],
    timeout_ms: int = 3000,
    fp_mode: str = "exact",
    allow_uf_fallback: bool = True,
    max_unroll: int = 1,
    dump_smt2_path: Optional[str] = None,
    verbose: bool = False,
    canonicalize: Optional[bool] = None,
    domain: str = "udomain",
    dump_ast: bool = False,
    dump_ast_stream: Any = None,
    enable_conditioned_unify: bool = False,
) -> ShaderEquivResult:
    """Verify all-path formal equivalence of direct_ssa and valhall_ssa in a JSON bundle."""
    t0 = time.perf_counter()
    spv_path = bundle.get("spv_path", "<memory>")
    stage = bundle.get("stage", "Unknown")
    direct_ssa = bundle.get("direct_ssa", {})
    valhall_ssa = bundle.get("valhall_ssa", {})

    # Auto-enable Stratum-1 canonicalization on large/branchy shaders when canonicalize is None.
    d_n0 = len(direct_ssa.get("nodes", []))
    v_n0 = len(valhall_ssa.get("nodes", []))
    d_b0 = len(direct_ssa.get("blocks", []))
    v_b0 = len(valhall_ssa.get("blocks", []))
    do_canon = (
        canonicalize
        if canonicalize is not None
        else (max(d_n0, v_n0) > 500 or max(d_b0, v_b0) > 4)
    )
    if do_canon and bundle.get("direct_ok", False) and bundle.get("valhall_ok", False):
        from tools.symbolic_ssa.passes import canonicalize_ssa_module
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

    env = SharedShaderEnv(bundle, domain=domain, fp_mode=fp_mode)

    def _collect_sym_bkeys(ssa_d: Dict[str, Any]) -> Set[str]:
        ks: Set[str] = set()
        for n in ssa_d.get("nodes", []):
            a = n.get("aux_str", "")
            if a.startswith("sym:"):
                ks.add(a[4:].split("[", 1)[0])
        return ks

    env.shared_sym_bkeys = _collect_sym_bkeys(bundle.get("direct_ssa", {})) & _collect_sym_bkeys(bundle.get("valhall_ssa", {}))
    d_eval = DagSsaZ3Evaluator(direct_ssa, env, is_valhall_side=False, fp_mode=fp_mode, max_unroll=max_unroll, domain=domain)
    v_eval = DagSsaZ3Evaluator(valhall_ssa, env, is_valhall_side=True, fp_mode=fp_mode, max_unroll=max_unroll, domain=domain)

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

    if stage == "Fragment" or not dsl.is_false(d_st.discarded) or not dsl.is_false(v_st.discarded):
        checks.append(("discarded", d_st.discarded, v_st.discarded, None, "discarded"))

    not_disc = _z3_not(d_st.discarded)

    all_rts = sorted(set(d_st.frag_colors.keys()) | set(v_st.frag_colors.keys()))
    for rt in all_rts:
        dc = d_st.frag_colors.get(rt, (bv32(0),) * 4)
        vc = v_st.frag_colors.get(rt, (bv32(0),) * 4)
        for ch, cname in enumerate(("r", "g", "b", "a")):
            checks.append((f"frag_color[{rt}].{cname}", dc[ch], vc[ch], not_disc, f"frag_color[{rt}]"))

    all_bis = sorted(set(d_st.builtin_outputs.keys()) | set(v_st.builtin_outputs.keys()))
    for bi in all_bis:
        do = d_st.builtin_outputs.get(bi)
        vo = v_st.builtin_outputs.get(bi)
        if do is None or vo is None:
            side = "direct" if do is not None else "valhall"
            msg = f"Asymmetric builtin_out[{bi}]: emitted only by {side}"
            res.asymmetric_outputs.append(f"builtin_out[{bi}]")
            res.notes.append(msg)
            checks.append((f"builtin_out[{bi}].present", dsl.BoolVal(do is not None), dsl.BoolVal(vo is not None), not_disc, f"builtin_out[{bi}]"))
            continue
        bi_guard = not_disc
        # Under Valhall's folded gl_mali_ViewportTransform + perspective divide,
        # constrain clip.w == 1.0f alongside identity viewport (scale=1, bias=0) for gl_Position.
        if stage == "Vertex" and bi == 0 and env.viewport_fau_words and isinstance(do, tuple) and len(do) >= 4:
            if fp_mode == "real" or domain == "udomain":
                vp_conds = [d_eval.domain.to_z3_real_f32(do[3]) == dsl.RealVal(1)]
            else:
                vp_conds = [_as_bv32(do[3]) == bv32(0x3F800000)]
                for c in range(3):
                    dbv = _as_bv32(do[c])
                    vp_conds.append(dsl.Not(dsl.fpIsNaN(bv32_to_fp32(dsl.lower(dbv)))))
                    vp_conds.append(dbv != bv32(0x80000000))
            bi_guard = _z3_and(bi_guard, dsl.And(*vp_conds))
        if isinstance(do, tuple) and isinstance(vo, tuple):
            for c in range(min(len(do), len(vo))):
                checks.append((f"builtin_out[{bi}][{c}]", _as_bv32(do[c]), _as_bv32(vo[c]), bi_guard, f"builtin_out[{bi}]"))
        else:
            checks.append((f"builtin_out[{bi}]", _as_bv32(do), _as_bv32(vo), bi_guard, f"builtin_out[{bi}]"))

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

    all_sym = sorted(set(d_st.sym_buf_stores.keys()) | set(v_st.sym_buf_stores.keys()))
    for stag in all_sym:
        dw, dv, doff = d_st.sym_buf_stores.get(stag, (dsl.BoolVal(False), bv32(0), bv32(0)))
        vw, vv, voff = v_st.sym_buf_stores.get(stag, (dsl.BoolVal(False), bv32(0), bv32(0)))
        checks.append((f"{stag}.written", dw, vw, None, stag))
        checks.append((f"{stag}.off", doff, voff, _z3_or(dw, vw), stag))
        checks.append((f"{stag}.value", dv, vv, _z3_or(dw, vw), stag))

    all_cs = sorted(set(d_st.const_stores.keys()) | set(v_st.const_stores.keys()))
    for bkey, off in all_cs:
        if _is_scratch_region(bkey):
            continue
        dw, dv = d_st.const_stores.get((bkey, off), (dsl.BoolVal(False), dsl.Select(env.get_buf_mem(bkey), bv32(off))))
        vw, vv = v_st.const_stores.get((bkey, off), (dsl.BoolVal(False), dsl.Select(env.get_buf_mem(bkey), bv32(off))))
        tag = f"{bkey}+0x{off:x}"
        checks.append((f"{tag}.written", dw, vw, None, tag))
        checks.append((f"{tag}.value", dv, vv, _z3_or(dw, vw), tag))

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
        probe_off = dsl.BitVec(probe_name, 32)
        env.inputs[probe_name] = probe_off
        checks.append((f"{bkey}[probe_off]", dsl.Select(d_mem, probe_off), dsl.Select(v_mem, probe_off), None, f"{bkey}:dyn"))

    all_im_keys = sorted(set(d_st.image_mems.keys()) | set(v_st.image_mems.keys()))
    for ikey in all_im_keys:
        d_im = d_st.image_mems.get(ikey, env.get_img_mem(ikey))
        v_im = v_st.image_mems.get(ikey, env.get_img_mem(ikey))
        probe_xy = dsl.BitVec(f"probe_{ikey.replace(':', '_')}_xy", 64)
        env.inputs[f"probe_{ikey.replace(':', '_')}_xy"] = probe_xy
        for c, cname in enumerate(("r", "g", "b", "a")):
            checks.append((f"img:{ikey}[xy].{cname}", dsl.Select(d_im[c], probe_xy), dsl.Select(v_im[c], probe_xy), None, f"img:{ikey}"))

    all_ae_keys = sorted(set(d_st.atomic_events.keys()) | set(v_st.atomic_events.keys()))
    for akey in all_ae_keys:
        if _is_scratch_region(akey):
            continue
        d_evs = d_st.atomic_events.get(akey, [])
        v_evs = v_st.atomic_events.get(akey, [])
        if len(d_evs) != len(v_evs):
            d_cnt = bv32(0)
            for de in d_evs:
                d_cnt = d_cnt + dsl.If(de[0], bv32(1), bv32(0))
            v_cnt = bv32(0)
            for ve in v_evs:
                v_cnt = v_cnt + dsl.If(ve[0], bv32(1), bv32(0))
            checks.append((f"atomic:{akey}.dyn_count", d_cnt, v_cnt, None, f"atomic:{akey}"))
        else:
            for idx in range(len(d_evs)):
                de, ve = d_evs[idx], v_evs[idx]
                checks.append((f"atomic:{akey}[{idx}].active", de[0], ve[0], None, f"atomic:{akey}"))
                ev_guard = _z3_or(de[0], ve[0])
                checks.append((f"atomic:{akey}[{idx}].addr", _as_bv64(de[2]), _as_bv64(ve[2]), ev_guard, f"atomic:{akey}"))
                checks.append((f"atomic:{akey}[{idx}].arg", de[3], ve[3], ev_guard, f"atomic:{akey}"))

    if not d_st.fallback_mem.eq(env.fallback_mem) or not v_st.fallback_mem.eq(env.fallback_mem):
        probe_addr = dsl.BitVec("probe_fallback_addr", 64)
        env.inputs["probe_fallback_addr"] = probe_addr
        checks.append(("fallback_mem[probe_addr]", dsl.Select(d_st.fallback_mem, probe_addr), dsl.Select(v_st.fallback_mem, probe_addr), None, "fallback_mem"))

    if not checks:
        res.status = "SKIP"
        res.notes.append("No comparable observable outputs extracted")
        res.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return res

    pipeline = build_default_pipeline(enable_conditioned_unify=enable_conditioned_unify)
    if dump_ast:
        _dump_check_asts(checks, preconditions, pipeline, dump_ast_stream or sys.stdout)

    smt2_chunks: Optional[List[str]] = [] if dump_smt2_path else None
    if smt2_chunks is not None:
        smt2_chunks.append(
            f"; Shader Equivalence Verification SMT-LIB2 Model\n"
            f"; Shader: {os.path.basename(spv_path)} | Stage: {stage} | Domain: {domain} | FP Mode: {fp_mode}\n"
            f"; Direct SSA: {res.direct_blocks}B/{res.direct_nodes}N | Valhall SSA: {res.valhall_blocks}B/{res.valhall_nodes}N\n"
            f"; LocalSize: {env.local_size} | MaxUnroll: {max_unroll} | Preconditions: {len(preconditions)}\n"
            f"(set-logic ALL)\n\n"
        )

    if verbose:
        print(
            f"[equiv] start shader={os.path.basename(spv_path)} stage={stage} domain={domain} fp_mode={fp_mode} "
            f"canon={do_canon} direct=({res.direct_blocks}B/{res.direct_nodes}N) "
            f"valhall=({res.valhall_blocks}B/{res.valhall_nodes}N) "
            f"checks={len(checks)} preconditions={len(preconditions)} "
            f"sym_slots={len(getattr(env, 'sym_buf_slot_fns', {}))} buf_arrays={len(getattr(env, 'buf_mems', {}))}",
            file=sys.stderr,
        )

    # Two-phase iterative-deepening channel scheduler:
    # Phase 1 sweeps all channels with a fast cap so multi-channel shaders (25-41
    # outputs) never starve a fast SAT channel behind earlier UNKNOWN channels.
    # Phase 2 allocates the full timeout_ms only to surviving UNKNOWN channels.
    use_two_phase = len(checks) > 4 and timeout_ms > 150
    fast_ms = min(100, max(30, timeout_ms // 4)) if use_two_phase else timeout_ms

    surviving_unknowns: List[Tuple[str, z3.ExprRef, z3.ExprRef, Optional[z3.BoolRef], str]] = []
    for name, lhs, rhs, guard, pc_key in checks:
        res.checked_outputs.append(name)
        st_rc, cex = _solve_miter_eq(
            lhs,
            rhs,
            guard,
            fast_ms,
            env,
            preconditions=preconditions,
            smt2_out=smt2_chunks if not use_two_phase else None,
            check_name=name,
            verbose=verbose,
            pipeline=pipeline,
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
            if use_two_phase:
                break
        elif st_rc == "UNKNOWN":
            surviving_unknowns.append((name, lhs, rhs, guard, pc_key))

    any_unknown = False
    if not res.mismatches and surviving_unknowns:
        if use_two_phase:
            for name, lhs, rhs, guard, pc_key in surviving_unknowns:
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
                    pipeline=pipeline,
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
                    break
                elif st_rc == "UNKNOWN":
                    any_unknown = True
        else:
            any_unknown = True

    if smt2_chunks is not None and dump_smt2_path:
        os.makedirs(os.path.dirname(os.path.abspath(dump_smt2_path)), exist_ok=True)
        with open(dump_smt2_path, "w", encoding="utf-8") as sf:
            sf.write("".join(smt2_chunks))

    if res.mismatches:
        res.status = "SAT"
    elif any_unknown:
        if domain == "bitvec" and fp_mode == "exact" and allow_uf_fallback:
            if verbose:
                print(
                    f"[equiv][fallback] shader={os.path.basename(spv_path)} exact FP left UNKNOWN; "
                    f"escalating to fp_mode=real",
                    file=sys.stderr,
                )
            real_res = verify_equivalence_bundle(
                bundle,
                timeout_ms=timeout_ms,
                fp_mode="real",
                allow_uf_fallback=False,
                domain=domain,
                max_unroll=max_unroll,
                dump_smt2_path=dump_smt2_path,
                verbose=verbose,
                canonicalize=do_canon,
            )
            if real_res.status in ("UNSAT", "BOUNDED_UNSAT"):
                return real_res
            uf_res = verify_equivalence_bundle(
                bundle,
                timeout_ms=timeout_ms,
                fp_mode="uf",
                allow_uf_fallback=False,
                domain=domain,
                max_unroll=max_unroll,
                dump_smt2_path=dump_smt2_path,
                verbose=verbose,
                canonicalize=do_canon,
            )
            if uf_res.status in ("UNSAT", "BOUNDED_UNSAT"):
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
    canonicalize: Optional[bool] = None,
    domain: str = "udomain",
    dump_ast: bool = False,
    enable_conditioned_unify: bool = False,
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
        return verify_equivalence_bundle(
            bundle,
            timeout_ms=timeout_ms,
            fp_mode=fp_mode,
            max_unroll=max_unroll,
            dump_smt2_path=dump_smt2_path,
            verbose=verbose,
            canonicalize=canonicalize,
            domain=domain,
            dump_ast=dump_ast,
            enable_conditioned_unify=enable_conditioned_unify,
        )
    finally:
        if os.path.exists(tmp_json):
            os.remove(tmp_json)


def main() -> int:
    ap = argparse.ArgumentParser(description="Z3 All-Path Formal Equivalence Verifier for ValhallSSA")
    ap.add_argument("spv_or_json", help="Path to .spv shader or pre-dumped equivalence bundle .json")
    ap.add_argument("--bin", default="./build/spirv2valhall", help="Path to spirv2valhall binary")
    ap.add_argument("--timeout-ms", type=int, default=3000, help="Per-output Z3 timeout in ms")
    ap.add_argument("--domain", choices=["bitvec", "udomain", "udomain_exact"], default="udomain", help="Symbolic value domain (default: udomain)")
    ap.add_argument("--fp-mode", choices=["exact", "real", "uf"], default="exact")
    ap.add_argument("--max-unroll", type=int, default=1, help="Bounded loop unroll depth (default: 1)")
    ap.add_argument("--dump-smt2", default=None, help="Write Z3 SMT-LIB2 miter queries to this file")
    ap.add_argument("--dump-ast", action="store_true", help="Pretty-print raw and simplified Z3 ASTs for every checked output")
    ap.add_argument("--canonicalize", action="store_true", default=None, help="Force Stratum-1 structural canonicalization before Z3")
    ap.add_argument("--no-canonicalize", action="store_false", dest="canonicalize", help="Disable automatic Stratum-1 canonicalization")
    ap.add_argument("--conditioned-unify", action="store_true", help="Enable WLP-justified conditioned subexpression unification")
    ap.add_argument("--cec", action="store_true", help="Run concrete diff + incremental CEC cutpoint sweeping")
    ap.add_argument("-v", "--verbose", action="store_true", help="Emit detailed Z3/simplifier diagnostics to stderr")
    args = ap.parse_args()

    if args.cec:
        from tools.verify.cec import verify_with_cec
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
        s2r = verify_with_cec(
            bundle,
            canonicalize=True if args.canonicalize is None else args.canonicalize,
            output_timeout_ms=args.timeout_ms,
            domain=args.domain,
            fp_mode=args.fp_mode,
        )
        print(
            f"[{s2r.status}] {os.path.basename(bundle.get('spv_path', args.spv_or_json))} "
            f"stage={bundle.get('stage', 'Unknown')} domain={args.domain} "
            f"concrete={s2r.concrete_status}({s2r.concrete_ms:.1f}ms) "
            f"cutpoints={s2r.cutpoints_proved}/{s2r.candidates_found} "
            f"cegar={s2r.cegar_refinements} elapsed={s2r.elapsed_ms:.1f}ms"
        )
        if s2r.direct_only_channels or s2r.valhall_only_channels:
            print(
                f"  CHANNEL_ASYMMETRY: direct_only={s2r.direct_only_channels} "
                f"valhall_only={s2r.valhall_only_channels}"
            )
        if s2r.concrete_mismatches:
            shown = s2r.concrete_mismatches[:5]
            for mm in shown:
                print(f"  MISMATCH {mm}")
            if s2r.sat_outputs > len(shown):
                print(f"  ... ({s2r.sat_outputs - len(shown)} more mismatched concrete outputs out of {s2r.checked_outputs} checked)")
        elif s2r.first_mismatch:
            print(f"  MISMATCH {s2r.first_mismatch}")
        return 0 if s2r.status in ("UNSAT", "BOUNDED_UNSAT") else 1

    if args.spv_or_json.endswith(".json"):
        with open(args.spv_or_json, "r", encoding="utf-8") as fh:
            bundle = json.load(fh)
        r = verify_equivalence_bundle(
            bundle,
            timeout_ms=args.timeout_ms,
            fp_mode=args.fp_mode,
            domain=args.domain,
            max_unroll=args.max_unroll,
            dump_smt2_path=args.dump_smt2,
            verbose=args.verbose,
            canonicalize=args.canonicalize,
            dump_ast=args.dump_ast,
            enable_conditioned_unify=args.conditioned_unify,
        )
    else:
        r = verify_spv_file(
            args.spv_or_json,
            spirv2valhall_bin=args.bin,
            timeout_ms=args.timeout_ms,
            fp_mode=args.fp_mode,
            domain=args.domain,
            max_unroll=args.max_unroll,
            dump_smt2_path=args.dump_smt2,
            verbose=args.verbose,
            canonicalize=args.canonicalize,
            dump_ast=args.dump_ast,
            enable_conditioned_unify=args.conditioned_unify,
        )

    print(f"[{r.status}] {os.path.basename(r.spv_path)} stage={r.stage} domain={args.domain} fp={r.fp_mode} "
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

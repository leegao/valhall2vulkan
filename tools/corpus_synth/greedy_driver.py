#!/usr/bin/env python3
"""
Greedy Submodular Corpus Synthesizer Driver.

Generates a compact suite of valid `N`-block (default 1), `M`-instruction
SPIR-V shaders that greedily maximize coverage of Valhall hardware instructions
(via `libMali` / `spirv2valhall`) or SPIR-V opcodes, with optional immediate
concolic equivalence verification via `tools.coverage.equiv_fuzzer`.
"""

from __future__ import annotations

import dataclasses
import os
import random
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from tools.corpus_synth import spv_builder as spv
from tools.corpus_synth.op_catalog import OP_CATALOG, OpSpec
from tools.corpus_synth.oracle import CoverageOracle, OracleResult


@dataclasses.dataclass
class SynthesizedShaderRecord:
    """Metadata and coverage summary for one synthesized SPIR-V shader."""

    shader_id: int
    spv_path: str
    num_blocks: int
    op_names: List[str]
    all_features: Set[str]
    new_features: Set[str]
    equiv_status: Optional[str] = None
    divergence_report: Optional[Any] = None


@dataclasses.dataclass
class CorpusSynthResult:
    """Summary of a greedy corpus synthesis run."""

    target: str
    shaders: List[SynthesizedShaderRecord]
    total_unique_features: Set[str]
    minimal_cover_indices: List[int]
    elapsed_ms: float


def emit_shader_bytes(
    ops: List[OpSpec],
    num_blocks: int = 1,
    seed: int = 0,
) -> bytes:
    """
    Emit a valid SPIR-V 1.3 compute shader containing `ops` distributed across
    `num_blocks` blocks (`1` = straight-line; `> 1` = structured acyclic
    selection diamonds) with anti-DCE sink accumulators written to SSBO out[0..3].
    """
    rng = random.Random(seed)
    b = spv.SpirvComputeBuilder()

    lbl_entry = b.alloc_id()
    b.emit(spv.OP_LABEL, lbl_entry)

    # Load 8 base U32 inputs and cast to S32/F32/Bool so every type pool starts populated.
    gid_x = b.load_gid_x()
    c15 = b.get_const_u32(15)
    u_inputs = []
    for i in range(8):
        idx_add = b.alloc_id()
        idx_mask = b.alloc_id()
        b.emit(spv.OP_IADD, b.t_u32, idx_add, gid_x, b.get_const_u32(i))
        b.emit(spv.OP_BITWISE_AND, b.t_u32, idx_mask, idx_add, c15)
        u_inputs.append(b.load_in_u32_dyn(idx_mask))
    s_inputs: List[int] = []
    f_inputs: List[int] = []
    for u in u_inputs[:4]:
        sid = b.alloc_id()
        b.emit(spv.OP_BITCAST, b.t_s32, sid, u)
        s_inputs.append(sid)
    for u in u_inputs[4:8]:
        fid = b.alloc_id()
        b.emit(spv.OP_BITCAST, b.t_f32, fid, u)
        f_inputs.append(fid)

    c0_u32 = b.get_const_u32(0)
    b0 = b.alloc_id()
    b.emit(spv.OP_INOT_EQUAL, b.t_bool, b0, u_inputs[0], c0_u32)

    pool: Dict[str, List[int]] = {
        "U32": list(u_inputs),
        "S32": list(s_inputs),
        "F32": list(f_inputs),
        "Bool": [b0],
    }

    acc_u = u_inputs[0]
    acc_f = f_inputs[0]

    def _emit_op_seq(seq: List[OpSpec], cur_u: int, cur_f: int, local_pool: Dict[str, List[int]]) -> Tuple[int, int]:
        for spec in seq:
            args = [rng.choice(local_pool[t]) for t in spec.in_types]
            res = spec.emit_fn(b, args)
            local_pool[spec.out_type].append(res)
            # Fold result into anti-DCE accumulator so libMali cannot eliminate it.
            if spec.out_type == "U32":
                nxt = b.alloc_id()
                b.emit(spv.OP_BITWISE_XOR, b.t_u32, nxt, cur_u, res)
                cur_u = nxt
            elif spec.out_type == "S32":
                bc = b.alloc_id()
                nxt = b.alloc_id()
                b.emit(spv.OP_BITCAST, b.t_u32, bc, res)
                b.emit(spv.OP_BITWISE_XOR, b.t_u32, nxt, cur_u, bc)
                cur_u = nxt
            elif spec.out_type == "F32":
                nxt = b.alloc_id()
                b.emit(spv.OP_FADD, b.t_f32, nxt, cur_f, res)
                cur_f = nxt
            elif spec.out_type == "Bool":
                sel = b.alloc_id()
                nxt = b.alloc_id()
                b.emit(spv.OP_SELECT, b.t_u32, sel, res, b.get_const_u32(0x9E3779B9), b.get_const_u32(1))
                b.emit(spv.OP_IADD, b.t_u32, nxt, cur_u, sel)
                cur_u = nxt
        return cur_u, cur_f

    if num_blocks <= 1:
        acc_u, acc_f = _emit_op_seq(ops, acc_u, acc_f, pool)
    else:
        # Build `num_diamonds` sequential structured selection diamonds (each adds 2-3 blocks).
        num_diamonds = max(1, (num_blocks - 1) // 2)
        chunk = max(1, len(ops) // (num_diamonds * 2))
        cursor = 0
        for d in range(num_diamonds):
            cond = b.alloc_id()
            b.emit(spv.OP_INOT_EQUAL, b.t_bool, cond, u_inputs[d % len(u_inputs)], c0_u32)
            lbl_then = b.alloc_id()
            lbl_else = b.alloc_id()
            lbl_merge = b.alloc_id()
            b.emit(spv.OP_SELECTION_MERGE, lbl_merge, 0)
            b.emit(spv.OP_BRANCH_CONDITIONAL, cond, lbl_then, lbl_else)

            t_ops = ops[cursor : cursor + chunk]
            cursor += len(t_ops)
            e_ops = ops[cursor : cursor + chunk] if d < num_diamonds - 1 else ops[cursor:]
            cursor += len(e_ops)

            b.emit(spv.OP_LABEL, lbl_then)
            t_pool = {k: list(v) for k, v in pool.items()}
            tu, tf = _emit_op_seq(t_ops, acc_u, acc_f, t_pool)
            b.emit(spv.OP_BRANCH, lbl_merge)

            b.emit(spv.OP_LABEL, lbl_else)
            e_pool = {k: list(v) for k, v in pool.items()}
            eu, ef = _emit_op_seq(e_ops, acc_u, acc_f, e_pool)
            b.emit(spv.OP_BRANCH, lbl_merge)

            b.emit(spv.OP_LABEL, lbl_merge)
            phi_u = b.alloc_id()
            phi_f = b.alloc_id()
            b.emit(spv.OP_PHI, b.t_u32, phi_u, tu, lbl_then, eu, lbl_else)
            b.emit(spv.OP_PHI, b.t_f32, phi_f, tf, lbl_then, ef, lbl_else)
            acc_u, acc_f = phi_u, phi_f
            pool["U32"].append(acc_u)
            pool["F32"].append(acc_f)

    # Store final anti-DCE accumulators to out[0..1]
    acc_f_bits = b.alloc_id()
    b.emit(spv.OP_BITCAST, b.t_u32, acc_f_bits, acc_f)
    b.store_out_u32(0, acc_u)
    b.store_out_u32(1, acc_f_bits)
    b.emit(spv.OP_RETURN)
    return b.build_bytes()


def _greedy_set_cover_indices(feature_sets: List[Set[str]]) -> List[int]:
    """Compute a minimal greedy set cover over synthesized shaders."""
    remaining: Set[str] = set().union(*feature_sets) if feature_sets else set()
    chosen: List[int] = []
    used: Set[int] = set()
    while remaining:
        best_i = -1
        best_gain = 0
        for i, fs in enumerate(feature_sets):
            if i in used:
                continue
            g = len(fs & remaining)
            if g > best_gain:
                best_gain = g
                best_i = i
        if best_i < 0:
            break
        used.add(best_i)
        chosen.append(best_i)
        remaining -= feature_sets[best_i]
    return chosen


def synthesize_corpus(
    target: str = "valhall-isa",
    num_shaders: int = 6,
    num_blocks: int = 1,
    num_insns: int = 12,
    out_dir: Optional[str] = None,
    bin_path: str = "./build/spirv2valhall",
    seed: int = 0xC0DE,
    candidates_per_shader: int = 4,
    verify_equiv: bool = False,
    domain: str = "udomain",
) -> CorpusSynthResult:
    """
    Greedily synthesize `num_shaders` SPIR-V shaders (each with `num_blocks`
    blocks and `num_insns` catalog operations) maximizing marginal coverage of
    `target` (`valhall-isa` or `spirv-ops`).
    """
    t0 = time.perf_counter()
    rng = random.Random(seed)
    oracle = CoverageOracle(target=target, bin_path=bin_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    global_covered: Set[str] = set()
    used_op_counts: Dict[str, int] = {spec.name: 0 for spec in OP_CATALOG}
    records: List[SynthesizedShaderRecord] = []

    for sidx in range(num_shaders):
        best_ops: Optional[List[OpSpec]] = None
        best_bytes: Optional[bytes] = None
        best_oracle: Optional[OracleResult] = None
        best_gain = -1

        for cidx in range(max(1, candidates_per_shader)):
            # Bias op sampling toward least-used catalog ops to maximize coverage diversity.
            ordered = sorted(OP_CATALOG, key=lambda sp: (used_op_counts[sp.name], rng.random()))
            cand_ops = ordered[:num_insns]
            rng.shuffle(cand_ops)
            cand_bytes = emit_shader_bytes(cand_ops, num_blocks=num_blocks, seed=seed + sidx * 31 + cidx)
            ores = oracle.evaluate(cand_bytes, cand_ops)
            if not ores.ok:
                continue
            gain = len(ores.features - global_covered)
            if gain > best_gain:
                best_gain = gain
                best_ops = cand_ops
                best_bytes = cand_bytes
                best_oracle = ores

        if best_ops is None or best_bytes is None or best_oracle is None:
            continue

        new_feats = best_oracle.features - global_covered
        global_covered.update(best_oracle.features)
        for sp in best_ops:
            used_op_counts[sp.name] += 1

        spv_path = ""
        if out_dir:
            spv_path = os.path.join(out_dir, f"synth_{sidx:03d}.spv")
            with open(spv_path, "wb") as fh:
                fh.write(best_bytes)

        eq_status: Optional[str] = None
        div_rep: Optional[Any] = None
        if verify_equiv and best_oracle.bundle is not None:
            from tools.coverage.equiv_fuzzer import run_concolic_equiv_fuzz
            from tools.symbolic_ssa.passes import canonicalize_ssa_module

            d_ssa, _ = canonicalize_ssa_module(best_oracle.bundle["direct_ssa"])
            v_ssa, _ = canonicalize_ssa_module(best_oracle.bundle["valhall_ssa"])
            ef = run_concolic_equiv_fuzz(
                best_oracle.bundle, d_ssa, v_ssa, seed_side="direct", k_init=16, time_budget_ms=500
            )
            eq_status = ef.status
            div_rep = ef.first_divergence

        records.append(
            SynthesizedShaderRecord(
                shader_id=sidx,
                spv_path=spv_path,
                num_blocks=num_blocks,
                op_names=[sp.name for sp in best_ops],
                all_features=best_oracle.features,
                new_features=new_feats,
                equiv_status=eq_status,
                divergence_report=div_rep,
            )
        )

    min_cover = _greedy_set_cover_indices([r.all_features for r in records])
    return CorpusSynthResult(
        target=target,
        shaders=records,
        total_unique_features=global_covered,
        minimal_cover_indices=min_cover,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )

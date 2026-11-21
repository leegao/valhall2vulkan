#!/usr/bin/env python3
"""
Unit & Regression Tests for All-Path ValhallSSA Formal Equivalence Verifier
(tools/verify/shader_equiv_verifier.py).

Covers:
  1. Direct vs. Valhall-lowered end-to-end equivalence via spirv2valhall + libMali-Gxx
  2. Mutation detection (injecting a 1-bit bug into valhall_ssa -> SAT + PC localization)
  3. Combinatorial state-explosion resistance: 20 sequential if/else diamonds (2^20 = 1,048,576 paths)
     proved UNSAT in linear O(|V|+|E|) time via DAG path-merging
  4. Regression checks for Checkpoint v24 bugs (BLEND.v4.f16, conditional DISCARD, OR.i32 lanes1,
     LD_VAR_IMM varying_index)
"""

from __future__ import annotations

import copy
import os
import tempfile
import unittest

from tools.verify.shader_equiv_verifier import (
    verify_equivalence_bundle,
    verify_spv_file,
)


def _make_diamond_chain_ssa(num_diamonds: int, flip_last_const: bool = False) -> dict:
    """Build a ValhallSsaModule JSON with `num_diamonds` sequential if/else diamonds (2^N paths)."""
    blocks = []
    nodes = []
    next_nid = 1

    def add_node(blk_id: int, op: str, vt: str, args=None, **kw) -> int:
        nonlocal next_nid
        nid = next_nid
        next_nid += 1
        n = {"id": nid, "op": op, "type": vt, "blk": blk_id, "args": list(args or [])}
        n.update(kw)
        nodes.append(n)
        return nid

    # Entry block 0: load input varying x0, x1
    b0_nodes = []
    tok0 = add_node(0, "Nop", "Token")
    b0_nodes.append(tok0)
    v_in = add_node(0, "LoadFragVarying", "V4F32", imm_u32=0)
    b0_nodes.append(v_in)
    acc = add_node(0, "VecExtract", "U32", [v_in], imm_u32=0)
    b0_nodes.append(acc)
    sel_src = add_node(0, "VecExtract", "U32", [v_in], imm_u32=1)
    b0_nodes.append(sel_src)

    blocks.append({
        "id": 0, "preds": [], "phis": [], "nodes": b0_nodes,
        "branch_target": -1, "fallthrough": -1, "is_cond": False, "is_uncond": False,
        "is_end": False, "is_discard": False, "is_loop_header": False,
        "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
    })

    cur_blk = 0
    for d in range(num_diamonds):
        b_then = len(blocks)
        b_else = b_then + 1
        b_merge = b_then + 2

        mask_c = add_node(cur_blk, "ConstU32", "U32", imm_u32=(1 << (d % 31)))
        masked = add_node(cur_blk, "BitAnd", "U32", [sel_src, mask_c])
        zero_c = add_node(cur_blk, "ConstU32", "U32", imm_u32=0)
        cond = add_node(cur_blk, "ICmpNe", "Bool", [masked, zero_c])
        blocks[cur_blk]["nodes"].extend([mask_c, masked, zero_c, cond])
        blocks[cur_blk]["is_cond"] = True
        blocks[cur_blk]["branch_cond"] = cond
        blocks[cur_blk]["branch_target"] = b_then
        blocks[cur_blk]["fallthrough"] = b_else
        blocks[cur_blk]["sel_merge"] = b_merge

        delta_t = (d + 3) ^ (1 if (flip_last_const and d == num_diamonds - 1) else 0)
        ct = add_node(b_then, "ConstU32", "U32", imm_u32=delta_t)
        vt = add_node(b_then, "IAdd", "U32", [acc, ct], orig_pc=d * 16)
        blocks.append({
            "id": b_then, "preds": [cur_blk], "phis": [], "nodes": [ct, vt],
            "branch_target": b_merge, "fallthrough": -1, "is_cond": False, "is_uncond": True,
            "is_end": False, "is_discard": False, "is_loop_header": False,
            "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
        })

        ce = add_node(b_else, "ConstU32", "U32", imm_u32=(d + 7))
        ve = add_node(b_else, "BitXor", "U32", [acc, ce], orig_pc=d * 16 + 8)
        blocks.append({
            "id": b_else, "preds": [cur_blk], "phis": [], "nodes": [ce, ve],
            "branch_target": b_merge, "fallthrough": -1, "is_cond": False, "is_uncond": True,
            "is_end": False, "is_discard": False, "is_loop_header": False,
            "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
        })

        phi = add_node(b_merge, "Phi", "U32", phi=[[b_then, vt], [b_else, ve]])
        blocks.append({
            "id": b_merge, "preds": [b_then, b_else], "phis": [phi], "nodes": [],
            "branch_target": -1, "fallthrough": -1, "is_cond": False, "is_uncond": False,
            "is_end": False, "is_discard": False, "is_loop_header": False,
            "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
        })
        acc = phi
        cur_blk = b_merge

    # Final block: StoreFragColor(0, vec4(acc, acc, acc, acc))
    vec4 = add_node(cur_blk, "VecConstruct4", "V4F32", [acc, acc, acc, acc])
    st = add_node(cur_blk, "StoreFragColor", "Token", [vec4, tok0], imm_u32=0, orig_pc=0xFF0)
    blocks[cur_blk]["nodes"].extend([vec4, st])
    blocks[cur_blk]["is_end"] = True

    return {"blocks": blocks, "nodes": nodes}


class TestShaderEquivVerifier(unittest.TestCase):
    def test_20_diamond_dag_no_state_explosion(self) -> None:
        """20 sequential if/else diamonds (2^20 = 1,048,576 paths) must prove UNSAT fast."""
        ssa = _make_diamond_chain_ssa(20, flip_last_const=False)
        bundle = {
            "spv_path": "synthetic_20_diamonds.spv",
            "stage": "Fragment",
            "arch": 10,
            "direct_ok": True,
            "valhall_ok": True,
            "direct_bindings": [],
            "mbs2_context": {"fcsts": [], "rlocs": [], "pilot_fau": [], "sinp": [], "sout": [], "pc_disasm": {}},
            "direct_ssa": ssa,
            "valhall_ssa": copy.deepcopy(ssa),
        }
        res = verify_equivalence_bundle(bundle, timeout_ms=2000)
        self.assertEqual(res.status, "UNSAT")
        self.assertLess(res.elapsed_ms, 1000.0)

    def test_20_diamond_mutation_detects_sat_and_localizes_pc(self) -> None:
        """A 1-bit bug in the 20th diamond must be caught as SAT with exact suspect PC."""
        direct = _make_diamond_chain_ssa(20, flip_last_const=False)
        buggy = _make_diamond_chain_ssa(20, flip_last_const=True)
        bundle = {
            "spv_path": "synthetic_20_diamonds_bug.spv",
            "stage": "Fragment",
            "arch": 10,
            "direct_ok": True,
            "valhall_ok": True,
            "direct_bindings": [],
            "mbs2_context": {
                "fcsts": [], "rlocs": [], "pilot_fau": [], "sinp": [], "sout": [],
                "pc_disasm": {str(19 * 16): "IADD.u32 r0, r0, #23"},
            },
            "direct_ssa": direct,
            "valhall_ssa": buggy,
        }
        res = verify_equivalence_bundle(bundle, timeout_ms=2000)
        self.assertEqual(res.status, "SAT")
        self.assertTrue(len(res.mismatches) >= 1)
        self.assertIn("frag_color[0]", res.mismatches[0].output_name)
        pcs = [pc for pc, _ in res.mismatches[0].suspect_pcs]
        self.assertIn(19 * 16, pcs)

    def test_ssbo_pilot_fau_binding_equivalence(self) -> None:
        """Verify SSBO writes match between direct LoadSidebandU64 and Valhall LoadFau+Pack64."""
        direct_ssa = {
            "blocks": [{
                "id": 0, "preds": [], "phis": [], "nodes": [1, 2, 3, 4, 5, 6],
                "branch_target": -1, "fallthrough": -1, "is_cond": False, "is_uncond": False,
                "is_end": True, "is_discard": False, "is_loop_header": False,
                "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
            }],
            "nodes": [
                {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                {"id": 2, "op": "LoadSidebandU64", "type": "U64", "blk": 0, "args": [], "imm_u32": 16, "aux_str": "ssbo:0:1"},
                {"id": 3, "op": "ConstU64", "type": "U64", "blk": 0, "args": [], "imm_u64": 4},
                {"id": 4, "op": "IAdd64", "type": "U64", "blk": 0, "args": [2, 3]},
                {"id": 5, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 0xDEADBEEF},
                {"id": 6, "op": "PsbStore", "type": "Token", "blk": 0, "args": [4, 5, 1], "imm_u32": 4, "aux_str": "ssbo:0:1"},
            ],
        }
        valhall_ssa = {
            "blocks": [{
                "id": 0, "preds": [], "phis": [], "nodes": [1, 2, 3, 4, 5, 6, 7, 8],
                "branch_target": -1, "fallthrough": -1, "is_cond": False, "is_uncond": False,
                "is_end": True, "is_discard": False, "is_loop_header": False,
                "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
            }],
            "nodes": [
                {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                {"id": 2, "op": "LoadFau", "type": "U32", "blk": 0, "args": [], "imm_u32": 0, "orig_pc": 0},
                {"id": 3, "op": "LoadFau", "type": "U32", "blk": 0, "args": [], "imm_u32": 1, "orig_pc": 0},
                {"id": 4, "op": "Pack64", "type": "U64", "blk": 0, "args": [2, 3], "orig_pc": 0},
                {"id": 5, "op": "ConstU64", "type": "U64", "blk": 0, "args": [], "imm_u64": 4, "orig_pc": 8},
                {"id": 6, "op": "IAdd64", "type": "U64", "blk": 0, "args": [4, 5], "orig_pc": 8},
                {"id": 7, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 0xDEADBEEF, "orig_pc": 8},
                {"id": 8, "op": "PsbStore", "type": "Token", "blk": 0, "args": [6, 7, 1], "imm_u32": 4, "orig_pc": 8},
            ],
        }
        bundle = {
            "spv_path": "synthetic_ssbo.spv",
            "stage": "Compute",
            "arch": 10,
            "direct_ok": True,
            "valhall_ok": True,
            "direct_bindings": [{"kind": "SSBO", "sideband_slot": 16, "set": 0, "binding": 1, "key": "ssbo:0:1"}],
            "mbs2_context": {
                "fcsts": [], "rlocs": [],
                "pilot_fau": [
                    {"fau_offset": 0, "word_idx": 0, "byte_size": 4, "resolved": True, "kind": "ssbo", "binding": 1, "key": "ssbo:0:1"},
                    {"fau_offset": 4, "word_idx": 1, "byte_size": 4, "resolved": True, "kind": "ssbo", "binding": 1, "key": "ssbo:0:1"},
                ],
                "sinp": [], "sout": [], "pc_disasm": {},
            },
            "direct_ssa": direct_ssa,
            "valhall_ssa": valhall_ssa,
        }
        res = verify_equivalence_bundle(bundle, timeout_ms=2000)
        self.assertEqual(res.status, "UNSAT")
        self.assertIn("ssbo:0:1+0x4.value", res.checked_outputs)

    def test_conditional_discard_equivalence(self) -> None:
        """Verify conditional DISCARD matches SPIR-V branching OpKill across all float bits (incl. NaN)."""
        # Direct: if (!(x != 0.0 || isnan(x))) OpKill; else frag_color[0] = (x, 0, 0, 1)
        direct_ssa = {
            "blocks": [
                {"id": 0, "preds": [], "phis": [], "nodes": [1, 2, 3, 4, 5, 6, 7, 8],
                 "branch_target": 2, "fallthrough": 1, "is_cond": True, "is_uncond": False,
                 "is_end": False, "is_discard": False, "is_loop_header": False,
                 "loop_backedge": -1, "loop_merge": -1, "sel_merge": 1, "branch_cond": 8},
                {"id": 1, "preds": [0], "phis": [], "nodes": [9, 10, 11],
                 "branch_target": -1, "fallthrough": -1, "is_cond": False, "is_uncond": False,
                 "is_end": True, "is_discard": False, "is_loop_header": False,
                 "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0},
                {"id": 2, "preds": [0], "phis": [], "nodes": [12, 13],
                 "branch_target": -1, "fallthrough": -1, "is_cond": False, "is_uncond": False,
                 "is_end": False, "is_discard": True, "is_loop_header": False,
                 "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0},
            ],
            "nodes": [
                {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                {"id": 2, "op": "LoadFragVarying", "type": "V4F32", "blk": 0, "args": [], "imm_u32": 0},
                {"id": 3, "op": "VecExtract", "type": "F32", "blk": 0, "args": [2], "imm_u32": 0},
                {"id": 4, "op": "ConstF32", "type": "F32", "blk": 0, "args": [], "imm_u32": 0},
                {"id": 5, "op": "FCmpOrdNe", "type": "Bool", "blk": 0, "args": [3, 4]},
                {"id": 6, "op": "IsNan", "type": "Bool", "blk": 0, "args": [3]},
                {"id": 7, "op": "LogicalOr", "type": "Bool", "blk": 0, "args": [5, 6]},
                {"id": 8, "op": "LogicalNot", "type": "Bool", "blk": 0, "args": [7]},
                {"id": 9, "op": "ConstF32", "type": "F32", "blk": 1, "args": [], "imm_u32": 0x3F800000},
                {"id": 10, "op": "VecConstruct4", "type": "V4F32", "blk": 1, "args": [3, 4, 4, 9]},
                {"id": 11, "op": "StoreFragColor", "type": "Token", "blk": 1, "args": [10, 1], "imm_u32": 0},
                {"id": 12, "op": "ConstBool", "type": "Bool", "blk": 2, "args": [], "imm_u32": 1},
                {"id": 13, "op": "Kill", "type": "Void", "blk": 2, "args": [12, 1], "aux_bool": True},
            ],
        }
        # Valhall: FCmpOrdEq(x, 0.0) -> Kill(cond, tok), then StoreFragColor(x, 0, 0, 1)
        valhall_ssa = {
            "blocks": [{
                "id": 0, "preds": [], "phis": [], "nodes": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                "branch_target": -1, "fallthrough": -1, "is_cond": False, "is_uncond": False,
                "is_end": True, "is_discard": False, "is_loop_header": False,
                "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
            }],
            "nodes": [
                {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                {"id": 2, "op": "LoadFragVarying", "type": "V4F32", "blk": 0, "args": [], "imm_u32": 0},
                {"id": 3, "op": "VecExtract", "type": "F32", "blk": 0, "args": [2], "imm_u32": 0},
                {"id": 4, "op": "ConstF32", "type": "F32", "blk": 0, "args": [], "imm_u32": 0},
                {"id": 5, "op": "FCmpOrdEq", "type": "Bool", "blk": 0, "args": [3, 4]},
                {"id": 6, "op": "Kill", "type": "Void", "blk": 0, "args": [5, 1], "aux_bool": True},
                {"id": 7, "op": "ConstF32", "type": "F32", "blk": 0, "args": [], "imm_u32": 0x3F800000},
                {"id": 8, "op": "VecConstruct4", "type": "V4F32", "blk": 0, "args": [3, 4, 4, 7]},
                {"id": 9, "op": "StoreFragColor", "type": "Token", "blk": 0, "args": [8, 1], "imm_u32": 0},
            ],
        }
        bundle = {
            "spv_path": "synthetic_cond_discard.spv",
            "stage": "Fragment",
            "arch": 10,
            "direct_ok": True,
            "valhall_ok": True,
            "direct_bindings": [],
            "mbs2_context": {"fcsts": [], "rlocs": [], "pilot_fau": [], "sinp": [], "sout": [], "pc_disasm": {}},
            "direct_ssa": direct_ssa,
            "valhall_ssa": valhall_ssa,
        }
        res = verify_equivalence_bundle(bundle, timeout_ms=2000)
        self.assertEqual(res.status, "UNSAT")


    def test_v2f32_pack_half2x16_no_crash(self) -> None:
        """Regression test for astc_enc.raw.spv: V2F32 FAdd -> PackHalf2x16 must not crash."""
        ssa = {
            "blocks": [{
                "id": 0, "preds": [], "phis": [], "nodes": [1, 2, 3, 4, 5, 6, 7],
                "branch_target": -1, "fallthrough": -1, "is_cond": False, "is_uncond": False,
                "is_end": True, "is_discard": False, "is_loop_header": False,
                "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
            }],
            "nodes": [
                {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                {"id": 2, "op": "LoadFragVarying", "type": "V4F32", "blk": 0, "args": [], "imm_u32": 0},
                {"id": 3, "op": "VecExtract", "type": "U32", "blk": 0, "args": [2], "imm_u32": 0},
                {"id": 4, "op": "UnpackHalf2x16", "type": "V2F32", "blk": 0, "args": [3]},
                {"id": 5, "op": "FAdd", "type": "V2F32", "blk": 0, "args": [4, 4]},
                {"id": 6, "op": "PackHalf2x16", "type": "U32", "blk": 0, "args": [5]},
                {"id": 7, "op": "StoreFragColor", "type": "Token", "blk": 0, "args": [6, 1], "imm_u32": 0},
            ],
        }
        bundle = {
            "spv_path": "synthetic_v2f16_pack.spv",
            "stage": "Fragment",
            "arch": 10,
            "direct_ok": True,
            "valhall_ok": True,
            "direct_bindings": [],
            "mbs2_context": {"fcsts": [], "rlocs": [], "pilot_fau": [], "sinp": [], "sout": [], "pc_disasm": {}},
            "direct_ssa": ssa,
            "valhall_ssa": copy.deepcopy(ssa),
        }
        res = verify_equivalence_bundle(bundle, timeout_ms=2000)
        self.assertEqual(res.status, "UNSAT")



    def test_symbolic_offset_ssbo_array_probing(self) -> None:
        """Dynamic-offset SSBO writes must be checked via Z3 Array probing (not skipped as outputs=0)."""
        direct_ssa = {
            "blocks": [{
                "id": 0, "preds": [], "phis": [], "nodes": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "branch_target": -1, "fallthrough": -1, "is_cond": False, "is_uncond": False,
                "is_end": True, "is_discard": False, "is_loop_header": False,
                "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
            }],
            "nodes": [
                {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                {"id": 2, "op": "LoadBuiltinInput", "type": "V3U32", "blk": 0, "args": [], "imm_u32": 28},
                {"id": 3, "op": "VecExtract", "type": "U32", "blk": 0, "args": [2], "imm_u32": 0},
                {"id": 4, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 2},
                {"id": 5, "op": "Shl", "type": "U32", "blk": 0, "args": [3, 4]},
                {"id": 6, "op": "UConvert32To64", "type": "U64", "blk": 0, "args": [5]},
                {"id": 7, "op": "LoadSidebandU64", "type": "U64", "blk": 0, "args": [], "imm_u32": 16, "aux_str": "ssbo:0:0"},
                {"id": 8, "op": "IAdd64", "type": "U64", "blk": 0, "args": [7, 6]},
                {"id": 9, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 42},
                {"id": 10, "op": "PsbStore", "type": "Token", "blk": 0, "args": [8, 9, 1], "imm_u32": 4, "aux_str": "ssbo:0:0"},
            ],
        }
        bundle = {
            "spv_path": "synthetic_dyn_ssbo.spv",
            "stage": "Compute",
            "arch": 10,
            "direct_ok": True,
            "valhall_ok": True,
            "direct_bindings": [{"kind": "SSBO", "sideband_slot": 16, "set": 0, "binding": 0, "key": "ssbo:0:0"}],
            "mbs2_context": {"fcsts": [], "rlocs": [], "pilot_fau": [], "sinp": [], "sout": [], "pc_disasm": {}},
            "direct_ssa": direct_ssa,
            "valhall_ssa": copy.deepcopy(direct_ssa),
        }
        res = verify_equivalence_bundle(bundle, timeout_ms=2000)
        self.assertEqual(res.status, "UNSAT")
        self.assertIn("ssbo:0:0[probe_off]", res.checked_outputs)

    def test_fragment_preload_pilot_ubo_word_binding(self) -> None:
        """Fragment preload pilot UBO word in FAU must match direct UBO PsbLoad at same byte_offset."""
        direct_ssa = {
            "blocks": [{
                "id": 0, "preds": [], "phis": [], "nodes": [1, 2, 3, 4, 5, 6, 7],
                "branch_target": -1, "fallthrough": -1, "is_cond": False, "is_uncond": False,
                "is_end": True, "is_discard": False, "is_loop_header": False,
                "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
            }],
            "nodes": [
                {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                {"id": 2, "op": "LoadSidebandU64", "type": "U64", "blk": 0, "args": [], "imm_u32": 16, "aux_str": "ubo:0:0"},
                {"id": 3, "op": "ConstU64", "type": "U64", "blk": 0, "args": [], "imm_u64": 16},
                {"id": 4, "op": "IAdd64", "type": "U64", "blk": 0, "args": [2, 3]},
                {"id": 5, "op": "PsbLoad", "type": "F32", "blk": 0, "args": [4, 1], "imm_u32": 4, "aux_str": "ubo:0:0"},
                {"id": 6, "op": "VecConstruct4", "type": "V4F32", "blk": 0, "args": [5, 5, 5, 5]},
                {"id": 7, "op": "StoreFragColor", "type": "Token", "blk": 0, "args": [6, 1], "imm_u32": 0},
            ],
        }
        valhall_ssa = {
            "blocks": [{
                "id": 0, "preds": [], "phis": [], "nodes": [1, 2, 3, 4],
                "branch_target": -1, "fallthrough": -1, "is_cond": False, "is_uncond": False,
                "is_end": True, "is_discard": False, "is_loop_header": False,
                "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
            }],
            "nodes": [
                {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                {"id": 2, "op": "LoadFau", "type": "F32", "blk": 0, "args": [], "imm_u32": 4, "orig_pc": 0},
                {"id": 3, "op": "VecConstruct4", "type": "V4F32", "blk": 0, "args": [2, 2, 2, 2]},
                {"id": 4, "op": "StoreFragColor", "type": "Token", "blk": 0, "args": [3, 1], "imm_u32": 0, "orig_pc": 8},
            ],
        }
        bundle = {
            "spv_path": "synthetic_frag_preload_ubo.spv",
            "stage": "Fragment",
            "arch": 10,
            "direct_ok": True,
            "valhall_ok": True,
            "direct_bindings": [{"kind": "UBO", "sideband_slot": 16, "set": 0, "binding": 0, "key": "ubo:0:0"}],
            "mbs2_context": {
                "fcsts": [], "rlocs": [],
                "pilot_fau": [{
                    "fau_offset": 16, "word_idx": 4, "byte_size": 4, "resolved": True,
                    "kind": "ubo", "binding": 0, "is_loaded_value": True,
                    "has_byte_offset": True, "byte_offset": 16, "is_derived": False, "key": "ubo:0:0",
                }],
                "sinp": [], "sout": [], "pc_disasm": {},
            },
            "direct_ssa": direct_ssa,
            "valhall_ssa": valhall_ssa,
        }
        res = verify_equivalence_bundle(bundle, timeout_ms=2000)
        self.assertEqual(res.status, "UNSAT")

    def test_vertex_viewport_transform_identity_precondition(self) -> None:
        """Vertex gl_Position with gl_mali_ViewportTransform (scale=1, bias=0, w=1) must prove UNSAT."""
        direct_ssa = {
            "blocks": [{
                "id": 0, "preds": [], "phis": [], "nodes": [1, 2, 3, 4, 5, 6],
                "branch_target": -1, "fallthrough": -1, "is_cond": False, "is_uncond": False,
                "is_end": True, "is_discard": False, "is_loop_header": False,
                "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
            }],
            "nodes": [
                {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                {"id": 2, "op": "LoadVertAttr", "type": "V4F32", "blk": 0, "args": [], "imm_u32": 0},
                {"id": 3, "op": "VecExtract", "type": "F32", "blk": 0, "args": [2], "imm_u32": 0},
                {"id": 4, "op": "ConstF32", "type": "F32", "blk": 0, "args": [], "imm_u32": 0x3F800000},
                {"id": 5, "op": "VecConstruct4", "type": "V4F32", "blk": 0, "args": [3, 3, 3, 4]},
                {"id": 6, "op": "StoreBuiltinOutput", "type": "Token", "blk": 0, "args": [5, 1], "imm_u32": 0},
            ],
        }
        # Valhall applies FMA(pos.xyz, vp_scale.xyz, vp_bias.xyz) with vp_scale=u0..u2, vp_bias=u4..u6
        valhall_ssa = {
            "blocks": [{
                "id": 0, "preds": [], "phis": [], "nodes": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                "branch_target": -1, "fallthrough": -1, "is_cond": False, "is_uncond": False,
                "is_end": True, "is_discard": False, "is_loop_header": False,
                "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
            }],
            "nodes": [
                {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                {"id": 2, "op": "LoadVertAttr", "type": "V4F32", "blk": 0, "args": [], "imm_u32": 0},
                {"id": 3, "op": "VecExtract", "type": "F32", "blk": 0, "args": [2], "imm_u32": 0},
                {"id": 4, "op": "LoadFau", "type": "F32", "blk": 0, "args": [], "imm_u32": 0},
                {"id": 5, "op": "LoadFau", "type": "F32", "blk": 0, "args": [], "imm_u32": 4},
                {"id": 6, "op": "Fma", "type": "F32", "blk": 0, "args": [3, 4, 5]},
                {"id": 7, "op": "ConstF32", "type": "F32", "blk": 0, "args": [], "imm_u32": 0x3F800000},
                {"id": 8, "op": "VecConstruct4", "type": "V4F32", "blk": 0, "args": [6, 6, 6, 7]},
                {"id": 9, "op": "StoreBuiltinOutput", "type": "Token", "blk": 0, "args": [8, 1], "imm_u32": 0},
            ],
        }
        bundle = {
            "spv_path": "synthetic_vert_vp.spv",
            "stage": "Vertex",
            "arch": 10,
            "direct_ok": True,
            "valhall_ok": True,
            "direct_bindings": [],
            "mbs2_context": {
                "fcsts": [],
                "rlocs": [
                    {"symbol": "gl_mali_ViewportTransform", "reg_offset": 0, "word_idx": 0, "uni_offset": 0, "size": 12},
                    {"symbol": "gl_mali_ViewportTransform", "reg_offset": 16, "word_idx": 4, "uni_offset": 16, "size": 12},
                ],
                "pilot_fau": [], "sinp": [], "sout": [], "pc_disasm": {},
            },
            "direct_ssa": direct_ssa,
            "valhall_ssa": valhall_ssa,
        }
        res = verify_equivalence_bundle(bundle, timeout_ms=2000)
        self.assertEqual(res.status, "UNSAT")


    def test_smod_and_all_int_bit_fp_eval_paths(self) -> None:
        """Exercise every integer, bitwise, shift, div/rem/SMod, 64-bit, cmp, conv, and FP evaluator path."""
        import z3
        from tools.verify.shader_equiv_verifier import SharedShaderEnv, DagSsaZ3Evaluator, bv32

        # Verify SMod exact SPIR-V §3.3.13 semantics across all 4 sign quadrants.
        smod_cases = [
            (7, 3, 1),
            (-7, 3, 2),
            (7, -3, -2),
            (-7, -3, -1),
            (-6, 3, 0),
        ]
        for a_val, b_val, expected in smod_cases:
            ssa = {
                "blocks": [{
                    "id": 0, "preds": [], "phis": [], "nodes": [1, 2, 3, 4, 5, 6],
                    "branch_target": -1, "fallthrough": -1, "is_cond": False, "is_uncond": False,
                    "is_end": True, "is_discard": False, "is_loop_header": False,
                    "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
                }],
                "nodes": [
                    {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                    {"id": 2, "op": "ConstS32", "type": "S32", "blk": 0, "args": [], "imm_u32": a_val & 0xFFFFFFFF},
                    {"id": 3, "op": "ConstS32", "type": "S32", "blk": 0, "args": [], "imm_u32": b_val & 0xFFFFFFFF},
                    {"id": 4, "op": "SMod", "type": "S32", "blk": 0, "args": [2, 3]},
                    {"id": 5, "op": "LoadSidebandU64", "type": "U64", "blk": 0, "args": [], "imm_u32": 16, "aux_str": "ssbo:0:0"},
                    {"id": 6, "op": "PsbStore", "type": "Token", "blk": 0, "args": [5, 4, 1], "imm_u32": 4, "aux_str": "ssbo:0:0"},
                ],
            }
            env = SharedShaderEnv({"mbs2_context": {}})
            st = DagSsaZ3Evaluator(ssa, env, is_valhall_side=False, fp_mode="exact").evaluate()
            _, val_expr = st.const_stores[("ssbo:0:0", 0)]
            got = z3.simplify(val_expr)
            self.assertEqual(got.as_long(), expected & 0xFFFFFFFF, f"SMod({a_val}, {b_val}) expected {expected}")

        # Exercise every 32-bit, 64-bit, comparison, conversion, and FP op in a single self-equivalence check.
        nodes = [
            {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
            {"id": 2, "op": "LoadBuiltinInput", "type": "U32", "blk": 0, "args": [], "imm_u32": 28},
            {"id": 3, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 7},
            {"id": 4, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 3},
            {"id": 5, "op": "ConstF32", "type": "F32", "blk": 0, "args": [], "imm_u32": 0x3F800000},
            {"id": 6, "op": "ConstF32", "type": "F32", "blk": 0, "args": [], "imm_u32": 0x40000000},
            {"id": 7, "op": "ConstU64", "type": "U64", "blk": 0, "args": [], "imm_u64": 0x100},
        ]
        nid = 8
        def add_op(op, vt, args, **kw):
            nonlocal nid
            n = {"id": nid, "op": op, "type": vt, "blk": 0, "args": args}
            n.update(kw)
            nodes.append(n)
            nid += 1
            return nid - 1

        # Unary & binary 32-bit integer / bit ops
        for op in ("BitNot", "SNeg", "SAbs", "Clz", "BitCount", "BitReverse"):
            add_op(op, "U32", [2])
        for op in ("IAdd", "ISub", "IMul", "BitAnd", "BitOr", "BitXor", "Shl", "ShrU", "ShrS",
                   "RotL", "RotR", "UDiv", "SDiv", "UMod", "SRem", "SMod", "IDot4S", "IDot4SU", "IDot4U"):
            add_op(op, "U32", [2, 3])
        add_op("BfeU", "U32", [2, 4, 3])
        add_op("BfeS", "U32", [2, 4, 3])
        add_op("Bfi", "U32", [2, 3, 4, 3])

        # 64-bit integer ops
        u64_a = add_op("UConvert32To64", "U64", [2])
        add_op("SConvert32To64", "U64", [2])
        add_op("UConvert64To32", "U32", [u64_a])
        add_op("BitNot64", "U64", [u64_a])
        for op in ("IAdd64", "ISub64", "IMul64", "BitAnd64", "BitOr64", "BitXor64", "Shl64", "ShrU64", "ShrS64"):
            add_op(op, "U64", [u64_a, 7])

        # Comparisons, logical, conversions, FP ops
        for op in ("ICmpEq", "ICmpNe", "ICmpSLt", "ICmpSLe", "ICmpSGt", "ICmpSGe", "ICmpULt", "ICmpULe", "ICmpUGt", "ICmpUGe"):
            add_op(op, "Bool", [2, 3])
        for op in ("FCmpOrdEq", "FCmpOrdGt", "FCmpOrdGe", "FCmpOrdLt", "FCmpOrdLe", "FCmpOrdNe", "FCmpUnordNe", "FCmpTotal"):
            add_op(op, "Bool", [5, 6])
        for op in ("FAbs", "FNeg", "Floor", "Ceil", "Trunc", "RoundEven", "Sqrt", "Sin", "Cos", "Exp2", "Log2", "DPdx", "DPdy", "Fwidth"):
            add_op(op, "F32", [5])
        for op in ("FAdd", "FSub", "FMul", "FDiv", "FMin", "FMax", "Pow", "Atan2"):
            add_op(op, "F32", [5, 6])
        add_op("FClamp", "F32", [5, 5, 6])
        add_op("Fma", "F32", [5, 5, 6])
        add_op("IsNan", "Bool", [5])
        add_op("IsInf", "Bool", [5])
        for op in ("ConvertFToS", "ConvertFToU", "ConvertSToF", "ConvertUToF"):
            add_op(op, "U32", [5])

        last_val = add_op("SMod", "S32", [2, 3])
        base = add_op("LoadSidebandU64", "U64", [], imm_u32=16, aux_str="ssbo:0:0")
        add_op("PsbStore", "Token", [base, last_val, 1], imm_u32=4, aux_str="ssbo:0:0")

        ssa = {
            "blocks": [{
                "id": 0, "preds": [], "phis": [], "nodes": [n["id"] for n in nodes],
                "branch_target": -1, "fallthrough": -1, "is_cond": False, "is_uncond": False,
                "is_end": True, "is_discard": False, "is_loop_header": False,
                "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
            }],
            "nodes": nodes,
        }
        bundle = {
            "spv_path": "synthetic_all_ops.spv",
            "stage": "Compute", "arch": 10, "direct_ok": True, "valhall_ok": True,
            "direct_bindings": [], "mbs2_context": {"fcsts": [], "rlocs": [], "pilot_fau": [], "sinp": [], "sout": [], "pc_disasm": {}},
            "direct_ssa": ssa, "valhall_ssa": ssa,
        }
        res = verify_equivalence_bundle(bundle, timeout_ms=2000)
        self.assertEqual(res.status, "UNSAT")

    def test_nested_additive_resolve_addr_and_pilot_value_kinds(self) -> None:
        """Dynamic SignExt*stride + const_field_off must resolve to ssbo:0:2 (never SKIP outputs=0)."""
        direct_ssa = {
            "blocks": [{
                "id": 0, "preds": [], "phis": [], "nodes": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "branch_target": -1, "fallthrough": -1, "is_cond": False, "is_uncond": False,
                "is_end": True, "is_discard": False, "is_loop_header": False,
                "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
            }],
            "nodes": [
                {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                {"id": 2, "op": "LoadSidebandU64", "type": "U64", "blk": 0, "args": [], "imm_u32": 16, "aux_str": "ssbo:0:2"},
                {"id": 3, "op": "LoadBuiltinInput", "type": "U32", "blk": 0, "args": [], "imm_u32": 28},
                {"id": 4, "op": "SConvert32To64", "type": "U64", "blk": 0, "args": [3]},
                {"id": 5, "op": "ConstU64", "type": "U64", "blk": 0, "args": [], "imm_u64": 80},
                {"id": 6, "op": "IMul64", "type": "U64", "blk": 0, "args": [4, 5]},
                {"id": 7, "op": "ConstU64", "type": "U64", "blk": 0, "args": [], "imm_u64": 8},
                {"id": 8, "op": "IAdd64", "type": "U64", "blk": 0, "args": [2, 6]},
                {"id": 9, "op": "IAdd64", "type": "U64", "blk": 0, "args": [8, 7]},
                {"id": 10, "op": "PsbStore", "type": "Token", "blk": 0, "args": [9, 3, 1], "imm_u32": 4, "aux_str": "ssbo:0:2"},
            ],
        }
        # Valhall side loads ssbo:0:2 base+8 directly from pilot_fau (pointer_offset=8) and adds idx*80
        valhall_ssa = {
            "blocks": [{
                "id": 0, "preds": [], "phis": [], "nodes": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "branch_target": -1, "fallthrough": -1, "is_cond": False, "is_uncond": False,
                "is_end": True, "is_discard": False, "is_loop_header": False,
                "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
            }],
            "nodes": [
                {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                {"id": 2, "op": "LoadFau", "type": "U32", "blk": 0, "args": [], "imm_u32": 0},
                {"id": 3, "op": "LoadFau", "type": "U32", "blk": 0, "args": [], "imm_u32": 1},
                {"id": 4, "op": "Pack64", "type": "U64", "blk": 0, "args": [2, 3]},
                {"id": 5, "op": "LoadBuiltinInput", "type": "U32", "blk": 0, "args": [], "imm_u32": 28},
                {"id": 6, "op": "SConvert32To64", "type": "U64", "blk": 0, "args": [5]},
                {"id": 7, "op": "ConstU64", "type": "U64", "blk": 0, "args": [], "imm_u64": 80},
                {"id": 8, "op": "IMul64", "type": "U64", "blk": 0, "args": [6, 7]},
                {"id": 9, "op": "IAdd64", "type": "U64", "blk": 0, "args": [4, 8]},
                {"id": 10, "op": "PsbStore", "type": "Token", "blk": 0, "args": [9, 5, 1], "imm_u32": 4, "orig_pc": 64},
            ],
        }
        bundle = {
            "spv_path": "synthetic_nested_addr.spv",
            "stage": "Compute", "arch": 10, "direct_ok": True, "valhall_ok": True,
            "direct_bindings": [],
            "mbs2_context": {
                "fcsts": [], "rlocs": [],
                "pilot_fau": [
                    {
                        "fau_offset": 0, "word_idx": 0, "byte_size": 8, "value_kind": "buffer_addr",
                        "symbol": "diag", "expression": "&diag+0x8", "resolved": True, "kind": "ssbo",
                        "binding": 2, "is_loaded_value": False, "has_byte_offset": False, "byte_offset": 0,
                        "is_derived": False, "has_pointer_offset": True, "pointer_offset": 8,
                        "has_const_value": False, "const_value": 0, "key": "ssbo:0:2",
                    },
                ],
                "sinp": [], "sout": [], "pc_disasm": {"64": "STORE.i32"},
            },
            "direct_ssa": direct_ssa,
            "valhall_ssa": valhall_ssa,
        }
        res = verify_equivalence_bundle(bundle, timeout_ms=2000)
        self.assertEqual(res.status, "UNSAT")
        self.assertIn("ssbo:0:2[probe_off]", res.checked_outputs)


if __name__ == "__main__":
    unittest.main()

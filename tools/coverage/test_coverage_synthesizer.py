#!/usr/bin/env python3
"""Unit tests for tools.coverage MaxSMT / Z3 coverage-guided input synthesizer."""

from __future__ import annotations

import unittest
from tools.symbolic_ssa.z3simplifier import dsl

from tools.coverage import collect_predicate_nodes, synthesize_coverage_vectors
from tools.verify.cec import run_concrete_diff


class TestCoverageSynthesizer(unittest.TestCase):
    def test_maxsmt_flips_magic_constant_branch_and_proves_dead_branch(self) -> None:
        ssa = {
            "blocks": [{
                "id": 0, "preds": [], "phis": [],
                "nodes": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                "branch_target": -1, "fallthrough": -1,
                "is_cond": False, "is_uncond": False, "is_end": True, "is_discard": False,
                "is_loop_header": False, "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
            }],
            "nodes": [
                {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                {"id": 2, "op": "LoadFragVarying", "type": "V4F32", "blk": 0, "args": [], "imm_u32": 0},
                {"id": 3, "op": "VecExtract", "type": "U32", "blk": 0, "args": [2], "imm_u32": 0},
                {"id": 4, "op": "VecExtract", "type": "U32", "blk": 0, "args": [2], "imm_u32": 1},
                {"id": 5, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 0x1337C0DE},
                {"id": 6, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 0xCAFEBABE},
                {"id": 7, "op": "ICmpEq", "type": "Bool", "blk": 0, "args": [3, 5]},
                {"id": 8, "op": "ICmpEq", "type": "Bool", "blk": 0, "args": [4, 6]},
                {"id": 9, "op": "LogicalAnd", "type": "Bool", "blk": 0, "args": [7, 8]},
                {"id": 10, "op": "Select", "type": "U32", "blk": 0, "args": [9, 5, 3]},
                {"id": 11, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 5},
                {"id": 12, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 6},
                {"id": 13, "op": "ICmpEq", "type": "Bool", "blk": 0, "args": [3, 11]},
                {"id": 14, "op": "ICmpEq", "type": "Bool", "blk": 0, "args": [3, 12]},
                {"id": 15, "op": "LogicalAnd", "type": "Bool", "blk": 0, "args": [13, 14]},
                {"id": 16, "op": "Select", "type": "U32", "blk": 0, "args": [15, 11, 10]},
                {"id": 17, "op": "VecConstruct4", "type": "V4F32", "blk": 0, "args": [16, 16, 16, 16]},
                {"id": 18, "op": "StoreFragColor", "type": "Token", "blk": 0, "args": [17, 1], "imm_u32": 0},
            ],
        }
        bundle = {
            "stage": "Fragment", "direct_ok": True, "valhall_ok": True,
            "mbs2_context": {}, "direct_bindings": [],
            "direct_ssa": ssa, "valhall_ssa": ssa,
        }
        preds = collect_predicate_nodes(ssa)
        self.assertEqual(len(preds), 2)

        cd = run_concrete_diff(bundle, ssa, ssa, k=32)
        res = synthesize_coverage_vectors(bundle, ssa, cd.d_node_traces, is_valhall_side=False)

        self.assertEqual(res.initial_covered, 2)
        self.assertEqual(res.final_covered, 3)
        self.assertEqual(res.dead_proved, 1)
        self.assertIn((9, True), res.covered_targets)
        self.assertIn((15, True), res.dead_targets)
        self.assertGreaterEqual(res.vectors_synthesized, 1)

    def test_maxsmt_flips_multiple_independent_fp_predicates_in_single_vector(self) -> None:
        ssa = {
            "blocks": [{
                "id": 0, "preds": [], "phis": [],
                "nodes": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                "branch_target": -1, "fallthrough": -1,
                "is_cond": False, "is_uncond": False, "is_end": True, "is_discard": False,
                "is_loop_header": False, "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
            }],
            "nodes": [
                {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                {"id": 2, "op": "LoadFragVarying", "type": "V4F32", "blk": 0, "args": [], "imm_u32": 0},
                {"id": 3, "op": "VecExtract", "type": "F32", "blk": 0, "args": [2], "imm_u32": 0},
                {"id": 4, "op": "VecExtract", "type": "F32", "blk": 0, "args": [2], "imm_u32": 1},
                {"id": 5, "op": "VecExtract", "type": "F32", "blk": 0, "args": [2], "imm_u32": 2},
                {"id": 6, "op": "ConstF32", "type": "F32", "blk": 0, "args": [], "imm_u32": 0x449A5000},
                {"id": 7, "op": "ConstF32", "type": "F32", "blk": 0, "args": [], "imm_u32": 0xC61A5100},
                {"id": 8, "op": "ConstF32", "type": "F32", "blk": 0, "args": [], "imm_u32": 0x422A0000},
                {"id": 9, "op": "FCmpOrdGt", "type": "Bool", "blk": 0, "args": [3, 6]},
                {"id": 10, "op": "FCmpOrdLt", "type": "Bool", "blk": 0, "args": [4, 7]},
                {"id": 11, "op": "FCmpOrdEq", "type": "Bool", "blk": 0, "args": [5, 8]},
                {"id": 12, "op": "Select", "type": "F32", "blk": 0, "args": [9, 6, 3]},
                {"id": 13, "op": "Select", "type": "F32", "blk": 0, "args": [10, 7, 4]},
                {"id": 14, "op": "Select", "type": "F32", "blk": 0, "args": [11, 8, 5]},
                {"id": 15, "op": "VecConstruct4", "type": "V4F32", "blk": 0, "args": [12, 13, 14, 14]},
                {"id": 16, "op": "StoreFragColor", "type": "Token", "blk": 0, "args": [15, 1], "imm_u32": 0},
            ],
        }
        bundle = {
            "stage": "Fragment", "direct_ok": True, "valhall_ok": True,
            "mbs2_context": {}, "direct_bindings": [],
            "direct_ssa": ssa, "valhall_ssa": ssa,
        }
        cd = run_concrete_diff(bundle, ssa, ssa, k=32)
        res = synthesize_coverage_vectors(bundle, ssa, cd.d_node_traces, is_valhall_side=False)

        self.assertEqual(res.total_targets, 6)
        self.assertEqual(res.final_covered, 6)
        self.assertEqual(res.vectors_synthesized, 1)


    def test_uf_and_image_array_override_decoding(self) -> None:
        import z3
        from tools.coverage import extract_concrete_overrides_from_model
        from tools.verify.shader_equiv_verifier import BV32, SharedShaderEnv, bv32
        from tools.symbolic_ssa import ConcreteShaderEnv

        bundle = {"stage": "Compute", "direct_ok": True, "valhall_ok": True, "mbs2_context": {}, "direct_bindings": []}
        env = SharedShaderEnv(bundle)
        uf_val = env.get_sym_buf_slot("ssbo:0:0", bv32(16))
        img_arrs = env.get_img_mem("img:0:0")
        s = z3.Solver()
        s.add(dsl.lower(uf_val == bv32(0xDEAD1234)))
        s.add(dsl.lower(dsl.Select(img_arrs[0], dsl.BitVecVal(7, 64)) == bv32(0xBEEF5678)))
        self.assertEqual(s.check(), z3.sat)
        ov = extract_concrete_overrides_from_model(s.model(), env)

        cenv = ConcreteShaderEnv(bundle, 0, overrides=ov)
        self.assertEqual(cenv.read_sym_buf_slot("ssbo:0:0", 16), 0xDEAD1234)
        self.assertEqual(cenv.read_init_img("img:0:0", 7)[0], 0xBEEF5678)

    def test_auto_shrinker_minimizes_inputs_and_localizes_frontier_pc(self) -> None:
        from tools.coverage import localize_and_shrink_divergence

        d_ssa = {
            "blocks": [{
                "id": 0, "preds": [], "phis": [], "nodes": [1, 2, 3, 4, 5, 6, 7, 8],
                "branch_target": -1, "fallthrough": -1, "is_cond": False, "is_uncond": False,
                "is_end": True, "is_discard": False, "is_loop_header": False,
                "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
            }],
            "nodes": [
                {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                {"id": 2, "op": "LoadFragVarying", "type": "V4F32", "blk": 0, "args": [], "imm_u32": 0},
                {"id": 3, "op": "VecExtract", "type": "U32", "blk": 0, "args": [2], "imm_u32": 0},
                {"id": 4, "op": "VecExtract", "type": "U32", "blk": 0, "args": [2], "imm_u32": 1},
                {"id": 5, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 1},
                {"id": 6, "op": "IAdd", "type": "U32", "blk": 0, "args": [3, 5]},
                {"id": 7, "op": "VecConstruct4", "type": "V4F32", "blk": 0, "args": [6, 6, 6, 6]},
                {"id": 8, "op": "StoreFragColor", "type": "Token", "blk": 0, "args": [7, 1], "imm_u32": 0},
            ],
        }
        # Bug in Valhall: ISub instead of IAdd at PC 0x10
        v_ssa = {
            "blocks": d_ssa["blocks"],
            "nodes": [
                {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                {"id": 2, "op": "LoadFragVarying", "type": "V4F32", "blk": 0, "args": [], "imm_u32": 0},
                {"id": 3, "op": "VecExtract", "type": "U32", "blk": 0, "args": [2], "imm_u32": 0},
                {"id": 4, "op": "VecExtract", "type": "U32", "blk": 0, "args": [2], "imm_u32": 1},
                {"id": 5, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 1},
                {"id": 6, "op": "ISub", "type": "U32", "blk": 0, "args": [3, 5], "orig_pc": 0x10},
                {"id": 7, "op": "VecConstruct4", "type": "V4F32", "blk": 0, "args": [6, 6, 6, 6]},
                {"id": 8, "op": "StoreFragColor", "type": "Token", "blk": 0, "args": [7, 1], "imm_u32": 0},
            ],
        }
        bundle = {
            "stage": "Fragment", "direct_ok": True, "valhall_ok": True,
            "mbs2_context": {"pc_disasm": {"16": "IADD.u32 r0, r0, #1 (buggy)"}},
            "direct_bindings": [], "direct_ssa": d_ssa, "valhall_ssa": v_ssa,
        }
        noisy_ov = {
            "frag_var_loc0_0": 42,
            "frag_var_loc0_1": 999,  # irrelevant
            "frag_var_loc0_2": 777,  # irrelevant
            "fau_sym_3": 12345,      # irrelevant
        }
        rep = localize_and_shrink_divergence(bundle, d_ssa, v_ssa, noisy_ov)
        self.assertTrue(rep.diverged)
        self.assertEqual(rep.original_override_count, 4)
        self.assertEqual(rep.shrunk_override_count, 0)  # Even x=0 diverges: 0+1 != 0-1!
        self.assertEqual(rep.frontier_valhall_nids, [6])
        self.assertEqual(rep.frontier_pcs, [(16, "IADD.u32 r0, r0, #1 (buggy)")])

    def test_maximal_reachability_disjoint_path_partition(self) -> None:
        from tools.coverage import solve_maximal_reachability

        # 2-branch CFG with 3 feasible paths (4th path x < 10 && x > 20 is infeasible):
        # bb0: if (x < 10) -> bb1 else -> bb2
        # bb2: if (x > 20) -> bb3 else -> bb4
        # bb1, bb3, bb4 all merge to bb5 (end)
        ssa = {
            "blocks": [
                {"id": 0, "preds": [], "phis": [], "nodes": [1, 2, 3, 4, 5],
                 "branch_target": 1, "fallthrough": 2, "is_cond": True, "is_uncond": False,
                 "is_end": False, "is_discard": False, "is_loop_header": False,
                 "loop_backedge": -1, "loop_merge": -1, "sel_merge": 5, "branch_cond": 5},
                {"id": 1, "preds": [0], "phis": [], "nodes": [],
                 "branch_target": 5, "fallthrough": -1, "is_cond": False, "is_uncond": True,
                 "is_end": False, "is_discard": False, "is_loop_header": False,
                 "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0},
                {"id": 2, "preds": [0], "phis": [], "nodes": [6, 7],
                 "branch_target": 3, "fallthrough": 4, "is_cond": True, "is_uncond": False,
                 "is_end": False, "is_discard": False, "is_loop_header": False,
                 "loop_backedge": -1, "loop_merge": -1, "sel_merge": 5, "branch_cond": 7},
                {"id": 3, "preds": [2], "phis": [], "nodes": [],
                 "branch_target": 5, "fallthrough": -1, "is_cond": False, "is_uncond": True,
                 "is_end": False, "is_discard": False, "is_loop_header": False,
                 "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0},
                {"id": 4, "preds": [2], "phis": [], "nodes": [],
                 "branch_target": 5, "fallthrough": -1, "is_cond": False, "is_uncond": True,
                 "is_end": False, "is_discard": False, "is_loop_header": False,
                 "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0},
                {"id": 5, "preds": [1, 3, 4], "phis": [], "nodes": [8, 9],
                 "branch_target": -1, "fallthrough": -1, "is_cond": False, "is_uncond": False,
                 "is_end": True, "is_discard": False, "is_loop_header": False,
                 "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0},
            ],
            "nodes": [
                {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                {"id": 2, "op": "LoadFragVarying", "type": "V4F32", "blk": 0, "args": [], "imm_u32": 0},
                {"id": 3, "op": "VecExtract", "type": "U32", "blk": 0, "args": [2], "imm_u32": 0},
                {"id": 4, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 10},
                {"id": 5, "op": "ICmpULt", "type": "Bool", "blk": 0, "args": [3, 4]},
                {"id": 6, "op": "ConstU32", "type": "U32", "blk": 2, "args": [], "imm_u32": 20},
                {"id": 7, "op": "ICmpUGt", "type": "Bool", "blk": 2, "args": [3, 6]},
                {"id": 8, "op": "VecConstruct4", "type": "V4F32", "blk": 5, "args": [3, 3, 3, 3]},
                {"id": 9, "op": "StoreFragColor", "type": "Token", "blk": 5, "args": [8, 1], "imm_u32": 0},
            ],
        }
        bundle = {
            "stage": "Fragment", "direct_ok": True, "valhall_ok": True,
            "mbs2_context": {}, "direct_bindings": [], "direct_ssa": ssa, "valhall_ssa": ssa,
        }
        res = solve_maximal_reachability(bundle, ssa, is_valhall_side=False)
        self.assertTrue(res.complete)
        self.assertEqual(res.stop_reason, "EXHAUSTED_ALL_PATHS")
        self.assertEqual(len(res.paths), 3)
        self.assertEqual(res.reached_blocks, {0, 1, 2, 3, 4, 5})
        # Verify all 3 visited-block paths are distinct
        visited_sets = {tuple(p.visited_blocks) for p in res.paths}
        self.assertEqual(visited_sets, {(0, 1, 5), (0, 2, 3, 5), (0, 2, 4, 5)})

    def test_hw_tex_uf_model_replay_matches_z3_branch(self) -> None:
        # Gap 1 & 5: Shader branches on TexSample(uv).r == 0x3F800000 (1.0f).
        # Z3 synthesizes a hw_tex_tex:0:0_c0 FuncInterp; ConcreteSsaInterpreter must replay it identically.
        ssa = {
            "blocks": [{
                "id": 0, "preds": [], "phis": [], "nodes": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "branch_target": -1, "fallthrough": -1, "is_cond": False, "is_uncond": False,
                "is_end": True, "is_discard": False, "is_loop_header": False,
                "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
            }],
            "nodes": [
                {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                {"id": 2, "op": "LoadFragVarying", "type": "V4F32", "blk": 0, "args": [], "imm_u32": 0},
                {"id": 3, "op": "TexSample", "type": "TupleValToken", "blk": 0, "args": [2, 1], "aux_str": "tex:0:0"},
                {"id": 4, "op": "TupleExtract", "type": "V4F32", "blk": 0, "args": [3], "imm_u32": 0},
                {"id": 5, "op": "VecExtract", "type": "U32", "blk": 0, "args": [4], "imm_u32": 0},
                {"id": 6, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 0x3F800000},
                {"id": 7, "op": "ICmpEq", "type": "Bool", "blk": 0, "args": [5, 6]},
                {"id": 8, "op": "Select", "type": "U32", "blk": 0, "args": [7, 6, 5]},
                {"id": 9, "op": "VecConstruct4", "type": "V4F32", "blk": 0, "args": [8, 8, 8, 8]},
                {"id": 10, "op": "StoreFragColor", "type": "Token", "blk": 0, "args": [9, 1], "imm_u32": 0},
            ],
        }
        bundle = {
            "stage": "Fragment", "direct_ok": True, "valhall_ok": True,
            "mbs2_context": {}, "direct_bindings": [], "direct_ssa": ssa, "valhall_ssa": ssa,
        }
        cd = run_concrete_diff(bundle, ssa, ssa, k=8)
        res = synthesize_coverage_vectors(bundle, ssa, cd.d_node_traces, is_valhall_side=False)
        self.assertIn((7, True), res.covered_targets)
        self.assertEqual(res.final_covered, 2)

    def test_fp_diff_policy_separates_benign_ulp_and_nan_from_real_bug(self) -> None:
        # Gap 2: Verify DiffPolicy distinguishes benign 1-ULP / NaN-payload / +/-0 from real bugs.
        from tools.coverage import DiffPolicy, classify_value_diff

        pol = DiffPolicy(max_ulp=2, ignore_nan_payload=True, treat_neg_zero_eq=True)
        # 1.0f (0x3F800000) vs 1.0f + 1 ULP (0x3F800001) on frag_color -> BENIGN_FP_ROUNDING
        k1, ud1 = classify_value_diff(0x3F800000, 0x3F800001, "frag_color[0].r", pol)
        self.assertEqual((k1, ud1), ("BENIGN_FP_ROUNDING", 1))
        # Quiet NaN payload difference -> NAN_PAYLOAD_ONLY
        k2, _ = classify_value_diff(0x7FC00000, 0xFFC00001, "frag_color[0].g", pol)
        self.assertEqual(k2, "NAN_PAYLOAD_ONLY")
        # +0.0 vs -0.0 -> BENIGN_FP_ROUNDING
        k3, _ = classify_value_diff(0x00000000, 0x80000000, "frag_color[0].b", pol)
        self.assertEqual(k3, "BENIGN_FP_ROUNDING")
        # 5 ULP difference -> REAL_DIVERGENCE
        k4, ud4 = classify_value_diff(0x3F800000, 0x3F800005, "frag_color[0].r", pol)
        self.assertEqual((k4, ud4), ("REAL_DIVERGENCE", 5))
        # Integer buffer write 1-LSB difference -> REAL_DIVERGENCE (never FP-relaxed)
        k5, _ = classify_value_diff(0x3F800000, 0x3F800001, "buf[ssbo:0:0:0]", pol)
        self.assertEqual(k5, "REAL_DIVERGENCE")

    def test_adaptive_loop_unrolling_reaches_iter2_predicate(self) -> None:
        # Gap 4: Loop starts at i=0, increments i=i+1; inner predicate (i == 1) is
        # False on iter 1 (max_unroll=1) and only flips True on iter 2 (adaptive_unroll=True).
        ssa = {
            "blocks": [
                {"id": 0, "preds": [], "phis": [], "nodes": [1, 2, 3, 4],
                 "branch_target": 1, "fallthrough": -1, "is_cond": False, "is_uncond": True,
                 "is_end": False, "is_discard": False, "is_loop_header": False,
                 "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0},
                {"id": 1, "preds": [0, 1], "phis": [5], "nodes": [6, 7, 8, 9],
                 "branch_target": 1, "fallthrough": 2, "is_cond": True, "is_uncond": False,
                 "is_end": False, "is_discard": False, "is_loop_header": True,
                 "loop_backedge": 1, "loop_merge": 2, "sel_merge": -1, "branch_cond": 9},
                {"id": 2, "preds": [1], "phis": [], "nodes": [10, 11],
                 "branch_target": -1, "fallthrough": -1, "is_cond": False, "is_uncond": False,
                 "is_end": True, "is_discard": False, "is_loop_header": False,
                 "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0},
            ],
            "nodes": [
                {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                {"id": 2, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 0},
                {"id": 3, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 1},
                {"id": 4, "op": "LoadFragVarying", "type": "V4F32", "blk": 0, "args": [], "imm_u32": 0},
                {"id": 5, "op": "Phi", "type": "U32", "blk": 1, "args": [], "phi": [[0, 2], [1, 8]]},
                {"id": 6, "op": "ICmpEq", "type": "Bool", "blk": 1, "args": [5, 3]},
                {"id": 7, "op": "Select", "type": "U32", "blk": 1, "args": [6, 3, 5]},
                {"id": 8, "op": "IAdd", "type": "U32", "blk": 1, "args": [5, 3]},
                {"id": 9, "op": "ICmpULt", "type": "Bool", "blk": 1, "args": [8, 3]},
                {"id": 10, "op": "VecConstruct4", "type": "V4F32", "blk": 2, "args": [7, 7, 7, 7]},
                {"id": 11, "op": "StoreFragColor", "type": "Token", "blk": 2, "args": [10, 1], "imm_u32": 0},
            ],
        }
        # Make loop continue condition depend on input x > 1 so iter 2 is feasible when x > 1
        ssa["nodes"].append({"id": 12, "op": "VecExtract", "type": "U32", "blk": 0, "args": [4], "imm_u32": 0})
        ssa["blocks"][0]["nodes"].append(12)
        ssa["nodes"][8] = {"id": 9, "op": "ICmpULt", "type": "Bool", "blk": 1, "args": [8, 12]}
        bundle = {
            "stage": "Fragment", "direct_ok": True, "valhall_ok": True,
            "mbs2_context": {}, "direct_bindings": [], "direct_ssa": ssa, "valhall_ssa": ssa,
        }
        # Pretend initial traces only ran x=0 (so iter 2 predicate 6==True was never hit)
        init_traces = {6: [False], 9: [False]}
        res = synthesize_coverage_vectors(
            bundle, ssa, init_traces, is_valhall_side=False,
            max_unroll=1, adaptive_unroll=True, max_adaptive_unroll=3,
        )
        self.assertIn((6, True), res.covered_targets)

if __name__ == "__main__":
    unittest.main()


class TestV30CleanupAndEquivFuzz(unittest.TestCase):
    def test_predicated_kill_collected_as_predicate(self) -> None:
        from tools.coverage.synthesizer import collect_predicate_nodes

        ssa = {
            "blocks": [{
                "id": 0, "preds": [], "phis": [], "nodes": [1, 2, 3, 4],
                "branch_target": -1, "fallthrough": -1,
                "is_cond": False, "is_uncond": False, "is_end": True, "is_discard": False,
                "is_loop_header": False, "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
            }],
            "nodes": [
                {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                {"id": 2, "op": "LoadBuiltinInput", "type": "U32", "blk": 0, "args": [], "imm_u32": 15},
                {"id": 3, "op": "ICmpEq", "type": "Bool", "blk": 0, "args": [2, 2]},
                {"id": 4, "op": "Kill", "type": "Token", "blk": 0, "args": [3, 1], "aux_bool": True},
            ],
        }
        preds = collect_predicate_nodes(ssa)
        self.assertEqual(preds, [(3, 0)])

    def test_concrete_symbolic_parity_idot4_and_atomics(self) -> None:
        import z3
        from tools.symbolic_ssa import ConcreteShaderEnv, ConcreteSsaInterpreter, SharedShaderEnv, SymbolicSsaInterpreter

        ssa = {
            "blocks": [{
                "id": 0, "preds": [], "phis": [], "nodes": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "branch_target": -1, "fallthrough": -1,
                "is_cond": False, "is_uncond": False, "is_end": True, "is_discard": False,
                "is_loop_header": False, "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
            }],
            "nodes": [
                {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                {"id": 2, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 0x01020304},
                {"id": 3, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 0xFF010203},
                {"id": 4, "op": "IDot4S", "type": "S32", "blk": 0, "args": [2, 3]},
                {"id": 5, "op": "IDot4U", "type": "U32", "blk": 0, "args": [2, 3]},
                {"id": 6, "op": "LoadSidebandU64", "type": "U64", "blk": 0, "args": [], "aux_str": "ssbo:0:0"},
                {"id": 7, "op": "AtomicRmw", "type": "TupleValToken", "blk": 0, "args": [6, 4, 1], "imm_u32": 0, "aux_str": "ssbo:0:0"},
                {"id": 8, "op": "TupleExtract", "type": "U32", "blk": 0, "args": [7], "imm_u32": 0},
                {"id": 9, "op": "TupleExtract", "type": "Token", "blk": 0, "args": [7], "imm_u32": 1},
                {"id": 10, "op": "PsbStore", "type": "Token", "blk": 0, "args": [6, 5, 9], "imm_u32": 4, "aux_str": "ssbo:0:0"},
            ],
        }
        bundle = {"stage": "Compute", "mbs2_context": {}, "direct_bindings": []}
        cenv = ConcreteShaderEnv(bundle, vec_idx=0, overrides={"mem:ssbo:0:0:0": 10})
        cvals, _ = ConcreteSsaInterpreter(ssa, is_valhall_side=False).execute_one(cenv)
        # IDot4S: 4*3 + 3*2 + 2*1 + 1*(-1) = 12 + 6 + 2 - 1 = 19
        self.assertEqual(cvals[4], 19)
        # IDot4U: 4*3 + 3*2 + 2*1 + 1*255 = 275
        self.assertEqual(cvals[5], 275)
        self.assertEqual(cvals[8], 10)

        senv = SharedShaderEnv(bundle)
        sev = SymbolicSsaInterpreter(ssa, senv, is_valhall_side=False, fp_mode="exact")
        sev.evaluate()
        s = z3.Solver()
        s.add(dsl.lower(sev.domain.lower_z3_bv32(sev.values[4]) != dsl.BitVecVal(19, 32)))
        self.assertEqual(s.check(), z3.unsat)
        s2 = z3.Solver()
        s2.add(dsl.lower(sev.domain.lower_z3_bv32(sev.values[5]) != dsl.BitVecVal(275, 32)))
        self.assertEqual(s2.check(), z3.unsat)

    def test_concolic_equiv_fuzz_detects_bug_and_passes_clean(self) -> None:
        from tools.coverage.equiv_fuzzer import run_concolic_equiv_fuzz
        from tools.coverage.shrinker import DiffPolicy

        def make_ssa(bug: bool):
            return {
                "blocks": [{
                    "id": 0, "preds": [], "phis": [], "nodes": [1, 2, 3, 4, 5, 6, 7],
                    "branch_target": -1, "fallthrough": -1,
                    "is_cond": False, "is_uncond": False, "is_end": True, "is_discard": False,
                    "is_loop_header": False, "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
                }],
                "nodes": [
                    {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                    {"id": 2, "op": "LoadBuiltinInput", "type": "U32", "blk": 0, "args": [], "imm_u32": 15},
                    {"id": 3, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 0x1337},
                    {"id": 4, "op": "ICmpEq", "type": "Bool", "blk": 0, "args": [2, 3]},
                    {"id": 5, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 99 if bug else 42},
                    {"id": 6, "op": "Select", "type": "U32", "blk": 0, "args": [4, 5, 2]},
                    {"id": 7, "op": "StoreBuiltinOutput", "type": "Token", "blk": 0, "args": [6, 1], "imm_u32": 1},
                ],
            }

        bundle = {"stage": "Vertex", "mbs2_context": {}, "direct_bindings": []}
        d_ssa = make_ssa(False)
        v_clean = make_ssa(False)
        v_bug = make_ssa(True)

        r_ok = run_concolic_equiv_fuzz(bundle, d_ssa, v_clean, seed_side="direct", k_init=8, time_budget_ms=1000)
        self.assertEqual(r_ok.status, "EQUIV_EXACT")
        self.assertEqual(r_ok.diverging_vectors, 0)

        # Bug triggers ONLY when bi_15 == 0x1337 (unreachable by random k_init=8, found by coverage seeder!)
        r_bug = run_concolic_equiv_fuzz(
            bundle, d_ssa, v_bug, seed_side="direct", k_init=8, time_budget_ms=2000, policy=DiffPolicy(max_ulp=0)
        )
        self.assertEqual(r_bug.status, "DIVERGED")
        self.assertGreater(r_bug.coverage_vectors, 0)
        self.assertIsNotNone(r_bug.first_divergence)
        self.assertTrue(r_bug.first_divergence.diverged)

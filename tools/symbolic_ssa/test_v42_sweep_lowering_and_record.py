"""Checkpoint v42 regression tests: sweep bug fixes, triage diagnostics, --record, and .spvasm support."""

from __future__ import annotations

import os
import tempfile
import unittest

from tools.coverage.equiv_fuzzer import run_concolic_equiv_fuzz
from tools.coverage.recorder import record_repro_spvasm
from tools.coverage.shrinker import (
    DiffPolicy,
    align_vertex_gl_position,
    compare_concrete_outputs,
)
from tools.coverage.triage import format_divergence_and_triage, triage_divergence
from tools.symbolic_ssa.env import ConcreteShaderEnv, _f32_to_u32
from tools.symbolic_ssa.passes import canonicalize_ssa_module
from tools.verify import (
    InstLoweringContext,
    LoweringSemanticsExtractor,
    SourceOperandDesc,
)
from tools.equivalence.shader_equiv_verifier import verify_spv_file
from tools.equivalence.spvasm_io import find_spirv_tool, load_equiv_bundle


class TestV42SweepLoweringAndRecord(unittest.TestCase):
    BIN_PATH = "./build/spirv2valhall"

    def test_align_vertex_gl_position_nan_clamp_and_k0fr2f_denom_floor(self) -> None:
        """align_vertex_gl_position must clamp NaN z_ndc to -1.0 and clamp denom >= 2^-15 (k0fr2f)."""
        bundle = {
            "stage": "Vertex",
            "mbs2_context": {
                "rlocs": [{"fau_offset": 0, "word_idx": 0, "symbol": "gl_mali_ViewportTransform"}],
            },
        }
        env = ConcreteShaderEnv(bundle, vec_idx=0)
        # Case 1: NaN in x propagates to z_ndc -> Valhall FADD.clamp_m1_1 clamps NaN to -1.0 (0xbf800000).
        d_out = {
            "builtin_out[0][0]": 0x7FC00000,
            "builtin_out[0][1]": _f32_to_u32(0.0),
            "builtin_out[0][2]": _f32_to_u32(0.5),
            "builtin_out[0][3]": _f32_to_u32(1.0),
        }
        v_out = {
            "builtin_out[0][0]": 0x7FC00000,
            "builtin_out[0][1]": 0x7FC00000,
            "builtin_out[0][2]": 0xBF800000,
            "builtin_out[0][3]": 0x7FC00000,
        }
        aligned = align_vertex_gl_position(d_out, v_out, env, policy=DiffPolicy(max_ulp=2))
        self.assertEqual(aligned["builtin_out[0][2]"], 0xBF800000)

        # Case 2: w == 0.0 -> denom clamped to k0fr2f = 0x38000000 (2^-15), so inv_w = 32768.0 (0x47000000).
        d_out_w0 = {
            "builtin_out[0][0]": _f32_to_u32(2.0),
            "builtin_out[0][1]": _f32_to_u32(-4.0),
            "builtin_out[0][2]": _f32_to_u32(-1.0),
            "builtin_out[0][3]": _f32_to_u32(0.0),
        }
        v_out_w0 = {
            "builtin_out[0][0]": _f32_to_u32(2.0 * 32768.0),
            "builtin_out[0][1]": _f32_to_u32(-4.0 * 32768.0),
            "builtin_out[0][2]": _f32_to_u32(-1.0 * 32768.0),
            "builtin_out[0][3]": _f32_to_u32(32768.0),
        }
        aligned_w0 = align_vertex_gl_position(d_out_w0, v_out_w0, env, policy=DiffPolicy(max_ulp=2))
        self.assertEqual(aligned_w0["builtin_out[0][3]"], _f32_to_u32(32768.0))
        self.assertEqual(aligned_w0["builtin_out[0][0]"], _f32_to_u32(2.0 * 32768.0))

    def test_pass_gvn_dce_imul_to_shl_dominance(self) -> None:
        """IMul(x, 4) -> Shl(x, 2) must never reuse a ConstU32(2) defined in a non-dominating branch block."""
        ssa = {
            "blocks": [
                {"id": 0, "preds": [], "succs": [1, 2], "nodes": [1, 2, 3, 4]},
                {"id": 1, "preds": [0], "succs": [3], "nodes": [10]},
                {"id": 2, "preds": [0], "succs": [3], "nodes": [20, 21, 22]},
                {"id": 3, "preds": [1, 2], "succs": [], "nodes": []},
            ],
            "nodes": [
                {"id": 1, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 0},
                {"id": 2, "op": "LoadVertAttr", "type": "V4F32", "blk": 0, "args": [], "imm_u32": 0},
                {"id": 3, "op": "VecExtract", "type": "U32", "blk": 0, "args": [2], "imm_u32": 0},
                {"id": 4, "op": "BranchCond", "type": "Void", "blk": 0, "args": [3], "imm_u32": 1, "imm_u32_2": 2},
                # Non-dominating ConstU32(2) in block 1 (not executed when taking block 2!)
                {"id": 10, "op": "ConstU32", "type": "U32", "blk": 1, "args": [], "imm_u32": 2},
                {"id": 20, "op": "ConstU32", "type": "U32", "blk": 2, "args": [], "imm_u32": 4},
                {"id": 21, "op": "IMul", "type": "U32", "blk": 2, "args": [3, 20]},
                {"id": 22, "op": "StoreBuiltinOutput", "type": "Token", "blk": 2, "args": [21, 1], "imm_u32": 1},
            ],
        }
        can, _ = canonicalize_ssa_module(ssa)
        entry_nodes = set(can["blocks"][0]["nodes"])
        for n in can["nodes"]:
            if n["op"] == "Shl":
                sh_arg = int(n["args"][1])
                self.assertIn(
                    sh_arg,
                    entry_nodes,
                    f"Shl shift constant v{sh_arg} must live in entry block 0 to dominate all uses",
                )

    def test_float_to_int_lowering_respects_round_mode(self) -> None:
        """F32_TO_U32/F32_TO_S32 lowering must emit RoundEven/Floor/Ceil/Trunc before ConvertFToU/S."""
        ext = LoweringSemanticsExtractor()
        for mnem, conv_op in [("F32_TO_U32", "ConvertFToU"), ("F32_TO_S32", "ConvertFToS")]:
            for rmode, exp_round in [
                ("none", "RoundEven"),
                ("round_down", "Floor"),
                ("round_up", "Ceil"),
                ("round_zero", "Trunc"),
            ]:
                ins = InstLoweringContext(
                    mnemonic=mnem,
                    variant="",
                    srcs=[SourceOperandDesc(reg_num=1)],
                    modifiers={"round": rmode},
                    has_dest=True,
                    dest_reg=0,
                )
                _, tr = ext.lower_instruction(ins)
                self.assertIn(exp_round, tr.emitted_ops, f"{mnem}.{rmode} should emit {exp_round}, got {tr.emitted_ops}")
                self.assertIn(conv_op, tr.emitted_ops, f"{mnem}.{rmode} should emit {conv_op}, got {tr.emitted_ops}")

    def test_fetch_src_u64_subword_swizzle(self) -> None:
        """AND.i64 with kFAFCFDFE.b2 must extract byte 2 (0xfc), not full 32-bit 0xfafcfdfe."""
        ext = LoweringSemanticsExtractor()
        ins = InstLoweringContext(
            mnemonic="AND",
            variant="i64",
            srcs=[
                SourceOperandDesc(kind="SmallConstant", imm_bits=0xFAFCFDFE, swizzle_mod="b2"),
                SourceOperandDesc(kind="Register", reg_num=6, is_64bit=True),
            ],
            modifiers={},
            has_dest=True,
            dest_reg=8,
        )
        _, tr = ext.lower_instruction(ins)
        self.assertIn("BfeU", tr.emitted_ops, f"Expected BfeU for .b2 swizzle on AND.i64, got {tr.emitted_ops}")
        self.assertIn(
            "UConvert32To64",
            tr.emitted_ops,
            f"Expected UConvert32To64 widening on AND.i64, got {tr.emitted_ops}",
        )

    def test_sample_shaders_fixed_in_v42(self) -> None:
        """Previously diverging sample shaders now pass equiv-fuzz after v42 fixes."""
        if not os.path.isfile(self.BIN_PATH):
            self.skipTest("spirv2valhall binary not built")
        for spv in [
            "samples/vert_rep_minimal.spv",
            "samples/vert_stress_xform_f6bc0cf0.spv",
            "samples/bcn_rgtc.spv",
            "samples/bcn_s3tc.spv",
        ]:
            bundle = load_equiv_bundle(spv, self.BIN_PATH)
            d_ssa, _ = canonicalize_ssa_module(bundle["direct_ssa"])
            v_ssa, _ = canonicalize_ssa_module(bundle["valhall_ssa"])
            ef = run_concolic_equiv_fuzz(
                bundle, d_ssa, v_ssa, k_init=64, time_budget_ms=5000, fp_mode="real", domain="udomain"
            )
            self.assertIn(ef.status, ("EQUIV_EXACT", "EQUIV_WITHIN_ULP"), f"{spv} failed: {ef.status}")

    def test_spvasm_record_and_roundtrip_consumption(self) -> None:
        """--record writes a valid .spvasm that both tools.coverage and shader_equiv_verifier consume."""
        if not os.path.isfile(self.BIN_PATH):
            self.skipTest("spirv2valhall binary not built")
        if not find_spirv_tool("spirv-as") or not find_spirv_tool("spirv-dis"):
            self.skipTest("spirv-as / spirv-dis not available on PATH or SPIRV_TOOLS_DIR")

        spv = "samples/frag_unique_05_d392e446.spv"
        bundle = load_equiv_bundle(spv, self.BIN_PATH)
        d_ssa, _ = canonicalize_ssa_module(bundle["direct_ssa"])
        v_ssa, _ = canonicalize_ssa_module(bundle["valhall_ssa"])
        ef = run_concolic_equiv_fuzz(
            bundle, d_ssa, v_ssa, k_init=64, time_budget_ms=5000, fp_mode="real", domain="udomain"
        )
        self.assertEqual(ef.status, "DIVERGED")
        self.assertIsNotNone(ef.first_divergence)
        rep = ef.first_divergence
        assert rep is not None

        triage = triage_divergence(
            bundle,
            d_ssa,
            v_ssa,
            rep,
            raw_direct_ssa=bundle["direct_ssa"],
            raw_valhall_ssa=bundle["valhall_ssa"],
        )
        self.assertEqual(triage.suspect_category, "MODELING_ENV_OR_ABI")
        lines = format_divergence_and_triage(rep, triage, trace_diff=True)
        self.assertTrue(any("[TRIAGE]" in ln for ln in lines))

        with tempfile.TemporaryDirectory() as td:
            out_asm = os.path.join(td, "repro.spvasm")
            rec = record_repro_spvasm(
                spv,
                out_asm,
                self.BIN_PATH,
                bundle,
                rep,
                triage,
                canonicalize=True,
                no_reduce=True,
            )
            self.assertTrue(os.path.isfile(rec.out_path))
            self.assertTrue(rec.verified_reproduces)

            # Consume .spvasm via tools.coverage loader
            b2 = load_equiv_bundle(out_asm, self.BIN_PATH)
            d2, _ = canonicalize_ssa_module(b2["direct_ssa"])
            v2, _ = canonicalize_ssa_module(b2["valhall_ssa"])
            div, _, _, _, _, _ = compare_concrete_outputs(
                b2, d2, v2, overrides=rep.shrunk_overrides, vec_idx=rep.vec_idx
            )
            self.assertTrue(div)

            # Consume .spvasm via shader_equiv_verifier
            res = verify_spv_file(out_asm, spirv2valhall_bin=self.BIN_PATH, timeout_ms=2000)
            self.assertNotEqual(res.status, "ERROR", f"shader_equiv_verifier failed on .spvasm: {res.notes}")


if __name__ == "__main__":
    unittest.main()

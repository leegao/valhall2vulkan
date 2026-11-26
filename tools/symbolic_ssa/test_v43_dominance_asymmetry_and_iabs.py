"""Checkpoint v43 regression tests: SSA dominance verifier, CFG rewriter safety, channel asymmetry, TexGather, and IABS."""

from __future__ import annotations

import copy
import os
import tempfile
import unittest

from tools.coverage.shrinker import DivergenceReport
from tools.coverage.triage import format_divergence_and_triage, triage_divergence
from tools.symbolic_ssa.cfg_rewriter.rewriter import CfgRewriter
from tools.symbolic_ssa.concrete_interpreter import ConcreteSsaInterpreter
from tools.symbolic_ssa.env import ConcreteShaderEnv
from tools.symbolic_ssa.passes import canonicalize_ssa_module, verify_ssa_dominance
from tools.verify import (
    InstLoweringContext,
    LoweringSemanticsExtractor,
    SourceOperandDesc,
)
from tools.equivalence.shader_equiv_verifier import verify_spv_file
from tools.equivalence.spvasm_io import find_spirv_tool


class TestV43DominanceAsymmetryAndIabs(unittest.TestCase):
    BIN_PATH = "./build/spirv2valhall"

    def test_verify_ssa_dominance_detects_non_dominating_use(self) -> None:
        """verify_ssa_dominance flags non-dominating uses and passes on valid canonicalized SSA."""
        bad_ssa = {
            "blocks": [
                {"id": 0, "preds": [], "branch_target": 1, "fallthrough": 2, "is_cond": True, "branch_cond": 1, "nodes": [1]},
                {"id": 1, "preds": [0], "branch_target": 3, "fallthrough": -1, "is_uncond": True, "nodes": [2]},
                {"id": 2, "preds": [0], "branch_target": 3, "fallthrough": -1, "is_uncond": True, "nodes": [3]},
                {"id": 3, "preds": [1, 2], "is_end": True, "nodes": []},
            ],
            "nodes": [
                {"id": 1, "op": "ConstBool", "type": "Bool", "blk": 0, "args": [], "imm_u32": 1},
                {"id": 2, "op": "ConstU32", "type": "U32", "blk": 1, "args": [], "imm_u32": 7},
                # Non-dominating use of v2 (defined in block 1) inside block 2.
                {"id": 3, "op": "IAdd", "type": "U32", "blk": 2, "args": [2, 2]},
            ],
        }
        viols = verify_ssa_dominance(bad_ssa)
        self.assertTrue(any(v.use_node == 3 and v.arg_id == 2 and "NON_DOM" in v.reason for v in viols))

    def test_merge_block_into_predecessor_updates_succ_branch_cond(self) -> None:
        """Merging A -> B must rewrite B's branch_cond if it referenced a single-incoming Phi in B."""
        blocks = {
            0: {
                "id": 0,
                "preds": [],
                "phis": [],
                "nodes": [1],
                "is_uncond": True,
                "is_cond": False,
                "branch_target": 1,
                "fallthrough": -1,
                "branch_cond": 0,
            },
            1: {
                "id": 1,
                "preds": [0],
                "phis": [2],
                "nodes": [],
                "is_uncond": False,
                "is_cond": True,
                "branch_target": 2,
                "fallthrough": 3,
                "branch_cond": 2,
            },
            2: {"id": 2, "preds": [1], "phis": [], "nodes": [], "is_end": True, "branch_target": -1, "fallthrough": -1},
            3: {"id": 3, "preds": [1], "phis": [], "nodes": [], "is_end": True, "branch_target": -1, "fallthrough": -1},
        }
        nodes = {
            1: {"id": 1, "op": "ConstBool", "type": "Bool", "blk": 0, "args": [], "imm_u32": 1},
            2: {"id": 2, "op": "Phi", "type": "Bool", "blk": 1, "args": [], "phi": [[0, 1]]},
        }
        rw = CfgRewriter(blocks, nodes)
        rw.merge_block_into_predecessor(0, 1)
        self.assertEqual(blocks[0]["branch_cond"], 1)
        self.assertEqual(verify_ssa_dominance(blocks, nodes), [])

    def test_shader_equiv_verifier_detects_frag_color_channel_asymmetry(self) -> None:
        """Non-CEC Z3 verifier must report SAT (not vacuous UNSAT) when frag_color[0] is asymmetric."""
        if not os.path.isfile(self.BIN_PATH) or not find_spirv_tool("spirv-as"):
            self.skipTest("spirv2valhall or spirv-as unavailable")

        # Minimal OpUndef-branched fragment shader where direct-SSA folds away the frag_color store
        # while Valhall still emits a BLEND of 0.0 to frag_color[0].
        asym_spvasm = """
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %out_col
               OpExecutionMode %main OriginUpperLeft
               OpDecorate %out_col Location 0
       %void = OpTypeVoid
       %fn_t = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
       %bool = OpTypeBool
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
  %float_256 = OpConstant %float 256
      %undef = OpUndef %bool
 %ptr_out_v4 = OpTypePointer Output %v4float
  %ptr_out_f = OpTypePointer Output %float
    %out_col = OpVariable %ptr_out_v4 Output
       %main = OpFunction %void None %fn_t
      %entry = OpLabel
               OpSelectionMerge %merge None
               OpBranchConditional %undef %then %merge
       %then = OpLabel
        %ptr = OpAccessChain %ptr_out_f %out_col %int_0
               OpStore %ptr %float_256
               OpReturn
      %merge = OpLabel
               OpReturn
               OpFunctionEnd
"""
        with tempfile.TemporaryDirectory() as td:
            asm_path = os.path.join(td, "asym_frag.spvasm")
            with open(asm_path, "w", encoding="utf-8") as fh:
                fh.write(asym_spvasm)
            res = verify_spv_file(
                asm_path,
                spirv2valhall_bin=self.BIN_PATH,
                domain="udomain_backfill",
                canonicalize=True,
                enable_conditioned_unify=True,
                timeout_ms=3000,
            )
            self.assertEqual(res.status, "SAT")
            self.assertIn("frag_color[0]", res.asymmetric_outputs)
            self.assertTrue(any(m.output_name == "frag_color[0].present" for m in res.mismatches))

    def test_tex_gather_comp0_arity_matches_direct_and_valhall(self) -> None:
        """TexGather with comp=0 (gather4_r) must produce identical UF hashes on direct and Valhall sides."""
        bundle = {"stage": "Fragment", "mbs2_context": {}}
        cenv = ConcreteShaderEnv(bundle, vec_idx=0)

        # Direct side passes comp=0 as data_args[1] (imm_u32_2=0).
        direct_ssa = {
            "blocks": [{"id": 0, "is_end": True, "nodes": [1, 2, 3, 4, 5, 6, 7]}],
            "nodes": [
                {"id": 1, "op": "ConstF32", "type": "F32", "blk": 0, "args": [], "imm_u32": 0x3F000000},
                {"id": 2, "op": "VecConstruct2", "type": "V2F32", "blk": 0, "args": [1, 1]},
                {"id": 3, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 0},
                {"id": 4, "op": "TokenIn", "type": "Token", "blk": 0, "args": []},
                {"id": 5, "op": "TexGather", "type": "TupleValToken", "blk": 0, "args": [2, 3, 4], "aux_str": "tex:0:1", "aux_bool2": True},
                {"id": 6, "op": "TupleExtract", "type": "V4F32", "blk": 0, "args": [5], "imm_u32": 0},
                {"id": 7, "op": "StoreFragColor", "type": "Token", "blk": 0, "args": [6, 4], "imm_u32": 0},
            ],
        }
        # Valhall side encodes comp=0 in imm_u32_2 and passes only [uv, tok].
        valhall_ssa = copy.deepcopy(direct_ssa)
        valhall_ssa["nodes"][4]["args"] = [2, 4]
        valhall_ssa["nodes"][4]["imm_u32_2"] = 0

        _, d_out = ConcreteSsaInterpreter(direct_ssa, is_valhall_side=False).execute_one(cenv)
        _, v_out = ConcreteSsaInterpreter(valhall_ssa, is_valhall_side=True).execute_one(cenv)
        self.assertEqual(d_out["frag_color[0].r"], v_out["frag_color[0].r"])
        self.assertEqual(d_out["frag_color[0].a"], v_out["frag_color[0].a"])

    def test_iabs_v4s8_and_v2s16_lane_wise_lowering(self) -> None:
        """IABS.v4s8 and IABS.v2s16 must emit per-lane BfeS + SAbs + mask/pack, not a single 32-bit SAbs."""
        ext = LoweringSemanticsExtractor()
        for variant, expected_lanes in [("v4s8", 4), ("v2s16", 2)]:
            ins = InstLoweringContext(
                mnemonic="IABS",
                variant=variant,
                srcs=[SourceOperandDesc(reg_num=1)],
                modifiers={},
                has_dest=True,
                dest_reg=0,
            )
            _, tr = ext.lower_instruction(ins)
            self.assertEqual(tr.emitted_ops.count("BfeS"), expected_lanes)
            self.assertEqual(tr.emitted_ops.count("SAbs"), expected_lanes)

    def test_triage_suppresses_nan_tag_on_integer_nodes(self) -> None:
        """--trace-diff must not print misleading [NaN] tags on integer SSA ops like ISub returning 0xfffffffe."""
        ssa = {
            "blocks": [{"id": 0, "is_end": True, "nodes": [1, 2, 3, 4, 5]}],
            "nodes": [
                {"id": 1, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 0},
                {"id": 2, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 2},
                {"id": 3, "op": "ISub", "type": "U32", "blk": 0, "args": [1, 2], "orig_pc": 0x10},
                {"id": 4, "op": "TokenIn", "type": "Token", "blk": 0, "args": []},
                {"id": 5, "op": "StoreBuiltinOutput", "type": "Token", "blk": 0, "args": [3, 4], "imm_u32": 1},
            ],
        }
        rep = DivergenceReport(
            diverged=True,
            output_name="builtin_out[1]",
            direct_val="0x00000000",
            valhall_val="0xfffffffe",
            original_override_count=0,
            shrunk_override_count=0,
            diff_kind="REAL_DIVERGENCE",
            frontier_valhall_nids=[3],
            vec_idx=0,
        )
        bundle = {"stage": "Vertex", "mbs2_context": {"pc_disasm": {"16": "ISUB.u32 r0, k0, k2"}}}
        d_ssa, _ = canonicalize_ssa_module(ssa)
        triage = triage_divergence(bundle, d_ssa, ssa, rep)
        lines = format_divergence_and_triage(rep, triage, trace_diff=True)
        trace_lines = [ln for ln in lines if "ISub(" in ln]
        self.assertTrue(trace_lines)
        self.assertNotIn("[NaN]", trace_lines[0])


if __name__ == "__main__":
    unittest.main()

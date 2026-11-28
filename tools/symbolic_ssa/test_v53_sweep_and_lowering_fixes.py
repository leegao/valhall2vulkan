"""Regression tests for Checkpoint v53 sweep, tooling, and ISA-lowering / canonicalizer fixes."""

from __future__ import annotations

import os
import tempfile
import unittest

from tools.coverage.panfrost_vs_ddk import run_panfrost_vs_ddk_equiv
from tools.equivalence.spvasm_io import assemble_spvasm
from tools.symbolic_ssa.passes import canonicalize_ssa_module
from tools.symbolic_ssa.z3utils import dsl


_FRAG_DEPTH_SPVASM = """\
; SPIR-V
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %v_in %frag_color %gl_FragDepth
               OpExecutionMode %main OriginUpperLeft
               OpExecutionMode %main DepthReplacing
               OpDecorate %v_in Location 0
               OpDecorate %frag_color Location 0
               OpDecorate %gl_FragDepth BuiltIn FragDepth
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
 %ptr_in_v4f = OpTypePointer Input %v4float
%ptr_out_v4f = OpTypePointer Output %v4float
  %ptr_out_f = OpTypePointer Output %float
       %v_in = OpVariable %ptr_in_v4f Input
 %frag_color = OpVariable %ptr_out_v4f Output
%gl_FragDepth = OpVariable %ptr_out_f Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
          %v = OpLoad %v4float %v_in
          %x = OpCompositeExtract %float %v 0
               OpStore %gl_FragDepth %x
               OpStore %frag_color %v
               OpReturn
               OpFunctionEnd
"""


class TestV53SweepAndLoweringFixes(unittest.TestCase):
    """Targeted regression tests for Checkpoint v53 fixes."""

    def test_z3expr_deep_dag_hash_no_recursion_error(self) -> None:
        """Deep Z3Expr ITE chains (like bcn_bc7.spv) must hash iteratively without RecursionError."""
        x = dsl.BitVec("x", 32)
        c = dsl.Bool("c")
        cur: dsl.Z3Expr = x
        for i in range(1500):
            cur = dsl.If(c, cur, dsl.BitVecVal(i, 32))
        h = hash(cur)
        self.assertIsInstance(h, int)
        self.assertEqual(h, hash(cur))

    def test_compare_modes_panfrost_vs_direct_and_ddk_vs_direct(self) -> None:
        """panfrost_vs_ddk must support --compare panfrost-vs-direct and ddk-vs-direct."""
        spv = "samples/comp_rep_minimal.spv"
        r_pan = run_panfrost_vs_ddk_equiv(
            spv, k_init=16, fp_range="vulkan", panfrost_backend="kraid", compare="panfrost-vs-direct"
        )
        self.assertEqual(r_pan.status, "EQUIV_EXACT", f"panfrost-vs-direct failed: {r_pan}")
        self.assertEqual(r_pan.compare, "panfrost-vs-direct")

        r_ddk = run_panfrost_vs_ddk_equiv(
            spv, k_init=16, fp_range="vulkan", compare="ddk-vs-direct"
        )
        self.assertEqual(r_ddk.status, "EQUIV_EXACT", f"ddk-vs-direct failed: {r_ddk}")
        self.assertEqual(r_ddk.compare, "ddk-vs-direct")

    def test_zs_emit_depth_operand_matches_direct(self) -> None:
        """ZS_EMIT must store src[0] (depth value), not sr_get(0) (coverage mask r60)."""
        with tempfile.TemporaryDirectory(prefix="v53_zs_") as td:
            spvasm = os.path.join(td, "frag_depth.spvasm")
            spv = os.path.join(td, "frag_depth.spv")
            with open(spvasm, "w", encoding="utf-8") as fh:
                fh.write(_FRAG_DEPTH_SPVASM)
            assemble_spvasm(spvasm, spv)
            r_pan = run_panfrost_vs_ddk_equiv(
                spv, k_init=32, fp_range="vulkan", panfrost_backend="kraid", compare="panfrost-vs-direct"
            )
            self.assertEqual(r_pan.status, "EQUIV_EXACT", f"ZS_EMIT panfrost-vs-direct failed: {r_pan}")
            r_ddk = run_panfrost_vs_ddk_equiv(
                spv, k_init=32, fp_range="vulkan", compare="ddk-vs-direct"
            )
            self.assertEqual(r_ddk.status, "EQUIV_EXACT", f"ZS_EMIT ddk-vs-direct failed: {r_ddk}")

    def test_pass_gvn_dce_cancels_panfrost_persp_var_mul_rcp_frag_w(self) -> None:
        """(1 / frag_w) * (persp_var * frag_w) must cancel to persp_var, not persp_var / frag_w."""
        mod = {
            "blocks": [{"id": 0, "phis": [], "nodes": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "succs": []}],
            "nodes": [
                {"id": 1, "op": "Nop", "type": "Token", "args": []},
                {"id": 2, "op": "LoadBuiltinInput", "type": "V4F32", "imm_u32": 15, "aux_str": "gl_FragCoord", "args": []},
                {"id": 3, "op": "VecExtract", "type": "F32", "imm_u32": 3, "args": [2]},
                {"id": 4, "op": "LoadFragVarying", "type": "V4F32", "imm_u32": 0, "aux_bool": False, "args": []},
                {"id": 5, "op": "VecExtract", "type": "F32", "imm_u32": 0, "args": [4]},
                {"id": 6, "op": "FMul", "type": "F32", "orig_pc": 0x10, "args": [5, 3]},
                {"id": 7, "op": "ConstF32", "type": "F32", "imm_u32": 0x3F800000, "args": []},
                {"id": 8, "op": "FDiv", "type": "F32", "orig_pc": 0x20, "args": [7, 3]},
                {"id": 9, "op": "FMul", "type": "F32", "orig_pc": 0x28, "args": [8, 6]},
                {"id": 10, "op": "VecConstruct4", "type": "V4F32", "args": [9, 9, 9, 9]},
                {"id": 11, "op": "StoreFragColor", "type": "Token", "imm_u32": 0, "args": [10, 1]},
                {"id": 12, "op": "Return", "type": "Void", "args": []},
            ],
        }
        can, _ = canonicalize_ssa_module(mod)
        ops = {n["op"] for n in can["nodes"]}
        self.assertNotIn("FDiv", ops, "FDiv(1, frag_w) should cancel cleanly with FMul(persp_var, frag_w)")
        vec_node = next(n for n in can["nodes"] if n["op"] == "VecConstruct4")
        # All 4 lanes should directly reference the VecExtract(LoadFragVarying, 0) node (id 5)
        self.assertEqual(vec_node["args"], [5, 5, 5, 5])


if __name__ == "__main__":
    unittest.main()

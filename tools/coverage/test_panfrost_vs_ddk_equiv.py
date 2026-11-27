"""CTest regression suite for Panfrost Valhall ISA->SSA vs. Mali DDK Valhall ISA->SSA equivalence fuzzing."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest

from tools.coverage.panfrost_vs_ddk import build_panfrost_vs_ddk_bundle, run_panfrost_vs_ddk_equiv


_SIMPLE_ALU_SPVASM = """\
; SPIR-V
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gl_gid
               OpExecutionMode %main LocalSize 16 1 1
               OpDecorate %gl_gid BuiltIn GlobalInvocationId
               OpDecorate %_arr_u ArrayStride 4
               OpDecorate %BufIn Block
               OpMemberDecorate %BufIn 0 Offset 0
               OpDecorate %BufOut Block
               OpMemberDecorate %BufOut 0 Offset 0
               OpDecorate %in_buf DescriptorSet 0
               OpDecorate %in_buf Binding 0
               OpDecorate %out_buf DescriptorSet 0
               OpDecorate %out_buf Binding 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %v3uint = OpTypeVector %uint 3
  %ptr_in_v3 = OpTypePointer Input %v3uint
     %gl_gid = OpVariable %ptr_in_v3 Input
     %_arr_u = OpTypeRuntimeArray %uint
      %BufIn = OpTypeStruct %_arr_u
     %BufOut = OpTypeStruct %_arr_u
     %ptr_in = OpTypePointer StorageBuffer %BufIn
    %ptr_out = OpTypePointer StorageBuffer %BufOut
   %ptr_u_sb = OpTypePointer StorageBuffer %uint
     %in_buf = OpVariable %ptr_in StorageBuffer
    %out_buf = OpVariable %ptr_out StorageBuffer
      %u32_0 = OpConstant %uint 0
      %u32_1 = OpConstant %uint 1
      %u32_2 = OpConstant %uint 2
       %main = OpFunction %void None %fn
      %entry = OpLabel
        %gid = OpLoad %v3uint %gl_gid
        %idx = OpCompositeExtract %uint %gid 0
       %base = OpIMul %uint %idx %u32_2
       %idx1 = OpIAdd %uint %base %u32_1
        %pa  = OpAccessChain %ptr_u_sb %in_buf %u32_0 %base
        %pb  = OpAccessChain %ptr_u_sb %in_buf %u32_0 %idx1
         %a  = OpLoad %uint %pa
         %b  = OpLoad %uint %pb
        %sum = OpIAdd %uint %a %b
        %res = OpBitwiseXor %uint %sum %a
        %po  = OpAccessChain %ptr_u_sb %out_buf %u32_0 %idx
               OpStore %po %res
               OpReturn
               OpFunctionEnd
"""


class TestPanfrostVsDdkEquiv(unittest.TestCase):
    def test_comp_rep_minimal_panfrost_vs_ddk_equiv_exact(self) -> None:
        """samples/comp_rep_minimal.spv compiles via native Vulkan Panfrost and DDK and is EQUIV_EXACT."""
        res = run_panfrost_vs_ddk_equiv(
            "samples/comp_rep_minimal.spv",
            k_init=32,
            fp_range="vulkan",
            canonicalize=True,
            arch=10,
        )
        self.assertEqual(res.status, "EQUIV_EXACT", f"Expected EQUIV_EXACT, got {res}")
        self.assertEqual(res.stage, "Compute")
        self.assertEqual(res.divs, 0)
        self.assertGreater(res.panfrost_insts, 0)
        self.assertGreater(res.ddk_insts, 0)

    def test_frag_rep_minimal_panfrost_vs_ddk_equiv_exact(self) -> None:
        """samples/frag_rep_minimal.spv compiles via native Vulkan Panfrost and DDK and is EQUIV_EXACT."""
        res = run_panfrost_vs_ddk_equiv(
            "samples/frag_rep_minimal.spv",
            k_init=32,
            fp_range="vulkan",
            canonicalize=True,
            arch=10,
        )
        self.assertEqual(res.status, "EQUIV_EXACT", f"Expected EQUIV_EXACT, got {res}")
        self.assertEqual(res.stage, "Fragment")
        self.assertEqual(res.divs, 0)
        self.assertGreater(res.panfrost_insts, 0)

    def test_vert_rep_minimal_panfrost_vs_ddk_dual_compile(self) -> None:
        """samples/vert_rep_minimal.spv dual-compiles natively on Vertex stage via --panfrost and DDK."""
        bundle, err, fail_kind = build_panfrost_vs_ddk_bundle("samples/vert_rep_minimal.spv", arch=10)
        self.assertIsNotNone(bundle, f"Vertex dual-compile failed ({fail_kind}): {err}")
        self.assertEqual(bundle["stage"], "Vertex")
        self.assertGreater(len(bundle.get("panfrost_pc_disasm", {})), 0)


    def test_comp_rep_minimal_panfrost_bifrost_backend_equiv_exact(self) -> None:
        """samples/comp_rep_minimal.spv compiles via --panfrost-backend bifrost and is EQUIV_EXACT."""
        res = run_panfrost_vs_ddk_equiv(
            "samples/comp_rep_minimal.spv",
            k_init=32,
            fp_range="vulkan",
            canonicalize=True,
            arch=11,
            panfrost_backend="bifrost",
        )
        self.assertEqual(res.status, "EQUIV_EXACT", f"Expected EQUIV_EXACT for bifrost, got {res}")
        self.assertEqual(res.panfrost_backend, "bifrost")
        self.assertEqual(res.divs, 0)

    def test_comp_rep_minimal_panfrost_cl_legacy_equiv_exact(self) -> None:
        """Legacy OpenCL --panfrost-cl path remains operational and EQUIV_EXACT on compute shaders."""
        res = run_panfrost_vs_ddk_equiv(
            "samples/comp_rep_minimal.spv",
            k_init=32,
            fp_range="vulkan",
            canonicalize=True,
            arch=10,
            panfrost_flag="--panfrost-cl",
        )
        self.assertEqual(res.status, "EQUIV_EXACT", f"Expected EQUIV_EXACT for --panfrost-cl, got {res}")
        self.assertEqual(res.divs, 0)

    def test_synthetic_compute_alu_panfrost_vs_ddk_equiv_exact(self) -> None:
        """Vulkan GLCompute SSBO ALU kernel is EQUIV_EXACT between Panfrost ISA->SSA and DDK ISA->SSA."""
        with tempfile.TemporaryDirectory(prefix="test_pan_ddk_") as td:
            spvasm = os.path.join(td, "alu.spvasm")
            spv = os.path.join(td, "alu.spv")
            with open(spvasm, "w") as f:
                f.write(_SIMPLE_ALU_SPVASM)
            subprocess.run(["spirv-as", spvasm, "-o", spv], check=True)
            res = run_panfrost_vs_ddk_equiv(
                spv,
                k_init=32,
                fp_range="vulkan",
                canonicalize=True,
                arch=10,
            )
            self.assertEqual(res.status, "EQUIV_EXACT", f"Expected EQUIV_EXACT, got {res}")
            self.assertEqual(res.divs, 0)
            self.assertEqual(res.triangle, "MATCH")


if __name__ == "__main__":
    unittest.main()

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



_CUBE_SAMPLE_SPVASM = """\
; SPIR-V
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main" %3 %7
               OpExecutionMode %2 OriginUpperLeft
               OpDecorate %3 Location 0
               OpDecorate %7 Location 0
               OpDecorate %8 DescriptorSet 0
               OpDecorate %8 Binding 1
       %void = OpTypeVoid
         %10 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v3float = OpTypeVector %float 3
    %v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
         %25 = OpTypeImage %float Cube 0 0 0 1 Unknown
         %26 = OpTypeSampledImage %25
%_ptr_UniformConstant_26 = OpTypePointer UniformConstant %26
          %3 = OpVariable %_ptr_Input_v4float Input
          %7 = OpVariable %_ptr_Output_v4float Output
          %8 = OpVariable %_ptr_UniformConstant_26 UniformConstant
          %2 = OpFunction %void None %10
         %28 = OpLabel
         %29 = OpLoad %v4float %3
         %33 = OpVectorShuffle %v3float %29 %29 0 1 2
         %34 = OpLoad %26 %8
         %35 = OpImageSampleImplicitLod %v4float %34 %33
               OpStore %7 %35
               OpReturn
               OpFunctionEnd
"""

_FEXP_SPVASM = """\
; SPIR-V
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %2 "main" %gl_gid
               OpExecutionMode %2 LocalSize 32 1 1
               OpDecorate %gl_gid BuiltIn GlobalInvocationId
               OpDecorate %_struct_4 Block
               OpMemberDecorate %_struct_4 0 Offset 0
               OpDecorate %_runtimearr_uint ArrayStride 4
               OpDecorate %6 DescriptorSet 0
               OpDecorate %6 Binding 0
               OpDecorate %7 DescriptorSet 0
               OpDecorate %7 Binding 1
       %void = OpTypeVoid
          %9 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
      %float = OpTypeFloat 32
     %v3uint = OpTypeVector %uint 3
     %uint_0 = OpConstant %uint 0
%_runtimearr_uint = OpTypeRuntimeArray %uint
  %_struct_4 = OpTypeStruct %_runtimearr_uint
%_ptr_sb_s = OpTypePointer StorageBuffer %_struct_4
%_ptr_sb_u = OpTypePointer StorageBuffer %uint
%_ptr_in_v3 = OpTypePointer Input %v3uint
%_ptr_in_u = OpTypePointer Input %uint
     %gl_gid = OpVariable %_ptr_in_v3 Input
          %6 = OpVariable %_ptr_sb_s StorageBuffer
          %7 = OpVariable %_ptr_sb_s StorageBuffer
          %2 = OpFunction %void None %9
         %40 = OpLabel
         %41 = OpAccessChain %_ptr_in_u %gl_gid %uint_0
         %42 = OpLoad %uint %41
         %48 = OpAccessChain %_ptr_sb_u %6 %uint_0 %42
         %52 = OpLoad %uint %48
         %60 = OpBitcast %float %52
         %66 = OpExtInst %float %1 Exp2 %60
         %67 = OpBitcast %uint %66
         %69 = OpAccessChain %_ptr_sb_u %7 %uint_0 %42
               OpStore %69 %67
               OpReturn
               OpFunctionEnd
"""

_IABS_V4S8_SPVASM = """\
; SPIR-V
               OpCapability Shader
               OpCapability Int8
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %2 "main" %gl_gid
               OpExecutionMode %2 LocalSize 32 1 1
               OpDecorate %gl_gid BuiltIn GlobalInvocationId
               OpDecorate %_struct_4 Block
               OpMemberDecorate %_struct_4 0 Offset 0
               OpDecorate %_runtimearr_uint ArrayStride 4
               OpDecorate %6 DescriptorSet 0
               OpDecorate %6 Binding 0
               OpDecorate %7 DescriptorSet 0
               OpDecorate %7 Binding 1
       %void = OpTypeVoid
          %9 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
       %char = OpTypeInt 8 1
     %v3uint = OpTypeVector %uint 3
     %v4char = OpTypeVector %char 4
     %uint_0 = OpConstant %uint 0
%_runtimearr_uint = OpTypeRuntimeArray %uint
  %_struct_4 = OpTypeStruct %_runtimearr_uint
%_ptr_sb_s = OpTypePointer StorageBuffer %_struct_4
%_ptr_sb_u = OpTypePointer StorageBuffer %uint
%_ptr_in_v3 = OpTypePointer Input %v3uint
%_ptr_in_u = OpTypePointer Input %uint
     %gl_gid = OpVariable %_ptr_in_v3 Input
          %6 = OpVariable %_ptr_sb_s StorageBuffer
          %7 = OpVariable %_ptr_sb_s StorageBuffer
          %2 = OpFunction %void None %9
         %46 = OpLabel
         %47 = OpAccessChain %_ptr_in_u %gl_gid %uint_0
         %48 = OpLoad %uint %47
         %54 = OpAccessChain %_ptr_sb_u %6 %uint_0 %48
         %58 = OpLoad %uint %54
         %72 = OpBitcast %v4char %58
         %73 = OpExtInst %v4char %1 SAbs %72
         %74 = OpBitcast %uint %73
         %76 = OpAccessChain %_ptr_sb_u %7 %uint_0 %48
               OpStore %76 %74
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



    def _assemble_and_check(self, spvasm_text: str, backend: str = "kraid") -> None:
        with tempfile.TemporaryDirectory(prefix="test_pan_ddk_") as td:
            spvasm = os.path.join(td, "shader.spvasm")
            spv = os.path.join(td, "shader.spv")
            with open(spvasm, "w") as f:
                f.write(spvasm_text)
            subprocess.run(["spirv-as", spvasm, "-o", spv], check=True)
            res = run_panfrost_vs_ddk_equiv(
                spv,
                k_init=32,
                fp_range="vulkan",
                canonicalize=True,
                arch=11,
                panfrost_backend=backend,
            )
            self.assertEqual(res.status, "EQUIV_EXACT", f"Expected EQUIV_EXACT, got {res}")
            self.assertEqual(res.divs, 0)
            self.assertEqual(res.triangle, "MATCH")

    def test_tex_single_cube_project_kraid_equiv_exact(self) -> None:
        """TEX_SINGLE.cube.project from Panfrost Kraid matches DDK and Direct-SSA."""
        self._assemble_and_check(_CUBE_SAMPLE_SPVASM, backend="kraid")

    def test_fadd_rscale_fexp_kraid_equiv_exact(self) -> None:
        """FADD_RSCALE + FEXP from Panfrost Kraid matches DDK and Direct-SSA."""
        self._assemble_and_check(_FEXP_SPVASM, backend="kraid")

    def test_iabs_v2s16_subword_swizzle_and_mkvec_v2i8_equiv_exact(self) -> None:
        """IABS.v2s16 (.b01/.b23) + MKVEC.v2i8 from Panfrost Kraid matches DDK and Direct-SSA."""
        self._assemble_and_check(_IABS_V4S8_SPVASM, backend="kraid")

    def test_panfrost_abi_bridge_idvs_and_viewport_devirt(self) -> None:
        """Explicit Panfrost ABI bridge splits IDVS vertex binary and devirtualizes viewport sysvals."""
        import json

        with tempfile.TemporaryDirectory(prefix="test_pan_bridge_") as td:
            out_json = os.path.join(td, "vert_equiv.json")
            subprocess.run(
                [
                    "./build/spirv2valhall",
                    "samples/vert_rep_minimal.spv",
                    "--panfrost",
                    "--opt-ssa",
                    "--dump-equiv-json",
                    out_json,
                    "--quiet",
                ],
                check=True,
            )
            with open(out_json) as f:
                data = json.load(f)
            bridge = data.get("panfrost_abi_bridge", {})
            self.assertTrue(bridge.get("idvs_split"), f"Expected idvs_split=True, got {bridge}")
            self.assertGreater(bridge.get("primary_insts", 0), 0)
            self.assertGreater(bridge.get("secondary_insts", 0), 0)
            self.assertEqual(bridge.get("viewport_words_devirt"), 6)


    def test_v52_vert_rep_minimal_idvs_varying_and_ncomps_equiv_exact(self) -> None:
        """v52: IDVS varying byte-offset map + pre-lowering ncomps undef-lane zero-pad -> EQUIV_EXACT."""
        r = run_panfrost_vs_ddk_equiv(
            "samples/vert_rep_minimal.spv",
            k_init=16,
            spirv2valhall_bin="./build/spirv2valhall",
            time_budget_ms=0,
            compute_triangle=True,
            panfrost_backend="kraid",
        )
        self.assertEqual(r.status, "EQUIV_EXACT", f"vert_rep_minimal failed: {r.status} divs={r.divs}/{r.vecs}")

    def test_v52_bcn_bc7_rodata_and_local_array_256_kraid_vs_direct_exact(self) -> None:
        """v52: BC7 Mode 2 Partition 62 (64-entry LocalVar array + carved Kraid .rodata) -> Kraid == Direct."""
        r = run_panfrost_vs_ddk_equiv(
            "samples/bcn_bc7.spv",
            k_init=40,
            spirv2valhall_bin="./build/spirv2valhall",
            time_budget_ms=0,
            compute_triangle=True,
            panfrost_backend="kraid",
            defau=False,
            opt_ssa=False,
        )
        self.assertEqual(
            r.direct_vs_panfrost,
            "EQUIV_EXACT",
            f"bcn_bc7 Kraid vs Direct failed: {r.direct_vs_panfrost} (pan_ddk={r.status})",
        )

    def test_v52_synthetic_kraid_lowering_fixes_equiv_exact(self) -> None:
        """v52: IADD.v2s16/v2u16 .b01, TEX_GRADIENT->IADD.u64, ADR .rodata, LD_VAR_SPECIAL, LD_VAR/FLAT."""
        from tools.corpus_synth.isa_target_synth import build_spvasm_for_isa

        targets = [
            "V2S8_TO_V2S16",
            "V2U8_TO_V2U16",
            "TEX_GRADIENT",
            "ADR",
            "LD_VAR_SPECIAL",
            "LD_VAR",
            "LD_VAR_FLAT",
        ]
        with tempfile.TemporaryDirectory(prefix="test_v52_synth_") as td:
            for mnem in targets:
                _, asm = build_spvasm_for_isa(mnem)
                spv_path = os.path.join(td, f"isa_{mnem.lower()}.spv")
                p_as = subprocess.run(
                    ["spirv-as", "--target-env", "vulkan1.2", "-", "-o", spv_path],
                    input=asm.encode(),
                    capture_output=True,
                )
                self.assertEqual(p_as.returncode, 0, f"spirv-as failed for {mnem}: {p_as.stderr.decode()}")
                r = run_panfrost_vs_ddk_equiv(
                    spv_path,
                    k_init=16,
                    spirv2valhall_bin="./build/spirv2valhall",
                    time_budget_ms=0,
                    compute_triangle=True,
                    panfrost_backend="kraid",
                )
                self.assertEqual(
                    r.status,
                    "EQUIV_EXACT",
                    f"{mnem} failed: status={r.status} divs={r.divs}/{r.vecs} tri={r.triangle}",
                )

    def test_cli_flags_and_maximal_reachability_seed(self) -> None:
        """CLI supports --domain udomain_backfill, --defau/--no-defau, --opt-ssa/--no-opt-ssa, --maximal-reachability, --maximal-reachability-seed, --slice-diff, --trace-diff, --record."""
        import io
        from contextlib import redirect_stdout
        from tools.coverage.panfrost_vs_ddk import main as pan_main

        with tempfile.TemporaryDirectory(prefix="test_pan_cli_") as td:
            spv_path = os.path.join(td, "alu.spv")
            subprocess.run(
                ["spirv-as", "--target-env", "vulkan1.2", "-", "-o", spv_path],
                input=_SIMPLE_ALU_SPVASM.encode(),
                check=True,
                capture_output=True,
            )
            vec_json = os.path.join(td, "vecs.json")
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = pan_main([
                    spv_path,
                    "--bin", "./build/spirv2valhall",
                    "--domain", "udomain_backfill",
                    "--defau",
                    "--opt-ssa",
                    "--maximal-reachability-seed",
                    "--k-init", "4",
                    "--time-budget-ms", "500",
                    "--slice-diff", "4",
                    "--trace-diff",
                    "--dump-vectors", vec_json,
                ])
            self.assertEqual(rc, 0, buf.getvalue())
            self.assertIn("EQUIV_EXACT", buf.getvalue())
            self.assertTrue(os.path.isfile(vec_json))

            buf_reach = io.StringIO()
            with redirect_stdout(buf_reach):
                rc_reach = pan_main([
                    spv_path,
                    "--bin", "./build/spirv2valhall",
                    "--domain", "udomain_backfill",
                    "--no-defau",
                    "--no-opt-ssa",
                    "--maximal-reachability",
                ])
            self.assertEqual(rc_reach, 0, buf_reach.getvalue())
            self.assertIn("[REACHABILITY]", buf_reach.getvalue())

if __name__ == "__main__":
    unittest.main()

"""Longitudinal compositional equivalence fuzzer (Checkpoint v48.2).

Composes chains of 4–8 distinct Valhall-emitting SPIR-V operations per shader
to catch cross-instruction interactions (register pressure, intermediate
rounding, 64-bit comparisons, subword bitcasts, atomics, and CFG phis) that
single-instruction ISA recipes cannot trigger.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from tools.coverage.equiv_fuzzer import run_concolic_equiv_fuzz


COMPOSED_COMPUTE_SHADERS: dict[str, str] = {
    "comp_01_v2f16_fma_round_iabs_s16": """
      %a16 = OpBitcast %v2half %u0
      %b16 = OpBitcast %v2half %u1
      %c16 = OpBitcast %v2half %u2
      %fma16 = OpExtInst %v2half %glsl Fma %a16 %b16 %c16
      %rnd16 = OpExtInst %v2half %glsl Floor %fma16
      %pk1 = OpBitcast %uint %rnd16
      %s16v = OpBitcast %v2short %pk1
      %abs16 = OpExtInst %v2short %glsl SAbs %s16v
      %lo16 = OpCompositeExtract %short %abs16 0
      %f32v = OpConvertSToF %float %lo16
      %res = OpBitcast %uint %f32v
    """,
    "comp_02_v4u8_iadd_idpadd_iabs_mkvec": """
      %b0 = OpBitcast %v4uchar %u0
      %b1 = OpBitcast %v4uchar %u1
      %sum8 = OpIAdd %v4uchar %b0 %b1
      %pk8 = OpBitcast %uint %sum8
      %dp = OpUDotAccSat %uint %pk8 %u2 %u3 PackedVectorFormat4x8Bit
      %s8v = OpBitcast %v4char %dp
      %a8 = OpExtInst %v4char %glsl SAbs %s8v
      %shuf = OpVectorShuffle %v4char %a8 %a8 3 1 2 0
      %res = OpBitcast %uint %shuf
    """,
    "comp_03_bfi_bfe_msb_popcnt_div": """
      %ext = OpBitFieldUExtract %uint %u0 %uint_4 %uint_12
      %ins = OpBitFieldInsert %uint %u1 %ext %uint_8 %uint_12
      %msb = OpExtInst %uint %glsl FindUMsb %ins
      %cnt = OpBitCount %uint %u2
      %den = OpBitwiseOr %uint %cnt %uint_1
      %q = OpUDiv %uint %ins %den
      %res = OpIAdd %uint %q %msb
    """,
    "comp_04_fp32_fma_clamp_csel_branch": """
      %f0 = OpBitcast %float %u0
      %f1 = OpBitcast %float %u1
      %f2 = OpBitcast %float %u2
      %m = OpExtInst %float %glsl Fma %f0 %f1 %f2
      %cl = OpExtInst %float %glsl FClamp %m %float_0 %float_1
      %cmp = OpFOrdGreaterThan %bool %cl %float_0_5
      %sel = OpSelect %float %cmp %f0 %f1
      %r1 = OpBitcast %uint %sel
      %res = OpBitwiseXor %uint %r1 %u3
    """,
    "comp_05_int64_arith_cmp_pack": """
      %v2a = OpCompositeConstruct %v2uint %u0 %u1
      %v2b = OpCompositeConstruct %v2uint %u2 %u3
      %q0 = OpBitcast %ulong %v2a
      %q1 = OpBitcast %ulong %v2b
      %qa = OpIAdd %ulong %q0 %q1
      %qm = OpIMul %ulong %qa %q0
      %gt = OpUGreaterThan %bool %qm %q1
      %up = OpBitcast %v2uint %qm
      %lo = OpCompositeExtract %uint %up 0
      %hi = OpCompositeExtract %uint %up 1
      %res = OpSelect %uint %gt %lo %hi
    """,
    "comp_06_atom_acmpxchg_mix": """
      %p_out1 = OpAccessChain %_ptr_StorageBuffer_uint %out_buf %uint_0 %idx1
      %old1 = OpAtomicIAdd %uint %p_out1 %uint_1 %uint_0 %u0
      %old2 = OpAtomicCompareExchange %uint %p_out1 %uint_1 %uint_0 %uint_0 %u1 %old1
      %res = OpIAdd %uint %old1 %old2
    """,
    "comp_07_f16_to_s32_s8_to_f32_chain": """
      %h2 = OpBitcast %v2half %u0
      %h0 = OpCompositeExtract %half %h2 0
      %i32 = OpConvertFToS %int %h0
      %b4 = OpBitcast %v4char %u1
      %b0 = OpCompositeExtract %char %b4 0
      %fb = OpConvertSToF %float %b0
      %ib = OpConvertFToS %int %fb
      %sum = OpIAdd %int %i32 %ib
      %res = OpBitcast %uint %sum
    """,
    "comp_08_nested_branch_loop_phi": """
      %c0 = OpUGreaterThan %bool %u0 %u1
      OpSelectionMerge %m_if None
      OpBranchConditional %c0 %b_then %b_else
      %b_then = OpLabel
      %t_val = OpIAdd %uint %u0 %u2
      OpBranch %m_if
      %b_else = OpLabel
      %e_val = OpISub %uint %u1 %u3
      OpBranch %m_if
      %m_if = OpLabel
      %phi = OpPhi %uint %t_val %b_then %e_val %b_else
      %res = OpBitwiseXor %uint %phi %u2
    """,
    "comp_09_int64_signed_cmp_shift_sub": """
      %v2a = OpCompositeConstruct %v2uint %u0 %u1
      %v2b = OpCompositeConstruct %v2uint %u2 %u3
      %q0 = OpBitcast %long %v2a
      %q1 = OpBitcast %long %v2b
      %qd = OpISub %long %q0 %q1
      %lt = OpSLessThan %bool %qd %q1
      %up = OpBitcast %v2uint %qd
      %lo = OpCompositeExtract %uint %up 0
      %hi = OpCompositeExtract %uint %up 1
      %res = OpSelect %uint %lt %lo %hi
    """,
    "comp_10_bitrev_clz_subword_pack": """
      %rev = OpBitReverse %uint %u0
      %lsb = OpExtInst %uint %glsl FindILsb %u1
      %b4 = OpBitcast %v4uchar %rev
      %b0 = OpCompositeExtract %uchar %b4 2
      %u8_32 = OpUConvert %uint %b0
      %res = OpIAdd %uint %u8_32 %lsb
    """,
    "comp_11_v2f16_min_max_add_pack": """
      %h0 = OpBitcast %v2half %u0
      %h1 = OpBitcast %v2half %u1
      %h2 = OpBitcast %v2half %u2
      %mn = OpExtInst %v2half %glsl FMin %h0 %h1
      %mx = OpExtInst %v2half %glsl FMax %mn %h2
      %ad = OpFAdd %v2half %mx %h0
      %res = OpBitcast %uint %ad
    """,
    "comp_12_counted_loop_phi_accum": """
      OpBranch %l_hdr
      %l_hdr = OpLabel
      %i_phi = OpPhi %uint %uint_0 %entry %i_next %l_cont
      %acc_phi = OpPhi %uint %u0 %entry %acc_next %l_cont
      %l_cond = OpULessThan %bool %i_phi %uint_4
      OpLoopMerge %l_exit %l_cont None
      OpBranchConditional %l_cond %l_body %l_exit
      %l_body = OpLabel
      %step = OpIMul %uint %acc_phi %u1
      %acc_next = OpBitwiseXor %uint %step %u2
      OpBranch %l_cont
      %l_cont = OpLabel
      %i_next = OpIAdd %uint %i_phi %uint_1
      OpBranch %l_hdr
      %l_exit = OpLabel
      %res = OpIAdd %uint %acc_phi %u3
    """,
}

COMPUTE_TEMPLATE = """OpCapability Shader
OpCapability Float16
OpCapability Int16
OpCapability Int8
OpCapability Int64
OpCapability DotProduct
OpCapability DotProductInput4x8BitPacked
OpExtension "SPV_KHR_integer_dot_product"
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %gl_gid
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %gl_gid BuiltIn GlobalInvocationId
OpDecorate %_arr_uint ArrayStride 4
OpMemberDecorate %Buf 0 Offset 0
OpDecorate %Buf Block
OpDecorate %in_buf DescriptorSet 0
OpDecorate %in_buf Binding 0
OpDecorate %out_buf DescriptorSet 0
OpDecorate %out_buf Binding 1
%void = OpTypeVoid
%fn_t = OpTypeFunction %void
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%int = OpTypeInt 32 1
%ulong = OpTypeInt 64 0
%long = OpTypeInt 64 1
%short = OpTypeInt 16 1
%ushort = OpTypeInt 16 0
%char = OpTypeInt 8 1
%uchar = OpTypeInt 8 0
%float = OpTypeFloat 32
%half = OpTypeFloat 16
%v2uint = OpTypeVector %uint 2
%v3uint = OpTypeVector %uint 3
%v2half = OpTypeVector %half 2
%v2short = OpTypeVector %short 2
%v4char = OpTypeVector %char 4
%v4uchar = OpTypeVector %uchar 4
%_arr_uint = OpTypeRuntimeArray %uint
%Buf = OpTypeStruct %_arr_uint
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%_ptr_StorageBuffer_uint = OpTypePointer StorageBuffer %uint
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
%_ptr_Input_uint = OpTypePointer Input %uint
%gl_gid = OpVariable %_ptr_Input_v3uint Input
%in_buf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%out_buf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%uint_3 = OpConstant %uint 3
%uint_4 = OpConstant %uint 4
%uint_8 = OpConstant %uint 8
%uint_12 = OpConstant %uint 12
%float_0 = OpConstant %float 0.0
%float_0_5 = OpConstant %float 0.5
%float_1 = OpConstant %float 1.0
%main = OpFunction %void None %fn_t
%entry = OpLabel
%p_gx = OpAccessChain %_ptr_Input_uint %gl_gid %uint_0
%gx = OpLoad %uint %p_gx
%base = OpShiftLeftLogical %uint %gx %uint_2
%idx0 = OpIAdd %uint %base %uint_0
%idx1 = OpIAdd %uint %base %uint_1
%idx2 = OpIAdd %uint %base %uint_2
%idx3 = OpIAdd %uint %base %uint_3
%p0 = OpAccessChain %_ptr_StorageBuffer_uint %in_buf %uint_0 %idx0
%p1 = OpAccessChain %_ptr_StorageBuffer_uint %in_buf %uint_0 %idx1
%p2 = OpAccessChain %_ptr_StorageBuffer_uint %in_buf %uint_0 %idx2
%p3 = OpAccessChain %_ptr_StorageBuffer_uint %in_buf %uint_0 %idx3
%u0 = OpLoad %uint %p0
%u1 = OpLoad %uint %p1
%u2 = OpLoad %uint %p2
%u3 = OpLoad %uint %p3
{BODY}
%po = OpAccessChain %_ptr_StorageBuffer_uint %out_buf %uint_0 %gx
OpStore %po %res
OpReturn
OpFunctionEnd
"""


@dataclass
class LongitudinalCaseResult:
    name: str
    status: str
    benign_fp: int


def run_longitudinal_suite(
    names: list[str] | None = None,
    k_init: int = 16,
    fp_range: str = "vulkan",
    bin_path: str = "./build/spirv2valhall",
) -> list[LongitudinalCaseResult]:
    selected = names if names is not None else sorted(COMPOSED_COMPUTE_SHADERS.keys())
    out: list[LongitudinalCaseResult] = []
    for name in selected:
        body = COMPOSED_COMPUTE_SHADERS[name]
        with tempfile.TemporaryDirectory() as td:
            asm = Path(td) / f"{name}.spvasm"
            spv = Path(td) / f"{name}.spv"
            eqj = Path(td) / "eq.json"
            asm.write_text(COMPUTE_TEMPLATE.format(BODY=body.strip()))
            subprocess.run(["spirv-as", "--target-env", "vulkan1.3", str(asm), "-o", str(spv)], check=True)
            subprocess.run([bin_path, "--opt-ssa", "--defau", "--dump-equiv-json", str(eqj), "--quiet", str(spv)], check=True)
            b = json.loads(eqj.read_text())
            ef = run_concolic_equiv_fuzz(b, b["direct_ssa"], b["valhall_ssa"], k_init=k_init, fp_range=fp_range)
            out.append(LongitudinalCaseResult(name=name, status=ef.status, benign_fp=ef.benign_fp_diffs))
    return out


if __name__ == "__main__":
    for r in run_longitudinal_suite():
        print(f"{r.name:38s}: {r.status} (benign_fp={r.benign_fp})")

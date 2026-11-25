"""
SPIR-V Assembly Harness Builders for Targeted Valhall ISA Synthesis.

Provides clean, reusable Compute (`LocalSize 32 1 1`), Fragment, and Vertex
Vulkan 1.3 SPIR-V templates that enforce the four Mali ISel synthesis
invariants:
  1. Anti-Pilot-hoisting (thread-varying `gl_GlobalInvocationID.x & 0xFFFF`).
  2. Anti-DCE sink (live store to `outbuf[gx]`, `out_col`, or `gl_Position`).
  3. Clean capability/extension emission (`Int8`, `Int16`, `Int64`, `Float16`,
     `DotProduct`, `GroupNonUniform*`, dynamic descriptor/array indexing).
  4. Three complexity tiers (`minimal`, `medium`, `branchy`).
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class IsaRecipe:
    """Executable SPIR-V assembly snippet that induces one or more Valhall mnemonics."""

    stage: str  # "compute", "fragment", "vertex"
    body: str
    caps: tuple[str, ...] = ()
    exts: tuple[str, ...] = ()
    needs_wg: bool = False
    needs_tex2d: bool = False
    needs_storage_img: bool = False
    needs_storage_img_array: bool = False
    needs_ssbo_array: bool = False
    needs_cube: bool = False
    needs_rodata: bool = False
    needs_frag_depth: bool = False
    needs_dyn_var_array: bool = False
    needs_dyn_var_flat_array: bool = False
    needs_dyn_attr_in_array: bool = False
    needs_dyn_attr_out_array: bool = False
    result_var: str = "%res"


def build_compute_spvasm(recipe: IsaRecipe, complexity: str = "minimal") -> str:
    """Wrap a compute `IsaRecipe` in a Vulkan 1.3 GLCompute SPIR-V harness."""
    caps = ["Shader"] + list(recipe.caps)
    if recipe.needs_storage_img or recipe.needs_storage_img_array:
        caps.append("StorageImageReadWithoutFormat")
        caps.append("StorageImageWriteWithoutFormat")
    if recipe.needs_storage_img_array:
        caps.append("StorageImageArrayDynamicIndexing")
    if recipe.needs_ssbo_array:
        caps.append("StorageBufferArrayDynamicIndexing")
    cap_lines = "\n".join(f"OpCapability {c}" for c in dict.fromkeys(caps))
    ext_lines = "\n".join(f'OpExtension "{e}"' for e in dict.fromkeys(recipe.exts))

    decorations = [
        "OpDecorate %gl_gid BuiltIn GlobalInvocationId",
        "OpDecorate %Buf Block",
        "OpMemberDecorate %Buf 0 Offset 0",
        "OpDecorate %RArr ArrayStride 4",
        "OpDecorate %inbuf DescriptorSet 0",
        "OpDecorate %inbuf Binding 0",
        "OpDecorate %outbuf DescriptorSet 0",
        "OpDecorate %outbuf Binding 1",
    ]
    extra_types = []
    extra_vars = []
    if recipe.needs_wg:
        extra_types.append("%WgArr = OpTypeArray %u32 %c32")
        extra_types.append("%pWgArr = OpTypePointer Workgroup %WgArr")
        extra_types.append("%pWgU32 = OpTypePointer Workgroup %u32")
        extra_vars.append("%wg = OpVariable %pWgArr Workgroup")
    if recipe.needs_tex2d:
        decorations.append("OpDecorate %tex2d DescriptorSet 0")
        decorations.append("OpDecorate %tex2d Binding 2")
        extra_types.append("%tImg2D = OpTypeImage %f32 2D 0 0 0 1 Unknown")
        extra_types.append("%tSampledImg2D = OpTypeSampledImage %tImg2D")
        extra_types.append("%pSampledImg2D = OpTypePointer UniformConstant %tSampledImg2D")
        extra_vars.append("%tex2d = OpVariable %pSampledImg2D UniformConstant")
    if recipe.needs_storage_img:
        decorations.append("OpDecorate %simg2d DescriptorSet 0")
        decorations.append("OpDecorate %simg2d Binding 3")
        extra_types.append("%tStorImg2D = OpTypeImage %u32 2D 0 0 0 2 Unknown")
        extra_types.append("%pStorImg2D = OpTypePointer UniformConstant %tStorImg2D")
        extra_vars.append("%simg2d = OpVariable %pStorImg2D UniformConstant")
    if recipe.needs_storage_img_array:
        decorations.append("OpDecorate %simg_arr DescriptorSet 0")
        decorations.append("OpDecorate %simg_arr Binding 4")
        extra_types.append("%tStorImgArrElem = OpTypeImage %u32 2D 0 0 0 2 Unknown")
        extra_types.append("%tStorImgArr = OpTypeArray %tStorImgArrElem %c4")
        extra_types.append("%pStorImgArr = OpTypePointer UniformConstant %tStorImgArr")
        extra_types.append("%pStorImgArrElem = OpTypePointer UniformConstant %tStorImgArrElem")
        extra_vars.append("%simg_arr = OpVariable %pStorImgArr UniformConstant")
    if recipe.needs_ssbo_array:
        decorations.append("OpDecorate %ssbo_arr DescriptorSet 0")
        decorations.append("OpDecorate %ssbo_arr Binding 5")
        extra_types.append("%ArrBuf = OpTypeArray %Buf %c4")
        extra_types.append("%pArrBuf = OpTypePointer StorageBuffer %ArrBuf")
        extra_vars.append("%ssbo_arr = OpVariable %pArrBuf StorageBuffer")
    if recipe.needs_rodata:
        extra_types.append("%RoArr = OpTypeArray %u32 %c16")
        extra_types.append("%pFnRoArr = OpTypePointer Function %RoArr")
        extra_types.append("%pFnU32 = OpTypePointer Function %u32")
        ro_elems = " ".join(f"%ro_{i}" for i in range(16))
        for i in range(16):
            extra_types.append(f"%ro_{i} = OpConstant %u32 {0x11110000 + i * 0x1357}")
        extra_types.append(f"%ro_init = OpConstantComposite %RoArr {ro_elems}")

    type_block = [
        "%void = OpTypeVoid",
        "%fn_t = OpTypeFunction %void",
        "%bool = OpTypeBool",
        "%u32 = OpTypeInt 32 0",
        "%s32 = OpTypeInt 32 1",
        "%f32 = OpTypeFloat 32",
        "%v2f32 = OpTypeVector %f32 2",
        "%v4f32 = OpTypeVector %f32 4",
        "%v2s32 = OpTypeVector %s32 2",
        "%v3u32 = OpTypeVector %u32 3",
        "%v4u32 = OpTypeVector %u32 4",
        "%st_f32_s32 = OpTypeStruct %f32 %s32",
    ]
    if "Int64" in caps:
        type_block.extend([
            "%u64 = OpTypeInt 64 0",
            "%s64 = OpTypeInt 64 1",
            "%c64_0 = OpConstant %u64 0",
            "%c64_1 = OpConstant %u64 1",
        ])
    if "Int16" in caps:
        type_block.extend([
            "%u16 = OpTypeInt 16 0",
            "%s16 = OpTypeInt 16 1",
            "%v2u16 = OpTypeVector %u16 2",
            "%v2s16 = OpTypeVector %s16 2",
        ])
    if "Int8" in caps:
        type_block.extend([
            "%u8 = OpTypeInt 8 0",
            "%s8 = OpTypeInt 8 1",
            "%v2u8 = OpTypeVector %u8 2",
            "%v2s8 = OpTypeVector %s8 2",
            "%v4u8 = OpTypeVector %u8 4",
            "%v4s8 = OpTypeVector %s8 4",
        ])
    if "Float16" in caps:
        type_block.extend(["%f16 = OpTypeFloat 16", "%v2f16 = OpTypeVector %f16 2"])
    type_block.extend([
        "%c0 = OpConstant %u32 0",
        "%c1 = OpConstant %u32 1",
        "%c2 = OpConstant %u32 2",
        "%c3 = OpConstant %u32 3",
        "%c4 = OpConstant %u32 4",
        "%c5 = OpConstant %u32 5",
        "%c15 = OpConstant %u32 15",
        "%c16 = OpConstant %u32 16",
        "%c27 = OpConstant %u32 27",
        "%c31 = OpConstant %u32 31",
        "%c32 = OpConstant %u32 32",
        "%c264 = OpConstant %u32 264",
        "%c_big = OpConstant %u32 305419896",
        "%c_mask16 = OpConstant %u32 65535",
        "%cf0 = OpConstant %f32 0.0",
        "%cf1 = OpConstant %f32 1.0",
        "%RArr = OpTypeRuntimeArray %u32",
        "%Buf = OpTypeStruct %RArr",
        "%pBuf = OpTypePointer StorageBuffer %Buf",
        "%pU32 = OpTypePointer StorageBuffer %u32",
        "%pInV3 = OpTypePointer Input %v3u32",
        "%pInU32 = OpTypePointer Input %u32",
    ] + extra_types)

    fn_locals = "%ro_var = OpVariable %pFnRoArr Function %ro_init\n" if recipe.needs_rodata else ""

    if complexity == "medium":
        pre = (
            "%a = OpBitwiseXor %u32 %raw_a %gx\n"
            "%b = OpIAdd %u32 %raw_b %c1\n"
            "%c = OpBitwiseOr %u32 %raw_c %c2\n"
            "%d = OpISub %u32 %raw_d %gx\n"
        )
        post = (
            "%mix1 = OpBitwiseXor %u32 %res %raw_d\n"
            "%final_res = OpIAdd %u32 %mix1 %gx\n"
        )
        core = f"{recipe.body}\n{post}"
    elif complexity == "branchy" and "OpSelectionMerge" not in recipe.body and not recipe.needs_wg:
        pre = (
            "%a = OpCopyObject %u32 %raw_a\n"
            "%b = OpCopyObject %u32 %raw_b\n"
            "%c = OpCopyObject %u32 %raw_c\n"
            "%d = OpCopyObject %u32 %raw_d\n"
        )
        core = (
            "%br_k = OpULessThan %bool %raw_a %raw_d\n"
            "OpSelectionMerge %br_merge None\n"
            "OpBranchConditional %br_k %br_then %br_else\n"
            "%br_then = OpLabel\n"
            f"{recipe.body}\n"
            "OpBranch %br_merge\n"
            "%br_else = OpLabel\n"
            "%alt_res = OpBitwiseXor %u32 %raw_b %raw_c\n"
            "OpBranch %br_merge\n"
            "%br_merge = OpLabel\n"
            "%final_res = OpPhi %u32 %res %br_then %alt_res %br_else\n"
        )
    else:
        pre = (
            "%a = OpCopyObject %u32 %raw_a\n"
            "%b = OpCopyObject %u32 %raw_b\n"
            "%c = OpCopyObject %u32 %raw_c\n"
            "%d = OpCopyObject %u32 %raw_d\n"
        )
        core = f"{recipe.body}\n%final_res = OpCopyObject %u32 %res\n"

    return f"""{cap_lines}
{ext_lines}
%ext = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %gl_gid
OpExecutionMode %main LocalSize 32 1 1
{chr(10).join(decorations)}
{chr(10).join(type_block)}
%gl_gid = OpVariable %pInV3 Input
%inbuf = OpVariable %pBuf StorageBuffer
%outbuf = OpVariable %pBuf StorageBuffer
{chr(10).join(extra_vars)}
%main = OpFunction %void None %fn_t
%entry = OpLabel
{fn_locals}%pgx = OpAccessChain %pInU32 %gl_gid %c0
%gx_raw = OpLoad %u32 %pgx
%gx = OpBitwiseAnd %u32 %gx_raw %c_mask16
%base = OpShiftLeftLogical %u32 %gx %c2
%i1 = OpIAdd %u32 %base %c1
%i2 = OpIAdd %u32 %base %c2
%i3 = OpIAdd %u32 %base %c3
%pa = OpAccessChain %pU32 %inbuf %c0 %base
%pb = OpAccessChain %pU32 %inbuf %c0 %i1
%pc = OpAccessChain %pU32 %inbuf %c0 %i2
%pd = OpAccessChain %pU32 %inbuf %c0 %i3
%raw_a = OpLoad %u32 %pa
%raw_b = OpLoad %u32 %pb
%raw_c = OpLoad %u32 %pc
%raw_d = OpLoad %u32 %pd
{pre}%fa = OpBitcast %f32 %a
%fb = OpBitcast %f32 %b
%fc = OpBitcast %f32 %c
%sa = OpBitcast %s32 %a
%sb = OpBitcast %s32 %b
%sc = OpBitcast %s32 %c
{core}%po = OpAccessChain %pU32 %outbuf %c0 %gx
OpStore %po %final_res
OpReturn
OpFunctionEnd
"""


def build_fragment_spvasm(recipe: IsaRecipe, complexity: str = "minimal") -> str:
    """Wrap a fragment `IsaRecipe` in a Vulkan 1.3 Fragment SPIR-V harness."""
    ep_vars = ["%in_v0", "%in_v1", "%in_flat", "%in_fc", "%out_col"]
    exec_modes = ["OpExecutionMode %main OriginUpperLeft"]
    decs = [
        "OpDecorate %in_v0 Location 0",
        "OpDecorate %in_v1 Location 1",
        "OpDecorate %in_flat Location 2",
        "OpDecorate %in_flat Flat",
        "OpDecorate %in_fc BuiltIn FragCoord",
        "OpDecorate %out_col Location 0",
    ]
    extra_types = []
    extra_vars = []
    if recipe.needs_frag_depth:
        ep_vars.append("%out_depth")
        exec_modes.append("OpExecutionMode %main DepthReplacing")
        decs.append("OpDecorate %out_depth BuiltIn FragDepth")
        extra_types.append("%pOutF32 = OpTypePointer Output %f32")
        extra_vars.append("%out_depth = OpVariable %pOutF32 Output")
    if recipe.needs_tex2d:
        decs.extend(["OpDecorate %tex2d DescriptorSet 0", "OpDecorate %tex2d Binding 0"])
        extra_types.extend([
            "%tImg2D = OpTypeImage %f32 2D 0 0 0 1 Unknown",
            "%tSampledImg2D = OpTypeSampledImage %tImg2D",
            "%pSampledImg2D = OpTypePointer UniformConstant %tSampledImg2D",
        ])
        extra_vars.append("%tex2d = OpVariable %pSampledImg2D UniformConstant")
    if recipe.needs_cube:
        decs.extend(["OpDecorate %texcube DescriptorSet 0", "OpDecorate %texcube Binding 1"])
        extra_types.extend([
            "%tImgCube = OpTypeImage %f32 Cube 0 0 0 1 Unknown",
            "%tSampledCube = OpTypeSampledImage %tImgCube",
            "%pSampledCube = OpTypePointer UniformConstant %tSampledCube",
        ])
        extra_vars.append("%texcube = OpVariable %pSampledCube UniformConstant")
    if recipe.needs_dyn_var_array or recipe.needs_dyn_var_flat_array:
        ep_vars.extend(["%in_varr", "%in_vidx"])
        decs.extend([
            "OpDecorate %in_varr Location 4",
            "OpDecorate %in_vidx Location 8",
            "OpDecorate %in_vidx Flat",
        ])
        if recipe.needs_dyn_var_flat_array:
            decs.append("OpDecorate %in_varr Flat")
        extra_types.extend([
            "%ArrV4 = OpTypeArray %v4f32 %c4",
            "%pInArrV4 = OpTypePointer Input %ArrV4",
            "%pInU32 = OpTypePointer Input %u32",
        ])
        extra_vars.extend([
            "%in_varr = OpVariable %pInArrV4 Input",
            "%in_vidx = OpVariable %pInU32 Input",
        ])

    post = "%final_res = OpFMul %v4f32 %res %v1\n" if complexity == "medium" else "%final_res = OpCopyObject %v4f32 %res\n"
    return f"""OpCapability Shader
%ext = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" {' '.join(ep_vars)}
{chr(10).join(exec_modes)}
{chr(10).join(decs)}
%void = OpTypeVoid
%fn_t = OpTypeFunction %void
%bool = OpTypeBool
%u32 = OpTypeInt 32 0
%s32 = OpTypeInt 32 1
%f32 = OpTypeFloat 32
%v2f32 = OpTypeVector %f32 2
%v3f32 = OpTypeVector %f32 3
%v4f32 = OpTypeVector %f32 4
%c0 = OpConstant %u32 0
%c3 = OpConstant %u32 3
%c4 = OpConstant %u32 4
%cs0 = OpConstant %s32 0
%cf0 = OpConstant %f32 0.0
%pInV4 = OpTypePointer Input %v4f32
%pOutV4 = OpTypePointer Output %v4f32
{chr(10).join(extra_types)}
%in_v0 = OpVariable %pInV4 Input
%in_v1 = OpVariable %pInV4 Input
%in_flat = OpVariable %pInV4 Input
%in_fc = OpVariable %pInV4 Input
%out_col = OpVariable %pOutV4 Output
{chr(10).join(extra_vars)}
%main = OpFunction %void None %fn_t
%entry = OpLabel
%v0 = OpLoad %v4f32 %in_v0
%v1 = OpLoad %v4f32 %in_v1
%vflat = OpLoad %v4f32 %in_flat
%fcoord = OpLoad %v4f32 %in_fc
{recipe.body}
{post}OpStore %out_col %final_res
OpReturn
OpFunctionEnd
"""


def build_vertex_spvasm(recipe: IsaRecipe, complexity: str = "minimal") -> str:
    """Wrap a vertex `IsaRecipe` in a Vulkan 1.3 Vertex SPIR-V harness."""
    ep_vars = ["%in_a0", "%in_a1", "%out_pos", "%out_v0"]
    decs = [
        "OpDecorate %in_a0 Location 0",
        "OpDecorate %in_a1 Location 1",
        "OpDecorate %out_pos BuiltIn Position",
        "OpDecorate %out_v0 Location 0",
    ]
    extra_types = []
    extra_vars = []
    if recipe.needs_dyn_attr_in_array:
        ep_vars.extend(["%in_arr", "%in_idx"])
        decs.extend(["OpDecorate %in_arr Location 2", "OpDecorate %in_idx Location 6"])
        extra_types.extend([
            "%ArrV4 = OpTypeArray %v4f32 %c4",
            "%pInArrV4 = OpTypePointer Input %ArrV4",
            "%pInU32 = OpTypePointer Input %u32",
        ])
        extra_vars.extend([
            "%in_arr = OpVariable %pInArrV4 Input",
            "%in_idx = OpVariable %pInU32 Input",
        ])
    if recipe.needs_dyn_attr_out_array:
        ep_vars.extend(["%in_oidx", "%out_arr"])
        decs.extend(["OpDecorate %in_oidx Location 7", "OpDecorate %out_arr Location 2"])
        extra_types.extend([
            "%OutArrV4 = OpTypeArray %v4f32 %c4",
            "%pOutArrV4 = OpTypePointer Output %OutArrV4",
            "%pInU32O = OpTypePointer Input %u32",
        ])
        extra_vars.extend([
            "%in_oidx = OpVariable %pInU32O Input",
            "%out_arr = OpVariable %pOutArrV4 Output",
        ])

    post = "%final_res = OpFMul %v4f32 %res %a1\n" if complexity == "medium" else "%final_res = OpCopyObject %v4f32 %res\n"
    return f"""OpCapability Shader
%ext = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" {' '.join(ep_vars)}
{chr(10).join(decs)}
%void = OpTypeVoid
%fn_t = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%f32 = OpTypeFloat 32
%v4f32 = OpTypeVector %f32 4
%c3 = OpConstant %u32 3
%c4 = OpConstant %u32 4
%pInV4 = OpTypePointer Input %v4f32
%pOutV4 = OpTypePointer Output %v4f32
{chr(10).join(extra_types)}
%in_a0 = OpVariable %pInV4 Input
%in_a1 = OpVariable %pInV4 Input
%out_pos = OpVariable %pOutV4 Output
%out_v0 = OpVariable %pOutV4 Output
{chr(10).join(extra_vars)}
%main = OpFunction %void None %fn_t
%entry = OpLabel
%a0 = OpLoad %v4f32 %in_a0
%a1 = OpLoad %v4f32 %in_a1
{recipe.body}
{post}OpStore %out_pos %final_res
OpStore %out_v0 %final_res
OpReturn
OpFunctionEnd
"""


def build_spvasm_from_recipe(recipe: IsaRecipe, complexity: str = "minimal") -> str:
    """Dispatch `recipe` to the matching stage SPIR-V harness builder."""
    if recipe.stage == "compute":
        return build_compute_spvasm(recipe, complexity)
    if recipe.stage == "fragment":
        return build_fragment_spvasm(recipe, complexity)
    if recipe.stage == "vertex":
        return build_vertex_spvasm(recipe, complexity)
    raise ValueError(f"Unknown stage '{recipe.stage}'")

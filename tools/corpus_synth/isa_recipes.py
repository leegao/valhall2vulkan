"""
Declarative SPIR-V Synthesis Recipes for Valhall ISA Mnemonics & Variants.

Contains:
  - `RECIPES`: 128 empirically verified Vulkan 1.3 SPIR-V recipes guaranteed to
    emit the target Valhall mnemonic in `EBIN[0]` (main shader) via `libMali-Gxx`.
  - `VARIANT_RECIPES`: Targeted SPIR-V recipes inducing specific non-default
    `(mnemonic, variant)` pairs (e.g., `IADD.v2u16`, `IADD.v4u8`, `FADD.v2f16`,
    `FMUL.v2f16`, `FMA.v2f16`, `FMIN.v2f16`, `FMAX.v2f16`, `IABS.v2s16`,
    `IDPADD.v4u8`).

Remaining mnemonics (internal/driver-only ops and ops that `libMali-Gxx`
canonicalizes or lowers differently for Vulkan 1.3) are covered deterministically
by `valhall_raw_synth.py` (201/201 mnemonics, 627/627 variants).
"""

from __future__ import annotations

from tools.corpus_synth.isa_spv_harness import IsaRecipe


def _cr(
    body: str,
    *,
    caps: tuple[str, ...] = (),
    exts: tuple[str, ...] = (),
    needs_wg: bool = False,
    needs_tex2d: bool = False,
    needs_storage_img: bool = False,
    needs_storage_img_array: bool = False,
    needs_ssbo_array: bool = False,
    needs_rodata: bool = False,
) -> IsaRecipe:
    return IsaRecipe(
        stage="compute",
        body=body,
        caps=caps,
        exts=exts,
        needs_wg=needs_wg,
        needs_tex2d=needs_tex2d,
        needs_storage_img=needs_storage_img,
        needs_storage_img_array=needs_storage_img_array,
        needs_ssbo_array=needs_ssbo_array,
        needs_rodata=needs_rodata,
    )


def _fr(
    body: str,
    *,
    needs_tex2d: bool = False,
    needs_cube: bool = False,
    needs_frag_depth: bool = False,
    needs_dyn_var_array: bool = False,
    needs_dyn_var_flat_array: bool = False,
) -> IsaRecipe:
    return IsaRecipe(
        stage="fragment",
        body=body,
        needs_tex2d=needs_tex2d,
        needs_cube=needs_cube,
        needs_frag_depth=needs_frag_depth,
        needs_dyn_var_array=needs_dyn_var_array,
        needs_dyn_var_flat_array=needs_dyn_var_flat_array,
    )


def _vr(
    body: str,
    *,
    needs_dyn_attr_in_array: bool = False,
    needs_dyn_attr_out_array: bool = False,
) -> IsaRecipe:
    return IsaRecipe(
        stage="vertex",
        body=body,
        needs_dyn_attr_in_array=needs_dyn_attr_in_array,
        needs_dyn_attr_out_array=needs_dyn_attr_out_array,
    )


# 128 empirically verified Vulkan 1.3 SPIR-V recipes emitting the target mnemonic in EBIN[0].
RECIPES: dict[str, IsaRecipe] = {
    # --- Integer Arithmetic ---
    "IADD": _cr("%res = OpIAdd %u32 %a %b"),
    "IADD_IMM": _cr("%res = OpIAdd %u32 %a %c_big"),
    "ISUB": _cr("%res = OpISub %u32 %a %b"),
    "IMUL": _cr("%res = OpIMul %u32 %a %b"),
    "IABS": _cr(
        "%v = OpBitcast %v4s8 %a\n%ab = OpExtInst %v4s8 %ext SAbs %v\n%res = OpBitcast %u32 %ab",
        caps=("Int8",),
    ),
    "HADD": _cr(
        "%ba = OpBitwiseAnd %u32 %a %b\n"
        "%bx = OpBitwiseXor %u32 %a %b\n"
        "%hs = OpShiftRightLogical %u32 %bx %c1\n"
        "%res = OpIAdd %u32 %ba %hs"
    ),
    "IDPADD": _cr(
        "%rs = OpSDotAccSat %s32 %a %b %sc PackedVectorFormat4x8Bit\n"
        "%res = OpBitcast %u32 %rs",
        caps=("DotProduct", "DotProductInput4x8BitPacked"),
        exts=("SPV_KHR_integer_dot_product",),
    ),
    "SHADD": _cr("%res = OpIAdd %u32 %a %b"),  # emitted by inbuf/outbuf SSBO indexing

    # --- Bitwise, Shifts, Rotates, Bitfields ---
    "AND": _cr("%res = OpBitwiseAnd %u32 %a %b"),
    "OR": _cr("%res = OpBitwiseOr %u32 %a %b"),
    "XOR": _cr("%res = OpBitwiseXor %u32 %a %b"),
    "NOT": _cr("%res = OpNot %u32 %a"),
    "LSHIFT": _cr("%res = OpShiftLeftLogical %u32 %a %b"),
    "RSHIFT": _cr("%res = OpShiftRightLogical %u32 %a %b"),
    "ARSHIFT": _cr("%t = OpShiftRightArithmetic %s32 %sa %b\n%res = OpBitcast %u32 %t"),
    "LSHIFT_AND": _cr("%t = OpShiftLeftLogical %u32 %a %b\n%res = OpBitwiseAnd %u32 %t %c"),
    "LSHIFT_OR": _cr("%t = OpShiftLeftLogical %u32 %a %b\n%res = OpBitwiseOr %u32 %t %c"),
    "LSHIFT_XOR": _cr("%t = OpShiftLeftLogical %u32 %a %b\n%res = OpBitwiseXor %u32 %t %c"),
    "RSHIFT_AND": _cr("%t = OpShiftRightLogical %u32 %a %b\n%res = OpBitwiseAnd %u32 %t %c"),
    "RSHIFT_OR": _cr("%t = OpShiftRightLogical %u32 %a %b\n%res = OpBitwiseOr %u32 %t %c"),
    "RSHIFT_XOR": _cr("%t = OpShiftRightLogical %u32 %a %b\n%res = OpBitwiseXor %u32 %t %c"),
    "LROT": _cr(
        "%t1 = OpShiftLeftLogical %u32 %a %c5\n"
        "%t2 = OpShiftRightLogical %u32 %a %c27\n"
        "%res = OpBitwiseOr %u32 %t1 %t2"
    ),
    "RROT": _cr(
        "%nb = OpISub %u32 %c32 %b\n"
        "%t1 = OpShiftRightLogical %u32 %a %b\n"
        "%t2 = OpShiftLeftLogical %u32 %a %nb\n"
        "%res = OpBitwiseOr %u32 %t1 %t2"
    ),
    "CLZ": _cr("%m = OpExtInst %u32 %ext FindUMsb %a\n%res = OpIAdd %u32 %m %b"),
    "BITREV": _cr("%res = OpBitReverse %u32 %a"),
    "POPCOUNT": _cr("%res = OpBitCount %u32 %a"),
    "MUX": _cr("%at = OpExtInst %f32 %ext Atan2 %fa %fb\n%res = OpBitcast %u32 %at"),

    # --- Moves, Packing, Swizzles, Rodata ---
    "MOV": _cr(
        "%k = OpULessThan %bool %a %b\n"
        "OpSelectionMerge %b_m None\n"
        "OpBranchConditional %k %b_t %b_f\n"
        "%b_t = OpLabel\n"
        "%po2 = OpAccessChain %pU32 %outbuf %c0 %i1\n"
        "OpStore %po2 %c\n"
        "OpBranch %b_m\n"
        "%b_f = OpLabel\n"
        "OpBranch %b_m\n"
        "%b_m = OpLabel\n"
        "%res = OpPhi %u32 %c %b_t %d %b_f"
    ),
    "MOV_IMM": _cr(
        "%k = OpULessThan %bool %a %b\n"
        "OpSelectionMerge %b_m None\n"
        "OpBranchConditional %k %b_t %b_f\n"
        "%b_t = OpLabel\n"
        "%po2 = OpAccessChain %pU32 %outbuf %c0 %i1\n"
        "OpStore %po2 %c_big\n"
        "OpBranch %b_m\n"
        "%b_f = OpLabel\n"
        "OpBranch %b_m\n"
        "%b_m = OpLabel\n"
        "%res = OpPhi %u32 %c_big %b_t %d %b_f"
    ),
    "MKVEC": _cr(
        "%va = OpBitcast %v4u8 %a\n"
        "%vb = OpBitcast %v4u8 %b\n"
        "%sh = OpVectorShuffle %v4u8 %va %vb 0 5 2 7\n"
        "%res = OpBitcast %u32 %sh",
        caps=("Int8",),
    ),
    "SWZ": _cr(
        "%va = OpBitcast %v2u16 %a\n"
        "%sh = OpVectorShuffle %v2u16 %va %va 1 0\n"
        "%res = OpBitcast %u32 %sh",
        caps=("Int16",),
    ),
    "ADR": _cr(
        "%ridx = OpBitwiseAnd %u32 %a %c15\n"
        "%rptr = OpAccessChain %pFnU32 %ro_var %ridx\n"
        "%res = OpLoad %u32 %rptr",
        needs_rodata=True,
    ),

    # --- Comparisons & Conditional Select ---
    "ICMP": _cr("%k = OpULessThan %bool %a %b\n%res = OpSelect %u32 %k %c1 %c0"),
    "ICMP_AND": _cr(
        "%k1 = OpULessThan %bool %a %b\n"
        "%k2 = OpIEqual %bool %b %c\n"
        "%ka = OpLogicalAnd %bool %k1 %k2\n"
        "%res = OpSelect %u32 %ka %a %d"
    ),
    "ICMP_OR": _cr(
        "%k1 = OpULessThan %bool %a %b\n"
        "%k2 = OpIEqual %bool %b %c\n"
        "%ko = OpLogicalOr %bool %k1 %k2\n"
        "%res = OpSelect %u32 %ko %a %d"
    ),
    "ICMP_MULTI": _vr(
        "%oidx0 = OpLoad %u32 %in_oidx\n"
        "%oidx = OpBitwiseAnd %u32 %oidx0 %c3\n"
        "%optr = OpAccessChain %pOutV4 %out_arr %oidx\n"
        "OpStore %optr %a0\n"
        "%res = OpFAdd %v4f32 %a0 %a1",
        needs_dyn_attr_out_array=True,
    ),
    "FCMP": _cr("%k = OpFOrdLessThan %bool %fa %fb\n%res = OpSelect %u32 %k %c1 %c0"),
    "FCMP_AND": _cr(
        "%k1 = OpFOrdLessThan %bool %fa %fb\n"
        "%k2 = OpFOrdEqual %bool %fb %fc\n"
        "%ka = OpLogicalAnd %bool %k1 %k2\n"
        "%res = OpSelect %u32 %ka %c1 %c0"
    ),
    "FCMP_OR": _cr(
        "%k1 = OpFOrdLessThan %bool %fa %fb\n"
        "%k2 = OpFOrdGreaterThan %bool %fb %fc\n"
        "%ko = OpLogicalOr %bool %k1 %k2\n"
        "%res = OpSelect %u32 %ko %a %d"
    ),
    "CSEL": _cr("%k = OpULessThan %bool %a %b\n%res = OpSelect %u32 %k %c %d"),

    # --- Floating-Point Core & Rounding ---
    "FADD": _cr("%t = OpFAdd %f32 %fa %fb\n%res = OpBitcast %u32 %t"),
    "FADD_IMM": _cr("%s = OpExtInst %f32 %ext Sin %fa\n%res = OpBitcast %u32 %s"),
    "FMUL": _cr("%t = OpFMul %f32 %fa %fb\n%res = OpBitcast %u32 %t"),
    "FMA": _cr("%t = OpExtInst %f32 %ext Fma %fa %fb %fc\n%res = OpBitcast %u32 %t"),
    "FMIN": _cr("%t = OpExtInst %f32 %ext FMin %fa %fb\n%res = OpBitcast %u32 %t"),
    "FMAX": _cr("%t = OpExtInst %f32 %ext FMax %fa %fb\n%res = OpBitcast %u32 %t"),
    "FABS": _cr("%t = OpExtInst %f32 %ext FAbs %fa\n%res = OpBitcast %u32 %t"),
    "FNEG": _cr("%t = OpFNegate %f32 %fa\n%res = OpBitcast %u32 %t"),
    "FROUND": _cr("%t = OpExtInst %f32 %ext Floor %fa\n%res = OpBitcast %u32 %t"),

    # --- Floating-Point SFU & Transcendentals ---
    "FRCP": _cr("%t = OpFDiv %f32 %cf1 %fa\n%res = OpBitcast %u32 %t"),
    "FRSQ": _cr("%t = OpExtInst %f32 %ext InverseSqrt %fa\n%res = OpBitcast %u32 %t"),
    "FREXPE": _cr(
        "%st = OpExtInst %st_f32_s32 %ext FrexpStruct %fa\n"
        "%e = OpCompositeExtract %s32 %st 1\n"
        "%res = OpBitcast %u32 %e"
    ),
    "FREXPM": _cr(
        "%st = OpExtInst %st_f32_s32 %ext FrexpStruct %fa\n"
        "%m = OpCompositeExtract %f32 %st 0\n"
        "%res = OpBitcast %u32 %m"
    ),
    "LDEXP": _cr("%t = OpExtInst %f32 %ext Ldexp %fa %sb\n%res = OpBitcast %u32 %t"),
    "FLOGD": _cr("%t = OpExtInst %f32 %ext Log2 %fa\n%res = OpBitcast %u32 %t"),
    "FADD_LSCALE": _cr("%t = OpExtInst %f32 %ext Log2 %fa\n%res = OpBitcast %u32 %t"),
    "FMUL_RSCALE": _cr("%t = OpExtInst %f32 %ext Pow %fa %fb\n%res = OpBitcast %u32 %t"),
    "FEXP": _cr("%t = OpExtInst %f32 %ext Exp2 %fa\n%res = OpBitcast %u32 %t"),
    "FSIN_TABLE": _cr("%t = OpExtInst %f32 %ext Sin %fa\n%res = OpBitcast %u32 %t"),
    "FCOS_TABLE": _cr("%t = OpExtInst %f32 %ext Cos %fa\n%res = OpBitcast %u32 %t"),
    "FATAN_ASSIST": _cr("%t = OpExtInst %f32 %ext Atan2 %fa %fb\n%res = OpBitcast %u32 %t"),
    "FATAN_TABLE": _cr("%t = OpExtInst %f32 %ext Atan2 %fa %fb\n%res = OpBitcast %u32 %t"),

    # --- Conversions ---
    "F32_TO_S32": _cr("%t = OpConvertFToS %s32 %fa\n%res = OpBitcast %u32 %t"),
    "F32_TO_U32": _cr("%res = OpConvertFToU %u32 %fa"),
    "S32_TO_F32": _cr("%t = OpConvertSToF %f32 %sa\n%res = OpBitcast %u32 %t"),
    "U32_TO_F32": _cr("%t = OpConvertUToF %f32 %a\n%res = OpBitcast %u32 %t"),
    "F16_TO_F32": _cr(
        "%u = OpExtInst %v2f32 %ext UnpackHalf2x16 %a\n"
        "%u0 = OpCompositeExtract %f32 %u 0\n"
        "%res = OpBitcast %u32 %u0"
    ),
    "F16_TO_S32": _cr(
        "%hv = OpBitcast %v2f16 %a\n%h = OpCompositeExtract %f16 %hv 0\n"
        "%s = OpConvertFToS %s32 %h\n%res = OpBitcast %u32 %s",
        caps=("Float16",),
    ),
    "F16_TO_U32": _cr(
        "%hv = OpBitcast %v2f16 %a\n%h = OpCompositeExtract %f16 %hv 0\n"
        "%res = OpConvertFToU %u32 %h",
        caps=("Float16",),
    ),
    "V2F32_TO_V2F16": _cr(
        "%v2 = OpCompositeConstruct %v2f32 %fa %fb\n"
        "%res = OpExtInst %u32 %ext PackHalf2x16 %v2"
    ),
    "V2F16_TO_V2S16": _cr(
        "%h2 = OpBitcast %v2f16 %a\n%s2 = OpConvertFToS %v2s16 %h2\n%res = OpBitcast %u32 %s2",
        caps=("Float16", "Int16"),
    ),
    "V2F16_TO_V2U16": _cr(
        "%h2 = OpBitcast %v2f16 %a\n%u2 = OpConvertFToU %v2u16 %h2\n%res = OpBitcast %u32 %u2",
        caps=("Float16", "Int16"),
    ),
    "V2S16_TO_V2F16": _cr(
        "%s2 = OpBitcast %v2s16 %a\n%h2 = OpConvertSToF %v2f16 %s2\n%res = OpBitcast %u32 %h2",
        caps=("Float16", "Int16"),
    ),
    "V2U16_TO_V2F16": _cr(
        "%u2 = OpBitcast %v2u16 %a\n%h2 = OpConvertUToF %v2f16 %u2\n%res = OpBitcast %u32 %h2",
        caps=("Float16", "Int16"),
    ),
    "S8_TO_F32": _cr(
        "%v4 = OpBitcast %v4s8 %a\n%e0 = OpCompositeExtract %s8 %v4 0\n"
        "%f0 = OpConvertSToF %f32 %e0\n%res = OpBitcast %u32 %f0",
        caps=("Int8",),
    ),
    "U8_TO_F32": _cr(
        "%v4 = OpBitcast %v4u8 %a\n%e0 = OpCompositeExtract %u8 %v4 0\n"
        "%f0 = OpConvertUToF %f32 %e0\n%res = OpBitcast %u32 %f0",
        caps=("Int8",),
    ),
    "S16_TO_F32": _cr(
        "%v2 = OpBitcast %v2s16 %a\n%e0 = OpCompositeExtract %s16 %v2 0\n"
        "%f0 = OpConvertSToF %f32 %e0\n%res = OpBitcast %u32 %f0",
        caps=("Int16",),
    ),
    "U16_TO_F32": _cr(
        "%v2 = OpBitcast %v2u16 %a\n%e0 = OpCompositeExtract %u16 %v2 0\n"
        "%f0 = OpConvertUToF %f32 %e0\n%res = OpBitcast %u32 %f0",
        caps=("Int16",),
    ),
    "S8_TO_S32": _cr(
        "%v4 = OpBitcast %v4s8 %a\n%e0 = OpCompositeExtract %s8 %v4 0\n"
        "%s0 = OpSConvert %s32 %e0\n%res = OpBitcast %u32 %s0",
        caps=("Int8",),
    ),
    "U8_TO_U32": _cr(
        "%v4 = OpBitcast %v4u8 %a\n%e0 = OpCompositeExtract %u8 %v4 0\n"
        "%res = OpUConvert %u32 %e0",
        caps=("Int8",),
    ),
    "S16_TO_S32": _cr(
        "%v2 = OpBitcast %v2s16 %a\n%e0 = OpCompositeExtract %s16 %v2 0\n"
        "%s0 = OpSConvert %s32 %e0\n%res = OpBitcast %u32 %s0",
        caps=("Int16",),
    ),
    "U16_TO_U32": _cr(
        "%v2 = OpBitcast %v2u16 %a\n%e0 = OpCompositeExtract %u16 %v2 0\n"
        "%res = OpUConvert %u32 %e0",
        caps=("Int16",),
    ),
    "V2S8_TO_V2F16": _cr(
        "%v4 = OpBitcast %v4s8 %a\n%v2 = OpVectorShuffle %v2s8 %v4 %v4 0 1\n"
        "%h2 = OpConvertSToF %v2f16 %v2\n%res = OpBitcast %u32 %h2",
        caps=("Int8", "Float16"),
    ),
    "V2U8_TO_V2F16": _cr(
        "%v4 = OpBitcast %v4u8 %a\n%v2 = OpVectorShuffle %v2u8 %v4 %v4 0 1\n"
        "%h2 = OpConvertUToF %v2f16 %v2\n%res = OpBitcast %u32 %h2",
        caps=("Int8", "Float16"),
    ),
    "V2S8_TO_V2S16": _cr(
        "%v4 = OpBitcast %v4s8 %a\n%v2 = OpVectorShuffle %v2s8 %v4 %v4 0 1\n"
        "%s2 = OpSConvert %v2s16 %v2\n%res = OpBitcast %u32 %s2",
        caps=("Int8", "Int16"),
    ),
    "V2U8_TO_V2U16": _cr(
        "%v4 = OpBitcast %v4u8 %a\n%v2 = OpVectorShuffle %v2u8 %v4 %v4 0 1\n"
        "%u2 = OpUConvert %v2u16 %v2\n%res = OpBitcast %u32 %u2",
        caps=("Int8", "Int16"),
    ),

    # --- Memory, Atomics, Subgroups, Sync, Control Flow ---
    "LOAD": _cr("%res = OpIAdd %u32 %a %b"),
    "STORE": _cr("%res = OpIAdd %u32 %a %b"),
    "LD_PKA": _cr(
        "%bidx = OpBitwiseAnd %u32 %gx %c3\n"
        "%pka_p = OpAccessChain %pU32 %ssbo_arr %bidx %c0 %gx\n"
        "%res = OpLoad %u32 %pka_p",
        needs_ssbo_array=True,
    ),
    "LEA_BUF_IMM": _vr(
        "%idx0 = OpLoad %u32 %in_idx\n"
        "%idx = OpBitwiseAnd %u32 %idx0 %c3\n"
        "%ptr = OpAccessChain %pInV4 %in_arr %idx\n"
        "%ld = OpLoad %v4f32 %ptr\n"
        "%res = OpFAdd %v4f32 %ld %a0",
        needs_dyn_attr_in_array=True,
    ),
    "ATOM": _cr(
        "%po2 = OpAccessChain %pU32 %outbuf %c0 %i1\n"
        "%ign = OpAtomicIAdd %u32 %po2 %c1 %c0 %a\n"
        "%res = OpIAdd %u32 %a %b"
    ),
    "ATOM_RETURN": _cr(
        "%po2 = OpAccessChain %pU32 %outbuf %c0 %i1\n"
        "%res = OpAtomicIAdd %u32 %po2 %c1 %c0 %a"
    ),
    "ATOM1": _cr(
        "%po2 = OpAccessChain %pU32 %outbuf %c0 %i1\n"
        "%ign = OpAtomicIIncrement %u32 %po2 %c1 %c0\n"
        "%res = OpIAdd %u32 %a %b"
    ),
    "ATOM1_RETURN": _cr(
        "%po2 = OpAccessChain %pU32 %outbuf %c0 %gx\n"
        "%res = OpAtomicIIncrement %u32 %po2 %c1 %c0"
    ),
    "AXCHG": _cr(
        "%po2 = OpAccessChain %pU32 %outbuf %c0 %i1\n"
        "%res = OpAtomicExchange %u32 %po2 %c1 %c0 %a"
    ),
    "ACMPXCHG": _cr(
        "%po2 = OpAccessChain %pU32 %outbuf %c0 %i1\n"
        "%res = OpAtomicCompareExchange %u32 %po2 %c1 %c0 %c0 %b %a"
    ),
    "CLPER": _cr(
        "%res = OpGroupNonUniformShuffleXor %u32 %c3 %a %b",
        caps=("GroupNonUniform", "GroupNonUniformShuffle"),
    ),
    "WMASK": _cr(
        "%k = OpIEqual %bool %a %b\n"
        "%bal = OpGroupNonUniformBallot %v4u32 %c3 %k\n"
        "%res = OpCompositeExtract %u32 %bal 0",
        caps=("GroupNonUniform", "GroupNonUniformBallot"),
    ),
    "BARRIER": _cr(
        "%widx = OpBitwiseAnd %u32 %gx %c31\n"
        "%pw1 = OpAccessChain %pWgU32 %wg %widx\n"
        "OpStore %pw1 %a\n"
        "OpControlBarrier %c2 %c2 %c264\n"
        "%widx2 = OpBitwiseXor %u32 %widx %c16\n"
        "%pw2 = OpAccessChain %pWgU32 %wg %widx2\n"
        "%res = OpLoad %u32 %pw2",
        needs_wg=True,
    ),
    "BRANCH": _cr(
        "%k = OpULessThan %bool %a %b\n"
        "OpSelectionMerge %b_m None\n"
        "OpBranchConditional %k %b_t %b_m\n"
        "%b_t = OpLabel\n"
        "%po2 = OpAccessChain %pU32 %outbuf %c0 %i1\n"
        "OpStore %po2 %c\n"
        "OpBranch %b_m\n"
        "%b_m = OpLabel\n"
        "%res = OpIAdd %u32 %a %b"
    ),
    "NOP": _fr(
        "%x0 = OpCompositeExtract %f32 %v0 0\n"
        "%k = OpFOrdLessThan %bool %x0 %cf0\n"
        "OpSelectionMerge %dm None\n"
        "OpBranchConditional %k %dk %dm\n"
        "%dk = OpLabel\n"
        "OpKill\n"
        "%dm = OpLabel\n"
        "%res = OpFAdd %v4f32 %v0 %v1"
    ),

    # --- Textures & Storage Images (Compute) ---
    "TEX_SINGLE": _cr(
        "%uv = OpCompositeConstruct %v2f32 %fa %fb\n"
        "%si = OpLoad %tSampledImg2D %tex2d\n"
        "%tx = OpImageSampleExplicitLod %v4f32 %si %uv Lod %cf0\n"
        "%tx0 = OpCompositeExtract %f32 %tx 0\n"
        "%res = OpBitcast %u32 %tx0",
        needs_tex2d=True,
    ),
    "TEX_FETCH": _cr(
        "%ic = OpCompositeConstruct %v2s32 %sa %sb\n"
        "%si = OpLoad %tSampledImg2D %tex2d\n"
        "%im = OpImage %tImg2D %si\n"
        "%tx = OpImageFetch %v4f32 %im %ic Lod %c0\n"
        "%tx0 = OpCompositeExtract %f32 %tx 0\n"
        "%res = OpBitcast %u32 %tx0",
        needs_tex2d=True,
    ),
    "TEX_GATHER": _cr(
        "%uv = OpCompositeConstruct %v2f32 %fa %fb\n"
        "%si = OpLoad %tSampledImg2D %tex2d\n"
        "%tx = OpImageGather %v4f32 %si %uv %c0\n"
        "%tx0 = OpCompositeExtract %f32 %tx 0\n"
        "%res = OpBitcast %u32 %tx0",
        needs_tex2d=True,
    ),
    "LD_TEX_IMM": _cr(
        "%ic = OpCompositeConstruct %v2s32 %sa %sb\n"
        "%sim = OpLoad %tStorImg2D %simg2d\n"
        "%tx = OpImageRead %v4u32 %sim %ic\n"
        "%res = OpCompositeExtract %u32 %tx 0",
        needs_storage_img=True,
    ),
    "LD_TEX": _cr(
        "%tidx = OpBitwiseAnd %u32 %a %c3\n"
        "%pimg = OpAccessChain %pStorImgArrElem %simg_arr %tidx\n"
        "%sim = OpLoad %tStorImgArrElem %pimg\n"
        "%ic = OpCompositeConstruct %v2s32 %sa %sb\n"
        "%tx = OpImageRead %v4u32 %sim %ic\n"
        "%res = OpCompositeExtract %u32 %tx 0",
        needs_storage_img_array=True,
    ),
    "LEA_TEX_IMM": _cr(
        "%ic = OpCompositeConstruct %v2s32 %sa %sb\n"
        "%sim = OpLoad %tStorImg2D %simg2d\n"
        "%tv = OpCompositeConstruct %v4u32 %a %b %c %d\n"
        "OpImageWrite %sim %ic %tv\n"
        "%res = OpIAdd %u32 %a %b",
        needs_storage_img=True,
    ),
    "LEA_TEX": _cr(
        "%tidx = OpBitwiseAnd %u32 %a %c3\n"
        "%pimg = OpAccessChain %pStorImgArrElem %simg_arr %tidx\n"
        "%sim = OpLoad %tStorImgArrElem %pimg\n"
        "%ic = OpCompositeConstruct %v2s32 %sa %sb\n"
        "%tv = OpCompositeConstruct %v4u32 %a %b %c %d\n"
        "OpImageWrite %sim %ic %tv\n"
        "%res = OpIAdd %u32 %a %b",
        needs_storage_img_array=True,
    ),
    "ST_CVT": _cr(
        "%ic = OpCompositeConstruct %v2s32 %sa %sb\n"
        "%sim = OpLoad %tStorImg2D %simg2d\n"
        "%tv = OpCompositeConstruct %v4u32 %a %b %c %d\n"
        "OpImageWrite %sim %ic %tv\n"
        "%res = OpIAdd %u32 %a %b",
        needs_storage_img=True,
    ),

    # --- Graphics (Fragment & Vertex) ---
    "TEX_GRADIENT": _fr(
        "%uv0 = OpVectorShuffle %v2f32 %v0 %v0 0 1\n"
        "%dx = OpVectorShuffle %v2f32 %v1 %v1 0 1\n"
        "%dy = OpVectorShuffle %v2f32 %v1 %v1 2 3\n"
        "%uv = OpFAdd %v2f32 %uv0 %dx\n"
        "%ti = OpLoad %tSampledImg2D %tex2d\n"
        "%res = OpImageSampleExplicitLod %v4f32 %ti %uv Grad %dx %dy",
        needs_tex2d=True,
    ),
    "LD_VAR_IMM": _fr("%res = OpFAdd %v4f32 %v0 %v1"),
    "LD_VAR": _fr(
        "%vidx0 = OpLoad %u32 %in_vidx\n"
        "%vidx = OpBitwiseAnd %u32 %vidx0 %c3\n"
        "%vptr = OpAccessChain %pInV4 %in_varr %vidx\n"
        "%vld = OpLoad %v4f32 %vptr\n"
        "%res = OpFAdd %v4f32 %vld %v0",
        needs_dyn_var_array=True,
    ),
    "LD_VAR_FLAT_IMM": _fr("%res = OpFAdd %v4f32 %v0 %vflat"),
    "LD_VAR_FLAT": _fr(
        "%vidx0 = OpLoad %u32 %in_vidx\n"
        "%vidx = OpBitwiseAnd %u32 %vidx0 %c3\n"
        "%vptr = OpAccessChain %pInV4 %in_varr %vidx\n"
        "%vld = OpLoad %v4f32 %vptr\n"
        "%res = OpFAdd %v4f32 %vld %v0",
        needs_dyn_var_flat_array=True,
    ),
    "LD_VAR_SPECIAL": _fr("%res = OpFAdd %v4f32 %v0 %fcoord"),
    "ATEST": _fr(
        "%x0 = OpCompositeExtract %f32 %v0 0\n"
        "%k = OpFOrdLessThan %bool %x0 %cf0\n"
        "OpSelectionMerge %dm None\n"
        "OpBranchConditional %k %dk %dm\n"
        "%dk = OpLabel\n"
        "OpKill\n"
        "%dm = OpLabel\n"
        "%res = OpFAdd %v4f32 %v0 %v1"
    ),
    "BLEND": _fr("%res = OpFAdd %v4f32 %v0 %v1"),
    "DISCARD": _fr(
        "%x0 = OpCompositeExtract %f32 %v0 0\n"
        "%k = OpFOrdLessThan %bool %x0 %cf0\n"
        "OpSelectionMerge %dm None\n"
        "OpBranchConditional %k %dk %dm\n"
        "%dk = OpLabel\n"
        "OpKill\n"
        "%dm = OpLabel\n"
        "%res = OpFAdd %v4f32 %v0 %v1"
    ),
    "ZS_EMIT": _fr(
        "%x0 = OpCompositeExtract %f32 %v0 0\n"
        "OpStore %out_depth %x0\n"
        "%res = OpFAdd %v4f32 %v0 %v1",
        needs_frag_depth=True,
    ),
    "CUBEFACE1": _fr(
        "%dir = OpVectorShuffle %v3f32 %v0 %v0 0 1 2\n"
        "%ci = OpLoad %tSampledCube %texcube\n"
        "%res = OpImageSampleImplicitLod %v4f32 %ci %dir",
        needs_cube=True,
    ),
    "CUBEFACE2": _fr(
        "%dir = OpVectorShuffle %v3f32 %v0 %v0 0 1 2\n"
        "%ci = OpLoad %tSampledCube %texcube\n"
        "%res = OpImageSampleImplicitLod %v4f32 %ci %dir",
        needs_cube=True,
    ),
    "CUBE_SSEL": _fr(
        "%dir = OpVectorShuffle %v3f32 %v0 %v0 0 1 2\n"
        "%ci = OpLoad %tSampledCube %texcube\n"
        "%res = OpImageSampleImplicitLod %v4f32 %ci %dir",
        needs_cube=True,
    ),
    "CUBE_TSEL": _fr(
        "%dir = OpVectorShuffle %v3f32 %v0 %v0 0 1 2\n"
        "%ci = OpLoad %tSampledCube %texcube\n"
        "%res = OpImageSampleImplicitLod %v4f32 %ci %dir",
        needs_cube=True,
    ),
    "VAR_TEX_SINGLE": _fr(
        "%uv = OpVectorShuffle %v2f32 %v0 %v0 0 1\n"
        "%ti = OpLoad %tSampledImg2D %tex2d\n"
        "%res = OpImageSampleImplicitLod %v4f32 %ti %uv",
        needs_tex2d=True,
    ),
    "VAR_TEX_GATHER": _fr(
        "%uv = OpVectorShuffle %v2f32 %v0 %v0 0 1\n"
        "%ti = OpLoad %tSampledImg2D %tex2d\n"
        "%res = OpImageGather %v4f32 %ti %uv %cs0",
        needs_tex2d=True,
    ),
    "LD_ATTR_IMM": _vr("%res = OpFAdd %v4f32 %a0 %a1"),
    "LD_ATTR": _vr(
        "%idx0 = OpLoad %u32 %in_idx\n"
        "%idx = OpBitwiseAnd %u32 %idx0 %c3\n"
        "%ptr = OpAccessChain %pInV4 %in_arr %idx\n"
        "%ld = OpLoad %v4f32 %ptr\n"
        "%res = OpFAdd %v4f32 %ld %a0",
        needs_dyn_attr_in_array=True,
    ),
    "LEA_ATTR_IMM": _vr("%res = OpFAdd %v4f32 %a0 %a1"),
}


# Targeted SPIR-V variant recipes for key non-default `(mnemonic, variant)` pairs.
VARIANT_RECIPES: dict[tuple[str, str], IsaRecipe] = {
    ("IADD", "v2u16"): _cr(
        "%va = OpBitcast %v2u16 %a\n%vb = OpBitcast %v2u16 %b\n"
        "%vt = OpIAdd %v2u16 %va %vb\n%res = OpBitcast %u32 %vt",
        caps=("Int16",),
    ),
    ("IADD", "v4u8"): _cr(
        "%va = OpBitcast %v4u8 %a\n%vb = OpBitcast %v4u8 %b\n"
        "%vt = OpIAdd %v4u8 %va %vb\n%res = OpBitcast %u32 %vt",
        caps=("Int8",),
    ),
    ("ISUB", "v2u16"): _cr(
        "%va = OpBitcast %v2u16 %a\n%vb = OpBitcast %v2u16 %b\n"
        "%vt = OpISub %v2u16 %va %vb\n%res = OpBitcast %u32 %vt",
        caps=("Int16",),
    ),
    ("IMUL", "v2u16"): _cr(
        "%va = OpBitcast %v2u16 %a\n%vb = OpBitcast %v2u16 %b\n"
        "%vt = OpIMul %v2u16 %va %vb\n%res = OpBitcast %u32 %vt",
        caps=("Int16",),
    ),
    ("IABS", "v2s16"): _cr(
        "%v = OpBitcast %v2s16 %a\n%ab = OpExtInst %v2s16 %ext SAbs %v\n%res = OpBitcast %u32 %ab",
        caps=("Int16",),
    ),
    ("IDPADD", "v4u8"): _cr(
        "%res = OpUDotAccSat %u32 %a %b %c PackedVectorFormat4x8Bit",
        caps=("DotProduct", "DotProductInput4x8BitPacked"),
        exts=("SPV_KHR_integer_dot_product",),
    ),
    ("FADD", "v2f16"): _cr(
        "%va = OpBitcast %v2f16 %a\n%vb = OpBitcast %v2f16 %b\n"
        "%vt = OpFAdd %v2f16 %va %vb\n%res = OpBitcast %u32 %vt",
        caps=("Float16",),
    ),
    ("FMUL", "v2f16"): _cr(
        "%va = OpBitcast %v2f16 %a\n%vb = OpBitcast %v2f16 %b\n"
        "%vt = OpFMul %v2f16 %va %vb\n%res = OpBitcast %u32 %vt",
        caps=("Float16",),
    ),
    ("FMA", "v2f16"): _cr(
        "%va = OpBitcast %v2f16 %a\n%vb = OpBitcast %v2f16 %b\n%vc = OpBitcast %v2f16 %c\n"
        "%vt = OpExtInst %v2f16 %ext Fma %va %vb %vc\n%res = OpBitcast %u32 %vt",
        caps=("Float16",),
    ),
    ("FMIN", "v2f16"): _cr(
        "%va = OpBitcast %v2f16 %a\n%vb = OpBitcast %v2f16 %b\n"
        "%vt = OpExtInst %v2f16 %ext FMin %va %vb\n%res = OpBitcast %u32 %vt",
        caps=("Float16",),
    ),
    ("FMAX", "v2f16"): _cr(
        "%va = OpBitcast %v2f16 %a\n%vb = OpBitcast %v2f16 %b\n"
        "%vt = OpExtInst %v2f16 %ext FMax %va %vb\n%res = OpBitcast %u32 %vt",
        caps=("Float16",),
    ),
    ("FROUND", "v2f16"): _cr(
        "%va = OpBitcast %v2f16 %a\n"
        "%vt = OpExtInst %v2f16 %ext Floor %va\n%res = OpBitcast %u32 %vt",
        caps=("Float16",),
    ),
    ("FADD", "clamp_0_1"): _cr(
        "%t = OpFAdd %f32 %fa %fb\n%cl = OpExtInst %f32 %ext FClamp %t %cf0 %cf1\n%res = OpBitcast %u32 %cl"
    ),
}

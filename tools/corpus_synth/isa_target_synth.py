"""
Targeted Single-Instruction Valhall ISA Synthesizer (Part 2B).

Synthesizes valid Vulkan 1.3 SPIR-V shaders guaranteed to induce a specific
Valhall ISA mnemonic in `EBIN[0]` (main shader, never hoisted into Pilot
`EBIN[1]`), verifies emission through `spirv2valhall` (`libMali-Gxx`), and
optionally validates the lowering rule immediately with `equiv_fuzzer`.

Supports three complexity levels (`minimal`, `medium`, `branchy`) across
Compute (`LocalSize 32 1 1` with thread-varying `gl_GlobalInvocationID.x`),
Fragment, and Vertex shader harnesses.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from tools.corpus_synth.oracle import _parse_valhall_mnemonic
from tools.corpus_synth.valhall_isel_catalog import get_rule_for_mnemonic
from tools.coverage.equiv_fuzzer import run_concolic_equiv_fuzz
from tools.coverage.shrinker import DiffPolicy, DivergenceReport
from tools.symbolic_ssa.passes import canonicalize_ssa_module
from tools.verify.spvasm_io import assemble_spvasm, load_equiv_bundle


@dataclasses.dataclass(frozen=True)
class IsaRecipe:
    """Executable SPIR-V assembly snippet that induces one or more Valhall mnemonics."""

    stage: str  # "compute", "fragment", "vertex"
    body: str
    caps: Tuple[str, ...] = ()
    exts: Tuple[str, ...] = ()
    needs_wg: bool = False
    needs_tex2d: bool = False
    needs_storage_img: bool = False
    needs_cube: bool = False
    needs_rodata: bool = False
    needs_frag_depth: bool = False
    result_var: str = "%res"


@dataclasses.dataclass
class IsaTargetResult:
    """Outcome of synthesizing and optionally fuzz-validating a target Valhall mnemonic."""

    mnemonic: str
    complexity: str
    stage: str
    spvasm: str
    spv_path: str
    compiled_ok: bool
    target_emitted: bool
    emitted_mnemonics: Set[str]
    emitted_variants: List[str]
    equiv_status: str = ""
    diverging_vectors: int = 0
    benign_fp_diffs: int = 0
    divergence_report: Optional[DivergenceReport] = None
    elapsed_ms: float = 0.0
    error: str = ""


def _cr(
    body: str,
    *,
    caps: Tuple[str, ...] = (),
    exts: Tuple[str, ...] = (),
    needs_wg: bool = False,
    needs_tex2d: bool = False,
    needs_storage_img: bool = False,
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
        needs_rodata=needs_rodata,
    )


def _fr(
    body: str,
    *,
    needs_tex2d: bool = False,
    needs_cube: bool = False,
    needs_frag_depth: bool = False,
) -> IsaRecipe:
    return IsaRecipe(
        stage="fragment",
        body=body,
        needs_tex2d=needs_tex2d,
        needs_cube=needs_cube,
        needs_frag_depth=needs_frag_depth,
    )


def _vr(body: str) -> IsaRecipe:
    return IsaRecipe(stage="vertex", body=body)


# Executable SPIR-V recipes keyed by canonical Valhall mnemonic.
RECIPES: Dict[str, IsaRecipe] = {
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
    "SHADD": _cr("%res = OpIAdd %u32 %a %b"),  # emitted automatically by inbuf/outbuf SSBO indexing

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
    "S16_TO_S32": _cr(
        "%v2 = OpBitcast %v2s16 %a\n%e0 = OpCompositeExtract %s16 %v2 0\n"
        "%s0 = OpSConvert %s32 %e0\n%res = OpBitcast %u32 %s0",
        caps=("Int16",),
    ),

    # --- Memory, Atomics, Subgroups, Sync, Control Flow ---
    "LOAD": _cr("%res = OpIAdd %u32 %a %b"),
    "STORE": _cr("%res = OpIAdd %u32 %a %b"),
    "ATOM": _cr(
        "%po2 = OpAccessChain %pU32 %outbuf %c0 %i1\n"
        "%ign = OpAtomicIAdd %u32 %po2 %c1 %c0 %a\n"
        "%res = OpIAdd %u32 %a %b"
    ),
    "ATOM_RETURN": _cr(
        "%po2 = OpAccessChain %pU32 %outbuf %c0 %i1\n"
        "%res = OpAtomicIAdd %u32 %po2 %c1 %c0 %a"
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
    "LEA_TEX_IMM": _cr(
        "%ic = OpCompositeConstruct %v2s32 %sa %sb\n"
        "%sim = OpLoad %tStorImg2D %simg2d\n"
        "%tv = OpCompositeConstruct %v4u32 %a %b %c %d\n"
        "OpImageWrite %sim %ic %tv\n"
        "%res = OpIAdd %u32 %a %b",
        needs_storage_img=True,
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
    "LD_VAR_IMM": _fr("%res = OpFAdd %v4f32 %v0 %v1"),
    "LD_VAR_FLAT_IMM": _fr("%res = OpFAdd %v4f32 %v0 %vflat"),
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
    "LD_ATTR_IMM": _vr("%res = OpFAdd %v4f32 %a0 %a1"),
    "LEA_ATTR_IMM": _vr("%res = OpFAdd %v4f32 %a0 %a1"),
}


def list_synthesizable_mnemonics() -> List[str]:
    """Return all Valhall mnemonics that have an executable synthesis recipe."""
    return sorted(RECIPES.keys())


def _build_compute_spvasm(recipe: IsaRecipe, complexity: str) -> str:
    caps = ["Shader"] + list(recipe.caps)
    if recipe.needs_storage_img:
        caps.append("StorageImageReadWithoutFormat")
        caps.append("StorageImageWriteWithoutFormat")
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
    if "Int16" in caps:
        type_block.extend(["%u16 = OpTypeInt 16 0", "%s16 = OpTypeInt 16 1", "%v2u16 = OpTypeVector %u16 2", "%v2s16 = OpTypeVector %s16 2"])
    if "Int8" in caps:
        type_block.extend(["%u8 = OpTypeInt 8 0", "%s8 = OpTypeInt 8 1", "%v4u8 = OpTypeVector %u8 4", "%v4s8 = OpTypeVector %s8 4"])
    if "Float16" in caps:
        type_block.extend(["%f16 = OpTypeFloat 16", "%v2f16 = OpTypeVector %f16 2"])
    type_block.extend([
        "%c0 = OpConstant %u32 0",
        "%c1 = OpConstant %u32 1",
        "%c2 = OpConstant %u32 2",
        "%c3 = OpConstant %u32 3",
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


def _build_fragment_spvasm(recipe: IsaRecipe, complexity: str) -> str:
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
%f32 = OpTypeFloat 32
%v2f32 = OpTypeVector %f32 2
%v3f32 = OpTypeVector %f32 3
%v4f32 = OpTypeVector %f32 4
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


def _build_vertex_spvasm(recipe: IsaRecipe, complexity: str) -> str:
    post = "%final_res = OpFMul %v4f32 %res %a1\n" if complexity == "medium" else "%final_res = OpCopyObject %v4f32 %res\n"
    return f"""OpCapability Shader
%ext = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %in_a0 %in_a1 %out_pos %out_v0
OpDecorate %in_a0 Location 0
OpDecorate %in_a1 Location 1
OpDecorate %out_pos BuiltIn Position
OpDecorate %out_v0 Location 0
%void = OpTypeVoid
%fn_t = OpTypeFunction %void
%f32 = OpTypeFloat 32
%v4f32 = OpTypeVector %f32 4
%pInV4 = OpTypePointer Input %v4f32
%pOutV4 = OpTypePointer Output %v4f32
%in_a0 = OpVariable %pInV4 Input
%in_a1 = OpVariable %pInV4 Input
%out_pos = OpVariable %pOutV4 Output
%out_v0 = OpVariable %pOutV4 Output
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


def build_spvasm_for_isa(mnemonic: str, complexity: str = "minimal") -> Tuple[str, str]:
    """Return `(stage, spvasm_text)` for `mnemonic` at the requested complexity."""
    mn = mnemonic.upper()
    recipe = RECIPES.get(mn)
    if recipe is None:
        rule = get_rule_for_mnemonic(mn)
        hint = f" (catalog tier={rule.tier}: {rule.spirv_pattern})" if rule else ""
        raise ValueError(f"No executable synthesis recipe for Valhall mnemonic '{mn}'{hint}")
    if recipe.stage == "compute":
        return recipe.stage, _build_compute_spvasm(recipe, complexity)
    if recipe.stage == "fragment":
        return recipe.stage, _build_fragment_spvasm(recipe, complexity)
    if recipe.stage == "vertex":
        return recipe.stage, _build_vertex_spvasm(recipe, complexity)
    raise ValueError(f"Unknown stage '{recipe.stage}'")


def synthesize_for_isa(
    mnemonic: str,
    *,
    complexity: str = "minimal",
    out_dir: Optional[str] = None,
    bin_path: str = "./build/spirv2valhall",
    verify_equiv: bool = False,
    k_init: int = 32,
    fp_range: str = "unconstrained",
    seed: int = 42,
) -> IsaTargetResult:
    """Synthesize a SPIR-V shader targeting `mnemonic`, verify `EBIN[0]` emission, and optionally fuzz-check."""
    t0 = time.perf_counter()
    mn = mnemonic.upper()
    stage, spvasm = build_spvasm_for_isa(mn, complexity=complexity)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        asm_path = os.path.join(out_dir, f"isa_{mn.lower()}_{complexity}.spvasm")
        spv_path = os.path.join(out_dir, f"isa_{mn.lower()}_{complexity}.spv")
        cleanup = False
    else:
        td = tempfile.mkdtemp(prefix=f"isa_{mn.lower()}_")
        asm_path = os.path.join(td, f"isa_{mn.lower()}_{complexity}.spvasm")
        spv_path = os.path.join(td, f"isa_{mn.lower()}_{complexity}.spv")
        cleanup = True

    with open(asm_path, "w", encoding="utf-8") as f:
        f.write(spvasm)

    try:
        assemble_spvasm(asm_path, spv_path, target_env="vulkan1.3")
        bundle = load_equiv_bundle(spv_path, bin_path, defau=True, opt_ssa=True)
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000.0
        return IsaTargetResult(
            mnemonic=mn,
            complexity=complexity,
            stage=stage,
            spvasm=spvasm,
            spv_path=spv_path if not cleanup else "",
            compiled_ok=False,
            target_emitted=False,
            emitted_mnemonics=set(),
            emitted_variants=[],
            elapsed_ms=elapsed,
            error=str(exc),
        )

    pc_disasm: Dict[str, str] = bundle.get("mbs2_context", {}).get("pc_disasm", {})
    emitted_variants: List[str] = []
    emitted_mnemonics: Set[str] = set()
    for line in pc_disasm.values():
        tok = _parse_valhall_mnemonic(line)
        if tok:
            emitted_variants.append(tok)
            emitted_mnemonics.add(tok.split(".")[0])

    target_emitted = mn in emitted_mnemonics
    equiv_status = ""
    divs = 0
    benign = 0
    div_rep: Optional[DivergenceReport] = None

    if verify_equiv and bundle.get("direct_ok") and bundle.get("valhall_ok"):
        d_ssa, _ = canonicalize_ssa_module(bundle["direct_ssa"])
        v_ssa, _ = canonicalize_ssa_module(bundle["valhall_ssa"])
        ef = run_concolic_equiv_fuzz(
            bundle,
            d_ssa,
            v_ssa,
            k_init=k_init,
            fp_range=fp_range,
            seed=seed,
            policy=DiffPolicy(),
        )
        equiv_status = ef.status
        divs = ef.diverging_vectors
        benign = ef.benign_fp_diffs
        div_rep = ef.first_divergence

    elapsed = (time.perf_counter() - t0) * 1000.0
    return IsaTargetResult(
        mnemonic=mn,
        complexity=complexity,
        stage=stage,
        spvasm=spvasm,
        spv_path=spv_path,
        compiled_ok=bool(bundle.get("direct_ok") and bundle.get("valhall_ok")),
        target_emitted=target_emitted,
        emitted_mnemonics=emitted_mnemonics,
        emitted_variants=sorted(set(emitted_variants)),
        equiv_status=equiv_status,
        diverging_vectors=divs,
        benign_fp_diffs=benign,
        divergence_report=div_rep,
        elapsed_ms=elapsed,
    )

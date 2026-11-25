"""
Mali Valhall Instruction Selection (ISel) SPIR-V Pattern Catalogue (Part A).

Maps every Valhall ISA mnemonic in `specs/lowering/valhall_lowering.json` (201
unique mnemonics) to its reverse-engineered Mali offline compiler lowering path
(`SPIR-V -> LLVM IR / llvm.bifrost.* -> Valhall ISA`, derived from
`libGLES_mali.so` r54p3-00eac0 and empirically verified against `libMali-Gxx`
via `build/spirv2valhall`).

Each entry is classified into one of three tiers:
  - `compute`: directly inducible in a Vulkan GLCompute SPIR-V shader.
  - `graphics`: inducible in a Vertex or Fragment SPIR-V shader (or texture/cube
    graphics pipeline).
  - `internal`: hardware/pilot/driver-internal, legacy, or non-Vulkan-SPIR-V
    instruction (not directly emitted from standard Vulkan SPIR-V).

Part B (`tools/corpus_synth` targeted single-instruction generator) consumes
`VALHALL_ISEL_CATALOG` to synthesize valid, anti-hoisted, anti-DCE shaders that
guarantee emission of a requested Valhall mnemonic for single-instruction
lowering validation in `equiv_fuzzer`.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclasses.dataclass(frozen=True)
class ValhallIselRule:
    """Declarative mapping from a Valhall ISA mnemonic to its inducing SPIR-V pattern."""

    mnemonic: str
    tier: str  # "compute", "graphics", "internal"
    category: str
    spirv_pattern: str
    llvm_ir_path: str
    constraints: str
    required_caps: Tuple[str, ...] = ()
    required_exts: Tuple[str, ...] = ()
    verified_in_malioc: bool = False


def _r(
    mnemonic: str,
    tier: str,
    category: str,
    spirv_pattern: str,
    llvm_ir_path: str,
    constraints: str = "Thread-varying operand (gl_GlobalInvocationID.x) to prevent Pilot hoisting",
    caps: Tuple[str, ...] = (),
    exts: Tuple[str, ...] = (),
    verified: bool = False,
) -> ValhallIselRule:
    return ValhallIselRule(
        mnemonic=mnemonic,
        tier=tier,
        category=category,
        spirv_pattern=spirv_pattern,
        llvm_ir_path=llvm_ir_path,
        constraints=constraints,
        required_caps=caps,
        required_exts=exts,
        verified_in_malioc=verified,
    )


VALHALL_ISEL_CATALOG: Tuple[ValhallIselRule, ...] = (
    # --- Integer Arithmetic ---
    _r("IADD", "compute", "int_alu", "OpIAdd %u32 %a %b (or %u64 / %v2u16 / %v4u8)", "add i32 / i64 / <2 x i16> / <4 x i8>", verified=True),
    _r("IADD_IMM", "compute", "int_alu", "OpIAdd %u32 %a %const_large", "add i32 %a, imm (outside small-const table)", verified=True),
    _r("ISUB", "compute", "int_alu", "OpISub %u32 %a %b", "sub i32 %a, %b", verified=True),
    _r("IMUL", "compute", "int_alu", "OpIMul %u32 %a %b", "mul i32 %a, %b", verified=True),
    _r("IABS", "compute", "int_alu", "GLSL.SAbs %s32 %sa (or %v2s16 / %v4s8)", "llvm.abs.i32 / llvm.abs.v4i8", caps=("Int8",), verified=True),
    _r("HADD", "compute", "int_alu", "OpIAdd (OpBitwiseAnd %a %b) (OpShiftRightLogical (OpBitwiseXor %a %b) 1)", "llvm.bifrost.haddu / hadds (DAG pattern match)", verified=True),
    _r("IDPADD", "compute", "int_alu", "OpSDot / OpUDot / OpSDotAccSat %a %b [%acc] PackedVectorFormat4x8Bit", "llvm.bifrost.idpadd.{ss,uu}[.sat].v4i8", caps=("DotProduct", "DotProductInput4x8BitPacked"), exts=("SPV_KHR_integer_dot_product",), verified=True),
    _r("SHADD", "compute", "int_alu", "OpAccessChain %pU32 %ssbo 0 %idx (64-bit base + (idx << 2))", "add i64 %base, (shl (zext/sext i32 %idx), 2)", verified=True),
    _r("IADDC", "compute", "int_alu", "64-bit multi-word add carry chain (OpIAdd %u64 or chained OpIAddCarry)", "llvm.bifrost.iaddc / uadd.with.overflow", caps=("Int64",)),
    _r("ISUBB", "compute", "int_alu", "64-bit multi-word sub borrow chain (OpISub %u64 or chained OpISubBorrow)", "llvm.bifrost.isubb / usub.with.overflow", caps=("Int64",)),
    _r("MMUL", "compute", "int_alu", "OpCooperativeMatrixMulAddKHR (8x8 / 4x4 matrix multiply-add)", "llvm.bifrost.mmul.{f32,v2f16,v4i8} / llvm.bifrost.coopmat.mul.add", caps=("CooperativeMatrixKHR",), exts=("SPV_KHR_cooperative_matrix",)),

    # --- Bitwise, Shifts, Rotates, Bitfields ---
    _r("AND", "compute", "bit_logic", "OpBitwiseAnd %u32 %a %b", "and i32 %a, %b", verified=True),
    _r("OR", "compute", "bit_logic", "OpBitwiseOr %u32 %a %b", "or i32 %a, %b", verified=True),
    _r("XOR", "compute", "bit_logic", "OpBitwiseXor %u32 %a %b", "xor i32 %a, %b", verified=True),
    _r("NOT", "compute", "bit_logic", "OpNot %u32 %a (non-folded, e.g. stored or added)", "xor i32 %a, -1", verified=True),
    _r("AND_IMM", "compute", "bit_logic", "OpBitwiseAnd %u32 %a %const_non_small", "and i32 %a, imm"),
    _r("OR_IMM", "compute", "bit_logic", "OpBitwiseOr %u32 %a %const_non_small", "or i32 %a, imm"),
    _r("XOR_IMM", "compute", "bit_logic", "OpBitwiseXor %u32 %a %const_non_small", "xor i32 %a, imm"),
    _r("LSHIFT", "compute", "bit_shift", "OpShiftLeftLogical %u32 %a %b", "shl i32 %a, %b", verified=True),
    _r("RSHIFT", "compute", "bit_shift", "OpShiftRightLogical %u32 %a %b", "lshr i32 %a, %b", verified=True),
    _r("ARSHIFT", "compute", "bit_shift", "OpShiftRightArithmetic %s32 %sa %b", "ashr i32 %sa, %b", verified=True),
    _r("LSHIFT_AND", "compute", "bit_shift_logic", "OpBitwiseAnd (OpShiftLeftLogical %a %b) %c", "and (shl %a, %b), %c", verified=True),
    _r("LSHIFT_OR", "compute", "bit_shift_logic", "OpBitwiseOr (OpShiftLeftLogical %a %b) %c", "or (shl %a, %b), %c", verified=True),
    _r("LSHIFT_XOR", "compute", "bit_shift_logic", "OpBitwiseXor (OpShiftLeftLogical %a %b) %c", "xor (shl %a, %b), %c", verified=True),
    _r("RSHIFT_AND", "compute", "bit_shift_logic", "OpBitwiseAnd (OpShiftRightLogical %a %b) %c", "and (lshr %a, %b), %c", verified=True),
    _r("RSHIFT_OR", "compute", "bit_shift_logic", "OpBitwiseOr (OpShiftRightLogical %a %b) %c", "or (lshr %a, %b), %c", verified=True),
    _r("RSHIFT_XOR", "compute", "bit_shift_logic", "OpBitwiseXor (OpShiftRightLogical %a %b) %c", "xor (lshr %a, %b), %c", verified=True),
    _r("ARSHIFT_AND", "compute", "bit_shift_logic", "OpBitwiseAnd (OpShiftRightArithmetic %sa %b) %c (when reg-pressure/mask allows fusion)", "and (ashr %sa, %b), %c"),
    _r("ARSHIFT_OR", "compute", "bit_shift_logic", "OpBitwiseOr (OpShiftRightArithmetic %sa %b) %c", "or (ashr %sa, %b), %c"),
    _r("ARSHIFT_XOR", "compute", "bit_shift_logic", "OpBitwiseXor (OpShiftRightArithmetic %sa %b) %c", "xor (ashr %sa, %b), %c"),
    _r("LROT", "compute", "bit_rot", "OpBitwiseOr (OpShiftLeftLogical %a 5) (OpShiftRightLogical %a 27)", "llvm.fshl.i32(%a, %a, 5)", constraints="Constant shift k or dynamic (a<<b)|(a>>(32-b))", verified=True),
    _r("RROT", "compute", "bit_rot", "OpBitwiseOr (OpShiftRightLogical %a %b) (OpShiftLeftLogical %a (OpISub 32 %b))", "llvm.fshr.i32(%a, %a, %b)", constraints="Dynamic shift %b required (constant RROT canonicalizes to LROT)", verified=True),
    _r("LROT_AND", "compute", "bit_rot", "OpBitwiseAnd (LROT(%a, %b)) %c", "and (fshl %a, %a, %b), %c"),
    _r("LROT_OR", "compute", "bit_rot", "OpBitwiseOr (LROT(%a, %b)) %c", "or (fshl %a, %a, %b), %c"),
    _r("LROT_XOR", "compute", "bit_rot", "OpBitwiseXor (LROT(%a, %b)) %c", "xor (fshl %a, %a, %b), %c"),
    _r("RROT_AND", "compute", "bit_rot", "OpBitwiseAnd (RROT(%a, %b)) %c", "and (fshr %a, %a, %b), %c"),
    _r("RROT_OR", "compute", "bit_rot", "OpBitwiseOr (RROT(%a, %b)) %c", "or (fshr %a, %a, %b), %c"),
    _r("RROT_XOR", "compute", "bit_rot", "OpBitwiseXor (RROT(%a, %b)) %c", "xor (fshr %a, %a, %b), %c"),
    _r("CLZ", "compute", "bit_unary", "GLSL.FindUMsb %u32 %a (or GLSL.FindSMsb)", "llvm.ctlz.i32 / llvm.bifrost.clz.m1", verified=True),
    _r("BITREV", "compute", "bit_unary", "OpBitReverse %u32 %a", "llvm.bitreverse.i32", verified=True),
    _r("POPCOUNT", "compute", "bit_unary", "OpBitCount %u32 %a", "llvm.ctpop.i32", verified=True),
    _r("MUX", "compute", "bit_mux", "OpBitFieldInsert / bitselect / GLSL.Atan2 sign-bit select", "llvm.bifrost.bitselect / select on sign/zero bit", verified=True),

    # --- Moves, Packing & Swizzles ---
    _r("MOV", "compute", "alu_mov", "Register copy across phi/branch or FAU-to-GPR move", "COPY / bitcast", verified=True),
    _r("MOV_IMM", "compute", "alu_mov", "OpConstant %u32 0x12345678 used in ALU/Store", "Constant i32 materialization", verified=True),
    _r("MKVEC", "compute", "alu_pack", "OpVectorShuffle %v4u8 %a %b 0 5 2 7 (or %v2u16 / packUnorm4x8)", "shufflevector / insertelement <4 x i8> or <2 x i16>", caps=("Int8",), verified=True),
    _r("SWZ", "compute", "alu_pack", "OpVectorShuffle %v2u16 %a %a 1 0 (or .h10 / .b0123 swizzle)", "shufflevector <2 x i16> / <4 x i8>", caps=("Int16",), verified=True),
    _r("ADR", "compute", "alu_adr", "Dynamic index into large OpConstantComposite array (in Function/Private storage)", "PC-relative rodata table address", constraints="Array >= 16 elements indexed by thread-varying %a"),

    # --- Comparisons & Conditional Select ---
    _r("ICMP", "compute", "cmp_int", "OpIEqual / OpINotEqual / OpULessThan / OpSLessThan", "icmp eq/ne/ult/slt i32", verified=True),
    _r("ICMP_AND", "compute", "cmp_int", "OpLogicalAnd (OpULessThan %a %b) (OpIEqual %b %c)", "and (icmp ...), (icmp ...)", verified=True),
    _r("ICMP_OR", "compute", "cmp_int", "OpLogicalOr (OpULessThan %a %b) (OpIEqual %b %c)", "or (icmp ...), (icmp ...)", verified=True),
    _r("FCMP", "compute", "cmp_float", "OpFOrdLessThan / OpFOrdEqual / OpIsNan", "fcmp olt/oeq/uno float", verified=True),
    _r("FCMP_AND", "compute", "cmp_float", "OpLogicalAnd (OpFOrdLessThan %fa %fb) (OpFOrdGreaterThan %fb %fc)", "and (fcmp ...), (fcmp ...)", verified=True),
    _r("FCMP_OR", "compute", "cmp_float", "OpLogicalOr (OpFOrdLessThan %fa %fb) (OpFOrdGreaterThan %fb %fc)", "or (fcmp ...), (fcmp ...)", verified=True),
    _r("CSEL", "compute", "cmp_csel", "OpSelect %u32 (OpULessThan %a %b) %c %d", "select (icmp/fcmp ...), %c, %d", verified=True),

    # --- Floating-Point Core & Rounding ---
    _r("FADD", "compute", "fp_alu", "OpFAdd %f32 %fa %fb (or %v2f16)", "fadd float / <2 x half>", verified=True),
    _r("FADD_IMM", "compute", "fp_alu", "OpFAdd %f32 %fa %const_fp (or GLSL.Sin/Cos range reduction)", "fadd float %fa, imm", verified=True),
    _r("FMUL", "compute", "fp_alu", "OpFMul %f32 %fa %fb (or %v2f16)", "fmul float / <2 x half>", verified=True),
    _r("FMA", "compute", "fp_alu", "GLSL.Fma %f32 %fa %fb %fc (or OpFAdd (OpFMul %fa %fb) %fc)", "llvm.fma.f32 / fmul+fadd contraction", verified=True),
    _r("FMIN", "compute", "fp_alu", "GLSL.FMin / GLSL.NMin %f32 %fa %fb", "llvm.minnum.f32", verified=True),
    _r("FMAX", "compute", "fp_alu", "GLSL.FMax / GLSL.NMax %f32 %fa %fb", "llvm.maxnum.f32", verified=True),
    _r("FABS", "compute", "fp_alu", "GLSL.FAbs %f32 %fa (when not folded into consumer .abs modifier)", "llvm.fabs.f32"),
    _r("FNEG", "compute", "fp_alu", "OpFNegate %f32 %fa (when not folded into consumer .neg modifier)", "fneg float"),
    _r("FROUND", "compute", "fp_alu", "GLSL.Floor / GLSL.Ceil / GLSL.Trunc / GLSL.RoundEven %f32 %fa", "llvm.floor/ceil/trunc/roundeven.f32 / llvm.bifrost.fround.even", verified=True),
    _r("FPCLASS", "compute", "fp_alu", "OpIsNan / OpIsInf when folded into fpclass check", "llvm.bifrost.isnan / fpclass"),

    # --- Floating-Point SFU & Transcendentals ---
    _r("FRCP", "compute", "fp_sfu", "OpFDiv %f32 1.0 %fa (or OpFDiv %fa %fb)", "llvm.bifrost.recip / llvm.bifrost.divide", verified=True),
    _r("FRSQ", "compute", "fp_sfu", "GLSL.InverseSqrt %f32 %fa", "llvm.bifrost.rsqrt", verified=True),
    _r("FSQRT", "compute", "fp_sfu", "GLSL.Sqrt %f32 %fa (lowered via FRSQ+FRCP on G710 or FSQRT where supported)", "llvm.bifrost.sqrt", verified=True),
    _r("FREXPE", "compute", "fp_sfu", "GLSL.FrexpStruct %fa (also emitted inside GLSL.Log2 / GLSL.Pow)", "llvm.bifrost.frexpe", verified=True),
    _r("FREXPM", "compute", "fp_sfu", "GLSL.FrexpStruct %fa (mantissa component)", "llvm.bifrost.frexpm", verified=True),
    _r("LDEXP", "compute", "fp_sfu", "GLSL.Ldexp %fa %sb (also emitted inside GLSL.Atan2 / GLSL.Exp2)", "llvm.bifrost.ldexp", verified=True),
    _r("ILOGB", "compute", "fp_sfu", "Integer unbiased exponent extraction (OpenCL ilogb / special log2 path)", "llvm.bifrost.ilogb"),
    _r("LOGB", "compute", "fp_sfu", "Float unbiased exponent extraction (OpenCL logb)", "llvm.bifrost.logb"),
    _r("FLOGD", "compute", "fp_sfu", "GLSL.Log2 / GLSL.Log / GLSL.Pow %f32 %fa", "llvm.bifrost.log2 / llvm.bifrost.log", verified=True),
    _r("FADD_LSCALE", "compute", "fp_sfu", "GLSL.Log2 / GLSL.Log / GLSL.Pow %f32 %fa (mantissa log scaling)", "llvm.bifrost.log2 expansion", verified=True),
    _r("FMUL_RSCALE", "compute", "fp_sfu", "GLSL.Sin / GLSL.Cos / GLSL.Pow (exponent-scaled multiply)", "llvm.bifrost.sin.gles / powr expansion", verified=True),
    _r("FEXP", "compute", "fp_sfu", "GLSL.Exp2 / GLSL.Exp / GLSL.Pow %f32 %fa", "llvm.bifrost.exp2 / llvm.bifrost.exp", verified=True),
    _r("FSIN_TABLE", "compute", "fp_sfu", "GLSL.Sin %f32 %fa", "llvm.bifrost.sin.gles / llvm.bifrost.sin.lookup", verified=True),
    _r("FCOS_TABLE", "compute", "fp_sfu", "GLSL.Cos %f32 %fa", "llvm.bifrost.cos.gles / llvm.bifrost.cos.lookup", verified=True),
    _r("FATAN_ASSIST", "compute", "fp_sfu", "GLSL.Atan2 %f32 %fa %fb (or GLSL.Atan)", "llvm.bifrost.atan2 / llvm.bifrost.atan", verified=True),
    _r("FATAN_TABLE", "compute", "fp_sfu", "GLSL.Atan2 %f32 %fa %fb (or GLSL.Atan)", "llvm.bifrost.atan2 / llvm.bifrost.atan", verified=True),
    _r("FSINCOS_OFFSET", "compute", "fp_sfu", "GLSL.Sin / GLSL.Cos (fast/high-precision LUT offset)", "llvm.bifrost.sin.fast / cos.fast"),
    _r("FMUL_SLICE", "compute", "fp_sfu", "GLSL.Sin / GLSL.Cos range-reduction slice multiply", "llvm.bifrost.sin.gles range reduction"),
    _r("FPOW_SC_DET", "compute", "fp_sfu", "GLSL.Pow %f32 %fa %fb (OpenCL/strict pow special-case detect)", "llvm.bifrost.powr / pow"),
    _r("FPOW_SC_APPLY", "compute", "fp_sfu", "GLSL.Pow %f32 %fa %fb (OpenCL/strict pow special-case apply)", "llvm.bifrost.powr / pow"),

    # --- Conversions ---
    _r("F32_TO_S32", "compute", "cvt", "OpConvertFToS %s32 %fa", "fptosi float to i32 / llvm.bifrost.fptosi.tz", verified=True),
    _r("F32_TO_U32", "compute", "cvt", "OpConvertFToU %u32 %fa", "fptoui float to i32 / llvm.bifrost.fptoui.tz", verified=True),
    _r("S32_TO_F32", "compute", "cvt", "OpConvertSToF %f32 %sa", "sitofp i32 to float", verified=True),
    _r("U32_TO_F32", "compute", "cvt", "OpConvertUToF %f32 %a", "uitofp i32 to float", verified=True),
    _r("F16_TO_F32", "compute", "cvt", "GLSL.UnpackHalf2x16 / OpFConvert %f32 %f16_val", "fpext half to float", caps=("Float16",), verified=True),
    _r("F32_TO_F16", "compute", "cvt", "OpFConvert %f16 %fa (scalar f16)", "fptrunc float to half", caps=("Float16",)),
    _r("V2F32_TO_V2F16", "compute", "cvt", "GLSL.PackHalf2x16 (OpCompositeConstruct %v2f32 %fa %fb)", "fptrunc <2 x float> to <2 x half>", verified=True),
    _r("F16_TO_S32", "compute", "cvt", "OpConvertFToS %s32 %f16_val", "fptosi half to i32", caps=("Float16",)),
    _r("F16_TO_U32", "compute", "cvt", "OpConvertFToU %u32 %f16_val", "fptoui half to i32", caps=("Float16",)),
    _r("V2F16_TO_V2S16", "compute", "cvt", "OpConvertFToS %v2s16 (OpBitcast %v2f16 %a)", "fptosi <2 x half> to <2 x i16>", caps=("Float16", "Int16"), verified=True),
    _r("V2F16_TO_V2U16", "compute", "cvt", "OpConvertFToU %v2u16 (OpBitcast %v2f16 %a)", "fptoui <2 x half> to <2 x i16>", caps=("Float16", "Int16"), verified=True),
    _r("V2S16_TO_V2F16", "compute", "cvt", "OpConvertSToF %v2f16 (OpBitcast %v2s16 %a)", "sitofp <2 x i16> to <2 x half>", caps=("Float16", "Int16"), verified=True),
    _r("V2U16_TO_V2F16", "compute", "cvt", "OpConvertUToF %v2f16 (OpBitcast %v2u16 %a)", "uitofp <2 x i16> to <2 x half>", caps=("Float16", "Int16"), verified=True),
    _r("S8_TO_F32", "compute", "cvt", "OpConvertSToF %f32 (OpCompositeExtract %s8 %v4s8 0)", "sitofp i8 to float", caps=("Int8",), verified=True),
    _r("U8_TO_F32", "compute", "cvt", "OpConvertUToF %f32 (OpCompositeExtract %u8 %v4u8 0)", "uitofp i8 to float", caps=("Int8",), verified=True),
    _r("S16_TO_F32", "compute", "cvt", "OpConvertSToF %f32 (OpCompositeExtract %s16 %v2s16 0)", "sitofp i16 to float", caps=("Int16",), verified=True),
    _r("U16_TO_F32", "compute", "cvt", "OpConvertUToF %f32 (OpCompositeExtract %u16 %v2u16 0)", "uitofp i16 to float", caps=("Int16",), verified=True),
    _r("S8_TO_S32", "compute", "cvt", "OpSConvert %s32 (OpCompositeExtract %s8 %v4s8 0) (non-folded)", "sext i8 to i32", caps=("Int8",)),
    _r("U8_TO_U32", "compute", "cvt", "OpUConvert %u32 (OpCompositeExtract %u8 %v4u8 0) (non-folded)", "zext i8 to i32", caps=("Int8",)),
    _r("S16_TO_S32", "compute", "cvt", "OpSConvert %s32 (OpCompositeExtract %s16 %v2s16 0)", "sext i16 to i32", caps=("Int16",), verified=True),
    _r("U16_TO_U32", "compute", "cvt", "OpUConvert %u32 (OpCompositeExtract %u16 %v2u16 0)", "zext i16 to i32", caps=("Int16",)),
    _r("V2S8_TO_V2F16", "compute", "cvt", "OpConvertSToF %v2f16 %v2s8", "sitofp <2 x i8> to <2 x half>", caps=("Float16", "Int8")),
    _r("V2U8_TO_V2F16", "compute", "cvt", "OpConvertUToF %v2f16 %v2u8", "uitofp <2 x i8> to <2 x half>", caps=("Float16", "Int8")),
    _r("V2S8_TO_V2S16", "compute", "cvt", "OpSConvert %v2s16 %v2s8", "sext <2 x i8> to <2 x i16>", caps=("Int16", "Int8")),
    _r("V2U8_TO_V2U16", "compute", "cvt", "OpUConvert %v2u16 %v2u8", "zext <2 x i8> to <2 x i16>", caps=("Int16", "Int8")),

    # --- Memory, Atomics, Subgroups, Control Flow ---
    _r("LOAD", "compute", "mem", "OpLoad %u32 (OpAccessChain %pU32 %ssbo 0 %idx)", "load i32, ptr addrspace(1)", verified=True),
    _r("STORE", "compute", "mem", "OpStore (OpAccessChain %pU32 %ssbo 0 %idx) %val", "store i32 %val, ptr addrspace(1)", verified=True),
    _r("LEA_BUF_IMM", "compute", "mem", "SSBO / UBO descriptor address calculation (Pilot/Main)", "llvm.bifrost.lea.buffer", verified=True),
    _r("LEA_BUF", "compute", "mem", "Dynamic descriptor-array SSBO/UBO address calculation", "llvm.bifrost.lea.buffer (dynamic idx)", caps=("RuntimeDescriptorArray",)),
    _r("ATOM", "compute", "mem_atomic", "OpAtomicIAdd (result unused)", "atomicrmw add (no return)", verified=True),
    _r("ATOM_RETURN", "compute", "mem_atomic", "OpAtomicIAdd (result used in output)", "atomicrmw add (with return)", verified=True),
    _r("ATOM1", "compute", "mem_atomic", "OpAtomicIIncrement / OpAtomicIDecrement (result unused)", "atomicrmw add 1 (no return)"),
    _r("ATOM1_RETURN", "compute", "mem_atomic", "OpAtomicIIncrement / OpAtomicIDecrement (result used)", "atomicrmw add 1 (with return)", verified=True),
    _r("AXCHG", "compute", "mem_atomic", "OpAtomicExchange %u32 %ptr %scope %sem %val", "atomicrmw xchg", verified=True),
    _r("ACMPXCHG", "compute", "mem_atomic", "OpAtomicCompareExchange %u32 %ptr %scope %eq %neq %val %cmp", "cmpxchg", verified=True),
    _r("CLPER", "compute", "subgroup", "OpGroupNonUniformShuffle / ShuffleXor / ShuffleUp / ShuffleDown", "llvm.bifrost.clper / llvm.bifrost.subgroup.shuffle*", caps=("GroupNonUniform", "GroupNonUniformShuffle"), verified=True),
    _r("CLPER_OR", "compute", "subgroup", "OpBitwiseOr (OpGroupNonUniformShuffle %a %lane) %b", "or (llvm.bifrost.clper ...), %b", caps=("GroupNonUniform", "GroupNonUniformShuffle")),
    _r("WMASK", "compute", "subgroup", "OpGroupNonUniformBallot %v4u32 Subgroup %pred", "llvm.bifrost.wmask", caps=("GroupNonUniform", "GroupNonUniformBallot"), verified=True),
    _r("BARRIER", "compute", "sync", "OpControlBarrier Workgroup Workgroup AcquireRelease|WorkgroupMemory with Workgroup shared array", "llvm.bifrost.barrier", constraints="LocalSize >= 32 (Valhall warp size is 16; <=16 elides BARRIER)"),
    _r("BRANCH", "compute", "cf", "OpSelectionMerge + OpBranchConditional on thread-varying predicate", "br i1 %cond, label %t, label %f", verified=True),
    _r("NOP", "compute", "cf", "Shader terminator (NOP.end) or scoreboard wait barrier", "Valhall terminator / wait slot", verified=True),

    # --- Textures & Storage Images (Compute + Graphics) ---
    _r("TEX_SINGLE", "compute", "tex", "OpImageSampleExplicitLod %v4f32 %sampled_img %uv Lod %cf0", "llvm.bifrost.texture", verified=True),
    _r("TEX_FETCH", "compute", "tex", "OpImageFetch %v4f32 %img %icoord Lod %c0", "llvm.bifrost.texture (fetch)", verified=True),
    _r("TEX_GATHER", "compute", "tex", "OpImageGather %v4f32 %sampled_img %uv %comp", "llvm.bifrost.texture (gather)", verified=True),
    _r("TEX_GRADIENT", "graphics", "tex", "Fragment OpImageSampleImplicitLod with explicit LOD/bias or non-uniform CF", "llvm.bifrost.texture (gradient)", constraints="Fragment stage"),
    _r("TEX_DUAL", "graphics", "tex", "Paired OpImageSampleImplicitLod on two textures with shared UV", "llvm.bifrost.dual.tex.frag", constraints="Fragment stage"),
    _r("LD_TEX_IMM", "compute", "tex", "OpImageRead %v4u32 %storage_img %icoord", "llvm.bifrost.ld.attr.tex", verified=True),
    _r("LD_TEX", "compute", "tex", "OpImageRead on dynamically indexed storage image array", "llvm.bifrost.ld.attr.tex (dynamic idx)", caps=("RuntimeDescriptorArray",)),
    _r("LEA_TEX_IMM", "compute", "tex", "OpImageWrite %storage_img %icoord %texel (address + ST_CVT)", "llvm.bifrost.lea.attr.tex.*", verified=True),
    _r("LEA_TEX", "compute", "tex", "OpImageWrite on dynamically indexed storage image array", "llvm.bifrost.lea.attr.tex.* (dynamic idx)", caps=("RuntimeDescriptorArray",)),
    _r("ST_CVT", "compute", "mem", "OpImageWrite %storage_img %icoord %texel (or Vertex gl_Position/XFB store)", "llvm.bifrost.st.cvt", verified=True),
    _r("LD_CVT", "graphics", "mem", "Typed format-converting load (buffer/attribute conversion)", "llvm.bifrost.ld.cvt"),

    # --- Graphics Stage-Specific (Vertex & Fragment) ---
    _r("CUBEFACE1", "graphics", "fp_cube", "OpImageSampleExplicitLod / ImplicitLod on OpTypeImage Cube %v3f32", "llvm.bifrost.cubemap", constraints="Cube sampled image"),
    _r("CUBEFACE2", "graphics", "fp_cube", "OpImageSampleExplicitLod / ImplicitLod on OpTypeImage Cube %v3f32", "llvm.bifrost.cubemap", constraints="Cube sampled image"),
    _r("CUBE_SSEL", "graphics", "fp_cube", "OpImageSampleExplicitLod / ImplicitLod on OpTypeImage Cube %v3f32", "llvm.bifrost.cubemap", constraints="Cube sampled image"),
    _r("CUBE_TSEL", "graphics", "fp_cube", "OpImageSampleExplicitLod / ImplicitLod on OpTypeImage Cube %v3f32", "llvm.bifrost.cubemap", constraints="Cube sampled image"),
    _r("LD_ATTR_IMM", "graphics", "vert", "Vertex OpLoad from Input location N", "llvm.bifrost.ld.attr", constraints="Vertex stage", verified=True),
    _r("LD_ATTR", "graphics", "vert", "Vertex dynamically indexed Input attribute array", "llvm.bifrost.ld.attr (dynamic)", constraints="Vertex stage"),
    _r("LEA_ATTR_IMM", "graphics", "vert", "Vertex Output / Position FIFO store address", "llvm.bifrost.lea.attr.*", constraints="Vertex stage", verified=True),
    _r("LEA_ATTR", "graphics", "vert", "Vertex dynamically indexed Output attribute array", "llvm.bifrost.lea.attr.* (dynamic)", constraints="Vertex stage"),
    _r("LD_VAR_IMM", "graphics", "frag", "Fragment OpLoad from perspective-interpolated Input varying", "llvm.bifrost.ld.var", constraints="Fragment stage", verified=True),
    _r("LD_VAR", "graphics", "frag", "Fragment dynamically indexed perspective varying array", "llvm.bifrost.ld.var (dynamic)", constraints="Fragment stage"),
    _r("LD_VAR_FLAT_IMM", "graphics", "frag", "Fragment OpLoad from Flat-decorated Input varying", "llvm.bifrost.ld.var.flat", constraints="Fragment stage", verified=True),
    _r("LD_VAR_FLAT", "graphics", "frag", "Fragment dynamically indexed Flat varying array", "llvm.bifrost.ld.var.flat (dynamic)", constraints="Fragment stage"),
    _r("LD_VAR_SPECIAL", "graphics", "frag", "Fragment OpLoad from BuiltIn FragCoord / PointCoord", "llvm.bifrost.ld.var.sys", constraints="Fragment stage", verified=True),
    _r("LD_VAR_BUF_IMM", "graphics", "frag", "Fragment varying load via varying buffer (IDVS/buf mode)", "llvm.bifrost.ld.var.buffer", constraints="Fragment stage"),
    _r("LD_VAR_BUF", "graphics", "frag", "Fragment dynamic varying load via varying buffer", "llvm.bifrost.ld.var.buffer", constraints="Fragment stage"),
    _r("LD_VAR_BUF_FLAT_IMM", "graphics", "frag", "Fragment flat varying load via varying buffer", "llvm.bifrost.ld.var.buffer", constraints="Fragment stage"),
    _r("LD_VAR_BUF_FLAT", "graphics", "frag", "Fragment dynamic flat varying load via varying buffer", "llvm.bifrost.ld.var.buffer", constraints="Fragment stage"),
    _r("VAR_TEX_SINGLE", "graphics", "frag_tex", "Fragment OpImageSampleImplicitLod directly on Input varying UV", "llvm.bifrost.var.tex", constraints="Fragment stage (fused varying+tex)"),
    _r("VAR_TEX_GRADIENT", "graphics", "frag_tex", "Fragment OpImageSampleExplicitLod/Grad directly on Input varying UV", "llvm.bifrost.var.tex", constraints="Fragment stage"),
    _r("VAR_TEX_GATHER", "graphics", "frag_tex", "Fragment OpImageGather directly on Input varying UV", "llvm.bifrost.var.tex", constraints="Fragment stage"),
    _r("VAR_TEX_DUAL", "graphics", "frag_tex", "Fragment paired texture sample directly on Input varying UV", "llvm.bifrost.var.tex", constraints="Fragment stage"),
    _r("VAR_TEX_BUF_SINGLE", "graphics", "frag_tex", "Fragment VAR_TEX_SINGLE in varying-buffer mode", "llvm.bifrost.var.tex", constraints="Fragment stage"),
    _r("VAR_TEX_BUF_GRADIENT", "graphics", "frag_tex", "Fragment VAR_TEX_GRADIENT in varying-buffer mode", "llvm.bifrost.var.tex", constraints="Fragment stage"),
    _r("VAR_TEX_BUF_GATHER", "graphics", "frag_tex", "Fragment VAR_TEX_GATHER in varying-buffer mode", "llvm.bifrost.var.tex", constraints="Fragment stage"),
    _r("VAR_TEX_BUF_DUAL", "graphics", "frag_tex", "Fragment VAR_TEX_DUAL in varying-buffer mode", "llvm.bifrost.var.tex", constraints="Fragment stage"),
    _r("ATEST", "graphics", "frag_out", "Fragment shader early/late alpha-coverage test before BLEND", "llvm.bifrost.atest", constraints="Fragment stage", verified=True),
    _r("BLEND", "graphics", "frag_out", "Fragment OpStore to Output color Location 0", "llvm.bifrost.blend / blend.opaque", constraints="Fragment stage", verified=True),
    _r("DISCARD", "graphics", "frag_out", "Fragment OpKill / OpDemoteToHelperInvocation", "llvm.bifrost.discard", constraints="Fragment stage", verified=True),
    _r("ZS_EMIT", "graphics", "frag_out", "Fragment OpStore to BuiltIn FragDepth / FragStencilRefEXT", "llvm.bifrost.zs.emit / z.emit / s.emit", constraints="Fragment stage"),
    _r("LD_TILE", "graphics", "frag_tile", "Fragment SubpassInput / tilebuffer read (OpImageRead InputAttachment)", "llvm.bifrost.ld.tile", constraints="Fragment stage + InputAttachment"),
    _r("ST_TILE", "graphics", "frag_tile", "Fragment explicit tilebuffer write (blend shader)", "llvm.bifrost.st.tile", constraints="Fragment blend shader"),

    # --- Internal / Pilot / Hardware-Specific / Legacy / Non-Standard-SPIR-V ---
    _r("LD_PKA", "internal", "pilot_mem", "Uniform/UBO indexed load emitted by Mali backend (packetized address)", "llvm.bifrost.ld.uniform", constraints="Backend-selected on large UBO arrays (seen in df118580)"),
    _r("LEA_PKA", "internal", "pilot_mem", "Packetized buffer address calculation", "Internal backend addressing"),
    _r("LD_CHECKED", "internal", "mem_checked", "Bounds-checked load (robustBufferAccess2 / driver robustness)", "llvm.bifrost.ld.checked", constraints="Requires MALI_DFC_ENABLE_ROBUSTNESS"),
    _r("LD_CHECKED_IMM", "internal", "mem_checked", "Bounds-checked immediate-offset load", "llvm.bifrost.ld.checked", constraints="Requires MALI_DFC_ENABLE_ROBUSTNESS"),
    _r("ST_CHECKED", "internal", "mem_checked", "Bounds-checked store", "llvm.bifrost.st.checked", constraints="Requires MALI_DFC_ENABLE_ROBUSTNESS"),
    _r("ST_CHECKED_IMM", "internal", "mem_checked", "Bounds-checked immediate-offset store", "llvm.bifrost.st.checked", constraints="Requires MALI_DFC_ENABLE_ROBUSTNESS"),
    _r("LD_MULTI", "internal", "mem_multi", "Multi-descriptor load (driver internal / cooperative/ray)", "Internal backend"),
    _r("ST_MULTI", "internal", "mem_multi", "Multi-descriptor store (driver internal)", "Internal backend"),
    _r("SEG_ADD", "internal", "mem_seg", "TLS/WLS segment-relative address add (large Function/Workgroup stack)", "llvm.bifrost.{private,local}.segment.base.address"),
    _r("SEG_SUB", "internal", "mem_seg", "TLS/WLS segment-relative address sub", "Internal segment arithmetic"),
    _r("JUMP", "internal", "cf_indirect", "Indirect jump / switch table jump", "llvm.bifrost.jump.arm"),
    _r("JUMP_EX", "internal", "cf_indirect", "Indirect jump to external shader (blend/ray-tracing callable)", "llvm.bifrost.jump.ex.arm"),
    _r("DOORBELL", "internal", "sys", "CSF firmware doorbell notification", "Internal hardware sync"),
    _r("EUREKA", "internal", "sys", "Hardware event/debug signal", "llvm.bifrost.eureka"),
    _r("KABOOM", "internal", "sys", "Hardware trap / abort instruction", "llvm.bifrost.kaboom"),
    _r("LD_GCLK", "internal", "sys", "GPU cycle counter (SPV_KHR_shader_clock OpReadClockKHR)", "llvm.bifrost.cycle.counter / system.timestamp", caps=("ShaderClockKHR",), exts=("SPV_KHR_shader_clock",)),
    _r("FLUSH", "internal", "fp_internal", "Explicit denormal flush-to-zero instruction", "llvm.bifrost.flush.denormal"),
    _r("NOT_OLD", "internal", "legacy", "Legacy Valhall v9/Bifrost NOT encoding alias", "Superseded by NOT on v10+"),
    _r("ICMP_MULTI", "internal", "cmp_internal", "Multi-predicate integer compare (backend-internal)", "Internal Valhall ISel"),
    _r("TEX_WEIGHTS", "internal", "tex_internal", "Texture filter weight query (ARM vendor extension)", "ARM internal texture extension"),
    _r("FRCP_APPROX", "internal", "fp_approx", "Low-precision reciprocal (OpenCL half_recip / fast-math)", "llvm.bifrost.half.recip / recip.fast"),
    _r("FRSQ_APPROX", "internal", "fp_approx", "Low-precision rsqrt (OpenCL half_rsqrt / fast-math)", "llvm.bifrost.half.rsqrt / rsqrt.fast"),
    _r("FRCBRT_APPROX_A", "internal", "fp_approx", "Cube-root approximation polynomial stage A (OpenCL cbrt)", "llvm.bifrost.cbrt"),
    _r("FRCBRT_APPROX_B", "internal", "fp_approx", "Cube-root approximation polynomial stage B (OpenCL cbrt)", "llvm.bifrost.cbrt"),
    _r("FRCBRT_APPROX_C", "internal", "fp_approx", "Cube-root approximation polynomial stage C (OpenCL cbrt)", "llvm.bifrost.cbrt"),
    _r("FEXPF", "internal", "fp_approx", "Fast exp2 helper (OpenCL half_exp2)", "llvm.bifrost.half.exp2 / exp2.fast"),
    _r("FEXP_TABLE", "internal", "fp_approx", "Legacy/alternate exp LUT (G710 uses FEXP)", "Internal SFU table"),
    _r("FLOG", "internal", "fp_approx", "Alternate log SFU (G710 uses FREXPE.log + FLOGD + FADD_LSCALE)", "llvm.bifrost.log.fast"),
    _r("FLOG_TABLE", "internal", "fp_approx", "Legacy/alternate log LUT", "Internal SFU table"),
    _r("FPOW", "internal", "fp_approx", "Direct pow SFU (G710 expands GLSL.Pow to FLOGD+FMUL_RSCALE+FEXP)", "llvm.bifrost.powr"),
    _r("FSINCOS", "internal", "fp_approx", "Combined sincos SFU (G710 uses FSIN_TABLE/FCOS_TABLE)", "llvm.bifrost.sinpi / cospi"),
    _r("FMUL_CSLICE", "internal", "fp_approx", "Trig cosine slice multiply (high-precision OpenCL trig)", "llvm.bifrost.cos.gles"),
    _r("FADD_RSCALE", "internal", "fp_approx", "Exponent-rescaled add (OpenCL math library)", "llvm.bifrost.private.fadd.rscale"),
    _r("FMA_RSCALE", "internal", "fp_approx", "Exponent-rescaled FMA (OpenCL math library)", "Internal math library"),
)

CATALOG_BY_MNEMONIC: Dict[str, ValhallIselRule] = {r.mnemonic: r for r in VALHALL_ISEL_CATALOG}


def get_rule_for_mnemonic(mnemonic: str) -> Optional[ValhallIselRule]:
    """Return the `ValhallIselRule` for a Valhall mnemonic (case-insensitive)."""
    return CATALOG_BY_MNEMONIC.get(mnemonic.upper())


def list_mnemonics_by_tier(tier: str) -> List[str]:
    """Return sorted mnemonics belonging to `tier` ('compute', 'graphics', 'internal')."""
    return sorted(r.mnemonic for r in VALHALL_ISEL_CATALOG if r.tier == tier)


def audit_lowering_spec_coverage(
    lowering_json_path: str | Path = "specs/lowering/valhall_lowering.json",
) -> Dict[str, List[str]]:
    """Compare `VALHALL_ISEL_CATALOG` against every unique mnemonic in `valhall_lowering.json`."""
    with open(lowering_json_path, encoding="utf-8") as f:
        spec = json.load(f)
    spec_mnems = {r["mnemonic"] for r in spec["rules"]}
    cat_mnems = set(CATALOG_BY_MNEMONIC.keys())
    return {
        "spec_total": sorted(spec_mnems),
        "missing_from_catalog": sorted(spec_mnems - cat_mnems),
        "extra_in_catalog": sorted(cat_mnems - spec_mnems),
    }

"""
Declarative SPIR-V & Valhall Idiom Catalog for Greedy Corpus Synthesis.

Each `OpSpec` emits 1–2 typed SPIR-V instructions into a `SpirvComputeBuilder`
using values drawn from the live typed pool `pool[ValType] -> [id, ...]`,
returning the newly produced `(result_type, result_id)`.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Dict, List, Tuple

from tools.corpus_synth import spv_builder as spv


@dataclasses.dataclass(frozen=True)
class OpSpec:
    """One synthesizable SPIR-V operation or multi-op Valhall lowering idiom."""

    name: str
    out_type: str  # "U32", "S32", "F32", "Bool"
    in_types: Tuple[str, ...]
    emit_fn: Callable[[spv.SpirvComputeBuilder, List[int]], int]
    tags: Tuple[str, ...] = ()


def _un_core(op: int, out_t_attr: str) -> Callable[[spv.SpirvComputeBuilder, List[int]], int]:
    def fn(b: spv.SpirvComputeBuilder, args: List[int]) -> int:
        res = b.alloc_id()
        b.emit(op, getattr(b, out_t_attr), res, args[0])
        return res
    return fn


def _bin_core(op: int, out_t_attr: str) -> Callable[[spv.SpirvComputeBuilder, List[int]], int]:
    def fn(b: spv.SpirvComputeBuilder, args: List[int]) -> int:
        res = b.alloc_id()
        b.emit(op, getattr(b, out_t_attr), res, args[0], args[1])
        return res
    return fn


def _tri_core(op: int, out_t_attr: str) -> Callable[[spv.SpirvComputeBuilder, List[int]], int]:
    def fn(b: spv.SpirvComputeBuilder, args: List[int]) -> int:
        res = b.alloc_id()
        b.emit(op, getattr(b, out_t_attr), res, args[0], args[1], args[2])
        return res
    return fn


def _glsl_ext(ext_id: int, out_t_attr: str) -> Callable[[spv.SpirvComputeBuilder, List[int]], int]:
    def fn(b: spv.SpirvComputeBuilder, args: List[int]) -> int:
        res = b.alloc_id()
        b.emit(spv.OP_EXT_INST, getattr(b, out_t_attr), res, b.glsl_ext, ext_id, *args)
        return res
    return fn


def _emit_bfe(signed: bool) -> Callable[[spv.SpirvComputeBuilder, List[int]], int]:
    op = spv.OP_BITFIELD_S_EXTRACT if signed else spv.OP_BITFIELD_U_EXTRACT
    t_attr = "t_s32" if signed else "t_u32"

    def fn(b: spv.SpirvComputeBuilder, args: List[int]) -> int:
        res = b.alloc_id()
        off = b.get_const_u32(5)
        cnt = b.get_const_u32(11)
        b.emit(op, getattr(b, t_attr), res, args[0], off, cnt)
        return res
    return fn


def _emit_bfi(b: spv.SpirvComputeBuilder, args: List[int]) -> int:
    res = b.alloc_id()
    off = b.get_const_u32(4)
    cnt = b.get_const_u32(8)
    b.emit(spv.OP_BITFIELD_INSERT, b.t_u32, res, args[0], args[1], off, cnt)
    return res


def _emit_pack_half2x16(b: spv.SpirvComputeBuilder, args: List[int]) -> int:
    vec = b.alloc_id()
    res = b.alloc_id()
    b.emit(spv.OP_COMPOSITE_CONSTRUCT, b.t_v2f32, vec, args[0], args[1])
    b.emit(spv.OP_EXT_INST, b.t_u32, res, b.glsl_ext, spv.GLSL_PACK_HALF2X16, vec)
    return res


def _emit_unpack_half2x16(b: spv.SpirvComputeBuilder, args: List[int]) -> int:
    vec = b.alloc_id()
    c0 = b.alloc_id()
    c1 = b.alloc_id()
    res = b.alloc_id()
    b.emit(spv.OP_EXT_INST, b.t_v2f32, vec, b.glsl_ext, spv.GLSL_UNPACK_HALF2X16, args[0])
    b.emit(spv.OP_COMPOSITE_EXTRACT, b.t_f32, c0, vec, 0)
    b.emit(spv.OP_COMPOSITE_EXTRACT, b.t_f32, c1, vec, 1)
    b.emit(spv.OP_FADD, b.t_f32, res, c0, c1)
    return res


def _emit_safe_udiv(op: int) -> Callable[[spv.SpirvComputeBuilder, List[int]], int]:
    def fn(b: spv.SpirvComputeBuilder, args: List[int]) -> int:
        c1 = b.get_const_u32(1)
        denom = b.alloc_id()
        res = b.alloc_id()
        b.emit(spv.OP_BITWISE_OR, b.t_u32, denom, args[1], c1)
        b.emit(op, b.t_u32, res, args[0], denom)
        return res
    return fn


def _emit_safe_sdiv(op: int) -> Callable[[spv.SpirvComputeBuilder, List[int]], int]:
    def fn(b: spv.SpirvComputeBuilder, args: List[int]) -> int:
        c1 = b.get_const_s32(1)
        denom = b.alloc_id()
        res = b.alloc_id()
        b.emit(spv.OP_BITWISE_OR, b.t_s32, denom, args[1], c1)
        b.emit(op, b.t_s32, res, args[0], denom)
        return res
    return fn


OP_CATALOG: List[OpSpec] = [
    # Integer arithmetic & bitwise
    OpSpec("OpIAdd", "U32", ("U32", "U32"), _bin_core(spv.OP_IADD, "t_u32"), ("int", "alu")),
    OpSpec("OpISub", "U32", ("U32", "U32"), _bin_core(spv.OP_ISUB, "t_u32"), ("int", "alu")),
    OpSpec("OpIMul", "U32", ("U32", "U32"), _bin_core(spv.OP_IMUL, "t_u32"), ("int", "mul")),
    OpSpec("OpSNegate", "S32", ("S32",), _un_core(spv.OP_SNEGATE, "t_s32"), ("int",)),
    OpSpec("OpUDiv", "U32", ("U32", "U32"), _emit_safe_udiv(spv.OP_UDIV), ("int", "div")),
    OpSpec("OpUMod", "U32", ("U32", "U32"), _emit_safe_udiv(spv.OP_UMOD), ("int", "div")),
    OpSpec("OpSDiv", "S32", ("S32", "S32"), _emit_safe_sdiv(spv.OP_SDIV), ("int", "div")),
    OpSpec("OpSRem", "S32", ("S32", "S32"), _emit_safe_sdiv(spv.OP_SREM), ("int", "div")),
    OpSpec("OpSMod", "S32", ("S32", "S32"), _emit_safe_sdiv(spv.OP_SMOD), ("int", "div")),
    OpSpec("OpBitwiseAnd", "U32", ("U32", "U32"), _bin_core(spv.OP_BITWISE_AND, "t_u32"), ("bit",)),
    OpSpec("OpBitwiseOr", "U32", ("U32", "U32"), _bin_core(spv.OP_BITWISE_OR, "t_u32"), ("bit",)),
    OpSpec("OpBitwiseXor", "U32", ("U32", "U32"), _bin_core(spv.OP_BITWISE_XOR, "t_u32"), ("bit",)),
    OpSpec("OpNot", "U32", ("U32",), _un_core(spv.OP_NOT, "t_u32"), ("bit",)),
    OpSpec("OpShiftLeftLogical", "U32", ("U32", "U32"), _bin_core(spv.OP_SHIFT_LEFT_LOGICAL, "t_u32"), ("shift",)),
    OpSpec("OpShiftRightLogical", "U32", ("U32", "U32"), _bin_core(spv.OP_SHIFT_RIGHT_LOGICAL, "t_u32"), ("shift",)),
    OpSpec("OpShiftRightArithmetic", "S32", ("S32", "U32"), _bin_core(spv.OP_SHIFT_RIGHT_ARITHMETIC, "t_s32"), ("shift",)),
    OpSpec("OpBitFieldUExtract", "U32", ("U32",), _emit_bfe(False), ("bitfield",)),
    OpSpec("OpBitFieldSExtract", "S32", ("S32",), _emit_bfe(True), ("bitfield",)),
    OpSpec("OpBitFieldInsert", "U32", ("U32", "U32"), _emit_bfi, ("bitfield",)),
    OpSpec("OpBitReverse", "U32", ("U32",), _un_core(spv.OP_BIT_REVERSE, "t_u32"), ("bit",)),
    OpSpec("OpBitCount", "U32", ("U32",), _un_core(spv.OP_BIT_COUNT, "t_u32"), ("bit",)),
    OpSpec("GLSL.SAbs", "S32", ("S32",), _glsl_ext(spv.GLSL_SABS, "t_s32"), ("int",)),
    OpSpec("GLSL.UMin", "U32", ("U32", "U32"), _glsl_ext(spv.GLSL_UMIN, "t_u32"), ("int", "minmax")),
    OpSpec("GLSL.UMax", "U32", ("U32", "U32"), _glsl_ext(spv.GLSL_UMAX, "t_u32"), ("int", "minmax")),
    OpSpec("GLSL.SMin", "S32", ("S32", "S32"), _glsl_ext(spv.GLSL_SMIN, "t_s32"), ("int", "minmax")),
    OpSpec("GLSL.SMax", "S32", ("S32", "S32"), _glsl_ext(spv.GLSL_SMAX, "t_s32"), ("int", "minmax")),
    OpSpec("GLSL.FindUMsb", "U32", ("U32",), _glsl_ext(spv.GLSL_FIND_UMSB, "t_u32"), ("bit", "clz")),
    OpSpec("GLSL.FindSMsb", "S32", ("S32",), _glsl_ext(spv.GLSL_FIND_SMSB, "t_s32"), ("bit", "clz")),

    # Comparisons, logic & selects
    OpSpec("OpIEqual", "Bool", ("U32", "U32"), _bin_core(spv.OP_IEQUAL, "t_bool"), ("cmp",)),
    OpSpec("OpULessThan", "Bool", ("U32", "U32"), _bin_core(spv.OP_ULESS_THAN, "t_bool"), ("cmp",)),
    OpSpec("OpSLessThan", "Bool", ("S32", "S32"), _bin_core(spv.OP_SLESS_THAN, "t_bool"), ("cmp",)),
    OpSpec("OpFOrdLessThan", "Bool", ("F32", "F32"), _bin_core(spv.OP_FORD_LESS_THAN, "t_bool"), ("fcmp",)),
    OpSpec("OpFOrdEqual", "Bool", ("F32", "F32"), _bin_core(spv.OP_FORD_EQUAL, "t_bool"), ("fcmp",)),
    OpSpec("OpIsNan", "Bool", ("F32",), _un_core(spv.OP_IS_NAN, "t_bool"), ("fcmp",)),
    OpSpec("OpIsInf", "Bool", ("F32",), _un_core(spv.OP_IS_INF, "t_bool"), ("fcmp",)),
    OpSpec("OpSelectU32", "U32", ("Bool", "U32", "U32"), _tri_core(spv.OP_SELECT, "t_u32"), ("csel",)),
    OpSpec("OpSelectF32", "F32", ("Bool", "F32", "F32"), _tri_core(spv.OP_SELECT, "t_f32"), ("csel",)),

    # Floating-point, transcendentals, conversions & f16 packing
    OpSpec("OpFAdd", "F32", ("F32", "F32"), _bin_core(spv.OP_FADD, "t_f32"), ("fp",)),
    OpSpec("OpFSub", "F32", ("F32", "F32"), _bin_core(spv.OP_FSUB, "t_f32"), ("fp",)),
    OpSpec("OpFMul", "F32", ("F32", "F32"), _bin_core(spv.OP_FMUL, "t_f32"), ("fp",)),
    OpSpec("OpFDiv", "F32", ("F32", "F32"), _bin_core(spv.OP_FDIV, "t_f32"), ("fp", "sfu")),
    OpSpec("OpFNegate", "F32", ("F32",), _un_core(spv.OP_FNEGATE, "t_f32"), ("fp",)),
    OpSpec("GLSL.FAbs", "F32", ("F32",), _glsl_ext(spv.GLSL_FABS, "t_f32"), ("fp",)),
    OpSpec("GLSL.Floor", "F32", ("F32",), _glsl_ext(spv.GLSL_FLOOR, "t_f32"), ("fp", "round")),
    OpSpec("GLSL.Ceil", "F32", ("F32",), _glsl_ext(spv.GLSL_CEIL, "t_f32"), ("fp", "round")),
    OpSpec("GLSL.Trunc", "F32", ("F32",), _glsl_ext(spv.GLSL_TRUNC, "t_f32"), ("fp", "round")),
    OpSpec("GLSL.RoundEven", "F32", ("F32",), _glsl_ext(spv.GLSL_ROUND_EVEN, "t_f32"), ("fp", "round")),
    OpSpec("GLSL.Sqrt", "F32", ("F32",), _glsl_ext(spv.GLSL_SQRT, "t_f32"), ("fp", "sfu")),
    OpSpec("GLSL.InverseSqrt", "F32", ("F32",), _glsl_ext(spv.GLSL_INVERSE_SQRT, "t_f32"), ("fp", "sfu")),
    OpSpec("GLSL.Sin", "F32", ("F32",), _glsl_ext(spv.GLSL_SIN, "t_f32"), ("fp", "sfu")),
    OpSpec("GLSL.Cos", "F32", ("F32",), _glsl_ext(spv.GLSL_COS, "t_f32"), ("fp", "sfu")),
    OpSpec("GLSL.Exp2", "F32", ("F32",), _glsl_ext(spv.GLSL_EXP2, "t_f32"), ("fp", "sfu")),
    OpSpec("GLSL.Log2", "F32", ("F32",), _glsl_ext(spv.GLSL_LOG2, "t_f32"), ("fp", "sfu")),
    OpSpec("GLSL.FMin", "F32", ("F32", "F32"), _glsl_ext(spv.GLSL_FMIN, "t_f32"), ("fp", "minmax")),
    OpSpec("GLSL.FMax", "F32", ("F32", "F32"), _glsl_ext(spv.GLSL_FMAX, "t_f32"), ("fp", "minmax")),
    OpSpec("GLSL.FClamp", "F32", ("F32", "F32", "F32"), _glsl_ext(spv.GLSL_FCLAMP, "t_f32"), ("fp", "minmax")),
    OpSpec("GLSL.Fma", "F32", ("F32", "F32", "F32"), _glsl_ext(spv.GLSL_FMA, "t_f32"), ("fp", "fma")),
    OpSpec("GLSL.FMix", "F32", ("F32", "F32", "F32"), _glsl_ext(spv.GLSL_FMIX, "t_f32"), ("fp", "fma")),
    OpSpec("GLSL.PackHalf2x16", "U32", ("F32", "F32"), _emit_pack_half2x16, ("f16", "swz")),
    OpSpec("GLSL.UnpackHalf2x16", "F32", ("U32",), _emit_unpack_half2x16, ("f16", "swz")),
    OpSpec("OpConvertFToS", "S32", ("F32",), _un_core(spv.OP_CONVERT_F_TO_S, "t_s32"), ("conv",)),
    OpSpec("OpConvertFToU", "U32", ("F32",), _un_core(spv.OP_CONVERT_F_TO_U, "t_u32"), ("conv",)),
    OpSpec("OpConvertSToF", "F32", ("S32",), _un_core(spv.OP_CONVERT_S_TO_F, "t_f32"), ("conv",)),
    OpSpec("OpConvertUToF", "F32", ("U32",), _un_core(spv.OP_CONVERT_U_TO_F, "t_f32"), ("conv",)),
]

CATALOG_BY_NAME: Dict[str, OpSpec] = {spec.name: spec for spec in OP_CATALOG}

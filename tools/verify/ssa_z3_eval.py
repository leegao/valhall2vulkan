#!/usr/bin/env python3
"""
Direct Valhall SSA-IR Z3 Evaluator (Task 3)
===========================================
Translates a `SymbolicSsaModule` (emitted by `LoweringSemanticsExtractor` in
Task 2) directly into Z3 QF_BV / QF_FP / QF_ABV expressions over `ValhallZ3State`.

Supports all 134 `tables::SsaOp` operations (scalar, 64-bit, V2/V3/V4 vectors,
tuples, IEEE-754 F32/FP16 pack/unpack, comparisons, conversions, subgroup,
control-flow, and memory/message side effects).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import z3

from tools.verify.clang_ast_evaluator import SymbolicSsaModule, SymbolicSsaNode
from tools.verify.panfrost_z3_model import (
    BV8,
    BV32,
    BV64,
    FP32_SORT,
    ValhallZ3State,
    _bitrev_bv32,
    _clz_bv,
    _fp_to_int_sat,
    _fp_total_le_bv,
    _hw_uf,
    _popcount_bv32,
    bv32,
    bv64,
    bv32_to_fp32,
    f16_bv_to_f32_bv,
    f32_bv_to_f16_bv,
    fp32_to_bv32,
)


Z3Val = Union[z3.ExprRef, Tuple[Any, ...]]


@dataclasses.dataclass
class SsaZ3EvalResult:
    """Result of evaluating a SymbolicSsaModule in Z3."""

    values: Dict[int, Z3Val]
    reg_writes: Dict[int, z3.BitVecRef]
    final_state: ValhallZ3State
    branch_cond: Optional[z3.BoolRef] = None
    trace_notes: List[str] = dataclasses.field(default_factory=list)


def _as_bv32(v: Z3Val) -> z3.BitVecRef:
    if isinstance(v, z3.BoolRef):
        return z3.If(v, bv32(1), bv32(0))
    if isinstance(v, z3.FPRef):
        return fp32_to_bv32(v)
    assert isinstance(v, z3.BitVecRef)
    if v.size() == 32:
        return v
    if v.size() < 32:
        return z3.ZeroExt(32 - v.size(), v)
    return z3.Extract(31, 0, v)


def _as_bv64(v: Z3Val) -> z3.BitVecRef:
    if isinstance(v, z3.BitVecRef):
        if v.size() == 64:
            return v
        if v.size() < 64:
            return z3.ZeroExt(64 - v.size(), v)
        return z3.Extract(63, 0, v)
    return z3.ZeroExt(32, _as_bv32(v))


def _as_bool(v: Z3Val) -> z3.BoolRef:
    if isinstance(v, z3.BoolRef):
        return v
    if isinstance(v, z3.BitVecRef):
        return v != z3.BitVecVal(0, v.size())
    raise TypeError(f"Cannot convert {type(v)} to BoolRef")


def _is_vec_val(v: Any) -> bool:
    return isinstance(v, tuple)


def _vec_len(val_type: str) -> int:
    if val_type.startswith("V2"):
        return 2
    if val_type.startswith("V3"):
        return 3
    if val_type.startswith("V4"):
        return 4
    return 1


def _fp_unary_bv32(op: str, x: z3.BitVecRef) -> z3.BitVecRef:
    f = bv32_to_fp32(x)
    if op == "FAbs":
        return x & bv32(0x7FFFFFFF)
    if op == "FNeg":
        return x ^ bv32(0x80000000)
    if op == "Floor":
        return fp32_to_bv32(z3.fpRoundToIntegral(z3.RTN(), f))
    if op == "Ceil":
        return fp32_to_bv32(z3.fpRoundToIntegral(z3.RTP(), f))
    if op == "Trunc":
        return fp32_to_bv32(z3.fpRoundToIntegral(z3.RTZ(), f))
    if op == "RoundEven":
        return fp32_to_bv32(z3.fpRoundToIntegral(z3.RNE(), f))
    if op == "Sqrt":
        return fp32_to_bv32(z3.fpSqrt(z3.RNE(), f))
    if op == "InverseSqrt":
        return _hw_uf("FRSQ_f32", BV32, BV32)(x)
    if op in ("Sin", "Cos", "Exp", "Exp2", "Log", "Log2", "Atan"):
        return _hw_uf(f"ssa_{op}_f32", BV32, BV32)(x)
    if op == "FrexpExp":
        return _hw_uf("ssa_FrexpExp_f32", BV32, BV32)(x)
    if op == "FrexpSignificand":
        return _hw_uf("ssa_FrexpSig_f32", BV32, BV32)(x)
    raise KeyError(f"Unknown FP unary op {op}")


def _fp_binary_bv32(op: str, a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BitVecRef:
    fa, fb = bv32_to_fp32(a), bv32_to_fp32(b)
    if op == "FAdd":
        return fp32_to_bv32(z3.fpAdd(z3.RNE(), fa, fb))
    if op == "FSub":
        return fp32_to_bv32(z3.fpSub(z3.RNE(), fa, fb))
    if op == "FMul":
        return fp32_to_bv32(z3.fpMul(z3.RNE(), fa, fb))
    if op == "FDiv":
        return fp32_to_bv32(z3.fpDiv(z3.RNE(), fa, fb))
    if op in ("FMin", "NMin"):
        return fp32_to_bv32(z3.fpMin(fa, fb))
    if op in ("FMax", "NMax"):
        return fp32_to_bv32(z3.fpMax(fa, fb))
    if op == "Pow":
        return _hw_uf("ssa_Pow_f32", BV32, BV32, BV32)(a, b)
    if op == "Ldexp":
        return _hw_uf("ssa_Ldexp_f32", BV32, BV32, BV32)(a, b)
    raise KeyError(f"Unknown FP binary op {op}")


def _fp_cmp(op: str, a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BoolRef:
    fa, fb = bv32_to_fp32(a), bv32_to_fp32(b)
    if op == "FCmpOrdEq":
        return z3.fpEQ(fa, fb)
    if op == "FCmpOrdGt":
        return z3.fpGT(fa, fb)
    if op == "FCmpOrdGe":
        return z3.fpGEQ(fa, fb)
    if op == "FCmpOrdLt":
        return z3.fpLT(fa, fb)
    if op == "FCmpOrdLe":
        return z3.fpLEQ(fa, fb)
    if op == "FCmpOrdNe":
        return z3.Or(z3.fpLT(fa, fb), z3.fpGT(fa, fb))
    if op == "FCmpUnordNe":
        return z3.Not(z3.fpEQ(fa, fb))
    if op == "FCmpTotal":
        return _fp_total_le_bv(32, a, b)
    raise KeyError(f"Unknown FP cmp op {op}")


def _int_cmp(op: str, a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BoolRef:
    if op == "ICmpEq":
        return a == b
    if op == "ICmpNe":
        return a != b
    if op == "ICmpSLt":
        return a < b
    if op == "ICmpSLe":
        return a <= b
    if op == "ICmpSGt":
        return a > b
    if op == "ICmpSGe":
        return a >= b
    if op == "ICmpULt":
        return z3.ULT(a, b)
    if op == "ICmpULe":
        return z3.ULE(a, b)
    if op == "ICmpUGt":
        return z3.UGT(a, b)
    if op == "ICmpUGe":
        return z3.UGE(a, b)
    raise KeyError(f"Unknown int cmp op {op}")


def _int_unary_bv32(op: str, x: z3.BitVecRef) -> z3.BitVecRef:
    if op == "BitNot":
        return ~x
    if op == "SAbs":
        return z3.If(x < bv32(0), -x, x)
    if op == "SNeg":
        return -x
    if op == "Clz":
        return _clz_bv(32, x, mask_zero=False)
    if op == "BitCount":
        return _popcount_bv32(x)
    if op == "BitReverse":
        return _bitrev_bv32(x)
    raise KeyError(f"Unknown int unary op {op}")


def _int_binary_bv32(op: str, a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BitVecRef:
    if op == "IAdd":
        return a + b
    if op == "ISub":
        return a - b
    if op == "IMul":
        return a * b
    if op == "BitAnd":
        return a & b
    if op == "BitOr":
        return a | b
    if op == "BitXor":
        return a ^ b
    if op == "Shl":
        return a << (b & bv32(31))
    if op == "ShrU":
        return z3.LShR(a, b & bv32(31))
    if op == "ShrS":
        return a >> (b & bv32(31))
    if op == "RotL":
        sh = b & bv32(31)
        return z3.RotateLeft(a, sh)
    if op == "RotR":
        sh = b & bv32(31)
        return z3.RotateRight(a, sh)
    raise KeyError(f"Unknown int binary op {op}")


def _eval_elementwise(
    node: SymbolicSsaNode, args: List[Z3Val], fn: Any
) -> Z3Val:
    vlen = _vec_len(node.val_type)
    is_vec = vlen > 1 or any(_is_vec_val(a) for a in args)
    if not is_vec:
        return fn(*args)
    if vlen == 1:
        vlen = max(len(a) for a in args if _is_vec_val(a))
    lanes = []
    for i in range(vlen):
        lane_args = [a[i] if _is_vec_val(a) else a for a in args]
        lanes.append(fn(*lane_args))
    return tuple(lanes)


def eval_ssa_module_z3(
    ssa: SymbolicSsaModule,
    raw_srcs: Optional[Sequence[z3.BitVecRef]] = None,
    dst_old: Optional[z3.BitVecRef] = None,
    dest_reg: int = 0,
    state: Optional[ValhallZ3State] = None,
    branch_offset: int = 0,
) -> SsaZ3EvalResult:
    """Directly evaluate every node of a SymbolicSsaModule in Z3."""
    st = dataclasses.replace(state) if state is not None else ValhallZ3State.symbolic()
    values: Dict[int, Z3Val] = {
        0: bv32(0),
        1: bv32(0),  # initial effect token
    }
    notes: List[str] = []

    for node in ssa.nodes[2:]:
        op = node.op
        args = [values[aid] for aid in node.args]

        if op == "InputReg":
            r = node.imm_u32 & 63
            if raw_srcs is not None and r < len(raw_srcs):
                values[node.id] = raw_srcs[r]
            elif dst_old is not None and r == (dest_reg & 63):
                values[node.id] = dst_old
            else:
                values[node.id] = z3.Select(st.regs, z3.BitVecVal(r, 6))
        elif op in ("ConstU32", "ConstS32", "ConstF32"):
            values[node.id] = bv32(node.imm_u32)
        elif op == "ConstU64":
            values[node.id] = bv64(node.imm_u64)
        elif op == "ConstBool":
            values[node.id] = z3.BoolVal(bool(node.aux_bool))
        elif op in ("Nop", "Copy", "Bitcast"):
            if args:
                v = args[0]
                if node.val_type == "U64" and isinstance(v, z3.BitVecRef) and v.size() == 32:
                    v = z3.ZeroExt(32, v)
                elif node.val_type in ("U32", "S32", "F32") and isinstance(v, z3.BitVecRef) and v.size() == 64:
                    v = z3.Extract(31, 0, v)
                values[node.id] = v
            else:
                values[node.id] = bv32(0)
        elif op == "Phi":
            values[node.id] = args[0] if args else bv32(0)

        # Vector & Tuple & 64-bit Pack/Split
        elif op in ("VecConstruct2", "VecConstruct3", "VecConstruct4"):
            values[node.id] = tuple(args)
        elif op == "VecExtract":
            vec = args[0]
            values[node.id] = vec[node.imm_u32] if isinstance(vec, tuple) else vec
        elif op == "VecShuffle":
            a, b = args[0], args[1]
            combined = list(a) + list(b)
            values[node.id] = (combined[node.imm_u32], combined[node.imm_u32_2])
        elif op == "Pack64":
            lo, hi = _as_bv32(args[0]), _as_bv32(args[1])
            values[node.id] = z3.Concat(hi, lo)
        elif op == "Split64":
            v64 = _as_bv64(args[0])
            values[node.id] = (z3.Extract(31, 0, v64), z3.Extract(63, 32, v64))
        elif op == "TupleExtract":
            tup = args[0]
            values[node.id] = tup[node.imm_u32] if isinstance(tup, tuple) else tup
        elif op == "UnpackHalf2x16":
            u = _as_bv32(args[0])
            values[node.id] = (
                f16_bv_to_f32_bv(z3.Extract(15, 0, u)),
                f16_bv_to_f32_bv(z3.Extract(31, 16, u)),
            )
        elif op == "PackHalf2x16":
            v2 = args[0]
            f0 = f32_bv_to_f16_bv(_as_bv32(v2[0]))
            f1 = f32_bv_to_f16_bv(_as_bv32(v2[1]))
            values[node.id] = z3.Concat(f1, f0)

        # 32-bit Integer / Bitwise
        elif op in ("BitNot", "SAbs", "SNeg", "Clz", "BitCount", "BitReverse"):
            values[node.id] = _eval_elementwise(node, args, lambda x: _int_unary_bv32(op, _as_bv32(x)))
        elif op in ("IAdd", "ISub", "IMul", "BitAnd", "BitOr", "BitXor", "Shl", "ShrU", "ShrS", "RotL", "RotR"):
            values[node.id] = _eval_elementwise(
                node, args, lambda x, y: _int_binary_bv32(op, _as_bv32(x), _as_bv32(y))
            )
        elif op in ("BfeU", "BfeS"):
            def _bfe(x: z3.BitVecRef, off: z3.BitVecRef, w: z3.BitVecRef) -> z3.BitVecRef:
                # Constant fast-path (vast majority of BfeU/BfeS calls in lowering)
                if isinstance(off, z3.BitVecNumRef) and isinstance(w, z3.BitVecNumRef):
                    o_i = off.as_long()
                    w_i = w.as_long()
                    if w_i == 0:
                        return bv32(0)
                    hi_i = min(31, o_i + w_i - 1)
                    sl = z3.Extract(hi_i, o_i, x)
                    ext_bits = 32 - sl.size()
                    if ext_bits == 0:
                        return sl
                    return z3.SignExt(ext_bits, sl) if op == "BfeS" else z3.ZeroExt(ext_bits, sl)
                shifted = (x >> (off & bv32(31))) if op == "BfeS" else z3.LShR(x, off & bv32(31))
                mask = (bv32(1) << (w & bv32(31))) - bv32(1)
                return shifted & mask
            values[node.id] = _eval_elementwise(node, args, _bfe)
        elif op == "Bfi":
            base, ins_v, off, w = _as_bv32(args[0]), _as_bv32(args[1]), _as_bv32(args[2]), _as_bv32(args[3])
            if isinstance(off, z3.BitVecNumRef) and isinstance(w, z3.BitVecNumRef):
                o_i, w_i = off.as_long(), w.as_long()
                if w_i >= 32:
                    values[node.id] = ins_v
                elif w_i == 0:
                    values[node.id] = base
                else:
                    mask_val = ((1 << w_i) - 1) << o_i
                    values[node.id] = (base & bv32(~mask_val)) | ((ins_v << o_i) & bv32(mask_val))
            else:
                mask = ((bv32(1) << (w & bv32(31))) - bv32(1)) << (off & bv32(31))
                values[node.id] = (base & ~mask) | ((ins_v << (off & bv32(31))) & mask)
        elif op in ("IDot4S", "IDot4SU", "IDot4U"):
            a32, b32 = _as_bv32(args[0]), _as_bv32(args[1])
            acc = bv32(0)
            for i in range(4):
                ba = z3.Extract(8 * i + 7, 8 * i, a32)
                bb = z3.Extract(8 * i + 7, 8 * i, b32)
                ea = z3.SignExt(24, ba) if op in ("IDot4S", "IDot4SU") else z3.ZeroExt(24, ba)
                eb = z3.SignExt(24, bb) if op == "IDot4S" else z3.ZeroExt(24, bb)
                acc = acc + (ea * eb)
            values[node.id] = acc

        # 64-bit Integer / Bitwise
        elif op == "BitNot64":
            values[node.id] = ~_as_bv64(args[0])
        elif op in ("IAdd64", "ISub64", "IMul64", "BitAnd64", "BitOr64", "BitXor64", "Shl64", "ShrU64", "ShrS64"):
            a64, b64 = _as_bv64(args[0]), _as_bv64(args[1])
            if op == "IAdd64":
                values[node.id] = a64 + b64
            elif op == "ISub64":
                values[node.id] = a64 - b64
            elif op == "IMul64":
                values[node.id] = a64 * b64
            elif op == "BitAnd64":
                values[node.id] = a64 & b64
            elif op == "BitOr64":
                values[node.id] = a64 | b64
            elif op == "BitXor64":
                values[node.id] = a64 ^ b64
            elif op == "Shl64":
                values[node.id] = a64 << (b64 & bv64(63))
            elif op == "ShrU64":
                values[node.id] = z3.LShR(a64, b64 & bv64(63))
            elif op == "ShrS64":
                values[node.id] = a64 >> (b64 & bv64(63))

        # Integer Comparisons & Logical & Select
        elif op.startswith("ICmp"):
            values[node.id] = _eval_elementwise(node, args, lambda x, y: _int_cmp(op, x, y))
        elif op == "LogicalAnd":
            values[node.id] = _eval_elementwise(node, args, lambda x, y: z3.And(_as_bool(x), _as_bool(y)))
        elif op == "LogicalOr":
            values[node.id] = _eval_elementwise(node, args, lambda x, y: z3.Or(_as_bool(x), _as_bool(y)))
        elif op == "LogicalNot":
            values[node.id] = _eval_elementwise(node, args, lambda x: z3.Not(_as_bool(x)))
        elif op == "Select":
            values[node.id] = _eval_elementwise(
                node, args, lambda c, t, f: z3.If(_as_bool(c), t, f)
            )

        # Floating-Point Ops
        elif op in ("FAbs", "FNeg", "Floor", "Ceil", "Trunc", "RoundEven", "Sqrt", "InverseSqrt",
                    "Sin", "Cos", "Exp", "Exp2", "Log", "Log2", "Atan", "FrexpExp", "FrexpSignificand"):
            values[node.id] = _eval_elementwise(node, args, lambda x: _fp_unary_bv32(op, _as_bv32(x)))
        elif op in ("FAdd", "FSub", "FMul", "FDiv", "FMin", "FMax", "NMin", "NMax", "Pow", "Ldexp"):
            values[node.id] = _eval_elementwise(
                node, args, lambda x, y: _fp_binary_bv32(op, _as_bv32(x), _as_bv32(y))
            )
        elif op == "FClamp":
            values[node.id] = _eval_elementwise(
                node,
                args,
                lambda x, lo, hi: fp32_to_bv32(
                    z3.fpMin(z3.fpMax(bv32_to_fp32(_as_bv32(x)), bv32_to_fp32(_as_bv32(lo))), bv32_to_fp32(_as_bv32(hi)))
                ),
            )
        elif op == "Fma":
            values[node.id] = _eval_elementwise(
                node,
                args,
                lambda a, b, c: fp32_to_bv32(
                    z3.fpFMA(z3.RNE(), bv32_to_fp32(_as_bv32(a)), bv32_to_fp32(_as_bv32(b)), bv32_to_fp32(_as_bv32(c)))
                ),
            )
        elif op == "IsNan":
            values[node.id] = _eval_elementwise(node, args, lambda x: z3.fpIsNaN(bv32_to_fp32(_as_bv32(x))))
        elif op == "IsInf":
            values[node.id] = _eval_elementwise(node, args, lambda x: z3.fpIsInf(bv32_to_fp32(_as_bv32(x))))
        elif op.startswith("FCmp"):
            values[node.id] = _eval_elementwise(node, args, lambda x, y: _fp_cmp(op, _as_bv32(x), _as_bv32(y)))

        # Conversions
        elif op == "ConvertFToS":
            values[node.id] = _eval_elementwise(
                node, args, lambda x: _fp_to_int_sat(bv32_to_fp32(_as_bv32(x)), 32, True, z3.RTZ())
            )
        elif op == "ConvertFToU":
            values[node.id] = _eval_elementwise(
                node, args, lambda x: _fp_to_int_sat(bv32_to_fp32(_as_bv32(x)), 32, False, z3.RTZ())
            )
        elif op == "ConvertSToF":
            values[node.id] = _eval_elementwise(
                node, args, lambda x: fp32_to_bv32(z3.fpSignedToFP(z3.RNE(), _as_bv32(x), FP32_SORT))
            )
        elif op == "ConvertUToF":
            values[node.id] = _eval_elementwise(
                node, args, lambda x: fp32_to_bv32(z3.fpUnsignedToFP(z3.RNE(), _as_bv32(x), FP32_SORT))
            )
        elif op == "SConvert32To64":
            values[node.id] = z3.SignExt(32, _as_bv32(args[0]))
        elif op == "UConvert32To64":
            values[node.id] = z3.ZeroExt(32, _as_bv32(args[0]))
        elif op == "UConvert64To32":
            values[node.id] = z3.Extract(31, 0, _as_bv64(args[0]))

        # Sideband, FAU, Builtins, Control Flow, Subgroup, Memory
        elif op == "LoadFau":
            values[node.id] = z3.Select(st.fau, z3.BitVecVal(node.imm_u32 & 63, 8))
        elif op == "LoadSidebandU32":
            values[node.id] = _hw_uf(f"sideband_u32_{node.imm_u32}", BV32, BV32)(bv32(node.imm_u32_2))
        elif op == "LoadSidebandU64":
            values[node.id] = _hw_uf(f"sideband_u64_{node.imm_u32}", BV32, BV64)(bv32(node.imm_u32))
        elif op == "LoadBuiltinInput":
            vlen = _vec_len(node.val_type)
            if vlen > 1:
                values[node.id] = tuple(
                    _hw_uf(f"builtin_{node.aux_str or node.imm_u32}_{i}", BV32, BV32)(z3.ZeroExt(27, st.lane_id))
                    for i in range(vlen)
                )
            else:
                values[node.id] = _hw_uf(f"builtin_{node.aux_str or node.imm_u32}", BV32, BV32)(z3.ZeroExt(27, st.lane_id))
        elif op == "Kill":
            cond = _as_bool(args[0]) if args else z3.BoolVal(True)
            st.discarded = z3.Or(st.discarded, cond)
            values[node.id] = bv32(0)
            notes.append("Kill")
        elif op == "ControlBarrier":
            st.barrier_epoch = st.barrier_epoch + bv32(1)
            values[node.id] = bv32(0)
            notes.append("ControlBarrier")
        elif op in ("Branch", "BranchCond", "Return"):
            if op == "Branch" and args:
                st.pc = _as_bv64(args[0])
            values[node.id] = bv32(0)
        elif op == "SubgroupBallot":
            c = _as_bv32(args[0])
            values[node.id] = _hw_uf("SubgroupBallot", BV32, z3.BitVecSort(5), BV32)(c, st.lane_id)
        elif op == "SubgroupShuffle":
            v, idx = _as_bv32(args[0]), _as_bv32(args[1])
            values[node.id] = _hw_uf("SubgroupShuffle", BV32, BV32, z3.BitVecSort(5), BV32)(v, idx, st.lane_id)
        elif op == "PsbLoad":
            addr = _as_bv64(args[0])
            eb = node.imm_u32 or 4
            if eb == 1:
                val = z3.ZeroExt(24, z3.Select(st.mem_global, addr))
            elif eb == 2:
                val = z3.ZeroExt(16, z3.Concat(z3.Select(st.mem_global, addr + bv64(1)), z3.Select(st.mem_global, addr)))
            else:
                val = z3.Concat(
                    z3.Select(st.mem_global, addr + bv64(3)),
                    z3.Select(st.mem_global, addr + bv64(2)),
                    z3.Select(st.mem_global, addr + bv64(1)),
                    z3.Select(st.mem_global, addr),
                )
            values[node.id] = (val, args[1] if len(args) > 1 else bv32(0))
        elif op == "PsbStore":
            addr = _as_bv64(args[0])
            val = _as_bv32(args[1])
            eb = node.imm_u32 or 4
            for j in range(min(eb, 4)):
                st.mem_global = z3.Store(st.mem_global, addr + bv64(j), z3.Extract(8 * j + 7, 8 * j, val))
            values[node.id] = args[2] if len(args) > 2 else bv32(0)
        elif op in ("AtomicRmw", "AtomicExchange", "AtomicCmpXchg"):
            addr = _as_bv64(args[0])
            arg = _as_bv32(args[1])
            uf = _hw_uf(f"ssa_{op}_{node.aux_str}", BV64, BV32, BV32)
            values[node.id] = (uf(addr, arg), args[-1] if args else bv32(0))
        elif op in ("LoadVertAttr", "LoadFragVarying"):
            uf = _hw_uf(f"ssa_{op}_{node.imm_u32}", BV32, BV32)
            values[node.id] = tuple(uf(bv32(c)) for c in range(4))
        elif op == "TexSample":
            uf = _hw_uf("ssa_TexSample", BV32, BV32, BV32)
            uv = args[0] if args else (bv32(0), bv32(0))
            u0 = _as_bv32(uv[0]) if isinstance(uv, tuple) else _as_bv32(uv)
            u1 = _as_bv32(uv[1]) if isinstance(uv, tuple) and len(uv) > 1 else bv32(0)
            values[node.id] = (tuple(uf(u0, u1) for _ in range(4)), args[-1] if args else bv32(0))
        elif op in ("StoreBuiltinOutput", "StoreFragColor", "StoreVertVarying"):
            values[node.id] = args[-1] if args else bv32(0)
        else:
            raise KeyError(f"Unhandled SsaOp '{op}' in eval_ssa_module_z3")

    reg_writes: Dict[int, z3.BitVecRef] = {}
    for r, sid in ssa.reg_writes.items():
        val = values[sid]
        reg_writes[r] = _as_bv32(val) if not isinstance(val, z3.BitVecRef) else val

    bcond = _as_bool(values[ssa.branch_cond]) if ssa.branch_cond is not None else None
    if bcond is not None:
        st.pc = z3.If(bcond, st.pc + bv64(8) + bv64(branch_offset), st.pc + bv64(8))
    return SsaZ3EvalResult(
        values=values,
        reg_writes=reg_writes,
        final_state=st,
        branch_cond=bcond,
        trace_notes=notes,
    )

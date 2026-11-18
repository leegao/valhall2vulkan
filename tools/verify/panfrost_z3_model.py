#!/usr/bin/env python3
"""
Frozen Panfrost Reference Semantics in Z3 (\Psi_pan)
===================================================
Exact operational semantics extracted from Mesa Panfrost:
  - kraid/swizzle.rs  (Swizzle, SwizzleByte, SwizzleWord, fold_u32, fold_u64)
  - kraid/ops.rs      (Foldable, PerCompFoldable, Mod, Clamp, Round, CmpOp, MuxOp)
  - kraid/encode_v9.rs (V9Instr mappings for all 627 isa-v9-v14.xml specs)

Encodes Valhall ISA instructions (v9-v14, all 201 unique mnemonics / 627 specs)
into Z3 QF_BV / QF_FP / QF_ABV expressions over ValhallZ3State.
"""

from __future__ import annotations

import dataclasses
import logging
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import z3

# ---------------------------------------------------------------------------
# Z3 Sorts & Constants
# ---------------------------------------------------------------------------
BV8 = z3.BitVecSort(8)
BV16 = z3.BitVecSort(16)
BV32 = z3.BitVecSort(32)
BV64 = z3.BitVecSort(64)
FP16_SORT = z3.FPSort(5, 11)
FP32_SORT = z3.FPSort(8, 24)


def bv32(val: int) -> z3.BitVecNumRef:
    return z3.BitVecVal(val & 0xFFFFFFFF, 32)


def bv64(val: int) -> z3.BitVecNumRef:
    return z3.BitVecVal(val & 0xFFFFFFFFFFFFFFFF, 64)


def bv16(val: int) -> z3.BitVecNumRef:
    return z3.BitVecVal(val & 0xFFFF, 16)


def bv8(val: int) -> z3.BitVecNumRef:
    return z3.BitVecVal(val & 0xFF, 8)


def _lane_width(variant: str) -> Tuple[int, int]:
    """Return (lane_bits, num_lanes) for a Valhall variant string."""
    v = variant.lower()
    if v in ("v2f16", "v2u16", "v2s16", "v2i16"):
        return (16, 2)
    if v in ("v4u8", "v4s8", "v4i8"):
        return (8, 4)
    if v in ("i64", "u64", "s64", "f64"):
        return (64, 1)
    return (32, 1)


def _is_signed_variant(variant: str) -> bool:
    v = variant.lower()
    return v in ("s8", "s16", "s32", "s64", "v2s8", "v2s16", "v4s8")


def _is_float_variant(variant: str) -> bool:
    v = variant.lower()
    return "f32" in v or "f16" in v or "f64" in v


def _infer_src_type_and_signed(mnem: str, variant: str, idx: int) -> Tuple[str, bool]:
    """Infer (src_dtype, signed_widen) matching kraid/swizzle.rs & ops.rs."""
    v = (variant or "").lower()
    if v:
        return (v, _is_signed_variant(v))
    # Empty-variant conversion / special instructions in isa-v9-v14.xml
    if mnem in ("F16_TO_F32", "F16_TO_S32", "F16_TO_U32"):
        return ("f16", False)
    if mnem in ("V2F16_TO_V2S16", "V2F16_TO_V2U16"):
        return ("v2f16", False)
    if mnem in ("F32_TO_F16", "F32_TO_S32", "F32_TO_U32", "V2F32_TO_V2F16"):
        return ("f32", False)
    if mnem in ("CUBEFACE1", "CUBEFACE2", "CUBE_SSEL", "CUBE_TSEL", "DISCARD"):
        return ("f32", False)
    if mnem in ("S8_TO_F32", "S8_TO_S32", "V2S8_TO_V2F16", "V2S8_TO_V2S16"):
        return ("s8", True)
    if mnem in ("S16_TO_F32", "S16_TO_S32", "V2S16_TO_V2F16"):
        return ("s16", True)
    if mnem in ("S32_TO_F32",):
        return ("s32", True)
    if mnem in ("U8_TO_F32", "U8_TO_U32", "V2U8_TO_V2F16", "V2U8_TO_V2U16"):
        return ("u8", False)
    if mnem in ("U16_TO_F32", "U16_TO_U32", "V2U16_TO_V2F16"):
        return ("u16", False)
    if mnem in ("U32_TO_F32",):
        return ("u32", False)
    return ("u32", False)


def _z3_rm(round_mode: str = "rte") -> Any:
    """Map Valhall/kraid FRound names to Z3 rounding modes."""
    rm = (round_mode or "rte").lower()
    if rm in ("rte", "none", "", "round_even"):
        return z3.RNE()
    if rm in ("rtp", "round_up"):
        return z3.RTP()
    if rm in ("rtn", "round_down"):
        return z3.RTN()
    if rm in ("rtz", "round_zero"):
        return z3.RTZ()
    if rm in ("rna", "round_na"):
        return z3.RNA()
    raise ValueError(f"Unsupported rounding mode '{round_mode}'")


def bv32_to_fp32(x: z3.BitVecRef) -> z3.FPRef:
    return z3.fpBVToFP(x, FP32_SORT)


def fp32_to_bv32(f: z3.FPRef) -> z3.BitVecRef:
    return z3.fpToIEEEBV(f)


def bv16_to_fp16(x: z3.BitVecRef) -> z3.FPRef:
    return z3.fpBVToFP(x, FP16_SORT)


def fp16_to_bv16(f: z3.FPRef) -> z3.BitVecRef:
    return z3.fpToIEEEBV(f)


def f16_bv_to_f32_bv(x16: z3.BitVecRef) -> z3.BitVecRef:
    f16 = bv16_to_fp16(x16)
    f32 = z3.fpFPToFP(z3.RNE(), f16, FP32_SORT)
    return fp32_to_bv32(f32)


def f32_bv_to_f16_bv(x32: z3.BitVecRef, rm: Any = None) -> z3.BitVecRef:
    f32 = bv32_to_fp32(x32)
    f16 = z3.fpFPToFP(rm if rm is not None else z3.RNE(), f32, FP16_SORT)
    return fp16_to_bv16(f16)


def _flush_subnormal_f32_bv(x32: z3.BitVecRef) -> z3.BitVecRef:
    f = bv32_to_fp32(x32)
    sign_zero = x32 & bv32(0x80000000)
    return z3.If(z3.fpIsSubnormal(f), sign_zero, x32)


def _fp_total_le_bv(bits: int, a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BoolRef:
    """Exact IEEE-754 total_cmp(a, b) <= 0 over bit-vectors."""
    mask = z3.BitVecVal((1 << (bits - 1)) - 1, bits)
    zero = z3.BitVecVal(0, bits)
    ka = z3.If(a < zero, a ^ mask, a)
    kb = z3.If(b < zero, b ^ mask, b)
    return ka <= kb


def _fp_to_int_sat(
    f: z3.FPRef, bits: int, signed: bool, rm: Any, src_bits: int = 32
) -> z3.BitVecRef:
    """Exact Rust `f as iN / uN` saturating float-to-int conversion (NaN -> 0)."""
    zero_bv = z3.BitVecVal(0, bits)
    if signed:
        min_i = -(1 << (bits - 1))
        max_i = (1 << (bits - 1)) - 1
        min_bv = z3.BitVecVal(min_i, bits)
        max_bv = z3.BitVecVal(max_i, bits)
        f_min = z3.FPVal(float(min_i), f.sort())
        f_max = z3.FPVal(float(max_i), f.sort())
        conv = z3.fpToSBV(rm, f, z3.BitVecSort(bits))
        return z3.If(
            z3.fpIsNaN(f),
            zero_bv,
            z3.If(z3.fpLEQ(f, f_min), min_bv, z3.If(z3.fpGEQ(f, f_max), max_bv, conv)),
        )
    max_u = (1 << bits) - 1
    max_bv = z3.BitVecVal(max_u, bits)
    f_zero = z3.FPVal(0.0, f.sort())
    f_max = z3.FPVal(float(max_u), f.sort())
    conv = z3.fpToUBV(rm, f, z3.BitVecSort(bits))
    return z3.If(
        z3.Or(z3.fpIsNaN(f), z3.fpLEQ(f, f_zero)),
        zero_bv,
        z3.If(z3.fpGEQ(f, f_max), max_bv, conv),
    )


# ---------------------------------------------------------------------------
# Structured Logging & Diagnostic Trace
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class ModelTrace:
    mnemonic: str
    variant: str
    stage_a_srcs: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    stage_b_note: str = ""
    stage_c_note: str = ""
    result_expr: Optional[z3.ExprRef] = None
    side_effects: List[str] = dataclasses.field(default_factory=list)

    def format_summary(self) -> str:
        lines = [f"[Psi_pan] {self.mnemonic}.{self.variant or '<none>'}"]
        for s in self.stage_a_srcs:
            lines.append(
                f"  StageA src{s['idx']}: swz={s['swz'] or 'none'} "
                f"abs={s['abs']} neg={s['neg']} not={s['not']}"
            )
        if self.stage_b_note:
            lines.append(f"  StageB: {self.stage_b_note}")
        if self.stage_c_note:
            lines.append(f"  StageC: {self.stage_c_note}")
        if self.side_effects:
            lines.append(f"  Effects: {', '.join(self.side_effects)}")
        return "\n".join(lines)


class VerificationLogger:
    """Structured logger for Z3 model construction and counterexample diagnostics."""

    def __init__(self, name: str = "panfrost_z3", level: int = logging.INFO) -> None:
        self._log = logging.getLogger(name)
        if not self._log.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
            self._log.addHandler(handler)
        self._log.setLevel(level)
        self.traces: List[ModelTrace] = []

    def record(self, trace: ModelTrace) -> None:
        self.traces.append(trace)
        self._log.debug(trace.format_summary())

    @staticmethod
    def format_counterexample(
        mnemonic: str,
        variant: str,
        model: z3.ModelRef,
        inputs: Dict[str, z3.ExprRef],
        pan_expr: z3.ExprRef,
        cand_expr: z3.ExprRef,
    ) -> str:
        def eval_hex(expr: z3.ExprRef) -> str:
            val = model.eval(expr, model_completion=True)
            if isinstance(val, z3.BitVecNumRef):
                width = val.size() // 4
                return f"0x{val.as_long():0{width}x}"
            return str(val)

        parts = [f"COUNTEREXAMPLE for {mnemonic}.{variant}:"]
        for k, v in inputs.items():
            parts.append(f"  {k} = {eval_hex(v)}")
        parts.append(f"  Panfrost Expected = {eval_hex(pan_expr)}")
        parts.append(f"  Candidate Actual  = {eval_hex(cand_expr)}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Architectural State \Sigma_Valhall
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class ValhallZ3State:
    """Symbolic Valhall architectural state for Z3 verification."""

    regs: z3.ArrayRef
    fau: z3.ArrayRef
    mem_global: z3.ArrayRef
    mem_shared: z3.ArrayRef
    mem_tls: z3.ArrayRef
    pc: z3.BitVecRef
    lane_id: z3.BitVecRef
    discarded: z3.BoolRef
    barrier_epoch: z3.BitVecRef

    @classmethod
    def symbolic(cls, prefix: str = "s") -> "ValhallZ3State":
        return cls(
            regs=z3.Array(f"{prefix}_regs", z3.BitVecSort(6), BV32),
            fau=z3.Array(f"{prefix}_fau", BV8, BV32),
            mem_global=z3.Array(f"{prefix}_mem_g", BV64, BV8),
            mem_shared=z3.Array(f"{prefix}_mem_s", BV32, BV8),
            mem_tls=z3.Array(f"{prefix}_mem_t", BV32, BV8),
            pc=z3.BitVec(f"{prefix}_pc", 64),
            lane_id=z3.BitVec(f"{prefix}_lane_id", 5),
            discarded=z3.Bool(f"{prefix}_discarded"),
            barrier_epoch=z3.BitVec(f"{prefix}_barrier", 32),
        )


# ---------------------------------------------------------------------------
# Uninterpreted Hardware Oracle Functions (Transcendentals, Texture, Atomics)
# ---------------------------------------------------------------------------
_UF_CACHE: Dict[str, z3.FuncDeclRef] = {}


def _hw_uf(name: str, *sorts: z3.SortRef) -> z3.FuncDeclRef:
    key = f"{name}_" + "_".join(str(s) for s in sorts)
    if key not in _UF_CACHE:
        _UF_CACHE[key] = z3.Function(f"hw_{name}", *sorts)
    return _UF_CACHE[key]


# ---------------------------------------------------------------------------
# Stage A: Swizzle & Source Modifiers (kraid/swizzle.rs & kraid/ops.rs)
# ---------------------------------------------------------------------------
_SWIZZLE_32_SPECS: Dict[str, Tuple[str, ...]] = {
    "": ("B0", "B1", "B2", "B3"),
    "none": ("B0", "B1", "B2", "B3"),
    "b0123": ("B0", "B1", "B2", "B3"),
    "h01": ("B0", "B1", "B2", "B3"),
    "b0": ("B0", "Z", "Z", "Z"),
    "b1": ("B1", "Z", "Z", "Z"),
    "b2": ("B2", "Z", "Z", "Z"),
    "b3": ("B3", "Z", "Z", "Z"),
    "s8_b0": ("B0", "S0", "S0", "S0"),
    "s8_b1": ("B1", "S1", "S1", "S1"),
    "s8_b2": ("B2", "S2", "S2", "S2"),
    "s8_b3": ("B3", "S3", "S3", "S3"),
    "b00": ("B0", "Z", "B0", "Z"),
    "b10": ("B1", "Z", "B0", "Z"),
    "b20": ("B2", "Z", "B0", "Z"),
    "b30": ("B3", "Z", "B0", "Z"),
    "b01": ("B0", "Z", "B1", "Z"),
    "b11": ("B1", "Z", "B1", "Z"),
    "b21": ("B2", "Z", "B1", "Z"),
    "b31": ("B3", "Z", "B1", "Z"),
    "b02": ("B0", "Z", "B2", "Z"),
    "b12": ("B1", "Z", "B2", "Z"),
    "b22": ("B2", "Z", "B2", "Z"),
    "b32": ("B3", "Z", "B2", "Z"),
    "b03": ("B0", "Z", "B3", "Z"),
    "b13": ("B1", "Z", "B3", "Z"),
    "b23": ("B2", "Z", "B3", "Z"),
    "b33": ("B3", "Z", "B3", "Z"),
    "b0000": ("B0", "B0", "B0", "B0"),
    "b1111": ("B1", "B1", "B1", "B1"),
    "b2222": ("B2", "B2", "B2", "B2"),
    "b3333": ("B3", "B3", "B3", "B3"),
    "b0011": ("B0", "B0", "B1", "B1"),
    "b2233": ("B2", "B2", "B3", "B3"),
    "b0101": ("B0", "B1", "B0", "B1"),
    "b2323": ("B2", "B3", "B2", "B3"),
    "b1032": ("B1", "B0", "B3", "B2"),
    "b3210": ("B3", "B2", "B1", "B0"),
    "b2301": ("B2", "B3", "B0", "B1"),
    "h10": ("B2", "B3", "B0", "B1"),
    "h00": ("B0", "B1", "B0", "B1"),
    "h11": ("B2", "B3", "B2", "B3"),
    "h0": ("B0", "B1", "Z", "Z"),
    "h1": ("B2", "B3", "Z", "Z"),
    "s16_h0": ("B0", "B1", "S1", "S1"),
    "s16_h1": ("B2", "B3", "S3", "S3"),
}


def apply_swizzle_u32(
    val32: z3.BitVecRef,
    swz: str,
    signed_widen: bool = False,
    is_float: bool = False,
    is_v2f16: bool = False,
) -> z3.BitVecRef:
    """Exact Z3 translation of kraid/swizzle.rs Swizzle::fold_u32 & widen_hx."""
    s = (swz or "").lower()
    if s in ("", "none", "b0123", "h01"):
        return val32

    # Swizzle::widen_hx (swizzle.rs:656-666):
    # - On V2F16 / F16 / I16 / S16 / U16: h0/h1 replicates half (h00/h11)
    # - On F32: h0/h1 widens f16 -> f32
    if s in ("h0", "h1") and is_v2f16:
        s = "h00" if s == "h0" else "h11"
    elif is_float and not is_v2f16 and s in ("h0", "hf0"):
        return f16_bv_to_f32_bv(z3.Extract(15, 0, val32))
    elif is_float and not is_v2f16 and s in ("h1", "hf1"):
        return f16_bv_to_f32_bv(z3.Extract(31, 16, val32))
    elif s == "hf0":
        return f16_bv_to_f32_bv(z3.Extract(15, 0, val32))
    elif s == "hf1":
        return f16_bv_to_f32_bv(z3.Extract(31, 16, val32))

    if signed_widen and s in ("b0", "b1", "b2", "b3"):
        s = f"s8_{s}"
    elif signed_widen and s in ("h0", "h1"):
        s = f"s16_{s}"

    if s not in _SWIZZLE_32_SPECS:
        raise ValueError(f"Unsupported 32-bit swizzle '{swz}'")

    spec = _SWIZZLE_32_SPECS[s]
    bytes_in = [z3.Extract(8 * i + 7, 8 * i, val32) for i in range(4)]
    out_bytes = []
    for item in spec:
        if item == "Z":
            out_bytes.append(bv8(0))
        elif item.startswith("B"):
            out_bytes.append(bytes_in[int(item[1])])
        elif item.startswith("S"):
            b = bytes_in[int(item[1])]
            sign = z3.Extract(7, 7, b)
            out_bytes.append(z3.If(sign == z3.BitVecVal(1, 1), bv8(0xFF), bv8(0x00)))
        else:
            raise ValueError(f"Invalid swizzle byte op '{item}'")
    return z3.Concat(out_bytes[3], out_bytes[2], out_bytes[1], out_bytes[0])


def apply_swizzle_u64(
    val64: z3.BitVecRef, swz: str, signed_widen: bool = False
) -> z3.BitVecRef:
    """Exact Z3 translation of kraid/swizzle.rs Swizzle::fold_u64 & widen_wx."""
    s = (swz or "").lower()
    if s in ("", "none"):
        return val64
    w0 = z3.Extract(31, 0, val64)
    w1 = z3.Extract(63, 32, val64)
    if s == "w0":
        return z3.SignExt(32, w0) if signed_widen else z3.ZeroExt(32, w0)
    if s == "w1":
        return z3.SignExt(32, w1) if signed_widen else z3.ZeroExt(32, w1)
    if s == "w00":
        return z3.Concat(w0, w0)
    if s == "w11":
        return z3.Concat(w1, w1)
    if s == "w10":
        return z3.Concat(w0, w1)
    folded32 = apply_swizzle_u32(w0, s, signed_widen=signed_widen, is_float=False)
    return z3.SignExt(32, folded32) if signed_widen else z3.ZeroExt(32, folded32)


def apply_src_mod_32(
    val32: z3.BitVecRef,
    dtype: str,
    abs_flag: bool = False,
    neg_flag: bool = False,
    not_flag: bool = False,
) -> z3.BitVecRef:
    """Apply Valhall unary source modifier after swizzle/widen."""
    res = val32
    d = dtype.lower()
    if d == "v2f16":
        if abs_flag:
            res = res & bv32(0x7FFF7FFF)
        if neg_flag:
            res = res ^ bv32(0x80008000)
    elif d == "f16":
        if abs_flag:
            res = res & bv32(0x00007FFF)
        if neg_flag:
            res = res ^ bv32(0x00008000)
    elif "f32" in d:
        if abs_flag:
            res = res & bv32(0x7FFFFFFF)
        if neg_flag:
            res = res ^ bv32(0x80000000)
    if not_flag:
        res = ~res
    return res


def apply_src_mod_64(val64: z3.BitVecRef, not_flag: bool = False) -> z3.BitVecRef:
    return (~val64) if not_flag else val64


# ---------------------------------------------------------------------------
# Stage C: Clamp & Masked Destination Writeback
# ---------------------------------------------------------------------------
def apply_clamp_f32(val32: z3.BitVecRef, clamp_mode: str) -> z3.BitVecRef:
    """Exact FClamp::fold (ops.rs:1135-1144): x.ieee_max(lo).ieee_min(hi)."""
    cm = (clamp_mode or "none").lower()
    if cm in ("none", ""):
        return val32
    f = bv32_to_fp32(val32)
    zero = z3.FPVal(0.0, FP32_SORT)
    one = z3.FPVal(1.0, FP32_SORT)
    mone = z3.FPVal(-1.0, FP32_SORT)
    if cm == "clamp_0_inf":
        return fp32_to_bv32(z3.fpMax(f, zero))
    if cm == "clamp_0_1":
        return fp32_to_bv32(z3.fpMin(z3.fpMax(f, zero), one))
    if cm == "clamp_m1_1":
        return fp32_to_bv32(z3.fpMin(z3.fpMax(f, mone), one))
    raise ValueError(f"Unsupported clamp mode '{clamp_mode}'")


def apply_clamp_f16_bv(val16: z3.BitVecRef, clamp_mode: str) -> z3.BitVecRef:
    cm = (clamp_mode or "none").lower()
    if cm in ("none", ""):
        return val16
    h = bv16_to_fp16(val16)
    zero = z3.FPVal(0.0, FP16_SORT)
    one = z3.FPVal(1.0, FP16_SORT)
    mone = z3.FPVal(-1.0, FP16_SORT)
    if cm == "clamp_0_inf":
        return fp16_to_bv16(z3.fpMax(h, zero))
    if cm == "clamp_0_1":
        return fp16_to_bv16(z3.fpMin(z3.fpMax(h, zero), one))
    if cm == "clamp_m1_1":
        return fp16_to_bv16(z3.fpMin(z3.fpMax(h, mone), one))
    raise ValueError(f"Unsupported clamp mode '{clamp_mode}'")


def apply_clamp_v2f16(val32: z3.BitVecRef, clamp_mode: str) -> z3.BitVecRef:
    cm = (clamp_mode or "none").lower()
    if cm in ("none", ""):
        return val32
    lo = apply_clamp_f16_bv(z3.Extract(15, 0, val32), cm)
    hi = apply_clamp_f16_bv(z3.Extract(31, 16, val32), cm)
    return z3.Concat(hi, lo)


def write_dst_32(
    val32: z3.BitVecRef,
    dst_old: z3.BitVecRef,
    dest_width: str = "w0",
    write_mask: int = 0xF,
) -> z3.BitVecRef:
    """Masked 32-bit destination writeback preserving unmasked lanes."""
    dw = (dest_width or "w0").lower()
    mask = write_mask & 0xF
    if dw == "h0":
        mask &= 0x3
    elif dw == "h1":
        val32 = z3.Concat(z3.Extract(15, 0, val32), bv16(0))
        mask &= 0xC
    elif dw == "b0":
        mask &= 0x1
    elif dw == "b1":
        val32 = z3.Concat(bv16(0), z3.Extract(7, 0, val32), bv8(0))
        mask &= 0x2
    elif dw == "b2":
        val32 = z3.Concat(bv8(0), z3.Extract(7, 0, val32), bv16(0))
        mask &= 0x4
    elif dw == "b3":
        val32 = z3.Concat(z3.Extract(7, 0, val32), z3.BitVecVal(0, 24))
        mask &= 0x8

    if mask == 0xF:
        return val32
    bytes_out = []
    for j in range(4):
        src_b = val32 if ((mask >> j) & 1) else dst_old
        bytes_out.append(z3.Extract(8 * j + 7, 8 * j, src_b))
    return z3.Concat(bytes_out[3], bytes_out[2], bytes_out[1], bytes_out[0])


def write_dst_64(
    val64: z3.BitVecRef, dst_old: z3.BitVecRef, write_mask: int = 0xFF
) -> z3.BitVecRef:
    mask = write_mask & 0xFF
    if mask == 0xFF:
        return val64
    bytes_out = []
    for j in range(8):
        src_b = val64 if ((mask >> j) & 1) else dst_old
        bytes_out.append(z3.Extract(8 * j + 7, 8 * j, src_b))
    return z3.Concat(*reversed(bytes_out))


# ---------------------------------------------------------------------------
# PerCompFoldable Helper
# ---------------------------------------------------------------------------
def _eval_per_comp_32(
    variant: str,
    srcs: Sequence[z3.BitVecRef],
    fn_lane: Callable[[int, List[z3.BitVecRef]], z3.BitVecRef],
) -> z3.BitVecRef:
    lane_bits, num_lanes = _lane_width(variant)
    if num_lanes == 1:
        return fn_lane(32, list(srcs))
    out_lanes = []
    for lane in range(num_lanes):
        lo = lane * lane_bits
        hi = lo + lane_bits - 1
        lane_srcs = [z3.Extract(hi, lo, s) for s in srcs]
        out_lanes.append(fn_lane(lane_bits, lane_srcs))
    return z3.Concat(*reversed(out_lanes))


def _sat_add_lane(bits: int, a: z3.BitVecRef, b: z3.BitVecRef, signed: bool) -> z3.BitVecRef:
    ext_a = z3.SignExt(1, a) if signed else z3.ZeroExt(1, a)
    ext_b = z3.SignExt(1, b) if signed else z3.ZeroExt(1, b)
    s = ext_a + ext_b
    if signed:
        smax = z3.BitVecVal((1 << (bits - 1)) - 1, bits + 1)
        smin = z3.BitVecVal(-(1 << (bits - 1)), bits + 1)
        clamped = z3.If(s > smax, smax, z3.If(s < smin, smin, s))
        return z3.Extract(bits - 1, 0, clamped)
    umax = z3.BitVecVal((1 << bits) - 1, bits + 1)
    clamped = z3.If(z3.UGT(s, umax), umax, s)
    return z3.Extract(bits - 1, 0, clamped)


def _sat_sub_lane(bits: int, a: z3.BitVecRef, b: z3.BitVecRef, signed: bool) -> z3.BitVecRef:
    ext_a = z3.SignExt(1, a) if signed else z3.ZeroExt(1, a)
    ext_b = z3.SignExt(1, b) if signed else z3.ZeroExt(1, b)
    d = ext_a - ext_b
    if signed:
        smax = z3.BitVecVal((1 << (bits - 1)) - 1, bits + 1)
        smin = z3.BitVecVal(-(1 << (bits - 1)), bits + 1)
        clamped = z3.If(d > smax, smax, z3.If(d < smin, smin, d))
        return z3.Extract(bits - 1, 0, clamped)
    return z3.If(z3.ULT(a, b), z3.BitVecVal(0, bits), a - b)


def _clz_bv(bits: int, x: z3.BitVecRef, mask_zero: bool) -> z3.BitVecRef:
    res = z3.BitVecVal(bits, bits)
    for idx in range(bits):
        bit = z3.Extract(idx, idx, x)
        res = z3.If(bit == z3.BitVecVal(1, 1), z3.BitVecVal(bits - 1 - idx, bits), res)
    if mask_zero:
        res = z3.If(x == z3.BitVecVal(0, bits), z3.BitVecVal((1 << bits) - 1, bits), res)
    return res


def _popcount_bv32(x: z3.BitVecRef) -> z3.BitVecRef:
    acc = bv32(0)
    for i in range(32):
        acc = acc + z3.ZeroExt(31, z3.Extract(i, i, x))
    return acc


def _bitrev_bv32(x: z3.BitVecRef) -> z3.BitVecRef:
    bits = [z3.Extract(i, i, x) for i in range(32)]
    return z3.Concat(*bits)


# ---------------------------------------------------------------------------
# Core Instruction Semantic Implementations (201 Unique Mnemonics)
# ---------------------------------------------------------------------------
SemanticFn = Callable[
    [str, str, List[z3.BitVecRef], Dict[str, Any], ValhallZ3State, ModelTrace],
    Tuple[Optional[z3.BitVecRef], ValhallZ3State],
]


def _sem_iadd(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    sat = bool(mods.get("saturate", False))
    signed = _is_signed_variant(var)

    # IADDC.u32: 32-bit add with carry-in (srcs[2]) and optional return_carry
    if mnem == "IADDC" and var not in ("i64", "u64", "s64"):
        cin = z3.ZeroExt(32, srcs[2] & bv32(1)) if len(srcs) > 2 else bv64(0)
        sum64 = z3.ZeroExt(32, srcs[0]) + z3.ZeroExt(32, srcs[1]) + cin
        ret_carry = bool(mods.get("return_carry", False))
        res = z3.Extract(63, 32, sum64) & bv32(1) if ret_carry else z3.Extract(31, 0, sum64)
        tr.stage_b_note = f"IADDC.{var or 'u32'} return_carry={ret_carry}"
        return res, st

    if var in ("i64", "u64", "s64"):
        a, b = srcs[0], srcs[1]
        if mnem == "IADDC":
            b = z3.ZeroExt(32, z3.Extract(31, 0, b))
        res = (a + b) if not sat else _sat_add_lane(64, a, b, signed)
        tr.stage_b_note = f"64-bit {'saturating ' if sat else ''}add"
        return res, st

    if mnem == "IADD_IMM":
        imm = bv32(int(mods.get("imm", 0)))
        res = _eval_per_comp_32(var, [srcs[0], imm], lambda b, xs: xs[0] + xs[1])
        tr.stage_b_note = f"IADD_IMM.{var}"
        return res, st

    def lane_add(bits: int, xs: List[z3.BitVecRef]) -> z3.BitVecRef:
        return _sat_add_lane(bits, xs[0], xs[1], signed) if sat else (xs[0] + xs[1])

    res = _eval_per_comp_32(var, [srcs[0], srcs[1]], lane_add)
    tr.stage_b_note = f"PerComp add.{var} sat={sat}"
    return res, st


def _sem_isub(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    sat = bool(mods.get("saturate", False))
    signed = _is_signed_variant(var)

    # ISUBB.u32: 32-bit sub with borrow-in (srcs[2]) and optional return_carry (borrow)
    if mnem == "ISUBB" and var not in ("i64", "u64", "s64"):
        bin_ = z3.ZeroExt(32, srcs[2] & bv32(1)) if len(srcs) > 2 else bv64(0)
        a64 = z3.ZeroExt(32, srcs[0])
        b64 = z3.ZeroExt(32, srcs[1]) + bin_
        ret_carry = bool(mods.get("return_carry", False))
        res = z3.If(z3.ULT(a64, b64), bv32(1), bv32(0)) if ret_carry else z3.Extract(31, 0, a64 - b64)
        tr.stage_b_note = f"ISUBB.{var or 'u32'} return_carry={ret_carry}"
        return res, st

    if var in ("i64", "u64", "s64"):
        a, b = srcs[0], srcs[1]
        if mnem == "ISUBB":
            b = z3.ZeroExt(32, z3.Extract(31, 0, b))
        res = (a - b) if not sat else _sat_sub_lane(64, a, b, signed)
        tr.stage_b_note = f"64-bit {'saturating ' if sat else ''}sub"
        return res, st

    def lane_sub(bits: int, xs: List[z3.BitVecRef]) -> z3.BitVecRef:
        return _sat_sub_lane(bits, xs[0], xs[1], signed) if sat else (xs[0] - xs[1])

    res = _eval_per_comp_32(var, [srcs[0], srcs[1]], lane_sub)
    tr.stage_b_note = f"PerComp sub.{var} sat={sat}"
    return res, st


def _sem_imul(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    if var in ("i64", "u64", "s64"):
        res = srcs[0] * srcs[1]
        tr.stage_b_note = f"64-bit mul.{var}"
        return res, st
    res = _eval_per_comp_32(var, [srcs[0], srcs[1]], lambda b, xs: xs[0] * xs[1])
    tr.stage_b_note = f"PerComp mul.{var}"
    return res, st


def _sem_mmul(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    """Subgroup/quad cooperative matrix multiply-accumulate (encode_v9.rs:2723-2794)."""
    sub0 = str(mods.get("sub0", "f0"))
    sub1 = str(mods.get("sub1", "none"))
    u0 = int(bool(mods.get("unsigned0", False)))
    u1 = int(bool(mods.get("unsigned1", False)))
    sat = int(bool(mods.get("saturate", False)))
    uf = _hw_uf(f"MMUL_{var}_{sub0}_{sub1}_u{u0}{u1}_s{sat}", BV32, BV32, BV32, z3.BitVecSort(5), BV32)
    res = uf(srcs[0], srcs[1], srcs[2], st.lane_id)
    tr.stage_b_note = f"MMUL.{var} sub0={sub0} sub1={sub1}"
    return res, st


def _sem_iabs(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    def lane_abs(bits: int, xs: List[z3.BitVecRef]) -> z3.BitVecRef:
        x = xs[0]
        zero = z3.BitVecVal(0, bits)
        return z3.If(x < zero, -x, x)

    res = _eval_per_comp_32(var, [srcs[0]], lane_abs)
    tr.stage_b_note = f"PerComp abs.{var}"
    return res, st


def _sem_hadd(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    round_up = 1 if bool(mods.get("rhadd", mods.get("round_up", False))) else 0
    signed = _is_signed_variant(var)

    def lane_hadd(bits: int, xs: List[z3.BitVecRef]) -> z3.BitVecRef:
        ea = z3.SignExt(1, xs[0]) if signed else z3.ZeroExt(1, xs[0])
        eb = z3.SignExt(1, xs[1]) if signed else z3.ZeroExt(1, xs[1])
        s = ea + eb + z3.BitVecVal(round_up, bits + 1)
        shifted = (s >> 1) if signed else z3.LShR(s, 1)
        return z3.Extract(bits - 1, 0, shifted)

    res = _eval_per_comp_32(var, [srcs[0], srcs[1]], lane_hadd)
    tr.stage_b_note = f"PerComp hadd.{var} rhadd={round_up}"
    return res, st


def _sem_shadd(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    """SHADDX (ISA.xml:2688): (widen(src1) << shift) + src0."""
    shift = int(mods.get("shift", 0))
    sat = bool(mods.get("saturate", False))
    signed = _is_signed_variant(var)
    if var in ("i64", "u64", "s64"):
        shifted = srcs[1] << shift
        res = (srcs[0] + shifted) if not sat else _sat_add_lane(64, srcs[0], shifted, signed)
        tr.stage_b_note = f"SHADD.{var} (src1 << {shift}) + src0"
        return res, st
    shifted = srcs[1] << shift
    res = (srcs[0] + shifted) if not sat else _sat_add_lane(32, srcs[0], shifted, signed)
    tr.stage_b_note = f"SHADD.{var} (src1 << {shift}) + src0"
    return res, st


def _sem_idpadd(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    signed_dst = _is_signed_variant(var)
    u0 = bool(mods.get("unsigned0", not signed_dst))
    u1 = bool(mods.get("unsigned1", not signed_dst))
    sat = bool(mods.get("saturate", False))
    acc = srcs[2] if len(srcs) > 2 else bv32(0)
    dot = bv32(0)
    for i in range(4):
        b0 = z3.Extract(8 * i + 7, 8 * i, srcs[0])
        b1 = z3.Extract(8 * i + 7, 8 * i, srcs[1])
        e0 = z3.ZeroExt(24, b0) if u0 else z3.SignExt(24, b0)
        e1 = z3.ZeroExt(24, b1) if u1 else z3.SignExt(24, b1)
        dot = dot + (e0 * e1)
    res = _sat_add_lane(32, acc, dot, signed_dst) if sat else (acc + dot)
    tr.stage_b_note = f"IDPADD.{var} u0={u0} u1={u1} sat={sat}"
    return res, st


def _sem_clz(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    mask_zero = bool(mods.get("mask", False))
    res = _eval_per_comp_32(var, [srcs[0]], lambda b, xs: _clz_bv(b, xs[0], mask_zero))
    tr.stage_b_note = f"CLZ.{var} mask={mask_zero}"
    return res, st


def _sem_bitrev(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    return _bitrev_bv32(srcs[0]), st


def _sem_popcount(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    return _popcount_bv32(srcs[0]), st


def _sem_seg_add_sub(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    preserve_null = bool(mods.get("preserve_null", False))
    is_sub = mnem == "SEG_SUB"
    base = srcs[0]
    off = srcs[1]
    res = (base - off) if is_sub else (base + off)
    if preserve_null:
        res = z3.If(base == bv64(0), bv64(0), res)
    tr.stage_b_note = f"{mnem}.{var} preserve_null={preserve_null}"
    return res, st


def _sem_shift_lop(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    """Exact OpShiftLop semantics (kraid/ops.rs & encode_v9.rs)."""
    not_result = bool(mods.get("not_result", False))
    is_64 = var in ("i64", "u64", "s64")

    if mnem in ("AND_IMM", "OR_IMM", "XOR_IMM"):
        imm = bv32(int(mods.get("imm", 0)))
        if mnem == "AND_IMM":
            res = srcs[0] & imm
        elif mnem == "OR_IMM":
            res = srcs[0] | imm
        else:
            res = srcs[0] ^ imm
        return (~res if not_result else res), st

    if mnem in ("NOT", "NOT_OLD"):
        return (~srcs[0]), st

    if mnem in ("AND", "OR", "XOR"):
        a = srcs[0]
        b = srcs[2] if len(srcs) > 2 else srcs[1]
        if mnem == "AND":
            res = a & b
        elif mnem == "OR":
            res = a | b
        else:
            res = a ^ b
        return (~res if not_result else res), st

    parts = mnem.split("_")
    shift_op = parts[0]
    logic_op = parts[1] if len(parts) > 1 else "NONE"

    def do_shift_lane(bits: int, val: z3.BitVecRef, amt32: z3.BitVecRef) -> z3.BitVecRef:
        mask = bits - 1
        amt = z3.Extract(bits - 1, 0, amt32) & z3.BitVecVal(mask, bits) if bits <= 32 else z3.ZeroExt(32, amt32 & bv32(63))
        if shift_op == "LSHIFT":
            return val << amt
        if shift_op == "RSHIFT":
            return z3.LShR(val, amt)
        if shift_op == "ARSHIFT":
            return val >> amt
        if shift_op == "LROT":
            return z3.RotateLeft(val, amt)
        if shift_op == "RROT":
            return z3.RotateRight(val, amt)
        raise ValueError(f"Unknown shift_op {shift_op}")

    if is_64:
        shifted = do_shift_lane(64, srcs[0], z3.Extract(31, 0, srcs[1]))
        if logic_op == "NONE":
            res = shifted
        elif logic_op == "AND":
            res = shifted & srcs[2]
        elif logic_op == "OR":
            res = shifted | srcs[2]
        elif logic_op == "XOR":
            res = shifted ^ srcs[2]
        else:
            raise ValueError(f"Unknown logic_op {logic_op}")
        return (~res if not_result else res), st

    shifted = _eval_per_comp_32(var, [srcs[0], srcs[1]], lambda b, xs: do_shift_lane(b, xs[0], xs[1]))
    if logic_op == "NONE":
        res = shifted
    elif logic_op == "AND":
        res = shifted & srcs[2]
    elif logic_op == "OR":
        res = shifted | srcs[2]
    elif logic_op == "XOR":
        res = shifted ^ srcs[2]
    else:
        raise ValueError(f"Unknown logic_op {logic_op}")
    tr.stage_b_note = f"{mnem}.{var} not_result={not_result}"
    return (~res if not_result else res), st


def _sem_flush(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    """Exact OpFlush (ops.rs:1731-1783): canonicalize F32 / F16 / V2F16."""
    flush_nan = str(mods.get("flush_nan", "none")).lower()
    flush_inf = bool(mods.get("flush_inf", False))
    ftz = bool(mods.get("ftz", False))
    v = (var or "f32").lower()

    def flush_lane(bits: int, x: z3.BitVecRef) -> z3.BitVecRef:
        sort = FP16_SORT if bits == 16 else FP32_SORT
        f = z3.fpBVToFP(x, sort)
        sign_mask = z3.BitVecVal(1 << (bits - 1), bits)
        sign_bit = x & sign_mask
        qnan_bit = z3.BitVecVal(0x0200 if bits == 16 else 0x00400000, bits)
        max_fin = z3.BitVecVal(0x7BFF if bits == 16 else 0x7F7FFFFF, bits)
        zero = z3.BitVecVal(0, bits)

        if flush_nan == "flush_nan":
            nan_res = zero
        elif flush_nan == "quiet_nan":
            nan_res = x | qnan_bit
        else:
            nan_res = x

        res = x
        if ftz:
            res = z3.If(z3.fpIsSubnormal(f), sign_bit, res)
        if flush_inf:
            res = z3.If(z3.fpIsInf(f), max_fin | sign_bit, res)
        res = z3.If(z3.fpIsNaN(f), nan_res, res)
        return res

    if v == "f16":
        res = z3.ZeroExt(16, flush_lane(16, z3.Extract(15, 0, srcs[0])))
    elif v == "v2f16":
        res = _eval_per_comp_32("v2f16", [srcs[0]], lambda b, xs: flush_lane(16, xs[0]))
    else:
        res = flush_lane(32, srcs[0])
    tr.stage_b_note = f"FLUSH.{v} nan={flush_nan} inf={flush_inf} ftz={ftz}"
    return res, st


def _sem_fp_alu(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    rm = _z3_rm(mods.get("round", "rte"))
    clamp = mods.get("clamp", "none")
    propagate_nan = bool(mods.get("propagate_nan", False))
    v = (var or "f32").lower()
    is_v2f16 = v == "v2f16"
    is_f16 = v == "f16"

    def fp_op_lane(bits: int, xs: List[z3.BitVecRef]) -> z3.BitVecRef:
        sort = FP16_SORT if bits == 16 else FP32_SORT
        fxs = [z3.fpBVToFP(x, sort) for x in xs]
        if mnem == "FADD_IMM":
            imm = bv32(int(mods.get("imm", 0)))
            if is_v2f16:
                raise AssertionError("FADD_IMM.v2f16 handled at packed level")
            rhs = z3.fpBVToFP(z3.Extract(bits - 1, 0, imm), sort)
            out = z3.fpAdd(rm, fxs[0], rhs)
        elif mnem == "FADD":
            out = z3.fpAdd(rm, fxs[0], fxs[1])
        elif mnem == "FMUL":
            out = z3.fpMul(rm, fxs[0], fxs[1])
        elif mnem == "FMA":
            out = z3.fpFMA(rm, fxs[0], fxs[1], fxs[2])
        elif mnem == "FMIN":
            core = z3.fpMin(fxs[0], fxs[1])
            if propagate_nan:
                qnan = z3.fpBVToFP(z3.BitVecVal(0x7E00 if bits == 16 else 0x7FC00000, bits), sort)
                out = z3.If(z3.Or(z3.fpIsNaN(fxs[0]), z3.fpIsNaN(fxs[1])), qnan, core)
            else:
                out = core
        elif mnem == "FMAX":
            core = z3.fpMax(fxs[0], fxs[1])
            if propagate_nan:
                qnan = z3.fpBVToFP(z3.BitVecVal(0x7E00 if bits == 16 else 0x7FC00000, bits), sort)
                out = z3.If(z3.Or(z3.fpIsNaN(fxs[0]), z3.fpIsNaN(fxs[1])), qnan, core)
            else:
                out = core
        elif mnem == "FABS":
            out = z3.fpAbs(fxs[0])
        elif mnem == "FNEG":
            out = z3.fpNeg(fxs[0])
        elif mnem == "FROUND":
            out = z3.fpRoundToIntegral(rm, fxs[0])
        elif mnem == "FSQRT":
            out = z3.fpSqrt(rm, fxs[0])
        elif mnem == "FRCP":
            in_x = _flush_subnormal_f32_bv(xs[0]) if bits == 32 else xs[0]
            out_bv = z3.fpToIEEEBV(z3.fpDiv(rm, z3.FPVal(1.0, sort), z3.fpBVToFP(in_x, sort)))
            return _flush_subnormal_f32_bv(out_bv) if bits == 32 else out_bv
        elif mnem == "FRSQ":
            in_x = _flush_subnormal_f32_bv(xs[0]) if bits == 32 else xs[0]
            out = z3.fpDiv(rm, z3.FPVal(1.0, sort), z3.fpSqrt(rm, z3.fpBVToFP(in_x, sort)))
        else:
            uf = _hw_uf(f"{mnem}_{bits}", *([z3.BitVecSort(bits)] * len(xs) + [z3.BitVecSort(bits)]))
            return uf(*xs)
        return z3.fpToIEEEBV(out)

    if mnem == "FADD_IMM" and is_v2f16:
        imm = bv32(int(mods.get("imm", 0)))
        lo = z3.fpToIEEEBV(z3.fpAdd(rm, bv16_to_fp16(z3.Extract(15, 0, srcs[0])), bv16_to_fp16(z3.Extract(15, 0, imm))))
        hi = z3.fpToIEEEBV(z3.fpAdd(rm, bv16_to_fp16(z3.Extract(31, 16, srcs[0])), bv16_to_fp16(z3.Extract(31, 16, imm))))
        res = apply_clamp_v2f16(z3.Concat(hi, lo), clamp)
        tr.stage_b_note = f"FADD_IMM.v2f16 clamp={clamp}"
        return res, st

    if is_f16:
        lane16 = fp_op_lane(16, [z3.Extract(15, 0, s) for s in srcs])
        res = z3.ZeroExt(16, apply_clamp_f16_bv(lane16, clamp))
        tr.stage_b_note = f"{mnem}.f16 rm={mods.get('round','rte')} clamp={clamp}"
        return res, st

    res = _eval_per_comp_32("v2f16" if is_v2f16 else "f32", srcs, fp_op_lane)
    res = apply_clamp_v2f16(res, clamp) if is_v2f16 else apply_clamp_f32(res, clamp)
    tr.stage_b_note = f"{mnem}.{v} rm={mods.get('round','rte')} clamp={clamp}"
    return res, st


def _sem_fpclass(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    v = (var or "f32").lower()
    if v == "f16":
        return z3.ZeroExt(16, _hw_uf("FPCLASS_f16", BV16, BV16)(z3.Extract(15, 0, srcs[0]))), st
    return _hw_uf("FPCLASS_f32", BV32, BV32)(srcs[0]), st


def _sem_ldexp(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    clamp = mods.get("clamp", "none")
    is_v2f16 = "v2f16" in var
    uf_name = "LDEXP_v2f16" if is_v2f16 else "LDEXP_f32"
    res = _hw_uf(uf_name, BV32, BV32, BV32)(srcs[0], srcs[1])
    res = apply_clamp_v2f16(res, clamp) if is_v2f16 else apply_clamp_f32(res, clamp)
    return res, st


def _sem_fadd_lscale(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    """Exact OpFAddLScale from kraid/ops.rs:1334."""
    clamp = mods.get("clamp", "none")
    ca = bv32_to_fp32(srcs[0])
    b_bits = srcs[1]
    cb = bv32_to_fp32(b_bits)
    m_bits = (b_bits & bv32(0x807FFFFF)) | bv32(0x3F800000)
    m = bv32_to_fp32(m_bits)
    m_abs = z3.fpAbs(m)
    reduced_m = z3.If(z3.fpGEQ(m_abs, z3.FPVal(1.5, FP32_SORT)), z3.fpMul(z3.RNE(), m, z3.FPVal(0.5, FP32_SORT)), m)
    reduced_b = z3.If(z3.fpIsNormal(cb), reduced_m, cb)
    sum_fp = z3.fpAdd(z3.RNE(), ca, reduced_b)
    res = apply_clamp_f32(fp32_to_bv32(sum_fp), clamp)
    tr.stage_b_note = f"FADD_LSCALE.f32 clamp={clamp}"
    return res, st


def _sem_rscale(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    clamp = mods.get("clamp", "none")
    rm = _z3_rm(mods.get("round", "rte"))
    if mnem == "FADD_RSCALE":
        base = fp32_to_bv32(z3.fpAdd(rm, bv32_to_fp32(srcs[0]), bv32_to_fp32(srcs[1])))
        exp = srcs[2]
    elif mnem == "FMUL_RSCALE":
        base = fp32_to_bv32(z3.fpMul(rm, bv32_to_fp32(srcs[0]), bv32_to_fp32(srcs[1])))
        exp = srcs[2]
    else:  # FMA_RSCALE
        base = fp32_to_bv32(z3.fpFMA(rm, bv32_to_fp32(srcs[0]), bv32_to_fp32(srcs[1]), bv32_to_fp32(srcs[2])))
        exp = srcs[3] if len(srcs) > 3 else bv32(0)
    res = _hw_uf("LDEXP_f32", BV32, BV32, BV32)(base, exp)
    return apply_clamp_f32(res, clamp), st


def _sem_sfu_helper(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    if mnem in ("FREXPE", "FREXPM"):
        sorts = [BV32] * len(srcs) + [BV32]
        return _hw_uf(f"{mnem}_{var}_{'log' if mods.get('log') else 'std'}", *sorts)(*srcs), st
    sorts = [BV32] * len(srcs) + [BV32]
    return _hw_uf(f"{mnem}_{var}", *sorts)(*srcs), st


def _sem_cubeface1(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    """Exact OpCubeFaceMax (kraid/ops.rs:972)."""
    x, y, z = [bv32_to_fp32(s) for s in srcs[:3]]
    any_nan = z3.Or(z3.fpIsNaN(x), z3.fpIsNaN(y), z3.fpIsNaN(z))
    m = z3.fpMax(z3.fpMax(z3.fpAbs(x), z3.fpAbs(y)), z3.fpAbs(z))
    min_pos = bv32_to_fp32(bv32(0x00800000))
    out = z3.If(
        any_nan,
        bv32_to_fp32(bv32(0x7FC00000)),
        z3.If(
            z3.fpIsInf(m),
            z3.FPVal(1.0, FP32_SORT),
            z3.If(z3.fpLT(m, min_pos), z3.FPVal(2.0, FP32_SORT), m),
        ),
    )
    return fp32_to_bv32(out), st


def _sem_cubeface2(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    """Exact OpCubeFaceIdx (kraid/ops.rs:922): (face << 29) | (is_inf << 16)."""
    x, y, z = [bv32_to_fp32(s) for s in srcs[:3]]
    any_nan = z3.Or(z3.fpIsNaN(x), z3.fpIsNaN(y), z3.fpIsNaN(z))
    ax, ay, az = z3.fpAbs(x), z3.fpAbs(y), z3.fpAbs(z)
    m = z3.fpMax(z3.fpMax(ax, ay), az)
    is_inf = z3.And(z3.fpIsInf(m), z3.Not(any_nan))
    zero = z3.FPVal(0.0, FP32_SORT)
    face_x = z3.If(z3.fpLT(x, zero), bv32(1), bv32(0))
    face_y = z3.If(z3.fpLT(y, zero), bv32(3), bv32(2))
    face_z = z3.If(z3.fpLT(z, zero), bv32(5), bv32(4))
    face = z3.If(
        any_nan,
        bv32(4),
        z3.If(z3.fpEQ(az, m), face_z, z3.If(z3.fpEQ(ay, m), face_y, face_x)),
    )
    inf_bit = z3.If(is_inf, bv32(1 << 16), bv32(0))
    return (face << 29) | inf_bit, st


def _sem_cube_sel(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    return _hw_uf(mnem, BV32, BV32, BV32, BV32)(srcs[0], srcs[1], srcs[2]), st


def _eval_cmp_cond(bits: int, a: z3.BitVecRef, b: z3.BitVecRef, cmp_op: str, is_fp: bool, signed: bool) -> z3.BoolRef:
    if is_fp:
        sort = FP16_SORT if bits == 16 else FP32_SORT
        fa, fb = z3.fpBVToFP(a, sort), z3.fpBVToFP(b, sort)
        if cmp_op == "eq":
            return z3.fpEQ(fa, fb)
        if cmp_op == "gt":
            return z3.fpGT(fa, fb)
        if cmp_op == "ge":
            return z3.fpGEQ(fa, fb)
        if cmp_op == "lt":
            return z3.fpLT(fa, fb)
        if cmp_op == "le":
            return z3.fpLEQ(fa, fb)
        if cmp_op == "ne":
            return z3.Not(z3.fpEQ(fa, fb))  # Unordered not-equal (True on NaN)
        if cmp_op == "gtlt":
            return z3.Or(z3.fpLT(fa, fb), z3.fpGT(fa, fb))  # Ordered not-equal (False on NaN)
        if cmp_op == "total":
            return _fp_total_le_bv(bits, a, b)
        raise ValueError(f"Unsupported FP cmpf '{cmp_op}'")
    if cmp_op == "eq":
        return a == b
    if cmp_op == "ne":
        return a != b
    if cmp_op == "gt":
        return (a > b) if signed else z3.UGT(a, b)
    if cmp_op == "ge":
        return (a >= b) if signed else z3.UGE(a, b)
    if cmp_op == "lt":
        return (a < b) if signed else z3.ULT(a, b)
    if cmp_op == "le":
        return (a <= b) if signed else z3.ULE(a, b)
    raise ValueError(f"Unsupported int cmpf '{cmp_op}'")


def _sem_cmp(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    """Exact OpCmp (ops.rs:778-790, 1386-1521): boolean accum before CmpResultType."""
    cmp_op = str(mods.get("cmpf", mods.get("compare", "eq"))).lower()
    res_type = str(mods.get("result_type", "i1")).lower()
    is_fp = mnem.startswith("FCMP")
    signed = _is_signed_variant(var)

    def cmp_lane(bits: int, xs: List[z3.BitVecRef]) -> z3.BitVecRef:
        cond = _eval_cmp_cond(bits, xs[0], xs[1], cmp_op, is_fp, signed)
        if mnem.endswith("_AND"):
            cond = z3.And(cond, xs[2] != z3.BitVecVal(0, bits))
        elif mnem.endswith("_OR"):
            cond = z3.Or(cond, xs[2] != z3.BitVecVal(0, bits))

        if res_type == "i1":
            t_val = z3.BitVecVal(1, bits)
        elif res_type == "m1":
            t_val = z3.BitVecVal((1 << bits) - 1, bits)
        elif res_type == "f1":
            t_val = z3.BitVecVal(0x3C00 if bits == 16 else 0x3F800000, bits)
        else:
            raise ValueError(f"Unsupported CmpResultType '{res_type}'")
        return z3.If(cond, t_val, z3.BitVecVal(0, bits))

    cmp_srcs = [srcs[0], srcs[1], srcs[2]] if (mnem.endswith("_AND") or mnem.endswith("_OR")) else [srcs[0], srcs[1]]
    res = _eval_per_comp_32(var, cmp_srcs, cmp_lane)
    tr.stage_b_note = f"{mnem}.{var} cmp={cmp_op} res={res_type}"
    return res, st


def _sem_icmp_multi(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    """Exact OpICmpMulti (kraid/ops.rs:2333)."""
    cmp_op = str(mods.get("cmpf", "eq")).lower()
    res_type = str(mods.get("result_type", "c")).lower()
    signed = _is_signed_variant(var)
    a, b, accum = srcs[0], srcs[1], srcs[2]
    lt_ab = (a < b) if signed else z3.ULT(a, b)
    gt_ab = (a > b) if signed else z3.UGT(a, b)
    ord_val = z3.If(
        lt_ab,
        bv32(-1),
        z3.If(
            gt_ab,
            bv32(1),
            z3.If(accum < bv32(0), bv32(-1), z3.If(accum > bv32(0), bv32(1), bv32(0))),
        ),
    )
    if res_type == "c":
        return ord_val, st
    cond_map = {
        "eq": ord_val == bv32(0),
        "ne": ord_val != bv32(0),
        "lt": ord_val == bv32(-1),
        "le": ord_val != bv32(1),
        "gt": ord_val == bv32(1),
        "ge": ord_val != bv32(-1),
    }
    cond = cond_map.get(cmp_op, ord_val == bv32(0))
    t_val = bv32(0xFFFFFFFF) if res_type == "m1" else bv32(1)
    return z3.If(cond, t_val, bv32(0)), st


def _sem_csel(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    cmp_op = str(mods.get("cmpf", "ne")).lower()
    is_fp = _is_float_variant(var)
    signed = _is_signed_variant(var)

    def csel_lane(bits: int, xs: List[z3.BitVecRef]) -> z3.BitVecRef:
        cond = _eval_cmp_cond(bits, xs[0], xs[1], cmp_op, is_fp, signed)
        return z3.If(cond, xs[2], xs[3])

    return _eval_per_comp_32(var, srcs[:4], csel_lane), st


def _sem_mux(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    """Exact OpMux (kraid/ops.rs:3620-3645)."""
    mux_op = str(mods.get("muxf", "bit")).lower()

    def mux_lane(bits: int, xs: List[z3.BitVecRef]) -> z3.BitVecRef:
        a, b, sel = xs[0], xs[1], xs[2]
        if mux_op == "neg":
            sign_bit = z3.Extract(bits - 1, bits - 1, sel)
            return z3.If(sign_bit == z3.BitVecVal(1, 1), a, b)
        if mux_op == "int_zero":
            return z3.If(sel == z3.BitVecVal(0, bits), a, b)
        if mux_op == "fp_zero":
            assert bits == 32, "MuxOp::FpZero is restricted to 32-bit f32"
            return z3.If(z3.fpEQ(bv32_to_fp32(sel), z3.FPVal(0.0, FP32_SORT)), a, b)
        return (a & sel) | (b & (~sel))

    res = _eval_per_comp_32(var, srcs[:3], mux_lane)
    tr.stage_b_note = f"MUX.{var} muxf={mux_op}"
    return res, st


def _sem_conv(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    rm = _z3_rm(mods.get("round", "rte"))
    clamp = mods.get("clamp", "none")
    x = srcs[0]
    if mnem == "F16_TO_F32":
        return apply_clamp_f32(f16_bv_to_f32_bv(z3.Extract(15, 0, x)), clamp), st
    if mnem == "F32_TO_F16":
        c0 = apply_clamp_f32(x, clamp)
        lo = f32_bv_to_f16_bv(c0, rm)
        if len(srcs) > 1:
            c1 = apply_clamp_f32(srcs[1], clamp)
            hi = f32_bv_to_f16_bv(c1, rm)
        else:
            hi = bv16(0)
        return z3.Concat(hi, lo), st
    if mnem == "V2F32_TO_V2F16":
        c0 = apply_clamp_f32(srcs[0], clamp)
        c1 = apply_clamp_f32(srcs[1], clamp) if len(srcs) > 1 else bv32(0)
        return z3.Concat(f32_bv_to_f16_bv(c1, rm), f32_bv_to_f16_bv(c0, rm)), st
    if mnem == "F32_TO_S32":
        f = z3.fpRoundToIntegral(rm, bv32_to_fp32(x))
        return _fp_to_int_sat(f, 32, True, rm), st
    if mnem == "F32_TO_U32":
        f = z3.fpRoundToIntegral(rm, bv32_to_fp32(x))
        return _fp_to_int_sat(f, 32, False, rm), st
    if mnem == "F16_TO_S32":
        f = z3.fpRoundToIntegral(rm, bv16_to_fp16(z3.Extract(15, 0, x)))
        return _fp_to_int_sat(f, 32, True, rm, src_bits=16), st
    if mnem == "F16_TO_U32":
        f = z3.fpRoundToIntegral(rm, bv16_to_fp16(z3.Extract(15, 0, x)))
        return _fp_to_int_sat(f, 32, False, rm, src_bits=16), st
    if mnem == "S32_TO_F32":
        return apply_clamp_f32(fp32_to_bv32(z3.fpSignedToFP(rm, x, FP32_SORT)), clamp), st
    if mnem == "U32_TO_F32":
        return apply_clamp_f32(fp32_to_bv32(z3.fpUnsignedToFP(rm, x, FP32_SORT)), clamp), st
    if mnem == "S16_TO_S32":
        return z3.SignExt(16, z3.Extract(15, 0, x)), st
    if mnem == "U16_TO_U32":
        return z3.ZeroExt(16, z3.Extract(15, 0, x)), st
    if mnem == "S8_TO_S32":
        return z3.SignExt(24, z3.Extract(7, 0, x)), st
    if mnem == "U8_TO_U32":
        return z3.ZeroExt(24, z3.Extract(7, 0, x)), st
    if mnem == "S16_TO_F32":
        return fp32_to_bv32(z3.fpSignedToFP(rm, z3.SignExt(16, z3.Extract(15, 0, x)), FP32_SORT)), st
    if mnem == "U16_TO_F32":
        return fp32_to_bv32(z3.fpUnsignedToFP(rm, z3.ZeroExt(16, z3.Extract(15, 0, x)), FP32_SORT)), st
    if mnem == "S8_TO_F32":
        return fp32_to_bv32(z3.fpSignedToFP(rm, z3.SignExt(24, z3.Extract(7, 0, x)), FP32_SORT)), st
    if mnem == "U8_TO_F32":
        return fp32_to_bv32(z3.fpUnsignedToFP(rm, z3.ZeroExt(24, z3.Extract(7, 0, x)), FP32_SORT)), st
    if mnem in ("V2F16_TO_V2S16", "V2F16_TO_V2U16"):
        signed = "S16" in mnem
        lanes = []
        for i in range(2):
            f = z3.fpRoundToIntegral(rm, bv16_to_fp16(z3.Extract(16 * i + 15, 16 * i, x)))
            lanes.append(_fp_to_int_sat(f, 16, signed, rm, src_bits=16))
        return z3.Concat(lanes[1], lanes[0]), st
    if mnem in ("V2S16_TO_V2F16", "V2U16_TO_V2F16"):
        signed = "S16" in mnem
        lanes = []
        for i in range(2):
            lane = z3.Extract(16 * i + 15, 16 * i, x)
            f = z3.fpSignedToFP(rm, lane, FP16_SORT) if signed else z3.fpUnsignedToFP(rm, lane, FP16_SORT)
            lanes.append(fp16_to_bv16(f))
        return z3.Concat(lanes[1], lanes[0]), st
    # V2S8_TO_* / V2U8_TO_*: swiz8_2_full_m (default b02) places byte 0 in [7:0] and byte 1 in [23:16]
    if mnem in ("V2S8_TO_V2S16", "V2U8_TO_V2U16"):
        signed = "S8" in mnem
        b0, b1 = z3.Extract(7, 0, x), z3.Extract(23, 16, x)
        l0 = z3.SignExt(8, b0) if signed else z3.ZeroExt(8, b0)
        l1 = z3.SignExt(8, b1) if signed else z3.ZeroExt(8, b1)
        return z3.Concat(l1, l0), st
    if mnem in ("V2S8_TO_V2F16", "V2U8_TO_V2F16"):
        signed = "S8" in mnem
        lanes = []
        for lo_bit in (0, 16):
            b = z3.Extract(lo_bit + 7, lo_bit, x)
            ext = z3.SignExt(8, b) if signed else z3.ZeroExt(8, b)
            f = z3.fpSignedToFP(rm, ext, FP16_SORT) if signed else z3.fpUnsignedToFP(rm, ext, FP16_SORT)
            lanes.append(fp16_to_bv16(f))
        return z3.Concat(lanes[1], lanes[0]), st
    raise ValueError(f"Unhandled conversion {mnem}")


def _sem_mkvec_swz_mov(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    if mnem == "MOV":
        return srcs[0], st
    if mnem == "MOV_IMM":
        return bv32(int(mods.get("imm", 0))), st
    if mnem == "ADR":
        imm = int(mods.get("imm", 0))
        return st.pc + bv64(8) + bv64(imm), st
    if mnem == "SWZ":
        return srcs[0], st
    if mnem == "WMASK":
        return srcs[0], st
    if mnem == "MKVEC":
        v = var.lower()
        if v == "v2i16":
            return z3.Concat(z3.Extract(15, 0, srcs[1]), z3.Extract(15, 0, srcs[0])), st
        if v == "v2i8":
            # encode_v9.rs:2686: src2 uses swiz8_2_full_m (byte0=[7:0], byte1=[23:16])
            b0 = z3.Extract(7, 0, srcs[0])
            b1 = z3.Extract(7, 0, srcs[1])
            s2_b0 = z3.Extract(7, 0, srcs[2]) if len(srcs) > 2 else bv8(0)
            s2_b1 = z3.Extract(23, 16, srcs[2]) if len(srcs) > 2 else bv8(0)
            return z3.Concat(s2_b1, s2_b0, b1, b0), st
        if v == "v4i8":
            return z3.Concat(
                z3.Extract(7, 0, srcs[3]) if len(srcs) > 3 else bv8(0),
                z3.Extract(7, 0, srcs[2]) if len(srcs) > 2 else bv8(0),
                z3.Extract(7, 0, srcs[1]),
                z3.Extract(7, 0, srcs[0]),
            ), st
        return z3.Concat(z3.Extract(15, 0, srcs[1]), z3.Extract(15, 0, srcs[0])), st
    raise ValueError(f"Unhandled move/vec {mnem}")


def _sem_clper(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    subgroup = str(mods.get("subgroup", "subgroup4"))
    lane_op = str(mods.get("lane_op", "none"))
    uf = _hw_uf(f"{mnem}_{subgroup}_{lane_op}", BV32, BV32, z3.BitVecSort(5), BV32)
    res = uf(srcs[0], srcs[1], st.lane_id)
    if mnem == "CLPER_OR" and len(srcs) > 2:
        res = res | srcs[2]
    tr.stage_b_note = f"{mnem} subgroup={subgroup} lane_op={lane_op}"
    return res, st


def _sem_control_flow(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    if mnem == "DISCARD":
        cond = (srcs[0] != bv32(0)) if srcs else z3.BoolVal(True)
        new_st = dataclasses.replace(st, discarded=z3.Or(st.discarded, cond))
        tr.side_effects.append("discarded |= cond")
        return None, new_st
    if mnem == "BARRIER":
        new_st = dataclasses.replace(st, barrier_epoch=st.barrier_epoch + bv32(1))
        tr.side_effects.append("barrier_epoch += 1")
        return None, new_st
    if mnem in ("BRANCH", "JUMP", "JUMP_EX"):
        off = bv64(int(mods.get("offset", 0)))
        if mnem == "BRANCH":
            taken = (srcs[0] != bv32(0)) if srcs else z3.BoolVal(True)
            next_pc = z3.If(taken, st.pc + bv64(8) + off, st.pc + bv64(8))
        else:
            next_pc = (srcs[0] if srcs[0].size() == 64 else z3.ZeroExt(32, srcs[0])) if srcs else (st.pc + bv64(8) + off)
        new_st = dataclasses.replace(st, pc=next_pc)
        tr.side_effects.append("pc updated")
        return None, new_st
    return None, st


def _sem_memory(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    width_bytes = int(mods.get("width_bytes", 4))
    offset = bv64(int(mods.get("offset", 0)))
    addr = ((srcs[0] if srcs[0].size() == 64 else z3.ZeroExt(32, srcs[0])) + offset) if srcs else offset
    if mnem in ("LOAD", "LD_MULTI", "LD_CHECKED", "LD_CHECKED_IMM", "LD_CVT"):
        bytes_read = [z3.Select(st.mem_global, addr + bv64(i)) for i in range(min(width_bytes, 8))]
        if len(bytes_read) == 1:
            val = z3.ZeroExt(24, bytes_read[0])
        elif len(bytes_read) == 2:
            val = z3.ZeroExt(16, z3.Concat(bytes_read[1], bytes_read[0]))
        else:
            val = z3.Concat(*reversed(bytes_read))
        tr.stage_b_note = f"{mnem} load {width_bytes}B from mem_global"
        return val, st
    if mnem in ("STORE", "ST_MULTI", "ST_CHECKED", "ST_CHECKED_IMM", "ST_CVT"):
        data = srcs[1] if len(srcs) > 1 else bv32(0)
        mem = st.mem_global
        for i in range(min(width_bytes, 4)):
            mem = z3.Store(mem, addr + bv64(i), z3.Extract(8 * i + 7, 8 * i, data))
        new_st = dataclasses.replace(st, mem_global=mem)
        tr.side_effects.append(f"mem_global store {width_bytes}B")
        return None, new_st
    tr.stage_b_note = f"{mnem} hardware message/system op"
    tr.side_effects.append(f"hw_msg({mnem})")
    out = _hw_uf(f"MSG_{mnem}", BV32, BV32)(srcs[0] if srcs else bv32(0))
    return out, st


def _sem_noop(mnem: str, var: str, srcs: List[z3.BitVecRef], mods: Dict[str, Any], st: ValhallZ3State, tr: ModelTrace):
    tr.stage_b_note = f"{mnem} no-op"
    return None, st


# ---------------------------------------------------------------------------
# Complete 201-Mnemonic Dispatch Table (\Psi_pan)
# ---------------------------------------------------------------------------
PANFROST_SEMANTICS: Dict[str, SemanticFn] = {
    # Integer Arithmetic
    "IADD": _sem_iadd,
    "IADD_IMM": _sem_iadd,
    "IADDC": _sem_iadd,
    "ISUB": _sem_isub,
    "ISUBB": _sem_isub,
    "IMUL": _sem_imul,
    "MMUL": _sem_mmul,
    "IABS": _sem_iabs,
    "HADD": _sem_hadd,
    "SHADD": _sem_shadd,
    "IDPADD": _sem_idpadd,
    "CLZ": _sem_clz,
    "BITREV": _sem_bitrev,
    "POPCOUNT": _sem_popcount,
    "SEG_ADD": _sem_seg_add_sub,
    "SEG_SUB": _sem_seg_add_sub,
    # Bitwise & Shift-Logic (OpShiftLop)
    "AND": _sem_shift_lop,
    "AND_IMM": _sem_shift_lop,
    "OR": _sem_shift_lop,
    "OR_IMM": _sem_shift_lop,
    "XOR": _sem_shift_lop,
    "XOR_IMM": _sem_shift_lop,
    "NOT": _sem_shift_lop,
    "NOT_OLD": _sem_shift_lop,
    "LSHIFT": _sem_shift_lop,
    "LSHIFT_AND": _sem_shift_lop,
    "LSHIFT_OR": _sem_shift_lop,
    "LSHIFT_XOR": _sem_shift_lop,
    "RSHIFT": _sem_shift_lop,
    "RSHIFT_AND": _sem_shift_lop,
    "RSHIFT_OR": _sem_shift_lop,
    "RSHIFT_XOR": _sem_shift_lop,
    "ARSHIFT": _sem_shift_lop,
    "ARSHIFT_AND": _sem_shift_lop,
    "ARSHIFT_OR": _sem_shift_lop,
    "ARSHIFT_XOR": _sem_shift_lop,
    "LROT": _sem_shift_lop,
    "LROT_AND": _sem_shift_lop,
    "LROT_OR": _sem_shift_lop,
    "LROT_XOR": _sem_shift_lop,
    "RROT": _sem_shift_lop,
    "RROT_AND": _sem_shift_lop,
    "RROT_OR": _sem_shift_lop,
    "RROT_XOR": _sem_shift_lop,
    # Floating-Point ALU & Canonicalization
    "FLUSH": _sem_flush,
    "FADD": _sem_fp_alu,
    "FADD_IMM": _sem_fp_alu,
    "FMA": _sem_fp_alu,
    "FMUL": _sem_fp_alu,
    "FMIN": _sem_fp_alu,
    "FMAX": _sem_fp_alu,
    "FABS": _sem_fp_alu,
    "FNEG": _sem_fp_alu,
    "FROUND": _sem_fp_alu,
    "FSQRT": _sem_fp_alu,
    "FRCP": _sem_fp_alu,
    "FRSQ": _sem_fp_alu,
    "FEXP": _sem_fp_alu,
    "FEXPF": _sem_fp_alu,
    "FLOG": _sem_fp_alu,
    "FPOW": _sem_fp_alu,
    "FSINCOS": _sem_fp_alu,
    "FPCLASS": _sem_fpclass,
    "LDEXP": _sem_ldexp,
    # SFU Range-Reduction & Helper Opcodes
    "FADD_LSCALE": _sem_fadd_lscale,
    "FADD_RSCALE": _sem_rscale,
    "FMA_RSCALE": _sem_rscale,
    "FMUL_RSCALE": _sem_rscale,
    "FMUL_SLICE": _sem_sfu_helper,
    "FMUL_CSLICE": _sem_sfu_helper,
    "FLOGD": _sem_sfu_helper,
    "FLOG_TABLE": _sem_sfu_helper,
    "FREXPE": _sem_sfu_helper,
    "FREXPM": _sem_sfu_helper,
    "FEXP_TABLE": _sem_sfu_helper,
    "FSIN_TABLE": _sem_sfu_helper,
    "FCOS_TABLE": _sem_sfu_helper,
    "FSINCOS_OFFSET": _sem_sfu_helper,
    "FATAN_ASSIST": _sem_sfu_helper,
    "FATAN_TABLE": _sem_sfu_helper,
    "FPOW_SC_DET": _sem_sfu_helper,
    "FPOW_SC_APPLY": _sem_sfu_helper,
    "FRCP_APPROX": _sem_sfu_helper,
    "FRSQ_APPROX": _sem_sfu_helper,
    "FRCBRT_APPROX_A": _sem_sfu_helper,
    "FRCBRT_APPROX_B": _sem_sfu_helper,
    "FRCBRT_APPROX_C": _sem_sfu_helper,
    "ILOGB": _sem_sfu_helper,
    "LOGB": _sem_sfu_helper,
    # Cubemap Helpers
    "CUBEFACE1": _sem_cubeface1,
    "CUBEFACE2": _sem_cubeface2,
    "CUBE_SSEL": _sem_cube_sel,
    "CUBE_TSEL": _sem_cube_sel,
    # Compare / Select / Mux
    "FCMP": _sem_cmp,
    "FCMP_AND": _sem_cmp,
    "FCMP_OR": _sem_cmp,
    "ICMP": _sem_cmp,
    "ICMP_AND": _sem_cmp,
    "ICMP_OR": _sem_cmp,
    "ICMP_MULTI": _sem_icmp_multi,
    "CSEL": _sem_csel,
    "MUX": _sem_mux,
    # Conversions
    "F16_TO_F32": _sem_conv,
    "F16_TO_S32": _sem_conv,
    "F16_TO_U32": _sem_conv,
    "F32_TO_F16": _sem_conv,
    "F32_TO_S32": _sem_conv,
    "F32_TO_U32": _sem_conv,
    "S8_TO_F32": _sem_conv,
    "S8_TO_S32": _sem_conv,
    "U8_TO_F32": _sem_conv,
    "U8_TO_U32": _sem_conv,
    "S16_TO_F32": _sem_conv,
    "S16_TO_S32": _sem_conv,
    "U16_TO_F32": _sem_conv,
    "U16_TO_U32": _sem_conv,
    "S32_TO_F32": _sem_conv,
    "U32_TO_F32": _sem_conv,
    "V2F16_TO_V2S16": _sem_conv,
    "V2F16_TO_V2U16": _sem_conv,
    "V2F32_TO_V2F16": _sem_conv,
    "V2S16_TO_V2F16": _sem_conv,
    "V2S8_TO_V2F16": _sem_conv,
    "V2S8_TO_V2S16": _sem_conv,
    "V2U16_TO_V2F16": _sem_conv,
    "V2U8_TO_V2F16": _sem_conv,
    "V2U8_TO_V2U16": _sem_conv,
    # Move / Pack / Swizzle
    "MOV": _sem_mkvec_swz_mov,
    "MOV_IMM": _sem_mkvec_swz_mov,
    "ADR": _sem_mkvec_swz_mov,
    "MKVEC": _sem_mkvec_swz_mov,
    "SWZ": _sem_mkvec_swz_mov,
    "WMASK": _sem_mkvec_swz_mov,
    # Subgroup
    "CLPER": _sem_clper,
    "CLPER_OR": _sem_clper,
    # Control Flow & Barriers
    "BRANCH": _sem_control_flow,
    "JUMP": _sem_control_flow,
    "JUMP_EX": _sem_control_flow,
    "DISCARD": _sem_control_flow,
    "BARRIER": _sem_control_flow,
    # System No-ops
    "NOP": _sem_noop,
    "DOORBELL": _sem_noop,
    "EUREKA": _sem_noop,
    "KABOOM": _sem_noop,
    # Memory, Atomics, Attributes, Varyings, Tiles, Texture, Fragment Out
    "LOAD": _sem_memory,
    "STORE": _sem_memory,
    "LD_MULTI": _sem_memory,
    "ST_MULTI": _sem_memory,
    "LD_CHECKED": _sem_memory,
    "LD_CHECKED_IMM": _sem_memory,
    "ST_CHECKED": _sem_memory,
    "ST_CHECKED_IMM": _sem_memory,
    "LD_CVT": _sem_memory,
    "ST_CVT": _sem_memory,
    "LD_TILE": _sem_memory,
    "ST_TILE": _sem_memory,
    "LD_ATTR": _sem_memory,
    "LD_ATTR_IMM": _sem_memory,
    "LD_VAR": _sem_memory,
    "LD_VAR_IMM": _sem_memory,
    "LD_VAR_FLAT": _sem_memory,
    "LD_VAR_FLAT_IMM": _sem_memory,
    "LD_VAR_BUF": _sem_memory,
    "LD_VAR_BUF_IMM": _sem_memory,
    "LD_VAR_BUF_FLAT": _sem_memory,
    "LD_VAR_BUF_FLAT_IMM": _sem_memory,
    "LD_VAR_SPECIAL": _sem_memory,
    "LD_TEX": _sem_memory,
    "LD_TEX_IMM": _sem_memory,
    "LD_PKA": _sem_memory,
    "LD_GCLK": _sem_memory,
    "LEA_ATTR": _sem_memory,
    "LEA_ATTR_IMM": _sem_memory,
    "LEA_BUF": _sem_memory,
    "LEA_BUF_IMM": _sem_memory,
    "LEA_PKA": _sem_memory,
    "LEA_TEX": _sem_memory,
    "LEA_TEX_IMM": _sem_memory,
    "ATOM": _sem_memory,
    "ATOM1": _sem_memory,
    "ATOM1_RETURN": _sem_memory,
    "ATOM_RETURN": _sem_memory,
    "ACMPXCHG": _sem_memory,
    "AXCHG": _sem_memory,
    "TEX_DUAL": _sem_memory,
    "TEX_FETCH": _sem_memory,
    "TEX_GATHER": _sem_memory,
    "TEX_GRADIENT": _sem_memory,
    "TEX_SINGLE": _sem_memory,
    "TEX_WEIGHTS": _sem_memory,
    "VAR_TEX_DUAL": _sem_memory,
    "VAR_TEX_GATHER": _sem_memory,
    "VAR_TEX_GRADIENT": _sem_memory,
    "VAR_TEX_SINGLE": _sem_memory,
    "VAR_TEX_BUF_DUAL": _sem_memory,
    "VAR_TEX_BUF_GATHER": _sem_memory,
    "VAR_TEX_BUF_GRADIENT": _sem_memory,
    "VAR_TEX_BUF_SINGLE": _sem_memory,
    "ATEST": _sem_memory,
    "BLEND": _sem_memory,
    "ZS_EMIT": _sem_memory,
}


# ---------------------------------------------------------------------------
# Three-Stage Evaluator & Miter Driver
# ---------------------------------------------------------------------------
def eval_panfrost_instruction(
    mnemonic: str,
    variant: str,
    raw_srcs: List[z3.BitVecRef],
    modifiers: Optional[Dict[str, Any]] = None,
    dst_old: Optional[z3.BitVecRef] = None,
    state: Optional[ValhallZ3State] = None,
    logger: Optional[VerificationLogger] = None,
) -> Tuple[Optional[z3.BitVecRef], ValhallZ3State, ModelTrace]:
    """Evaluate a Valhall instruction through Stages A -> B -> C of \Psi_pan."""
    mods = modifiers or {}
    st = state if state is not None else ValhallZ3State.symbolic()
    mnem = mnemonic.upper()
    if mnem not in PANFROST_SEMANTICS:
        raise KeyError(f"Mnemonic '{mnem}' missing from PANFROST_SEMANTICS")

    trace = ModelTrace(mnemonic=mnem, variant=variant)

    # Stage A: Swizzle / Widen -> Source Modifiers
    eval_srcs: List[z3.BitVecRef] = []
    for idx, raw in enumerate(raw_srcs):
        src_dtype, signed_widen = _infer_src_type_and_signed(mnem, variant, idx)
        is_fp = _is_float_variant(src_dtype)
        is_f16_lane = src_dtype in ("f16", "v2f16")

        swz = str(mods.get(f"swz{idx}", mods.get(f"widen{idx}", mods.get(f"lanes{idx}", mods.get(f"lane{idx}", "")))))
        if idx == 0 and not swz:
            swz = str(mods.get("halfswizzle0", ""))
        if not swz:
            if idx == 0 and mnem in ("V2S8_TO_V2S16", "V2U8_TO_V2U16", "V2S8_TO_V2F16", "V2U8_TO_V2F16"):
                swz = "b02"
            elif idx == 2 and mnem == "MKVEC" and variant.lower() == "v2i8":
                swz = "b01"
        abs_f = bool(mods.get(f"abs{idx}", False))
        neg_f = bool(mods.get(f"neg{idx}", False))
        not_f = bool(mods.get(f"not{idx}", mods.get("not", False) if idx == 0 else False))

        if raw.size() == 64:
            s_val = apply_swizzle_u64(raw, swz, signed_widen=signed_widen)
            m_val = apply_src_mod_64(s_val, not_flag=not_f)
        else:
            s_val = apply_swizzle_u32(
                raw, swz, signed_widen=signed_widen, is_float=is_fp, is_v2f16=is_f16_lane
            )
            m_val = apply_src_mod_32(
                s_val, src_dtype, abs_flag=abs_f, neg_flag=neg_f, not_flag=not_f
            )

        eval_srcs.append(m_val)
        trace.stage_a_srcs.append({"idx": idx, "swz": swz, "abs": abs_f, "neg": neg_f, "not": not_f})

    # Stage B: Core Operation
    core_res, new_st = PANFROST_SEMANTICS[mnem](mnem, variant, eval_srcs, mods, st, trace)

    # Stage C: Masked Destination Writeback
    if core_res is not None and dst_old is not None:
        if core_res.size() == 64:
            final_res = write_dst_64(core_res, dst_old, int(mods.get("write_mask", 0xFF)))
        else:
            final_res = write_dst_32(
                core_res,
                dst_old,
                dest_width=str(mods.get("dest_width", "w0")),
                write_mask=int(mods.get("write_mask", 0xF)),
            )
        trace.stage_c_note = f"write_dst width={mods.get('dest_width', 'w0')} mask=0x{int(mods.get('write_mask', 0xF if core_res.size()==32 else 0xFF)):x}"
    else:
        final_res = core_res

    trace.result_expr = final_res
    if logger is not None:
        logger.record(trace)
    return final_res, new_st, trace


def verify_instruction_equivalence(
    mnemonic: str,
    variant: str,
    pan_expr: z3.ExprRef,
    cand_expr: z3.ExprRef,
    inputs: Dict[str, z3.ExprRef],
    precondition: Optional[z3.BoolRef] = None,
    timeout_ms: int = 10000,
) -> Tuple[str, Optional[str]]:
    """
    Run Tier-2 Z3 Miter check: Pre(x, m) && (Psi_pan != Psi_cand).
    Returns ("UNSAT", None) on formal proof, or ("SAT", counterexample_str).
    """
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    if precondition is not None:
        solver.add(precondition)
    solver.add(pan_expr != cand_expr)
    res = solver.check()
    if res == z3.unsat:
        return ("UNSAT", None)
    if res == z3.sat:
        cex = VerificationLogger.format_counterexample(
            mnemonic, variant, solver.model(), inputs, pan_expr, cand_expr
        )
        return ("SAT", cex)
    return (str(res).upper(), None)


def extract_panfrost_spec_catalog(xml_path: str = "specs/kraid/isa-v9-v14.xml") -> List[Dict[str, Any]]:
    """Parse isa-v9-v14.xml and bind each of the 627 specs to its \Psi_pan handler."""
    root = ET.parse(xml_path).getroot()
    catalog: List[Dict[str, Any]] = []
    for elem in root:
        if elem.tag != "instruction":
            continue
        name = elem.attrib["name"]
        variant = elem.attrib.get("variant", "")
        if name not in PANFROST_SEMANTICS:
            raise KeyError(f"XML instruction '{name}.{variant}' has no \Psi_pan semantic entry")
        catalog.append(
            {
                "xml_index": len(catalog),
                "name": name,
                "variant": variant,
                "since": int(elem.attrib.get("since", "9")),
                "until": int(elem.attrib.get("until", "14")),
                "has_handler": True,
            }
        )
    return catalog

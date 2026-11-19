"""Per-instruction operand descriptions and XML encoding-field mappings.

Renders the concise ``Operands:`` bullet list placed immediately after
``SYNTAX_BODY`` in Chapter 3 of ``valhall-isa-reference.tex``. Each bullet
names the operand exactly as it appears in the syntax line, maps it to the
underlying XML bitfield(s) when the field name is non-obvious (e.g.,
``@sr -> staging_register_index``, ``dest -> dest_register_index``,
``#attribute_table_index -> encoded_attribute_table_index``), and explains the
operand's semantic role in that instruction.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Mapping

from .fmt_helpers import md_inline_to_tex, tex_esc

# Per-mnemonic semantic descriptions keyed by syntax operand name:
#   "dest", "sw", "sr", "src0", "src1", "src2", "src3", "offset", "imm1w",
#   "shift", "attribute_index", "attribute_table_index", "varying_index",
#   "varying_table_index", "texture_index", "texture_table_index",
#   "buffer_index", "buffer_table_index".
_OP_ROLES: dict[str, dict[str, str]] = {
    # 1. Integer Arithmetic
    "BITREV": {
        "dest": "32-bit GPR receiving the bit-reversed word.",
        "src0": "32-bit input integer whose bit order is reversed.",
    },
    "CLZ": {
        "dest": "Destination GPR receiving per-lane leading-zero counts (`W` or `~0` on zero if `.mask`).",
        "src0": "Input integer (`u32`, `v2u16`, or `v4u8`).",
    },
    "HADD": {
        "dest": "Destination GPR receiving per-lane halving averages `(src0 + src1 + round) >> 1`.",
        "src0": "First integer operand.",
        "src1": "Second integer operand.",
    },
    "IABS": {
        "dest": "Destination GPR receiving per-lane wrapping absolute values.",
        "src0": "Signed integer input (`s32`, `v2s16`, or `v4s8`).",
    },
    "IADD": {
        "dest": "Destination GPR (or register pair for `s64`/`u64`) receiving the sum.",
        "src0": "First addend (supports Stage A widening/swizzle).",
        "src1": "Second addend (supports Stage A widening/swizzle).",
    },
    "IADDC": {
        "dest": "32-bit destination GPR receiving the low sum word (default) or carry-out bit (`.c`).",
        "src0": "First 32-bit unsigned addend.",
        "src1": "Second 32-bit unsigned addend.",
        "src2": "Carry-in operand (only bit 0, `src2 & 1`, is added).",
    },
    "IADD_IMM": {
        "dest": "Destination GPR receiving `src0 + #imm1w`.",
        "src0": "Input register (`k0` encodes the `MOV_IMM` alias).",
        "imm1w": "Inline 32-bit immediate addend (interpreted per lane for `v2i16`/`v4i8`).",
    },
    "IDPADD": {
        "dest": "32-bit destination GPR receiving `src2 + dot4x8(src0, src1)`.",
        "src0": "Packed 4x8-bit integer vector A (`b0..b3`).",
        "src1": "Packed 4x8-bit integer vector B (`b0..b3`).",
        "src2": "32-bit accumulator addend.",
    },
    "IMUL": {
        "dest": "Destination GPR (or 64-bit pair for `s64`/`u64` widening multiply) receiving the product.",
        "src0": "First multiplicand (must use `.w0`/`.w1` for `s64`/`u64`).",
        "src1": "Second multiplicand (must use `.w0`/`.w1` for `s64`/`u64`).",
    },
    "ISUB": {
        "dest": "Destination GPR (or 64-bit pair) receiving `src0 - src1`.",
        "src0": "Minuend (supports Stage A widening/swizzle).",
        "src1": "Subtrahend (supports Stage A widening/swizzle).",
    },
    "ISUBB": {
        "dest": "32-bit destination GPR receiving the low difference word (default) or borrow-out bit (`.c`).",
        "src0": "32-bit unsigned minuend.",
        "src1": "32-bit unsigned subtrahend.",
        "src2": "Borrow-in operand (only bit 0, `src2 & 1`, is subtracted).",
    },
    "MMUL": {
        "dest": "Per-lane slice of cooperative output matrix tile `D = A * B + C`.",
        "src0": "Per-lane slice of cooperative input matrix tile `A`.",
        "src1": "Per-lane slice of cooperative input matrix tile `B`.",
        "src2": "Per-lane slice of cooperative accumulator matrix tile `C`.",
    },
    "POPCOUNT": {
        "dest": "32-bit destination GPR receiving the number of set bits in `src0` (`0..32`).",
        "src0": "32-bit input bitmask.",
    },
    "SEG_ADD": {
        "dest": "64-bit destination register pair receiving the offset segment pointer.",
        "src0": "64-bit base segment pointer (preserved as `0` when `.preserve_null` is set).",
        "src1": "Signed/unsigned offset widened to 64 bits and added to `src0`.",
    },
    "SEG_SUB": {
        "dest": "64-bit destination register pair receiving the offset segment pointer.",
        "src0": "64-bit base segment pointer (preserved as `0` when `.preserve_null` is set).",
        "src1": "Offset widened to 64 bits and subtracted from `src0`.",
    },
    "SHADD": {
        "dest": "64-bit destination register pair receiving `src0 + (widen64(src1) << #shift)`.",
        "src0": "64-bit base address or integer.",
        "src1": "Index/offset widened to 64 bits before shifting.",
        "shift": "3-bit unsigned left-shift amount (`0..7`, e.g. `#2` for 4-byte stride).",
    },

    # 2. Bitwise & Shift-Logic
    "AND": {
        "dest": "Destination GPR receiving `src0 & src2` (or `~(src0 & src2)` with `.not_result`).",
        "src0": "First bitwise operand.",
        "src2": "Second bitwise operand (note: encoded in `src2`, not `src1`; supports `.not2`).",
    },
    "OR": {
        "dest": "Destination GPR receiving `src0 | src2` (or NOR with `.not_result`).",
        "src0": "First bitwise operand.",
        "src2": "Second bitwise operand (encoded in `src2`, not `src1`; supports `.not2`).",
    },
    "XOR": {
        "dest": "Destination GPR receiving `src0 ^ src2` (or XNOR with `.not_result`).",
        "src0": "First bitwise operand.",
        "src2": "Second bitwise operand (encoded in `src2`, not `src1`; supports `.not2`).",
    },
    "AND_IMM": {
        "dest": "32-bit destination GPR receiving `src0 & #imm1w`.",
        "src0": "32-bit input operand.",
        "imm1w": "Inline 32-bit bitwise AND mask.",
    },
    "OR_IMM": {
        "dest": "32-bit destination GPR receiving `src0 | #imm1w`.",
        "src0": "32-bit input operand.",
        "imm1w": "Inline 32-bit bitwise OR mask.",
    },
    "XOR_IMM": {
        "dest": "32-bit destination GPR receiving `src0 ^ #imm1w`.",
        "src0": "32-bit input operand.",
        "imm1w": "Inline 32-bit bitwise XOR mask.",
    },
    "NOT": {
        "dest": "Destination GPR receiving `~src0`.",
        "src0": "Input operand to invert bitwise.",
    },
    "NOT_OLD": {
        "dest": "Destination GPR receiving `~src0`.",
        "src0": "Input operand to invert bitwise.",
    },

    # 3. Floating-Point ALU
    "FABS": {
        "dest": "Destination GPR receiving `|src0|` (sign bit cleared).",
        "src0": "Floating-point input.",
    },
    "FADD": {
        "dest": "Destination GPR receiving `src0 + src1` (optionally narrowed via `dest_width_narrow`).",
        "src0": "First floating-point addend (supports `.abs`/`.neg`/swizzle).",
        "src1": "Second floating-point addend (supports `.abs`/`.neg`/swizzle).",
    },
    "FADD_IMM": {
        "dest": "Destination GPR receiving `src0 + #imm1w`.",
        "src0": "Floating-point input.",
        "imm1w": "Inline 32-bit IEEE-754 float immediate (or two packed FP16 values for `v2f16`).",
    },
    "FEXP": {
        "dest": "Destination GPR receiving `2^x`.",
        "src0": "Signed Q8.24 fixed-point exponent (`x * 2^24`) for `FEXP.f32` (unused on `FEXP.f16`).",
        "src1": "Original FP32 input for overflow/underflow/NaN edge cases on `FEXP.f32` (or direct FP16 exponent on `FEXP.f16`).",
    },
    "FEXPF": {
        "dest": "32-bit destination GPR receiving `exp2(src0)`.",
        "src0": "Direct FP32 exponent input (no Q8.24 pre-scaling).",
    },
    "FLOG": {
        "dest": "Destination GPR receiving `log2(src0)`.",
        "src0": "Floating-point input (`f16` on v12+, `f32` on v14+).",
    },
    "FLUSH": {
        "dest": "Destination GPR receiving canonicalized floating-point output.",
        "src0": "Floating-point input subject to `.ftz`, `.flush_inf`, `.flush_nan`, or `.quiet_nan`.",
    },
    "FMA": {
        "dest": "Destination GPR receiving `fma(src0, src1, src2)` (with optional clamp/narrowing).",
        "src0": "First multiplicand.",
        "src1": "Second multiplicand.",
        "src2": "Addend accumulated with single rounding.",
    },
    "FMAX": {
        "dest": "Destination GPR receiving `max(src0, src1)` per NaN handling mode.",
        "src0": "First floating-point operand.",
        "src1": "Second floating-point operand.",
    },
    "FMIN": {
        "dest": "Destination GPR receiving `min(src0, src1)` per NaN handling mode.",
        "src0": "First floating-point operand.",
        "src1": "Second floating-point operand.",
    },
    "FMUL": {
        "dest": "Destination GPR receiving `src0 * src1`.",
        "src0": "First multiplicand.",
        "src1": "Second multiplicand.",
    },
    "FNEG": {
        "dest": "Destination GPR receiving `-src0` (sign bit flipped).",
        "src0": "Floating-point input.",
    },
    "FPCLASS": {
        "dest": "Destination GPR receiving IEEE-754 classification bitmask.",
        "src0": "Floating-point value to classify.",
    },
    "FPOW": {
        "dest": "Destination GPR receiving `pow(src0, src1)` in FP16.",
        "src0": "FP16 base operand.",
        "src1": "FP16 exponent operand.",
    },
    "FRCP": {
        "dest": "Destination GPR receiving reciprocal `1.0 / src0`.",
        "src0": "Floating-point divisor (subnormals flush to `0.0` on `f32`).",
    },
    "FROUND": {
        "dest": "Destination GPR receiving rounded integral float value.",
        "src0": "Floating-point input rounded according to `round` modifier.",
    },
    "FRSQ": {
        "dest": "Destination GPR receiving reciprocal square root `1.0 / sqrt(src0)`.",
        "src0": "Floating-point input.",
    },
    "FSINCOS": {
        "dest": "Destination GPR receiving `sin` or `cos` of the reconstructed angle (v14+).",
        "src0": "Primary reduced angle operand.",
        "src1": "Auxiliary quadrant/range-reduction operand.",
    },
    "FSQRT": {
        "dest": "32-bit destination GPR receiving `sqrt(src0)`.",
        "src0": "FP32 input value.",
    },
    "LDEXP": {
        "dest": "Destination GPR receiving `src0 * 2^src1`.",
        "src0": "Floating-point significand/value.",
        "src1": "Signed integer power-of-two exponent adjustment.",
    },

    # 4. SFU Range-Reduction & Helpers
    "FADD_LSCALE": {
        "dest": "FP32 destination GPR receiving `src0 + reduce_log(src1)` (typically `m - 1` when `src0 = -1.0`).",
        "src0": "FP32 addend (usually `k1f.neg` = `-1.0`).",
        "src1": "FP32 input whose normal mantissa is reduced into `[0.75, 1.5)`.",
    },
    "FADD_RSCALE": {
        "dest": "FP32 destination GPR receiving `(src0 + src2) * 2^clamp(src3, -512, 512)`.",
        "src0": "First FP32 addend.",
        "src2": "Second FP32 addend (encoded in `src2`; `src1` is fixed to `1.0`).",
        "src3": "Signed integer exponent scale clamped to `[-512, 512]`.",
    },
    "FATAN_ASSIST": {
        "dest": "32-bit destination GPR receiving packed `(i16 octant/exp, f16 seed)` assist word.",
        "src0": "FP32 `y` argument of `atan2(y, x)`.",
        "src1": "FP32 `x` argument of `atan2(y, x)`.",
    },
    "FATAN_TABLE": {
        "dest": "FP32 destination GPR receiving the octant offset constant (`0`, `±pi/2`, `±pi`).",
        "src0": "32-bit assist/index word produced by `FATAN_ASSIST`.",
    },
    "FCOS_TABLE": {
        "dest": "FP32 destination GPR receiving `cos(idx * pi / 32)`.",
        "src0": "Fixed-point angle word whose low 6 bits select the 64-entry cosine LUT entry.",
    },
    "FEXP_TABLE": {
        "dest": "FP32 destination GPR receiving `2^(idx / 16)` (or low-order correction with `.low`).",
        "src0": "Input whose low 4 bits (`u4`) index the 16-entry exponential LUT.",
    },
    "FLOGD": {
        "dest": "FP32 destination GPR receiving `log2(m) / (m - 1)` for reduced mantissa `m`.",
        "src0": "FP32 input whose mantissa is reduced into `[0.75, 1.5)`.",
    },
    "FLOG_TABLE": {
        "dest": "FP32 destination GPR receiving the polynomial coefficient from the log LUT.",
        "src0": "FP32 input whose top mantissa bits index the table.",
    },
    "FMA_RSCALE": {
        "dest": "FP32 destination GPR receiving `(src0 * src1 + src2) * 2^clamp(src3, -512, 512)`.",
        "src0": "First FP32 multiplicand.",
        "src1": "Second FP32 multiplicand.",
        "src2": "FP32 addend.",
        "src3": "Signed integer exponent scale clamped to `[-512, 512]`.",
    },
    "FMUL_CSLICE": {
        "dest": "FP32 destination GPR receiving the cosine range-reduction slice product.",
        "src0": "FP32 angle input (supports `lane0`/`.abs`/`.neg`).",
        "src1": "Slice selector / extended-precision constant operand.",
    },
    "FMUL_RSCALE": {
        "dest": "FP32 destination GPR receiving `(src0 * src1) * 2^clamp(src3, -512, 512)`.",
        "src0": "First FP32 multiplicand.",
        "src1": "Second FP32 multiplicand (`src2` is fixed to `-0.0`).",
        "src3": "Signed integer exponent scale clamped to `[-512, 512]`.",
    },
    "FMUL_SLICE": {
        "dest": "FP32 destination GPR receiving the sliced `2/pi` range-reduction product.",
        "src0": "FP32 input angle.",
        "src1": "Slice index / reduction operand.",
    },
    "FPOW_SC_APPLY": {
        "dest": "FP32 destination GPR receiving either the special-case override or `src0`.",
        "src0": "Computed general-case power result `exp2(y * log2(|x|))`.",
        "src1": "Special-case tag word produced by `FPOW_SC_DET`.",
    },
    "FPOW_SC_DET": {
        "dest": "32-bit destination GPR receiving the encoded special-case tag for `FPOW_SC_APPLY`.",
        "src0": "FP32 base `x` of `pow(x, y)` / `powr(x, y)`.",
        "src1": "FP32 exponent `y` of `pow(x, y)` / `powr(x, y)`.",
    },
    "FRCBRT_APPROX_A": {
        "dest": "FP32 destination GPR receiving the initial `x^(-1/3)` approximation seed.",
        "src0": "FP32 input `x`.",
    },
    "FRCBRT_APPROX_B": {
        "dest": "FP32 destination GPR receiving the stage-B `x^(-1/3)` refinement value.",
        "src0": "Intermediate FP32 value from stage A.",
    },
    "FRCBRT_APPROX_C": {
        "dest": "FP32 destination GPR receiving the stage-C `x^(-1/3)` refinement value.",
        "src0": "Intermediate FP32 value from stage B.",
    },
    "FRCP_APPROX": {
        "dest": "FP32 destination GPR receiving the low-precision `1/x` seed for Newton-Raphson.",
        "src0": "FP32 input `x`.",
    },
    "FREXPE": {
        "dest": "Signed integer destination GPR receiving the unbiased exponent `e` (adjusted for `.sqrt`/`.log`).",
        "src0": "Floating-point input.",
    },
    "FREXPM": {
        "dest": "Floating-point destination GPR receiving the normalized significand (`[0.5, 1.0)` or `[0.75, 1.5)` with `.log`).",
        "src0": "Floating-point input.",
    },
    "FRSQ_APPROX": {
        "dest": "FP32 destination GPR receiving the low-precision `1/sqrt(x)` seed for Newton-Raphson.",
        "src0": "FP32 input `x`.",
    },
    "FSINCOS_OFFSET": {
        "dest": "FP32 destination GPR receiving the trig quadrant angle offset (scaled by `2^24` if `.scale`).",
        "src0": "Input whose low bits select the trig offset LUT entry.",
    },
    "FSIN_TABLE": {
        "dest": "FP32 destination GPR receiving `sin(idx * pi / 32)`.",
        "src0": "Fixed-point angle word whose low 6 bits select the 64-entry sine LUT entry.",
    },
    "ILOGB": {
        "dest": "Signed integer destination GPR receiving `floor(log2(|src0|))` (`INT32_MIN` on 0, `INT32_MAX` on Inf).",
        "src0": "Floating-point input.",
    },
    "LOGB": {
        "dest": "Floating-point destination GPR receiving `floor(log2(|src0|))` (`-Inf` on 0, `+Inf` on Inf).",
        "src0": "Floating-point input.",
    },

    # 5. Cubemap Helpers
    "CUBEFACE1": {
        "dest": "FP32 destination GPR receiving major-axis magnitude `max(|x|, |y|, |z|)` (`1.0` on Inf, `2.0` on zero/subnormal).",
        "src0": "FP32 cubemap direction `x` component.",
        "src1": "FP32 cubemap direction `y` component.",
        "src2": "FP32 cubemap direction `z` component.",
    },
    "CUBEFACE2": {
        "dest": "32-bit destination GPR receiving `(face_idx << 29) | (any_inf << 16)`.",
        "src0": "FP32 cubemap direction `x` component.",
        "src1": "FP32 cubemap direction `y` component.",
        "src2": "FP32 cubemap direction `z` component.",
    },
    "CUBE_SSEL": {
        "dest": "FP32 destination GPR receiving the signed horizontal (`S`) numerator for the selected face.",
        "src0": "FP32 cubemap direction `z` component.",
        "src1": "FP32 cubemap direction `x` component.",
        "src2": "Face word from `CUBEFACE2` (uses bits `[31:29]`).",
    },
    "CUBE_TSEL": {
        "dest": "FP32 destination GPR receiving the signed vertical (`T`) numerator for the selected face.",
        "src0": "FP32 cubemap direction `y` component.",
        "src1": "FP32 cubemap direction `z` component.",
        "src2": "Face word from `CUBEFACE2` (uses bits `[31:29]`).",
    },

    # 6. Compare, Select & Mux
    "CSEL": {
        "dest": "Destination GPR receiving `src2` when `cmpf(src0, src1)` is true, else `src3`.",
        "src0": "Left-hand comparison operand.",
        "src1": "Right-hand comparison operand.",
        "src2": "Value selected when the comparison is true.",
        "src3": "Value selected when the comparison is false.",
    },
    "FCMP": {
        "dest": "Destination GPR receiving comparison result formatted per `result_type` (`.i1`, `.m1`, or `.f1`).",
        "src0": "Left-hand floating-point comparison operand.",
        "src1": "Right-hand floating-point comparison operand.",
    },
    "FCMP_AND": {
        "dest": "Destination GPR receiving `cmpf(src0, src1) && (src2 != 0)` formatted per `result_type`.",
        "src0": "Left-hand floating-point comparison operand.",
        "src1": "Right-hand floating-point comparison operand.",
        "src2": "Existing boolean/mask predicate combined with logical AND.",
    },
    "FCMP_OR": {
        "dest": "Destination GPR receiving `cmpf(src0, src1) || (src2 != 0)` formatted per `result_type`.",
        "src0": "Left-hand floating-point comparison operand.",
        "src1": "Right-hand floating-point comparison operand.",
        "src2": "Existing boolean/mask predicate combined with logical OR (`k0` encodes standalone `FCMP`).",
    },
    "ICMP": {
        "dest": "Destination GPR receiving integer comparison result formatted per `result_type`.",
        "src0": "Left-hand integer comparison operand.",
        "src1": "Right-hand integer comparison operand.",
    },
    "ICMP_AND": {
        "dest": "Destination GPR receiving `cmpf(src0, src1) && (src2 != 0)` formatted per `result_type`.",
        "src0": "Left-hand integer comparison operand.",
        "src1": "Right-hand integer comparison operand.",
        "src2": "Existing boolean/mask predicate combined with logical AND.",
    },
    "ICMP_MULTI": {
        "dest": "Destination GPR receiving 3-way carry (`-1`/`0`/`+1` with `.c`) or chained multi-word comparison result.",
        "src0": "Left-hand 32-bit word of the multi-word integer.",
        "src1": "Right-hand 32-bit word of the multi-word integer.",
        "src2": "Incoming 3-way comparison carry from the less-significant word (`0` for lowest word).",
    },
    "ICMP_OR": {
        "dest": "Destination GPR receiving `cmpf(src0, src1) || (src2 != 0)` formatted per `result_type`.",
        "src0": "Left-hand integer comparison operand.",
        "src1": "Right-hand integer comparison operand.",
        "src2": "Existing boolean/mask predicate combined with logical OR (`k0` encodes standalone `ICMP`).",
    },
    "MUX": {
        "dest": "Destination GPR receiving lane/bit-selected output from `src0` and `src1`.",
        "src0": "Value selected when the `mux` test succeeds (or where mask bits are `1` in `.bit` mode).",
        "src1": "Value selected when the `mux` test fails (or where mask bits are `0` in `.bit` mode).",
        "src2": "Selector value or bitwise mask evaluated according to `mux` mode (`.int_zero`, `.neg`, `.fp_zero`, `.bit`).",
    },

    # 8. Move, Pack & Swizzle
    "ADR": {
        "dest": "32-bit destination GPR receiving `(PC + 8 + #imm1w) & 0xFFFFFFFF`.",
        "imm1w": "Signed 32-bit PC-relative byte offset from the next instruction (`PC + 8`).",
    },
    "MKVEC": {
        "dest": "32-bit destination GPR receiving the packed halfwords or bytes.",
        "src0": "Source supplying lane 0 (selected via `lane0`).",
        "src1": "Source supplying lane 1 (selected via `lane1`).",
        "src2": "Source supplying upper 16 bits `[31:16]` in `MKVEC.v2i8` mode (unused in `v2i16`).",
    },
    "MOV": {
        "dest": "Destination GPR receiving `src0` (optionally halfword-masked via `dest_width`).",
        "src0": "32-bit source register, uniform, or inline constant.",
    },
    "MOV_IMM": {
        "dest": "Destination GPR receiving `#imm1w` (optionally halfword-masked via `dest_width`).",
        "imm1w": "Inline 32-bit immediate value.",
    },
    "SWZ": {
        "dest": "32-bit destination GPR receiving swizzled 16-bit or 8-bit lanes of `src0`.",
        "src0": "Packed `v2i16` or `v4i8` source register.",
    },

    # 9. Subgroup & Cross-Lane
    "CLPER": {
        "dest": "Destination GPR receiving the value read from the target peer lane (or `inactive_result` if inactive).",
        "src0": "Per-lane data value shared across the subgroup.",
        "src1": "Byte 0 (`src1.b0`) supplies the target lane index, XOR mask, or shift delta per `lane_op`.",
    },
    "CLPER_OR": {
        "dest": "Destination GPR receiving `subgroupPermute(src0, src1.b0) | src2` (v14+).",
        "src0": "Per-lane data value shared across the subgroup.",
        "src1": "Byte 0 supplies the peer-lane index or shuffle operand.",
        "src2": "32-bit value bitwise-ORed into the shuffled result.",
    },
    "WMASK": {
        "dest": "32-bit destination GPR receiving the active-lane ballot bitmask within `subgroup`.",
        "src0": "Per-lane predicate (non-zero sets this lane's bit in the ballot mask).",
    },

    # 10. Control Flow
    "BRANCH": {
        "src0": "Condition register tested via `combine` (and inverted if `.not`).",
        "offset": "Signed PC-relative byte offset added to `PC + 8` when the branch is taken.",
    },
    "DISCARD": {
        "src0": "Left-hand comparison operand; lanes matching `cmpf(src0, src1)` are marked discarded/helper.",
        "src1": "Right-hand comparison operand.",
    },
    "JUMP": {
        "src0": "Condition register tested via `combine` (and inverted if `.not`).",
        "src1": "Target shader address (`0` terminates the shader; absolute or PC-relative per `address_mode`).",
    },
    "JUMP_EX": {
        "src0": "Condition register guarding the call/return/tail-call.",
        "src1": "Target shader address (for `.call` / `.replace`).",
        "src2": "Auxiliary call/stack operand.",
    },

    # 11. Memory Load / Store & LEA
    "LOAD": {
        "sw": "Contiguous destination staging register range receiving the loaded data.",
        "src0": "64-bit base GPU virtual address (`[rN:rN+1]` or 64-bit FAU uniform).",
        "offset": "Signed byte offset added to `src0`.",
    },
    "STORE": {
        "sr": "Contiguous source staging register range holding the data to write.",
        "src0": "64-bit base GPU virtual address (`[rN:rN+1]` or 64-bit FAU uniform).",
        "offset": "Signed byte offset added to `src0`.",
    },
    "LD_CHECKED": {
        "sw": "Destination staging register range receiving loaded data (or `0` / unchanged on OOB).",
        "src0": "64-bit base GPU virtual address.",
        "src1": "Dynamic unsigned 32-bit byte offset.",
        "src2": "32-bit buffer byte-size upper bound (`src1 + access_bytes <= src2`).",
    },
    "LD_CHECKED_IMM": {
        "sw": "Destination staging register range receiving loaded data (or `0` / unchanged on OOB).",
        "src0": "64-bit base GPU virtual address.",
        "offset": "Immediate unsigned byte offset (assembled from `offset_lo` and `offset_hi`).",
        "src2": "32-bit buffer byte-size upper bound (`#offset + access_bytes <= src2`).",
    },
    "ST_CHECKED": {
        "sr": "Source staging register range holding the data to store.",
        "src0": "64-bit base GPU virtual address.",
        "src1": "Dynamic unsigned 32-bit byte offset.",
        "src2": "32-bit buffer byte-size upper bound (OOB stores are silently dropped).",
    },
    "ST_CHECKED_IMM": {
        "sr": "Source staging register range holding the data to store.",
        "src0": "64-bit base GPU virtual address.",
        "offset": "Immediate unsigned byte offset (assembled from `offset_lo` and `offset_hi`).",
        "src2": "32-bit buffer byte-size upper bound (OOB stores are silently dropped).",
    },
    "LD_CVT": {
        "sw": "Destination staging register range receiving format-converted texel/attribute components.",
        "src0": "64-bit base GPU virtual address (typically from `LEA_TEX`/`LEA_ATTR`).",
        "offset": "Unsigned byte offset added to `src0`.",
        "src2": "32-bit hardware format conversion descriptor (typically `r2` from `LEA_TEX`/`LEA_ATTR`).",
    },
    "ST_CVT": {
        "sr": "Source staging register range holding register-format components to encode and store.",
        "src0": "64-bit base GPU virtual address (typically from `LEA_TEX`/`LEA_ATTR`).",
        "offset": "Unsigned byte offset added to `src0`.",
        "src2": "32-bit hardware format conversion descriptor (typically `r2` from `LEA_TEX`/`LEA_ATTR`).",
    },
    "LD_GCLK": {
        "sw": "2-register destination staging range (`@rN:rN+1`) receiving the 64-bit GPU clock value.",
    },
    "LD_MULTI": {
        "sw": "Destination staging register range (`sr_count` 32-bit words) receiving bulk-loaded data.",
        "src0": "64-bit base GPU virtual address.",
        "offset": "Signed byte offset added to `src0`.",
    },
    "ST_MULTI": {
        "sr": "Source staging register range (`sr_count` 32-bit words) written sequentially to memory.",
        "src0": "64-bit base GPU virtual address.",
        "offset": "Signed byte offset added to `src0`.",
    },
    "LD_PKA": {
        "sw": "Destination staging register range receiving the loaded buffer data.",
        "src0": "32-bit byte offset within the buffer.",
        "src1": "32-bit buffer descriptor handle (table + index).",
    },
    "LEA_ATTR": {
        "sw": "3-register destination range: `@sw[0:1]` = 64-bit attribute VA, `@sw[2]` = 32-bit format conversion descriptor.",
        "src0": "Vertex index (`VertexIndex`).",
        "src1": "Instance index (`InstanceIndex`).",
        "src2": "Dynamic attribute descriptor handle.",
    },
    "LEA_ATTR_IMM": {
        "sw": "3-register destination range: `@sw[0:1]` = 64-bit attribute VA, `@sw[2]` = 32-bit format conversion descriptor.",
        "src0": "Vertex index (`VertexIndex`).",
        "src1": "Instance index (`InstanceIndex`).",
        "attribute_index": "Immediate index of the Attribute descriptor within the table.",
        "attribute_table_index": "Immediate SRT table index holding the Attribute descriptor array.",
    },
    "LEA_BUF": {
        "sw": "2-register destination range (`@sw[0:1]`) receiving the 64-bit effective buffer GPU VA.",
        "src0": "32-bit byte offset (or vertex index for IDVS output buffers).",
        "src1": "Dynamic buffer descriptor handle.",
    },
    "LEA_BUF_IMM": {
        "sw": "2-register destination range (`@sw[0:1]`) receiving the 64-bit effective buffer GPU VA.",
        "src0": "32-bit byte offset (or vertex index for IDVS output buffers).",
        "buffer_index": "Immediate index of the Buffer descriptor within the table.",
        "buffer_table_index": "Immediate SRT table index holding the Buffer descriptor array.",
    },
    "LEA_PKA": {
        "sw": "2-register destination range (`@sw[0:1]`) receiving the 64-bit effective PKA buffer GPU VA.",
        "src0": "32-bit byte offset within the buffer.",
        "src1": "32-bit PKA buffer descriptor handle.",
    },
    "LEA_TEX": {
        "sw": "3-register destination range: `@sw[0:1]` = 64-bit texel VA, `@sw[2]` = 32-bit format conversion descriptor.",
        "src0": "Packed `u16vec2(x, y)` texel coordinates.",
        "src1": "3D slice / array layer `z` coordinate (and optional MSAA sample index).",
        "src2": "Dynamic image/texture descriptor handle.",
    },
    "LEA_TEX_IMM": {
        "sw": "3-register destination range: `@sw[0:1]` = 64-bit texel VA, `@sw[2]` = 32-bit format conversion descriptor.",
        "src0": "Packed `u16vec2(x, y)` texel coordinates.",
        "src1": "3D slice / array layer `z` coordinate.",
        "texture_index": "Immediate index of the Image/Texture descriptor within the table.",
        "texture_table_index": "Immediate SRT table index holding the Image/Texture descriptor array.",
    },

    # 12. Atomics
    "ACMPXCHG": {
        "sw": "Destination staging register(s) receiving the pre-modification memory value.",
        "sr": "Source staging register pair holding `(compare_value, swap_value)`.",
        "src0": "64-bit base GPU virtual address.",
        "offset": "Unsigned byte offset added to `src0`.",
    },
    "ATOM": {
        "sr": "Source staging register(s) holding the atomic operand value (no return value written).",
        "src0": "64-bit base GPU virtual address.",
        "offset": "Unsigned byte offset added to `src0`.",
    },
    "ATOM1": {
        "src0": "64-bit base GPU virtual address for unary atomic RMW (`ainc`/`adec`/etc.).",
        "offset": "Unsigned byte offset added to `src0`.",
    },
    "ATOM1_RETURN": {
        "sw": "Destination staging register(s) receiving the pre-modification memory value.",
        "src0": "64-bit base GPU virtual address.",
        "offset": "Unsigned byte offset added to `src0`.",
    },
    "ATOM_RETURN": {
        "sw": "Destination staging register(s) receiving the pre-modification memory value.",
        "sr": "Source staging register(s) holding the atomic operand value.",
        "src0": "64-bit base GPU virtual address.",
        "offset": "Unsigned byte offset added to `src0`.",
    },
    "AXCHG": {
        "sw": "Destination staging register(s) receiving the previous memory value.",
        "sr": "Source staging register(s) holding the replacement value to swap into memory.",
        "src0": "64-bit base GPU virtual address.",
        "offset": "Unsigned byte offset added to `src0`.",
    },

    # 13. Attributes, Varyings & Fragment
    "ATEST": {
        "sw": "Destination staging register (pinned `r60`) receiving the updated MSAA sample coverage mask.",
        "src0": "Incoming sample coverage mask (pinned `r60`).",
        "src1": "Fragment alpha value (`f32`) for alpha-to-coverage.",
        "src2": "Hardware alpha-test datum (`atest_datum.w0` FAU special).",
    },
    "BLEND": {
        "sr": "Contiguous source staging registers holding the fragment RGBA output color.",
        "src0": "64-bit blend descriptor (`blend_descriptor_0..7` FAU special).",
        "offset": "Unsigned PC-relative byte offset to the blend shader continuation (or return offset).",
        "src2": "Sample coverage mask (typically `r60^`).",
    },
    "LD_ATTR": {
        "sw": "Destination staging register range receiving `vecsize` converted attribute components.",
        "src0": "Vertex index (`VertexIndex`, preloaded in `r60`).",
        "src1": "Instance index (`InstanceIndex`, preloaded in `r61`).",
        "src2": "Dynamic attribute descriptor handle.",
    },
    "LD_ATTR_IMM": {
        "sw": "Destination staging register range receiving `vecsize` converted attribute components.",
        "src0": "Vertex index (`VertexIndex`, preloaded in `r60`).",
        "src1": "Instance index (`InstanceIndex`, preloaded in `r61`).",
        "attribute_index": "Immediate index of the Attribute descriptor within the table.",
        "attribute_table_index": "Immediate SRT table index holding the Attribute descriptor array.",
    },
    "LD_TEX": {
        "sw": "Destination staging register range receiving the loaded texel components.",
        "src0": "Packed `u16vec2(x, y)` integer texel coordinates.",
        "src1": "3D slice / array layer `z` coordinate.",
        "src2": "Dynamic image/texture descriptor handle.",
    },
    "LD_TEX_IMM": {
        "sw": "Destination staging register range receiving the loaded texel components.",
        "src0": "Packed `u16vec2(x, y)` integer texel coordinates.",
        "src1": "3D slice / array layer `z` coordinate.",
        "texture_index": "Immediate index of the Image/Texture descriptor within the table.",
        "texture_table_index": "Immediate SRT table index holding the Image/Texture descriptor array.",
    },
    "LD_TILE": {
        "sw": "Destination staging register range receiving pixel data read from on-chip tile SRAM.",
        "src0": "Pixel coordinate / sample selector.",
        "src1": "Sample coverage mask.",
        "src2": "Tilebuffer conversion descriptor / render-target selector.",
    },
    "LD_VAR": {
        "sw": "Destination staging register range receiving interpolated varying components.",
        "src0": "Fragment sample/pixel coordinate (`r61`, or explicit offset for `interpolateAtOffset`).",
        "src1": "Dynamic varying descriptor handle.",
    },
    "LD_VAR_IMM": {
        "sw": "Destination staging register range receiving interpolated varying components.",
        "src0": "Fragment sample/pixel coordinate (`r61`, or explicit offset).",
        "varying_index": "Immediate varying slot index within the varying table.",
        "varying_table_index": "Immediate SRT table index for varyings (typically `#0xb`).",
    },
    "LD_VAR_BUF": {
        "sw": "Destination staging register range receiving interpolated varying components from the varying buffer.",
        "src0": "Fragment sample/pixel coordinate (`r61`).",
        "src1": "Dynamic byte offset within the descriptor-less varying buffer.",
    },
    "LD_VAR_BUF_IMM": {
        "sw": "Destination staging register range receiving interpolated varying components from the varying buffer.",
        "src0": "Fragment sample/pixel coordinate (`r61`).",
        "offset": "Immediate unsigned byte offset within the varying buffer.",
    },
    "LD_VAR_BUF_FLAT": {
        "sw": "Destination staging register range receiving uninterpolated vertex data from the varying buffer.",
        "src0": "Dynamic byte offset (and vertex index when `.explicit_vertex` is set).",
    },
    "LD_VAR_BUF_FLAT_IMM": {
        "sw": "Destination staging register range receiving uninterpolated provoking-vertex data.",
        "offset": "Immediate unsigned byte offset within the varying buffer.",
    },
    "LD_VAR_FLAT": {
        "sw": "Destination staging register range receiving uninterpolated flat varying components.",
        "src0": "Dynamic varying descriptor handle.",
    },
    "LD_VAR_FLAT_IMM": {
        "sw": "Destination staging register range receiving uninterpolated flat varying components.",
        "varying_index": "Immediate varying slot index within the varying table.",
        "varying_table_index": "Immediate SRT table index for varyings (typically `#0xb`).",
    },
    "LD_VAR_SPECIAL": {
        "sw": "Destination staging register range receiving `point`, `bary`, `frag_w`, or `frag_z`.",
        "src0": "Sample position / pixel coordinate (`r61` for `.sample`, `k0` for `.center`).",
    },
    "ST_TILE": {
        "sr": "Source staging register range holding the RGBA pixel value to write into tile SRAM.",
        "src0": "Pixel coordinate / sample selector.",
        "src1": "Sample coverage mask.",
        "src2": "Tilebuffer conversion descriptor / render-target selector.",
    },
    "ZS_EMIT": {
        "sw": "Destination staging register (pinned `r60`) receiving the post-depth/stencil coverage mask.",
        "src0": "Shader-computed depth value `gl_FragDepth` (`f32`, used when `.z` is set).",
        "src1": "Shader-computed 8-bit stencil reference (`src1 & 0xFF`, used when `.stencil` is set).",
        "src2": "Incoming sample coverage mask (pinned `r60`).",
    },

    # 14. Texture & Varying-Texture
    "TEX_SINGLE": {
        "sw": "Destination staging register range receiving sampled RGBA texels (per `write_mask`).",
        "sr": "Source staging register range holding packed coordinates `(u, v, [w/layer], [lod/bias], [dref], [offset])`.",
        "src0": "Texture/sampler descriptor handle (32-bit FAU uniform or 64-bit register pair).",
    },
    "TEX_DUAL": {
        "sw": "Destination staging register range receiving two consecutive 4-component sample results.",
        "sr": "Source staging register range holding shared texture coordinates and LOD parameters.",
        "src0": "Dual texture/sampler descriptor handle.",
    },
    "TEX_FETCH": {
        "sw": "Destination staging register range receiving unfiltered texel components.",
        "sr": "Source staging register range holding integer coordinates `(x, [y], [z/layer], [lod/sample])`.",
        "src0": "Texture descriptor handle.",
    },
    "TEX_GATHER": {
        "sw": "Destination staging register range receiving 4 gathered values from the 2x2 texel footprint.",
        "sr": "Source staging register range holding `(u, v, [layer], [dref], [offset])`.",
        "src0": "Texture/sampler descriptor handle.",
    },
    "TEX_GRADIENT": {
        "sw": "Destination staging register range receiving sampled texels (or fixed-point 8.8 LOD / derivatives).",
        "sr": "Source staging register range holding coordinates and explicit `(dPdx, dPdy)` gradients (or UV for LOD query).",
        "src0": "Texture/sampler descriptor handle.",
    },
    "TEX_WEIGHTS": {
        "sw": "Destination staging register range receiving the 4 bilinear filter weights (v9 only).",
        "sr": "Source staging register range holding `(u, v)` texture coordinates.",
        "src0": "Texture/sampler descriptor handle.",
    },
}

# Populate shift-logic family roles cleanly
for _sh_base, _sh_desc in (
    ("LSHIFT", "logical left shift"),
    ("RSHIFT", "logical right shift"),
    ("ARSHIFT", "arithmetic right shift"),
    ("LROT", "rotate left"),
    ("RROT", "rotate right"),
):
    for _lop, _lop_desc in (
        ("", ""),
        ("_AND", "bitwise AND"),
        ("_OR", "bitwise OR"),
        ("_XOR", "bitwise XOR"),
    ):
        _mn = _sh_base + _lop
        _d = {
            "dest": f"Destination GPR receiving the {_sh_desc}" + (f" combined via {_lop_desc}" if _lop else "") + " (inverted if `.not_result`).",
            "src0": "Input integer value to shift/rotate.",
            "src1": "8-bit shift/rotate count per lane (selected via `lanes1`, masked modulo lane width `W`).",
        }
        if _lop:
            _d["src2"] = f"Second operand combined via {_lop_desc} after shifting (inverted if `.not2`)."
        _OP_ROLES[_mn] = _d

# Populate conversion family roles cleanly
for _conv_mn, _conv_desc in (
    ("F16_TO_F32", "FP16 input lane (`lane0`) promoted losslessly to FP32."),
    ("F16_TO_S32", "FP16 input lane (`lane0`) rounded and saturated to signed 32-bit integer."),
    ("F16_TO_U32", "FP16 input lane (`lane0`) rounded and saturated to unsigned 32-bit integer."),
    ("F32_TO_F16", "FP32 input value narrowed to FP16 into `dest.h0`, `dest.h1`, or replicated `dest.h01`."),
    ("F32_TO_S32", "FP32 input rounded per `round` and saturated to `[-2^31, 2^31-1]` (NaN -> `0`)."),
    ("F32_TO_U32", "FP32 input rounded per `round` and saturated to `[0, 2^32-1]` (NaN/<0 -> `0`)."),
    ("S16_TO_F32", "Signed 16-bit lane (`lane0`) converted exactly to FP32."),
    ("S16_TO_S32", "Signed 16-bit lane (`lane0`) sign-extended to 32-bit integer."),
    ("S32_TO_F32", "Signed 32-bit integer converted to FP32 (rounded per `round`)."),
    ("S8_TO_F32", "Signed 8-bit byte lane (`lane0`) converted exactly to FP32."),
    ("S8_TO_S32", "Signed 8-bit byte lane (`lane0`) sign-extended to 32-bit integer."),
    ("U16_TO_F32", "Unsigned 16-bit lane (`lane0`) converted exactly to FP32."),
    ("U16_TO_U32", "Unsigned 16-bit lane (`lane0`) zero-extended to 32-bit integer."),
    ("U32_TO_F32", "Unsigned 32-bit integer converted to FP32 (rounded per `round`)."),
    ("U8_TO_F32", "Unsigned 8-bit byte lane (`lane0`) converted exactly to FP32."),
    ("U8_TO_U32", "Unsigned 8-bit byte lane (`lane0`) zero-extended to 32-bit integer."),
    ("V2F16_TO_V2S16", "Packed `v2f16` input converted per lane to saturating `v2s16`."),
    ("V2F16_TO_V2U16", "Packed `v2f16` input converted per lane to saturating `v2u16`."),
    ("V2S16_TO_V2F16", "Packed `v2s16` input converted per lane to `v2f16`."),
    ("V2S8_TO_V2F16", "Packed `v4s8` input (two bytes selected by `swz0`, default `b02`) converted to `v2f16`."),
    ("V2S8_TO_V2S16", "Packed `v4s8` input (two bytes selected by `swz0`, default `b02`) sign-extended to `v2s16`."),
    ("V2U16_TO_V2F16", "Packed `v2u16` input converted per lane to `v2f16`."),
    ("V2U8_TO_V2F16", "Packed `v4u8` input (two bytes selected by `swz0`, default `b02`) converted to `v2f16`."),
    ("V2U8_TO_V2U16", "Packed `v4u8` input (two bytes selected by `swz0`, default `b02`) zero-extended to `v2u16`."),
):
    _OP_ROLES[_conv_mn] = {
        "dest": "32-bit destination GPR receiving the converted value.",
        "src0": _conv_desc,
    }
_OP_ROLES["V2F32_TO_V2F16"] = {
    "dest": "32-bit destination GPR receiving packed `v2f16(src0, src1)`.",
    "src0": "FP32 input converted into the low halfword (`dest.h0`).",
    "src1": "FP32 input converted into the high halfword (`dest.h1`).",
}

# Populate fused VAR_TEX_* family roles cleanly
for _vmn, _is_buf, _kind in (
    ("VAR_TEX_SINGLE", False, "single texture sample"),
    ("VAR_TEX_DUAL", False, "dual texture sample"),
    ("VAR_TEX_GATHER", False, "2x2 texture gather"),
    ("VAR_TEX_GRADIENT", False, "gradient/LOD texture operation"),
    ("VAR_TEX_BUF_SINGLE", True, "single texture sample"),
    ("VAR_TEX_BUF_DUAL", True, "dual texture sample"),
    ("VAR_TEX_BUF_GATHER", True, "2x2 texture gather"),
    ("VAR_TEX_BUF_GRADIENT", True, "gradient/LOD texture operation"),
):
    _OP_ROLES[_vmn] = {
        "sw": f"Destination staging register range receiving the {_kind} result.",
        "src0": "Texture/sampler descriptor handle.",
        "src2": (
            "Descriptor-less varying buffer byte offset providing interpolated UV coordinates (encoded in `src2`)."
            if _is_buf
            else "Varying descriptor handle providing interpolated UV coordinates (encoded in `src2`)."
        ),
    }


def _field_mapping_for_operand(
    tag: str, name: str, direction: str, field_names: set[str]
) -> str:
    """Return a concise field-mapping hint when the syntax name differs from the XML field."""
    if tag == "dest":
        parts = []
        if "dest_register_index" in field_names:
            parts.append("`dest_register_index`")
        for dw in ("dest_width", "dest_width_narrow", "dest_width_replicate"):
            if dw in field_names:
                parts.append(f"`{dw}`")
        return ", ".join(parts)
    if tag in ("sr", "staging"):
        is_write = direction == "output" or "write" in name or name in ("sw", "sr_dst")
        if is_write and "staging_register_write_index" in field_names:
            cnt = " + `sr_write_count`" if "sr_write_count" in field_names else ""
            return f"`staging_register_write_index`{cnt}"
        parts = []
        if "staging_register_index" in field_names:
            parts.append("`staging_register_index`")
        if "sr_write_count" in field_names and "sr_count" not in field_names:
            parts.append("`sr_write_count`")
        elif "sr_count" in field_names:
            parts.append("`sr_count`")
        return " + ".join(parts)
    if tag == "src":
        return ""  # src0/src1/src2/src3 match their XML field names directly
    # Immediate / label operands
    if name == "offset" and "offset_lo" in field_names and "offset_hi" in field_names:
        return "`offset_lo` + `offset_hi`"
    enc_table = f"encoded_{name}"
    if enc_table in field_names:
        return f"`{enc_table}`"
    if name in field_names:
        return f"`{name}`"
    return ""


def _fmt_op_token(tag: str, name: str, direction: str) -> str:
    if tag == "dest":
        return r"\dst{dest}"
    if tag in ("sr", "staging"):
        is_write = direction == "output" or "write" in name or name in ("sw", "sr_dst")
        return (r"\dst{@" if is_write else r"\src{@") + tex_esc(name) + "}"
    if tag == "src":
        return r"\src{" + tex_esc(name) + "}"
    return r"\imm{\#" + tex_esc(name) + "}"


def render_operands_list_tex(mnem: str, ins_els: ET.Element | list[ET.Element], field_names: set[str]) -> str:
    """Render the Operands section placed immediately after SYNTAX_BODY."""
    if isinstance(ins_els, ET.Element):
        ins_list = [ins_els]
    else:
        ins_list = list(ins_els)
    ops: list[tuple[str, str, str]] = []
    seen_names: set[str] = set()
    # Prefer the variant with the most operands first so canonical order (e.g. dest, src0, src1) is preserved
    for ins_el in sorted(ins_list, key=lambda el: -len(list(el.find("syntax") or []))):
        syn_el = ins_el.find("syntax")
        if syn_el is None:
            continue
        for ch in syn_el:
            if ch.tag == "name":
                continue
            nm = ch.attrib.get("name", ch.tag)
            if nm not in seen_names:
                seen_names.add(nm)
                ops.append((ch.tag, nm, ch.attrib.get("direction", "")))

    if not ops:
        return (
            r"\textbf{Operands:} \textit{None} "
            r"(control/system instruction with no register or immediate operands; configured via modifiers and \texttt{flow})."
        )

    roles: Mapping[str, str] = _OP_ROLES.get(mnem, {})
    lines = [r"\textbf{Operands:}", r"\begin{tightitemize}\footnotesize"]
    for tag, name, direction in ops:
        tok_tex = _fmt_op_token(tag, name, direction)
        fmap = _field_mapping_for_operand(tag, name, direction, field_names)
        fmap_tex = f" ({md_inline_to_tex(fmap)})" if fmap else ""
        desc = roles.get(name)
        if not desc:
            if tag == "dest":
                desc = "Destination general-purpose register."
            elif tag in ("sr", "staging"):
                desc = "Destination staging register range." if direction == "output" else "Source staging register range."
            elif tag == "src":
                desc = f"Source operand `{name}` (GPR, FAU uniform, or inline constant)."
            else:
                desc = f"Immediate operand `{name}`."
        lines.append(f"  \\item \\texttt{{{tok_tex}}}{fmap_tex}: {md_inline_to_tex(desc)}")
    lines.append(r"\end{tightitemize}")
    return "\n".join(lines)

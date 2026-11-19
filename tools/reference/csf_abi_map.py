"""Execution Context (CSF ABI) bullet mappings and syntax-aligned Equivalent GLSL
operational semantics (with flag variants) for every canonical Valhall mnemonic
and relevant Chapter 2 modifier enum.

All Equivalent GLSL snippets are written directly in terms of the instruction's
XML <syntax> operand and modifier names (dest, src0, src1, src2, src3, @sr,
#offset, #imm, cmpf, result_type, saturate, clamp, round, etc.).
"""

from __future__ import annotations

from .fmt_helpers import tex_esc

# ============================================================================
# 1. Syntax-Aligned Equivalent GLSL (Operational Semantics & Flag Variants)
# ============================================================================
_EQUIV_GLSL: dict[str, list[str]] = {
    # 1. Integer Arithmetic
    "BITREV": [
        "dest = bitfieldReverse(src0); // dest[k] = src0[31 - k] for k in 0..31",
    ],
    "CLZ": [
        "// default:       dest[i] = (src0[i] == 0u) ? W : uint(W - 1 - findMSB(src0[i]));",
        "// .mask:         dest[i] = (src0[i] == 0u) ? ~0u : uint(W - 1 - findMSB(src0[i]));",
    ],
    "HADD": [
        "// .round_down:   dest[i] = (ext(src0[i]) + ext(src1[i])) >> 1; // exact (W+1)-bit sum",
        "// .round_up:     dest[i] = (ext(src0[i]) + ext(src1[i]) + 1) >> 1;",
    ],
    "IABS": [
        "dest[i] = (src0[i] < 0) ? uint(-src0[i]) : uint(src0[i]); // wraps INT_MIN -> INT_MIN",
    ],
    "IADD": [
        "// default:       dest[i] = widen0(src0)[i] + widen1(src1)[i]; // wrapping mod 2^W",
        "// .sat (s32):    dest[i] = clamp(int64_t(widen0(src0)[i]) + int64_t(widen1(src1)[i]), MIN_W, MAX_W);",
    ],
    "IADDC": [
        "// default:       dest = uint((uint64_t(src0) + uint64_t(src1) + uint64_t(src2 & 1u)) & 0xFFFFFFFFul);",
        "// .c:            dest = uint((uint64_t(src0) + uint64_t(src1) + uint64_t(src2 & 1u)) >> 32);",
    ],
    "IADD_IMM": [
        "dest[i] = src0[i] + imm1w[i]; // per-element wrapping add with inline 32-bit immediate",
    ],
    "IDPADD": [
        "// default:       dest = src2 + sum_{k=0..3}(ext(src0.b[k]) * ext(src1.b[k])); // wrapping",
        "// .sat:          dest = clamp(int64_t(src2) + int64_t(dot4x8(src0, src1)), MIN_32, MAX_32);",
    ],
    "IMUL": [
        "// default / .sat:dest[i] = sat_or_wrap(widen0(src0)[i] * widen1(src1)[i]);",
        "// .u64 / .s64:   dest = ext64(widen0(src0)) * ext64(widen1(src1)); // 32x32->64 widening",
    ],
    "ISUB": [
        "// default:       dest[i] = widen0(src0)[i] - widen1(src1)[i]; // wrapping mod 2^W",
        "// .sat (u32):    dest[i] = (widen0(src0)[i] >= widen1(src1)[i]) ? (a - b) : 0u;",
    ],
    "ISUBB": [
        "// default:       dest = src0 - src1 - (src2 & 1u);",
        "// .c:            dest = (uint64_t(src0) < uint64_t(src1) + uint64_t(src2 & 1u)) ? 1u : 0u;",
    ],
    "MMUL": [
        "// .v2f16 / .f16: dest = src2 + sum_{k=0..3}(A_sub(src0, sub0)[row,k] * B_sub(src1, sub1)[k,col]);",
        "// .v4s8.sat:     dest = sat32(src2 + sum_{k=0..15}(A(src0)[row,k] * B(src1)[k,col])); // v14",
    ],
    "POPCOUNT": [
        "dest = uint(bitCount(src0)); // exact 32-bit population count in 0..32",
    ],
    "SEG_ADD": [
        "// default:       dest = src0 + widen1(src1);",
        "// .preserve_null:dest = (src0 == 0ul) ? 0ul : (src0 + widen1(src1));",
    ],
    "SEG_SUB": [
        "// default:       dest = src0 - widen1(src1);",
        "// .preserve_null:dest = (src0 == 0ul) ? 0ul : (src0 - widen1(src1));",
    ],
    "SHADD": [
        "// .u64:          dest = src0 + (uint64_t(widen1(src1)) << #shift);",
        "// .s64:          dest = int64_t(src0) + (int64_t(widen1(src1)) << #shift);",
    ],

    # 2. Bitwise (non-shift)
    "AND": [
        "// default:       res[i] = widen0(src0)[i] & (not2 ? ~src2[i] : src2[i]);",
        "// .not_result:   dest[i] = not_result ? ~res[i] : res[i];",
    ],
    "AND_IMM": [
        "dest = src0 & #imm1w; // inline 32-bit bitwise AND (v11+)",
    ],
    "OR": [
        "// default:       res[i] = widen0(src0)[i] | (not2 ? ~src2[i] : src2[i]);",
        "// .not_result:   dest[i] = not_result ? ~res[i] : res[i]; // NOR when .not_result",
    ],
    "OR_IMM": [
        "dest = src0 | #imm1w; // inline 32-bit bitwise OR (v11+)",
    ],
    "XOR": [
        "// default:       res[i] = widen0(src0)[i] ^ (not2 ? ~src2[i] : src2[i]);",
        "// .not_result:   dest[i] = not_result ? ~res[i] : res[i]; // XNOR when .not_result",
    ],
    "XOR_IMM": [
        "dest = src0 ^ #imm1w; // inline 32-bit bitwise XOR (v11+)",
    ],
    "NOT": [
        "dest = ~src0; // bitwise complement",
    ],
    "NOT_OLD": [
        "dest = ~src0; // legacy v9--v13 SFU bitwise complement",
    ],

    # 3. Floating-Point ALU
    "FABS": [
        "dest = uintBitsToFloat(floatBitsToUint(src0) & 0x7FFFFFFFu); // clears IEEE sign bit",
    ],
    "FADD": [
        "// default:       dest[i] = clamp(round_rm(mod(src0)[i] + mod(src1)[i])); // mod = widen/swz -> abs -> neg",
        "// .dest_width:   narrows FP32->FP16 into dest.h0/h1/h01 (or lane-masks v2f16)",
    ],
    "FADD_IMM": [
        "dest[i] = round_rne(src0[i] + uintBitsToFloat(#imm1w)[i]);",
    ],
    "FEXP": [
        "dest = exp2(float(int(src0)) / 16777216.0); // Q8.24 fixed-point src0 + FP32 range src1",
    ],
    "FEXPF": [
        "dest = exp2(src0); // direct FP32 exp2 (v14+)",
    ],
    "FLOG": [
        "dest = log2(src0); // 0 -> -Inf, <0 -> NaN (v12+ f16, v14+ f32)",
    ],
    "FLUSH": [
        "// .ftz:          if (isSubnormal(src0[i])) dest[i] = copysign(0.0, src0[i]);",
        "// .flush_inf:    if (isinf(src0[i]))       dest[i] = copysign(MAX_FINITE, src0[i]);",
        "// .flush_nan:    if (isnan(src0[i]))       dest[i] = +0.0; // .quiet_nan -> src0[i] | qbit",
    ],
    "FMA": [
        "// default:       dest[i] = clamp(round_rm(fma(mod(src0)[i], mod(src1)[i], mod(src2)[i])));",
        "// .dest_width:   Stage C narrows FP32->FP16 (h0/h1/h01) or masks v2f16 halfword",
    ],
    "FMAX": [
        "// .nan_suppress: dest[i] = isnan(a) ? b : (isnan(b) ? a : (totalOrd(a)>totalOrd(b) ? a : b));",
        "// .nan_propagate:dest[i] = (isnan(a)||isnan(b)) ? quieten(winNaN(a,b)) : maxTotal(a,b); // then clamp",
    ],
    "FMIN": [
        "// .nan_suppress: dest[i] = isnan(a) ? b : (isnan(b) ? a : (totalOrd(a)<totalOrd(b) ? a : b));",
        "// .nan_propagate:dest[i] = (isnan(a)||isnan(b)) ? quieten(winNaN(a,b)) : minTotal(a,b); // then clamp",
    ],
    "FMUL": [
        "dest[i] = clamp(round_rm(fma(mod(src0)[i], mod(src1)[i], -0.0))); // preserves -0.0",
    ],
    "FNEG": [
        "dest = uintBitsToFloat(floatBitsToUint(src0) ^ 0x80000000u); // flips IEEE sign bit",
    ],
    "FPCLASS": [
        "dest = classifyIEEE754Mask(src0); // bitmask: sign, zero, subnormal, normal, inf, nan",
    ],
    "FPOW": [
        "dest = placeReplicate(dest_width, pow(lane(src0), lane(src1))); // v12+ scalar f16",
    ],
    "FRCP": [
        "dest = ftz(1.0 / ftz(src0)); // < 1 ULP; input & output subnormals flush to +-0.0 on f32",
    ],
    "FROUND": [
        "// .round_down:   dest[i] = floor(src0[i]); // .round_up: ceil; .round_zero: trunc",
        "// default / .na: dest[i] = roundEven(src0[i]); // .round_na: sign(x)*floor(abs(x)+0.5)",
    ],
    "FRSQ": [
        "dest = ftz(inversesqrt(ftz(src0))); // <= 1 ULP; -0.0 -> -Inf; <0 -> NaN",
    ],
    "FSINCOS": [
        "// .sin:          dest = sin(reconstructAngle(src0, src1)); // v14+",
        "// .cos:          dest = cos(reconstructAngle(src0, src1)); // v14+",
    ],
    "FSQRT": [
        "dest = sqrt(src0); // -0.0 -> -0.0, <0 -> NaN (v14+)",
    ],
    "LDEXP": [
        "// default:       dest[i] = round_rm(ldexp(src0[i], int(src1[i])));",
        "// .inf / .inf0:  overrides +-Inf (exp < -16384) and +-0.0 (exp >= +16384) to +-1.0",
    ],

    # 4. SFU Range-Reduction & Helpers
    "FADD_LSCALE": [
        "dest = mod(src0) + reduceToLogInterval_0_75_to_1_5(src1); // log2(x) reduction",
    ],
    "FADD_RSCALE": [
        "// default:       dest = float((double(src0) + double(src1)) * exp2(double(int(src2))));",
        "// .n_add:        suppresses (+Inf + -Inf) NaN during expm1 range reduction",
    ],
    "FATAN_ASSIST": [
        "dest = packAtanOctantAssist(src0 /*y*/, src1 /*x*/); // packed (i16 octant/exp, f16 seed)",
    ],
    "FATAN_TABLE": [
        "dest = ATAN_OCTANT_OFFSET_LUT[src0 & 0x7u]; // 0, +-pi/2, +-pi",
    ],
    "FCOS_TABLE": [
        "dest = cos(float(src0 & 0x3Fu) * (PI / 32.0)); // 64-entry cosine LUT",
    ],
    "FEXP_TABLE": [
        "// default:       dest = exp2(float(src0 & 0xFu) / 16.0);",
        "// .low:          dest = EXP2_LOW_CORRECTION_LUT[src0 & 0xFu];",
    ],
    "FLOGD": [
        "float m = reduceLog(src0); dest = (m == 1.0) ? (1.0/log(2.0)) : log2(m)/(m - 1.0);",
    ],
    "FLOG_TABLE": [
        "// .base2:        dest = LOG2_POLY_LUT[precision][mantissaIdx(src0)];",
        "// .natural:      dest = LN_POLY_LUT[precision][mantissaIdx(src0)];",
    ],
    "FMA_RSCALE": [
        "// default:       dest = float(fma(double(src0), double(src1), double(src2)) * exp2(double(int(src3))));",
        "// .n:            double p = (src0==0.0||src1==0.0) ? 0.0 : double(src0)*double(src1); dest = float((p+src2)*exp2(src3));",
    ],
    "FMUL_CSLICE": [
        "dest = payneHanekCosSlice(lane(src0), src1); // extended-precision cosine range reduction",
    ],
    "FMUL_RSCALE": [
        "// default:       dest = float((double(src0) * double(src1)) * exp2(double(int(src2))));",
        "// .left:         dest = isZeroInf(src0, src1) ? src0 : float((double(src0)*double(src1))*exp2(src2));",
    ],
    "FMUL_SLICE": [
        "dest = payneHanekSlice(src0, src1); // sliced 2/pi range-reduction product",
    ],
    "FPOW_SC_APPLY": [
        "dest = hasSpecialOverride(src1) ? decodeOverride(src1) : src0;",
    ],
    "FPOW_SC_DET": [
        "// .pow:          dest = detectPowSpecialCase(src0, src1); // pow(0,0)=1, (-x)^odd = -|x|^odd",
        "// .powr:         dest = detectPowrSpecialCase(src0, src1); // powr(0,0)=NaN, x<0 -> NaN",
    ],
    "FRCBRT_APPROX_A": ["dest = approxInvCbrtStageA(src0); // initial x^(-1/3) seed"],
    "FRCBRT_APPROX_B": ["dest = approxInvCbrtStageB(src0); // stage B refinement"],
    "FRCBRT_APPROX_C": ["dest = approxInvCbrtStageC(src0); // stage C refinement"],
    "FRCP_APPROX": ["dest = approxRcp12Bit(src0); // Newton-Raphson 1/x seed"],
    "FREXPE": [
        "// default:       int e; frexp(src0[i], e); dest[i] = neg_result ? -e : e;",
        "// .sqrt / .log:  returns even sqrt exponent (e-125)>>1 or log exponent (e-127)+top_mant_bit",
    ],
    "FREXPM": [
        "// default:       int e; dest[i] = frexp(src0[i], e); // in [0.5, 1.0)",
        "// .log:          dest[i] = isinf(src0[i]) ? copysign(1.0, src0[i]) : normTo_0_75_1_5(src0[i]);",
    ],
    "FRSQ_APPROX": ["dest = approxRsqrt12Bit(src0); // Newton-Raphson 1/sqrt(x) seed"],
    "FSINCOS_OFFSET": [
        "// default:       dest = SINCOS_OFFSET_LUT[src0 & 0x7u];",
        "// .scale:        dest = ldexp(SINCOS_OFFSET_LUT[src0 & 0x7u], 24);",
    ],
    "FSIN_TABLE": [
        "dest = ((src0 & 0x3Fu) == 0u) ? -0.0 : sin(float(src0 & 0x3Fu) * (PI / 32.0));",
    ],
    "ILOGB": [
        "dest[i] = (src0[i] == 0.0) ? INT32_MIN : (isinf(src0[i]) ? INT32_MAX : int(floor(log2(abs(src0[i])))));",
    ],
    "LOGB": [
        "dest[i] = (src0[i] == 0.0) ? -INF : (isinf(src0[i]) ? +INF : floor(log2(abs(src0[i]))));",
    ],

    # 5. Cubemap Helpers
    "CUBEFACE1": [
        "float m = max(abs(src0), max(abs(src1), abs(src2))); dest = isinf(m) ? 1.0 : ((m == 0.0) ? 2.0 : m);",
    ],
    "CUBEFACE2": [
        "dest = (majorAxisFaceIndex(src0, src1, src2) << 29) | (anyInf(src0, src1, src2) ? 0x10000u : 0u);",
    ],
    "CUBE_SSEL": [
        "dest = cubeSelectS(src0 /*Z*/, src1 /*X*/, src2 >> 29 /*face*/); // +X -> -Z, -Z -> -X",
    ],
    "CUBE_TSEL": [
        "dest = cubeSelectT(src0 /*Y*/, src1 /*Z*/, src2 >> 29 /*face*/); // +Y -> +Z, +X -> -Y",
    ],

    # 6. Compare, Select & Mux
    "CSEL": [
        "dest[i] = cmpf(src0[i], src1[i]) ? src2[i] : src3[i]; // cmpf in {eq,ne,lt,le,gt,ge,gtlt,total}",
    ],
    "FCMP": [
        "// For each lane i in 0..N-1: bool c = cmpf(mod(src0)[i], mod(src1)[i]);",
        "// result_type:   .i1 -> c?1u:0u; .m1 -> c?~0u:0u; .f1 -> c?1.0f:0.0f",
    ],
    "FCMP_AND": [
        "// For each lane i in 0..N-1: bool c = cmpf(mod(src0)[i], mod(src1)[i]) && (src2[i] != 0);",
        "// result_type:   .i1 -> c?1u:0u; .m1 -> c?~0u:0u; .f1 -> c?1.0f:0.0f",
    ],
    "FCMP_OR": [
        "// For each lane i in 0..N-1: bool c = cmpf(mod(src0)[i], mod(src1)[i]) || (src2[i] != 0);",
        "// result_type:   .i1 -> c?1u:0u; .m1 -> c?~0u:0u; .f1 -> c?1.0f:0.0f",
    ],
    "ICMP": [
        "// For each lane i in 0..N-1: bool c = cmpf(widen0(src0)[i], widen1(src1)[i]);",
        "// result_type:   .i1 -> c?1u:0u; .m1 -> c?~0u:0u; .f1 -> c?1.0f:0.0f",
    ],
    "ICMP_AND": [
        "// For each lane i in 0..N-1: bool c = cmpf(widen0(src0)[i], widen1(src1)[i]) && (src2[i] != 0);",
        "// result_type:   .i1 -> c?1u:0u; .m1 -> c?~0u:0u; .f1 -> c?1.0f:0.0f",
    ],
    "ICMP_MULTI": [
        "// .c (low word): dest = (src0 < src1) ? -1 : ((src0 > src1) ? +1 : int(src2));",
        "// .i1 (hi word): dest = (cmpf(src0, src1) || (src0 == src1 && carryMatches(cmpf, src2))) ? 1u : 0u;",
    ],
    "ICMP_OR": [
        "// For each lane i in 0..N-1: bool c = cmpf(widen0(src0)[i], widen1(src1)[i]) || (src2[i] != 0);",
        "// result_type:   .i1 -> c?1u:0u; .m1 -> c?~0u:0u; .f1 -> c?1.0f:0.0f",
    ],
    "MUX": [
        "// .int_zero(def):dest[i] = (src2[i] == 0) ? src0[i] : src1[i]; // .neg: (src2<0); .fp_zero: (f32==0.0)",
        "// .bit:          dest[i] = (src0[i] & src2[i]) | (src1[i] & ~src2[i]);",
    ],

    # 7. Conversions
    "F16_TO_F32": ["dest = unpackHalf2x16(src0)[lane0]; // exact IEEE f16 -> f32 promotion"],
    "F16_TO_S32": ["float h = unpackHalf2x16(src0)[lane0]; dest = isnan(h) ? 0 : int(clamp(round_rm(h), INT32_MIN, INT32_MAX));"],
    "F16_TO_U32": ["float h = unpackHalf2x16(src0)[lane0]; dest = (isnan(h) || h < 0.0) ? 0u : uint(clamp(round_rm(h), 0.0, UINT32_MAX));"],
    "F32_TO_F16": [
        "// .h0 / .h1:     dest.lane = floatToHalfBits(round_rm(clamp(src0))); // preserves peer halfword",
        "// .h01:          uint h = floatToHalfBits(round_rm(clamp(src0))); dest = h | (h << 16);",
    ],
    "F32_TO_S32": ["dest = isnan(src0) ? 0 : int(clamp(round_rm(src0), INT32_MIN, INT32_MAX)); // .round_zero=trunc"],
    "F32_TO_U32": ["dest = (isnan(src0) || src0 < 0.0) ? 0u : uint(clamp(round_rm(src0), 0.0, UINT32_MAX));"],
    "S16_TO_F32": ["dest = float(int16_t(lane0(src0))); // exact s16 -> f32"],
    "S16_TO_S32": ["dest = bitfieldExtract(int(src0), lane0 * 16, 16); // sign-extend h0 or h1"],
    "S32_TO_F32": ["dest = round_rm(float(int(src0))); // default RNE; supports .round_up/down/zero/na"],
    "S8_TO_F32": ["dest = float(bitfieldExtract(int(src0), lane0 * 8, 8)); // exact s8 -> f32"],
    "S8_TO_S32": ["dest = bitfieldExtract(int(src0), lane0 * 8, 8); // sign-extend b0..b3"],
    "U16_TO_F32": ["dest = float(bitfieldExtract(src0, lane0 * 16, 16)); // exact u16 -> f32"],
    "U16_TO_U32": ["dest = bitfieldExtract(src0, lane0 * 16, 16); // zero-extend h0 or h1"],
    "U32_TO_F32": ["dest = round_rm(float(uint(src0)));"],
    "U8_TO_F32": ["dest = float(bitfieldExtract(src0, lane0 * 8, 8)); // exact u8 -> f32"],
    "U8_TO_U32": ["dest = bitfieldExtract(src0, lane0 * 8, 8); // zero-extend b0..b3"],
    "V2F16_TO_V2S16": ["dest[i] = isnan(h[i]) ? 0 : int16_t(clamp(round_rm(h[i]), -32768.0, 32767.0)); // i in {0,1}"],
    "V2F16_TO_V2U16": ["dest[i] = (isnan(h[i]) || h[i] < 0.0) ? 0u : uint16_t(clamp(round_rm(h[i]), 0.0, 65535.0));"],
    "V2F32_TO_V2F16": ["dest = packHalf2x16(vec2(round_rm(clamp(src0)), round_rm(clamp(src1)))); // clamp BEFORE narrow"],
    "V2S16_TO_V2F16": ["dest[i] = round_rm(float16_t(int16_t(swz0(src0)[i]))); // i in {0,1}"],
    "V2S8_TO_V2F16": ["dest = f16vec2(float16_t(s8(src0, swz0.x)), float16_t(s8(src0, swz0.y))); // default b02"],
    "V2S8_TO_V2S16": ["dest = i16vec2(int16_t(s8(src0, swz0.x)), int16_t(s8(src0, swz0.y))); // default b02"],
    "V2U16_TO_V2F16": ["dest[i] = round_rm(float16_t(uint16_t(swz0(src0)[i]))); // >=65520 -> +Inf"],
    "V2U8_TO_V2F16": ["dest = f16vec2(float16_t(u8(src0, swz0.x)), float16_t(u8(src0, swz0.y))); // default b02"],
    "V2U8_TO_V2U16": ["dest = maskWrite(dest_width, u16vec2(u8(src0, swz0.x), u8(src0, swz0.y)));"],

    # 8. Move, Pack & Swizzle
    "ADR": ["dest = uint((PC + 8ul + int64_t(#offset)) & 0xFFFFFFFFul); // PC-relative address"],
    "MKVEC": [
        "// .v2i8:         dest = u8(src0, lane0) | (u8(src1, lane1) << 8) | ((src2 & 0xFFFFu) << 16);",
        "// .v2i16:        dest = u16(src0, lane0) | (u16(src1, lane1) << 16);",
    ],
    "MOV": ["dest = maskWrite(dest_width, src0); // full 32-bit or h0/h1 halfword lane mask"],
    "MOV_IMM": ["dest = maskWrite(dest_width, #imm1w); // inline 32-bit constant load"],
    "SWZ": ["dest = u8vec4(src0[lanes0.x], src0[lanes0.y], src0[lanes0.z], src0[lanes0.w]);"],

    # 9. Subgroup & Cross-Lane
    "CLPER": [
        "uint tgt = laneOp(lane_id, src1.b0, subgroup_size); // none/xor/accumulate/shift/rotate/low/prefix",
        "dest = laneActive(tgt) ? laneVal(src0, tgt) : inactiveConst(inactive_result);",
    ],
    "CLPER_OR": ["dest = subgroupPermute(src0, lane_select, src1.b0) | src2; // v14 fused shuffle+OR"],
    "WMASK": ["dest = subgroupBallot(src0 != 0u).x & ((1u << subgroup_size) - 1u);"],

    # 10. Control Flow, Barriers & System
    "BARRIER": ["barrier(); memoryBarrierShared(); // synchronizes all warps in WG_SIZE workgroup"],
    "BRANCH": [
        "bool cond = combine(src0); // none: !=0, h0/h1, and: both!=0, lowbits",
        "if (branch_negate ? !cond : cond) PC = PC + 8 + #offset;",
    ],
    "DISCARD": ["if (cmpf(src0, src1)) discard; // marks active fragment lane discarded/helper"],
    "DOORBELL": ["// privileged HW CSF doorbell notification (no GLSL equivalent)"],
    "EUREKA": ["// privileged HW job-management event signal (no GLSL equivalent)"],
    "JUMP": ["if (combine(src0)) { if (src1 == 0u) return; PC = (address_mode==absolute) ? src1 : PC + 8 + src1; }"],
    "JUMP_EX": [
        "// .call:         hw_stack.push(PC + 8); PC = target(src1, src2);",
        "// .return:       PC = hw_stack.pop(); // .replace: tail-call without push",
    ],
    "KABOOM": ["// hw_trap(KABOOM): raises immediate GPU shader fault (unreachable sentinel)"],
    "NOP": [
        "// .wait*:        scoreboardWait(flow_wait_mask); // stalls until async slots drain",
        "// .end:          return; // terminates shader warp",
    ],

    # 11. Memory Load / Store & LEA
    "LOAD": [
        "// .i8/.i16:      @sr = (extend == .sext) ? sext32(mem[src0 + #offset]) : zext32(mem[src0 + #offset]);",
        "// .i32..i128:    @sr = mem[src0 + #offset]; // access hint: none/istream/estream/force",
    ],
    "STORE": [
        "// general:       mem[src0 + #offset] = @sr; // access hint: none/istream/estream/force",
        "// IDVS graphics: .istream -> TILER_CTX::geometry_buffer (gl_Position); .estream -> VARY_SIZE",
    ],
    "LD_CHECKED": [
        "// default:       @sr = (src1 + width <= src2) ? mem[src0 + src1] : 0;",
        "// .oob_ignore:   if (src1 + width <= src2) @sr = mem[src0 + src1]; // v14 leaves @sr unchanged on OOB",
    ],
    "LD_CHECKED_IMM": ["@sr = (#offset + width <= src1) ? mem[src0 + #offset] : 0;"],
    "ST_CHECKED": ["if (src1 + width <= src2) mem[src0 + src1] = @sr; // OOB silently dropped"],
    "ST_CHECKED_IMM": ["if (#offset + width <= src1) mem[src0 + #offset] = @sr; // OOB silently dropped"],
    "LD_CVT": ["@sr = convertPixelFormat(mem[src0 + #offset], cvt_desc=src1, register_format, vecsize);"],
    "ST_CVT": ["mem[src0 + #offset] = encodePixelFormat(@sr, cvt_desc=src1, register_format, vecsize);"],
    "LD_GCLK": [
        "// .cycle_counter:   @sr = clockARB(); // 64-bit per-core GPU cycle counter",
        "// .system_timestamp:@sr = clockRealtimeEXT(); // 64-bit cross-core timestamp",
    ],
    "LD_MULTI": ["for (int i = 0; i < sr_count; ++i) @sr[i] = mem_u32[src0 + #offset + i*4]; // v14 bulk load"],
    "ST_MULTI": ["for (int i = 0; i < sr_count; ++i) mem_u32[src0 + #offset + i*4] = @sr[i]; // v14 bulk store"],
    "LD_PKA": ["@sr = mem[resolveBufferDesc(src1).address + src0]; // UBO/SSBO descriptor load"],
    "LEA_ATTR": ["(@sr[0:1], @sr[2]) = (attrAddr(handle=src2, vert=src0, inst=src1), attrFormatDesc(src2));"],
    "LEA_ATTR_IMM": ["(@sr[0:1], @sr[2]) = (attrAddr(SRT[#table].Attr[#index], idx=src0), attrFormatDesc);"],
    "LEA_BUF": ["@sr[0:1] = resolveBufferDesc(src1).address + uint64_t(src0);"],
    "LEA_BUF_IMM": ["@sr[0:1] = SRT[#table].Buffer[#index].address + uint64_t(src0);"],
    "LEA_PKA": ["@sr[0:1] = resolvePkaBuffer(src1).address + uint64_t(src0);"],
    "LEA_TEX": ["(@sr[0:1], @sr[2]) = (texelAddr(handle=src2, xy=src0, z=src1), texFormatDesc(src2));"],
    "LEA_TEX_IMM": ["(@sr[0:1], @sr[2]) = (texelAddr(SRT[#table].Tex[#index], xy=src0, z=src1), texFormatDesc);"],

    # 12. Atomics
    "ACMPXCHG": ["@sr[0] = atomicCompSwap(mem[src0 + #offset], @sr[0] /*cmp*/, @sr[1] /*swap*/);"],
    "ATOM": ["atomicOp(atom_opc, mem[src0 + #offset], @sr); // aadd/asmin/asmax/aumin/aumax/aand/aor/axor; no return"],
    "ATOM1": ["atomicUnary(atom1_opc, mem[src0 + #offset]); // ainc/adec/asmax1/aumax1/aor1; no return"],
    "ATOM1_RETURN": ["@sr = atomicUnaryReturn(atom1_opc, mem[src0 + #offset]); // returns old value"],
    "ATOM_RETURN": ["@sr = atomicOpReturn(atom_opc, mem[src0 + #offset], @sr); // returns old value"],
    "AXCHG": ["@sr = atomicExchange(mem[src0 + #offset], @sr); // swaps & returns old value"],

    # 13. Attributes, Varyings & Fragment
    "ATEST": ["@sr = applyAlphaToCoverageAndEarlyZ(src0 /*cov*/, src1 /*alpha*/, src2 /*datum*/); // no implicit discard"],
    "BLEND": ["tileSRAM[rt] = blend(src0 /*blend_desc*/, @sr /*RGBA*/, tileSRAM[rt], cov=src1); // .end terminates warp"],
    "LD_ATTR": ["@sr = fetchAndConvertVertexAttr(handle=src2, vert=src0, inst=src1, register_format, vecsize);"],
    "LD_ATTR_IMM": ["@sr = fetchAndConvertVertexAttr(SRT[#table].Attr[#index], vert=src0, inst=src1);"],
    "LD_TEX": ["@sr = imageLoad(images[src2], ivec2(src0.h0, src0.h1), z=src1);"],
    "LD_TEX_IMM": ["@sr = imageLoad(SRT[#table].Image[#index], ivec2(src0.h0, src0.h1), z=src1);"],
    "LD_TILE": [
        "// default:       @sr = subpassLoad(tileColorSRAM[src2], pixel=src0, cov=src1);",
        "// .z_stencil:    @sr = subpassLoad(tileDepthStencilSRAM, pixel=src0); // .last_read frees tile",
    ],
    "LD_VAR": ["@sr = interpolateVarying(handle=src1, pixel=src0, sample_position, update_mode);"],
    "LD_VAR_BUF": ["@sr = interpolateVaryingBuf(byte_off=src1, pixel=src0, source_format, sample_position);"],
    "LD_VAR_BUF_FLAT": ["@sr = varyingBuf[vertex(src0)][byte_off=src0]; // .explicit_vertex indexes pervertexEXT"],
    "LD_VAR_BUF_FLAT_IMM": ["@sr = varyingBuf[provoking_vtx][#offset]; // v14 immediate flat varying"],
    "LD_VAR_BUF_IMM": ["@sr = interpolateVaryingBuf(byte_off=#offset, pixel=src0, update_mode);"],
    "LD_VAR_FLAT": ["@sr = readFlatVarying(handle=src0, vertex); // .vertex0/1/2 selects triangle vertex"],
    "LD_VAR_FLAT_IMM": ["@sr = readFlatVarying(SRT[#table].Attr[#index], vertex);"],
    "LD_VAR_IMM": ["@sr = interpolateVarying(SRT[#table].Attr[#index], pixel=src0, sample_position, update_mode);"],
    "LD_VAR_SPECIAL": [
        "// .frag_w / .frag_z: @sr = gl_FragCoord.w / gl_FragCoord.z;",
        "// .point / .bary:    @sr = gl_PointCoord / gl_BaryCoordEXT.xy;",
    ],
    "ST_TILE": ["tileColorSRAM[src2][pixel=src0, cov=src1] = encodeRTFormat(@sr, register_format, vecsize);"],
    "ZS_EMIT": [
        "// .z:            gl_FragDepth = src0; @sr = lateDepthTest(src2 /*cov*/, src0);",
        "// .z.stencil:    gl_FragDepth = src0; gl_FragStencilRefARB = int(src1 & 0xFFu); @sr = lateZSTest(...);",
    ],

    # 14. Texture & Varying-Texture
    "TEX_DUAL": ["@sr_write[0:3] = sample(tex0, @sr_read, lod_mode); @sr_write[4:7] = sample(tex1, @sr_read, lod_mode);"],
    "TEX_FETCH": ["@sr_write = texelFetch(src0 /*handle*/, ivec(@sr_read), lod); // .array adds layer"],
    "TEX_GATHER": [
        "// .gather4_{r,g,b,a}: @sr_write = textureGather(src0, uv(@sr_read), gather_component);",
        "// .compare:           @sr_write = textureGather(src0, uv(@sr_read), dref(@sr_read));",
    ],
    "TEX_GRADIENT": [
        "// default:            @sr_write = textureGrad(src0, uv(@sr_read), dPdx(@sr_read), dPdy(@sr_read));",
        "// .disable_lod_*:     @sr_write = textureQueryLod(src0, uv(@sr_read)); // .derivative -> dFdx/dFdy",
    ],
    "TEX_SINGLE": [
        "// .computed/.explicit:@sr_write = texture[Lod](src0, coords(@sr_read), lod_mode); // .skip skips helpers",
        "// .project / .compare:divides by q (textureProj) / compares against Dref (shadow PCF)",
    ],
    "TEX_WEIGHTS": ["@sr_write = bilinearWeights2x2(src0, uv(@sr_read)); // v9 only"],
    "VAR_TEX_BUF_DUAL": ["uv = interpVaryBuf(src2, src1); @sr_write[0:3] = sample(tex0, uv); @sr_write[4:7] = sample(tex1, uv);"],
    "VAR_TEX_BUF_GATHER": ["@sr_write = textureGather(src0, interpVaryBuf(src2, src1), gather_component);"],
    "VAR_TEX_BUF_GRADIENT": ["@sr_write = evalTexGradOrLod(src0, interpVaryBuf(src2, src1));"],
    "VAR_TEX_BUF_SINGLE": ["@sr_write = texture(src0, interpVaryBuf(src2, src1)); // .project/.array/.compare"],
    "VAR_TEX_DUAL": ["uv = interpVaryDesc(src2, src1); @sr_write[0:3] = sample(tex0, uv); @sr_write[4:7] = sample(tex1, uv);"],
    "VAR_TEX_GATHER": ["@sr_write = textureGather(src0, interpVaryDesc(src2, src1), gather_component);"],
    "VAR_TEX_GRADIENT": ["@sr_write = evalTexGradOrLod(src0, interpVaryDesc(src2, src1));"],
    "VAR_TEX_SINGLE": ["@sr_write = texture(src0, interpVaryDesc(src2, src1)); // fused LD_VAR + TEX_SINGLE"],
}

_SHIFT_GLSL = {
    "LSHIFT": "(src0[i] << (amt & (W - 1)))",
    "RSHIFT": "(uint(src0[i]) >> (amt & (W - 1)))",
    "ARSHIFT": "(int(src0[i]) >> (amt & (W - 1)))",
    "LROT": "((src0[i] << s) | (src0[i] >> ((W - s) & (W - 1))))",
    "RROT": "((src0[i] >> s) | (src0[i] << ((W - s) & (W - 1))))",
}
_LOGIC_GLSL = {
    "": "",
    "_AND": " & (not2 ? ~src2[i] : src2[i])",
    "_OR": " | (not2 ? ~src2[i] : src2[i])",
    "_XOR": " ^ (not2 ? ~src2[i] : src2[i])",
}
for _sh, _expr in _SHIFT_GLSL.items():
    for _lop, _lexpr in _LOGIC_GLSL.items():
        _mn = _sh + _lop
        if _mn in _EQUIV_GLSL:
            continue
        _amt_note = "uint amt = extractByte(src1, lanes1); uint s = amt & (W - 1);"
        if _lop:
            _EQUIV_GLSL[_mn] = [
                f"// For each lane i in 0..N-1: {_amt_note}",
                f"// default:       dest[i] = ({_expr}){_lexpr};",
                f"// .not_result:   dest[i] = ~(({_expr}){_lexpr});",
            ]
        else:
            _EQUIV_GLSL[_mn] = [
                f"// For each lane i in 0..N-1: {_amt_note}",
                f"// default / .not:dest[i] = not_result ? ~({_expr}) : ({_expr});",
            ]


def equiv_glsl_tex(mnem: str) -> str:
    """Return LaTeX-escaped Equivalent GLSL lines for *mnem*."""
    return r"\\".join(tex_esc(g) for g in _EQUIV_GLSL[mnem])


# ============================================================================
# 2. Per-Instruction Execution Context Bullets (Split Stage Preloads + Descriptors)
# ============================================================================

_BULLET_COMPUTE_PRELOAD = (
    r"\textbf{Compute stage preloads (\hyperref[sec:csf-spd]{\texttt{SPD\_k}}, \texttt{r16..r23}):} "
    r"Hardware pre-fills \texttt{r55..r57} with \texttt{WorkGroupID.xyz + JOB\_OFFSET} (from "
    r"\hyperref[sec:csf-regs]{\texttt{COMPUTESR::JOB\_OFFSET\_X/Y/Z}} \texttt{r34..r36}) and \texttt{r60..r62} with "
    r"\texttt{LocalInvocationID.xyz} or \texttt{GlobalInvocationID.xyz} (from \texttt{WG\_SIZE} \texttt{r33} \& "
    r"\texttt{JOB\_SIZE} \texttt{r37..r39}, selected by \texttt{ShaderProgram::preload}) so compute shaders can derive "
    r"thread/element indices without memory loads."
)

_BULLET_VERTEX_PRELOAD = (
    r"\textbf{Vertex stage preloads (\hyperref[sec:csf-spd]{\texttt{SPD\_0}/\texttt{SPD\_1}}, \texttt{r16..r19}):} "
    r"The hardware Index Fetcher reads \hyperref[sec:csf-regs]{\texttt{IDVSSR::INDEX\_BUFFER}} (\texttt{r54:r55}) and pre-fills "
    r"\texttt{r59, r60} with \texttt{VertexIndex} (\texttt{+ VERTEX\_OFFSET + GLOBAL\_ATTR\_OFF}) and \texttt{r61} with "
    r"\texttt{InstanceIndex} (\texttt{+ INSTANCE\_OFFSET}) so Position and Varying shaders can pull vertex attributes directly."
)

_BULLET_FRAGMENT_PRELOAD = (
    r"\textbf{Fragment stage preloads \& FP modes (\hyperref[sec:csf-spd]{\texttt{SPD\_2}}, \texttt{r20:r21}):} "
    r"Hardware pre-fills \texttt{r59} with \texttt{floatBitsToUint(FragCoord.z)} and \texttt{r60} with the MSAA sample "
    r"coverage mask (\texttt{gl\_SampleMaskIn[0]}), while \texttt{ShaderProgram} word~0 latches 32/64 GPR allocation and "
    r"warp-wide \texttt{flush\_to\_zero\_mode}, \texttt{suppress\_na\_n}, and \texttt{suppress\_inf}."
)

_BULLET_FAU_INLINE = (
    r"\textbf{Inline FAU \& Special Operands (\hyperref[sec:csf-tsd-fau]{\texttt{FAU\_k}}, \texttt{r8..r15}):} "
    r"Any 8-bit source with \texttt{mode=2} reads zero-latency uniform \texttt{u0..u31} from \texttt{FAU\_k} "
    r"(\texttt{slots64 = bits[63:56]}); \texttt{mode=3} reads \texttt{SmallConstant} (Table~1.2) or \texttt{FauSpecial} "
    r"(\texttt{pc}, \texttt{lane\_id}, \texttt{lane\_width}, \texttt{tls\_ptr}, \texttt{wls\_ptr}, \texttt{blend\_descriptor\_0..7})."
)

_BULLET_SRT_ATTR = (
    r"\textbf{Attribute descriptor chain (\hyperref[sec:csf-srt]{\texttt{IDVSSR::SRT\_0}}, \texttt{r0:r1}):} "
    r"Masks 64B-aligned VA (\texttt{\& \textasciitilde{}0x3F}, \texttt{count=bits[5:0]}) $\to$ \texttt{Resource[table]} (16B) $\to$ "
    r"\hyperref[sec:csf-buf-attr]{\texttt{Attribute[index]}} (32B: \texttt{table}, \texttt{buffer\_index}, \texttt{offset}, "
    r"\texttt{stride}, \texttt{format}) $\to$ \texttt{Resource[Attribute::table]} $\to$ \texttt{Buffer[Attribute::buffer\_index]} "
    r"(32B: \texttt{address}, \texttt{size})."
)

_BULLET_SRT_BUF = (
    r"\textbf{Buffer descriptor chain (\hyperref[sec:csf-srt]{\texttt{SRT\_k}}, \texttt{r0:r1..r6:r7}):} "
    r"Masks 64B-aligned VA (\texttt{\& \textasciitilde{}0x3F}, \texttt{count=bits[5:0]}) $\to$ \texttt{Resource[table]} "
    r"(16B, \texttt{contains\_descriptors=1}) $\to$ \hyperref[sec:csf-buf-attr]{\texttt{Buffer[index]}} (32B, \texttt{type=9}: "
    r"64-bit \texttt{address}, \texttt{size}). Precondition: \texttt{table < count} and \texttt{index*32 < Resource[table].size}."
)

_BULLET_DIRECT_MEM = (
    r"\textbf{Direct 64-bit GPU VA memory/atomics:} Base addresses typically come from "
    r"\hyperref[sec:csf-tsd-fau]{\texttt{FAU\_k}} (\texttt{u0..u31} populated by a Pilot shader chasing "
    r"\texttt{SRT\_0} $\to$ \texttt{Resource[]} $\to$ \texttt{Buffer[]}), \texttt{FauSpecial::tls\_ptr}/\texttt{wls\_ptr} "
    r"backed by \hyperref[sec:csf-tsd-fau]{\texttt{TSD\_k}} (\texttt{r24..r31}), or \texttt{FauSpecial::pc} for embedded constant LUTs."
)

_BULLET_IDVS_STREAM = (
    r"\textbf{Graphics fixed-function streaming (\texttt{STORE.istream}/\texttt{.estream}):} In IDVS Position shaders, "
    r"\texttt{STORE.istream} writes 16B \texttt{gl\_Position} into \hyperref[sec:csf-tiler]{\texttt{TILER\_CTX::geometry\_buffer}} "
    r"(\texttt{r40:r41}); in Varying shaders, \texttt{STORE.estream} writes per-vertex varyings into "
    r"\hyperref[sec:csf-tiler]{\texttt{IDVSSR::VARY\_SIZE}} (\texttt{r48:r49})."
)

_BULLET_CHECKED_MEM = (
    r"\textbf{Hardware bounds check (v12+):} Chases \hyperref[sec:csf-srt]{\texttt{SRT\_k}} / "
    r"\hyperref[sec:csf-tsd-fau]{\texttt{FAU\_k}} base VA and enforces \texttt{offset + width <= Buffer::size} "
    r"(or bound register); OOB loads return \texttt{0} (unless v14 \texttt{.oob\_ignore}) and OOB stores drop silently."
)

_BULLET_TEX = (
    r"\textbf{Texture \& Sampler chain (\hyperref[sec:csf-srt]{\texttt{SRT\_2}} Frag / \texttt{SRT\_0} Comp/Vert):} "
    r"64B-aligned VA (\texttt{\& \textasciitilde{}0x3F}) $\to$ \texttt{Resource[table]} $\to$ "
    r"\hyperref[sec:csf-tex-samp]{\texttt{Texture[tex\_idx]}} (32B, \texttt{type=2}) $\to$ \texttt{GenericPlane[mip]} "
    r"(32B, \texttt{type=10}: \texttt{pointer}, \texttt{row\_stride}, Morton 16$\times$16 / AFBC) + "
    r"\texttt{Sampler[samp\_idx]} (32B, \texttt{type=1}: filter, wrap, LOD clamp/bias, compare)."
)

_BULLET_VAR = (
    r"\textbf{Varying buffer \& barycentric cache (\hyperref[sec:csf-tiler]{\texttt{IDVSSR::VARY\_SIZE}}, \texttt{r48:r49}):} "
    r"Reads per-vertex varying RAM written by Varying EBIN \texttt{STORE.estream}; descriptor forms (\texttt{LD\_VAR*}) also "
    r"chase \hyperref[sec:csf-srt]{\texttt{SRT\_2}} $\to$ \texttt{Resource[table]} $\to$ \texttt{Attribute[index]} and reuse "
    r"cached barycentric weights via \texttt{.store}/\texttt{.retrieve}."
)

_BULLET_TILE = (
    r"\textbf{On-chip 16$\times$16 Tile SRAM (\hyperref[sec:csf-fbd]{\texttt{FRAGMENTSR::FBD\_POINTER}}, \texttt{r40:r41}):} "
    r"64B-aligned VA (\texttt{\& \textasciitilde{}0x3F}, \texttt{type[0]|zs\_ext[1]|rt\_cnt-1[5:2]}) $\to$ "
    r"\texttt{FramebufferParameters} (128B) + optional \texttt{ZSCRCExtension} (64B, for \texttt{.z\_stencil}) + "
    r"\texttt{RenderTarget[rt]} (64B: \texttt{internal\_format}, \texttt{clear\_color}, Morton 16$\times$16 \texttt{writeback\_buffer})."
)

_BULLET_BLEND = (
    r"\textbf{Blend descriptor chain (\hyperref[sec:csf-zsd-blend]{\texttt{IDVSSR::BLEND\_DESC}}, \texttt{r50:r51}):} "
    r"16B-aligned VA (\texttt{\& \textasciitilde{}0xF}, \texttt{count=bits[3:0]}) $\to$ \texttt{Blend[rt]} (16B: fixed-function "
    r"equation or programmable blend shader PC), selected explicitly via \texttt{src0 = FauSpecial::blend\_descriptor\_0..7}."
)

_BULLET_ZSD = (
    r"\textbf{Depth/Stencil \& coverage state (\hyperref[sec:csf-zsd-blend]{\texttt{IDVSSR::ZSD}}, \texttt{r52:r53}):} "
    r"Interacts with 32B \texttt{Depth/stencil} (\texttt{depth\_func}, \texttt{stencil\_from\_shader}), "
    r"\hyperref[sec:csf-tiler]{\texttt{DCD2}} (\texttt{r38}: \texttt{no\_shader\_depth\_read} Early-Z vs Late-Z), "
    r"\texttt{DCD0}/\texttt{DCD1} (\texttt{r57:r58}: \texttt{shader\_modifies\_coverage}, \texttt{alpha\_to\_coverage}), "
    r"and \texttt{FauSpecial::atest\_datum}."
)

_BULLET_TSD = (
    r"\textbf{Thread/Workgroup Local Storage (\hyperref[sec:csf-tsd-fau]{\texttt{TSD\_k}}, \texttt{r24..r31}):} "
    r"32B \texttt{LocalStorage} descriptor (\texttt{tls\_size}, \texttt{wls\_instances}, \texttt{wls\_size\_base/scale}, "
    r"\texttt{tls\_base\_pointer}, \texttt{wls\_base\_pointer}) exposed inline via \texttt{FauSpecial::tls\_ptr}/\texttt{wls\_ptr}."
)

_BULLET_BARRIER = (
    r"\textbf{Workgroup synchronization (\hyperref[sec:csf-regs]{\texttt{COMPUTESR::WG\_SIZE}}, \texttt{r33}):} "
    r"Synchronizes all warps in the \texttt{(x, y, z)} workgroup and orders \hyperref[sec:csf-tsd-fau]{\texttt{TSD\_k}} WLS "
    r"shared-memory visibility. Must encode \texttt{slot7} + \texttt{.wait}."
)

_BULLET_JUMP_EX = (
    r"\textbf{Hardware call-stack requirement (\hyperref[sec:csf-spd]{\texttt{SPD\_k}} \& \hyperref[sec:csf-tsd-fau]{\texttt{TSD\_k}}):} "
    r"Requires \texttt{ShaderProgram::shader\_contains\_jump\_ex = 1} (word~0 bit~29) and per-thread stack backing in \texttt{TSD\_k}."
)

_EXEC_CTX_OVERRIDES: dict[str, list[str]] = {
    "LD_ATTR": [_BULLET_SRT_ATTR, _BULLET_VERTEX_PRELOAD],
    "LD_ATTR_IMM": [_BULLET_SRT_ATTR, _BULLET_VERTEX_PRELOAD],
    "LEA_ATTR": [_BULLET_SRT_ATTR, _BULLET_VERTEX_PRELOAD],
    "LEA_ATTR_IMM": [_BULLET_SRT_ATTR, _BULLET_VERTEX_PRELOAD],
    "LD_PKA": [_BULLET_SRT_BUF, _BULLET_FAU_INLINE],
    "LEA_BUF": [_BULLET_SRT_BUF, _BULLET_FAU_INLINE],
    "LEA_BUF_IMM": [_BULLET_SRT_BUF, _BULLET_FAU_INLINE],
    "LEA_PKA": [_BULLET_SRT_BUF, _BULLET_FAU_INLINE],
    "LOAD": [_BULLET_DIRECT_MEM, _BULLET_FAU_INLINE],
    "STORE": [_BULLET_DIRECT_MEM, _BULLET_IDVS_STREAM, _BULLET_FAU_INLINE],
    "LD_MULTI": [_BULLET_DIRECT_MEM, _BULLET_FAU_INLINE],
    "ST_MULTI": [_BULLET_DIRECT_MEM, _BULLET_FAU_INLINE],
    "ATOM": [_BULLET_DIRECT_MEM, _BULLET_FAU_INLINE],
    "ATOM1": [_BULLET_DIRECT_MEM, _BULLET_FAU_INLINE],
    "ATOM_RETURN": [_BULLET_DIRECT_MEM, _BULLET_FAU_INLINE],
    "ATOM1_RETURN": [_BULLET_DIRECT_MEM, _BULLET_FAU_INLINE],
    "ACMPXCHG": [_BULLET_DIRECT_MEM, _BULLET_FAU_INLINE],
    "AXCHG": [_BULLET_DIRECT_MEM, _BULLET_FAU_INLINE],
    "LD_CHECKED": [_BULLET_CHECKED_MEM, _BULLET_FAU_INLINE],
    "LD_CHECKED_IMM": [_BULLET_CHECKED_MEM, _BULLET_FAU_INLINE],
    "ST_CHECKED": [_BULLET_CHECKED_MEM, _BULLET_FAU_INLINE],
    "ST_CHECKED_IMM": [_BULLET_CHECKED_MEM, _BULLET_FAU_INLINE],
    "LD_CVT": [_BULLET_TEX, _BULLET_FAU_INLINE],
    "ST_CVT": [_BULLET_TEX, _BULLET_FAU_INLINE],
    "LEA_TEX": [_BULLET_TEX, _BULLET_FAU_INLINE],
    "LEA_TEX_IMM": [_BULLET_TEX, _BULLET_FAU_INLINE],
    "LD_TEX": [_BULLET_TEX, _BULLET_FAU_INLINE],
    "LD_TEX_IMM": [_BULLET_TEX, _BULLET_FAU_INLINE],
    "TEX_SINGLE": [_BULLET_TEX, _BULLET_FRAGMENT_PRELOAD],
    "TEX_DUAL": [_BULLET_TEX, _BULLET_FRAGMENT_PRELOAD],
    "TEX_FETCH": [_BULLET_TEX, _BULLET_FAU_INLINE],
    "TEX_GATHER": [_BULLET_TEX, _BULLET_FRAGMENT_PRELOAD],
    "TEX_GRADIENT": [_BULLET_TEX, _BULLET_FRAGMENT_PRELOAD],
    "TEX_WEIGHTS": [_BULLET_TEX, _BULLET_FRAGMENT_PRELOAD],
    "LD_VAR": [_BULLET_VAR, _BULLET_FRAGMENT_PRELOAD],
    "LD_VAR_IMM": [_BULLET_VAR, _BULLET_FRAGMENT_PRELOAD],
    "LD_VAR_BUF": [_BULLET_VAR, _BULLET_FRAGMENT_PRELOAD],
    "LD_VAR_BUF_IMM": [_BULLET_VAR, _BULLET_FRAGMENT_PRELOAD],
    "LD_VAR_BUF_FLAT": [_BULLET_VAR, _BULLET_FRAGMENT_PRELOAD],
    "LD_VAR_BUF_FLAT_IMM": [_BULLET_VAR, _BULLET_FRAGMENT_PRELOAD],
    "LD_VAR_FLAT": [_BULLET_VAR, _BULLET_FRAGMENT_PRELOAD],
    "LD_VAR_FLAT_IMM": [_BULLET_VAR, _BULLET_FRAGMENT_PRELOAD],
    "LD_VAR_SPECIAL": [_BULLET_VAR, _BULLET_FRAGMENT_PRELOAD],
    "VAR_TEX_SINGLE": [_BULLET_VAR, _BULLET_TEX, _BULLET_FRAGMENT_PRELOAD],
    "VAR_TEX_DUAL": [_BULLET_VAR, _BULLET_TEX, _BULLET_FRAGMENT_PRELOAD],
    "VAR_TEX_GATHER": [_BULLET_VAR, _BULLET_TEX, _BULLET_FRAGMENT_PRELOAD],
    "VAR_TEX_GRADIENT": [_BULLET_VAR, _BULLET_TEX, _BULLET_FRAGMENT_PRELOAD],
    "VAR_TEX_BUF_SINGLE": [_BULLET_VAR, _BULLET_TEX, _BULLET_FRAGMENT_PRELOAD],
    "VAR_TEX_BUF_DUAL": [_BULLET_VAR, _BULLET_TEX, _BULLET_FRAGMENT_PRELOAD],
    "VAR_TEX_BUF_GATHER": [_BULLET_VAR, _BULLET_TEX, _BULLET_FRAGMENT_PRELOAD],
    "VAR_TEX_BUF_GRADIENT": [_BULLET_VAR, _BULLET_TEX, _BULLET_FRAGMENT_PRELOAD],
    "LD_TILE": [_BULLET_TILE, _BULLET_FRAGMENT_PRELOAD],
    "ST_TILE": [_BULLET_TILE, _BULLET_FRAGMENT_PRELOAD],
    "BLEND": [_BULLET_BLEND, _BULLET_TILE, _BULLET_FRAGMENT_PRELOAD],
    "ATEST": [_BULLET_ZSD, _BULLET_FRAGMENT_PRELOAD],
    "DISCARD": [_BULLET_ZSD, _BULLET_FRAGMENT_PRELOAD],
    "ZS_EMIT": [_BULLET_ZSD, _BULLET_FRAGMENT_PRELOAD],
    "SEG_ADD": [_BULLET_TSD, _BULLET_FAU_INLINE],
    "SEG_SUB": [_BULLET_TSD, _BULLET_FAU_INLINE],
    "BARRIER": [_BULLET_BARRIER, _BULLET_COMPUTE_PRELOAD],
    "JUMP_EX": [_BULLET_JUMP_EX, _BULLET_FAU_INLINE],
    "JUMP": [
        r"\textbf{Indirect / blend-shader dispatch (\hyperref[sec:csf-zsd-blend]{\texttt{IDVSSR::BLEND\_DESC}}):} "
        r"Used for programmable blend-shader calls/returns (\texttt{Blend[rt]::internal.shader.pc}); jumping to \texttt{0} terminates warp execution.",
        _BULLET_FAU_INLINE,
    ],
    "ADR": [
        r"\textbf{Shader BO base (\hyperref[sec:csf-spd]{\texttt{SPD\_k::binary}}, \texttt{r16..r23}):} "
        r"Computes PC-relative addresses inside the shader binary object (e.g., embedded constant LUTs or blend-shader return labels).",
        _BULLET_FAU_INLINE,
    ],
    "LD_GCLK": [
        r"\textbf{Hardware clock source:} Reads per-core GPU cycle counter or cross-core system timestamp directly (no \texttt{SRT}/\texttt{FAU}/\texttt{TSD} descriptor dereference; tracked on \texttt{slot0..2}).",
    ],
}


def execution_context_bullets(mnem: str, category: str) -> list[str]:
    """Return bulleted LaTeX strings for the Execution Context section of *mnem*."""
    if mnem in _EXEC_CTX_OVERRIDES:
        return _EXEC_CTX_OVERRIDES[mnem]
    if "Floating-Point" in category or "SFU" in category or "Cubemap" in category:
        return [_BULLET_COMPUTE_PRELOAD, _BULLET_VERTEX_PRELOAD, _BULLET_FRAGMENT_PRELOAD, _BULLET_FAU_INLINE]
    return [_BULLET_COMPUTE_PRELOAD, _BULLET_VERTEX_PRELOAD, _BULLET_FAU_INLINE]


def render_execution_context_tex(mnem: str, category: str) -> str:
    """Render the bulleted Execution Context list for *mnem*."""
    bullets = execution_context_bullets(mnem, category)
    return "\n".join(f"  \\item {b}" for b in bullets)


# ============================================================================
# 3. Chapter 2 Enum -> Execution Context Cross-References
# ============================================================================
ENUM_CSF_ABI_NOTES: dict[str, str] = {
    "flow_control_m": (
        r"Scoreboard waits (\texttt{.wait0..wait}) synchronize async \texttt{SRT}/\texttt{LOAD}/\texttt{TEX}/\texttt{LD\_VAR} messages; "
        r"\texttt{.end} signals warp completion to \hyperref[sec:csf-modes]{CSF \texttt{RUN\_COMPUTE}/\texttt{RUN\_IDVS}/\texttt{RUN\_FRAGMENT}} "
        r"and triggers \texttt{CSWAIT} or Tiler/Tile-SRAM writeback."
    ),
    "access_load_m": (
        r"Controls L1/L2 cache allocation on direct 64-bit VA loads (backed by \hyperref[sec:csf-tsd-fau]{\texttt{FAU\_k}} buffer VAs "
        r"or \texttt{TSD\_k} TLS/WLS pools)."
    ),
    "access_store_m": (
        r"In IDVS graphics, \texttt{STORE.istream} routes \texttt{gl\_Position} to "
        r"\hyperref[sec:csf-tiler]{\texttt{TILER\_CTX::geometry\_buffer}} (\texttt{r40:r41}) and \texttt{STORE.estream} writes "
        r"per-vertex varyings into \hyperref[sec:csf-tiler]{\texttt{IDVSSR::VARY\_SIZE}} (\texttt{r48:r49})."
    ),
    "oob_mode_m": (
        r"Works with \hyperref[sec:csf-buf-attr]{\texttt{Buffer::size}} (32B descriptor in \texttt{SRT\_k}) for hardware-checked "
        r"\texttt{robustBufferAccess} bounds enforcement."
    ),
    "register_file_format_general_m": (
        r"Governs staging register conversion against \hyperref[sec:csf-buf-attr]{\texttt{Attribute::format}}, "
        r"\hyperref[sec:csf-fbd]{\texttt{RenderTarget::internal\_format}}, and \hyperref[sec:csf-zsd-blend]{\texttt{Blend[rt]}}."
    ),
    "sample_position_m": (
        r"Reads sample coordinates configured by \hyperref[sec:csf-fbd]{\texttt{FramebufferParameters::sample\_locations}} and "
        r"\hyperref[sec:csf-tiler]{\texttt{DCD0::multisample\_enable}} / \texttt{evaluate\_per\_sample}."
    ),
    "update_mode_m": (
        r"Controls the hardware barycentric cache backed by \hyperref[sec:csf-tiler]{\texttt{IDVSSR::VARY\_SIZE}} (\texttt{r48:r49})."
    ),
    "tex_dimensionality_m": (
        r"Must match \hyperref[sec:csf-tex-samp]{\texttt{Texture::dimension}} (word 0 bits 4:6) fetched via \texttt{SRT\_2}/\texttt{SRT\_0}."
    ),
    "tex_lod_mode_m": (
        r"Interacts with \hyperref[sec:csf-tex-samp]{\texttt{Sampler}} LOD clamp (\texttt{minimum\_lod}/\texttt{maximum\_lod}), "
        r"\texttt{lod\_bias}, and \texttt{Texture::levels} via \texttt{SRT\_2}."
    ),
    "tex_gather_component_m": (
        r"Gathers the selected channel from \hyperref[sec:csf-tex-samp]{\texttt{Texture}} $\to$ \texttt{GenericPlane} (Morton 16$\times$16 detiled) via \texttt{SRT\_2}."
    ),
    "ls_z_stencil_m": (
        r"Targets \hyperref[sec:csf-fbd]{\texttt{FRAGMENTSR::FBD\_POINTER}} (\texttt{zs\_crc\_ext=1}) $\to$ \texttt{ZSCRCExtension} (64B) "
        r"instead of \texttt{RenderTarget[rt]}."
    ),
    "ls_z_last_read_m": (
        r"Signals final read of \hyperref[sec:csf-fbd]{\texttt{RenderTarget[rt]}} on-chip 16$\times$16 Tile SRAM."
    ),
    "z_m": (
        r"Exports shader depth to \hyperref[sec:csf-zsd-blend]{\texttt{IDVSSR::ZSD}} (\texttt{r52:r53}) and requires "
        r"\hyperref[sec:csf-tiler]{\texttt{DCD2::no\_shader\_depth\_read=0}} (Late-Z)."
    ),
    "stencil_m": (
        r"Exports shader stencil ref to \hyperref[sec:csf-zsd-blend]{\texttt{IDVSSR::ZSD}} (\texttt{stencil\_from\_shader=1}, word 0 bit 30)."
    ),
    "segadd_preserve_null_m": (
        r"Preserves null pointers during \hyperref[sec:csf-tsd-fau]{\texttt{TSD\_k}} TLS/WLS segmented address arithmetic."
    ),
    "jump_stack_mode_m": (
        r"Requires \hyperref[sec:csf-spd]{\texttt{SPD\_k::shader\_contains\_jump\_ex=1}} and \hyperref[sec:csf-tsd-fau]{\texttt{TSD\_k}} call-stack backing."
    ),
    "clper_subgroup_size_m": (
        r"Operates within Mali's 16-lane hardware warp (\texttt{FauSpecial::lane\_width=16}, \hyperref[sec:csf-tsd-fau]{Section~\ref{sec:csf-tsd-fau}})."
    ),
}


def csf_abi_for_enum(ename: str) -> str:
    """Return optional Execution Context cross-reference prose for Chapter 2 enum *ename*."""
    return ENUM_CSF_ABI_NOTES.get(ename, "")

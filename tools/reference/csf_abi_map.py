"""Execution Context (CSF ABI) bullet mappings and syntax-aligned Pseudo-GLSL
operational semantics for every canonical Valhall mnemonic and relevant Chapter 2
modifier enum.

All Pseudo-GLSL snippets express operational semantics as GLSL-flavored pseudocode
written directly in terms of each instruction's XML <syntax> operands and
modifiers (dest, @sw, @sr, src0..src3, #offset, #imm1w, #shift, cmpf,
result_type, saturate, clamp, round, etc.).
"""

from __future__ import annotations

from .fmt_helpers import tex_esc

# ============================================================================
# 1. Syntax-Aligned Pseudo-GLSL (Operational Semantics)
# ============================================================================
_PSEUDO_GLSL: dict[str, list[str]] = {
    # 1. Integer Arithmetic
    "BITREV": [
        "uint out_bits = 0u;",
        "for (int k = 0; k < 32; ++k) out_bits |= ((src0 >> k) & 1u) << (31 - k);",
        "dest = out_bits; // equivalent to bitfieldReverse(src0)",
    ],
    "CLZ": [
        "for (int i = 0; i < N; ++i) {",
        "    if (src0[i] == 0u) dest[i] = flag_mask ? ~0u : uint(W);",
        "    else               dest[i] = uint(W - 1 - findMSB(src0[i]));",
        "}",
    ],
    "HADD": [
        "for (int i = 0; i < N; ++i) {",
        "    int64_t sum = ext(src0[i]) + ext(src1[i]) + (round == round_up ? 1 : 0);",
        "    dest[i] = T(sum >> 1); // exact (W+1)-bit sum, no overflow",
        "}",
    ],
    "IABS": [
        "for (int i = 0; i < N; ++i)",
        "    dest[i] = (src0[i] < 0) ? uint(-src0[i]) : uint(src0[i]); // INT_MIN wraps",
    ],
    "IADD": [
        "for (int i = 0; i < N; ++i) {",
        "    auto sum = ext(widen0(src0)[i]) + ext(widen1(src1)[i]);",
        "    dest[i] = flag_sat ? clamp(sum, MIN_W, MAX_W) : wrap_W(sum);",
        "}",
    ],
    "IADDC": [
        "uint64_t sum = uint64_t(src0) + uint64_t(src1) + uint64_t(src2 & 1u);",
        "dest = (return_carry == c) ? uint(sum >> 32) : uint(sum & 0xFFFFFFFFul);",
    ],
    "IADD_IMM": [
        "for (int i = 0; i < N; ++i)",
        "    dest[i] = wrap_W(src0[i] + #imm1w[i]); // MOV_IMM when src0 == k0",
    ],
    "IDPADD": [
        "int64_t acc = int64_t(src2);",
        "for (int k = 0; k < 4; ++k) acc += ext(src0.b[k]) * ext(src1.b[k]);",
        "dest = flag_sat ? clamp(acc, MIN_32, MAX_32) : wrap_32(acc);",
    ],
    "IMUL": [
        "if (is_64bit) dest = ext64(widen0(src0)) * ext64(widen1(src1)); // 32x32->64",
        "else for (int i = 0; i < N; ++i) {",
        "    auto prod = ext(widen0(src0)[i]) * ext(widen1(src1)[i]);",
        "    dest[i] = flag_sat ? clamp(prod, MIN_W, MAX_W) : wrap_W(prod);",
        "}",
    ],
    "ISUB": [
        "for (int i = 0; i < N; ++i) {",
        "    auto diff = ext(widen0(src0)[i]) - ext(widen1(src1)[i]);",
        "    dest[i] = flag_sat ? clamp(diff, MIN_W, MAX_W) : wrap_W(diff);",
        "}",
    ],
    "ISUBB": [
        "uint64_t rhs = uint64_t(src1) + uint64_t(src2 & 1u);",
        "if (return_carry == c) dest = (uint64_t(src0) < rhs) ? 1u : 0u;",
        "else                   dest = uint((uint64_t(src0) - rhs) & 0xFFFFFFFFul);",
    ],
    "MMUL": [
        "// Cooperative subgroup matrix multiply-accumulate across warp lanes:",
        "mat_tile D = mat_tile(src2) + submat(src0, sub0) * submat(src1, sub1);",
        "dest = flag_sat ? saturate(D)[lane_slice] : D[lane_slice];",
    ],
    "POPCOUNT": [
        "dest = uint(bitCount(src0)); // exact count of 1-bits in 0..32",
    ],
    "SEG_ADD": [
        "if (flag_preserve_null && src0 == 0ul) dest = 0ul;",
        "else                                   dest = src0 + widen64(src1);",
    ],
    "SEG_SUB": [
        "if (flag_preserve_null && src0 == 0ul) dest = 0ul;",
        "else                                   dest = src0 - widen64(src1);",
    ],
    "SHADD": [
        "dest = src0 + (widen64(src1) << #shift); // #shift in 0..7, wraps mod 2^64",
    ],

    # 2. Bitwise (non-shift)
    "AND": [
        "for (int i = 0; i < N; ++i) {",
        "    auto b = flag_not2 ? ~src2[i] : src2[i];",
        "    auto r = widen0(src0)[i] & b;",
        "    dest[i] = flag_not_result ? ~r : r;",
        "}",
    ],
    "AND_IMM": [
        "dest = maskWrite(dest_width, src0 & #imm1w);",
    ],
    "OR": [
        "for (int i = 0; i < N; ++i) {",
        "    auto b = flag_not2 ? ~src2[i] : src2[i];",
        "    auto r = widen0(src0)[i] | b;",
        "    dest[i] = flag_not_result ? ~r : r; // NOR when .not_result",
        "}",
    ],
    "OR_IMM": [
        "dest = maskWrite(dest_width, src0 | #imm1w);",
    ],
    "XOR": [
        "for (int i = 0; i < N; ++i) {",
        "    auto b = flag_not2 ? ~src2[i] : src2[i];",
        "    auto r = widen0(src0)[i] ^ b;",
        "    dest[i] = flag_not_result ? ~r : r; // XNOR when .not_result",
        "}",
    ],
    "XOR_IMM": [
        "dest = maskWrite(dest_width, src0 ^ #imm1w);",
    ],
    "NOT": [
        "dest = ~widen0(src0); // alias of OR(src0, 0) with .not_result",
    ],
    "NOT_OLD": [
        "dest = ~src0; // legacy v9--v13 bitwise complement",
    ],

    # 3. Floating-Point ALU
    "FABS": [
        "float v = uintBitsToFloat(floatBitsToUint(src0) & 0x7FFFFFFFu);",
        "dest = narrowOrMask(dest_width_narrow, clamp_mode(round_rm(v)));",
    ],
    "FADD": [
        "for (int i = 0; i < N; ++i) {",
        "    auto r = clamp_mode(round_rm(mod(src0)[i] + mod(src1)[i]));",
        "    dest[i] = narrowOrMask(dest_width_narrow, r);",
        "}",
    ],
    "FADD_IMM": [
        "for (int i = 0; i < N; ++i)",
        "    dest[i] = round_rne(src0[i] + #imm1w[i]);",
    ],
    "FEXP": [
        "if (is_f32) dest = ftz_out(specialCaseFexp(src1, exp2(float(int(src0)) / 16777216.0)));",
        "else        dest = replicate(dest_width_replicate, exp2(lane1(src1))); // v12+ f16",
    ],
    "FEXPF": [
        "dest = exp2(src0); // direct FP32 exp2 (v14+)",
    ],
    "FLOG": [
        "dest = (src0 == 0.0) ? -INF : ((src0 < 0.0) ? NaN : log2(src0));",
    ],
    "FLUSH": [
        "for (int i = 0; i < N; ++i) {",
        "    auto x = src0[i];",
        "    if (flag_ftz && isSubnormal(x))   x = copysign(0.0, x);",
        "    if (flag_flush_inf && isinf(x))   x = copysign(MAX_FINITE, x);",
        "    if (flag_flush_nan && isnan(x))   x = +0.0;",
        "    else if (flag_quiet_nan && isnan(x)) x = quieten(x);",
        "    dest[i] = x;",
        "}",
    ],
    "FMA": [
        "for (int i = 0; i < N; ++i) {",
        "    auto r = clamp_mode(round_rm(fma(mod(src0)[i], mod(src1)[i], mod(src2)[i])));",
        "    dest[i] = narrowOrMask(dest_width_narrow, r);",
        "}",
    ],
    "FMAX": [
        "for (int i = 0; i < N; ++i) {",
        "    auto a = mod(src0)[i], b = mod(src1)[i];",
        "    auto r = flag_nan_suppress ? (isnan(a) ? b : (isnan(b) ? a : maxTotal(a, b)))",
        "                               : ((isnan(a) || isnan(b)) ? quieten(winNaN(a, b)) : maxTotal(a, b));",
        "    dest[i] = clamp_mode(r);",
        "}",
    ],
    "FMIN": [
        "for (int i = 0; i < N; ++i) {",
        "    auto a = mod(src0)[i], b = mod(src1)[i];",
        "    auto r = flag_nan_suppress ? (isnan(a) ? b : (isnan(b) ? a : minTotal(a, b)))",
        "                               : ((isnan(a) || isnan(b)) ? quieten(winNaN(a, b)) : minTotal(a, b));",
        "    dest[i] = clamp_mode(r);",
        "}",
    ],
    "FMUL": [
        "for (int i = 0; i < N; ++i)",
        "    dest[i] = clamp_mode(round_rm(fma(mod(src0)[i], mod(src1)[i], -0.0)));",
    ],
    "FNEG": [
        "float v = uintBitsToFloat(floatBitsToUint(src0) ^ 0x80000000u);",
        "dest = narrowOrMask(dest_width_narrow, clamp_mode(round_rm(v)));",
    ],
    "FPCLASS": [
        "dest = classifyIEEE754Bitmask(src0); // bits: sign, zero, subnormal, normal, inf, qnan, snan",
    ],
    "FPOW": [
        "dest = replicate(dest_width_replicate, pow(lane0(src0), lane1(src1))); // v12+ f16",
    ],
    "FRCP": [
        "if (is_f32) dest = ftz(1.0 / ftz(src0)); // < 1 ULP, flushes subnormals",
        "else        dest = float16_t(1.0h / lane0(src0)); // preserves f16 subnormals",
    ],
    "FROUND": [
        "for (int i = 0; i < N; ++i) {",
        "    if      (round == round_down) dest[i] = floor(src0[i]);",
        "    else if (round == round_up)   dest[i] = ceil(src0[i]);",
        "    else if (round == round_zero) dest[i] = trunc(src0[i]);",
        "    else if (round == round_na)   dest[i] = sign(src0[i]) * floor(abs(src0[i]) + 0.5);",
        "    else                          dest[i] = roundEven(src0[i]);",
        "}",
    ],
    "FRSQ": [
        "float x = is_f32 ? ftz(src0) : float(lane0(src0));",
        "dest = (x < 0.0) ? NaN : ((x == 0.0) ? copysign(INF, x) : inversesqrt(x));",
    ],
    "FSINCOS": [
        "float theta = reconstructAngle(src0, src1);",
        "dest = (func == sin) ? sin(theta) : cos(theta); // v14+",
    ],
    "FSQRT": [
        "dest = (src0 < 0.0) ? NaN : sqrt(src0); // preserves -0.0 -> -0.0 (v14+)",
    ],
    "LDEXP": [
        "for (int i = 0; i < N; ++i) {",
        "    int e = int(src1[i]);",
        "    if (flag_inf_override && ((isinf(src0[i]) && e < -16384) || (src0[i] == 0.0 && e >= 16384)))",
        "        dest[i] = copysign(1.0, src0[i]);",
        "    else dest[i] = round_rm(ldexp(src0[i], e));",
        "}",
    ],

    # 4. SFU Range-Reduction & Helpers
    "FADD_LSCALE": [
        "float m = isNormal(src1) ? reduceTo_0_75_to_1_5(src1) : src1;",
        "dest = clamp_mode(round_rm(mod(src0) + m)); // src0=-1.0 yields m - 1",
    ],
    "FADD_RSCALE": [
        "double s = double(src0) + double(src2);",
        "if (special == n_add && isinf(src0) && isinf(src2) && sign(src0) != sign(src2)) s = 0.0;",
        "dest = clamp_mode(round_rm(float(s * exp2(double(clamp(int(src3), -512, 512))))));",
    ],
    "FATAN_ASSIST": [
        "dest = packAtanOctantAssist(src0 /*y*/, src1 /*x*/); // (i16 octant/exp, f16 seed)",
    ],
    "FATAN_TABLE": [
        "dest = ATAN_OCTANT_OFFSET_LUT[src0 & 0x7u]; // 0, +-pi/2, +-pi",
    ],
    "FCOS_TABLE": [
        "if (isnan(src0))      dest = src0;",
        "else if (isinf(src0)) dest = NaN;",
        "else if (signExp(src0) != (0, 146)) dest = -0.0;",
        "else { uint idx = floatBitsToUint(src0) & 0x3Fu; dest = cos(float(idx) * PI / 32.0); }",
    ],
    "FEXP_TABLE": [
        "uint idx = src0 & 0xFu;",
        "dest = (adj == low) ? EXP2_LOW_CORRECTION_LUT[idx] : exp2(float(idx) / 16.0);",
    ],
    "FLOGD": [
        "float m = reduceTo_0_75_to_1_5(src0);",
        "dest = (m == 1.0) ? (1.0 / log(2.0)) : (log2(m) / (m - 1.0));",
    ],
    "FLOG_TABLE": [
        "uint idx = topMantissaBits(src0);",
        "dest = (base == natural) ? LN_POLY_LUT[precision][idx] : LOG2_POLY_LUT[precision][idx];",
    ],
    "FMA_RSCALE": [
        "double p = (special == n && (src0 == 0.0 || src1 == 0.0)) ? copysign(0.0, src0 * src1)",
        "                                                         : double(src0) * double(src1);",
        "int e = clamp(int(src3), -512, 512);",
        "dest = (special == left && isZeroTimesInf(src0, src1)) ? src0",
        "                                                       : clamp_mode(round_rm(float((p + double(src2)) * exp2(double(e)))));",
    ],
    "FMUL_CSLICE": [
        "dest = payneHanekCosSlice(mod(lane0(src0)), src1);",
    ],
    "FMUL_RSCALE": [
        "int e = clamp(int(src3), -512, 512);",
        "if (special == left && isZeroTimesInf(src0, src1)) dest = src0;",
        "else dest = clamp_mode(round_rm(float(double(src0) * double(src1) * exp2(double(e)))));",
    ],
    "FMUL_SLICE": [
        "dest = payneHanekTwoOverPiSlice(src0, src1);",
    ],
    "FPOW_SC_APPLY": [
        "dest = hasSpecialOverride(src1) ? decodeSpecialOverride(src1) : src0;",
    ],
    "FPOW_SC_DET": [
        "dest = (func == powr) ? detectPowrSpecialCase(src0, src1) : detectPowSpecialCase(src0, src1);",
    ],
    "FRCBRT_APPROX_A": ["dest = approxInvCbrtStageA(src0); // initial x^(-1/3) seed"],
    "FRCBRT_APPROX_B": ["dest = approxInvCbrtStageB(src0); // stage-B refinement"],
    "FRCBRT_APPROX_C": ["dest = approxInvCbrtStageC(src0); // stage-C refinement"],
    "FRCP_APPROX": ["dest = approxRcp12Bit(src0); // Newton-Raphson 1/x seed"],
    "FREXPE": [
        "for (int i = 0; i < N; ++i) {",
        "    int e; frexp(src0[i], e);",
        "    if (mode == sqrt) e = (e - 125) >> 1;",
        "    else if (mode == log) e = (e - 127) + (topMantBit(src0[i]) ? 1 : 0);",
        "    dest[i] = flag_neg_result ? -e : e;",
        "}",
    ],
    "FREXPM": [
        "for (int i = 0; i < N; ++i) {",
        "    if (mode == log) dest[i] = isinf(src0[i]) ? copysign(1.0, src0[i]) : reduceTo_0_75_to_1_5(src0[i]);",
        "    else { int e; dest[i] = frexp(src0[i], e); } // in [0.5, 1.0)",
        "}",
    ],
    "FRSQ_APPROX": ["dest = approxRsqrt12Bit(src0); // Newton-Raphson 1/sqrt(x) seed"],
    "FSINCOS_OFFSET": [
        "float off = SINCOS_OFFSET_LUT[src0 & 0x7u];",
        "dest = (scale == scale_24) ? ldexp(off, 24) : off;",
    ],
    "FSIN_TABLE": [
        "if (isnan(src0))      dest = src0;",
        "else if (isinf(src0)) dest = NaN;",
        "else if (signExp(src0) != (0, 146)) dest = -0.0;",
        "else { uint idx = floatBitsToUint(src0) & 0x3Fu; dest = (idx == 0u) ? -0.0 : sin(float(idx) * PI / 32.0); }",
    ],
    "ILOGB": [
        "for (int i = 0; i < N; ++i) {",
        "    if (src0[i] == 0.0)    dest[i] = INT32_MIN;",
        "    else if (isinf(src0[i])) dest[i] = INT32_MAX;",
        "    else                   dest[i] = int(floor(log2(abs(src0[i]))));",
        "}",
    ],
    "LOGB": [
        "for (int i = 0; i < N; ++i)",
        "    dest[i] = (src0[i] == 0.0) ? -INF : (isinf(src0[i]) ? +INF : floor(log2(abs(src0[i]))));",
    ],

    # 5. Cubemap Helpers
    "CUBEFACE1": [
        "float m = max(abs(src0), max(abs(src1), abs(src2)));",
        "if (isnan(src0) || isnan(src1) || isnan(src2)) dest = NaN;",
        "else if (isinf(m)) dest = 1.0;",
        "else if (m < FLT_MIN) dest = 2.0;",
        "else dest = m;",
    ],
    "CUBEFACE2": [
        "uint face = majorAxisFaceIndex(src0, src1, src2); // |z| >= |y| >= |x|; NaN -> 4",
        "uint inf_bit = (anyInf(src0, src1, src2) && !anyNaN(src0, src1, src2)) ? 0x10000u : 0u;",
        "dest = (face << 29) | inf_bit;",
    ],
    "CUBE_SSEL": [
        "uint face = src2 >> 29; // src0 = z, src1 = x",
        "dest = (face == 0u) ? -src0 : (face == 1u) ? src0 : (face == 5u) ? -src1 : src1;",
    ],
    "CUBE_TSEL": [
        "uint face = src2 >> 29; // src0 = y, src1 = z",
        "dest = (face == 2u) ? src1 : (face == 3u) ? -src1 : -src0;",
    ],

    # 6. Compare, Select & Mux
    "CSEL": [
        "for (int i = 0; i < N; ++i)",
        "    dest[i] = cmpf(src0[i], src1[i]) ? src2[i] : src3[i];",
    ],
    "FCMP": [
        "for (int i = 0; i < N; ++i) {",
        "    bool c = cmpf(mod(src0)[i], mod(src1)[i]);",
        "    dest[i] = (result_type == m1) ? (c ? ~0u : 0u) : (result_type == f1 ? (c ? 1.0 : 0.0) : (c ? 1u : 0u));",
        "}",
    ],
    "FCMP_AND": [
        "for (int i = 0; i < N; ++i) {",
        "    bool c = cmpf(mod(src0)[i], mod(src1)[i]) && (src2[i] != 0);",
        "    dest[i] = formatResult(result_type, c); // .i1: 1/0, .m1: ~0/0, .f1: 1.0/0.0",
        "}",
    ],
    "FCMP_OR": [
        "for (int i = 0; i < N; ++i) {",
        "    bool c = cmpf(mod(src0)[i], mod(src1)[i]) || (src2[i] != 0);",
        "    dest[i] = formatResult(result_type, c); // .i1: 1/0, .m1: ~0/0, .f1: 1.0/0.0",
        "}",
    ],
    "ICMP": [
        "for (int i = 0; i < N; ++i) {",
        "    bool c = cmpf(widen0(src0)[i], widen1(src1)[i]);",
        "    dest[i] = formatResult(result_type, c);",
        "}",
    ],
    "ICMP_AND": [
        "for (int i = 0; i < N; ++i) {",
        "    bool c = cmpf(widen0(src0)[i], widen1(src1)[i]) && (src2[i] != 0);",
        "    dest[i] = formatResult(result_type, c);",
        "}",
    ],
    "ICMP_MULTI": [
        "int ord = (src0 < src1) ? -1 : ((src0 > src1) ? +1 : int(src2));",
        "if (result_type == c) dest = uint(ord); // 3-way carry to next word",
        "else                  dest = formatResult(result_type, matchesCmpf(cmpf, ord));",
    ],
    "ICMP_OR": [
        "for (int i = 0; i < N; ++i) {",
        "    bool c = cmpf(widen0(src0)[i], widen1(src1)[i]) || (src2[i] != 0);",
        "    dest[i] = formatResult(result_type, c);",
        "}",
    ],
    "MUX": [
        "for (int i = 0; i < N; ++i) {",
        "    if (mux == bit)          dest[i] = (src0[i] & src2[i]) | (src1[i] & ~src2[i]);",
        "    else if (mux == neg)     dest[i] = (int(src2[i]) < 0) ? src0[i] : src1[i];",
        "    else if (mux == fp_zero) dest[i] = (asFloat(src2[i]) == 0.0) ? src0[i] : src1[i];",
        "    else                     dest[i] = (src2[i] == 0u) ? src0[i] : src1[i]; // .int_zero",
        "}",
    ],

    # 7. Conversions
    "F16_TO_F32": ["dest = unpackHalf2x16(src0)[lane0]; // exact f16 -> f32 widening"],
    "F16_TO_S32": [
        "float h = unpackHalf2x16(src0)[lane0];",
        "dest = isnan(h) ? 0 : int(clamp(round_rm(h), float(INT32_MIN), float(INT32_MAX)));",
    ],
    "F16_TO_U32": [
        "float h = unpackHalf2x16(src0)[lane0];",
        "dest = (isnan(h) || h <= 0.0) ? 0u : uint(clamp(round_rm(h), 0.0, float(UINT32_MAX)));",
    ],
    "F32_TO_F16": [
        "uint h = floatToHalfBits(round_rm(clamp_mode(src0)));",
        "if (dest_width_narrow == h01) dest = h | (h << 16);",
        "else                          dest = writeHalf(dest, dest_width_narrow, h);",
    ],
    "F32_TO_S32": [
        "if (isnan(src0)) dest = 0;",
        "else             dest = int(clamp(round_rm(src0), double(INT32_MIN), double(INT32_MAX)));",
    ],
    "F32_TO_U32": [
        "if (isnan(src0) || src0 <= 0.0) dest = 0u;",
        "else                            dest = uint(clamp(round_rm(src0), 0.0, double(UINT32_MAX)));",
    ],
    "S16_TO_F32": ["dest = float(bitfieldExtract(int(src0), lane0 * 16, 16));"],
    "S16_TO_S32": ["dest = bitfieldExtract(int(src0), lane0 * 16, 16);"],
    "S32_TO_F32": ["dest = round_rm(float(int(widen0(src0))));"],
    "S8_TO_F32": ["dest = float(bitfieldExtract(int(src0), lane0 * 8, 8));"],
    "S8_TO_S32": ["dest = bitfieldExtract(int(src0), lane0 * 8, 8);"],
    "U16_TO_F32": ["dest = float(bitfieldExtract(uint(src0), lane0 * 16, 16));"],
    "U16_TO_U32": ["dest = bitfieldExtract(uint(src0), lane0 * 16, 16);"],
    "U32_TO_F32": ["dest = round_rm(float(uint(widen0(src0))));"],
    "U8_TO_F32": ["dest = float(bitfieldExtract(uint(src0), lane0 * 8, 8));"],
    "U8_TO_U32": ["dest = bitfieldExtract(uint(src0), lane0 * 8, 8);"],
    "V2F16_TO_V2S16": [
        "vec2 h = unpackHalf2x16(src0);",
        "for (int i = 0; i < 2; ++i) dest[i] = isnan(h[i]) ? 0 : int16_t(clamp(round_rm(h[i]), -32768.0, 32767.0));",
    ],
    "V2F16_TO_V2U16": [
        "vec2 h = unpackHalf2x16(src0);",
        "for (int i = 0; i < 2; ++i) dest[i] = (isnan(h[i]) || h[i] <= 0.0) ? 0u : uint16_t(clamp(round_rm(h[i]), 0.0, 65535.0));",
    ],
    "V2F32_TO_V2F16": [
        "dest = packHalf2x16(vec2(round_rm(clamp_mode(src0)), round_rm(clamp_mode(src1))));",
    ],
    "V2S16_TO_V2F16": [
        "for (int i = 0; i < 2; ++i) dest[i] = round_rm(float16_t(int16_t(swz0(src0)[i])));",
    ],
    "V2S8_TO_V2F16": [
        "dest = f16vec2(float16_t(s8(src0, swz0.x)), float16_t(s8(src0, swz0.y)));",
    ],
    "V2S8_TO_V2S16": [
        "dest = i16vec2(int16_t(s8(src0, swz0.x)), int16_t(s8(src0, swz0.y)));",
    ],
    "V2U16_TO_V2F16": [
        "for (int i = 0; i < 2; ++i) dest[i] = round_rm(float16_t(uint16_t(swz0(src0)[i])));",
    ],
    "V2U8_TO_V2F16": [
        "dest = f16vec2(float16_t(u8(src0, swz0.x)), float16_t(u8(src0, swz0.y)));",
    ],
    "V2U8_TO_V2U16": [
        "dest = maskWrite(dest_width, u16vec2(u8(src0, swz0.x), u8(src0, swz0.y)));",
    ],

    # 8. Move, Pack & Swizzle
    "ADR": ["dest = uint((PC + 8ul + int64_t(#imm1w)) & 0xFFFFFFFFul);"],
    "MKVEC": [
        "if (is_v2i8) dest = u8(src0, lane0) | (u8(src1, lane1) << 8) | ((src2 & 0xFFFFu) << 16);",
        "else         dest = u16(src0, lane0) | (u16(src1, lane1) << 16); // v2i16",
    ],
    "MOV": ["dest = maskWrite(dest_width, src0);"],
    "MOV_IMM": ["dest = maskWrite(dest_width, #imm1w);"],
    "SWZ": [
        "for (int i = 0; i < N; ++i) dest[i] = src0[lanes0[i]];",
    ],

    # 9. Subgroup & Cross-Lane
    "CLPER": [
        "uint tgt = laneOp(lane_id, src1 & 0xFFu, subgroup_size); // none/xor/accumulate/shift/rotate/low/prefix",
        "dest = laneActive(tgt) ? readLane(src0, tgt) : inactiveConst(inactive_result);",
    ],
    "CLPER_OR": [
        "uint tgt = laneOp(lane_id, src1 & 0xFFu, subgroup_size);",
        "dest = (laneActive(tgt) ? readLane(src0, tgt) : inactiveConst(inactive_result)) | src2;",
    ],
    "WMASK": [
        "uint mask = 0u;",
        "for (uint k = 0u; k < subgroup_size; ++k) if (laneActive(k) && readLane(src0, k) != 0u) mask |= (1u << k);",
        "dest = mask;",
    ],

    # 10. Control Flow, Barriers & System
    "BARRIER": [
        "barrier();",
        "memoryBarrierShared(); // synchronize all warps in WG_SIZE workgroup",
    ],
    "BRANCH": [
        "bool cond = combine(src0); // none: !=0, h0/h1, and: both!=0, lowbits",
        "if (flag_not ? !cond : cond) PC = PC + 8 + #offset;",
    ],
    "DISCARD": [
        "if (cmpf(src0, src1)) discard; // mark active fragment lane discarded/helper",
    ],
    "DOORBELL": [
        "csf_doorbell(); // signal privileged CSF firmware doorbell",
    ],
    "EUREKA": [
        "csf_signal_event(); // signal privileged CSF job-management event",
    ],
    "JUMP": [
        "bool cond = combine(src0);",
        "if (flag_not ? !cond : cond) {",
        "    if (src1 == 0u) return; // terminate shader warp",
        "    PC = (address_mode == absolute) ? src1 : (PC + 8 + src1);",
        "}",
    ],
    "JUMP_EX": [
        "if (stack_mode == call)        { hw_stack.push(PC + 8); PC = resolveTarget(src1, src2); }",
        "else if (stack_mode == ret)    { PC = hw_stack.pop(); }",
        "else /* replace (tail-call) */ { PC = resolveTarget(src1, src2); }",
    ],
    "KABOOM": [
        "gpu_fault_trap(KABOOM); // raise immediate shader fault",
    ],
    "NOP": [
        "if (flow_wait_mask != 0u) scoreboardWait(flow_wait_mask);",
        "if (flow == end)          return; // terminate shader warp",
    ],

    # 11. Memory Load / Store & LEA
    "LOAD": [
        "uint64_t addr = src0 + int64_t(#offset);",
        "if (width < 32) @sw = (extend == sext) ? sext32(mem[addr]) : zext32(mem[addr]);",
        "else            @sw = mem_read(addr, width, access_hint);",
    ],
    "STORE": [
        "uint64_t addr = src0 + int64_t(#offset);",
        "mem_write(addr, @sr, width, access_hint); // .istream/.estream for IDVS outputs",
    ],
    "LD_CHECKED": [
        "if (uint64_t(src1) + access_bytes <= uint64_t(src2)) @sw = mem_read(src0 + src1, width);",
        "else if (!flag_oob_ignore)                           @sw = 0;",
    ],
    "LD_CHECKED_IMM": [
        "if (uint64_t(#offset) + access_bytes <= uint64_t(src2)) @sw = mem_read(src0 + #offset, width);",
        "else if (!flag_oob_ignore)                              @sw = 0;",
    ],
    "ST_CHECKED": [
        "if (uint64_t(src1) + access_bytes <= uint64_t(src2)) mem_write(src0 + src1, @sr, width);",
    ],
    "ST_CHECKED_IMM": [
        "if (uint64_t(#offset) + access_bytes <= uint64_t(src2)) mem_write(src0 + #offset, @sr, width);",
    ],
    "LD_CVT": [
        "@sw = decodePixelFormat(mem_read(src0 + #offset), cvt_desc = src2, register_format, vecsize);",
    ],
    "ST_CVT": [
        "mem_write(src0 + #offset, encodePixelFormat(@sr, cvt_desc = src2, register_format, vecsize));",
    ],
    "LD_GCLK": [
        "if (clock_source == cycle_counter) @sw[0:1] = readPerCoreCycleCounter64();",
        "else                               @sw[0:1] = readSystemTimestamp64();",
    ],
    "LD_MULTI": [
        "for (int i = 0; i < sr_count; ++i) @sw[i] = mem_u32[src0 + #offset + i * 4];",
    ],
    "ST_MULTI": [
        "for (int i = 0; i < sr_count; ++i) mem_u32[src0 + #offset + i * 4] = @sr[i];",
    ],
    "LD_PKA": [
        "BufferDesc b = resolveBufferDesc(src1);",
        "@sw = (src0 + access_bytes <= b.size) ? mem_read(b.address + src0, width) : 0;",
    ],
    "LEA_ATTR": [
        "@sw[0:1] = attrAddress(handle = src2, vert = src0, inst = src1);",
        "@sw[2]   = attrFormatDesc(handle = src2);",
    ],
    "LEA_ATTR_IMM": [
        "AttrDesc a = SRT[#attribute_table_index].Attr[#attribute_index];",
        "@sw[0:1] = attrAddress(a, vert = src0, inst = src1); @sw[2] = a.format_desc;",
    ],
    "LEA_BUF": [
        "@sw[0:1] = resolveBufferDesc(src1).address + uint64_t(src0);",
    ],
    "LEA_BUF_IMM": [
        "@sw[0:1] = SRT[#buffer_table_index].Buffer[#buffer_index].address + uint64_t(src0);",
    ],
    "LEA_PKA": [
        "@sw[0:1] = resolvePkaBuffer(src1).address + uint64_t(src0);",
    ],
    "LEA_TEX": [
        "@sw[0:1] = texelAddress(handle = src2, xy = src0, z = src1);",
        "@sw[2]   = texFormatDesc(handle = src2);",
    ],
    "LEA_TEX_IMM": [
        "TexDesc t = SRT[#texture_table_index].Tex[#texture_index];",
        "@sw[0:1] = texelAddress(t, xy = src0, z = src1); @sw[2] = t.format_desc;",
    ],

    # 12. Atomics
    "ACMPXCHG": [
        "uint64_t addr = src0 + #offset;",
        "@sw = atomicCompSwap(mem[addr], @sr[0] /*cmp*/, @sr[1] /*swap*/);",
    ],
    "ATOM": [
        "atomicOp(atom_opc, mem[src0 + #offset], @sr); // fire-and-forget RMW",
    ],
    "ATOM1": [
        "atomicUnary(atom1_opc, mem[src0 + #offset]); // ainc/adec/asmax1/aumax1/aor1",
    ],
    "ATOM1_RETURN": [
        "@sw = atomicUnaryReturn(atom1_opc, mem[src0 + #offset]); // returns old value",
    ],
    "ATOM_RETURN": [
        "@sw = atomicOpReturn(atom_opc, mem[src0 + #offset], @sr); // returns old value",
    ],
    "AXCHG": [
        "@sw = atomicExchange(mem[src0 + #offset], @sr); // swaps & returns old value",
    ],

    # 13. Attributes, Varyings & Fragment
    "ATEST": [
        "@sw = evalAlphaToCoverageAndEarlyZ(cov = src0, alpha = src1, datum = src2);",
    ],
    "BLEND": [
        "tileSRAM[rt] = blendColor(desc = src0, rgba = @sr, dst = tileSRAM[rt], cov = src2);",
        "if (flow == end) return;",
    ],
    "LD_ATTR": [
        "@sw = fetchConvertAttr(handle = src2, vert = src0, inst = src1, register_format, vecsize);",
    ],
    "LD_ATTR_IMM": [
        "AttrDesc a = SRT[#attribute_table_index].Attr[#attribute_index];",
        "@sw = fetchConvertAttr(a, vert = src0, inst = src1, register_format, vecsize);",
    ],
    "LD_TEX": [
        "@sw = imageLoad(handle = src2, ivec2(src0.h0, src0.h1), z = src1, register_format, vecsize);",
    ],
    "LD_TEX_IMM": [
        "TexDesc t = SRT[#texture_table_index].Tex[#texture_index];",
        "@sw = imageLoad(t, ivec2(src0.h0, src0.h1), z = src1, register_format, vecsize);",
    ],
    "LD_TILE": [
        "if (flag_z_stencil) @sw = readTileDepthStencilSRAM(pixel = src0, cov = src1);",
        "else                @sw = decodeTileColorSRAM(cvt = src2, pixel = src0, cov = src1);",
    ],
    "LD_VAR": [
        "@sw = interpolateVarying(handle = src1, pixel = src0, sample_position, update_mode);",
    ],
    "LD_VAR_BUF": [
        "@sw = interpolateVaryingBuf(byte_off = src1, pixel = src0, source_format, sample_position, update_mode);",
    ],
    "LD_VAR_BUF_FLAT": [
        "@sw = readVaryingBufFlat(off_and_vtx = src0, flag_explicit_vertex);",
    ],
    "LD_VAR_BUF_FLAT_IMM": [
        "@sw = readVaryingBufFlat(provoking_vertex, byte_off = #offset);",
    ],
    "LD_VAR_BUF_IMM": [
        "@sw = interpolateVaryingBuf(byte_off = #offset, pixel = src0, source_format, update_mode);",
    ],
    "LD_VAR_FLAT": [
        "@sw = readFlatVarying(handle = src0, vertex);",
    ],
    "LD_VAR_FLAT_IMM": [
        "AttrDesc v = SRT[#varying_table_index].Attr[#varying_index];",
        "@sw = readFlatVarying(v, vertex);",
    ],
    "LD_VAR_IMM": [
        "AttrDesc v = SRT[#varying_table_index].Attr[#varying_index];",
        "@sw = interpolateVarying(v, pixel = src0, sample_position, update_mode);",
    ],
    "LD_VAR_SPECIAL": [
        "if      (varying_name == frag_w) @sw = gl_FragCoord.w;",
        "else if (varying_name == frag_z) @sw = gl_FragCoord.z;",
        "else if (varying_name == point)  @sw = gl_PointCoord;",
        "else /* bary */                  @sw = gl_BaryCoordEXT.xy;",
    ],
    "ST_TILE": [
        "tileColorSRAM[pixel = src0, cov = src1] = encodeTileColor(@sr, cvt = src2, register_format, vecsize);",
    ],
    "ZS_EMIT": [
        "if (flag_z)       gl_FragDepth = src0;",
        "if (flag_stencil) gl_FragStencilRefARB = int(src1 & 0xFFu);",
        "@sw = evalLateDepthStencilTest(cov = src2, z = src0, stencil = src1);",
    ],

    # 14. Texture & Varying-Texture
    "TEX_DUAL": [
        "@sw[0:3] = textureSample(tex0_of(src0), coords = @sr, lod_mode);",
        "@sw[4:7] = textureSample(tex1_of(src0), coords = @sr, lod_mode);",
    ],
    "TEX_FETCH": [
        "@sw = texelFetch(handle = src0, icoords = @sr, flag_array, flag_texel_offset);",
    ],
    "TEX_GATHER": [
        "if (flag_compare) @sw = textureGatherCompare(handle = src0, uv_dref = @sr);",
        "else              @sw = textureGather(handle = src0, uv = @sr, gather_component);",
    ],
    "TEX_GRADIENT": [
        "if (flag_disable_lod_clamp) @sw = packFixed8_8(textureQueryLod(handle = src0, uv = @sr));",
        "else if (flag_derivative)   @sw = evalScreenSpaceDerivatives(uv = @sr);",
        "else                        @sw = textureGrad(handle = src0, uv_grads = @sr);",
    ],
    "TEX_SINGLE": [
        "@sw = textureSample(handle = src0, coords = @sr, dim, lod_mode, flag_project, flag_compare);",
    ],
    "TEX_WEIGHTS": [
        "@sw = bilinearFootprintWeights2x2(handle = src0, uv = @sr); // v9 only",
    ],
    "VAR_TEX_BUF_DUAL": [
        "vec2 uv = interpolateVaryingBuf(byte_off = src2);",
        "@sw[0:3] = textureSample(tex0_of(src0), uv); @sw[4:7] = textureSample(tex1_of(src0), uv);",
    ],
    "VAR_TEX_BUF_GATHER": [
        "@sw = textureGather(handle = src0, interpolateVaryingBuf(byte_off = src2), gather_component);",
    ],
    "VAR_TEX_BUF_GRADIENT": [
        "@sw = evalTexGradOrLod(handle = src0, interpolateVaryingBuf(byte_off = src2));",
    ],
    "VAR_TEX_BUF_SINGLE": [
        "@sw = textureSample(handle = src0, interpolateVaryingBuf(byte_off = src2), dim, flag_project, flag_compare);",
    ],
    "VAR_TEX_DUAL": [
        "vec2 uv = interpolateVarying(handle = src2);",
        "@sw[0:3] = textureSample(tex0_of(src0), uv); @sw[4:7] = textureSample(tex1_of(src0), uv);",
    ],
    "VAR_TEX_GATHER": [
        "@sw = textureGather(handle = src0, interpolateVarying(handle = src2), gather_component);",
    ],
    "VAR_TEX_GRADIENT": [
        "@sw = evalTexGradOrLod(handle = src0, interpolateVarying(handle = src2));",
    ],
    "VAR_TEX_SINGLE": [
        "@sw = textureSample(handle = src0, interpolateVarying(handle = src2), dim, flag_project, flag_compare);",
    ],
}

_SHIFT_PSEUDO = {
    "LSHIFT": "(src0[i] << s)",
    "RSHIFT": "(uint(src0[i]) >> s)",
    "ARSHIFT": "(int(src0[i]) >> s)",
    "LROT": "((src0[i] << s) | (src0[i] >> ((W - s) & (W - 1))))",
    "RROT": "((src0[i] >> s) | (src0[i] << ((W - s) & (W - 1))))",
}
_LOGIC_PSEUDO = {
    "": None,
    "_AND": "&",
    "_OR": "|",
    "_XOR": "^",
}
for _sh, _expr in _SHIFT_PSEUDO.items():
    for _lop, _op in _LOGIC_PSEUDO.items():
        _mn = _sh + _lop
        if _mn in _PSEUDO_GLSL:
            continue
        if _op is not None:
            _PSEUDO_GLSL[_mn] = [
                "for (int i = 0; i < N; ++i) {",
                "    uint s = extractByte(src1, lanes1, i) & (W - 1);",
                f"    auto r = {_expr} {_op} (flag_not2 ? ~src2[i] : src2[i]);",
                "    dest[i] = flag_not_result ? ~r : r;",
                "}",
            ]
        else:
            _PSEUDO_GLSL[_mn] = [
                "for (int i = 0; i < N; ++i) {",
                "    uint s = extractByte(src1, lanes1, i) & (W - 1);",
                f"    auto r = {_expr};",
                "    dest[i] = flag_not_result ? ~r : r;",
                "}",
            ]

# Backward-compatible alias
_EQUIV_GLSL = _PSEUDO_GLSL


def pseudo_glsl_tex(mnem: str) -> str:
    """Return LaTeX-escaped Pseudo-GLSL operational semantics lines for *mnem*."""
    return r"\\".join(tex_esc(g) for g in _PSEUDO_GLSL[mnem])


equiv_glsl_tex = pseudo_glsl_tex


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

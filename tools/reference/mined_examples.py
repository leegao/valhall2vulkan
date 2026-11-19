"""Curated SPIR-V -> Valhall ISA worked examples mined from real shaders.

Every example in ``MINED_SPIRV_EXAMPLES`` was produced by compiling real Vulkan
SPIR-V binaries from ``spirv.tar.xz`` through Arm Mali Offline Compiler
(``malioc`` v8.8.1, Mali-G710 v10 and Immortalis-G925 v14) using
``tools/moc_hook/moc_hook.c``, disassembling the SPIR-V with ``spirv-dis``, and
disassembling the emitted Valhall ``OBJC`` bytecode with the generated Valhall
decoder.

Each entry maps a canonical Valhall mnemonic to a tuple:
    (title, stage_core, spirv_lines, valhall_lines, explanation_md)
"""

from __future__ import annotations

from .fmt_helpers import md_inline_to_tex, tex_esc, tex_esc_literal

# mnemonic -> (title, stage_core, spirv_lines, valhall_lines, explanation_md)
MINED_SPIRV_EXAMPLES: dict[str, tuple[str, str, list[str], list[str], str]] = {
    # ------------------------------------------------------------------
    # 1. Integer Arithmetic
    # ------------------------------------------------------------------
    "HADD": (
        "Halving signed vertex-index divide-by-2",
        "Vertex shader, Mali-G710 (v10)",
        [
            "%v = OpLoad %int %gl_VertexIndex",
            "%half = OpShiftRightArithmetic %int %v %int_1",
            "%f = OpConvertSToF %float %half",
        ],
        [
            "HADD.s32.round_down r0, r60^, k0",
            "S32_TO_F32 r0, r0^",
        ],
        "Instead of emitting a shift-unit `ARSHIFT` instruction, `malioc` lowers `(gl_VertexIndex >> 1)` on v10 to `HADD.s32.round_down r0, r60^, k0`, computing `(r60 + 0) >> 1` on the integer add pipe with zero extra constant setup (since `k0` is the inline zero constant and `r60` is the preloaded `VertexIndex`).",
    ),
    "IABS": (
        "Signed integer absolute value before float conversion",
        "Vertex shader, Mali-G710 (v10)",
        [
            "%a = OpExtInst %int %glsl SAbs %idx",
            "%f = OpConvertUToF %float %a",
        ],
        [
            "IABS.s32 r1, r0",
            "U32_TO_F32 r2, r1",
            "FMUL.f32.clamp_0_inf r2, r2^, u4.w0",
        ],
        "`GLSL.std.450 SAbs` maps 1:1 to `IABS.s32`. Because the result is guaranteed non-negative (except `INT32_MIN`, which wraps to `0x80000000`), `malioc` feeds `r1` directly into `U32_TO_F32`.",
    ),
    "IADD": (
        "Buffer byte-offset calculation from vertex attribute and UBO base",
        "Vertex shader, Mali-G710 (v10)",
        [
            "%idx = OpLoad %uint %attr_in",
            "%off = OpIAdd %uint %ubo_base %idx",
            "%byte_off = OpIMul %uint %off %uint_16",
        ],
        [
            "IADD.u32 r0, u3.w0, r0^",
            "IMUL.u32 r0, r0^, k13121110.b0",
            "LD_PKA.i32.slot0.wait0 @r0, r0^, u12.w1",
        ],
        "`OpIAdd` lowers to `IADD.u32` reading the uniform base directly from FAU (`u3.w0`) and killing the last use of `r0` (`r0^`). The resulting index is scaled by 16 (`k13121110.b0 = 0x10`) for `LD_PKA`.",
    ),
    "IADD_IMM": (
        "Materializing 32-bit integer constants via MOV_IMM alias",
        "Compute shader, Mali-G710 (v10)",
        [
            "OpStore %out_x %uint_1000",
            "OpStore %out_y %uint_4294966296 ; -1000",
            "OpStore %out_z %uint_0",
            "OpStore %out_w %uint_0",
        ],
        [
            "MOV_IMM.i32 r2, #0x0",
            "MOV.i32 r3, r2",
            "MOV_IMM.i32 r1, #0xfffffc18",
            "MOV_IMM.i32 r0, #0x3e8",
            "STORE.i128.slot0.end @r0:r1:r2:r3, u0, #0x0",
        ],
        "Valhall encodes 32-bit immediate loads (`MOV_IMM.i32`) as `IADD_IMM.i32` with `src0 = k0` (0). Here four scalar stores to a 16-byte SSBO struct are coalesced into one `STORE.i128.end` after loading `#0x3e8` (`1000`) and `#0xfffffc18` (`-1000`).",
    ),
    "IMUL": (
        "Struct stride multiplication using FAU byte constant",
        "Vertex shader, Mali-G710 (v10)",
        [
            "%sum = OpIAdd %uint %base %attr_idx",
            "%stride_off = OpIMul %uint %sum %uint_16",
        ],
        [
            "IADD.u32 r0, u3.w0, r0^",
            "IMUL.u32 r0, r0^, k13121110.b0",
        ],
        "`OpIMul` lowers to `IMUL.u32` with inline FAU small constant `k13121110.b0` (byte 0 = `0x10` = 16), avoiding a separate immediate-load instruction.",
    ),
    "ISUB": (
        "Subtracting BaseInstance from InstanceIndex",
        "Vertex shader, Mali-G710 (v10)",
        [
            "%inst = OpLoad %int %gl_InstanceIndex",
            "%base = OpLoad %int %ubo_base_inst",
            "%rel  = OpISub %int %inst %base",
            "%f    = OpConvertSToF %float %rel",
        ],
        [
            "LD_PKA.i32.slot0 @r0, u15.w0, u13.w1",
            "ISUB.u32 r0, r61^, r0^",
            "S32_TO_F32 r0, r0^",
        ],
        "`r61` holds hardware-preloaded `gl_InstanceIndex`. After `LD_PKA` fetches the base offset into `r0`, `ISUB.u32` computes the relative instance index (`r61 - r0`) before converting to float.",
    ),
    "SHADD": (
        "Scaled 64-bit buffer address calculation for 32-bit element load",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%ptr = OpAccessChain %_ptr_Uniform_uint %ssbo %int_0 %idx",
            "%val = OpLoad %uint %ptr",
        ],
        [
            "SHADD.u64 [r0:r1], sample, [r0^:r1^].w0, #0x2",
            "LOAD.i32.slot0.wait0 @r0, [r0^:r1^], #0x0",
        ],
        "`SHADD.u64` fuses `base_64 + (uint64(index_32) << 2)` into a single 64-bit address instruction, zero-extending the 32-bit element index via `.w0` and scaling by `sizeof(uint) == 4` (`#0x2`).",
    ),

    # ------------------------------------------------------------------
    # 2. Bitwise & Shift-Logic
    # ------------------------------------------------------------------
    "AND_IMM": (
        "Inline 32-bit bitmask clear on v11+ (Immortalis-G925)",
        "Vertex shader, Immortalis-G925 (v14)",
        [
            "%masked = OpBitwiseAnd %uint %flags %uint_0xffffffdf",
            "%eq     = OpIEqual %bool %masked %expected",
        ],
        [
            "AND_IMM.i32 r0, r0^, #0xffffffdf",
            "CSEL.u32.eq r0, r0^, u12.w1, r2^, u12.w0",
        ],
        "On Valhall v11+, `OpBitwiseAnd` with a 32-bit constant mask that does not fit in FAU small constants lowers directly to a single `AND_IMM.i32` instruction (clearing bit 5 via `#0xffffffdf`) rather than `MOV_IMM` + `AND`.",
    ),
    "ARSHIFT_OR": (
        "Signed arithmetic right shift of bitfield value",
        "Vertex shader, Mali-G710 (v10)",
        [
            "%shr = OpShiftRightArithmetic %int %val %int_21",
        ],
        [
            "ARSHIFT.i32 r4, r0, k17161514.b1",
        ],
        "Standalone `ARSHIFT.i32` is encoded as `ARSHIFT_OR` with `src2 = k0` (0). Here `k17161514.b1` supplies the inline shift count `0x15` (21) directly from the FAU small-constant table.",
    ),
    "LSHIFT_AND": (
        "Bitmasking vertex attribute bits via AND alias",
        "Vertex shader, Mali-G710 (v10)",
        [
            "%bit1 = OpBitwiseAnd %int %bits %int_2",
            "%bit0 = OpBitwiseAnd %int %bits %int_1",
        ],
        [
            "AND.i32 r1, k03020100.b2, r0",
            "AND.i32 r0, k01000000.b3, r0^",
        ],
        "Standalone `AND.i32` encodes as `LSHIFT_AND` with shift amount 0. Notice `malioc` pulls small bitmasks (`2` from `k03020100.b2`, `1` from `k01000000.b3`) directly from FAU byte lanes.",
    ),
    "LSHIFT_OR": (
        "Combining two comparison masks for fragment discard",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%is_nz  = OpFOrdNotEqual %bool %v %float_0",
            "%is_nan = OpFUnordNotEqual %bool %v %v",
            "%either = OpLogicalOr %bool %is_nz %is_nan",
        ],
        [
            "FCMP.f32.discard.m1.gtlt r0, r2, k0",
            "FCMP.f32.m1.ne r1, r2, r2",
            "OR.i32 r0, r0^, r1^",
        ],
        "Two `.m1` (`0xFFFFFFFF` / `0x00000000`) comparison masks are combined with `OR.i32` (encoded as `LSHIFT_OR` with shift 0).",
    ),
    "LSHIFT_XOR": (
        "Inverting a boolean mask via NOT alias",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%keep = OpLogicalNot %bool %cond",
            "OpBranchConditional %keep %kill_blk %cont_blk",
        ],
        [
            "NOT.i32 r1, r0",
            "DISCARD.f32.wait0126.ne r1^, k0",
        ],
        "Because `.m1` booleans use all-ones (`0xFFFFFFFF`) for true and `0` for false, `OpLogicalNot` lowers to `NOT.i32` (encoded via shift-logic with `.not_result`) before testing against `k0` in `DISCARD`.",
    ),
    "RSHIFT_AND": (
        "Fused dynamic bit-extraction: (bits >> shift) & mask",
        "Vertex shader, Mali-G710 (v10)",
        [
            "%s = OpShiftRightLogical %uint %word %bit_idx",
            "%m = OpBitwiseAnd %uint %s %ubo_mask",
        ],
        [
            "RSHIFT_AND.i32 r6, r11^, r6^.b0, u13.w0",
            "CSEL.u32.eq r6, r6^, k0, u13.w1, r7^",
        ],
        "`malioc` fuses `OpShiftRightLogical` + `OpBitwiseAnd` into a single `RSHIFT_AND.i32` instruction: `r11` is logically right-shifted by byte 0 of `r6` (`r6^.b0`) and masked with `u13.w0` in one cycle.",
    ),
    "RSHIFT_OR": (
        "Logical right shift for index division by 32",
        "Vertex shader, Mali-G710 (v10)",
        [
            "%word_idx = OpShiftRightLogical %uint %bit_idx %uint_5",
            "%addr_idx = OpIAdd %uint %word_idx %base",
        ],
        [
            "RSHIFT.i32 r11, r6, k07060504.b1",
            "IADD.u32 r11, r11^, u3.w0",
        ],
        "`RSHIFT.i32` (encoded as `RSHIFT_OR` with `src2 = k0`) shifts `r6` right by 5 (`k07060504.b1 = 0x05`) to compute a 32-bit word index.",
    ),
    "XOR_IMM": (
        "Inverting a comparison mask with inline 32-bit XOR on v14",
        "Compute shader, Immortalis-G925 (v14)",
        [
            "%oob = OpLogicalOr %bool %lt_min %gt_max",
            "%in_bounds = OpLogicalNot %bool %oob",
            "OpBranchConditional %in_bounds %body %exit",
        ],
        [
            "FCMP.f32.m1.lt r12, r15, u2.w0",
            "OR.i32 r12, r12^, r50^",
            "XOR_IMM.i32 r12, r12^, #0xffffffff",
            "BRANCH.not r12, 0x248",
        ],
        "On Immortalis-G925 (v14), `malioc` emits `XOR_IMM.i32 r12, r12^, #0xffffffff` to flip all 32 bits of the `.m1` predicate mask before branching.",
    ),

    # ------------------------------------------------------------------
    # 3. Floating-Point ALU & Canonicalization
    # ------------------------------------------------------------------
    "FADD": (
        "Clamping a varying channel to [0, 1] using FADD.clamp_0_1",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%c = OpExtInst %float %glsl FClamp %alpha %float_0 %float_1",
            "%out = OpFMul %float %c %fade",
        ],
        [
            "FADD.f32.wait1.clamp_0_1 r1, r7^, k0.neg",
            "FMUL.f32.wait.clamp_0_1 r7, r1^, r0^",
        ],
        "`FClamp(x, 0.0, 1.0)` lowers to `FADD.f32.clamp_0_1 r1, r7^, k0.neg` (adding `-0.0` preserves the sign of `0.0` while applying the free hardware `[0, 1]` output clamp).",
    ),
    "FADD_IMM": (
        "Adding a non-FAU float epsilon constant inline",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%denom = OpFAdd %float %val %float_0_01 ; 0x3c23d70a",
            "%inv   = OpFDiv %float %float_1 %denom",
        ],
        [
            "FADD_IMM.f32 r63, r61, #0x3c23d70a",
            "FRCP.f32 r63, r63^",
        ],
        "Because `0.01f` (`0x3c23d70a`) is not in the FAU small-constant table, `malioc` uses `FADD_IMM.f32` to embed the IEEE-754 bit pattern directly in the instruction word before `FRCP.f32`.",
    ),
    "FEXP": (
        "Lowering GLSL.std.450 Exp2 via Q8.24 fixed-point pre-scaling",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%e = OpExtInst %float %glsl Exp2 %x",
        ],
        [
            "LDEXP.f32 r55, r4, u9.w1          ; scale r4 by 2^24",
            "F32_TO_S32 r55, r55^              ; convert to signed Q8.24",
            "FEXP.f32 r55, r55^, r4            ; 2^(r55 / 2^24), range/NaN from r4",
        ],
        "`FEXP.f32` does not take raw FP32 directly: `malioc` scales `x` by $2^{24}$ (`LDEXP` or `FMUL_RSCALE`), converts to signed Q8.24 fixed-point (`F32_TO_S32`), and passes both the fixed-point value (`src0 = r55^`) and original float (`src1 = r4`, used for overflow/underflow/NaN edge cases) into `FEXP.f32`.",
    ),
    "FMA": (
        "3D vector normalization and dot product with fused multiply-add",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%len2 = OpDot %float %n %n",
            "%inv  = OpExtInst %float %glsl InverseSqrt %len2",
            "%ndotl = OpExtInst %float %glsl Fma %ny %ly %nx_lx",
            "%half_lambert = OpExtInst %float %glsl FClamp (fma(%ndotl, 0.5, 0.5)) 0.0 1.0",
        ],
        [
            "FMUL.f32 r3, r0, r0",
            "FMA.f32 r3, r1, r1, r3^",
            "FMA.f32 r3, r2, r2, r3^",
            "FRSQ.f32 r3, r3^",
            "FMA.f32.wait1.clamp_0_1 r0, r0^, k0fr2f.h1, k0fr2f.h1",
        ],
        "`OpDot` lowers to `FMUL` + two `FMA` instructions; `fma(d, 0.5, 0.5)` clamped to `[0, 1]` fuses into a single `FMA.f32.clamp_0_1` using `k0fr2f.h1` (`0.5f`).",
    ),
    "FMAX": (
        "2x2 textureGather max reduction with NaN suppression",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%g = OpImageGather %v4float %tex %uv %int_0 None",
            "%m0 = OpExtInst %float %glsl FMax %g.y %g.x",
            "%m1 = OpExtInst %float %glsl FMax %g.z %g.w",
            "%res = OpExtInst %float %glsl FMax %m0 %m1",
        ],
        [
            "TEX_GATHER.slot0.wait0.skip.tex2d.dst32.rgba.wide.float_coordinates.gather4_r @r0:r1:r2:r3, @r0:r1, u0",
            "FMAX.f32.nan_suppress r0, r1^, r0^",
            "FMAX.f32.nan_suppress r2, r2^, r3^",
            "FMAX.f32.wait.nan_suppress r4, r0^, r2^",
        ],
        "Vulkan/GLSL `FMax` specifies that if one operand is NaN, the non-NaN operand is returned; `malioc` maps this directly to `FMAX.f32.nan_suppress`.",
    ),
    "FMIN": (
        "2x2 textureGather min reduction with NaN suppression",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%g = OpImageGather %v4float %tex %uv %int_0 None",
            "%m0 = OpExtInst %float %glsl FMin %g.y %g.x",
            "%m1 = OpExtInst %float %glsl FMin %g.z %g.w",
            "%res = OpExtInst %float %glsl FMin %m0 %m1",
        ],
        [
            "TEX_GATHER.slot0.wait0.skip.tex2d.dst32.rgba.wide.float_coordinates.gather4_r @r0:r1:r2:r3, @r0:r1, u0",
            "FMIN.f32.nan_suppress r0, r1^, r0^",
            "FMIN.f32.nan_suppress r2, r2^, r3^",
            "FMIN.f32.nan_suppress r0, r0^, r2^",
        ],
        "Just like `FMAX`, `GLSL.std.450 FMin` uses `.nan_suppress` so a single NaN texel does not poison the 2x2 footprint minimum.",
    ),
    "FMUL": (
        "Perspective divide preparation and clamped alpha multiply",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%scaled = OpFMul %float %u %inv_w",
            "%clamped = OpExtInst %float %glsl FClamp %scaled %float_0 %float_1",
        ],
        [
            "FMUL.f32.wait.clamp_0_1 r7, r1^, r0^",
        ],
        "`FMUL.f32` is an encoding alias of `FMA.f32` with addend `-0.0`, supporting full source modifiers (`.abs`, `.neg`) and output `.clamp_0_1`.",
    ),
    "FRCP": (
        "Perspective division by interpolated 1/w",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%w    = OpLoad %float %frag_w",
            "%invw = OpFDiv %float %float_1 %w",
            "%u    = OpFMul %float %u_times_w %invw",
        ],
        [
            "LD_VAR_SPECIAL.slot0.wait0.frag_w.f32.sample.clobber @r2, r61^",
            "FRCP.f32 r3, r2",
            "FMUL.f32 r1, r3, r1^",
        ],
        "Floating-point reciprocal `1.0 / w` lowers directly to a single SFU `FRCP.f32` instruction (< 1 ULP accuracy, with FTZ on `f32`).",
    ),
    "FROUND": (
        "Ceiling a shifted LOD level with .round_up",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%shifted = OpFSub %float %lod %float_0_5",
            "%ceiled  = OpExtInst %float %glsl Ceil %shifted",
        ],
        [
            "FADD.f32 r3, r2, k0fr2f.h1.neg",
            "FROUND.f32.round_up r3, r3^",
        ],
        "`GLSL.std.450 Ceil` maps directly to `FROUND.f32.round_up` (while `Floor`, `Trunc`, and `RoundEven` select `.round_down`, `.round_zero`, and default RNE respectively).",
    ),
    "FRSQ": (
        "Vector normalization via SFU reciprocal square root",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%dot  = OpDot %float %n %n",
            "%rinv = OpExtInst %float %glsl InverseSqrt %dot",
            "%nx   = OpFMul %float %n.x %rinv",
        ],
        [
            "FMUL.f32 r3, r0, r0",
            "FMA.f32 r3, r1, r1, r3^",
            "FMA.f32 r3, r2, r2, r3^",
            "FRSQ.f32 r3, r3^",
            "FMUL.f32 r0, r0^, r3",
        ],
        "`GLSL.std.450 InverseSqrt` lowers directly to a single `FRSQ.f32` instruction on the SFU pipe.",
    ),
    "LDEXP": (
        "Power-of-two exponent scaling before FEXP.f32",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%exp_res = OpExtInst %float %glsl Exp2 %x",
        ],
        [
            "LDEXP.f32 r55, r4, u9.w1          ; r55 = r4 * 2^24",
            "F32_TO_S32 r55, r55^",
            "FEXP.f32 r55, r55^, r4",
        ],
        "`malioc` uses `LDEXP.f32` with exponent `24` (from uniform `u9.w1`) to shift the binary point of `r4` left by 24 bits before converting to Q8.24 fixed-point for `FEXP.f32`.",
    ),

    # ------------------------------------------------------------------
    # 4. SFU Range-Reduction & Helpers
    # ------------------------------------------------------------------
    "FADD_LSCALE": (
        "Complete 5-instruction GLSL Log2 lowering sequence",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%y = OpExtInst %float %glsl Log2 %x",
        ],
        [
            "FREXPE.f32.log r4, r2             ; unbiased exponent e (for m in [0.75, 1.5))",
            "S32_TO_F32 r4, r4^                ; float(e)",
            "FLOGD.f32 r6, r2                  ; log2(m) / (m - 1)",
            "FADD_LSCALE.f32 r2, k1f.neg, r2^  ; m - 1 = (-1.0) + reduce_log(r2)",
            "FMA.f32 r2, r6^, r2^, r4^         ; log2(x) = (log2(m)/(m-1)) * (m-1) + e",
        ],
        "This real `malioc` sequence shows why Valhall has three dedicated log helpers: `FREXPE.log` extracts the integer exponent $e$, `FLOGD` evaluates the smooth divided difference $\\log_2(m)/(m-1)$ on $m \\in [0.75, 1.5)$, `FADD_LSCALE` computes $m - 1$ by adding $-1.0$ (`k1f.neg`) to the reduced mantissa, and a final `FMA` reconstructs $\\log_2(x) = \\frac{\\log_2(m)}{m-1}(m-1) + e$.",
    ),
    "FCOS_TABLE": (
        "Cosine table lookup in GLSL Cos range-reduction",
        "Vertex shader, Mali-G710 (v10)",
        [
            "%c = OpExtInst %float %glsl Cos %angle",
        ],
        [
            "FMUL_RSCALE.f32 r7, r2, r2, k1n   ; r7 = (r2 * r2) * 2^(-1) = r2^2 / 2",
            "FSIN_TABLE.u6 r63, r5             ; sin(idx * pi / 32)",
            "FCOS_TABLE.u6 r5, r5^             ; cos(idx * pi / 32)",
            "FMUL.f32 r7, r7^, r5.neg          ; -(r2^2 / 2) * cos(idx * pi / 32)",
            "FMA.f32.clamp_m1_1 r2, r2^, r63^.neg, r7^",
        ],
        "Valhall splits $\\cos(\\theta_0 + \\delta) \\approx \\cos(\\theta_0) - \\delta\\sin(\\theta_0) - \\frac{\\delta^2}{2}\\cos(\\theta_0)$ using 64-entry LUTs `FCOS_TABLE.u6` and `FSIN_TABLE.u6` indexed by the 6-bit quadrant/step index `r5`, finishing with `.clamp_m1_1` to keep the result within $[-1, 1]$.",
    ),
    "FLOGD": (
        "Logarithm divided-difference helper in GLSL Log2",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%y = OpExtInst %float %glsl Log2 %x",
        ],
        [
            "FREXPE.f32.log r4, r2",
            "S32_TO_F32 r4, r4^",
            "FLOGD.f32 r6, r2",
            "FADD_LSCALE.f32 r2, k1f.neg, r2^",
            "FMA.f32 r2, r6^, r2^, r4^",
        ],
        "`FLOGD.f32` evaluates $\\log_2(m)/(m-1)$ for the mantissa $m \\in [0.75, 1.5)$ of `r2`, avoiding catastrophic cancellation near $m = 1$ when multiplied by $(m - 1)$ from `FADD_LSCALE`.",
    ),
    "FMA_RSCALE": (
        "FP64-intermediate multiply with power-of-two scaling (FMUL_RSCALE alias)",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%y = OpExtInst %float %glsl Exp2 %x",
        ],
        [
            "FMUL_RSCALE.f32 r57, r57^, u3.w1, r4",
            "F32_TO_S32 r5, r57",
            "FEXP.f32 r57, r5^, r57^",
        ],
        "`FMUL_RSCALE.f32` (an alias of `FMA_RSCALE` with `src2 = -0.0`) multiplies `r57 * u3.w1` and scales by $2^{r4}$ in exact FP64 with a single rounding to FP32 — avoiding intermediate overflow/underflow before `FEXP.f32`.",
    ),
    "FMUL_RSCALE": (
        "Computing delta^2 / 2 during trig polynomial evaluation",
        "Vertex shader, Mali-G710 (v10)",
        [
            "%c = OpExtInst %float %glsl Cos %angle",
        ],
        [
            "FMUL_RSCALE.f32 r7, r2, r2, k1n   ; (r2 * r2) * 2^(-1)",
            "FSIN_TABLE.u6 r63, r5",
            "FCOS_TABLE.u6 r5, r5^",
        ],
        "Passing `k1n` (`-1`) as the exponent scale `src3` divides `r2 * r2` by 2 in exact FP64 without needing a separate `0.5f` constant multiply.",
    ),
    "FREXPE": (
        "Extracting the .log-biased exponent for Log2",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%y = OpExtInst %float %glsl Log2 %x",
        ],
        [
            "FREXPE.f32.log r4, r2",
            "S32_TO_F32 r4, r4^",
            "FLOGD.f32 r6, r2",
            "FADD_LSCALE.f32 r2, k1f.neg, r2^",
            "FMA.f32 r2, r6^, r2^, r4^",
        ],
        "With `.log`, `FREXPE` increments the unbiased exponent by 1 whenever the mantissa is $\\ge 1.5$ so that the reduced mantissa $m$ lies in $[0.75, 1.5)$ instead of $[1.0, 2.0)$, matching `FLOGD` and `FADD_LSCALE`.",
    ),
    "FSIN_TABLE": (
        "Sine table lookup paired with FCOS_TABLE in trig lowering",
        "Vertex shader, Mali-G710 (v10)",
        [
            "%c = OpExtInst %float %glsl Cos %angle",
        ],
        [
            "FSIN_TABLE.u6 r63, r5             ; sin(idx * pi / 32)",
            "FCOS_TABLE.u6 r5, r5^             ; cos(idx * pi / 32)",
            "FMUL.f32 r7, r7^, r5.neg",
            "FMA.f32.clamp_m1_1 r2, r2^, r63^.neg, r7^",
        ],
        "`FSIN_TABLE.u6` reads the 6-bit index in `r5` (low 6 bits of the range-reduced fixed-point angle) and returns $\\sin(\\text{idx} \\cdot \\pi / 32)$ for the linear Taylor term $-\\delta \\sin(\\theta_0)$.",
    ),

    # ------------------------------------------------------------------
    # 5. Cubemap Helpers
    # ------------------------------------------------------------------
    "CUBEFACE1": (
        "Full 10-instruction SPIR-V Cube texture sampling lowering",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%cube_img = OpTypeImage %float Cube 0 0 0 1 Unknown",
            "%color    = OpImageSampleImplicitLod %v4float %sampled_cube %dir_xyz None",
        ],
        [
            "CUBEFACE2 r6, r5, r4, r3          ; face_idx<<29 | inf_flag<<16 from (x=r5,y=r4,z=r3)",
            "CUBE_SSEL r7, r3, r5, r6          ; select face S numerator from (z=r3, x=r5, face=r6)",
            "CUBEFACE1 r5, r5^, r4, r3         ; major axis |M| = max(|x|,|y|,|z|)",
            "FRCP.f32 r5, r5^                  ; 1 / |M|",
            "FMUL.f32 r5, r5^, k0fr2f.h1       ; 0.5 / |M|",
            "FMA.f32.clamp_0_1 r7, r5, r7^, k0fr2f.h1 ; s = clamp(S/(2|M|) + 0.5, 0, 1)",
            "MUX.i32.bit r8, r7^, r6, u1.w1    ; pack 3-bit face_idx into s",
            "CUBE_TSEL r3, r4^, r3^, r6^       ; select face T numerator from (y=r4, z=r3, face=r6)",
            "FMA.f32.clamp_0_1 r9, r5^, r3^, k0fr2f.h1 ; t = clamp(T/(2|M|) + 0.5, 0, 1)",
            "TEX_SINGLE.slot0.reconverge.skip.cube.dst32.rgba.wide.computed @r4:r5:r6:r7, @r8:r9, u2",
        ],
        "Valhall does not accept raw 3D `(x, y, z)` vectors in `TEX_SINGLE.cube`. Instead, `malioc` lowers `OpImageSampleImplicitLod` on a `Cube` image using all four cubemap helpers: `CUBEFACE1` computes the major-axis magnitude $|M|$ (returning `1.0` on Inf and `2.0` on 0 to prevent NaN division), `FRCP` + `FMUL 0.5` computes $0.5/|M|$, and `FMA.clamp_0_1` maps both `S` and `T` into $[0, 1]$ before `TEX_SINGLE.cube`.",
    ),
    "CUBEFACE2": (
        "Encoding the 3-bit cubemap face index in bits [31:29]",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%color = OpImageSampleImplicitLod %v4float %sampled_cube %dir_xyz None",
        ],
        [
            "CUBEFACE2 r6, r5, r4, r3          ; r6[31:29] = face (0..5), r6[16] = any_inf",
            "CUBE_SSEL r7, r3, r5, r6",
            "MUX.i32.bit r8, r7^, r6, u1.w1    ; merge r6[31:29] into s coordinate",
            "CUBE_TSEL r3, r4^, r3^, r6^",
        ],
        "`CUBEFACE2` places the 3-bit face ID in bits `[31:29]` so it can be consumed directly by `CUBE_SSEL`/`CUBE_TSEL` and bit-blended into the texture unit's packed cube `s` coordinate via `MUX.i32.bit`.",
    ),
    "CUBE_SSEL": (
        "Selecting the signed S numerator for the active cube face",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%color = OpImageSampleImplicitLod %v4float %sampled_cube %dir_xyz None",
        ],
        [
            "CUBEFACE2 r6, r5, r4, r3          ; face from (x=r5, y=r4, z=r3)",
            "CUBE_SSEL r7, r3, r5, r6          ; src0=z, src1=x, src2=face",
            "FMA.f32.clamp_0_1 r7, r5, r7^, k0fr2f.h1",
        ],
        "Notice the operand order: `CUBE_SSEL` takes `z` in `src0`, `x` in `src1`, and the `CUBEFACE2` output in `src2`, selecting and negating the horizontal face coordinate (`-z` for `+X`, `+z` for `-X`, `+x` for `±Y/+Z`, `-x` for `-Z`).",
    ),
    "CUBE_TSEL": (
        "Selecting the signed T numerator for the active cube face",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%color = OpImageSampleImplicitLod %v4float %sampled_cube %dir_xyz None",
        ],
        [
            "CUBE_TSEL r3, r4^, r3^, r6^       ; src0=y, src1=z, src2=face",
            "FMA.f32.clamp_0_1 r9, r5^, r3^, k0fr2f.h1",
            "TEX_SINGLE.slot0.reconverge.skip.cube.dst32.rgba.wide.computed @r4:r5:r6:r7, @r8:r9, u2",
        ],
        "`CUBE_TSEL` takes `y` in `src0`, `z` in `src1`, and the `CUBEFACE2` face word in `src2`, outputting `+z` for `+Y`, `-z` for `-Y`, and `-y` for `±X/±Z`.",
    ),

    # ------------------------------------------------------------------
    # 6. Compare, Select & Mux
    # ------------------------------------------------------------------
    "CSEL": (
        "NaN-to-zero sanitization of texture coordinates via CSEL.f32.ne",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%is_nan = OpIsNan %bool %u",
            "%safe_u = OpSelect %float %is_nan %float_0 %u",
        ],
        [
            "CSEL.f32.ne r1, r1^, r1^, k0, r1^",
            "CSEL.f32.discard.ne r0, r0^, r0^, k0, r0^",
        ],
        "Self-comparing `r1 != r1` (`CSEL.f32.ne`) is true if and only if `r1` is NaN; hardware selects `k0` (`0.0f`) when NaN and preserves `r1^` otherwise in a single cycle.",
    ),
    "FCMP_AND": (
        "Fused float comparison AND existing boolean predicate",
        "Vertex shader, Mali-G710 (v10)",
        [
            "%gt = OpFOrdGreaterThan %bool %v %thresh",
            "%both = OpLogicalAnd %bool %gt %ubo_flag",
            "%sel = OpSelect %float %both %val_a %val_b",
        ],
        [
            "FCMP_AND.f32.m1.gt r1, r0, u4.w0, u4.w1",
            "CSEL.u32.eq r1, r1^, k0, r2^, u3.w0",
        ],
        "`FCMP_AND` folds `(r0 > u4.w0) && (u4.w1 != 0)` into one instruction, writing an `.m1` mask to `r1` that drives `CSEL.u32.eq`.",
    ),
    "FCMP_OR": (
        "Testing float non-zero or NaN for branch/discard",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%nz  = OpFOrdNotEqual %bool %x %float_0",
            "%nan = OpFUnordNotEqual %bool %x %x",
        ],
        [
            "FCMP.f32.discard.m1.gtlt r0, r2, k0",
            "FCMP.f32.m1.ne r1, r2, r2",
        ],
        "Standalone `FCMP` is encoded as `FCMP_OR` with `src2 = k0`. Note the distinction: `.gtlt` tests ordered not-equal (`x < 0 || x > 0`, false on NaN), whereas `.ne` tests unordered not-equal (true on NaN).",
    ),
    "ICMP_AND": (
        "Chained bit-test and uniform-flag guard",
        "Vertex shader, Mali-G710 (v10)",
        [
            "%b = OpBitwiseAnd %int %x %int_2",
            "%c = OpLogicalAnd %bool (OpIEqual %b %int_0) %flag",
        ],
        [
            "AND.i32 r1, k03020100.b2, r0",
            "ICMP_AND.u32.m1.eq r2, r1, k0, u3.w1",
        ],
        "`ICMP_AND.u32.m1.eq` evaluates `(r1 == 0) && (u3.w1 != 0)` and outputs an all-ones/zero mask directly into `r2`.",
    ),
    "ICMP_OR": (
        "Short-circuit OR of two integer equality tests",
        "Vertex shader, Mali-G710 (v10)",
        [
            "%eq0 = OpIEqual %bool %bit0 %int_0",
            "%any = OpLogicalOr %bool %eq0 %prev_cond",
        ],
        [
            "ICMP_OR.u32.m1.eq r0, r0^, k0, r2",
            "BRANCH r1, 0x78",
        ],
        "`ICMP_OR.u32.m1.eq` fuses `(r0 == 0) || (r2 != 0)` into a single instruction, avoiding a separate `OR.i32`.",
    ),
    "MUX": (
        "Bitwise merge of cubemap face index into S coordinate",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%color = OpImageSampleImplicitLod %v4float %sampled_cube %dir_xyz None",
        ],
        [
            "CUBEFACE2 r6, r5, r4, r3",
            "FMA.f32.clamp_0_1 r7, r5, r7^, k0fr2f.h1",
            "MUX.i32.bit r8, r7^, r6, u1.w1",
        ],
        "`MUX.i32.bit` computes `(r7 & u1.w1) | (r6 & ~u1.w1)` bit-by-bit, splicing the 3-bit cube face ID from `r6[31:29]` into the projected `s` coordinate `r7` in one cycle.",
    ),

    # ------------------------------------------------------------------
    # 7. Conversions
    # ------------------------------------------------------------------
    "F16_TO_F32": (
        "Exact FP16-to-FP32 lane promotion",
        "Vertex shader, Mali-G710 (v10)",
        [
            "%v = OpExtInst %v2float %glsl UnpackHalf2x16 %packed",
        ],
        [
            "F16_TO_F32 r14, r47^.h0",
        ],
        "Extracts the low 16-bit half-float lane (`.h0`) from `r47` and promotes it losslessly to IEEE-754 FP32 in `r14`.",
    ),
    "F32_TO_S32": (
        "Truncating float-to-signed-int conversion (.round_zero)",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%scaled = OpFMul %float %v %float_256",
            "%clamped = OpExtInst %float %glsl FMin %scaled %max_val",
            "%ival = OpConvertFToS %int %clamped",
        ],
        [
            "FMUL.f32 r3, r3^, k255f256f.h1",
            "FMIN.f32.nan_propagate.clamp_0_inf r3, r3^, u3.w1",
            "F32_TO_S32.round_zero r3, r3^",
        ],
        "SPIR-V `OpConvertFToS` requires truncation toward zero, which `malioc` encodes explicitly via `.round_zero` (default without `.round_zero` is round-to-nearest-even, as used in `FEXP` Q8.24 conversion).",
    ),
    "F32_TO_U32": (
        "Clamped float-to-unsigned-int conversion",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%pos = OpExtInst %float %glsl FMax %v %float_0",
            "%u   = OpConvertFToU %uint %pos",
        ],
        [
            "FADD.f32.clamp_0_inf r2, r2^, k0.neg",
            "F32_TO_U32 r2, r2^",
        ],
        "`F32_TO_U32` saturates negative/NaN inputs to `0` and values $\\ge 2^{32}-1$ to `0xFFFFFFFF`.",
    ),
    "S16_TO_F32": (
        "Converting fixed-point 8.8 LOD result from TEX_GRADIENT to float",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%lod = OpImageQueryLod %v2float %sampler %uv",
        ],
        [
            "TEX_GRADIENT.slot0.discard.skip.tex2d.dst32.r.wide.disable_lod_clamp @r2, @r0:r1, u2",
            "S16_TO_F32 r2, r2^.h0",
            "FMUL.f32 r2, r2^, u3.w0           ; multiply by 1/256",
        ],
        "`TEX_GRADIENT.disable_lod_clamp` returns the raw LOD as a signed 8.8 fixed-point 16-bit integer in `r2.h0`; `S16_TO_F32` converts `r2^.h0` exactly to FP32 before scaling by $1/256$.",
    ),
    "S16_TO_S32": (
        "Sign-extending a 16-bit uniform halfword for discard test",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%flag16 = OpBitFieldSExtract %int %ubo_word %int_0 %int_16",
            "OpBranchConditional (OpINotEqual %flag16 %int_0) %kill %ok",
        ],
        [
            "S16_TO_S32.discard r0, u0.w0.h0",
            "DISCARD.f32.wait0126.ne r0^, k0",
        ],
        "`S16_TO_S32` reads the low 16 bits of uniform `u0.w0` (`.h0`) and sign-extends them to 32 bits in `r0`.",
    ),
    "S32_TO_F32": (
        "Converting signed vertex index to float",
        "Vertex shader, Mali-G710 (v10)",
        [
            "%vid = OpLoad %int %gl_VertexIndex",
            "%f   = OpConvertSToF %float %vid",
        ],
        [
            "S32_TO_F32 r0, r60^",
        ],
        "`OpConvertSToF` maps directly to `S32_TO_F32` (default RNE rounding) consuming the preloaded `VertexIndex` in `r60`.",
    ),
    "U16_TO_F32": (
        "Unpacking packed 16-bit pixel coordinates to float",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%fc = OpLoad %v4float %gl_FragCoord",
        ],
        [
            "U16_TO_F32 r3, r59.h1",
            "U16_TO_F32 r2, r59^.h0",
        ],
        "Hardware packs integer fragment `(x, y)` coordinates into halfwords `.h0` and `.h1` of `r59`; `U16_TO_F32` converts each halfword losslessly to FP32.",
    ),
    "U32_TO_F32": (
        "Converting unsigned integer to float",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%f = OpConvertUToF %float %uval",
        ],
        [
            "F32_TO_S32.round_zero r2, r2^",
            "U32_TO_F32 r2, r2^",
            "FMUL.f32 r2, r2^, u3.w0",
        ],
        "`OpConvertUToF` lowers 1:1 to `U32_TO_F32` with default round-to-nearest-even.",
    ),
    "U8_TO_F32": (
        "Unpacking a RGBA8_UNORM uint32 word into 4 float channels",
        "Vertex shader, Mali-G710 (v10)",
        [
            "%b0 = OpBitFieldUExtract %uint %packed %uint_0 %uint_8",
            "%r  = OpFMul %float (OpConvertUToF %float %b0) %inv_255",
            "%b1 = OpBitFieldUExtract %uint %packed %uint_8 %uint_8",
            "%g  = OpFMul %float (OpConvertUToF %float %b1) %inv_255",
        ],
        [
            "LD_PKA.i32.slot0.wait0 @r0, r0^, u12.w1",
            "U8_TO_F32 r1, r0.b0",
            "U8_TO_F32 r62, r0.b1",
            "U8_TO_F32 r63, r0.b2",
            "U8_TO_F32 r61, r0^.b3",
            "FMUL.f32 r7, r1^, u13.w0          ; multiply by 1/255",
        ],
        "`malioc` fuses each `OpBitFieldUExtract(..., 8)` + `OpConvertUToF` into a single `U8_TO_F32` selecting bytes `.b0` through `.b3` of the loaded 32-bit word.",
    ),
    "V2F32_TO_V2F16": (
        "Packing two FP32 values into packed FP16 (PackHalf2x16)",
        "Vertex shader, Mali-G710 (v10)",
        [
            "%packed = OpExtInst %uint %glsl PackHalf2x16 %v2f",
        ],
        [
            "V2F32_TO_V2F16.wait1 r57.h0, r57^, r57^",
        ],
        "`GLSL.std.450 PackHalf2x16` lowers directly to `V2F32_TO_V2F16`, narrowing two FP32 sources into `dest.h0` and `dest.h1` in a single instruction (v9--v10).",
    ),

    # ------------------------------------------------------------------
    # 8. Move, Pack & Swizzle
    # ------------------------------------------------------------------
    "MKVEC": (
        "Packing two FP16 constants into a v2f16 register for BLEND",
        "Fragment shader, Mali-G710 (v10)",
        [
            "OpStore %frag_out %vec4_1_1_1_1",
        ],
        [
            "MOV_IMM.i32.discard r62, #0x3c003c00 ; (1.0h, 1.0h)",
            "MKVEC.v2i16.wait r63, r62.h0, r62.h1",
            "BLEND.slot0.end.v4.f16 @r62:r63, blend_descriptor_0, 0x68, r60^",
        ],
        "`MKVEC.v2i16` combines two 16-bit halfwords (`r62.h0` and `r62.h1`) into a single 32-bit register `r63` to form the 64-bit `v4.f16` RGBA output for `BLEND`.",
    ),
    "MOV": (
        "Copying FAU uniforms into contiguous staging registers for BLEND",
        "Fragment shader, Mali-G710 (v10)",
        [
            "OpStore %out_color (OpCompositeConstruct %v4float %u_r %u_g %u_b %var_a)",
        ],
        [
            "LD_VAR_IMM.slot0.discard.f32.sample.store @r3, r61^, #0x0, #0xb",
            "MOV.i32 r2, u0.w0",
            "MOV.i32 r1, u0.w1",
            "MOV.i32.wait r0, u1.w0",
            "BLEND.slot0.end.v4.f32 @r0:r1:r2:r3, blend_descriptor_0, 0x28, r60^",
        ],
        "`BLEND` requires its RGBA inputs in a contiguous staging-register range (`@r0:r1:r2:r3`), so `malioc` uses `MOV.i32` to stage uniform channels from `u0..u1` alongside the varying alpha in `r3`.",
    ),
    "MOV_IMM": (
        "Loading packed FP16 constant colors for BLEND",
        "Fragment shader, Mali-G710 (v10)",
        [
            "OpStore %out_color %const_v4f16_1_1_1_0",
        ],
        [
            "MOV_IMM.i32.discard r1.h1, #0x0",
            "MOV_IMM.i32 r0, #0x3c003c00",
            "MOV_IMM.i32.wait r1.h0, #0x3c003c00",
            "BLEND.slot0.end.v4.f16 @r0:r1, blend_descriptor_0, 0x20, r60^",
        ],
        "`MOV_IMM.i32` loads 32-bit bit patterns (`0x3c003c00` = two FP16 `1.0h` lanes) and uses halfword write masks (`r1.h1`, `r1.h0`) to construct `(1.0h, 1.0h, 1.0h, 0.0h)`.",
    ),

    # ------------------------------------------------------------------
    # 9. Subgroup & Cross-Lane
    # ------------------------------------------------------------------
    "CLPER": (
        "Computing fragment screen-space derivative OpDPdx via 2x2 quad shuffle",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%dx = OpDPdx %float %var_x",
        ],
        [
            "LD_VAR_IMM.slot0.wait0.v4.f32.sample.store @r12:r13:r14:r15, r61^, #0x0, #0xb",
            "CLPER.subgroup4.zero r51, r12, k01000000.b3 ; read lane 1 of 2x2 quad",
            "CLPER.subgroup4.zero r1, r12, k0.b0         ; read lane 0 of 2x2 quad",
            "; followed by FADD.f32 dest, r51, r1.neg   -> lane1 - lane0",
        ],
        "Fragment shaders execute in 2x2 pixel quads (`subgroup4`). `OpDPdx` lowers to two `CLPER.subgroup4.zero` instructions fetching quad-lane 1 (`k01000000.b3 = 1`) and quad-lane 0 (`k0.b0 = 0`), then subtracting them.",
    ),

    # ------------------------------------------------------------------
    # 10. Control Flow, Barriers & System
    # ------------------------------------------------------------------
    "BRANCH": (
        "Conditional forward skip after fragment discard check",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%le = OpFOrdLessThanEqual %bool %alpha %thresh",
            "OpSelectionMerge %merge None",
            "OpBranchConditional %le %merge %shaded",
        ],
        [
            "FCMP.f32.discard.m1.le r8, r7, u0.w0",
            "MOV_IMM.i32 r0, #0x0",
            "BRANCH r8^, 0x88                  ; skip shading block if r8 != 0",
            "... ; shaded block ending with .reconverge",
        ],
        "`BRANCH` tests `r8^ != 0` and jumps forward by `+0x88` bytes when true, reconverging at the merge block (marked with `.reconverge` on the last instruction of the taken/fallthrough body).",
    ),
    "DISCARD": (
        "Lowering fragment discard (OpKill / OpDemoteToHelperInvocation)",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%kill = OpFOrdLessThan %bool %val %neg_thresh",
            "OpBranchConditional %kill %discard_blk %cont_blk",
            "%discard_blk = OpLabel ; OpKill",
        ],
        [
            "DISCARD.f32.wait0126.lt r0^, u0.w1.neg",
            "ATEST.slot0.wait0 @r60, r60, k0, atest_datum.w0",
        ],
        "`DISCARD.f32.lt` directly evaluates `r0 < -u0.w1` and marks failing fragment lanes as helper/discarded without branching, followed by `ATEST` to commit updated coverage to `r60`.",
    ),
    "NOP": (
        "Explicit scoreboard barrier before ATEST",
        "Fragment shader, Mali-G710 (v10)",
        [
            "; Fragment prologue before alpha/depth test",
        ],
        [
            "NOP.wait0126",
            "ATEST.slot0 @r60, r60, k0, atest_datum.w0",
        ],
        "`NOP.wait0126` performs no computation but stalls warp execution until asynchronous scoreboard slots 0, 1, 2, and 6 drain — required before issuing `ATEST` or `DISCARD`.",
    ),

    # ------------------------------------------------------------------
    # 11. Memory Load / Store & LEA
    # ------------------------------------------------------------------
    "LD_CHECKED": (
        "Robust bounds-checked SSBO load on Immortalis-G925 (v14)",
        "Compute shader, Immortalis-G925 (v14)",
        [
            "%idx = OpIAdd %uint (OpShiftLeftLogical %uint %wg_x %uint_5) %local_x",
            "%ptr = OpAccessChain %_ptr_Uniform_float %ssbo %int_0 %idx",
            "%val = OpLoad %float %ptr",
        ],
        [
            "LSHIFT.i32 r0, r57^, k0B0A0908.b1",
            "IADD.u32 r6, r0, k0F0E0D0C.b0",
            "LD_CHECKED.i32.slot0.wait0 @r7, u7, r6, k1n",
        ],
        "On Valhall v12+ (Immortalis-G925), `malioc` lowers robust Vulkan SSBO loads to `LD_CHECKED.i32`, which checks the byte offset `r6` against the buffer bound in hardware and returns `0` on out-of-bounds access without a branch.",
    ),
    "LD_PKA": (
        "Loading from a uniform/storage buffer via descriptor handle",
        "Vertex shader, Mali-G710 (v10)",
        [
            "%ptr = OpAccessChain %_ptr_Uniform_uint %ubo %int_0 %idx",
            "%val = OpLoad %uint %ptr",
        ],
        [
            "IMUL.u32 r0, r0^, k13121110.b0",
            "LD_PKA.i32.slot0.wait0 @r0, r0^, u12.w1",
        ],
        "`LD_PKA.i32` loads 32 bits at byte offset `r0^` from the buffer descriptor identified by handle `u12.w1`, performing hardware descriptor resolution and bounds checking in one message.",
    ),
    "LEA_BUF_IMM": (
        "Computing IDVS varying-buffer output address in a Vertex shader",
        "Vertex shader, Mali-G710 (v10)",
        [
            "OpStore %out_uv %attr_uv",
        ],
        [
            "LD_ATTR_IMM.slot0.v2.f32 @r0:r1, r60^, r61^, #0x1, #0x1",
            "LEA_BUF_IMM.slot0 @r4:r5, r59^, #0x0, #0x3d",
            "STORE.i128.slot0.end.estream @r0:r1:r2:r3, [r4^:r5^], #0x0",
        ],
        "In Mali IDVS vertex/varying shaders, `LEA_BUF_IMM` resolves buffer descriptor `(table=#0x3d, index=#0x0)` for vertex index `r59^` into a 64-bit GPU VA in `@r4:r5`, which `STORE.i128.estream` writes to.",
    ),
    "LEA_TEX_IMM": (
        "Computing storage-image texel address + conversion descriptor for OpImageWrite",
        "Compute shader, Mali-G710 (v10)",
        [
            "OpImageWrite %storage_img %coord_xy %rgba_val None",
        ],
        [
            "SWZ.v2i16 r0.h1, r1^.h10          ; pack (x, y) into u16vec2",
            "LEA_TEX_IMM.slot0.wait02 @r0:r1:r2, r0^, k0, #0x6, #0x4",
            "ST_CVT.slot0.reconverge.estream.v4.f32 @r8:r9:r10:r11, [r0^:r1^], #0x0, r2^",
        ],
        "`OpImageWrite` is a two-message sequence on Valhall: `LEA_TEX_IMM` resolves image descriptor `(table=#0x4, index=#0x6)` at `(x, y)` into 3 staging registers (`@r0:r1` = 64-bit texel address, `r2` = 32-bit format conversion descriptor), and `ST_CVT` converts and stores `@r8:r9:r10:r11`.",
    ),
    "LOAD": (
        "Direct 64-bit virtual-address load with immediate struct offset",
        "Vertex shader, Mali-G710 (v10)",
        [
            "%ptr = OpAccessChain %_ptr_Uniform_uint %ssbo %int_0 %idx %int_113 ; offset 0x1c4",
            "%val = OpLoad %uint %ptr",
        ],
        [
            "IMUL.u32 r1, r0, k13121110.b0",
            "IADD.s64 [r2:r3], u8, [r0:r1].w1",
            "LOAD.i32.slot0.wait0 @r1, [r2^:r3^], #0x1c4",
        ],
        "Once the 64-bit base pointer `[r2:r3]` is computed by adding the strided element offset to base VA `u8`, `LOAD.i32` folds the constant struct member offset `#0x1c4` directly into the load instruction.",
    ),
    "ST_CHECKED": (
        "Robust bounds-checked SSBO store on Immortalis-G925 (v14)",
        "Compute shader, Immortalis-G925 (v14)",
        [
            "%ptr = OpAccessChain %_ptr_Uniform_float %ssbo %int_0 %idx",
            "OpStore %ptr %val",
        ],
        [
            "LSHIFT.i32.wait012 r2, r0, k07060504.b0 ; byte offset = idx << 2",
            "ST_CHECKED.i32.slot0 @r1, u2, r2, k1n",
        ],
        "`ST_CHECKED.i32` writes `r1` to base `u2` at byte offset `r2`, silently dropping the store if out-of-bounds under Vulkan robust buffer access.",
    ),
    "ST_CVT": (
        "Format-converting storage-image store paired with LEA_TEX_IMM",
        "Compute shader, Mali-G710 (v10)",
        [
            "OpImageWrite %storage_img %coord_xy %rgba_val None",
        ],
        [
            "LEA_TEX_IMM.slot0.wait02 @r0:r1:r2, r0^, k0, #0x6, #0x4",
            "ST_CVT.slot0.reconverge.estream.v4.f32 @r8:r9:r10:r11, [r0^:r1^], #0x0, r2^",
        ],
        "`ST_CVT` encodes the four FP32 channels in `@r8:r9:r10:r11` into the storage image's memory format using conversion descriptor `r2^` produced by `LEA_TEX_IMM`.",
    ),
    "STORE": (
        "Coalesced 128-bit compute SSBO store with .end",
        "Compute shader, Mali-G710 (v10)",
        [
            "OpStore %out_buf_0 %uint_1000",
            "OpStore %out_buf_1 %uint_4294966296",
            "OpStore %out_buf_2 %uint_0",
            "OpStore %out_buf_3 %uint_0",
        ],
        [
            "STORE.i128.slot0.end @r0:r1:r2:r3, u0, #0x0",
        ],
        "`STORE.i128.slot0.end` writes 16 bytes from `@r0:r1:r2:r3` to the 64-bit base VA in `u0` and simultaneously terminates the shader warp via `.end`.",
    ),

    # ------------------------------------------------------------------
    # 12. Atomics
    # ------------------------------------------------------------------
    "ATOM": (
        "Fire-and-forget unsigned atomic maximum (return value unused)",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%old = OpAtomicUMax %uint %ptr %uint_1 %uint_0 %val ; %old unused",
        ],
        [
            "IADD.u64.wait0 [r50:r51], u22, [r48:r49].w1",
            "ATOM.i32.slot0.aumax @r48, [r50^:r51^], #0x0",
        ],
        "Because the return value of `OpAtomicUMax` is unused in SPIR-V, `malioc` selects non-returning `ATOM.i32.aumax` instead of `ATOM_RETURN`, avoiding a destination staging register writeback and scoreboard stall.",
    ),

    # ------------------------------------------------------------------
    # 13. Attributes, Varyings & Fragment
    # ------------------------------------------------------------------
    "ATEST": (
        "Committing fragment alpha/discard coverage before termination",
        "Fragment shader, Mali-G710 (v10)",
        [
            "OpKill ; unconditional discard fragment shader",
        ],
        [
            "DISCARD.f32.wait0126.ne k1n, k0",
            "ATEST.slot0.end @r60, r60, k0, atest_datum.w0",
        ],
        "Even in a 2-instruction unconditional-discard shader, `DISCARD` must be followed by `ATEST` (updating pinned coverage register `r60` against `atest_datum.w0`) before terminating with `.end`.",
    ),
    "BLEND": (
        "Writing FP16 RGBA output to the tilebuffer and ending the warp",
        "Fragment shader, Mali-G710 (v10)",
        [
            "OpStore %out_color %rgba_const",
        ],
        [
            "MOV_IMM.i32 r0, #0x3c003c00",
            "MOV_IMM.i32.wait r1.h0, #0x3c003c00",
            "BLEND.slot0.end.v4.f16 @r0:r1, blend_descriptor_0, 0x20, r60^",
        ],
        "`BLEND` emits the final fragment color (`@r0:r1` as 4x FP16) to render target 0 via `blend_descriptor_0` masked by sample coverage `r60^`, and terminates the fragment warp with `.end`.",
    ),
    "LD_ATTR_IMM": (
        "Fetching a vec2 vertex input attribute in a Vertex shader",
        "Vertex shader, Mali-G710 (v10)",
        [
            "%uv = OpLoad %v2float %in_location_1",
        ],
        [
            "LD_ATTR_IMM.slot0.v2.f32 @r0:r1, r60^, r61^, #0x1, #0x1",
        ],
        "`LD_ATTR_IMM` fetches 2 FP32 components (`v2.f32`) into `@r0:r1` using hardware-preloaded `VertexIndex` (`r60^`) and `InstanceIndex` (`r61^`) from attribute descriptor `(table=#0x1, index=#0x1)`.",
    ),
    "LD_TEX_IMM": (
        "Direct storage-image texel read (OpImageRead)",
        "Compute shader, Mali-G710 (v10)",
        [
            "%texel = OpImageRead %v4uint %img %coord None",
        ],
        [
            "LD_TEX_IMM.slot0.u32 @r1, k0, k0, #0x4, #0x4",
        ],
        "`OpImageRead` lowers to `LD_TEX_IMM`, reading unsigned 32-bit texel data from descriptor `(table=#0x4, index=#0x4)` at integer coordinates `(src0, src1)`.",
    ),
    "LD_VAR_FLAT_IMM": (
        "Loading a flat-qualified fragment input varying",
        "Fragment shader, Mali-G710 (v10)",
        [
            "OpDecorate %in_flat Flat",
            "%v = OpLoad %float %in_flat",
        ],
        [
            "LD_VAR_FLAT_IMM.slot1.f32 @r4, #0x1, #0xb",
        ],
        "Because `%in_flat` carries `OpDecorate Flat`, `malioc` emits `LD_VAR_FLAT_IMM` (no barycentric coordinate operand needed) reading directly from the provoking vertex at `(table=#0xb, index=#0x1)`.",
    ),
    "LD_VAR_IMM": (
        "Interpolating a fragment varying and caching barycentrics with .store/.retrieve",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%v0 = OpLoad %float %in_loc_1",
            "%v1 = OpLoad %v4float %in_loc_0",
        ],
        [
            "LD_VAR_IMM.slot1.discard.f32.sample.store @r0, r61, #0x1, #0xb",
            "LD_VAR_IMM.slot0.wait0.v4.f32.retrieve @r4:r5:r6:r7, r61^, #0x0, #0xb",
        ],
        "The first `LD_VAR_IMM` uses `.sample.store` to compute and cache the perspective barycentric weights in hardware; the second uses `.retrieve` to reuse those cached weights without recomputing `1/w`.",
    ),
    "LD_VAR_SPECIAL": (
        "Loading gl_FragCoord.w via .frag_w special varying",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%p_w = OpAccessChain %_ptr_Input_float %gl_FragCoord %int_3",
            "%w   = OpLoad %float %p_w",
        ],
        [
            "LD_VAR_SPECIAL.slot0.wait0.frag_w.f32.center.clobber @r2, k0",
        ],
        "`gl_FragCoord.w` does not use a user varying slot; `malioc` fetches it directly from the fixed-function rasterizer via `LD_VAR_SPECIAL.frag_w`.",
    ),
    "ZS_EMIT": (
        "Exporting shader-computed gl_FragDepth",
        "Fragment shader, Mali-G710 (v10)",
        [
            "OpDecorate %gl_FragDepth BuiltIn FragDepth",
            "OpStore %gl_FragDepth %depth_val",
        ],
        [
            "TEX_SINGLE.slot0.wait0.skip.tex2d.dst32.r.wide.computed_force_delta @r0, @r0:r1, u1",
            "ZS_EMIT.slot0.wait0.z @r60, r0, k0, r60^",
        ],
        "Writing `gl_FragDepth` lowers to `ZS_EMIT.z`, sending the sampled depth `r0` (with stencil unused = `k0`) and current coverage `r60^` to the late-ZS unit and receiving the updated post-depth-test coverage mask back into `@r60`.",
    ),

    # ------------------------------------------------------------------
    # 14. Texture & Varying-Texture
    # ------------------------------------------------------------------
    "TEX_FETCH": (
        "Unfiltered integer texelFetch with explicit LOD 0",
        "Vertex shader, Mali-G710 (v10)",
        [
            "%t = OpImageFetch %v4float %img %ivec2_xy Lod %int_0",
        ],
        [
            "TEX_FETCH.slot0.tex2d.dst32.rgba @r20:r21:r22:r23, @r10:r11, [r4:r5]",
        ],
        "`OpImageFetch` lowers directly to `TEX_FETCH.tex2d.dst32.rgba`, reading integer coordinates from `@r10:r11` and the image descriptor handle from `[r4:r5]` without filtering.",
    ),
    "TEX_GATHER": (
        "2x2 red-channel textureGather for min/max depth reduction",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%g = OpImageGather %v4float %sampled_img %uv %int_0 None",
        ],
        [
            "TEX_GATHER.slot0.wait0.skip.tex2d.dst32.rgba.wide.float_coordinates.gather4_r @r0:r1:r2:r3, @r0:r1, u0",
        ],
        "`OpImageGather` with component `0` (`R`) lowers to `TEX_GATHER` with `.gather4_r` and `.float_coordinates`, returning the 4 footprint red texels into `@r0:r1:r2:r3`.",
    ),
    "TEX_GRADIENT": (
        "Querying hardware texture LOD via OpImageQueryLod",
        "Fragment shader, Mali-G710 (v10)",
        [
            "%lod = OpImageQueryLod %v2float %sampled_img %uv",
        ],
        [
            "TEX_GRADIENT.slot0.discard.skip.tex2d.dst32.r.wide.disable_lod_clamp @r2, @r0:r1, u2",
            "S16_TO_F32 r2, r2^.h0",
            "FMUL.f32 r2, r2^, u3.w0",
        ],
        "Instead of sampling texels, `TEX_GRADIENT` with `.disable_lod_clamp` queries the texture unit's implicit mipmap LOD calculation and returns the signed 8.8 fixed-point LOD in `r2.h0`.",
    ),
    "TEX_SINGLE": (
        "Sampling a 2D texture array with explicit LOD 0 in a Vertex shader",
        "Vertex shader, Mali-G710 (v10)",
        [
            "%color = OpImageSampleExplicitLod %v4float %tex2d_arr %uv_layer Lod %float_0",
        ],
        [
            "MOV_IMM.i32 r0, #0x3f000000       ; u = 0.5",
            "MOV.i32 r1, r0                    ; v = 0.5",
            "MOV_IMM.i32 r2, #0x40000000       ; layer = 2.0",
            "TEX_SINGLE.slot0.wait0.tex2d.dst32.rgba.wide.array @r0:r1:r2:r3, @r0:r1:r2, u3",
        ],
        "`OpImageSampleExplicitLod` on a `2DArray` sampler lowers to `TEX_SINGLE` with `.tex2d.array` (default LOD mode = zero/explicit) reading `(u, v, layer)` from `@r0:r1:r2` and writing 4 FP32 channels to `@r0:r1:r2:r3`.",
    ),
}


def render_mined_example_tex(mnem: str) -> str:
    """Render a mined SPIR-V -> Valhall ISA worked example block if available."""
    entry = MINED_SPIRV_EXAMPLES.get(mnem)
    if entry is None:
        return ""
    title, stage_core, spv_lines, vh_lines, explain_md = entry
    spv_tex = r"\\".join(tex_esc_literal(line) for line in spv_lines)
    vh_tex = r"\\".join(tex_esc_literal(line) for line in vh_lines)
    return (
        f"\\par\\textbf{{Mined SPIR-V $\\to$ Valhall ISA Example ({tex_esc(title)} --- {tex_esc(stage_core)}):}}\n"
        f"\\codebox{{\\textbf{{; SPIR-V (input)}}\\\\{spv_tex}\\\\[3pt]\\textbf{{; Valhall ISA (malioc output)}}\\\\{vh_tex}}}\n"
        f"{md_inline_to_tex(explain_md)}"
    )

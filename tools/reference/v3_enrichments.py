#!/usr/bin/env python3
"""v3 Enrichments for valhall-isa-reference.tex:
1. Per-entry descriptions & color-coded worked examples for all 128 *_m enums (596 values).
2. Uncommon instruction field explanations with worked example, SPIR-V op, and GLSL snippet.
3. Appendix A.1 (all 12 DDK-only opcodes: usage, induced semantics, AArch64 disasm, C/pseudocode).
4. Appendix A.2 (all 12 DDK-vs-Panfrost diffs: AArch64 disasm AND C/pseudocode decompilation).
"""
import re
from .fmt_helpers import tex_esc_literal as tex_esc, mnem, imod, dst, src, imm

# ============================================================================
# 1. Per-Entry Descriptions & Worked Examples for ALL 128 *_m Enums
# ============================================================================

FLOW_CONTROL_ENTRIES = {
    "none": ("No scoreboard wait or warp control action; continue immediately to next instruction.",
             f"{mnem('FADD.f32')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    "wait0": ("Stall warp until all asynchronous messages in scoreboard slot 0 complete.",
              f"{mnem('FADD.f32')}{imod('.wait0')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    "wait1": ("Stall warp until all asynchronous messages in scoreboard slot 1 complete.",
              f"{mnem('FADD.f32')}{imod('.wait1')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    "wait01": ("Stall warp until both scoreboard slots 0 and 1 drain completely.",
               f"{mnem('FMUL.f32')}{imod('.wait01')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    "wait2": ("Stall warp until all asynchronous messages in scoreboard slot 2 complete.",
              f"{mnem('IADD.s32')}{imod('.wait2')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    "wait02": ("Stall warp until both scoreboard slots 0 and 2 drain completely.",
               f"{mnem('FMA.f32')}{imod('.wait02')} {dst('r0')}, {src('r1')}, {src('r2')}, {src('r3')}"),
    "wait12": ("Stall warp until both scoreboard slots 1 and 2 drain completely.",
               f"{mnem('FMA.f32')}{imod('.wait12')} {dst('r0')}, {src('r1')}, {src('r2')}, {src('r3')}"),
    "wait012": ("Stall warp until scoreboard slots 0, 1, and 2 all complete.",
                f"{mnem('MOV.i32')}{imod('.wait012')} {dst('r0')}, {src('r1')}"),
    "wait0126": ("Stall warp until slots 0, 1, 2, and tile/blend slot 6 all complete.",
                 f"{mnem('NOP')}{imod('.wait0126')}"),
    "wait": ("Full scoreboard barrier: stall until all in-flight slots (0--7) complete.",
             f"{mnem('NOP')}{imod('.wait')}"),
    "reconverge": ("Signal hardware warp reconvergence point after divergent control flow.",
                   f"{mnem('FADD.f32')}{imod('.reconverge')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    "wait_resource": ("Wait for hardware resource availability (v10--v13 barrier/queue sync).",
                      f"{mnem('NOP')}{imod('.wait_resource')}"),
    "discard": ("Mark active fragment lanes as helper/discarded (suppresses side effects \& writes).",
                f"{mnem('NOP')}{imod('.discard')}"),
    "continue": ("Continue hardware-managed loop/warp execution (v14+).",
                 f"{mnem('NOP')}{imod('.continue')}"),
    "end": ("Terminate shader warp execution and release register/tile resources.",
            f"{mnem('NOP')}{imod('.end')}"),
}

CLPER_INACTIVE_ENTRIES = {
    "zero": ("Return 0x00000000 for inactive source lanes.", f"{mnem('CLPER')}{imod('.zero')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    "umax": ("Return 0xFFFFFFFF (UINT32\\_MAX) for inactive lanes.", f"{mnem('CLPER')}{imod('.umax')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    "i1": ("Return 0x00000001 (int32 1) for inactive lanes.", f"{mnem('CLPER')}{imod('.i1')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    "v2i1": ("Return 0x00010001 (v2i16 {1, 1}) for inactive lanes.", f"{mnem('CLPER')}{imod('.v2i1')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    "smin": ("Return 0x80000000 (INT32\\_MIN) for inactive lanes.", f"{mnem('CLPER')}{imod('.smin')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    "smax": ("Return 0x7FFFFFFF (INT32\\_MAX) for inactive lanes.", f"{mnem('CLPER')}{imod('.smax')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    "v2smin": ("Return 0x80008000 (v2s16 {INT16\\_MIN, INT16\\_MIN}).", f"{mnem('CLPER')}{imod('.v2smin')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    "v2smax": ("Return 0x7FFF7FFF (v2s16 {INT16\\_MAX, INT16\\_MAX}).", f"{mnem('CLPER')}{imod('.v2smax')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    "v4smin": ("Return 0x80808080 (v4s8 {-128, -128, -128, -128}).", f"{mnem('CLPER')}{imod('.v4smin')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    "v4smax": ("Return 0x7F7F7F7F (v4s8 {127, 127, 127, 127}).", f"{mnem('CLPER')}{imod('.v4smax')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    "f1": ("Return 0x3F800000 (FP32 +1.0) for inactive lanes.", f"{mnem('CLPER')}{imod('.f1')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    "v2f1": ("Return 0x3C003C00 (v2f16 {+1.0, +1.0}) for inactive lanes.", f"{mnem('CLPER')}{imod('.v2f1')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    "inf": ("Return 0x7F800000 (FP32 +Inf) for inactive lanes (min identity).", f"{mnem('CLPER')}{imod('.inf')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    "ninf": ("Return 0xFF800000 (FP32 -Inf) for inactive lanes (max identity).", f"{mnem('CLPER')}{imod('.ninf')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    "v2inf": ("Return 0x7C007C00 (v2f16 {+Inf, +Inf}) for inactive lanes.", f"{mnem('CLPER')}{imod('.v2inf')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    "v2ninf": ("Return 0xFC00FC00 (v2f16 {-Inf, -Inf}) for inactive lanes.", f"{mnem('CLPER')}{imod('.v2ninf')} {dst('r0')}, {src('r1')}, {src('r2')}"),
}

CLPER_LANEOP_ENTRIES = {
    "none": ("Direct lane index lookup: read from lane \texttt{src1 \& (N-1)}.", f"{mnem('CLPER')}{imod('.none')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    "xor": ("Butterfly XOR shuffle: read from lane \texttt{my\_lane \textasciicircum{} src1}.", f"{mnem('CLPER')}{imod('.xor')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    "accumulate": ("Tree-reduction accumulate helper across subgroup lanes.", f"{mnem('CLPER')}{imod('.accumulate')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    "shift": ("Subgroup shift up/down: read from \texttt{my\_lane - src1} (clamped/inactive if OOB).", f"{mnem('CLPER')}{imod('.shift')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    "rotate": ("Cyclic subgroup rotate: read from \texttt{(my\_lane + src1) mod N}.", f"{mnem('CLPER')}{imod('.rotate')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    "low": ("Broadcast lowest active lane index in the subgroup.", f"{mnem('CLPER')}{imod('.low')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    "low_alt": ("Alternate low-lane broadcast mode for quad/clustered ops.", f"{mnem('CLPER')}{imod('.low_alt')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    "prefix": ("Inclusive/exclusive prefix-scan lane index computation.", f"{mnem('CLPER')}{imod('.prefix')} {dst('r0')}, {src('r1')}, {src('r2')}"),
}

EXACT_ENUM_ENTRIES = {
    "flow_control_m": FLOW_CONTROL_ENTRIES,
    "clper_inactive_result_m": CLPER_INACTIVE_ENTRIES,
    "clper_lane_op_m": CLPER_LANEOP_ENTRIES,
    "clamp_m": {
        "none": ("No output clamping; pass IEEE-754 result through unchanged.", f"{mnem('FADD.f32')} {dst('r0')}, {src('r1')}, {src('r2')}"),
        "clamp_0_inf": ("Clamp to $[+0.0, +\\infty]$; negative values, $-0.0$, and NaN become $+0.0$.", f"{mnem('FADD.f32')}{imod('.clamp_0_inf')} {dst('r0')}, {src('r1')}, {src('r2')}"),
        "clamp_m1_1": ("Clamp to $[-1.0, +1.0]$; NaN becomes $-1.0$ (\\texttt{0xBF800000}) per DDK.", f"{mnem('FMA.f32')}{imod('.clamp_m1_1')} {dst('r0')}, {src('r1')}, {src('r2')}, {src('r3')}"),
        "clamp_0_1": ("Clamp to $[+0.0, +1.0]$ (saturate); $-0.0$ and NaN become $+0.0$.", f"{mnem('FMA.f32')}{imod('.clamp_0_1')} {dst('r0')}, {src('r1')}, {src('r2')}, {src('r3')}"),
    },
    "sem_m": {
        "nan_suppress": ("IEEE-754 minNum/maxNum: non-NaN wins over NaN; $-0.0 < +0.0$ deterministically.", f"{mnem('FMIN.f32')}{imod('.nan_suppress')} {dst('r0')}, {src('r1')}, {src('r2')}"),
        "nan_propagate": ("Propagate NaN: if either input is NaN, return quietened winning NaN preserving payload.", f"{mnem('FMAX.f32')}{imod('.nan_propagate')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    },
    "cmpf_m": {
        "eq": ("Equal ($a == b$); false if either float input is NaN.", f"{mnem('FCMP.f32')}{imod('.eq.i1')} {dst('r0')}, {src('r1')}, {src('r2')}"),
        "gt": ("Greater than ($a > b$); false on NaN.", f"{mnem('ICMP.s32')}{imod('.gt.i1')} {dst('r0')}, {src('r1')}, {src('r2')}"),
        "ge": ("Greater than or equal ($a \\ge b$); false on NaN.", f"{mnem('FCMP.f32')}{imod('.ge.i1')} {dst('r0')}, {src('r1')}, {src('r2')}"),
        "ne": ("Not equal ($a \\ne b$); true if either float input is NaN (unordered).", f"{mnem('FCMP.f32')}{imod('.ne.i1')} {dst('r0')}, {src('r1')}, {src('r2')}"),
        "lt": ("Less than ($a < b$); false on NaN.", f"{mnem('ICMP.u32')}{imod('.lt.i1')} {dst('r0')}, {src('r1')}, {src('r2')}"),
        "le": ("Less than or equal ($a \\le b$); false on NaN.", f"{mnem('FCMP.f32')}{imod('.le.i1')} {dst('r0')}, {src('r1')}, {src('r2')}"),
        "gtlt": ("Ordered not-equal ($a < b \\lor a > b$); false if either float is NaN.", f"{mnem('FCMP.f32')}{imod('.gtlt.i1')} {dst('r0')}, {src('r1')}, {src('r2')}"),
        "total": ("IEEE-754 totalOrder predicate (orders $-0.0 < +0.0$ and NaN bit patterns).", f"{mnem('FCMP.f32')}{imod('.total.i1')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    },
    "cmp_result_type_m": {
        "i1": ("Integer boolean result: \\texttt{1} if true, \\texttt{0} if false.", f"{mnem('ICMP.u32')}{imod('.eq.i1')} {dst('r0')}, {src('r1')}, {src('r2')}"),
        "f1": ("Floating-point boolean result: \\texttt{1.0} (\\texttt{0x3F800000}) if true, \\texttt{0.0} if false.", f"{mnem('FCMP.f32')}{imod('.lt.f1')} {dst('r0')}, {src('r1')}, {src('r2')}"),
        "m1": ("Bitmask boolean result: all-ones (\\texttt{0xFFFFFFFF} / \\texttt{-1}) if true, \\texttt{0} if false.", f"{mnem('ICMP.s32')}{imod('.ge.m1')} {dst('r0')}, {src('r1')}, {src('r2')}"),
        "c": ("Carry/chained comparison result (legal on \\texttt{ICMP\\_MULTI} only per DDK \\texttt{ModInfo}).", f"{mnem('ICMP_MULTI.u32')}{imod('.eq.c')} {dst('r0')}, {src('r1')}, {src('r2')}, {src('r3')}"),
    },
    "round_m": {
        "none": ("Round to nearest ties-to-even (RTE, default IEEE-754 mode).", f"{mnem('FADD.f32')} {dst('r0')}, {src('r1')}, {src('r2')}"),
        "round_up": ("Round toward $+\\infty$ (RTP / ceiling).", f"{mnem('FADD.f32')}{imod('.round_up')} {dst('r0')}, {src('r1')}, {src('r2')}"),
        "round_down": ("Round toward $-\\infty$ (RTN / floor).", f"{mnem('FADD.f32')}{imod('.round_down')} {dst('r0')}, {src('r1')}, {src('r2')}"),
        "round_zero": ("Round toward zero (RTZ / truncate).", f"{mnem('FADD.f32')}{imod('.round_zero')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    },
    "round_extended_m": {
        "none": ("Round to nearest ties-to-even (RTE).", f"{mnem('S32_TO_F32')} {dst('r0')}, {src('r1')}"),
        "round_up": ("Round toward $+\\infty$ (RTP).", f"{mnem('S32_TO_F32')}{imod('.round_up')} {dst('r0')}, {src('r1')}"),
        "round_down": ("Round toward $-\\infty$ (RTN).", f"{mnem('S32_TO_F32')}{imod('.round_down')} {dst('r0')}, {src('r1')}"),
        "round_zero": ("Round toward zero (RTZ).", f"{mnem('S32_TO_F32')}{imod('.round_zero')} {dst('r0')}, {src('r1')}"),
        "round_na": ("Round to nearest ties-away-from-zero (RNA).", f"{mnem('S32_TO_F32')}{imod('.round_na')} {dst('r0')}, {src('r1')}"),
    },
    "round_integer_m": {
        "none": ("Round float to integer using nearest ties-to-even (RTE).", f"{mnem('F32_TO_S32')} {dst('r0')}, {src('r1')}"),
        "round_up": ("Round float to integer toward $+\\infty$ (\\texttt{ceil}).", f"{mnem('F32_TO_S32')}{imod('.round_up')} {dst('r0')}, {src('r1')}"),
        "round_down": ("Round float to integer toward $-\\infty$ (\\texttt{floor}).", f"{mnem('F32_TO_S32')}{imod('.round_down')} {dst('r0')}, {src('r1')}"),
        "round_zero": ("Truncate fractional part toward zero (GLSL \\texttt{int(x)}).", f"{mnem('F32_TO_S32')}{imod('.round_zero')} {dst('r0')}, {src('r1')}"),
        "round_na": ("Round to nearest integer, ties away from zero.", f"{mnem('F32_TO_S32')}{imod('.round_na')} {dst('r0')}, {src('r1')}"),
    },
    "ldexp_round_m": {
        "none": ("Scale by $2^e$ with round-to-nearest ties-to-even (RTE).", f"{mnem('LDEXP.f32')} {dst('r0')}, {src('r1')}, {src('r2')}"),
        "round_up": ("Scale by $2^e$ rounding toward $+\\infty$.", f"{mnem('LDEXP.f32')}{imod('.round_up')} {dst('r0')}, {src('r1')}, {src('r2')}"),
        "round_down": ("Scale by $2^e$ rounding toward $-\\infty$.", f"{mnem('LDEXP.f32')}{imod('.round_down')} {dst('r0')}, {src('r1')}, {src('r2')}"),
        "round_zero": ("Scale by $2^e$ rounding toward zero.", f"{mnem('LDEXP.f32')}{imod('.round_zero')} {dst('r0')}, {src('r1')}, {src('r2')}"),
        "round_na": ("Scale by $2^e$ rounding ties away from zero.", f"{mnem('LDEXP.f32')}{imod('.round_na')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    },
    "fma_rscale_special_32_m": {
        "none": ("Standard FP64-widened $2^e$-scaled FMA/FADD/FMUL.", f"{mnem('FMA_RSCALE.f32')} {dst('r0')}, {src('r1')}, {src('r2')}, {src('r3')}, {src('r4')}"),
        "n": ("If $|a|=0$ or $|b|=0$, force product to $\\pm 0.0$ instead of $0\\cdot\\infty=\\text{NaN}$.", f"{mnem('FMA_RSCALE.f32')}{imod('.n')} {dst('r0')}, {src('r1')}, {src('r2')}, {src('r3')}, {src('r4')}"),
        "left": ("If $0\\cdot\\infty$ or $\\infty\\cdot 0$, return left operand $a$ unmodified.", f"{mnem('FMUL_RSCALE.f32')}{imod('.left')} {dst('r0')}, {src('r1')}, {src('r2')}, {src('r3')}"),
        "n_add": ("Suppress $\\infty + (-\\infty)$ NaN during range reduction (valid on FMA/FADD\\_RSCALE only).", f"{mnem('FADD_RSCALE.f32')}{imod('.n_add')} {dst('r0')}, {src('r1')}, {src('r2')}, {src('r3')}"),
    },
    "frexp_special_m": {
        "none": ("Standard IEEE \\texttt{frexp}: mantissa in $[0.5, 1.0)$, exponent $e - 126$.", f"{mnem('FREXPM.f32')} {dst('r0')}, {src('r1')}"),
        "sqrt": ("Square-root decomposition: mantissa in $[0.25, 1.0)$, even exponent $(e-125)\\gg 1$.", f"{mnem('FREXPE.f32')}{imod('.sqrt')} {dst('r0')}, {src('r1')}"),
        "log": ("Logarithm decomposition: mantissa in $[0.75, 1.5)$; $\\pm\\infty \\to \\pm 1.0$ on \\texttt{FREXPM}.", f"{mnem('FREXPM.f32')}{imod('.log')} {dst('r0')}, {src('r1')}"),
    },
    "ldexp_inf_m": {
        "none": ("Standard \\texttt{ldexp(x, e)} without extreme-exponent $\\pm\\infty/\\pm 0$ overrides.", f"{mnem('LDEXP.f32')} {dst('r0')}, {src('r1')}, {src('r2')}"),
        "inf": ("For $e < -16384$: $\\pm\\infty \\to \\pm 1.0$ and finite $\\to \\pm 0.0$.", f"{mnem('LDEXP.f32')}{imod('.inf')} {dst('r0')}, {src('r1')}, {src('r2')}"),
        "inf0": ("Also for $e \\ge +16384$: $\\pm 0.0 \\to \\pm 1.0$ and non-zero $\\to \\pm\\infty$.", f"{mnem('LDEXP.f32')}{imod('.inf0')} {dst('r0')}, {src('r1')}, {src('r2')}"),
    },
    "access_load_m": {
        "none": ("Default L1/L2 cached load policy.", f"{mnem('LOAD.i32')} {dst('@r0')}, {src('r2')}, {imm('#0')}"),
        "istream": ("Inner-streaming hint: bypass/deprioritize L1 cache allocation.", f"{mnem('LOAD.i32')}{imod('.istream')} {dst('@r0')}, {src('r2')}, {imm('#0')}"),
        "estream": ("Outer-streaming hint: streaming access through L2/outer caches.", f"{mnem('LOAD.i32')}{imod('.estream')} {dst('@r0')}, {src('r2')}, {imm('#0')}"),
        "force": ("Force cache coherence / bypass stale cache lines (volatile/coherent read).", f"{mnem('LOAD.i32')}{imod('.force')} {dst('@r0')}, {src('r2')}, {imm('#0')}"),
    },
    "access_store_m": {
        "none": ("Default write-back cached store policy.", f"{mnem('STORE.i32')} {src('@r0')}, {src('r2')}, {imm('#0')}"),
        "istream": ("Inner-streaming store hint: avoid polluting L1 cache.", f"{mnem('STORE.i32')}{imod('.istream')} {src('@r0')}, {src('r2')}, {imm('#0')}"),
        "estream": ("Outer-streaming store hint: stream past L2 toward memory.", f"{mnem('STORE.i32')}{imod('.estream')} {src('@r0')}, {src('r2')}, {imm('#0')}"),
        "force": ("Force write-through/coherent visibility (coherent SSBO/image store).", f"{mnem('STORE.i32')}{imod('.force')} {src('@r0')}, {src('r2')}, {imm('#0')}"),
    },
    "atom_operation_m": {
        "aaddu": ("Atomic unsigned add (saturating/wrapping per opcode): \\texttt{*p += v}.", f"{mnem('ATOM.i32')}{imod('.aaddu')} {src('@r0')}, {src('r2')}, {imm('#0')}"),
        "aadds": ("Atomic signed add: \\texttt{*p += (int)v}.", f"{mnem('ATOM.i32')}{imod('.aadds')} {src('@r0')}, {src('r2')}, {imm('#0')}"),
        "aadd": ("Atomic integer add (wrapping two's complement): \\texttt{*p += v}.", f"{mnem('ATOM.i32')}{imod('.aadd')} {src('@r0')}, {src('r2')}, {imm('#0')}"),
        "asmin": ("Atomic signed minimum: \\texttt{*p = min((int)*p, (int)v)}.", f"{mnem('ATOM.i32')}{imod('.asmin')} {src('@r0')}, {src('r2')}, {imm('#0')}"),
        "asmax": ("Atomic signed maximum: \\texttt{*p = max((int)*p, (int)v)}.", f"{mnem('ATOM.i32')}{imod('.asmax')} {src('@r0')}, {src('r2')}, {imm('#0')}"),
        "aumin": ("Atomic unsigned minimum: \\texttt{*p = min((uint)*p, (uint)v)}.", f"{mnem('ATOM.i32')}{imod('.aumin')} {src('@r0')}, {src('r2')}, {imm('#0')}"),
        "aumax": ("Atomic unsigned maximum: \\texttt{*p = max((uint)*p, (uint)v)}.", f"{mnem('ATOM.i32')}{imod('.aumax')} {src('@r0')}, {src('r2')}, {imm('#0')}"),
        "aand": ("Atomic bitwise AND: \\texttt{*p \\&= v}.", f"{mnem('ATOM.i32')}{imod('.aand')} {src('@r0')}, {src('r2')}, {imm('#0')}"),
        "aor": ("Atomic bitwise OR: \\texttt{*p |= v}.", f"{mnem('ATOM.i32')}{imod('.aor')} {src('@r0')}, {src('r2')}, {imm('#0')}"),
        "axor": ("Atomic bitwise XOR: \\texttt{*p \\^= v}.", f"{mnem('ATOM.i32')}{imod('.axor')} {src('@r0')}, {src('r2')}, {imm('#0')}"),
        "axchg": ("Atomic exchange: \\texttt{old = *p; *p = v}.", f"{mnem('ATOM_RETURN.i32')}{imod('.axchg')} {dst('@r0')}, {src('r2')}, {imm('#0')}"),
    },
    "atom1_operation_m": {
        "ainc": ("Unary atomic increment: \\texttt{*p += 1}.", f"{mnem('ATOM1.i32')}{imod('.ainc')} {src('r2')}, {imm('#0')}"),
        "adec": ("Unary atomic decrement: \\texttt{*p -= 1}.", f"{mnem('ATOM1.i32')}{imod('.adec')} {src('r2')}, {imm('#0')}"),
        "aumax1": ("Unary atomic unsigned max with 1: \\texttt{*p = max(*p, 1u)}.", f"{mnem('ATOM1.i32')}{imod('.aumax1')} {src('r2')}, {imm('#0')}"),
        "asmax1": ("Unary atomic signed max with 1: \\texttt{*p = max((int)*p, 1)}.", f"{mnem('ATOM1.i32')}{imod('.asmax1')} {src('r2')}, {imm('#0')}"),
        "aor1": ("Unary atomic bitwise OR with 1: \\texttt{*p |= 1}.", f"{mnem('ATOM1.i32')}{imod('.aor1')} {src('r2')}, {imm('#0')}"),
    },
    "branch_combine_m": {
        "none": ("Test full 32-bit predicate operand \\texttt{src0 != 0}.", f"{mnem('BRANCH')} {src('r0')}, {imm('#+4')}"),
        "h0": ("Test low 16-bit halfword \\texttt{src0[15:0] != 0}.", f"{mnem('BRANCH')}{imod('.h0')} {src('r0')}, {imm('#+4')}"),
        "h1": ("Test high 16-bit halfword \\texttt{src0[31:16] != 0}.", f"{mnem('BRANCH')}{imod('.h1')} {src('r0')}, {imm('#+4')}"),
        "and": ("Branch if both halfwords are non-zero (\\texttt{h0 != 0 \\&\\& h1 != 0}).", f"{mnem('BRANCH')}{imod('.and')} {src('r0')}, {imm('#+4')}"),
        "lowbits": ("Test low bit(s) of predicate lane(s) for warp condition.", f"{mnem('BRANCH')}{imod('.lowbits')} {src('r0')}, {imm('#+4')}"),
    },
    "mux_cmpf_m": {
        "neg": ("Select \\texttt{src0} if sign bit (bit 31) of \\texttt{src2} is 1, else \\texttt{src1}.", f"{mnem('MUX.i32')}{imod('.neg')} {dst('r0')}, {src('r1')}, {src('r2')}, {src('r3')}"),
        "int_zero": ("Select \\texttt{src0} if integer \\texttt{src2 == 0}, else \\texttt{src1}.", f"{mnem('MUX.i32')}{imod('.int_zero')} {dst('r0')}, {src('r1')}, {src('r2')}, {src('r3')}"),
        "fp_zero": ("Select \\texttt{src0} if float \\texttt{src2 == ±0.0}, else \\texttt{src1}.", f"{mnem('MUX.i32')}{imod('.fp_zero')} {dst('r0')}, {src('r1')}, {src('r2')}, {src('r3')}"),
        "bit": ("Per-bit blend: \\texttt{(src0 \\& src2) | (src1 \\& ~src2)}.", f"{mnem('MUX.i32')}{imod('.bit')} {dst('r0')}, {src('r1')}, {src('r2')}, {src('r3')}"),
    },
    "message_slot_index_m": {
        "slot0": ("Track async message completion in scoreboard slot 0.", f"{mnem('LOAD.i32')}{imod('.slot0')} {dst('@r0')}, {src('r2')}, {imm('#0')}"),
        "slot1": ("Track async message completion in scoreboard slot 1.", f"{mnem('TEX_SINGLE')}{imod('.slot1')} {dst('@r0:r3')}, {src('@r4:r5')}"),
        "slot2": ("Track async message completion in scoreboard slot 2.", f"{mnem('LD_VAR_BUF.f32')}{imod('.slot2')} {dst('@r0:r3')}, {src('r4')}"),
        "slot7": ("Track async message in slot 7 (non-blocking / fire-and-forget write).", f"{mnem('STORE.i32')}{imod('.slot7')} {src('@r0')}, {src('r2')}, {imm('#0')}"),
    },
    "jump_stack_mode_m": {
        "return": ("Pop return address from hardware call stack and branch.", f"{mnem('JUMP_EX')}{imod('.return')} {src('r0')}"),
        "call": ("Push next PC onto hardware call stack and jump to target.", f"{mnem('JUMP_EX')}{imod('.call')} {src('r0')}, {imm('#+16')}"),
        "none": ("Indirect jump without modifying hardware call stack.", f"{mnem('JUMP_EX')} {src('r0')}, {src('r1')}"),
        "replace": ("Tail-call: replace top-of-stack frame without pushing.", f"{mnem('JUMP_EX')}{imod('.replace')} {src('r0')}, {src('r1')}"),
    },
    "register_file_format_general_m": {
        "auto32": ("Format determined by descriptor; write 32-bit GPRs.", f"{mnem('LD_ATTR')}{imod('.auto32.v4')} {dst('@r0:r3')}, {src('r4')}"),
        "f32": ("Convert descriptor data to 32-bit IEEE-754 float in GPRs.", f"{mnem('LD_ATTR')}{imod('.f32.v4')} {dst('@r0:r3')}, {src('r4')}"),
        "f16": ("Convert descriptor data to packed FP16 pairs in GPRs.", f"{mnem('LD_ATTR')}{imod('.f16.v4')} {dst('@r0:r1')}, {src('r4')}"),
        "s32": ("Sign-extend integer attribute/tile data to 32-bit signed int.", f"{mnem('LD_ATTR')}{imod('.s32.v4')} {dst('@r0:r3')}, {src('r4')}"),
        "s16": ("Convert integer attribute/tile data to packed signed 16-bit.", f"{mnem('LD_ATTR')}{imod('.s16.v4')} {dst('@r0:r1')}, {src('r4')}"),
        "u32": ("Zero-extend integer attribute/tile data to 32-bit unsigned int.", f"{mnem('LD_ATTR')}{imod('.u32.v4')} {dst('@r0:r3')}, {src('r4')}"),
        "u16": ("Convert integer attribute/tile data to packed unsigned 16-bit.", f"{mnem('LD_ATTR')}{imod('.u16.v4')} {dst('@r0:r1')}, {src('r4')}"),
    },
    "varying_name_m": {
        "point": ("Read \\texttt{gl\\_PointCoord} $(s, t)$ inside point sprite.", f"{mnem('LD_VAR_SPECIAL')}{imod('.point')} {dst('@r0:r1')}, {src('r2')}"),
        "bary": ("Read perspective/noperspective barycentric weights $(i, j)$.", f"{mnem('LD_VAR_SPECIAL')}{imod('.bary')} {dst('@r0:r1')}, {src('r2')}"),
        "frag_w": ("Read reciprocal clip-space $1/w$ (\\texttt{gl\\_FragCoord.w}).", f"{mnem('LD_VAR_SPECIAL')}{imod('.frag_w')} {dst('@r0')}, {src('r2')}"),
        "frag_z": ("Read interpolated window-space depth (\\texttt{gl\\_FragCoord.z}).", f"{mnem('LD_VAR_SPECIAL')}{imod('.frag_z')} {dst('@r0')}, {src('r2')}"),
    },
    "tex_dimensionality_m": {
        "tex1d": ("1D texture coordinate $(u)$.", f"{mnem('TEX_SINGLE')}{imod('.tex1d')} {dst('@r0:r3')}, {src('@r4')}"),
        "tex2d": ("2D texture coordinates $(u, v)$.", f"{mnem('TEX_SINGLE')}{imod('.tex2d')} {dst('@r0:r3')}, {src('@r4:r5')}"),
        "tex3d": ("3D volumetric texture coordinates $(u, v, w)$.", f"{mnem('TEX_SINGLE')}{imod('.tex3d')} {dst('@r0:r3')}, {src('@r4:r6')}"),
        "cube": ("Cubemap direction/face-projected coordinates.", f"{mnem('TEX_SINGLE')}{imod('.cube')} {dst('@r0:r3')}, {src('@r4:r6')}"),
    },
    "tex_lod_mode_m": {
        "none": ("Implicit zero LOD (base mip level 0, no derivatives).", f"{mnem('TEX_SINGLE')}{imod('.tex2d')} {dst('@r0:r3')}, {src('@r4:r5')}"),
        "computed": ("Hardware 2x2 quad finite-difference LOD (\\texttt{texture()}).", f"{mnem('TEX_SINGLE')}{imod('.tex2d.computed')} {dst('@r0:r3')}, {src('@r4:r5')}"),
        "explicit": ("Explicit LOD supplied in staging register (\\texttt{textureLod()}).", f"{mnem('TEX_SINGLE')}{imod('.tex2d.explicit')} {dst('@r0:r3')}, {src('@r4:r6')}"),
        "computed_bias": ("Computed quad LOD plus shader bias (\\texttt{texture(..., bias)}).", f"{mnem('TEX_SINGLE')}{imod('.tex2d.computed_bias')} {dst('@r0:r3')}, {src('@r4:r6')}"),
        "grdesc": ("Explicit $dP/dx, dP/dy$ gradients in staging regs (\\texttt{textureGrad()}).", f"{mnem('TEX_SINGLE')}{imod('.tex2d.grdesc')} {dst('@r0:r3')}, {src('@r4:r9')}"),
        "computed_force_delta": ("Computed LOD forcing quad delta evaluation even in helper lanes.", f"{mnem('TEX_SINGLE')}{imod('.tex2d.computed_force_delta')} {dst('@r0:r3')}, {src('@r4:r5')}"),
        "computed_bias_force_delta": ("Computed LOD + bias with forced quad delta evaluation.", f"{mnem('TEX_SINGLE')}{imod('.tex2d.computed_bias_force_delta')} {dst('@r0:r3')}, {src('@r4:r6')}"),
    },
}

def describe_enum_entry(ename: str, vname: str, venc: str, sample_ins: str):
    """Return (short_description_tex, color_coded_example_tex) for ANY (enum, value)."""
    if ename in EXACT_ENUM_ENTRIES and vname in EXACT_ENUM_ENTRIES[ename]:
        return EXACT_ENUM_ENTRIES[ename][vname]

    ins = sample_ins or "OP"
    mod_tok = "" if vname == "none" else f".{vname}"

    # Pattern-based exact semantic generators for every remaining enum family
    if ename == "ls_multi_sr_count_m":
        bits = (int(venc) + 1) * 32
        regs = int(venc) + 1
        r_end = regs - 1
        sr = f"@r0" if regs == 1 else f"@r0:r{r_end}"
        return (f"Transfer {regs} consecutive 32-bit staging register(s) ({bits} bits total).",
                f"{mnem(ins)}{imod('.' + vname)} {dst(sr)}, {src('r60')}, {imm('#0')}")

    if ename in ("dest_width_m", "dest_width_narrow_m", "dest_width_replicate_m", "dest_width_single_lane_16_bit_m"):
        if vname == "none":
            return ("Write full 32-bit destination register (both halfwords).", f"{mnem(ins + '.f32')} {dst('r0')}, {src('r1')}")
        if vname == "h0":
            return ("Write low 16-bit halfword \\texttt{dst[15:0]}; zero-fill upper \\texttt{dst[31:16]}.", f"{mnem(ins + '.f16')} {dst('r0')}{imod('.h0')}, {src('r1')}")
        if vname == "h1":
            return ("Write high 16-bit halfword \\texttt{dst[31:16]}; preserve/mask low halfword per lane-mask rule.", f"{mnem(ins + '.f16')} {dst('r0')}{imod('.h1')}, {src('r1')}")
        if vname == "h01":
            return ("Replicate/pack 16-bit result into both \\texttt{dst[15:0]} and \\texttt{dst[31:16]}.", f"{mnem(ins + '.f16')} {dst('r0')}{imod('.h01')}, {src('r1')}")

    if ename in ("widen_m", "widen_int_m", "lane16_m", "lane8_m", "lane_all_m", "load_lane_16_m", "load_lane_32_m"):
        m_lane = {
            "none": "Use full 32-bit operand without narrowing/widening.",
            "h0": "Select low 16-bit halfword \\texttt{[15:0]} (widen to 32-bit if FP/int widen).",
            "h1": "Select high 16-bit halfword \\texttt{[31:16]} (widen to 32-bit if FP/int widen).",
            "b0": "Select byte 0 \\texttt{[7:0]} (zero/sign-extended to lane width).",
            "b1": "Select byte 1 \\texttt{[15:8]} (zero/sign-extended to lane width).",
            "b2": "Select byte 2 \\texttt{[23:16]} (zero/sign-extended to lane width).",
            "b3": "Select byte 3 \\texttt{[31:24]} (zero/sign-extended to lane width).",
            "w0": "Select 32-bit word 0 \\texttt{[31:0]} (extend to 64-bit).",
            "w1": "Select 32-bit word 1 \\texttt{[63:32]} (extend to 64-bit).",
            "d0": "Select full 64-bit doubleword \\texttt{[63:0]}.",
        }
        d = m_lane.get(vname, f"Select lane/width \\texttt{{{tex_esc(vname)}}}.")
        return (d, f"{mnem(ins)} {dst('r0')}, {src('r1')}{imod(mod_tok)}")

    if ename in ("swiz_m", "swiz_int_m", "swiz8_2_full_m", "lanes_int_m", "shift_bitwise_i16_lanes_m", "shift_bitwise_i32_lanes_m"):
        if vname.startswith("h"):
            d = f"16-bit halfword swizzle \\texttt{{{vname}}}: lane 0 $\\leftarrow$ h{vname[1]}, lane 1 $\\leftarrow$ h{vname[2]}."
        elif vname.startswith("b") and len(vname) == 5:
            d = f"4-byte swizzle \\texttt{{{vname}}}: output bytes [0..3] $\\leftarrow$ input bytes [{vname[1]}, {vname[2]}, {vname[3]}, {vname[4]}]."
        elif vname.startswith("b") and len(vname) == 3:
            d = f"2-byte swizzle \\texttt{{{vname}}}: low byte $\\leftarrow$ b{vname[1]}, high byte $\\leftarrow$ b{vname[2]}."
        elif vname.startswith("b") and len(vname) == 2:
            d = f"Single-byte extract \\texttt{{{vname}}}: select byte {vname[1]} zero/sign-extended to 32 bits."
        else:
            d = f"Lane swizzle/extract \\texttt{{{tex_esc(vname)}}}."
        return (d, f"{mnem(ins)} {dst('r0')}, {src('r1')}{imod('.' + vname)}, {src('r2')}")

    if "write_mask" in ename:
        ch = vname.replace("primary_", "").replace("secondary_", "").upper()
        prefix = "Primary " if "primary_" in vname else ("Secondary " if "secondary_" in vname else "")
        return (f"{prefix}channel write mask: write only {ch} component(s) to consecutive staging GPRs.",
                f"{mnem(ins)}{imod('.' + vname)} {dst('@r0:r3')}, {src('@r4:r5')}")

    if "width_m" in ename and any(x in ename for x in ("tex_", "var_tex_")):
        bits = "16-bit packed" if "16" in vname else "32-bit"
        pref = "Primary " if "primary_" in vname else ("Secondary " if "secondary_" in vname else "")
        return (f"{pref}texture return format \\texttt{{{tex_esc(vname)}}} ({bits} staging register elements).",
                f"{mnem(ins)}{imod('.' + vname)} {dst('@r0:r3')}, {src('@r4:r5')}")

    if "lod_mode" in ename:
        if vname in EXACT_ENUM_ENTRIES["tex_lod_mode_m"]:
            desc, _ = EXACT_ENUM_ENTRIES["tex_lod_mode_m"][vname]
            return (desc, f"{mnem(ins)}{imod(mod_tok)} {dst('@r0:r3')}, {src('@r4:r5')}")

    if "dimensionality" in ename:
        desc, _ = EXACT_ENUM_ENTRIES["tex_dimensionality_m"].get(vname, (f"Dimensionality {vname}.", ""))
        return (desc, f"{mnem(ins)}{imod('.' + vname)} {dst('@r0:r3')}, {src('@r4:r5')}")

    if "sample_position" in ename or "sample_and_update" in ename:
        sm = {
            "center": "Evaluate varying at pixel center $(0.5, 0.5)$.",
            "center_or_none": "Evaluate at pixel center (default / flat varying).",
            "centroid": "Evaluate varying at MSAA sample centroid (avoids extrapolation outside primitive).",
            "sample": "Evaluate varying at current MSAA sample position (\\texttt{interpolateAtSample}).",
            "explicit": "Evaluate varying at explicit $(x, y)$ offset register (\\texttt{interpolateAtOffset}).",
            "none": "No sample position override (flat or retrieve-cached).",
            "center_store": "Evaluate at pixel center and store barycentrics to varying cache.",
            "centroid_store": "Evaluate at centroid and store barycentrics to varying cache.",
            "sample_store": "Evaluate at per-sample position and store barycentrics.",
            "explicit_store": "Evaluate at explicit offset and store barycentrics.",
            "center_clobber": "Evaluate at pixel center, clobbering existing cached barycentrics.",
            "sample_clobber": "Evaluate at sample position, clobbering cached barycentrics.",
            "retrieve": "Reuse previously stored barycentrics from the varying cache without recomputing.",
        }
        return (sm.get(vname, f"Sample/update mode \\texttt{{{tex_esc(vname)}}}."),
                f"{mnem(ins)}{imod(mod_tok)} {dst('@r0:r3')}, {src('r4')}")

    if "update_mode" in ename:
        um = {
            "store": "Compute barycentric/varying coordinates and store in hardware varying cache.",
            "retrieve": "Reuse previously cached barycentric coordinates (saves ALU/interpolation power).",
            "retrieve_or_none": "Reuse cached barycentrics if available (default).",
            "clobber": "Compute fresh coordinates and overwrite hardware varying cache.",
            "clobber_or_none": "Overwrite varying cache (default for special varyings).",
            "none": "Do not read or modify the hardware varying cache.",
        }
        return (um.get(vname, f"Varying cache update mode \\texttt{{{tex_esc(vname)}}}."),
                f"{mnem(ins)}{imod(mod_tok)} {dst('@r0:r3')}, {src('r4')}")

    if ename == "clper_or_lane_select_m":
        return (f"Select lane pair \\texttt{{{tex_esc(vname)}}} for cross-lane bitwise OR reduction.",
                f"{mnem('CLPER_OR')}{imod('.' + vname)} {dst('r0')}, {src('r1')}, {src('r2')}")

    if ename == "clper_subgroup_size_m":
        n = vname.replace("subgroup", "")
        return (f"Partition warp into independent {n}-lane subgroups for shuffle/reduction.",
                f"{mnem(ins)}{imod('.' + vname)} {dst('r0')}, {src('r1')}, {src('r2')}")

    if ename == "tex_gather_component_m":
        c = vname[-1].upper()
        return (f"Gather 2x2 footprint of channel {c} (\\texttt{{textureGather(..., { 'RGBA'.index(c) })}}).",
                f"{mnem(ins)}{imod('.' + vname)} {dst('@r0:r3')}, {src('@r4:r5')}")

    if ename == "vecsize_varying_m":
        vn = {"none": "1 component (scalar).", "v2": "2 components (vec2).", "v3": "3 components (vec3).", "v4": "4 components (vec4)."}
        return (vn.get(vname, f"Vector size {vname}."), f"{mnem(ins)}{imod(mod_tok)} {dst('@r0:r3')}, {src('r4')}")

    if ename == "vertex_m":
        vm = {"none": "Default provoking vertex.", "vertex0": "Read flat varying from triangle vertex 0.", "vertex1": "Read flat varying from triangle vertex 1.", "vertex2": "Read flat varying from triangle vertex 2."}
        return (vm.get(vname, f"Vertex selector {vname}."), f"{mnem(ins)}{imod(mod_tok)} {dst('@r0:r3')}, {src('r4')}")

    # Clean general fallback for remaining 2-3 value flag enums (abs, neg, not, skip, array, compare, project, etc.)
    flag_descs = {
        "abs": "Clear sign bit ($|x|$) before operation (after widen, before neg).",
        "neg": "Flip sign bit ($-x$) before operation (after abs).",
        "not": "Bitwise invert operand or negate branch predicate ($\\sim x$ / $!p$).",
        "sat": "Clamp integer result to signed/unsigned type range instead of wrapping.",
        "sticky": "OR-reduce shifted-out bits into LSB (round-to-odd / sticky bit for FP64 emulation).",
        "conservative": "Conservative warp branch hint when lanes may diverge.",
        "mask": "Apply lane/bitmask variant of count-leading-zeros.",
        "low": "Select low half of exponential lookup table entry.",
        "ftz": "Flush subnormal float inputs/outputs to signed zero ($\\pm 0.0$).",
        "flush_inf": "Flush $\\pm\\infty$ to finite clamp/zero per \\texttt{FLUSH} mode.",
        "flush_nan": "Flush NaN inputs to $+0.0$.",
        "quiet_nan": "Convert signaling NaNs (sNaN) to quiet NaNs (qNaN) by setting bit 22.",
        "round_zero": "Round toward zero (RTZ) instead of default RTE.",
        "round_down": "Halving add floor: $(a + b) \\gg 1$.",
        "round_up": "Halving add ceil: $(a + b + 1) \\gg 1$.",
        "pow": "Evaluate \\texttt{pow(x, y)} special-case detection (negative base allowed for integer $y$).",
        "powr": "Evaluate OpenCL \\texttt{powr(x, y)} special-case detection (negative base $\\to$ NaN).",
        "c": "Return carry-out / borrow-out bit (0 or 1) instead of sum/difference.",
        "relative": "PC-relative jump target offset.",
        "absolute": "Absolute 64-bit GPU virtual address jump target.",
        "diverge": "Mark indirect jump as potentially divergent across warp lanes.",
        "cycle_counter": "Read 64-bit GPU shader-core cycle counter.",
        "system_timestamp": "Read 64-bit global system timestamp counter.",
        "flat32": "Flat (uninterpolated) 32-bit varying read from provoking vertex.",
        "flat16": "Flat (uninterpolated) 16-bit varying read from provoking vertex.",
        "src_flat32": "Source buffer holds flat 32-bit varying values.",
        "src_flat16": "Source buffer holds flat 16-bit varying values.",
        "src_f32": "Source buffer holds FP32 varying values.",
        "src_f16": "Source buffer holds FP16 varying values.",
        "last_read": "Signal final tile-buffer read so hardware may release tile memory.",
        "z_stencil": "Access combined depth/stencil tile buffer instead of color tile.",
        "oob_ignore": "Silently return 0 on OOB load or drop OOB store without faulting.",
        "preserve_null": "If base address is 0 (NULL), return 0 instead of adding offset.",
        "sext": "Sign-extend narrow loaded value (8/16-bit) to 32 bits.",
        "zext": "Zero-extend narrow loaded value (8/16-bit) to 32 bits.",
        "zext_or_none": "Zero-extend narrow load (or no extension for 32-bit+ loads).",
        "stencil": "Emit fragment stencil output to tile depth/stencil unit.",
        "z": "Emit fragment depth ($z$) output to tile depth/stencil unit.",
        "f0": "Select submatrix half 0 for matrix multiply.",
        "f1": "Select submatrix half 1 for matrix multiply.",
        "red": "Range-reduced logarithm table lookup.",
        "base2": "Base-2 logarithm ($\\log_2$) lookup table.",
        "natural": "Natural logarithm ($\\ln$) lookup table.",
        "high": "High-precision logarithm table mode.",
        "offset": "Apply phase offset in sin/cos table lookup.",
        "cos": "Compute cosine in \\texttt{FSINCOS}.",
        "sin": "Compute sine in \\texttt{FSINCOS}.",
        "scale": "Apply $\\pi$-scaling in \\texttt{FSINCOS\\_OFFSET}.",
        "array": "Enable texture array layer index coordinate (\\texttt{sampler2DArray}).",
        "compare": "Enable shadow depth comparison reference coordinate (\\texttt{sampler2DShadow}).",
        "float_coordinates": "Use normalized floating-point texture coordinates in $[0, 1]$.",
        "integer_coordinates": "Use unnormalized integer texel coordinates (\\texttt{texelFetch}).",
        "derivative": "Return screen-space coordinate derivatives instead of filtered texels.",
        "force_delta": "Force 2x2 quad derivative calculation even on inactive/helper lanes.",
        "footprint_2x2": "Evaluate 2x2 bilinear texel weight footprint.",
        "skip": "Skip texture fetch for helper/discarded fragment lanes.",
        "disable_lod_bias": "Ignore sampler/shader LOD bias during gradient calculation.",
        "disable_lod_clamp": "Ignore sampler min/max LOD clamp bounds.",
        "project": "Divide texture coordinates by homogeneous $q$ (\\texttt{textureProj}).",
        "texel_offset": "Add signed 4-bit texel offset $(dx, dy, dz)$ (\\texttt{textureOffset}).",
        "wide": "Use 32-bit wide texture/sampler descriptor indices.",
        "v4u8": "Interpret 32-bit operands as 4 packed unsigned 8-bit lanes.",
        "explicit_vertex": "Select flat varying vertex index explicitly via register.",
        "none": f"Default behavior (flag disabled for \\texttt{{{tex_esc(ename)}}}).",
    }
    desc = flag_descs.get(vname, f"Modifier option \\texttt{{{tex_esc(vname)}}} (encoding {venc}).")
    ex = f"{mnem(ins)}{imod(mod_tok)} {dst('r0')}, {src('r1')}"
    return (desc, ex)


# ============================================================================
# 2. Uncommon Instruction Fields Documentation (Chapter 3)
# ============================================================================

COMMON_FIELDS = {
    "instruction_class", "fau_page_index", "flow", "opcode1", "opcode2", "opcode3",
    "src0", "src1", "src2", "src3", "dest_register_index", "dest_width",
    "widen0", "widen1", "widen2", "swz0", "swz1", "swz2",
    "lanes0", "lanes1", "lanes2", "lane0", "lane1",
    "neg0", "neg1", "neg2", "abs0", "abs1", "abs2", "not", "not2",
    "round", "clamp", "saturate", "datatype",
}

UNCOMMON_FIELD_DOCS = {
    "staging_register_control_32": (
        "Controls the read/write staging register bank and count for async message instructions.",
        f"{mnem('LOAD.i128')}{imod('.slot0')} {dst('@r0:r3')}, {src('r4')}, {imm('#0')}",
        "OpLoad (vec4)", "vec4 v = ssbo.data[idx];"
    ),
    "sr_count": (
        "Number of consecutive 32-bit staging registers transferred by a load/store/texture message.",
        f"{mnem('LD_MULTI.i128')} {dst('@r0:r3')}, {src('r4')}, {imm('#0')}",
        "OpLoad (wide vector)", "uvec4 x = buf[i];"
    ),
    "sr_write_count": (
        "Number of consecutive 32-bit staging registers written back by a dual-staging message (e.g. ATOM\\_RETURN / TEX).",
        f"{mnem('TEX_SINGLE')}{imod('.tex2d.f32.rgba')} {dst('@r0:r3')}, {src('@r4:r5')}",
        "OpImageSampleImplicitLod", "vec4 c = texture(tex, uv);"
    ),
    "staging_register_index": (
        "Base GPR index of the contiguous staging register window (`@rN:rM`) for message payload/return.",
        f"{mnem('STORE.i64')}{imod('.slot7')} {src('@r0:r1')}, {src('r2')}, {imm('#8')}",
        "OpStore (uvec2)", "ssbo.pairs[i] = uvec2(a, b);"
    ),
    "staging_register_write_index": (
        "Separate destination staging register window base when read and write staging windows differ.",
        f"{mnem('ATOM_RETURN.i32')}{imod('.aadd')} {dst('@r0')}, {src('@r1')}, {src('r2')}, {imm('#0')}",
        "OpAtomicIAdd", "uint old = atomicAdd(ssbo.cnt, 1u);"
    ),
    "message_slot_index": (
        "Asynchronous scoreboard slot (`slot0`--`slot2`, `slot7`) tracking completion of this message.",
        f"{mnem('LOAD.i32')}{imod('.slot0')} {dst('@r0')}, {src('r2')}, {imm('#0')}",
        "OpLoad", "uint v = ssbo.arr[i];"
    ),
    "double_width_instruction": (
        "Indicates 64-bit / dual-slot wide instruction encoding variant.",
        f"{mnem('IADD.u64')} {dst('[r0:r1]')}, {src('[r2:r3]')}, {src('[r4:r5]')}",
        "OpIAdd (64-bit)", "uint64_t z = x + y;"
    ),
    "skip": (
        "Skips texture/memory fetch on helper/discarded fragment lanes while preserving 2x2 quad derivatives.",
        f"{mnem('TEX_SINGLE')}{imod('.tex2d.computed.skip')} {dst('@r0:r3')}, {src('@r4:r5')}",
        "OpImageSampleImplicitLod (after OpDemoteToHelperInvocation)", "if (a < 0.0) discard; vec4 c = texture(s, uv);"
    ),
    "wide_indices": (
        "Uses 32-bit wide texture/sampler table indices instead of compact 16-bit packed indices.",
        f"{mnem('TEX_SINGLE')}{imod('.tex2d.wide')} {dst('@r0:r3')}, {src('@r4:r5')}, {src('r6')}",
        "OpImageSampleImplicitLod (bindless)", "vec4 c = texture(textures[nonuniformEXT(i)], uv);"
    ),
    "sample_position": (
        "Selects MSAA interpolation location (`center`, `centroid`, `sample`, or `explicit` offset).",
        f"{mnem('LD_VAR_BUF.f32')}{imod('.centroid.v2')} {dst('@r0:r1')}, {src('r2')}",
        "OpInterpolateAtCentroid", "vec2 p = interpolateAtCentroid(v_uv);"
    ),
    "encoded_sample": (
        "Compressed encoding of MSAA sample interpolation position (`center_or_none`, `centroid`, `sample`, `explicit`).",
        f"{mnem('LD_VAR.f32')}{imod('.sample.v4')} {dst('@r0:r3')}, {src('r4')}",
        "OpInterpolateAtSample", "vec4 v = interpolateAtSample(v_col, 0);"
    ),
    "encoded_sample_and_update": (
        "Jointly encodes MSAA sample position and hardware varying-cache update mode (`center_store`, `retrieve`, etc.).",
        f"{mnem('VAR_TEX_BUF_SINGLE')}{imod('.center_store.tex2d')} {dst('@r0:r3')}, {src('r4')}",
        "OpImageSampleImplicitLod (fused varying)", "vec4 c = texture(s, v_uv);"
    ),
    "update": (
        "Controls hardware varying-cache lifecycle (`store`, `retrieve`, `clobber`, `none`) to avoid redundant barycentric math.",
        f"{mnem('LD_VAR_BUF.f32')}{imod('.retrieve.v4')} {dst('@r0:r3')}, {src('r4')}",
        "OpLoad (Input varying)", "vec4 a = v_pos; vec4 b = v_norm;"
    ),
    "encoded_update": (
        "Compressed bitfield encoding for varying-cache update mode (`store`, `retrieve`, `clobber`).",
        f"{mnem('LD_VAR_SPECIAL')}{imod('.bary.store')} {dst('@r0:r1')}, {src('r2')}",
        "OpLoad (BuiltIn BaryCoordKHR)", "vec3 b = gl_BaryCoordEXT;"
    ),
    "combine": (
        "Combines 16-bit halfword predicates (`h0`, `h1`, `and`, `lowbits`) before evaluating a branch.",
        f"{mnem('BRANCH')}{imod('.and')} {src('r0')}, {imm('#+8')}",
        "OpBranchConditional (packed predicate)", "if (condA && condB) { ... }"
    ),
    "array_enable": (
        "Enables array-layer coordinate in texture staging registers (`sampler2DArray` / `samplerCubeArray`).",
        f"{mnem('TEX_SINGLE')}{imod('.tex2d.array')} {dst('@r0:r3')}, {src('@r4:r6')}",
        "OpImageSampleImplicitLod (Arrayed)", "vec4 c = texture(texArr, vec3(uv, layer));"
    ),
    "compare_enable": (
        "Enables depth-reference shadow comparison (`Dref` in staging registers).",
        f"{mnem('TEX_SINGLE')}{imod('.tex2d.compare')} {dst('@r0')}, {src('@r4:r6')}",
        "OpImageSampleDrefImplicitLod", "float s = texture(shadowMap, vec3(uv, refZ));"
    ),
    "projection_enable": (
        "Divides texture coordinates by homogeneous $q$ prior to filtering.",
        f"{mnem('TEX_SINGLE')}{imod('.tex2d.project')} {dst('@r0:r3')}, {src('@r4:r6')}",
        "OpImageSampleProjImplicitLod", "vec4 c = textureProj(tex, uvw);"
    ),
    "texel_offset": (
        "Applies a signed 4-bit per-axis texel offset $(dx, dy, dz)$ packed in staging registers.",
        f"{mnem('TEX_SINGLE')}{imod('.tex2d.texel_offset')} {dst('@r0:r3')}, {src('@r4:r6')}",
        "OpImageSampleImplicitLod (ConstOffset)", "vec4 c = textureOffset(tex, uv, ivec2(1, -1));"
    ),
    "lod_mode": (
        "Selects mipmap LOD calculation mode (`none`, `computed`, `explicit`, `computed_bias`, `grdesc`).",
        f"{mnem('TEX_SINGLE')}{imod('.tex2d.explicit')} {dst('@r0:r3')}, {src('@r4:r6')}",
        "OpImageSampleExplicitLod", "vec4 c = textureLod(tex, uv, 2.0);"
    ),
    "lod_bias_disable": (
        "Disables sampler/shader LOD bias when computing explicit gradients (`TEX_GRADIENT`).",
        f"{mnem('TEX_GRADIENT')}{imod('.tex2d.disable_lod_bias')} {dst('@r0:r1')}, {src('@r4:r5')}",
        "OpImageQueryLod", "vec2 lod = textureQueryLod(tex, uv);"
    ),
    "lod_clamp_disable": (
        "Disables sampler min/max LOD clamp bounds when querying raw mip level.",
        f"{mnem('TEX_GRADIENT')}{imod('.tex2d.disable_lod_clamp')} {dst('@r0:r1')}, {src('@r4:r5')}",
        "OpImageQueryLod", "vec2 lod = textureQueryLod(tex, uv);"
    ),
    "derivative_enable": (
        "Returns raw screen-space derivatives ($dP/dx, dP/dy$) instead of scalar LOD.",
        f"{mnem('TEX_GRADIENT')}{imod('.tex2d.derivative')} {dst('@r0:r3')}, {src('@r4:r5')}",
        "OpDPdx / OpDPdy", "vec2 dx = dFdx(uv);"
    ),
    "force_delta_enable": (
        "Forces 2x2 quad derivative evaluation even across helper/divergent lanes.",
        f"{mnem('TEX_GRADIENT')}{imod('.tex2d.force_delta')} {dst('@r0:r1')}, {src('@r4:r5')}",
        "OpImageQueryLod", "vec2 lod = textureQueryLod(tex, uv);"
    ),
    "oob_mode": (
        "Out-of-bounds checked buffer behavior: `none` (standard) or `oob_ignore` (return 0 / drop write).",
        f"{mnem('LD_CHECKED.i32')}{imod('.oob_ignore')} {dst('@r0')}, {src('r2')}, {imm('#0')}",
        "OpLoad (RobustBufferAccess)", "uint v = ssbo.arr[idx]; // robust"
    ),
    "access": (
        "Memory cache streaming/coherence policy (`none`, `istream`, `estream`, `force`).",
        f"{mnem('STORE.i32')}{imod('.force')} {src('@r0')}, {src('r2')}, {imm('#0')}",
        "OpStore (MakePointerVisible / Coherent)", "coherent buffer B { uint x; }; x = 1u;"
    ),
    "source_format": (
        "Declares the storage precision of the varying buffer (`src_f32`, `src_f16`, `src_flat32`, `src_flat16`).",
        f"{mnem('LD_VAR_BUF.f32')}{imod('.src_f16.v2')} {dst('@r0:r1')}, {src('r2')}",
        "OpLoad (RelaxedPrecision Input)", "mediump in vec2 v_uv;"
    ),
    "register_format": (
        "Target GPR data conversion format (`auto32`, `f32`, `f16`, `s32`, `s16`, `u32`, `u16`).",
        f"{mnem('LD_TILE')}{imod('.f16.v4')} {dst('@r0:r1')}, {src('r2')}",
        "OpImageRead (SubpassData)", "f16vec4 c = subpassLoad(inputAtt);"
    ),
    "result_type": (
        "Encodes boolean output representation (`i1` = 0/1, `f1` = 0.0/1.0, `m1` = 0/-1, `c` = carry).",
        f"{mnem('FCMP.f32')}{imod('.lt.m1')} {dst('r0')}, {src('r1')}, {src('r2')}",
        "OpFOrdLessThan + OpSelect", "int mask = (a < b) ? -1 : 0;"
    ),
    "sticky": (
        "OR-reduces shifted-out rounding bits into LSB (round-to-odd for double-precision emulation).",
        f"{mnem('FADD.f32')}{imod('.sticky')} {dst('r0')}, {src('r1')}, {src('r2')}",
        "OpFAdd (emulated FP64)", "double c = a + b; // lowered"
    ),
    "special": (
        "Selects `FMA_RSCALE`/`FREXP` special-case override (`n`, `left`, `n_add`, `sqrt`, `log`).",
        f"{mnem('FREXPM.f32')}{imod('.log')} {dst('r0')}, {src('r1')}",
        "GLSL.std.450 Log2 (range reduction)", "float y = log2(x);"
    ),
    "inf": (
        "Handles extreme `LDEXP` exponents ($|e| \\ge 16384$): `.inf` maps $\\pm\\infty \\to \\pm 1.0$; `.inf0` also maps $\\pm 0 \\to \\pm 1.0$.",
        f"{mnem('LDEXP.f32')}{imod('.inf0')} {dst('r0')}, {src('r1')}, {src('r2')}",
        "GLSL.std.450 Ldexp", "float y = ldexp(x, e);"
    ),
    "sem": (
        "IEEE-754 NaN handling on `FMIN`/`FMAX`: `nan_suppress` (minNum/maxNum) vs `nan_propagate` (payload-preserving).",
        f"{mnem('FMAX.f32')}{imod('.nan_propagate')} {dst('r0')}, {src('r1')}, {src('r2')}",
        "GLSL.std.450 NMax / FMax", "float m = max(a, b);"
    ),
    "preserve_null": (
        "Preserves NULL (0) pointer on segmented address add/sub instead of adding segment base.",
        f"{mnem('SEG_ADD')}{imod('.preserve_null')} {dst('[r0:r1]')}, {src('r2')}",
        "OpPtrAccessChain (PhysicalStorageBuffer)", "if (ptr != null) val = ptr.x;"
    ),
    "inactive_result": (
        "Identity value substituted by `CLPER` when the source lane in the subgroup is inactive.",
        f"{mnem('CLPER')}{imod('.subgroup16.xor.inf')} {dst('r0')}, {src('r1')}, {src('r2')}",
        "OpGroupNonUniformFMin", "float m = subgroupMin(x);"
    ),
    "lane_op": (
        "Subgroup lane-index transformation in `CLPER` (`none`, `xor`, `shift`, `rotate`, `low`, `prefix`).",
        f"{mnem('CLPER')}{imod('.subgroup16.xor.zero')} {dst('r0')}, {src('r1')}, {src('r2')}",
        "OpGroupNonUniformShuffleXor", "uint v = subgroupShuffleXor(x, 1u);"
    ),
    "subgroup": (
        "Subgroup partition width (`subgroup2`, `subgroup4`, `subgroup8`, `subgroup16`).",
        f"{mnem('CLPER')}{imod('.subgroup8.rotate.zero')} {dst('r0')}, {src('r1')}, {src('r2')}",
        "OpGroupNonUniformRotateKHR", "uint v = subgroupRotate(x, 1);"
    ),
    "lane_select": (
        "Selects which pair of warp lanes (`lanes04`..`lanesBF`) is OR-combined by `CLPER_OR`.",
        f"{mnem('CLPER_OR')}{imod('.subgroup16.lanes04')} {dst('r0')}, {src('r1')}, {src('r2')}",
        "OpGroupNonUniformBitwiseOr", "uint v = subgroupOr(x);"
    ),
    "return_carry": (
        "When `.c` is set on `IADDC`/`ISUBB`, returns the 1-bit carry/borrow instead of the 32-bit sum/diff.",
        f"{mnem('IADDC.u32')}{imod('.c')} {dst('r0')}, {src('r1')}, {src('r2')}",
        "OpIAddCarry", "uint carry; uint s = uaddCarry(a, b, carry);"
    ),
    "neg_result": (
        "Negates the final output of `FREXPE` (used for reciprocal-sqrt / inverse exponent scaling).",
        f"{mnem('FREXPE.f32')}{imod('.sqrt.neg')} {dst('r0')}, {src('r1')}",
        "GLSL.std.450 InverseSqrt", "float y = inversesqrt(x);"
    ),
    "flush_to_zero_mode": (
        "Flushes subnormal values to signed zero in `FLUSH`.",
        f"{mnem('FLUSH.f32')}{imod('.ftz')} {dst('r0')}, {src('r1')}",
        "ExecutionMode DenormFlushToZero", "// FP32 FTZ canonicalization"
    ),
    "inf_mode": (
        "Controls whether $\\pm\\infty$ is flushed by `FLUSH`.",
        f"{mnem('FLUSH.f32')}{imod('.flush_inf')} {dst('r0')}, {src('r1')}",
        "None (HW canonicalization)", "// Clamp/flush Inf in special math"
    ),
    "nan_mode": (
        "Controls NaN canonicalization in `FLUSH` (`none`, `flush_nan`, `quiet_nan`).",
        f"{mnem('FLUSH.f32')}{imod('.quiet_nan')} {dst('r0')}, {src('r1')}",
        "None (HW sNaN quietening)", "// Quieten signaling NaN"
    ),
    "encoded_extend": (
        "Controls sign-extension (`sext`) vs zero-extension (`zext_or_none`) for narrow 8/16-bit loads.",
        f"{mnem('LOAD.i16')}{imod('.sext')} {dst('@r0')}, {src('r2')}, {imm('#0')}",
        "OpLoad + OpSConvert", "int x = int(ssbo.i16_arr[i]);"
    ),
    "extend": (
        "Virtual extension mode (`sext`, `zext`, `none`) decoded from `encoded_extend` and `memory_width`.",
        f"{mnem('LOAD.i8')}{imod('.sext')} {dst('@r0')}, {src('r2')}, {imm('#0')}",
        "OpLoad + OpSConvert", "int x = int(ssbo.i8_arr[i]);"
    ),
    "write_mask": (
        "Channel enable bitmask (`r`..`rgba`) controlling which texture components are written to staging GPRs.",
        f"{mnem('TEX_SINGLE')}{imod('.tex2d.f32.rg')} {dst('@r0:r1')}, {src('@r4:r5')}",
        "OpImageSampleImplicitLod + OpVectorShuffle", "vec2 rg = texture(tex, uv).rg;"
    ),
    "primary_write_mask": (
        "Channel mask (`primary_r`..`primary_rgba`) for the first texture fetch in `TEX_DUAL`.",
        f"{mnem('TEX_DUAL')}{imod('.primary_rgb.secondary_r')} {dst('@r0:r3')}, {src('@r4:r5')}",
        "2x OpImageSampleImplicitLod (fused)", "vec3 a = texture(t0, uv).rgb; float b = texture(t1, uv).r;"
    ),
    "secondary_write_mask": (
        "Channel mask (`secondary_r`..`secondary_rgba`) for the second texture fetch in `TEX_DUAL`.",
        f"{mnem('TEX_DUAL')}{imod('.primary_rg.secondary_rg')} {dst('@r0:r3')}, {src('@r4:r5')}",
        "2x OpImageSampleImplicitLod (fused)", "vec2 a = texture(t0, uv).rg; vec2 b = texture(t1, uv).rg;"
    ),
    "gather_component": (
        "Selects the R, G, B, or A channel gathered across the 2x2 texel footprint.",
        f"{mnem('TEX_GATHER')}{imod('.tex2d.gather4_g')} {dst('@r0:r3')}, {src('@r4:r5')}",
        "OpImageGather", "vec4 g = textureGather(tex, uv, 1);"
    ),
    "coordinate_mode": (
        "Selects normalized float coordinates vs unnormalized integer texel indices.",
        f"{mnem('TEX_FETCH')}{imod('.tex2d.integer_coordinates')} {dst('@r0:r3')}, {src('@r4:r5')}",
        "OpImageFetch", "vec4 c = texelFetch(tex, ivec2(xy), 0);"
    ),
    "z_last_read": (
        "Marks the final tile-buffer read in a fragment shader so hardware can reclaim tile storage.",
        f"{mnem('LD_TILE')}{imod('.f32.v4.last_read')} {dst('@r0:r3')}, {src('r4')}",
        "OpImageRead (SubpassData)", "vec4 c = subpassLoad(att);"
    ),
    "z_stencil": (
        "Targets depth/stencil tile buffer instead of color attachment in `LD_TILE`.",
        f"{mnem('LD_TILE')}{imod('.z_stencil')} {dst('@r0')}, {src('r4')}",
        "OpImageRead (Depth/Stencil subpass)", "float d = subpassLoad(depthAtt).x;"
    ),
    "z": (
        "Emits shader-computed fragment depth (`gl_FragDepth`) in `ZS_EMIT`.",
        f"{mnem('ZS_EMIT')}{imod('.z')} {src('r0')}, {src('r1')}",
        "OpStore (BuiltIn FragDepth)", "gl_FragDepth = d;"
    ),
    "stencil": (
        "Emits shader-computed fragment stencil ref (`gl_FragStencilRefARB`) in `ZS_EMIT`.",
        f"{mnem('ZS_EMIT')}{imod('.stencil')} {src('r0')}, {src('r1')}",
        "OpStore (BuiltIn FragStencilRefEXT)", "gl_FragStencilRefARB = s;"
    ),
    "sub0": (
        "Selects submatrix half (`f0`, `f1`, `none`) of source matrix A in `MMUL`.",
        f"{mnem('MMUL.v2f16')}{imod('.f0')} {dst('r0')}, {src('r1')}, {src('r2')}, {src('r3')}",
        "OpCooperativeMatrixMulAddKHR", "// 4x4/8x8 cooperative matrix multiply"
    ),
    "sub1": (
        "Selects submatrix half (`f0`, `f1`, `none`) of source matrix B in `MMUL`.",
        f"{mnem('MMUL.v2f16')}{imod('.f1')} {dst('r0')}, {src('r1')}, {src('r2')}, {src('r3')}",
        "OpCooperativeMatrixMulAddKHR", "// 4x4/8x8 cooperative matrix multiply"
    ),
    "atom_opc": (
        "Selects the atomic read-modify-write ALU operation (`aadd`, `asmin`, `aumax`, `aand`, `aor`, `axor`, `axchg`).",
        f"{mnem('ATOM.i32')}{imod('.aumax')} {src('@r0')}, {src('r2')}, {imm('#0')}",
        "OpAtomicUMax", "atomicMax(ssbo.val, x);"
    ),
    "atom1_opc": (
        "Selects unary atomic operation (`ainc`, `adec`, `aumax1`, `asmax1`, `aor1`) with implicit operand 1.",
        f"{mnem('ATOM1.i32')}{imod('.ainc')} {src('r2')}, {imm('#0')}",
        "OpAtomicIIncrement", "atomicAdd(ssbo.cnt, 1u);"
    ),
    "address_mode": (
        "Selects PC-relative vs absolute 64-bit jump address calculation.",
        f"{mnem('JUMP')}{imod('.absolute')} {src('r0')}, {src('[r2:r3]')}",
        "OpSwitch / indirect function call", "// Jump table dispatch"
    ),
    "stack_mode": (
        "Controls hardware call-stack push/pop/replace on `JUMP_EX` (`call`, `return`, `replace`, `none`).",
        f"{mnem('JUMP_EX')}{imod('.call')} {src('r0')}, {imm('#+32')}",
        "OpFunctionCall", "float y = helperFunc(x);"
    ),
    "diverge": (
        "Signals potentially divergent indirect jump across warp lanes in `JUMP_EX`.",
        f"{mnem('JUMP_EX')}{imod('.diverge')} {src('r0')}, {src('[r2:r3]')}",
        "OpFunctionCall (divergent)", "if (gl_LocalInvocationID.x < k) f();"
    ),
    "conservative": (
        "Conservative warp-execution hint on `BRANCH`/`JUMP` when reconvergence is non-trivial.",
        f"{mnem('BRANCH')}{imod('.conservative')} {src('r0')}, {imm('#+12')}",
        "OpBranchConditional", "while (cond) { ... }"
    ),
    "source": (
        "Selects global clock source in `LD_GCLK` (`cycle_counter` vs `system_timestamp`).",
        f"{mnem('LD_GCLK')}{imod('.system_timestamp')} {dst('[r0:r1]')}",
        "OpReadClockKHR", "uint64_t t = clockARB();"
    ),
    "muxf": (
        "Selects the `MUX` comparison predicate (`neg`, `int_zero`, `fp_zero`, `bit`).",
        f"{mnem('MUX.i32')}{imod('.bit')} {dst('r0')}, {src('r1')}, {src('r2')}, {src('r3')}",
        "OpBitwiseOr / OpBitwiseAnd", "uint z = (a & mask) | (b & ~mask);"
    ),
    "bitwise_operation": (
        "Selects the bitwise ALU op (`and`, `or`, `xor`, `lshift_and`, etc.) fused with shifts.",
        f"{mnem('LSHIFT_OR.i32')} {dst('r0')}, {src('r1')}, {src('r2')}, {src('r3')}",
        "OpShiftLeftLogical + OpBitwiseOr", "uint y = (a << s) | b;"
    ),
    "shift_operation": (
        "Selects shift direction/type (`lshift`, `rshift`, `arshift`) in fused shift-bitwise instructions.",
        f"{mnem('RSHIFT_AND.i32')} {dst('r0')}, {src('r1')}, {src('r2')}, {src('r3')}",
        "OpShiftRightLogical + OpBitwiseAnd", "uint y = (a >> s) & m;"
    ),
    "not_result": (
        "Bitwise-inverts the final output of a bitwise/shift-bitwise instruction (~result).",
        f"{mnem('AND.i32')}{imod('.not_result')} {dst('r0')}, {src('r1')}, {src('r2')}",
        "OpBitwiseAnd + OpNot", "uint y = ~(a & b);"
    ),
    "offset": (
        "Signed byte offset added to base address register in memory load/store/atomic instructions.",
        f"{mnem('LOAD.i32')} {dst('@r0')}, {src('[r2:r3]')}, {imm('#16')}",
        "OpAccessChain + OpLoad", "uint y = ssbo.arr[4];"
    ),
    "vecsize": (
        "Component count (`none`=1, `v2`, `v3`, `v4`) for attribute, varying, or blend transfers.",
        f"{mnem('LD_ATTR')}{imod('.f32.v3')} {dst('@r0:r2')}, {src('r4')}",
        "OpLoad (vec3 Input)", "in vec3 a_normal;"
    ),
    "dimensionality": (
        "Texture dimensionality (`tex1d`, `tex2d`, `tex3d`, `cube`).",
        f"{mnem('TEX_SINGLE')}{imod('.tex3d.f32.rgba')} {dst('@r0:r3')}, {src('@r4:r6')}",
        "OpImageSampleImplicitLod (Dim=3D)", "vec4 c = texture(volTex, uvw);"
    ),
    "memory_width": (
        "Memory transfer width (`i8`, `i16`, `i24`, `i32`, `i48`, `i64`, `i96`, `i128`).",
        f"{mnem('LOAD.i64')} {dst('@r0:r1')}, {src('[r2:r3]')}, {imm('#0')}",
        "OpLoad (uvec2)", "uvec2 v = ssbo.pairs[i];"
    ),
    "unsigned0": ("Treats `src0` as unsigned integer in mixed-sign multiply/dot-product.", f"{mnem('IDPADD')}{imod('.v4u8')} {dst('r0')}, {src('r1')}, {src('r2')}, {src('r3')}", "OpUDotKHR", "uint d = dot4x8EXT(a, b);"),
    "unsigned1": ("Treats `src1` as unsigned integer in mixed-sign multiply/dot-product.", f"{mnem('IDPADD')}{imod('.v4u8')} {dst('r0')}, {src('r1')}, {src('r2')}, {src('r3')}", "OpSUDotKHR", "int d = dot4x8EXT(a, b);"),
    "mode": ("Selects logarithm table base (`red`, `base2`, `natural`) in `FLOG_TABLE`.", f"{mnem('FLOG_TABLE')}{imod('.base2')} {dst('r0')}, {src('r1')}", "GLSL.std.450 Log2", "float y = log2(x);"),
    "precision": ("Selects high vs low precision table lookup in `FLOG_TABLE`.", f"{mnem('FLOG_TABLE')}{imod('.high')} {dst('r0')}, {src('r1')}", "GLSL.std.450 Log", "float y = log(x);"),
    "func": ("Selects `pow` vs `powr` special-case detection in `FPOW_SC_DET`.", f"{mnem('FPOW_SC_DET')}{imod('.pow')} {dst('r0')}, {src('r1')}, {src('r2')}", "GLSL.std.450 Pow", "float y = pow(x, p);"),
    "operation": ("Selects `sin` vs `cos` output in `FSINCOS`.", f"{mnem('FSINCOS')}{imod('.sin')} {dst('r0')}, {src('r1')}", "GLSL.std.450 Sin", "float y = sin(x);"),
    "scale": ("Enables $\\pi$-scaling in `FSINCOS_OFFSET`.", f"{mnem('FSINCOS_OFFSET')}{imod('.scale')} {dst('r0')}, {src('r1')}", "GLSL.std.450 Sin", "float y = sin(x);"),
    "adj": ("Selects low/adjusted table entry in `FEXP_TABLE`.", f"{mnem('FEXP_TABLE')}{imod('.low')} {dst('r0')}, {src('r1')}", "GLSL.std.450 Exp2", "float y = exp2(x);"),
    "footprint": ("Selects 2x2 bilinear footprint in `TEX_WEIGHTS`.", f"{mnem('TEX_WEIGHTS')}{imod('.footprint_2x2')} {dst('@r0:r3')}, {src('@r4:r5')}", "OpImageSampleFootprintNV", "// Bilinear weight query"),
    "vertex": ("Selects provoking triangle vertex (`vertex0`..`vertex2`) for flat varying read.", f"{mnem('LD_VAR_FLAT_IMM.f32')}{imod('.vertex0')} {dst('@r0:r3')}, {imm('#0')}", "OpLoad (Flat Input)", "flat in vec4 v_flat;"),
    "vertex_type": ("Enables explicit register-supplied vertex index on `LD_VAR_BUF_FLAT`.", f"{mnem('LD_VAR_BUF_FLAT.f32')}{imod('.explicit_vertex')} {dst('@r0:r3')}, {src('r4')}", "OpLoad (PerVertexKHR)", "pervertexEXT in vec4 v[3]; vec4 x = v[i];"),
    "varying_name": ("Selects fixed-function special varying (`point`, `bary`, `frag_w`, `frag_z`).", f"{mnem('LD_VAR_SPECIAL')}{imod('.frag_z')} {dst('@r0')}, {src('r2')}", "OpLoad (BuiltIn FragCoord)", "float z = gl_FragCoord.z;"),
    "descriptor_type": ("Selects descriptor table binding class (attribute, buffer, texture, varying).", f"{mnem('LD_BUFFER.i32')} {dst('@r0')}, {src('r2')}, {imm('#0')}", "OpLoad (StorageBuffer)", "uint v = ssbo.arr[i];"),
    "varying_table_index": ("Descriptor table index selecting the varying buffer table.", f"{mnem('LD_VAR_BUF_IMM.f32')} {dst('@r0:r3')}, {imm('#0')}, {imm('#0')}", "OpLoad (Input)", "in vec4 v_col;"),
    "attribute_table_index": ("Descriptor table index selecting the vertex attribute buffer table.", f"{mnem('LD_ATTR_IMM')}{imod('.f32.v4')} {dst('@r0:r3')}, {imm('#0')}, {imm('#0')}", "OpLoad (Vertex Input)", "layout(location=0) in vec4 a_pos;"),
    "texture_table_index": ("Descriptor table index selecting the texture/sampler resource table.", f"{mnem('TEX_SINGLE')}{imod('.tex2d')} {dst('@r0:r3')}, {src('@r4:r5')}, {imm('#0')}", "OpSampledImage", "vec4 c = texture(tex, uv);"),
    "buffer_table_index": ("Descriptor table index selecting the storage/uniform buffer table.", f"{mnem('LD_BUFFER.i32')} {dst('@r0')}, {src('r2')}, {imm('#0')}", "OpLoad (StorageBuffer)", "uint v = ssbo.data[i];"),
    "cmpf": ("Comparison predicate (`eq`, `ne`, `lt`, `le`, `gt`, `ge`, `gtlt`, `total`).", f"{mnem('CSEL.f32')}{imod('.lt')} {dst('r0')}, {src('r1')}, {src('r2')}, {src('r3')}, {src('r4')}", "OpFOrdLessThan + OpSelect", "float y = (a < b) ? c : d;"),
    "mask": ("Enables bitmask-mode leading-zero count in `CLZ`.", f"{mnem('CLZ.u32')}{imod('.mask')} {dst('r0')}, {src('r1')}", "GLSL.std.450 FindUMsb", "int msb = findMSB(x);"),
    "shift": ("Inline shift count (0--31) in `SHADD` / shift-add instructions.", f"{mnem('SHADD.u64')} {dst('[r0:r1]')}, {src('[r2:r3]')}, {src('r4')}, {imm('#2')}", "OpPtrAccessChain", "uint x = ssbo.arr[i]; // base + (i<<2)"),
    "imm1w": ("32-bit inline immediate operand encoded directly in the instruction word (bits 8:39).", f"{mnem('IADD_IMM.i32')} {dst('r0')}, {src('r1')}, {imm('#0x10')}", "OpIAdd (constant)", "int y = x + 16;"),

    # Core & previously-omitted instruction fields (now documented for EVERY field!)
    "instruction_class": (
        "4-bit major instruction class selector (`bits 48:51`) partitioning the 64-bit Valhall opcode space (e.g., `FMA`, `CVT`, `SFU`, `LS`, `TEX`, `BRANCH`).",
        f"{mnem('FADD.f32')} {dst('r0')}, {src('r1')}, {src('r2')}",
        "N/A (encoding class)", "// Hardware execution pipe / decode class"
    ),
    "fau_page_index": (
        "Selects the FAU (Fast Access Uniform) page (`0..3`) when a source operand references hardware small constants, thread IDs, or lane constants.",
        f"{mnem('FADD.f32')} {dst('r0')}, {src('r1')}, {src('k1f')}",
        "OpConstant / BuiltIn", "float y = x + 1.0;"
    ),
    "flow": (
        "4-bit embedded flow-control & scoreboard wait modifier (`bits 59:62`: `none`, `wait0..wait0126`, `wait`, `reconverge`, `discard`, `end`).",
        f"{mnem('FADD.f32')}{imod('.wait0.reconverge')} {dst('r0')}, {src('r1')}, {src('r2')}",
        "OpControlBarrier / OpReturn", "// Scoreboard wait + warp reconvergence"
    ),
    "opcode1": (
        "Primary sub-opcode within the instruction class uniquely identifying the operation family.",
        f"{mnem('IADD.s32')} {dst('r0')}, {src('r1')}, {src('r2')}",
        "OpIAdd (operation selector)", "// Primary opcode bits"
    ),
    "opcode2": (
        "Secondary sub-opcode further specializing instructions sharing `instruction_class` and `opcode1` (e.g., conversions, SFU tables, moves).",
        f"{mnem('F32_TO_S32')}{imod('.round_zero')} {dst('r0')}, {src('r1')}",
        "OpConvertFToS", "int y = int(x);"
    ),
    "opcode3": (
        "Tertiary sub-opcode disambiguating specialized multi-op families (`FPOW_SC_*`, SFU lookup tables, `ICMP_MULTI`).",
        f"{mnem('FPOW_SC_DET.f32')}{imod('.pow')} {dst('r0')}, {src('r1')}, {src('r2')}",
        "GLSL.std.450 Pow", "float z = pow(x, y);"
    ),
    "datatype": (
        "Selects the lane data type (`f32`, `f16`, `v2f16`, `s32`, `u32`, `v2s16`, `v2u16`, `v4s8`, `v4u8`, `s64`, `u64`, etc.).",
        f"{mnem('IADD.v2s16')} {dst('r0')}, {src('r1')}, {src('r2')}",
        "OpTypeInt / OpTypeFloat", "i16vec2 c = a + b;"
    ),
    "dest_register_index": (
        "6-bit destination GPR index (`r0..r63`) receiving the Stage C writeback result (even-aligned pair `[rN:rN+1]` for 64-bit ops).",
        f"{mnem('FMA.f32')} {dst('r0')}, {src('r1')}, {src('r2')}, {src('r3')}",
        "SSA Result ID", "float r0 = fma(r1, r2, r3);"
    ),
    "dest_width": (
        "Stage C destination lane write-mask (`none` = full 32-bit, `h0` = low 16-bit, `h1` = high 16-bit without bit-shifting).",
        f"{mnem('FADD.v2f16')} {dst('r0')}{imod('.h1')}, {src('r1')}, {src('r2')}",
        "OpVectorInsertDynamic / WriteMask", "dst.y = a.y + b.y;"
    ),
    "src0": (
        "Primary source operand (`r0..r63`, `u0..u63` uniform, `k*` FAU constant, or `t0..t2` temporary; `^` marks discard/last-use).",
        f"{mnem('FADD.f32')} {dst('r0')}, {src('r1')}, {src('r2')}",
        "OpFAdd (Operand 0)", "float c = a + b;"
    ),
    "src1": (
        "Secondary source operand (register, uniform, or FAU constant; in shifts, supplies the shift count byte).",
        f"{mnem('LSHIFT.i32')} {dst('r0')}, {src('r1')}, {src('r2')}{imod('.b0')}",
        "OpShiftLeftLogical (Shift)", "uint y = a << s;"
    ),
    "src2": (
        "Third source operand (FMA addend, `MUX`/`CSEL` selector, `IADDC` carry-in, or bitwise mask in fused shift-logic).",
        f"{mnem('FMA.f32')} {dst('r0')}, {src('r1')}, {src('r2')}, {src('r3')}",
        "GLSL.std.450 Fma (Addend)", "float d = fma(a, b, c);"
    ),
    "src3": (
        "Fourth source operand (false-branch value in `CSEL`, exponent scale in `FMA_RSCALE`, or fourth matrix/select input).",
        f"{mnem('CSEL.f32')}{imod('.lt')} {dst('r0')}, {src('r1')}, {src('r2')}, {src('r3')}, {src('r4')}",
        "OpSelect (Object 2)", "float r = (a < b) ? x : y;"
    ),
    "widen0": (
        "Stage A widening extractor on `src0` (promotes FP16 `h0`/`h1` to FP32, or sign/zero-extends `b0..b3`/`h0..h1`/`w0..w1` to 32/64-bit before `abs`/`neg`).",
        f"{mnem('IADD.s32')} {dst('r0')}, {src('r1')}{imod('.h0')}, {src('r2')}",
        "OpSConvert / OpFConvert", "int c = int(int16_t(a)) + b;"
    ),
    "widen1": (
        "Stage A widening extractor on `src1` (promotes narrow byte/half/word lane before Stage B).",
        f"{mnem('IMUL.u64')} {dst('[r0:r1]')}, {src('[r2:r3]')}{imod('.w0')}, {src('[r4:r5]')}{imod('.w0')}",
        "OpUConvert / OpUMulExtended", "uint64_t p = uint64_t(a) * uint64_t(b);"
    ),
    "widen2": (
        "Stage A widening extractor on `src2` (e.g., promoting FP16 addend `h0`/`h1` to FP32 in `FMA.f32`).",
        f"{mnem('FMA.f32')} {dst('r0')}, {src('r1')}, {src('r2')}, {src('r3')}{imod('.h0')}",
        "OpFConvert + GLSL.std.450 Fma", "float d = fma(a, b, float(c16));"
    ),
    "swz0": (
        "Stage A lane swizzle on `src0` (`h00`/`h01`/`h10`/`h11` for 16-bit vectors, or `bXY` byte-to-halfword widening).",
        f"{mnem('FADD.v2f16')} {dst('r0')}, {src('r1')}{imod('.h10')}, {src('r2')}",
        "OpVectorShuffle", "f16vec2 c = a.yx + b;"
    ),
    "swz1": (
        "Stage A lane swizzle on `src1` (`h00`/`h01`/`h10`/`h11` or `bXY` byte-to-halfword widening).",
        f"{mnem('FADD.v2f16')} {dst('r0')}, {src('r1')}, {src('r2')}{imod('.h00')}",
        "OpVectorShuffle", "f16vec2 c = a + b.xx;"
    ),
    "swz2": (
        "Stage A lane swizzle on `src2` (e.g., `h00`/`h01`/`h10`/`h11` on `FMA.v2f16` addend).",
        f"{mnem('FMA.v2f16')} {dst('r0')}, {src('r1')}, {src('r2')}, {src('r3')}{imod('.h11')}",
        "OpVectorShuffle", "f16vec2 d = fma(a, b, c.yy);"
    ),
    "lanes0": (
        "Stage A 4-byte vector swizzle on `src0` (`b0123`, `b3210`, `b0000`, etc.) for packed 8-bit `v4*8` operations.",
        f"{mnem('SWZ.v4i8')} {dst('r0')}, {src('r1')}{imod('.b3210')}",
        "OpVectorShuffle (4x8-bit)", "u8vec4 y = x.wzyx;"
    ),
    "lanes1": (
        "Stage A byte-lane selector on `src1` (selects shift-count byte `b0..b3` on 32/64-bit shifts, `bXY` on `v2i16` shifts, or 4-byte swizzle on `v4*8`).",
        f"{mnem('LSHIFT.i32')} {dst('r0')}, {src('r1')}, {src('r2')}{imod('.b1')}",
        "OpBitFieldUExtract + OpShiftLeftLogical", "uint y = a << ((s >> 8) & 0xFFu);"
    ),
    "lane0": (
        "Stage A single-lane extractor on `src0` (`b0..b3` for 8-bit or `h0`/`h1` for 16-bit narrow conversions and `MKVEC`).",
        f"{mnem('F16_TO_F32')} {dst('r0')}, {src('r1')}{imod('.h1')}",
        "OpCompositeExtract", "float y = float(x.y);"
    ),
    "lane1": (
        "Stage A single-lane extractor on `src1` (`b0..b3` or `h0`/`h1`) for `MKVEC` packing and narrow binary ops.",
        f"{mnem('MKVEC.v2i8')} {dst('r0')}, {src('r1')}{imod('.b0')}, {src('r2')}{imod('.b1')}, {src('r3')}{imod('.b01')}",
        "OpCompositeConstruct", "u8vec4 v = u8vec4(a.x, b.y, c.x, c.y);"
    ),
    "abs0": (
        "Stage A floating-point absolute value on `src0` (clears IEEE sign bit after swizzle/widen, before `neg0`).",
        f"{mnem('FADD.f32')} {dst('r0')}, {src('r1')}{imod('.abs')}, {src('r2')}",
        "GLSL.std.450 FAbs", "float c = abs(a) + b;"
    ),
    "abs1": (
        "Stage A floating-point absolute value on `src1` (clears IEEE sign bit after swizzle/widen, before `neg1`).",
        f"{mnem('FMA.f32')} {dst('r0')}, {src('r1')}, {src('r2')}{imod('.abs')}, {src('r3')}",
        "GLSL.std.450 FAbs", "float d = fma(a, abs(b), c);"
    ),
    "abs2": (
        "Stage A floating-point absolute value on `src2` (clears IEEE sign bit of FMA addend after swizzle/widen).",
        f"{mnem('FMA.f32')} {dst('r0')}, {src('r1')}, {src('r2')}, {src('r3')}{imod('.abs')}",
        "GLSL.std.450 FAbs", "float d = fma(a, b, abs(c));"
    ),
    "neg": (
        "Negation flag on segmented pointer arithmetic (`SEG_ADD` vs `SEG_SUB`: subtracts widened `src1` when set) or unary FP negation.",
        f"{mnem('SEG_SUB.u64')} {dst('[r0:r1]')}, {src('[r2:r3]')}, {src('r4')}{imod('.w0')}",
        "OpPtrAccessChain (negative index)", "shared_arr[idx - off]"
    ),
    "neg0": (
        "Stage A floating-point negation on `src0` (flips IEEE sign bit after swizzle/widen and `abs0`).",
        f"{mnem('FADD.f32')} {dst('r0')}, {src('r1')}{imod('.neg')}, {src('r2')}",
        "OpFNegate", "float c = -a + b;"
    ),
    "neg1": (
        "Stage A floating-point negation on `src1` (flips IEEE sign bit after swizzle/widen and `abs1`).",
        f"{mnem('FMA.f32')} {dst('r0')}, {src('r1')}, {src('r2')}{imod('.neg')}, {src('r3')}",
        "OpFNegate", "float d = fma(a, -b, c);"
    ),
    "neg2": (
        "Stage A floating-point negation on `src2` (flips IEEE sign bit of FMA addend after swizzle/widen and `abs2`).",
        f"{mnem('FMA.f32')} {dst('r0')}, {src('r1')}, {src('r2')}, {src('r3')}{imod('.neg')}",
        "OpFNegate", "float d = fma(a, b, -c);"
    ),
    "not": (
        "Boolean negation of the evaluated branch condition on `BRANCH`/`JUMP`/`JUMP_EX` (inverts the boolean predicate, NOT bitwise `~src0`).",
        f"{mnem('BRANCH')}{imod('.not')} {src('k0')}, {imm('#0x10')}",
        "OpLogicalNot + OpBranchConditional", "if (!cond) goto target;"
    ),
    "not2": (
        "Stage A bitwise complement (`~src2`) applied to `src2` before the bitwise/shift-logic operation.",
        f"{mnem('AND.i32')} {dst('r0')}, {src('r1')}, {src('r2')}{imod('.not')}",
        "OpNot + OpBitwiseAnd", "uint y = a & ~b;"
    ),
    "round": (
        "Rounding mode (`none`/RNE, `round_up`, `round_down`, `round_zero`, `round_na`) applied in FP ALU, conversions, `HADD`, `LDEXP`, and Stage C narrowing.",
        f"{mnem('FROUND.f32')}{imod('.round_down')} {dst('r0')}, {src('r1')}",
        "GLSL.std.450 Floor / Ceil / Trunc", "float y = floor(x);"
    ),
    "clamp": (
        "Stage B post-op floating-point clamp (`none`, `clamp_0_inf`, `clamp_m1_1`, `clamp_0_1`) with `-0.0 < +0.0` and NaN suppression (`NaN -> lo`).",
        f"{mnem('FMA.f32')}{imod('.clamp_0_1')} {dst('r0')}, {src('r1')}, {src('r2')}, {src('r3')}",
        "GLSL.std.450 FClamp", "float y = clamp(fma(a, b, c), 0.0, 1.0);"
    ),
    "saturate": (
        "Stage B integer saturation flag (`.sat`): clamps signed/unsigned overflow or underflow to `[MIN_N, MAX_N]` / `[0, 2^N - 1]` instead of wrapping.",
        f"{mnem('IADD.s32')}{imod('.sat')} {dst('r0')}, {src('r1')}, {src('r2')}",
        "OpSDotAccSat / saturating add", "int c = clamp(int64_t(a) + int64_t(b), INT_MIN, INT_MAX);"
    ),
    "vertex_id": (
        "2-bit immediate triangle vertex index (`0..2`) selecting which provoking vertex to read in flat varying loads.",
        f"{mnem('LD_VAR_FLAT_IMM.f32')} {dst('@r0:r3')}, {imm('#0')}, {imm('#1')}",
        "OpLoad (Flat / PerVertexKHR)", "flat in vec4 v_flat;"
    ),
}

FIELD_DOCS = UNCOMMON_FIELD_DOCS

def get_field_doc(fname: str):
    if fname in UNCOMMON_FIELD_DOCS:
        return UNCOMMON_FIELD_DOCS[fname]
    # Clean prefix/suffix matches for variants like offset_hi, offset_lo, memory_width_large, encoded_*_table_index
    if fname in ("offset_hi", "offset_lo"):
        return UNCOMMON_FIELD_DOCS["offset"]
    if fname.startswith("memory_width"):
        return UNCOMMON_FIELD_DOCS["memory_width"]
    if fname.endswith("_table_index"):
        return (f"Encoded descriptor table index for \\texttt{{{tex_esc(fname)}}}.",
                f"{mnem('LD_BUFFER.i32')} {dst('@r0')}, {src('r2')}, {imm('#0')}",
                "OpAccessChain (DescriptorSet)", "uint v = buf[i];")
    if fname.endswith("_index") and fname not in ("dest_register_index",):
        return (f"Resource/slot index field \\texttt{{{tex_esc(fname)}}} selecting the target descriptor or slot.",
                f"{mnem('LD_VAR_BUF_IMM.f32')} {dst('@r0:r3')}, {imm('#0')}",
                "OpLoad (Descriptor)", "vec4 v = v_in;")
    if fname in ("primary_width", "secondary_width", "register_width"):
        return ("Selects 16-bit vs 32-bit GPR element width for the corresponding message return window.",
                f"{mnem('TEX_DUAL')}{imod('.primary_f16.secondary_f32')} {dst('@r0:r2')}, {src('@r4:r5')}",
                "OpImageSampleImplicitLod (RelaxedPrecision)", "mediump vec2 a = texture(t0, uv).rg;")
    if fname in ("dest_width_narrow", "dest_width_replicate"):
        return ("Destination halfword lane-mask / replication mode (`h0`, `h1`, `h01`, `none`).",
                f"{mnem('F32_TO_F16')} {dst('r0')}{imod('.h01')}, {src('r1')}",
                "OpFConvert (f32 -> f16)", "float16_t h = float16_t(x);")
    if fname == "signedness":
        return ("Selects signed vs unsigned interpretation for integer memory/comparison/conversion.",
                f"{mnem('LD_ATTR')}{imod('.s32.v4')} {dst('@r0:r3')}, {src('r4')}",
                "OpLoad (ivec4)", "in ivec4 a_idx;")
    if fname == "compare":
        return ("Enables depth comparison predicate in texture/shadow lookup.",
                f"{mnem('TEX_SINGLE')}{imod('.tex2d.compare')} {dst('@r0')}, {src('@r4:r6')}",
                "OpImageSampleDrefImplicitLod", "float s = texture(shadow, vec3(uv, z));")
    if fname == "lane":
        return ("Sub-word lane selector (`b0`..`b3`, `h0`, `h1`, `w0`, `d0`) for narrow checked/unpacked memory transfers.",
                f"{mnem('LOAD.i16')}{imod('.h1')} {dst('@r0')}, {src('r2')}, {imm('#0')}",
                "OpLoad (uint16_t)", "uint16_t x = ssbo.u16[i];")
    if fname == "coord_select":
        return ("Selects which coordinate pair is read from staging registers.",
                f"{mnem('TEX_GRADIENT')}{imod('.tex2d')} {dst('@r0:r1')}, {src('@r4:r5')}",
                "OpImageQueryLod", "vec2 lod = textureQueryLod(tex, uv);")
    if fname == "coordinate_or_derivative":
        return ("Selects whether `TEX_GRADIENT` computes coordinate LOD, forced quad deltas, or explicit derivatives.",
                f"{mnem('TEX_GRADIENT')}{imod('.tex2d.derivative')} {dst('@r0:r3')}, {src('@r4:r5')}",
                "OpDPdx / OpImageQueryLod", "vec2 d = dFdx(uv);")
    return None

# Backwards-compatible alias
get_uncommon_field_doc = get_field_doc

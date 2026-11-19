"""Frozen Valhall ISA semantics for the reference generator.

Per-mnemonic category, operational behavior, and Panfrost references for every
canonical Valhall mnemonic. Extracted from docs/notes.md Part I (Workstream 1:
Mesa Panfrost Compiler Semantics) and frozen here so gen_valhall_spec_tex.py no
longer parses the Markdown notes at build time.

Each ISA_SEMANTICS entry provides:
  category -- Chapter 3 section heading (one of CAT_ORDER)
  behavior -- operational semantics paragraph (Markdown inline)
  notes    -- Panfrost file:line references (Markdown inline)

To regenerate after editing docs/notes.md, re-run the extraction (see the
generator history) or update the entries below by hand.
"""

CAT_ORDER = [
    'Integer Arithmetic',
    'Bitwise & Shift-Logic',
    'Floating-Point ALU & Canonicalization',
    'SFU Range-Reduction & Helpers',
    'Cubemap Helpers',
    'Compare, Select & Mux',
    'Conversions',
    'Move, Pack & Swizzle',
    'Subgroup & Cross-Lane',
    'Control Flow, Barriers & System',
    'Memory Load / Store & LEA',
    'Atomics',
    'Attributes, Varyings & Fragment',
    'Texture & Varying-Texture',
]

ISA_SEMANTICS = {
    'BITREV': {
        "category": 'Integer Arithmetic',
        "behavior": 'Reverses the bit order of the 32-bit unsigned integer `src0` and writes the result to `dest`: for every bit index $k \\in \\{0, \\dots, 31\\}$, sets `dest[k] = src0[31 - k]`.',
        "notes": 'Exact bit-for-bit reversal (`hw_tests.rs:1265-1272`). (`kraid/ops.rs:331-355`, `encode_v9.rs:811-828`, `nir.rs:1255-1261`, `hw_tests.rs:1265-1272`)',
    },
    'CLZ': {
        "category": 'Integer Arithmetic',
        "behavior": 'Counts leading zero bits independently in each SIMD element $i \\in \\{0, \\dots, N-1\\}$ of `src0` (where $N=1$ for `u32`, $N=2$ for `v2u16`, and $N=4$ for `v4u8` on v9--v10, with lane width $W = 32/N$). When `src0[i] != 0`, writes `leading_zeros_W(src0[i])` to `dest[i]`; when `src0[i] == 0`, writes `W` by default or all-ones (`~0_W`) if `.mask` is set.',
        "notes": 'When `src0[i] == 0`: returns `bits` (32/16/8) if `!mask`, or `0xFFFFFFFF`/`0xFFFF`/`0xFF` if `mask` (`hw_tests.rs:1279-1282`). (`kraid/ops.rs:696-732`, `encode_v9.rs:902-920`, `nir.rs:1240-1279`, `hw_tests.rs:1275-1315`)',
    },
    'HADD': {
        "category": 'Integer Arithmetic',
        "behavior": 'Computes the overflow-free halving average independently for each SIMD element $i \\in \\{0, \\dots, N-1\\}$ (where $N=1$ for `s32`/`u32`, $N=2$ for `v2s16`/`v2u16`, $N=4$ for `v4s8`/`v4u8`; arch <= v10): `dest[i] = (ext(src0[i]) + ext(src1[i]) + (round == round_up ? 1 : 0)) >> 1` using an exact $(W+1)$-bit intermediate sum.',
        "notes": 'Removed in v11 (`hw_tests.rs:1768`). Arithmetic shift right for signed, logical shift right for unsigned. (`kraid/ops.rs:2157-2203`, `encode_v9.rs:1710-1734`, `nir.rs:1071-1090`, `hw_tests.rs:1765-1812`)',
    },
    'IABS': {
        "category": 'Integer Arithmetic',
        "behavior": 'Computes the wrapping signed absolute value for each SIMD element $i \\in \\{0, \\dots, N-1\\}$ of `src0` ($N=1$ for `s32`, $N=2$ for `v2s16`, $N=4$ for `v4s8` on v9--v10): `dest[i] = (src0[i] < 0) ? -src0[i] : src0[i]` (wrapping on `INT_MIN`).',
        "notes": 'For `INT_MIN` (`-2^(N-1)`), `unsigned_abs()` returns `2^(N-1)` (identical bit pattern to `INT_MIN`, wrapping, non-saturating; verified in `hw_tests.rs:1815-1845`). (`kraid/ops.rs:2211-2240`, `encode_v9.rs:1736-1753`, `nir.rs:1064-1070`, `hw_tests.rs:1815-1845`)',
    },
    'IADD': {
        "category": 'Integer Arithmetic',
        "behavior": 'Adds Stage A widened/swizzled `src0` and `src1` independently for each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `s32`/`u32`/`s64`/`u64`, $N=2$ for `v2s16`/`v2u16`, $N=4$ for `v4s8`/`v4u8` on v9--v10): `dest[i] = sat(widen0(src0)[i] + widen1(src1)[i])` when `.sat` is set, or wraps modulo $2^W$ otherwise.',
        "notes": '64-bit `IADD.sat` is NOT supported by HW (`kraid/ops.rs:2281-2284`, `hw_tests.rs:1885`). `v4s8`/`v4u8` only exist on arch <= v10 (`hw_tests.rs:1876`). (`kraid/ops.rs:2250-2298`, `encode_v9.rs:1755-1793`, `nir.rs:1091-1105`, `hw_tests.rs:1848-1902`)',
    },
    'IADDC': {
        "category": 'Integer Arithmetic',
        "behavior": 'Computes 32-bit unsigned addition with 1-bit carry-in: `sum64 = uint64(src0) + uint64(src1) + uint64(src2 & 1)`. By default writes the low 32-bit sum `sum64[31:0]` to `dest`; when `.c` (`return_carry`) is set, writes the 1-bit carry-out `(sum64 >> 32) & 1` to `dest` instead.',
        "notes": 'Unsigned 32-bit only; carry-in uses LSB of `src2`. Used on Bifrost/Valhall as FMA-unit `IADD` when `src2=0` (`bi_schedule.c:537`). (`isa-v9-v14.xml`, `bifrost/bifrost/ISA.xml:1474`, `bi_schedule.c:537-544`)',
    },
    'IADD_IMM': {
        "category": 'Integer Arithmetic',
        "behavior": 'Adds the inline 32-bit immediate `imm1w` to `src0` with wrapping arithmetic for each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `i32`, $N=2$ for `v2i16`, $N=4$ for `v4i8` on v9--v10): `dest[i] = src0[i] + imm1w[i]`.',
        "notes": 'With `src0 = 0`, canonical immediate move (`ISA.xml:62`). `v4i8` only arch <= v10. (`kraid/encode_v9.rs:1766-1782`, `bifrost/valhall/ISA.xml`)',
    },
    'IDPADD': {
        "category": 'Integer Arithmetic',
        "behavior": 'Computes the 4-element 8-bit integer dot product of bytes $k \\in \\{0, 1, 2, 3\\}$ of `src0` and `src1` and accumulates into 32-bit `src2`: `dot = sum_{k=0..3}(ext(src0.b[k]) * ext(src1.b[k]))`; writes `sat(src2 + dot)` if `.sat` is set, or wrapping `src2 + dot` otherwise.',
        "notes": 'On `v4s8` (v9-v13), both inputs are signed (`V4S8`), dst is `S32`. On v14, `unsigned0`/`unsigned1` override per-src signedness while dst remains `S32`. On `v4u8`, all unsigned. (`kraid/ops.rs:2354-2423`, `encode_v9.rs:1820-1865`, `hw_tests.rs:1952-1989`)',
    },
    'IMUL': {
        "category": 'Integer Arithmetic',
        "behavior": 'Multiplies Stage A widened/swizzled `src0` and `src1` independently for each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `s32`/`u32`/`s64`/`u64`, $N=2$ for `v2s16`/`v2u16`, $N=4$ for `v4s8`/`v4u8` on v9--v10): `dest[i] = sat(widen0(src0)[i] * widen1(src1)[i])` under `.sat`, or low $W$ bits otherwise. For 64-bit `s64`/`u64`, performs a $32 \\times 32 \\to 64$ widening multiply requiring `.w0`/`.w1` on both sources.',
        "notes": '64-bit `IMUL.sat` NOT supported (`ops.rs:2465-2468`); `IMUL.s64/u64` requires widening swizzle (`None` swizzle unsupported, `hw_tests.rs:2022-2027`). (`kraid/ops.rs:2433-2483`, `encode_v9.rs:1867-1896`, `nir.rs:1106-1136`, `hw_tests.rs:1992-2060`)',
    },
    'ISUB': {
        "category": 'Integer Arithmetic',
        "behavior": 'Subtracts Stage A widened/swizzled `src1` from `src0` independently for each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `s32`/`u32`/`s64`/`u64`, $N=2$ for `v2s16`/`v2u16`, $N=4$ for `v4s8`/`v4u8` on v9--v10): `dest[i] = sat(widen0(src0)[i] - widen1(src1)[i])` when `.sat` is set, or wraps modulo $2^W$ otherwise.',
        "notes": '64-bit `ISUB.sat` NOT supported by HW (`kraid/ops.rs:2523-2526`, `hw_tests.rs:2093`). Unsigned sat clamps underflow to 0. `v4s8`/`v4u8` arch <= v10. (`kraid/ops.rs:2492-2540`, `encode_v9.rs:1898-1918`, `nir.rs:1137-1159`, `hw_tests.rs:2063-2110`)',
    },
    'ISUBB': {
        "category": 'Integer Arithmetic',
        "behavior": '32-bit unsigned subtract with borrow-in: `diff64 = zext64(src0) - zext64(src1) - (src2 & 1)`; if `return_carry == c`, returns borrow-out (`src0 < src1 + bin ? 1 : 0`), else `diff64[31:0]`.',
        "notes": 'Unsigned 32-bit only; borrow-in uses LSB of `src2`. (`isa-v9-v14.xml`, `bifrost/bifrost/ISA.xml:1615`)',
    },
    'MMUL': {
        "category": 'Integer Arithmetic',
        "behavior": 'Subgroup/quad cooperative matrix multiply-accumulate `D = A * B + C` across lanes: `f32` (4x4*4x4->4x4), `v2f16` (4x4 submat of 4x8 -> 4x4 f32), `f16` (4x4 submat of 4x8 * 4x8 -> 4x8 f16, v14), `v4s8`/`v4u8` (4x16*16x4 -> 4x4 s32/u32).',
        "notes": 'Arch >= v11 (`f16` variant and mixed-sign/sat `v4s8` on v14). Cross-lane subgroup operation. (`kraid/ops.rs:3411-3550`, `encode_v9.rs:2723-2819`, `nir.rs:1907-1957`)',
    },
    'POPCOUNT': {
        "category": 'Integer Arithmetic',
        "behavior": 'Counts the total number of `1` bits in the 32-bit word `src0` and writes the exact unsigned count (in the range $0 \\dots 32$) to `dest`.',
        "notes": 'Exact 32-bit popcount (`hw_tests.rs:2138-2145`). (`kraid/ops.rs:3738-3762`, `encode_v9.rs:2897-2914`, `nir.rs:1248-1254`, `hw_tests.rs:2138-2145`)',
    },
    'SEG_ADD': {
        "category": 'Integer Arithmetic',
        "behavior": 'Computes 64-bit segment pointer addition: widens `src1` to 64 bits and adds it to the 64-bit base pointer `src0` (`dest = src0 + widen64(src1)`), wrapping modulo $2^{64}$. When `.preserve_null` is set and `src0 == 0`, returns `0` so null pointers remain null.',
        "notes": 'Used for WLS/TLS segment pointer arithmetic; `preserve_null` keeps null pointer `0` as `0`. (`bifrost/valhall/ISA.xml:2666-2684`, `bifrost/compiler.h:720-722`, `isa-v9-v14.xml`)',
    },
    'SEG_SUB': {
        "category": 'Integer Arithmetic',
        "behavior": 'Computes 64-bit segment pointer subtraction: widens `src1` to 64 bits and subtracts it from the 64-bit base pointer `src0` (`dest = src0 - widen64(src1)`), wrapping modulo $2^{64}$. When `.preserve_null` is set and `src0 == 0`, returns `0` so null pointers remain null.',
        "notes": 'Same null-preservation rule as `SEG_ADD`. (`bifrost/valhall/ISA.xml:2666-2684`, `isa-v9-v14.xml`)',
    },
    'SHADD': {
        "category": 'Integer Arithmetic',
        "behavior": 'Computes 64-bit shift-and-add address arithmetic (`s64` or `u64`): widens `src1` to 64 bits, shifts it left by the 3-bit immediate `#shift` ($0 \\dots 7$), and adds it to the 64-bit base `src0` (`dest = src0 + (widen64(src1) << #shift)`), wrapping modulo $2^{64}$.',
        "notes": 'Only `s64` and `u64` variants. Wraps modulo `2^64`. (`bifrost/valhall/ISA.xml:2686-2707`, `bifrost/bir.c:80,128-130`, `isa-v9-v14.xml`)',
    },
    'AND': {
        "category": 'Bitwise & Shift-Logic',
        "behavior": 'Computes bitwise AND independently across each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `i32`/`i64`, $N=2$ for `v2i16`, $N=4$ for `v4i8` on v9--v10): `res = widen0(src0)[i] & (not2 ? ~src2[i] : src2[i])`, then writes `not_result ? ~res : res` to `dest[i]` (`src1` unused/fixed 0).',
        "notes": 'Second operand is `src2` (NOT `src1`) in XML encoding (`encode_v9.rs:2982-2985`). Supports `i32`, `i64`, `v2i16`, `v4i8`. (`kraid/ops.rs:3944-4025`, `encode_v9.rs:2916-3093`, `hw_tests.rs:2148-2198`)',
    },
    'AND_IMM': {
        "category": 'Bitwise & Shift-Logic',
        "behavior": 'Computes a 32-bit bitwise AND between `src0` and the inline 32-bit immediate `#imm1w` (`dest = src0 & #imm1w`) on Valhall v11+. Does not support source swizzles or `.not_result`; supports `dest_width` halfword write masking.',
        "notes": 'No `not_result` or `src0` swizzle supported (`encode_v9.rs:3043`). (`kraid/encode_v9.rs:2919-2926, 3036-3054`)',
    },
    'ARSHIFT': {
        "category": 'Bitwise & Shift-Logic',
        "behavior": 'For each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `i32`/`i64`, $N=2$ for `v2i16`, $N=4$ for `v4i8` on v9--v10, lane width $W$), computes arithmetic right shift `s = (int_W(val) >> (amt & (W - 1)))` where `val = src0[i]` and the 8-bit shift count `amt` is extracted from `src1` via `lanes1`; writes `not_result ? ~s : s` to `dest[i]`.',
        "notes": 'Shift amount is masked modulo lane width (`wrapping_shl`/`wrapping_shr`/`rotate_*`, `kraid/ops.rs:3984-4009`). On `v2i16`, `src1` is `V2I8` (`ops.rs:3953`) with byte lanes selected by `lanes1` (default `b02`: byte 0 -> lane 0, byte 2 -> lane 1). (`kraid/ops.rs:3878-4025`, `encode_v9.rs:2940-3093`, `hw_tests.rs:2148-2198`)',
    },
    'ARSHIFT_AND': {
        "category": 'Bitwise & Shift-Logic',
        "behavior": 'For each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `i32`/`i64`, $N=2$ for `v2i16`, $N=4$ for `v4i8` on v9--v10, lane width $W$), computes arithmetic right shift `s = (int_W(val) >> (amt & (W - 1)))` where `val = src0[i]` and the 8-bit shift count `amt` is extracted from `src1` via `lanes1`, then combines with `src2[i]` via bitwise AND `res = s & (not2 ? ~src2[i] : src2[i])` and writes `not_result ? ~res : res` to `dest[i]`.',
        "notes": 'Shift amount is masked modulo lane width (`wrapping_shl`/`wrapping_shr`/`rotate_*`, `kraid/ops.rs:3984-4009`). On `v2i16`, `src1` is `V2I8` (`ops.rs:3953`) with byte lanes selected by `lanes1` (default `b02`: byte 0 -> lane 0, byte 2 -> lane 1). (`kraid/ops.rs:3878-4025`, `encode_v9.rs:2940-3093`, `hw_tests.rs:2148-2198`)',
    },
    'ARSHIFT_OR': {
        "category": 'Bitwise & Shift-Logic',
        "behavior": 'For each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `i32`/`i64`, $N=2$ for `v2i16`, $N=4$ for `v4i8` on v9--v10, lane width $W$), computes arithmetic right shift `s = (int_W(val) >> (amt & (W - 1)))` where `val = src0[i]` and the 8-bit shift count `amt` is extracted from `src1` via `lanes1`, then combines with `src2[i]` via bitwise OR `res = s | (not2 ? ~src2[i] : src2[i])` and writes `not_result ? ~res : res` to `dest[i]`.',
        "notes": 'Shift amount is masked modulo lane width (`wrapping_shl`/`wrapping_shr`/`rotate_*`, `kraid/ops.rs:3984-4009`). On `v2i16`, `src1` is `V2I8` (`ops.rs:3953`) with byte lanes selected by `lanes1` (default `b02`: byte 0 -> lane 0, byte 2 -> lane 1). (`kraid/ops.rs:3878-4025`, `encode_v9.rs:2940-3093`, `hw_tests.rs:2148-2198`)',
    },
    'ARSHIFT_XOR': {
        "category": 'Bitwise & Shift-Logic',
        "behavior": 'For each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `i32`/`i64`, $N=2$ for `v2i16`, $N=4$ for `v4i8` on v9--v10, lane width $W$), computes arithmetic right shift `s = (int_W(val) >> (amt & (W - 1)))` where `val = src0[i]` and the 8-bit shift count `amt` is extracted from `src1` via `lanes1`, then combines with `src2[i]` via bitwise XOR `res = s ^ (not2 ? ~src2[i] : src2[i])` and writes `not_result ? ~res : res` to `dest[i]`.',
        "notes": 'Shift amount is masked modulo lane width (`wrapping_shl`/`wrapping_shr`/`rotate_*`, `kraid/ops.rs:3984-4009`). On `v2i16`, `src1` is `V2I8` (`ops.rs:3953`) with byte lanes selected by `lanes1` (default `b02`: byte 0 -> lane 0, byte 2 -> lane 1). (`kraid/ops.rs:3878-4025`, `encode_v9.rs:2940-3093`, `hw_tests.rs:2148-2198`)',
    },
    'LROT': {
        "category": 'Bitwise & Shift-Logic',
        "behavior": 'For each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `i32`/`i64`, $N=2$ for `v2i16`, $N=4$ for `v4i8` on v9--v10, lane width $W$), computes rotate left `s = rotate_left(val, amt & (W - 1))` where `val = src0[i]` and the 8-bit shift count `amt` is extracted from `src1` via `lanes1`; writes `not_result ? ~s : s` to `dest[i]`.',
        "notes": 'Shift amount is masked modulo lane width (`wrapping_shl`/`wrapping_shr`/`rotate_*`, `kraid/ops.rs:3984-4009`). On `v2i16`, `src1` is `V2I8` (`ops.rs:3953`) with byte lanes selected by `lanes1` (default `b02`: byte 0 -> lane 0, byte 2 -> lane 1). (`kraid/ops.rs:3878-4025`, `encode_v9.rs:2940-3093`, `hw_tests.rs:2148-2198`)',
    },
    'LROT_AND': {
        "category": 'Bitwise & Shift-Logic',
        "behavior": 'For each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `i32`/`i64`, $N=2$ for `v2i16`, $N=4$ for `v4i8` on v9--v10, lane width $W$), computes rotate left `s = rotate_left(val, amt & (W - 1))` where `val = src0[i]` and the 8-bit shift count `amt` is extracted from `src1` via `lanes1`, then combines with `src2[i]` via bitwise AND `res = s & (not2 ? ~src2[i] : src2[i])` and writes `not_result ? ~res : res` to `dest[i]`.',
        "notes": 'Shift amount is masked modulo lane width (`wrapping_shl`/`wrapping_shr`/`rotate_*`, `kraid/ops.rs:3984-4009`). On `v2i16`, `src1` is `V2I8` (`ops.rs:3953`) with byte lanes selected by `lanes1` (default `b02`: byte 0 -> lane 0, byte 2 -> lane 1). (`kraid/ops.rs:3878-4025`, `encode_v9.rs:2940-3093`, `hw_tests.rs:2148-2198`)',
    },
    'LROT_OR': {
        "category": 'Bitwise & Shift-Logic',
        "behavior": 'For each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `i32`/`i64`, $N=2$ for `v2i16`, $N=4$ for `v4i8` on v9--v10, lane width $W$), computes rotate left `s = rotate_left(val, amt & (W - 1))` where `val = src0[i]` and the 8-bit shift count `amt` is extracted from `src1` via `lanes1`, then combines with `src2[i]` via bitwise OR `res = s | (not2 ? ~src2[i] : src2[i])` and writes `not_result ? ~res : res` to `dest[i]`.',
        "notes": 'Shift amount is masked modulo lane width (`wrapping_shl`/`wrapping_shr`/`rotate_*`, `kraid/ops.rs:3984-4009`). On `v2i16`, `src1` is `V2I8` (`ops.rs:3953`) with byte lanes selected by `lanes1` (default `b02`: byte 0 -> lane 0, byte 2 -> lane 1). (`kraid/ops.rs:3878-4025`, `encode_v9.rs:2940-3093`, `hw_tests.rs:2148-2198`)',
    },
    'LROT_XOR': {
        "category": 'Bitwise & Shift-Logic',
        "behavior": 'For each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `i32`/`i64`, $N=2$ for `v2i16`, $N=4$ for `v4i8` on v9--v10, lane width $W$), computes rotate left `s = rotate_left(val, amt & (W - 1))` where `val = src0[i]` and the 8-bit shift count `amt` is extracted from `src1` via `lanes1`, then combines with `src2[i]` via bitwise XOR `res = s ^ (not2 ? ~src2[i] : src2[i])` and writes `not_result ? ~res : res` to `dest[i]`.',
        "notes": 'Shift amount is masked modulo lane width (`wrapping_shl`/`wrapping_shr`/`rotate_*`, `kraid/ops.rs:3984-4009`). On `v2i16`, `src1` is `V2I8` (`ops.rs:3953`) with byte lanes selected by `lanes1` (default `b02`: byte 0 -> lane 0, byte 2 -> lane 1). (`kraid/ops.rs:3878-4025`, `encode_v9.rs:2940-3093`, `hw_tests.rs:2148-2198`)',
    },
    'LSHIFT': {
        "category": 'Bitwise & Shift-Logic',
        "behavior": 'For each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `i32`/`i64`, $N=2$ for `v2i16`, $N=4$ for `v4i8` on v9--v10, lane width $W$), computes logical left shift `s = (val << (amt & (W - 1)))` where `val = src0[i]` and the 8-bit shift count `amt` is extracted from `src1` via `lanes1`; writes `not_result ? ~s : s` to `dest[i]`.',
        "notes": 'Shift amount is masked modulo lane width (`wrapping_shl`/`wrapping_shr`/`rotate_*`, `kraid/ops.rs:3984-4009`). On `v2i16`, `src1` is `V2I8` (`ops.rs:3953`) with byte lanes selected by `lanes1` (default `b02`: byte 0 -> lane 0, byte 2 -> lane 1). (`kraid/ops.rs:3878-4025`, `encode_v9.rs:2940-3093`, `hw_tests.rs:2148-2198`)',
    },
    'LSHIFT_AND': {
        "category": 'Bitwise & Shift-Logic',
        "behavior": 'For each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `i32`/`i64`, $N=2$ for `v2i16`, $N=4$ for `v4i8` on v9--v10, lane width $W$), computes logical left shift `s = (val << (amt & (W - 1)))` where `val = src0[i]` and the 8-bit shift count `amt` is extracted from `src1` via `lanes1`, then combines with `src2[i]` via bitwise AND `res = s & (not2 ? ~src2[i] : src2[i])` and writes `not_result ? ~res : res` to `dest[i]`.',
        "notes": 'Shift amount is masked modulo lane width (`wrapping_shl`/`wrapping_shr`/`rotate_*`, `kraid/ops.rs:3984-4009`). On `v2i16`, `src1` is `V2I8` (`ops.rs:3953`) with byte lanes selected by `lanes1` (default `b02`: byte 0 -> lane 0, byte 2 -> lane 1). (`kraid/ops.rs:3878-4025`, `encode_v9.rs:2940-3093`, `hw_tests.rs:2148-2198`)',
    },
    'LSHIFT_OR': {
        "category": 'Bitwise & Shift-Logic',
        "behavior": 'For each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `i32`/`i64`, $N=2$ for `v2i16`, $N=4$ for `v4i8` on v9--v10, lane width $W$), computes logical left shift `s = (val << (amt & (W - 1)))` where `val = src0[i]` and the 8-bit shift count `amt` is extracted from `src1` via `lanes1`, then combines with `src2[i]` via bitwise OR `res = s | (not2 ? ~src2[i] : src2[i])` and writes `not_result ? ~res : res` to `dest[i]`.',
        "notes": 'Shift amount is masked modulo lane width (`wrapping_shl`/`wrapping_shr`/`rotate_*`, `kraid/ops.rs:3984-4009`). On `v2i16`, `src1` is `V2I8` (`ops.rs:3953`) with byte lanes selected by `lanes1` (default `b02`: byte 0 -> lane 0, byte 2 -> lane 1). (`kraid/ops.rs:3878-4025`, `encode_v9.rs:2940-3093`, `hw_tests.rs:2148-2198`)',
    },
    'LSHIFT_XOR': {
        "category": 'Bitwise & Shift-Logic',
        "behavior": 'For each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `i32`/`i64`, $N=2$ for `v2i16`, $N=4$ for `v4i8` on v9--v10, lane width $W$), computes logical left shift `s = (val << (amt & (W - 1)))` where `val = src0[i]` and the 8-bit shift count `amt` is extracted from `src1` via `lanes1`, then combines with `src2[i]` via bitwise XOR `res = s ^ (not2 ? ~src2[i] : src2[i])` and writes `not_result ? ~res : res` to `dest[i]`.',
        "notes": 'Shift amount is masked modulo lane width (`wrapping_shl`/`wrapping_shr`/`rotate_*`, `kraid/ops.rs:3984-4009`). On `v2i16`, `src1` is `V2I8` (`ops.rs:3953`) with byte lanes selected by `lanes1` (default `b02`: byte 0 -> lane 0, byte 2 -> lane 1). (`kraid/ops.rs:3878-4025`, `encode_v9.rs:2940-3093`, `hw_tests.rs:2148-2198`)',
    },
    'NOT': {
        "category": 'Bitwise & Shift-Logic',
        "behavior": 'Computes the bitwise complement of `src0` (`dest = ~widen(src0)`) across all lanes. Encoded as an alias of `OR` with `src2 = 0` and `.not_result` enabled.',
        "notes": 'Bitwise NOT across all bits of destination width. (`kraid/nir.rs:1160-1171`, `encode_v9.rs:3060-3066`)',
    },
    'NOT_OLD': {
        "category": 'Bitwise & Shift-Logic',
        "behavior": 'Legacy v9--v13 unary bitwise complement on `i32` or `i64`: inverts every bit of `src0` and writes `dest = ~src0`. Superseded by shift-logic `NOT` on newer architectures.',
        "notes": 'Superseded by shift-logic `NOT` (`bifrost/valhall/ISA.xml:137`). (`bifrost/valhall/ISA.xml:137-138`, `isa-v9-v14.xml`)',
    },
    'OR': {
        "category": 'Bitwise & Shift-Logic',
        "behavior": 'Computes bitwise OR independently across each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `i32`/`i64`, $N=2$ for `v2i16`, $N=4$ for `v4i8` on v9--v10): `res = widen0(src0)[i] | (not2 ? ~src2[i] : src2[i])`, then writes `not_result ? ~res : res` to `dest[i]`.',
        "notes": 'Second operand is `src2`. `OpShiftLop(None, None)` encodes as `OR src0, 0` (`encode_v9.rs:3060-3066`). (`kraid/ops.rs:3944-4025`, `encode_v9.rs:2916-3093`, `hw_tests.rs:2148-2198`)',
    },
    'OR_IMM': {
        "category": 'Bitwise & Shift-Logic',
        "behavior": 'Computes a 32-bit bitwise OR between `src0` and the inline 32-bit immediate `#imm1w` (`dest = src0 | #imm1w`) on Valhall v11+. Does not support source swizzles or `.not_result`; supports `dest_width` halfword write masking.',
        "notes": 'No `not_result` or `src0` swizzle supported. (`kraid/encode_v9.rs:2919-2926, 3036-3054`)',
    },
    'RROT': {
        "category": 'Bitwise & Shift-Logic',
        "behavior": 'For each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `i32`/`i64`, $N=2$ for `v2i16`, $N=4$ for `v4i8` on v9--v10, lane width $W$), computes rotate right `s = rotate_right(val, amt & (W - 1))` where `val = src0[i]` and the 8-bit shift count `amt` is extracted from `src1` via `lanes1`; writes `not_result ? ~s : s` to `dest[i]`.',
        "notes": 'Shift amount is masked modulo lane width (`wrapping_shl`/`wrapping_shr`/`rotate_*`, `kraid/ops.rs:3984-4009`). On `v2i16`, `src1` is `V2I8` (`ops.rs:3953`) with byte lanes selected by `lanes1` (default `b02`: byte 0 -> lane 0, byte 2 -> lane 1). (`kraid/ops.rs:3878-4025`, `encode_v9.rs:2940-3093`, `hw_tests.rs:2148-2198`)',
    },
    'RROT_AND': {
        "category": 'Bitwise & Shift-Logic',
        "behavior": 'For each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `i32`/`i64`, $N=2$ for `v2i16`, $N=4$ for `v4i8` on v9--v10, lane width $W$), computes rotate right `s = rotate_right(val, amt & (W - 1))` where `val = src0[i]` and the 8-bit shift count `amt` is extracted from `src1` via `lanes1`, then combines with `src2[i]` via bitwise AND `res = s & (not2 ? ~src2[i] : src2[i])` and writes `not_result ? ~res : res` to `dest[i]`.',
        "notes": 'Shift amount is masked modulo lane width (`wrapping_shl`/`wrapping_shr`/`rotate_*`, `kraid/ops.rs:3984-4009`). On `v2i16`, `src1` is `V2I8` (`ops.rs:3953`) with byte lanes selected by `lanes1` (default `b02`: byte 0 -> lane 0, byte 2 -> lane 1). (`kraid/ops.rs:3878-4025`, `encode_v9.rs:2940-3093`, `hw_tests.rs:2148-2198`)',
    },
    'RROT_OR': {
        "category": 'Bitwise & Shift-Logic',
        "behavior": 'For each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `i32`/`i64`, $N=2$ for `v2i16`, $N=4$ for `v4i8` on v9--v10, lane width $W$), computes rotate right `s = rotate_right(val, amt & (W - 1))` where `val = src0[i]` and the 8-bit shift count `amt` is extracted from `src1` via `lanes1`, then combines with `src2[i]` via bitwise OR `res = s | (not2 ? ~src2[i] : src2[i])` and writes `not_result ? ~res : res` to `dest[i]`.',
        "notes": 'Shift amount is masked modulo lane width (`wrapping_shl`/`wrapping_shr`/`rotate_*`, `kraid/ops.rs:3984-4009`). On `v2i16`, `src1` is `V2I8` (`ops.rs:3953`) with byte lanes selected by `lanes1` (default `b02`: byte 0 -> lane 0, byte 2 -> lane 1). (`kraid/ops.rs:3878-4025`, `encode_v9.rs:2940-3093`, `hw_tests.rs:2148-2198`)',
    },
    'RROT_XOR': {
        "category": 'Bitwise & Shift-Logic',
        "behavior": 'For each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `i32`/`i64`, $N=2$ for `v2i16`, $N=4$ for `v4i8` on v9--v10, lane width $W$), computes rotate right `s = rotate_right(val, amt & (W - 1))` where `val = src0[i]` and the 8-bit shift count `amt` is extracted from `src1` via `lanes1`, then combines with `src2[i]` via bitwise XOR `res = s ^ (not2 ? ~src2[i] : src2[i])` and writes `not_result ? ~res : res` to `dest[i]`.',
        "notes": 'Shift amount is masked modulo lane width (`wrapping_shl`/`wrapping_shr`/`rotate_*`, `kraid/ops.rs:3984-4009`). On `v2i16`, `src1` is `V2I8` (`ops.rs:3953`) with byte lanes selected by `lanes1` (default `b02`: byte 0 -> lane 0, byte 2 -> lane 1). (`kraid/ops.rs:3878-4025`, `encode_v9.rs:2940-3093`, `hw_tests.rs:2148-2198`)',
    },
    'RSHIFT': {
        "category": 'Bitwise & Shift-Logic',
        "behavior": 'For each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `i32`/`i64`, $N=2$ for `v2i16`, $N=4$ for `v4i8` on v9--v10, lane width $W$), computes logical right shift `s = (uint_W(val) >> (amt & (W - 1)))` where `val = src0[i]` and the 8-bit shift count `amt` is extracted from `src1` via `lanes1`; writes `not_result ? ~s : s` to `dest[i]`.',
        "notes": 'Shift amount is masked modulo lane width (`wrapping_shl`/`wrapping_shr`/`rotate_*`, `kraid/ops.rs:3984-4009`). On `v2i16`, `src1` is `V2I8` (`ops.rs:3953`) with byte lanes selected by `lanes1` (default `b02`: byte 0 -> lane 0, byte 2 -> lane 1). (`kraid/ops.rs:3878-4025`, `encode_v9.rs:2940-3093`, `hw_tests.rs:2148-2198`)',
    },
    'RSHIFT_AND': {
        "category": 'Bitwise & Shift-Logic',
        "behavior": 'For each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `i32`/`i64`, $N=2$ for `v2i16`, $N=4$ for `v4i8` on v9--v10, lane width $W$), computes logical right shift `s = (uint_W(val) >> (amt & (W - 1)))` where `val = src0[i]` and the 8-bit shift count `amt` is extracted from `src1` via `lanes1`, then combines with `src2[i]` via bitwise AND `res = s & (not2 ? ~src2[i] : src2[i])` and writes `not_result ? ~res : res` to `dest[i]`.',
        "notes": 'Shift amount is masked modulo lane width (`wrapping_shl`/`wrapping_shr`/`rotate_*`, `kraid/ops.rs:3984-4009`). On `v2i16`, `src1` is `V2I8` (`ops.rs:3953`) with byte lanes selected by `lanes1` (default `b02`: byte 0 -> lane 0, byte 2 -> lane 1). (`kraid/ops.rs:3878-4025`, `encode_v9.rs:2940-3093`, `hw_tests.rs:2148-2198`)',
    },
    'RSHIFT_OR': {
        "category": 'Bitwise & Shift-Logic',
        "behavior": 'For each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `i32`/`i64`, $N=2$ for `v2i16`, $N=4$ for `v4i8` on v9--v10, lane width $W$), computes logical right shift `s = (uint_W(val) >> (amt & (W - 1)))` where `val = src0[i]` and the 8-bit shift count `amt` is extracted from `src1` via `lanes1`, then combines with `src2[i]` via bitwise OR `res = s | (not2 ? ~src2[i] : src2[i])` and writes `not_result ? ~res : res` to `dest[i]`.',
        "notes": 'Shift amount is masked modulo lane width (`wrapping_shl`/`wrapping_shr`/`rotate_*`, `kraid/ops.rs:3984-4009`). On `v2i16`, `src1` is `V2I8` (`ops.rs:3953`) with byte lanes selected by `lanes1` (default `b02`: byte 0 -> lane 0, byte 2 -> lane 1). (`kraid/ops.rs:3878-4025`, `encode_v9.rs:2940-3093`, `hw_tests.rs:2148-2198`)',
    },
    'RSHIFT_XOR': {
        "category": 'Bitwise & Shift-Logic',
        "behavior": 'For each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `i32`/`i64`, $N=2$ for `v2i16`, $N=4$ for `v4i8` on v9--v10, lane width $W$), computes logical right shift `s = (uint_W(val) >> (amt & (W - 1)))` where `val = src0[i]` and the 8-bit shift count `amt` is extracted from `src1` via `lanes1`, then combines with `src2[i]` via bitwise XOR `res = s ^ (not2 ? ~src2[i] : src2[i])` and writes `not_result ? ~res : res` to `dest[i]`.',
        "notes": 'Shift amount is masked modulo lane width (`wrapping_shl`/`wrapping_shr`/`rotate_*`, `kraid/ops.rs:3984-4009`). On `v2i16`, `src1` is `V2I8` (`ops.rs:3953`) with byte lanes selected by `lanes1` (default `b02`: byte 0 -> lane 0, byte 2 -> lane 1). (`kraid/ops.rs:3878-4025`, `encode_v9.rs:2940-3093`, `hw_tests.rs:2148-2198`)',
    },
    'XOR': {
        "category": 'Bitwise & Shift-Logic',
        "behavior": 'Computes bitwise XOR independently across each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `i32`/`i64`, $N=2$ for `v2i16`, $N=4$ for `v4i8` on v9--v10): `res = widen0(src0)[i] ^ (not2 ? ~src2[i] : src2[i])`, then writes `not_result ? ~res : res` to `dest[i]`.',
        "notes": 'Second operand is `src2`. (`kraid/ops.rs:3944-4025`, `encode_v9.rs:2916-3093`, `hw_tests.rs:2148-2198`)',
    },
    'XOR_IMM': {
        "category": 'Bitwise & Shift-Logic',
        "behavior": 'Computes a 32-bit bitwise XOR between `src0` and the inline 32-bit immediate `#imm1w` (`dest = src0 ^ #imm1w`) on Valhall v11+. Does not support source swizzles or `.not_result`; supports `dest_width` halfword write masking.',
        "notes": 'No `not_result` or `src0` swizzle supported. (`kraid/encode_v9.rs:2919-2926, 3036-3054`)',
    },
    'FABS': {
        "category": 'Floating-Point ALU & Canonicalization',
        "behavior": 'Computes the floating-point absolute value of `src0` by clearing its IEEE-754 sign bit (`dest = |src0|`), then applies optional `round`, `clamp`, and `dest_width_narrow` modifiers. Encoded as an alias of `FADD(abs(src0), -0.0)`.',
        "notes": 'Lowered in `kraid/nir.rs:766-775` as `OpFAdd(-0.0, src0.fabs())`. Because it is an FADD alias, adding `-0.0` canonicalizes sNaN->qNaN and applies FTZ if global FTZ is enabled. (`kraid/nir.rs:766-775`, `isa-v9-v14.xml`)',
    },
    'FADD': {
        "category": 'Floating-Point ALU & Canonicalization',
        "behavior": 'Adds floating-point operands `src0` and `src1` independently for each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `f32`, $N=2$ for packed `v2f16`), rounding via `round` and applying optional post-op `clamp`: `dest[i] = clamp(round_rm(src0[i] + src1[i]))`.',
        "notes": 'Clamp suppresses NaN (`x.ieee_max(lo).ieee_min(hi)`, `ops.rs:1136-1143`). `dest_width_narrow` narrows f32->f16 using `round` (`dst_mod_prop.rs:226-229`, `builder.rs:50-65`). (`kraid/ops.rs:1267-1302`, `encode_v9.rs:1271-1315`, `dst_mod_prop.rs:199-236`, `hw_tests.rs:1486-1519`)',
    },
    'FADD_IMM': {
        "category": 'Floating-Point ALU & Canonicalization',
        "behavior": 'Adds the inline 32-bit floating-point immediate `imm1w` to `src0` under IEEE-754 RNE for each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `f32`, $N=2$ for `v2f16`): `dest[i] = round_rne(src0[i] + imm1w[i])`.',
        "notes": 'Always RNE, unclamped (`encode_v9.rs:1282-1302`). (`kraid/encode_v9.rs:1282-1302`, `bifrost/valhall/ISA.xml:40-42`)',
    },
    'FEXP': {
        "category": 'Floating-Point ALU & Canonicalization',
        "behavior": 'Computes base-2 exponential $2^x$. For `FEXP.f32`, `src0` supplies a signed Q8.24 fixed-point exponent (`int(round(x * 2^24))`) and `src1` supplies the original FP32 `x` for overflow/underflow/NaN propagation; subnormal outputs flush to zero. For `FEXP.f16` (v12+), computes $2^{\\text{src1}}$ directly from the FP16 lane in `src1` (`src0` is unused) and supports `dest_width_replicate`.',
        "notes": '` ULP (`hw_tests.rs:2297`). (`kraid/ops.rs:1589-1635`, `encode_v9.rs:1361-1400`, `builder.rs:353-384`, `hw_tests.rs:2295-2345`)',
    },
    'FEXPF': {
        "category": 'Floating-Point ALU & Canonicalization',
        "behavior": 'Directly computes the single-precision base-2 exponential `dest = exp2(src0)` on Valhall v14+ without requiring Q8.24 fixed-point pre-scaling in software. Handles standard IEEE-754 overflow to `+Inf`, underflow to `+0.0`, and NaN propagation.',
        "notes": 'Replaces the 3-instruction `FMA_RSCALE` + `F32_TO_S32` + `FEXP.f32` sequence on v14+. (`isa-v9-v14.xml`)',
    },
    'FLOG': {
        "category": 'Floating-Point ALU & Canonicalization',
        "behavior": 'Directly computes the base-2 logarithm `dest = log2(src0)` (`f16` on v12+, `f32` on v14+) in a single SFU instruction. Returns `-Inf` for `±0.0`, `NaN` for negative inputs, and `+Inf` for `+Inf`.',
        "notes": 'Negative -> NaN, `0` -> `-Inf`, `+Inf` -> `+Inf`. (`kraid/ops.rs:1640-1656`, `encode_v9.rs:1402-1419`, `nir.rs:931-942`, `isa-v9-v14.xml`)',
    },
    'FLUSH': {
        "category": 'Floating-Point ALU & Canonicalization',
        "behavior": 'Canonicalizes IEEE-754 special values independently for each floating-point element $i \\in \\{0, \\dots, N-1\\}$ of `src0` ($N=1$ `f32`, $N=2$ `v2f16`): NaN maps to `+0.0` (`.flush_nan`) or quietened NaN (`.quiet_nan`); `±Inf` clamps to `±MAX_FINITE` (`.flush_inf`); subnormals flush to `±0.0` (`.ftz`); normal finite values pass through unchanged.',
        "notes": '`qnan_bit` is `0x00400000` (f32) / `0x0200` (f16); `MAX_FINITE` is `0x7F7FFFFF` / `0x7BFF`. Exact bit-for-bit verified (`hw_tests.rs:1570-1596`). (`kraid/ops.rs:1681-1783`, `encode_v9.rs:1440-1470`, `hw_tests.rs:1570-1596`)',
    },
    'FMA': {
        "category": 'Floating-Point ALU & Canonicalization',
        "behavior": 'Computes single-rounded fused multiply-add independently for each floating-point element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `f32`, $N=2$ for `v2f16`): `dest[i] = clamp(round_rm(fma(src0[i], src1[i], src2[i])))`.',
        "notes": 'Single rounding for `a*b+c`; `dest_width_narrow` narrows f32->f16 using `round` (`dst_mod_prop.rs:566-569`). (`kraid/ops.rs:1788-1824`, `encode_v9.rs:1472-1495`, `hw_tests.rs:1599-1631`)',
    },
    'FMAX': {
        "category": 'Floating-Point ALU & Canonicalization',
        "behavior": 'Computes IEEE-754 maximum independently for each floating-point element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ `f32`, $N=2$ `v2f16`) with deterministic `-0.0 < +0.0`. Under `nan_suppress` (default), a non-NaN wins over NaN; under `nan_propagate`, returns the quietened winning NaN preserving sign and payload (`| qbit`). Optional `clamp` runs after `sem`.',
        "notes": 'Verified in `libGLES_mali.so` `_mali_max*_sf32` (`0x2e28300, 0x2e28590`) and `kraid/ops.rs:1862-1902`.',
    },
    'FMIN': {
        "category": 'Floating-Point ALU & Canonicalization',
        "behavior": 'Computes IEEE-754 minimum independently for each floating-point element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ `f32`, $N=2$ `v2f16`) with deterministic `-0.0 < +0.0`. Under `nan_suppress` (default), a non-NaN wins over NaN; under `nan_propagate`, returns the quietened winning NaN preserving sign and payload (`| qbit`). Optional `clamp` runs after `sem`.',
        "notes": 'Verified in `libGLES_mali.so` `_mali_min_sf32` (`0x2e28500`) and `kraid/ops.rs:1907-1947`.',
    },
    'FMUL': {
        "category": 'Floating-Point ALU & Canonicalization',
        "behavior": 'Multiplies floating-point `src0` and `src1` independently for each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ `f32`, $N=2$ `v2f16`): `dest[i] = clamp(round_rm(src0[i] * src1[i]))` (hardware alias of `FMA(src0, src1, -0.0)` so `(+0.0)*(-0.0)` preserves `-0.0`).',
        "notes": 'In `kraid/nir.rs:896-904`, `nir_op_fmul` lowers to `OpFma(a, b, -0.0)`. `a*b + (-0.0)` preserves `-0.0` sign for `(+0)*(-0)`. (`kraid/nir.rs:896-904`, `isa-v9-v14.xml`)',
    },
    'FNEG': {
        "category": 'Floating-Point ALU & Canonicalization',
        "behavior": 'Negates floating-point `src0` by flipping its IEEE-754 sign bit (`dest = -src0`), then applies optional `round`, `clamp`, and `dest_width_narrow` modifiers. Encoded as an alias of `FADD(-src0, -0.0)`.',
        "notes": 'Lowered in `kraid/nir.rs:776-784` as `OpFAdd(-0.0, src0.fneg())`. (`kraid/nir.rs:776-784`, `isa-v9-v14.xml`)',
    },
    'FPCLASS': {
        "category": 'Floating-Point ALU & Canonicalization',
        "behavior": 'Classifies each floating-point element of `src0` (`f32` or `f16`) into an IEEE-754 class bitmask written to `dest`, distinguishing sign, zero, subnormal, normal, infinity, and quiet/signaling NaN.',
        "notes": 'SFU table unit instruction; also used in Bifrost/Valhall transcendental special-case detection. (`bifrost/valhall/ISA.xml:2286-2297`, `bifrost/bi_lower_swizzle.c:182`)',
    },
    'FPOW': {
        "category": 'Floating-Point ALU & Canonicalization',
        "behavior": 'Computes the FP16 power function `dest = pow(src0, src1)` directly on Valhall v12+ for scalar half-precision inputs selected by `lane0`/`lane1`, writing to `dest` per `dest_width_replicate`.',
        "notes": 'Arch v12-v14 only; `f32` pow is lowered via `flog2_32` + `fexp_32_to` (`kraid/nir.rs:943-954`). (`isa-v9-v14.xml`, `kraid/nir.rs:943-954`)',
    },
    'FRCP': {
        "category": 'Floating-Point ALU & Canonicalization',
        "behavior": 'Computes the floating-point reciprocal `dest = 1.0 / src0` (`f32` or `f16`) on the SFU with `< 1 ULP` accuracy. On `f32`, subnormal inputs and subnormal outputs are flushed to signed zero (`±0.0`); on `f16`, subnormals are preserved.',
        "notes": 'Hardware precision is `< 1 ULP` (`hw_tests.rs:1697`). No `round` or `clamp` modifier. (`kraid/ops.rs:1952-1976`, `encode_v9.rs:1581-1598`, `hw_tests.rs:1688-1700`)',
    },
    'FROUND': {
        "category": 'Floating-Point ALU & Canonicalization',
        "behavior": 'Rounds each floating-point element $i \\in \\{0, \\dots, N-1\\}$ of `src0` ($N=1$ `f32`, $N=2$ `v2f16`) to an integral floating-point value using `round` (`round_integer_m`: `none`=RNE, `round_up`=ceil, `round_down`=floor, `round_zero`=trunc, `round_na`=ties-away-from-zero).',
        "notes": '`v2f16` only on arch <= v10 (`hw_tests.rs:1716-1719`). On arch >= v11, `round_up`/`round_down` does NOT auto-FTZ inputs (`kraid/nir.rs:849-868`). (`kraid/ops.rs:2053-2077`, `encode_v9.rs:1651-1669`, `nir.rs:839-876`, `hw_tests.rs:1703-1730`)',
    },
    'FRSQ': {
        "category": 'Floating-Point ALU & Canonicalization',
        "behavior": 'Computes the floating-point reciprocal square root `dest = 1.0 / sqrt(src0)` (`f32` or `f16`) on the SFU with `<= 1 ULP` accuracy. Returns `+Inf` for `+0.0`, `-Inf` for `-0.0`, and `NaN` for negative inputs; on `f32`, subnormal inputs are flushed to zero.',
        "notes": 'Hardware precision is `1 ULP` (`hw_tests.rs:1742`). Negative non-zero -> NaN, `+0.0` -> `+Inf`, `-0.0` -> `-Inf`. (`kraid/ops.rs:2082-2103`, `encode_v9.rs:1671-1688`, `hw_tests.rs:1733-1745`)',
    },
    'FSINCOS': {
        "category": 'Floating-Point ALU & Canonicalization',
        "behavior": '`FSINCOS.f32` (v14+): computes `sin` or `cos` (selected by `operation` modifier: `cos`/`sin`) from range-reduced inputs `src0`, `src1`.',
        "notes": 'Distinct `cos` vs `sin` behavior controlled by `operation` modifier. (`isa-v9-v14.xml`)',
    },
    'FSQRT': {
        "category": 'Floating-Point ALU & Canonicalization',
        "behavior": 'Computes the single-precision IEEE-754 square root `dest = sqrt(src0)` directly on Valhall v14+. Preserves `-0.0 -> -0.0`, returns `+Inf` for `+Inf`, and produces `NaN` for strictly negative inputs.',
        "notes": 'IEEE-754 `sqrt`: negative non-zero -> NaN, `-0.0` -> `-0.0`, `+Inf` -> `+Inf`. (`isa-v9-v14.xml`)',
    },
    'LDEXP': {
        "category": 'Floating-Point ALU & Canonicalization',
        "behavior": 'Scales each floating-point element $i \\in \\{0, \\dots, N-1\\}$ of `src0` by $2^{\\text{src1}[i]}$ ($N=1$ `f32`, $N=2$ `v2f16`; `src1[i]` signed integer), rounding via `round`. Under `.inf`, `exp < -16384` maps `±Inf -> ±1.0` and finite `-> ±0.0`; under `.inf0`, `exp >= +16384` also maps `±0.0 -> ±1.0` and non-zero `-> ±Inf`.',
        "notes": 'Verified in `libGLES_mali.so` `_mali_ldexp_{sf32,inf_sf32,inf0_sf32}` (`0x2e288c0..0x2e28b60`).',
    },
    'FADD_LSCALE': {
        "category": 'SFU Range-Reduction & Helpers',
        "behavior": 'Reduces normal FP32 `src1` into the logarithm mantissa interval $[0.75, 1.5)$ (forcing exponent to 0 so $m \\in [1.0, 2.0)$, then halving $m$ if $|m| \\ge 1.5$; non-normals pass unchanged) and computes `dest = clamp(round_rm(src0 + reduce_log(src1)))`. Paired with `src0 = -1.0` to compute $m - 1$ for `log2`.',
        "notes": 'Used in `flog2_32_to` with `src0 = -1.0` (`builder.rs:410-415`). Verified in `hw_tests.rs:1522-1550`. (`kraid/ops.rs:1308-1354`, `encode_v9.rs:1317-1339`, `builder.rs:409-415`, `hw_tests.rs:1522-1550`)',
    },
    'FADD_RSCALE': {
        "category": 'SFU Range-Reduction & Helpers',
        "behavior": 'Computes `dest = clamp(round_rm((src0 + src2) * 2^clamp(src3, -512, 512)))` in exact FP64 with a single rounding to FP32 (encoded as `FMA_RSCALE` with `src1 = 1.0`). Note that the second addend is `src2` and the signed integer exponent scale is `src3`.',
        "notes": '`libGLES_mali.so` `ModInfo` allows `.special=n_add` only on `FADD_RSCALE` (`0x2e28e80`).',
    },
    'FATAN_ASSIST': {
        "category": 'SFU Range-Reduction & Helpers',
        "behavior": 'First-stage SFU helper for `atan2(y, x)` (`src0 = y`, `src1 = x`): determines the octant and ratio exponent and packs them into `dest` as `(i16 octant/exp, f16 seed)` for subsequent polynomial evaluation and `FATAN_TABLE`.',
        "notes": 'Paired with `FATAN_TABLE.f32`. (`bifrost/valhall/ISA.xml:2490-2498`, `isa-v9-v14.xml`)',
    },
    'FATAN_TABLE': {
        "category": 'SFU Range-Reduction & Helpers',
        "behavior": 'Second-stage SFU lookup table for `atan2`: indexes the 8-entry octant offset table using the low 3 bits of `src0` and writes the FP32 offset constant (`0`, `±pi/2`, or `±pi`) to `dest`.',
        "notes": 'Paired with `FATAN_ASSIST.f32`. (`bifrost/valhall/ISA.xml:2498`, `isa-v9-v14.xml`)',
    },
    'FCOS_TABLE': {
        "category": 'SFU Range-Reduction & Helpers',
        "behavior": '64-entry cosine lookup table for trigonometric range reduction: if `src0` is NaN returns `src0`; if `src0` is Inf returns `NaN`; if `(sign, exp) != (0, 146)` returns `-0.0`; otherwise indexes `idx = bits(src0) & 0x3F` and writes `cos(idx * pi / 32)` to `dest` (with exact `+0.0` at `idx = 16` and `-0.0` at `idx = 48`).',
        "notes": 'Exact hardware match verified in `hw_tests.rs:1553-1567` (with `offset=false`). (`kraid/ops.rs:1526-1584`, `encode_v9.rs:1341-1359`, `nir.rs:1996-2003`, `hw_tests.rs:1553-1567`)',
    },
    'FEXP_TABLE': {
        "category": 'SFU Range-Reduction & Helpers',
        "behavior": '16-entry exponential lookup table indexed by the low 4 bits (`u4`) of `src0`: writes `2^(idx / 16)` in FP32 to `dest` by default, or the low-order correction table entry when `.low` (`adj`) is set.',
        "notes": 'Legacy SFU table helper (`bifrost/valhall/ISA.xml:2346`, `compiler.h:684`). (`bifrost/valhall/ISA.xml:2346`, `bifrost/compiler.h:684`, `isa-v9-v14.xml`)',
    },
    'FLOGD': {
        "category": 'SFU Range-Reduction & Helpers',
        "behavior": 'Logarithm divided-difference helper (`f32`): reduces `src0` to its mantissa $m \\in [0.75, 1.5)$ and writes `dest = log2(m) / (m - 1)` (with the removable singularity at $m = 1$ evaluating to $1 / \\ln 2$). Multiplied by $(m - 1)$ from `FADD_LSCALE` and added to `FREXPE.log` to compute `log2(x)` without cancellation near 1.',
        "notes": 'Paired with `FREXPE.log` + `FADD_LSCALE(-1.0, x)` + `FMA` to compute `log2(x)` to <= 3 ULP (`builder.rs:387-418`, `hw_tests.rs:2348-2393`). (`kraid/ops.rs:1658-1677`, `encode_v9.rs:1421-1438`, `builder.rs:404-417`, `hw_tests.rs:2348-2393`)',
    },
    'FLOG_TABLE': {
        "category": 'SFU Range-Reduction & Helpers',
        "behavior": 'Logarithm lookup table (`f32`): returns table coefficient for `log2`/`ln`/reduced log based on `mode` (`red`/`base2`/`natural`) and `precision` (`none`/`high`/`low`).',
        "notes": 'Used in Bifrost/Valhall legacy log sequences (`bi_opt_mod_props.c:42,61`). (`bifrost/valhall/ISA.xml:2298`, `bifrost/compiler.h:781-782`, `isa-v9-v14.xml`)',
    },
    'FMA_RSCALE': {
        "category": 'SFU Range-Reduction & Helpers',
        "behavior": 'Computes `dest = clamp(round_rm((src0 * src1 + src2) * 2^clamp(src3, -512, 512)))` in exact FP64 with a single rounding to FP32, preventing intermediate overflow or underflow during transcendental range reduction. The `.special` modifier controls zero-times-infinity handling: `.n` forces `0 * anything -> ±0`; `.left` returns `src0` on `0 * Inf`; `.n_add` suppresses `(+Inf) + (-Inf)` NaNs.',
        "notes": 'Only RNE (`none`) and RTZ (`round_zero`) rounding. Verified in `libGLES_mali.so` `_mali_fma_po2_scale_sf32` (`0x2e28c20`) and `kraid/ops.rs:1828-1857`.',
    },
    'FMUL_CSLICE': {
        "category": 'SFU Range-Reduction & Helpers',
        "behavior": 'Extended-precision cosine range-reduction slice multiply helper (`f32`): multiplies `src0` (supporting `lane0`, `.abs`, `.neg`) against the Payne-Hanek slice constant selected by `src1` and writes the partial product to `dest`.',
        "notes": 'SFU helper opcode; not emitted by kraid. (`isa-v9-v14.xml`)',
    },
    'FMUL_RSCALE': {
        "category": 'SFU Range-Reduction & Helpers',
        "behavior": 'Computes `dest = clamp(round_rm((src0 * src1) * 2^clamp(src3, -512, 512)))` in exact FP64 with a single rounding to FP32 (encoded as `FMA_RSCALE` with `src2 = -0.0`). The `.special` modifier selects `.n` (`0 * anything -> ±0`) or `.left` (`0 * Inf -> src0` unscaled).',
        "notes": '`libGLES_mali.so` `ModInfo` allows `n`/`left` only (forbids `n_add`; `_mali_mul_left_po2_scale_sf32` `0x2e29550`).',
    },
    'FMUL_SLICE': {
        "category": 'SFU Range-Reduction & Helpers',
        "behavior": 'Trigonometric range-reduction multiply helper (`f32`): computes sliced product for Payne-Hanek / Cody-Waite `sin`/`cos` reduction.',
        "notes": 'SFU helper opcode; not emitted by kraid. (`isa-v9-v14.xml`, `bifrost/bifrost/ISA.xml:1290`)',
    },
    'FPOW_SC_APPLY': {
        "category": 'SFU Range-Reduction & Helpers',
        "behavior": 'Power special-case applicator (`f32`): inspects the special-case tag word in `src1` (produced by `FPOW_SC_DET`); if a special-case override is flagged, writes the decoded IEEE-754 override value to `dest`, otherwise passes `src0` through unchanged.',
        "notes": 'Paired with `FPOW_SC_DET`. (`bifrost/bifrost/ISA.xml:4634`, `isa-v9-v14.xml`)',
    },
    'FPOW_SC_DET': {
        "category": 'SFU Range-Reduction & Helpers',
        "behavior": 'Power special-case detector (`f32`): inspects base `src0` and exponent `src1` for edge cases (`0`, `±1`, `±Inf`, `NaN`, and negative bases) according to `func` (`.pow` vs `.powr`) and writes a packed override tag to `dest` for `FPOW_SC_APPLY`.',
        "notes": 'Paired with `FPOW_SC_APPLY` (`bifrost/compiler.h:686`). (`bifrost/compiler.h:686`, `bifrost/bifrost/ISA.xml:4689`, `isa-v9-v14.xml`)',
    },
    'FRCBRT_APPROX_A': {
        "category": 'SFU Range-Reduction & Helpers',
        "behavior": 'Stage-A reciprocal cube-root (`x^(-1/3)`) SFU helper (`f32`): computes the initial low-precision seed approximation of `src0^(-1/3)` in `dest` for the multi-stage cube-root sequence.',
        "notes": 'SFU table seed instruction (`bifrost/ISA.xml:4700`). (`bifrost/bifrost/ISA.xml:4700`, `isa-v9-v14.xml`)',
    },
    'FRCBRT_APPROX_B': {
        "category": 'SFU Range-Reduction & Helpers',
        "behavior": 'Stage-B reciprocal cube-root (`x^(-1/3)`) SFU helper (`f32`): applies the second-stage polynomial refinement to the intermediate value `src0` and writes the refined estimate to `dest`.',
        "notes": 'SFU table seed instruction (`bifrost/ISA.xml:4722`). (`bifrost/bifrost/ISA.xml:4722`, `isa-v9-v14.xml`)',
    },
    'FRCBRT_APPROX_C': {
        "category": 'SFU Range-Reduction & Helpers',
        "behavior": 'Stage-C reciprocal cube-root (`x^(-1/3)`) SFU helper (`f32`): applies the final refinement step to `src0` and writes the resulting `x^(-1/3)` approximation to `dest`.',
        "notes": 'SFU table seed instruction (`bifrost/ISA.xml:4726`). (`bifrost/bifrost/ISA.xml:4726`, `isa-v9-v14.xml`)',
    },
    'FRCP_APPROX': {
        "category": 'SFU Range-Reduction & Helpers',
        "behavior": 'Low-precision reciprocal seed (`f32`): computes an initial ~12-bit approximation of `1.0 / src0` in `dest`, intended as the starting estimate for Newton-Raphson refinement.',
        "notes": 'SFU table seed instruction. (`bifrost/valhall/ISA.xml`, `isa-v9-v14.xml`)',
    },
    'FREXPE': {
        "category": 'SFU Range-Reduction & Helpers',
        "behavior": 'Extracts the signed integer exponent $E$ of each floating-point element $i \\in \\{0, \\dots, N-1\\}$ of `src0` ($N=1$ `f32`, $N=2$ `v2f16`), returning `0` for `±0`, `±Inf`, and `NaN`: `none` gives `e - 126` (`[0.5, 1.0)`), `sqrt` gives `(e - 125) >> 1` (`[0.25, 1.0)`), and `log` gives `(e - 127) + top_mant_bit` (`[0.75, 1.5)`). Negates the result when `neg_result=neg`.',
        "notes": 'Bit-exact verified in `libGLES_mali.so` `_mali_frexpe_*` (`0x2e291d0..0x2e293c8`). `v2f16` arch v9--v10 only.',
    },
    'FREXPM': {
        "category": 'SFU Range-Reduction & Helpers',
        "behavior": 'Extracts the normalized significand $M$ of each floating-point element $i \\in \\{0, \\dots, N-1\\}$ of `src0` ($N=1$ `f32`, $N=2$ `v2f16`): passes `±0` unchanged and quietens `NaN`; maps finite values into `[0.5, 1.0)` (`none`), `[0.25, 1.0)` (`sqrt`), or `[0.75, 1.5)` (`log`). On `±Inf`, `none`/`sqrt` return `±Inf` whereas `log` returns `±1.0` (`x & 0xbf800000`).',
        "notes": 'Bit-exact verified in `libGLES_mali.so` `_mali_frexpm_*` (`0x2e28fb0..0x2e2914c`). `v2f16` arch v9--v10 only.',
    },
    'FRSQ_APPROX': {
        "category": 'SFU Range-Reduction & Helpers',
        "behavior": 'Low-precision reciprocal square-root seed (`f32`): computes an initial ~12-bit approximation of `1.0 / sqrt(src0)` in `dest`, intended as the starting estimate for Newton-Raphson refinement.',
        "notes": 'SFU table seed instruction. (`bifrost/valhall/ISA.xml`, `isa-v9-v14.xml`)',
    },
    'FSINCOS_OFFSET': {
        "category": 'SFU Range-Reduction & Helpers',
        "behavior": 'Trigonometric angle-offset lookup table (`u6`): indexes the 8-entry quadrant offset table using the low bits of `src0` and writes the FP32 offset to `dest` (scaled by $2^{24}$ when `.scale` is set).',
        "notes": 'SFU table helper (`bifrost/valhall/ISA.xml:2340`, `compiler.h:691`). (`bifrost/valhall/ISA.xml:2340`, `bifrost/compiler.h:691`, `isa-v9-v14.xml`)',
    },
    'FSIN_TABLE': {
        "category": 'SFU Range-Reduction & Helpers',
        "behavior": '64-entry sine lookup table for trigonometric range reduction: if `src0` is NaN returns `src0`; if `src0` is Inf returns `NaN`; if `(sign, exp) != (0, 146)` returns `-0.0`; otherwise indexes `idx = bits(src0) & 0x3F` and writes `sin(idx * pi / 32)` to `dest` (with exact `-0.0` at `idx = 0` and `+0.0` at `idx = 32`).',
        "notes": 'Exact hardware match verified in `hw_tests.rs:1748-1762` (with `offset=false`). (`kraid/ops.rs:2107-2147`, `encode_v9.rs:1690-1708`, `nir.rs:1988-1995`, `hw_tests.rs:1748-1762`)',
    },
    'ILOGB': {
        "category": 'SFU Range-Reduction & Helpers',
        "behavior": 'Extracts the unbiased base-2 integer exponent `floor(log2(|src0[i]|))` for each floating-point element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ `f32 -> s32`, $N=2$ `v2f16 -> v2s16`; arch <= v10).',
        "notes": 'IEEE-754 `ilogb` special cases: `0 -> FP_ILOGB0`, `Inf -> INT_MAX`, `NaN -> FP_ILOGBNAN`. (`bifrost/bifrost/ISA.xml:5778-5788`, `isa-v9-v14.xml`)',
    },
    'LOGB': {
        "category": 'SFU Range-Reduction & Helpers',
        "behavior": 'Extracts the unbiased base-2 floating-point exponent `floor(log2(|src0[i]|))` for each floating-point element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ `f32`, $N=2$ `v2f16`; arch <= v10).',
        "notes": 'IEEE-754 `logb` special cases: `+-0 -> -Inf`, `+-Inf -> +Inf`, `NaN -> NaN`. (`bifrost/bifrost/ISA.xml:7386-7396`, `isa-v9-v14.xml`)',
    },
    'CUBEFACE1': {
        "category": 'Cubemap Helpers',
        "behavior": 'Computes the cubemap major-axis magnitude from 3D direction `(x=src0, y=src1, z=src2)`: `m = max(|x|, |y|, |z|)`. Writes `NaN` if any input is NaN, `1.0` if `m` is infinite, `2.0` if `m` is zero or subnormal (preventing divide-by-zero), and `m` otherwise to `dest`.',
        "notes": 'Verified in `hw_tests.rs:1443-1450` (`Ulp(1)`). (`kraid/ops.rs:948-994`, `encode_v9.rs:1074-1094`, `nir.rs:1958-1971`, `hw_tests.rs:1443-1450`)',
    },
    'CUBEFACE2': {
        "category": 'Cubemap Helpers',
        "behavior": 'Determines the dominant cubemap face index (`0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z`) from 3D direction `(x=src0, y=src1, z=src2)` with tie-breaking priority $|z| \\ge |y| \\ge |x|$ (NaN defaults to face `4`). Packs the 3-bit face index into bits `[31:29]` and sets bit `16` if any finite/infinite component is `±Inf` (and no component is NaN): `dest = (face << 29) | (is_inf << 16)`.',
        "notes": 'Tie-breaking priority: `|z| >= |y| >= |x|`. Exact match verified in `hw_tests.rs:1433-1440`. (`kraid/ops.rs:898-944`, `encode_v9.rs:1052-1072`, `nir.rs:1958-1971`, `hw_tests.rs:1433-1440`)',
    },
    'CUBE_SSEL': {
        "category": 'Cubemap Helpers',
        "behavior": 'Selects signed cubemap S coordinate from `(src0=Z, src1=X, src2=CUBEFACE2)`: `face 0 -> -Z`, `1 -> +Z`, `2/3/4 -> +X`, `5 -> -X`; when `is_inf` (bit 16 of `src2`) is set, normalizes `±Inf -> ±1.0` and finite `-> ±0.0`.',
        "notes": 'Verified in `libGLES_mali.so` (`0x1ecbc3c`) and `malioc` G710.',
    },
    'CUBE_TSEL': {
        "category": 'Cubemap Helpers',
        "behavior": 'Selects signed cubemap T coordinate from `(src0=Y, src1=Z, src2=CUBEFACE2)`: `face 2 -> +Z`, `3 -> -Z`, `0/1/4/5 -> -Y`; when `is_inf` (bit 16 of `src2`) is set, normalizes `±Inf -> ±1.0` and finite `-> ±0.0`.',
        "notes": 'Verified in `libGLES_mali.so` (`0x1ecbc98`) and `malioc` G710.',
    },
    'CSEL': {
        "category": 'Compare, Select & Mux',
        "behavior": 'Compares `src0[i]` and `src1[i]` using predicate `cmpf` independently for each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `f32`/`s32`/`u32`, $N=2$ for `v2f16`/`v2s16`/`v2u16`) and writes `src2[i]` to `dest[i]` if true, or `src3[i]` if false.',
        "notes": '`src2`/`src3` are raw bit payloads (`VNIN`, `ops.rs:859`). Exact match verified in `hw_tests.rs:1391-1430`. (`kraid/ops.rs:851-894`, `encode_v9.rs:1026-1050`, `nir.rs:1222-1239`, `hw_tests.rs:1391-1430`)',
    },
    'FCMP': {
        "category": 'Compare, Select & Mux',
        "behavior": 'Compares floating-point `src0[i]` and `src1[i]` using predicate `cmpf` independently for each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ `f32`, $N=2$ `v2f16`) and encodes the boolean into `dest[i]` via `result_type` (`i1` $\\to$ `1`, `m1` $\\to$ all-ones, `f1` $\\to$ `1.0`, false $\\to$ `0`).',
        "notes": '`ne` is unordered (`True` on NaN); `gtlt` is ordered not-equal (`False` on NaN); `total` is IEEE-754 `total_cmp(a, b) <= 0` (`ops.rs:1489-1521`). (`kraid/ops.rs:741-790, 1386-1522`, `encode_v9.rs:933-1024`, `hw_tests.rs:1318-1388`)',
    },
    'FCMP_AND': {
        "category": 'Compare, Select & Mux',
        "behavior": 'Compares floating-point `src0[i]` and `src1[i]` via `cmpf` for each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ `f32`, $N=2$ `v2f16`), combines the predicate with `(src2[i] != 0)` using logical AND, and formats the boolean into `dest[i]` according to `result_type`.',
        "notes": '`src2[i]` tested as `!= 0` per lane before formatting via `result_type` (`ops.rs:785-786`). (`kraid/ops.rs:741-790`, `encode_v9.rs:974-982`, `opt_fuse.rs:188-238`, `hw_tests.rs:1318-1388`)',
    },
    'FCMP_OR': {
        "category": 'Compare, Select & Mux',
        "behavior": 'Compares floating-point `src0[i]` and `src1[i]` via `cmpf` for each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ `f32`, $N=2$ `v2f16`), combines the predicate with `(src2[i] != 0)` using logical OR, and formats the boolean into `dest[i]` according to `result_type`.',
        "notes": '`src2[i]` tested as `!= 0` per lane before formatting via `result_type`. (`kraid/ops.rs:741-790`, `encode_v9.rs:983-991`, `opt_fuse.rs:188-238`, `hw_tests.rs:1318-1388`)',
    },
    'ICMP': {
        "category": 'Compare, Select & Mux',
        "behavior": 'Compares Stage A widened/swizzled integer lanes `src0[i]` and `src1[i]` via `cmpf` independently for each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `s32`/`u32`, $N=2$ for `v2s16`/`v2u16`, $N=4$ for `v4s8`/`v4u8` on v9--v10) and encodes the boolean into `dest[i]` via `result_type`.',
        "notes": '`v4s8`/`v4u8` only arch <= v10 (`hw_tests.rs:1351`). `gtlt`/`total` invalid for ints. (`kraid/ops.rs:741-790, 1463-1522`, `encode_v9.rs:994-1002`, `hw_tests.rs:1318-1388`)',
    },
    'ICMP_AND': {
        "category": 'Compare, Select & Mux',
        "behavior": 'For each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `s32`/`u32`, $N=2$ for `v2s16`/`v2u16`, $N=4$ for `v4s8`/`v4u8` on v9--v10), compares Stage A widened/swizzled `src0[i]` and `src1[i]` using predicate `cmpf`, combines the boolean with `(src2[i] != 0)` via logical AND, and formats the result into `dest[i]` according to `result_type` (`i1` $\\to$ `1`, `m1` $\\to$ all-ones, `f1` $\\to$ `1.0`, false $\\to$ `0`).',
        "notes": '`v4s8`/`v4u8` only arch <= v10. (`kraid/ops.rs:741-790`, `encode_v9.rs:1003-1011`, `hw_tests.rs:1318-1388`)',
    },
    'ICMP_MULTI': {
        "category": 'Compare, Select & Mux',
        "behavior": 'Multi-word chained integer compare (`s32`/`u32`): `ord = cmp(src0, src1).then((src2 as i32).cmp(&0))`; if `result_type == c` -> `ord as i8 as u32` (`-1=0xFFFFFFFF`, `0`, `1`), else `res_type(cmpf(ord))`.',
        "notes": '`ICMP_MULTI.s32.c` does NOT exist (`kraid/nir.rs:1192`, `hw_tests.rs:1928`); used for 64-bit comparisons (`nir.rs:1186-1209`). (`kraid/ops.rs:2303-2349`, `encode_v9.rs:1795-1818`, `nir.rs:1186-1209`, `hw_tests.rs:1905-1949`)',
    },
    'ICMP_OR': {
        "category": 'Compare, Select & Mux',
        "behavior": 'For each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ for `s32`/`u32`, $N=2$ for `v2s16`/`v2u16`, $N=4$ for `v4s8`/`v4u8` on v9--v10), compares Stage A widened/swizzled `src0[i]` and `src1[i]` using predicate `cmpf`, combines the boolean with `(src2[i] != 0)` via logical OR, and formats the result into `dest[i]` according to `result_type` (`i1` $\\to$ `1`, `m1` $\\to$ all-ones, `f1` $\\to$ `1.0`, false $\\to$ `0`).',
        "notes": '`v4s8`/`v4u8` only arch <= v10. (`kraid/ops.rs:741-790`, `encode_v9.rs:1012-1020`, `hw_tests.rs:1318-1388`)',
    },
    'MUX': {
        "category": 'Compare, Select & Mux',
        "behavior": 'Selects between `src0[i]` and `src1[i]` using selector `src2[i]` independently for each SIMD element $i \\in \\{0, \\dots, N-1\\}$ ($N=1$ `i32`, $N=2$ `v2i16`, $N=4$ `v4i8`): `.neg` selects `src0[i]` if the sign bit of `src2[i]` is 1; `.int_zero` (default) if `src2[i] == 0`; `.fp_zero` (`i32` only) if `f32(src2) == ±0.0`; `.bit` blends bitwise `(src0[i] & src2[i]) | (src1[i] & ~src2[i])`.',
        "notes": '`fp_zero` is `i32`-only (`ops.rs:3639`, `hw_tests.rs:2121`) and treats both `+0.0` and `-0.0` as true. Exact match verified in `hw_tests.rs:2113-2135`. (`kraid/ops.rs:3573-3647`, `encode_v9.rs:2858-2885`, `opt_fuse.rs:150-180`, `hw_tests.rs:2113-2135`)',
    },
    'F16_TO_F32': {
        "category": 'Conversions',
        "behavior": 'Widens the FP16 halfword selected by `lane0` (`.h0` or `.h1`) of `src0` losslessly to IEEE-754 single-precision FP32 and writes the result to `dest` (exact conversion with no rounding or clamping).',
        "notes": 'Exact IEEE-754 f16->f32 widening (`hw_tests.rs:1453-1459`). (`kraid/ops.rs:1064-1088`, `encode_v9.rs:1150-1166`, `hw_tests.rs:1453-1459`)',
    },
    'F16_TO_S32': {
        "category": 'Conversions',
        "behavior": 'Saturating f16->s32 (arch <= v10): `x = round_rm(f16(src0.lane0))`; NaN->0, `-Inf -> INT32_MIN`, `+Inf -> INT32_MAX`, else `x as i32`.',
        "notes": 'Arch v9-v10 only (`kraid/ops.rs:1196`, `nir.rs:727`). (`kraid/ops.rs:1196-1221`, `encode_v9.rs:1168-1198`, `nir.rs:726-734`)',
    },
    'F16_TO_U32': {
        "category": 'Conversions',
        "behavior": 'Converts the FP16 halfword selected by `lane0` of `src0` to a saturating unsigned 32-bit integer in `dest` (arch <= v10): rounds per `round`, returning `0` for `NaN` or `<= 0.0`, `UINT32_MAX` for `+Inf`, and `uint32(x)` otherwise.',
        "notes": 'Arch v9-v10 only (`kraid/ops.rs:1196`, `nir.rs:727`). (`kraid/ops.rs:1196-1221`, `encode_v9.rs:1168-1198`, `nir.rs:726-734`)',
    },
    'F32_TO_F16': {
        "category": 'Conversions',
        "behavior": 'Narrows FP32 `src0` to IEEE-754 FP16 after applying optional `clamp` and `round` modifiers, and writes the 16-bit result into `dest.h0`, `dest.h1` (preserving the other halfword), or replicated into both halves (`dest.h01`) per `dest_width_narrow` (v11--v14 alias of `FADD.f32(src0, -0.0)`).',
        "notes": 'Arch v11-v14 only (v9-v10 uses `V2F32_TO_V2F16` or `FADD.f32.hf0/hf1`). Single source (`src1` fixed to `-0.0`). (`kraid/builder.rs:50-65`, `nir.rs:642-663`, `isa-v9-v14.xml`)',
    },
    'F32_TO_S32': {
        "category": 'Conversions',
        "behavior": 'Converts FP32 `src0` to a saturating signed 32-bit integer in `dest`: rounds per `round` (default RNE; `.round_zero` for truncation), returning `0` on `NaN`, `INT32_MIN` for values `<= -2^31`, `INT32_MAX` for values `>= 2^31 - 1`, and `int32(x)` otherwise.',
        "notes": 'Matches Rust `f32 as i32` saturation (`ops.rs:1249-1262`, `hw_tests.rs:1462-1483`). (`kraid/ops.rs:1226-1262`, `encode_v9.rs:1212-1246`, `hw_tests.rs:1462-1483`)',
    },
    'F32_TO_U32': {
        "category": 'Conversions',
        "behavior": 'Converts FP32 `src0` to a saturating unsigned 32-bit integer in `dest`: rounds per `round` (default RNE; `.round_zero` for truncation), returning `0` on `NaN` or values `<= 0.0`, `UINT32_MAX` for values `>= 2^32 - 1`, and `uint32(x)` otherwise.',
        "notes": 'Matches Rust `f32 as u32` saturation (`ops.rs:1249-1262`, `hw_tests.rs:1462-1483`). (`kraid/ops.rs:1226-1262`, `encode_v9.rs:1212-1246`, `hw_tests.rs:1462-1483`)',
    },
    'S16_TO_F32': {
        "category": 'Conversions',
        "behavior": 'Converts the signed 16-bit integer selected by `lane0` (`.h0` or `.h1`) of `src0` exactly to IEEE-754 FP32 and writes the result to `dest` (arch <= v10).',
        "notes": 'Arch v9-v10 only (`kraid/nir.rs:740`); on v11+ use `S32_TO_F32` with `widen0=h0`. (`kraid/ops.rs:2575-2600`, `encode_v9.rs:1992-1995`, `nir.rs:738-749`)',
    },
    'S16_TO_S32': {
        "category": 'Conversions',
        "behavior": 'Extracts the signed 16-bit halfword selected by `lane0` (`.h0` or `.h1`) of `src0`, sign-extends it to 32 bits, and writes the result to `dest`.',
        "notes": 'Exact sign extension. (`isa-v9-v14.xml`, `bifrost/valhall/ISA.xml`)',
    },
    'S32_TO_F32': {
        "category": 'Conversions',
        "behavior": 'Converts signed 32-bit integer `src0` (with optional Stage A widening) to IEEE-754 FP32 in `dest`, applying the selected `round` mode (default RNE; no clamp modifier).',
        "notes": 'Supports all 5 rounding modes (`round_extended_m`). (`kraid/ops.rs:2575-2600`, `encode_v9.rs:1960-2010`, `nir.rs:738-749`)',
    },
    'S8_TO_F32': {
        "category": 'Conversions',
        "behavior": 'Converts the signed 8-bit byte selected by `lane0` (`.b0`..`.b3`) of `src0` exactly to IEEE-754 FP32 and writes the result to `dest` (arch <= v10).',
        "notes": 'Arch v9-v10 only (`kraid/nir.rs:740`). (`kraid/ops.rs:2575-2600`, `encode_v9.rs:1984-1987`, `nir.rs:738-749`)',
    },
    'S8_TO_S32': {
        "category": 'Conversions',
        "behavior": 'Extracts the signed 8-bit byte selected by `lane0` (`.b0`..`.b3`) of `src0`, sign-extends it to 32 bits, and writes the result to `dest`.',
        "notes": 'Exact sign extension. (`isa-v9-v14.xml`, `bifrost/valhall/ISA.xml`)',
    },
    'U16_TO_F32': {
        "category": 'Conversions',
        "behavior": 'Converts the unsigned 16-bit integer selected by `lane0` (`.h0` or `.h1`) of `src0` exactly to IEEE-754 FP32 and writes the result to `dest` (arch <= v10).',
        "notes": 'Arch v9-v10 only (`kraid/nir.rs:692`); on v11+ use `U32_TO_F32` with `widen0=h0`. (`kraid/ops.rs:2575-2600`, `encode_v9.rs:1996-1999`, `nir.rs:690-701`)',
    },
    'U16_TO_U32': {
        "category": 'Conversions',
        "behavior": 'Extracts the unsigned 16-bit halfword selected by `lane0` (`.h0` or `.h1`) of `src0`, zero-extends it to 32 bits, and writes the result to `dest`.',
        "notes": 'Exact zero extension. (`isa-v9-v14.xml`, `bifrost/valhall/ISA.xml`)',
    },
    'U32_TO_F32': {
        "category": 'Conversions',
        "behavior": 'Converts unsigned 32-bit integer `src0` (with optional Stage A widening) to IEEE-754 FP32 in `dest`, applying the selected `round` mode (default RNE; no clamp modifier).',
        "notes": 'Supports all 5 rounding modes (`round_extended_m`). (`kraid/ops.rs:2575-2600`, `encode_v9.rs:1960-2010`, `nir.rs:690-701`)',
    },
    'U8_TO_F32': {
        "category": 'Conversions',
        "behavior": 'Converts the unsigned 8-bit byte selected by `lane0` (`.b0`..`.b3`) of `src0` exactly to IEEE-754 FP32 and writes the result to `dest` (arch <= v10).',
        "notes": 'Arch v9-v10 only (`kraid/nir.rs:692`). (`kraid/ops.rs:2575-2600`, `encode_v9.rs:1988-1991`, `nir.rs:690-701`)',
    },
    'U8_TO_U32': {
        "category": 'Conversions',
        "behavior": 'Extracts the unsigned 8-bit byte selected by `lane0` (`.b0`..`.b3`) of `src0`, zero-extends it to 32 bits, and writes the result to `dest`.',
        "notes": 'Exact zero extension. (`isa-v9-v14.xml`, `bifrost/valhall/ISA.xml`)',
    },
    'V2F16_TO_V2S16': {
        "category": 'Conversions',
        "behavior": 'Converts each half-precision lane $i \\in \\{0, 1\\}$ of swizzled `src0` to a saturating signed 16-bit integer `dest[i] = sat_i16(round_rm(src0[i]))` (NaN maps to `0`; arch <= v10).',
        "notes": 'Arch v9-v10 only. (`isa-v9-v14.xml`, `bifrost/valhall/ISA.xml`)',
    },
    'V2F16_TO_V2U16': {
        "category": 'Conversions',
        "behavior": 'Converts each half-precision lane $i \\in \\{0, 1\\}$ of swizzled `src0` to a saturating unsigned 16-bit integer `dest[i] = sat_u16(round_rm(src0[i]))` (NaN or `<= 0` maps to `0`; arch <= v10).',
        "notes": 'Arch v9-v10 only. (`isa-v9-v14.xml`, `bifrost/valhall/ISA.xml`)',
    },
    'V2F32_TO_V2F16': {
        "category": 'Conversions',
        "behavior": 'Two-source FP32-to-FP16 packing conversion (v9--v10): clamps and rounds `src0` to FP16 into `dest.h0` and clamps and rounds `src1` to FP16 into `dest.h1` in a single instruction.',
        "notes": 'Arch v9-v10 only (`kraid/nir.rs:653, 676`). Clamp is applied BEFORE f32->f16 rounding (`ops.rs:4629-4638`). (`kraid/ops.rs:4597-4645`, `encode_v9.rs:3462-3482`, `nir.rs:652-689`)',
    },
    'V2S16_TO_V2F16': {
        "category": 'Conversions',
        "behavior": 'Converts each signed 16-bit integer lane $i \\in \\{0, 1\\}$ of swizzled `src0` to IEEE-754 FP16 `dest[i] = round_rm(f16(src0[i]))` (arch <= v10).',
        "notes": 'Arch v9-v10 only (`kraid/ops.rs:2542`). (`kraid/ops.rs:2542-2570`, `encode_v9.rs:1945-1949`, `nir.rs:750-765`)',
    },
    'V2S8_TO_V2F16': {
        "category": 'Conversions',
        "behavior": 'Extracts two signed 8-bit bytes selected by `swz0` (`bXY`, default `b02`) from `src0` and converts them exactly to packed FP16 lanes `dest.h0 = f16(s8(src0.b[X]))` and `dest.h1 = f16(s8(src0.b[Y]))` (arch <= v10).',
        "notes": 'Arch v9-v10 only (`kraid/ops.rs:2542`, `encode_v9.rs:1933`). (`kraid/ops.rs:2542-2570`, `encode_v9.rs:1937-1940`)',
    },
    'V2S8_TO_V2S16': {
        "category": 'Conversions',
        "behavior": 'Selects two signed 8-bit bytes from `src0` via `swz0` (default `b02`: bytes 0 and 2), sign-extends each to 16 bits, and writes the packed `v2s16` pair to `dest`.',
        "notes": 'Exact per-lane sign extension. (`isa-v9-v14.xml`, `bifrost/valhall/ISA.xml`)',
    },
    'V2U16_TO_V2F16': {
        "category": 'Conversions',
        "behavior": 'Converts each unsigned 16-bit integer lane $i \\in \\{0, 1\\}$ of swizzled `src0` to IEEE-754 FP16 `dest[i] = round_rm(f16(src0[i]))` (arch <= v10).',
        "notes": 'Arch v9-v10 only; values `>= 65520` overflow to `+Inf` in RNE. (`kraid/ops.rs:2542-2570`, `encode_v9.rs:1950-1954`, `nir.rs:750-765`)',
    },
    'V2U8_TO_V2F16': {
        "category": 'Conversions',
        "behavior": 'Extracts two unsigned 8-bit bytes selected by `swz0` (`bXY`, default `b02`) from `src0` and converts them exactly to packed FP16 lanes `dest.h0 = f16(u8(src0.b[X]))` and `dest.h1 = f16(u8(src0.b[Y]))` (arch <= v10).',
        "notes": 'Arch v9-v10 only. (`kraid/ops.rs:2542-2570`, `encode_v9.rs:1941-1944`)',
    },
    'V2U8_TO_V2U16': {
        "category": 'Conversions',
        "behavior": 'Selects two unsigned 8-bit bytes from `src0` via `swz0` (default `b02`: bytes 0 and 2), zero-extends each to 16 bits, and writes the packed `v2u16` pair to `dest` (respecting `dest_width`).',
        "notes": 'Exact per-lane zero extension. (`isa-v9-v14.xml`, `bifrost/valhall/ISA.xml`)',
    },
    'ADR': {
        "category": 'Move, Pack & Swizzle',
        "behavior": 'Computes a 32-bit PC-relative shader code address by adding the signed inline offset `#imm1w` to the address of the following instruction (`PC + 8`) and writes `(PC + 8 + #imm1w) & 0xFFFFFFFF` to `dest`.',
        "notes": 'Destination is **32-bit** (`#[dst_type(I32)]` in `kraid/ops.rs:127`); `imm1w` is byte offset from `PC + 8` (`encode_v9.rs:162-164`). (`kraid/ops.rs:126-141`, `encode_v9.rs:162-164, 582-593`)',
    },
    'MKVEC': {
        "category": 'Move, Pack & Swizzle',
        "behavior": '`MKVEC.v2i16`: `Concat(src1.lane1, src0.lane0)`. `MKVEC.v2i8`: `Concat(src2[15:0], src1.lane1, src0.lane0)` (bits `[35:24]` reserved zero; virtual `lanes2=b01` in XML and fixed `b01` in `libGLES_mali.so` `ModInfo`).',
        "notes": 'All 3 workstreams reconciled (`isa-v9-v14.xml`, `libGLES_mali.so` `ModInfo`, 313 ASTC/ETC2/BC6 `malioc` emissions).',
    },
    'MOV': {
        "category": 'Move, Pack & Swizzle',
        "behavior": 'Copies the 32-bit value `src0` (GPR, FAU uniform, or inline constant) into `dest`, optionally masking the write to `.h0` or `.h1` via `dest_width`.',
        "notes": 'No swizzle or source modifiers (`encode_v9.rs:2821-2856`). (`kraid/ops.rs:3555-3569`, `encode_v9.rs:2821-2856`)',
    },
    'MOV_IMM': {
        "category": 'Move, Pack & Swizzle',
        "behavior": 'Loads the inline 32-bit immediate `#imm1w` into `dest` (interpreted as `i32`, `v2i16`, or `v4i8`), optionally masking the write to `.h0` or `.h1` via `dest_width`.',
        "notes": 'Alias of `IADD_IMM` with `src0 = 0`. (`kraid/encode_v9.rs:2842-2847`, `isa-v9-v14.xml`)',
    },
    'SWZ': {
        "category": 'Move, Pack & Swizzle',
        "behavior": 'Permutes packed 16-bit (`v2i16`) or 8-bit (`v4i8`, arch <= v10) lanes of `src0` according to `lanes0` and writes the swizzled vector to `dest`. Encoded as an alias of `IADD` with `src1 = 0`.',
        "notes": 'In kraid, `OpSwz` is a virtual op lowered via copy/`IADD`/`ShiftLop`/`ARShift`/`MKVEC` (`lower_mkvec_swz.rs`). (`kraid/ops.rs:4159-4204`, `lower_mkvec_swz.rs:137-499`, `hw_tests.rs:2239-2289`)',
    },
    'CLPER': {
        "category": 'Subgroup & Cross-Lane',
        "behavior": 'Cross-lane permute: reads `src0` from target lane computed from `lane_id` and `src1` (8-bit via `lanes1`) using `lane_op` (`none`/`xor`/`accumulate`/`shift`/`rotate`/`low`/`low_alt`/`prefix`) within `subgroup`; returns `inactive_result` constant if target lane is inactive/OOB.',
        "notes": 'Used for `ddx`/`ddy` (`builder.rs:263-288`) and subgroup shuffles/reductions. (`kraid/ops.rs:537-690`, `encode_v9.rs:830-900`, `builder.rs:260-289`, `bifrost/valhall/ISA.xml:23`)',
    },
    'CLPER_OR': {
        "category": 'Subgroup & Cross-Lane',
        "behavior": 'Cross-lane permute + bitwise OR (v14+): permutes `src0` across `subgroup` using `lane_select` (`lanes04`..`lanesBF`) and `src1` (`lanes1`), then ORs with `src2`.',
        "notes": 'Has `lane_select` (NOT `lane_op` or `inactive_result`). (`isa-v9-v14.xml`)',
    },
    'WMASK': {
        "category": 'Subgroup & Cross-Lane',
        "behavior": 'Computes a subgroup ballot bitmask over active lanes within the hardware subgroup (`subgroup2`, `subgroup4`, `subgroup8`, or `subgroup16`): bit $k$ of `dest` is `1` if lane $k$ is active and its `src0 != 0`, and `0` otherwise.',
        "notes": 'Cross-lane subgroup operation; NOT an identity move! Implements `nir_intrinsic_ballot`. (`kraid/ops.rs:5087-5103`, `encode_v9.rs:3831-3848`, `nir.rs:1885-1893`, `bifrost/valhall/ISA.xml:2172-2188`)',
    },
    'BARRIER': {
        "category": 'Control Flow, Barriers & System',
        "behavior": 'Synchronizes execution and workgroup-shared (WLS) memory visibility across all warps in the current compute workgroup; advances the hardware barrier epoch before any warp proceeds past the barrier.',
        "notes": 'Implements `nir_intrinsic_barrier` with `SCOPE_WORKGROUP` (`kraid/nir.rs:1898-1904`). (`kraid/ops.rs:317-327`, `encode_v9.rs:738-746`, `nir.rs:1894-1906`, `bifrost/valhall/ISA.xml:13`)',
    },
    'BRANCH': {
        "category": 'Control Flow, Barriers & System',
        "behavior": 'Evaluates `base = combine(src0)` (`none`: `!=0`, `h0`: `lo16!=0`, `h1`: `hi16!=0`, `and`: `==0xFFFFFFFF`, `lowbits`: `bits[20:3] not in {0, 0x3FFFF}`); `taken = not ? !base : base`; `pc = taken ? pc + 8 + offset : pc + 8`.',
        "notes": 'Unconditional branch is encoded as `BRANCH.not 0, target` (`kraid/ops.rs:513-515`). `not` is a boolean branch-condition negation, NOT a bitwise NOT on `src0`! (`kraid/ops.rs:476-533`, `encode_v9.rs:772-792`, `opt_fuse.rs:240-290`, `bifrost/valhall/ISA.xml:19`)',
    },
    'DISCARD': {
        "category": 'Control Flow, Barriers & System',
        "behavior": 'Fragment discard: evaluates `FCMP` condition `cmpf(src0, src1)` (both `f32`, with `widen0/1`, `abs0/1`, `neg0/1`); if true, discards fragment (`discarded = true`) and terminates thread.',
        "notes": 'Takes **2 FP32 sources** and a **`cmpf`** modifier (`ops.rs:1039-1060`, `encode_v9.rs:1129-1148`, `opt_fuse.rs:98-142`). Only valid in fragment shaders. (`kraid/ops.rs:1039-1060`, `encode_v9.rs:1129-1148`, `opt_fuse.rs:98-142`, `bifrost/valhall/ISA.xml:37`)',
    },
    'DOORBELL': {
        "category": 'Control Flow, Barriers & System',
        "behavior": 'Issues a privileged hardware CSF doorbell message to the command-stream firmware/job manager (system/firmware use only; no user-shader register operands).',
        "notes": 'Not emitted by graphics/compute compilers; system message side effect. (`bifrost/bifrost/ISA.xml:3851`, `isa-v9-v14.xml`)',
    },
    'EUREKA': {
        "category": 'Control Flow, Barriers & System',
        "behavior": 'Signals a privileged hardware job-management event notification to the CSF firmware (system/firmware use only; no user-shader register operands).',
        "notes": 'System/job-management message. (`bifrost/bifrost/ISA.xml:3855`, `isa-v9-v14.xml`)',
    },
    'JUMP': {
        "category": 'Control Flow, Barriers & System',
        "behavior": 'Conditional indirect control transfer: evaluates `taken = not ? !combine(src0) : combine(src0)`. If `taken`, jumps to the shader address in `src1` (absolute or PC-relative per `address_mode`; jumping to address `0` terminates shader execution); otherwise falls through to `PC + 8`.',
        "notes": '`src1` is 32-bit address (`OpJump.address: S32`, `ops.rs:2609`). Used for blend shader calls (`lower_blend_call.rs:51-57`). (`kraid/ops.rs:2605-2630`, `encode_v9.rs:2015-2045`, `lower_blend_call.rs:31-58`)',
    },
    'JUMP_EX': {
        "category": 'Control Flow, Barriers & System',
        "behavior": 'Extended indirect jump/call/return: conditional jump with `not`/`combine`/`conservative`, divergence hint (`diverge`), and call-stack operation (`stack_mode`: `none`/`call`/`return`/`replace`) using `src1` and `src2`.',
        "notes": 'Supports hardware call/return stack management. (`isa-v9-v14.xml`, `bifrost/compiler.h:715`, `bifrost/bifrost/ISA.xml:1621`)',
    },
    'KABOOM': {
        "category": 'Control Flow, Barriers & System',
        "behavior": 'Triggers an immediate hardware shader fault/trap message to the CSF job manager; used as an unreachable-code or fatal-assertion sentinel.',
        "notes": 'Triggers hardware fault/abort; not a benign no-op. (`bifrost/bifrost/ISA.xml:6225`, `isa-v9-v14.xml`)',
    },
    'NOP': {
        "category": 'Control Flow, Barriers & System',
        "behavior": 'Architectural no-operation that performs no arithmetic or memory access, used to carry `flow` control actions such as scoreboard waits (`.wait`, `.wait0126`), `.reconverge`, or warp termination (`.end`).',
        "notes": 'No register or memory side effects. (`kraid/ops.rs:3651-3661`, `encode_v9.rs:2887-2895`, `bifrost/valhall/ISA.xml:135`)',
    },
    'LD_CHECKED': {
        "category": 'Memory Load / Store & LEA',
        "behavior": 'Bounds-checked load (v12+): loads `i8`..`i128` from base `src0` + index/offset `src1` checked against bound `src2`; on OOB, returns 0 (or `oob_mode` behavior on v14).',
        "notes": 'Robust buffer access hardware instruction. (`isa-v9-v14.xml`)',
    },
    'LD_CHECKED_IMM': {
        "category": 'Memory Load / Store & LEA',
        "behavior": 'Bounds-checked memory load (v12+) at 64-bit base address `src0` plus immediate unsigned byte offset `#offset` (`offset_lo` + `offset_hi`): if `#offset + access_bytes <= src2`, loads into `@sw`; otherwise returns `0` (or leaves `@sw` unchanged when `.oob_ignore` is set on v14).',
        "notes": 'Robust buffer access with immediate offset. (`isa-v9-v14.xml`)',
    },
    'LD_CVT': {
        "category": 'Memory Load / Store & LEA',
        "behavior": 'Load with format conversion: reads from `src0 + offset` and converts memory pixel/vertex format using internal conversion descriptor `src2` into `vecsize` components of `register_format`.',
        "notes": 'Paired with `LEA_TEX`/`LEA_ATTR` (`ISA.xml:70`). NOT a raw byte load! (`kraid/ops.rs:2707-2734`, `encode_v9.rs:2065-2085`, `bifrost/valhall/ISA.xml:70`)',
    },
    'LD_GCLK': {
        "category": 'Memory Load / Store & LEA',
        "behavior": 'Reads the 64-bit hardware GPU clock (`cycle_counter` for per-core shader cycles or `system_timestamp` for the cross-core global timestamp) asynchronously into the 2-register staging pair `@sw[0:1]`.',
        "notes": 'Non-deterministic architectural clock read. (`kraid/ops.rs:2767-2797`, `bifrost/valhall/ISA.xml:72`)',
    },
    'LD_MULTI': {
        "category": 'Memory Load / Store & LEA',
        "behavior": 'Multi-register load (v14+): loads `sr_count` (`ls_multi_sr_count_m`) consecutive 32-bit staging registers from `src0 + sext64(offset)`.',
        "notes": 'Wide bulk load beyond `LOAD.i128`. (`isa-v9-v14.xml`)',
    },
    'LD_PKA': {
        "category": 'Memory Load / Store & LEA',
        "behavior": 'Loads `i8` through `i128` bytes from the buffer descriptor identified by handle `src1` at byte offset `src0` into staging register range `@sw`, with hardware descriptor resolution and bounds checking.',
        "notes": 'If `handle[31:25] == 0x7F`, loads descriptor; if `0`, loads buffer data. (`kraid/ops.rs:2802-2828`, `bifrost/valhall/ISA.xml:74-81`)',
    },
    'LEA_ATTR': {
        "category": 'Memory Load / Store & LEA',
        "behavior": 'Computes effective address + conversion descriptor (3 staging regs: `addr_lo`, `addr_hi`, `cvt_desc`) for vertex attribute at dynamic handle.',
        "notes": 'Returns 3 staging registers (`ISA.xml:103`). (`bifrost/valhall/ISA.xml:103`, `isa-v9-v14.xml`)',
    },
    'LEA_ATTR_IMM': {
        "category": 'Memory Load / Store & LEA',
        "behavior": 'Resolves the Attribute descriptor at immediate `(table=#attribute_table_index, index=#attribute_index)` for `(vertex=src0, instance=src1)` and writes 3 staging registers to `@sw`: `@sw[0:1]` holds the 64-bit attribute GPU VA and `@sw[2]` holds the 32-bit format conversion descriptor for `LD_CVT`/`ST_CVT`.',
        "notes": 'Returns 3 staging registers (`ISA.xml:105`). (`bifrost/valhall/ISA.xml:105`, `isa-v9-v14.xml`)',
    },
    'LEA_BUF': {
        "category": 'Memory Load / Store & LEA',
        "behavior": 'Resolves the Buffer descriptor identified by dynamic handle `src1`, adds byte offset (or vertex index) `src0`, and writes the resulting 64-bit GPU virtual address into the 2-register staging pair `@sw[0:1]`.',
        "notes": 'Returns 64-bit address in 2 staging registers. (`kraid/ops.rs:3161-3185`, `bifrost/valhall/ISA.xml:107`)',
    },
    'LEA_BUF_IMM': {
        "category": 'Memory Load / Store & LEA',
        "behavior": 'Resolves the Buffer descriptor at immediate `(table=#buffer_table_index, index=#buffer_index)`, adds byte offset (or vertex index) `src0`, and writes the resulting 64-bit GPU virtual address into `@sw[0:1]`.',
        "notes": 'Immediate-handle form of `LEA_BUF`. (`bifrost/valhall/ISA.xml:109`, `isa-v9-v14.xml`)',
    },
    'LEA_PKA': {
        "category": 'Memory Load / Store & LEA',
        "behavior": 'Resolves the simple/PKA buffer descriptor identified by handle `src1`, adds byte offset `src0`, and writes the resulting 64-bit GPU virtual address into the 2-register staging pair `@sw[0:1]`.',
        "notes": 'Address-generation companion to `LD_PKA`. (`kraid/ops.rs:3187-3211`, `bifrost/valhall/ISA.xml:111`)',
    },
    'LEA_TEX': {
        "category": 'Memory Load / Store & LEA',
        "behavior": 'Computes effective address + conversion descriptor (3 staging regs) for image texel at packed 16-bit coordinates and dynamic handle.',
        "notes": 'Returns `(addr_lo, addr_hi, cvt_desc)` for `LD_CVT`/`ST_CVT` (`ISA.xml:113`). (`kraid/ops.rs:3213-3239`, `bifrost/valhall/ISA.xml:113`)',
    },
    'LEA_TEX_IMM': {
        "category": 'Memory Load / Store & LEA',
        "behavior": 'Resolves the Image/Texture descriptor at immediate `(table=#texture_table_index, index=#texture_index)` for coordinates `(xy=src0, z=src1)` and writes 3 staging registers to `@sw`: `@sw[0:1]` holds the 64-bit texel GPU VA and `@sw[2]` holds the 32-bit format conversion descriptor for `LD_CVT`/`ST_CVT`.',
        "notes": 'Immediate-handle form of `LEA_TEX` (`ISA.xml:115`). (`bifrost/valhall/ISA.xml:115`, `isa-v9-v14.xml`)',
    },
    'LOAD': {
        "category": 'Memory Load / Store & LEA',
        "behavior": 'Loads `1..16` bytes (`i8`/`i16`/`i24`/`i32`/`i48`/`i64`/`i96`/`i128`) from `src0 + sext64(offset)` into staging register(s) `sr_dst`; applies `extend` (`sext` vs `zext_or_none`) and sub-register `lane` (`b0..b3`/`h0..h1`) for narrow loads.',
        "notes": 'Little-endian memory; supports up to 128-bit (4 staging regs, even-aligned when `count > 1`). (`kraid/ops.rs:3241-3269`, `encode_v9.rs:2642-2663`, `bifrost/valhall/ISA.xml:117-124`)',
    },
    'STORE': {
        "category": 'Memory Load / Store & LEA',
        "behavior": 'Stores `1..16` bytes (`i8`..`i128`) from staging register(s) `sr_src` (with optional `lane` for `i8`/`i16`) to `src0 + sext64(offset)`.',
        "notes": 'Little-endian memory; supports up to 128-bit. (`kraid/ops.rs:4072-4105`, `encode_v9.rs:3119-3140`, `isa-v9-v14.xml`)',
    },
    'ST_CHECKED': {
        "category": 'Memory Load / Store & LEA',
        "behavior": 'Bounds-checked store (v12+): stores `i8`..`i128` to base `src0` + offset `src1` checked against bound `src2`; OOB stores are dropped.',
        "notes": 'Robust buffer access store. (`isa-v9-v14.xml`)',
    },
    'ST_CHECKED_IMM': {
        "category": 'Memory Load / Store & LEA',
        "behavior": 'Bounds-checked memory store (v12+) at 64-bit base address `src0` plus immediate unsigned byte offset `#offset` (`offset_lo` + `offset_hi`): if `#offset + access_bytes <= src2`, writes `@sr` to memory; out-of-bounds stores are silently dropped.',
        "notes": 'Robust buffer access store with immediate offset. (`isa-v9-v14.xml`)',
    },
    'ST_CVT': {
        "category": 'Memory Load / Store & LEA',
        "behavior": 'Store with format conversion: converts `vecsize` components of `register_format` from `sr_src` using conversion descriptor `src2` and writes to `src0 + offset`.',
        "notes": 'Used with `LEA_TEX_IMM` for image stores (`ISA.xml:142`). NOT a raw byte store! (`kraid/ops.rs:4038-4067`, `encode_v9.rs:3095-3117`, `bifrost/valhall/ISA.xml:142`)',
    },
    'ST_MULTI': {
        "category": 'Memory Load / Store & LEA',
        "behavior": 'Multi-register store (v14+): stores `sr_count` (`ls_multi_sr_count_m`) consecutive 32-bit staging registers to `src0 + sext64(offset)`.',
        "notes": 'Wide bulk store beyond `STORE.i128`. (`isa-v9-v14.xml`)',
    },
    'ACMPXCHG': {
        "category": 'Atomics',
        "behavior": 'Atomic compare-and-swap (`i32`/`i64`) at `src0 + offset`: `sr_src` holds `(compare, swap)` pair (`V2I32`/`V2I64`); if `*addr == compare`, `*addr = swap`; returns old `*addr` in `sr_dst`.',
        "notes": '`sr_src` is 2x width (`V2IN`, `ops.rs:100`). (`kraid/ops.rs:96-122`, `encode_v9.rs:565-580`)',
    },
    'ATOM': {
        "category": 'Atomics',
        "behavior": 'Fire-and-forget atomic RMW (`i32`/`i64`) at `src0 + offset` with operand `sr_src` and op `atom_opc` (`aadd`/`aaddu`/`aadds`/`asmin`/`asmax`/`aumin`/`aumax`/`aand`/`aor`/`axor`/`axchg`). No return value.',
        "notes": '`atom_opc` selects the exact RMW operation; `aaddu`/`aadds` are 64-bit atomics with 32-bit widened operand. (`kraid/ops.rs:176-276`, `encode_v9.rs:620-700`)',
    },
    'ATOM1': {
        "category": 'Atomics',
        "behavior": 'Fire-and-forget unary atomic RMW (`i32`/`i64`) at `src0 + offset` with implicit operand `1`/`-1`: `ainc` (`+1`), `adec` (`-1`), `aor1` (`|1`), `asmax1` (`smax 1`), `aumax1` (`umax 1`).',
        "notes": 'No `sr_src` operand; constant `1`/`-1` implicit (`ops.rs:190-199`). (`kraid/ops.rs:189-237, 281-313`, `encode_v9.rs:702-736`)',
    },
    'ATOM1_RETURN': {
        "category": 'Atomics',
        "behavior": 'Executes a unary atomic read-modify-write (`i32` or `i64`, selected by `atom1_opc`: `ainc`, `adec`, `asmax1`, `aumax1`, `aor1`) at GPU VA `src0 + #offset` and returns the pre-modification memory value in `@sw`.',
        "notes": 'Returns old value in `sr_dst`. (`kraid/ops.rs:281-313`, `encode_v9.rs:725-735`)',
    },
    'ATOM_RETURN': {
        "category": 'Atomics',
        "behavior": 'Atomic RMW (`i32`/`i64`) at `src0 + offset` with operand `sr_src` and op `atom_opc`; returns pre-modification memory value in `sr_dst`.',
        "notes": 'Returns old value in `sr_dst`. (`kraid/ops.rs:242-276`, `encode_v9.rs:689-698`)',
    },
    'AXCHG': {
        "category": 'Atomics',
        "behavior": 'Atomically swaps the 32-bit or 64-bit memory location at GPU VA `src0 + #offset` with the replacement value in `@sr` and returns the previous memory value in `@sw`.',
        "notes": 'Encoded from `OpAtom { atom_op: AtomOp::Xchg }` (`encode_v9.rs:679-688`). (`kraid/ops.rs:186, 242-276`, `encode_v9.rs:679-688`)',
    },
    'ATEST': {
        "category": 'Attributes, Varyings & Fragment',
        "behavior": 'Performs hardware alpha-to-coverage and early-Z coverage update in fragment shaders: evaluates `(coverage=src0 [r60], alpha=src1 [f32], datum=src2 [atest_datum])` and writes the updated MSAA sample coverage mask to `@sw` (pinned `r60`). Does not implicitly discard helper lanes.',
        "notes": 'Does NOT implicitly discard; must precede `ZS_EMIT`/`BLEND` (`ISA.xml:2`). (`kraid/ops.rs:145-173`, `encode_v9.rs:595-618`, `bifrost/valhall/ISA.xml:2`)',
    },
    'BLEND': {
        "category": 'Attributes, Varyings & Fragment',
        "behavior": 'Blends color `sr_src` (`r0..`) with render target using descriptor `src0` (64-bit) and coverage `src2` (`r60`). In fixed-function path, writes tilebuffer and either terminates (`.end`) or branches by `offset`; in blend-shader path, falls through to next instruction.',
        "notes": 'Dual-path control-flow semantics (`ISA.xml:17`, `lower_blend_call.rs:8-58`). (`kraid/ops.rs:361-402`, `encode_v9.rs:748-770`, `lower_blend_call.rs:8-58`, `bifrost/valhall/ISA.xml:17`)',
    },
    'LD_ATTR': {
        "category": 'Attributes, Varyings & Fragment',
        "behavior": 'Loads `vecsize` components of `register_format` from vertex attribute descriptor at dynamic handle `(src2)` for `(vertex_id=src0, instance_id=src1)`.',
        "notes": 'Handle must be subgroup-uniform (`ISA.xml:66`). (`kraid/ops.rs:2668-2694`, `bifrost/valhall/ISA.xml:66`)',
    },
    'LD_ATTR_IMM': {
        "category": 'Attributes, Varyings & Fragment',
        "behavior": 'Fetches `vecsize` vertex-attribute components for `(vertex=src0, instance=src1)` from the Attribute descriptor at immediate `(table=#attribute_table_index, index=#attribute_index)`, converts them to `register_format`, and writes them into `@sw`.',
        "notes": 'Immediate-handle form of `LD_ATTR` (`ISA.xml:68`). (`bifrost/valhall/ISA.xml:68`, `isa-v9-v14.xml`)',
    },
    'LD_TEX': {
        "category": 'Attributes, Varyings & Fragment',
        "behavior": 'Loads `vecsize` components of `register_format` from image/texture descriptor at dynamic handle (`src2`) at integer coords (`src0`, `src1`).',
        "notes": 'Direct image texel load (`ISA.xml:83`). (`kraid/ops.rs:2841-2864`, `bifrost/valhall/ISA.xml:83`)',
    },
    'LD_TEX_IMM': {
        "category": 'Attributes, Varyings & Fragment',
        "behavior": 'Reads `vecsize` texel components at integer coordinates `(xy=src0, z=src1)` from the Image/Texture descriptor at immediate `(table=#texture_table_index, index=#texture_index)`, converts them to `register_format`, and writes them into `@sw`.',
        "notes": 'Immediate-handle form of `LD_TEX` (`ISA.xml:85`). (`bifrost/valhall/ISA.xml:85`, `isa-v9-v14.xml`)',
    },
    'LD_TILE': {
        "category": 'Attributes, Varyings & Fragment',
        "behavior": 'Loads pixel data from tilebuffer at `(pixel=src0, coverage=src1, conversion=src2)` into `sr_dst`; `z_stencil` loads depth/stencil instead of color.',
        "notes": 'Used for framebuffer fetch and blend shaders (`ISA.xml:87`). (`kraid/ops.rs:2877-2911`, `bifrost/valhall/ISA.xml:87`)',
    },
    'LD_VAR': {
        "category": 'Attributes, Varyings & Fragment',
        "behavior": 'Interpolates varying via descriptor at dynamic handle (`src1`) at `sample_position` (`center`/`centroid`/`sample`/`explicit`), updating hidden barycentric cache per `update` (`store`/`retrieve`/`clobber`/`none`).',
        "notes": 'Software varying buffer interpolation (`ops.rs:2934-2999`). (`kraid/ops.rs:2914-2999`, `bifrost/valhall/ISA.xml:89`)',
    },
    'LD_VAR_BUF': {
        "category": 'Attributes, Varyings & Fragment',
        "behavior": 'Interpolates varying directly from hardware varying buffer at dynamic byte offset (`src1`) with `source_format` (`f32`/`f16`, v10+) conversion to `register_format`.',
        "notes": 'Descriptor-less varying interpolation (`ops.rs:3001-3042`). (`kraid/ops.rs:3001-3042`, `isa-v9-v14.xml`)',
    },
    'LD_VAR_BUF_FLAT': {
        "category": 'Attributes, Varyings & Fragment',
        "behavior": 'Fetches flat varying directly from hardware varying buffer at dynamic offset without conversion (v14+ native; v9-v13 lowers to `LD_VAR_BUF`).',
        "notes": 'Descriptor-less flat varying fetch (`ops.rs:3044-3071`). (`kraid/ops.rs:3044-3071`, `bifrost/valhall/ISA.xml:91`)',
    },
    'LD_VAR_BUF_FLAT_IMM': {
        "category": 'Attributes, Varyings & Fragment',
        "behavior": 'Fetches an uninterpolated (flat) provoking-vertex varying from the descriptor-less varying buffer at immediate byte offset `#offset` into `@sw` without interpolation or format conversion.',
        "notes": 'Descriptor-less flat varying fetch (`ISA.xml:93`). (`bifrost/valhall/ISA.xml:93`, `isa-v9-v14.xml`)',
    },
    'LD_VAR_BUF_IMM': {
        "category": 'Attributes, Varyings & Fragment',
        "behavior": 'Interpolates a fragment varying from the descriptor-less varying buffer at immediate byte offset `#offset` using pixel/sample coordinate `src0`, converting from `source_format` to `register_format` into `@sw` and updating the barycentric cache per `update_mode`.',
        "notes": 'Descriptor-less varying interpolation. (`isa-v9-v14.xml`)',
    },
    'LD_VAR_FLAT': {
        "category": 'Attributes, Varyings & Fragment',
        "behavior": 'Fetches an uninterpolated (flat) varying from the varying descriptor identified by dynamic handle `src0` for the selected triangle `vertex` (`vertex0`/`vertex1`/`vertex2`) into `@sw` without format conversion.',
        "notes": 'Software varying buffer flat fetch (`ops.rs:3073-3100`). (`kraid/ops.rs:3073-3100`, `bifrost/valhall/ISA.xml:95`)',
    },
    'LD_VAR_FLAT_IMM': {
        "category": 'Attributes, Varyings & Fragment',
        "behavior": 'Fetches an uninterpolated (flat) varying from the varying descriptor at immediate `(table=#varying_table_index, index=#varying_index)` for the selected triangle `vertex` into `@sw` without format conversion.',
        "notes": 'Software varying buffer flat fetch (`ISA.xml:97`). (`bifrost/valhall/ISA.xml:97`, `isa-v9-v14.xml`)',
    },
    'LD_VAR_IMM': {
        "category": 'Attributes, Varyings & Fragment',
        "behavior": 'Interpolates `vecsize` varying components from the varying descriptor at immediate `(table=#varying_table_index, index=#varying_index)` at pixel/sample coordinate `src0`, converting to `register_format` into `@sw` and caching/reusing barycentrics per `update_mode`.',
        "notes": 'Software varying buffer interpolation (`ISA.xml:99`). (`bifrost/valhall/ISA.xml:99`, `isa-v9-v14.xml`)',
    },
    'LD_VAR_SPECIAL': {
        "category": 'Attributes, Varyings & Fragment',
        "behavior": 'Reads a fixed-function rasterizer varying selected by `varying_name` (`point` for `gl_PointCoord`, `bary` for barycentric coordinates, `frag_w` for `gl_FragCoord.w`, or `frag_z` for `gl_FragCoord.z`) at `sample_position` (`src0`) into `@sw`.',
        "notes": '`frag_z` does not support `explicit` sample position (`ops.rs:3111`). (`kraid/ops.rs:3104-3150`, `isa-v9-v14.xml`)',
    },
    'ST_TILE': {
        "category": 'Attributes, Varyings & Fragment',
        "behavior": 'Encodes `vecsize` color components from `@sr` using `register_format` and the tilebuffer conversion descriptor in `src2`, and stores the result into on-chip tile color SRAM at pixel `src0` masked by sample coverage `src1`.',
        "notes": 'Used internally in blend shaders (`ISA.xml:144`). (`kraid/ops.rs:4118-4146`, `encode_v9.rs:3142-3165`, `bifrost/valhall/ISA.xml:144`)',
    },
    'ZS_EMIT': {
        "category": 'Attributes, Varyings & Fragment',
        "behavior": 'Emits programmatic depth (`src0`, if `.z`) and/or stencil (`src1`, if `.stencil`) with coverage `src2` (`r60`), returning updated coverage in `sr_dst` (`r60`).',
        "notes": 'Must execute after `DISCARD` and `ATEST` (`ops.rs:5106`, `encode_v9.rs:3863-3865`). (`kraid/ops.rs:5111-5146`, `encode_v9.rs:3850-3875`, `bifrost/valhall/ISA.xml:174`)',
    },
    'TEX_DUAL': {
        "category": 'Texture & Varying-Texture',
        "behavior": 'Paired dual texture sample issuing two samples with shared coordinates and independent `primary`/`secondary` widths and write masks.',
        "notes": 'Combines two texture fetches into one message (`ISA.xml:146`). (`kraid/ops.rs:4343-4391`, `bifrost/valhall/ISA.xml:146`)',
    },
    'TEX_FETCH': {
        "category": 'Texture & Varying-Texture',
        "behavior": 'Performs an unfiltered integer-coordinate texel fetch (`texelFetch`) from texture descriptor `src0` using integer coordinates `(x, [y], [z/layer], [lod/sample])` in `@sr`, writing `write_mask` channels into `@sw`.',
        "notes": 'No sampler filtering (`ISA.xml:148`). (`kraid/ops.rs:4393-4435`, `bifrost/valhall/ISA.xml:148`)',
    },
    'TEX_GATHER': {
        "category": 'Texture & Varying-Texture',
        "behavior": 'Samples the 2x2 bilinear footprint around floating-point UV coordinates in `@sr` from texture descriptor `src0` and returns the 4 values of the selected `gather_component` (`r`/`g`/`b`/`a`, or shadow compare results when `.compare` is set) into `@sw`.',
        "notes": '2D/Cube only (`ISA.xml:150`). (`kraid/ops.rs:4437-4482`, `bifrost/valhall/ISA.xml:150`)',
    },
    'TEX_GRADIENT': {
        "category": 'Texture & Varying-Texture',
        "behavior": 'Samples texture descriptor `src0` using explicit `(dPdx, dPdy)` screen-space gradients packed in `@sr` (`textureGrad`), or—when `.disable_lod_clamp` / `.derivative` is set—returns the hardware-computed 8.8 fixed-point mipmap LOD (`textureQueryLod`) or derivatives into `@sw`.',
        "notes": 'Supports explicit gradient sampling (`ISA.xml:152`). (`kraid/ops.rs:4504-4549`, `bifrost/valhall/ISA.xml:152`)',
    },
    'TEX_SINGLE': {
        "category": 'Texture & Varying-Texture',
        "behavior": 'Standard filtered texture sample via sampler/texture handle `src0` and staging coordinates `sr_src`, writing masked channels to `sr_dst`.',
        "notes": 'Hardware texture unit message; `skip` disables helper-invocation sampling when derivatives not needed. (`kraid/ops.rs:4551-4595`, `encode_v9.rs:3442-3459`, `bifrost/valhall/ISA.xml:154`)',
    },
    'TEX_WEIGHTS': {
        "category": 'Texture & Varying-Texture',
        "behavior": 'Texture footprint/filter weights query (arch v9 only): returns bilinear/footprint weights for `footprint` (`2x2`, etc.) at coordinates in `sr_src`.',
        "notes": 'Removed after v9. (`isa-v9-v14.xml`)',
    },
    'VAR_TEX_BUF_DUAL': {
        "category": 'Texture & Varying-Texture',
        "behavior": 'Fuses descriptor-less varying-buffer UV interpolation (at byte offset `src2`) with a dual texture sample (`TEX_DUAL` via handle `src0`), writing both 4-component sample results into `@sw` in a single message.',
        "notes": 'Fragment-only (`encode_v9.rs:3535-3585`). (`kraid/ops.rs:4649-4708`, `encode_v9.rs:3535-3585`, `bifrost/valhall/ISA.xml:155`)',
    },
    'VAR_TEX_BUF_GATHER': {
        "category": 'Texture & Varying-Texture',
        "behavior": 'Fuses descriptor-less varying-buffer UV interpolation (at byte offset `src2`) with a 2x2 texture gather (`TEX_GATHER` via handle `src0`), writing the 4 gathered component values into `@sw` in a single message.',
        "notes": 'Fragment-only; 2D (`encode_v9.rs:3587-3618`). (`kraid/ops.rs:4710-4762`, `encode_v9.rs:3587-3618`, `bifrost/valhall/ISA.xml:157`)',
    },
    'VAR_TEX_BUF_GRADIENT': {
        "category": 'Texture & Varying-Texture',
        "behavior": 'Fuses descriptor-less varying-buffer UV interpolation (at byte offset `src2`) with a texture gradient/LOD operation (`TEX_GRADIENT` via handle `src0`), writing the result into `@sw` in a single message.',
        "notes": 'Fragment-only (`encode_v9.rs:3620-3650`). (`kraid/ops.rs:4764-4817`, `encode_v9.rs:3620-3650`, `bifrost/valhall/ISA.xml:159`)',
    },
    'VAR_TEX_BUF_SINGLE': {
        "category": 'Texture & Varying-Texture',
        "behavior": 'Fuses descriptor-less varying-buffer UV interpolation (at byte offset `src2`, `source_format` f32/f16 on v10+) with a filtered texture sample (`TEX_SINGLE` via handle `src0`), writing the sampled texel into `@sw` in a single message.',
        "notes": 'Fragment-only (`encode_v9.rs:3519-3533, 3652-3683`). (`kraid/ops.rs:4819-4870`, `encode_v9.rs:3652-3683`, `bifrost/valhall/ISA.xml:161`)',
    },
    'VAR_TEX_DUAL': {
        "category": 'Texture & Varying-Texture',
        "behavior": 'Fuses descriptor-based varying UV interpolation (via varying handle `src2`) with a dual texture sample (`TEX_DUAL` via texture handle `src0`), writing both sample results into `@sw` in a single message.',
        "notes": 'Fragment-only; FP32 varyings (`ISA.xml:163`). (`kraid/ops.rs:4872-4929`, `encode_v9.rs:3685-3734`, `bifrost/valhall/ISA.xml:163`)',
    },
    'VAR_TEX_GATHER': {
        "category": 'Texture & Varying-Texture',
        "behavior": 'Fuses descriptor-based varying UV interpolation (via varying handle `src2`) with a 2x2 texture gather (`TEX_GATHER` via texture handle `src0`), writing the 4 gathered component values into `@sw` in a single message.',
        "notes": 'Fragment-only; 2D (`encode_v9.rs:3749`). (`kraid/ops.rs:4931-4981`, `encode_v9.rs:3736-3766`, `bifrost/valhall/ISA.xml:165`)',
    },
    'VAR_TEX_GRADIENT': {
        "category": 'Texture & Varying-Texture',
        "behavior": 'Fuses descriptor-based varying UV interpolation (via varying handle `src2`) with a texture gradient/LOD operation (`TEX_GRADIENT` via texture handle `src0`), writing the result into `@sw` in a single message.',
        "notes": 'Fragment-only; FP32 varyings (`ISA.xml:167`). (`kraid/ops.rs:4983-5034`, `encode_v9.rs:3768-3797`, `bifrost/valhall/ISA.xml:167`)',
    },
    'VAR_TEX_SINGLE': {
        "category": 'Texture & Varying-Texture',
        "behavior": 'Fuses descriptor-based varying UV interpolation (via varying handle `src2`) with a filtered texture sample (`TEX_SINGLE` via texture handle `src0`), writing the sampled texel into `@sw` in a single message.',
        "notes": 'Fragment-only; FP32 varyings (`ISA.xml:169`). (`kraid/ops.rs:5036-5083`, `encode_v9.rs:3799-3829`, `bifrost/valhall/ISA.xml:169`)',
    },
}


def category_of(mnem: str) -> str:
    """Return the Chapter 3 category for *mnem*, or "Other" if unknown."""
    entry = ISA_SEMANTICS.get(mnem)
    return entry["category"] if entry else "Other"


def behavior_of(mnem: str) -> tuple:
    """Return (behavior, notes) Markdown for *mnem*, or ("", "") if unknown."""
    entry = ISA_SEMANTICS.get(mnem)
    if entry is None:
        return "", ""
    return entry["behavior"], entry["notes"]


def missing_semantics(mnemonics) -> list:
    """Return sorted mnemonics that have no frozen semantics entry."""
    return sorted(m for m in mnemonics if m not in ISA_SEMANTICS)


# ============================================================================
# Per-Mnemonic SPIR-V Mapping, Inducing GLSL, and Worked Examples (All 201 Ops)
# ============================================================================
from .fmt_helpers import fmt_asm_line

# Format: mnem -> (spirv_op, glsl_snippet, [(label, asm_tex, walkthrough_md), ...])
INST_EXTRAS = {
    # 1. Integer Arithmetic
    "BITREV": (
        "OpBitReverse", "uint y = bitfieldReverse(x);",
        [
            ("Canonical 32-bit reversal",
             fmt_asm_line("BITREV.i32", "", "r0", ["r1"]),
             "With `r1 = 0x12345678` (`0001_0010_0011_0100_0101_0110_0111_1000`): bit `k` moves to bit `31 - k`, producing `r0 = 0x1E6A2C48`."),
            ("Palindrome & MSB/LSB swap",
             fmt_asm_line("BITREV.i32", "", "r0", ["r1"]),
             "With `r1 = 0x80000001`: bit 31 and bit 0 swap cleanly -> `r0 = 0x80000001`; for `r1 = 0x00000003` -> `r0 = 0xC0000000`."),
        ],
    ),
    "CLZ": (
        "GLSL.std.450 FindUMsb (31 - clz)", "int m = findMSB(x);",
        [
            ("Non-zero input",
             fmt_asm_line("CLZ.u32", "", "r0", ["r1"]),
             "With `r1 = 0x00100000` (bit 20 set): 11 zero bits precede bit 20 -> `r0 = 11`. GLSL `findMSB(r1)` is then `31 - 11 = 20`."),
            ("Zero input with .mask vs default",
             fmt_asm_line("CLZ.u32", ".mask", "r0", ["r1"]),
             "With `r1 = 0x00000000`: without `.mask`, hardware returns `r0 = 32` (lane width); with `.mask`, hardware returns `r0 = 0xFFFFFFFF` (`-1`, matching GLSL `findMSB(0) == -1`)."),
        ],
    ),
    "HADD": (
        "None (lowered from `(a+b)>>1` / `findMSB(int)`)", "int h = (a >> 1) + (b >> 1) + ((a & b) & 1);",
        [
            ("Overflow-free unsigned ceil average",
             fmt_asm_line("HADD.u32", ".round_up", "r0", ["r1", "r2"]),
             "With `r1 = 0xFFFFFFFF, r2 = 0xFFFFFFFF`: exact 33-bit sum `(2^32-1 + 2^32-1 + 1) >> 1 = 0xFFFFFFFF` with zero overflow."),
            ("Signed floor vs ceil rounding",
             fmt_asm_line("HADD.s32", ".round_down", "r0", ["r1", "r2"]),
             "With `r1 = 1, r2 = 2`: `.round_down` computes `(1 + 2) >> 1 = 1`, whereas `.round_up` computes `(1 + 2 + 1) >> 1 = 2` (v9--v10 only)."),
        ],
    ),
    "IABS": (
        "GLSL.std.450 SAbs", "int y = abs(x);",
        [
            ("Negative 32-bit signed integer",
             fmt_asm_line("IABS.s32", "", "r0", ["r1"]),
             "With `r1 = 0xFFFFFFFB` (`-5`): negates to `r0 = 5` (`0x00000005`). Positive inputs pass through unchanged."),
            ("INT32_MIN wrapping edge case",
             fmt_asm_line("IABS.s32", "", "r0", ["r1"]),
             "With `r1 = 0x80000000` (`-2147483648`): `IABS` is non-saturating, so `unsigned_abs()` returns `0x80000000` (identical bit pattern, NOT `0x7FFFFFFF`)."),
        ],
    ),
    "IADD": (
        "OpIAdd", "int c = a + b;",
        [
            ("Widening + saturating 32-bit signed add",
             fmt_asm_line("IADD.s32", ".sat", "r0", ["r1.h0", "r2"]),
             "With `r1 = 0x00007FFF` (`+32767`), `r2 = 0x7FFFFFF0`: Stage A sign-extends `r1.h0` to `0x00007FFF`; Stage B clamps overflow to `0x7FFFFFFF`."),
            ("Packed SIMD v2s16 with lane-mask writeback",
             fmt_asm_line("IADD.v2s16", "", "r0.h1", ["r1.h01", "r2.h01"]),
             "With `r1 = [10, 20]` (`0x0014000A`), `r2 = [5, -3]` (`0xFFFD0005`), `r0_old = 0x0000BEEF`: lane 1 computes `20 + (-3) = 17` (`0x0011`); Stage C `.h1` writes `r0 = 0x0011BEEF`."),
        ],
    ),
    "IADDC": (
        "OpIAddCarry", "uint sum = uaddCarry(a, b, carry);",
        [
            ("Carry-out extraction (.c)",
             fmt_asm_line("IADDC.u32", ".c", "r0", ["r1", "r2", "r3"]),
             "With `r1 = 0xFFFFFFFF, r2 = 0, r3 = 1`: 64-bit sum is `0x1_00000000`; `.c` returns the carry-out bit `r0 = 1`."),
            ("32-bit sum with carry-in (no .c)",
             fmt_asm_line("IADDC.u32", "", "r0", ["r1", "r2", "r3"]),
             "With `r1 = 0xFFFFFFFF, r2 = 5, r3 = 0x00000003` (LSB=1): `0xFFFFFFFF + 5 + 1 = 0x1_00000005`; without `.c`, returns low 32 bits `r0 = 5`."),
        ],
    ),
    "IADD_IMM": (
        "OpIAdd (constant RHS)", "uint y = x + 0x12345u;",
        [
            ("32-bit inline constant add",
             fmt_asm_line("IADD_IMM.i32", "", "r0", ["r1"], ["#0x100"]),
             "With `r1 = 0x00000040`: adds 32-bit inline immediate -> `r0 = 0x00000140` (wraps mod `2^32` on overflow)."),
            ("Packed v2i16 per-lane immediate add",
             fmt_asm_line("IADD_IMM.v2i16", "", "r0", ["r1"], ["#0x00020001"]),
             "With `r1 = 0x00100020` (`[16, 32]`): adds `1` to lane 0 and `2` to lane 1 independently -> `r0 = 0x00120021`."),
        ],
    ),
    "IDPADD": (
        "OpSDotAccSat / OpUDotAccSat", "int d = dotPacked4x8EXT(a, b) + acc;",
        [
            ("Signed 4x8 dot product with saturation",
             fmt_asm_line("IDPADD.v4s8", ".sat", "r0", ["r1", "r2", "r3"]),
             "With `r1 = 0x01020304`, `r2 = 0x01010101`, `r3 = 10`: dot `1*1 + 2*1 + 3*1 + 4*1 = 10`, accumulated with `r3` -> `r0 = 20`."),
            ("Negative signed lanes & saturation clamp",
             fmt_asm_line("IDPADD.v4s8", ".sat", "r0", ["r1", "r2", "r3"]),
             "With `r1 = 0x7F7F7F7F` (`[127,127,127,127]`), `r2 = 0x80808080` (`[-128,...]`), `r3 = 0x80000000` (`INT32_MIN`): dot is `-65024`; `.sat` clamps underflow to `r0 = 0x80000000`."),
        ],
    ),
    "IMUL": (
        "OpIMul / OpUMulExtended / OpSMulExtended", "uint64_t p = uint64_t(a) * uint64_t(b);",
        [
            ("32x32 -> 64-bit widening unsigned multiply",
             fmt_asm_line("IMUL.u64", "", "[r0:r1]", ["[r2:r3].w0", "[r4:r5].w0"]),
             "With `r2 = 0x80000000, r4 = 4`: Stage A zero-extends `.w0` to 64-bit; Stage B computes `0x00000002_00000000` in `[r0:r1]` (`r0 = 0`, `r1 = 2`)."),
            ("32-bit saturating signed multiply",
             fmt_asm_line("IMUL.s32", ".sat", "r0", ["r1", "r2"]),
             "With `r1 = 0x00010000` (`65536`), `r2 = 0x00010000` (`65536`): exact product `2^32` overflows `s32`; `.sat` clamps to `r0 = 0x7FFFFFFF`."),
        ],
    ),
    "ISUB": (
        "OpISub", "uint d = a - b;",
        [
            ("Unsigned saturating underflow clamp",
             fmt_asm_line("ISUB.u32", ".sat", "r0", ["r1", "r2"]),
             "With `r1 = 5, r2 = 12`: unsigned underflow clamps to `r0 = 0` under `.sat` (without `.sat`, wraps to `0xFFFFFFF9`)."),
            ("Signed saturating overflow on subtracting negative",
             fmt_asm_line("ISUB.s32", ".sat", "r0", ["r1", "r2"]),
             "With `r1 = 0x7FFFFFF0, r2 = -32` (`0xFFFFFFE0`): exact `0x7FFFFFF0 - (-32) = 0x80000010` overflows `s32` -> `.sat` clamps to `r0 = 0x7FFFFFFF`."),
        ],
    ),
    "ISUBB": (
        "OpISubBorrow", "uint diff = usubBorrow(a, b, borrow);",
        [
            ("Borrow-out extraction (.c)",
             fmt_asm_line("ISUBB.u32", ".c", "r0", ["r1", "r2", "r3"]),
             "With `r1 = 3, r2 = 3, r3 = 1` (borrow-in): `3 < 3 + 1`, so `.c` returns borrow-out `r0 = 1`."),
            ("32-bit difference with borrow-in (no .c)",
             fmt_asm_line("ISUBB.u32", "", "r0", ["r1", "r2", "r3"]),
             "With `r1 = 10, r2 = 4, r3 = 1`: computes `10 - 4 - 1 = 5` -> `r0 = 5` (multi-word 64/128-bit subtraction)."),
        ],
    ),
    "MMUL": (
        "OpCooperativeMatrixMulAddKHR", "coopMatMulAdd(A, B, C, ...);",
        [
            ("FP16 submatrix multiply-accumulate into FP32",
             fmt_asm_line("MMUL.v2f16", "", "r0", ["r1.f0", "r2.f1", "r3"]),
             "Each lane in a 4x4 subgroup computes one output element of `D = A[f0] * B[f1] + C` using 4x4 FP16 submatrices accumulated in FP32 `r0`."),
            ("v14 mixed-sign INT8 saturating matrix multiply",
             fmt_asm_line("MMUL.v4s8", ".sat", "r0", ["r1", "r2", "r3"]),
             "Multiplies 4x16 `v4s8` tile `A` by 16x4 `B`, accumulates into signed 32-bit `r3`, and saturates to `[INT32_MIN, INT32_MAX]` (v14+)."),
        ],
    ),
    "POPCOUNT": (
        "OpBitCount", "int c = bitCount(x);",
        [
            ("Mixed bit pattern",
             fmt_asm_line("POPCOUNT.i32", "", "r0", ["r1"]),
             "With `r1 = 0xF0F00003`: counts set bits `4 + 4 + 2 = 10` -> `r0 = 10`."),
            ("Boundary values (0 and all-ones)",
             fmt_asm_line("POPCOUNT.i32", "", "r0", ["r1"]),
             "With `r1 = 0x00000000` -> `r0 = 0`; with `r1 = 0xFFFFFFFF` -> `r0 = 32`."),
        ],
    ),
    "SEG_ADD": (
        "OpPtrAccessChain (TLS/WLS segment)", "shared_arr[idx] // WLS base + offset",
        [
            ("Null-preserving pointer offset",
             fmt_asm_line("SEG_ADD.u64", ".preserve_null", "[r0:r1]", ["[r2:r3]", "r4.w0"]),
             "With `[r2:r3] = 0x0`, `r4 = 0x20`: `.preserve_null` detects null base and keeps `[r0:r1] = 0x0` instead of `0x20`."),
            ("Non-null WLS pointer advance",
             fmt_asm_line("SEG_ADD.u64", ".preserve_null", "[r0:r1]", ["[r2:r3]", "r4.w0"]),
             "With `[r2:r3] = 0x00000001_00001000`, `r4 = 0x40`: zero-extends `r4.w0` and adds -> `[r0:r1] = 0x00000001_00001040`."),
        ],
    ),
    "SEG_SUB": (
        "OpPtrAccessChain (negative offset)", "shared_arr[idx - 1] // WLS base - offset",
        [
            ("Non-null pointer decrement",
             fmt_asm_line("SEG_SUB.u64", ".preserve_null", "[r0:r1]", ["[r2:r3]", "r4.w0"]),
             "With `[r2:r3] = 0x1000`, `r4 = 0x20`: subtracts widened `r4` -> `[r0:r1] = 0x0FE0`."),
            ("Null pointer preservation",
             fmt_asm_line("SEG_SUB.u64", ".preserve_null", "[r0:r1]", ["[r2:r3]", "r4.w0"]),
             "With `[r2:r3] = 0`, `r4 = 0x10`: `.preserve_null` keeps `[r0:r1] = 0` rather than wrapping to `0xFFFFFFFF_FFFFFFF0`."),
        ],
    ),
    "SHADD": (
        "OpIAdd + OpShiftLeftLogical (64-bit addr)", "uint64_t addr = base + (uint64_t(idx) << 2);",
        [
            ("Unsigned array stride addressing",
             fmt_asm_line("SHADD.u64", "", "[r0:r1]", ["[r2:r3]", "r4.w0"], ["#2"]),
             "With `[r2:r3] = 0x1000`, `r4 = 5`, `#shift = 2`: `0x1000 + (5 << 2) = 0x1014` (indexes 4-byte elements)."),
            ("Signed negative index stride",
             fmt_asm_line("SHADD.s64", "", "[r0:r1]", ["[r2:r3]", "r4.w0"], ["#3"]),
             "With `[r2:r3] = 0x2000`, `r4 = 0xFFFFFFFE` (`-2`), `#shift = 3`: sign-extends `-2` to 64-bit, shifts `<< 3` (`-16`), adds -> `[r0:r1] = 0x1FF0`."),
        ],
    ),

    # 2. Bitwise & Shift-Logic (specific overrides; pattern generator covers fused variants)
    "AND": (
        "OpBitwiseAnd", "uint y = a & ~b;",
        [
            ("Bit-clear using src2.not",
             fmt_asm_line("AND.i32", "", "r0", ["r1", "r2.not"]),
             "With `r1 = 0xFF, r2 = 0x0F`: Stage A inverts `src2` (`~0x0F = 0xFFFFFFF0`); Stage B computes `0xFF & 0xFFFFFFF0 = 0xF0`."),
            ("NAND via .not_result",
             fmt_asm_line("AND.i32", ".not_result", "r0", ["r1", "r2"]),
             "With `r1 = 0x0000FFFF, r2 = 0x00FF00FF`: `r1 & r2 = 0x000000FF`, then `.not_result` complements -> `r0 = 0xFFFFFF00`."),
        ],
    ),
    "AND_IMM": (
        "OpBitwiseAnd (constant)", "uint y = x & 0x00FF00FFu;",
        [
            ("32-bit inline bitmask (v11+)",
             fmt_asm_line("AND_IMM.i32", "", "r0", ["r1"], ["#0x00FF00FF"]),
             "With `r1 = 0x12345678`: masks with 32-bit inline immediate -> `r0 = 0x00340078` without consuming an FAU slot."),
            ("Sign-bit extraction",
             fmt_asm_line("AND_IMM.i32", "", "r0", ["r1"], ["#0x80000000"]),
             "With `r1 = 0xBF800000` (`-1.0f`): isolates IEEE sign bit -> `r0 = 0x80000000`."),
        ],
    ),
    "OR": (
        "OpBitwiseOr", "uint y = a | b;",
        [
            ("NOR via .not_result",
             fmt_asm_line("OR.i32", ".not_result", "r0", ["r1", "r2"]),
             "With `r1 = 0xF0, r2 = 0x0A`: `0xF0 | 0x0A = 0xFA`, then `.not_result` inverts -> `r0 = 0xFFFFFF05`."),
            ("Bitwise OR with inverted second operand",
             fmt_asm_line("OR.i32", "", "r0", ["r1", "r2.not"]),
             "With `r1 = 0x0000000F, r2 = 0xFFFFFFF0`: `~r2 = 0x0000000F`, so `0xF | 0xF = 0x0000000F`."),
        ],
    ),
    "OR_IMM": (
        "OpBitwiseOr (constant)", "uint y = x | 0x80000000u;",
        [
            ("Force negative sign bit (v11+)",
             fmt_asm_line("OR_IMM.i32", "", "r0", ["r1"], ["#0x80000000"]),
             "With `r1 = 0x3F800000` (`+1.0f`): sets bit 31 -> `r0 = 0xBF800000` (`-1.0f`)."),
            ("Set flag bits in packed bitfield",
             fmt_asm_line("OR_IMM.i32", "", "r0", ["r1"], ["#0x00000005"]),
             "With `r1 = 0x00000010`: sets bits 0 and 2 -> `r0 = 0x00000015`."),
        ],
    ),
    "XOR": (
        "OpBitwiseXor", "uint y = a ^ b;",
        [
            ("Bitwise toggle",
             fmt_asm_line("XOR.i32", "", "r0", ["r1", "r2"]),
             "With `r1 = 0xAAAA5555, r2 = 0xFFFFFFFF`: flips all 32 bits -> `r0 = 0x5555AAAA`."),
            ("XNOR via .not_result",
             fmt_asm_line("XOR.i32", ".not_result", "r0", ["r1", "r2"]),
             "With `r1 = 0x000000FF, r2 = 0x000000FF`: `r1 ^ r2 = 0`, then `.not_result` yields `r0 = 0xFFFFFFFF` (bitwise equality mask)."),
        ],
    ),
    "XOR_IMM": (
        "OpBitwiseXor (constant)", "uint y = x ^ 0x80000000u;",
        [
            ("Flip FP32 sign bit (v11+)",
             fmt_asm_line("XOR_IMM.i32", "", "r0", ["r1"], ["#0x80000000"]),
             "With `r1 = 0xBF800000` (`-1.0f`): toggles sign bit -> `r0 = 0x3F800000` (`+1.0f`)."),
            ("Toggle dual FP16 sign bits",
             fmt_asm_line("XOR_IMM.i32", "", "r0", ["r1"], ["#0x80008000"]),
             "With `r1 = 0x3C00BC00` (`[+1.0h, -1.0h]`): toggles both halfword sign bits -> `r0 = 0xBC003C00` (`[-1.0h, +1.0h]`)."),
        ],
    ),
    "NOT": (
        "OpNot", "uint y = ~x;",
        [
            ("32-bit bitwise complement",
             fmt_asm_line("NOT.i32", "", "r0", ["r1"]),
             "With `r1 = 0x0000FFFF`: flips every bit -> `r0 = 0xFFFF0000`."),
            ("Zero and all-ones complement",
             fmt_asm_line("NOT.i32", "", "r0", ["r1"]),
             "With `r1 = 0x00000000` -> `r0 = 0xFFFFFFFF`; with `r1 = 0xFFFFFFFF` -> `r0 = 0x00000000`."),
        ],
    ),
    "NOT_OLD": (
        "OpNot (legacy v9--v13)", "uint y = ~x; // legacy SFU form",
        [
            ("Legacy SFU bitwise complement (v9--v13)",
             fmt_asm_line("NOT_OLD.i32", "", "r0", ["r1"]),
             "With `r1 = 0x12345678`: bitwise complement -> `r0 = 0xEDCBA987` (removed in v14 in favor of `NOT`)."),
            ("Packed v2i16 complement",
             fmt_asm_line("NOT_OLD.i32", "", "r0.h0", ["r1"]),
             "With `r1 = 0x123400FF`, `r0_old = 0xAAAA0000`: complements low halfword to `0xFF00` and `.h0` writes `r0 = 0xAAAAFF00`."),
        ],
    ),

    # 3. Floating-Point ALU
    "FABS": (
        "GLSL.std.450 FAbs", "float y = abs(x);",
        [
            ("Clear negative sign bit",
             fmt_asm_line("FABS.f32", "", "r0", ["r1"]),
             "With `r1 = 0xBF800000` (`-1.0f`): clears bit 31 -> `r0 = 0x3F800000` (`+1.0f`)."),
            ("Negative zero & negative NaN",
             fmt_asm_line("FABS.f32", "", "r0", ["r1"]),
             "With `r1 = 0x80000000` (`-0.0f`) -> `r0 = 0x00000000` (`+0.0f`); with `r1 = 0xFFC00001` (`-qNaN`) -> clears sign bit."),
        ],
    ),
    "FADD": (
        "OpFAdd", "float c = clamp(a + b, 0.0, 1.0);",
        [
            ("Clamped FP32 add with fused Stage C FP16 narrowing",
             fmt_asm_line("FADD.f32", ".clamp_0_1", "r0.h1", ["r1", "r2"]),
             "With `r1 = 1.0f, r2 = 0.5f, r0_old = 0x0000BEEF`: Stage B clamps `1.5f` to `1.0f`; Stage C `dest_width_narrow=h1` narrows to `0x3C00` in `r0[31:16]` -> `r0 = 0x3C00BEEF`."),
            ("FP16 widening source + rounding mode",
             fmt_asm_line("FADD.f32", ".round_down", "r0", ["r1.h0.neg", "r2"]),
             "With `r1.h0 = 2.0h` (`0x4000`), `r2 = 5.0f`: Stage A widens `2.0h -> 2.0f` then negates to `-2.0f`; Stage B computes `5.0f + (-2.0f) = 3.0f` under RTN."),
        ],
    ),
    "FADD_IMM": (
        "OpFAdd (constant)", "float y = x + 1.5;",
        [
            ("Inline FP32 constant addition",
             fmt_asm_line("FADD_IMM.f32", "", "r0", ["r1"], ["#0x3FC00000"]),
             "With `r1 = 2.5f`: adds inline `1.5f` (`0x3FC00000`) under RNE -> `r0 = 4.0f` (`0x40800000`)."),
            ("Signed zero rule (`-0.0 + +0.0`)",
             fmt_asm_line("FADD_IMM.f32", "", "r0", ["r1"], ["#0x00000000"]),
             "With `r1 = -0.0f` (`0x80000000`) and inline `+0.0f`: IEEE RNE yields `+0.0f` (`0x00000000`), canonicalizing `-0.0`."),
        ],
    ),
    "FEXP": (
        "GLSL.std.450 Exp2", "float y = exp2(x);",
        [
            ("Q8.24 fixed-point + FP32 input",
             fmt_asm_line("FEXP.f32", "", "r0", ["r1", "r2"]),
             "With `r1 = 0x02000000` (`+2.0` in Q8.24), `r2 = 2.0f`: computes `2^2.0 = 4.0f` (`0x40800000`)."),
            ("Negative exponent & underflow",
             fmt_asm_line("FEXP.f32", "", "r0", ["r1", "r2"]),
             "With `r1 = 0xFE000000` (`-2.0` in Q8.24), `r2 = -2.0f`: computes `2^-2.0 = 0.25f` (`0x3E800000`); inputs `<= -127.0` flush to `+0.0f`."),
        ],
    ),
    "FEXPF": (
        "GLSL.std.450 Exp2 (v14+)", "float y = exp2(x);",
        [
            ("Direct FP32 exp2 (v14+)",
             fmt_asm_line("FEXPF.f32", "", "r0", ["r1"]),
             "With `r1 = 3.0f`: direct floating-point `exp2(3.0) = 8.0f` (`0x41000000`) without needing Q8.24 pre-scaling."),
            ("Special values (`-Inf`, `+Inf`, `NaN`)",
             fmt_asm_line("FEXPF.f32", "", "r0", ["r1"]),
             "With `r1 = -Inf` -> `+0.0f`; with `r1 = +Inf` (or `>= 128.0f`) -> `+Inf`; with `r1 = NaN` -> `qNaN`."),
        ],
    ),
    "FLOG": (
        "GLSL.std.450 Log2 (v12+ f16 / v14+ f32)", "float y = log2(x);",
        [
            ("Exact power of two",
             fmt_asm_line("FLOG.f32", "", "r0", ["r1"]),
             "With `r1 = 8.0f`: computes `log2(8.0) = 3.0f` (`0x40400000`); with `r1 = 0.5f` -> `-1.0f`."),
            ("Domain boundary values",
             fmt_asm_line("FLOG.f32", "", "r0", ["r1"]),
             "With `r1 = ±0.0f` -> `-Inf` (`0xFF800000`); with `r1 = -1.0f` -> `qNaN`; with `r1 = +Inf` -> `+Inf`."),
        ],
    ),
    "FLUSH": (
        "None (FP canonicalization / FTZ)", "// Compiler-inserted FTZ/NaN/Inf flush",
        [
            ("Infinity clamping + sNaN quietening",
             fmt_asm_line("FLUSH.f32", ".ftz.flush_inf.quiet_nan", "r0", ["r1"]),
             "With `r1 = 0xFF800000` (`-Inf`): `.flush_inf` clamps to `-MAX_FINITE` (`0xFF7FFFFF`); an sNaN `0x7F800001` becomes `0x7FC00001`."),
            ("Subnormal FTZ preserving sign",
             fmt_asm_line("FLUSH.f32", ".ftz", "r0", ["r1"]),
             "With `r1 = 0x80000001` (smallest negative FP32 subnormal): `.ftz` flushes to `-0.0f` (`0x80000000`), preserving sign bit."),
        ],
    ),
    "FMA": (
        "GLSL.std.450 Fma", "float d = fma(a, -b, c);",
        [
            ("Single-rounded multiply-subtract",
             fmt_asm_line("FMA.f32", "", "r0", ["r1", "r2.neg", "r3"]),
             "With `r1 = 2.0f, r2 = 3.0f, r3 = 10.0f`: Stage A negates `r2` to `-3.0f`; single-rounded `2.0 * (-3.0) + 10.0 = 4.0f`."),
            ("Clamped FMA with fused FP16 replication (.h01)",
             fmt_asm_line("FMA.f32", ".clamp_0_1", "r0.h01", ["r1", "r2", "r3"]),
             "With `r1 = 0.5f, r2 = 1.0f, r3 = 0.75f`: exact `1.25f` clamps to `1.0f`, narrows to `0x3C00`, and `.h01` replicates -> `r0 = 0x3C003C00`."),
        ],
    ),
    "FMAX": (
        "GLSL.std.450 NMax / FMax", "float m = max(a, b);",
        [
            ("Signed zero ordering & NaN propagation",
             fmt_asm_line("FMAX.f32", ".nan_propagate", "r0", ["r1", "r2"]),
             "With `r1 = -0.0f, r2 = +0.0f`: deterministically returns `+0.0f`. With `r1 = 0xFF800001` (sNaN), `r2 = 1.0f`: `.nan_propagate` returns quietened `0xFFC00001`."),
            ("NaN suppression (IEEE maxNum)",
             fmt_asm_line("FMAX.f32", ".nan_suppress", "r0", ["r1", "r2"]),
             "With `r1 = NaN, r2 = -5.0f`: `.nan_suppress` ignores NaN and returns `-5.0f` (`0xC0A00000`)."),
        ],
    ),
    "FMIN": (
        "GLSL.std.450 NMin / FMin", "float m = min(a, b);",
        [
            ("Deterministic `-0.0 < +0.0` & NaN suppression",
             fmt_asm_line("FMIN.f32", ".nan_suppress", "r0", ["r1", "r2"]),
             "With `r1 = -0.0f, r2 = +0.0f`: deterministically returns `-0.0f` (`0x80000000`). With `r1 = NaN, r2 = 1.0f`: returns `1.0f`."),
            ("NaN propagation + post-op clamp order",
             fmt_asm_line("FMIN.f32", ".nan_propagate.clamp_0_1", "r0", ["r1", "r2"]),
             "With `r1 = NaN, r2 = 0.5f`: `sem` runs BEFORE `clamp` (propagating NaN), then `.clamp_0_1` maps `NaN -> +0.0f` (`0x00000000`)."),
        ],
    ),
    "FMUL": (
        "OpFMul", "float c = a * b;",
        [
            ("Signed-zero preservation (FMA alias)",
             fmt_asm_line("FMUL.f32", "", "r0", ["r1", "r2"]),
             "Encoded as `FMA(r1, r2, -0.0)`: with `r1 = +0.0f, r2 = -0.0f`, produces `-0.0f` (`0x80000000`)."),
            ("Packed v2f16 multiply with swizzle",
             fmt_asm_line("FMUL.v2f16", "", "r0", ["r1.h01", "r2.h00"]),
             "With `r1 = [3.0h, 2.0h]`, `r2 = [0.0h, 4.0h]` (`h0 = 4.0h`): broadcasts `r2.h0` to both lanes -> `r0 = [12.0h, 8.0h]`."),
        ],
    ),
    "FNEG": (
        "OpFNegate", "float y = -x;",
        [
            ("Negate positive float",
             fmt_asm_line("FNEG.f32", "", "r0", ["r1"]),
             "With `r1 = 0x3F800000` (`+1.0f`): flips sign bit -> `r0 = 0xBF800000` (`-1.0f`)."),
            ("Negate zero and infinity",
             fmt_asm_line("FNEG.f32", "", "r0", ["r1"]),
             "With `r1 = +0.0f` (`0x00000000`) -> `-0.0f` (`0x80000000`); with `r1 = -Inf` (`0xFF800000`) -> `+Inf` (`0x7F800000`)."),
        ],
    ),
    "FPCLASS": (
        "OpIsNan / OpIsInf", "bool b = isinf(x) || isnan(x);",
        [
            ("Classify infinity / NaN",
             fmt_asm_line("FPCLASS.f32", "", "r0", ["r1"]),
             "With `r1 = 0x7F800000` (`+Inf`): sets the hardware positive-infinity class bit in `r0`."),
            ("Classify subnormal vs zero",
             fmt_asm_line("FPCLASS.f32", "", "r0", ["r1"]),
             "With `r1 = 0x80000000` (`-0.0f`): sets the negative-zero bit distinct from negative-subnormal (`0x80000001`)."),
        ],
    ),
    "FPOW": (
        "GLSL.std.450 Pow (f16, v12+)", "float16_t z = pow(x, y);",
        [
            ("Scalar FP16 power with .h01 replication",
             fmt_asm_line("FPOW.f16", "", "r0.h01", ["r1.h0", "r2.h0"]),
             "With `r1.h0 = 2.0h, r2.h0 = 3.0h`: computes `2^3 = 8.0h` (`0x4800`) and `.h01` replicates -> `r0 = 0x48004800`."),
            ("Zero and negative base edge cases",
             fmt_asm_line("FPOW.f16", "", "r0.h0", ["r1.h0", "r2.h0"]),
             "With `r1.h0 = 4.0h, r2.h0 = 0.5h`: returns `2.0h` (`0x4000`); negative base with non-integer exponent returns `qNaN`."),
        ],
    ),
    "FRCP": (
        "OpFDiv (`1.0 / x`)", "float y = 1.0 / x;",
        [
            ("Normal FP32 reciprocal",
             fmt_asm_line("FRCP.f32", "", "r0", ["r1"]),
             "With `r1 = 4.0f`: returns `0.25f` (`0x3E800000`, `< 1 ULP`)."),
            ("Subnormal FTZ & signed zero",
             fmt_asm_line("FRCP.f32", "", "r0", ["r1"]),
             "With `r1 = -0.0f` (or any negative subnormal, flushed to `-0.0f` on input): returns `-Inf` (`0xFF800000`). With `r1 = +Inf`: returns `+0.0f`."),
        ],
    ),
    "FROUND": (
        "GLSL.std.450 RoundEven / Floor / Ceil / Trunc", "float y = floor(x);",
        [
            ("Floor of negative fractional float",
             fmt_asm_line("FROUND.f32", ".round_down", "r0", ["r1"]),
             "With `r1 = -1.25f`: `.round_down` returns `-2.0f` (`0xC0000000`); `-0.0f` preserves `-0.0f`."),
            ("Ties-to-even (RNE) vs ties-away (.round_na)",
             fmt_asm_line("FROUND.f32", ".round_na", "r0", ["r1"]),
             "With `r1 = 2.5f`: default RNE returns `2.0f` (even), whereas `.round_na` rounds ties away from zero -> `3.0f`."),
        ],
    ),
    "FRSQ": (
        "GLSL.std.450 InverseSqrt", "float y = inversesqrt(x);",
        [
            ("Exact square reciprocal",
             fmt_asm_line("FRSQ.f32", "", "r0", ["r1"]),
             "With `r1 = 4.0f`: returns `0.5f` (`0x3F000000`, `<= 1 ULP`)."),
            ("Zero and negative inputs",
             fmt_asm_line("FRSQ.f32", "", "r0", ["r1"]),
             "With `r1 = -0.0f` -> `-Inf` (`0xFF800000`); with `r1 = -1.0f` -> `qNaN`; with `r1 = +Inf` -> `+0.0f`."),
        ],
    ),
    "FSINCOS": (
        "GLSL.std.450 Sin / Cos (v14+)", "float y = sin(x);",
        [
            ("Hardware sine evaluation (v14+)",
             fmt_asm_line("FSINCOS.f32", ".sin", "r0", ["r1", "r2"]),
             "Evaluates hardware `sin` from range-reduced quadrant/angle inputs `r1, r2`: at `angle = pi/2` returns `+1.0f` (`0x3F800000`)."),
            ("Hardware cosine mode (.cos)",
             fmt_asm_line("FSINCOS.f32", ".cos", "r0", ["r1", "r2"]),
             "With `.cos` on `angle = 0.0f`: returns `+1.0f`; at `angle = pi` returns `-1.0f` (`0xBF800000`)."),
        ],
    ),
    "FSQRT": (
        "GLSL.std.450 Sqrt (v14+)", "float y = sqrt(x);",
        [
            ("Positive normal square root (v14+)",
             fmt_asm_line("FSQRT.f32", "", "r0", ["r1"]),
             "With `r1 = 9.0f`: returns `3.0f` (`0x40400000`); v9--v13 lower `sqrt(x)` to `FRSQ` + `FRCP`."),
            ("Signed zero & negative domain",
             fmt_asm_line("FSQRT.f32", "", "r0", ["r1"]),
             "With `r1 = -0.0f`: IEEE-754 requires `sqrt(-0.0) = -0.0f` (`0x80000000`). With `r1 = -4.0f` -> `qNaN`."),
        ],
    ),
    "LDEXP": (
        "GLSL.std.450 Ldexp", "float y = ldexp(x, exp);",
        [
            ("Standard power-of-two scaling",
             fmt_asm_line("LDEXP.f32", "", "r0", ["r1", "r2"]),
             "With `r1 = 1.5f, r2 = 3`: adds `3` to the exponent -> `1.5 * 2^3 = 12.0f` (`0x41400000`)."),
            ("Extreme-exponent override (.inf0)",
             fmt_asm_line("LDEXP.f32", ".inf0", "r0", ["r1", "r2"]),
             "With `r1 = +0.0f, r2 = 20000 (>= 16384)`: normal `ldexp(0, 20000)` is `0.0f`, but `.inf0` overrides it to `+1.0f` (`0x3F800000`)."),
        ],
    ),

    # 4. SFU Range-Reduction & Helpers
    "FADD_LSCALE": (
        "None (lowered from GLSL.std.450 Log2)", "float y = log2(x); // emits FADD_LSCALE(-1.0, x)",
        [
            ("Log2 mantissa reduction (`m - 1`)",
             fmt_asm_line("FADD_LSCALE.f32", "", "r0", ["k1f.neg", "r1"]),
             "With `r1 = 6.0f` (`1.5 * 2^2`): reduces `r1` into `[0.75, 1.5)` -> `0.75f`, adds `-1.0f` -> `r0 = -0.25f` (`0xBE800000`)."),
            ("Input already in `[0.75, 1.5)`",
             fmt_asm_line("FADD_LSCALE.f32", "", "r0", ["k1f.neg", "r1"]),
             "With `r1 = 1.25f`: stays `1.25f`, adds `-1.0f` -> `r0 = +0.25f` (`0x3E800000`), feeding the `log2` polynomial."),
        ],
    ),
    "FADD_RSCALE": (
        "None (scaled FP add)", "float y = (a + b) * 0.5; // scale = -1",
        [
            ("Exact FP64-widened midpoint",
             fmt_asm_line("FADD_RSCALE.f32", "", "r0", ["r1", "r2", "k1n"]),
             "With `r1 = 3.0f, r2 = 5.0f, scale = -1`: evaluates `(3 + 5) * 2^-1 = 4.0f` (`0x40800000`) without intermediate FP32 overflow."),
            ("Infinity cancellation override (.n_add)",
             fmt_asm_line("FADD_RSCALE.f32", ".n_add", "r0", ["r1", "r2", "k0"]),
             "Used in `expm1`: prevents `+Inf + (-Inf)` from producing NaN during range reduction when `.n_add` is set."),
        ],
    ),
    "FATAN_ASSIST": (
        "None (lowered from GLSL.std.450 Atan/Atan2)", "float y = atan(a, b);",
        [
            ("First-quadrant atan2 assist",
             fmt_asm_line("FATAN_ASSIST.f32", "", "r0", ["r1", "r2"]),
             "With `y = r1 = 1.0f, x = r2 = 2.0f`: packs octant/exponent assist metadata `(i16, f16)` into `r0` for `FATAN_TABLE`."),
            ("Steep slope (`|y| > |x|`) octant swap",
             fmt_asm_line("FATAN_ASSIST.f32", "", "r0", ["r1", "r2"]),
             "With `y = r1 = 4.0f, x = r2 = 1.0f`: records octant swap flag so `FATAN_TABLE` applies the `pi/2 - atan(x/y)` identity."),
        ],
    ),
    "FATAN_TABLE": (
        "None (lowered from GLSL.std.450 Atan/Atan2)", "float y = atan(x);",
        [
            ("Octant 0 base offset lookup",
             fmt_asm_line("FATAN_TABLE.f32", "", "r0", ["r1"]),
             "Given `r1` from `FATAN_ASSIST` in octant 0 (`0 <= y <= x`): returns base angle offset `0.0f`."),
            ("Octant 1/2 (`pi/2` or `pi`) offset lookup",
             fmt_asm_line("FATAN_TABLE.f32", "", "r0", ["r1"]),
             "Given `r1` in octant with `x < 0, y > 0`: returns `+pi` (`0x40490FDB`) or `+pi/2` (`0x3FC90FDB`) for quadrant reconstruction."),
        ],
    ),
    "FCOS_TABLE": (
        "None (lowered from GLSL.std.450 Cos/Sin)", "float y = cos(x);",
        [
            ("Table index 0 (`cos(0) = 1.0`)",
             fmt_asm_line("FCOS_TABLE.u6", "", "r0", ["r1"]),
             "With `r1 = 0x49400000` (`idx = 0`, exp `127+19`): returns `cos(0) = +1.0f` (`0x3F800000`)."),
            ("Table index 16 (`cos(pi/2) = 0.0`)",
             fmt_asm_line("FCOS_TABLE.u6", "", "r0", ["r1"]),
             "With `r1 = 0x49400010` (`idx = 16`): returns `cos(16*pi/32) = +0.0f` (`0x00000000`)."),
        ],
    ),
    "FEXP_TABLE": (
        "None (legacy SFU exp2 table)", "// Legacy 4-bit exp2 LUT helper",
        [
            ("4-bit fractional `2^(idx/16)` lookup",
             fmt_asm_line("FEXP_TABLE.u4", "", "r0", ["r1"]),
             "With `r1 & 0xF = 8`: looks up `2^(8/16) = sqrt(2) ~= 1.4142135f` (`0x3FB504F3`)."),
            ("Adjusted low-precision entry (.low)",
             fmt_asm_line("FEXP_TABLE.u4", ".low", "r0", ["r1"]),
             "With `.low` and `r1 & 0xF = 0`: returns the low-word correction term for extended-precision `exp2` reconstruction."),
        ],
    ),
    "FLOGD": (
        "None (lowered from GLSL.std.450 Log2)", "float y = log2(x);",
        [
            ("Reduced mantissa `m = 1.0f`",
             fmt_asm_line("FLOGD.f32", "", "r0", ["r1"]),
             "With `r1 = 1.0f`: evaluates `lim_{m->1} log2(m)/(m - 1) = 1/ln(2) ~= 1.442695f` (`0x3FB8AA3B`)."),
            ("Upper interval edge `m = 1.5f`",
             fmt_asm_line("FLOGD.f32", "", "r0", ["r1"]),
             "With `r1 = 1.5f` (or `6.0f`, reduced to `0.75f`): evaluates 16-entry cubic LUT `log2(m)/(m - 1)` for `FMA` with `FADD_LSCALE`."),
        ],
    ),
    "FLOG_TABLE": (
        "None (legacy SFU log table)", "// Legacy log LUT helper",
        [
            ("Base-2 log LUT entry",
             fmt_asm_line("FLOG_TABLE.f32", ".base2", "r0", ["r1"]),
             "Indexes the base-2 logarithm coefficient table using the top mantissa bits of `r1`."),
            ("Natural log high-precision LUT entry",
             fmt_asm_line("FLOG_TABLE.f32", ".natural.high", "r0", ["r1"]),
             "Selects the high-precision natural-log (`ln`) table coefficient for `log(x)` lowering."),
        ],
    ),
    "FMA_RSCALE": (
        "None (scaled FMA for Exp2/Sin/Cube)", "float y = fma(a, b, c) * exp2(float(s));",
        [
            ("Overflow-free intermediate product scaling",
             fmt_asm_line("FMA_RSCALE.f32", "", "r0", ["r1", "r2", "r3", "r4"]),
             "With `r1 = 2^100, r2 = 2^50, r3 = 0, r4 = -150`: single-rounded FP64 evaluates `2^150 * 2^-150 = 1.0f` without intermediate FP32 overflow."),
            ("Special `0 * Inf` override (.n)",
             fmt_asm_line("FMA_RSCALE.f32", ".n", "r0", ["k0", "r_inf", "k1f", "k0"]),
             "With `0.0 * +Inf + 1.0`: standard FMA yields `NaN`, whereas `.n` forces the product to `0.0`, returning `1.0f`."),
        ],
    ),
    "FMUL_CSLICE": (
        "None (SFU trig range reduction)", "// Hardware trig slice helper",
        [
            ("Cosine Payne-Hanek slice product",
             fmt_asm_line("FMUL_CSLICE", "", "r0", ["r1.h0", "r2"]),
             "Computes the high-order cosine range-reduction slice product of `r1.h0` and `r2` for extended-precision trig."),
            ("Upper halfword slice selection",
             fmt_asm_line("FMUL_CSLICE", "", "r0", ["r1.h1", "r2"]),
             "Selects `r1.h1` as the second chunk in a multi-slice `2/pi` reduction sequence."),
        ],
    ),
    "FMUL_RSCALE": (
        "None (scaled FP mul for Sin/Pow)", "float y = (a * b) * 0.5;",
        [
            ("Exact FP64-widened scaled multiply",
             fmt_asm_line("FMUL_RSCALE.f32", "", "r0", ["r1", "r2", "k1n"]),
             "With `r1 = 2^80, r2 = 2^60, scale = -140`: produces exact `1.0f` (`0x3F800000`) despite `r1 * r2 = 2^140 > FLT_MAX`."),
            ("Left-operand preservation on `0 * Inf` (.left)",
             fmt_asm_line("FMUL_RSCALE.f32", ".left", "r0", ["r1", "r2", "k1n"]),
             "With `r1 = 0.0f, r2 = +Inf`: `.left` returns `r1` (`0.0f`) instead of `NaN`."),
        ],
    ),
    "FMUL_SLICE": (
        "None (SFU trig range reduction)", "// Hardware Payne-Hanek slice helper",
        [
            ("Payne-Hanek `2/pi` slice multiply",
             fmt_asm_line("FMUL_SLICE.f32", "", "r0", ["r1", "r2"]),
             "Extracts a partial product slice of `x * (2/pi)` for large-angle `sin(x)` / `cos(x)` range reduction."),
            ("Negated slice accumulation",
             fmt_asm_line("FMUL_SLICE.f32", "", "r0", ["r1", "r2.neg"]),
             "Negates `r2` in Stage A to compute the residual subtraction term during multi-word trig reduction."),
        ],
    ),
    "FPOW_SC_APPLY": (
        "None (lowered from GLSL.std.450 Pow)", "float z = pow(x, y);",
        [
            ("Normal path passthrough",
             fmt_asm_line("FPOW_SC_APPLY", "", "r0", ["r1", "r2"]),
             "When `r2` from `FPOW_SC_DET` indicates no special case: passes computed `exp2(y * log2(x))` in `r1` straight through to `r0`."),
            ("Special-case override (`pow(0.0, 0.0) = 1.0`)",
             fmt_asm_line("FPOW_SC_APPLY", "", "r0", ["r1", "r2"]),
             "When `FPOW_SC_DET` flags `x == 0, y == 0`: overrides `r1` (`NaN`) with `r0 = 1.0f` (`0x3F800000`)."),
        ],
    ),
    "FPOW_SC_DET": (
        "None (lowered from GLSL.std.450 Pow)", "float z = pow(x, y);",
        [
            ("GLSL/OpenCL `pow` mode (.pow)",
             fmt_asm_line("FPOW_SC_DET.f32", ".pow", "r0", ["r1", "r2"]),
             "With `r1 = -2.0f, r2 = 3.0f` (odd integer): detects negative-base odd-integer sign flip for `FPOW_SC_APPLY`."),
            ("OpenCL `powr` mode (.powr)",
             fmt_asm_line("FPOW_SC_DET.f32", ".powr", "r0", ["r1", "r2"]),
             "With `r1 = 0.0f, r2 = 0.0f`: under `.powr`, flags `NaN` (unlike `.pow`, which returns `1.0f`)."),
        ],
    ),
    "FRCBRT_APPROX_A": (
        "None (SFU cube-root seed A)", "// Compiler cbrt approximation stage A",
        [
            ("Initial `x^(-1/3)` seed for `x = 8.0f`",
             fmt_asm_line("FRCBRT_APPROX_A.f32", "", "r0", ["r1"]),
             "With `r1 = 8.0f`: returns initial reciprocal cube-root seed `~0.5f` for Newton-Raphson refinement."),
            ("Negative input preservation",
             fmt_asm_line("FRCBRT_APPROX_A.f32", "", "r0", ["r1"]),
             "With `r1 = -27.0f`: produces negative seed `~-0.333f` preserving sign of the cube root."),
        ],
    ),
    "FRCBRT_APPROX_B": (
        "None (SFU cube-root seed B)", "// Compiler cbrt approximation stage B",
        [
            ("Second-stage cbrt polynomial refinement",
             fmt_asm_line("FRCBRT_APPROX_B.f32", "", "r0", ["r1"]),
             "Refines the exponent/mantissa seed from `FRCBRT_APPROX_A` to increase bit accuracy before final FMA steps."),
            ("Subnormal/edge handling",
             fmt_asm_line("FRCBRT_APPROX_B.f32", "", "r0", ["r1"]),
             "Handles normalized intermediate from stage A across all finite FP32 exponents."),
        ],
    ),
    "FRCBRT_APPROX_C": (
        "None (SFU cube-root seed C)", "// Compiler cbrt approximation stage C",
        [
            ("Third-stage cbrt refinement",
             fmt_asm_line("FRCBRT_APPROX_C.f32", "", "r0", ["r1"]),
             "Completes the 3-stage hardware reciprocal cube-root seed pipeline (`A -> B -> C`)."),
            ("Final FMA pairing",
             fmt_asm_line("FRCBRT_APPROX_C.f32", "", "r0", ["r1"]),
             "Outputs the final refined reciprocal cube-root approximation ready for `FMUL` to recover `cbrt(x) = x * x^(-2/3)`."),
        ],
    ),
    "FRCP_APPROX": (
        "None (SFU reciprocal seed)", "// Newton-Raphson 1/x seed",
        [
            ("Initial Newton-Raphson seed for `1/4.0`",
             fmt_asm_line("FRCP_APPROX.f32", "", "r0", ["r1"]),
             "With `r1 = 4.0f`: returns fast low-precision estimate `~0.25f` used as `y0` in `y1 = y0 * (2 - x * y0)`."),
            ("FP64 division lowering",
             fmt_asm_line("FRCP_APPROX.f32", "", "r0", ["r1"]),
             "Provides the ~12-bit seed for emulated FP64 reciprocal before 3 Newton-Raphson iterations."),
        ],
    ),
    "FREXPE": (
        "GLSL.std.450 Frexp (exponent)", "int e; float m = frexp(x, e);",
        [
            ("Standard vs `.log` exponent extraction",
             fmt_asm_line("FREXPE.f32", ".log", "r0", ["r1"]),
             "With `r1 = 3.0f` (`0x40400000`, biased `e = 128`, mantissa `1.5`): default returns `2` (`0.75 * 2^2`); `.log` returns `2` (`0.75 * 2^2` in `[0.75, 1.5)`)."),
            ("Square-root even exponent with `.neg`",
             fmt_asm_line("FREXPE.f32", ".sqrt.neg", "r0", ["r1"]),
             "With `r1 = 16.0f` (`2^4`): `.sqrt` halves exponent to `2` (`4^2`), and `.neg` negates -> `r0 = -2` (for `inversesqrt`)."),
        ],
    ),
    "FREXPM": (
        "GLSL.std.450 Frexp (mantissa)", "int e; float m = frexp(x, e);",
        [
            ("Standard `[0.5, 1.0)` mantissa",
             fmt_asm_line("FREXPM.f32", "", "r0", ["r1"]),
             "With `r1 = 3.0f`: returns `r0 = 0.75f` (`0x3F400000`), paired with `FREXPE = 2` since `0.75 * 2^2 = 3.0`."),
            ("DDK `.log` infinity override",
             fmt_asm_line("FREXPM.f32", ".log", "r0", ["r1"]),
             "With `r1 = +Inf` (`0x7F800000`): default returns `+Inf`, whereas `.log` returns `+1.0f` (`0x3F800000`) and `-Inf -> -1.0f`."),
        ],
    ),
    "FRSQ_APPROX": (
        "None (SFU rsqrt seed)", "// Newton-Raphson 1/sqrt(x) seed",
        [
            ("Initial `1/sqrt(4.0)` seed",
             fmt_asm_line("FRSQ_APPROX.f32", "", "r0", ["r1"]),
             "With `r1 = 4.0f`: returns low-precision seed `~0.5f` for Newton-Raphson `y1 = y0 * (1.5 - 0.5 * x * y0^2)`."),
            ("Vector normalization fast path",
             fmt_asm_line("FRSQ_APPROX.f32", "", "r0", ["r1"]),
             "Used as the initial estimate for software-refined FP64 `inversesqrt`."),
        ],
    ),
    "FSINCOS_OFFSET": (
        "None (SFU trig offset LUT)", "// Extended trig offset lookup",
        [
            ("Scaled quadrant offset (.scale)",
             fmt_asm_line("FSINCOS_OFFSET.u6", ".scale", "r0", ["r1"]),
             "Reads 8-entry quadrant-symmetric angle offset LUT for `r1 & 0x3F`; `.scale` adds `+24` to the FP32 exponent."),
            ("Unscaled base offset",
             fmt_asm_line("FSINCOS_OFFSET.u6", "", "r0", ["r1"]),
             "Returns the unscaled quadrant offset constant directly for `FMA` angle reconstruction."),
        ],
    ),
    "FSIN_TABLE": (
        "None (lowered from GLSL.std.450 Sin/Cos)", "float y = sin(x);",
        [
            ("Index 16 (`sin(pi/2) = +1.0`)",
             fmt_asm_line("FSIN_TABLE.u6", "", "r0", ["r1"]),
             "With `r1 = 0x49400010` (`idx = 16`, exp `127+19`): returns `sin(16*pi/32) = +1.0f` (`0x3F800000`)."),
            ("Index 0 signed-zero edge case",
             fmt_asm_line("FSIN_TABLE.u6", "", "r0", ["r1"]),
             "With `r1 = 0x49400000` (`idx = 0`): returns `-0.0f` (`0x80000000`) so `FMA` preserves `sin(-0.0) = -0.0`."),
        ],
    ),
    "ILOGB": (
        "None (OpenCL `ilogb`, v9--v10)", "// OpenCL int e = ilogb(x);",
        [
            ("Normal positive float (v9--v10)",
             fmt_asm_line("ILOGB.f32", "", "r0", ["r1"]),
             "With `r1 = 6.0f` (`1.5 * 2^2`): returns integer unbiased exponent `floor(log2(6.0)) = 2`."),
            ("Special values (`0`, `Inf`, `NaN`)",
             fmt_asm_line("ILOGB.f32", "", "r0", ["r1"]),
             "With `r1 = ±0.0f` -> `FP_ILOGB0` (`0x80000000`); with `r1 = ±Inf` -> `INT32_MAX` (`0x7FFFFFFF`); removed in v11."),
        ],
    ),
    "LOGB": (
        "None (OpenCL `logb`, v9--v10)", "// OpenCL float e = logb(x);",
        [
            ("Normal float unbiased exponent (v9--v10)",
             fmt_asm_line("LOGB.f32", "", "r0", ["r1"]),
             "With `r1 = 6.0f`: returns float unbiased exponent `2.0f` (`0x40000000`)."),
            ("Zero and infinity",
             fmt_asm_line("LOGB.f32", "", "r0", ["r1"]),
             "With `r1 = ±0.0f` -> `-Inf` (`0xFF800000`); with `r1 = ±Inf` -> `+Inf` (`0x7F800000`); removed in v11."),
        ],
    ),

    # 5. Cubemap Helpers
    "CUBEFACE1": (
        "None (lowered from OpImageSample* Cube, v9--v10)", "vec4 c = texture(cubeSamp, dir);",
        [
            ("Major-axis magnitude extraction",
             fmt_asm_line("CUBEFACE1", "", "r0", ["r1", "r2", "r3"]),
             "With `(x, y, z) = (1.0, -4.0, 2.0)`: returns `max(|x|, |y|, |z|) = 4.0f` (`0x40800000`) for `FRCP` projection."),
            ("Infinity and zero clamping",
             fmt_asm_line("CUBEFACE1", "", "r0", ["r1", "r2", "r3"]),
             "If any component is `±Inf`, returns `1.0f`; if all components are `±0.0`/subnormal, returns `2.0f` to avoid division by zero."),
        ],
    ),
    "CUBEFACE2": (
        "None (lowered from OpImageSample* Cube, v9--v10)", "vec4 c = texture(cubeSamp, dir);",
        [
            ("`-Z` face index packing",
             fmt_asm_line("CUBEFACE2", "", "r0", ["r1", "r2", "r3"]),
             "With `(0.0, 0.0, -1.0)`: selects face 5 (`-Z`) -> `r0 = (5 << 29) = 0xA0000000`."),
            ("`+X` face with infinity flag",
             fmt_asm_line("CUBEFACE2", "", "r0", ["r1", "r2", "r3"]),
             "With `(+Inf, 1.0, 0.0)`: selects face 0 (`+X`) and sets bit 16 (`0x00010000`) to signal an infinite major axis."),
        ],
    ),
    "CUBE_SSEL": (
        "None (lowered from OpImageSample* Cube, v9--v10)", "vec4 c = texture(cubeSamp, dir);",
        [
            ("`+X` face horizontal coordinate (`-Z`)",
             fmt_asm_line("CUBE_SSEL", "", "r0", ["r3", "r1", "r4"]),
             "Operands `(Z, X, face)`: with `face = 0` (`+X`, `r4 = 0x00000000`), selects and negates `Z` -> `r0 = -r3`."),
            ("`-Z` face horizontal coordinate (`-X`)",
             fmt_asm_line("CUBE_SSEL", "", "r0", ["r3", "r1", "r4"]),
             "With `face = 5` (`-Z`, `r4 = 0xA0000000`): selects and negates `X` -> `r0 = -r1`."),
        ],
    ),
    "CUBE_TSEL": (
        "None (lowered from OpImageSample* Cube, v9--v10)", "vec4 c = texture(cubeSamp, dir);",
        [
            ("`+Y` face vertical coordinate (`+Z`)",
             fmt_asm_line("CUBE_TSEL", "", "r0", ["r2", "r3", "r4"]),
             "Operands `(Y, Z, face)`: with `face = 2` (`+Y`, `r4 = 0x40000000`), returns `+Z` (`r0 = r3`)."),
            ("`+X` face vertical coordinate (`-Y`)",
             fmt_asm_line("CUBE_TSEL", "", "r0", ["r2", "r3", "r4"]),
             "With `face = 0` (`+X`, `r4 = 0x00000000`): returns `-Y` (`r0 = -r2`)."),
        ],
    ),

    # 6. Compare, Select & Mux
    "CSEL": (
        "OpSelect + OpFOrd*/OpS*/OpU*", "float r = (a < b) ? x : y;",
        [
            ("Floating-point conditional select (true branch)",
             fmt_asm_line("CSEL.f32", ".lt", "r0", ["r1", "r2", "r3", "r4"]),
             "With `r1 = 1.0f, r2 = 2.0f, r3 = 0xAAAA, r4 = 0xBBBB`: `1.0 < 2.0` is true -> selects `r0 = 0xAAAA`."),
            ("NaN comparison falls through to false branch",
             fmt_asm_line("CSEL.f32", ".lt", "r0", ["r1", "r2", "r3", "r4"]),
             "With `r1 = NaN, r2 = 2.0f`: ordered `.lt` evaluates to false -> selects `r0 = r4` (`0xBBBB`)."),
        ],
    ),
    "FCMP": (
        "OpFOrdEqual/LessThan/... / OpFUnordNotEqual", "bool b = (a != b);",
        [
            ("Unordered `.ne` on NaN with `.m1` bitmask",
             fmt_asm_line("FCMP.f32", ".ne.m1", "r0", ["r1", "r2"]),
             "With `r1 = NaN, r2 = 1.0f`: `.ne` is unordered (True on NaN), `.m1` formats True as `r0 = 0xFFFFFFFF`."),
            ("Ordered `.gtlt` on NaN with `.f1` float boolean",
             fmt_asm_line("FCMP.f32", ".gtlt.f1", "r0", ["r1", "r2"]),
             "With `r1 = NaN, r2 = 1.0f`: `.gtlt` is ordered (False on NaN) -> `r0 = 0.0f`; for `(2.0f, 1.0f)` -> `r0 = 1.0f` (`0x3F800000`)."),
        ],
    ),
    "FCMP_AND": (
        "OpLogicalAnd + OpFOrd*", "bool b = prev && (a < b);",
        [
            ("Short-circuit chain with false accumulator",
             fmt_asm_line("FCMP_AND.f32", ".lt.i1", "r0", ["r1", "r2", "r3"]),
             "With `r1 = 1.0f, r2 = 2.0f, r3 = 0`: `(1.0 < 2.0) && (0 != 0)` is False -> `r0 = 0`."),
            ("Both predicates true with `.m1` mask",
             fmt_asm_line("FCMP_AND.f32", ".le.m1", "r0", ["r1", "r2", "r3"]),
             "With `r1 = 2.0f, r2 = 2.0f, r3 = 0xFFFFFFFF`: `(2.0 <= 2.0) && true` is True -> `r0 = 0xFFFFFFFF`."),
        ],
    ),
    "FCMP_OR": (
        "OpLogicalOr + OpFOrd*", "bool b = prev || (a > b);",
        [
            ("Previous condition true overrides false comparison",
             fmt_asm_line("FCMP_OR.f32", ".gt.i1", "r0", ["r1", "r2", "r3"]),
             "With `r1 = 1.0f, r2 = 2.0f, r3 = 1`: `(1.0 > 2.0) || (1 != 0)` is True -> `r0 = 1`."),
            ("Chained range check (both false)",
             fmt_asm_line("FCMP_OR.f32", ".gt.i1", "r0", ["r1", "r2", "r3"]),
             "With `r1 = 1.0f, r2 = 2.0f, r3 = 0`: `(1.0 > 2.0) || false` is False -> `r0 = 0`."),
        ],
    ),
    "ICMP": (
        "OpIEqual / OpSLessThan / OpULessThan / ...", "bool b = (a >= b);",
        [
            ("Signed comparison with negative value",
             fmt_asm_line("ICMP.s32", ".ge.i1", "r0", ["r1", "r2"]),
             "With `r1 = 0xFFFFFFFF` (`-1`), `r2 = 0`: signed `-1 >= 0` is False -> `r0 = 0`."),
            ("Unsigned comparison on same bit pattern",
             fmt_asm_line("ICMP.u32", ".ge.m1", "r0", ["r1", "r2"]),
             "With `r1 = 0xFFFFFFFF` (`4294967295u`), `r2 = 0`: unsigned `>=` is True -> `r0 = 0xFFFFFFFF` under `.m1`."),
        ],
    ),
    "ICMP_AND": (
        "OpLogicalAnd + OpI*/OpS*/OpU*", "bool b = prev && (a == b);",
        [
            ("Chained equality (both true)",
             fmt_asm_line("ICMP_AND.u32", ".eq.m1", "r0", ["r1", "r2", "r3"]),
             "With `r1 = 5, r2 = 5, r3 = 1`: `(5 == 5) && (1 != 0)` is True -> `r0 = 0xFFFFFFFF`."),
            ("64-bit equality high-word step (false)",
             fmt_asm_line("ICMP_AND.u32", ".eq.i1", "r0", ["r1", "r2", "r3"]),
             "With `r1 = 0, r2 = 1, r3 = 1` (low words matched, high words differ): returns `r0 = 0`."),
        ],
    ),
    "ICMP_MULTI": (
        "OpULessThan / OpSLessThan (64-bit chained)", "bool b = (a64 < b64);",
        [
            ("64-bit unsigned `<` low-word step (`.c` 3-way carry)",
             fmt_asm_line("ICMP_MULTI.u32", ".lt.c", "r0", ["r1", "r2", "k0"]),
             "With `r1 = 3, r2 = 5, acc = 0`: `3 < 5` outputs 3-way carry `r0 = -1` (`0xFFFFFFFF`) for the high-word step."),
            ("64-bit unsigned `<` high-word tie-breaker",
             fmt_asm_line("ICMP_MULTI.u32", ".lt.i1", "r0", ["r3", "r4", "r0"]),
             "With equal high words `r3 = 7, r4 = 7` and low-word carry `r0 = -1`: tie-breaks using carry `< 0` -> returns `r0 = 1` (`true`)."),
        ],
    ),
    "ICMP_OR": (
        "OpLogicalOr + OpI*/OpS*/OpU*", "bool b = prev || (a != b);",
        [
            ("Both false",
             fmt_asm_line("ICMP_OR.u32", ".ne.m1", "r0", ["r1", "r2", "r3"]),
             "With `r1 = 4, r2 = 4, r3 = 0`: `(4 != 4) || (0 != 0)` is False -> `r0 = 0`."),
            ("64-bit inequality (`!=`) high-word step",
             fmt_asm_line("ICMP_OR.u32", ".ne.i1", "r0", ["r1", "r2", "r3"]),
             "With `r1 = 0, r2 = 0, r3 = 1` (low words differed): `(0 != 0) || (1 != 0)` is True -> `r0 = 1`."),
        ],
    ),
    "MUX": (
        "OpSelect / OpenCL `bitselect`", "uint r = (a & mask) | (b & ~mask);",
        [
            ("Per-bit blend (`.bit`)",
             fmt_asm_line("MUX.i32", ".bit", "r0", ["r1", "r2", "r3"]),
             "With `r1 = 0x12345678, r2 = 0xA0000000, r3 = 0x1FFFFFFF`: takes bits `[28:0]` from `r1` and `[31:29]` from `r2` -> `r0 = 0xB2345678`."),
            ("Default `.int_zero` selector (encoding 1)",
             fmt_asm_line("MUX.i32", ".int_zero", "r0", ["r1", "r2", "r3"]),
             "With `r1 = 0x1111, r2 = 0x2222, r3 = 0`: `r3 == 0` selects `src0` -> `r0 = 0x1111` (for `r3 != 0`, selects `src1 = 0x2222`)."),
        ],
    ),

    # 7. Conversions
    "F16_TO_F32": (
        "OpFConvert (f16 -> f32) / UnpackHalf2x16", "float y = float(x16);",
        [
            ("Upper halfword FP16 -> FP32",
             fmt_asm_line("F16_TO_F32", "", "r0", ["r1.h1"]),
             "With `r1 = 0x3C000000`: extracts `h1 = 0x3C00` (`1.0h`) -> exact `r0 = 0x3F800000` (`1.0f`)."),
            ("FP16 subnormal exact promotion",
             fmt_asm_line("F16_TO_F32", "", "r0", ["r1.h0"]),
             "With `r1.h0 = 0x0001` (`2^-24`, smallest FP16 subnormal): normalizes exactly in FP32 -> `r0 = 0x33800000` (`2^-24f`)."),
        ],
    ),
    "F16_TO_S32": (
        "OpConvertFToS (f16 -> s32, v9--v10)", "int y = int(x16);",
        [
            ("Negative half-float truncation (v9--v10)",
             fmt_asm_line("F16_TO_S32", ".round_zero", "r0", ["r1.h0"]),
             "With `r1.h0 = 0xC280` (`-3.25h`): `.round_zero` truncates toward zero -> `r0 = -3` (`0xFFFFFFFD`)."),
            ("NaN and infinity saturation",
             fmt_asm_line("F16_TO_S32", ".round_zero", "r0", ["r1.h0"]),
             "With `r1.h0 = 0x7E00` (`NaN`) -> `r0 = 0`; with `r1.h0 = 0xFC00` (`-Inf`) -> `r0 = 0x80000000` (`INT32_MIN`)."),
        ],
    ),
    "F16_TO_U32": (
        "OpConvertFToU (f16 -> u32, v9--v10)", "uint y = uint(x16);",
        [
            ("Positive half-float conversion (v9--v10)",
             fmt_asm_line("F16_TO_U32", ".round_zero", "r0", ["r1.h0"]),
             "With `r1.h0 = 0x4580` (`5.5h`): truncates to `r0 = 5`."),
            ("Negative / NaN clamp to 0",
             fmt_asm_line("F16_TO_U32", ".round_zero", "r0", ["r1.h0"]),
             "With `r1.h0 = 0xC000` (`-2.0h`) or `NaN`: clamps to `r0 = 0`; `+Inf` saturates to `0xFFFFFFFF`."),
        ],
    ),
    "F32_TO_F16": (
        "OpFConvert (f32 -> f16, v11+)", "float16_t y = float16_t(x);",
        [
            ("Upper-half write preserving low half (v11+)",
             fmt_asm_line("F32_TO_F16", "", "r0.h1", ["r1"]),
             "With `r1 = 1.0f`, `r0_old = 0x0000BEEF`: converts `1.0f -> 0x3C00` in upper half -> `r0 = 0x3C00BEEF`."),
            ("Pre-narrowing clamp + broadcast (`.clamp_0_1` + `.h01`)",
             fmt_asm_line("F32_TO_F16", ".clamp_0_1", "r0.h01", ["r1"]),
             "With `r1 = 100000.0f` (`> FP16 MAX 65504`): `.clamp_0_1` runs BEFORE narrowing (clamping to `1.0f`, not `+Inf`) -> `r0 = 0x3C003C00`."),
        ],
    ),
    "F32_TO_S32": (
        "OpConvertFToS", "int y = int(x);",
        [
            ("Truncate toward zero (GLSL `int(x)`)",
             fmt_asm_line("F32_TO_S32", ".round_zero", "r0", ["r1"]),
             "With `r1 = -3.9f`: `.round_zero` truncates to `r0 = -3` (`0xFFFFFFFD`)."),
            ("NaN & out-of-range saturation",
             fmt_asm_line("F32_TO_S32", ".round_zero", "r0", ["r1"]),
             "With `r1 = NaN` -> `r0 = 0`; with `r1 = +Inf` (or `>= 2^31`) -> `r0 = 0x7FFFFFFF`; with `r1 = -Inf` -> `r0 = 0x80000000`."),
        ],
    ),
    "F32_TO_U32": (
        "OpConvertFToU", "uint y = uint(x);",
        [
            ("Positive float truncation",
             fmt_asm_line("F32_TO_U32", ".round_zero", "r0", ["r1"]),
             "With `r1 = 42.8f`: truncates to `r0 = 42` (`0x0000002A`)."),
            ("Negative, NaN, and overflow saturation",
             fmt_asm_line("F32_TO_U32", ".round_zero", "r0", ["r1"]),
             "With `r1 = -1.5f` or `NaN` -> `r0 = 0`; with `r1 = 1e20f` or `+Inf` -> `r0 = 0xFFFFFFFF`."),
        ],
    ),
    "S16_TO_F32": (
        "OpConvertSToF (s16 -> f32, v9--v10)", "float y = float(int16_t(x));",
        [
            ("Negative 16-bit integer (v9--v10)",
             fmt_asm_line("S16_TO_F32", "", "r0", ["r1.h0"]),
             "With `r1.h0 = 0xFFFE` (`-2`): exact conversion -> `r0 = -2.0f` (`0xC0000000`)."),
            ("Upper halfword `INT16_MIN`",
             fmt_asm_line("S16_TO_F32", "", "r0", ["r1.h1"]),
             "With `r1 = 0x80000000` (`h1 = -32768`): exact conversion -> `r0 = -32768.0f` (`0xC7000000`)."),
        ],
    ),
    "S16_TO_S32": (
        "OpSConvert (s16 -> s32)", "int y = int(int16_t(x));",
        [
            ("Sign-extend negative upper halfword",
             fmt_asm_line("S16_TO_S32", "", "r0", ["r1.h1"]),
             "With `r1 = 0x80010000`: sign-extends `0x8001` (`-32767`) -> `r0 = 0xFFFF8001`."),
            ("Positive low halfword",
             fmt_asm_line("S16_TO_S32", "", "r0", ["r1.h0"]),
             "With `r1 = 0xFFFF7FFF`: sign-extends `h0 = 0x7FFF` (`+32767`) -> `r0 = 0x00007FFF`."),
        ],
    ),
    "S32_TO_F32": (
        "OpConvertSToF", "float y = float(x);",
        [
            ("Exact small negative integer",
             fmt_asm_line("S32_TO_F32", "", "r0", ["r1"]),
             "With `r1 = -10` (`0xFFFFFFF6`): exact conversion -> `r0 = -10.0f` (`0xC1200000`)."),
            ("Inexact 32-bit integer rounding",
             fmt_asm_line("S32_TO_F32", ".round_zero", "r0", ["r1"]),
             "With `r1 = 0x01000001` (`2^24 + 1`, exceeds 24-bit mantissa): `.round_zero` rounds down to `16777216.0f` (`0x4B800000`)."),
        ],
    ),
    "S8_TO_F32": (
        "OpConvertSToF (s8 -> f32, v9--v10)", "float y = float(int8_t(x));",
        [
            ("Negative byte 2 extraction (v9--v10)",
             fmt_asm_line("S8_TO_F32", "", "r0", ["r1.b2"]),
             "With `r1 = 0x00FC0000` (`b2 = -4`): exact conversion -> `r0 = -4.0f` (`0xC0800000`)."),
            ("Positive byte 0 extraction",
             fmt_asm_line("S8_TO_F32", "", "r0", ["r1.b0"]),
             "With `r1 = 0x0000007F` (`b0 = +127`): exact conversion -> `r0 = 127.0f` (`0x42FE0000`)."),
        ],
    ),
    "S8_TO_S32": (
        "OpSConvert (s8 -> s32)", "int y = int(int8_t(x));",
        [
            ("Negative byte sign extension",
             fmt_asm_line("S8_TO_S32", "", "r0", ["r1.b0"]),
             "With `r1.b0 = 0xFE` (`-2`): sign-extends bit 7 across `[31:8]` -> `r0 = 0xFFFFFFFE`."),
            ("Byte 3 positive extension",
             fmt_asm_line("S8_TO_S32", "", "r0", ["r1.b3"]),
             "With `r1 = 0x7F000000`: sign-extends `0x7F` (`+127`) -> `r0 = 0x0000007F`."),
        ],
    ),
    "U16_TO_F32": (
        "OpConvertUToF (u16 -> f32, v9--v10)", "float y = float(uint16_t(x));",
        [
            ("Low halfword unsigned -> FP32 (v9--v10)",
             fmt_asm_line("U16_TO_F32", "", "r0", ["r1.h0"]),
             "With `r1.h0 = 0x0100` (`256`): exact conversion -> `r0 = 256.0f` (`0x43800000`)."),
            ("Max uint16 (`65535`)",
             fmt_asm_line("U16_TO_F32", "", "r0", ["r1.h1"]),
             "With `r1 = 0xFFFF0000`: exact conversion (fits in 24-bit mantissa) -> `r0 = 65535.0f` (`0x477FFF00`)."),
        ],
    ),
    "U16_TO_U32": (
        "OpUConvert (u16 -> u32)", "uint y = uint(uint16_t(x));",
        [
            ("Zero-extend upper halfword",
             fmt_asm_line("U16_TO_U32", "", "r0", ["r1.h1"]),
             "With `r1 = 0xABCD1234`: extracts `h1` and zero-extends -> `r0 = 0x0000ABCD`."),
            ("Zero-extend low halfword (masks upper bits)",
             fmt_asm_line("U16_TO_U32", "", "r0", ["r1.h0"]),
             "With `r1 = 0xFFFF8000`: zero-extends `0x8000` -> `r0 = 0x00008000` (not sign-extended)."),
        ],
    ),
    "U32_TO_F32": (
        "OpConvertUToF", "float y = float(u);",
        [
            ("Exact unsigned integer",
             fmt_asm_line("U32_TO_F32", "", "r0", ["r1"]),
             "With `r1 = 100`: returns `100.0f` (`0x42C80000`)."),
            ("`UINT32_MAX` rounding under RNE vs RTZ",
             fmt_asm_line("U32_TO_F32", ".round_zero", "r0", ["r1"]),
             "With `r1 = 0xFFFFFFFF`: default RNE rounds up to `2^32` (`0x4F800000`), whereas `.round_zero` rounds down to `0x4F7FFFFF`."),
        ],
    ),
    "U8_TO_F32": (
        "OpConvertUToF / UnpackUnorm4x8 (v9--v10)", "vec4 v = unpackUnorm4x8(u);",
        [
            ("Byte 3 unsigned -> FP32 (v9--v10)",
             fmt_asm_line("U8_TO_F32", "", "r0", ["r1.b3"]),
             "With `r1 = 0xFF000000`: converts byte 3 (`255u`) -> `r0 = 255.0f` (`0x437F0000`)."),
            ("Byte 0 zero conversion",
             fmt_asm_line("U8_TO_F32", "", "r0", ["r1.b0"]),
             "With `r1 = 0xFFFFFF00`: converts byte 0 (`0u`) -> `r0 = +0.0f` (`0x00000000`)."),
        ],
    ),
    "U8_TO_U32": (
        "OpUConvert (u8 -> u32)", "uint y = uint(uint8_t(x));",
        [
            ("Zero-extend byte 1",
             fmt_asm_line("U8_TO_U32", "", "r0", ["r1.b1"]),
             "With `r1 = 0x0000AB00`: zero-extends byte 1 (`0xAB`) -> `r0 = 0x000000AB`."),
            ("Zero-extend byte 3 (MSB set)",
             fmt_asm_line("U8_TO_U32", "", "r0", ["r1.b3"]),
             "With `r1 = 0xFE000000`: zero-extends `0xFE` -> `r0 = 0x000000FE` (`254`, unlike `S8_TO_S32`)."),
        ],
    ),
    "V2F16_TO_V2S16": (
        "OpConvertFToS (v2f16 -> v2s16, v9--v10)", "i16vec2 y = i16vec2(x);",
        [
            ("Per-lane truncation toward zero (v9--v10)",
             fmt_asm_line("V2F16_TO_V2S16", ".round_zero", "r0", ["r1.h01"]),
             "With `r1 = [2.5h, -1.5h]`: truncates each halfword -> `r0 = [2, -1]` (`0x0002FFFF`)."),
            ("Overflow and NaN saturation",
             fmt_asm_line("V2F16_TO_V2S16", ".round_zero", "r0", ["r1.h01"]),
             "With `r1 = [+Inf, NaN]`: `+Inf` saturates to `INT16_MAX` (`0x7FFF`), `NaN` becomes `0` -> `r0 = 0x7FFF0000`."),
        ],
    ),
    "V2F16_TO_V2U16": (
        "OpConvertFToU (v2f16 -> v2u16, v9--v10)", "u16vec2 y = u16vec2(x);",
        [
            ("Negative lane clamps to 0 (v9--v10)",
             fmt_asm_line("V2F16_TO_V2U16", ".round_zero", "r0", ["r1.h01"]),
             "With `r1 = [3.0h, -2.0h]`: saturates negative lane to 0 -> `r0 = [3, 0]` (`0x00030000`)."),
            ("Swizzled lanes + `+Inf` saturation",
             fmt_asm_line("V2F16_TO_V2U16", ".round_zero", "r0", ["r1.h10"]),
             "With `r1 = [1.0h, +Inf]`: `h10` swaps lanes, saturating `+Inf` to `0xFFFF` -> `r0 = 0x0001FFFF`."),
        ],
    ),
    "V2F32_TO_V2F16": (
        "GLSL.std.450 PackHalf2x16 (v9--v10)", "uint p = packHalf2x16(vec2(a, b));",
        [
            ("Pre-narrowing `.clamp_0_1` packing (v9--v10)",
             fmt_asm_line("V2F32_TO_V2F16", ".clamp_0_1", "r0", ["r1", "r2"]),
             "With `r1 = 1.0f, r2 = 2.0f`: clamps BEFORE narrowing (`1.0f, 1.0f`) -> packs `r0 = 0x3C003C00`."),
            ("Rounding mode control on pack",
             fmt_asm_line("V2F32_TO_V2F16", ".round_zero", "r0", ["r1", "r2"]),
             "With `r1 = 65536.0f` (`> FP16 MAX 65504`), `r2 = -0.5f`: `.round_zero` yields `MAX_FP16` (`0x7BFF`) in `h0` and `-0.5h` (`0xB800`) in `h1` -> `r0 = 0xB8007BFF`."),
        ],
    ),
    "V2S16_TO_V2F16": (
        "OpConvertSToF (v2s16 -> v2f16, v9--v10)", "f16vec2 y = f16vec2(x);",
        [
            ("Per-lane signed int16 -> FP16 (v9--v10)",
             fmt_asm_line("V2S16_TO_V2F16", "", "r0", ["r1.h01"]),
             "With `r1 = 0x0002FFFF` (`[2, -1]`): converts per-lane -> `r0 = 0x4000BC00` (`[2.0h, -1.0h]`)."),
            ("Inexact large int16 rounding",
             fmt_asm_line("V2S16_TO_V2F16", ".round_zero", "r0", ["r1.h01"]),
             "With `r1 = [2049, 0]` (exceeds 11-bit FP16 significand): `.round_zero` rounds `2049` down to `2048.0h` (`0x6800`)."),
        ],
    ),
    "V2S8_TO_V2F16": (
        "OpConvertSToF (v2s8 -> v2f16, v9--v10)", "f16vec2 y = f16vec2(x.xz);",
        [
            ("Default `b02` stride extraction (v9--v10)",
             fmt_asm_line("V2S8_TO_V2F16", "", "r0", ["r1.b02"]),
             "Default `b02` extracts bytes 0 & 2: with `r1 = 0x00FE0001` (`b2=-2, b0=+1`), yields `r0 = 0xC0003C00` (`[-2.0h, 1.0h]`)."),
            ("Adjacent bytes `b01` extraction",
             fmt_asm_line("V2S8_TO_V2F16", "", "r0", ["r1.b01"]),
             "With `r1 = 0x000004FC` (`b1=+4, b0=-4`): yields `r0 = 0x4400C400` (`[+4.0h, -4.0h]`)."),
        ],
    ),
    "V2S8_TO_V2S16": (
        "OpSConvert (v2s8 -> v2s16)", "i16vec2 y = i16vec2(x.xz);",
        [
            ("Default `b02` sign extension",
             fmt_asm_line("V2S8_TO_V2S16", "", "r0", ["r1.b02"]),
             "With `r1 = 0x00FE00FF` (`b2 = -2, b0 = -1`): sign-extends bytes 0 & 2 -> `r0 = 0xFFFEFFFF`."),
            ("Broadcast single byte `b33`",
             fmt_asm_line("V2S8_TO_V2S16", "", "r0", ["r1.b33"]),
             "With `r1 = 0x80000000` (`b3 = -128`): sign-extends byte 3 into both halfwords -> `r0 = 0xFF80FF80`."),
        ],
    ),
    "V2U16_TO_V2F16": (
        "OpConvertUToF (v2u16 -> v2f16, v9--v10)", "f16vec2 y = f16vec2(x);",
        [
            ("Normal uint16 pair (v9--v10)",
             fmt_asm_line("V2U16_TO_V2F16", "", "r0", ["r1.h01"]),
             "With `r1 = 0x00040001` (`[4, 1]`): yields `r0 = 0x44003C00` (`[4.0h, 1.0h]`)."),
            ("Overflow to `+Inf` above `65519`",
             fmt_asm_line("V2U16_TO_V2F16", "", "r0", ["r1.h01"]),
             "With `r1 = 0xFFFF0000` (`[65535, 0]`): under RNE, `65535 >= 65520` overflows FP16 to `+Inf` (`0x7C00`) -> `r0 = 0x7C000000`."),
        ],
    ),
    "V2U8_TO_V2F16": (
        "OpConvertUToF (v2u8 -> v2f16, v9--v10)", "f16vec2 y = f16vec2(u.xz);",
        [
            ("Default `b02` unsigned byte pair (v9--v10)",
             fmt_asm_line("V2U8_TO_V2F16", "", "r0", ["r1.b02"]),
             "With `r1 = 0x00020001` (`b2 = 2, b0 = 1`): yields `r0 = 0x40003C00` (`[2.0h, 1.0h]`)."),
            ("High-bit byte (`0xFF = 255.0h`)",
             fmt_asm_line("V2U8_TO_V2F16", "", "r0", ["r1.b13"]),
             "With `r1 = 0xFF000000` (`b3 = 255, b1 = 0`): yields `r0 = 0x5BF80000` (`[255.0h, 0.0h]`)."),
        ],
    ),
    "V2U8_TO_V2U16": (
        "OpUConvert (v2u8 -> v2u16)", "u16vec2 y = u16vec2(u.zx);",
        [
            ("ASTC swizzle + lane-mask idiom",
             fmt_asm_line("V2U8_TO_V2U16", "", "r0.h1", ["r1.b20"]),
             "With `r1 = 0x000500AB`, `r0_old = 0x00001234`: `b20` puts `b2=5` in lane 0 and `b0=0xAB` in lane 1; `.h1` masks lane 1 -> `r0 = 0x00AB1234`."),
            ("Full 2-lane unpack (`b01`)",
             fmt_asm_line("V2U8_TO_V2U16", "", "r0", ["r1.b01"]),
             "With `r1 = 0x0000FEDC`: zero-extends `b0=0xDC` and `b1=0xFE` -> `r0 = 0x00FE00DC`."),
        ],
    ),

    # 8. Move, Pack & Swizzle
    "ADR": (
        "None (PC-relative address for blend/call)", "// Compiler PC-relative label calculation",
        [
            ("Forward return-address calculation",
             fmt_asm_line("ADR", "", "r54", [], ["#0x10"]),
             "At `PC = 0x1000`: computes 32-bit `r54 = (0x1000 + 8 + 0x10) & 0xFFFFFFFF = 0x1018`."),
            ("Negative backward offset",
             fmt_asm_line("ADR", "", "r54", [], ["#-0x18"]),
             "At `PC = 0x1020`: computes `0x1020 + 8 - 0x18 = 0x1010` (PC-relative label before current instruction)."),
        ],
    ),
    "MKVEC": (
        "OpCompositeConstruct / PackUnorm4x8", "uint p = packUnorm4x8(v);",
        [
            ("4-byte pack with fixed virtual `b01` on `src2`",
             fmt_asm_line("MKVEC.v2i8", "", "r0", ["r1.b0", "r2.b0", "r3.b01"]),
             "With `r1 = 0x11, r2 = 0x22, r3 = 0xAABB4433`: packs `Concat(r3[15:0], r2.b0, r1.b0) = 0x44332211`."),
            ("Halfword vector pack (`MKVEC.v2i16`)",
             fmt_asm_line("MKVEC.v2i16", "", "r0", ["r1.h1", "r2.h0"]),
             "With `r1 = 0xAAAA0000`, `r2 = 0x0000BBBB`: places `r1.h1` in low half and `r2.h0` in high half -> `r0 = 0xBBBBAAAA`."),
        ],
    ),
    "MOV": (
        "OpCopyObject", "uint y = x; // register move",
        [
            ("Full 32-bit register copy",
             fmt_asm_line("MOV.i32", "", "r0", ["r1"]),
             "With `r1 = 0xDEADBEEF`: copies all 32 bits -> `r0 = 0xDEADBEEF` (also loads FAU constants like `k1f`)."),
            ("Halfword lane-masked copy (`r0.h0`)",
             fmt_asm_line("MOV.i32", "", "r0.h0", ["r1"]),
             "With `r1 = 0x12345678`, `r0_old = 0xAAAA0000`: `.h0` writes only `r1[15:0]` -> `r0 = 0xAAAA5678`."),
        ],
    ),
    "MOV_IMM": (
        "OpConstant", "uint y = 0xE3D0E3D0u;",
        [
            ("ASTC halfword-masked immediate load",
             fmt_asm_line("MOV_IMM.i32", "", "r4.h1", [], ["#0xE3D0E3D0"]),
             "With `r4_old = 0x00001234`: writes upper half `0xE3D0` into `r4[31:16]`, preserving `r4[15:0]` -> `r4 = 0xE3D01234`."),
            ("Full 32-bit immediate constant load",
             fmt_asm_line("MOV_IMM.i32", "", "r0", [], ["#0x3F800000"]),
             "Loads arbitrary 32-bit constant `0x3F800000` (`1.0f`) directly into `r0` without FAU traffic."),
        ],
    ),
    "SWZ": (
        "OpVectorShuffle", "u8vec4 y = x.wzyx;",
        [
            ("Byte-reverse endianness swap (ETC2 idiom)",
             fmt_asm_line("SWZ.v4i8", "", "r0", ["r1.b3210"]),
             "With `r1 = 0x11223344`: reverses all 4 bytes -> `r0 = 0x44332211`."),
            ("Splat single byte across all 4 lanes",
             fmt_asm_line("SWZ.v4i8", "", "r0", ["r1.b1111"]),
             "With `r1 = 0x11223344`: replicates byte 1 (`0x33`) to all 4 lanes -> `r0 = 0x33333333`."),
        ],
    ),

    # 9. Subgroup & Cross-Lane
    "CLPER": (
        "OpGroupNonUniformShuffle* / OpDPdx / OpDPdy", "float d = dFdx(v); // or subgroupShuffleXor(v, 1)",
        [
            ("16-lane butterfly XOR shuffle",
             fmt_asm_line("CLPER", ".subgroup16.xor.zero", "r0", ["r1", "r2.b0"]),
             "Each lane `L` reads `r1` from lane `L ^ (r2 & 0xF)`: with `r2 = 1`, lanes `(0,1), (2,3), ...` swap values; inactive lanes return `0` (`.zero`)."),
            ("Subgroup min reduction identity (`.inf`)",
             fmt_asm_line("CLPER", ".subgroup16.xor.inf", "r0", ["r1", "r2.b0"]),
             "When partner lane `L ^ r2` is inactive under divergence, `.inf` substitutes `+Inf` (`0x7F800000`) so subsequent `FMIN` is unaffected."),
        ],
    ),
    "CLPER_OR": (
        "OpGroupNonUniformShuffle* + OpBitwiseOr (v14+)", "// v14 fused subgroup shuffle + OR",
        [
            ("Fused subgroup permute + bitwise OR (v14+)",
             fmt_asm_line("CLPER_OR", ".subgroup16.lanes04", "r0", ["r1", "r2.b0", "r3"]),
             "Permutes `r1` from the lane pair selected by `lanes04` + `r2.b0`, then computes `r0 = permuted | r3` in one instruction."),
            ("Quad bitmask accumulation",
             fmt_asm_line("CLPER_OR", ".subgroup4.lanes15", "r0", ["r1", "r2.b0", "r3"]),
             "Accumulates cross-lane bitflags across a 4-lane quad without needing a separate `OR.i32` instruction."),
        ],
    ),
    "WMASK": (
        "OpGroupNonUniformBallot / All / Any", "uvec4 m = subgroupBallot(v > 0.0);",
        [
            ("16-lane warp ballot",
             fmt_asm_line("WMASK", ".subgroup16", "r0", ["r1"]),
             "If active lanes `0, 1, 4` have `r1 != 0` and all other lanes have `r1 == 0`: returns `r0 = 0x00000013` (`(1<<0)|(1<<1)|(1<<4)`)."),
            ("4-lane quad ballot",
             fmt_asm_line("WMASK", ".subgroup4", "r0", ["r1"]),
             "Evaluates non-zero predicate independently within each 4-lane fragment quad (bits `0..3` per quad)."),
        ],
    ),

    # 10. Control Flow, Barriers & System
    "BARRIER": (
        "OpControlBarrier", "barrier();",
        [
            ("Compute workgroup barrier",
             fmt_asm_line("BARRIER", ".slot7.wait", "", []),
             "Synchronizes all warps in the compute workgroup and increments the barrier epoch (requires `slot7` + `.wait`)."),
            ("Shared-memory publish + barrier",
             fmt_asm_line("BARRIER", ".slot7.wait", "", []),
             "Placed after `STORE.i32.slot0` to WLS/shared memory so all warps observe prior writes before subsequent `LOAD`s."),
        ],
    ),
    "BRANCH": (
        "OpBranch / OpBranchConditional", "if (cond) { ... }",
        [
            ("Panfrost canonical unconditional branch",
             fmt_asm_line("BRANCH", ".not", "", ["k0"], ["#0x10"]),
             "`combine=none` tests `k0 != 0` (`False`); `.not` negates boolean to `True` -> `PC = PC + 8 + 0x10`."),
            ("Conditional halfword-AND branch",
             fmt_asm_line("BRANCH", "", "", ["r4.and"], ["#0x20"]),
             "With `r4 = 0x00010001`: both halfwords non-zero -> taken (`PC += 8 + 0x20`). With `r4 = 0x00010000`: falls through (`PC += 8`)."),
        ],
    ),
    "DISCARD": (
        "OpKill / OpDemoteToHelperInvocation", "if (alpha < 0.5) discard;",
        [
            ("Alpha-test fragment discard",
             fmt_asm_line("DISCARD.f32", ".lt", "", ["r3", "k0fr2f.h1"]),
             "Evaluates `r3 < 0.5f`; active lanes where true are marked discarded/helper and suppress color/depth/SSBO writes."),
            ("Unconditional discard via `k0 == k0`",
             fmt_asm_line("DISCARD.f32", ".eq", "", ["k0", "k0"]),
             "Always evaluates true (`0.0 == 0.0`), discarding all currently active lanes in the divergent branch."),
        ],
    ),
    "DOORBELL": (
        "None (firmware/job-management)", "// System/firmware doorbell message",
        [
            ("Firmware doorbell notification",
             fmt_asm_line("DOORBELL", "", "", []),
             "Signals the CSF/job-management firmware doorbell from privileged microcode (never emitted by Vulkan/OpenGL compilers)."),
            ("With scoreboard drain",
             fmt_asm_line("DOORBELL", ".wait", "", []),
             "Drains all in-flight asynchronous slots before ringing the hardware doorbell."),
        ],
    ),
    "EUREKA": (
        "None (firmware/job-management)", "// System/firmware event notification",
        [
            ("Hardware event signal",
             fmt_asm_line("EUREKA", "", "", []),
             "Emits a hardware job-management event notification from system shaders."),
            ("Synchronized event signal",
             fmt_asm_line("EUREKA", ".wait", "", []),
             "Waits for all outstanding scoreboard slots before asserting the hardware event."),
        ],
    ),
    "JUMP": (
        "OpFunctionCall / OpReturn (indirect)", "// Indirect blend shader call / return",
        [
            ("Unconditional absolute indirect jump (blend shader)",
             fmt_asm_line("JUMP", ".absolute", "", ["k1n", "r54"]),
             "Since `k1n != 0`, jumps to 32-bit absolute shader PC in `r54` (`r54 == 0` terminates execution cleanly)."),
            ("Predicated indirect return",
             fmt_asm_line("JUMP", ".absolute", "", ["r0", "r54"]),
             "Jumps to `r54` only if predicate `r0 != 0`; otherwise falls through to `PC + 8`."),
        ],
    ),
    "JUMP_EX": (
        "OpFunctionCall / OpReturn (stack)", "// Hardware call-stack indirect jump",
        [
            ("Hardware call-stack push (`.call`)",
             fmt_asm_line("JUMP_EX", ".call", "", ["k1n", "r1", "r2"]),
             "Pushes return address `PC + 8` onto the hardware call stack and jumps to target address."),
            ("Hardware call-stack return (`.return`)",
             fmt_asm_line("JUMP_EX", ".return", "", ["k1n", "k0", "k0"]),
             "Pops the saved return PC from the hardware call stack and resumes the caller."),
        ],
    ),
    "KABOOM": (
        "OpUnreachable / hardware fault", "// Hardware abort trap",
        [
            ("Unreachable trap",
             fmt_asm_line("KABOOM", "", "", []),
             "Immediately raises a hardware shader fault (`hw_trap(KABOOM)`); used as a poison/unreachable sentinel."),
            ("Divergent abort guard",
             fmt_asm_line("KABOOM", ".end", "", []),
             "Faults the shader job if execution ever reaches an impossible control-flow path."),
        ],
    ),
    "NOP": (
        "OpNop", "// Flow-control carrier (.wait/.end)",
        [
            ("Multi-slot scoreboard wait",
             fmt_asm_line("NOP", ".wait012", "", []),
             "Performs no register or memory update; stalls the warp until scoreboard slots 0, 1, and 2 all drain."),
            ("Shader termination carrier (`.end`)",
             fmt_asm_line("NOP", ".wait.end", "", []),
             "Drains all pending async stores/messages (`.wait`) and terminates the shader warp (`.end`)."),
        ],
    ),

    # 11. Memory Load / Store & LEA
    "LOAD": (
        "OpLoad (PhysicalStorageBuffer / SSBO / Shared)", "int x = buf.data[i];",
        [
            ("Sign-extended 8-bit load (`.sext`)",
             fmt_asm_line("LOAD.i8", ".slot0.sext", "@r1", ["[r2:r3]"], ["#0x4"]),
             "Reads 1 byte at `[r2:r3] + 4`: if memory holds `0xFE`, `.sext` sign-extends to `r1 = 0xFFFFFFFE` (`-2`)."),
            ("128-bit streaming vector load",
             fmt_asm_line("LOAD.i128", ".estream.slot0", "@r0:r3", ["[r4:r5]"], ["#0x10"]),
             "Loads 16 bytes little-endian from `[r4:r5] + 16` into `@r0:r3` with outer-streaming cache hint (`.estream`)."),
        ],
    ),
    "STORE": (
        "OpStore (PhysicalStorageBuffer / SSBO / Shared)", "buf.data[i] = val;",
        [
            ("64-bit little-endian store",
             fmt_asm_line("STORE.i64", ".slot0", "@r0:r1", ["[r2:r3]"], ["#0x8"]),
             "Stores 8 bytes from staging pair `@r0:r1` little-endian to address `[r2:r3] + 8` on scoreboard slot 0."),
            ("Coherent 32-bit fire-and-forget store (`.force.slot7`)",
             fmt_asm_line("STORE.i32", ".force.slot7", "@r0", ["[r2:r3]"], ["#0"]),
             "Writes `r0` with cache-coherent visibility (`.force`) tracked on non-blocking `slot7`."),
        ],
    ),
    "LD_CHECKED": (
        "OpLoad (robustBufferAccess, v12+)", "uint x = ssbo[idx]; // Mali-G720 v12",
        [
            ("In-bounds hardware-checked load (v12+)",
             fmt_asm_line("LD_CHECKED.i32", ".slot0", "@r12", ["u3", "r47^", "k1n"]),
             "Checks byte offset `r47 < bound`: when in-bounds, loads 4 bytes from base `u3 + r47` into `@r12`."),
            ("Out-of-bounds zero return",
             fmt_asm_line("LD_CHECKED.i32", ".slot0", "@r12", ["u3", "r47", "r48"]),
             "When `r47 >= r48` (with default `oob_mode=none`): suppresses the memory transaction and writes `@r12 = 0`."),
        ],
    ),
    "LD_CHECKED_IMM": (
        "OpLoad (robustBufferAccess imm, v12+)", "uint64_t x = ssbo[4]; // Mali-G720 v12",
        [
            ("Immediate-offset bounds-checked 64-bit load",
             fmt_asm_line("LD_CHECKED_IMM.i64", ".slot1", "@r62:r63", ["[r8:r9]", "r10"], ["#0x20"]),
             "Checks `0x20 + 8 <= r10`: if in-bounds, loads 8 bytes at `[r8:r9] + 0x20` into `@r62:r63` on `slot1`."),
            ("Out-of-bounds zero fill",
             fmt_asm_line("LD_CHECKED_IMM.i64", ".slot1", "@r62:r63", ["[r8:r9]", "r10"], ["#0x20"]),
             "If buffer size `r10 = 0x10` (`< 0x20`): suppresses load and zeroes both `@r62 = 0, @r63 = 0`."),
        ],
    ),
    "ST_CHECKED": (
        "OpStore (robustBufferAccess, v12+)", "ssbo[idx] = val; // Mali-G720 v12",
        [
            ("In-bounds checked store",
             fmt_asm_line("ST_CHECKED.i32", ".slot0", "@r0", ["u3", "r1", "r2"]),
             "Checks `r1 < r2`: when in-bounds, writes `@r0` to `u3 + r1`."),
            ("Out-of-bounds silent drop",
             fmt_asm_line("ST_CHECKED.i32", ".slot0", "@r0", ["u3", "r1", "r2"]),
             "When `r1 >= r2`: silently drops the store without faulting or corrupting adjacent memory."),
        ],
    ),
    "ST_CHECKED_IMM": (
        "OpStore (robustBufferAccess imm, v12+)", "ssbo[4] = val; // Mali-G720 v12",
        [
            ("Immediate-offset checked store",
             fmt_asm_line("ST_CHECKED_IMM.i32", ".slot0", "@r0", ["u3", "r2"], ["#0x10"]),
             "Checks `0x10 < r2`: if in-bounds, writes 4 bytes from `@r0` to `u3 + 0x10`."),
            ("Out-of-bounds silent drop",
             fmt_asm_line("ST_CHECKED_IMM.i32", ".slot0", "@r0", ["u3", "r2"], ["#0x10"]),
             "If `r2 <= 0x10`: suppresses the write cleanly with no memory side effect."),
        ],
    ),
    "LD_CVT": (
        "OpImageRead (paired with LEA_TEX)", "vec4 p = imageLoad(img, coord);",
        [
            ("Converted RGBA8 -> FP32 texel load",
             fmt_asm_line("LD_CVT", ".f32.v4.slot0", "@r0:r3", ["[r4:r5]", "r6"], ["#0"]),
             "Loads texel at address `[r4:r5]` (from `LEA_TEX`) and converts pixel format via descriptor `r6` into 4x FP32 `@r0:r3`."),
            ("Relaxed-precision FP16 image load",
             fmt_asm_line("LD_CVT", ".f16.v4.slot0", "@r0:r1", ["[r4:r5]", "r6"], ["#0"]),
             "Converts the same texel directly into 4x packed FP16 values in just 2 registers `@r0:r1`."),
        ],
    ),
    "ST_CVT": (
        "OpImageWrite (paired with LEA_TEX)", "imageStore(img, coord, color);",
        [
            ("Converted FP32 -> RGBA8 texel store",
             fmt_asm_line("ST_CVT", ".f32.v4.slot0", "@r0:r3", ["[r4:r5]", "r6"], ["#0"]),
             "Converts 4x FP32 `@r0:r3` using conversion descriptor `r6` (e.g., UNORM8 quantization) and stores to `[r4:r5]`."),
            ("Integer storage image store",
             fmt_asm_line("ST_CVT", ".u32.v4.slot0", "@r0:r3", ["[r4:r5]", "r6"], ["#0"]),
             "Writes 4x `u32` channels into an integer `uimage2D` texel address returned by `LEA_TEX`."),
        ],
    ),
    "LD_GCLK": (
        "OpReadClockKHR", "uint64_t t = clockARB();",
        [
            ("GPU shader-core cycle counter",
             fmt_asm_line("LD_GCLK", ".cycle_counter.slot0", "@r0:r1", []),
             "Reads the 64-bit GPU cycle counter into `@r0:r1` (`r0` = low 32 bits, `r1` = high 32 bits) for fine-grained profiling."),
            ("Global system timestamp",
             fmt_asm_line("LD_GCLK", ".system_timestamp.slot0", "@r0:r1", []),
             "Reads the monotonically increasing 64-bit system-wide timestamp synchronized across cores."),
        ],
    ),
    "LD_MULTI": (
        "OpLoad (bulk >128-bit, v14+)", "// v14 bulk multi-register load",
        [
            ("256-bit (8-GPR) bulk load (v14+)",
             fmt_asm_line("LD_MULTI", ".i256.slot0", "@r0:r7", ["[r8:r9]"], ["#0"]),
             "Loads 8 consecutive 32-bit registers (32 bytes) from `[r8:r9]` in a single v14 memory message."),
            ("Offset 160-bit (5-GPR) load",
             fmt_asm_line("LD_MULTI", ".i160.slot1", "@r0:r4", ["[r8:r9]"], ["#0x20"]),
             "Loads 20 bytes from `[r8:r9] + 0x20` into `@r0:r4` on scoreboard `slot1`."),
        ],
    ),
    "ST_MULTI": (
        "OpStore (bulk >128-bit, v14+)", "// v14 bulk multi-register store",
        [
            ("256-bit (8-GPR) bulk store (v14+)",
             fmt_asm_line("ST_MULTI", ".i256.slot0", "@r0:r7", ["[r8:r9]"], ["#0"]),
             "Stores 8 consecutive 32-bit registers (32 bytes) to `[r8:r9]` in a single v14 store message."),
            ("Streaming 192-bit (6-GPR) store",
             fmt_asm_line("ST_MULTI", ".i192.estream.slot7", "@r0:r5", ["[r8:r9]"], ["#0x10"]),
             "Streams 24 bytes from `@r0:r5` to `[r8:r9] + 0x10` on `slot7`."),
        ],
    ),
    "LD_PKA": (
        "OpLoad (UBO/SSBO descriptor table)", "vec4 v = ubo.mat[i];",
        [
            ("128-bit descriptor-indexed buffer load",
             fmt_asm_line("LD_PKA.i128", ".slot0", "@r0:r3", ["r4", "u0"]),
             "Loads 16 bytes from PKA buffer descriptor `u0` at byte offset `r4` into `@r0:r3`."),
            ("32-bit scalar descriptor load",
             fmt_asm_line("LD_PKA.i32", ".slot0", "@r0", ["r4", "u0"]),
             "Loads 4 bytes from descriptor `u0` at dynamic offset `r4` into `@r0`."),
        ],
    ),
    "LEA_ATTR": (
        "None (attribute address + cvt desc)", "// Dynamic vertex attribute address",
        [
            ("Dynamic vertex attribute address lookup",
             fmt_asm_line("LEA_ATTR", ".slot0", "@r0:r2", ["r60", "r61", "r4"]),
             "Computes `(addr_lo, addr_hi, cvt_desc)` in `@r0:r2` for `(vertex_id=r60, instance_id=r61)` and dynamic attribute handle `r4`."),
            ("Paired with `LD_CVT` / `ST_CVT`",
             fmt_asm_line("LEA_ATTR", ".slot1", "@r4:r6", ["r60", "r61", "r8"]),
             "Produces the 64-bit address `[r4:r5]` and format descriptor `r6` consumed by a subsequent `ST_CVT` in vertex-output shaders."),
        ],
    ),
    "LEA_ATTR_IMM": (
        "None (attribute address + cvt desc)", "// Static vertex/varying output address",
        [
            ("Static varying output address (table 0, index 1)",
             fmt_asm_line("LEA_ATTR_IMM", ".slot0", "@r0:r2", ["r59"], ["#0", "#1"]),
             "Computes 64-bit output varying address `[r0:r1]` + conversion descriptor `r2` for immediate `(table=0, index=1)`."),
            ("Vertex shader position output (table 0, index 0)",
             fmt_asm_line("LEA_ATTR_IMM", ".slot0", "@r0:r2", ["r59"], ["#0", "#0"]),
             "Resolves the per-vertex position buffer slot before `ST_CVT.f32.v4`."),
        ],
    ),
    "LEA_BUF": (
        "OpAccessChain (dynamic SSBO base)", "ssbo[dyn_idx].val = x;",
        [
            ("Dynamic SSBO address resolution",
             fmt_asm_line("LEA_BUF", ".slot0", "@r0:r1", ["r2", "r3"]),
             "Computes 64-bit GPU VA `@r0:r1` for dynamic buffer descriptor handle `r3` plus byte offset `r2`."),
            ("Bindless SSBO array indexing",
             fmt_asm_line("LEA_BUF", ".slot1", "@r4:r5", ["r6", "r7"]),
             "Resolves address of `ssbos[nonuniformEXT(i)]` for subsequent `LOAD`/`STORE`/`ATOM`."),
        ],
    ),
    "LEA_BUF_IMM": (
        "OpAccessChain (static SSBO base)", "ssbo.val[i] = x;",
        [
            ("Static SSBO base + dynamic byte offset",
             fmt_asm_line("LEA_BUF_IMM", ".slot0", "@r0:r1", ["r2"], ["#0", "#0"]),
             "Computes 64-bit effective address of immediate buffer `(table=0, index=0)` + byte offset `r2` into `@r0:r1`."),
            ("Second descriptor binding (`table=0, index=2`)",
             fmt_asm_line("LEA_BUF_IMM", ".slot0", "@r2:r3", ["k0"], ["#0", "#2"]),
             "With offset `k0 = 0`: retrieves the base 64-bit GPU pointer of buffer binding 2."),
        ],
    ),
    "LEA_PKA": (
        "OpAccessChain (PKA buffer address)", "// Simple/PKA buffer effective address",
        [
            ("PKA buffer address resolution",
             fmt_asm_line("LEA_PKA", ".slot0", "@r0:r1", ["r2", "r3"]),
             "Computes 64-bit effective address `@r0:r1` for PKA buffer handle `r3` + byte offset `r2`."),
            ("Zero-offset base pointer query",
             fmt_asm_line("LEA_PKA", ".slot0", "@r0:r1", ["k0", "u0"]),
             "Extracts the 64-bit base pointer of uniform-supplied PKA handle `u0` into `@r0:r1`."),
        ],
    ),
    "LEA_TEX": (
        "None (image texel address + cvt desc)", "imageStore(imgs[i], xy, c);",
        [
            ("Dynamic storage image texel address",
             fmt_asm_line("LEA_TEX", ".slot0", "@r0:r2", ["r3", "r4", "r5"]),
             "Computes `(addr_lo, addr_hi, cvt_desc)` in `@r0:r2` for dynamic image handle `r5` at packed 16-bit coords `(r3, r4)`."),
            ("Image atomic address setup",
             fmt_asm_line("LEA_TEX", ".slot0", "@r2:r4", ["r0", "k0", "r5"]),
             "Resolves the exact 64-bit texel pointer `[r2:r3]` for `imageAtomicAdd` via subsequent `ATOM_RETURN.i32`."),
        ],
    ),
    "LEA_TEX_IMM": (
        "None (image texel address + cvt desc)", "imageStore(img, xy, c);",
        [
            ("Static 2D storage image texel address",
             fmt_asm_line("LEA_TEX_IMM", ".slot0", "@r0:r2", ["r3", "r4"], ["#0", "#0"]),
             "Computes `(addr_lo, addr_hi, cvt_desc)` for static image `(table=0, index=0)` at coords `(r3, r4)` for `ST_CVT`/`LD_CVT`."),
            ("Static 3D/array image layer address",
             fmt_asm_line("LEA_TEX_IMM", ".slot0", "@r0:r2", ["r3", "r4"], ["#0", "#1"]),
             "Passes packed `(x, y)` in `r3` and `(z/layer, msaa_sample)` in `r4` for binding `(0, 1)`."),
        ],
    ),

    # 12. Atomics
    "ACMPXCHG": (
        "OpAtomicCompareExchange", "uint old = atomicCompSwap(x, cmp, val);",
        [
            ("Successful 32-bit CAS",
             fmt_asm_line("ACMPXCHG.i32", ".slot0", "@r0", ["[r2:r3]"], ["#0"]),
             "With `*addr == 10`, compare `r0 = 10`, swap `r1 = 42`: updates `*addr = 42` and returns old value `r0 = 10`."),
            ("Failed 32-bit CAS (memory unchanged)",
             fmt_asm_line("ACMPXCHG.i32", ".slot0", "@r0", ["[r2:r3]"], ["#0"]),
             "With `*addr == 99`, compare `r0 = 10`, swap `r1 = 42`: leaves `*addr = 99` untouched and returns observed `r0 = 99`."),
        ],
    ),
    "ATOM": (
        "OpAtomicIAdd/Min/Max/And/Or/Xor (unused result)", "atomicAdd(buf[i], v); // result unused",
        [
            ("Fire-and-forget atomic add",
             fmt_asm_line("ATOM.i32", ".aadd.slot0", "@r0", ["[r2:r3]"], ["#0"]),
             "With `*addr = 100`, `r0 = 5`: atomically updates `*addr = 105` without returning the old value."),
            ("Fire-and-forget atomic bitwise OR",
             fmt_asm_line("ATOM.i32", ".aor.slot7", "@r0", ["[r2:r3]"], ["#0x4"]),
             "Atomically sets bits `r0` in `*([r2:r3] + 4)` on non-blocking `slot7`."),
        ],
    ),
    "ATOM1": (
        "OpAtomicIIncrement / IDecrement (unused result)", "atomicAdd(buf[i], 1u); // result unused",
        [
            ("Fire-and-forget atomic increment",
             fmt_asm_line("ATOM1.i32", ".ainc.slot0", "", ["[r2:r3]"], ["#0"]),
             "With `*addr = 7`: atomically increments `*addr` to `8` without needing a value staging register."),
            ("Fire-and-forget atomic decrement",
             fmt_asm_line("ATOM1.i32", ".adec.slot7", "", ["[r2:r3]"], ["#0x8"]),
             "Atomically decrements `*([r2:r3] + 8)` by `1`."),
        ],
    ),
    "ATOM1_RETURN": (
        "OpAtomicIIncrement / IDecrement", "uint old = atomicAdd(buf[i], 1u);",
        [
            ("Atomic counter allocation (`.ainc`)",
             fmt_asm_line("ATOM1_RETURN.i32", ".ainc.slot0", "@r0", ["[r2:r3]"], ["#0"]),
             "With `*addr = 12`: atomically sets `*addr = 13` and returns pre-increment ticket `@r0 = 12`."),
            ("Atomic reference-count decrement (`.adec`)",
             fmt_asm_line("ATOM1_RETURN.i32", ".adec.slot0", "@r0", ["[r2:r3]"], ["#0"]),
             "With `*addr = 1`: atomically sets `*addr = 0` and returns old count `@r0 = 1`."),
        ],
    ),
    "ATOM_RETURN": (
        "OpAtomicIAdd / SMin / UMax / And / Or / Xor", "uint old = atomicMax(buf[i], v);",
        [
            ("Atomic unsigned maximum (`.aumax`)",
             fmt_asm_line("ATOM_RETURN.i32", ".aumax.slot0", "@r0", ["[r2:r3]"], ["#0"]),
             "With `*addr = 10`, `r0 = 25`: updates `*addr = 25` and returns previous value `@r0 = 10`."),
            ("Atomic signed minimum (`.asmin`)",
             fmt_asm_line("ATOM_RETURN.i32", ".asmin.slot0", "@r0", ["[r2:r3]"], ["#0x4"]),
             "With `*addr = 5`, `r0 = -3`: updates `*addr = -3` and returns previous value `@r0 = 5`."),
        ],
    ),
    "AXCHG": (
        "OpAtomicExchange", "uint old = atomicExchange(buf[i], v);",
        [
            ("32-bit atomic swap",
             fmt_asm_line("AXCHG.i32", ".slot0", "@r0", ["[r2:r3]"], ["#0"]),
             "With `*addr = 0x1111`, `r0 = 0x2222`: stores `0x2222` to `*addr` and returns `@r0 = 0x1111`."),
            ("64-bit spinlock/pointer exchange",
             fmt_asm_line("AXCHG.i64", ".slot0", "@r0:r1", ["[r2:r3]"], ["#0"]),
             "Atomically swaps 64-bit value at `[r2:r3]` with `@r0:r1`, returning the prior 64-bit word in `@r0:r1`."),
        ],
    ),

    # 13. Attributes, Varyings & Fragment
    "ATEST": (
        "None (alpha-to-coverage)", "// Fragment alpha-to-coverage / early-Z",
        [
            ("Early-Z / alpha-to-coverage update",
             fmt_asm_line("ATEST", ".slot0", "@r60", ["r60", "r3", "k0"]),
             "Combines input sample coverage `r60` with RT0 alpha `r3` and writes updated coverage mask back to `@r60`."),
            ("Discard check after ATEST",
             fmt_asm_line("ATEST", ".slot0.wait0", "@r60", ["r60", "r3", "k0"]),
             "Does NOT implicitly discard the lane; shaders test `@r60 == 0` afterward if skipping expensive shading."),
        ],
    ),
    "BLEND": (
        "OpStore (Fragment color output)", "outColor = vec4(r, g, b, a);",
        [
            ("Fixed-function RT0 RGBA32F blend + shader end",
             fmt_asm_line("BLEND", ".f32.v4.slot0.end", "@r0:r3", ["u0", "r60"], ["#0"]),
             "Blends RGBA `@r0:r3` with coverage `r60` using blend descriptor `u0` for RT0; `.end` terminates the fragment warp."),
            ("MRT secondary render target (`#1`, no `.end`)",
             fmt_asm_line("BLEND", ".f16.v4.slot0", "@r4:r5", ["u1", "r60"], ["#1"]),
             "Emits packed FP16 RGBA `@r4:r5` to render target 1 before issuing the final RT0 `.end` blend."),
        ],
    ),
    "LD_ATTR": (
        "OpLoad (dynamic vertex input)", "in vec4 vAttr; // dynamic index",
        [
            ("Dynamic vertex attribute fetch (FP32 vec4)",
             fmt_asm_line("LD_ATTR", ".f32.v4.slot0", "@r0:r3", ["r60", "r61", "r4"]),
             "Loads 4x FP32 vertex attribute for `(vertex_id=r60, instance_id=r61)` at dynamic descriptor handle `r4` into `@r0:r3`."),
            ("Integer vertex attribute (u32 vec2)",
             fmt_asm_line("LD_ATTR", ".u32.v2.slot0", "@r0:r1", ["r60", "r61", "r4"]),
             "Loads 2x unsigned 32-bit joint indices into `@r0:r1`."),
        ],
    ),
    "LD_ATTR_IMM": (
        "OpLoad (vertex input)", "layout(location=0) in vec4 aPos;",
        [
            ("Static vertex position attribute (`table=0, index=0`)",
             fmt_asm_line("LD_ATTR_IMM", ".f32.v4.slot0", "@r0:r3", ["r60", "r61"], ["#0", "#0"]),
             "Loads 4x FP32 vertex attribute from immediate `(table=0, index=0)` into `@r0:r3`."),
            ("Relaxed-precision normal (`f16 vec3`)",
             fmt_asm_line("LD_ATTR_IMM", ".f16.v3.slot0", "@r4:r5", ["r60", "r61"], ["#0", "#1"]),
             "Converts attribute 1 to 3x FP16 packed into `@r4:r5`."),
        ],
    ),
    "LD_TEX": (
        "OpImageRead (dynamic handle)", "vec4 p = imageLoad(imgs[i], xy);",
        [
            ("Direct texel load from bindless image handle",
             fmt_asm_line("LD_TEX", ".f32.v4.slot0", "@r0:r3", ["r4", "r5", "r6"]),
             "Reads unfiltered texel from dynamic image handle `r6` at integer coordinates `(r4, r5)` into 4x FP32 `@r0:r3`."),
            ("Unsigned integer image read",
             fmt_asm_line("LD_TEX", ".u32.v4.slot0", "@r0:r3", ["r4", "r5", "r6"]),
             "Reads raw `uvec4` texel values from `uimage2D` without float normalization."),
        ],
    ),
    "LD_TEX_IMM": (
        "OpImageRead", "vec4 p = imageLoad(img, xy);",
        [
            ("Static `uimage2D` texel read",
             fmt_asm_line("LD_TEX_IMM", ".u32.v4.slot0", "@r0:r3", ["r4", "r5"], ["#0", "#0"]),
             "Reads texel `(r4.h0, r4.h1)` from static image `(table=0, index=0)` into `@r0:r3`."),
            ("FP16 storage image read (`table=0, index=2`)",
             fmt_asm_line("LD_TEX_IMM", ".f16.v4.slot0", "@r0:r1", ["r4", "r5"], ["#0", "#2"]),
             "Reads 4x FP16 channels from binding 2 into `@r0:r1`."),
        ],
    ),
    "LD_TILE": (
        "OpImageRead (subpassInput / framebuffer fetch)", "vec4 fb = subpassLoad(inColor);",
        [
            ("Color subpass / framebuffer fetch",
             fmt_asm_line("LD_TILE", ".f32.v4.slot0", "@r0:r3", ["r59", "r60", "u0"]),
             "Reads on-chip tilebuffer color at `(pixel=r59, coverage=r60, conv_desc=u0)` into 4x FP32 `@r0:r3`."),
            ("Depth/stencil tilebuffer read (`.z_stencil.last_read`)",
             fmt_asm_line("LD_TILE", ".f32.z_stencil.last_read.slot0", "@r0", ["r59", "r60", "u1"]),
             "Reads depth from the Z/S tilebuffer and marks it `.last_read` so tile storage can be reclaimed."),
        ],
    ),
    "LD_VAR": (
        "OpLoad (interpolated fragment input)", "in vec4 vCol; // dynamic handle",
        [
            ("Center interpolation saving barycentrics (`.store`)",
             fmt_asm_line("LD_VAR", ".f32.v4.center.store.slot0", "@r0:r3", ["r59", "r4"]),
             "Interpolates varying handle `r4` at pixel center and caches barycentric weights in hardware (`.store`)."),
            ("Centroid interpolation reusing cache (`.retrieve`)",
             fmt_asm_line("LD_VAR", ".f32.v4.centroid.retrieve.slot0", "@r4:r7", ["r59", "r5"]),
             "Interpolates a second varying at centroid reusing cached barycentric weights (`.retrieve`)."),
        ],
    ),
    "LD_VAR_BUF": (
        "OpLoad (hardware varying buffer)", "in vec4 vCol;",
        [
            ("Descriptor-less varying buffer interpolation",
             fmt_asm_line("LD_VAR_BUF.f32", ".v4.center.store.slot0", "@r0:r3", ["r59", "r4"]),
             "Interpolates 4x FP32 varying at dynamic byte offset `r4` in the hardware varying buffer and stores barycentrics."),
            ("FP16-stored varying widened to FP32 (`.src_f16`)",
             fmt_asm_line("LD_VAR_BUF.f32", ".src_f16.v2.center.retrieve.slot0", "@r0:r1", ["r59", "r4"]),
             "Reads `mediump` FP16 varying storage (`.src_f16`) and delivers interpolated FP32 `vec2` in `@r0:r1`."),
        ],
    ),
    "LD_VAR_BUF_FLAT": (
        "OpLoad (flat varying buffer, v14+)", "flat in uint vId;",
        [
            ("Provoking-vertex flat varying load (v14+)",
             fmt_asm_line("LD_VAR_BUF_FLAT", ".u32.slot0", "@r0", ["r4"]),
             "Fetches uninterpolated 32-bit flat varying from the varying buffer at dynamic offset `r4`."),
            ("Per-vertex explicit index (`.explicit_vertex`)",
             fmt_asm_line("LD_VAR_BUF_FLAT", ".f32.v4.explicit_vertex.slot0", "@r0:r3", ["r4"]),
             "Implements `GL_EXT_fragment_shader_barycentric` `pervertexEXT` by selecting triangle vertex `0..2` dynamically."),
        ],
    ),
    "LD_VAR_BUF_FLAT_IMM": (
        "OpLoad (flat varying buffer imm, v14+)", "flat in uint vId;",
        [
            ("Immediate-offset flat 32-bit varying (v14+)",
             fmt_asm_line("LD_VAR_BUF_FLAT_IMM", ".u32.slot0", "@r0", [], ["#0x10"]),
             "Fetches uninterpolated flat varying at immediate byte offset `0x10` into `@r0`."),
            ("Flat 16-bit varying (`flat16`)",
             fmt_asm_line("LD_VAR_BUF_FLAT_IMM", ".flat16.slot0", "@r0", [], ["#0x08"]),
             "Reads 16-bit flat varying at immediate offset `0x08`."),
        ],
    ),
    "LD_VAR_BUF_IMM": (
        "OpLoad (varying buffer imm)", "in vec2 vUV;",
        [
            ("Cached barycentric varying load (`.retrieve`)",
             fmt_asm_line("LD_VAR_BUF_IMM.f32", ".v2.center.retrieve.slot0", "@r0:r1", ["r59"], ["#0x10"]),
             "Interpolates `vec2` varying at immediate byte offset `0x10` reusing cached barycentric coordinates."),
            ("Per-sample MSAA interpolation (`.sample`)",
             fmt_asm_line("LD_VAR_BUF_IMM.f32", ".v4.sample.store.slot0", "@r0:r3", ["r59"], ["#0x00"]),
             "Evaluates `sample in vec4` at the active MSAA sample position rather than pixel center."),
        ],
    ),
    "LD_VAR_FLAT": (
        "OpLoad (flat varying, dynamic)", "flat in uvec4 vData;",
        [
            ("Dynamic-handle flat varying load",
             fmt_asm_line("LD_VAR_FLAT", ".u32.v4.slot0", "@r0:r3", ["r4"]),
             "Fetches 4x `u32` uninterpolated flat varying via descriptor handle `r4`."),
            ("Vertex-1 flat read (`.vertex1`)",
             fmt_asm_line("LD_VAR_FLAT", ".f32.v4.vertex1.slot0", "@r0:r3", ["r4"]),
             "Reads triangle vertex 1's attribute directly without perspective interpolation."),
        ],
    ),
    "LD_VAR_FLAT_IMM": (
        "OpLoad (flat varying)", "flat in uvec4 vData;",
        [
            ("Immediate descriptor flat load (`table=0, index=1`)",
             fmt_asm_line("LD_VAR_FLAT_IMM", ".u32.v4.slot0", "@r0:r3", [], ["#0", "#1"]),
             "Fetches 4x `u32` uninterpolated flat varying at immediate `(table=0, index=1)`."),
            ("Explicit provoking vertex 2 (`.vertex2`)",
             fmt_asm_line("LD_VAR_FLAT_IMM", ".f32.v4.vertex2.slot0", "@r0:r3", [], ["#0", "#2"]),
             "Reads vertex 2 of the current primitive for `pervertexEXT`."),
        ],
    ),
    "LD_VAR_IMM": (
        "OpLoad (interpolated varying)", "layout(location=0) in vec4 vCol;",
        [
            ("Center-interpolated `vec4` varying",
             fmt_asm_line("LD_VAR_IMM", ".f32.v4.center.store.slot0", "@r0:r3", ["r59"], ["#0", "#0"]),
             "Interpolates `vec4` varying at immediate `(table=0, index=0)` into `@r0:r3` and caches barycentrics."),
            ("Centroid-interpolated `vec2` varying",
             fmt_asm_line("LD_VAR_IMM", ".f32.v2.centroid.clobber.slot0", "@r4:r5", ["r59"], ["#0", "#1"]),
             "Interpolates `centroid in vec2` at `(table=0, index=1)` without preserving the prior barycentric cache."),
        ],
    ),
    "LD_VAR_SPECIAL": (
        "OpLoad (FragCoord / PointCoord / BaryCoord)", "float w = gl_FragCoord.w;",
        [
            ("Reciprocal clip-space W (`gl_FragCoord.w`)",
             fmt_asm_line("LD_VAR_SPECIAL", ".frag_w.center.clobber.slot0", "@r0", ["r59"]),
             "Loads perspective-correct `1/w` (`gl_FragCoord.w`) at pixel center into `@r0`."),
            ("Barycentric coordinates (`gl_BaryCoordEXT`)",
             fmt_asm_line("LD_VAR_SPECIAL", ".bary.center.store.slot0", "@r0:r1", ["r59"]),
             "Loads `(i, j)` barycentric weights into `@r0:r1` (third weight is `1 - i - j`)."),
        ],
    ),
    "ST_TILE": (
        "None (blend shader tilebuffer write)", "// Internal blend shader tile store",
        [
            ("Programmable blend shader color write",
             fmt_asm_line("ST_TILE", ".f32.v4.slot0", "@r0:r3", ["r59", "r60", "u0"]),
             "Converts and writes blended RGBA `@r0:r3` directly to on-chip tilebuffer at `(pixel=r59, cov=r60, desc=u0)`."),
            ("Packed FP16 tilebuffer write",
             fmt_asm_line("ST_TILE", ".f16.v4.slot0", "@r0:r1", ["r59", "r60", "u0"]),
             "Writes 4x FP16 channels from `@r0:r1` into an `R16G16B16A16_SFLOAT` tile attachment."),
        ],
    ),
    "ZS_EMIT": (
        "OpStore (FragDepth / FragStencilRefEXT)", "gl_FragDepth = d;",
        [
            ("Programmatic fragment depth export (`.z`)",
             fmt_asm_line("ZS_EMIT", ".z.slot0", "@r60", ["r0", "k0", "r60"]),
             "Emits shader-computed `gl_FragDepth = r0` with coverage `r60`, returning post-depth-test coverage in `@r60`."),
            ("Combined depth + stencil ref export (`.z.stencil`)",
             fmt_asm_line("ZS_EMIT", ".z.stencil.slot0", "@r60", ["r0", "r1", "r60"]),
             "Exports both depth `r0` and 8-bit stencil reference `r1` simultaneously, updating `@r60`."),
        ],
    ),

    # 14. Texture & Varying-Texture
    "TEX_DUAL": (
        "2x OpImageSample* (shared coords)", "vec4 a = texture(s0, uv); vec4 b = texture(s1, uv);",
        [
            ("Paired 2D albedo + normal sample",
             fmt_asm_line("TEX_DUAL", ".tex2d.computed.slot0", "@r0:r7", ["@r8:r9", "u0"]),
             "Issues two filtered 2D samples sharing UV `@r8:r9`, writing primary RGBA to `@r0:r3` and secondary RGBA to `@r4:r7`."),
            ("Mixed-mask dual sample (`.primary_rgb.secondary_r`)",
             fmt_asm_line("TEX_DUAL", ".tex2d.primary_rgb.secondary_r.slot0", "@r0:r3", ["@r8:r9", "u0"]),
             "Packs primary `.rgb` into `@r0:r2` and secondary `.r` (e.g., roughness) into `@r3`, saving 4 staging GPRs."),
        ],
    ),
    "TEX_FETCH": (
        "OpImageFetch", "vec4 c = texelFetch(samp, ivec2(xy), lod);",
        [
            ("Unfiltered integer-coord 2D fetch",
             fmt_asm_line("TEX_FETCH", ".tex2d.dst32.rgba.slot0", "@r0:r3", ["@r4:r6", "u0"]),
             "Fetches exact texel at integer `(x=r4, y=r5, lod=r6)` via handle `u0` into 4x 32-bit `@r0:r3` without filtering."),
            ("2D array texel fetch (`.array`)",
             fmt_asm_line("TEX_FETCH", ".tex2d.array.dst32.rgba.slot0", "@r0:r3", ["@r4:r7", "u0"]),
             "Adds integer array layer in `@r7` for `texelFetch(sampler2DArray, ivec3(x,y,layer), lod)`."),
        ],
    ),
    "TEX_GATHER": (
        "OpImageGather / OpImageDrefGather", "vec4 g = textureGather(samp, uv, 0);",
        [
            ("2x2 red-channel gather (`.gather4_r`)",
             fmt_asm_line("TEX_GATHER", ".tex2d.gather4_r.dst32.rgba.slot0", "@r0:r3", ["@r4:r5", "u0"]),
             "Gathers the red channel of the 4 texels in the 2x2 bilinear footprint around `uv = (@r4, @r5)` into `@r0:r3`."),
            ("Shadow depth-compare gather (`.compare`)",
             fmt_asm_line("TEX_GATHER", ".tex2d.compare.dst32.rgba.slot0", "@r0:r3", ["@r4:r6", "u0"]),
             "Compares the 2x2 depth footprint against `Dref = @r6`, returning 4 binary PCF results (`0.0` or `1.0`) in `@r0:r3`."),
        ],
    ),
    "TEX_GRADIENT": (
        "OpImageSampleExplicitLod (Grad)", "vec4 c = textureGrad(samp, uv, ddx, ddy);",
        [
            ("Explicit anisotropic gradient sample",
             fmt_asm_line("TEX_GRADIENT", ".tex2d.dst32.rgba.slot0", "@r0:r3", ["@r4:r9", "u0"]),
             "Samples texture at `uv = (@r4, @r5)` using explicit `dPdx = (@r6, @r7)` and `dPdy = (@r8, @r9)`."),
            ("Query raw mipmap LOD (`.disable_lod_bias.disable_lod_clamp`)",
             fmt_asm_line("TEX_GRADIENT", ".tex2d.disable_lod_bias.disable_lod_clamp.slot0", "@r0:r1", ["@r4:r5", "u0"]),
             "Returns `(clamped_lod, unclamped_lod)` for GLSL `textureQueryLod(samp, uv)`."),
        ],
    ),
    "TEX_SINGLE": (
        "OpImageSampleImplicitLod / ExplicitLod", "vec4 c = texture(samp, uv);",
        [
            ("Implicit quad-LOD 2D sample with `.skip`",
             fmt_asm_line("TEX_SINGLE", ".tex2d.computed.dst32.rgba.skip.slot0", "@r0:r3", ["@r4:r5", "u0"]),
             "Computes 2x2 quad LOD from `@r4:r5`, skips memory fetch on helper lanes (`.skip`), and returns RGBA in `@r0:r3`."),
            ("Hardware-projected cubemap sample (v11+ `.cube.project`)",
             fmt_asm_line("TEX_SINGLE", ".cube.project.computed.dst32.rgba.slot0", "@r0:r3", ["@r4:r6", "u0"]),
             "Takes 3D direction `(x, y, z)` in `@r4:r6` directly on v11+, eliminating v9--v10 `CUBEFACE1/2` + `CUBE_SSEL/TSEL`."),
        ],
    ),
    "TEX_WEIGHTS": (
        "None (v9 footprint weights query)", "// v9-only filter weights query",
        [
            ("2x2 bilinear weight query (v9 only)",
             fmt_asm_line("TEX_WEIGHTS", ".tex2d.footprint_2x2.slot0", "@r0:r3", ["@r4:r5", "u0"]),
             "Returns the 4 normalized bilinear filter weights for the 2x2 texel footprint at `@r4:r5` (removed in v10+)."),
            ("Sub-texel center check",
             fmt_asm_line("TEX_WEIGHTS", ".tex2d.footprint_2x2.slot0", "@r0:r3", ["@r4:r5", "u0"]),
             "At exact texel midpoint `(frac=0.5, 0.5)`: returns `[0.25f, 0.25f, 0.25f, 0.25f]` in `@r0:r3`."),
        ],
    ),
    "VAR_TEX_BUF_DUAL": (
        "OpLoad (in) + 2x OpImageSample*", "vec4 a=texture(s0,vUV); vec4 b=texture(s1,vUV);",
        [
            ("Fused varying-buffer interpolation + dual sample",
             fmt_asm_line("VAR_TEX_BUF_DUAL", ".tex2d.computed.slot0", "@r0:r7", ["u0", "r59", "r4"]),
             "Interpolates `vUV` directly from varying buffer offset `r4` and feeds both texture units without intermediate UV GPRs."),
            ("FP16 varying source (`.src_f16`)",
             fmt_asm_line("VAR_TEX_BUF_DUAL", ".tex2d.src_f16.computed.slot0", "@r0:r7", ["u0", "r59", "r4"]),
             "Reads `mediump` FP16 UVs from the varying buffer and issues paired filtered samples."),
        ],
    ),
    "VAR_TEX_BUF_GATHER": (
        "OpLoad (in) + OpImageGather", "vec4 g = textureGather(s, vUV, 0);",
        [
            ("Fused varying-buffer gather (red channel)",
             fmt_asm_line("VAR_TEX_BUF_GATHER", ".tex2d.gather4_r.slot0", "@r0:r3", ["u0", "r59", "r4"]),
             "Interpolates `vUV` from varying buffer offset `r4` and gathers the 2x2 red-channel footprint in one message."),
            ("Fused shadow-compare gather (`.compare`)",
             fmt_asm_line("VAR_TEX_BUF_GATHER", ".tex2d.compare.slot0", "@r0:r3", ["u0", "r59", "r4"]),
             "Interpolates `(u, v, refZ)` from varying buffer and returns 4 shadow-comparison results."),
        ],
    ),
    "VAR_TEX_BUF_GRADIENT": (
        "OpLoad (in) + OpImageSampleExplicitLod", "vec4 c = textureGrad(s, vUV, dx, dy);",
        [
            ("Fused varying-buffer gradient query",
             fmt_asm_line("VAR_TEX_BUF_GRADIENT", ".tex2d.slot0", "@r0:r3", ["u0", "r59", "r4"]),
             "Interpolates `vUV` from varying buffer and computes screen-space texture gradients / LOD in one fused message."),
            ("Forced quad derivatives (`.force_delta`)",
             fmt_asm_line("VAR_TEX_BUF_GRADIENT", ".tex2d.force_delta.slot0", "@r0:r1", ["u0", "r59", "r4"]),
             "Evaluates 2x2 quad derivatives across helper lanes for accurate mip selection."),
        ],
    ),
    "VAR_TEX_BUF_SINGLE": (
        "OpLoad (in) + OpImageSampleImplicitLod", "vec4 c = texture(s, vUV);",
        [
            ("Fused varying-buffer 2D texture sample",
             fmt_asm_line("VAR_TEX_BUF_SINGLE", ".tex2d.computed.slot0", "@r0:r3", ["u0", "r59", "r4"]),
             "Interpolates `vUV` from varying buffer offset `r4` directly into `TEX_SINGLE`, saving an entire `LD_VAR_BUF` + scoreboard wait."),
            ("Fused projective texture sample (`.project`)",
             fmt_asm_line("VAR_TEX_BUF_SINGLE", ".tex2d.project.computed.slot0", "@r0:r3", ["u0", "r59", "r4"]),
             "Interpolates homogeneous `(u, v, q)` and performs hardware `textureProj(s, vUVQ)` in one instruction."),
        ],
    ),
    "VAR_TEX_DUAL": (
        "OpLoad (in) + 2x OpImageSample*", "vec4 a=texture(s0,vUV); vec4 b=texture(s1,vUV);",
        [
            ("Descriptor varying + dual 2D texture sample",
             fmt_asm_line("VAR_TEX_DUAL", ".tex2d.center.slot0", "@r0:r7", ["u0", "r59", "u1"]),
             "Interpolates descriptor-based varying `u1` at pixel center and issues two paired samples via `u0`."),
            ("Centroid-interpolated dual sample",
             fmt_asm_line("VAR_TEX_DUAL", ".tex2d.centroid.slot0", "@r0:r7", ["u0", "r59", "u1"]),
             "Uses MSAA centroid interpolation for `vUV` to prevent out-of-primitive UV extrapolation on both textures."),
        ],
    ),
    "VAR_TEX_GATHER": (
        "OpLoad (in) + OpImageGather", "vec4 g = textureGather(s, vUV, 0);",
        [
            ("Descriptor varying + 2x2 red gather",
             fmt_asm_line("VAR_TEX_GATHER", ".tex2d.gather4_r.center.slot0", "@r0:r3", ["u0", "r59", "u1"]),
             "Fuses descriptor-based `vUV` interpolation (`u1`) with `textureGather(..., 0)` into `@r0:r3`."),
            ("Green-channel gather (`.gather4_g`)",
             fmt_asm_line("VAR_TEX_GATHER", ".tex2d.gather4_g.center.slot0", "@r0:r3", ["u0", "r59", "u1"]),
             "Gathers the 2x2 green channel footprint directly from interpolated `vUV`."),
        ],
    ),
    "VAR_TEX_GRADIENT": (
        "OpLoad (in) + OpImageSampleExplicitLod", "vec4 c = textureGrad(s, vUV, dx, dy);",
        [
            ("Descriptor varying + explicit gradient/LOD",
             fmt_asm_line("VAR_TEX_GRADIENT", ".tex2d.center.slot0", "@r0:r3", ["u0", "r59", "u1"]),
             "Fuses descriptor-based `vUV` interpolation (`u1`) with hardware texture gradient evaluation."),
            ("Raw derivative output (`.derivative`)",
             fmt_asm_line("VAR_TEX_GRADIENT", ".tex2d.derivative.center.slot0", "@r0:r3", ["u0", "r59", "u1"]),
             "Outputs raw `(dU/dx, dV/dx, dU/dy, dV/dy)` derivatives directly into `@r0:r3`."),
        ],
    ),
    "VAR_TEX_SINGLE": (
        "OpLoad (in) + OpImageSampleImplicitLod", "vec4 c = texture(s, vUV);",
        [
            ("Descriptor varying + implicit-LOD 2D sample",
             fmt_asm_line("VAR_TEX_SINGLE", ".tex2d.computed.center.slot0", "@r0:r3", ["u0", "r59", "u1"]),
             "Fuses descriptor-based `vUV` interpolation (`u1`) directly into `TEX_SINGLE` (`u0`), returning RGBA in `@r0:r3`."),
            ("Arrayed cubemap / 2D-array fused sample (`.array`)",
             fmt_asm_line("VAR_TEX_SINGLE", ".tex2d.array.computed.center.slot0", "@r0:r3", ["u0", "r59", "u1"]),
             "Interpolates `(u, v, layer)` from varying `u1` and samples a `sampler2DArray` in a single message."),
        ],
    ),
}

# Programmatically populate all 20 fused shift/rotate-logic mnemonics cleanly (2 examples each!)
_SH_META = {
    "LSHIFT": ("OpShiftLeftLogical", "<<", "0x00000010", "4", "0x00000100", "0x00000001", "33", "0x00000002"),
    "RSHIFT": ("OpShiftRightLogical", ">> (uint)", "0x80000000", "4", "0x08000000", "0x80000000", "32", "0x80000000"),
    "ARSHIFT": ("OpShiftRightArithmetic", ">> (int)", "0x80000000", "4", "0xF8000000", "0x80000000", "31", "0xFFFFFFFF"),
    "LROT": ("None (lowered from `(x<<s)|(x>>(32-s))`)", "rotl", "0x80000001", "1", "0x00000003", "0x12345678", "8", "0x34567812"),
    "RROT": ("None (lowered from `(x>>s)|(x<<(32-s))`)", "rotr", "0x00000003", "1", "0x80000001", "0x12345678", "8", "0x78123456"),
}
_LOP_META = {
    "": ("", "", "", None),
    "_AND": (" + OpBitwiseAnd", " & c", " & 0x0FFFFFFF", lambda s: hex(int(s, 16) & 0x0FFFFFFF)),
    "_OR": (" + OpBitwiseOr", " | c", " | 0x0000000F", lambda s: hex(int(s, 16) | 0x0000000F)),
    "_XOR": (" + OpBitwiseXor", " ^ c", " ^ 0x000000FF", lambda s: hex(int(s, 16) ^ 0x000000FF)),
}
for _sh, (_spv_sh, _gl_sh, _v_in, _amt, _v_sh, _v2_in, _amt2, _v2_sh) in _SH_META.items():
    for _lop, (_spv_l, _gl_l, _op_txt, _fn_l) in _LOP_META.items():
        _mn = _sh + _lop
        if _mn in INST_EXTRAS:
            continue
        _spv = _spv_sh + _spv_l
        if _sh in ("LROT", "RROT"):
            _gl = f"uint y = ((x {'<<' if _sh == 'LROT' else '>>'} s) | (x {'>>' if _sh == 'LROT' else '<<'} (32u - s))){_gl_l};"
        else:
            _t_cast = "int" if _sh == "ARSHIFT" else "uint"
            _gl = f"{_t_cast} y = (a {_gl_sh.split()[0]} s){_gl_l};"
        if _lop:
            _asm1 = fmt_asm_line(f"{_mn}.i32", "", "r0", ["r1", "r2.b0", "r3"])
            _res1 = _fn_l(_v_sh).upper().replace("0X", "0x")
            _walk1 = (
                f"With `r1 = {_v_in}`, `r2.b0 = {_amt}` (masked mod 32): shift yields `{_v_sh}`, "
                f"then `{_op_txt.strip()}` yields `r0 = {_res1}`."
            )
            _asm2 = fmt_asm_line(f"{_mn}.i32", ".not_result", "r0", ["r1", "r2.b0", "r3"])
            _res2_raw = int(_fn_l(_v2_sh), 16)
            _res2_inv = (~_res2_raw) & 0xFFFFFFFF
            _walk2 = (
                f"With `r1 = {_v2_in}`, `r2.b0 = {_amt2}` (effective shift `{int(_amt2) & 31}`): shift yields `{_v2_sh}`, "
                f"`{_op_txt.strip()}` yields `0x{_res2_raw:08X}`, and `.not_result` inverts -> `r0 = 0x{_res2_inv:08X}`."
            )
            INST_EXTRAS[_mn] = (
                _spv,
                _gl,
                [
                    ("Standard fused shift + bitwise op", _asm1, _walk1),
                    ("Modulo-32 shift count & .not_result", _asm2, _walk2),
                ],
            )
        else:
            _asm1 = fmt_asm_line(f"{_mn}.i32", "", "r0", ["r1", "r2.b0"])
            _walk1 = f"With `r1 = {_v_in}`, `r2.b0 = {_amt}`: yields `r0 = {_v_sh}`."
            _asm2 = fmt_asm_line(f"{_mn}.i32", "", "r0", ["r1", "r2.b1"])
            _walk2 = (
                f"With `r1 = {_v2_in}`, `r2.b1 = {_amt2}` (shift count masked mod 32 -> `{int(_amt2) & 31}`): "
                f"yields `r0 = {_v2_sh}`."
            )
            INST_EXTRAS[_mn] = (
                _spv,
                _gl,
                [
                    ("Canonical 32-bit shift/rotate", _asm1, _walk1),
                    ("Byte-lane selector & modulo-32 masking", _asm2, _walk2),
                ],
            )


# ============================================================================
# Chapter 2.1--2.50: Curated Enumerations with Multiple Combined Walkthroughs
# ============================================================================
# Format: (ename, title_tex, desc_md, [(label, asm_tex, walkthrough_md), ...])
KEY_ENUMS = [
    ("flow_control_m", "Flow Control (flow\\_control\\_m, Bits 59:62)",
     "Embedded scheduling, scoreboard waits, warp reconvergence, and shader termination.",
     [
         ("Wait on slot0 + warp reconvergence",
          fmt_asm_line("FADD.f32", ".wait0.reconverge", "r0", ["r1", "r2"]),
          "Stalls until the asynchronous message in `slot0` writes `r1`, executes `FADD.f32`, and marks the hardware reconvergence point after a divergent branch."),
         ("Full scoreboard drain + shader end",
          fmt_asm_line("NOP", ".wait.end", "", []),
          "Drains all in-flight asynchronous slots (`slot0..slot7`) so all SSBO/tile writes complete before releasing the warp (`.end`)."),
     ]),
    ("widen_m", "FP32 Source Widen (widen\\_m)",
     "Widens an FP16 lane to exact IEEE-754 FP32 before applying `abs`/`neg`. Default: `none`.",
     [
         ("Widen upper FP16 lane then negate",
          fmt_asm_line("FADD.f32", "", "r0", ["r1.h1.neg", "r2"]),
          "With `r1 = 0x3C000000` (`h1 = 1.0h`), `r2 = 5.0f`: Stage A widens `h1 -> 1.0f` first, then applies `.neg -> -1.0f`, then adds `r2` -> `r0 = 4.0f`."),
         ("Widen low FP16 subnormal + abs",
          fmt_asm_line("FMA.f32", "", "r0", ["r1.h0.abs", "r2", "r3"]),
          "With `r1.h0 = 0x8001` (`-2^-24h`): Stage A promotes to exact FP32 `-2^-24f`, `.abs` clears sign -> `+2^-24f`, then multiplies by `r2` and adds `r3`."),
     ]),
    ("swiz_m", "FP16 Vector Swizzle (swiz\\_m)",
     "Halfword swizzle for `v2f16` operands (`hXY` places half `X` in `[15:0]`, `Y` in `[31:16]`). Default: `h01`.",
     [
         ("Swap + broadcast in v2f16 add",
          fmt_asm_line("FADD.v2f16", "", "r0", ["r1.h10", "r2.h00"]),
          "With `r1 = [2.0h, 1.0h]` (`0x40003C00`), `r2 = [4.0h, 0.5h]` (`0x44003800`): `r1.h10 = [1.0h, 2.0h]`, `r2.h00 = [0.5h, 0.5h]` -> `r0 = [1.5h, 2.5h]`."),
         ("Broadcast high lane with negation",
          fmt_asm_line("FMUL.v2f16", "", "r0", ["r1.h01", "r2.h11.neg"]),
          "With `r1 = [3.0h, 2.0h]`, `r2 = [4.0h, 1.0h]`: `r2.h11` broadcasts `4.0h` and `.neg` gives `[-4.0h, -4.0h]` -> `r0 = [-12.0h, -8.0h]`."),
     ]),
    ("widen_int_m", "Integer Source Widen (widen\\_int\\_m)",
     "Sign-extends (`s32`/`s64`) or zero-extends (`u32`/`u64`) narrow byte/half/word lanes. On 64-bit ops, byte/half swizzles fold low 32 bits then sign-extend to 64. Default: `none`.",
     [
         ("Signed byte widen on IADD.s32",
          fmt_asm_line("IADD.s32", "", "r0", ["r1.b0", "r2"]),
          "With `r1 = 0x000000FE` (`-2` in `b0`), `r2 = 10`: Stage A sign-extends `b0 -> 0xFFFFFFFE`; Stage B computes `r0 = 8`."),
         ("Unsigned word widen on IMUL.u64",
          fmt_asm_line("IMUL.u64", "", "[r0:r1]", ["[r2:r3].w0", "[r4:r5].w0"]),
          "With `r2 = 0x80000000, r4 = 4`: zero-extends `.w0` to `0x00000000_80000000` (not sign-extended) -> `[r0:r1] = 0x00000002_00000000`."),
     ]),
    ("swiz_int_m", "16-Bit Integer Swizzle (swiz\\_int\\_m)",
     "Halfword swizzle or 8-to-16 byte widening (`bXY`: sign-extends on `v2s16`, zero-extends on `v2u16`). Default: `h01`.",
     [
         ("Signed vs unsigned byte-pair widen (`b02`)",
          fmt_asm_line("IADD.v2s16", "", "r0", ["r1.b02", "k0"]),
          "With `r1 = 0x00FE00FF` (`b0 = -1, b2 = -2`): `v2s16` sign-extends both bytes -> `r0 = 0xFFFEFFFF` (whereas `v2u16` yields `0x00FE00FF`)."),
         ("Swap halfwords (`h10`) before subtraction",
          fmt_asm_line("ISUB.v2s16", "", "r0", ["r1.h10", "r2.h01"]),
          "With `r1 = [10, 3]` (`0x000A0003`), `r2 = [2, 1]` (`0x00020001`): `r1.h10 = [3, 10]`, subtracting `[2, 1]` -> `r0 = [1, 9]` (`0x00010009`)."),
     ]),
    ("swiz8_2_full_m", "2-Byte Full Swizzle (swiz8\\_2\\_full\\_m)",
     "Extracts any 2 of 4 bytes into `[7:0]` and `[23:16]`. Default `b02` on `V2*8_TO_*`; virtual fixed `b01` on `MKVEC.v2i8`.",
     [
         ("Cross-lane byte swap (`b20`) on signed widen",
          fmt_asm_line("V2S8_TO_V2S16", "", "r0", ["r1.b20"]),
          "With `r1 = 0x000500FE`: `b20` puts `b2 = +5` in lane 0 and `b0 = -2` in lane 1 -> `r0 = 0xFFFE0005`."),
         ("Duplicate high byte (`b33`) on unsigned widen",
          fmt_asm_line("V2U8_TO_V2U16", "", "r0", ["r1.b33"]),
          "With `r1 = 0xAB123456`: `b33` zero-extends byte 3 (`0xAB`) into both lanes -> `r0 = 0x00AB00AB`."),
     ]),
    ("lanes_int_m", "8-Bit Vector Lanes (lanes\\_int\\_m)",
     "4-byte permutation for `v4s8`/`v4u8`/`v4i8` (arch v9--v10). Default: `b0123`.",
     [
         ("Endianness byte reversal (`b3210`)",
          fmt_asm_line("SWZ.v4i8", "", "r0", ["r1.b3210"]),
          "With `r1 = 0x11223344`: `b3210` reverses all 4 bytes -> `r0 = 0x44332211`."),
         ("Byte broadcast (`b0000`) in packed add",
          fmt_asm_line("IADD.v4u8", ".sat", "r0", ["r1.b0123", "r2.b0000"]),
          "With `r1 = 0xF0201000`, `r2.b0 = 0x20`: broadcasts `0x20` to all 4 bytes, saturating byte 3 (`0xF0+0x20 -> 0xFF`) -> `r0 = 0xFF403020`."),
     ]),
    ("lane8_m", "8-Bit Single Lane (lane8\\_m)",
     "Extracts a single byte `b0..b3`. On `v4*8` replicates across 4 lanes; on `v2*16` widens and replicates to 2x16. Default: `b0`.",
     [
         ("Extract byte 2 and zero-extend",
          fmt_asm_line("U8_TO_U32", "", "r0", ["r1.b2"]),
          "With `r1 = 0x00AB0000`: extracts byte 2 (`0xAB`) -> `r0 = 0x000000AB`."),
         ("Extract byte 3 and sign-extend",
          fmt_asm_line("S8_TO_S32", "", "r0", ["r1.b3"]),
          "With `r1 = 0xFE000000`: extracts byte 3 (`-2`) and sign-extends -> `r0 = 0xFFFFFFFE`."),
     ]),
    ("lane16_m", "16-Bit Single Lane (lane16\\_m)",
     "Extracts a single 16-bit halfword `h0` or `h1`. Default: `h0`.",
     [
         ("Extract upper FP16 halfword (`h1`)",
          fmt_asm_line("F16_TO_F32", "", "r0", ["r1.h1"]),
          "With `r1 = 0x40003C00`: extracts `h1 = 0x4000` (`2.0h`) -> `r0 = 0x40000000` (`2.0f`)."),
         ("Extract low signed 16-bit halfword (`h0`)",
          fmt_asm_line("S16_TO_S32", "", "r0", ["r1.h0"]),
          "With `r1 = 0x1234FFFC`: extracts `h0 = 0xFFFC` (`-4`) and sign-extends -> `r0 = 0xFFFFFFFC`."),
     ]),
    ("shift_bitwise_i32_lanes_m", "32/64-Bit Shift Amount Lane (shift\\_bitwise\\_i32\\_lanes\\_m)",
     "Selects unsigned 8-bit shift count byte from `src1` (masked modulo lane width: 32 or 64). Default: `b0`.",
     [
         ("Read shift amount from byte 1 (`b1`)",
          fmt_asm_line("LSHIFT.i32", "", "r0", ["r1", "r2.b1"]),
          "With `r1 = 1, r2 = 0x00000300`: reads `b1 = 3` -> `r0 = 1 << 3 = 8`."),
         ("Modulo-32 wrap on shift count >= 32",
          fmt_asm_line("RSHIFT.i32", "", "r0", ["r1", "r2.b0"]),
          "With `r1 = 0x80000000, r2.b0 = 33`: hardware masks `33 & 31 = 1` -> `r0 = 0x40000000` (NOT 0)."),
     ]),
    ("shift_bitwise_i16_lanes_m", "16-Bit Shift Amount Lanes (shift\\_bitwise\\_i16\\_lanes\\_m)",
     "Selects two unsigned 8-bit shift counts (lane 0 & lane 1, masked mod 16) on `v2i16` shifts. Default: `b02`.",
     [
         ("Independent per-lane shift counts (`b02`)",
          fmt_asm_line("LSHIFT.v2i16", "", "r0", ["r1", "r2.b02"]),
          "With `r1 = 0x00010001, r2 = 0x00110010` (`b0=16 -> 0`, `b2=17 -> 1`): lane 0 shifts by 0, lane 1 shifts by 1 -> `r0 = 0x00020001`."),
         ("Shared shift count (`b00`)",
          fmt_asm_line("RSHIFT.v2i16", "", "r0", ["r1", "r2.b00"]),
          "With `r1 = 0x00800040, r2.b0 = 2`: shifts both 16-bit lanes right by 2 -> `r0 = 0x00200010`."),
     ]),
    ("abs_m", "Pre-Source Floating-Point Abs (abs\\_m)",
     "Clears IEEE sign bit(s) after swizzle (`& 0x7FFFFFFF` f32, `& 0x7FFF7FFF` f16/v2f16). Default: `none`.",
     [
         ("Pre-source `.abs` on FP32 add",
          fmt_asm_line("FADD.f32", "", "r0", ["r1.abs", "r2"]),
          "With `r1 = -3.0f, r2 = 1.0f`: Stage A clears sign bit of `r1` (`+3.0f`) -> `r0 = 4.0f`."),
         ("Combined `.abs.neg` (always negative magnitude)",
          fmt_asm_line("FADD.f32", "", "r0", ["r1.abs.neg", "r2"]),
          "With `r1 = +3.0f, r2 = 10.0f`: `.abs` runs before `.neg`, forcing `-|r1| = -3.0f` -> `r0 = 7.0f`."),
     ]),
    ("neg_m", "Pre-Source Floating-Point Neg (neg\\_m)",
     "Flips IEEE sign bit(s) after swizzle (`^ 0x80000000` f32, `^ 0x80008000` f16/v2f16). Default: `none`.",
     [
         ("Floating-point subtraction via `src1.neg`",
          fmt_asm_line("FADD.f32", "", "r0", ["r1", "r2.neg"]),
          "With `r1 = 5.0f, r2 = 2.0f`: Stage A negates `r2` (`-2.0f`) -> `r0 = 3.0f` (Valhall has no separate `FSUB` opcode)."),
         ("FMA negated product (`-a*b + c`)",
          fmt_asm_line("FMA.f32", "", "r0", ["r1.neg", "r2", "r3"]),
          "With `r1 = 3.0f, r2 = 4.0f, r3 = 20.0f`: Stage A negates `r1` to `-3.0f` -> `r0 = -12.0 + 20.0 = 8.0f`."),
     ]),
    ("not_m", "Bitwise NOT Modifier (not\\_m)",
     "Bitwise complement `~x` on `src2` (`not2`) or result (`not_result`) of shift-logic ops. Default: `none`.",
     [
         ("Combined `src2.not` and `.not_result`",
          fmt_asm_line("AND.i32", ".not_result", "r0", ["r1", "r2.not"]),
          "With `r1 = 0x0F, r2 = 0x03`: `~(0x0F & ~0x03) = ~(0x0C) = 0xFFFFFFF3`."),
         ("Bitwise BIC (`a & ~b`) without `.not_result`",
          fmt_asm_line("AND.i32", "", "r0", ["r1", "r2.not"]),
          "With `r1 = 0xFFFF00FF, r2 = 0x00FF000F`: clears masked bits -> `r0 = 0xFF0000F0`."),
     ]),
    ("clamp_m", "Floating-Point Clamp (clamp\\_m)",
     "Post-op IEEE-754 clamp with deterministic `-0.0 < +0.0` and NaN suppression (`NaN -> lo`). Default: `none`.",
     [
         ("`.clamp_0_1` on `-0.0` and `NaN`",
          fmt_asm_line("FMA.f32", ".clamp_0_1", "r0", ["r1", "r2", "r3"]),
          "With result `-0.0f` or `NaN`: `.clamp_0_1` clamps to `+0.0f` (`0x00000000`), never `-0.0f` or `NaN`."),
         ("`.clamp_m1_1` NaN-to-low-bound rule",
          fmt_asm_line("FADD.f32", ".clamp_m1_1", "r0", ["r1", "r2"]),
          "With result `NaN` or `-2.5f`: `.clamp_m1_1` returns `-1.0f` (`0xBF800000`); with `+3.0f`, returns `+1.0f` (`0x3F800000`)."),
     ]),
    ("round_m", "Floating-Point Rounding (round\\_m)",
     "Standard 4 IEEE-754 rounding modes for FP ALU and Stage C `dest_width_narrow`. Default: `none` (RNE).",
     [
         ("Stage C FP32 -> FP16 narrowing with `.round_zero`",
          fmt_asm_line("FADD.f32", ".round_zero", "r0.h0", ["r1", "r2"]),
          "With `1.0f + 2^-11`: RNE rounds up to `0x3C02` in FP16, whereas `.round_zero` truncates to `0x3C01`."),
         ("Directed rounding `.round_up` vs `.round_down`",
          fmt_asm_line("FADD.f32", ".round_up", "r0", ["r1", "r2"]),
          "With `1.0f + 2^-25`: RNE returns `1.0f` (`0x3F800000`), whereas `.round_up` rounds toward `+Inf` -> `0x3F800001`."),
     ]),
    ("round_integer_m", "Integer / Extended Rounding (round\\_integer\\_m)",
     "Adds `round_na` (ties away from zero) for `FROUND` and float-to-int conversions. Default: `none` (RNE).",
     [
         ("Halfway tie: RNE vs `.round_na`",
          fmt_asm_line("FROUND.f32", ".round_na", "r0", ["r1"]),
          "With `r1 = 2.5f`: RNE returns `2.0f` (even), whereas `.round_na` rounds ties away from zero -> `3.0f` (`-2.5f -> -3.0f`)."),
         ("GLSL `int(x)` truncation with `.round_zero`",
          fmt_asm_line("F32_TO_S32", ".round_zero", "r0", ["r1"]),
          "With `r1 = -3.9f`: `.round_zero` truncates to `-3`, whereas `.round_down` (`floor`) would yield `-4`."),
     ]),
    ("hadd_round_m", "Halving-Add Rounding (hadd\\_round\\_m)",
     "Selects floor (`round_down`) vs ceil (`round_up`) for v9--v10 `HADD`. Default: `round_down`.",
     [
         ("Odd sum floor vs ceil",
          fmt_asm_line("HADD.u32", ".round_up", "r0", ["r1", "r2"]),
          "With `r1 = 1, r2 = 2`: `round_down` gives `(1+2)>>1 = 1`; `round_up` gives `(1+2+1)>>1 = 2`."),
         ("Signed negative odd sum",
          fmt_asm_line("HADD.s32", ".round_down", "r0", ["r1", "r2"]),
          "With `r1 = -3, r2 = -2`: arithmetic shift `(-5) >> 1 = -3` under `.round_down`, vs `(-5 + 1) >> 1 = -2` under `.round_up`."),
     ]),
    ("saturate_m", "Integer Saturation (saturate\\_m)",
     "Clamps integer overflow/underflow to `[MIN_N, MAX_N]` or `[0, 2^N - 1]`. Default: `none`.",
     [
         ("Signed positive overflow clamp",
          fmt_asm_line("IADD.s32", ".sat", "r0", ["r1", "r2"]),
          "With `r1 = 0x7FFFFFFF, r2 = 1`: wraps to `0x80000000` without `.sat`; saturates to `0x7FFFFFFF` with `.sat`."),
         ("Unsigned subtraction underflow clamp",
          fmt_asm_line("ISUB.u32", ".sat", "r0", ["r1", "r2"]),
          "With `r1 = 10, r2 = 25`: `.sat` clamps underflow to `r0 = 0` instead of wrapping to `0xFFFFFFF1`."),
     ]),
    ("sem_m", "FMIN/FMAX NaN Semantics (sem\\_m)",
     "Selects NaN suppression (default) vs payload-preserving NaN propagation (`| qbit`). Applied before `clamp`.",
     [
         ("Payload-preserving NaN propagation",
          fmt_asm_line("FMIN.f32", ".nan_propagate", "r0", ["r1", "r2"]),
          "With `r1 = 0xFF800001` (sNaN), `r2 = 1.0f`: `nan_suppress` returns `1.0f`; `nan_propagate` sets bit 22 and returns `0xFFC00001`."),
         ("Deterministic signed zero regardless of `sem`",
          fmt_asm_line("FMAX.f32", ".nan_suppress", "r0", ["r1", "r2"]),
          "With `r1 = -0.0f, r2 = +0.0f`: both `nan_suppress` and `nan_propagate` deterministically return `+0.0f` (`FMIN` returns `-0.0f`)."),
     ]),
    ("flush_ftz_m", "FLUSH Denormal Mode (flush\\_ftz\\_m)",
     "Flushes subnormal inputs to signed zero in `FLUSH`. Default: `none`.",
     [
         ("Negative subnormal FTZ",
          fmt_asm_line("FLUSH.f32", ".ftz", "r0", ["r1"]),
          "With `r1 = 0x80000001` (negative subnormal): `.ftz` flushes to `-0.0f` (`0x80000000`), preserving sign."),
         ("Normal value passes unchanged",
          fmt_asm_line("FLUSH.f32", ".ftz", "r0", ["r1"]),
          "With `r1 = 0x00800000` (`FLT_MIN`, smallest normal): exponent is non-zero, so `r0 = 0x00800000` unchanged."),
     ]),
    ("flush_inf_m", "FLUSH Infinity Mode (flush\\_inf\\_m)",
     "Clamps `±Inf` to `±MAX_FINITE` in `FLUSH`. Default: `none`.",
     [
         ("Clamp `+Inf` to `+FLT_MAX`",
          fmt_asm_line("FLUSH.f32", ".flush_inf", "r0", ["r1"]),
          "With `r1 = 0x7F800000` (`+Inf`): `.flush_inf` clamps to `0x7F7FFFFF` (`+3.4028235e38f`)."),
         ("Clamp `-Inf` to `-FLT_MAX`",
          fmt_asm_line("FLUSH.f32", ".flush_inf", "r0", ["r1"]),
          "With `r1 = 0xFF800000` (`-Inf`): `.flush_inf` clamps to `0xFF7FFFFF` (`-3.4028235e38f`)."),
     ]),
    ("flush_nan_m", "FLUSH NaN Mode (flush\\_nan\\_m)",
     "Flushes NaN to `+0.0` (`flush_nan`) or quietens sNaN (`quiet_nan`). Default: `none`.",
     [
         ("Flush NaN to `+0.0`",
          fmt_asm_line("FLUSH.f32", ".flush_nan", "r0", ["r1"]),
          "With `r1 = 0x7FC00000` (qNaN) or `0xFF800001` (sNaN): `.flush_nan` returns `+0.0f` (`0x00000000`)."),
         ("Quieten sNaN preserving payload (`.quiet_nan`)",
          fmt_asm_line("FLUSH.f32", ".quiet_nan", "r0", ["r1"]),
          "With `r1 = 0x7F800042` (sNaN): `.quiet_nan` sets quiet bit 22 -> `r0 = 0x7FC00042`."),
     ]),
    ("fma_rscale_special_32_m", "FMA\\_RSCALE Special Modes (fma\\_rscale\\_special\\_32\\_m)",
     "Controls `0 * Inf` and `Inf` addend overrides in `FMA_RSCALE`/`FADD_RSCALE`/`FMUL_RSCALE`. Default: `none`.",
     [
         ("Suppress `0 * Inf` NaN with `.n`",
          fmt_asm_line("FMA_RSCALE.f32", ".n", "r0", ["k0", "r_inf", "k1f", "k0"]),
          "With `0 * Inf + 1.0`: default yields `NaN`; `.n` forces product to `0.0`, yielding `1.0f`."),
         ("Preserve left operand with `.left` on `FMUL_RSCALE`",
          fmt_asm_line("FMUL_RSCALE.f32", ".left", "r0", ["r1", "r_inf", "k0"]),
          "With `r1 = -0.0f, r_inf = +Inf`: `.left` returns `r1` (`-0.0f`) instead of `NaN`."),
     ]),
    ("frexp_special_m", "FREXP Normalization Mode (frexp\\_special\\_m)",
     "Selects significand interval: `none` `[0.5, 1)`, `sqrt` `[0.25, 1)` (`4^E`), `log` `[0.75, 1.5)`. Default: `none`.",
     [
         ("Infinity override under `.log`",
          fmt_asm_line("FREXPM.f32", ".log", "r0", ["r_inf"]),
          "On `+Inf`: `none` and `sqrt` return `+Inf`, whereas `.log` returns `+1.0f` (`0x3F800000`)."),
         ("Even-exponent `.sqrt` decomposition",
          fmt_asm_line("FREXPM.f32", ".sqrt", "r0", ["r1"]),
          "With `r1 = 2.0f`: `.sqrt` normalizes into `[0.25, 1.0)` -> `0.5f` with `FREXPE.sqrt = 1` (`0.5 * 4^1 = 2.0`)."),
     ]),
    ("ldexp_inf_m", "LDEXP Special Threshold Mode (ldexp\\_inf\\_m)",
     "Overrides `±Inf`/`±0.0` at extreme exponents `exp < -16384` (`.inf`) and `exp >= +16384` (`.inf0`). Default: `none`.",
     [
         ("Negative extreme exponent (`.inf`)",
          fmt_asm_line("LDEXP.f32", ".inf", "r0", ["r_inf", "r_neg20000"]),
          "With `r_inf = +Inf, exp = -20000 (< -16384)`: `.inf` returns `+1.0f` (used in `atan2` denominator scaling)."),
         ("Positive extreme exponent (`.inf0`)",
          fmt_asm_line("LDEXP.f32", ".inf0", "r0", ["k0", "r_pos20000"]),
          "With `k0 = +0.0f, exp = +20000 (>= 16384)`: `.inf0` overrides `0 * 2^20000` to `+1.0f` (`0x3F800000`)."),
     ]),
    ("cmpf_m", "Comparison Condition (cmpf\\_m)",
     "Condition code for `FCMP*`, `ICMP*`, `CSEL`, and `DISCARD.f32`. FP `ne` is unordered; `gtlt`/`total` are FP-only.",
     [
         ("Ordered `.gtlt` vs unordered `.ne` on NaN",
          fmt_asm_line("FCMP.f32", ".gtlt.i1", "r0", ["r_nan", "k0"]),
          "On `(NaN, 0.0)`: `.ne` returns `1` (unordered), whereas `.gtlt` returns `0` (ordered not-equal)."),
         ("IEEE `.total` ordering of `-0.0` vs `+0.0`",
          fmt_asm_line("FCMP.f32", ".total.i1", "r0", ["r_neg0", "k0"]),
          "Standard `.lt` says `-0.0 < +0.0` is `0` (false), whereas `.total` (IEEE `totalOrder`) returns `1` (`-0.0 < +0.0`)."),
     ]),
    ("cmp_result_type_m", "Comparison Result Format (cmp\\_result\\_type\\_m)",
     "Formats boolean comparison output (`i1`: 1, `f1`: 1.0, `m1`: all-ones, `c`: 3-way `{-1,0,+1}` on `ICMP_MULTI.u32` ONLY). Default: `i1`.",
     [
         ("Three boolean encodings (`.i1`, `.f1`, `.m1`)",
          fmt_asm_line("ICMP.u32", ".eq.m1", "r0", ["r1", "r1"]),
          "On equal operands: `.i1` writes `0x00000001`, `.f1` writes `0x3F800000` (`1.0f`), `.m1` writes `0xFFFFFFFF` (`-1`)."),
         ("3-way chained carry (`.c`) on `ICMP_MULTI.u32`",
          fmt_asm_line("ICMP_MULTI.u32", ".lt.c", "r0", ["r1", "r2", "k0"]),
          "Outputs `-1` (`0xFFFFFFFF`) if `r1 < r2`, `0` if `r1 == r2`, or `+1` if `r1 > r2` for 64-bit chained comparison."),
     ]),
    ("mux_cmpf_m", "Mux Condition (mux\\_cmpf\\_m)",
     "Condition tested on `src2` selector in `MUX`. Default: `int_zero` (encoding `1`, NOT `bit`!).",
     [
         ("Default `.int_zero` selection",
          fmt_asm_line("MUX.i32", ".int_zero", "r0", ["r1", "r2", "r3"]),
          "With `r1 = 0xAAAA, r2 = 0xBBBB, r3 = 0`: `int_zero` sees `r3 == 0` -> selects `r1 = 0xAAAA` (if `r3 != 0`, selects `r2`)."),
         ("Per-bit blend (`.bit`) vs sign-bit select (`.neg`)",
          fmt_asm_line("MUX.i32", ".bit", "r0", ["r1", "r2", "r3"]),
          "With `r1 = 0x0000FFFF, r2 = 0xFFFF0000, r3 = 0x0F0F0F0F`: blends per bit -> `r0 = 0xF0F00F0F`. Under `.neg`, bit 31 of `r3` chooses the entire word."),
     ]),
    ("dest_width_m", "Destination Lane Write-Mask (dest\\_width\\_m)",
     "Pure 16-bit lane write-mask on 32-bit/packed ops (NO bit shift!). Default: `none` (full 32-bit).",
     [
         ("Upper halfword write-mask (`r0.h1`)",
          fmt_asm_line("FADD.v2f16", "", "r0.h1", ["r1", "r2"]),
          "Computes both FP16 lanes; `.h1` writes ONLY lane 1 (`val32[31:16]`) into `r0[31:16]`, preserving `r0[15:0]`."),
         ("Low halfword write-mask (`r0.h0`) on `MOV_IMM`",
          fmt_asm_line("MOV_IMM.i32", "", "r0.h0", [], ["#0xAAAA1234"]),
          "With `r0_old = 0xDEAD0000`: `.h0` writes `0x1234` to `r0[15:0]` and preserves `r0[31:16]` -> `r0 = 0xDEAD1234`."),
     ]),
    ("dest_width_narrow_m", "Narrowing Destination Writeback (dest\\_width\\_narrow\\_m)",
     "Fused FP32->FP16 narrowing using the instruction's `round` mode into `h0`, `h1`, or replicated `h01`. Default: `none`.",
     [
         ("Narrow + broadcast to both halfwords (`.h01`)",
          fmt_asm_line("FADD.f32", ".round_zero", "r0.h01", ["r1", "r2"]),
          "With `r1 = 1.0f, r2 = 1.0f`: narrows `2.0f -> 0x4000` and `.h01` replicates to both halves -> `r0 = 0x40004000`."),
         ("Narrow into upper half preserving low half (`.h1`)",
          fmt_asm_line("FMA.f32", "", "r0.h1", ["r1", "r2", "r3"]),
          "With `r0_old = 0x00003C00` and FP32 result `4.0f`: narrows to `0x4400` in `r0[31:16]` -> `r0 = 0x44003C00`."),
     ]),
    ("dest_width_replicate_m", "Replicating Scalar FP16 Destination (dest\\_width\\_replicate\\_m)",
     "Places scalar FP16 output (`FRCP/FRSQ/FEXP/FLOG/FPOW.f16`) into `h0`, `h1`, or replicated `h01`. Default: `none`.",
     [
         ("Broadcast scalar FP16 reciprocal (`.h01`)",
          fmt_asm_line("FRCP.f16", "", "r0.h01", ["r1.h0"]),
          "With `r1.h0 = 2.0h`: `1/2.0 = 0.5h` (`0x3800`), replicated to `r0 = 0x38003800`."),
         ("Write scalar FP16 into upper half only (`.h1`)",
          fmt_asm_line("FLOG.f16", "", "r0.h1", ["r1.h1"]),
          "With `r1.h1 = 4.0h`, `r0_old = 0x00003C00`: writes `log2(4.0h) = 2.0h` (`0x4000`) to `r0[31:16]` -> `r0 = 0x40003C00`."),
     ]),
    ("branch_combine_m", "Branch Condition Combine (branch\\_combine\\_m)",
     "Predicate reduction on `src0` before optional `.not` boolean negation. Default: `none` (`!= 0`).",
     [
         ("Both halfwords non-zero (`.and`)",
          fmt_asm_line("BRANCH", "", "", ["r4.and"], ["#0x20"]),
          "With `r4 = 0x00010001`: both halfwords non-zero -> taken; with `r4 = 0x0000FFFF`: upper half is 0 -> not taken."),
         ("Single halfword predicate (`.h1`)",
          fmt_asm_line("BRANCH", "", "", ["r4.h1"], ["#0x10"]),
          "Tests `r4[31:16] != 0` while ignoring `r4[15:0]` (useful after `v2*16` comparisons)."),
     ]),
    ("branch_negate_m", "Branch Condition Negate (branch\\_negate\\_m)",
     "Negates the boolean result of `combine(src0)` (NOT a bitwise NOT on `src0`). Default: `none`.",
     [
         ("Panfrost unconditional branch (`.not k0`)",
          fmt_asm_line("BRANCH", ".not", "", ["k0"], ["#0x10"]),
          "`k0 != 0` is `False`; `.not` flips it to `True` -> always branches to `PC + 8 + 0x10`."),
         ("Branch-if-zero (`if (r0 == 0)`)",
          fmt_asm_line("BRANCH", ".not", "", ["r0"], ["#0x28"]),
          "Taken when `r0 == 0` (`combine` = False, `.not` = True); falls through for any non-zero `r0`."),
     ]),
    ("jump_address_mode_m", "Jump Address Mode (jump\\_address\\_mode\\_m)",
     "Selects PC-relative vs absolute target address on `JUMP` (v11+ fixes `absolute`).",
     [
         ("Absolute blend-shader jump (`.absolute`)",
          fmt_asm_line("JUMP", ".absolute", "", ["k1n", "r54"]),
          "Jumps directly to 32-bit shader PC in `r54` (`r54 == 0` terminates the shader cleanly)."),
         ("PC-relative jump table (v9--v10)",
          fmt_asm_line("JUMP", "", "", ["k1n", "r2"]),
          "On v9--v10 with relative mode (`0`): adds signed offset `r2` to `PC + 8`."),
     ]),
    ("clper_subgroup_size_m", "Subgroup Size (clper\\_subgroup\\_size\\_m)",
     "Partitions 16-lane warp into 2-, 4-, 8-, or 16-lane subgroups for `CLPER*` and `WMASK`.",
     [
         ("4-lane fragment quad derivative (`.subgroup4`)",
          fmt_asm_line("CLPER", ".subgroup4.xor.zero", "r0", ["r1", "k01000000.b3"]),
          "Exchanges `r1` with horizontal quad neighbor (`lane ^ 1` inside each 4-lane quad, used for `dFdx`)."),
         ("Full 16-lane warp reduction (`.subgroup16`)",
          fmt_asm_line("WMASK", ".subgroup16", "r0", ["r1"]),
          "Collects predicate bits from all 16 lanes of the warp into a single 16-bit ballot mask."),
     ]),
    ("clper_lane_op_m", "CLPER Lane Operation (clper\\_lane\\_op\\_m)",
     "Computes target lane from `(lane_id, src1.b0)` (`none`, `xor`, `accumulate`, `shift`, `rotate`, v11+ `low`/`prefix`).",
     [
         ("Butterfly reduction (`.accumulate`)",
          fmt_asm_line("CLPER", ".subgroup16.accumulate.zero", "r0", ["r1", "r2.b0"]),
          "Reads partner lane in a binary-tree butterfly reduction for `subgroupAdd` / `subgroupMax`."),
         ("Cyclic subgroup rotate (`.rotate`)",
          fmt_asm_line("CLPER", ".subgroup16.rotate.zero", "r0", ["r1", "r2.b0"]),
          "With `r2.b0 = 1`: lane `L` reads `r1` from lane `(L + 1) & 15`, wrapping lane 15 back to lane 0."),
     ]),
    ("clper_inactive_result_m", "CLPER Inactive Lane Result (clper\\_inactive\\_result\\_m)",
     "16 hardware constants returned when a `CLPER` target lane is inactive or out of bounds. Default: `zero`.",
     [
         ("Additive identity (`.zero`) under divergence",
          fmt_asm_line("CLPER", ".subgroup16.xor.zero", "r0", ["r1", "r2.b0"]),
          "If target lane `L ^ r2.b0` is masked off by divergence, returns `0` so `IADD`/`FADD` reductions stay exact."),
         ("Unsigned min identity (`.umax`) & FP max identity (`.ninf`)",
          fmt_asm_line("CLPER", ".subgroup16.xor.umax", "r0", ["r1", "r2.b0"]),
          "Substitutes `0xFFFFFFFF` for inactive lanes in `subgroupMin(uint)` (or `-Inf` with `.ninf` for `subgroupMax(float)`)."),
     ]),
    ("sign_extend_or_none_m", "Load Extension Mode (sign\\_extend\\_or\\_none\\_m)",
     "Sign- or zero-extends narrow `LOAD.i8`/`i16` into 32-bit staging register. Default: `none`/`zext_or_none`.",
     [
         ("Sign-extended 8-bit load (`.sext`)",
          fmt_asm_line("LOAD.i8", ".sext.slot0", "@r1", ["[r2:r3]"], ["#0"]),
          "Loading byte `0x80` (`-128`): `.sext` writes `0xFFFFFF80` into `r1`."),
         ("Zero-extended 16-bit load (default)",
          fmt_asm_line("LOAD.i16", ".slot0", "@r1", ["[r2:r3]"], ["#0x2"]),
          "Loading halfword `0x8001`: default `zext_or_none` writes `0x00008001` (unsigned `32769`)."),
     ]),
    ("access_load_m", "Memory Cache Access Hint (access\\_load\\_m)",
     "Cache allocation/streaming policy (`none`, `istream`, `estream`, `force`). Default: `none`.",
     [
         ("Outer-streaming sequential read (`.estream`)",
          fmt_asm_line("LOAD.i128", ".estream.slot0", "@r0:r3", ["[r4:r5]"], ["#0"]),
          "Hints evict-first streaming cache behavior for large sequential buffer scans to avoid thrashing L2."),
         ("Coherent volatile read (`.force`)",
          fmt_asm_line("LOAD.i32", ".force.slot0", "@r0", ["[r4:r5]"], ["#0"]),
          "Bypasses stale L1 lines to observe writes from other workgroups (`coherent` SSBO)."),
     ]),
    ("oob_mode_m", "Checked Load OOB Mode (oob\\_mode\\_m, v14)",
     "Controls out-of-bounds behavior on v14 `LD_CHECKED*` (`none`: return 0; `oob_ignore`: disable OOB zeroing).",
     [
         ("Standard robust OOB zeroing (`none`)",
          fmt_asm_line("LD_CHECKED.i32", ".slot0", "@r0", ["u3", "r1", "r2"]),
          "With `r1 >= r2`: suppresses the memory read and writes `r0 = 0` per Vulkan `robustBufferAccess`."),
         ("Suppress OOB zeroing (`.oob_ignore`, v14)",
          fmt_asm_line("LD_CHECKED.i32", ".oob_ignore.slot0", "@r0", ["u3", "r1", "r2"]),
          "Leaves destination unmodified on OOB when the shader tracks bounds separately."),
     ]),
    ("atom_operation_m", "Binary Atomic Operation (atom\\_operation\\_m)",
     "Selects RMW operation for `ATOM` / `ATOM_RETURN` (`aadd`, `asmin`, `asmax`, `aumin`, `aumax`, `aand`, `aor`, `axor`, `axchg`, `aaddu`, `aadds`).",
     [
         ("Signed max returning previous value (`.asmax`)",
          fmt_asm_line("ATOM_RETURN.i32", ".asmax.slot0", "@r0", ["[r2:r3]"], ["#0"]),
          "With `*addr = -5, r0 = 3`: updates `*addr = 3` and returns previous value `@r0 = -5`."),
         ("Bitwise XOR without return (`ATOM` + `.axor`)",
          fmt_asm_line("ATOM.i32", ".axor.slot7", "@r0", ["[r2:r3]"], ["#0x4"]),
          "Atomically toggles bits `*([r2:r3] + 4) ^= r0` on non-blocking `slot7`."),
     ]),
    ("atom1_operation_m", "Unary Atomic Operation (atom1\\_operation\\_m)",
     "Implicit-operand RMW opcode for `ATOM1` / `ATOM1_RETURN` (`ainc`, `adec`, `asmax1`, `aumax1`, `aor1`).",
     [
         ("Work-queue ticket increment (`.ainc`)",
          fmt_asm_line("ATOM1_RETURN.i32", ".ainc.slot0", "@r0", ["[r2:r3]"], ["#0"]),
          "Atomically increments `*addr` by `1` and returns the old ticket in `@r0` without needing an input value register."),
         ("Set dirty flag bit 0 (`.aor1`)",
          fmt_asm_line("ATOM1.i32", ".aor1.slot7", "", ["[r2:r3]"], ["#0"]),
          "Atomically sets bit 0 (`*addr |= 1`) with zero value-register traffic."),
     ]),
    ("register_file_format_general_m", "Register Conversion Format (register\\_file\\_format\\_general\\_m)",
     "Staging register data format for `LD_CVT`, `ST_CVT`, `LD_ATTR*`, `LD_VAR*`, `LD_TILE`, `BLEND`.",
     [
         ("FP32 vertex attribute unpack (`.f32`)",
          fmt_asm_line("LD_ATTR_IMM", ".f32.v4.slot0", "@r0:r3", ["r60", "r61"], ["#0", "#0"]),
          "Converts e.g. `R8G8B8A8_UNORM` vertex buffer data into 4x IEEE FP32 in `@r0:r3`."),
         ("Packed FP16 tilebuffer fetch (`.f16`)",
          fmt_asm_line("LD_TILE", ".f16.v4.slot0", "@r0:r1", ["r59", "r60", "u0"]),
          "Packs 4x converted FP16 color channels into just 2 GPRs `@r0:r1`."),
     ]),
    ("vecsize_varying_m", "Vector Size (vecsize\\_varying\\_m)",
     "Number of channels (`v1`/`none`, `v2`, `v3`, `v4`) transferred by varying/attribute/tile/blend messages.",
     [
         ("2-channel UV varying load (`.v2`)",
          fmt_asm_line("LD_VAR_IMM", ".f32.v2.center.slot0", "@r0:r1", ["r59"], ["#0", "#0"]),
          "Interpolates 2 FP32 channels (`vec2`) into staging pair `@r0:r1`."),
         ("3-channel normal attribute load (`.v3`)",
          fmt_asm_line("LD_ATTR_IMM", ".f32.v3.slot0", "@r0:r2", ["r60", "r61"], ["#0", "#1"]),
          "Loads 3 FP32 channels (`vec3`) into `@r0:r2`, leaving `r3` untouched."),
     ]),
    ("sample_position_m", "Varying Sample Position (sample\\_position\\_m)",
     "Selects pixel center, centroid, per-sample, or explicit offset interpolation.",
     [
         ("MSAA centroid interpolation (`.centroid`)",
          fmt_asm_line("LD_VAR_IMM", ".f32.v4.centroid.store.slot0", "@r0:r3", ["r59"], ["#0", "#0"]),
          "Evaluates `centroid in vec4` inside the covered portion of the primitive to avoid extrapolation artifacts."),
         ("Explicit sub-pixel offset (`.explicit`)",
          fmt_asm_line("LD_VAR_IMM", ".f32.v2.explicit.clobber.slot0", "@r0:r1", ["r4"], ["#0", "#0"]),
          "Implements GLSL `interpolateAtOffset(v, ofs)` using caller-supplied offset `r4`."),
     ]),
    ("update_mode_m", "Varying Cache Update Mode (update\\_mode\\_m)",
     "Controls hidden barycentric interpolation cache (`store`, `retrieve`, `clobber`).",
     [
         ("First varying saves barycentrics (`.store`), second reuses (`.retrieve`)",
          fmt_asm_line("LD_VAR_IMM", ".f32.v4.center.retrieve.slot0", "@r4:r7", ["r59"], ["#0", "#1"]),
          "Reuses barycentric weights saved by a prior `.store` varying load at the same sample position, saving interpolation cycles."),
         ("One-off special varying without polluting cache (`.clobber`)",
          fmt_asm_line("LD_VAR_SPECIAL", ".frag_w.center.clobber.slot0", "@r0", ["r59"]),
          "Computes `frag_w` directly when no subsequent `.retrieve` depends on the cache."),
     ]),
    ("tex_dimensionality_m", "Texture Dimensionality (tex\\_dimensionality\\_m)",
     "Selects 1D, 2D, 3D, or Cube texture addressing.",
     [
         ("Hardware-projected cubemap (v11+ `.cube.project`)",
          fmt_asm_line("TEX_SINGLE", ".cube.project.dst32.rgba.slot0", "@r0:r3", ["@r4:r6", "u0"]),
          "Samples a cubemap from 3D direction `@r4:r6` directly in hardware (eliminates v9--v10 `CUBEFACE1/2` + `CUBE_SSEL/TSEL`)."),
         ("3D volumetric texture (`.tex3d`)",
          fmt_asm_line("TEX_SINGLE", ".tex3d.computed.dst32.rgba.slot0", "@r0:r3", ["@r4:r6", "u0"]),
          "Samples a 3D LUT using trilinear `(u, v, w)` coordinates in `@r4:r6`."),
     ]),
    ("tex_lod_mode_m", "Texture LOD Mode (tex\\_lod\\_mode\\_m)",
     "Controls mipmap level-of-detail selection (`computed`, `explicit`, `zero`, `computed_bias`, etc.).",
     [
         ("Explicit mipmap LOD (`.explicit`)",
          fmt_asm_line("TEX_SINGLE", ".tex2d.explicit.dst32.rgba.slot0", "@r0:r3", ["@r4:r6", "u0"]),
          "Reads explicit 8.8 fixed-point LOD from `@r6` (implements GLSL `textureLod(samp, uv, lod)`)."),
         ("Computed quad LOD + shader bias (`.computed_bias`)",
          fmt_asm_line("TEX_SINGLE", ".tex2d.computed_bias.dst32.rgba.slot0", "@r0:r3", ["@r4:r6", "u0"]),
          "Computes 2x2 finite-difference LOD and adds bias `@r6` (implements `texture(samp, uv, bias)`)."),
     ]),
    ("tex_gather_component_m", "Texture Gather Component (tex\\_gather\\_component\\_m)",
     "Selects channel (`gather4_r`, `gather4_g`, `gather4_b`, `gather4_a`) for `TEX_GATHER`.",
     [
         ("Green-channel 2x2 gather (`.gather4_g`)",
          fmt_asm_line("TEX_GATHER", ".tex2d.gather4_g.dst32.rgba.slot0", "@r0:r3", ["@r4:r5", "u0"]),
          "Implements GLSL `textureGather(samp, uv, 1)`, returning the 4 green texels in counter-clockwise order."),
         ("Alpha-channel 2x2 gather (`.gather4_a`)",
          fmt_asm_line("TEX_GATHER", ".tex2d.gather4_a.dst32.rgba.slot0", "@r0:r3", ["@r4:r5", "u0"]),
          "Implements GLSL `textureGather(samp, uv, 3)` for custom alpha-cutout filtering."),
     ]),
]


# ============================================================================
# Chapter 2.51.X: Worked Examples for ALL 78 Secondary Modifier Enumerations
# ============================================================================
SECONDARY_ENUM_WALKTHROUGHS = {
    "access_store_m": (
        "Coherent SSBO store vs streaming store",
        fmt_asm_line("STORE.i32", ".force.slot7", "@r0", ["[r2:r3]"], ["#0"]),
        "With `.force`, bypasses write-back caching so other workgroups immediately observe the store; `.estream` instead streams bulk writes past L2.",
    ),
    "branch_conservative_m": (
        "Conservative loop back-edge branch",
        fmt_asm_line("BRANCH", ".conservative", "", ["r0"], ["#-0x18"]),
        "Hints to the warp scheduler that divergence across lanes is non-trivial so reconvergence tracking stays conservative.",
    ),
    "clper_or_lane_select_m": (
        "Fused cross-lane shuffle + OR (v14)",
        fmt_asm_line("CLPER_OR", ".subgroup16.lanes04", "r0", ["r1", "r2.b0", "r3"]),
        "Selects the `lanes04` pair routing in v14 `CLPER_OR`, permutes `r1`, and bitwise-ORs with `r3` in a single cycle.",
    ),
    "clz_mask_m": (
        "GLSL `findMSB(0) == -1` via `.mask`",
        fmt_asm_line("CLZ.u32", ".mask", "r0", ["r1"]),
        "When `r1 = 0`: default `none` returns `32`, whereas `.mask` returns `0xFFFFFFFF` (`-1`). For `r1 = 0x00010000`, both return `15`.",
    ),
    "dest_width_single_lane_16_bit_m": (
        "Single-lane 16-bit writeback preserving peer lane",
        fmt_asm_line("F32_TO_F16", "", "r0.h1", ["r1"]),
        "With `r1 = 2.0f`, `r0_old = 0x00003C00`: `.h1` writes `0x4000` to `r0[31:16]` while preserving `r0[15:0]` -> `r0 = 0x40003C00`.",
    ),
    "encoded_sample_and_update_m": (
        "Joint sample-position + cache-store encoding",
        fmt_asm_line("VAR_TEX_BUF_SINGLE", ".center_store.tex2d.slot0", "@r0:r3", ["u0", "r59", "r4"]),
        "Interpolates `vUV` at pixel center and stores the barycentric weights in hardware cache for subsequent `.retrieve` varying loads.",
    ),
    "encoded_sample_position_none_m": (
        "Per-sample MSAA varying load",
        fmt_asm_line("LD_VAR_IMM", ".f32.v4.sample.slot0", "@r0:r3", ["r59"], ["#0", "#0"]),
        "Encodes `sample` (`2`) instead of default `center_or_none` (`0`) to interpolate at each MSAA sample location.",
    ),
    "encoded_update_mode_special_m": (
        "Caching barycentrics from `LD_VAR_SPECIAL`",
        fmt_asm_line("LD_VAR_SPECIAL", ".bary.store.slot0", "@r0:r1", ["r59"]),
        "Uses `.store` (`1`) instead of default `clobber_or_none` (`0`) so later varying loads can `.retrieve` the same weights.",
    ),
    "fadd_sticky_m": (
        "Round-to-odd sticky bit for emulated FP64",
        fmt_asm_line("FADD.f32", ".sticky", "r0", ["r1", "r2"]),
        "OR-reduces shifted-out bits into bit 0 of the mantissa so a subsequent rounding step avoids double-rounding error.",
    ),
    "fexp_adj_m": (
        "Low-word correction in legacy `FEXP_TABLE`",
        fmt_asm_line("FEXP_TABLE.u4", ".low", "r0", ["r1"]),
        "Selects the `.low` table bank to fetch the fine-precision correction term paired with the main `2^(idx/16)` entry.",
    ),
    "fma_rscale_round_m": (
        "Truncated scaled FMA (`.round_zero`)",
        fmt_asm_line("FMA_RSCALE.f32", ".round_zero", "r0", ["r1", "r2", "r3", "r4"]),
        "Evaluates `fma(r1, r2, r3) * 2^r4` in FP64 and rounds toward zero instead of default RNE.",
    ),
    "fpow_sc_det_function_m": (
        "`pow(0,0) = 1` vs `powr(0,0) = NaN`",
        fmt_asm_line("FPOW_SC_DET.f32", ".powr", "r0", ["r1", "r2"]),
        "With `r1 = 0.0f, r2 = 0.0f`: `.pow` signals `1.0f` override, whereas `.powr` signals `NaN` per OpenCL `powr` semantics.",
    ),
    "iaddc_return_carry_m": (
        "Extracting 1-bit carry-out for 64-bit add",
        fmt_asm_line("IADDC.u32", ".c", "r0", ["r1", "r2", "k0"]),
        "With `r1 = 0xFFFFFFFF, r2 = 2`: `none` returns sum `1`, whereas `.c` returns carry-out `1` to feed the high-word `IADDC`.",
    ),
    "jump_divergence_mode_m": (
        "Divergent indirect function call",
        fmt_asm_line("JUMP_EX", ".call.diverge", "", ["r0", "r1", "r2"]),
        "Marks `.diverge` when lanes in the warp may jump to different target addresses via a function pointer.",
    ),
    "jump_stack_mode_m": (
        "Hardware call-stack push and tail-call replace",
        fmt_asm_line("JUMP_EX", ".replace", "", ["k1n", "r1", "r2"]),
        "`.call` pushes `PC+8`, `.return` pops it, and `.replace` performs a tail call without growing the hardware stack.",
    ),
    "lane_all_m": (
        "Selecting sub-word lane in checked/narrow load",
        fmt_asm_line("LOAD.i8", ".b2.slot0", "@r0", ["[r2:r3]"], ["#0"]),
        "Loads 1 byte from memory and inserts it into byte lane 2 (`r0[23:16]`), leaving other bytes of `r0` intact.",
    ),
    "ld_gclk_source_m": (
        "Reading GPU cycle counter vs system clock",
        fmt_asm_line("LD_GCLK", ".system_timestamp.slot0", "@r0:r1", []),
        "`.cycle_counter` reads per-core shader cycles; `.system_timestamp` reads the cross-core monotonic wall clock.",
    ),
    "ld_var_buf_encoded_update_mode_none_m": (
        "Default `.retrieve_or_none` in varying buffer",
        fmt_asm_line("LD_VAR_BUF_IMM.f32", ".v4.center.store.slot0", "@r0:r3", ["r59"], ["#0"]),
        "First varying uses `.store` (`0`); subsequent varyings default to `retrieve_or_none` (`1`) to reuse cached barycentrics.",
    ),
    "ld_var_buf_flat_format_m": (
        "Flat 16-bit vs 32-bit varying buffer read (v14)",
        fmt_asm_line("LD_VAR_BUF_FLAT_IMM", ".flat16.slot0", "@r0", [], ["#0x04"]),
        "Reads a 16-bit flat varying (`flat16`) instead of a 32-bit word (`flat32`) from varying buffer offset `0x04`.",
    ),
    "ld_var_buf_source_format_m": (
        "FP16 varying buffer widened to FP32 registers",
        fmt_asm_line("LD_VAR_BUF.f32", ".src_f16.v2.center.slot0", "@r0:r1", ["r59", "r4"]),
        "Declares that the varying buffer stores `f16` (`src_f16`) while delivering interpolated `f32` results in `@r0:r1`.",
    ),
    "ldexp_round_m": (
        "Rounding under subnormal underflow in `LDEXP`",
        fmt_asm_line("LDEXP.f32", ".round_down", "r0", ["r1", "r2"]),
        "With `r1 = -1.5f, r2 = -150`: rounds underflowing negative subnormal toward `-Inf` (`0x80000001`) instead of flushing to `-0.0`.",
    ),
    "load_lane_16_m": (
        "Inserting a 16-bit load into `r0.h1`",
        fmt_asm_line("LOAD.i16", ".h1.slot0", "@r0", ["[r2:r3]"], ["#0x2"]),
        "Loads 16 bits from `[r2:r3] + 2` directly into `r0[31:16]` without overwriting `r0[15:0]`.",
    ),
    "load_lane_32_m": (
        "32-bit vs 64-bit (`d0`) load lane mode",
        fmt_asm_line("LOAD.i32", ".slot0", "@r0", ["[r2:r3]"], ["#0"]),
        "Default `none` writes a 32-bit word into `@r0`; `d0` selects full 64-bit doubleword lane addressing.",
    ),
    "ls_multi_sr_count_m": (
        "Bulk 8-register (256-bit) transfer on v14",
        fmt_asm_line("LD_MULTI", ".i256.slot0", "@r0:r7", ["[r8:r9]"], ["#0"]),
        "Encoding `7` (`.i256`) transfers `(7 + 1) * 32 = 256` bits (8 consecutive staging registers `@r0:r7`) in one message.",
    ),
    "ls_z_last_read_m": (
        "Reclaiming tile storage on final `LD_TILE`",
        fmt_asm_line("LD_TILE", ".f32.v4.last_read.slot0", "@r0:r3", ["r59", "r60", "u0"]),
        "`.last_read` signals that no further tilebuffer reads will occur for this pixel so hardware can release tile resources early.",
    ),
    "ls_z_stencil_m": (
        "Reading depth/stencil tilebuffer instead of color",
        fmt_asm_line("LD_TILE", ".f32.z_stencil.slot0", "@r0", ["r59", "r60", "u1"]),
        "With `.z_stencil`, targets the on-chip depth/stencil attachment rather than a color render target.",
    ),
    "message_slot_index_m": (
        "Overlapping async load (`slot0`) and texture (`slot1`)",
        fmt_asm_line("LOAD.i32", ".slot0", "@r0", ["[r2:r3]"], ["#0"]),
        "Assigning independent messages to `slot0` and `slot1` lets the shader wait on `.wait0` while `slot1` remains in flight (`slot7` is non-blocking).",
    ),
    "round_extended_m": (
        "Ties-away-from-zero int-to-float conversion",
        fmt_asm_line("S32_TO_F32", ".round_na", "r0", ["r1"]),
        "For `r1 = 0x01000001` (halfway between two FP32 values): `.round_na` rounds away from zero while default RNE rounds to even.",
    ),
    "sample_position_none_m": (
        "Centroid vs uninterpolated `none` sample position",
        fmt_asm_line("LD_VAR_BUF.f32", ".v4.centroid.slot0", "@r0:r3", ["r59", "r4"]),
        "Selects `centroid` for MSAA-safe interpolation, or `none` when interpolation position is unused.",
    ),
    "segadd_preserve_null_m": (
        "Keeping NULL pointer zero on offset add",
        fmt_asm_line("SEG_ADD.u64", ".preserve_null", "[r0:r1]", ["[r2:r3]", "r4.w0"]),
        "With `[r2:r3] = 0, r4 = 16`: `.preserve_null` keeps `[r0:r1] = 0`; with `none`, computes `[r0:r1] = 16`.",
    ),
    "sign_extend_or_none_compressed_encoding_m": (
        "Compressed 1-bit sign-extend flag on narrow `LOAD`",
        fmt_asm_line("LOAD.i8", ".sext.slot0", "@r0", ["[r2:r3]"], ["#0"]),
        "Encoding `0` (`.sext`) sign-extends `0xFE -> 0xFFFFFFFE`; encoding `1` (`zext_or_none`) zero-extends `0xFE -> 0x000000FE`.",
    ),
    "stencil_m": (
        "Exporting programmatic fragment stencil reference",
        fmt_asm_line("ZS_EMIT", ".stencil.slot0", "@r60", ["k0", "r1", "r60"]),
        "With `.stencil`, writes `r1[7:0]` as `gl_FragStencilRefARB` and returns updated coverage in `@r60`.",
    ),
    "submatrix_sel_m": (
        "Selecting 4x4 submatrix halves in `MMUL.v2f16`",
        fmt_asm_line("MMUL.v2f16", "", "r0", ["r1.f0", "r2.f1", "r3"]),
        "`.f0` selects the first 4x4 FP16 half of matrix A and `.f1` selects the second 4x4 half of matrix B.",
    ),
    "table_flog_mode_m": (
        "Selecting natural vs base-2 log polynomial",
        fmt_asm_line("FLOG_TABLE.f32", ".natural", "r0", ["r1"]),
        "`.base2` fetches `log2` LUT terms, `.natural` fetches `ln` terms, and `.red` fetches reduced-range coefficients.",
    ),
    "table_flog_precision_m": (
        "High vs low precision `FLOG_TABLE` lookup",
        fmt_asm_line("FLOG_TABLE.f32", ".base2.high", "r0", ["r1"]),
        "`.high` returns the upper-order polynomial coefficient; `.low` returns the fine refinement coefficient.",
    ),
    "table_sincos_offset_m": (
        "Applying quadrant angle offset in `FSINCOS`",
        fmt_asm_line("FSINCOS.f32", ".sin.offset", "r0", ["r1", "r2"]),
        "Enables the quadrant phase offset adjustment during v14 hardware `FSINCOS` evaluation.",
    ),
    "table_sincos_operation_m": (
        "Selecting `sin` vs `cos` output on `FSINCOS`",
        fmt_asm_line("FSINCOS.f32", ".cos", "r0", ["r1", "r2"]),
        "Encoding `0` computes `cos(x)`; encoding `1` computes `sin(x)` from the same range-reduced inputs.",
    ),
    "table_sincos_scale_m": (
        "Adding `+24` exponent scale in `FSINCOS_OFFSET`",
        fmt_asm_line("FSINCOS_OFFSET.u6", ".scale", "r0", ["r1"]),
        "`.scale` boosts the returned FP32 offset exponent by `+24` to align with fixed-point range reduction.",
    ),
    "tex_array_enable_m": (
        "Sampling a `sampler2DArray` layer",
        fmt_asm_line("TEX_SINGLE", ".tex2d.array.computed.slot0", "@r0:r3", ["@r4:r6", "u0"]),
        "`.array` tells the texture unit to read the array layer index from staging register `@r6` after `(u, v)`.",
    ),
    "tex_compare_enable_m": (
        "Hardware shadow-map PCF comparison",
        fmt_asm_line("TEX_SINGLE", ".tex2d.compare.computed.slot0", "@r0", ["@r4:r6", "u0"]),
        "`.compare` reads depth reference `Dref` from `@r6` and returns filtered shadow factor in `[0.0, 1.0]`.",
    ),
    "tex_coordinate_mode_m": (
        "Unnormalized integer coordinates for `texelFetch`",
        fmt_asm_line("TEX_FETCH", ".tex2d.integer_coordinates.slot0", "@r0:r3", ["@r4:r6", "u0"]),
        "`.integer_coordinates` interprets `@r4:r5` as raw integer texel indices `(x, y)` rather than normalized `[0, 1]` floats.",
    ),
    "tex_coordinate_or_derivative_m": (
        "Requesting raw screen-space derivatives in `TEX_GRADIENT`",
        fmt_asm_line("TEX_GRADIENT", ".tex2d.derivative.slot0", "@r0:r3", ["@r4:r5", "u0"]),
        "`.derivative` outputs `(dU/dx, dV/dx, dU/dy, dV/dy)` instead of scalar LOD; `.force_delta` forces quad deltas across helper lanes.",
    ),
    "tex_derivative_enable_m": (
        "Returning derivatives from `VAR_TEX_GRADIENT`",
        fmt_asm_line("VAR_TEX_GRADIENT", ".tex2d.derivative.center.slot0", "@r0:r3", ["u0", "r59", "u1"]),
        "Enables raw derivative output directly from interpolated varying coordinates.",
    ),
    "tex_dual_array_enable_m": (
        "Arrayed dual texture sample",
        fmt_asm_line("TEX_DUAL", ".tex2d.array.computed.slot0", "@r0:r7", ["@r8:r10", "u0"]),
        "Passes shared `(u, v, layer)` in `@r8:r10` to both paired texture lookups simultaneously.",
    ),
    "tex_dual_lod_mode_m": (
        "Explicit LOD on dual texture sample",
        fmt_asm_line("TEX_DUAL", ".tex2d.explicit.slot0", "@r0:r7", ["@r8:r10", "u0"]),
        "Applies explicit LOD `@r10` to both paired texture fetches in `TEX_DUAL`.",
    ),
    "tex_dual_primary_width_m": (
        "FP16 primary output width in `TEX_DUAL`",
        fmt_asm_line("TEX_DUAL", ".tex2d.primary_f16.secondary_f32.slot0", "@r0:r5", ["@r6:r7", "u0"]),
        "Packs the primary texture's RGBA into 2 FP16 registers (`@r0:r1`) while keeping the secondary texture in 4 FP32 registers (`@r2:r5`).",
    ),
    "tex_dual_primary_write_mask_m": (
        "RGB-only primary write mask in `TEX_DUAL`",
        fmt_asm_line("TEX_DUAL", ".tex2d.primary_rgb.secondary_r.slot0", "@r0:r3", ["@r4:r5", "u0"]),
        "Suppresses unused primary alpha writeback, writing only `.rgb` to `@r0:r2`.",
    ),
    "tex_dual_secondary_width_m": (
        "FP16 secondary output width in `TEX_DUAL`",
        fmt_asm_line("TEX_DUAL", ".tex2d.primary_f32.secondary_f16.slot0", "@r0:r5", ["@r6:r7", "u0"]),
        "Returns secondary texture channels as packed FP16 pairs (`@r4:r5`) to cut register pressure.",
    ),
    "tex_dual_secondary_write_mask_m": (
        "Single-channel secondary mask in `TEX_DUAL`",
        fmt_asm_line("TEX_DUAL", ".tex2d.primary_rgba.secondary_r.slot0", "@r0:r4", ["@r5:r6", "u0"]),
        "Writes only the red channel of the second texture into `@r4` (e.g., ambient occlusion map).",
    ),
    "tex_footprint_m": (
        "2x2 bilinear footprint weights query (v9)",
        fmt_asm_line("TEX_WEIGHTS", ".tex2d.footprint_2x2.slot0", "@r0:r3", ["@r4:r5", "u0"]),
        "Returns the 4 normalized bilinear weights for the 2x2 footprint surrounding `@r4:r5`.",
    ),
    "tex_force_delta_enable_m": (
        "Forcing quad delta evaluation on helper lanes",
        fmt_asm_line("VAR_TEX_BUF_GRADIENT", ".tex2d.force_delta.slot0", "@r0:r1", ["u0", "r59", "r4"]),
        "Ensures valid 2x2 finite differences even when neighboring lanes in the quad are helper/discarded pixels.",
    ),
    "tex_gradient_skip_m": (
        "Skipping helper lanes on `TEX_GRADIENT`",
        fmt_asm_line("TEX_GRADIENT", ".tex2d.skip.slot0", "@r0:r3", ["@r4:r9", "u0"]),
        "Skips texture memory fetches on discarded/helper lanes while preserving quad LOD math.",
    ),
    "tex_gradient_width_m": (
        "FP16 return format on `TEX_GRADIENT`",
        fmt_asm_line("TEX_GRADIENT", ".tex2d.f16.rgba.slot0", "@r0:r1", ["@r4:r9", "u0"]),
        "Returns the filtered RGBA result packed as 4x FP16 in `@r0:r1` instead of 4x FP32.",
    ),
    "tex_gradient_write_mask_m": (
        "Two-channel `.rg` mask on `TEX_GRADIENT`",
        fmt_asm_line("TEX_GRADIENT", ".tex2d.f32.rg.slot0", "@r0:r1", ["@r4:r9", "u0"]),
        "Writes only `.rg` components into `@r0:r1` (e.g., 2-channel normal map or `textureQueryLod`).",
    ),
    "tex_lod_bias_disable_m": (
        "Ignoring sampler LOD bias for raw `textureQueryLod`",
        fmt_asm_line("TEX_GRADIENT", ".tex2d.disable_lod_bias.slot0", "@r0:r1", ["@r4:r5", "u0"]),
        "Suppresses sampler descriptor LOD bias when querying unbiased mipmap level.",
    ),
    "tex_lod_clamp_disable_m": (
        "Ignoring sampler min/max LOD clamp",
        fmt_asm_line("TEX_GRADIENT", ".tex2d.disable_lod_clamp.slot0", "@r0:r1", ["@r4:r5", "u0"]),
        "Returns the unclamped mipmap level `lambda` required by the second component of GLSL `textureQueryLod`.",
    ),
    "tex_projection_enable_m": (
        "Homogeneous projective divide (`textureProj`)",
        fmt_asm_line("TEX_SINGLE", ".tex2d.project.computed.slot0", "@r0:r3", ["@r4:r6", "u0"]),
        "Divides `(u, v)` by homogeneous coordinate `q = @r6` inside the texture unit before filtering.",
    ),
    "tex_skip_m": (
        "Skipping helper-lane memory traffic (`.skip`)",
        fmt_asm_line("TEX_SINGLE", ".tex2d.computed.skip.slot0", "@r0:r3", ["@r4:r5", "u0"]),
        "Helper lanes participate in 2x2 quad LOD derivation, then `.skip` suppresses their actual cache fetch.",
    ),
    "tex_texel_offset_m": (
        "Signed 4-bit texel offset (`textureOffset`)",
        fmt_asm_line("TEX_SINGLE", ".tex2d.texel_offset.computed.slot0", "@r0:r3", ["@r4:r6", "u0"]),
        "Applies packed signed 4-bit `(dx, dy)` offset in `@r6` (range `[-8, +7]`) before sampling.",
    ),
    "tex_texel_offset_si_m": (
        "Immediate-style texel offset on `TEX_FETCH`/`TEX_GATHER`",
        fmt_asm_line("TEX_GATHER", ".tex2d.gather4_r.texel_offset.slot0", "@r0:r3", ["@r4:r6", "u0"]),
        "Offsets the 2x2 gather footprint by signed `(dx, dy)` for GLSL `textureGatherOffset`.",
    ),
    "tex_wide_indices_enable_m": (
        "32-bit bindless descriptor indices (`.wide`)",
        fmt_asm_line("TEX_SINGLE", ".tex2d.wide.computed.slot0", "@r0:r3", ["@r4:r5", "r6"]),
        "Allows full 32-bit texture/sampler table indices in `r6` for large Vulkan bindless descriptor arrays.",
    ),
    "tex_width_m": (
        "Signed integer (`s32`) vs FP16 (`f16`) texture return",
        fmt_asm_line("TEX_SINGLE", ".tex2d.f16.rgba.slot0", "@r0:r1", ["@r4:r5", "u0"]),
        "`.f16` returns 4x FP16 in `@r0:r1`; `.s32`/`.u32` returns integer texels for `isampler2D`/`usampler2D`.",
    ),
    "tex_write_mask_m": (
        "Single-channel `.r` mask saving staging GPRs",
        fmt_asm_line("TEX_SINGLE", ".tex2d.f32.r.slot0", "@r0", ["@r4:r5", "u0"]),
        "Writes only the red channel to `@r0` instead of allocating 4 staging registers `@r0:r3`.",
    ),
    "update_mode_none_m": (
        "Varying cache lifecycle (`store` -> `retrieve` -> `none`)",
        fmt_asm_line("LD_VAR_IMM", ".f32.v4.center.store.slot0", "@r0:r3", ["r59"], ["#0", "#0"]),
        "`.store` populates the barycentric cache; `.retrieve` reuses it; `.none` leaves cache state untouched.",
    ),
    "update_mode_special_m": (
        "Special varying cache update mode",
        fmt_asm_line("LD_VAR_SPECIAL", ".frag_z.center.clobber.slot0", "@r0", ["r59"]),
        "Uses `.clobber` when interpolating `gl_FragCoord.z` independently of user varyings.",
    ),
    "v4u8_m": (
        "Unsigned 4x8-bit lane interpretation on `IDPADD`/`MMUL`",
        fmt_asm_line("IDPADD.v4s8", ".v4u8.sat", "r0", ["r1", "r2", "r3"]),
        "Overrides a source operand from signed `s8` to unsigned `u8` for mixed-sign `OpSUDotKHR` dot products (v14).",
    ),
    "var_tex_array_enable_m": (
        "Fused arrayed varying-texture sample",
        fmt_asm_line("VAR_TEX_SINGLE", ".tex2d.array.computed.center.slot0", "@r0:r3", ["u0", "r59", "u1"]),
        "Interpolates `(u, v, layer)` directly from varying `u1` and samples a `sampler2DArray` in one instruction.",
    ),
    "var_tex_buf_source_format_m": (
        "FP16 UV varying buffer fused into texture sample",
        fmt_asm_line("VAR_TEX_BUF_SINGLE", ".tex2d.src_f16.computed.slot0", "@r0:r3", ["u0", "r59", "r4"]),
        "Reads `mediump` FP16 UVs from varying buffer offset `r4` and samples texture `u0` without intermediate GPRs.",
    ),
    "var_tex_compare_enable_m": (
        "Fused varying interpolation + shadow comparison",
        fmt_asm_line("VAR_TEX_BUF_SINGLE", ".tex2d.compare.computed.slot0", "@r0", ["u0", "r59", "r4"]),
        "Interpolates shadow-map `(u, v, refZ)` from varying buffer and performs hardware PCF depth comparison.",
    ),
    "var_tex_dimensionality_m": (
        "Fused 3D varying-texture lookup (`.tex3d`)",
        fmt_asm_line("VAR_TEX_SINGLE", ".tex3d.computed.center.slot0", "@r0:r3", ["u0", "r59", "u1"]),
        "Interpolates 3D `(u, v, w)` varying coordinates directly into a volumetric texture sample.",
    ),
    "var_tex_dual_array_enable_m": (
        "Fused arrayed dual texture sample",
        fmt_asm_line("VAR_TEX_BUF_DUAL", ".tex2d.array.computed.slot0", "@r0:r7", ["u0", "r59", "r4"]),
        "Interpolates `(u, v, layer)` once and feeds two paired `sampler2DArray` lookups.",
    ),
    "var_tex_dual_lod_mode_m": (
        "Computed quad LOD vs base-mip (`none`) on `VAR_TEX_DUAL`",
        fmt_asm_line("VAR_TEX_DUAL", ".tex2d.computed.center.slot0", "@r0:r7", ["u0", "r59", "u1"]),
        "`.computed` enables mipmapping via 2x2 quad derivatives; `none` forces mip level 0.",
    ),
    "var_tex_lod_mode_m": (
        "Computed quad LOD vs mip-0 on `VAR_TEX_SINGLE`",
        fmt_asm_line("VAR_TEX_BUF_SINGLE", ".tex2d.computed.slot0", "@r0:r3", ["u0", "r59", "r4"]),
        "Uses `.computed` for standard `texture(s, vUV)` or `none` when sampling a single-level texture.",
    ),
    "var_tex_projection_enable_m": (
        "Fused projective varying-texture sample",
        fmt_asm_line("VAR_TEX_SINGLE", ".tex2d.project.computed.center.slot0", "@r0:r3", ["u0", "r59", "u1"]),
        "Interpolates homogeneous `(u, v, q)` and divides by `q` before filtering (`textureProj(s, vUVQ)`).",
    ),
    "varying_name_m": (
        "Reading fixed-function `gl_PointCoord` vs `gl_FragCoord.w`",
        fmt_asm_line("LD_VAR_SPECIAL", ".point.center.clobber.slot0", "@r0:r1", ["r59"]),
        "`.point` reads point-sprite `(s, t)`, `.bary` reads barycentrics, `.frag_w` reads `1/w`, and `.frag_z` reads depth `z`.",
    ),
    "vertex_m": (
        "Selecting triangle vertex 1 for `pervertexEXT`",
        fmt_asm_line("LD_VAR_FLAT_IMM", ".f32.v4.vertex1.slot0", "@r0:r3", [], ["#0", "#0"]),
        "Reads vertex 1's uninterpolated attribute directly instead of the default provoking vertex (`none`/`vertex0`).",
    ),
    "vertex_type_m": (
        "Dynamic register-indexed triangle vertex (v14)",
        fmt_asm_line("LD_VAR_BUF_FLAT", ".f32.v4.explicit_vertex.slot0", "@r0:r3", ["r4"]),
        "With `.explicit_vertex`, reads the triangle vertex index (`0..2`) dynamically from a register instead of fixing the provoking vertex.",
    ),
    "z_m": (
        "Exporting programmatic fragment depth (`gl_FragDepth`)",
        fmt_asm_line("ZS_EMIT", ".z.slot0", "@r60", ["r0", "k0", "r60"]),
        "With `.z`, submits `r0` as the fragment's depth value to the late-Z test and returns updated coverage in `@r60`.",
    ),
}


def extras_of(mnem: str) -> tuple[str, str, list[tuple[str, str, str]]]:
    """Return (spirv_op, glsl_snippet, [(label, asm_tex, walkthrough_md), ...])."""
    return INST_EXTRAS[mnem]


def secondary_enum_walkthroughs(ename: str) -> list[tuple[str, str, str]]:
    """Return worked example walkthrough(s) for a Section 2.51.X secondary enum."""
    entry = SECONDARY_ENUM_WALKTHROUGHS.get(ename)
    if entry is None:
        return []
    if isinstance(entry, list):
        return entry
    return [entry]

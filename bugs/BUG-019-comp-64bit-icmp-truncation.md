# BUG-019 (`COMP-1`): 64-Bit Integer Comparison Truncated to 32 Bits in `spirv2ssa`

- **Status:** FIXED in Checkpoint v48.3
- **Layer:** `spirv2ssa` (`src/spirv2ssa/spirv2ssa_ops.cpp:626`)
- **Discovery Shader:** `comp_05_int64_arith_cmp_pack` & `comp_09_int64_signed_cmp_shift_sub` (`tools/corpus_synth/longitudinal_fuzz.py`)
- **Root Cause:** `ValhallSSA` `ICmp*` ops are 32-bit; `spirv2ssa` emitted 32-bit `ICmpUGt`/`ICmpSLt` directly on `U64`/`S64` operands, causing `ConcreteBitvecDomain.icmp()` to truncate the upper 32 bits.
- **Fix:** Lowered all 10 64-bit integer comparisons (`Eq`, `Ne`, `UGt`, `UGe`, `ULt`, `ULe`, `SGt`, `SGe`, `SLt`, `SLe`) via `Split64` inside `emit_cmp` (`hi_strict || (hi_eq && lo_unsigned_cmp)`).
- **Regression Test:** `TestCorpusSynth.test_v48_3_longitudinal_12_and_transcendental_ulp` (`comp_05`, `comp_09`).

# BUG-001 through BUG-013: Fixed Bugs (Checkpoints v48 – v48.2)

*(Individual files for BUG-001..BUG-013 are linked from [`bugs/README.md`](README.md); this file documents BUG-001.)*

## BUG-001 (`RC-1`): `spirv2ssa` `SpvOpBitcast` Missing `Float16` `UnpackHalf2x16` / `PackHalf2x16`
- **Status:** FIXED in Checkpoint v48 (`53b2c3f`)
- **Layer:** `spirv2ssa` (`src/spirv2ssa/spirv2ssa_ops.cpp:839–877`)
- **Affected Targets (13):** `FADD.v2f16`, `FMUL.v2f16`, `FMA.v2f16`, `FMIN.v2f16`, `FMAX.v2f16`, `FROUND.v2f16`, `F16_TO_S32`, `V2F16_TO_V2S16`, `V2F16_TO_V2U16`, `V2S16_TO_V2F16`, `V2U16_TO_V2F16`, `V2S8_TO_V2F16`, `V2U8_TO_V2F16`.
- **Witness:** `FADD.v2f16` `D=0x80800000`, `V=0x44800000` at `0x20: FADD.v2f16 r2, r2^.h01, r3^.h01`.
- **Root Cause:** `sw==32 && dw==16` emitted raw `BfeU` (`ValType::U32`) instead of `UnpackHalf2x16` (`ValType::V2F32`) when destination scalar element type was `SpirvTypeKind::Float`; symmetrically, `sw==16 && dw==32` emitted `BitOr` instead of `PackHalf2x16`.
- **Fix:** Check `SpirvTypeKind::Float` in both `32->16` and `16->32` branches of `SpvOpBitcast`.
- **Regression Test:** `TestCorpusSynth.test_v48_lowering_and_opt_fixes` (`FADD.v2f16`).

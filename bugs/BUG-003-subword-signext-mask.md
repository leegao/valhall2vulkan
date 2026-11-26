# BUG-003 (`RC-3`): `spirv2ssa` Subword Arithmetic Missing Sign-Extension & Width-Masking
- **Status:** FIXED in Checkpoint v48 (`53b2c3f`)
- **Layer:** `spirv2ssa` (`src/spirv2ssa/spirv2ssa_ops.cpp:133, 735, 883`)
- **Targets (3):** `S16_TO_F32`, `IABS.v2s16`, `IADD.v4u8`.
- **Witness:** `S16_TO_F32`: `D=0x47586a00` (`+55402.0`), `V=0xc61e5800` (`-10134.0`); `IADD.v4u8`: `D=0x0193105b`, `V=0x00920f5b` (inter-byte carry bleed).
- **Root Cause & Fix:** Added `BfeS` sign-extension on subword signed inputs (`sw < 32`) in `SpvOpConvertSToF` and `kGlslSAbs`, and `BfeS`/`BfeU` width masking on subword results in `SpvOpIAdd` and `kGlslSAbs`.

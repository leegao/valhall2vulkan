# BUG-002 (`RC-2`): `spirv2ssa` `SpvOpBitcast` Missing `32 <-> 8` Branch
- **Status:** FIXED in Checkpoint v48 (`53b2c3f`)
- **Layer:** `spirv2ssa` (`src/spirv2ssa/spirv2ssa_ops.cpp:839–877`)
- **Targets (6):** `S8_TO_F32`, `U8_TO_F32`, `V2S8_TO_V2S16`, `V2U8_TO_V2U16`, `MKVEC`, `IABS` (`v4s8`).
- **Witness:** `S8_TO_F32`: `D=0x4e7c0000` (float of entire 32-bit word), `V=0x00000000` (byte 0).
- **Root Cause & Fix:** `sw==32 && dw==8` and `sw==8 && dw==32` fell through to `else` (verbatim 32-bit copy). Added explicit 4-byte `BfeU` unpack and masked shift-or pack branches.

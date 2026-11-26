# BUG-020 (`COMP-2`): `LOAD.i16` / `LOAD.i8` Subword Lane Clobber (`.h0`/`.h1`/`.b0..3`) + `Float16` Intermediate Quantization

- **Status:** FIXED in Checkpoint v48.3
- **Layer:** Valhall Pass 1 (`src/valhall_pass1_handlers.cpp:440`, `abi::sr_set`) + `spirv2ssa` (`src/spirv2ssa/spirv2ssa_ops.cpp:587`)
- **Discovery Shader:** `comp_01_v2f16_fma_round_iabs_s16` (`tools/corpus_synth/longitudinal_fuzz.py`)
- **Witness (`vec=1`):** `Direct = 0x44800000` (`1024.0f`), `Valhall = 0x00000000` (`0.0f`).
- **Root Cause:** `libMali` packed two 16-bit loads into the same register `r63` (`0x20: LOAD.i16.slot0.h1 @r63` followed by `0x40: LOAD.i16.slot0.wait0.h0 @r63`), preserving `r63.h1` while writing `r63.h0`. Pass 1's `abi::sr_set()` blindly overwrote the full 32-bit `r63`, clobbering `r63.h1` with `0`. Additionally, `spirv2ssa` did not quantize `Float16` `GLSL.std.450` ext-inst outputs back to IEEE-754 `binary16`.
- **Fix:**
  1. In `abi::sr_set()`, when `ins.mnemonic == Mnemonic::LOAD` with `extend == "none"` and `lane` in `{h0, h1, b0..b3}`, merge into `prev = current_reg(blk, dst_reg)` via `SsaOp::Bfi`.
  2. In `lower_glsl450_ext_inst()`, quantize `Float16` result lanes via `PackHalf2x16` + `UnpackHalf2x16`.
- **Regression Test:** `TestCorpusSynth.test_v48_3_longitudinal_12_and_transcendental_ulp` (`comp_01`, `comp_11`).

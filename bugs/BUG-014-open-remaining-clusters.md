# BUG-014 through BUG-018: Single-Instruction Sweep Clusters (All Closed in v49 — 129/129)

## BUG-014 (`NEW-2`): `ICMP_MULTI` Vertex Output Varying Array Slot Offset Mapping (1 target: `ICMP_MULTI`)
- **Status:** CLOSED (v49)
- **Root Cause:** `spirv2ssa_mem.cpp` `flush_stage_outputs()` packed only the first 4 lanes of a `vec4[4]` vertex output array into a single `StoreVertVarying(loc)` and dropped elements `1..3` (`vert_var[3..5]`), while `isa_recipes.py` left unwritten elements uninitialized (creating UB vs Valhall TLS loads).
- **Fix:** Emit one `StoreVertVarying(loc + e * elem_slots)` per array element in `src/spirv2ssa/spirv2ssa_mem.cpp` and initialize all 4 elements of `out_arr` in `tools/corpus_synth/isa_recipes.py` (`isa_spv_harness.py` vertex constants `%c0/%c1/%c2`).

## BUG-015 (`RC-5`): Cubemap Software Projection vs. 3D `TexSample(vec3)` Uninterpreted Function (5 targets: `CUBEFACE1`, `CUBEFACE2`, `CUBE_SSEL`, `CUBE_TSEL`, `FMA_RSCALE`)
- **Status:** CLOSED (v49)
- **Root Cause:** Direct `spirv2ssa` passed raw 3D `(x, y, z)` to `TexSample` (`arity=3`), whereas Mali Valhall projects 3D cube vectors via `CUBEFACE1/2 + FRCP + CUBE_SSEL/TSEL + FMA_RSCALE + MUX` into `(s_packed, t)` (`arity=2`) for `TEX_SINGLE.cube`.
- **Fix:** Tag `Dim=Cube` `TexSample`/`TexSampleLod` with `imm_u32_2 = 1` in `src/spirv2ssa/spirv2ssa_mem.cpp` and project Direct 3D `(x, y, z)` into Valhall's exact `(s_packed, t)` in `tools/symbolic_ssa/interpreter.py` before calling `hw_uf`.

## BUG-016 (`RC-6`): Branchy Transcendental SSBO ULP Tolerance & `Pow` Negative-Base UB (6 targets: `FADD_IMM`, `FADD_LSCALE`, `FCOS_TABLE`, `FLOGD`, `FMUL_RSCALE`, `FSIN_TABLE`)
- **Status:** CLOSED (v49)
- **Root Cause:** (a) In `tools/coverage/equiv_fuzzer.py`, `_has_fp_ssbo` only checked direct `F32` / `Bitcast(F32)` operands of `PsbStore` and missed `Phi`/`Select` merges in `--complexity branchy` shaders; (b) `isa_target_synth.py` passed an explicit `DiffPolicy()` that bypassed the transcendental SSBO augmentation; (c) `FMUL_RSCALE` (`Pow(x, y)`) fed negative `x` (SPIR-V UB) and compounded `Exp2(y * Log2(x))` ULP up to 20 ULP.
- **Fix:** Traverse `Phi`/`Select`/`Copy`/`Bitcast` in `equiv_fuzzer.py`, augment any `max_ulp > 0` policy with `ssbo:` (`max_ulp=32` when `Pow` is present, `8` otherwise, `max_abs_err=2^-11`), and clamp `FMUL_RSCALE` base to `FAbs(%fa) + 1.0`.

## BUG-017 (`RC-8`): `FATAN_ASSIST` / `FATAN_TABLE` / `LDEXP.inf` Lowering (3 targets: `FATAN_ASSIST`, `FATAN_TABLE`, `MUX`)
- **Status:** CLOSED (v49)
- **Root Cause:** `specs/lowering/valhall_lowering.json` lowered both `FATAN_ASSIST` and `FATAN_TABLE` to 1-arg `Atan`, and `LDEXP.f32.inf`/`.inf0` read all 32 bits of the exponent operand instead of sign-extending the low 16-bit `int16` (`h0`).
- **Fix:** Lower `LDEXP.f32` with `inf`/`inf0` via `BfeS(%1, 0, 16)`, lower `FATAN_ASSIST` to packed `{h0: int16(-frexp_exp(max(|y|,|x|))), h1: f16(clamp(|y'|/|x'|, 0, 65504))}`, and lower `FATAN_TABLE` to `Atan(f16_to_f32(h1))`.

## BUG-018 (`RC-10c` + `RC-13a/b` + `LSHIFT_XOR`/`RSHIFT_XOR`): `TEX_GRADIENT`, `CLPER`, `LD_PKA`, and Shift-XOR Fusion (5 targets)
- **Status:** CLOSED (v49)
- **Root Causes & Fixes:**
  - `TEX_GRADIENT`: `src/valhall_pass1_handlers.cpp` now records `TEX_GRADIENT.derivative` `(ddx.x, ddx.y, ddy.x, ddy.y)` in `Pass1Context::grad_desc_map` and unpacks it on `TEX_SINGLE.grdesc` into `TexSampleLod(uv, ddx.x, ddx.y, ddy.x, ddy.y, tok)`.
  - `CLPER`: Mask `OpGroupNonUniformShuffleXor` mask operand to `%b & 15` (Valhall subgroup size = 16) in `tools/corpus_synth/isa_recipes.py` and mask `bi_40` (`SubgroupLocalInvocationId`) to `& 0xF` in `concrete_shader_env.py`.
  - `LD_PKA`: Use `0x01000000` (16 MB per slot) descriptor-array stride in `src/spirv2ssa/spirv2ssa_mem.cpp`, propagate `fau_fcst_words` in `equiv_bundle.cpp`, and resolve `LD_PKA.i64` SRT table loads (`0x3e000000 | rack`, `offset = desc_idx * 0x20 + 8 + (bidx << 5)`) to `LoadSidebandU64(buf_sym) + (bidx << 24)` in `src/valhall_pass1_handlers.cpp`.
  - `LSHIFT_XOR` / `RSHIFT_XOR`: Use `%d` instead of `%c` in `isa_recipes.py` so `libMali` does not factor `^ %c` out of the `branchy` `CSEL`, enabling `LSHIFT_XOR` and `RSHIFT_XOR` emission.

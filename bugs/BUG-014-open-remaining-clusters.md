# BUG-014 through BUG-018: Remaining 16 Open Single-Instruction Divergences (Out of 142)

## BUG-014 (`NEW-2`): `ICMP_MULTI` Vertex Output Varying Array Slot Offset Mapping (1 target: `ICMP_MULTI`)
- **Status:** OPEN
- **Root Cause:** `libMali` writes a `vec4[4]` vertex output array at `#0x20/#0x30/#0x40/#0x50` (`vert_var[2..5]`), while Direct `spirv2ssa` `OutputVar` store drops array indices on `StoreVertVarying`.

## BUG-015 (`RC-5`): Cubemap Software Projection vs. 3D `TexSample(vec3)` Uninterpreted Function (5 targets: `CUBEFACE1`, `CUBEFACE2`, `CUBE_SSEL`, `CUBE_TSEL`, `FMA_RSCALE`)
- **Status:** OPEN
- **Root Cause:** `hw_uf` hashes 3D `(x,y,z)` on Direct vs. projected 2D+face `(s,t,face)` on Valhall. Fix: canonicalize Direct `Dim=Cube` coordinates via standard Vulkan face projection before `hw_uf`.

## BUG-016 (`RC-6`): Transcendental Polynomial Absolute-Error Tolerance Near Zero Crossings (4 targets: `FADD_IMM`, `FSIN_TABLE`, `FCOS_TABLE`, `FMUL_RSCALE`)
- **Status:** OPEN (benign FP precision)
- **Root Cause:** Near zero crossings (e.g., `sin(pi)`), Valhall's degree-3 minimax polynomial exceeds 8 ULP while staying within Vulkan's GLSL.std.450 absolute-error spec (`<= 2^-11`). Fix: support absolute-error tolerance alongside ULP for transcendental FP outputs.

## BUG-017 (`RC-8`): `FATAN_ASSIST` / `FATAN_TABLE` Lowered to 1-Arg `Atan` (3 targets: `FATAN_ASSIST`, `FATAN_TABLE`, `MUX`)
- **Status:** OPEN
- **Root Cause:** `specs/lowering/valhall_lowering.json:2415–2447` lowers both `FATAN_ASSIST` and `FATAN_TABLE` to 1-arg `Atan`, yielding `NaN` on `Atan2`. Fix: model `FATAN_ASSIST` (`h0` exponent shift, `h1` `f16` table tangent) and `FATAN_TABLE` (`atan(c)`).

## BUG-018 (`RC-10c` + `RC-13a/b`): `TEX_GRADIENT` (`grdesc`), `CLPER` (`mask >= 16` UB), `LD_PKA` (SRT Table `u1.w0`) (3 targets)
- **Status:** OPEN
- **Root Causes:** (a) `TEX_GRADIENT`: `TEX_SINGLE.grdesc` needs gradient-descriptor unpacking to match Direct `TexSampleLod(uv, ddx, ddy)`; (b) `CLPER`: fuzzer feeds random `mask >= 16` (SPIR-V UB) where Valhall's 16-way `CSEL` ladder defaults to `mask=0`; (c) `LD_PKA`: dynamic SSBO array descriptor table `u1.w0 = 0x3e000004` needs SRT table resolution in Pass 1.

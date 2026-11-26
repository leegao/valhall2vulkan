# BUG-006 (`RC-10d`): `TEX_FETCH` Missing `lod` Arg & Single-Sampler `tex_idx` Resolution
- **Status:** FIXED in Checkpoint v48 (`53b2c3f`)
- **Layer:** Valhall Pass 1 (`src/valhall_pass1_handlers.cpp:763–795`)
- **Targets (2):** `TEX_FETCH`, `LD_TEX`.
- **Root Cause & Fix:** Appended `ConstU32(0)` LOD only for `Mnemonic::TEX_FETCH` when arity is 1 (preserving `LD_TEX_IMM` arity for `comp_unique_03_fba18fbe.spv`), and added single-sampler fallback when `has_tex_idx` is false.

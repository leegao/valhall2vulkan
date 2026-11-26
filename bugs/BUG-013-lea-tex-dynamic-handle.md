# BUG-013 (`RC-10e`): `lea_tex()` Missing Single-Sampler Fallback for Dynamic Handles
- **Status:** FIXED in Checkpoint v48.2
- **Layer:** Valhall Pass 1 (`src/valhall_pass1_handlers.cpp:854`)
- **Targets (1):** `LEA_TEX`.
- **Root Cause & Fix:** Added `if (!has_tex_idx && ctx.opts.samplers.size() == 1u)` fallback in `lea_tex()`, resolving `tex:0:4`.

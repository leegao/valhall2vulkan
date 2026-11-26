# BUG-009 (`RC-10a/b` + `NEW-1`): `VAR_TEX_GATHER` `is_gather=false` & Valhall Gather4 Channel Order
- **Status:** FIXED in Checkpoints v48 (`53b2c3f`) & v48.1 (`fc000b2`)
- **Layer:** Valhall Pass 1 (`specs/lowering/valhall_lowering.json` + `src/valhall_pass1_handlers.cpp:389`)
- **Targets (2):** `TEX_GATHER`, `VAR_TEX_GATHER`.
- **Root Cause & Fix:** Fixed `VAR_TEX_GATHER` `is_gather=true` and applied Valhall's hardware gather4 permutation `kGather4Perm[4] = {3, 2, 0, 1}` (hardware `(r,g,b,a)` = SPIR-V `(w,z,x,y)`) in `extract()`, guarded to `TupleExtract` texture outputs.

# BUG-010 (`RC-7` + `NEW-3` + `NEW-4`): Subgroup Waterfall Loop Non-Termination & Dynamic `InputVar` Indexing
- **Status:** FIXED in Checkpoint v48.1 (`fc000b2`)
- **Layer:** `concrete_bitvec.py:265`, `valhall_pass1_handlers.cpp:625`, `interpreter.py:693`, `spirv2ssa_mem.cpp:681`
- **Targets (2):** `LD_ATTR`, `LEA_BUF_IMM` (plus unblocked `LD_VAR`/`LD_VAR_FLAT` -> `BUG-011`).
- **Root Cause & Fix:** Modeled single-lane `subgroup_ballot` (`1 << (lane_id & 15)`) and `subgroup_shuffle` identity (`(lane & 15) == (lane_id & 15)`), wired dynamic index registers in Pass 1 & interpreter, and passed `loc + ptr.dynamic_index` in `spirv2ssa_mem.cpp:lower_load_ptr()`.

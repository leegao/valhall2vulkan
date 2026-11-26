# BUG-004 (`RC-4`): `pass_gvn_dce.py` `IMul -> Shl` Intra-Block Dominance Violation
- **Status:** FIXED in Checkpoint v48 (`53b2c3f`)
- **Layer:** Symbolic SSA Optimizer (`tools/symbolic_ssa/passes/pass_gvn_dce.py:189–229`)
- **Targets (5):** `ATOM`, `ATOM1`, `BRANCH`, `MOV`, `MOV_IMM`.
- **Witness:** `ATOM`: Direct addr `0x000df3a4` (`57146 * 16 + 4`), Valhall addr `0x0000df3e` (`57146 << 0 + 4`).
- **Root Cause & Fix:** `shift_const_cache` reused a `ConstU32(4)` defined later in Block 0 (`pc=0x18`) for an earlier `IMul` (`pc=0x10`), causing linear interpretation to read default `0`. Allocated shift constants at index `0` of `entry_bid`.

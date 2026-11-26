# BUG-011 (`NEW-5`): `sinp_addr_to_loc` Missing Array-Element Varying Slots
- **Status:** FIXED in Checkpoint v48.2
- **Layer:** ShaderEnv (`tools/symbolic_ssa/domains/shader_env/base.py:249`)
- **Targets (2):** `LD_VAR`, `LD_VAR_FLAT`.
- **Root Cause & Fix:** MBS2 `sinp` only lists base slots of varying arrays (`addr=0 -> loc=4`, next entry `addr=4 -> loc=0`). Expanded contiguous unmapped slots `base_hw + k -> base_loc + k`.

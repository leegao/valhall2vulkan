# BUG-007 (`RC-11`): `ACMPXCHG` Double-Swapped Staging Registers
- **Status:** FIXED in Checkpoint v48 (`53b2c3f`)
- **Layer:** Valhall Pass 1 (`specs/lowering/valhall_lowering.json:6450`)
- **Targets (1):** `ACMPXCHG`.
- **Witness:** `D=('atomiccmpxchg', 4, 1098907648, 1056964608)`, `V=('atomiccmpxchg', 4, 1056964608, 1098907648)`.
- **Root Cause & Fix:** Passed `@sr_get(0), @sr_get(1)` instead of `@sr_get(1), @sr_get(0)` into `@atomic_cmpxchg`.

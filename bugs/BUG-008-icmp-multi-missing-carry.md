# BUG-008 (`RC-12`): `ICMP_MULTI` Dropped Carry / Tie-Breaker Operand `%2`
- **Status:** FIXED in Checkpoint v48 (`53b2c3f`)
- **Layer:** Valhall Pass 1 (`specs/lowering/valhall_lowering.json` `ICMP_MULTI`)
- **Targets (1):** `ICMP_MULTI` (value mismatch resolved; exposed `BUG-014`).
- **Root Cause & Fix:** Combined `ICmpEq(%a, %b)` with `%prev = %2` via `Select(%eq, %prev, %cmp_res)`.

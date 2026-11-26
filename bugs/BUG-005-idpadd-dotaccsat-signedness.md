# BUG-005 (`RC-9`): Missing `SpvOp*DotAccSat` + `handle_int_idpadd` Signedness Bug
- **Status:** FIXED in Checkpoint v48 (`53b2c3f`) — **formally proven `UNSAT` in Z3 (`test_all_paths_miter`)**
- **Layer:** `spirv2ssa` (`spirv2ssa_ops.cpp:1018`) + Valhall Pass 1 (`valhall_pass1_handlers.cpp:969`)
- **Targets (2):** `IDPADD`, `IDPADD.v4u8`.
- **Root Cause & Fix:** Added `SpvOpSDotAccSat`/`UDotAccSat`/`SUDotAccSat` in `spirv2ssa_ops.cpp`, and fixed `var.find("s") == 0` -> `var.find("s") != std::string::npos` in `handle_int_idpadd` (since `"v4s8"` has `'s'` at index 2).

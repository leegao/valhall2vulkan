; Shader Equivalence Verification SMT-LIB2 Model
; Shader: etc2_compress.spv | Stage: Compute | FP Mode: exact
; Direct SSA: 447B/16812N | Valhall SSA: 262B/25839N
; LocalSize: (8, 8, 1) | MaxUnroll: 1 | Preconditions: 6
(set-logic ALL)

; --- ssbo:0:1[probe_off]: simplified to false (UNSAT) ---
; --- ssbo:0:2[probe_off]: simplified to false (UNSAT) ---
; --- fallback_mem[probe_addr]: simplified to false (UNSAT) ---

; Shader Equivalence Verification SMT-LIB2 Model
; Shader: comp_rep_minimal.spv | Stage: Compute | FP Mode: exact
; Direct SSA: 1B/23N | Valhall SSA: 1B/26N
; LocalSize: (1, 1, 1) | MaxUnroll: 1 | Preconditions: 0
(set-logic ALL)

; --- ssbo:0:1+0x0.written: syntactically identical (UNSAT) ---
; --- ssbo:0:1+0x0.value: syntactically identical (UNSAT) ---
; --- ssbo:0:1+0x4.written: syntactically identical (UNSAT) ---
; --- ssbo:0:1+0x4.value: syntactically identical (UNSAT) ---
; --- ssbo:0:1+0x8.written: syntactically identical (UNSAT) ---
; --- ssbo:0:1+0x8.value: syntactically identical (UNSAT) ---
; --- ssbo:0:1+0xc.written: syntactically identical (UNSAT) ---
; --- ssbo:0:1+0xc.value: syntactically identical (UNSAT) ---

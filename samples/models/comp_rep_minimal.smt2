; Shader Equivalence Verification SMT-LIB2 Model
; Shader: comp_rep_minimal.spv | Stage: Compute | FP Mode: exact
; Direct SSA: 1B/23N | Valhall SSA: 1B/26N
; LocalSize: (1, 1, 1) | MaxUnroll: 1 | Preconditions: 0
(set-logic ALL)

; === Output Miter Check: ssbo:0:1+0x0.written ===
(push 1)
; benchmark generated from python API
(set-info :status unknown)
(check-sat)
(check-sat)
(pop 1)
; === Output Miter Check: ssbo:0:1+0x0.value ===
(push 1)
; benchmark generated from python API
(set-info :status unknown)
(declare-fun mem_ssbo_0_1 () (Array (_ BitVec 32) (_ BitVec 32)))
(assert
 (not (= (select mem_ssbo_0_1 (_ bv0 32)) (_ bv1000 32))))
(check-sat)
(check-sat)
(pop 1)
; === Output Miter Check: ssbo:0:1+0x4.written ===
(push 1)
; benchmark generated from python API
(set-info :status unknown)
(check-sat)
(check-sat)
(pop 1)
; === Output Miter Check: ssbo:0:1+0x4.value ===
(push 1)
; benchmark generated from python API
(set-info :status unknown)
(declare-fun mem_ssbo_0_1 () (Array (_ BitVec 32) (_ BitVec 32)))
(assert
 (not (= (select mem_ssbo_0_1 (_ bv4 32)) (_ bv4294966296 32))))
(check-sat)
(check-sat)
(pop 1)
; === Output Miter Check: ssbo:0:1+0x8.written ===
(push 1)
; benchmark generated from python API
(set-info :status unknown)
(check-sat)
(check-sat)
(pop 1)
; === Output Miter Check: ssbo:0:1+0x8.value ===
(push 1)
; benchmark generated from python API
(set-info :status unknown)
(declare-fun mem_ssbo_0_1 () (Array (_ BitVec 32) (_ BitVec 32)))
(assert
 (not (= (select mem_ssbo_0_1 (_ bv8 32)) (_ bv0 32))))
(check-sat)
(check-sat)
(pop 1)
; === Output Miter Check: ssbo:0:1+0xc.written ===
(push 1)
; benchmark generated from python API
(set-info :status unknown)
(check-sat)
(check-sat)
(pop 1)
; === Output Miter Check: ssbo:0:1+0xc.value ===
(push 1)
; benchmark generated from python API
(set-info :status unknown)
(declare-fun mem_ssbo_0_1 () (Array (_ BitVec 32) (_ BitVec 32)))
(assert
 (not (= (select mem_ssbo_0_1 (_ bv12 32)) (_ bv0 32))))
(check-sat)
(check-sat)
(pop 1)
; === Output Miter Check: fallback_mem[probe_addr] ===
(push 1)
; benchmark generated from python API
(set-info :status unknown)
(declare-fun probe_fallback_addr () (_ BitVec 64))
(declare-fun fau_sym () (Array (_ BitVec 32) (_ BitVec 32)))
(declare-fun mem_fallback () (Array (_ BitVec 64) (_ BitVec 32)))
(assert
 (let ((?x72 (select fau_sym (_ bv1 32))))
(let ((?x96 (concat ?x72 (_ bv12 32))))
(let ((?x77 (concat ?x72 (_ bv8 32))))
(let ((?x66 (concat ?x72 (_ bv0 32))))
(let ((?x59 (store mem_fallback ?x66 (_ bv1000 32))))
(let ((?x146 (store ?x59 (concat ?x72 (_ bv4 32)) (_ bv4294966296 32))))
(let ((?x92 (store ?x146 ?x77 (_ bv0 32))))
(let ((?x143 (select mem_fallback probe_fallback_addr)))
(not (= ?x143 (select (store ?x92 ?x96 (_ bv0 32)) probe_fallback_addr))))))))))))
(check-sat)
(check-sat)
(pop 1)

# Formal Specification & Architecture of `shader_equiv_verifier`

## Table of Contents

1. [Executive Overview & Threat Model](#1-executive-overview--threat-model)
2. [End-to-End Equivalence Pipeline](#2-end-to-end-equivalence-pipeline)
3. [Formal Syntax of Serialized `ValhallSsaModule`](#3-formal-syntax-of-serialized-valhallssamodule)
4. [Shared Symbolic Execution Environment](#4-shared-symbolic-execution-environment)
5. [Denotational Semantics of `BlockEffectState`](#5-denotational-semantics-of-blockeffectstate)
6. [Whole-CFG Topological DAG Path-Merging](#6-whole-cfg-topological-dag-path-merging)
7. [Bounded Loop Unrolling & `BOUNDED_UNSAT` Soundness](#7-bounded-loop-unrolling--bounded_unsat-soundness)
8. [Memory Model: Disjoint 4 GB Regions, Additive Base Recovery & Shared-Memory Unification](#8-memory-model-disjoint-4-gb-regions-additive-base-recovery--shared-memory-unification)
9. [Compute Execution-Environment Preconditions (`LocalSize`)](#9-compute-execution-environment-preconditions-localsize)
10. [Fragment Discard vs. Demote-to-Helper-Invocation Semantics](#10-fragment-discard-vs-demote-to-helper-invocation-semantics)
11. [Atomic Read-Modify-Write State & Ordered Event Logs](#11-atomic-read-modify-write-state--ordered-event-logs)
12. [Storage-Image Write Normalization (`PsbStore` vs. `StoreTexel`)](#12-storage-image-write-normalization-psbstore-vs-storetexel)
13. [Vertex Viewport Transform Precondition & Symbolic Epilogue Roadmap](#13-vertex-viewport-transform-precondition--symbolic-epilogue-roadmap)
14. [Theory Stratification & Layer-2 Pluggable FP Proof-Obligation Discharger](#14-theory-stratification--layer-2-pluggable-fp-proof-obligation-discharger)
15. [Per-Output Miter Decomposition, Fail-Closed Asymmetry & Counterexample Localization](#15-per-output-miter-decomposition-fail-closed-asymmetry--counterexample-localization)
16. [SMT-LIB2 Model Export & Reproducibility](#16-smt-lib2-model-export--reproducibility)
17. [Empirical Evaluation on the 25-Sample Corpus](#17-empirical-evaluation-on-the-25-sample-corpus)
18. [Soundness Boundaries, Known Limitations & Future Work](#18-soundness-boundaries-known-limitations--future-work)

---

## 1. Executive Overview & Threat Model

`shader_equiv_verifier` is a static, all-path formal equivalence checker that proves (or refutes with a concrete counterexample) functional equivalence between two independent lowering paths that both target the project's canonical SSA intermediate representation, **`ValhallSsaModule`**:

1. **Reference Path (`direct_ssa`)**: Direct lowering from Vulkan SPIR-V 1.6 bytecode (`*.spv`) into `ValhallSsaModule` via `src/spirv2ssa/` (`SpirvLowerer`).
2. **Compiled Hardware Path (`valhall_ssa`)**: Compiling the exact same SPIR-V shader through ARM's reference Mali Offline Compiler (`libMali-Gxx` / `malioc_inproc`) to Valhall machine code + MBS2 metadata, disassembling and constructing a CFG (`valhall_cfg`), and lifting the Valhall ISA back into `ValhallSsaModule` via `valhall_pass1` (`src/valhall_pass1*.cpp`).

```
                       +------------------------+
                       |   Input SPIR-V (.spv)  |
                       +-----------+------------+
                                   |
                 +-----------------+-----------------+
                 |                                   |
                 v                                   v
      +---------------------+            +-----------------------+
      | src/spirv2ssa/      |            | libMali-Gxx (malioc)  |
      | Direct SPIR-V ->    |            | Valhall EBIN + MBS2   |
      | ValhallSsaModule    |            +-----------+-----------+
      +----------+----------+                        |
                 |                                   v
                 |                       +-----------------------+
                 |                       | valhall_cfg +         |
                 |                       | valhall_pass1 lifting |
                 |                       | -> ValhallSsaModule   |
                 |                       +-----------+-----------+
                 |                                   |
                 +-----------------+-----------------+
                                   |
                                   v
                 +-----------------------------------+
                 | equiv_bundle.cpp (JSON Bundle)    |
                 | - direct_ssa, valhall_ssa         |
                 | - direct_bindings, mbs2_context   |
                 |   (local_size, fcsts, rlocs,      |
                 |    pilot_fau, sinp, sout, pc_map) |
                 +-----------------+-----------------+
                                   |
                                   v
                 +-------------------------------------+
                 | shader_equiv_verifier.py            |
                 | 1. SymbolicShaderEnv + Preconditions|
                 | 2. Whole-CFG Topological DAG Eval   |
                 | 3. Per-Output Z3 Miter Checks       |
                 | 4. Layer-2 FP Discharger + UF       |
                 | 5. orig_pc BFS Localization         |
                 +-------------------------------------+
```

### Verification Objective

Let $\mathcal{P}_D$ denote `direct_ssa` and $\mathcal{P}_V$ denote `valhall_ssa`. Let $\mathcal{I}$ be the vector of all shared symbolic shader inputs (vertex attributes, fragment varyings, builtins, initial buffer memory arrays, initial image arrays, unbound FAU words, and subgroup `lane_id`). Let $\Phi(\mathcal{I})$ be the conjunction of execution-environment preconditions (e.g., Vulkan `LocalSize` bounds on compute invocation IDs).

For every externally observable output channel $o \in \mathcal{O}(\mathcal{P}_D) \cup \mathcal{O}(\mathcal{P}_V)$ with active guard $G_o(\mathcal{I})$, the verifier checks the validity of the miter implication:

$$\forall \mathcal{I}.\; \Phi(\mathcal{I}) \land G_o(\mathcal{I}) \implies [\![ \mathcal{P}_D ]\!]_o(\mathcal{I}) = [\![ \mathcal{P}_V ]\!]_o(\mathcal{I})$$

Equivalently, Z3 searches for a satisfying assignment (counterexample) to the negated formula:

$$\exists \mathcal{I}.\; \Phi(\mathcal{I}) \land G_o(\mathcal{I}) \land \left([\![ \mathcal{P}_D ]\!]_o(\mathcal{I}) \neq [\![ \mathcal{P}_V ]\!]_o(\mathcal{I})\right)$$

### Verdict Lattice

The verifier returns one of six mutually exclusive verdicts:

| Verdict | Mathematical Meaning | Soundness Guarantee |
|---|---|---|
| **`UNSAT`** | Every output miter is unsatisfiable on loop-free CFGs in exact QF_BV/QF_FP (or sound UF-FP). | **Complete & sound** equivalence for all inputs satisfying $\Phi(\mathcal{I})$. |
| **`BOUNDED_UNSAT`** | Every output miter is unsatisfiable, and at least one CFG contains a backedge unrolled to depth $K = \text{max\\_unroll}$. | **Sound up to $K$ loop iterations**; divergences requiring $> K$ iterations are not ruled out. |
| **`SAT`** | At least one output miter has a concrete counterexample $\mathcal{I}^*$ in exact theory. | **True refutation** under the modeled environment $\Phi(\mathcal{I})$ (either a real compiler/lifter bug or a missing environment precondition). |
| **`UNKNOWN`** | At least one output timed out / hit `rlimit` in exact QF_FP, and neither Layer-2 FP discharge nor UF-FP fallback proved `UNSAT`. | Inconclusive; exact SMT-LIB2 queries are exportable via `--dump-smt2` for offline analysis. |
| **`SKIP`** | No comparable observable outputs were extracted on either side. | Vacuous; hardened in v25.1 so atomic-only and image-write-only shaders never `SKIP`. |
| **`ERROR`** | Either `direct_ssa` or `valhall_ssa` failed to lower (`direct_ok == false` or `valhall_ok == false`). | Pipeline failure before SMT construction. |

---

## 2. End-to-End Equivalence Pipeline

The C++ driver `tools/spirv2valhall/equiv_bundle.cpp` constructs the self-contained JSON bundle consumed by `shader_equiv_verifier.py`:

1. **SPIR-V Direct Lowering**: Reads raw SPIR-V words and runs `spirv2ssa::lower_spirv_to_valhall_ssa(spv_words, dopts)` with `dopts.tolerant = true`, producing `direct_res.ssa` and canonical resource bindings (`direct_bindings`: `ssbo:<set>:<binding>`, `ubo:<set>:<binding>`, `push_const`, `tex:<set>:<binding>`, `img:<set>:<binding>`).
2. **Valhall Extraction & Pass-1 Lifting**: Selects the primary EBIN (plus `VertexVarying` secondary EBIN for vertex shaders), builds `cfg::ControlFlowGraph`, propagates MBS2 `kern.wg_{x,y,z}` into `ValhallTranslateOptions`, and runs `compiler::run_pass1_lowering(pri_cfg, sec_cfg, vopts)` to produce `p1_res.ssa`.
3. **MBS2 Context Export**: Serializes all ABI metadata needed to align Valhall's low-level FAU/slot references with SPIR-V's high-level bindings:
   - `local_size`: `[wg_x, wg_y, wg_z]` from `mbs2.kern` (added in v25.1).
   - `fcsts`: Module-level and EBIN-level constant FAU words (`word_idx -> uint32`).
   - `rlocs`: Uniform buffer relocations (`word_idx -> (ubo:0:binding, uni_offset)`), including `gl_mali_ViewportTransform`.
   - `pilot_fau`: Symbolically analyzed pilot-shader FAU mappings (`buffer_addr`, `symbol_value`, `const_int`, `buffer_size`, `descriptor_addr`, `abi_preload`, with optional `pointer_offset` / `byte_offset`).
   - `sinp` / `sout`: Valhall attribute/varying slot-to-SPIR-V-location tables.
   - `pc_disasm`: Map from Valhall instruction byte PC (`orig_pc`) to formatted disassembly string for counterexample attribution.

---

## 3. Formal Syntax of Serialized `ValhallSsaModule`

A serialized `ValhallSsaModule` is a tuple $M = (B, N)$ where:

- $B = [b_0, b_1, \dots, b_{m-1}]$ is an ordered list of basic blocks in reverse postorder (RPO), where $b_0$ is the unique entry block (`id = 0`). Each block $b_i$ specifies:
  - `preds`: predecessor block IDs.
  - `phis`: SSA node IDs of $\phi$-nodes at block entry.
  - `nodes`: ordered SSA node IDs in the block body.
  - Terminator metadata: `is_cond`, `branch_cond`, `branch_target`, `fallthrough`, `is_end`, `is_discard`.
- $N$ is a map from integer `SsaId` to node records $n = \langle \text{id}, \text{op}, \text{type}, \text{args}, \text{imm\\_u32}, \text{imm\\_u32\\_2}, \text{imm\\_u64}, \text{aux\\_bool}, \text{aux\\_bool2}, \text{aux\\_str}, \text{phi}, \text{orig\\_pc} \rangle$.

### Value Sort Domain

Every SSA value evaluates into the Python/Z3 domain `Z3Val`:

$$\text{Z3Val} ::= \text{BitVec}(32) \mid \text{BitVec}(64) \mid \text{Bool} \mid \text{Tuple}(\text{Z3Val}^k)$$

Floating-point values (`F32`) are stored canonically as 32-bit IEEE-754 bitvectors `BitVec(32)` rather than raw `FPSort(8, 24)`. This eliminates unnecessary sort conversions across `Bitcast`, `Load`, `Store`, `Pack64`, `FAbs` (`x & 0x7FFFFFFF`), and `FNeg` (`x ^ 0x80000000`), and guarantees that bit-exact equality `lhs == rhs` distinguishes `-0.0` (`0x80000000`) from `+0.0` (`0x00000000`) and preserves NaN payloads. When IEEE-754 arithmetic is evaluated in `exact` mode, `bv32_to_fp32` / `fp32_to_bv32` (`z3.fpBVToFP` / `z3.fpToIEEEBV`) convert at the operation boundary.

---

## 4. Shared Symbolic Execution Environment

`SymbolicShaderEnv` constructs a **single shared universe** of symbolic inputs so that both `direct_ssa` and `valhall_ssa` read identical symbolic values for every external input:

1. **Builtins (`LoadBuiltinInput`)**: Indexed by SPIR-V `SpvBuiltIn` integer ID:
   - `V4F32` (e.g., `FragCoord`, `bi_15`): 4-tuple `(bi_15_0, bi_15_1, bi_15_2, bi_15_3)` of `BV32`.
   - `V3U32` (e.g., `WorkgroupId` = 26, `LocalInvocationId` = 27, `GlobalInvocationId` = 28): 3-tuple `(bi_k_0, bi_k_1, bi_k_2)` of `BV32`.
   - `Bool` (e.g., `FrontFacing` = 17): `Bool("bi_17")`.
   - Scalar `U32` (e.g., `LocalInvocationIndex` = 29, `VertexIndex` = 42, `InstanceIndex` = 43): `BitVec("bi_k", 32)`.
2. **Stage Inputs (`LoadVertAttr`, `LoadFragVarying`)**: 4-tuples of `BV32` named `vert_attr_loc<L>_<c>` or `frag_var_loc<L>_<c>`. On the Valhall side, hardware slot indices are translated through `sinp_addr_to_loc` so both sides reference the same SPIR-V location `L`.
3. **FAU Uniform Resolution (`LoadFau`)**: Resolved in strict priority order:
   1. `fcst_by_word[w]` $\to$ concrete `bv32(val)` (compiler-emitted constant).
   2. `viewport_fau_words[w]` $\to$ `0x3F800000` (scale) or `0x00000000` (bias) under the current identity viewport precondition.
   3. `pilot_by_word[w]`:
      - `const_int` $\to$ concrete low/high 32-bit word.
      - `symbol_value` (non-derived, with `byte_offset`) $\to$ `Select(init_buf_mem[key], bv32(byte_offset))`.
      - `buffer_addr` (non-derived) $\to$ low or high 32-bit word of `(get_buf_base(key) + pointer_offset)`.
      - `buffer_size` / `descriptor_addr` / `abi_preload` / `is_derived` $\to$ symbolic `Select(fau_sym, bv32(w))`.
   4. `rloc_by_word[w]` $\to$ `Select(init_buf_mem[ubo:0:binding], bv32(uni_offset))`.
   5. Fallback $\to$ `Select(fau_sym, bv32(w))`.
4. **Initial Memory Arrays**:
   - Per-buffer word-addressed arrays: `mem_<safe_key> : Array(BV32, BV32)`.
   - Per-image 4-channel arrays: `img_<safe_key>_c{0..3} : Array(BV64, BV32)` indexed by packed `Concat(y, x)`.
   - Unresolved-address fallback array: `mem_fallback : Array(BV64, BV32)`.

---

## 5. Denotational Semantics of `BlockEffectState`

At the exit of every basic block $b$, the observable side-effect state is represented by a `BlockEffectState` tuple:

$$\sigma = \langle \tau, \delta, C, B, V, S_c, M, I, F, A, P \rangle$$

where:

- $\tau \in \text{Bool}$ (`terminated`): Whether a hard terminating discard (`OpKill` / `OpTerminateInvocation`) has executed on the path.
- $\delta \in \text{Bool}$ (`demoted`): Whether `OpDemoteToHelperInvocation` has executed on the path.
- `discarded` $= \tau \lor \delta$: Combined fragment discard predicate.
- $C : \mathbb{N} \to \text{BV32}^4$ (`frag_colors`): Render-target color outputs `frag_color[rt]`.
- $B : \mathbb{N} \to \text{Z3Val}$ (`builtin_outputs`): Builtin outputs (`gl_Position` = 0, `gl_PointSize` = 1, `gl_FragDepth` = 22, etc.).
- $V : \mathbb{N} \to \text{BV32}^4$ (`vert_varyings`): Vertex shader varying outputs by SPIR-V location.
- $S_c : (\text{BufKey} \times \mathbb{N}) \to (\text{Bool} \times \text{BV32})$ (`const_stores`): Structured constant-offset store map `(written_cond, val)`.
- $M : \text{BufKey} \to \text{Array}(\text{BV32}, \text{BV32})$ (`buf_mems`): Per-buffer dynamic-offset memory arrays.
- $I : \text{ImgKey} \to \text{Array}(\text{BV64}, \text{BV32})^4$ (`image_mems`): Storage-image 4-channel arrays.
- $F : \text{Array}(\text{BV64}, \text{BV32})$ (`fallback_mem`): Fallback memory array for unresolved 64-bit addresses.
- $A : \text{BufKey} \to [(\text{Bool}, \text{Tag}, \text{Addr}, \text{BV32}, \text{BV32})]$ (`atomic_events`): Ordered per-buffer atomic RMW event log.
- $P : \text{OutKey} \to 2^{\text{SsaId}}$ (`output_pcs`): Provenance set of sink SSA node IDs for counterexample PC attribution.

### State Merging Operator

Given incoming edges $\{(c_1, \sigma_1), \dots, (c_k, \sigma_k)\}$ entering a merge block $b$, the merged state $\sigma = \text{Merge}([(c_1, \sigma_1), \dots, (c_k, \sigma_k)])$ is computed by cascading `z3.If` over each component with aggressive syntactic short-circuiting (`_z3_if`):

- If $k = 1$, clone $\sigma_1$ directly without introducing `If` nodes.
- If $c_i \equiv \text{False}$, skip $\sigma_i$.
- If $a \text{ eq } b$ (pointer/AST equality in Z3), return $a$ directly without wrapping in `z3.If(c, a, b)`.

---

## 6. Whole-CFG Topological DAG Path-Merging

Naive path-by-path symbolic execution explores $\mathcal{O}(2^k)$ paths on a shader with $k$ sequential `if/else` diamonds — completely intractable on real-world encoders like `etc2_astc_enc.spv` (3,132 blocks) or `etc2_bc6.spv` (1,846 blocks).

`SymbolicSsaInterpreter.evaluate()` instead performs **single-pass whole-CFG topological DAG path-merging**:

1. **Reachability Propagation**:
   - Block $b_0$ starts with $R(b_0) = \text{True}$ and $\sigma_{\text{in}}(b_0) = \sigma_{\text{init}}$.
   - For each block $b_i$ ($i > 0$) with already-evaluated incoming edges $E_{\text{in}}(b_i) = \{(p_j, e_j)\}$:
     $$R(b_i) = \bigvee_{(p_j, e_j) \in E_{\text{in}}(b_i)} e_j, \qquad \sigma_{\text{in}}(b_i) = \text{Merge}\bigl([(e_j, \sigma_{\text{out}}(p_j))]\bigr)$$
2. **$\phi$-Node Resolution**:
   For each $\phi$-node $v = \phi[(p_1, v_1), \dots, (p_r, v_r)]$ in $b_i$, restricted to already-evaluated predecessors, its symbolic value is folded over the incoming edge predicates $e_{p_j \to b_i}$:
   $$[\![ v ]\!] = \text{If}(e_{p_r \to b_i}, [\![ v_r ]\!], \dots \text{If}(e_{p_2 \to b_i}, [\![ v_2 ]\!], [\![ v_1 ]\!])\dots)$$
3. **Node Evaluation & Outgoing Edges**:
   Every node in $b_i$ is evaluated once in program order. Outgoing edges are guarded by the non-terminated control-flow mask:
   $$R_{\text{cf}}(b_i) = R(b_i) \land \neg \sigma_{\text{out}}(b_i).\tau$$
   For a conditional branch with condition $c$:
   - Forward edge to `branch_target` ($> b_i$): $R_{\text{cf}}(b_i) \land c$.
   - Forward edge to `fallthrough` ($> b_i$): $R_{\text{cf}}(b_i) \land \neg c$.
4. **Exit State Aggregation**:
   All exit blocks (`is_end`, `is_discard`, or no successors) are merged into the final shader state $\sigma_{\text{final}}$.

### Complexity

Each block and each SSA node is visited **once** per unroll pass, yielding $\mathcal{O}(|B| + |N|)$ symbolic construction time and linear DAG expression sharing inside Z3's hash-consed AST.

---

## 7. Bounded Loop Unrolling & `BOUNDED_UNSAT` Soundness

When a CFG contains a backedge $u \to h$ with $h \le u$ (or a block lists a predecessor $p \ge b$):

1. `SymbolicSsaInterpreter` sets `has_loops = True`.
2. On the initial pass (`max_unroll = 1`), forward edges through the loop body execute **1 iteration** and backedges are recorded in `backedges = [(src, hdr, bcond)]`.
3. When configured with `--max-unroll K` ($K > 1$), as long as `backedges` is non-empty (i.e., at least one backedge condition is not syntactically `False`) and `unroll_iter < K`:
   - Active backedges `(src, hdr, bcond)` are appended to `edge_conds[hdr]`.
   - The loop slice `[b for b in blocks if b.id >= min_hdr]` is re-evaluated with `allow_backedges=True`, updating header $\phi$-nodes and accumulating loop-carried state into `blk_out_state`.
   - `max_unroll_depth` records the actual number of iterations unrolled.
4. **Verdict Semantics**:
   - If every output miter is `UNSAT` and `has_loops == True`, the verifier **never** claims full `UNSAT`; it reports **`BOUNDED_UNSAT`** alongside `max_unroll_depth`.
   - **Soundness**: A `BOUNDED_UNSAT` verdict is mathematically sound for all executions whose loop trip counts are $\le K$ (or whose loop bodies are dead/infeasible after $K$ iterations). A `SAT` verdict on a loop-bearing shader is a genuine counterexample on the $K$-bounded execution trace.

---

## 8. Memory Model: Disjoint 4 GB Regions, Additive Base Recovery & Shared-Memory Unification

### Disjoint 4 GB Virtual Address Partitioning

To prevent spurious cross-buffer aliasing in Z3's flat memory space (e.g., an SSBO write colliding with a UBO or push-constant read), `SymbolicShaderEnv` assigns every canonical buffer key $k$ a unique, disjoint 4 GB virtual address region:

$$\text{Base}(k) = r_k \cdot 2^{32}, \qquad r_k \in \{1, 2, 3, \dots\}$$

Thus `ssbo:0:0` occupies `[0x0000_0001_0000_0000, 0x0000_0001_FFFF_FFFF]`, `ssbo:0:1` occupies `[0x0000_0002_0000_0000, 0x0000_0002_FFFF_FFFF]`, etc.

### Address Resolution (`resolve_addr`)

Given a 64-bit Z3 address expression $a$, `resolve_addr(a, hint_key)` recovers `(buf_key, const_offset, off32)` via a 4-tier cascade:

1. **Constant Folding**: If $a$ simplifies to a concrete 64-bit integer $c$, extract region `(c >> 32) << 32` and offset `c & 0xFFFFFFFF`.
2. **High-32-Bit Extraction**: If `Extract(63, 32, a)` simplifies to a registered region $r_k$, recover `buf_key = base_to_buf[r_k]` and symbolic `off32 = simplify(Extract(31, 0, a))`.
3. **Additive-Chain AST Walk (`_find_base_region_in_add`)**: Walks nested `Z3_OP_BADD` chains (e.g., `base + SignExt(32, idx * stride) + field_off`) up to 64 AST nodes to recover a unique registered region base even when signed offsets cause carry/borrow ambiguity in bit 32.
4. **Hint / Fallback**: Uses `hint_key` (`aux_str`) if present; otherwise routes to `fallback_mem` (never silently dropped).

### `wls_shared` $\leftrightarrow$ `workgroup:<id>` Unification (v25.1)

In SPIR-V compute shaders, shared memory variables (`StorageClassWorkgroup`) are lowered by `direct_ssa` to `LoadSidebandU64(imm=7, aux_str="workgroup:<var_id>")`. On the Valhall side, Mali lowers all workgroup-local storage accesses to hardware WLS (`LoadSidebandU64(imm=7, aux_str="")`, keyed `"wls_shared"`).

Prior to v25.1, `"workgroup:<id>"` and `"wls_shared"` were allocated **different** 4 GB regions, and final output checks skipped `"wls_shared"` while **checking** `"workgroup:<id>"` — causing spurious `SAT` counterexamples on shared-memory compute encoders like `etc2_bc6.spv`.

In v25.1:
- If `direct_ssa` references a unique `workgroup:<id>` key, `SymbolicShaderEnv` aliases `buf_base_map["wls_shared"] = buf_base_map[unique_wg_key]` and routes `get_buf_mem("wls_shared")` to the same Z3 array.
- Both `tls_private`, `wls_shared`, and `workgroup:*` are treated as thread/workgroup-local scratch at final external-output comparison: their writes are still fully tracked in `buf_mems` and `const_stores` during execution (so any load from shared memory feeding an SSBO/image output is verified), while dead post-exit scratch contents are excluded from external output checks.

---

## 9. Compute Execution-Environment Preconditions (`LocalSize`)

Under the Vulkan / SPIR-V execution model (§1.9, §3.6), a compute shader with `OpExecutionMode LocalSize X Y Z` guarantees for every invocation:

$$0 \le_u \text{LocalInvocationId}.x < X, \qquad 0 \le_u \text{LocalInvocationId}.y < Y, \qquad 0 \le_u \text{LocalInvocationId}.z < Z$$

$$\text{GlobalInvocationId}.c = \text{WorkgroupId}.c \cdot \text{LocalSize}.c + \text{LocalInvocationId}.c \qquad (c \in \{x, y, z\})$$

$$\text{LocalInvocationIndex} = \text{LocalInvocationId}.z \cdot (X \cdot Y) + \text{LocalInvocationId}.y \cdot X + \text{LocalInvocationId}.x$$

Without these preconditions, Z3 is free to pick out-of-range invocation IDs such as `bi_27_0 = 0xC0000000` (as observed in v25 on `etc2_astc_enc.spv` and `etc2_bc6.spv`), breaking Mali's strength reductions (e.g., eliding `& (X-1)` masks or packing `(x, y, z)` into 16-bit lanes).

In v25.1:
- `equiv_bundle.cpp` exports `"local_size": [wg_x, wg_y, wg_z]` inside `mbs2_context`.
- `SymbolicShaderEnv.build_preconditions()` generates the exact Z3 `ULT` and identity constraints above for all active `bi_27_*`, `bi_28_*`, `bi_29` inputs, and `_solve_miter_eq` asserts them before checking `query`.

---

## 10. Fragment Discard vs. Demote-to-Helper-Invocation Semantics

SPIR-V 1.6 distinguishes two fundamentally different fragment discard instructions:

- **`OpKill` / `OpTerminateInvocation`** (`SsaOp::Kill` with `aux_bool2 = false`): Immediately **terminates** the invocation. Neither subsequent stores nor subsequent control-flow transitions execute.
- **`OpDemoteToHelperInvocation`** (`SsaOp::Kill` with `aux_bool2 = true`, tagged by `spirv2ssa.cpp` in v25.1): Demotes the invocation to a **helper invocation**. Subsequent observable side-effects (color outputs, depth outputs, SSBO/image stores, atomics) are **masked off**, while control flow, ALU instructions, subgroup operations, and screen-space derivatives (`OpDPdx`/`OpDPdy`) **continue executing**.

`BlockEffectState` models both predicates cleanly:

$$\text{active\\_store\\_mask}(b) = R(b) \land \neg(\tau \lor \delta)$$
$$R_{\text{cf}}(b) = R(b) \land \neg \tau$$

Every observable store (`PsbStore`, `StoreTexel`, `StoreFragColor`, `StoreBuiltinOutput`, `StoreVertVarying`, `Atomic*`) is predicated by $\text{active\\_store\\_mask}(b)$, while outgoing CFG edges are predicated by $R_{\text{cf}}(b)$. Fragment discard equivalence compares $\text{discarded} = \tau \lor \delta$.

---

## 11. Atomic Read-Modify-Write State & Ordered Event Logs

Prior to v25.1, `AtomicRmw`, `AtomicExchange`, and `AtomicCmpXchg` returned an uninterpreted function value without modifying `buf_mems` or recording any observable effect — causing atomic-only compute shaders (e.g., `comp_unique_02_d7242030.spv`) to terminate with `[SKIP] outputs=0`.

In v25.1, every atomic instruction performs two synchronized state updates under $m = \text{active\\_store\\_mask}(b)$:

1. **Exact Single-Invocation RMW Memory Transition**:
   Reads $v_{\text{old}} = \text{Select}(M[k], \text{off})$, computes $v_{\text{new}}$:
   - `AtomicExchange`: $v_{\text{new}} = \text{arg}$
   - `AtomicCmpXchg`: $v_{\text{new}} = \text{If}(v_{\text{old}} = \text{cmp}, \text{arg}, v_{\text{old}})$
   - `aadd` / `asub` / `aand` / `aor` / `axor`: $v_{\text{old}} \oplus \text{arg}$
   - `asmin` / `aumin` / `asmax` / `aumax`: signed/unsigned min/max
   - Float atomics (`afadd`, `afmin`, `afmax`): `_hw_uf("atomic_post_<tag>", BV32, BV32, BV32)(v_old, arg)`
   And updates $M[k] \leftarrow \text{If}(m, \text{Store}(M[k], \text{off}, v_{\text{new}}), M[k])$, returning $(v_{\text{old}}, \text{tok})$.
2. **Ordered Per-Buffer Atomic Event Log**:
   Appends $(m, \text{op\\_tag}, \text{off\\_or\\_addr}, \text{arg}, \text{cmp})$ to `st.atomic_events[k]`. At verification time, for every non-scratch buffer $k$, the verifier checks event count equality and per-event `(active, addr, arg)` equivalence — guaranteeing that atomic-only compute shaders are rigorously checked and never `SKIP`.

---

## 12. Storage-Image Write Normalization (`PsbStore` vs. `StoreTexel`)

`src/spirv2ssa/spirv2ssa_mem.cpp::lower_image_write` lowers `OpImageWrite` into per-channel `PsbStore` nodes on `aux_str = "img:<set>:<binding>"` with packed coordinate address `Pack64(x, y) + c * 4`. Conversely, `valhall_pass1` lowers Valhall `ST_TEX` / image stores into `StoreTexel` with `(coord_vec2, texel_vec4)`.

In v25.1, `SymbolicSsaInterpreter._eval_node` intercepts `PsbStore` whenever `aux_str.startswith("img:")`, extracts the channel index $c \in \{0, 1, 2, 3\}$ and base coordinate `xy_addr = Concat(y, x)`, and writes directly into `st.image_mems[aux_str][c]` under `active_store_mask`. Both pipelines now populate the exact same 4-channel `Array(BV64, BV32)` image state.

---

## 13. Vertex Viewport Transform Precondition & Symbolic Epilogue Roadmap

ARM Mali's vertex shader compiler folds the fixed-function Vulkan viewport transform and perspective division directly into the vertex position shader via the hidden uniform `gl_mali_ViewportTransform`:

$$\text{win}.xyz = \text{FMA}\!\left(\frac{\text{clip}.xyz}{\text{clip}.w},\; \text{vp\\_scale}.xyz,\; \text{vp\\_bias}.xyz\right)$$

Currently, `shader_equiv_verifier.py` handles this via an **identity viewport precondition**:
- `vp_scale = (1.0f, 1.0f, 1.0f, 1.0f)`, `vp_bias = (0.0f, 0.0f, 0.0f, 0.0f)`.
- `clip.w == 1.0f` (`0x3F800000`), and `clip.xyz` are neither NaN nor `-0.0f` (ensuring `FMA(x, 1.0f, +0.0f) == x` bit-for-bit under IEEE-754 RNE).

As tracked in `TODO.md` §1.1 (`TODO(leegao@)`), lifting the symbolic viewport epilogue onto `direct_ssa` will remove this restriction and verify vertex shaders over arbitrary viewports and non-unit `clip.w`.

---

## 14. Theory Stratification & Layer-2 Pluggable FP Proof-Obligation Discharger

Exact IEEE-754 bit-blasting in Z3 (`QF_FP`) is decidable but exponentially expensive on deep floating-point DAGs (e.g., BC6/BC7/ASTC encoders and PBR fragment shaders). The verifier therefore uses a **two-layer stratified architecture**:

### Layer 1: 3-Tier Theory Stratification (`exact` $\rightarrow$ `real` $\rightarrow$ `uf`)
1. **Tier 1 (`fp_mode="exact"`)**: Every output miter is first checked in exact IEEE-754 `QF_BV`/`QF_FP` (`timeout_ms`, `rlimit=2,000,000`) with bit-exact identities (`FAbs`, `FNeg`, `FMA(x, 1.0, +0.0)`). Any `SAT` counterexample is reported immediately.
2. **Layer-2 Obligation Dispatch**: If an output returns `UNKNOWN`, Layer 1 submits an `FpObligation` to `FpProofDischargerRegistry`.
3. **Tier 2 (`fp_mode="real"`, v25.2)**: If exact mode times out, the verifier retries in **exact real arithmetic** (`RealSort()` bridged to `BV32` via cancelling `bv2real`/`real2bv` functions with exact binary32 rational constant decoding). This verifies **lowering equivalence** over $\mathbb{R}$ (ignoring IEEE-754 rounding / NaN / `-0.0` differences) with baked-in algebraic and transcendental reductions:
   - `Fma(a, b, c) = a * b + c` (unifies Mali FMA contraction with `FAdd(FMul(a, b), c)`).
   - `InverseSqrt(x) = 1 / real\_sqrt(x)`, `Exp(x) = real\_exp2(x * log2(e))`, `Log(x) = real\_log2(x) * ln(2)`.
   - `Pow(x, y)` reduces concrete exponents (`0 -> 1`, `1 -> x`, `2 -> x*x`, `0.5 -> real\_sqrt(x)`, `-0.5 -> 1/real\_sqrt(x)`, `-1 -> 1/x`) and general `Pow(x, y) = real\_exp2(y * real\_log2(x))`.
   - Exact base points: `Sqrt(0)=0`, `Sqrt(1)=1`, `Exp2(0)=1`, `Exp2(1)=2`, `Log2(1)=0`, `Log2(2)=1`, `Sin(0)=0`, `Cos(0)=1`.
   - `real UNSAT` proves algebraic lowering equivalence over $\mathbb{R}$; `real SAT` during automatic fallback is logged as diagnostic evidence without emitting a false exact `SAT`.
4. **Tier 3 (`fp_mode="uf"`)**: If `real` also cannot prove equivalence, Layer 1 runs the **sound UF-FP fallback**, replacing FP ops with uninterpreted functions (`fp_FAdd`, `fp_FMul`, `fp_Fma`, etc.). `UF UNSAT` implies exact `UNSAT`; `UF SAT` conservatively reports **`UNKNOWN`**.

### Layer 2: Pluggable `FpProofDischargerRegistry` (`tools/verify/fp_proof_discharger.py`)
Defines the `FpProofBackend` abstract interface (`name`, `is_available()`, `discharge(obligation, timeout_ms)`) and registers three ordered backends:
1. **`Z3FpExactBackend`**: Fast local `QF_FP` check (`rlimit=500,000`) for bit-exact (`ulp_bound == 0`) algebraic/identity lemmas.
2. **`OrToolsCpSatBackend`** *(Implemented in v25.2)*: Extracts variable interval bounds `[lo, hi]` from Z3 preconditions (`ULT`/`ULE`/`UGT`/`UGE`) and translates bounded `BitVec`/`Bool` expressions (`BVAdd`, `BVSub`, `BVMul`, power-of-two `Band`, constant shifts, `ZeroExt`, `Extract`, `If`, comparisons) into `ortools.sat.python.cp_model.CpModel`. Supports both exact equality (`lhs != rhs`) and approximate ULP-tolerance checks (`|lhs - rhs| > ulp_bound`), returning `DISCHARGED` on `INFEASIBLE` and `COUNTEREXAMPLE` on `FEASIBLE`/`OPTIMAL`.
3. **`ExternalSmtProverBackend`** (`TODO(leegao@)`): Subprocess adapter hook for Bitwuzla, CVC5, or Gappa on `obligation.to_smt2()`.

---

## 15. Per-Output Miter Decomposition, Fail-Closed Asymmetry & Counterexample Localization

Rather than conjoining all outputs into a single monolithic SMT query, `verify_equivalence_bundle` decomposes the shader into **independent per-output miter checks**:

1. `discarded` (Fragment or non-false discard)
2. `frag_color[rt].{r,g,b,a}` (guarded by `~discarded`)
3. `builtin_out[bi]` / `builtin_out[bi][c]` (guarded by `~discarded` + viewport precondition if applicable)
4. `vert_var[loc][0..3]`
5. Constant-offset buffer stores: `<bkey>+0x<off>.written` and `<bkey>+0x<off>.value` (guarded by `dw || vw`)
6. Dynamic-offset buffer stores: `<bkey>[probe_off]` via universally quantified probe variable `probe_<bkey>_off`
7. Storage-image writes: `img:<ikey>[xy].{r,g,b,a}` via probe variable `probe_<ikey>_xy`
8. Ordered atomic event logs: `atomic:<bkey>.count`, `atomic:<bkey>[i].{active,addr,arg}`
9. Unresolved-address fallback memory: `fallback_mem[probe_addr]`

### Fail-Closed Asymmetric Output Handling (v25.1)
If a `builtin_out[bi]` is emitted by only one side (`do is None` or `vo is None`), the verifier **never silently skips it**. It records the asymmetry in `res.asymmetric_outputs` and `res.notes`, and emits a fail-closed check `builtin_out[bi].present` (`BoolVal(do is not None) == BoolVal(vo is not None)` guarded by `~discarded`).

### Counterexample Localization (`collect_pcs_for_nodes`)
When a miter returns `SAT`, `v_eval.collect_pcs_for_nodes(v_st.output_pcs[pc_key])` runs a bounded reverse BFS (up to 2,048 SSA nodes) from the diverging Valhall sink node IDs, collects all reachable `orig_pc` values, and resolves the last 16 PCs against `mbs2_context["pc_disasm"]` to pinpoint the exact Valhall instructions responsible for the divergence.

---

## 16. SMT-LIB2 Model Export & Reproducibility

Passing `--dump-smt2 <path>` (or `dump_smt2_path` in Python) exports every non-trivial miter query into a standalone, solver-agnostic SMT-LIB2 file wrapped in `(push 1) ... (check-sat) (pop 1)` scopes with header metadata (shader name, stage, FP mode, SSA sizes, `LocalSize`, `MaxUnroll`, and precondition count). Every sample in `samples/` has its corresponding v25.2 model archived in `samples/models/<name>.smt2` (indexed in `samples/models/README.md`) so any counterexample or timeout can be replayed directly with `z3`, `bitwuzla`, or `cvc5`.

---

## 17. Empirical Evaluation on the 25-Sample Corpus

On the expanded 25-sample stress corpus (`--timeout-ms 5000`, `max_unroll=1`, Checkpoint v25.2):
- **`UNSAT`**: **3** (up from 2 in v25.1): `frag_rep_minimal.spv` (3.6 ms), `comp_rep_minimal.spv` (6.0 ms — flipped from `SAT` after the 64-bit `BufferAddr` high-word fix), and `frag_stress_discard_f814d888.spv` (13.3 ms).
- **`BOUNDED_UNSAT`**: **1**: `etc2_compress.spv` (2,192.9 ms).
- **`SAT`**: **7** (down from 8 in v25.1) — all 7 expose concrete, actionable lowering/ABI gaps (`builtin_out[1]` `gl_PointSize` on 3 vertex shaders after `ClipDistance`/`CullDistance` were eliminated; `atomic:ssbo:0:1.count` on `comp_unique_02_d7242030.spv` after all 159 atomics resolved from `fallback_mem` to `ssbo:0:1`; `fallback_mem` on 2 image/texel-pointer compute shaders; Mali `BLEND` `.a = 1.0f` fill on `frag_unique_01_fca361fe.spv`).
- **`UNKNOWN`**: **14** — large FP/texture/BCn encoders hitting exact `QF_FP` `rlimit=2,000,000` (with Tier-2 `fp_mode="real"` now separating genuine algebraic divergences on 3 fragment shaders from integer/loop-heavy `UNKNOWN` on BCn encoders).
- **`SKIP` / `ERROR`**: **0 / 0**.

---

## 18. Soundness Boundaries, Known Limitations & Future Work

1. **Single-Invocation Sequential Model**: Cross-thread data races and `OpControlBarrier` workgroup memory visibility are currently modeled sequentially; adding barrier-epoch array havoc is tracked in `TODO.md` §3.
2. **Bounded Loop Unrolling**: `BOUNDED_UNSAT` guarantees equivalence only up to `max_unroll` iterations. Inductive loop-invariant / bisimulation proofs remain future work (`TODO.md` §3).
3. **Viewport Precondition**: Vertex `gl_Position` equivalence currently assumes identity viewport and `clip.w == 1.0f` until symbolic `gl_mali_ViewportTransform` epilogue lifting is implemented (`TODO.md` §1.1).
4. **Uninterpreted Transcendentals & Derivatives**: `Sin`, `Cos`, `Exp2`, `Log2`, `InverseSqrt`, `DPdx`, `DPdy`, `TexSample`, and `Subgroup*` are modeled as uninterpreted functions — sound for structural equivalence when both sides emit identical operands, but unable to prove algebraic/ULP approximations without Layer-2 external provers (`TODO.md` §1.2–1.3).

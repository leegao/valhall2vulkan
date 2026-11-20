# Valhall SSA Dataflow & Abstract Interpretation Framework (`src/dataflow/`)
## Design Document & Implementation Guide

**Version:** Checkpoint v22  
**Location:** `src/dataflow/` (operating over `src/ssa/valhall_ssa.{h,cpp}`)  
**Primary Consumers:** `tools/spirv2valhall/isa_enricher.cpp`, Pilot Shader FAU chasing, future SSA optimizations & formal verification passes

---

## Table of Contents

1. [Executive Overview & Architectural Philosophy](#1-executive-overview--architectural-philosophy)
2. [Module Map & Dependency Graph](#2-module-map--dependency-graph)
3. [Lattice Theory & Compile-Time Contract (`ValidateLatticeDomain`)](#3-lattice-theory--compile-time-contract-validatelatticedomain)
4. [Prepared Domain Combinators (`src/dataflow/domains.h`)](#4-prepared-domain-combinators-srcdataflowdomainsh)
5. [Symbolic Memory & Disjoint Region Model (`src/dataflow/memory_domain.h`)](#5-symbolic-memory--disjoint-region-model-srcdataflowmemory_domainh)
6. [SSA Abstract State (`src/dataflow/abstract_domain.h`)](#6-ssa-abstract-state-srcdataflowabstract_domainh)
7. [Directional Chaotic Worklist Engine (`src/dataflow/worklist.{h,cpp}`)](#7-directional-chaotic-worklist-engine-srcdataflowworklisthcpp)
8. [Backwards SSA Liveness Analysis (`src/dataflow/liveness.{h,cpp}`)](#8-backwards-ssa-liveness-analysis-srcdataflowlivenesshcpp)
9. [SSA Transfer Visitor & Extensional Equalities (`src/dataflow/transfer.{h,cpp}`)](#9-ssa-transfer-visitor--extensional-equalities-srcdataflowtransferhcpp)
10. [Implementation Guide: What You Must Implement vs. What You Get for Free](#10-implementation-guide-what-you-must-implement-vs-what-you-get-for-free)
11. [Worked Case Study 1: Pilot Shader SRT Pointer Chasing (`pointer_analysis.cpp`)](#11-worked-case-study-1-pilot-shader-srt-pointer-chasing-pointer_analysiscpp)
12. [Worked Case Study 2: Diamond CFG, `FiniteSetDomain` & Non-Singleton Dereference](#12-worked-case-study-2-diamond-cfg-finitesetdomain--non-singleton-dereference)
13. [Worked Case Study 3: 64-Bit Split/Pack Reconstruction & VA Masking](#13-worked-case-study-3-64-bit-splitpack-reconstruction--va-masking)
14. [Worked Case Study 4: Writing a Brand-New Analysis (Uniformity / Divergence)](#14-worked-case-study-4-writing-a-brand-new-analysis-uniformity--divergence)
15. [Performance, Memory Scaling & Soundness Guarantees](#15-performance-memory-scaling--soundness-guarantees)
16. [Design Q&A: Deep-Dive Answers to Architectural Review Questions](#16-design-qa-deep-dive-answers-to-architectural-review-questions)

---

## 1. Executive Overview & Architectural Philosophy

### 1.1 Why SSA-Level Dataflow (and Not Raw ISA-Level)?

Arm's Valhall GPU ISA encodes 200+ mnemonics with highly irregular operand layouts (for example, `AND` places its second operand in `src2` while `AND_IMM` uses `imm`; 64-bit pointers are split across adjacent 32-bit GPR pairs `rN:rN+1`; conditional moves, FAU uniforms, and register-kill modifiers `^` interact in subtle ways). Writing dataflow analyses directly on raw decoded ISA instructions forces every analysis to re-implement:
1. Operand-slot decoding and 64-bit register-pair assembly/splitting.
2. Mutable 64-register file snapshots per basic block ($O(|B| \cdot 64)$ state copying).
3. Register clobbering vs. live-range kill bookkeeping.

By lifting Valhall ISA into **`ValhallSsaModule`** (`src/ssa/valhall_ssa.{h,cpp}`) first—and stamping every lowered `SsaNode` with `(orig_inst_idx, orig_pc)`—the dataflow framework gains four decisive advantages:
1. **Single Static Assignment ($O(|V|)$ Memory):** Every `SsaId` has exactly one defining `SsaNode` in the entire module. Rather than replicating register arrays per basic block ($O(|B| \cdot |V|)$ memory, which OOMed on 69 KB shaders like `astc_enc.raw.spv`), `run_ssa_dataflow` stores a **single shared `VectorDomain<DomainT>`** of length $|V|$ and tracks only the small mutable side-channel memory maps (`fau_mem`, `sym_mem`) per block.
2. **Normalized Semantic Operations (`SsaOpKind`):** All 200+ Valhall mnemonics collapse into 28 clean semantic `SsaOpKind` groups (`Const`, `Copy`, `Phi`, `Select`, `IntAdd`, `BitAnd`, `Pack64`, `Split64`, `PsbLoad`, `PsbStore`, etc.).
3. **Explicit Control-Flow Merges (`Phi`):** Divergent definitions are merged explicitly at block entry via `SsaOp::Phi`, so every dominating `SsaId` is automatically path-consistent.
4. **Clean Separation of Concerns:** `src/dataflow/` has **zero** dependency on ISA decoding, CFG construction, or Pass 1 lowering. Consumers that start from raw ISA (such as `tools/spirv2valhall/isa_enricher.cpp`) explicitly lower ISA $\to$ CFG $\to$ SSA (`tolerant_ssa = true`), call the pure SSA passes, and project findings back via `orig_inst_idx`.

### 1.2 Core Design Principles

1. **Composable Lattice Combinators (`domains.h`):** Never hand-write map/vector/tuple join-meet-equals loops inside an analysis. Compose `VectorDomain`, `MapDomain`, `TupleDomain`, `BitsetDomain`, and `FiniteSetDomain<T, K>` like algebraic building blocks.
2. **Fail-Fast Compile-Time Contracts (`ValidateLatticeDomain`):** Missing or mistyped lattice methods fail immediately with readable `static_assert` diagnostics instead of 200-line template backtraces.
3. **Domain-Independent Directional Worklist (`worklist.{h,cpp}`):** The chaotic iteration engine lives in `worklist.cpp` (compiled once, not templated per domain) and drives both `DataflowDirection::Forward` and `DataflowDirection::Backward` analyses.
4. **Extensional Equalities by Default (`SsaDomainTransferVisitor`):** Every domain automatically inherits copy propagation, `Phi`/`Select` joins, and algebraic identities (`x+0 == x`, `x&~0 == x`, `x^x == 0`, etc.) simply by opting into `try_as_const_u64` / `make_const_u64`.

---

## 2. Module Map & Dependency Graph

```text
                        ┌────────────────────────────────────┐
                        │     src/ssa/valhall_ssa.{h,cpp}    │
                        │  ValhallSsaModule, SsaBlock,       │
                        │  SsaNode (orig_inst_idx, orig_pc)  │
                        └─────────────────┬──────────────────┘
                                          │
          ┌───────────────────────────────┼───────────────────────────────┐
          ▼                               ▼                               ▼
┌─────────────────────┐      ┌─────────────────────────┐      ┌───────────────────────┐
│     domains.h       │      │     worklist.{h,cpp}    │      │   memory_domain.h     │
│ • ValidateLattice   │      │ • DataflowDirection     │      │ • MemoryRegionKind    │
│ • domain_equals     │      │ • run_cfg_worklist()    │      │ • SymbolicMemoryLoc   │
│ • VectorDomain      │      │   (Forward / Backward)  │      │ • may_alias()         │
│ • MapDomain         │      └────────────┬────────────┘      └───────────┬───────────┘
│ • TupleDomain       │                   │                               │
│ • BitsetDomain      │                   │                               │
│ • FiniteSetDomain   │                   │                               │
└─────────┬───────────┘                   │                               │
          │                               │                               │
          ├───────────────────────────────┼───────────────────────────────┤
          ▼                               ▼                               ▼
┌─────────────────────┐      ┌─────────────────────────┐      ┌───────────────────────┐
│  liveness.{h,cpp}   │      │   abstract_domain.h     │      │  tracked_val.{h,cpp}  │
│ • BitsetDomain      │      │ • AbstractState<D>      │      │ • TrackedVal (19 kind)│
│ • run_ssa_liveness  │      │   = VectorDomain<D>     │      │ • TrackedValDomain    │
│ • Phi-edge uses     │      │   + MapDomain<u32, D>   │      │ • TrackedValSetDomain │
│ • killed_after_node │      │   + MapDomain<SymLoc,D> │      │   (FiniteSet<..., 4>) │
└─────────┬───────────┘      └────────────┬────────────┘      └───────────┬───────────┘
          │                               │                               │
          └───────────────────────────────┼───────────────────────────────┘
                                          ▼
                        ┌────────────────────────────────────┐
                        │       transfer.{h,cpp}             │
                        │ • SsaOpKind + classify_ssa_op()    │
                        │ • SsaTransferVisitorBase           │
                        │ • SsaDomainTransferVisitor<D>      │
                        │ • run_ssa_dataflow<D>(dir, ...)    │
                        └─────────────────┬──────────────────┘
                                          ▼
                        ┌────────────────────────────────────┐
                        │    pointer_analysis.{h,cpp}        │
                        │ • PointerAnalysisEnv               │
                        │ • SsaPointerTransferVisitor        │
                        │ • run_ssa_pointer_analysis()       │
                        └────────────────────────────────────┘
```

| File | Responsibility | Compiled In |
|---|---|---|
| `src/dataflow/domains.h` | `ValidateLatticeDomain`, `domain_equals`, `VectorDomain`, `MapDomain`, `TupleDomain`, `BitsetDomain`, `FiniteSetDomain<T, MaxK>` | Header-only |
| `src/dataflow/memory_domain.h` | `MemoryRegionKind`, `SymbolicMemoryLoc`, `may_alias()`, `Hash`, `format()` | Header-only |
| `src/dataflow/abstract_domain.h` | `AbstractState<DomainT>` composing `VectorDomain` + `MapDomain`s | Header-only |
| `src/dataflow/worklist.{h,cpp}` | Domain-independent directional CFG worklist solver (`Forward`, `Backward`) | `libvalhall2spirv.a` |
| `src/dataflow/liveness.{h,cpp}` | Backwards SSA liveness using `BitsetDomain` + `run_cfg_worklist(Backward)` | `libvalhall2spirv.a` |
| `src/dataflow/transfer.{h,cpp}` | `SsaOpKind` classifier, `SsaTransferVisitorBase`, `SsaDomainTransferVisitor<D>`, `run_ssa_dataflow<D>` | `libvalhall2spirv.a` |
| `src/dataflow/tracked_val.{h,cpp}` | Reusable 19-kind `TrackedVal` value lattice, `TrackedValDomain`, `TrackedValSetDomain` | `libvalhall2spirv.a` |
| `src/dataflow/pointer_analysis.{h,cpp}` | Pure SSA pointer/ABI/Pilot FAU analysis over `TrackedValSetDomain` | `libvalhall2spirv.a` |

---

## 3. Lattice Theory & Compile-Time Contract (`ValidateLatticeDomain`)

### 3.1 Mathematical Requirements

A bounded lattice is a tuple $(\mathcal{L}, \sqsubseteq, \sqcup, \sqcap, \bot, \top)$ satisfying for all $a, b, c \in \mathcal{L}$:
- **Partial Order ($\sqsubseteq$):** Reflexive ($a \sqsubseteq a$), Antisymmetric ($a \sqsubseteq b \land b \sqsubseteq a \implies a = b$), Transitive ($a \sqsubseteq b \land b \sqsubseteq c \implies a \sqsubseteq c$).
- **Bounds:** $\bot \sqsubseteq a \sqsubseteq \top$.
- **Least Upper Bound ($\sqcup$, `join`):** Commutative, Associative, Idempotent ($a \sqcup a = a$), $\bot \sqcup a = a$, $\top \sqcup a = \top$, and $a \sqsubseteq b \iff a \sqcup b = b$.
- **Greatest Lower Bound ($\sqcap$, `meet`):** Dual of `join` ($\top \sqcap a = a$, $\bot \sqcap a = \bot$, $a \sqsubseteq b \iff a \sqcap b = a$).
- **Ascending Chain Condition (ACC):** Every strictly ascending chain $a_0 \sqsubset a_1 \sqsubset \dots \sqsubset a_h$ has finite length $h < \infty$. Combined with monotone transfer functions ($x \sqsubseteq y \implies f(x) \sqsubseteq f(y)$), Tarski's fixed-point theorem guarantees chaotic worklist iteration terminates in finite steps at the least fixpoint.

### 3.2 C++17 Interface Contract (`ValidateLatticeDomain<DomainT>`)

Every domain struct `D` passed to `AbstractState<D>`, `VectorDomain<D>`, `MapDomain<K, D>`, `TupleDomain<...>`, `FiniteSetDomain<D, K>`, or `SsaDomainTransferVisitor<D>` is checked at instantiation by `ValidateLatticeDomain<D>`:

```cpp
struct MyDomain {
    // REQUIRED (enforced with clear static_assert messages):
    using Value = ...;
    static Value bottom();
    static Value top();
    static Value join(const Value& a, const Value& b);
    static Value meet(const Value& a, const Value& b);
    static bool  leq(const Value& a, const Value& b);

    // OPTIONAL (1): Equality
    // If omitted, `domain_equals<MyDomain>(a, b)` automatically computes:
    //   MyDomain::leq(a, b) && MyDomain::leq(b, a)
    static bool  equals(const Value& a, const Value& b);

    // OPTIONAL (2): Constant-Folding Hooks
    // If provided, `SsaDomainTransferVisitor<MyDomain>` automatically folds
    // constants and discharges algebraic extensional equalities (x+0 == x,
    // x&~0 == x, x|0 == x, x^0 == x, x*1 == x, x-x == 0, x^x == 0, x&0 == 0).
    static bool  try_as_const_u64(const Value& v, uint64_t* out);
    static Value make_const_u64(uint64_t c);
};
```

**Why `domain_equals<D>` matters:** Simple flat or small-enum domains (like `SignDomain` in `tests/test_dataflow.cpp`) do not need to write boilerplate `equals()`, while complex struct domains (like `TrackedValDomain` or `BitsetDomain`) can provide a fast specialized `equals()` that avoids calling `leq()` twice.

---

## 4. Prepared Domain Combinators (`src/dataflow/domains.h`)

### 4.1 `VectorDomain<ElemDomainT>`

Pointwise lattice over a dense 0-indexed `std::vector<ElemValue>`, where out-of-bounds indices implicitly map to `ElemDomainT::bottom()`.

- **Adversarial Reallocation Guard (Power-of-2 Growth):** Calling `set(idx, v)` or `ensure_size(n)` resizes the backing vector to `next_power_of_two(needed)` (`1, 2, 4, 8, 16, ...`) filled with `ElemDomainT::bottom()`. This prevents $O(N^2)$ reallocation blowup when `SsaId`s are populated incrementally.
- **Operations:**
  - `get(idx)`, `set(idx, v)`, `join_at(idx, v)` (returns `bool changed`)
  - `join_into(dst, other)`, `meet_into(dst, other)` (in-place, zero unnecessary copies)
  - `join(a, b)`, `meet(a, b)`, `leq(a, b)`, `equals(a, b)`

### 4.2 `MapDomain<KeyT, ValDomainT>`

Pointwise lattice over a sparse `std::map<KeyT, ValValue>`, where absent keys implicitly map to `ValDomainT::bottom()`.

- **Canonical Sparsity:** `set(k, bottom())` erases `k`; `meet_into` erases keys whose meet drops to `bottom()`. As a result, `equals(a, b)` can check `a.entries.size() == b.entries.size()` in $O(1)$ before comparing entries.
- **Operations:** `get(k)`, `set(k, v)`, `join_at(k, v)`, `clear()`, `join_into`, `meet_into`, `join`, `meet`, `leq`, `equals`.

### 4.3 `TupleDomain<D1, D2, ...>`

Variadic Cartesian product lattice over `std::tuple<typename Domains::Value...>`:

$$\text{join}((a_1, \dots, a_k), (b_1, \dots, b_k)) = (\text{join}_1(a_1, b_1), \dots, \text{join}_k(a_k, b_k))$$

```cpp
// Example: combine sign analysis with a live/taint bitset in 1 line!
using CombinedDomain = TupleDomain<SignDomain, BitsetDomain>;
static_assert(ValidateLatticeDomain<CombinedDomain>::valid, "");
```

All six lattice operations (`bottom`, `top`, `join`, `meet`, `leq`, `equals`) are generated automatically via C++17 fold expressions over `std::index_sequence_for<Domains...>`.

### 4.4 `BitsetDomain`

Compact powerset lattice $(\mathcal{P}(\{0, \dots, N-1\}), \subseteq, \cup, \cap, \emptyset, \mathcal{U})$ backed by `std::vector<uint64_t>` (64 elements per 8-byte word):
- `bottom()` = empty bitset; `top()` = universal bitset (`is_top_val = true`).
- `join(a, b)` = bitwise OR ($\cup$); `meet(a, b)` = bitwise AND ($\cap$); `diff(a, b)` = bitwise AND-NOT ($\setminus$).
- `leq(a, b)` = `(aw & ~bw) == 0` for all words ($\subseteq$).
- Zero heap churn inside fixpoint loops via `clear_all()`, `or_with(other)`, and `or_diff_with(other, minus)`.

### 4.5 `FiniteSetDomain<ElemDomainT, MaxK = 4>`

A bounded finite powerset domain that tracks up to `MaxK` (default 4) distinct concrete values from `ElemDomainT` before widening to `Top`:

```text
                        ⊤_S (|S| > 4 or ⊤ ∈ S)
              /            |               |            \
        {a, b, c, d}   {a, b, c}       {a, b}           ...
              \            |               |            /
               {a}        {b}             {c}           {d}
                  \        |               |           /
                           ⊥_S = ∅
```

**Why `FiniteSetDomain` is a game-changer for shader dataflow:**
- Suppose a shader computes `off = cond ? 16 : 32;` (or a `Phi` merges two buffer addresses `&bufA` and `&bufB`).
- In a flat lattice, `join(16, 32)` immediately collapses to `Top`, losing all precision!
- In `FiniteSetDomain<TrackedValDomain, 4>`, `join({16}, {32}) = {16, 32}` (cardinality 2 $\le 4$). Subsequent arithmetic (`off + 8`) automatically lifts pointwise via `lift_unary` to `{24, 40}`!
- **Free Lift & Lower:**
  - `lift_value(x)` / `singleton(x)`: wraps a single `ElemValue` into `{x}`.
  - `try_as_singleton(&out)`: extracts the single element when `size() == 1` (e.g., to check whether a memory address is uniquely resolved before dereferencing!).
  - `lift_unary(S, fn)` & `lift_binary(A, B, fn)`: project any pointwise transfer function `ElemValue -> ElemValue` (or `-> FiniteSetValue`) across the set, widening to `Top` cleanly if cardinality exceeds `MaxK`.
- **Automatic Constant-Folding Forwarding:** When `ElemDomainT` provides `try_as_const_u64`/`make_const_u64`, `FiniteSetDomain` forwards them for singleton constant sets, so `SsaDomainTransferVisitor<FiniteSetDomain<TrackedValDomain, 4>>` inherits all extensional equalities automatically!

---

## 5. Symbolic Memory & Disjoint Region Model (`src/dataflow/memory_domain.h`)

Raw string keys (like `"SRT[4]+0x00"`) cannot support structured alias analysis or pointer escape queries. `memory_domain.h` replaces them with `SymbolicMemoryLoc`:

```cpp
enum class MemoryRegionKind : uint8_t {
    FauUniform = 0,   // FAU_0 uniform bank (RO in Main shaders; written in Pilot shaders)
    SrtTable,         // Shader Resource Table (SRT_0[table_id])
    Buffer,           // UBO / SSBO buffer descriptor (SRT[table_id]->Desc[entry_id])
    DataSection,      // Embedded read-only EBIN .data / LUT section
    Tls,              // Per-thread scratch / spill stack (thread_local_pointer)
    Wls,              // Workgroup-local shared memory
    Other,
};
```

### Disjointness Soundness Assumptions (`SymbolicMemoryLoc::may_alias`)

| Pair of Locations | `may_alias()` | Rationale |
|---|---|---|
| Different `MemoryRegionKind` (e.g., `Tls` vs. `Buffer`) | `false` | GPU address spaces are hardware-disjoint. |
| `SrtTable(rack=0)` vs. `SrtTable(rack=4)` | `false` | Distinct SRT rack tables occupy disjoint descriptor allocations. |
| `Buffer(rack=4, desc=0)` vs. `Buffer(rack=4, desc=1)` | `false` | Distinct Vulkan descriptor bindings are modeled as non-aliasing by default. |
| Same buffer, known offsets `+0x00` vs. `+0x10` | `false` | Distinct known byte offsets within the same buffer. |
| Same buffer, known `+0x00` vs. unknown `+?` | `true` | Dynamic offset may alias any offset in the same buffer. |

**Why `fau_mem` is tracked separately and easy to analyze:**
- In **Main shaders** (Compute, Vertex, Fragment), `FAU_0` is **strictly read-only (RO)**—it is populated once before dispatch by the Pilot shader and the driver's RLOC/FCST constant uploads. Therefore, `LoadFau(w)` in a Main shader is completely flow-insensitive and never invalidated by `PsbStore`s to `Buffer`/`Tls`/`Wls`.
- In **Pilot shaders**, `FAU_0` is the *output destination* (`FauDstBase` in `r0:r1`), written via `PsbStore(FauDstBase + off, val)` and never aliased by `SRT_0` reads.

---

## 6. SSA Abstract State (`src/dataflow/abstract_domain.h`)

`AbstractState<DomainT>` composes three combinators:

```cpp
template <typename DomainT>
struct AbstractState {
    static_assert(ValidateLatticeDomain<DomainT>::valid, "");
    using SsaEnvDomain = VectorDomain<DomainT>;
    using FauMemDomain = MapDomain<uint32_t, DomainT>;
    using SymMemDomain = MapDomain<SymbolicMemoryLoc, DomainT>;

    typename SsaEnvDomain::Value ssa_vals; // SsaId -> Value (power-of-2 dense vector)
    typename FauMemDomain::Value fau_mem;  // FAU byte_off -> Value (RO in Main, RW in Pilot)
    typename SymMemDomain::Value sym_mem;  // SymbolicMemoryLoc -> Value (disjoint regions)
    ...
};
```

Notice how `join_with()`, `meet_with()`, and `equals()` are now each **3 one-line delegations** to `SsaEnvDomain`, `FauMemDomain`, and `SymMemDomain`—zero duplicated map/vector traversal code!

**Path Consistency of `SsaId`s:** Because every `SsaId` is defined at a single dominating `SsaNode`, all paths reaching a merge block $B$ agree on every `SsaId` defined in a dominator of $B$. Any value that differs across incoming paths is represented by an explicit `Phi` node in `B.phis`, which joins its incoming `SsaId`s explicitly via `on_phi`.

---

## 7. Directional Chaotic Worklist Engine (`src/dataflow/worklist.{h,cpp}`)

The worklist engine is completely decoupled from `DomainT` and compiled once in `src/dataflow/worklist.cpp`:

```cpp
enum class DataflowDirection : uint8_t { Forward = 0, Backward = 1 };

WorklistResult run_cfg_worklist(
    const compiler::ValhallSsaModule& mod,
    DataflowDirection dir,
    const std::function<bool(uint32_t block_id)>& transfer_block,
    uint32_t max_iterations = 64);
```

- **`Forward`:** Seeds worklist with blocks $0, 1, \dots, |B|-1$. Whenever `transfer_block(b)` returns `true`, enqueues `fallthrough_block` and `branch_target_block` (deduplicated via `in_worklist[b]`).
- **`Backward`:** Seeds worklist with blocks $|B|-1, \dots, 0$. Whenever `transfer_block(b)` returns `true`, enqueues all `blk.preds`.
- **Termination Guard:** Caps total block visits at `max_iterations * |B|` and reports `res.converged = worklist.empty()`.

---

## 8. Backwards SSA Liveness Analysis (`src/dataflow/liveness.{h,cpp}`)

Built directly on `BitsetDomain` and `run_cfg_worklist(mod, DataflowDirection::Backward, ...)`:

1. **Precompute per-block `BitsetValue` sets:**
   - `phi_defs[B]` = $\{p \mid p \in B.\text{phis}\}$
   - `body_def[B]` = $\{n \mid n \in B.\text{nodes}\}$
   - `body_ue_use[B]` = upward-exposed uses in `B.nodes` and `B.branch_cond`
2. **Fixpoint Equations (inside `run_cfg_worklist` callback):**
   $$\text{LIVE\_OUT}[B] = \bigcup_{S \in \text{succ}(B)} \Big( (\text{LIVE\_IN}[S] \setminus \text{PHI\_DEFS}[S]) \cup \text{PHI\_USES}(B \to S) \Big)$$
   $$\text{LIVE\_IN}[B] = \text{UE\_USE}[B] \cup (\text{LIVE\_OUT}[B] \setminus \text{BODY\_DEF}[B])$$
3. **Why $\phi$-edge uses matter:** In SSA, a $\phi$-node `%m = Phi((B1, %p1), (B2, %p2))` in block `B3` uses `%p1` on the *edge* `B1 -> B3` and `%p2` on the *edge* `B2 -> B3`. Thus `%p1` is live-out of `B1` (and NOT live-out of `B2`, and NOT live-in of `B3`).
4. **Per-Node Kill Sets (`killed_after_node[n]`):** After fixpoint convergence, a single reverse sweep over `B.nodes` starting from `LIVE_OUT[B] ∪ {branch_cond}` records every `arg` of `n` that is not live after `n`.

---

## 9. SSA Transfer Visitor & Extensional Equalities (`src/dataflow/transfer.{h,cpp}`)

### 9.1 `SsaOpKind` Classification

`classify_ssa_op(tables::SsaOp)` maps all 134 `tables::SsaOp` enum values into 28 semantic `SsaOpKind`s (paired with `is_ssa_op_64bit()` and `classify_ssa_shift_kind()`). `dispatch_ssa_transfer(node, visitor)` switches cleanly on `SsaOpKind` and invokes the matching virtual hook on `SsaTransferVisitorBase`:

| `SsaOpKind` | Grouped `tables::SsaOp`s | Visitor Hook |
|---|---|---|
| `Const` | `ConstBool`, `ConstU32`, `ConstS32`, `ConstF32`, `ConstU64` | `on_const(dst, node)` |
| `Copy` | `Copy`, `Bitcast` | `on_copy(dst, src, node)` |
| `Phi` | `Phi` | `on_phi(dst, node)` |
| `Select` | `Select` | `on_select(dst, cond, t, f, node)` |
| `IntAdd` / `IntSub` / `IntMul` | `IAdd[64]`, `ISub[64]`, `IMul[64]` | `on_iadd` / `on_isub` / `on_imul` |
| `BitAnd` / `BitOr` / `BitXor` / `BitNot` | `BitAnd[64]`, `BitOr[64]`, `BitXor[64]`, `BitNot[64]` | `on_and` / `on_or` / `on_xor` / `on_not` |
| `Shift` | `Shl[64]`, `ShrU[64]`, `ShrS[64]`, `RotL`, `RotR` | `on_shift(dst, lhs, rhs, kind, is64, node)` |
| `Pack64` / `Split64` / `TupleExtract` | `Pack64`, `Split64`, `TupleExtract` | `on_pack64` / `on_split64` / `on_tuple_extract` |
| `Convert` / `Vec` | `UConvert*`, `SConvert*`, `Convert*`, `Vec*`, `Pack/UnpackHalf2x16` | `on_convert` / `on_vec` |
| `LoadFau` / `LoadBuiltin` / `LoadSideband` | `LoadFau`, `LoadBuiltinInput`, `LoadSidebandU32/U64` | `on_load_fau` / `on_load_builtin` / `on_load_sideband` |
| `PsbLoad` / `PsbStore` | `PsbLoad`, `PsbStore` | `on_psb_load` / `on_psb_store` |
| `LoadVertAttr` / `LoadFragVarying` / `TexSample` / `StoreOutput` / `Atomic` | Stage I/O & atomics | `on_load_vert_attr` / `on_load_frag_varying` / `on_tex_sample` / `on_store_output` / `on_atomic` |

### 9.2 Extensional Equalities (`SsaDomainTransferVisitor<DomainT>`)

Every subclass of `SsaDomainTransferVisitor<DomainT>` inherits these sound semantic identities for free:
- **Copy / Alias:** `Copy(x)` and `Bitcast(x)` copy `state.val(x)` directly.
- **Control-Flow Joins:** `Phi` joins all non-self incoming values; `Select(c, x, x)` copies `x` and `Select(c, a, b)` computes `join(a, b)`.
- **Algebraic Identities (when `try_as_const_u64`/`make_const_u64` are present):**
  - Identity elements: `x + 0`, `0 + x`, `x - 0`, `x * 1`, `1 * x`, `x | 0`, `0 | x`, `x ^ 0`, `0 ^ x`, `x & ~0`, `~0 & x`, `x << 0`, `x >> 0` $\to$ `eval(x)`.
  - Idempotence: `x & x`, `x | x` $\to$ `eval(x)`.
  - Annihilators & self-cancellation: `x * 0`, `0 * x`, `x & 0`, `0 & x`, `x - x`, `x ^ x` $\to$ `make_const(0)`.
  - Constant folding: all integer, bitwise, and shift ops fold when both operands are known constants.

---

## 10. Implementation Guide: What You Must Implement vs. What You Get for Free

When building a new SSA dataflow analysis on `src/dataflow/`, follow this checklist:

### What You MUST Implement
1. **Your Value Lattice `MyDomain`:**
   - `using Value = ...;`
   - `bottom()`, `top()`, `join(a, b)`, `meet(a, b)`, `leq(a, b)` (validated at compile time by `ValidateLatticeDomain<MyDomain>`).
   - *(Optional)* `equals(a, b)` if faster than `leq(a, b) && leq(b, a)`.
   - *(Optional)* `try_as_const_u64(v, &out)` and `make_const_u64(c)` if your domain tracks integer constants and wants automatic constant folding + algebraic identities.
2. **Your Transfer Visitor `MyVisitor : public SsaDomainTransferVisitor<MyDomain>`:**
   - Override **only** the `on_*` hooks where your analysis has domain-specific semantics (e.g., `on_load_builtin` and `on_load_fau` for uniformity analysis; `on_iadd` and `on_psb_load` for pointer analysis).
   - For any case you don't handle specially inside an overridden hook, delegate back to `SsaDomainTransferVisitor<MyDomain>::on_*(...)` to preserve constant folding and extensional equalities.
3. **Entry State Seeding & Runner Call:**
   - Construct `AbstractState<MyDomain> entry(mod.nodes.size())`, seed `entry.fau_mem` / `entry.sym_mem` if needed, and call `run_ssa_dataflow<MyDomain>(mod, dir, entry, visitor, liveness_ptr)`.

### What You Do NOT Need to Implement (Inherited for Free!)
- CFG traversal, worklist deduplication, convergence detection, or loop-iteration caps (`worklist.cpp`).
- `O(|V|)` shared SSA environment storage, power-of-2 vector growth, or per-block memory-map joins (`abstract_domain.h`, `transfer.h`).
- `tables::SsaOp` decoding or 134-way opcode switching (`classify_ssa_op`, `dispatch_ssa_transfer`).
- `Copy`, `Bitcast`, `Phi`, `Select`, and algebraic identity rules (`SsaDomainTransferVisitor`).
- Finite-set lifting (`FiniteSetDomain<MyDomain, 4>`), tuple products (`TupleDomain`), dense vectors (`VectorDomain`), sparse maps (`MapDomain`), or bitsets (`BitsetDomain`).
- Backwards SSA liveness or $\phi$-edge kill tracking (`run_ssa_liveness`).

---

## 11. Worked Case Study 1: Pilot Shader SRT Pointer Chasing (`pointer_analysis.cpp`)

Let's trace how `SsaPointerTransferVisitor` chases `_7163_input_rgba8_buffer` in `astc_enc.raw.spv`'s Pilot Shader (`EBIN[1]`):

```text
ISA PC  | Lowered SSA Nodes (ValhallSsaModule)             | TrackedValSet after transfer
--------+--------------------------------------------------+---------------------------------------------
preamble| %1 = LoadSidebandU64(3)  [is_fau_dst_va(3)]      | { FauDstBase(+0x0) }
preamble| %2 = LoadSidebandU64(2)  [is_srt_base_va(2)]     | { SrtBase(+0x0) }
0x0008  | %3 = ConstU64(0x40)                              | { ConstInt(0x40) }
0x0008  | %4 = IAdd64(%2, %3)                              | { SrtBase(+0x40) }
0x0008  | %5 = PsbLoad(%4, elem_bytes=8)                   | { SrtRackPtr(rack=4, +0x0) }  [0x40/16 = 4]
0x0008  | %6 = TupleExtract(%5, 0)                         | { SrtRackPtr(rack=4, +0x0) }
0x0048  | %7 = Split64(%6) -> lo=%8, hi=%9                 | lo={SrtRackPtr(4)}, hi={SrtRackPtr(4).hi}
0x0048  | %10 = BitAnd(%9, ConstU32(0xffffff))             | { SrtRackPtr(4).hi }  (VA mask preserved!)
0x0048  | %11 = Pack64(%8, %10)                            | { SrtRackPtr(rack=4, +0x0) }
0x0050  | %12 = IAdd64(%11, ConstU64(0x8))                 | { SrtRackPtr(rack=4, +0x8) }
0x0050  | %13 = PsbLoad(%12, elem_bytes=8)                 | { BufferAddr(4, 0, "_7163_input_rgba8_buffer") }
0x0050  | %14 = TupleExtract(%13, 0)                       | { BufferAddr(4, 0, "_7163_input_rgba8_buffer") }
0x0080  | %15 = PsbStore(%1, %14, elem_bytes=8)            | fau_mem[0x00] = { BufferAddr(4, 0, "_7163_...") }
```

Every step uses `TrackedValSetDomain::lift_unary` / `lift_binary`, and `PsbStore` records `PilotFauWrite{0x00, 8, BufferAddr(4, 0, "_7163_input_rgba8_buffer")}` which `isa_enricher.cpp` then seeds into `env.initial_fau[0x00]` when analyzing `EBIN[0]`!

---

## 12. Worked Case Study 2: Diamond CFG, `FiniteSetDomain` & Non-Singleton Dereference

Consider a diamond CFG `B0 -> {B1, B2} -> B3` (tested in `tests/test_dataflow.cpp::test_ssa_diamond_cfg_phi_finiteset_and_liveness`):

```text
  B0: %srt = LoadSidebandU64(2)             -> { SrtBase(+0x0) }
  B1: %p1  = IAdd64(%srt, ConstU64(0x10))   -> { SrtBase(+0x10) }
  B2: %p2  = IAdd64(%srt, ConstU64(0x20))   -> { SrtBase(+0x20) }
  B3: %pm  = Phi((B1, %p1), (B2, %p2))      -> { SrtBase(+0x10), SrtBase(+0x20) }  [size = 2!]
      %u   = IAdd64(%pm, ConstU64(0))       -> { SrtBase(+0x10), SrtBase(+0x20) }  [+0 preserved!]
      %ld  = PsbLoad(%u, 8)                 -> Top  [non-singleton addr deref -> Top!]
```

- At `%pm`, `FiniteSetDomain<TrackedValDomain, 4>` keeps **both** concrete offsets `{SrtBase(+0x10), SrtBase(+0x20)}` instead of immediately widening.
- If a consumer calls `df.val(%pm)` (`collapse_tracked_val_set`), it smoothly folds via `TrackedValDomain::join` to `SrtBase(+?)`.
- At `%ld = PsbLoad(%u, 8)`, `addr_set.try_as_singleton(&single_addr)` returns `false` because `%u` has cardinality 2, so `on_psb_load` soundly returns `TrackedValSetDomain::top()`.

---

## 13. Worked Case Study 3: 64-Bit Split/Pack Reconstruction & VA Masking

Valhall hardware stores 64-bit pointers in pairs of 32-bit registers (`r4:r5`), so Pass 1 frequently emits:
1. `Split64(ptr64)` $\to$ `TupleExtract(0)` (low 32-bit `r4`) and `TupleExtract(1)` (high 32-bit `r5`).
2. `BitAnd(hi, 0xffffff)` (clearing upper tag bits on `r5` before dereferencing).
3. `Pack64(lo, masked_hi)` to rebuild the 64-bit address for `PsbLoad`.

In `SsaPointerTransferVisitor`:
- `on_split64` copies the 64-bit `TrackedValSet` onto the `TupleU32x2` node.
- `on_tuple_extract(0)` preserves `is_hi_half = false`; `on_tuple_extract(1)` sets `is_hi_half = true` (or shifts `const_val >> 32` for `ConstInt`).
- `on_pack64(lo, hi)` folds `(hi << 32) | lo` if both are `ConstInt`; otherwise it recovers the pointer/FAU/symbol from `lo` (clearing `is_hi_half`)—even if `hi` was widened to `Top` by an obscure high-word operation!

---

## 14. Worked Case Study 4: Writing a Brand-New Analysis (Uniformity / Divergence)

Here is a complete, minimal **Uniformity Analysis** (determining whether an SSA value is workgroup-uniform or thread-divergent) in ~35 lines of C++ using the framework:

```cpp
struct UniformityDomain {
    enum class Value : uint8_t { Bottom = 0, Uniform, Divergent }; // Top == Divergent
    static Value bottom() { return Value::Bottom; }
    static Value top()    { return Value::Divergent; }
    static bool  leq(const Value& a, const Value& b) {
        return a == Value::Bottom || b == Value::Divergent || a == b;
    }
    static Value join(const Value& a, const Value& b) {
        if (a == Value::Bottom) return b;
        if (b == Value::Bottom) return a;
        return (a == Value::Uniform && b == Value::Uniform) ? Value::Uniform : Value::Divergent;
    }
    static Value meet(const Value& a, const Value& b) {
        if (a == Value::Divergent) return b;
        if (b == Value::Divergent) return a;
        return (a == Value::Uniform || b == Value::Uniform) ? Value::Uniform : Value::Bottom;
    }
    // Note: `equals()` is omitted -> auto-implemented by `domain_equals<UniformityDomain>`!
};

class UniformityVisitor : public SsaDomainTransferVisitor<UniformityDomain> {
    using V = UniformityDomain::Value;
    void on_const(compiler::SsaId d, const compiler::SsaNode&) override { assign(d, V::Uniform); }
    void on_load_fau(compiler::SsaId d, uint32_t, const compiler::SsaNode&) override { assign(d, V::Uniform); }
    void on_load_builtin(compiler::SsaId d, uint32_t bi, const compiler::SsaNode&) override {
        // WorkgroupId is uniform across the workgroup; Local/GlobalInvocationId is divergent
        assign(d, (bi == 26 /*WorkgroupId*/) ? V::Uniform : V::Divergent);
    }
    void on_iadd(compiler::SsaId d, compiler::SsaId a, compiler::SsaId b, bool, const compiler::SsaNode&) override {
        assign(d, UniformityDomain::join(eval(a), eval(b)));
    }
    // `Copy`, `Bitcast`, `Phi`, `Select` are inherited automatically!
};
```

Calling `run_ssa_forward_dataflow<UniformityDomain>(mod, entry, visitor)` immediately solves uniformity across arbitrary loops and branches with zero boilerplate.

---

## 15. Performance, Memory Scaling & Soundness Guarantees

| Property | Complexity / Bound | Verified Measurement (`astc_enc.raw.spv`, 69 KB, 8,628 insts) |
|---|---|---|
| SSA value environment (`ssa_vals`) | $O(|V|)$ memory (single shared vector, power-of-2 growth) | ~4 MB RAM (down from 7.6 GB per-block copies) |
| Side-channel memory (`fau_mem`, `sym_mem`) | $O(|B| \cdot |\mu|)$ memory ($|\mu| \le 20$ typically) | < 100 KB RAM |
| Backwards SSA Liveness (`BitsetDomain`) | $O(|B| \cdot |V| / 64)$ bytes | < 1 MB RAM, < 5 ms |
| End-to-end compile + SSA lower + liveness + pointer analysis + 8,808-line disassembly | Monotone fixpoint, finite height $h \le 5$ | **2.24 seconds total** |

---

## 16. Design Q&A: Deep-Dive Answers to Architectural Review Questions

1. **What do optional `try_as_const_u64` and `make_const_u64` do?**
   They are detected via SFINAE (`detail::has_const_helpers<DomainT>`) by `SsaDomainTransferVisitor<DomainT>`. When a domain implements them, the base visitor automatically performs 32/64-bit constant folding (`IAdd`, `ISub`, `IMul`, `BitAnd`, `BitOr`, `BitXor`, `BitNot`, `Shl`/`ShrU`/`ShrS`/`RotL`/`RotR`) and discharges algebraic extensional equalities (`x+0 == x`, `x&~0 == x`, `x|0 == x`, `x^0 == x`, `x*1 == x`, `x-x == 0`, `x^x == 0`, `x&0 == 0`). Domains that don't track constants (like `SignDomain` or `UniformityDomain`) simply omit them.
2. **Are any of the `TrackedVal::has_offset()` kinds multi-pointers (pointers to pointers)?**
   Yes! Mali Valhall uses a **two-level pointer hierarchy**:
   - `SrtBase` (`SRT_0`) is a pointer-to-pointer: dereferencing `SRT_0 + 16*rack` yields `SrtRackPtr(rack)`.
   - `SrtRackPtr(rack)` (and `DescriptorAddr(rack, desc)`) is a pointer to a 32-byte descriptor struct containing a 64-bit pointer at `+0x08`: dereferencing it yields `BufferAddr(rack, desc)`.
   - `BufferAddr` is a leaf pointer to user data (`UBO`/`SSBO`), whose dereference yields `SymbolValue`.
   - `FauDstBase`, `DataSectionPtr`, `TlsPtr`, and `WlsPtr` are single-level pointers to scalar/vector data.
3. **Why does `same_base_identity()` return unconditionally `true` for `FauDstBase`, `SrtBase`, `DataSectionPtr`, `TlsPtr`, and `WlsPtr`?**
   Because every shader invocation has **exactly one** `FAU_0` bank, **one** `SRT_0` root table, **one** embedded `.data` section, **one** per-thread TLS scratch frame, and **one** per-workgroup WLS shared-memory block. There is no secondary index (`rack` or `desc_idx`) that could distinguish two different `SrtBase` or `TlsPtr` bases.
4. **What scenarios could `same_base_identity()` miss?**
   - **Buffer Device Address (`VK_KHR_buffer_device_address` / physical storage buffers):** Two `BufferAddr`s loaded from an SSBO (pointer chasing through user memory, not through `SRT_0`) both have `rack = -1, desc_idx = -1, symbol_name = ""` and could collide or be over-merged unless tagged with an allocation-site `SsaId`.
   - **Dynamic descriptor indexing (`VK_EXT_descriptor_indexing`):** If `desc_idx` is runtime-computed, `SrtRackPtr + dynamic_idx` has `offset_known = false`, so `deref_pointer` widens to `Top` rather than tracking a symbolic array of buffers.
   - **Aliased Vulkan bindings:** If two different `(rack, desc_idx)` descriptors are bound to overlapping ranges of the same `VkBuffer` by the host application, `same_base_identity` treats them as distinct bases.
5. **How precise is `deref_pointer()`?**
   It is **field-sensitive and exact** for Mali's fixed CSF descriptor layouts (`SRT_0` 16B entries, 32B buffer descriptors with `+0x04 = size` and `+0x08 = 64-bit VA`, and UBO/SSBO constant member offsets resolved via MBS2 type metadata). It is **conservative (`Top`)** for dynamic/unknown offsets into `SrtRackPtr`, non-singleton address sets in `FiniteSetDomain`, and stores/loads through arbitrary user-pointer chains (`BufferAddr` $\to$ `SymbolValue` $\to$ `PsbLoad` yields `Top`).
6. **Why does `on_const` handle `ConstF32` as `ConstInt(node.imm_u32)`?**
   In Valhall ISA, GPRs are untyped 32-bit registers and `MOV.i32` / `Bitcast` frequently move raw IEEE-754 float bit patterns (stored in `node.imm_u32`) through integer/bitwise ops (e.g. sign-bit flipping via `XOR #0x80000000`). Keeping the raw 32-bit bit pattern lets `Bitcast` and bitwise ops fold accurately. The `default:` branch in `on_const` is defensive future-proofing in case a new `SsaOpKind::Const` variant (e.g. `ConstF64` or `ConstVec`) is added later.
7. **Why do we track `LoadBuiltinInput` (`AbiPreload`) in pointer analysis?**
   Two reasons: (a) `isa_enricher.cpp` tags every instruction that reads live ABI registers (`gl_GlobalInvocationID.x/y/z`, `gl_WorkGroupID`, `gl_VertexIndex`, `gl_FragCoord`)—and because SSA separates the initial builtin `SsaId` from any later register reuse, ABI tags automatically stop the instant a register is clobbered! (b) Knowing which operands originate from `GlobalInvocationID` / `LocalInvocationID` is the foundation for future thread-divergence and buffer-index stride analysis.
8. **What do `on_split64`, `on_tuple_extract`, `on_pack64`, `on_vec`, `on_convert`, and `on_psb_load` do—and when do their fallback `Top` branches trigger?**
   - **`on_split64(dst, src64)`:** Pass 1 models splitting a 64-bit value into two 32-bit GPRs as `tup = Split64(src64)` followed by `lo = TupleExtract(tup, 0)` and `hi = TupleExtract(tup, 1)`. Assigning `eval(src64)` onto `tup` propagates the full 64-bit `TrackedValSet` to both `TupleExtract` children without losing pointer identity.
   - **`on_tuple_extract` `else` branch:** Only reachable if a future multi-element tuple (`elem_idx >= 2`, e.g., a 3-tuple or 4-tuple) is added to `ValhallSsaModule`; currently `TupleU32x2` and `TupleValToken` only use indices `0` and `1`.
   - **`on_pack64(dst, lo, hi)`:** Recombines two 32-bit GPRs back into a 64-bit address (`Pack64(r4, r5)`). It checks `lo` first (because Valhall's low 32-bit half carries the base pointer identity and offset, while `hi` is often masked with `AND #0xffffff` or `MOV`ed from a constant upper page), falling back to `hi` only if `lo` wasn't a pointer (e.g., if `lo` was overwritten by an offset calculation while `hi` still carries the base).
   - **`on_vec` `Top` fallback:** Triggered for `VecConstruct2/3/4`, `VecShuffle`, `PackHalf2x16`, and `UnpackHalf2x16` (everything in `SsaOpKind::Vec` other than `VecExtract`).
   - **`on_convert` `Top` fallback:** Triggered for float/int numeric conversions (`ConvertFToS`, `ConvertFToU`, `ConvertSToF`, `ConvertUToF`), which do not preserve integer bit patterns or pointer identities.
   - **`on_psb_load(dst, addr, elem_bytes)`:** Models Valhall's Physical Storage Buffer memory load (`LOAD.i32/i64/i128 [addr]`), returning a `TupleValToken` whose element 0 is the loaded value. When `addr` is a singleton pointer, it chases `SRT_0 -> SRT[rack] -> BufferAddr/BufferSize -> SymbolValue`; when `addr` is a non-singleton set, it soundly widens to `Top`.

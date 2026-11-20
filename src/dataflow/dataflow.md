# Valhall SSA Dataflow & Abstract Interpretation Framework
## Design Document & Implementation Guide

## Table of Contents

- [Overview](#overview)
- [Module Map & Dependency Graph](#module-map--dependency-graph)
- [Domain Theory](#domain-theory)
- [Standard Domain Combinators](#standard-domain-combinators)
- [Symbolic Memory Domain](#symbolic-memory-domain)
- [SSA Abstract State](#ssa-abstract-state)
- [Fixpoint Solver](#fixpoint-solver)
- [Backwards SSA Liveness Analysis](#backwards-ssa-liveness-analysis)
- [Transfer Functions](#transfer-functions)
- [Implementation Guide: What You Must Implement vs. What You Get for Free](#implementation-guide-what-you-must-implement-vs-what-you-get-for-free)
- [Worked Case Study: Pilot Shader SRT Pointer Chasing](#worked-case-study-pilot-shader-srt-pointer-chasing)
- [Worked Case Study: Diamond CFG, `FiniteSetDomain` & Non-Singleton Dereference](#worked-case-study-diamond-cfg-finitesetdomain--non-singleton-dereference)
- [Worked Case Study: 64-Bit Split/Pack Reconstruction & VA Masking](#worked-case-study-64-bit-splitpack-reconstruction--va-masking)
- [Worked Case Study: Writing a Brand-New Analysis (Uniformity / Divergence)](#worked-case-study-writing-a-brand-new-analysis-uniformity--divergence)
- [Performance, Memory Scaling & Soundness Guarantees](#performance-memory-scaling--soundness-guarantees)

## Overview

The Valhall SSA dataflow framework provides a reusable abstract interpretation infrastructure over `ValhallSsaModule` (`src/ssa/valhall_ssa.{h,cpp}`). It decouples lattice definitions, composable domain combinators, CFG fixpoint iteration, and SSA instruction transfer dispatch so new static analyses (pointer chasing, liveness, uniformity, range analysis, alias/escape analysis) can be authored concisely without duplicating solver or IR-traversal boilerplate.

### Core Design Principles

- **Domain Combinators (`domains.h`):** Compose `VectorDomain`, `MapDomain`, `TupleDomain`, `BitsetDomain`, and `FiniteSetDomain<T, K>` like algebraic building blocks rather than hand-writing map/vector/tuple join-meet-equals loops inside each analysis.
- **Extensional Equalities by Default (`SsaDomainTransferVisitor`):** Every domain automatically inherits copy propagation, `Phi`/`Select` joins, and algebraic identities (`x+0 == x`, `x&~0 == x`, `x^x == 0`, etc.) simply by opting into `try_as_const_u64` / `make_const_u64`.

## Module Map & Dependency Graph

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
│ • MapDomain         │      └────────────┬────────────┘      │ • offset arithmetic   │
│ • TupleDomain       │                   │                   └───────────┬───────────┘
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

## Domain Theory

### Mathematical Requirements

A complete lattice / CPO is a tuple $(\mathcal{L}, \sqsubseteq, \sqcup, \sqcap, \bot, \top)$ satisfying for all $a, b, c \in \mathcal{L}$:
- **Partial Order ($\sqsubseteq$):** Reflexive ($a \sqsubseteq a$), Antisymmetric ($a \sqsubseteq b \land b \sqsubseteq a \implies a = b$), Transitive ($a \sqsubseteq b \land b \sqsubseteq c \implies a \sqsubseteq c$).
- **Bounds:** $\bot \sqsubseteq a \sqsubseteq \top$.
- **Least Upper Bound ($\sqcup$, `join`):** Commutative, Associative, Idempotent ($a \sqcup a = a$), $\bot \sqcup a = a$, $\top \sqcup a = \top$, and $a \sqsubseteq b \iff a \sqcup b = b$.
- **Greatest Lower Bound ($\sqcap$, `meet`):** Dual of `join` ($\top \sqcap a = a$, $\bot \sqcap a = \bot$, $a \sqsubseteq b \iff a \sqcap b = a$).
- **Ascending Chain Condition (ACC):** Every strictly ascending chain $a_0 \sqsubset a_1 \sqsubset \dots \sqsubset a_h$ has finite length $h < \infty$. Over a finite-height domain, domain-continuous functions converge to a fixpoint in finitely many steps.

### Interface

Every domain struct `D` passed to `AbstractState<D>`, `VectorDomain<D>`, `MapDomain<K, D>`, `TupleDomain<...>`, `FiniteSetDomain<D, K>`, or `SsaDomainTransferVisitor<D>` is checked at instantiation by `ValidateLatticeDomain<D>`:

```cpp
struct MyDomain {
    // REQUIRED (enforced with static_assert diagnostics):
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

## Standard Domain Combinators

### `VectorDomain<ElemDomainT>`

Pointwise domain over a dense 0-indexed `std::vector<ElemValue>`, where out-of-bounds indices implicitly map to `ElemDomainT::bottom()`.

- **Additional Operations:**
  - `get(idx)`, `set(idx, v)`, `join_at(idx, v)` (returns `bool changed`)
  - `join_into(dst, other)`, `meet_into(dst, other)` (in-place mutation)

### `MapDomain<KeyT, ValDomainT>`

Pointwise domain over a sparse `std::map<KeyT, ValValue>`, where absent keys implicitly map to `ValDomainT::bottom()`. Setting or meeting an entry to `bottom()` erases the key to maintain canonical sparsity.

- **Additional Operations:** `get(k)`, `set(k, v)`, `join_at(k, v)`, `clear()`, `join_into`, `meet_into`.

### `TupleDomain<D1, D2, ...>`

Variadic Cartesian product domain over `std::tuple<typename Domains::Value...>`:

$$\text{join}((a_1, \dots, a_k), (b_1, \dots, b_k)) = (\text{join}_1(a_1, b_1), \dots, \text{join}_k(a_k, b_k))$$

```cpp
using CombinedDomain = TupleDomain<SignDomain, BitsetDomain>;
static_assert(ValidateLatticeDomain<CombinedDomain>::valid, "");
```

All six domain operations (`bottom`, `top`, `join`, `meet`, `leq`, `equals`) are generated automatically via C++17 fold expressions over `std::index_sequence_for<Domains...>`.

### `BitsetDomain`

Compact powerset domain $(\mathcal{P}(\{0, \dots, N-1\}), \subseteq, \cup, \cap, \emptyset, \mathcal{U})$ backed by `std::vector<uint64_t>` (64 elements per word).

- **Additional Operations:** `set(bit)`, `clear(bit)`, `test(bit)`, `clear_all()`, `or_with(other)`, `or_diff_with(other, minus)`, `BitsetDomain::diff(a, b)` ($\setminus$).

### `FiniteSetDomain<ElemDomainT, MaxK = 4>`

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

**Motivation:**
- When a shader computes `off = cond ? 16 : 32;` (or a `Phi` merges two buffer addresses `&bufA` and `&bufB`), a flat lattice immediately collapses `join(16, 32)` to `Top`.
- In `FiniteSetDomain<TrackedValDomain, 4>`, `join({16}, {32}) = {16, 32}` (cardinality 2 $\le 4$), and subsequent arithmetic (`off + 8`) lifts pointwise via `lift_unary` to `{24, 40}`.
- **Lift & Lower:**
  - **Implicit singleton lift:** `FiniteSetValue` provides a non-`explicit` converting constructor and assignment operator from `const ElemValue&`, so singletons lift automatically (e.g., `TrackedValSet s = TrackedVal::make_const(42);` or `assign(dst, TrackedVal::make_srt_base());`). Explicit `lift_value(x)` / `singleton(x)` are also available.
  - `try_as_singleton(&out)`: extracts the unique element when `size() == 1` (e.g., to check whether a memory address is uniquely resolved before dereferencing).
  - `lift_unary(S, fn)` & `lift_binary(A, B, fn)`: project any pointwise transfer function `ElemValue -> ElemValue` (or `-> FiniteSetValue`) across the set, widening to `Top` cleanly if cardinality exceeds `MaxK`.
- **Constant-Folding Forwarding:** When `ElemDomainT` provides `try_as_const_u64`/`make_const_u64`, `FiniteSetDomain` forwards them for singleton constant sets, so `SsaDomainTransferVisitor<FiniteSetDomain<TrackedValDomain, 4>>` inherits all extensional equalities automatically.

## Symbolic Memory Domain

`memory_domain.h` models GPU address spaces using structured `SymbolicMemoryLoc` keys rather than opaque strings:

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

### Pointer Arithmetic & Disjointness (`SymbolicMemoryLoc`)

- **Pointer wrangling methods:** `with_added_offset(delta)`, `with_unknown_offset()`, and `with_masked_offset(mask)` encapsulate constant offset addition, dynamic-index widening, and VA bit-masking directly on the memory location.
- **Disjointness (`may_alias()`):**

| Pair of Locations | `may_alias()` | Rationale |
|---|---|---|
| Different `MemoryRegionKind` (e.g., `Tls` vs. `Buffer`) | `false` | GPU address spaces are hardware-disjoint. |
| `SrtTable(rack=0)` vs. `SrtTable(rack=4)` | `false` | Distinct SRT rack tables occupy disjoint descriptor allocations. |
| `Buffer(rack=4, desc=0)` vs. `Buffer(rack=4, desc=1)` | `false` | Distinct Vulkan descriptor bindings are modeled as non-aliasing by default. |
| Same buffer, known offsets `+0x00` vs. `+0x10` | `false` | Distinct known byte offsets within the same buffer. |
| Same buffer, known `+0x00` vs. unknown `+?` | `true` | Dynamic offset may alias any offset in the same buffer. |

**Why `fau_mem` is tracked separately:**
- In **Main shaders** (Compute, Vertex, Fragment), `FAU_0` is **strictly read-only (RO)**—it is populated once before dispatch by the Pilot shader and the driver's RLOC/FCST constant uploads. Therefore, `LoadFau(w)` in a Main shader is flow-insensitive and never invalidated by `PsbStore`s to `Buffer`/`Tls`/`Wls`.
- In **Pilot shaders**, `FAU_0` is the output destination (`FauDstBase` in `r0:r1`), written via `PsbStore(FauDstBase + off, val)` and never aliased by `SRT_0` reads.

## SSA Abstract State

`AbstractState<DomainT>` (`src/dataflow/abstract_domain.h`) composes three standard domain combinators:

```cpp
template <typename DomainT>
struct AbstractState {
    static_assert(ValidateLatticeDomain<DomainT>::valid, "");
    using SsaEnvDomain = VectorDomain<DomainT>;
    using FauMemDomain = MapDomain<uint32_t, DomainT>;
    using SymMemDomain = MapDomain<SymbolicMemoryLoc, DomainT>;

    typename SsaEnvDomain::Value ssa_vals; // SsaId -> Value
    typename FauMemDomain::Value fau_mem;  // FAU byte_off -> Value (RO in Main, RW in Pilot)
    typename SymMemDomain::Value sym_mem;  // SymbolicMemoryLoc -> Value (disjoint regions)
    ...
};
```

## Fixpoint Solver

The chaotic worklist engine (`src/dataflow/worklist.{h,cpp}`) is decoupled from `DomainT` and supports both forward and backward passes:

```cpp
enum class DataflowDirection : uint8_t { Forward = 0, Backward = 1 };

WorklistResult run_cfg_worklist(
    const compiler::ValhallSsaModule& mod,
    DataflowDirection dir,
    const std::function<bool(uint32_t block_id)>& transfer_block,
    uint32_t max_iterations = 64);
```

- **`Forward`:** Seeds worklist with blocks $0, 1, \dots, |B|-1$; whenever `transfer_block(b)` returns `true`, enqueues `fallthrough_block` and `branch_target_block`.
- **`Backward`:** Seeds worklist with blocks $|B|-1, \dots, 0$; whenever `transfer_block(b)` returns `true`, enqueues all `blk.preds`.
- **Termination Guard:** Caps total block visits at `max_iterations * |B|` and reports `res.converged = worklist.empty()`.

## Backwards SSA Liveness Analysis

Built directly on `BitsetDomain` and `run_cfg_worklist(mod, DataflowDirection::Backward, ...)`:

1. **Precompute per-block `BitsetValue` sets:**
   - `phi_defs[B]` = $\{p \mid p \in B.\text{phis}\}$
   - `body_def[B]` = $\{n \mid n \in B.\text{nodes}\}$
   - `body_ue_use[B]` = upward-exposed uses in `B.nodes` and `B.branch_cond`
2. **Fixpoint Equations (inside `run_cfg_worklist` callback):**
   $$\text{LIVE\\_OUT}[B] = \bigcup_{S \in \text{succ}(B)} \Big( (\text{LIVE\\_IN}[S] \setminus \text{PHI\\_DEFS}[S]) \cup \text{PHI\\_USES}(B \to S) \Big)$$
   $$\text{LIVE\\_IN}[B] = \text{UE\\_USE}[B] \cup (\text{LIVE\\_OUT}[B] \setminus \text{BODY\\_DEF}[B])$$
3. **Per-Node Kill Sets (`killed_after_node[n]`):** After fixpoint convergence, a single reverse sweep over `B.nodes` starting from `LIVE_OUT[B] ∪ {branch_cond}` records every `arg` of `n` that is not live after `n`.

## Transfer Functions

### `SsaOpKind` Classification

`classify_ssa_op(tables::SsaOp)` maps all 134 `tables::SsaOp` enum values into 28 semantic `SsaOpKind`s (paired with `is_ssa_op_64bit()` and `classify_ssa_shift_kind()`). `dispatch_ssa_transfer(node, visitor)` switches on `SsaOpKind` and invokes the matching virtual hook on `SsaTransferVisitorBase`:

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

### Extensional Equalities (`SsaDomainTransferVisitor<DomainT>`)

Every subclass of `SsaDomainTransferVisitor<DomainT>` inherits these identities by default:
- **Copy / Alias:** `Copy(x)` and `Bitcast(x)` copy `state.val(x)` directly.
- **Control-Flow Joins:** `Phi` joins all non-self incoming values; `Select(c, x, x)` copies `x` and `Select(c, a, b)` computes `join(a, b)`.
- **Algebraic Identities (when `try_as_const_u64`/`make_const_u64` are present):**
  - Identity elements: `x + 0`, `0 + x`, `x - 0`, `x * 1`, `1 * x`, `x | 0`, `0 | x`, `x ^ 0`, `0 ^ x`, `x & ~0`, `~0 & x`, `x << 0`, `x >> 0` $\to$ `eval(x)`.
  - Idempotence: `x & x`, `x | x` $\to$ `eval(x)`.
  - Annihilators & self-cancellation: `x * 0`, `0 * x`, `x & 0`, `0 & x`, `x - x`, `x ^ x` $\to$ `make_const(0)`.
  - Constant folding: integer, bitwise, and shift ops fold when both operands are known constants.

## Implementation Guide: What You Must Implement vs. What You Get for Free

### What You Must Implement
1. **Value Domain `MyDomain`:**
   - `using Value = ...;`
   - `bottom()`, `top()`, `join(a, b)`, `meet(a, b)`, `leq(a, b)`.
   - *(Optional)* `equals(a, b)` if faster than `leq(a, b) && leq(b, a)`.
   - *(Optional)* `try_as_const_u64(v, &out)` and `make_const_u64(c)` to opt into constant folding and algebraic identities.
2. **Transfer Visitor `MyVisitor : public SsaDomainTransferVisitor<MyDomain>`:**
   - Override only the `on_*` hooks with domain-specific semantics, delegating unhandled cases back to `SsaDomainTransferVisitor<MyDomain>::on_*(...)`.
3. **Entry State & Runner Call:**
   - Construct `AbstractState<MyDomain> entry(mod.nodes.size())`, seed `entry.fau_mem`/`sym_mem` if needed, and call `run_ssa_dataflow<MyDomain>(mod, dir, entry, visitor, liveness_ptr)`.

### What You Get for Free
- CFG traversal, worklist deduplication, convergence detection, and iteration bounds (`worklist.cpp`).
- $O(|V|)$ shared SSA environment storage and per-block memory-map joins (`abstract_domain.h`, `transfer.h`).
- `tables::SsaOp` classification and dispatch (`classify_ssa_op`, `dispatch_ssa_transfer`).
- `Copy`, `Bitcast`, `Phi`, `Select`, and algebraic identity rules (`SsaDomainTransferVisitor`).
- Domain combinators (`FiniteSetDomain`, `TupleDomain`, `VectorDomain`, `MapDomain`, `BitsetDomain`) and backwards SSA liveness (`run_ssa_liveness`).

## Worked Case Study: Pilot Shader SRT Pointer Chasing

Tracing `_7163_input_rgba8_buffer` in `astc_enc.raw.spv`'s Pilot Shader (`EBIN[1]`):

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
0x0048  | %10 = BitAnd(%9, ConstU32(0xffffff))             | { SrtRackPtr(4).hi }  (VA mask preserved)
0x0048  | %11 = Pack64(%8, %10)                            | { SrtRackPtr(rack=4, +0x0) }
0x0050  | %12 = IAdd64(%11, ConstU64(0x8))                 | { SrtRackPtr(rack=4, +0x8) }
0x0050  | %13 = PsbLoad(%12, elem_bytes=8)                 | { BufferAddr(4, 0, "_7163_input_rgba8_buffer") }
0x0050  | %14 = TupleExtract(%13, 0)                       | { BufferAddr(4, 0, "_7163_input_rgba8_buffer") }
0x0080  | %15 = PsbStore(%1, %14, elem_bytes=8)            | fau_mem[0x00] = { BufferAddr(4, 0, "_7163_...") }
```

Pointer offset arithmetic at `0x0008`, `0x0048`, and `0x0050` delegates directly to `SymbolicMemoryLoc::with_added_offset()` and `SymbolicMemoryLoc::with_masked_offset()`.

## Worked Case Study: Diamond CFG, `FiniteSetDomain` & Non-Singleton Dereference

Consider a diamond CFG `B0 -> {B1, B2} -> B3` (verified in `tests/test_dataflow.cpp`):

```text
  B0: %srt = LoadSidebandU64(2)             -> { SrtBase(+0x0) }
  B1: %p1  = IAdd64(%srt, ConstU64(0x10))   -> { SrtBase(+0x10) }
  B2: %p2  = IAdd64(%srt, ConstU64(0x20))   -> { SrtBase(+0x20) }
  B3: %pm  = Phi((B1, %p1), (B2, %p2))      -> { SrtBase(+0x10), SrtBase(+0x20) }  [size = 2]
      %u   = IAdd64(%pm, ConstU64(0))       -> { SrtBase(+0x10), SrtBase(+0x20) }
      %ld  = PsbLoad(%u, 8)                 -> Top  [non-singleton addr deref -> Top]
```

- At `%pm`, `FiniteSetDomain<TrackedValDomain, 4>` retains both concrete offsets `{SrtBase(+0x10), SrtBase(+0x20)}`.
- Calling `df.val(%pm)` (`collapse_tracked_val_set`) folds them via `TrackedValDomain::join` to `SrtBase(+?)`.
- At `%ld = PsbLoad(%u, 8)`, `addr_set.try_as_singleton(&single_addr)` returns `false`, so `on_psb_load` soundly returns `Top`.

## Worked Case Study: 64-Bit Split/Pack Reconstruction & VA Masking

Valhall stores 64-bit pointers in pairs of 32-bit registers (`r4:r5`), so Pass 1 emits `Split64(ptr64)` $\to$ `TupleExtract(0)` (low 32 bits) and `TupleExtract(1)` (high 32 bits), often followed by `BitAnd(hi, 0xffffff)` and `Pack64(lo, masked_hi)`:
- `on_split64` copies the 64-bit `TrackedValSet` onto the `TupleU32x2` node.
- `on_tuple_extract(0)` sets `is_hi_half = false`; `on_tuple_extract(1)` sets `is_hi_half = true` (or shifts `const_val >> 32` for `ConstInt`).
- `on_pack64(lo, hi)` folds `(hi << 32) | lo` if both are `ConstInt`; otherwise it recovers the pointer/FAU/symbol from `lo` (clearing `is_hi_half`), even if `hi` was widened to `Top`.

## Worked Case Study: Writing a Brand-New Analysis (Uniformity / Divergence)

A complete workgroup-uniformity analysis in ~30 lines of C++:

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
};

class UniformityVisitor : public SsaDomainTransferVisitor<UniformityDomain> {
    using V = UniformityDomain::Value;
    void on_const(compiler::SsaId d, const compiler::SsaNode&) override { assign(d, V::Uniform); }
    void on_load_fau(compiler::SsaId d, uint32_t, const compiler::SsaNode&) override { assign(d, V::Uniform); }
    void on_load_builtin(compiler::SsaId d, uint32_t bi, const compiler::SsaNode&) override {
        assign(d, (bi == 26 /*WorkgroupId*/) ? V::Uniform : V::Divergent);
    }
    void on_iadd(compiler::SsaId d, compiler::SsaId a, compiler::SsaId b, bool, const compiler::SsaNode&) override {
        assign(d, UniformityDomain::join(eval(a), eval(b)));
    }
};
```

## Performance, Memory Scaling & Soundness Guarantees

| Property | Complexity / Bound | Verified Measurement (`astc_enc.raw.spv`, 69 KB, 8,628 insts) |
|---|---|---|
| SSA value environment (`ssa_vals`) | $O(|V|)$ memory (single shared vector) | ~4 MB RAM |
| Side-channel memory (`fau_mem`, `sym_mem`) | $O(|B| \cdot |\mu|)$ memory ($|\mu| \le 20$ typically) | < 100 KB RAM |
| Backwards SSA Liveness (`BitsetDomain`) | $O(|B| \cdot |V| / 64)$ bytes | < 1 MB RAM, < 5 ms |
| End-to-end compile + SSA lower + liveness + pointer analysis + 8,808-line disassembly | Finite height $h \le 5$ | **2.24 seconds total** |

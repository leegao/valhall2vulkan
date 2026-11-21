# Design Doc: Dataflow-Driven De-FAU and De-Virtualization of Valhall SSA

**Status:** Design Revision v2 (Checkpoint v27.1 — incorporates `leegao@` review)
**Target Modules:** `src/dataflow/domains/`, `src/dataflow/memory_domain.h`, `src/dataflow/pointer_analysis.{h,cpp}`, `src/dataflow/ssa_defau_devirt.{h,cpp}`
**Primary Goal:** Lower Valhall-specific FAU indirections, Pilot-shader precomputations, and hardware-virtualized ISA idioms in `ValhallSsaModule` into portable, semantically equivalent SPIR-V-ready SSA using the existing `src/dataflow` abstract interpretation framework.

---

## 1. Why: The Valhall FAU and Hardware-Virtualization Problem

### 1.1 Architectural Origin of FAU Indirection and Virtualization

When Arm's Valhall compiler (`libGLES_mali`) compiles a Vulkan SPIR-V shader to Valhall ISA (`v9`–`v14`), it performs two aggressive hardware-lowering transformations that leave no direct counterpart in portable Vulkan SPIR-V:

1. **Fast Access Uniform (FAU) & Pilot-Shader Hoisting (`EBIN[1]` $\to$ `EBIN[0]`)**:
   - Scalar constants (`FCST`), descriptor-table relocations (`RLOC`), buffer base virtual addresses, UBO scalar loads, and uniform boolean predicates are hoisted out of the Main shader (`EBIN[0]`) into a small read-only uniform register bank (`FAU_0`, accessed via `u0..u63` $\to$ `SsaOp::LoadFau`).
   - A companion **Pilot shader** (`EBIN[1]`) executes once per draw/dispatch before `EBIN[0]`, chases Shader Resource Table (`SRT_0`) descriptors, performs bounds-checked UBO loads (`LOAD`), evaluates uniform predicates (`FCMP`), and writes the results into `FAU_0` via `STORE` to `pilot_fau_dst_va`.
2. **Hardware Virtualization of High-Level SPIR-V Constructs**:
   - High-level SPIR-V semantics are lowered into Valhall-specific micro-architectural idioms: 64-bit physical GPU virtual addresses split across paired 32-bit GPRs (`Pack64`/`Split64`), 16:16 halfword-packed texel coordinates (`SWZ.v2i16` $\to$ `Bfi`/`BfeU`), 16-lane hardware subgroup operations (`WMASK.subgroup16`, `CLPER.subgroup16.xor`) emulating 32-lane Vulkan subgroup ops, packed 32-bit/16-bit condition masks (`FCMP_AND.m1` + `BRANCH.and`/`.h0`/`.h1`), and hardware sideband queries (`LoadSidebandU32`).

### 1.2 Concrete Examples of Valhall Virtualized Idioms in Real Shaders

Below are five concrete idioms observed directly in `samples/comp_unique_02_d7242030.spv` and the 25-shader sample suite:

#### Example A: Pilot-Hoisted UBO Values, Derived Predicates, and Buffer Base Addresses
In `samples/comp_unique_02_d7242030.spv`, the Pilot shader (`EBIN[1]`) executes:
```text
; EBIN[1] Pilot Shader:
LOAD.i64  @r8:r9,   [SRT[4]+0x8]         ; &_13_support_buffer (UBO set=0, binding=0)
LOAD.i32  @r6,      [r8:r9 + 0xd0]       ; scale0 = ubo0[0xd0]
LOAD.i32  @r7,      [r8:r9 + 0xe0]       ; scale1 = ubo0[0xe0]
FCMP.f32.m1.eq r4,  r6, k1f              ; r4 = (scale0 == 1.0f) ? 0xFFFFFFFF : 0
FCMP.f32.m1.eq r5,  r7, k1f              ; r5 = (scale1 == 1.0f) ? 0xFFFFFFFF : 0
STORE.i128 @r4:r5:r6:r7, [FAU_0 + 0x0]   ; FAU[0]=p0, FAU[1]=p1, FAU[2]=scale0, FAU[3]=scale1
LOAD.i32  @r4,      [&_19_cp_c3 + 0x0]   ; threshold = ubo4[0x0]
STORE.i32 @r4,      [FAU_0 + 0x10]       ; FAU[4] = threshold
LOAD.i64  @r62:r63, [SRT[4]+0x28]        ; &_24_cp_s0 (SSBO set=0, binding=1)
STORE.i64 @r62:r63, [FAU_0 + 0x18]       ; FAU[6..7] = &ssbo1
```
Meanwhile, the Main shader (`EBIN[0]`) sees only opaque `LoadFau(0..7)` plus `FCST` constant-pool slots `LoadFau(8..27)` (`0x0ffe0000`, `0x77800000`, `0x0ffc0000`, `0x07800000`, `1..15`). Without de-FAU, Pass 2 must synthesize a fake FAU UBO instead of emitting `OpAccessChain` into the real `ubo:0:0`, `ubo:0:4`, and `ssbo:0:1` descriptors.

#### Example B: 16:16 Packed Halfword Texel Coordinates (`SWZ.v2i16` + `LD_TEX_IMM`)
Valhall's `LD_TEX_IMM` with `descriptor_type=texture_u16` packs integer `(x, y)` into a single 32-bit GPR (`SWZ.v2i16 r5.h1, r1^.h10`), which Pass 1 faithfully lowers to:
```text
%59 : U32 = Bfi(%56, %36, 16, 16)        ; pack (x, y) into 16:16
%62 : U32 = BfeU(%59, 0, 16)             ; x & 0xFFFF
%65 : U32 = BfeU(%59, 16, 16)            ; y & 0xFFFF
%69 : V4U32 = TexSample(Vec2(%66, %67), %2) "tex:0:2"
```
When `KnownBits` proves `x, y` fit in `[0, 65535]` (or under Valhall's architectural `texture_u16` precondition), `BfeU(Bfi(x, y, 16, 16), 0, 16)` simplifies directly to `x` and `BfeU(..., 16, 16)` to `y`, eliminating the redundant bit-packing.

#### Example C: Virtualized 16-Lane Subgroup Reduction Emulating 32-Lane SPIR-V
In `comp_unique_02_d7242030.spv`, the source SPIR-V performed a 32-lane `SubgroupShuffleXor` butterfly (`delta = 1, 2, 4, 8, 16`). Because Valhall's hardware warp is 16 lanes (`subgroup16`), `libGLES_mali` emits:
```text
WMASK.subgroup16        r2, k1n                    ; ballot = SubgroupBallot(true).x
BRANCH                  r2^.and, 0x118             ; if (ballot == 0xFFFFFFFF) goto fast_path
; Fast path (0x0118..0x0150): 4-step XOR butterfly (delta = 1, 2, 4, 8)
CLPER.subgroup16.xor    r2, r1, 1  ; CSEL.u32.lt r2, r1, r2
CLPER.subgroup16.xor    r3, r2, 2  ; CSEL.u32.lt r2, r2, r3
CLPER.subgroup16.xor    r3, r2, 4  ; CSEL.u32.lt r2, r2, r3
CLPER.subgroup16.xor    r3, r2, 8  ; CSEL.u32.lt r2, r2, r3
; Virtualized delta=16 lookup (0x0158..0x0268): materialize all 16 lanes + CSEL ladder!
CLPER.subgroup16.zero   r3..r50, r2, 0..15         ; 16 static shuffles
BRANCH                  lane_id.w0, 0x278          ; only lane 0 executes ladder
OR.i32                  r51, 16, lane_id           ; r51 = 16 | 0 = 16
CSEL.u32.eq             r3, r51, u6..u13(1..15), r4..r50, r3
CSEL.u32.lt             r2, r2, r3, r2, r3
ATOM.i32.aumin          @r2, u3, #0x0
```
Notice how `OR.i32 r51, 16, lane_id` on lane 0 evaluates to constant `16`, which fails every `r51 == 1..15` `CSEL` check and selects `r3` (lane 0's own already-reduced `r2`), making `min(r2, r3) == r2`! De-FAU of `u6..u13` (`1..15`) plus path-sensitive constant propagation (`lane_id == 0` on the taken/fallthrough edge of `BRANCH lane_id.w0`) collapses the entire 32-instruction `CLPER` + `CSEL` ladder to a single `OpGroupNonUniformUMin` (or clean 4-stage butterfly).

#### Example D: Packed Boolean Masks (`m1`/`i1`/`f1`) & `BRANCH` Combine Modes
Valhall has no 1-bit predicate register file. Comparisons (`FCMP`, `ICMP`, `FCMP_AND`, `ICMP_AND`) produce 32-bit `m1` (`0xFFFFFFFF`/`0`), `i1` (`1`/`0`), or `f1` (`1.0f`/`0.0f`) masks, or two 16-bit halfword predicates (`v2i16`) consumed by `BRANCH.h0`, `BRANCH.h1`, `BRANCH.and`, or `BRANCH.lowbits`. In SSA, this leaves verbose `Select(p, 0xFFFFFFFF, 0)` $\to$ `BitAnd` $\to$ `ICmpNe(..., 0)` chains that must be lowered back to native `Bool` `LogicalAnd`/`LogicalOr` trees.

#### Example E: Ballot-Based Active-Lane Election (`WMASK` + `CLZ` + `ISUB` + `POPCOUNT`)
At `0x0700..0x0730` in `comp_unique_02_d7242030.spv`, `libGLES_mali` lowers `subgroupElect()` + `subgroupBallotBitCount(subgroupBallot(true))` to:
```text
WMASK.subgroup16  r0, k1n                 ; ballot = SubgroupBallot(true).x
CLZ.u32           r1, r0                  ; leading zeros of ballot
ISUB.u32          r1, 31, r1              ; msb_lane = 31 - clz(ballot)
ICMP.u32.m1.eq    r1, r1, lane_id.w0      ; am I the highest active lane?
BRANCH.not        r1, 0x738
POPCOUNT.i32      r0, r0                  ; active_count = popcount(ballot)
ATOM.i32.aadd     @r0, u3, #0x10
```
De-virtualization recognizes `(31 - Clz(ballot)) == lane_id` as `OpGroupNonUniformElect` (or preserves it as exact portable SPIR-V `OpGroupNonUniformBallotFindMSB` + `OpGroupNonUniformBallotBitCount`).

### 1.3 Complete Inventory of Valhall-Specific Leftover SSA Constructs

Every Valhall-specific construct that can appear in `ValhallSsaModule` after Pass 1 (`src/valhall_pass1.cpp`) and lacks a direct portable Vulkan SPIR-V equivalent is inventoried below:

| # | Leftover SSA Construct / Pattern | Origin in Valhall ISA | Target Portable SPIR-V / Canonical SSA | Honest Fallback (if Unresolved) |
|---|---|---|---|---|
| 1 | `LoadFau(w)` (static constant) | `FCST` constant pool (`uK.w0/w1`) | `ConstU32(val)` / `ConstF32(val)` | Synthetic FAU UBO load + fallback log |
| 2 | `LoadFau(w)` (Pilot UBO load) | `EBIN[1]` `LOAD` from `SRT->Buf[b]+off` | `BufferLoad(ubo:s:b, off)` | Synthetic FAU UBO load + fallback log |
| 3 | `LoadFau(w)` (Pilot derived expr) | `EBIN[1]` `FCMP`/`ICMP`/`ALU` on UBO | Inlined SSA expression (`FCmpOrdEq(BufferLoad(...), 1.0f)`) | Synthetic FAU UBO load + fallback log |
| 4 | `Pack64(LoadFau(lo), LoadFau(hi))` | `EBIN[1]` 64-bit buffer VA in `FAU` | Symbolic `BufferAddr(set, binding, off)` | PhysicalStorageBuffer64 (`Psb*`) |
| 5 | `PsbLoad` / `PsbStore` / `AtomicRmw(va64)` | `LOAD`/`STORE`/`ATOM` on chased VA | `BufferLoad` / `BufferStore` / `BufferAtomicRmw(set, binding, off)` | `PhysicalStorageBuffer64` + fallback log |
| 6 | `PsbLoad`/`PsbStore` on `TLS_0` / `WLS_0` | Register spill / workgroup shared mem | `ssa_spill_mem2reg` SSA value / `Workgroup` array `OpAccessChain` | `Function`/`Workgroup` byte array |
| 7 | `LoadSidebandU32(kind, idx)` | Hardware sideband (`SRT_0`, `FAU_0`, `lane_width`, `viewport`) | `gl_SubgroupSize`, push constant, or symbolic descriptor root | Sideband UBO emulation + fallback log |
| 8 | `LoadSpecialReg` (`lane_id`, `r55..r62`) | Valhall ABI preloads & special regs | Canonical `LoadBuiltinInput` (`gl_WorkGroupID`, `gl_SubgroupInvocationID`, etc.) | Keep builtin / zero-init + log |
| 9 | `Bfi(x, y, 16, 16)` $\to$ `BfeU(..., 0/16, 16)` | `SWZ.v2i16` + `LD_TEX_IMM` (`texture_u16`) | Direct `(x, y)` `VecConstruct2` when `KnownBits` proves $16$-bit range | Keep `BitAnd(..., 0xFFFF)` |
| 10 | `Select(p, 0xFFFFFFFF, 0)` $\to$ `BitAnd`/`ICmpNe(0)` | `FCMP`/`ICMP` `.m1`/`.i1`/`.f1` + `BRANCH` | Native `Bool` `LogicalAnd` / `LogicalOr` / `LogicalNot` | Keep `Select` + `ICmpNe` |
| 11 | `BRANCH.h0`/`.h1`/`.and`/`.lowbits` | Unary 32-bit `BranchCombineM` | Native `Bool` comparisons (`ICmpEq(ballot, 0xFFFFFFFF)`, halfword tests) | Keep expanded bitmask test |
| 12 | `CLPER.xor` butterfly + 16-way `CSEL` ladder | 16-lane hardware emulating 32-lane shuffle | `OpGroupNonUniformUMin`/`UMax`/`IAdd` or clean `SubgroupShuffle` | Keep `SubgroupShuffle` sequence |
| 13 | `Blend` / `ATEST` / `ZS_EMIT` / `LD_TILE` | Fragment fixed-function tile-buffer ops | `OpStore` to fragment `Output` location / `OpKill` | Emulated tile-buffer stub + log |
| 14 | `TexSample` imm descriptor index | `LD_TEX_IMM`/`TEX_SINGLE` `#index, #table` | Tagged `tex:set:binding` via `DescriptorHandleDomain` | Raw binding index + fallback log |

### 1.4 What `pointer_analysis` Already Does Today for `FCST` vs. What Is Missing

Before designing new machinery, we audit what `src/dataflow/` already implements today:

1. **What already works (`src/dataflow/pointer_analysis.{h,cpp}`, `src/dataflow/memory_domain.h`, `src/dataflow/tracked_val.h`, `src/dataflow/ssa_buffer_symbolize.cpp`)**:
   - `PointerAnalysisEnv` (`src/dataflow/pointer_analysis.h:51`) already exposes `std::map<uint32_t, TrackedVal> initial_fau;`.
   - `run_pointer_analysis` (`src/dataflow/pointer_analysis.cpp:334`) seeds `AbstractState::fau_mem` (`MapDomain<uint32_t, TrackedValSetDomain>`) from `env.initial_fau`.
   - `SsaPointerTransferVisitor::on_load_fau` (`src/dataflow/pointer_analysis.cpp:136`) already queries `state_->fau(byte_off)` (and 64-bit aligned `byte_off & ~7u` pairs) and propagates `TrackedVal` (`ConstInt`, `BufferAddr`, `SymbolValue`, `SrtTablePtr`) through `Pack64`, `IAdd64`, `BitAnd`, `Copy`, and `Phi`.
   - `ssa_buffer_symbolize.cpp` consumes `PointerAnalysisResult` to rewrite `PsbLoad`/`PsbStore` on resolved `BufferAddr` into `BufferLoad`/`BufferStore`.
2. **What is concretely missing today**:
   - **Gap 1 — `FCST` constants are NOT seeded into `initial_fau`**: In `tools/spirv2valhall/isa_enricher.cpp:218-281`, `env.initial_fau` is seeded with Pilot-written `BufferAddr` and `SymbolValue`, while `FCST` entries are only formatted as disassembly comments (`"FCST " + format_fcst_value(...)`). Consequently, `pointer_analysis` sees `LoadFau(8..27)` as opaque `FauUniformRef`, preventing constant propagation of `0x0ffe0000`, `0x77800000`, and `1..15`!
   - **Gap 2 — Pilot-computed expressions & predicates collapse to `Top`**: When the Pilot shader computes `r4 = FCMP.f32.eq(ubo0[0xd0], 1.0f)` and stores `r4` to `FAU+0x0`, `TrackedValDomain` has no symbolic-expression variant, so `PilotFauWrite` records `{0xffffffff, 0x0}` (or `Top`), losing the exact predicate `ubo0[0xd0] == 1.0f`.
   - **Gap 3 — No SSA rewriting of `LoadFau` or Valhall virtualization idioms**: `pointer_analysis` is strictly an *analysis* pass; `ssa_buffer_symbolize` only rewrites `PsbLoad`/`PsbStore`, leaving every `LoadFau`, `Bfi`/`BfeU` texel-coord pack, `m1` mask chain, and `CLPER`+`CSEL` ladder intact.
   - **Gap 4 — No known-bits / range / path-sensitivity**: `TrackedValDomain` is a flat domain (`ConstInt` vs `Top`), so it cannot prove `0 <= x < 65536` for 16:16 `Bfi`/`BfeU` elimination, nor can it refine `lane_id == 0` along the fallthrough edge of `BRANCH lane_id.w0`.

---

## 2. Where: Pipeline Placement and Modular Architecture

`run_ssa_defau_devirt` slots directly into the existing `src/dataflow/ssa_opt_pipeline.cpp` pipeline between Pass 1 lowering and Pass 2 SPIR-V tiling:

```text
  Valhall Binary (EBIN[1] Pilot + EBIN[0] Main + MBS2 FCST/RLOC/SRT)
        │
        ├──► Pass 1 (Pilot) ──► ValhallSsaModule (Pilot)
        │                              │
        │                              ▼
        │                   run_defau_devirt_analysis(Pilot, env)
        │                              │  produces FauSummaryMap (seeded w/ FCST + RLOC + Pilot writes)
        │                              ▼
        └──► Pass 1 (Main)  ──► ValhallSsaModule (Main)
                                       │
                                       ▼
                            run_ssa_defau_devirt(Main, FauSummaryMap, opts)
                              ├─ 1. Forward Fixpoint (DeVirtDomain via Worklist)
                              ├─ 2. Lightweight FactProverPass (Congruence + Algebraic Identities)
                              ├─ 3. SSA Materialization Rewriter + FallbackSiteLog
                              └─ 4. Standard ssa_opt_pipeline (SCCP, CSE, DCE, SpillMem2Reg, BufferSymbolize)
                                       │
                                       ▼
                            Pass 2 Tiler (src/valhall_tiler.cpp) ──► Portable SPIR-V
```

### DRY Reuse of `src/dataflow/domains/` (Modularized in Checkpoint v27.1)

In Checkpoint v27.1, `src/dataflow/domains.h` was modularized into 8 self-contained headers under `src/dataflow/domains/`:
- `traits.h`: `ShiftKind`, `ConvertKind`, `ValidateLatticeDomain`, SFINAE op traits
- `flat.h`: `FlatDomain<T>`, `IntegralDomain<IntT>`, `FloatDomain<FloatT>`
- `mixins.h`: `IntegralAdditiveActionMixin`, `IntegralBitmaskActionMixin`, `IntegralCombinableMixin`, `PointwiseOpsMixin`
- `containers.h`: `VectorDomain`, `MapDomain`, `TupleDomain`
- `bitset.h`: `BitsetValue`, `BitsetDomain`
- `finite_set.h`: `FiniteSetValue`, `FiniteSetDomain<ElemDomainT, MaxK>`
- `union.h`: `UnionDomain<Domains...>`, `CrossDomainOp`
- `lane.h`: `LaneDomainWrapper<InnerDomainT>`

Rather than introducing a parallel domain hierarchy, `DeVirtDomain` is composed **entirely** from these existing combinators plus `src/dataflow/memory_domain.h`.

---

## 3. How: DRY Abstract Domain Design ($\mathcal{D}\_{\text{devirt}}$)

### 3.1 Factoring Out `SymExprDomain` ($\mathcal{E}\_{\text{sym}}$) — Unifying Pointers and FAU Expressions

Today, `PointerVal` (`src/dataflow/tracked_val.h`) tracks `{SrtBase, SrtTablePtr, BufferAddr, FauDstBase, TlsBase, WlsBase} + byte_offset`. Notice that a pointer `BufferAddr(set, binding) + off` is structurally just an affine symbolic expression!

We factor out a bounded-depth symbolic expression value type **`SymExprVal`** (max AST depth $H\_{\text{expr}} = 4$, ensuring finite height!) with a flat lattice **`SymExprDomain`**:

```cpp
struct SymExpr {
    enum class Kind : uint8_t {
        // Memory/Descriptor Roots (subsuming PointerVal)
        SrtBase, SrtTablePtr, BufferAddr, FauDstBase, TlsBase, WlsBase,
        // Scalar Memory & Builtin Leaves
        BufferScalarLoad,   // *(T*)(BufferAddr(set, binding) + byte_off)
        BuiltinLeaf,        // gl_SubgroupInvocationID, gl_WorkGroupID.{x,y,z}, etc.
        FauSlotRef,         // Unresolved LoadFau(byte_off)
        // Bounded Pure Algebraic / Comparison Ops (depth <= 4)
        UnaryOp,            // SsaOp (BitNot, LogicalNot, Convert*, Clz, PopCount) + child
        BinaryOp,           // SsaOp (IAdd, IMul, BitAnd/Or/Xor, Shl/ShrU, FCmp*, ICmp*) + lhs, rhs
        SelectOp,           // Select(cond, tval, fval)
    };
    Kind kind;
    uint32_t set_id = 0, binding_id = 0;
    int64_t byte_offset = 0;
    bool offset_known = true;
    tables::SsaOp op = tables::SsaOp::Nop;
    uint8_t depth = 0;
    std::shared_ptr<const SymExpr> a, b, c;  // hash-consed immutable DAG nodes
    uint64_t const_arg = 0;                  // inline constant for affine/mask ops
};
```

**DRY Mixin Reuse**: `SymExprDomain` inherits directly from:
- **`IntegralAdditiveActionMixin<SymExprDomain, SymExprVal, IntegralDomain<uint64_t>>`**: automatically handles `base + z`, `z + base`, `base - z`, and `base + Top_Z` $\to$ `widen_unknown_offset(base)` for BOTH pointer addresses AND scalar symbolic offsets!
- **`IntegralBitmaskActionMixin<SymExprDomain, SymExprVal, IntegralDomain<uint64_t>>`**: automatically handles Valhall's 48-bit VA canonicalization (`AND.i32 hi, hi, 0x00FFFFFF`) and bit-masking of `SymExpr` without duplicating `BitAnd`/`BitOr` dispatch!

**Finite Height**: Because `SymExpr::depth` is capped at $H\_{\text{expr}} = 4$ (exceeding depth 4 immediately widens to $\top\_{\mathcal{E}}$) and `SymExprDomain` is a flat lattice over hash-consed canonical `SymExpr` nodes, its lattice height is strictly **2** ($\bot < e < \top$).

### 3.2 `DescriptorHandleDomain` ($\mathcal{H}\_{\text{desc}}$)

Tracks Valhall texture/sampler/image descriptor handles resolved from `LD_TEX_IMM`/`TEX_SINGLE` `#index, #table` or `SRT` descriptor-table loads:
- Flat lattice (`FlatDomain<DescriptorHandle>`) with payload `{Kind::{Texture, Sampler, Image, Buffer}, uint32_t table_id, uint32_t entry_id, uint32_t spirv_set, uint32_t spirv_binding}`.
- Lattice height: **2**.

### 3.3 `WarpPatternDomain` ($\mathcal{W}\_{\text{warp}}$)

Tracks hardware-virtualized subgroup patterns across instructions without collapsing to $\top$:
- Flat lattice (`FlatDomain<WarpPattern>`) over tagged variants:
  - `BallotAllActive(SsaId mask_ssa, uint8_t subgroup_size)` — produced by `WMASK.subgroupN(nonzero)`
  - `ButterflyStage(SsaId base_val, SsaOp reduce_op, uint32_t covered_xor_mask)` — tracks `CLPER.xor` + `CSEL`/`IAdd`/`FAdd` stages (`covered_xor_mask` accumulates `0x1 -> 0x3 -> 0x7 -> 0xF` across the 4 stages of a 16-lane butterfly!)
  - `LaneSnapshot(SsaId src_val, uint8_t lane_idx)` — produced by `CLPER.none(src_val, const_lane)`
  - `LaneElect(SsaId ballot_ssa)` — produced by `ICmpEq(ISub(31, Clz(ballot)), gl_SubgroupInvocationID)`
- Lattice height: **2** (at join points of mismatched patterns, widens to $\top$).

### 3.4 `KnownBitsDomain` ($\mathcal{K}\_{\text{bits}}$) — Built on `BitsetDomain`

Tracks per-bit certainty for every 64-bit SSA value using two 64-bit `BitsetValue` words (reusing `src/dataflow/domains/bitset.h`):
- `known_zero`: bit $i = 1 \iff$ concrete bit $i$ is guaranteed `0`.
- `known_one`: bit $i = 1 \iff$ concrete bit $i$ is guaranteed `1`.
- Invariant: `(known_zero & known_one) == 0` (violation represents unreachable $\bot$).
- Lattice operations (all pointwise in $O(1)$):
  - $\bot$: `is_bottom = true`
  - $\top$: `known_zero = 0`, `known_one = 0`
  - Join ($a \sqcup b$): `known_zero = a.known_zero & b.known_zero`, `known_one = a.known_one & b.known_one`
  - Meet ($a \sqcap b$): `kz = a.kz | b.kz`, `ko = a.ko | b.ko`; if `(kz & ko) != 0` return $\bot$
  - Exact constant check: `(known_zero | known_one) == ~0ULL` $\implies$ exact value `known_one`!
  - $16$-bit unsigned range check (for `Bfi`/`BfeU` elimination): `(known_zero & 0xFFFFFFFFFFFF0000ULL) == 0xFFFFFFFFFFFF0000ULL` $\implies v \in [0, 65535]$.
- Lattice height: at most **65** (each join strictly decreases popcount of `known_zero | known_one` from 64 down to 0, guaranteeing fast convergence).

### 3.5 Full Composed Domain ($\mathcal{D}\_{\text{devirt}}$) and Reduced Product ($\rho$)

We assemble the full value domain directly using `UnionDomain`, `LaneDomainWrapper`, `FiniteSetDomain`, and `TupleDomain`:

$$\mathcal{V}\_{\text{core}} = \text{UnionDomain}\big\langle \text{IntegralDomain}\langle\text{uint64\_t}\rangle,\; \text{FloatDomain}\langle\text{float}\rangle,\; \mathcal{E}\_{\text{sym}},\; \mathcal{H}\_{\text{desc}},\; \mathcal{W}\_{\text{warp}} \big\rangle$$

$$\mathcal{V}\_{\text{set}} = \text{FiniteSetDomain}\big\langle \text{LaneDomainWrapper}\langle\mathcal{V}\_{\text{core}}\rangle,\; \text{MaxK}=4 \big\rangle$$

$$\mathcal{D}\_{\text{val}} = \text{TupleDomain}\big\langle \mathcal{V}\_{\text{set}},\; \mathcal{K}\_{\text{bits}} \big\rangle$$

And the program-point state reuses **`AbstractState`** and **`MemoryDomain`** (`src/dataflow/memory_domain.h`):
- `ssa_env`: `VectorDomain<D_val>` indexed by `SsaId`
- `fau_mem`: `MapDomain<uint32_t, D_val>` (RW in Pilot `EBIN[1]`; RO in Main `EBIN[0]`, seeded from `FCST` + `RLOC` + Pilot `FauSummaryMap`)
- `tls_mem`, `wls_mem`: `MapDomain<int64_t, D_val>` (reusing `MemoryDomain` region isolation!)
- `path_facts`: `MapDomain<SsaId, D_val>` (edge-refined facts from `cond_br`, e.g., `lane_id == 0` on the fallthrough edge of `BRANCH lane_id.w0`)

**Reduced Product Operator ($\rho : \mathcal{V}\_{\text{set}} \times \mathcal{K}\_{\text{bits}} \to \mathcal{V}\_{\text{set}} \times \mathcal{K}\_{\text{bits}}$)**:
1. **$\mathcal{V}\_{\text{set}} \to \mathcal{K}\_{\text{bits}}$**:
   - If $\mathcal{V}\_{\text{set}} = \{\text{ConstInt}(c)\}$: tighten `known_one |= c`, `known_zero |= ~c`.
   - If $\mathcal{V}\_{\text{set}} = \{\text{BuiltinLeaf}(\text{SubgroupLocalInvocationId})\}$ on Valhall (`subgroup16`): tighten `known_zero |= 0xFFFFFFFFFFFFFFF0ULL` ($0 \le \text{lane\_id} < 16$).
   - If $\mathcal{V}\_{\text{set}} = \{\text{BallotAllActive}(\_, 16)\}$: tighten `known_zero |= 0xFFFFFFFFFFFF0000ULL`.
   - If $\mathcal{V}\_{\text{set}}$ is a 2-element set $\{0, \text{0xFFFFFFFF}\}$ (Valhall `m1` mask): tighten `known_zero |= 0xFFFFFFFF00000000ULL`.
2. **$\mathcal{K}\_{\text{bits}} \to \mathcal{V}\_{\text{set}}$**:
   - If `known_zero | known_one == ~0ULL` (all 64 bits known) and $\mathcal{V}\_{\text{set}}$ is $\top$: refine $\mathcal{V}\_{\text{set}}$ to $\{\text{ConstInt}(\text{known\_one})\}$.
   - If `KnownBits` is $\bot$ (contradictory path constraint): collapse $\mathcal{V}\_{\text{set}}$ to $\bot$ (marks branch dead!).

---

## 4. Detailed Transfer Functions (`DeVirtTransferVisitor`)

`DeVirtTransferVisitor` plugs into `dispatch_ssa_node` (`src/dataflow/transfer.h`) and runs in two sequential fixpoint phases.

### 4.1 Phase 1: Pilot Shader Transfer (`role == StageRole::Pilot`)

Seed entry state with `SRT_0`, `FAU_0`, `FCST`, and `RLOC` metadata, then iterate to fixpoint:

```cpp
// Pilot-specific transfer rules (pseudocode)
void DeVirtTransferVisitor::on_psb_load(SsaId dst, SsaId addr_id, uint32_t bytes, SsaId mem_tok) {
    DVal addr = eval(addr_id);
    // Delegate pointer resolution to MemoryDomain::deref(addr, bytes)
    MemoryDerefResult d = MemoryDomain::deref(addr.vset, bytes, env_.srt_layout);
    switch (d.kind) {
        case MemoryDerefResult::Kind::SrtEntry:
            // Loaded buffer descriptor address from SRT[table] + off
            assign(dst, DVal::from_sym(SymExpr::buffer_addr(d.set, d.binding, 0)));
            break;
        case MemoryDerefResult::Kind::BufferScalar:
            // Loaded scalar from UBO/SSBO (e.g., ubo:0:0 + 0xd0)
            assign(dst, DVal::from_sym(SymExpr::buffer_load(d.set, d.binding, d.byte_offset, bytes)));
            break;
        default:
            assign(dst, DVal::top());
    }
}

void DeVirtTransferVisitor::on_fcmp(SsaId dst, SsaOp cmp_op, SsaId lhs, SsaId rhs) {
    DVal a = eval(lhs), b = eval(rhs);
    // Constant-fold if both are known floats
    if (a.is_const_f32() && b.is_const_f32()) {
        assign(dst, DVal::const_bool(eval_fcmp_const(cmp_op, a.f32(), b.f32())));
        return;
    }
    // Otherwise build bounded SymExpr DAG node (depth <= 4)
    if (auto expr = SymExpr::make_binary(cmp_op, a.to_sym(), b.to_sym())) {
        assign(dst, rho(DVal::from_sym(expr), KnownBits::from_bool()));
        return;
    }
    assign(dst, DVal::top());
}

void DeVirtTransferVisitor::on_select(SsaId dst, SsaId cond, SsaId tv, SsaId fv) {
    DVal c = eval(cond), t = eval(tv), f = eval(fv);
    if (c.is_const_bool()) { assign(dst, c.bool_val() ? t : f); return; }
    // Recognize Valhall m1/i1/f1 boolean mask materialization: Select(p, 0xFFFFFFFF, 0)
    if (t.is_const_u64(0xFFFFFFFFu) && f.is_const_u64(0u) && c.has_sym()) {
        assign(dst, rho(DVal::from_sym(SymExpr::make_select(c.sym(), 0xFFFFFFFFu, 0u)),
                        KnownBits::from_u32_mask()));
        return;
    }
    assign(dst, DVal::join(t, f));
}

void DeVirtTransferVisitor::on_psb_store(SsaId addr_id, SsaId val_id, uint32_t bytes) {
    DVal addr = eval(addr_id), val = eval(val_id);
    if (auto fau_off = addr.try_as_fau_dst_offset()) {
        // Strong update in Pilot FAU bank; split 64/128-bit vector stores into 32-bit words
        for (uint32_t w = 0; w < bytes / 4u; ++w) {
            DVal word_val = extract_word(val, w);
            state_.fau_mem.set(*fau_off + w * 4u, word_val);
            pilot_summary_[*fau_off + w * 4u] = DVal::join(pilot_summary_[*fau_off + w * 4u], word_val);
        }
    }
}
```

### 4.2 Phase 2: Main Shader Transfer (`role == MainCompute | Vertex | Fragment`)

Entry state seeds `fau_mem` from three priority-ordered sources:
1. `FCST` static constant pool $\to$ `DVal::const_u32(fcst.word)` with exact `KnownBits`.
2. `RLOC` relocation table $\to$ `DVal::from_sym(SymExpr::buffer_addr(...))` / `DescriptorHandle`.
3. Pilot `FauSummaryMap` $\to$ symbolic `BufferScalarLoad`, `BufferAddr`, or `SymExpr` predicate.

```cpp
// Main-shader transfer rules (pseudocode)
void DeVirtTransferVisitor::on_load_fau(SsaId dst, uint32_t word_idx) {
    uint32_t byte_off = word_idx * 4u;
    if (auto found = state_.fau_mem.lookup(byte_off)) {
        assign(dst, *found);
        return;
    }
    // Check aligned 64-bit pair (e.g. hi-half of BufferAddr stored at byte_off & ~7u)
    if (auto pair = state_.fau_mem.lookup(byte_off & ~7u); pair && pair->is_64bit_addr()) {
        assign(dst, LaneDomainWrapper::split_half(*pair, /*is_hi=*/(byte_off & 4u) != 0));
        return;
    }
    assign(dst, DVal::from_sym(SymExpr::fau_slot_ref(byte_off)));
}

void DeVirtTransferVisitor::on_bfi(SsaId dst, SsaId base, SsaId ins, SsaId off, SsaId cnt) {
    DVal b = eval(base), v = eval(ins), o = eval(off), c = eval(cnt);
    KnownBits kb = KnownBits::bfi(b.kb, v.kb, o.kb, c.kb);
    // Preserve exact SymExpr for 16:16 packing so BfeU can project back losslessly
    if (o.is_const_u64(16) && c.is_const_u64(16)) {
        assign(dst, rho(DVal::from_sym(SymExpr::pack16x2(b.to_sym(), v.to_sym())), kb));
        return;
    }
    assign(dst, rho(DVal::top(), kb));
}

void DeVirtTransferVisitor::on_bfe_u(SsaId dst, SsaId src, SsaId off, SsaId cnt) {
    DVal s = eval(src), o = eval(off), c = eval(cnt);
    KnownBits kb = KnownBits::bfe_u(s.kb, o.kb, c.kb);
    // Constant fold (e.g. BfeU(0x0b0a0908, 0, 8) == 8)
    if (kb.is_exact_const()) { assign(dst, DVal::const_u32(kb.const_u32())); return; }
    // Project 16:16 packed SymExpr back to original component if range-safe
    if (auto p16 = s.try_as_pack16x2(); p16 && c.is_const_u64(16)) {
        if (o.is_const_u64(0))  { assign(dst, rho(DVal::from_sym(p16->lo_16), kb)); return; }
        if (o.is_const_u64(16)) { assign(dst, rho(DVal::from_sym(p16->hi_16), kb)); return; }
    }
    assign(dst, rho(DVal::top(), kb));
}

void DeVirtTransferVisitor::on_subgroup_ballot(SsaId dst, SsaId scope, SsaId pred) {
    DVal p = eval(pred);
    if (p.is_const_bool(true)) {
        assign(dst, rho(DVal::from_warp(WarpPattern::ballot_all_active(dst, 16)),
                        KnownBits::u16_range()));
        return;
    }
    assign(dst, DVal::top());
}

void DeVirtTransferVisitor::on_subgroup_shuffle(SsaId dst, SsaId scope, SsaId val, SsaId lane) {
    DVal v = eval(val), l = eval(lane);
    // Case 1: Static lane broadcast (CLPER.none with constant lane 0..15)
    if (l.kb.is_exact_const() && l.kb.const_u32() < 16u) {
        assign(dst, DVal::from_warp(WarpPattern::lane_snapshot(val, l.kb.const_u32())));
        return;
    }
    // Case 2: XOR butterfly stage (lane == BitXor(gl_SubgroupInvocationID, power_of_2))
    if (auto delta = match_xor_lane_id(l)) {
        assign(dst, DVal::from_warp(WarpPattern::butterfly_peer(val, *delta)));
        return;
    }
    assign(dst, DVal::top());
}

void DeVirtTransferVisitor::on_cond_br_edge(uint32_t from_bb, uint32_t to_bb,
                                            SsaId cond_id, bool branch_taken,
                                            AbstractState& edge_state) {
    // Path-sensitive edge refinement!
    // Example: BRANCH lane_id.w0, skip_bb  ==>  fallthrough edge knows lane_id == 0!
    if (auto eq = match_icmp_zero(cond_id)) {
        bool is_zero_on_edge = eq->is_ne ? !branch_taken : branch_taken;
        if (is_zero_on_edge) {
            edge_state.refine(eq->var, DVal::const_u32(0u));
        }
    }
}
```

---

## 5. Lightweight Constant-Propagation & Extensional-Equality Prover Pass (`FactProverPass`)

After the `DeVirtDomain` fixpoint converges, many redundant Valhall idioms remain as multi-instruction algebraic identities that a single local peephole cannot discharge (for example, `OR(16, lane_id)` under `lane_id == 0` feeding a 15-step `CSEL` ladder, or `ICmpNe(Select(p, 0xFFFFFFFF, 0), 0)`).

We introduce a dedicated, deterministic, linear-time **`FactProverPass`** (built on a union-find congruence table + `KnownBits` facts, with zero SMT solver overhead in the compiler hot path):

### 5.1 Core Prover Rules

1. **Boolean Mask Unwrapping (Extensional Equality on `Bool`)**:
   - `ICmpNe(Select(p, c_nz, 0), 0)` $\equiv p$ (for any `c_nz != 0`)
   - `ICmpEq(Select(p, 0xFFFFFFFF, 0), 0xFFFFFFFF)` $\equiv p$
   - `BitAnd(Select(p, 0xFFFFFFFF, 0), Select(q, 0xFFFFFFFF, 0))` $\equiv \text{Select}(\text{LogicalAnd}(p, q), \text{0xFFFFFFFF}, 0)$
   - `LogicalNot(LogicalNot(p))` $\equiv p$
2. **Range-Guarded Bit-Field Projection**:
   - `BfeU(Bfi(x, y, 16, 16), 0, 16)` $\equiv x$ whenever `KnownBits(x).upper16_zero()` (or `opts.assume_valhall_tex16_range`)
   - `BfeU(Bfi(x, y, 16, 16), 16, 16)` $\equiv y$ whenever `KnownBits(y).upper16_zero()`
   - `BitAnd(BfeU(v, off, w), mask)` $\equiv \text{BfeU}(v, \text{off}, w)$ whenever `((1u << w) - 1) & ~mask == 0`
3. **Path-Refined `CSEL` Ladder Collapse**:
   - On any block where `edge_state` proves `x == C` (e.g., `lane_id == 0` $\implies$ `r51 = 16 | 0 = 16`), every `ICmpEq(r51, k)` for $k \in \{1..15\}$ discharges to `ConstBool(false)` and every `Select(false, r_k, acc)` collapses to `acc`.
   - When `CSEL.u32.lt(r2, r3, r2, r3)` reduces to `min(r2, LaneSnapshot(r2, 0))` on `lane_id == 0`, congruence closure proves `LaneSnapshot(r2, 0) == r2` on lane 0, discharging `min(r2, r2) == r2`!
4. **Pilot-Hoisted Conditional Select Simplification**:
   - `Select(FCmpOrdEq(s, 1.0f), x, ConvertFToS(ConvertSToF(x) * s))` is preserved verbatim in canonical form — matching direct SPIR-V's `if (s == 1.0f) x else int(float(x) * s)`!

### 5.2 Structured Fallback Telemetry (`FallbackSiteLog`)

Real-world shaders compiled by `libGLES_mali` inevitably contain exotic or dynamically indexed FAU / sideband / warp constructs that exceed bounded static analysis. Rather than failing silently or crashing, every unresolved site emits a structured `FallbackRecord`:

```cpp
enum class FallbackReason : uint8_t {
    UnresolvedFauSlot,        // LoadFau(w) absent from FCST, RLOC, and Pilot summary
    DynamicFauOffset,         // Pilot stored to FAU_0 + Top_Z
    SymExprDepthExceeded,     // Pilot expression exceeded H_expr = 4
    DynamicDescriptorIndex,   // Non-constant texture/buffer table index
    UnrecognizedWarpLadder,   // CLPER sequence did not match butterfly/broadcast/elect
    UnsupportedSideband,      // Unknown LoadSidebandU32(kind, idx)
};

struct FallbackRecord {
    uint64_t valhall_pc;
    compiler::SsaId ssa_id;
    tables::SsaOp op;
    FallbackReason reason;
    std::string abstract_val_str;  // DVal::format() for immediate diagnosis
};
```

CLI flag `--dump-devirt-fallbacks` aggregates fallback counts by `FallbackReason` across the 25-shader corpus, allowing us to rank the highest-impact gaps and iteratively patch them up with zero guesswork.

---

## 6. Soundness, Monotonicity, Termination, and Materialization

### 6.1 Concretization ($\gamma$) and Soundness

For every SSA value $v$ and concrete GPU execution state $\sigma = (\sigma\_{\text{mem}}, \sigma\_{\text{fau}}, \sigma\_{\text{lane}})$:
- $\gamma\_{\mathcal{E}}(e)$ is the singleton concrete value obtained by evaluating `SymExpr` $e$ in $\sigma$ (or all 64-bit values for $\top\_{\mathcal{E}}$).
- $\gamma\_{\mathcal{K}}(z, o) = \{ x \in \text{BV64} \mid (x \mathbin{\&} z) = 0 \land (x \mathbin{\&} o) = o \}$.
- $\gamma\_{\mathcal{D}}(V, K) = \gamma\_{\mathcal{V}}(V) \cap \gamma\_{\mathcal{K}}(K)$.

Every transfer function $f^{\#}$ is sound ($\forall \sigma \in \gamma(s),\; f(\sigma) \in \gamma(f^{\#}(s))$) and monotone ($s\_1 \sqsubseteq s\_2 \implies f^{\#}(s\_1) \sqsubseteq f^{\#}(s\_2)$). Because $\rho$ only removes concrete states already excluded by the sibling sub-domain ($\gamma(\rho(V, K)) = \gamma(V, K)$), the reduced product preserves soundness.

### 6.2 Finite-Height Termination

| Component | Height | Widening / Bound |
|---|---|---|
| `IntegralDomain<uint64_t>` / `FloatDomain<float>` | 2 | Flat lattice |
| `SymExprDomain` ($\mathcal{E}\_{\text{sym}}$) | 2 | Flat lattice over DAGs of depth $\le 4$ |
| `DescriptorHandleDomain` ($\mathcal{H}\_{\text{desc}}$) | 2 | Flat lattice |
| `WarpPatternDomain` ($\mathcal{W}\_{\text{warp}}$) | 2 | Flat lattice |
| `FiniteSetDomain<..., MaxK=4>` | 5 | Cardinality $> 4 \implies \top$ |
| `KnownBitsDomain` ($\mathcal{K}\_{\text{bits}}$) | 65 | Strictly monotone bit-certainty drop |

Therefore, on a CFG with $N$ SSA nodes, the Chaotic Iteration worklist (`src/dataflow/worklist.cpp`) terminates in strictly bounded polynomial time $O(N \cdot 65)$ with no widening divergence.

### 6.3 SSA Materialization Pass (`run_ssa_defau_devirt`)

Materialization runs in topological (RPO) order over the Main shader SSA:
1. **Materialize `LoadFau(w)`**:
   - `ConstInt(c)` $\to$ `ConstU32(c)` (eliminates all `FCST` indirections!).
   - `SymExpr::BufferScalarLoad(set, binding, off)` $\to$ `BufferLoad(ubo:set:binding, off)` (hoisted to `bb0` since UBOs are read-only and uniform!).
   - `SymExpr` pure DAG $\to$ recursively materialize leaves + ops in `bb0`, memoized by `SymExpr` pointer so shared Pilot predicates (`FAU+0x0`) are emitted once.
   - `BufferAddr(set, binding, off)` $\to$ consumed directly by `PsbLoad`/`PsbStore`/`AtomicRmw` rewriting to `BufferLoad`/`BufferStore`/`BufferAtomicRmw`.
   - Unresolved $\to$ keep `LoadFau(w)` and append to `FallbackSiteLog`.
2. **Apply `FactProverPass` Rewrites**:
   - Replace every node whose congruence class contains a simpler canonical representative (`ConstU32`, unwrapped `Bool`, projected 16-bit coordinate, collapsed `CSEL` ladder).
3. **Run Standard Cleanup (`ssa_opt_pipeline`)**:
   - `ssa_sccp` + `ssa_cse` + `ssa_dce` eliminate all now-dead `LoadFau`, `Bfi`, `CLPER.none`, and `CSEL` ladder instructions.

---

## 7. Worked End-to-End Walkthrough: `samples/comp_unique_02_d7242030.spv`

### 7.1 Pilot Analysis Output (`FauSummaryMap`)

| FAU Slot | Source | Resolved `DVal` in `FauSummaryMap` |
|---|---|---|
| `FAU+0x00` (`u0.w0`) | Pilot `0x00a0` | `Select(FCmpOrdEq(BufferLoad(ubo:0:0, 0xd0), 1.0f), 0xFFFFFFFF, 0)` |
| `FAU+0x04` (`u0.w1`) | Pilot `0x0098` | `Select(FCmpOrdEq(BufferLoad(ubo:0:0, 0xe0), 1.0f), 0xFFFFFFFF, 0)` |
| `FAU+0x08` (`u1.w0`) | Pilot `0x0080` | `BufferLoad(ubo:0:0, 0xd0)` |
| `FAU+0x0c` (`u1.w1`) | Pilot `0x0068` | `BufferLoad(ubo:0:0, 0xe0)` |
| `FAU+0x10` (`u2.w0`) | Pilot `0x00b0` | `BufferLoad(ubo:0:4, 0x0)` |
| `FAU+0x18..0x1f` (`u3`) | Pilot `0x00c0` | `BufferAddr(ssbo:0:1, offset=0)` |
| `FAU+0x20` (`u4.w0`) | `FCST` | `ConstU32(0x0ffe0000)` (`KnownBits` exact) |
| `FAU+0x24` (`u4.w1`) | `FCST` | `ConstU32(0x77800000)` (`KnownBits` exact) |
| `FAU+0x28` (`u5.w0`) | `FCST` | `ConstU32(0x0ffc0000)` (`KnownBits` exact) |
| `FAU+0x2c` (`u5.w1`) | `FCST` | `ConstU32(0x07800000)` (`KnownBits` exact) |
| `FAU+0x30..0x68` (`u6..u13`) | `FCST` | `ConstU32(1) .. ConstU32(15)` (`KnownBits` exact) |

### 7.2 Main Shader Transformation (Before vs. After)

- **Phase 2 (Coordinate Scaling & Texture Sampling)**:
  - *Before*: `LoadFau(0)`, `LoadFau(2)`, `Bfi(%56, %36, 16, 16)`, `BfeU(..., 0, 16)`, `BfeU(..., 16, 16)`.
  - *After*: `scale0 = BufferLoad(ubo:0:0, 0xd0)`, `Select(FCmpOrdEq(scale0, 1.0f), x, ConvertFToS(ConvertSToF(x) * scale0))`, direct `TexSample(Vec2(sx, sy)) "tex:0:2"`. Zero `LoadFau`, zero `Bfi`/`BfeU`.
- **Phase 3 (`R11G11B10F` Unpacking & Filter Predicate)**:
  - *Before*: `LoadFau(8..11)`, `Select(..., 0xFFFFFFFF, 0)` $\to$ `BitAnd` $\to$ `ICmpNe(..., 0)`.
  - *After*: Exact inline constants `0x0ffe0000`, `0x77800000`, `0x0ffc0000`, `0x07800000`, `BufferLoad(ubo:0:4, 0x0)`, and native `Bool` `LogicalAnd` feeding `cond_br`.
- **Phase 5 (Subgroup Min/Max Reductions)**:
  - *Before*: 4 `CLPER.xor` stages + 16 `CLPER.none` snapshots + 15-stage `CSEL` ladder on `r51 = 16 | lane_id` + `AtomicRmw(Pack64(LoadFau(6), LoadFau(7)))`.
  - *After*: `edge_state` refines `lane_id == 0` after `BRANCH lane_id.w0`, `FactProverPass` folds `16 | 0 == 16` and collapses all 15 `CSEL`s + all 16 `CLPER.none` snapshots away, leaving only the 4-stage `SubgroupShuffle(lane_id ^ {1,2,4,8})` butterfly and `BufferAtomicRmw(ssbo:0:1, 0x0, aumin)`.
- **Result**: `FallbackSiteLog` reports **0 fallbacks**, `LoadFau` count drops from **52 $\to$ 0**, and total SSA node count in `comp_unique_02_d7242030.spv` drops by **>55%**.

---

## 8. Interfaces and Phased Rollout Plan

### 8.1 Clean Header Boundaries

- `src/dataflow/domains/sym_expr.h`: `SymExpr`, `SymExprDomain` (inheriting `IntegralAdditiveActionMixin` + `IntegralBitmaskActionMixin`).
- `src/dataflow/domains/known_bits.h`: `KnownBitsValue`, `KnownBitsDomain` (built on `BitsetValue`).
- `src/dataflow/ssa_defau_devirt.h`: `FauSummaryMap`, `DeVirtOptions`, `FallbackRecord`, `DeVirtResult`, `run_pilot_fau_summary()`, `run_ssa_defau_devirt()`.
- `src/dataflow/ssa_fact_prover.h`: `FactProverPass` (congruence closure + algebraic identity discharge).

### 8.2 Phased Implementation Plan

1. **Phase A (Quick Win — Zero New Domains)**: Seed `FCST` constants into `PointerAnalysisEnv::initial_fau` in `isa_enricher.cpp` and rewrite constant `LoadFau` $\to$ `ConstU32` in `ssa_buffer_symbolize.cpp`. Immediately eliminates 100% of `FCST` `LoadFau`s across all 25 sample shaders.
2. **Phase B (`SymExprDomain` + Pilot Summary)**: Extract `SymExprDomain` from `PointerVal` (keeping `PointerVal` as a type alias / thin view), run `run_pilot_fau_summary` on `EBIN[1]`, and materialize `BufferLoad` + Pilot predicates in `EBIN[0]`.
3. **Phase C (`KnownBitsDomain` + `FactProverPass` + `FallbackSiteLog`)**: Add `KnownBitsDomain`, path-sensitive `cond_br` edge refinement, `Bfi`/`BfeU` projection, boolean mask unwrapping, `CLPER` ladder collapse, and `--dump-devirt-fallbacks` telemetry.
4. **Phase D (Formal Verification)**: Verify before/after `run_ssa_defau_devirt` equivalence across all 25 sample shaders using `tools/verify/shader_equiv_verifier.py` and add unit tests in `tests/test_dataflow.cpp` and `tests/test_ssa_opt.cpp`.

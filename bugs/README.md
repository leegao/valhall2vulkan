# `valhall2spirv` Bug Catalog (Checkpoints v47.2 – v48.3)

This directory catalogues **all 20 distinct lowering, optimizer, and modeling bugs** root-caused during the v47.2–v48.3 full 142-case ISA/variant sweep and longitudinal compositional fuzzing campaign.

- **Sweep Progression:** `86/142` (60.6%, v47.2) $\to$ `118/142` (83.1%, v48) $\to$ `122/142` (85.9%, v48.1) $\to$ `126/142` (88.7%, v48.2) $\to$ **`129/142` (90.8%, v48.3)** — **+43 targets fixed, 0 regressions**.
- **Longitudinal Compositional Suite (`tools/corpus_synth/longitudinal_fuzz.py`):** **12/12** multi-instruction composed shaders pass `EQUIV_EXACT` (v48.3); **2** newly uncovered compositional bugs (`BUG-019`, `BUG-020`, both fixed in v48.3).

## Master Index

| ID | Cluster | Status | Layer | Targets Affected | Report |
|---|---|---|---|---|---|
| **BUG-001** | `RC-1` | **FIXED (v48)** | `spirv2ssa` | 13 (`v2f16` ops & `f16` casts) | [`BUG-001-bitcast-f16-unpack.md`](BUG-001-bitcast-f16-unpack.md) |
| **BUG-002** | `RC-2` | **FIXED (v48)** | `spirv2ssa` | 6 (`32<->8` casts, `MKVEC`, `IABS`) | [`BUG-002-bitcast-32-8-missing.md`](BUG-002-bitcast-32-8-missing.md) |
| **BUG-003** | `RC-3` | **FIXED (v48)** | `spirv2ssa` | 3 (`S16_TO_F32`, `IABS.v2s16`, `IADD.v4u8`) | [`BUG-003-subword-signext-mask.md`](BUG-003-subword-signext-mask.md) |
| **BUG-004** | `RC-4` | **FIXED (v48)** | SSA Opt | 5 (`ATOM`, `ATOM1`, `BRANCH`, `MOV`, `MOV_IMM`) | [`BUG-004-gvn-imul-shl-dominance.md`](BUG-004-gvn-imul-shl-dominance.md) |
| **BUG-005** | `RC-9` | **FIXED (v48)** | `spirv2ssa` + Pass1 | 2 (`IDPADD`, `IDPADD.v4u8`; Z3 `UNSAT`) | [`BUG-005-idpadd-dotaccsat-signedness.md`](BUG-005-idpadd-dotaccsat-signedness.md) |
| **BUG-006** | `RC-10d` | **FIXED (v48)** | Pass1 | 2 (`TEX_FETCH`, `LD_TEX`) | [`BUG-006-tex-fetch-lod-and-idx.md`](BUG-006-tex-fetch-lod-and-idx.md) |
| **BUG-007** | `RC-11` | **FIXED (v48)** | Pass1 | 1 (`ACMPXCHG`) | [`BUG-007-acmpxchg-swapped-sr.md`](BUG-007-acmpxchg-swapped-sr.md) |
| **BUG-008** | `RC-12` | **FIXED (v48)** | Pass1 | 1 (`ICMP_MULTI` carry `%2`) | [`BUG-008-icmp-multi-missing-carry.md`](BUG-008-icmp-multi-missing-carry.md) |
| **BUG-009** | `RC-10a/b` + `NEW-1` | **FIXED (v48.1)** | Pass1 | 2 (`TEX_GATHER`, `VAR_TEX_GATHER`) | [`BUG-009-gather4-channel-perm.md`](BUG-009-gather4-channel-perm.md) |
| **BUG-010** | `RC-7` + `NEW-3` + `NEW-4` | **FIXED (v48.1)** | Pass1 + `spirv2ssa` + Interp | 2 (`LD_ATTR`, `LEA_BUF_IMM`) | [`BUG-010-waterfall-and-dyn-inputvar.md`](BUG-010-waterfall-and-dyn-inputvar.md) |
| **BUG-011** | `NEW-5` | **FIXED (v48.2)** | ShaderEnv | 2 (`LD_VAR`, `LD_VAR_FLAT`) | [`BUG-011-sinp-array-slot-expansion.md`](BUG-011-sinp-array-slot-expansion.md) |
| **BUG-012** | `RC-13c` | **FIXED (v48.2)** | ShaderEnv | 1 (`LD_VAR_SPECIAL` / `FragCoord.xy`) | [`BUG-012-fragcoord-samplepos-invariant.md`](BUG-012-fragcoord-samplepos-invariant.md) |
| **BUG-013** | `RC-10e` | **FIXED (v48.2)** | Pass1 | 1 (`LEA_TEX`) | [`BUG-013-lea-tex-dynamic-handle.md`](BUG-013-lea-tex-dynamic-handle.md) |
| **BUG-014** | `NEW-2` | **OPEN** | `spirv2ssa` / MBS2 | 1 (`ICMP_MULTI` vertex varying slot map) | [`BUG-014-open-remaining-clusters.md`](BUG-014-open-remaining-clusters.md) |
| **BUG-015** | `RC-5` | **OPEN** | Equiv Model | 5 (`CUBEFACE1/2`, `CUBE_SSEL/TSEL`, `FMA_RSCALE`) | [`BUG-014-open-remaining-clusters.md`](BUG-014-open-remaining-clusters.md) |
| **BUG-016** | `RC-6` | **FIXED (v48.3, 3/4)** | Fuzzer ULP | 4 (`FADD_IMM`, `FSIN_TABLE`, `FCOS_TABLE`, `FMUL_RSCALE`) | [`BUG-014-open-remaining-clusters.md`](BUG-014-open-remaining-clusters.md) |
| **BUG-017** | `RC-8` | **OPEN** | Pass1 | 3 (`FATAN_ASSIST`, `FATAN_TABLE`, `MUX`) | [`BUG-014-open-remaining-clusters.md`](BUG-014-open-remaining-clusters.md) |
| **BUG-018** | `RC-10c` + `RC-13a/b` | **OPEN** | Pass1 / Interp | 3 (`TEX_GRADIENT`, `CLPER`, `LD_PKA`) | [`BUG-014-open-remaining-clusters.md`](BUG-014-open-remaining-clusters.md) |
| **BUG-019** | `COMP-1` | **FIXED (v48.3)** | `spirv2ssa` | Longitudinal `comp_05`, `comp_09` (`ICmp*` 64-bit truncation) | [`BUG-019-comp-64bit-icmp-truncation.md`](BUG-019-comp-64bit-icmp-truncation.md) |
| **BUG-020** | `COMP-2` | **FIXED (v48.3)** | Pass1 + `spirv2ssa` | Longitudinal `comp_01`, `comp_11` (`LOAD.i16/i8` subword lane merge + `f16` quantization) | [`BUG-020-comp-v2f16-fma-rounding.md`](BUG-020-comp-v2f16-fma-rounding.md) |

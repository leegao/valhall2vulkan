# Valhall ISA Divergence Triage Playbook (`tools/corpus_synth/triage.md`)

This playbook defines the deterministic, step-by-step root-cause workflow for triaging any `equiv=DIVERGED` failure reported by `tools.corpus_synth --target-isa ... --verify-equiv`, followed by an analysis of remaining debugging gaps and tool proposals.

---

## Part 1: Step-by-Step Root-Cause Process (Instruction -> Root Cause)

### Step 1: Isolate & Dump the Minimal Repro with `--slice-diff`
Run the failing mnemonic alone with `--slice-diff 6`, `--output-shaders`, and `--fp-range vulkan`:
```bash
python3 -m tools.corpus_synth.cli \
  --target-isa <MNEMONIC> \
  --mode spirv \
  --verify-equiv \
  --k-init 16 \
  --fp-range vulkan \
  --slice-diff 6 \
  --output-shaders /tmp/triage_<mn>
```
This immediately produces:
1. The exact `.spvasm` + `.spv` at `/tmp/triage_<mn>/isa_<mn>_minimal.spvasm`.
2. The failing `(vec_idx, output_name, direct_val, valhall_val)` witness and `frontier_pcs`.
3. The 6-hop side-by-side Direct vs Valhall concrete backward slice (`[SLICE-DIFF]`) highlighting `== MATCH` vs `!= DIFF` vs `<< FIRST-DIVERGE`.

### Step 2: Classify the Failure Bucket in 30 Seconds
Look at the `[MISMATCH]` + `[SLICE-DIFF]` output and classify into one of 5 root-cause buckets:

| Signature | Bucket | Where to Look |
|---|---|---|
| `direct=0x...` matches input/constant, `valhall=None` (or vice versa) | **A. Channel / ABI / Output Sink Asymmetry** | `tools/symbolic_ssa/concrete_interpreter.py`, `tools/spirv2valhall/equiv_bundle.cpp` |
| Direct slice lacks the SPIR-V op (passes input straight through) while Valhall computes it | **B. `spirv2ssa` Missing / Incomplete Op Lowering** | `src/spirv2ssa/spirv2ssa_ops.cpp`, `src/spirv2ssa/spirv2ssa_mem.cpp` |
| Direct & Valhall inputs match (`== MATCH`), but a single Valhall PC produces the wrong concrete value (`<< FIRST-DIVERGE`) | **C. Valhall ISA -> SSA Lowering Rule Bug** | `specs/lowering/valhall_lowering.json`, `src/valhall_pass1_handlers.cpp` |
| Raw SSA agrees (`--no-canonicalize` passes), canonicalized SSA diverges | **D. SSA Canonicalizer / `ssa-opt` Bug** | `tools/symbolic_ssa/passes.py`, `src/dataflow/ssa_*.cpp` |
| `ulp_diff` is small (1–3 ULP) on `Sin`/`Cos`/`Pow`/`Atan2` | **E. Expected Transcendental Precision / ULP Tolerance** | `tools/coverage/shrinker.py` (`DiffPolicy`) or `tools/symbolic_ssa/concrete_interpreter.py` |

### Step 3: Deep Inspection & Shrinking via `tools.coverage`
Run `tools.coverage` on the dumped `.spvasm` with `--trace-diff`, `--slice-diff 8`, and `--shrink` (which invokes `spirv-reduce` automatically):
```bash
python3 -m tools.coverage.cli \
  /tmp/triage_<mn>/isa_<mn>_minimal.spvasm \
  --equiv-fuzz \
  --k-init 16 \
  --fp-range vulkan \
  --trace-diff \
  --slice-diff 8 \
  --shrink
```
And inspect the raw disassembly + SSA side-by-side if needed:
```bash
./build/spirv2valhall /tmp/triage_<mn>/isa_<mn>_minimal.spv --mods
./build/spirv2valhall /tmp/triage_<mn>/isa_<mn>_minimal.spv --direct-ssa --opt-ssa
./build/spirv2valhall /tmp/triage_<mn>/isa_<mn>_minimal.spv --dump-ssa --defau --opt-ssa
```

### Step 4: Concrete Worked Triage Examples from the v47 Sweep

1. **`MKVEC` (Bucket B — `spirv2ssa` missing `OpVectorShuffle` on `v4u8`)**:
   - `--slice-diff 4` shows at `h1`:
     - `D: %36 TupleExtract = 0x3f000000` (loaded straight from `inbuf[0]`, never shuffled!)
     - `V: %90 Bfi = 0x41000000` (`[0x28:MKVEC.v2i8 r2, r2^.b0, r3^.b1, r4^.b01]`)
   - Root cause immediately localized to `src/spirv2ssa/spirv2ssa_ops.cpp` (`OpBitcast %v4u8` / `OpVectorShuffle %v4u8`).

2. **`ACMPXCHG` (Bucket C — Valhall `ACMPXCHG` operand/address mismatch)**:
   - `[MISMATCH] out=atomic:ssbo:0:1 direct=(('atomiccmpxchg', 36, 1065353216, 16),) valhall=(('atomiccmpxchg', 6, 16, 0),)`
   - Notice BOTH the element offset (`36` vs `6`) AND the `(cmp, val)` argument positions (`(1065353216, 16)` vs `(16, 0)`) differ.
   - Root cause localized to `src/valhall_pass1_handlers.cpp` (`ACMPXCHG` staging-register extraction) + SSBO offset scaling.

3. **`S8_TO_F32` / `U8_TO_F32` / `V2S16_TO_V2F16` / `V2U16_TO_V2F16` (Bucket B/C — packed subword `OpBitcast` + conversion)**:
   - `frontier_pcs: 0x20(S8_TO_F32 r2, r2^.b0)` with `direct=0x4e7c0000` vs `valhall=0x00000000`.
   - `OpBitcast %v4s8 %a` -> `OpCompositeExtract %s8 0` -> `OpConvertSToF` is reading the full 32-bit word on Direct vs byte 0 on Valhall.

4. **`LD_VAR` / `LD_VAR_FLAT` / `LD_ATTR` / `LEA_BUF_IMM` (Bucket A — dynamic-array ABI / `valhall=None`)**:
   - `direct=0x80000000 valhall=None` (and ~6.4s timeout in `equiv_fuzzer` from loop/branch path explosion in Mali's dynamic-index switch ladder).
   - Check `--dump-ssa --defau` to see why the dynamic varying/attribute load failed to resolve to a concrete output channel.

5. **`FSIN_TABLE` / `FCOS_TABLE` / `FADD_IMM` (Bucket E — 1–3 ULP trig polynomial delta)**:
   - `direct=0x3ef57744 valhall=0x3ef57747` (3 ULP) and `direct=0x3f60a940 valhall=0x3f60a93f` (1 ULP) — expected Mali LUT/polynomial approximation vs host `sinf`/`cosf`.

---

## Part 2: Remaining Gaps & Proposed New Debugging Tools

Even with `--slice-diff <depth>`, `--trace-diff`, `--dump-shaders`, and `spirv-reduce`, four concrete debugging gaps remain when triaging large sweeps:

### 1. Concrete Op-Level Equivalence Matrix (`--op-matrix`) — *High Impact*
- **Gap**: Many failures in the 201-op sweep share the exact same underlying root cause (e.g., `CUBEFACE1`/`CUBEFACE2`/`CUBE_SSEL`/`CUBE_TSEL` all fail on the same cube-projection chain; `V2S16_TO_V2F16`/`V2U16_TO_V2F16`/`V2S8_TO_V2F16`/`V2U8_TO_V2F16` all fail on packed `v2f16` `OpBitcast`).
- **Proposal**: Cluster sweep failures automatically by `(first_diverging_op, valhall_frontier_mnemonic, diff_kind)` and emit a deduplicated root-cause table showing "N mnemonics blocked by 1 root cause" with the smallest repro per cluster.

### 2. Interactive / Textual Value-Provenance Tree (`--explain-node <ssa_id>`)
- **Gap**: `--slice-diff` aligns by hop depth, which works great for 1–6 hops, but when Mali expands a single SPIR-V op (like `Atan2` or `Pow`) into 20+ Valhall instructions, Direct hop 1 corresponds to Valhall hop 12.
- **Proposal**: Align slices by **semantic landmarks** (shared concrete values and SSBO/FAU/input loads) rather than raw BFS hop index, collapsing intermediate `Copy`/`Bitcast`/`Bfe`/`Bfi` chains into a single line.

### 3. Pilot-Shader & FAU/SRT Sideband Inspector (`--dump-fau-state <vec_idx>`)
- **Gap**: Failures involving `LD_PKA`, `LEA_BUF_IMM`, `LD_TEX`, `LEA_TEX`, or `LD_VAR_SPECIAL` (`SHADD.u64 [r4:r5], sample, ...`) depend on exact concrete FAU uniform words and pilot-shader-computed buffer bases.
- **Proposal**: Print the concrete `ConcreteShaderEnv` FAU table (`u0..uN`), SRT descriptor addresses, and pilot-shader register outputs for `vec_idx` right beside the failing Valhall instruction.

### 4. Automated `--no-opt-ssa` / `--no-defau` / `--no-canonicalize` Pass Bisector (`--bisect-pass`)
- **Gap**: Determining whether a bug is in Pass 1 (`valhall_pass1`), `defau` (`ssa_defau_devirt`), C++ `ssa-opt`, or Python `canonicalize_ssa_module` currently requires running 4 manual commands.
- **Proposal**: A single `--bisect-pass` flag in `tools.coverage` that toggles each pass in pipeline order (`raw_pass1` -> `+defau` -> `+opt_ssa` -> `+canonicalize`) on the failing witness `(vec_idx, overrides)` and reports the exact first pass where Direct and Valhall diverge.

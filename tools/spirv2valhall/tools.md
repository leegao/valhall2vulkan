# `tools/spirv2valhall` — CLI Reference & Pipeline Guide

`build/spirv2valhall` is the primary C++ driver for compiling SPIR-V shaders into Mali Valhall ISA, inspecting CSF Execution Contexts & Descriptor Graphs (MBS2), lowering Valhall ISA or SPIR-V to `ValhallSSA`, running De-FAU devirtualization and SSA optimizations, lifting Valhall ISA back to SPIR-V, and dumping dual-SSA equivalence bundles.

It supports four compilation/input backends:
1. **Arm `libMali-Gxx` In-Process Compiler (default)**: Compiles Vulkan Compute, Vertex, and Fragment SPIR-V in-process via `malioc_inproc`, reads the live `cmpbe_chunk_MBS2*`, runs pilot-shader dataflow analysis, and prints MBS2-enriched Valhall disassembly.
2. **Mesa Panfrost Compiler (`--panfrost` and `--panfrost-cl`)**: Compiles Vulkan compute SPIR-V (`--panfrost`, via built-in Vulkan->OpenCL SPIR-V translator) or OpenCL 2.2 SPIR-V (`--panfrost-cl`) using `third_party/prebuilts/panfrost/panfrost_compile`, extracts Valhall EBIN bytecode from `bifrost_precompiled_kernel_info`, disassembles it, and optionally lowers to `ValhallSSA` or lifts to SPIR-V.
3. **Direct SPIR-V -> ValhallSSA (`--direct-ssa`)**: Lowers SPIR-V directly through `spirv2ssa` without invoking any GPU compiler.
4. **Direct Raw Valhall ISA (`--direct-isa <FILE>`)**: Decodes and Pass-1-lowers raw `.valhall` or `.tsv` 64-bit instruction streams in batch JSONL mode (used by `tools.corpus_synth --mode raw`).

---

## Quick Start Examples

### 1. Compile Vulkan SPIR-V via `libMali-Gxx` and Print Context + Enriched Valhall ISA
```bash
./build/spirv2valhall samples/comp_unique_03_fba18fbe.spv
./build/spirv2valhall --no-context --mods samples/frag_rep_minimal.spv
./build/spirv2valhall --context-only samples/vert_rep_minimal.spv
```

### 2. Lower to ValhallSSA (with De-FAU Devirtualization & SSA Opt) or Dump Dual Equiv JSON
```bash
# Print lowered + devirtualized + optimized ValhallSSA
./build/spirv2valhall --dump-ssa --defau --opt-ssa samples/comp_unique_03_fba18fbe.spv

# Write dual Direct + Valhall SSA equivalence bundle for tools.coverage / tools.equivalence
./build/spirv2valhall --defau --opt-ssa --dump-equiv-json /tmp/bundle.json samples/comp_unique_03_fba18fbe.spv

# Lower input SPIR-V directly via spirv2ssa (bypassing libMali)
./build/spirv2valhall --direct-ssa --opt-ssa samples/comp_unique_03_fba18fbe.spv
```

### 3. Compile via Mesa Panfrost (`--panfrost` and `--panfrost-cl`)
```bash
# Vulkan compute SPIR-V -> Panfrost Valhall ISA
./build/spirv2valhall --panfrost -a 10 shader_compute.spv

# Vulkan compute SPIR-V -> Panfrost Valhall ISA -> ValhallSSA (+ JSON)
./build/spirv2valhall --panfrost -a 10 --dump-ssa --opt-ssa --dump-ssa-json /tmp/pf_ssa.json shader_compute.spv

# OpenCL 2.2 SPIR-V kernel -> Panfrost Valhall ISA & ValhallSSA
./build/spirv2valhall --panfrost-cl -a 10 kernel_cl.spv
./build/spirv2valhall --panfrost-cl -a 10 --dump-ssa kernel_cl.spv
```

### 4. Roundtrip Lift Valhall ISA Back to SPIR-V
```bash
./build/spirv2valhall --dump-spirv --out-spirv /tmp/lifted samples/comp_unique_03_fba18fbe.spv
```

### 5. Batch-Verify Raw 64-Bit Valhall Instruction Streams
```bash
./build/spirv2valhall --direct-isa tests/fixtures/direct_isa_sample.tsv
```

---

## Complete CLI Flag Reference (`./build/spirv2valhall`)

| Flag | Default | Description |
| :--- | :--- | :--- |
| `<input.spv>` | *(required unless `--direct-isa`)* | Input SPIR-V binary (`.spv`). Stage is inferred automatically. |
| `-c, --core <CORE>` | `Mali-G710` | Target Mali GPU core for `libMali-Gxx`. |
| `-a, --arch <N>` | inferred (`10` for Panfrost) | Valhall architecture version (`9`–`14`). |
| `-o, --out-header <FILE>` | `None` | Persist decompiled MBS2 C header (`cmpbe_chunks.h` format). |
| `--out-bin <PREFIX>` | `None` | Write raw EBIN bytecode to `<PREFIX>_0.bin`, `<PREFIX>_1.bin`, ... |
| `--dump-spirv` | `False` | Lift Valhall ISA back to SPIR-V via `valhall2spirv` and print summary/disassembly. |
| `--out-spirv <PREFIX>` | `None` | Write lifted SPIR-V binary(ies) to `<PREFIX>_0.spv`, ... |
| `--dump-ssa` | `False` | Lower Valhall ISA to `ValhallSSA` and print SSA instead of ISA disassembly. |
| `--dump-ssa-json <FILE>` | `None` | Write lowered (and optionally optimized) `ValhallSSA` JSON to `<FILE>`. |
| `--direct-ssa` | `False` | Lower input `.spv` directly via `spirv2ssa` (bypassing `libMali`) and print SSA. |
| `--direct-isa <FILE>` | `None` | Bypass `libMali`: decode and Pass-1-lower `.valhall` or `.tsv` raw ISA streams (JSONL output). |
| `--dump-equiv-json <FILE>` | `None` | Write dual `direct_ssa` + `valhall_ssa` + MBS2/Pilot metadata equivalence bundle JSON. |
| `--opt-ssa` | `False` | Run C++ verification SSA optimization pipeline (`ssa_opt_pipeline`) on `ValhallSSA` / `spirv2ssa`. |
| `--defau` | `False` | Run De-FAU devirtualization + fact prover on lowered `ValhallSSA`. |
| `--dump-devirt-fallbacks` | `False` | Run `--defau` and print unresolved FAU fallback diagnostics to stderr. |
| `--opt-fp-mode <strict\|fast>` | `strict` | FP algebraic canonicalization mode for `--opt-ssa`. |
| `--no-fp-tile-rewrites` | `False` | Disable declarative `valhall_tiles.json` FP rewrites in SSA opt. |
| `--no-fp-cpp-rewrites` | `False` | Disable procedural C++ FP rewrites in SSA opt. |
| `--no-context` | `False` | Omit the CSF Execution Context & Descriptor Graph header. |
| `--context-only` | `False` | Print only the CSF Execution Context & Descriptor Graph header. |
| `--no-enrich` | `False` | Print raw Valhall disassembly without MBS2/ABI/SRT comment tags. |
| `--mali-lib <PATH>` | `mali_offline_compiler/graphics/libMali-Gxx_r55p0-00rel0.so` | Path to `libMali-Gxx` shared library. |
| `--panfrost` | `False` | Compile Vulkan compute SPIR-V via Mesa Panfrost (`spirv -> isa -> ssa`). |
| `--panfrost-cl` | `False` | Compile OpenCL 2.2 SPIR-V via Mesa Panfrost (`spirv -> isa -> ssa`). |
| `--panfrost-bin <PATH>` | auto (`third_party/prebuilts/panfrost/panfrost_compile`) | Explicit path to `panfrost_compile` binary (also honors `PANFROST_COMPILE_BIN`). |
| `--no-strip-caps` | `False` | Do not strip unused `Float64`/`Interlock` capabilities before compiling with `libMali`. |
| `--mods` | `False` | Include decoded modifier `key=value` pairs on each disassembled instruction. |
| `--quiet` | `False` | Suppress stdout text output (useful with `--out-bin` / `--dump-equiv-json`). |
| `-v, --verbose` | `False` | Print stage inference and compiler invocation diagnostics to stderr. |

---

## Running Unit Tests

```bash
ctest --test-dir build --output-on-failure -R "test_spirv2valhall|test_moc_roundtrip"
```

# `tools.corpus_synth` — Usage Guide & Flag Reference

`tools.corpus_synth` is the Valhall ISA & SPIR-V corpus synthesizer for `valhall2spirv`. It supports two primary workflows:
1. **Targeted ISA Synthesis (`--target-isa`)**: Deterministically synthesizes shaders targeting specific Valhall mnemonics, variants, and hardware field mutations, and validates lowering equivalence via `equiv_fuzzer`.
2. **Greedy Coverage-Maximizing Synthesis (default)**: Synthesizes multi-block SPIR-V shaders maximizing unique Valhall ISA or SPIR-V opcode coverage.

---

## Quick Start Examples

### 1. Audit Full 201-Mnemonic / 627-Variant / 17-Fragment Coverage
```bash
python3 -m tools.corpus_synth.cli --coverage-audit
```

### 2. Single-Instruction Equivalence Validation with Backward-Slice Diff
```bash
python3 -m tools.corpus_synth.cli \
  --target-isa ACMPXCHG \
  --mode spirv \
  --verify-equiv \
  --k-init 16 \
  --fp-range vulkan \
  --slice-diff 6 \
  --output-shaders /tmp/repro_acmpxchg
```
- Compiles the `ACMPXCHG` recipe via `build/spirv2valhall`.
- Runs `equiv_fuzzer` on 16 Vulkan-range vectors.
- On mismatch, prints the 6-hop Direct vs Valhall concrete backward slice (`--slice-diff 6`) and dumps `/tmp/repro_acmpxchg/isa_acmpxchg_minimal.spvasm`.

### 3. Sweep All 129 SPIR-V-Inducible Mnemonics
```bash
python3 -m tools.corpus_synth.cli \
  --target-isa ALL \
  --mode spirv \
  --verify-equiv \
  --k-init 16 \
  --fp-range vulkan \
  --output-shaders /tmp/isa_sweep_failures
```

### 4. Pin Intermediate SSA Values to Catch Optimizer / Intermediate Divergences
```bash
python3 -m tools.corpus_synth.cli \
  --target-isa HADD \
  --mode spirv \
  --materialize-intermediates \
  --verify-equiv \
  --k-init 16 \
  --fp-range vulkan \
  --slice-diff 5
```

### 5. Generate Semantically Equivalent `spirv-fuzz` Mutants
```bash
python3 -m tools.corpus_synth.cli \
  --target-isa HADD,RROT,FMA \
  --mode spirv \
  --fuzz \
  --fuzz-variants 3 \
  --verify-equiv \
  --k-init 16 \
  --fp-range vulkan
```
*(Automatically locates `third_party/prebuilts/spirv-tools/spirv-fuzz` if installed.)*

### 6. Full 627-Variant & 17-Fragment Raw Valhall Decode + Pass-1 Sweep
```bash
python3 -m tools.corpus_synth.cli \
  --target-isa ALL \
  --mode raw \
  --variants \
  --mutate-fields
```
*(Verifies all 13,233 encoded Valhall instruction streams across all 201 mnemonics and 7,987 branch points in ~3 seconds via `spirv2valhall --direct-isa`.)*

### 7. Greedy Multi-Block Corpus Synthesis
```bash
python3 -m tools.corpus_synth.cli \
  --target valhall-isa \
  --num-shaders 8 \
  --blocks 3 \
  --insns 16 \
  --verify-equiv \
  --output-shaders /tmp/greedy_corpus
```

---

## Complete Flag Reference

| Flag | Default | Description |
|---|---|---|
| `--target-isa <SPEC>` | `None` | Target mnemonic(s): single (`HADD`), comma-separated (`HADD,RROT`), `ALL`, `COMPUTE`, `GRAPHICS`, or `INTERNAL`. |
| `--mode {auto,spirv,raw,both}` | `auto` | `spirv`: Vulkan SPIR-V via `libMali-Gxx` (129 mnemonics). `raw`: 64-bit Valhall encoder via `spirv2valhall --direct-isa` (201 mnemonics). `auto`: `spirv` if recipe exists else `raw`. `both`: run both. |
| `--complexity {minimal,medium,branchy}` | `minimal` | Shader wrapper complexity (`minimal`: tightest; `medium`: pre/post register mixing; `branchy`: structured `OpSelectionMerge` diamond). |
| `--mutate-fields` | `False` | Attach all applicable cross-instruction field fragments (17 fragments) and instruction branch-point mutations. |
| `--variants` | `False` | Cover all 627 XML `(mnemonic, variant, since..until)` specs instead of only the primary variant per mnemonic. |
| `--coverage-audit` | `False` | Print JSON coverage report (mnemonics, variants, tiers, fragments, branch points) and exit. |
| `--verify-equiv` | `False` | Run concolic `equiv_fuzzer` (Direct `spirv2ssa` vs Valhall `valhall_pass1`) on synthesized SPIR-V shaders. |
| `--max-coverage-seeds` | `False` | Enable maximal-reachability / higher-budget concolic seeding (`max_paths=32`, `time_budget_ms=5000`) during `--verify-equiv`. |
| `--materialize-intermediates` | `False` | Store scalar intermediate SSA IDs (`%u32`/`%s32`/`%f32`) from compute recipe bodies into `outbuf` so optimizations cannot DCE them and `equiv_fuzzer` checks intermediates. |
| `--slice-diff <DEPTH>` | `0` | On `--verify-equiv` divergence, print an N-hop side-by-side Direct vs Valhall concrete backward-slice diff aligned by shared concrete values. |
| `--dump-shaders` | `False` | Print generated `.spvasm` text to stdout. (Note: on `--verify-equiv` mismatch/failure, the failing `.spvasm` is dumped automatically.) |
| `--output-shaders <DIR>` | `None` | Write generated `.spvasm` and `.spv` files (and any auto-dumped failing shaders) to `<DIR>`. |
| `--fuzz` | `False` | Invoke `spirv-fuzz` on each synthesized SPIR-V shader to generate semantically equivalent fuzzed mutants and verify them. |
| `--fuzz-variants <N>` | `2` | Number of `spirv-fuzz` mutants generated per seed shader when `--fuzz` is set. |
| `--k-init <N>` | `32` | Number of initial concrete input vectors for `--verify-equiv`. |
| `--fp-range {unconstrained,vulkan}` | `unconstrained` | Floating-point input sampling range (`vulkan` restricts to finite Vulkan-safe floats). |
| `--domain {bitvec,udomain,udomain_exact}` | `udomain` | Symbolic domain for concolic execution. |
| `--target {valhall-isa,spirv-ops}` | `valhall-isa` | Coverage objective in greedy synthesis mode (when `--target-isa` is omitted). |
| `--num-shaders`, `--blocks`, `--insns`, `--candidates` | `6, 1, 12, 4` | Greedy corpus synthesis parameters. |
| `--bin <PATH>` | `./build/spirv2valhall` | Path to `spirv2valhall` binary. |
| `--seed <INT>` | `0xC0DE` | PRNG seed for synthesis and `spirv-fuzz`. |
| `--out-dir <DIR>` | `None` | Legacy alias for `--output-shaders`. |

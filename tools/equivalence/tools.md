# `tools/equivalence` — Shader-Level Formal Equivalence Verification

`tools/equivalence` provides end-to-end formal equivalence checking between **Direct SPIR-V SSA** (`spirv2ssa`) and **Lowered Valhall SSA** (`libMali` -> Pass 1 -> De-FAU / SSA opt). It houses the Z3 all-paths miter verifier, cutpoint-guided combinational equivalence checker (CEC), and shared SPIR-V/bundle I/O helpers.

For the full theoretical design, domain models (`udomain`, `udomain_backfill`, `real`/`bv` FP modes), and proof architecture, see [`shader-equiv-verifier.md`](shader-equiv-verifier.md).

---

## Modules & CLI Entrypoints

| Module | Role | CLI Entrypoint |
| :--- | :--- | :--- |
| `shader_equiv_verifier.py` | Full shader-level Z3 equivalence prover (`PROVEN` / `COUNTEREXAMPLE` / `UNKNOWN` / `SKIP`) with canonicalization, backfill domain, conditioned HM unification, and CEC dispatch. | `python3 -m tools.equivalence.shader_equiv_verifier ...` |
| `cec.py` | Fast concrete differential execution + cutpoint-guided CEC (matches intermediate SSA nodes by concrete signature and proves cutpoint-by-cutpoint equivalence in Z3). | Library (`verify_with_cec`, `run_concrete_diff`) & `--cec` mode of `shader_equiv_verifier` |
| `spvasm_io.py` | Locates prebuilt/system `spirv-as`, `spirv-dis`, `spirv-val`, `spirv-opt`, `spirv-reduce`, `spirv-fuzz`; assembles `.spvasm` and invokes `build/spirv2valhall --dump-equiv-json` to produce dual-SSA bundles. | Library (`load_equiv_bundle`, `assemble_spvasm`, `find_spirv_tool`) |

---

## CLI Usage: `tools.equivalence.shader_equiv_verifier`

### Basic Invocation

```bash
# Verify a single SPIR-V binary, .spvasm assembly, or pre-dumped bundle .json
python3 -m tools.equivalence.shader_equiv_verifier path/to/shader.spv

# Fast concrete diff + cutpoint-guided CEC
python3 -m tools.equivalence.shader_equiv_verifier --cec path/to/shader.spvasm

# All-path Z3 miter with backfill domain, canonicalization, and conditioned unification
python3 -m tools.equivalence.shader_equiv_verifier \
  --domain udomain_backfill --canonicalize --conditioned-unify \
  bugs/frag_unique_05_d392e446_frag_color_0_.a.spvasm

# Batch-verify a directory of shaders with JSON summary output
python3 -m tools.equivalence.shader_equiv_verifier \
  --cec --fp-mode real --timeout-ms 10000 \
  samples/
```

### Key CLI Flags

| Flag | Default | Description |
| :--- | :--- | :--- |
| `inputs` | *(required)* | One or more `.spv`, `.spvasm`, `.json` equiv bundles, or directories of shaders. |
| `--cec` | `False` | Run fast concrete differential fuzzing followed by cutpoint-guided CEC (`tools.equivalence.cec.verify_with_cec`) before/instead of monolithic Z3 miter. |
| `--domain <mode>` | `udomain` | Symbolic domain: `udomain` (fast bitvector + uninterpreted FP/HW ops), `udomain_backfill` (iterative concrete-witness backfill for UF congruence), or `exact`. |
| `--fp-mode <mode>` | `exact` | Floating-point Z3 theory: `exact` (IEEE-754 `FPSort`), `real` (rational arithmetic approximation for fast algebraic proofs), or `uf` (uninterpreted FP ops). |
| `--canonicalize` / `--no-canonicalize` | `True` | Run `tools.symbolic_ssa.passes` canonicalization (GVN, DCE, CFG chain contraction, algebraic normalization) on both sides before solving. |
| `--conditioned-unify` | `False` | Enable path-conditioned Herbrand/HM unification across control-flow merges. |
| `--max-unroll <N>` | `1` | Bounded loop unrolling depth for symbolic path enumeration. |
| `--timeout-ms <MS>` | `15000` | Per-shader Z3 solver timeout in milliseconds. |
| `--spirv2valhall <PATH>` | `build/spirv2valhall` | Path to the `spirv2valhall` binary used to generate equiv bundles from `.spv`/`.spvasm`. |
| `--opt-ssa` / `--defau` | `True` | Passed through to `spirv2valhall` when dumping equiv bundles. |
| `--verbose` | `False` | Print detailed per-output Z3 proof status, counterexample assignments, and timing breakdown. |

### Reading Output & Exit Codes

* **`PROVEN`** (exit `0`): Z3 returned `UNSAT` for the miter `direct_out != valhall_out` across all observable outputs (SSBOs, fragment colors, vertex varyings, `gl_Position`, atomics, kills).
* **`COUNTEREXAMPLE`** (exit `1`): Z3 or concrete diff found an input assignment where observable outputs diverge; prints the diverging output channel and concrete witness.
* **`UNKNOWN`** (exit `2`): Solver timed out or hit an uninterpreted-function abstraction gap.
* **`SKIP`** (exit `0`): Shader produced no comparable observable outputs or failed compilation.

---

## Programmatic API Examples

```python
from tools.equivalence import load_equiv_bundle, verify_with_cec, verify_spv_file

# 1. One-shot formal verification of a .spv or .spvasm file
res = verify_spv_file("samples/vert_rep_minimal.spv", domain="udomain_backfill", canonicalize=True)
print(res.status, res.elapsed_ms)

# 2. Load a dual-SSA bundle and run cutpoint-guided CEC directly
bundle = load_equiv_bundle("bugs/frag_unique_05_d392e446_frag_color_0_.a.spvasm")
cec_res = verify_with_cec(bundle, bundle["direct_ssa"], bundle["valhall_ssa"])
print(cec_res.status, cec_res.matched_cutpoints)
```

---

## Running Unit Tests

```bash
ctest --test-dir build --output-on-failure -R "test_shader_equiv|test_cec|test_canonicalize"
```

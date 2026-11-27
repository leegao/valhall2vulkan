# `tools.coverage` & `tools.coverage.equiv_fuzzer` CLI Reference

`tools.coverage` provides concolic input synthesis, disjoint maximal-reachability path enumeration, concrete equivalence fuzzing (`direct_ssa` vs. `valhall_ssa`), automated root-cause triage (`[TRIAGE]` + `--trace-diff` + `--slice-diff`), and `spirv-reduce`-backed `.spvasm` repro recording (`--record`) for `.spv`, `.spvasm`, or pre-dumped equivalence bundle `.json` inputs.

---

## Quick Start

```bash
# 1. Concolic equivalence fuzzing with trace-diff and 8-hop backward slice diff
python3 -m tools.coverage samples/comp_unique_03_fba18fbe.spv \
  --equiv-fuzz --fp-range vulkan --k-init 64 --trace-diff --slice-diff 8

# 2. Record a shrunken .spvasm reproducer into bugs/ on divergence
python3 -m tools.coverage path/to/failing.spv \
  --equiv-fuzz --fp-range vulkan --record bugs/ \
  --record-timeout-s 20 --record-steps 120

# 3. Maximal-reachability disjoint path partition (exact IEEE-754 + Unique-Call UF Pruning)
python3 -m tools.coverage samples/comp_unique_03_fba18fbe.spv \
  --fp-mode exact --maximal-reachability --dump-vectors /tmp/paths.json

# 4. Predicate-coverage input synthesis + auto-shrinking any discovered divergence
python3 -m tools.coverage samples/comp_unique_03_fba18fbe.spv \
  --k-init 64 --fp-mode exact --shrink --dump-vectors /tmp/vecs.json

# 5. Maximal-reachability-seeded coverage synthesis
python3 -m tools.coverage samples/comp_unique_03_fba18fbe.spv \
  --maximal-reachability-seed --domain udomain_backfill --dump-vectors /tmp/vecs.json

# 6. Cross-compiler equivalence fuzzing (Mesa Panfrost vs Mali DDK)
python3 -m tools.coverage.panfrost_vs_ddk samples/comp_unique_03_fba18fbe.spv \
  --maximal-reachability-seed --trace-diff --slice-diff 8 --record bugs/
```

If Z3 is installed in a local wheel directory, prepend `PYTHONPATH=/path/to/z3_site_packages:.` before `python3`.

---

## Operating Modes

| Mode Flag | What It Does | Exit Code |
| :--- | :--- | :--- |
| *(default)* | Runs `k_init` corner + PRNG vectors, then uses incremental Z3 + Unique-Call UF Pruning + AST-size greedy packing to flip uncovered branch/`Select`/`Kill` predicates and prove dead polarities UNSAT. | `0` (or `1` on load error) |
| `--maximal-reachability` | Enumerates pairwise-disjoint satisfiable path conditions $\{PC_0, \dots, PC_{m-1}\}$ prioritizing unvisited CFG blocks. Reports `complete=True (EXHAUSTED_ALL_PATHS)` when the blocked solver returns UNSAT. | `0` |
| `--equiv-fuzz` | Runs `tools.coverage.equiv_fuzzer.run_concolic_equiv_fuzz`: seeds inputs from `--side` (`direct` or `valhall`) via (1) `k_init` corner+PRNG vectors, (2) maximal-reachability witnesses, and (3) predicate-coverage vectors within `--time-budget-ms`, concretely tests both `direct_ssa` and `valhall_ssa` under `DiffPolicy`, and auto-shrinks the first `REAL_DIVERGENCE`. | `0` if equivalent (`EQUIV_EXACT` / `EQUIV_WITHIN_ULP` / `SKIP`), `1` if `DIVERGED` |

---

## Complete CLI Flag Reference (`python3 -m tools.coverage`)

| Flag | Default | Description |
| :--- | :--- | :--- |
| `spv_or_json` | *(required)* | Input `.spv`, `.spvasm` (assembled via `tools.equivalence.spvasm_io`), or pre-dumped `--dump-equiv-json` bundle. |
| `--bin <PATH>` | `./build/spirv2valhall` | Path to `spirv2valhall` binary used to lift `.spv`/`.spvasm` inputs. |
| `--side {direct,valhall}` | `direct` | SSA side targeted for synthesis/reachability or used to seed `--equiv-fuzz`. |
| `--k-init <N>` | `32` | Number of initial concrete seed vectors (edge-case corners + deterministic PRNG). |
| `--max-rounds <N>` | `12` | Maximum synthesis rounds in default coverage mode. |
| `--domain {bitvec,udomain,udomain_backfill,udomain_exact}` | `udomain_backfill` | Symbolic value domain for concolic solving. |
| `--fp-mode {exact,real,uf}` | `exact` | Z3 floating-point theory: `exact` (`QF_FP`), `real` (`QF_NRA` algebraic), or `uf` (uninterpreted FP ops). |
| `--fp-range {unconstrained,vulkan}` | `unconstrained` | Concrete input FP sampling range (`vulkan` restricts to finite Vulkan-safe floats). |
| `--canonicalize` / `--no-canonicalize` | `True` | Enable or disable pre-SMT structural canonicalization (`tools.symbolic_ssa.passes`). |
| `--defau` / `--no-defau` | `True` | Enable or disable De-FAU devirtualization when invoking `spirv2valhall`. |
| `--opt-ssa` / `--no-opt-ssa` | `True` | Enable or disable C++ `ssa-opt` passes when invoking `spirv2valhall`. |
| `--maximal-reachability` | `False` | Solve and report pairwise-disjoint reachability conditions exercising all paths. |
| `--maximal-reachability-seed` | `False` | Use maximal reachability to find maximal path vector seeds, then run maximal coverage (and equivalence fuzzing when `--equiv-fuzz` is active). |
| `--equiv-fuzz` | `False` | Seed concrete vectors from `--side` and concretely test Direct vs Valhall equivalence. |
| `--shrink` | `False` | Auto-shrink diverging inputs (always enabled in `--equiv-fuzz`). |
| `--slice-diff <DEPTH>` | `0` | On divergence, print an N-hop side-by-side Direct vs Valhall concrete backward-slice diff aligned by shared values. |
| `--trace-diff` | `False` | Print concrete Valhall frontier node evaluations and Valhall disassembly for root-cause triage. |
| `--record <PATH_OR_DIR>` | `None` | Record a shrunken `.spvasm` divergence reproducer (via `spirv-reduce` + `spirv-val`) to `<PATH_OR_DIR>` (e.g., `bugs/`). |
| `--no-reduce` | `False` | With `--record`, skip `spirv-reduce` and disassemble the original shader directly. |
| `--record-timeout-s <SEC>` | `15.0` | Maximum wall-clock seconds for `spirv-reduce` during `--record`. |
| `--record-steps <N>` | `120` | Maximum `spirv-reduce` step limit during `--record`. |
| `--max-ulp <N>` | `2` | Maximum IEEE-754 FP32 ULP difference treated as `BENIGN_FP_ROUNDING` by `DiffPolicy`. (Automatically augmented for transcendental SSBO stores when `max_ulp > 0`.) |
| `--max-paths <N>` | `32` | Maximum disjoint paths to enumerate in `--maximal-reachability` and `--equiv-fuzz`. |
| `--timeout-ms <MS>` | `500` | Per-solve Z3 timeout in milliseconds. |
| `--time-budget-ms <MS>` | `5000` | Total wall-clock budget in milliseconds for synthesis, reachability, or `--equiv-fuzz`. |
| `--max-unroll <N>` | `1` | Bounded loop unrolling depth (coverage mode adaptively increases up to 3 when loops remain uncovered). |
| `--dump-vectors <FILE>` | `None` | Write synthesized override dicts (or disjoint path records in `--maximal-reachability`) to JSON. |

---

## Panfrost vs. DDK Cross-Compiler CLI (`python3 -m tools.coverage.panfrost_vs_ddk`)

`tools.coverage.panfrost_vs_ddk` accepts one or more `.spv`/`.spvasm` `shaders` and shares the same concolic, reachability, triage, and recording flags as `python3 -m tools.coverage` (`--bin`, `--side {direct,valhall,panfrost,ddk}`, `--k-init` default `64`, `--max-rounds`, `--domain` default `udomain_backfill`, `--fp-mode` default `real`, `--fp-range` default `vulkan`, `--canonicalize`/`--no-canonicalize`, `--defau`/`--no-defau`, `--opt-ssa`/`--no-opt-ssa`, `--maximal-reachability`, `--maximal-reachability-seed`, `--equiv-fuzz`, `--shrink`, `--slice-diff`, `--trace-diff`, `--record`, `--no-reduce`, `--record-timeout-s`, `--record-steps`, `--max-ulp`, `--max-paths`, `--timeout-ms`, `--time-budget-ms` default `3000`, `--max-unroll`, `--dump-vectors`), plus Panfrost-specific options:

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--arch <N>` | `11` | Valhall architecture version. |
| `--panfrost-cl` | `False` | Use OpenCL `--panfrost-cl` backend instead of native Vulkan `--panfrost`. |
| `--panfrost-backend {auto,kraid,bifrost}` | `auto` | Backend for `panfrost_vk_compile`. |
| `--panfrost-robust` | `False` | Enable `--robust` in `panfrost_vk_compile`. |
| `--panfrost-no-idvs` | `False` | Pass `--no-idvs` to `panfrost_vk_compile`. |
| `--no-triangle` | `False` | Skip 3-way Direct attribution on divergence. |
| `--json-out <FILE>` | `""` | Write full `PanfrostVsDdkResult` list to JSON. |

---

## Interpreting Output

### `[EQUIV-FUZZ]`, `[SHRINK]`, `[TRIAGE]`, and `[SLICE-DIFF]`

```text
[EQUIV-FUZZ] bcn_bc7.spv status=DIVERGED seed=direct vecs=32(init=32,reach=0,cov=0) blocks=0/525 preds=326->326/2746 benign_fp=0 divs=11 elapsed=185.7ms
  [SHRINK] vec=0 output=ssbo:0:0+0xf2000000 direct=0xff000000 valhall=None kind=REAL_DIVERGENCE ulp=None overrides=0->0 cone=(15264,5895)
    frontier_pcs: 0x30(ARSHIFT.i32 r1, r2, k03020100.b2), 0x38(AND.i32 r2, r2^, u5.w1), ...
  [TRIAGE] suspect=LOWERING_OR_SEMANTICS
```

- **`status`**:
  - `EQUIV_EXACT`: every output and memory/image/atomic side effect matched bit-for-bit on all tested vectors.
  - `EQUIV_WITHIN_ULP`: no real divergences; only FP32 outputs differed within `DiffPolicy` (`max_ulp`, `max_abs_err`, or NaN payload bits).
  - `DIVERGED`: at least one `REAL_DIVERGENCE` was observed.
  - `SKIP`: neither side produced observable outputs on the tested vectors.
- **`[TRIAGE] suspect=...`**: Classifies failures into `MODELING_CANONICALIZER`, `LOWERING_ADDR_OR_SWIZZLE`, `MODELING_ENV_OR_ABI`, `SPIRV2SSA_OR_MODELING_TEX_GATHER`, or `LOWERING_OR_SEMANTICS`.
- **`[SLICE-DIFF depth=N]`**: Walks backward `N` hops from the diverging sink on both Direct and Valhall SSA, marking `== MATCH`, `== SHARED`, `!= DIFF`, and `<< FIRST-DIVERGE`.

---

## Programmatic API (`tools.coverage.equiv_fuzzer`)

```python
from tools.coverage import DiffPolicy, run_concolic_equiv_fuzz
from tools.symbolic_ssa.passes import canonicalize_ssa_module

d_ssa, _ = canonicalize_ssa_module(bundle["direct_ssa"])
v_ssa, _ = canonicalize_ssa_module(bundle["valhall_ssa"])
res = run_concolic_equiv_fuzz(
    bundle,
    d_ssa,
    v_ssa,
    seed_side="direct",
    k_init=32,
    time_budget_ms=5000,
    fp_mode="exact",
    fp_range="vulkan",
    policy=DiffPolicy(max_ulp=2),
    auto_shrink=True,
)
print(res.status, res.diverging_vectors, res.first_divergence)
```

---

## Running Unit Tests

```bash
ctest --test-dir build --output-on-failure -R "test_coverage_synthesizer"
```

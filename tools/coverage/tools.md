# `tools.coverage` & `tools.coverage.equiv_fuzzer` CLI Reference

`tools.coverage` provides concolic input synthesis, disjoint maximal-reachability path enumeration, concrete equivalence fuzzing (`direct_ssa` vs. `valhall_ssa`), and counterexample shrinking/frontier localization for `ValhallSsaModule` bundles or `.spv` shaders.

## Quick Start

```bash
# 1. Concolic equivalence fuzzing (seed from direct SSA, test direct vs. valhall, auto-shrink)
python3 -m tools.coverage samples/comp_unique_03_fba18fbe.spv --canonicalize --equiv-fuzz --side direct

# 2. Maximal-reachability disjoint path partition (exact IEEE-754 + Unique-Call UF Pruning)
python3 -m tools.coverage samples/comp_unique_03_fba18fbe.spv --canonicalize --fp-mode exact --maximal-reachability --dump-vectors /tmp/paths.json

# 3. Predicate-coverage input synthesis + auto-shrinking any discovered divergence
python3 -m tools.coverage samples/comp_unique_03_fba18fbe.spv --canonicalize --k-init 64 --fp-mode exact --shrink --dump-vectors /tmp/vecs.json
```

If Z3 is installed in a local wheel directory, prepend `PYTHONPATH=/path/to/z3_site_packages:.` before `python3`.

## Operating Modes

| Mode Flag | What It Does | Exit Code |
| :--- | :--- | :--- |
| *(default)* | Runs `k_init` corner + PRNG vectors, then uses incremental Z3 + Unique-Call UF Pruning + AST-size greedy packing to flip uncovered branch/`Select`/`Kill` predicates and prove dead polarities UNSAT. | `0` (or `1` on load error) |
| `--maximal-reachability` | Enumerates pairwise-disjoint satisfiable path conditions $\{PC_0, \dots, PC_{m-1}\}$ prioritizing unvisited CFG blocks. Reports `complete=True (EXHAUSTED_ALL_PATHS)` when the blocked solver returns UNSAT. | `0` |
| `--equiv-fuzz` | Runs `tools.coverage.equiv_fuzzer.run_concolic_equiv_fuzz`: seeds inputs from `--side` (`direct` or `valhall`) via (1) `k_init` corner+PRNG vectors, (2) maximal-reachability witnesses, and (3) predicate-coverage vectors within `--time-budget-ms`, concretely tests both `direct_ssa` and `valhall_ssa` under `DiffPolicy`, and auto-shrinks the first `REAL_DIVERGENCE`. | `0` if equivalent (`EQUIV_EXACT` / `EQUIV_WITHIN_ULP` / `SKIP`), `1` if `DIVERGED` |

## CLI Flags

| Flag | Default | Description |
| :--- | :--- | :--- |
| `spv_or_json` | *(required)* | Input `.spv` shader (compiled on the fly via `--bin`) or pre-dumped `--dump-equiv-json` bundle. |
| `--bin` | `./build/spirv2valhall` | Path to `spirv2valhall` binary used to lift `.spv` inputs. |
| `--side` | `direct` | SSA side (`direct` or `valhall`) targeted for synthesis/reachability or used to seed `--equiv-fuzz`. |
| `--canonicalize` | `False` | Run the 6-pass pre-SMT structural canonicalization pipeline (`tools.symbolic_ssa.passes`) on both sides before analysis. |
| `--fp-mode` | `exact` | Z3 floating-point theory: `exact` (`QF_FP`), `real` (`QF_NRA` algebraic), or `uf` (uninterpreted FP ops). |
| `--k-init` | `32` | Number of initial concrete seed vectors (edge-case corners including `0`, `-0`, `±1`, `qNaN`, `±Inf` + deterministic PRNG). |
| `--max-rounds` | `12` | Maximum synthesis rounds in default coverage mode. |
| `--max-paths` | `32` | Maximum disjoint paths to enumerate in `--maximal-reachability` and `--equiv-fuzz`. |
| `--max-unroll` | `1` | Bounded loop unrolling depth (coverage mode adaptively increases up to 3 when loops remain uncovered). |
| `--timeout-ms` | `500` | Per-solve Z3 timeout in milliseconds. |
| `--time-budget-ms` | `5000` | Total wall-clock budget in milliseconds for synthesis, reachability, or `--equiv-fuzz`. |
| `--max-ulp` | `2` | Maximum IEEE-754 FP32 ULP difference treated as `BENIGN_FP_ROUNDING` by `DiffPolicy`. |
| `--shrink` | `False` | In default coverage mode, auto-shrink and localize the first concrete divergence found across `k_init` or synthesized vectors (always enabled in `--equiv-fuzz`). |
| `--dump-vectors` | `None` | Write synthesized override dicts (or disjoint path records in `--maximal-reachability`) to a JSON file. |

## Interpreting Output

### `[EQUIV-FUZZ]` Output Line

```text
[EQUIV-FUZZ] bcn_bc7.spv status=DIVERGED seed=direct vecs=32(init=32,reach=0,cov=0) blocks=0/525 preds=326->326/2746 benign_fp=0 divs=11 elapsed=185.7ms
  [SHRINK] output=ssbo:0:0+0xf2000000 direct=0xff000000 valhall=None kind=REAL_DIVERGENCE ulp=None overrides=0->0 cone=(15264,5895)
    frontier_pcs: 0x30(ARSHIFT.i32 r1, r2, k03020100.b2), 0x38(AND.i32 r2, r2^, u5.w1), ...
```

- **`status`**:
  - `EQUIV_EXACT`: every output and memory/image/atomic side effect matched bit-for-bit on all tested vectors.
  - `EQUIV_WITHIN_ULP`: no real divergences; only FP32 outputs differed within `--max-ulp` or NaN payload bits.
  - `DIVERGED`: at least one `REAL_DIVERGENCE` was observed.
  - `SKIP`: neither side produced observable outputs on the tested vectors.
- **`vecs=32(init=32,reach=0,cov=0)`**: total vectors executed, broken down by phase (`init` = Phase 1 corner+PRNG, `reach` = Phase 2 maximal-reachability witnesses, `cov` = Phase 3 predicate-coverage witnesses). When Phase 1 (`init`) already finds a divergence, Phases 2 and 3 are skipped so the report returns immediately (which is why `blocks=0/525` and `reach=0,cov=0`).
- **`preds=326->326/2746`**: predicate polarity coverage (`initial -> final / total_targets`, where each conditional branch, `Select`, or predicated `Kill` contributes 2 polarities).
- **`divs=11`**: number of tested vectors that produced a `REAL_DIVERGENCE`.
- **`[SHRINK] output=... direct=0xff000000 valhall=None`**:
  - **`valhall=None` (or `direct=None`)** means **asymmetric output/store presence**: one side wrote to that observable output key (`ssbo:0:0+0xf2000000` or `tex:0:6+0x3d42`) on the witness vector, while the other side **did not write to that key at all**. Common causes:
    1. **Unresolved buffer/image provenance on the Valhall side:** `direct_ssa` emitted `PsbStore` with `aux_str="ssbo:0:0"` or `"tex:0:6"`, whereas `valhall_ssa` emitted `PsbStore` with `aux_str=""` (raw 64-bit pointer not resolved to the same named buffer key, landing in `fallback_mem` instead of `ssbo:0:0`).
    2. **Control-flow or predicate divergence:** Valhall skipped the store block or took a different branch/discard path.
    3. **Address offset mismatch:** Both sides wrote to `ssbo:0:0`, but at different byte offsets.
  - **`overrides=0->0`**: delta-shrinker minimized the override dict from `N` to `M` entries (`0->0` means the divergence already reproduced on the default seed vector `vec_idx` with zero Z3 overrides needed).
  - **`cone=(15264,5895)`**: backward SSA cone sizes (number of nodes feeding sink ops) on `(direct_ssa, valhall_ssa)`.
  - **`frontier_pcs`**: earliest Valhall PCs (and disassembly) in the diverging backward cone whose input values matched some node in `direct_ssa` but whose output value did not—pointing directly to the first suspicious lowered Valhall instructions.

### Programmatic API (`tools.coverage.equiv_fuzzer`)

```python
from tools.coverage import DiffPolicy, run_concolic_equiv_fuzz
from tools.symbolic_ssa.passes import canonicalize_ssa_module

d_ssa, _ = canonicalize_ssa_module(bundle["direct_ssa"])
v_ssa, _ = canonicalize_ssa_module(bundle["valhall_ssa"])
res = run_concolic_equiv_fuzz(
    bundle,
    d_ssa,
    v_ssa,
    seed_side="valhall",      # or "direct"
    k_init=32,
    time_budget_ms=5000,
    fp_mode="exact",
    policy=DiffPolicy(max_ulp=2),
    auto_shrink=True,
)
print(res.status, res.diverging_vectors, res.first_divergence)
```

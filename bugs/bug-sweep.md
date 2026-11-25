# Bug Sweep & Repro Maintenance Guide

This guide explains how to run equivalence sweeps with `tools.coverage` and `tools.verify.shader_equiv_verifier`, read the `[SHRINK]` / `[TRIAGE]` / `--trace-diff` diagnostics, root-cause divergences efficiently using the **Cluster-First Playbook**, record shrunken `.spvasm` repros into `bugs/`, and keep `bugs/manifest.md` up to date.

---

## 1. Environment Prerequisites

Before running sweeps or recording `.spvasm` files, ensure:

1. `build/spirv2valhall` is up to date:
   ```bash
   cmake --build build -j$(nproc)
   ```
2. `spirv-as`, `spirv-dis`, `spirv-val`, and `spirv-reduce` are on `PATH` (or set `SPIRV_TOOLS_DIR=/path/to/spirv-tools/bin`).
3. Python can import `z3` and the repo root (`PYTHONPATH=.:/path/to/pydeps`).

---

## 2. Running a Single-Shader Check or Sweep

### Fast Concolic Equivalence Fuzzing (`tools.coverage`)

```bash
python3 -m tools.coverage \
  --canonicalize --k-init 64 --domain udomain --fp-mode real \
  --equiv-fuzz --trace-diff \
  samples/bcn_bc7.spv
```

* Accepts `.spv`, `.spvasm`, or `.json` (pre-dumped equiv bundle).
* Exit code `0` on `EQUIV_EXACT` / `EQUIV_WITHIN_ULP` / `SKIP`; exit code `1` on `DIVERGED`.
* Add `--trace-diff` when triaging a specific failure to print concrete Valhall frontier node evaluations.

### Formal Z3 / CEC Verification (`tools.verify.shader_equiv_verifier`)

```bash
# Fast concrete diff + cutpoint-guided CEC
python3 -m tools.verify.shader_equiv_verifier --cec bugs/frag_unique_05_d392e446_frag_color_0_.a.spvasm

# All-path Z3 miter proof
python3 -m tools.verify.shader_equiv_verifier \
  --domain udomain_backfill --canonicalize --conditioned-unify \
  bugs/frag_unique_05_d392e446_frag_color_0_.a.spvasm
```

---

## 3. Reading `[SHRINK]`, `[TRIAGE]`, and `--trace-diff` Output

1. **Summary line (`[EQUIV-FUZZ]`)**: Check `divs=N` out of `vecs=64`. `divs=1..4` usually signals an edge-case input (overflow, `w==0`, extreme trig `|x|>50`, or a rare branch); `divs=32..64` usually signals a systematic ABI/descriptor or instruction-lowering bug.
2. **Witness line (`[SHRINK]`)**:
   * `vec=<idx>` + `overrides=0->0` gives the exact deterministic input vector to replay.
   * `output=... direct=... valhall=...` shows the first diverging channel and decoded FP32 values (e.g., `0x3efffffd(0.5)`).
   * `cone=(D, V)` shows backward dependency cone sizes.
3. **Triage line (`[TRIAGE]`)**:
   * `MODELING_CANONICALIZER`: Raw (uncanonicalized) SSA already agrees on this witness — run pass-by-pass bisection and `verify_ssa_dominance(can_ssa)` first!
   * `LOWERING_ADDR_OR_SWIZZLE`: A `PsbLoad` read out-of-bounds of `ebin_rodata` — inspect 64-bit operand/swizzle lowering in `specs/lowering/valhall_lowering.json`.
   * `MODELING_ENV_OR_ABI`: Known ABI/modeling gap (`TEX_GRADIENT`, `CUBEFACE`, `SHADD sample`, `subnormal_ftz`, `concrete_step_limit_truncated`, `unresolved_pilot_fau`, `extreme_trig_range`, or `nan_trig_clamp`).
   * `SPIRV2SSA_OR_MODELING_TEX_GATHER`: `TexGather` / `OpImageGather` mismatch.
   * `LOWERING_OR_SEMANTICS`: No known modeling gap matched — likely a real ISA lowering or `spirv2ssa` bug.
4. **`cone_ops_diff`**: Highlights SSA ops unique to one side's backward cone.
5. **`frontier_trace` (`--trace-diff`)**: Lists the earliest non-constant Valhall SSA nodes whose concrete value never appears in the Direct SSA cone, alongside their exact concrete operand values, output value, and Valhall disassembly.

---

## 4. Cluster-First Root-Cause Playbook (Improved in v44)

Instead of debugging shaders in arbitrary filename order, follow this 4-step triage funnel:

1. **Sort by `(suspect, divs)`**: Start with `suspect=LOWERING_OR_SEMANTICS` and the **smallest `divs` count (`divs=1..6`)**. Low-divergence shaders have only 1–2 bugs left, so fixing one root cause often flips multiple shaders straight to `EQUIV`!
2. **Check `unresolved_pilot_fau` and `mbs2_context` first**: Before tracing arithmetic, check whether `signals=[...]` reports `unresolved_pilot_fau(...)`, `concrete_step_limit_truncated(...)`, `extreme_trig_range(...)`, or `nan_trig_clamp(...)`. A single unresolved FAU slot (like `BaseVertex`/`BaseInstance` at `byte_offset=76/80`) can explain 10+ shaders at once.
3. **Find the first F32 `NaN`/`Inf` origin on NaN-vs-finite diffs**: Whenever one side is `NaN` and the other is finite, scan FP32-typed nodes (`n["type"] == "F32"`) in topological order for the first node whose inputs are finite and whose output is `NaN`/`Inf`. This immediately distinguishes `FMA.clamp_m1_1` NaN suppression from `FRCP(FRSQ(+Inf))` overflow or `FMIN`/`FMAX` NaN-propagation bugs.
4. **Walk backward 5–6 hops from the diverging store on both sides**: When both sides are finite with different magnitudes/signs (like `d5ded472` `25.59` vs `-13.30`), dump the 6-hop backward slice from the diverging `Store*` node on Direct and Valhall side-by-side. The first `Select`/`Phi`/`CSEL` or ALU op whose values disagree is the exact divergence point.

---

## 5. Recording Shrunken `.spvasm` Repros (`--record`) & Ops Workflow

```bash
python3 -m tools.coverage \
  --canonicalize --k-init 64 --domain udomain --fp-mode real \
  --equiv-fuzz --record bugs/ \
  --record-timeout-s 20 --record-steps 120 \
  path/to/failing.spv
```

* `recorder.py` runs `spirv-reduce` inside an isolated temp directory (`cwd=td`), validates every candidate with `spirv-val`, rejects candidates that slip from a value divergence into an `OpUndef` channel-asymmetry (`None`), and checkpoints `best.spv` so timeouts still emit the smallest verified repro.
* **Opening a bug**: Record into `bugs/`, verify replay via `tools.coverage`, and add an `OPEN` row to `bugs/manifest.md`.
* **Closing a bug**: Verify the repro now passes, add a targeted unit test in `tools/symbolic_ssa/test_v*.py` registered in `CMakeLists.txt`, move the row in `bugs/manifest.md` to **Closed Bugs**, and verify `cmake --build build`, `ctest --test-dir build --output-on-failure`, and `ruff check` all pass with 0 errors.

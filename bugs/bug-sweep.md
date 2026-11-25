# Bug Sweep & Repro Maintenance Guide

This guide explains how to run equivalence sweeps with `tools.coverage` and `tools.verify.shader_equiv_verifier`, read the `[SHRINK]` / `[TRIAGE]` / `--trace-diff` diagnostics, record shrunken `.spvasm` repros into `bugs/`, and keep `bugs/manifest.md` up to date.

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

The standard sweep command used across Checkpoint v42 and v43 is:

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

Once you have a small `.spv` or `.spvasm` repro, you can cross-check with both concrete-equivalence checking (`--cec`) and all-path Z3 proving:

```bash
# Fast concrete diff + cutpoint-guided CEC
python3 -m tools.verify.shader_equiv_verifier --cec bugs/frag_unique_05_d392e446_frag_color_0_.a.spvasm

# All-path Z3 miter proof
python3 -m tools.verify.shader_equiv_verifier \
  --domain udomain_backfill --canonicalize --conditioned-unify \
  bugs/frag_unique_05_d392e446_frag_color_0_.a.spvasm
```

### Batch Sweep Across `samples/*.spv` + Corpus

To run a reproducible 75-shader sweep (all 25 `samples/*.spv` + 50 random shaders from a corpus directory using `seed=42`):

```bash
python3 - << 'EOF'
import concurrent.futures, glob, json, os, random, subprocess, sys

samples = sorted(glob.glob("samples/*.spv"))
corpus_all = sorted(glob.glob("/path/to/spirv_samples/*.spv"))
corpus50 = random.Random(42).sample(corpus_all, min(50, len(corpus_all)))
targets = [("samples", p) for p in samples] + [("corpus50", p) for p in corpus50]

def run_one(item):
    grp, path = item
    cmd = [
        sys.executable, "-m", "tools.coverage",
        "--canonicalize", "--k-init", "64", "--domain", "udomain",
        "--fp-mode", "real", "--equiv-fuzz", path,
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
        return {"group": grp, "name": os.path.basename(path), "rc": p.returncode, "out": (p.stdout + p.stderr).strip()}
    except subprocess.TimeoutExpired:
        return {"group": grp, "name": os.path.basename(path), "rc": -1, "out": "TIMEOUT"}

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    rows = list(ex.map(run_one, targets))

with open("/tmp/sweep_results.json", "w") as f:
    json.dump(rows, f, indent=2)
print(f"Wrote {len(rows)} results to /tmp/sweep_results.json")
EOF
```

---

## 3. Reading `[SHRINK]`, `[TRIAGE]`, and `--trace-diff` Output

Example output from `samples/bcn_bc7.spv`:

```text
[EQUIV-FUZZ] bcn_bc7.spv status=DIVERGED seed=direct vecs=64(init=64,reach=0,cov=0) blocks=0/19 preds=357->357/1430 benign_fp=0 divs=1 elapsed=156.2ms
  [SHRINK] vec=32 output=ssbo:0:0+0xea475f6c direct=0xff7bce63 valhall=0xffffc652 kind=REAL_DIVERGENCE ulp=None overrides=0->0 cone=(3530,3429)
    frontier_pcs: 0x0(ICMP.s32.m1.ge r0, r61, u4.w0), 0x30(ARSHIFT.i32 r1, r2, k03020100.b2), ...
  [TRIAGE] suspect=LOWERING_OR_SEMANTICS
    cone_ops_diff: direct_only=[ConstS32,FDiv,ICmpSLe,IMul64,LogicalOr,SDiv,SMod] valhall_only=[Bfi,BitAnd64,BitNot,BitXor,ICmpSLt,ICmpUGt,LoadFau,Pack64]
    frontier_trace:
      0x38 v64 LoadFau() -> 0xfffffffc  ; AND.i32 r2, r2^, u5.w1
      0x22d8 v9227 ShrU(v9219=0x416f59f4, v9226=0x00000003) -> 0x082deb3e  ; RSHIFT_AND.i32 r2, r2^, k03020100.b3, u10.w0
```

Read it top to bottom:

1. **Summary line (`[EQUIV-FUZZ]`)**: `divs=1` out of `vecs=64` tells you this is an edge-case divergence (not a systematic every-vector bug like `divs=64`).
2. **Witness line (`[SHRINK]`)**:
   * `vec=32` + `overrides=0->0` gives the exact deterministic input vector needed to replay the failure.
   * `output=ssbo:0:0+0xea475f6c direct=0xff7bce63 valhall=0xffffc652` shows the first diverging channel and both concrete values (FP channels also print decoded floats like `0x3efffffd(0.5)`).
   * `cone=(3530,3429)` gives the backward dependency cone sizes (Direct SSA nodes, Valhall SSA nodes).
3. **Triage line (`[TRIAGE]`)**:
   * `suspect=MODELING_CANONICALIZER`: Raw (uncanonicalized) SSA already agrees on this witness — inspect `tools/symbolic_ssa/passes/` first!
   * `suspect=LOWERING_ADDR_OR_SWIZZLE`: A `PsbLoad` in the Valhall cone read out-of-bounds of `ebin_rodata` — inspect 64-bit operand/swizzle lowering in `specs/lowering/valhall_lowering.json`.
   * `suspect=MODELING_ENV_OR_ABI`: Known ABI/modeling gap (`TEX_GRADIENT`, `CUBEFACE`, `SHADD sample`, `subnormal_ftz`, or `concrete_step_limit_truncated`).
   * `suspect=SPIRV2SSA_OR_MODELING_TEX_GATHER`: `TexGather` / `OpImageGather` mismatch.
   * `suspect=LOWERING_OR_SEMANTICS`: No known modeling gap matched — likely a real ISA lowering or `spirv2ssa` bug.
4. **`cone_ops_diff`**: Highlights SSA ops unique to one side's backward cone (e.g., `RoundEven` missing on Valhall, or `Bfi`/`BitAnd64` only on Valhall).
5. **`frontier_trace` (`--trace-diff`)**: Lists the earliest Valhall SSA nodes whose concrete value never appears in the Direct SSA cone, alongside their exact concrete input values, output value, and Valhall disassembly. Look here for unexpected bitmasks, wrong lane extracts, or `LoadSidebandU64() -> 0x00000000`.

---

## 4. Recording Shrunken `.spvasm` Repros (`--record`)

Whenever you uncover a new divergence cluster worth tracking:

```bash
python3 -m tools.coverage \
  --canonicalize --k-init 64 --domain udomain --fp-mode real \
  --equiv-fuzz --record bugs/ \
  --record-timeout-s 20 --record-steps 120 \
  path/to/failing.spv
```

* `--record bugs/` automatically names the file `<shader_stem>_<sanitized_output>.spvasm` inside `bugs/`.
* `spirv-reduce` runs inside an isolated temp directory (`cwd=td`), validates every candidate with `spirv-val`, rejects candidates that slip from a value mismatch into a `CHANNEL_ASYMMETRY` (`None`), and checkpoints the smallest verified `.spv` so even a timeout returns the best reduction found so far.
* Pass `--no-reduce` if you want an immediate full disassembly without running `spirv-reduce`.

---

## 5. Ops Workflow: Keeping `bugs/` Up to Date

1. **When opening a new bug**:
   * Record a `.spvasm` repro into `bugs/` using `--record bugs/`.
   * Verify it replays cleanly with `python3 -m tools.coverage --canonicalize --equiv-fuzz bugs/<file>.spvasm`.
   * Add an `OPEN` row to `bugs/manifest.md` with the next `BUG-1xx` ID, layer, repro path, and root-cause hypothesis.
2. **When fixing a bug**:
   * Verify the recorded `bugs/<file>.spvasm` now returns `EQUIV_EXACT` or `EQUIV_WITHIN_ULP` (or `UNSAT` in `shader_equiv_verifier`).
   * Add a targeted unit/regression test in `tools/symbolic_ssa/test_v*.py` and register it in `CMakeLists.txt`.
   * Move the entry in `bugs/manifest.md` from **Open Bugs** to **Closed Bugs** with the checkpoint version, fix location, and test name.
   * Either keep the `.spvasm` in `bugs/` if a regression test references it directly, or remove it once covered by a unit test so `bugs/` stays focused on active repros.
3. **Before committing**:
   * Run `cmake --build build -j$(nproc)` and `ctest --test-dir build --output-on-failure`.
   * Run `ruff check` on all modified/added Python files to keep lint clean.

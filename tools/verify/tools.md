# `tools/verify` — Instruction-Level Formal Verification Guide

`tools/verify` contains the **single-instruction formal verification** infrastructure for `valhall2spirv`: it extracts symbolic Z3 semantics directly from `specs/lowering/valhall_lowering.json` and `src/valhall_pass1_handlers.cpp` (via Clang AST JSON) and proves equivalence against an independent Mesa/Panfrost Z3 reference oracle across all 627 Valhall XML instruction variants.

> **Note:** Shader-level equivalence verification (`shader_equiv_verifier.py`, `cec.py`, `spvasm_io.py`, and `shader-equiv-verifier.md`) lives in [`tools/equivalence/`](../equivalence/tools.md).

---

## Modules & Responsibilities

| Module | Role |
| :--- | :--- |
| `all_paths_miter.py` | End-to-end all-paths Z3 miter driver: compares extracted `valhall2spirv` Pass-1 lowering traces against `panfrost_z3_model.py` across every modifier combination and all 627 XML instructions. |
| `lowering_semantics_extractor.py` | Symbolically executes `@INLINE_OPS` rules (from `tools.specs.dsl_parser` / `valhall_lowering_dsl.py`) and C++ host handlers (via `clang_ast_evaluator.py`) to produce per-path `LoweringTrace` objects. |
| `clang_ast_evaluator.py` | Symbolic interpreter over Clang AST JSON (`tools/verify/generated/valhall_host_handlers.json`) for C++ Pass-1 handlers in `src/valhall_pass1_handlers.cpp`. |
| `operands_spec_loader.py` | Loads declarative operand fetchers from `specs/lowering/valhall_lowering.json["operands"]` into `CppFunctionDef` ASTs for `CppSymbolicEvaluator`. |
| `panfrost_z3_model.py` | Independent Z3 bitvector/IEEE-754 reference semantics derived from Mesa Panfrost (`bifrost`/`valhall` ISA) covering source modifiers, swizzles, clamps, and opcodes. |
| `ssa_z3_eval.py` | Evaluates extracted single-instruction `SymbolicSsaModule` graphs into Z3 expressions using `tools.symbolic_ssa`. |

---

## Quick Start & CLI / Script Usage

Always run from the repository root with `PYTHONPATH=.:/path/to/z3_site_packages`.

### 1. Run the Full 627-Instruction All-Paths Miter Proof
```bash
python3 -m tools.verify.all_paths_miter
```
Or programmatically for a single mnemonic or full sweep:
```python
from tools.verify import check_instruction_miter, run_all_627_miter

# Check a single instruction across all modifier paths
res = check_instruction_miter("IADD.u32")
print(res.mnemonic, res.status, res.paths_checked)

# Run the full 627-instruction miter
summary = run_all_627_miter()
print(summary)
```

### 2. Inspect Extracted Symbolic Lowering Traces for an Instruction
```python
from tools.verify import LoweringSemanticsExtractor

ext = LoweringSemanticsExtractor()
traces = ext.extract_instruction("FMA.f32")
for i, tr in enumerate(traces):
    print(f"Path {i}: cond={tr.path_condition} nodes={len(tr.ssa.nodes)}")
```

### 3. Evaluate Panfrost Z3 Reference Oracle Directly
```python
from tools.verify.panfrost_z3_model import PANFROST_SEMANTICS, eval_panfrost_instruction

print(f"Panfrost Z3 oracle covers {len(PANFROST_SEMANTICS)} mnemonics")
```

### 4. Regenerating `valhall_host_handlers.json` (Requires LLVM-21 `libclang`)
When `third_party/prebuilts/llvm-21` is present, CMake automatically builds `clang_ast_dumper` and regenerates `tools/verify/generated/valhall_host_handlers.json` whenever `src/valhall_pass1_handlers.cpp` changes:
```bash
cmake -B build && cmake --build build --target valhall_host_handlers_ast
ctest --test-dir build --output-on-failure -R "test_clang_ast_dumper"
```

---

## Running Unit Tests

```bash
ctest --test-dir build --output-on-failure \
  -R "test_panfrost_z3_model|test_lowering_extractor|test_all_paths_miter|test_clang_ast_dumper"
```

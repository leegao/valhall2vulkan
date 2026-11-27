# `tools/specs` — Code Generators & Lowering DSL Guide

`tools/specs` contains all specification-driven code generators and the LL(1) lowering DSL compiler for `valhall2spirv`. Every generated C++ header and source file under `src/generated/` is produced deterministically by these tools from declarative XML and JSON specs in `specs/`.

---

## Tools & Modules Overview

| File | Input Spec(s) | Generated Output(s) |
| :--- | :--- | :--- |
| `gen_valhall_spec.py` | `specs/valhall/ISA.xml`, `specs/kraid/isa-v9-v14.xml`, `specs/kraid/ops.rs`, `specs/kraid/encode_v9.rs`, `specs/csf/v11.xml` (+ `v12..v14`) | `src/generated/valhall_isa_generated.{h,cpp}`, `src/generated/valhall_csf_descriptors_generated.{h,cpp}` |
| `gen_valhall_tables.py` | `specs/kraid/isa-v9-v14.xml`, `specs/lowering/valhall_lowering.json`, `specs/lowering/valhall_tiles.json` | `src/generated/valhall_tables_generated.{h,cpp}`, `src/generated/valhall_operands_generated.cpp`, `src/generated/valhall_pass1_lowering_generated.inc` |
| `gen_spirv_builder.py` | `specs/spirv/spirv.core.grammar.json` | `src/generated/valhall_spirv_builder_generated.{h,cpp}` |
| `gen_spirv_parser.py` | `specs/spirv/spirv.core.grammar.json` | `src/generated/valhall_spirv_parser_generated.{h,cpp}` |
| `dsl_parser.py` | Library (`specs/lowering/spec.md` LL(1) grammar) | Parses `@INLINE_OPS` and operand-fetcher bodies into strict typed ASTs (`parse_inline_ops`). |
| `valhall_lowering_dsl.py` | Library (consumes `dsl_parser.py` ASTs) | Compiles `@INLINE_OPS` rules and operand fetchers into C++ code for `gen_valhall_tables.py`. |
| `isa_spec_index.py` | Library (`specs/kraid/isa-v9-v14.xml`) | Shared Valhall ISA XML specification reader (`IsaSpecIndex`, `load_isa_spec_index`). |

---

## CLI Usage

All generators can be run directly from the repository root or via CMake's `regenerate_specs` target.

### 1. Regenerate All Specs via CMake
```bash
cmake --build build --target regenerate_specs
```
*(Note: `gen_valhall_tables.py` also runs automatically during `cmake --build build` whenever `specs/lowering/valhall_lowering.json`, `specs/lowering/valhall_tiles.json`, `tools/specs/gen_valhall_tables.py`, or `tools/specs/valhall_lowering_dsl.py` changes.)*

### 2. `gen_valhall_tables.py` — Lowering Tables, Operands & Structural Validation
```bash
# Regenerate lowering tables, operands, and Pass-1 inline lowering switch
python3 tools/specs/gen_valhall_tables.py

# Verify src/generated/valhall_operands_generated.cpp is up to date without overwriting
python3 tools/specs/gen_valhall_tables.py --check-operands

# Custom spec paths / output directory
python3 tools/specs/gen_valhall_tables.py \
  --xml specs/kraid/isa-v9-v14.xml \
  --lowering specs/lowering/valhall_lowering.json \
  --tiles specs/lowering/valhall_tiles.json \
  --out-dir src/generated
```
Validates 100% coverage of all 627 XML instructions, checks modifier names and values against the ISA XML, and compiles all `@INLINE_OPS` rules via `valhall_lowering_dsl.py`.

### 3. `gen_valhall_spec.py` — Valhall ISA Decoder & CSF Descriptors
```bash
python3 tools/specs/gen_valhall_spec.py \
  --valhall-xml specs/valhall/ISA.xml \
  --kraid-xml specs/kraid/isa-v9-v14.xml \
  --kraid-ops specs/kraid/ops.rs \
  --kraid-encode specs/kraid/encode_v9.rs \
  --csf-xml specs/csf/v11.xml \
  --out-dir src/generated
```

### 4. `gen_spirv_builder.py` & `gen_spirv_parser.py` — SPIR-V Grammar Codegen
```bash
# Regenerate typed SPIR-V 1.x binary emitter (879 instructions, 42 value enums, 16 bit enums)
python3 tools/specs/gen_spirv_builder.py

# Regenerate typed SPIR-V instruction views and parser (879 opcodes, 58 typed views)
python3 tools/specs/gen_spirv_parser.py
```

### 5. `dsl_parser.py` — Standalone Lowering DSL Self-Check
```bash
python3 tools/specs/dsl_parser.py
```
Parses every `@INLINE_OPS` rule and operand definition in `specs/lowering/valhall_lowering.json` to verify strict LL(1) grammar compliance.

---

## Running Unit Tests

```bash
ctest --test-dir build --output-on-failure \
  -R "test_isa_decoder|test_csf_descriptors|test_spirv_builder|test_lowering_extractor"
```

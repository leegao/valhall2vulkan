#!/usr/bin/env python3
"""
Automated SPIR-V Parser & Metadata Table Code Generator.

Parses specs/spirv/spirv.core.grammar.json (Khronos SPIR-V Core Grammar) and
generates:
  - src/generated/valhall_spirv_parser_generated.h
  - src/generated/valhall_spirv_parser_generated.cpp

Uses third_party/spirv-reflect/include/spirv/unified1/spirv.h enum definitions
directly so opcode/decoration/builtin/storage-class constants are never duplicated.
Also generates zero-allocation typed operand views directly from the grammar's
operand specifications so consumers never index raw instruction word arrays.
"""

import json
import re
import sys
from pathlib import Path

TERMINATOR_OPS = {
    "OpBranch",
    "OpBranchConditional",
    "OpSwitch",
    "OpKill",
    "OpReturn",
    "OpReturnValue",
    "OpUnreachable",
    "OpTerminateInvocation",
    "OpIgnoreIntersectionKHR",
    "OpTerminateRayKHR",
    "OpEmitMeshTasksEXT",
}

# Core SPIR-V instructions for which typed operand view structs + parsers are generated
# directly from spirv.core.grammar.json operand definitions.
TYPED_VIEW_OPS = [
    "OpBranch",
    "OpBranchConditional",
    "OpSwitch",
    "OpSelectionMerge",
    "OpLoopMerge",
    "OpReturnValue",
    "OpVariable",
    "OpPhi",
    "OpAccessChain",
    "OpInBoundsAccessChain",
    "OpPtrAccessChain",
    "OpInBoundsPtrAccessChain",
    "OpArrayLength",
    "OpLoad",
    "OpStore",
    "OpCopyMemory",
    "OpFunctionCall",
    "OpExtInst",
    "OpAtomicLoad",
    "OpAtomicStore",
    "OpAtomicExchange",
    "OpAtomicCompareExchange",
    "OpAtomicCompareExchangeWeak",
    "OpAtomicIIncrement",
    "OpAtomicIDecrement",
    "OpAtomicIAdd",
    "OpAtomicISub",
    "OpAtomicSMin",
    "OpAtomicUMin",
    "OpAtomicSMax",
    "OpAtomicUMax",
    "OpAtomicAnd",
    "OpAtomicOr",
    "OpAtomicXor",
    "OpAtomicFlagTestAndSet",
    "OpAtomicFlagClear",
    "OpAtomicFMinEXT",
    "OpAtomicFMaxEXT",
    "OpAtomicFAddEXT",
    "OpImageSampleImplicitLod",
    "OpImageSampleExplicitLod",
    "OpImageSampleDrefImplicitLod",
    "OpImageSampleDrefExplicitLod",
    "OpImageSampleProjImplicitLod",
    "OpImageSampleProjExplicitLod",
    "OpImageFetch",
    "OpImageGather",
    "OpImageDrefGather",
    "OpImageRead",
    "OpImageWrite",
    "OpVectorShuffle",
    "OpCompositeConstruct",
    "OpCompositeExtract",
    "OpCompositeInsert",
    "OpSelect",
    "OpBitFieldInsert",
    "OpBitFieldSExtract",
    "OpBitFieldUExtract",
]


def _collect_enum_entries(operand_kinds, kind_name):
    out = []
    seen = set()
    for ok in operand_kinds:
        if ok.get("kind") != kind_name:
            continue
        for e in ok.get("enumerants", []):
            val = e.get("value")
            if isinstance(val, str):
                val = int(val, 0)
            if val in seen:
                continue
            seen.add(val)
            out.append((val, e["enumerant"]))
    out.sort(key=lambda x: x[0])
    return out


def _sanitize_field_name(operand: dict, fallback_idx: int) -> str:
    kind = operand.get("kind", "")
    if kind == "IdResultType":
        return "result_type"
    if kind == "IdResult":
        return "result_id"
    raw = operand.get("name", "")
    if not raw:
        raw = kind or f"operand_{fallback_idx}"
    # Strip trailing variadic ellipses or comma-separated lists like "Operand 1, Operand 2, ..."
    if "," in raw:
        raw = raw.split(",")[0].strip()
    raw = raw.replace("'", "").replace("\n", " ").strip()
    # Convert CamelCase or spaces to snake_case
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", raw)
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    if not s or s[0].isdigit():
        s = f"op_{s}"
    if s in ("default", "operator", "switch", "return", "register", "signed", "unsigned"):
        s = f"{s}_val"
    return s


def _build_typed_view(opname: str, ins: dict) -> list[str]:
    """Generate a typed view struct + static parse() method from a grammar instruction entry."""
    short = opname[2:] if opname.startswith("Op") else opname
    struct_name = f"Spv{short}View"
    operands = ins.get("operands", [])

    fields: list[tuple[str, str, str]] = []  # (kind_tag, field_name, orig_kind)
    used_names: set[str] = set()
    for idx, opnd in enumerate(operands):
        quant = opnd.get("quantifier", "")
        fname = _sanitize_field_name(opnd, idx)
        if fname in used_names:
            fname = f"{fname}_{idx}"
        used_names.add(fname)
        if quant == "*":
            fields.append(("span", fname, opnd.get("kind", "")))
        elif quant == "?":
            fields.append(("opt", fname, opnd.get("kind", "")))
        else:
            fields.append(("req", fname, opnd.get("kind", "")))

    req_count = sum(1 for k, _, _ in fields if k == "req")
    extra_opcodes = {
        "OpAccessChain": ["SpvOpInBoundsAccessChain"],
        "OpPtrAccessChain": ["SpvOpInBoundsPtrAccessChain"],
        "OpAtomicCompareExchange": ["SpvOpAtomicCompareExchangeWeak"],
    }.get(opname, [])
    lines = [
        f"struct {struct_name} {{",
        "    static constexpr bool kIsSpvView = true;",
        f"    static constexpr SpvOp kOpcode = Spv{opname};",
        "",
    ]
    for k, fname, _ in fields:
        if k == "req":
            lines.append(f"    uint32_t {fname} = 0;")
        elif k == "opt":
            lines.append(f"    bool has_{fname} = false;")
            lines.append(f"    uint32_t {fname} = 0;")
        else:
            lines.append(f"    SpvWordSpan {fname}{{}};")

    lines.append("")
    match_expr = f"op == static_cast<uint16_t>(Spv{opname})"
    for eo in extra_opcodes:
        match_expr += f" || op == static_cast<uint16_t>({eo})"
    lines.append(f"    static constexpr bool matches_opcode(uint16_t op) {{ return {match_expr}; }}")
    lines.append("")
    lines.append(f"    static bool parse(const uint32_t* words, size_t word_count, {struct_name}* out) {{")
    lines.append(f"        if (!words || !out || word_count < {1 + req_count}u) return false;")
    lines.append("        if (!matches_opcode(static_cast<uint16_t>(words[0] & 0xFFFFu))) return false;")
    lines.append(f"        *out = {struct_name}{{}};")
    lines.append("        size_t idx = 1u;")
    for k, fname, _ in fields:
        if k == "req":
            lines.append(f"        out->{fname} = words[idx++];")
        elif k == "opt":
            lines.append(f"        if (idx < word_count) {{ out->has_{fname} = true; out->{fname} = words[idx++]; }}")
        else:
            lines.append(f"        if (idx < word_count) {{ out->{fname} = {{words + idx, word_count - idx}}; idx = word_count; }}")
    lines.append("        return true;")
    lines.append("    }")
    lines.append("")
    lines.append(f"    static {struct_name} must_parse(const uint32_t* words, size_t word_count) {{")
    lines.append(f"        {struct_name} out{{}};")
    lines.append("        if (!parse(words, word_count, &out)) {")
    lines.append("            const uint16_t got_op = (words && word_count > 0u) ? static_cast<uint16_t>(words[0] & 0xFFFFu) : 0u;")
    lines.append(f'            std::fprintf(stderr, "FATAL: {struct_name}::must_parse failed (got_op=%u, word_count=%zu, expected_op=%u, min_words={1 + req_count})\\n",')
    lines.append(f"                         unsigned(got_op), word_count, unsigned(kOpcode));")
    lines.append("            std::abort();")
    lines.append("        }")
    lines.append("        return out;")
    lines.append("    }")

    # Special helper for OpSwitch: decode (literal, label) pairs using selector bit width.
    if opname == "OpSwitch":
        lines.append("")
        lines.append("    std::vector<std::pair<uint64_t, uint32_t>> decode_cases(uint32_t selector_width_bits) const {")
        lines.append("        std::vector<std::pair<uint64_t, uint32_t>> out;")
        lines.append("        if (selector_width_bits == 64u) {")
        lines.append("            for (size_t k = 0; k + 2u < target.count; k += 3u) {")
        lines.append("                uint64_t lit = uint64_t(target[k]) | (uint64_t(target[k + 1u]) << 32);")
        lines.append("                out.emplace_back(lit, target[k + 2u]);")
        lines.append("            }")
        lines.append("        } else {")
        lines.append("            for (size_t k = 0; k + 1u < target.count; k += 2u) {")
        lines.append("                out.emplace_back(uint64_t(target[k]), target[k + 1u]);")
        lines.append("            }")
        lines.append("        }")
        lines.append("        return out;")
        lines.append("    }")

    # Special helper for OpPhi: typed (variable, parent) pair accessors.
    if opname == "OpPhi":
        lines.append("")
        lines.append("    size_t incoming_count() const { return variable.count / 2u; }")
        lines.append("    uint32_t incoming_value(size_t i) const { return variable[2u * i]; }")
        lines.append("    uint32_t incoming_parent(size_t i) const { return variable[2u * i + 1u]; }")

    lines.append("};")
    lines.append("")
    return lines


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    grammar_path = repo_root / "specs" / "spirv" / "spirv.core.grammar.json"
    out_dir = repo_root / "src" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not grammar_path.exists():
        print(f"Error: {grammar_path} not found", file=sys.stderr)
        return 1

    with open(grammar_path, "r", encoding="utf-8") as f:
        grammar = json.load(f)

    instructions = grammar["instructions"]
    operand_kinds = grammar["operand_kinds"]

    # Deduplicate by opcode (keep first canonical entry) and index by opname
    by_opcode = {}
    by_opname = {}
    for ins in instructions:
        op = int(ins["opcode"])
        by_opname.setdefault(ins["opname"], ins)
        if op in by_opcode:
            continue
        ops = ins.get("operands", [])
        has_rt = any(o.get("kind") == "IdResultType" for o in ops)
        has_rid = any(o.get("kind") == "IdResult" for o in ops)
        min_words = 1 + sum(1 for o in ops if o.get("quantifier") not in ("?", "*"))
        by_opcode[op] = {
            "opcode": op,
            "opname": ins["opname"],
            "has_rt": has_rt,
            "has_rid": has_rid,
            "min_words": min_words,
            "is_term": ins["opname"] in TERMINATOR_OPS,
        }

    sorted_ops = [by_opcode[k] for k in sorted(by_opcode.keys())]
    builtins = _collect_enum_entries(operand_kinds, "BuiltIn")
    exec_models = _collect_enum_entries(operand_kinds, "ExecutionModel")
    storage_classes = _collect_enum_entries(operand_kinds, "StorageClass")
    decorations = _collect_enum_entries(operand_kinds, "Decoration")

    h_lines = [
        "// AUTO-GENERATED BY tools/gen_spirv_parser.py -- DO NOT EDIT MANUALLY",
        "// Derived from specs/spirv/spirv.core.grammar.json",
        "#pragma once",
        "#include <spirv/unified1/spirv.h>",
        "#include <cstddef>",
        "#include <cstdint>",
        "#include <cstdio>",
        "#include <cstdlib>",
        "#include <utility>",
        "#include <vector>",
        "",
        "namespace valhall::spirv_parser {",
        "",
        "struct SpvWordSpan {",
        "    const uint32_t* data = nullptr;",
        "    size_t count = 0;",
        "",
        "    uint32_t operator[](size_t i) const { return data[i]; }",
        "    bool empty() const { return count == 0; }",
        "    const uint32_t* begin() const { return data; }",
        "    const uint32_t* end() const { return data + count; }",
        "};",
        "",
        "struct SpvOpInfo {",
        "    uint16_t opcode;",
        "    const char* name;",
        "    bool has_result_type;",
        "    bool has_result_id;",
        "    uint8_t min_words;",
        "    bool is_terminator;",
        "};",
        "",
        "struct DecodedSpvInst {",
        "    SpvOp opcode = SpvOpNop;",
        "    uint16_t word_count = 0;",
        "    uint32_t result_type = 0;",
        "    uint32_t result_id = 0;",
        "    uint16_t operand_start = 1;",
        "    const uint32_t* words = nullptr;",
        "",
        "    size_t operand_count() const {",
        "        return (word_count > operand_start) ? static_cast<size_t>(word_count - operand_start) : 0u;",
        "    }",
        "    uint32_t operand(size_t i) const { return words[operand_start + i]; }",
        "    SpvWordSpan operands() const {",
        "        return {words ? words + operand_start : nullptr, operand_count()};",
        "    }",
        "};",
        "",
        "const SpvOpInfo* lookup_spv_op_info(uint16_t opcode);",
        "const char* spv_op_name(uint16_t opcode);",
        "const char* spv_builtin_name(uint32_t builtin);",
        "const char* spv_execution_model_name(uint32_t exec_model);",
        "const char* spv_storage_class_name(uint32_t storage_class);",
        "const char* spv_decoration_name(uint32_t decoration);",
        "",
        "/// Decode a single SPIR-V instruction header and populate result_type / result_id / operand_start.",
        "bool decode_spv_instruction(const uint32_t* words, size_t remaining_words, DecodedSpvInst* out);",
        "",
    ]

    generated_views = 0
    for opname in TYPED_VIEW_OPS:
        ins = by_opname.get(opname)
        if ins is None:
            print(f"Warning: {opname} not found in grammar", file=sys.stderr)
            continue
        h_lines.extend(_build_typed_view(opname, ins))
        generated_views += 1

    h_lines.extend([
        "/// Shared prefix view for all value-producing SPIR-V atomic instructions.",
        "struct SpvAtomicHeaderView {",
        "    static constexpr bool kIsSpvView = true;",
        "    uint32_t result_type = 0;",
        "    uint32_t result_id = 0;",
        "    uint32_t pointer = 0;",
        "    uint32_t memory = 0;",
        "    uint32_t semantics = 0;",
        "",
        "    static constexpr bool matches_opcode(uint16_t op) {",
        "        switch (static_cast<SpvOp>(op)) {",
        "            case SpvOpAtomicLoad: case SpvOpAtomicExchange:",
        "            case SpvOpAtomicCompareExchange: case SpvOpAtomicCompareExchangeWeak:",
        "            case SpvOpAtomicIIncrement: case SpvOpAtomicIDecrement:",
        "            case SpvOpAtomicIAdd: case SpvOpAtomicISub:",
        "            case SpvOpAtomicSMin: case SpvOpAtomicUMin:",
        "            case SpvOpAtomicSMax: case SpvOpAtomicUMax:",
        "            case SpvOpAtomicAnd: case SpvOpAtomicOr: case SpvOpAtomicXor:",
        "            case SpvOpAtomicFlagTestAndSet:",
        "            case SpvOpAtomicFMinEXT: case SpvOpAtomicFMaxEXT: case SpvOpAtomicFAddEXT:",
        "                return true;",
        "            default: return false;",
        "        }",
        "    }",
        "    static bool parse(const uint32_t* words, size_t word_count, SpvAtomicHeaderView* out) {",
        "        if (!words || !out || word_count < 6u) return false;",
        "        if (!matches_opcode(static_cast<uint16_t>(words[0] & 0xFFFFu))) return false;",
        "        out->result_type = words[1]; out->result_id = words[2];",
        "        out->pointer = words[3]; out->memory = words[4]; out->semantics = words[5];",
        "        return true;",
        "    }",
        "    static SpvAtomicHeaderView must_parse(const uint32_t* words, size_t word_count) {",
        "        SpvAtomicHeaderView out{};",
        "        if (!parse(words, word_count, &out)) {",
        '            std::fprintf(stderr, "FATAL: SpvAtomicHeaderView::must_parse failed (word_count=%zu)\\n", word_count);',
        "            std::abort();",
        "        }",
        "        return out;",
        "    }",
        "};",
        "",
        "/// Shared operand view for binary read-modify-write SPIR-V atomic instructions.",
        "struct SpvAtomicRmwView {",
        "    static constexpr bool kIsSpvView = true;",
        "    uint32_t result_type = 0;",
        "    uint32_t result_id = 0;",
        "    uint32_t pointer = 0;",
        "    uint32_t memory = 0;",
        "    uint32_t semantics = 0;",
        "    uint32_t value = 0;",
        "",
        "    static constexpr bool matches_opcode(uint16_t op) {",
        "        switch (static_cast<SpvOp>(op)) {",
        "            case SpvOpAtomicIAdd: case SpvOpAtomicISub:",
        "            case SpvOpAtomicSMin: case SpvOpAtomicUMin:",
        "            case SpvOpAtomicSMax: case SpvOpAtomicUMax:",
        "            case SpvOpAtomicAnd: case SpvOpAtomicOr: case SpvOpAtomicXor:",
        "            case SpvOpAtomicFMinEXT: case SpvOpAtomicFMaxEXT: case SpvOpAtomicFAddEXT:",
        "                return true;",
        "            default: return false;",
        "        }",
        "    }",
        "    static bool parse(const uint32_t* words, size_t word_count, SpvAtomicRmwView* out) {",
        "        if (!words || !out || word_count < 7u) return false;",
        "        if (!matches_opcode(static_cast<uint16_t>(words[0] & 0xFFFFu))) return false;",
        "        out->result_type = words[1]; out->result_id = words[2];",
        "        out->pointer = words[3]; out->memory = words[4]; out->semantics = words[5];",
        "        out->value = words[6];",
        "        return true;",
        "    }",
        "    static SpvAtomicRmwView must_parse(const uint32_t* words, size_t word_count) {",
        "        SpvAtomicRmwView out{};",
        "        if (!parse(words, word_count, &out)) {",
        '            std::fprintf(stderr, "FATAL: SpvAtomicRmwView::must_parse failed (word_count=%zu)\\n", word_count);',
        "            std::abort();",
        "        }",
        "        return out;",
        "    }",
        "};",
        "",
        "/// Shared prefix view for image sample/fetch/gather/read instructions.",
        "struct SpvImageOpHeaderView {",
        "    static constexpr bool kIsSpvView = true;",
        "    uint32_t result_type = 0;",
        "    uint32_t result_id = 0;",
        "    uint32_t image = 0;",
        "    uint32_t coordinate = 0;",
        "",
        "    static constexpr bool matches_opcode(uint16_t op) {",
        "        switch (static_cast<SpvOp>(op)) {",
        "            case SpvOpImageSampleImplicitLod: case SpvOpImageSampleExplicitLod:",
        "            case SpvOpImageSampleDrefImplicitLod: case SpvOpImageSampleDrefExplicitLod:",
        "            case SpvOpImageSampleProjImplicitLod: case SpvOpImageSampleProjExplicitLod:",
        "            case SpvOpImageFetch: case SpvOpImageGather:",
        "            case SpvOpImageDrefGather: case SpvOpImageRead:",
        "                return true;",
        "            default: return false;",
        "        }",
        "    }",
        "    static bool parse(const uint32_t* words, size_t word_count, SpvImageOpHeaderView* out) {",
        "        if (!words || !out || word_count < 5u) return false;",
        "        if (!matches_opcode(static_cast<uint16_t>(words[0] & 0xFFFFu))) return false;",
        "        out->result_type = words[1]; out->result_id = words[2];",
        "        out->image = words[3]; out->coordinate = words[4];",
        "        return true;",
        "    }",
        "    static SpvImageOpHeaderView must_parse(const uint32_t* words, size_t word_count) {",
        "        SpvImageOpHeaderView out{};",
        "        if (!parse(words, word_count, &out)) {",
        "            uint16_t got_op = (words && word_count > 0) ? static_cast<uint16_t>(words[0] & 0xFFFFu) : 0u;",
        "            std::fprintf(stderr, \"FATAL: SpvImageOpHeaderView parse failed (got opcode=%u [%s], words=%zu)\\n\",",
        "                         static_cast<unsigned>(got_op), spv_op_name(got_op), word_count);",
        "            std::abort();",
        "        }",
        "        return out;",
        "    }",
        "};",
        "",
        "} // namespace valhall::spirv_parser",
        "",
    ])

    cpp_lines = [
        "// AUTO-GENERATED BY tools/gen_spirv_parser.py -- DO NOT EDIT MANUALLY",
        "// Derived from specs/spirv/spirv.core.grammar.json",
        '#include "valhall_spirv_parser_generated.h"',
        "#include <algorithm>",
        "",
        "namespace valhall::spirv_parser {",
        "",
        "namespace {",
        "",
        "constexpr SpvOpInfo kSpvOpTable[] = {",
    ]
    for info in sorted_ops:
        cpp_lines.append(
            f'    {{{info["opcode"]}u, "{info["opname"]}", '
            f'{"true" if info["has_rt"] else "false"}, '
            f'{"true" if info["has_rid"] else "false"}, '
            f'{info["min_words"]}u, '
            f'{"true" if info["is_term"] else "false"}}},'
        )
    cpp_lines.extend([
        "};",
        f"constexpr size_t kSpvOpTableCount = {len(sorted_ops)}u;",
        "",
        "} // namespace",
        "",
        "const SpvOpInfo* lookup_spv_op_info(uint16_t opcode) {",
        "    const SpvOpInfo* begin = kSpvOpTable;",
        "    const SpvOpInfo* end = kSpvOpTable + kSpvOpTableCount;",
        "    const SpvOpInfo* it = std::lower_bound(",
        "        begin, end, opcode,",
        "        [](const SpvOpInfo& info, uint16_t key) { return info.opcode < key; });",
        "    if (it != end && it->opcode == opcode) return it;",
        "    return nullptr;",
        "}",
        "",
        "const char* spv_op_name(uint16_t opcode) {",
        "    const SpvOpInfo* info = lookup_spv_op_info(opcode);",
        '    return info ? info->name : "OpUnknown";',
        "}",
        "",
        "bool decode_spv_instruction(const uint32_t* words, size_t remaining_words, DecodedSpvInst* out) {",
        "    if (!words || remaining_words == 0 || !out) return false;",
        "    uint16_t wc = static_cast<uint16_t>(words[0] >> 16);",
        "    uint16_t op = static_cast<uint16_t>(words[0] & 0xFFFFu);",
        "    if (wc == 0 || wc > remaining_words) return false;",
        "    out->opcode = static_cast<SpvOp>(op);",
        "    out->word_count = wc;",
        "    out->words = words;",
        "    out->result_type = 0;",
        "    out->result_id = 0;",
        "    uint16_t idx = 1;",
        "    if (const SpvOpInfo* info = lookup_spv_op_info(op)) {",
        "        if (info->has_result_type && idx < wc) out->result_type = words[idx++];",
        "        if (info->has_result_id && idx < wc) out->result_id = words[idx++];",
        "    }",
        "    out->operand_start = idx;",
        "    return true;",
        "}",
        "",
    ])

    def emit_name_switch(fn_name, arg_name, entries, default_str):
        lines = [f"const char* {fn_name}(uint32_t {arg_name}) {{", f"    switch ({arg_name}) {{"]
        for val, name in entries:
            lines.append(f'        case {val}u: return "{name}";')
        lines.append(f'        default: return "{default_str}";')
        lines.append("    }")
        lines.append("}")
        lines.append("")
        return lines

    cpp_lines.extend(emit_name_switch("spv_builtin_name", "builtin", builtins, "UnknownBuiltIn"))
    cpp_lines.extend(emit_name_switch("spv_execution_model_name", "exec_model", exec_models, "UnknownExecModel"))
    cpp_lines.extend(emit_name_switch("spv_storage_class_name", "storage_class", storage_classes, "UnknownStorageClass"))
    cpp_lines.extend(emit_name_switch("spv_decoration_name", "decoration", decorations, "UnknownDecoration"))
    cpp_lines.append("} // namespace valhall::spirv_parser")
    cpp_lines.append("")

    (out_dir / "valhall_spirv_parser_generated.h").write_text("\n".join(h_lines), encoding="utf-8")
    (out_dir / "valhall_spirv_parser_generated.cpp").write_text("\n".join(cpp_lines), encoding="utf-8")
    print(f"[OK] Generated valhall_spirv_parser_generated.{{h,cpp}} ({len(sorted_ops)} opcodes, {generated_views} typed views)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

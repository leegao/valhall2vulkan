#!/usr/bin/env python3
"""
Automated SPIR-V Builder & Grammar Code Generator.

Parses specs/spirv/spirv.core.grammar.json (Khronos SPIR-V Core Grammar) and
generates:
  - src/generated/valhall_spirv_builder_generated.h
  - src/generated/valhall_spirv_builder_generated.cpp

All 879 SPIR-V instructions are emitted as SpirvBuilder::op_<Name>(...) methods
with __attribute__((weak)) linkage (VALHALL_WEAK) and section-level dead-code
elimination (-ffunction-sections + -Wl,--gc-sections) so unused ops are stripped
from final linked binaries.
"""

import json
import os
import re
import sys
from pathlib import Path

CPP_KEYWORDS = {
    "alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor",
    "bool", "break", "case", "catch", "char", "char8_t", "char16_t", "char32_t",
    "class", "compl", "concept", "const", "consteval", "constexpr", "constinit",
    "const_cast", "continue", "co_await", "co_return", "co_yield", "decltype",
    "default", "delete", "do", "double", "dynamic_cast", "else", "enum",
    "explicit", "export", "extern", "false", "float", "for", "friend", "goto",
    "if", "inline", "int", "long", "mutable", "namespace", "new", "noexcept",
    "not", "not_eq", "nullptr", "operator", "or", "or_eq", "private",
    "protected", "public", "register", "reinterpret_cast", "requires", "return",
    "short", "signed", "sizeof", "static", "static_assert", "static_cast",
    "struct", "switch", "template", "this", "thread_local", "throw", "true",
    "try", "typedef", "typeid", "typename", "union", "unsigned", "using",
    "virtual", "void", "volatile", "wchar_t", "while", "xor", "xor_eq",
}


def sanitize_ident(name: str, prefix_digit: str = "V_") -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", name)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "arg"
    if s[0].isdigit():
        s = prefix_digit + s
    if s in CPP_KEYWORDS:
        s = s + "_val"
    return s


def sanitize_param_name(name: str, idx: int) -> str:
    if not name:
        return f"arg{idx}"
    s = name.lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s).strip("_")
    if not s:
        return f"arg{idx}"
    if s[0].isdigit():
        s = "p_" + s
    if s in CPP_KEYWORDS:
        s = s + "_val"
    return s


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    grammar_path = repo_root / "specs" / "spirv" / "spirv.core.grammar.json"
    out_dir = repo_root / "src" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not grammar_path.exists():
        print(f"Error: {grammar_path} not found", file=sys.stderr)
        return 1

    with open(grammar_path, "r", encoding="utf-8") as f:
        grammar = json.load(f)

    magic_number = grammar["magic_number"]
    major = grammar["major_version"]
    minor = grammar["minor_version"]
    revision = grammar["revision"]

    operand_kinds = grammar["operand_kinds"]
    instructions = grammar["instructions"]

    kind_map = {ok["kind"]: ok for ok in operand_kinds}
    value_enums = [ok for ok in operand_kinds if ok["category"] == "ValueEnum"]
    bit_enums = [ok for ok in operand_kinds if ok["category"] == "BitEnum"]

    # Determine which ValueEnums can carry extra parameter words in any enumerant
    value_enum_has_params = set()
    for ve in value_enums:
        for e in ve.get("enumerants", []):
            if e.get("parameters"):
                value_enum_has_params.add(ve["kind"])
                break

    # Capability dependency / extension tables (from the grammar's Capability kind)
    cap_kind = kind_map.get("Capability")
    cap_deps = {}
    cap_exts = {}
    if cap_kind:
        for e in cap_kind.get("enumerants", []):
            cap_deps[e["enumerant"]] = list(e.get("capabilities") or [])
            cap_exts[e["enumerant"]] = list(e.get("extensions") or [])

    # Capabilities that only make sense for Kernel/OpenCL modules; this builder
    # targets shader execution models, so declaring them would be invalid.
    KERNEL_ONLY_CAPS = {"Kernel", "Addresses"}

    def resolve_capabilities(names):
        """Pick the first shader-compatible capability and expand transitive deps."""
        chosen = next((n for n in names if n not in KERNEL_ONLY_CAPS), None)
        if chosen is None:
            return []
        out, seen, queue = [], set(), [chosen]
        while queue:
            c = queue.pop(0)
            if c in seen or c in KERNEL_ONLY_CAPS:
                continue
            seen.add(c)
            out.append(c)
            queue.extend(cap_deps.get(c, []))
        return out

    def resolve_extensions(inst):
        """Instruction extensions plus the extension(s) providing its capabilities."""
        exts = []
        for e in inst.get("extensions") or []:
            if e not in exts:
                exts.append(e)
        for c in resolve_capabilities(inst.get("capabilities") or []):
            ce = cap_exts.get(c, [])
            if not ce:
                continue
            # Prefer the ratified KHR spelling over deprecated EXT/NV variants.
            pick = next((x for x in ce if "KHR" in x), ce[0])
            if pick not in exts:
                exts.append(pick)
        return exts

    # =========================================================================
    # Generate valhall_spirv_builder_generated.h
    # =========================================================================
    h_lines = []
    h_lines.append("#pragma once")
    h_lines.append("// AUTO-GENERATED by tools/gen_spirv_builder.py from specs/spirv/spirv.core.grammar.json")
    h_lines.append(f"// SPIR-V Core Grammar Version {major}.{minor} Rev {revision}")
    h_lines.append("// DO NOT EDIT MANUALLY.")
    h_lines.append("")
    h_lines.append("#include <cstdint>")
    h_lines.append("#include <cstddef>")
    h_lines.append("#include <optional>")
    h_lines.append("#include <string>")
    h_lines.append("#include <utility>")
    h_lines.append("#include <vector>")
    h_lines.append("")
    h_lines.append("#if defined(__GNUC__) || defined(__clang__)")
    h_lines.append("#define VALHALL_WEAK __attribute__((weak))")
    h_lines.append("#else")
    h_lines.append("#define VALHALL_WEAK")
    h_lines.append("#endif")
    h_lines.append("")
    h_lines.append("namespace valhall::spirv {")
    h_lines.append("")
    h_lines.append(f"constexpr uint32_t kSpvMagicNumber = {magic_number}u;")
    h_lines.append(f"constexpr uint32_t kSpvMajorVersion = {major}u;")
    h_lines.append(f"constexpr uint32_t kSpvMinorVersion = {minor}u;")
    h_lines.append(f"constexpr uint32_t kSpvRevision = {revision}u;")
    h_lines.append("constexpr uint32_t kSpvVersion13   = 0x00010300u;")
    h_lines.append(f"constexpr uint32_t kSpvVersionLatest = ({major}u << 16) | ({minor}u << 8);")
    h_lines.append("constexpr uint32_t kSpvGeneratorId = 0x001B0001u;")
    h_lines.append("")

    # Emit ValueEnums
    for ve in value_enums:
        kind = ve["kind"]
        h_lines.append(f"enum class {kind} : uint32_t {{")
        seen_names = set()
        prefix = "Dim" if kind == "Dim" else "V_"
        for e in ve.get("enumerants", []):
            ename = sanitize_ident(e["enumerant"], prefix_digit=prefix)
            val = e["value"]
            if ename not in seen_names:
                h_lines.append(f"    {ename} = {val}u,")
                seen_names.add(ename)
            for alias in e.get("aliases", []):
                aname = sanitize_ident(alias, prefix_digit=prefix)
                if aname not in seen_names:
                    h_lines.append(f"    {aname} = {val}u,")
                    seen_names.add(aname)
        h_lines.append("};")
        h_lines.append("")

    # Emit BitEnums as Mask namespaces (preserving ImageOperandsMask / MemoryAccessMask naming)
    for be in bit_enums:
        kind = be["kind"]
        h_lines.append(f"namespace {kind}Mask {{")
        seen_names = set()
        for e in be.get("enumerants", []):
            ename = sanitize_ident(e["enumerant"], prefix_digit="Mask_")
            val = e["value"]
            if ename not in seen_names:
                h_lines.append(f"    constexpr uint32_t {ename} = {val}u;")
                seen_names.add(ename)
            for alias in e.get("aliases", []):
                aname = sanitize_ident(alias, prefix_digit="Mask_")
                if aname not in seen_names:
                    h_lines.append(f"    constexpr uint32_t {aname} = {val}u;")
                    seen_names.add(aname)
        h_lines.append(f"}} // namespace {kind}Mask")
        h_lines.append("")

    # Emit Op enum
    h_lines.append("enum class Op : uint16_t {")
    seen_ops = set()
    for inst in instructions:
        opname = inst["opname"]
        opcode = inst["opcode"]
        if opname not in seen_ops:
            h_lines.append(f"    {opname} = {opcode},")
            seen_ops.add(opname)
        for alias in inst.get("aliases", []):
            if alias not in seen_ops:
                h_lines.append(f"    {alias} = {opcode},")
                seen_ops.add(alias)
    h_lines.append("};")
    h_lines.append("")

    # Emit GLSL.std.450 enum (used by OpExtInst)
    h_lines.append("// GLSL.std.450 extended instruction opcodes")
    h_lines.append("enum class GlslStd450 : uint32_t {")
    glsl_ops = [
        ("Round", 1), ("RoundEven", 2), ("Trunc", 3), ("FAbs", 4), ("SAbs", 5),
        ("FSign", 6), ("SSign", 7), ("Floor", 8), ("Ceil", 9), ("Fract", 10),
        ("Radians", 11), ("Degrees", 12), ("Sin", 13), ("Cos", 14), ("Tan", 15),
        ("Asin", 16), ("Acos", 17), ("Atan", 18), ("Atan2", 25), ("Pow", 26),
        ("Exp", 27), ("Log", 28), ("Exp2", 29), ("Log2", 30), ("Sqrt", 31),
        ("InverseSqrt", 32), ("FMin", 37), ("UMin", 38), ("SMin", 39),
        ("FMax", 40), ("UMax", 41), ("SMax", 42), ("FClamp", 43), ("UClamp", 44),
        ("SClamp", 45), ("FMix", 46), ("Fma", 50), ("Frexp", 51), ("Ldexp", 53),
        ("PackSnorm4x8", 54), ("PackUnorm4x8", 55), ("PackSnorm2x16", 56),
        ("PackUnorm2x16", 57), ("PackHalf2x16", 58), ("UnpackSnorm2x16", 60),
        ("UnpackUnorm2x16", 61), ("UnpackHalf2x16", 62), ("UnpackSnorm4x8", 63),
        ("UnpackUnorm4x8", 64), ("FindILsb", 73), ("FindSMsb", 74), ("FindUMsb", 75),
        ("NMin", 79), ("NMax", 80), ("NClamp", 81),
    ]
    for gname, gval in glsl_ops:
        h_lines.append(f"    {gname} = {gval},")
    h_lines.append("};")
    h_lines.append("")

    # Build method declarations and definitions for all instructions
    method_decls = []
    method_defs = []

    for inst in instructions:
        opname = inst["opname"]
        cls = inst.get("class", "")
        operands = inst.get("operands", [])

        has_result_type = any(op["kind"] == "IdResultType" for op in operands)
        has_result = any(op["kind"] == "IdResult" for op in operands)
        ret_type = "uint32_t" if has_result else "void"

        # Method name: op_IAdd for OpIAdd, etc.
        short_name = opname[2:] if opname.startswith("Op") else opname
        method_name = f"op_{short_name}"

        params = []  # list of (decl_str, call_str, arg_kind, pname, quant)
        used_pnames = set()

        def unique_pname(raw_name: str, idx: int) -> str:
            base = sanitize_param_name(raw_name, idx)
            cand = base
            n = 2
            while cand in used_pnames:
                cand = f"{base}_{n}"
                n += 1
            used_pnames.add(cand)
            return cand

        # First pass: collect parameters
        raw_params = []
        for idx, op in enumerate(operands):
            okind = op["kind"]
            quant = op.get("quantifier", "")
            oname = op.get("name", okind)
            if okind == "IdResultType":
                pname = unique_pname("result_type", idx)
                raw_params.append(("uint32_t", pname, okind, quant, False))
            elif okind == "IdResult":
                continue
            elif okind in ("IdRef", "IdScope", "IdMemorySemantics"):
                pname = unique_pname(oname, idx)
                if quant == "*":
                    raw_params.append(("const std::vector<uint32_t>&", pname, okind, quant, False))
                elif quant == "?":
                    raw_params.append(("std::optional<uint32_t>", pname, okind, quant, True))
                else:
                    raw_params.append(("uint32_t", pname, okind, quant, False))
            elif okind == "LiteralInteger" or okind == "LiteralExtInstInteger" or okind == "LiteralFloat":
                pname = unique_pname(oname, idx)
                if quant == "*":
                    raw_params.append(("const std::vector<uint32_t>&", pname, okind, quant, False))
                elif quant == "?":
                    raw_params.append(("std::optional<uint32_t>", pname, okind, quant, True))
                else:
                    raw_params.append(("uint32_t", pname, okind, quant, False))
            elif okind == "LiteralString":
                pname = unique_pname(oname, idx)
                if quant == "?":
                    raw_params.append(("const std::optional<std::string>&", pname, okind, quant, True))
                else:
                    raw_params.append(("const std::string&", pname, okind, quant, False))
            elif okind == "LiteralContextDependentNumber":
                pname = unique_pname(oname, idx)
                raw_params.append(("const std::vector<uint32_t>&", pname, okind, quant, False))
            elif okind == "LiteralSpecConstantOpInteger":
                pname = unique_pname(oname, idx)
                raw_params.append(("uint32_t", pname, okind, quant, False))
                extra_name = unique_pname("spec_operands", idx)
                raw_params.append(("const std::vector<uint32_t>&", extra_name, "ExtraWords", "*", True))
            elif okind in ("PairLiteralIntegerIdRef", "PairIdRefLiteralInteger", "PairIdRefIdRef"):
                pname = unique_pname(oname, idx)
                raw_params.append(("const std::vector<std::pair<uint32_t, uint32_t>>&", pname, okind, quant, False))
            elif okind in kind_map and kind_map[okind]["category"] == "ValueEnum":
                pname = unique_pname(oname, idx)
                if quant == "*":
                    raw_params.append((f"const std::vector<{okind}>&", pname, okind, quant, False))
                elif quant == "?":
                    raw_params.append((f"std::optional<{okind}>", pname, okind, quant, True))
                else:
                    raw_params.append((okind, pname, okind, quant, False))
                    if okind in value_enum_has_params:
                        extra_name = unique_pname(oname + "_extra", idx)
                        raw_params.append(("const std::vector<uint32_t>&", extra_name, "ExtraWords", "*", True))
            elif okind in kind_map and kind_map[okind]["category"] == "BitEnum":
                pname = unique_pname(oname + "_mask", idx)
                if quant == "?":
                    raw_params.append(("std::optional<uint32_t>", pname, okind, quant, True))
                else:
                    raw_params.append(("uint32_t", pname, okind, quant, False))
                extra_name = unique_pname(oname + "_extra", idx)
                raw_params.append(("const std::vector<uint32_t>&", extra_name, "ExtraWords", "*", True))
            else:
                pname = unique_pname(oname, idx)
                raw_params.append(("const std::vector<uint32_t>&", pname, okind, quant, False))

        # Assign defaults only to contiguous trailing optional/variadic parameters
        can_default = [False] * len(raw_params)
        for i in range(len(raw_params) - 1, -1, -1):
            ptype, pname, okind, quant, is_opt = raw_params[i]
            if is_opt or quant == "*":
                can_default[i] = True
            else:
                break

        decl_param_strs = []
        def_param_strs = []
        for i, (ptype, pname, okind, quant, is_opt) in enumerate(raw_params):
            def_param_strs.append(f"{ptype} {pname}")
            if can_default[i]:
                if ptype.startswith("std::optional<"):
                    decl_param_strs.append(f"{ptype} {pname} = std::nullopt")
                elif ptype.startswith("const std::optional<"):
                    decl_param_strs.append(f"{ptype} {pname} = std::nullopt")
                elif ptype.startswith("const std::vector<"):
                    decl_param_strs.append(f"{ptype} {pname} = {{}}")
                else:
                    decl_param_strs.append(f"{ptype} {pname}")
            else:
                decl_param_strs.append(f"{ptype} {pname}")

        decl_sig = f"    {ret_type} {method_name}({', '.join(decl_param_strs)});"
        def_sig = f"VALHALL_WEAK {ret_type} SpirvBuilder::{method_name}({', '.join(def_param_strs)})"
        method_decls.append(decl_sig)

        # Build body
        body = []
        body.append(f"{def_sig} {{")

        # Special routing for Capability and Extension to keep deduplication
        if opname == "OpCapability":
            body.append(f"    add_capability({raw_params[0][1]});")
            body.append("}")
            method_defs.append("\n".join(body))
            continue
        if opname == "OpExtension":
            body.append(f"    add_extension({raw_params[0][1]});")
            body.append("}")
            method_defs.append("\n".join(body))
            continue

        # Grammar-driven capability / extension requirements
        for c in resolve_capabilities(inst.get("capabilities") or []):
            body.append(f"    add_capability(Capability::{c});")
        for e in resolve_extensions(inst):
            body.append(f'    add_extension("{e}");')

        # Validate every SSA ID operand (result type + IdRef/IdScope/IdMemorySemantics)
        for (ptype, pname, okind, quant, is_opt) in raw_params:
            if okind not in ("IdResultType", "IdRef", "IdScope", "IdMemorySemantics"):
                continue
            if quant == "*":
                body.append(f'    for (auto v : {pname}) validate_id(v, "{opname}", "{pname}");')
            elif quant == "?":
                body.append(f'    if ({pname}.has_value()) validate_id(*{pname}, "{opname}", "{pname}");')
            else:
                body.append(f'    validate_id({pname}, "{opname}", "{pname}");')

        if has_result:
            body.append("    uint32_t id = alloc_id();")

        body.append("    std::vector<uint32_t> args;")
        # Assemble args in exact grammar operand order
        # Track ExtraWords that belong to the immediately preceding operand
        rp_idx = 0
        for op in operands:
            okind = op["kind"]
            quant = op.get("quantifier", "")
            if okind == "IdResult":
                body.append("    args.push_back(id);")
                continue
            ptype, pname, _, _, _ = raw_params[rp_idx]
            rp_idx += 1

            if okind == "IdResultType":
                body.append(f"    args.push_back({pname});")
            elif okind in ("IdRef", "IdScope", "IdMemorySemantics", "LiteralInteger", "LiteralExtInstInteger", "LiteralFloat"):
                if quant == "*":
                    body.append(f"    args.insert(args.end(), {pname}.begin(), {pname}.end());")
                elif quant == "?":
                    body.append(f"    if ({pname}.has_value()) args.push_back(*{pname});")
                else:
                    body.append(f"    args.push_back({pname});")
            elif okind == "LiteralString":
                if quant == "?":
                    body.append(f"    if ({pname}.has_value()) {{ auto sw = encode_string(*{pname}); args.insert(args.end(), sw.begin(), sw.end()); }}")
                else:
                    body.append(f"    {{ auto sw = encode_string({pname}); args.insert(args.end(), sw.begin(), sw.end()); }}")
            elif okind == "LiteralContextDependentNumber":
                body.append(f"    args.insert(args.end(), {pname}.begin(), {pname}.end());")
            elif okind == "LiteralSpecConstantOpInteger":
                _, extra_pname, _, _, _ = raw_params[rp_idx]
                rp_idx += 1
                body.append(f"    args.push_back({pname});")
                body.append(f"    args.insert(args.end(), {extra_pname}.begin(), {extra_pname}.end());")
            elif okind in ("PairLiteralIntegerIdRef", "PairIdRefLiteralInteger", "PairIdRefIdRef"):
                body.append(f"    for (const auto& pr : {pname}) {{ args.push_back(pr.first); args.push_back(pr.second); }}")
            elif okind in kind_map and kind_map[okind]["category"] == "ValueEnum":
                if quant == "*":
                    body.append(f"    for (auto v : {pname}) args.push_back(static_cast<uint32_t>(v));")
                elif quant == "?":
                    body.append(f"    if ({pname}.has_value()) args.push_back(static_cast<uint32_t>(*{pname}));")
                else:
                    body.append(f"    args.push_back(static_cast<uint32_t>({pname}));")
                    if okind in value_enum_has_params:
                        _, extra_pname, _, _, _ = raw_params[rp_idx]
                        rp_idx += 1
                        body.append(f"    args.insert(args.end(), {extra_pname}.begin(), {extra_pname}.end());")
            elif okind in kind_map and kind_map[okind]["category"] == "BitEnum":
                _, extra_pname, _, _, _ = raw_params[rp_idx]
                rp_idx += 1
                if quant == "?":
                    body.append(f"    if ({pname}.has_value()) {{")
                    body.append(f"        args.push_back(*{pname});")
                    body.append(f"        args.insert(args.end(), {extra_pname}.begin(), {extra_pname}.end());")
                    body.append("    }")
                else:
                    body.append(f"    args.push_back({pname});")
                    body.append(f"    args.insert(args.end(), {extra_pname}.begin(), {extra_pname}.end());")
            else:
                body.append(f"    args.insert(args.end(), {pname}.begin(), {pname}.end());")

        # Choose target section buffer
        if opname in ("OpExecutionMode", "OpExecutionModeId"):
            target_buf = "execution_modes_words_"
        elif opname in ("OpName", "OpMemberName", "OpString", "OpSource", "OpSourceContinued", "OpSourceExtension", "OpModuleProcessed"):
            target_buf = "debug_names_words_"
        elif cls == "Annotation":
            target_buf = "decorations_words_"
        elif cls in ("Type-Declaration", "Constant-Creation"):
            target_buf = "types_consts_globals_words_"
        elif opname == "OpVariable":
            # Route Function-storage variables to local_vars_words_ and others to types_consts_globals_words_
            target_buf = None
        else:
            target_buf = "body_words_"

        if target_buf is not None:
            body.append(f"    append_op({target_buf}, Op::{opname}, args);")
        else:
            sc_pname = raw_params[1][1]
            body.append(f"    if ({sc_pname} == StorageClass::Function) {{")
            body.append(f"        append_op(local_vars_words_, Op::{opname}, args);")
            body.append("    } else {")
            body.append(f"        append_op(types_consts_globals_words_, Op::{opname}, args);")
            body.append("    }")

        if opname == "OpLabel":
            body.append("    in_entry_block_ = false;")
            body.append("    current_label_id_ = id;")

        if has_result:
            body.append("    return id;")
        body.append("}")
        method_defs.append("\n".join(body))

    h_lines.append("// Macro included inside class SpirvBuilder to declare all generated op_<OpName> methods")
    h_lines.append("#define VALHALL_SPIRV_BUILDER_GENERATED_OPS \\")
    for i, d in enumerate(method_decls):
        suffix = " \\" if i + 1 < len(method_decls) else ""
        h_lines.append(f"{d}{suffix}")
    h_lines.append("")
    h_lines.append("} // namespace valhall::spirv")
    h_lines.append("")

    gen_h_path = out_dir / "valhall_spirv_builder_generated.h"
    with open(gen_h_path, "w", encoding="utf-8") as f:
        f.write("\n".join(h_lines))

    # =========================================================================
    # Generate valhall_spirv_builder_generated.cpp
    # =========================================================================
    cpp_lines = []
    cpp_lines.append("// AUTO-GENERATED by tools/gen_spirv_builder.py from specs/spirv/spirv.core.grammar.json")
    cpp_lines.append(f"// SPIR-V Core Grammar Version {major}.{minor} Rev {revision}")
    cpp_lines.append("// DO NOT EDIT MANUALLY.")
    cpp_lines.append("//")
    cpp_lines.append("// Every op_<OpName> method is emitted with VALHALL_WEAK (__attribute__((weak)))")
    cpp_lines.append("// and compiled with -ffunction-sections so unused ops are stripped by -Wl,--gc-sections.")
    cpp_lines.append("")
    cpp_lines.append('#include "valhall_spirv_builder.h"')
    cpp_lines.append("")
    cpp_lines.append("namespace valhall::spirv {")
    cpp_lines.append("")
    for mdef in method_defs:
        cpp_lines.append(mdef)
        cpp_lines.append("")
    cpp_lines.append("} // namespace valhall::spirv")
    cpp_lines.append("")

    gen_cpp_path = out_dir / "valhall_spirv_builder_generated.cpp"
    with open(gen_cpp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(cpp_lines))

    print(
        f"Generated {gen_h_path.relative_to(repo_root)} and {gen_cpp_path.relative_to(repo_root)}: "
        f"{len(instructions)} instructions, {len(value_enums)} value enums, {len(bit_enums)} bit enums"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

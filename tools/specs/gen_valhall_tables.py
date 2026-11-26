#!/usr/bin/env python3
"""
Build-Time Spec Validator & Code Generator for Two-Pass Valhall-to-SPIR-V Compiler
==================================================================================
Inputs:
  - specs/kraid/isa-v9-v14.xml
  - specs/lowering/valhall_lowering.json (Pass 1: DirectSsaOp, @LOOKUP, INLINE_OPS, HANDLER)
  - specs/lowering/valhall_tiles.json    (Pass 2: Canonical & Fusion DP Tiles)

Outputs:
  - src/generated/valhall_tables_generated.h
  - src/generated/valhall_tables_generated.cpp
  - src/generated/valhall_pass1_lowering_generated.inc
"""

import argparse
import json
import re
import os
import sys
import xml.etree.ElementTree as ET


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from valhall_lowering_dsl import (
    parse_ssa_op_spec,
    gen_fetch_expr,
    gen_dest_vt_expr,
    gen_store_stmt,
    gen_lookup_key_expr,
    compile_inline_ops_rule,
    compile_when_expr,
    compile_operands_cpp,
)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="specs/kraid/isa-v9-v14.xml")
    ap.add_argument("--lowering", default="specs/lowering/valhall_lowering.json")
    ap.add_argument("--tiles", default="specs/lowering/valhall_tiles.json")
    ap.add_argument("--out-dir", default="src/generated")
    ap.add_argument("--check-operands", action="store_true", help="Verify valhall_operands_generated.cpp matches spec")
    return ap.parse_args()


def load_xml_instructions(xml_path):
    root = ET.parse(xml_path).getroot()
    insns = []
    for elem in root:
        if elem.tag != "instruction":
            continue
        since = int(elem.attrib.get("since", "9"))
        until = int(elem.attrib.get("until", "14"))
        arch_mask = 0
        for a in range(since, until + 1):
            arch_mask |= (1 << a)
        if arch_mask == 0:
            continue
        valid_fields = set()
        for ch in elem.iter():
            if ch.tag in ("modifier", "field") and "name" in ch.attrib:
                valid_fields.add(ch.attrib["name"])
        insns.append({
            "index": len(insns),
            "name": elem.attrib["name"],
            "variant": elem.attrib.get("variant", ""),
            "alias": elem.attrib.get("alias", ""),
            "alias_variant": elem.attrib.get("alias_variant", ""),
            "since": since,
            "until": until,
            "arch_mask": arch_mask,
            "valid_fields": valid_fields,
        })
    return insns




def validate_lowering_spec(insns, lowering_json, lowering_path):
    """Fail-loud structural validation of valhall_lowering.json against the XML ISA spec."""
    fields_by_mnem = {}
    for ins in insns:
        fields_by_mnem.setdefault(ins["name"], set()).update(ins.get("valid_fields", set()))

    allowed_rule_keys = {
        "mnemonic", "variants", "op_class", "ssa_op", "dest_type", "src_types",
        "splits_block", "is_barrier_or_cf", "effects", "arch_range", "when",
        "inline_ops", "handler", "comment", "_comment",
    }
    mod_ref_re = re.compile(r"\$modifiers\.([A-Za-z0-9_]+)")
    mod_eq_re = re.compile(r"@mod_eq\(\s*([A-Za-z0-9_]+)\s*,")
    lookup_tables = lowering_json.get("lookup_tables", {})
    errors = []

    for ridx, r in enumerate(lowering_json.get("rules", [])):
        mnem = r.get("mnemonic", "<missing>")
        unknown_keys = sorted(set(r.keys()) - allowed_rule_keys)
        if unknown_keys:
            errors.append(f"rule[{ridx}] {mnem}: unrecognized keys {unknown_keys}")
        valid_fields = fields_by_mnem.get(mnem, set())
        if not valid_fields:
            errors.append(f"rule[{ridx}] {mnem}: mnemonic not found in XML ISA spec")
            continue
        # Validate every $modifiers.<key> and @mod_eq(<key>, ...) reference
        scan_parts = [str(r.get("ssa_op", "")), str(r.get("when", ""))]
        scan_parts.extend(str(x) for x in r.get("inline_ops", []))
        blob = "\n".join(scan_parts)
        for key in sorted(set(mod_ref_re.findall(blob)) | set(mod_eq_re.findall(blob))):
            if key not in valid_fields:
                errors.append(
                    f"rule[{ridx}] {mnem}: references unknown modifier '{key}' "
                    f"(valid: {sorted(valid_fields)})"
                )
        # Validate @LOOKUP(table, ...) references an existing lookup_table
        for m in re.finditer(r"@LOOKUP\(\s*([A-Za-z0-9_]+)\s*,", blob):
            tbl = m.group(1)
            if tbl not in lookup_tables:
                errors.append(f"rule[{ridx}] {mnem}: references unknown lookup table '{tbl}'")

    if errors:
        print(f"[ERROR] Structural validation failed for {lowering_path} ({len(errors)} issue(s)):", file=sys.stderr)
        for e in errors[:30]:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)


def parse_arch_range(rule):
    ar = rule.get("arch_range", [9, 14])
    if not isinstance(ar, list) or len(ar) != 2:
        raise ValueError(f"Rule {rule.get('mnemonic')} has invalid arch_range: {ar!r}")
    lo, hi = int(ar[0]), int(ar[1])
    if lo < 9 or hi > 14 or lo > hi:
        raise ValueError(f"Rule {rule.get('mnemonic')} arch_range out of [9, 14]: {ar!r}")
    mask = 0
    for a in range(lo, hi + 1):
        mask |= (1 << a)
    return lo, hi, mask


def match_lowering_rules(ins, rules):
    name = ins["name"]
    var = ins["variant"]
    ins_mask = ins["arch_mask"]
    exact_matches = []
    wildcard_matches = []
    for idx, r in enumerate(rules):
        if r["mnemonic"] != name:
            continue
        _, _, r_mask = parse_arch_range(r)
        if (r_mask & ins_mask) == 0:
            continue
        vars_list = r.get("variants", ["*"])
        if var in vars_list:
            exact_matches.append((idx, r))
        elif "*" in vars_list:
            wildcard_matches.append((idx, r))
    return exact_matches + wildcard_matches


def parse_effects_mask(effects_list):
    cap_bits = 0
    ext_bits = 0
    exec_bits = 0
    flags = 0
    cap_map = {
        "Int64": 1 << 0,
        "Int16": 1 << 1,
        "Int8": 1 << 2,
        "Float16": 1 << 3,
        "Float64": 1 << 4,
        "PhysicalStorageBufferAddresses": 1 << 5,
        "GroupNonUniform": 1 << 6,
        "GroupNonUniformBallot": 1 << 7,
        "GroupNonUniformShuffle": 1 << 8,
        "Sampled1D": 1 << 9,
        "Image1D": 1 << 10,
    }
    ext_map = {
        "SPV_KHR_physical_storage_buffer": 1 << 0,
    }
    exec_map = {
        "DepthReplacing": 1 << 0,
    }
    for e in effects_list:
        if e == "uses_sideband":
            flags |= (1 << 0)
        elif e == "uses_glsl_std450":
            flags |= (1 << 1)
        elif e.startswith("cap:"):
            c = e[4:]
            if c not in cap_map:
                raise ValueError(f"Unknown capability effect '{c}'")
            cap_bits |= cap_map[c]
        elif e.startswith("ext:"):
            x = e[4:]
            if x not in ext_map:
                raise ValueError(f"Unknown extension effect '{x}'")
            ext_bits |= ext_map[x]
        elif e.startswith("exec:"):
            m = e[5:]
            if m not in exec_map:
                raise ValueError(f"Unknown exec_mode effect '{m}'")
            exec_bits |= exec_map[m]
        else:
            raise ValueError(f"Unknown effect string '{e}'")
    return cap_bits, ext_bits, exec_bits, flags


def flatten_pattern(pat, all_nodes, leaf_vars):
    if isinstance(pat, str):
        if pat not in leaf_vars:
            leaf_vars[pat] = len(leaf_vars)
        idx = len(all_nodes)
        all_nodes.append({
            "is_leaf": True,
            "leaf_slot": leaf_vars[pat],
            "ssa_op": "Nop",
            "has_imm": False,
            "imm_val": 0,
            "has_imm_offset": False,
            "imm_offset": 0,
            "has_imm_width": False,
            "imm_width": 0,
            "same_type_as_inner": False,
            "child_indices": [],
        })
        return idx

    idx = len(all_nodes)
    node = {
        "is_leaf": False,
        "leaf_slot": -1,
        "ssa_op": pat["op"],
        "has_imm": "imm" in pat,
        "imm_val": int(pat.get("imm", 0)),
        "has_imm_offset": "imm_offset" in pat,
        "imm_offset": int(pat.get("imm_offset", 0)),
        "has_imm_width": "imm_width" in pat,
        "imm_width": int(pat.get("imm_width", 0)),
        "same_type_as_inner": bool(pat.get("same_type_as_inner", False)),
        "child_indices": [],
    }
    all_nodes.append(node)
    for ch in pat.get("args", []):
        c_idx = flatten_pattern(ch, all_nodes, leaf_vars)
        node["child_indices"].append(c_idx)
    return idx


def main():
    args = parse_args()
    insns = load_xml_instructions(args.xml)
    with open(args.lowering, "r", encoding="utf-8") as f:
        lowering_json = json.load(f)

    if args.check_operands:
        ops_cpp_path = os.path.join(args.out_dir, "valhall_operands_generated.cpp")
        expected_cpp = compile_operands_cpp(lowering_json)
        if not os.path.isfile(ops_cpp_path):
            print(f"ERROR: {ops_cpp_path} does not exist", file=sys.stderr)
            sys.exit(1)
        with open(ops_cpp_path, "r", encoding="utf-8") as f:
            if f.read() != expected_cpp:
                print(f"ERROR: {ops_cpp_path} is out of date relative to {args.lowering}", file=sys.stderr)
                sys.exit(1)
        print(f"[gen_valhall_tables] {ops_cpp_path} is up to date ({len(lowering_json.get('operands', []))} operands)")
        return
    with open(args.tiles, "r", encoding="utf-8") as f:
        tiles_json = json.load(f)

    validate_lowering_spec(insns, lowering_json, args.lowering)
    rules = lowering_json["rules"]
    lookup_tables = lowering_json.get("lookup_tables", {})
    host_functions = lowering_json.get("host_functions", {})
    tiles = tiles_json["tiles"]

    canonical_ssa_ops = set()
    for t in tiles:
        if t["kind"] == "canonical":
            canonical_ssa_ops.add(t["ssa_op"])

    parsed_rules = []
    for r in rules:
        kind, val = parse_ssa_op_spec(r["ssa_op"])
        parsed_rules.append((kind, val))

    uncovered = []
    missing_canon_tiles = set()
    flat_candidates = []
    spec_ranges = []
    for ins in insns:
        cands = match_lowering_rules(ins, rules)
        if not cands:
            uncovered.append((ins["index"], ins["name"], ins["variant"]))
            spec_ranges.append((0, 0))
        else:
            off = len(flat_candidates)
            for ridx, _r in cands:
                flat_candidates.append(ridx)
                kind, val = parsed_rules[ridx]
                if kind == "DirectSsaOp" and val not in canonical_ssa_ops:
                    missing_canon_tiles.add(val)
            spec_ranges.append((off, len(cands)))

    if uncovered:
        print(f"[ERROR] {len(uncovered)} XML instructions lack lowering rules in {args.lowering}:", file=sys.stderr)
        for idx, nm, var in uncovered[:20]:
            print(f"  spec_index={idx}: {nm}.{var}", file=sys.stderr)
        sys.exit(1)

    if missing_canon_tiles:
        print(f"[ERROR] SSA ops missing canonical tiles in {args.tiles}: {sorted(missing_canon_tiles)}", file=sys.stderr)
        sys.exit(1)

    op_classes = sorted({r["op_class"] for r in rules})
    direct_ssa_ops = {val for kind, val in parsed_rules if kind == "DirectSsaOp"}
    ssa_ops = sorted(canonical_ssa_ops | direct_ssa_ops)
    emit_kinds = sorted({t["emit"] for t in tiles})

    def pascal(s):
        return "".join(p[0].upper() + p[1:] for p in s.replace("-", "_").split("_") if p)

    # Flatten fusion tile patterns
    fusion_tiles = [t for t in tiles if t["kind"] == "fusion"]
    fusion_nodes = []
    fusion_specs = []
    for ft in fusion_tiles:
        leaf_vars = {}
        start_idx = len(fusion_nodes)
        flatten_pattern(ft["pattern"], fusion_nodes, leaf_vars)
        node_count = len(fusion_nodes) - start_idx
        ssa_rw = ft.get("ssa_rewrite") or {}
        has_ssa_rw = bool(ssa_rw)
        rw_forward = int(ssa_rw.get("forward_leaf", -1))
        rw_op = ssa_rw.get("op", "Nop")
        rw_args = [int(x) for x in ssa_rw.get("args", [])]
        fusion_specs.append({
            "name": ft["name"],
            "cost": int(ft["cost"]),
            "emit": ft["emit"],
            "effects": ft.get("effects", []),
            "root_node": start_idx,
            "node_count": node_count,
            "num_leaves": len(leaf_vars),
            "requires_fast_math": bool(ft.get("fast_math", False)),
            "has_ssa_rewrite": has_ssa_rw,
            "ssa_rewrite_forward_leaf": rw_forward,
            "ssa_rewrite_op": rw_op,
            "ssa_rewrite_args": rw_args,
        })

    os.makedirs(args.out_dir, exist_ok=True)
    h_path = os.path.join(args.out_dir, "valhall_tables_generated.h")
    cpp_path = os.path.join(args.out_dir, "valhall_tables_generated.cpp")
    inc_path = os.path.join(args.out_dir, "valhall_pass1_lowering_generated.inc")

    h_lines = [
        "// AUTO-GENERATED BY tools/gen_valhall_tables.py -- DO NOT EDIT MANUALLY",
        "// Derived from specs/kraid/isa-v9-v14.xml, specs/lowering/valhall_lowering.json, specs/lowering/valhall_tiles.json",
        "#pragma once",
        "#include <cstdint>",
        "#include <cstddef>",
        "",
        "namespace valhall::tables {",
        "",
        "enum class OpClass : uint16_t {",
    ]
    for i, oc in enumerate(op_classes):
        h_lines.append(f"    {pascal(oc)} = {i},")
    h_lines.extend([
        "};",
        "",
        "enum class SsaOp : uint16_t {",
    ])
    for i, so in enumerate(ssa_ops):
        h_lines.append(f"    {so} = {i},")
    h_lines.extend([
        "};",
        "",
        "enum class LoweringKind : uint8_t {",
        "    DirectSsaOp = 0,",
        "    DynamicSsaOp = 1,",
        "    Handler = 2,",
        "    InlineOps = 3,",
        "};",
        "",
        "enum class TileEmitKind : uint16_t {",
    ])
    for i, ek in enumerate(emit_kinds):
        h_lines.append(f"    {ek} = {i},")
    h_lines.extend([
        "};",
        "",
        "struct EffectMask {",
        "    uint32_t cap_bits = 0;",
        "    uint32_t ext_bits = 0;",
        "    uint32_t exec_bits = 0;",
        "    uint32_t flags = 0;",
        "};",
        "",
        "struct LoweringRuleSpec {",
        "    const char* mnemonic;",
        "    uint16_t rule_index;",
        "    OpClass op_class;",
        "    LoweringKind kind;",
        "    SsaOp ssa_op;",
        "    const char* handler_name;",
        "    bool splits_block;",
        "    bool is_barrier_or_cf;",
        "    uint32_t arch_mask;",
        "    bool has_when;",
        "    EffectMask effects;",
        "};",
        "",
        "struct CanonicalTileSpec {",
        "    const char* name;",
        "    SsaOp ssa_op;",
        "    uint8_t cost;",
        "    TileEmitKind emit_kind;",
        "    EffectMask effects;",
        "};",
        "",
        "struct FusionPatternNode {",
        "    bool is_leaf;",
        "    int8_t leaf_slot;",
        "    SsaOp ssa_op;",
        "    bool has_imm;",
        "    uint32_t imm_val;",
        "    bool has_imm_offset;",
        "    uint32_t imm_offset;",
        "    bool has_imm_width;",
        "    uint32_t imm_width;",
        "    bool same_type_as_inner;",
        "    uint8_t num_children;",
        "    int16_t children[4];",
        "};",
        "",
        "struct FusionTileSpec {",
        "    const char* name;",
        "    uint8_t cost;",
        "    TileEmitKind emit_kind;",
        "    EffectMask effects;",
        "    uint16_t root_node;",
        "    uint16_t node_count;",
        "    uint8_t num_leaves;",
        "    bool requires_fast_math;",
        "    bool has_ssa_rewrite;",
        "    int8_t ssa_rewrite_forward_leaf;",
        "    SsaOp ssa_rewrite_op;",
        "    uint8_t ssa_rewrite_num_args;",
        "    uint8_t ssa_rewrite_args[3];",
        "};",
        "",
        f"constexpr size_t kNumXmlInstructions = {len(insns)};",
        f"constexpr size_t kNumLoweringRules = {len(rules)};",
        f"constexpr size_t kNumCanonicalTiles = {sum(1 for t in tiles if t['kind'] == 'canonical')};",
        f"constexpr size_t kNumFusionTiles = {len(fusion_specs)};",
        "",
        "const LoweringRuleSpec* lookup_lowering_rule(uint16_t spec_index);",
        "const LoweringRuleSpec* get_lowering_rule(uint16_t rule_index);",
        "const uint16_t* lookup_lowering_rule_candidates(uint16_t spec_index, size_t* out_count);",
        "const LoweringRuleSpec* lookup_lowering_rule_by_name(const char* mnemonic);",
        "const CanonicalTileSpec* lookup_canonical_tile(SsaOp op);",
        "const FusionTileSpec* get_fusion_tiles(size_t* out_count);",
        "const FusionPatternNode* get_fusion_pattern_nodes(size_t* out_count);",
        "const char* op_class_to_string(OpClass c);",
        "const char* ssa_op_to_string(SsaOp op);",
        "",
        "} // namespace valhall::tables",
        "",
    ])

    cpp_lines = [
        "// AUTO-GENERATED BY tools/gen_valhall_tables.py -- DO NOT EDIT MANUALLY",
        "#include \"valhall_tables_generated.h\"",
        "#include <cstring>",
        "",
        "namespace valhall::tables {",
        "",
        "static const LoweringRuleSpec kLoweringRules[] = {",
    ]
    for ridx, r in enumerate(rules):
        cb, eb, exb, fl = parse_effects_mask(r.get("effects", []))
        kind, val = parsed_rules[ridx]
        ssa_op_enum = val if kind == "DirectSsaOp" else "Nop"
        hname = r["ssa_op"]
        _, _, r_arch_mask = parse_arch_range(r)
        has_when = bool(r.get("when"))
        cpp_lines.append(
            f'    {{ "{r["mnemonic"]}", {ridx}u, OpClass::{pascal(r["op_class"])}, '
            f'LoweringKind::{kind}, SsaOp::{ssa_op_enum}, "{hname}", '
            f'{"true" if r.get("splits_block") else "false"}, '
            f'{"true" if r.get("is_barrier_or_cf") else "false"}, '
            f'0x{r_arch_mask:x}u, {"true" if has_when else "false"}, '
            f'{{ 0x{cb:x}u, 0x{eb:x}u, 0x{exb:x}u, 0x{fl:x}u }} }},'
        )
    cpp_lines.extend([
        "};",
        "",
        "struct SpecRuleRange { uint32_t offset; uint16_t count; };",
        "static const uint16_t kSpecRuleCandidates[] = {",
    ])
    for i in range(0, len(flat_candidates), 16):
        chunk = ", ".join(str(x) for x in flat_candidates[i:i+16])
        cpp_lines.append(f"    {chunk},")
    cpp_lines.extend([
        "};",
        "",
        "static const SpecRuleRange kSpecRuleRanges[] = {",
    ])
    for i in range(0, len(spec_ranges), 8):
        chunk = ", ".join(f"{{ {off}u, {cnt}u }}" for off, cnt in spec_ranges[i:i+8])
        cpp_lines.append(f"    {chunk},")
    cpp_lines.extend([
        "};",
        "",
        "static const CanonicalTileSpec kCanonicalTiles[] = {",
    ])
    canon_tiles = [t for t in tiles if t["kind"] == "canonical"]
    for t in canon_tiles:
        cb, eb, exb, fl = parse_effects_mask(t.get("effects", []))
        cpp_lines.append(
            f'    {{ "{t["name"]}", SsaOp::{t["ssa_op"]}, {int(t["cost"])}u, '
            f'TileEmitKind::{t["emit"]}, {{ 0x{cb:x}u, 0x{eb:x}u, 0x{exb:x}u, 0x{fl:x}u }} }},'
        )
    cpp_lines.extend([
        "};",
        "",
        "static const FusionPatternNode kFusionPatternNodes[] = {",
    ])
    for n in fusion_nodes:
        ch = list(n["child_indices"]) + [-1] * (4 - len(n["child_indices"]))
        cpp_lines.append(
            f'    {{ {"true" if n["is_leaf"] else "false"}, {n["leaf_slot"]}, SsaOp::{n["ssa_op"]}, '
            f'{"true" if n["has_imm"] else "false"}, {n["imm_val"]}u, '
            f'{"true" if n["has_imm_offset"] else "false"}, {n["imm_offset"]}u, '
            f'{"true" if n["has_imm_width"] else "false"}, {n["imm_width"]}u, '
            f'{"true" if n["same_type_as_inner"] else "false"}, {len(n["child_indices"])}u, '
            f'{{ {ch[0]}, {ch[1]}, {ch[2]}, {ch[3]} }} }},'
        )
    cpp_lines.extend([
        "};",
        "",
        "static const FusionTileSpec kFusionTiles[] = {",
    ])
    for fs in fusion_specs:
        cb, eb, exb, fl = parse_effects_mask(fs["effects"])
        rwa = list(fs["ssa_rewrite_args"]) + [0] * (3 - len(fs["ssa_rewrite_args"]))
        cpp_lines.append(
            f'    {{ "{fs["name"]}", {fs["cost"]}u, TileEmitKind::{fs["emit"]}, '
            f'{{ 0x{cb:x}u, 0x{eb:x}u, 0x{exb:x}u, 0x{fl:x}u }}, '
            f'{fs["root_node"]}u, {fs["node_count"]}u, {fs["num_leaves"]}u, '
            f'{"true" if fs["requires_fast_math"] else "false"}, '
            f'{"true" if fs["has_ssa_rewrite"] else "false"}, '
            f'{fs["ssa_rewrite_forward_leaf"]}, SsaOp::{fs["ssa_rewrite_op"]}, '
            f'{len(fs["ssa_rewrite_args"])}u, {{ {rwa[0]}u, {rwa[1]}u, {rwa[2]}u }} }},'
        )
    cpp_lines.extend([
        "};",
        "",
        "const LoweringRuleSpec* lookup_lowering_rule(uint16_t spec_index) {",
        f"    if (spec_index >= {len(insns)}) return nullptr;",
        "    const auto& rng = kSpecRuleRanges[spec_index];",
        "    if (rng.count == 0) return nullptr;",
        "    return &kLoweringRules[kSpecRuleCandidates[rng.offset]];",
        "}",
        "",
        "const LoweringRuleSpec* get_lowering_rule(uint16_t rule_index) {",
        f"    if (rule_index >= {len(rules)}) return nullptr;",
        "    return &kLoweringRules[rule_index];",
        "}",
        "",
        "const uint16_t* lookup_lowering_rule_candidates(uint16_t spec_index, size_t* out_count) {",
        f"    if (spec_index >= {len(insns)}) {{ if (out_count) *out_count = 0; return nullptr; }}",
        "    const auto& rng = kSpecRuleRanges[spec_index];",
        "    if (out_count) *out_count = rng.count;",
        "    return &kSpecRuleCandidates[rng.offset];",
        "}",
        "",
        "const LoweringRuleSpec* lookup_lowering_rule_by_name(const char* mnemonic) {",
        "    if (!mnemonic) return nullptr;",
        f"    for (size_t i = 0; i < {len(rules)}; ++i) {{",
        "        if (std::strcmp(kLoweringRules[i].mnemonic, mnemonic) == 0) return &kLoweringRules[i];",
        "    }",
        "    return nullptr;",
        "}",
        "",
        "const CanonicalTileSpec* lookup_canonical_tile(SsaOp op) {",
        f"    for (size_t i = 0; i < {len(canon_tiles)}; ++i) {{",
        "        if (kCanonicalTiles[i].ssa_op == op) return &kCanonicalTiles[i];",
        "    }",
        "    return nullptr;",
        "}",
        "",
        "const FusionTileSpec* get_fusion_tiles(size_t* out_count) {",
        f"    if (out_count) *out_count = {len(fusion_specs)};",
        "    return kFusionTiles;",
        "}",
        "",
        "const FusionPatternNode* get_fusion_pattern_nodes(size_t* out_count) {",
        f"    if (out_count) *out_count = {len(fusion_nodes)};",
        "    return kFusionPatternNodes;",
        "}",
        "",
        "const char* op_class_to_string(OpClass c) {",
        "    switch (c) {",
    ])
    for oc in op_classes:
        cpp_lines.append(f'        case OpClass::{pascal(oc)}: return "{oc}";')
    cpp_lines.extend([
        "    }",
        '    return "unknown";',
        "}",
        "",
        "const char* ssa_op_to_string(SsaOp op) {",
        "    switch (op) {",
    ])
    for so in ssa_ops:
        cpp_lines.append(f'        case SsaOp::{so}: return "{so}";')
    cpp_lines.extend([
        "    }",
        '    return "Unknown";',
        "}",
        "",
        "} // namespace valhall::tables",
        "",
    ])

    # Generate valhall_pass1_lowering_generated.inc
    inc_lines = [
        "// AUTO-GENERATED BY tools/gen_valhall_tables.py -- DO NOT EDIT MANUALLY",
        "// Included inside valhall_pass1.cpp",
        "inline std::string get_mod_str(const DecodedValhallInst& ins, const char* key) {",
        "    auto it = ins.modifier_strs.find(key);",
        '    return (it != ins.modifier_strs.end()) ? it->second : "";',
        "}",
        "inline SsaId make_f32_const_inline(Pass1Context& ctx, uint32_t blk, float c) {",
        "    uint32_t bits = 0; std::memcpy(&bits, &c, sizeof(float));",
        "    return ctx.sb.emit_const_f32_bits(blk, bits);",
        "}",
        "inline SsaId make_v2f_const_inline(Pass1Context& ctx, uint32_t blk, float c) {",
        "    SsaId s = make_f32_const_inline(ctx, blk, c);",
        "    return ctx.emit_op(blk, tables::SsaOp::VecConstruct2, ValType::V2F32, { s, s });",
        "}",
        "",
    ]

    # Emit lookup table helper functions
    for tname, tmap in lookup_tables.items():
        if tname == "_default":
            continue
        default_val = tmap.get("_default")
        tkind = tmap.get("_kind")
        entries = [(k, v) for k, v in tmap.items() if k not in ("_default", "_kind")]
        if not entries:
            raise ValueError(f"Lookup table '{tname}' is empty")
        sample_v = entries[0][1]
        is_u32 = (tkind == "u32") or (tkind is None and (isinstance(sample_v, int) or (isinstance(sample_v, str) and (sample_v.isdigit() or sample_v.startswith("0x")))))
        if tkind == "string":
            def_s = default_val if default_val is not None else entries[0][1]
            inc_lines.append(f"inline const char* lookup_{tname}(const std::string& key) {{")
            for k, v in entries:
                inc_lines.append(f'    if (key == "{k}") return "{v}";')
            inc_lines.append(f'    return "{def_s}";')
            inc_lines.append("}")
        elif is_u32:
            ret_t = "uint32_t"
            def_cpp = f"static_cast<uint32_t>({default_val if default_val is not None else entries[0][1]})"
            inc_lines.append(f"inline {ret_t} lookup_{tname}(const std::string& key) {{")
            for k, v in entries:
                inc_lines.append(f'    if (key == "{k}") return static_cast<uint32_t>({v});')
            inc_lines.append(f"    return {def_cpp};")
            inc_lines.append("}")
        else:
            # SsaOp table: validate every entry
            for k, v in entries:
                if v not in canonical_ssa_ops:
                    raise ValueError(f"Lookup table '{tname}' maps '{k}' -> unknown SsaOp '{v}'")
            def_op = default_val if default_val is not None else entries[0][1]
            if def_op not in canonical_ssa_ops:
                raise ValueError(f"Lookup table '{tname}' default '{def_op}' is not a valid SsaOp")
            inc_lines.append(f"inline tables::SsaOp lookup_{tname}(const std::string& key) {{")
            for k, v in entries:
                inc_lines.append(f'    if (key == "{k}") return tables::SsaOp::{v};')
            inc_lines.append(f"    return tables::SsaOp::{def_op};")
            inc_lines.append("}")

    when_cases = []
    for ridx, r in enumerate(rules):
        if r.get("when"):
            wexpr = compile_when_expr(r["when"], lookup_tables, canonical_ssa_ops, host_functions)
            when_cases.append(f"        case {ridx}u: return ({wexpr}); // {r['mnemonic']}")

    inc_lines.extend([
        "",
        "inline bool eval_rule_when_generated(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule) {",
        "    (void)ctx; (void)blk; (void)ins;",
        "    switch (rule->rule_index) {",
    ] + when_cases + [
        "        default: return true;",
        "    }",
        "}",
        "",
        "inline bool lower_rule_generated(Pass1Context& ctx, uint32_t blk, const DecodedValhallInst& ins, const tables::LoweringRuleSpec* rule) {",
        "    std::string var = variant_to_string(ins.canonical_variant);",
        '    bool is_v2f16 = (ins.canonical_variant == Variant::V2F16 || ins.variant == Variant::V2F16 || var == "v2f16");',
        '    bool is_signed = (var.find("s") == 0 || var.find("v2s") == 0 || var.find("v4s") == 0);',
        "    (void)var;",
        "    (void)is_v2f16;",
        "    (void)is_signed;",
        "    switch (rule->rule_index) {",
    ])

    num_direct = 0
    num_dynamic = 0
    num_inline = 0
    num_handler = 0
    for ridx, r in enumerate(rules):
        kind, val = parsed_rules[ridx]
        if kind == "Handler":
            num_handler += 1
            inc_lines.append(f"        case {ridx}u: {val}(ctx, blk, ins, rule); return true;")
            continue

        if kind == "InlineOps":
            num_inline += 1
            body = compile_inline_ops_rule(r, lookup_tables, canonical_ssa_ops, host_functions)
            inc_lines.append(f"        case {ridx}u: {{ {body} }} // {r['mnemonic']} INLINE_OPS")
            continue

        dest_t = r["dest_type"]
        src_ts = r.get("src_types", [])

        if kind == "DirectSsaOp" and val == "Nop":
            num_direct += 1
            inc_lines.append(f"        case {ridx}u: return true; // {r['mnemonic']} Nop")
            continue

        if kind == "DirectSsaOp" and val == "Copy":
            num_direct += 1
            if len(src_ts) != 1:
                raise ValueError(f"Copy rule {r['mnemonic']} must have 1 src_type, got {src_ts}")
            fetch_e = gen_fetch_expr(src_ts[0], 0)
            store_s = gen_store_stmt(dest_t, fetch_e)
            inc_lines.append(f"        case {ridx}u: {{ {store_s} return true; }} // {r['mnemonic']} Copy")
            continue

        vt_expr = gen_dest_vt_expr(dest_t)
        fetch_exprs = [gen_fetch_expr(st, i) for i, st in enumerate(src_ts)]
        args_init = ", ".join(fetch_exprs)

        if kind == "DynamicSsaOp":
            num_dynamic += 1
            subkind = val[0]
            if subkind == "LOOKUP":
                op_line = f"tables::SsaOp op = {gen_lookup_key_expr(val[2], lookup_tables, val[1])};"
            else:
                op_line = f"tables::SsaOp op = {val[1]}(ctx, blk, ins, vt, args, rule->effects);"
        else:
            num_direct += 1
            op_line = f"tables::SsaOp op = tables::SsaOp::{val};"

        store_s = gen_store_stmt(dest_t, "out")
        inc_lines.append(
            f"        case {ridx}u: {{ "
            f"ValType vt = {vt_expr}; "
            f"std::vector<SsaId> args = {{ {args_init} }}; "
            f"{op_line} "
            f"SsaId out = ctx.emit_op(blk, op, vt, std::move(args), rule->effects); "
            f"{store_s} return true; }}"
        )

    inc_lines.extend([
        "        default: return false;",
        "    }",
        "}",
        "",
    ])

    with open(h_path, "w", encoding="utf-8") as f:
        f.write("\n".join(h_lines))
    with open(cpp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(cpp_lines))
    with open(inc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(inc_lines))
    ops_cpp_path = os.path.join(args.out_dir, "valhall_operands_generated.cpp")
    if lowering_json.get("operands"):
        with open(ops_cpp_path, "w", encoding="utf-8") as f:
            f.write(compile_operands_cpp(lowering_json))

    print(f"[OK] Validated 100% coverage: {len(insns)} XML instructions -> {len(rules)} lowering rules "
          f"(direct={num_direct}, dynamic_lookup={num_dynamic}, inline_ops={num_inline}, handler={num_handler}), "
          f"{len(lookup_tables)} lookup tables, {len(canon_tiles)} canonical tiles + {len(fusion_specs)} fusion tiles.")


if __name__ == "__main__":
    main()

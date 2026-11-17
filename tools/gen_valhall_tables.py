#!/usr/bin/env python3
"""
Build-Time Spec Validator & Code Generator for Two-Pass Valhall-to-SPIR-V Compiler
==================================================================================
Inputs:
  - specs/kraid/isa-v9-v14.xml
  - specs/lowering/valhall_lowering.json (Pass 1)
  - specs/lowering/valhall_tiles.json    (Pass 2)

Outputs:
  - src/generated/valhall_tables_generated.h
  - src/generated/valhall_tables_generated.cpp
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="specs/kraid/isa-v9-v14.xml")
    ap.add_argument("--lowering", default="specs/lowering/valhall_lowering.json")
    ap.add_argument("--tiles", default="specs/lowering/valhall_tiles.json")
    ap.add_argument("--out-dir", default="src/generated")
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
        insns.append({
            "index": len(insns),
            "name": elem.attrib["name"],
            "variant": elem.attrib.get("variant", ""),
            "alias": elem.attrib.get("alias", ""),
            "alias_variant": elem.attrib.get("alias_variant", ""),
            "since": since,
            "until": until,
            "arch_mask": arch_mask,
        })
    return insns


def match_lowering_rule(ins, rules):
    name = ins["name"]
    var = ins["variant"]
    wildcard_match = None
    for idx, r in enumerate(rules):
        if r["mnemonic"] != name:
            continue
        vars_list = r.get("variants", ["*"])
        if var in vars_list:
            return idx, r
        if "*" in vars_list and wildcard_match is None:
            wildcard_match = (idx, r)
    return wildcard_match if wildcard_match is not None else (None, None)


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
    """Flatten a JSON pattern AST into FusionTileNode entries (pre-order). Returns root index."""
    if isinstance(pat, str):
        # Leaf capture variable like "$a"
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
    with open(args.tiles, "r", encoding="utf-8") as f:
        tiles_json = json.load(f)

    rules = lowering_json["rules"]
    tiles = tiles_json["tiles"]

    canonical_ssa_ops = set()
    for t in tiles:
        if t["kind"] == "canonical":
            canonical_ssa_ops.add(t["ssa_op"])

    uncovered = []
    missing_canon_tiles = set()
    spec_to_rule = []
    for ins in insns:
        ridx, r = match_lowering_rule(ins, rules)
        if r is None:
            uncovered.append((ins["index"], ins["name"], ins["variant"]))
            spec_to_rule.append(-1)
        else:
            spec_to_rule.append(ridx)
            if r["ssa_op"] not in canonical_ssa_ops:
                missing_canon_tiles.add(r["ssa_op"])

    if uncovered:
        print(f"[ERROR] {len(uncovered)} XML instructions lack lowering rules in {args.lowering}:", file=sys.stderr)
        for idx, nm, var in uncovered[:20]:
            print(f"  spec_index={idx}: {nm}.{var}", file=sys.stderr)
        sys.exit(1)

    if missing_canon_tiles:
        print(f"[ERROR] SSA ops missing canonical tiles in {args.tiles}: {sorted(missing_canon_tiles)}", file=sys.stderr)
        sys.exit(1)

    op_classes = sorted({r["op_class"] for r in rules})
    ssa_ops = sorted(canonical_ssa_ops | {r["ssa_op"] for r in rules})
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
        fusion_specs.append({
            "name": ft["name"],
            "cost": int(ft["cost"]),
            "emit": ft["emit"],
            "effects": ft.get("effects", []),
            "root_node": start_idx,
            "node_count": node_count,
            "num_leaves": len(leaf_vars),
        })

    os.makedirs(args.out_dir, exist_ok=True)
    h_path = os.path.join(args.out_dir, "valhall_tables_generated.h")
    cpp_path = os.path.join(args.out_dir, "valhall_tables_generated.cpp")

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
        "    OpClass op_class;",
        "    SsaOp ssa_op;",
        "    bool splits_block;",
        "    bool is_barrier_or_cf;",
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
        "};",
        "",
        f"constexpr size_t kNumXmlInstructions = {len(insns)};",
        f"constexpr size_t kNumLoweringRules = {len(rules)};",
        f"constexpr size_t kNumCanonicalTiles = {sum(1 for t in tiles if t['kind'] == 'canonical')};",
        f"constexpr size_t kNumFusionTiles = {len(fusion_specs)};",
        "",
        "const LoweringRuleSpec* lookup_lowering_rule(uint16_t spec_index);",
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
    for r in rules:
        cb, eb, exb, fl = parse_effects_mask(r.get("effects", []))
        cpp_lines.append(
            f'    {{ "{r["mnemonic"]}", OpClass::{pascal(r["op_class"])}, SsaOp::{r["ssa_op"]}, '
            f'{"true" if r.get("splits_block") else "false"}, '
            f'{"true" if r.get("is_barrier_or_cf") else "false"}, '
            f'{{ 0x{cb:x}u, 0x{eb:x}u, 0x{exb:x}u, 0x{fl:x}u }} }},'
        )
    cpp_lines.extend([
        "};",
        "",
        "static const uint16_t kSpecIndexToRuleIndex[] = {",
    ])
    for i in range(0, len(spec_to_rule), 16):
        chunk = ", ".join(str(x) for x in spec_to_rule[i:i+16])
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
        cpp_lines.append(
            f'    {{ "{fs["name"]}", {fs["cost"]}u, TileEmitKind::{fs["emit"]}, '
            f'{{ 0x{cb:x}u, 0x{eb:x}u, 0x{exb:x}u, 0x{fl:x}u }}, '
            f'{fs["root_node"]}u, {fs["node_count"]}u, {fs["num_leaves"]}u }},'
        )
    cpp_lines.extend([
        "};",
        "",
        "const LoweringRuleSpec* lookup_lowering_rule(uint16_t spec_index) {",
        f"    if (spec_index >= {len(insns)}) return nullptr;",
        "    return &kLoweringRules[kSpecIndexToRuleIndex[spec_index]];",
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

    with open(h_path, "w", encoding="utf-8") as f:
        f.write("\n".join(h_lines))
    with open(cpp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(cpp_lines))

    print(f"[OK] Validated 100% coverage: {len(insns)}/627 XML instructions -> {len(rules)} lowering rules, "
          f"{len(canon_tiles)} canonical tiles + {len(fusion_specs)} fusion tiles ({len(fusion_nodes)} pattern nodes).")


if __name__ == "__main__":
    main()

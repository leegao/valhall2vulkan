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
import os
import re
import sys
import xml.etree.ElementTree as ET


IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
HANDLER_RE = re.compile(r"^@HANDLER\(([A-Za-z_][A-Za-z0-9_]*)\)$")
LOOKUP_RE = re.compile(r"^@LOOKUP\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([^)]+?)\s*\)$")

VALTYPE_MAP = {
    "void": "ValType::Void",
    "bool": "ValType::Bool",
    "u32": "ValType::U32",
    "s32": "ValType::S32",
    "f32": "ValType::F32",
    "u64": "ValType::U64",
    "s64": "ValType::S64",
    "v2f32": "ValType::V2F32",
    "v2u32": "ValType::V2U32",
    "v2s32": "ValType::V2S32",
    "v4f32": "ValType::V4F32",
    "v4u32": "ValType::V4U32",
    "v4s32": "ValType::V4S32",
    "v3u32": "ValType::V3U32",
    "v2bool": "ValType::V2Bool",
    "tuple_u32x2": "ValType::TupleU32x2",
    "tuple_val_token": "ValType::TupleValToken",
    "token": "ValType::Token",
    "fp_vt": "(is_v2f16 ? ValType::V2F32 : ValType::F32)",
    "v2i_vt": "(is_signed ? ValType::V2S32 : ValType::V2U32)",
}


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


def parse_ssa_op_spec(ssa_op_str):
    if ssa_op_str == "@INLINE_OPS":
        return ("InlineOps", "INLINE_OPS")
    m = HANDLER_RE.match(ssa_op_str)
    if m:
        return ("Handler", m.group(1))
    lm = LOOKUP_RE.match(ssa_op_str)
    if lm:
        return ("DynamicSsaOp", ("LOOKUP", lm.group(1), lm.group(2)))
    if ssa_op_str.startswith("@"):
        fn = ssa_op_str[1:]
        if not IDENT_RE.match(fn):
            raise ValueError(f"Invalid dynamic ssa_op selector '{ssa_op_str}'")
        return ("DynamicSsaOp", ("FN", fn))
    if not IDENT_RE.match(ssa_op_str):
        raise ValueError(f"Invalid ssa_op '{ssa_op_str}'")
    return ("DirectSsaOp", ssa_op_str)


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


def gen_fetch_expr(src_type, idx):
    if src_type == "f32":
        return f"fetch_src_f32(ctx.sb, blk, ins, {idx})"
    if src_type == "f32_or_v2f16":
        return f"(is_v2f16 ? fetch_src_v2f16(ctx.sb, blk, ins, {idx}) : fetch_src_f32(ctx.sb, blk, ins, {idx}))"
    if src_type == "u32":
        return f"fetch_src_int32(ctx.sb, blk, ins, {idx}, false)"
    if src_type == "s32_raw":
        return f"fetch_src_int32(ctx.sb, blk, ins, {idx}, true)"
    if src_type == "s32":
        return f"ctx.sb.emit_bitcast(blk, ValType::S32, fetch_src_int32(ctx.sb, blk, ins, {idx}, true))"
    if src_type == "i32_by_var":
        return f"fetch_src_int32(ctx.sb, blk, ins, {idx}, (var.find(\"s\") == 0))"
    if src_type == "u64":
        return f"fetch_src_u64(ctx.sb, blk, ins, {idx}, false)"
    if src_type == "s64":
        return f"fetch_src_u64(ctx.sb, blk, ins, {idx}, true)"
    if src_type == "i64_by_var":
        return f"fetch_src_u64(ctx.sb, blk, ins, {idx}, (var.find(\"s64\") != std::string::npos))"
    if src_type == "v2f16":
        return f"fetch_src_v2f16(ctx.sb, blk, ins, {idx})"
    if src_type == "v2s16":
        return f"fetch_src_v2i16(ctx.sb, blk, ins, {idx}, true)"
    if src_type == "v2u16":
        return f"fetch_src_v2i16(ctx.sb, blk, ins, {idx}, false)"
    if src_type == "v2i16_by_var":
        return f"fetch_src_v2i16(ctx.sb, blk, ins, {idx}, is_signed)"
    if src_type == "v4i8_by_var":
        return f"fetch_src_v4i8(ctx.sb, blk, ins, {idx}, is_signed)"
    if src_type == "v4s8":
        return f"fetch_src_v4i8(ctx.sb, blk, ins, {idx}, true)"
    if src_type == "v4u8":
        return f"fetch_src_v4i8(ctx.sb, blk, ins, {idx}, false)"
    raise ValueError(f"Unsupported src_type '{src_type}'")


def gen_dest_vt_expr(dest_type):
    if dest_type == "f32":
        return "ValType::F32"
    if dest_type == "f32_or_v2f16":
        return "(is_v2f16 ? ValType::V2F32 : ValType::F32)"
    if dest_type == "u32":
        return "ValType::U32"
    if dest_type == "s32":
        return "ValType::S32"
    if dest_type == "u64":
        return "ValType::U64"
    if dest_type == "v2f16":
        return "ValType::V2F32"
    if dest_type == "v2s16":
        return "ValType::V2S32"
    if dest_type == "v2u16":
        return "ValType::V2U32"
    if dest_type == "v2i16_by_var":
        return "(is_signed ? ValType::V2S32 : ValType::V2U32)"
    if dest_type == "none":
        return "ValType::Void"
    raise ValueError(f"Unsupported dest_type '{dest_type}'")


def gen_store_stmt(dest_type, val_expr):
    if dest_type == "f32":
        return f"store_dest_f32(ctx.sb, blk, ins, {val_expr});"
    if dest_type == "f32_or_v2f16":
        return f"if (is_v2f16) store_dest_v2f16(ctx.sb, blk, ins, {val_expr}); else store_dest_f32(ctx.sb, blk, ins, {val_expr});"
    if dest_type == "u32":
        return f"store_dest_u32(ctx.sb, blk, ins, {val_expr});"
    if dest_type == "s32":
        return f"store_dest_u32(ctx.sb, blk, ins, ctx.sb.emit_bitcast(blk, ValType::U32, {val_expr}));"
    if dest_type == "u64":
        return f"store_dest_u64(ctx.sb, blk, ins, {val_expr});"
    if dest_type == "v2f16":
        return f"store_dest_v2f16(ctx.sb, blk, ins, {val_expr});"
    if dest_type == "v2s16":
        return f"store_dest_v2i16(ctx.sb, blk, ins, {val_expr}, true);"
    if dest_type == "v2u16":
        return f"store_dest_v2i16(ctx.sb, blk, ins, {val_expr}, false);"
    if dest_type == "v2i16_by_var":
        return f"store_dest_v2i16(ctx.sb, blk, ins, {val_expr}, is_signed);"
    if dest_type == "none":
        return ""
    raise ValueError(f"Unsupported dest_type '{dest_type}' for store")


def gen_lookup_key_expr(key_expr, lookup_tables, table_name):
    if table_name not in lookup_tables:
        raise ValueError(f"Unknown lookup table '{table_name}'")
    if key_expr.startswith("$modifiers."):
        mod_name = key_expr[len("$modifiers."):]
        return f'lookup_{table_name}(get_mod_str(ins, "{mod_name}"))'
    raise ValueError(f"Unsupported @LOOKUP key expression '{key_expr}'")


# ============================================================================
# Milestone 2: inline_ops DSL Compiler (Python -> C++17)
# ============================================================================

def split_top_level_args(s):
    args = []
    depth = 0
    cur = []
    for ch in s:
        if ch in ("(", "["):
            depth += 1
            cur.append(ch)
        elif ch in (")", "]"):
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            tok = "".join(cur).strip()
            if tok:
                args.append(tok)
            cur = []
        else:
            cur.append(ch)
    tok = "".join(cur).strip()
    if tok:
        args.append(tok)
    return args


def compile_cond(cond_str):
    c = cond_str.strip()
    neg = False
    if c.startswith("!"):
        neg = True
        c = c[1:].strip()
    if c == "$has_src0":
        expr = "(!ins.srcs.empty())"
    elif c == "$has_src1":
        expr = "(ins.srcs.size() >= 2)"
    elif c == "$has_src2":
        expr = "(ins.srcs.size() >= 3)"
    elif c == "$has_src3":
        expr = "(ins.srcs.size() >= 4)"
    elif c == "$has_dest":
        expr = "ins.has_dest"
    elif c == "$is_v2f16":
        expr = "is_v2f16"
    elif c == "$is_signed":
        expr = "is_signed"
    elif c == "$is_64bit":
        expr = "ins.dest_is_64bit"
    elif c == "$src0_swz_h1":
        expr = '(get_src_swz(ins, 0) == "h1")'
    elif c == "$is_frag_stage":
        expr = "(ctx.opts.stage == ShaderStage::Fragment)"
    elif c == "$dest_is_64bit":
        expr = "ins.dest_is_64bit"
    elif c.startswith("$has_field."):
        fn = c[len("$has_field."):]
        expr = f'(ins.fields.count("{fn}") > 0)'
    elif c.startswith("$mod_eq(") and c.endswith(")"):
        parts = split_top_level_args(c[len("$mod_eq("):-1])
        expr = f'(get_mod_str(ins, "{parts[0]}") == "{parts[1]}")'
    elif c.startswith("$mod_ne(") and c.endswith(")"):
        parts = split_top_level_args(c[len("$mod_ne("):-1])
        expr = f'(get_mod_str(ins, "{parts[0]}") != "{parts[1]}")'
    elif c == "$dest_is_64bit":
        expr = "ins.dest_is_64bit"
    elif c.startswith("$has_field."):
        fn = c[len("$has_field."):]
        expr = f'(ins.fields.count("{fn}") > 0)'
    elif c.startswith("$mod_active(") and c.endswith(")"):
        k = c[len("$mod_active("):-1].strip()
        expr = f'(!get_mod_str(ins, "{k}").empty() && get_mod_str(ins, "{k}") != "none")'
    elif c.startswith("$mod_eq(") and c.endswith(")"):
        parts = split_top_level_args(c[len("$mod_eq("):-1])
        expr = f'(get_mod_str(ins, "{parts[0]}") == "{parts[1]}")'
    elif c.startswith("$mod_ne(") and c.endswith(")"):
        parts = split_top_level_args(c[len("$mod_ne("):-1])
        expr = f'(get_mod_str(ins, "{parts[0]}") != "{parts[1]}")'
    elif c == "$has_imm_nonzero":
        expr = "(ins.has_imm && ins.imm_signed != 0)"
    elif c == "$sr_write_present":
        expr = "ins.sr_write.present"
    elif c == "$is_vert_stage":
        expr = "(ctx.opts.stage == ShaderStage::Vertex)"
    elif c == "$is_64_atom":
        expr = '(var.find("64") != std::string::npos || ins.dest_is_64bit || (ins.sr_write.present && ins.sr_write.count >= 2) || (ins.sr_read.present && ins.sr_read.count >= 2 && ins.canonical_mnemonic != Mnemonic::ACMPXCHG))'
    elif c == "$is_gather":
        expr = "(ins.canonical_mnemonic == Mnemonic::TEX_GATHER || ins.mnemonic == Mnemonic::TEX_GATHER)"
    elif c == "$is_flat_var":
        expr = "(ins.canonical_mnemonic == Mnemonic::LD_VAR_FLAT_IMM || ins.canonical_mnemonic == Mnemonic::LD_VAR_FLAT)"
    elif c.startswith("$is_zero(") and c.endswith(")"):
        vn = c[len("$is_zero("):-1].strip().lstrip("%")
        expr = f"(v_{vn} == 0u)"
    elif c.startswith("$lt_sr_read(") and c.endswith(")"):
        vn = c[len("$lt_sr_read("):-1].strip().lstrip("%")
        expr = f"(v_{vn} < (ins.sr_read.present ? std::max<uint32_t>(1u, ins.sr_read.count) : 1u))"
    elif c.startswith("$lt_u32(") and c.endswith(")"):
        parts = split_top_level_args(c[len("$lt_u32("):-1])
        a0 = parts[0].strip()
        if a0.startswith("#lookup_u32_raw(") and a0.endswith(")"):
            lp_args = split_top_level_args(a0[len("#lookup_u32_raw("):-1])
            a0_cpp = f'lookup_{lp_args[0]}(get_mod_str(ins, "{lp_args[1][len("$modifiers."):]}"))'
        elif a0.startswith("%"):
            a0_cpp = f"v_{a0[1:]}"
        else:
            a0_cpp = a0
        expr = f"({a0_cpp} < {parts[1].strip()})"
    elif c.startswith("$fields."):
        fn = c[len("$fields."):]
        expr = f'(ins.fields.count("{fn}") && ins.fields.at("{fn}") != 0)'
    else:
        raise ValueError(f"Unknown inline_ops condition '{cond_str}'")
    return f"!({expr})" if neg else expr


def compile_atom(atom, src_types, declared_vars, lookup_tables):
    a = atom.strip()
    if a.startswith("%") and a[1:].isdigit():
        idx = int(a[1:])
        if idx >= len(src_types):
            raise ValueError(f"inline_ops references %{idx}, but src_types only has {len(src_types)} entries")
        return gen_fetch_expr(src_types[idx], idx)
    if a.startswith("%"):
        vname = a[1:]
        if vname not in declared_vars:
            raise ValueError(f"inline_ops references undeclared variable '{a}'")
        return f"v_{vname}"
    if a == "#raw0":
        return "fetch_raw_src_u32(ctx.sb, blk, ins, 0)"
    if a == "#imm_u32":
        return "ctx.sb.emit_const_u32(blk, static_cast<uint32_t>(ins.imm_unsigned))"
    if a == "#imm_s32":
        return "ctx.sb.emit_const_u32(blk, static_cast<uint32_t>(ins.imm_signed))"
    if a == "#imm_f32":
        return "ctx.sb.emit_const_f32_bits(blk, static_cast<uint32_t>(ins.imm_unsigned))"
    if a == "#imm_u64":
        return "ctx.sb.emit_const_u64(blk, static_cast<uint64_t>(ins.imm_signed))"
    if a == "#imm_u64_mask63":
        return "ctx.sb.emit_const_u64(blk, static_cast<uint64_t>(ins.imm_unsigned & 63u))"
    if a == "#token":
        return "ctx.sb.current_token(blk)"
    if a.startswith("#u64_mul4(") and a.endswith(")"):
        vn = a[len("#u64_mul4("):-1].strip().lstrip("%")
        return f"ctx.sb.emit_const_u64(blk, uint64_t(v_{vn}) * 4ULL)"
    if a.startswith("#f32_alpha_or_zero(") and a.endswith(")"):
        vn = a[len("#f32_alpha_or_zero("):-1].strip().lstrip("%")
        return f"ctx.sb.emit_const_f32_bits(blk, (v_{vn} == 3u) ? 0x3f800000u : 0u)"
    if a == "#branch_target_pc_u64":
        return "ctx.sb.emit_const_u64(blk, static_cast<uint64_t>(ins.has_branch_offset ? ins.branch_target_pc : (int64_t(ins.pc) + 8 + ins.imm_signed)))"
    if a == "#const_bool(true)":
        return "ctx.sb.emit_const_bool(blk, true)"
    if a == "#const_bool(false)":
        return "ctx.sb.emit_const_bool(blk, false)"
    if a.startswith("#field_u32(") and a.endswith(")"):
        fn = a[len("#field_u32("):-1].strip()
        return f'ctx.sb.emit_const_u32(blk, ins.fields.at("{fn}"))'
    if a.startswith("#field_u32_mul16(") and a.endswith(")"):
        fn = a[len("#field_u32_mul16("):-1].strip()
        return f'ctx.sb.emit_const_u32(blk, ins.fields.at("{fn}") * 16u)'
    if a == "#fp_zero":
        return "(is_v2f16 ? make_v2f_const_inline(ctx, blk, 0.0f) : ctx.sb.emit_const_f32_bits(blk, 0u))"
    if a.startswith("#u32(") and a.endswith(")"):
        val = a[5:-1].strip()
        return f"ctx.sb.emit_const_u32(blk, static_cast<uint32_t>({val}))"
    if a.startswith("#s32(") and a.endswith(")"):
        val = a[5:-1].strip()
        return f"ctx.sb.emit_const_s32(blk, static_cast<int32_t>({val}))"
    if a.startswith("#u64(") and a.endswith(")"):
        val = a[5:-1].strip()
        return f"ctx.sb.emit_const_u64(blk, static_cast<uint64_t>({val}))"
    if a.startswith("#f32(") and a.endswith(")"):
        val = a[5:-1].strip()
        if val.startswith("0x") or val.startswith("0X"):
            return f"ctx.sb.emit_const_f32_bits(blk, static_cast<uint32_t>({val}))"
        return f"make_f32_const_inline(ctx, blk, static_cast<float>({val}))"
    if a.startswith("#lookup_u32(") and a.endswith(")"):
        inner = a[len("#lookup_u32("):-1]
        parts = split_top_level_args(inner)
        if len(parts) != 2:
            raise ValueError(f"Invalid #lookup_u32 syntax: '{a}'")
        call_expr = gen_lookup_key_expr(parts[1], lookup_tables, parts[0])
        return f"ctx.sb.emit_const_u32(blk, {call_expr})"
    raise ValueError(f"Unknown inline_ops atom '{a}'")



def compile_uint32_arg(tok, declared_vars, lookup_tables):
    t = tok.strip()
    if t.startswith("%"):
        vn = t[1:]
        if vn not in declared_vars:
            raise ValueError(f"Undeclared uint32_t var '{t}'")
        return f"v_{vn}"
    if t == "$src0_swz_h1":
        return '(get_src_swz(ins, 0) == "h1" ? 1u : 0u)'
    if t.startswith("#lookup_u32_raw(") and t.endswith(")"):
        parts = split_top_level_args(t[len("#lookup_u32_raw("):-1])
        return gen_lookup_key_expr(parts[1], lookup_tables, parts[0])
    return f"static_cast<uint32_t>({t})"

def compile_bool_arg(tok):
    t = tok.strip()
    if t in ("true", "false"):
        return t
    return compile_cond(t)

def compile_loop_bound(bound_str):
    b = bound_str.strip()
    if b == "$sr_write_count":
        return "(ins.sr_write.present ? std::max<uint32_t>(1u, ins.sr_write.count) : 1u)"
    if b == "$sr_write_count_max4":
        return "(ins.sr_write.present ? std::min<uint32_t>(4u, std::max<uint32_t>(1u, ins.sr_write.count)) : 4u)"
    if b == "$sr_read_count":
        return "(ins.sr_read.present ? std::max<uint32_t>(1u, ins.sr_read.count) : 1u)"
    return f"static_cast<uint32_t>({b})"

def compile_inline_rhs(rhs, lhs_vt, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions):
    r = rhs.strip()
    # Trailing flags: [barrier], [side_effect], [aux_bool=...]
    barrier = "false"
    side_eff = "false"
    aux_bool = None
    while r.endswith("]"):
        lb = r.rfind("[")
        if lb < 0:
            break
        flag = r[lb+1:-1].strip()
        r = r[:lb].strip()
        if flag == "barrier":
            barrier = "true"
        elif flag == "side_effect":
            side_eff = "true"
        elif flag.startswith("aux_bool="):
            aux_bool = flag[len("aux_bool="):].strip()
        else:
            raise ValueError(f"Unknown inline_ops node flag '[{flag}]'")

    # Simple variable or atom alias
    if r.startswith("%") or r.startswith("#"):
        return compile_atom(r, src_types, declared_vars, lookup_tables)

    # Function-like call: op(args...)
    lp = r.find("(")
    if lp < 0 or not r.endswith(")"):
        raise ValueError(f"Invalid inline_ops RHS '{rhs}'")

    # Handle @LOOKUP(table, key)(args...)
    if r.startswith("@LOOKUP("):
        # Find matching paren for @LOOKUP(...)
        depth = 0
        close_idx = -1
        for idx, ch in enumerate(r):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    close_idx = idx
                    break
        if close_idx < 0 or close_idx + 1 >= len(r) or r[close_idx + 1] != "(":
            raise ValueError(f"Invalid @LOOKUP call in inline_ops: '{rhs}'")
        lookup_part = r[:close_idx + 1]
        args_part = r[close_idx + 2:-1]
        lm = LOOKUP_RE.match(lookup_part)
        if not lm:
            raise ValueError(f"Invalid @LOOKUP syntax: '{lookup_part}'")
        op_expr = gen_lookup_key_expr(lm.group(2), lookup_tables, lm.group(1))
        arg_exprs = [compile_atom(a, src_types, declared_vars, lookup_tables) for a in split_top_level_args(args_part)]
        return f"ctx.emit_op(blk, {op_expr}, {lhs_vt}, {{ {', '.join(arg_exprs)} }}, rule->effects, {barrier}, {side_eff})"

    fn_name = r[:lp].strip()
    args_str = r[lp+1:-1].strip()
    arg_tokens = split_top_level_args(args_str)

    if fn_name.startswith("@"):
        hname = fn_name[1:]
        if hname not in host_functions:
            raise ValueError(f"Unknown host @FN '{fn_name}' (not declared in host_functions)")
        sig = host_functions[hname]
        expected_args = sig["args"]
        cpp_args = ["ctx", "blk", "ins", "rule"]
        tok_idx = 0
        for atype in expected_args:
            if atype == "ValType" and tok_idx < len(arg_tokens) and arg_tokens[tok_idx].strip() in VALTYPE_MAP:
                cpp_args.append(VALTYPE_MAP[arg_tokens[tok_idx].strip()])
                tok_idx += 1
            elif atype == "ValType":
                cpp_args.append(lhs_vt)
            else:
                if tok_idx >= len(arg_tokens):
                    raise ValueError(f"@{hname} expected {expected_args}, got {arg_tokens}")
                tok = arg_tokens[tok_idx]
                tok_idx += 1
                if atype == "SsaId":
                    if tok.strip().startswith("@"):
                        cpp_args.append(compile_inline_rhs(tok, lhs_vt, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions))
                    else:
                        cpp_args.append(compile_atom(tok, src_types, declared_vars, lookup_tables))
                elif atype == "uint32_t":
                    cpp_args.append(compile_uint32_arg(tok, declared_vars, lookup_tables))
                elif atype == "bool":
                    cpp_args.append(compile_bool_arg(tok))
                else:
                    raise ValueError(f"Unsupported host_functions arg type '{atype}'")
        if tok_idx != len(arg_tokens):
            raise ValueError(f"@{hname} received extra args: {arg_tokens[tok_idx:]}")
        return f"abi::{hname}({', '.join(cpp_args)})"


    if fn_name == "load_sideband_u64":
        if len(arg_tokens) != 1:
            raise ValueError(f"load_sideband_u64() expects 1 arg, got {arg_tokens}")
        return f"ctx.emit_load_sideband_u64(blk, {arg_tokens[0]}u)"

    if fn_name == "split64_lo":
        if len(arg_tokens) != 1:
            raise ValueError(f"split64_lo() expects 1 arg, got {arg_tokens}")
        a0 = compile_atom(arg_tokens[0], src_types, declared_vars, lookup_tables)
        return f"([&]() {{ SsaId _lo = 0, _hi = 0; ctx.emit_split64(blk, {a0}, &_lo, &_hi); return _lo; }})()"

    if fn_name == "bitcast":
        if len(arg_tokens) != 1:
            raise ValueError(f"bitcast() expects 1 arg, got {arg_tokens}")
        a0 = compile_atom(arg_tokens[0], src_types, declared_vars, lookup_tables)
        return f"ctx.sb.emit_bitcast(blk, {lhs_vt}, {a0})"

    if fn_name == "extract":
        if len(arg_tokens) != 2:
            raise ValueError(f"extract() expects 2 args, got {arg_tokens}")
        vec = compile_atom(arg_tokens[0], src_types, declared_vars, lookup_tables)
        idx_tok = arg_tokens[1].strip()
        if idx_tok == "$src0_swz_h1":
            idx_expr = '(get_src_swz(ins, 0) == "h1" ? 1u : 0u)'
        else:
            idx_expr = f"static_cast<uint32_t>({idx_tok})"
        return f"([&]() {{ SsaNode ex{{}}; ex.op = tables::SsaOp::VecExtract; ex.type = {lhs_vt}; ex.imm_u32 = {idx_expr}; ex.args = {{ {vec} }}; return ctx.sb.emit(blk, std::move(ex)); }})()"

    if fn_name == "shuffle2":
        if len(arg_tokens) != 4:
            raise ValueError(f"shuffle2() expects 4 args, got {arg_tokens}")
        a0 = compile_atom(arg_tokens[0], src_types, declared_vars, lookup_tables)
        a1 = compile_atom(arg_tokens[1], src_types, declared_vars, lookup_tables)
        return (f"([&]() {{ SsaNode sh{{}}; sh.op = tables::SsaOp::VecShuffle; sh.type = {lhs_vt}; "
                f"sh.args = {{ {a0}, {a1} }}; sh.imm_u32 = {arg_tokens[2]}; sh.imm_u32_2 = {arg_tokens[3]}; "
                f"return ctx.sb.emit(blk, std::move(sh)); }})()")

    if fn_name == "bfi":
        if len(arg_tokens) != 4:
            raise ValueError(f"bfi() expects 4 args, got {arg_tokens}")
        a0 = compile_atom(arg_tokens[0], src_types, declared_vars, lookup_tables)
        a1 = compile_atom(arg_tokens[1], src_types, declared_vars, lookup_tables)
        off = arg_tokens[2]
        wid = arg_tokens[3]
        return (f"([&]() {{ SsaNode bfi{{}}; bfi.op = tables::SsaOp::Bfi; bfi.type = ValType::U32; "
                f"bfi.imm_u32 = {off}; bfi.imm_u32_2 = {wid}; "
                f"bfi.args = {{ {a0}, {a1}, ctx.sb.emit_const_u32(blk, {off}u), ctx.sb.emit_const_u32(blk, {wid}u) }}; "
                f"return ctx.sb.emit(blk, std::move(bfi)); }})()")

    # Direct SsaOp call
    if fn_name not in canonical_ssa_ops and fn_name not in ("Kill", "ControlBarrier"):
        raise ValueError(f"Unknown SsaOp '{fn_name}' in inline_ops")
    arg_exprs = [(compile_inline_rhs(a, lhs_vt, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions) if a.strip().startswith("@") else compile_atom(a, src_types, declared_vars, lookup_tables)) for a in arg_tokens]
    if aux_bool is not None:
        ab_expr = "true" if aux_bool == "true" else ("false" if aux_bool == "false" else compile_cond(aux_bool))
        return (f"([&]() {{ SsaNode n{{}}; n.op = tables::SsaOp::{fn_name}; n.type = {lhs_vt}; "
                f"n.args = {{ {', '.join(arg_exprs)} }}; n.effects = rule->effects; "
                f"n.is_barrier_or_cf = {barrier}; n.is_side_effect = {side_eff}; n.aux_bool = {ab_expr}; "
                f"return ctx.sb.emit(blk, std::move(n)); }})()")
    return f"ctx.emit_op(blk, tables::SsaOp::{fn_name}, {lhs_vt}, {{ {', '.join(arg_exprs)} }}, rule->effects, {barrier}, {side_eff})"


def tokenize_inline_ops(ops_list):
    """Split inline_ops list into flat token stream of statements, '{', '}', 'if (...)', 'else'."""
    raw = " ".join(s.strip() for s in ops_list)
    tokens = []
    i = 0
    n = len(raw)
    while i < n:
        if raw[i].isspace():
            i += 1
            continue
        if raw[i] in ("{", "}"):
            tokens.append(raw[i])
            i += 1
            continue
        if raw.startswith("if", i) and (i + 2 < n and (raw[i+2].isspace() or raw[i+2] == "(")):
            lp = raw.find("(", i)
            depth = 0
            rp = -1
            for j in range(lp, n):
                if raw[j] == "(": depth += 1
                elif raw[j] == ")":
                    depth -= 1
                    if depth == 0:
                        rp = j
                        break
            if rp < 0:
                raise ValueError(f"Unclosed if(...) in inline_ops: {raw[i:]}")
            tokens.append(("IF", raw[lp+1:rp].strip()))
            i = rp + 1
            continue
        if raw.startswith("for", i) and (i + 3 < n and (raw[i+3].isspace() or raw[i+3] == "(")):
            lp = raw.find("(", i)
            depth = 0
            rp = -1
            for j in range(lp, n):
                if raw[j] == "(": depth += 1
                elif raw[j] == ")":
                    depth -= 1
                    if depth == 0:
                        rp = j
                        break
            if rp < 0:
                raise ValueError(f"Unclosed for(...) in inline_ops: {raw[i:]}")
            tokens.append(("FOR", raw[lp+1:rp].strip()))
            i = rp + 1
            continue
        if raw.startswith("else", i) and (i + 4 == n or raw[i+4].isspace() or raw[i+4] == "{"):
            tokens.append(("ELSE",))
            i += 4
            continue
        # Read until ';' or '}'
        semi = raw.find(";", i)
        rbrace = raw.find("}", i)
        if semi < 0 and rbrace < 0:
            stmt = raw[i:].strip()
            if stmt:
                tokens.append(("STMT", stmt))
            break
        end = semi if (semi >= 0 and (rbrace < 0 or semi < rbrace)) else rbrace
        stmt = raw[i:end].strip()
        if stmt:
            tokens.append(("STMT", stmt))
        i = end + (1 if raw[end] == ";" else 0)
    return tokens


def compile_inline_ops_rule(rule, lookup_tables, canonical_ssa_ops, host_functions):
    ops_list = rule.get("inline_ops", [])
    if not ops_list:
        raise ValueError(f"INLINE_OPS rule {rule['mnemonic']} has empty inline_ops")

    src_types = rule.get("src_types", [])
    dest_type = rule["dest_type"]
    tokens = tokenize_inline_ops(ops_list)

    # Pre-scan all declared variables so they can be declared at top scope if assigned in if/else
    declared_vars = {}
    for tok in tokens:
        if isinstance(tok, tuple) and tok[0] == "FOR":
            lhs_f = tok[1].split(":", 1)[0].strip()
            if lhs_f.startswith("%"):
                declared_vars[lhs_f[1:]] = "uint32_t"
    for tok in tokens:
        if isinstance(tok, tuple) and tok[0] == "STMT":
            stmt = tok[1]
            if "=" in stmt and stmt.lstrip().startswith("%"):
                lhs = stmt.split("=", 1)[0].strip()
                if ":" in lhs:
                    vname, vtype = [x.strip() for x in lhs.split(":", 1)]
                    vname = vname[1:]
                    if vtype not in VALTYPE_MAP:
                        raise ValueError(f"Unknown type '{vtype}' in inline_ops for {rule['mnemonic']}")
                    declared_vars[vname] = VALTYPE_MAP[vtype]

    lines = []
    # Pre-declare all variables at rule scope so if/else reassignments work cleanly
    for vname, vt_cpp in declared_vars.items():
        if vt_cpp == "uint32_t":
            lines.append(f"uint32_t v_{vname} = 0; (void)v_{vname};")
        else:
            lines.append(f"SsaId v_{vname} = kInvalidSsa; (void)v_{vname};")

    for tok in tokens:
        if tok == "{":
            lines.append("{")
        elif tok == "}":
            lines.append("}")
        elif isinstance(tok, tuple) and tok[0] == "FOR":
            parts = tok[1].split(":", 1)
            vname = parts[0].strip().lstrip("%")
            rng = parts[1].strip()
            assert rng.startswith("0 ..")
            bound = compile_loop_bound(rng[4:].strip())
            lines.append(f"for (v_{vname} = 0; v_{vname} < {bound}; ++v_{vname})")
        elif isinstance(tok, tuple) and tok[0] == "IF":
            lines.append(f"if ({compile_cond(tok[1])})")
        elif isinstance(tok, tuple) and tok[0] == "ELSE":
            lines.append("else")
        elif isinstance(tok, tuple) and tok[0] == "STMT":
            stmt = tok[1]
            if stmt == "return":
                lines.append("return true;")
            elif stmt.startswith("return "):
                expr = compile_atom(stmt[len("return "):], src_types, declared_vars, lookup_tables)
                st = gen_store_stmt(dest_type, expr)
                lines.append(f"{st} return true;")
            elif stmt.startswith("sr_set(") and stmt.endswith(")"):
                parts = split_top_level_args(stmt[len("sr_set("):-1])
                if len(parts) != 2:
                    raise ValueError(f"sr_set() expects 2 args, got {parts}")
                expr = compile_atom(parts[1], src_types, declared_vars, lookup_tables)
                lines.append(f"ctx.sb.set_reg(blk, static_cast<uint8_t>((ins.sr_write.present ? ins.sr_write.base_reg : 0) + ({parts[0]})), {expr});")
            elif stmt.startswith("set_branch_cond(") and stmt.endswith(")"):
                inner = stmt[len("set_branch_cond("):-1]
                expr = compile_atom(inner, src_types, declared_vars, lookup_tables)
                lines.append(f"ctx.mod.blocks[blk].branch_cond = {expr};")
            elif stmt.startswith("set_token(") and stmt.endswith(")"):
                inner = stmt[len("set_token("):-1]
                expr = compile_atom(inner, src_types, declared_vars, lookup_tables)
                lines.append(f"ctx.sb.set_token(blk, {expr});")
            elif "=" in stmt and stmt.lstrip().startswith("%"):
                lhs, rhs = stmt.split("=", 1)
                lhs = lhs.strip()
                if ":" in lhs:
                    vname, vtype = [x.strip() for x in lhs.split(":", 1)]
                    vname = vname[1:]
                    vt_cpp = VALTYPE_MAP[vtype]
                else:
                    vname = lhs[1:].strip()
                    if vname not in declared_vars:
                        raise ValueError(f"Reassignment to undeclared '%{vname}' in {rule['mnemonic']}")
                    vt_cpp = declared_vars[vname]
                rhs_cpp = compile_inline_rhs(rhs, vt_cpp, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions)
                lines.append(f"v_{vname} = {rhs_cpp};")
            else:
                # Standalone void op like Kill(#token)[barrier][side_effect]
                rhs_cpp = compile_inline_rhs(stmt, "ValType::Void", src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions)
                lines.append(f"(void)({rhs_cpp});")

    if "out" in declared_vars and dest_type != "none":
        st = gen_store_stmt(dest_type, "v_out")
        lines.append(st)
    lines.append("return true;")
    return " ".join(lines)


def main():
    args = parse_args()
    insns = load_xml_instructions(args.xml)
    with open(args.lowering, "r", encoding="utf-8") as f:
        lowering_json = json.load(f)
    with open(args.tiles, "r", encoding="utf-8") as f:
        tiles_json = json.load(f)

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
    spec_to_rule = []
    for ins in insns:
        ridx, r = match_lowering_rule(ins, rules)
        if r is None:
            uncovered.append((ins["index"], ins["name"], ins["variant"]))
            spec_to_rule.append(-1)
        else:
            spec_to_rule.append(ridx)
            kind, val = parsed_rules[ridx]
            if kind == "DirectSsaOp" and val not in canonical_ssa_ops:
                missing_canon_tiles.add(val)

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
    for ridx, r in enumerate(rules):
        cb, eb, exb, fl = parse_effects_mask(r.get("effects", []))
        kind, val = parsed_rules[ridx]
        ssa_op_enum = val if kind == "DirectSsaOp" else "Nop"
        hname = r["ssa_op"]
        cpp_lines.append(
            f'    {{ "{r["mnemonic"]}", {ridx}u, OpClass::{pascal(r["op_class"])}, '
            f'LoweringKind::{kind}, SsaOp::{ssa_op_enum}, "{hname}", '
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

    inc_lines.extend([
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

    print(f"[OK] Validated 100% coverage: {len(insns)} XML instructions -> {len(rules)} lowering rules "
          f"(direct={num_direct}, dynamic_lookup={num_dynamic}, inline_ops={num_inline}, handler={num_handler}), "
          f"{len(lookup_tables)} lookup tables, {len(canon_tiles)} canonical tiles + {len(fusion_specs)} fusion tiles.")


if __name__ == "__main__":
    main()

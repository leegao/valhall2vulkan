#!/usr/bin/env python3
"""
Valhall Pass 1 Lowering DSL Parser & C++17 Code Generator
=========================================================
Parses and type-checks:
  - DirectSsaOp rules
  - @LOOKUP(table, key) rules
  - @INLINE_OPS rules (with @FN uniform host ABI, $FN 0-arg metavariables,
    #FN constant constructors, if/else, and for loops)
  - @HANDLER(fn) rules
"""

import re

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
        return f'fetch_src_int32(ctx.sb, blk, ins, {idx}, (var.find("s") == 0))'
    if src_type == "u64":
        return f"fetch_src_u64(ctx.sb, blk, ins, {idx}, false)"
    if src_type == "s64":
        return f"fetch_src_u64(ctx.sb, blk, ins, {idx}, true)"
    if src_type == "i64_by_var":
        return f'fetch_src_u64(ctx.sb, blk, ins, {idx}, (var.find("s64") != std::string::npos))'
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
    mapping = {
        "f32": "ValType::F32",
        "f32_or_v2f16": "(is_v2f16 ? ValType::V2F32 : ValType::F32)",
        "u32": "ValType::U32",
        "s32": "ValType::S32",
        "u64": "ValType::U64",
        "v2f16": "ValType::V2F32",
        "v2s16": "ValType::V2S32",
        "v2u16": "ValType::V2U32",
        "v2i16_by_var": "(is_signed ? ValType::V2S32 : ValType::V2U32)",
        "none": "ValType::Void",
    }
    if dest_type not in mapping:
        raise ValueError(f"Unsupported dest_type '{dest_type}'")
    return mapping[dest_type]


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


def compile_host_call_expr(fn_name, arg_tokens, lhs_vt, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions, expected_ret=None):
    if fn_name not in host_functions:
        raise ValueError(f"Unknown host ABI function '{fn_name}' (not declared in host_functions)")
    sig = host_functions[fn_name]
    if expected_ret is not None and sig["ret"] != expected_ret:
        raise ValueError(f"@{fn_name} returns {sig['ret']}, expected {expected_ret}")
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
                raise ValueError(f"@{fn_name} expected {expected_args}, got {arg_tokens}")
            tok = arg_tokens[tok_idx].strip()
            tok_idx += 1
            cpp_args.append(compile_typed_arg(tok, atype, lhs_vt, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions))
    if tok_idx != len(arg_tokens):
        raise ValueError(f"@{fn_name} received extra args: {arg_tokens[tok_idx:]}")
    return f"abi::{fn_name}({', '.join(cpp_args)})"


def compile_typed_arg(tok, atype, lhs_vt, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions):
    t = tok.strip()
    if atype == "SsaId":
        return compile_ssa_expr(t, lhs_vt, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions)
    if atype == "uint32_t":
        if t.startswith("%"):
            vn = t[1:]
            if vn not in declared_vars:
                raise ValueError(f"Undeclared uint32_t var '{t}'")
            return f"v_{vn}"
        if t.startswith("$"):
            return compile_host_call_expr(t[1:], [], lhs_vt, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions, expected_ret="uint32_t")
        if t.startswith("@") and "(" in t and t.endswith(")"):
            lp = t.find("(")
            return compile_host_call_expr(t[1:lp], split_top_level_args(t[lp+1:-1]), lhs_vt, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions, expected_ret="uint32_t")
        if t.startswith("#lookup_u32_raw(") and t.endswith(")"):
            parts = split_top_level_args(t[len("#lookup_u32_raw("):-1])
            return gen_lookup_key_expr(parts[1], lookup_tables, parts[0])
        return f"static_cast<uint32_t>({t})"
    if atype == "int32_t":
        return f"static_cast<int32_t>({t})"
    if atype == "uint64_t":
        return f"static_cast<uint64_t>({t})"
    if atype == "float":
        return f"static_cast<float>({t})"
    if atype == "bool":
        if t in ("true", "false"):
            return t
        return compile_cond(t, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions)
    if atype == "const char*":
        if t.startswith('"') and t.endswith('"'):
            return t
        return f'"{t}"'
    raise ValueError(f"Unsupported host_functions arg type '{atype}'")


def compile_ssa_expr(expr_str, lhs_vt, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions):
    a = expr_str.strip()
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
    if a.startswith("#lookup_u32(") and a.endswith(")"):
        inner = a[len("#lookup_u32("):-1]
        parts = split_top_level_args(inner)
        if len(parts) != 2:
            raise ValueError(f"Invalid #lookup_u32 syntax: '{a}'")
        call_expr = gen_lookup_key_expr(parts[1], lookup_tables, parts[0])
        return f"ctx.sb.emit_const_u32(blk, {call_expr})"
    if a.startswith("$"):
        return compile_host_call_expr(a[1:], [], lhs_vt, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions, expected_ret="SsaId")
    if a.startswith("#"):
        inner = a[1:]
        if "(" in inner and inner.endswith(")"):
            lp = inner.find("(")
            return compile_host_call_expr(inner[:lp], split_top_level_args(inner[lp+1:-1]), lhs_vt, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions, expected_ret="SsaId")
        return compile_host_call_expr(inner, [], lhs_vt, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions, expected_ret="SsaId")
    if a.startswith("@"):
        return compile_inline_rhs(a, lhs_vt, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions)
    raise ValueError(f"Unknown inline_ops SsaId expression '{a}'")


def compile_cond(cond_str, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions):
    c = cond_str.strip()
    neg = False
    if c.startswith("!"):
        neg = True
        c = c[1:].strip()
    if c.startswith("$"):
        expr = compile_host_call_expr(c[1:], [], "ValType::Void", src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions, expected_ret="bool")
    elif c.startswith("@") and "(" in c and c.endswith(")"):
        lp = c.find("(")
        expr = compile_host_call_expr(c[1:lp], split_top_level_args(c[lp+1:-1]), "ValType::Void", src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions, expected_ret="bool")
    else:
        raise ValueError(f"Unknown inline_ops condition '{cond_str}' (must be $fn or @fn(...))")
    return f"!({expr})" if neg else expr


def compile_loop_bound(bound_str, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions):
    return compile_typed_arg(bound_str.strip(), "uint32_t", "ValType::Void", src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions)


def compile_inline_rhs(rhs, lhs_vt, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions):
    r = rhs.strip()
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

    if r.startswith("%") or r.startswith("#") or r.startswith("$"):
        return compile_ssa_expr(r, lhs_vt, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions)

    lp = r.find("(")
    if lp < 0 or not r.endswith(")"):
        raise ValueError(f"Invalid inline_ops RHS '{rhs}'")

    if r.startswith("@LOOKUP("):
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
        arg_exprs = [compile_ssa_expr(a, lhs_vt, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions) for a in split_top_level_args(args_part)]
        return f"ctx.emit_op(blk, {op_expr}, {lhs_vt}, {{ {', '.join(arg_exprs)} }}, rule->effects, {barrier}, {side_eff})"

    fn_name = r[:lp].strip()
    args_str = r[lp+1:-1].strip()
    arg_tokens = split_top_level_args(args_str)

    if fn_name.startswith("@"):
        return compile_host_call_expr(fn_name[1:], arg_tokens, lhs_vt, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions)

    if fn_name not in canonical_ssa_ops and fn_name not in ("Kill", "ControlBarrier"):
        raise ValueError(f"Unknown SsaOp '{fn_name}' in inline_ops")
    arg_exprs = [compile_ssa_expr(a, lhs_vt, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions) for a in arg_tokens]
    if aux_bool is not None:
        ab_expr = "true" if aux_bool == "true" else ("false" if aux_bool == "false" else compile_cond(aux_bool, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions))
        return (f"([&]() {{ SsaNode n{{}}; n.op = tables::SsaOp::{fn_name}; n.type = {lhs_vt}; "
                f"n.args = {{ {', '.join(arg_exprs)} }}; n.effects = rule->effects; "
                f"n.is_barrier_or_cf = {barrier}; n.is_side_effect = {side_eff}; n.aux_bool = {ab_expr}; "
                f"return ctx.sb.emit(blk, std::move(n)); }})()")
    return f"ctx.emit_op(blk, tables::SsaOp::{fn_name}, {lhs_vt}, {{ {', '.join(arg_exprs)} }}, rule->effects, {barrier}, {side_eff})"


def tokenize_inline_ops(ops_list):
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
                if raw[j] == "(":
                    depth += 1
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
                if raw[j] == "(":
                    depth += 1
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
            bound = compile_loop_bound(rng[4:].strip(), src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions)
            lines.append(f"for (v_{vname} = 0; v_{vname} < {bound}; ++v_{vname})")
        elif isinstance(tok, tuple) and tok[0] == "IF":
            lines.append(f"if ({compile_cond(tok[1], src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions)})")
        elif isinstance(tok, tuple) and tok[0] == "ELSE":
            lines.append("else")
        elif isinstance(tok, tuple) and tok[0] == "STMT":
            stmt = tok[1]
            if stmt == "return":
                lines.append("return true;")
            elif stmt.startswith("return "):
                expr = compile_ssa_expr(stmt[len("return "):], "ValType::Void", src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions)
                st = gen_store_stmt(dest_type, expr)
                lines.append(f"{st} return true;")
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
                rhs_cpp = compile_inline_rhs(stmt, "ValType::Void", src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions)
                if stmt.lstrip().startswith("@"):
                    lines.append(f"{rhs_cpp};")
                else:
                    lines.append(f"(void)({rhs_cpp});")

    if "out" in declared_vars and dest_type != "none":
        st = gen_store_stmt(dest_type, "v_out")
        lines.append(st)
    lines.append("return true;")
    return " ".join(lines)

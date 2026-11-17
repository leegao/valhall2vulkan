#!/usr/bin/env python3
"""
Valhall Pass 1 Lowering DSL Compiler & C++17 Code Generator
============================================================
Consumes the strict LL(1) AST from dsl_parser.py and compiles:
  - DirectSsaOp rules
  - @LOOKUP(table, key) rules
  - @INLINE_OPS rules (with @FN uniform host ABI, $FN 0-arg metavariables,
    #FN constant constructors, if/else, and for loops)
  - @HANDLER(fn) rules
"""

import re
from dsl_parser import (
    AssignStmt,
    BareIdent,
    Cond,
    DollarCall,
    ExprStmt,
    ForStmt,
    HashExpr,
    HostCall,
    IfStmt,
    Literal,
    LookupCall,
    ModifierKey,
    ReturnStmt,
    SrcRef,
    SsaCall,
    VarRef,
    parse_inline_ops,
)

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


# ---------------------------------------------------------------------------
# AST-based @INLINE_OPS Compiler
# ---------------------------------------------------------------------------

def _collect_decls(stmts, declared_vars, mnemonic):
    for st in stmts:
        if isinstance(st, ForStmt):
            declared_vars[st.var] = "uint32_t"
            _collect_decls(st.body, declared_vars, mnemonic)
        elif isinstance(st, AssignStmt):
            if st.val_type is not None:
                if st.val_type not in VALTYPE_MAP:
                    raise ValueError(f"Unknown type '{st.val_type}' in inline_ops for {mnemonic}")
                declared_vars[st.var] = VALTYPE_MAP[st.val_type]
        elif isinstance(st, IfStmt):
            _collect_decls(st.then_body, declared_vars, mnemonic)
            if st.else_body is not None:
                _collect_decls(st.else_body, declared_vars, mnemonic)


def _compile_host_call_ast(fn_name, arg_nodes, lhs_vt, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions, expected_ret=None):
    if fn_name not in host_functions:
        raise ValueError(f"Unknown host ABI function '{fn_name}' (not declared in host_functions)")
    sig = host_functions[fn_name]
    if expected_ret is not None and sig["ret"] != expected_ret:
        raise ValueError(f"@{fn_name} returns {sig['ret']}, expected {expected_ret}")
    expected_args = sig["args"]
    cpp_args = ["ctx", "blk", "ins", "rule"]
    tok_idx = 0
    for atype in expected_args:
        if atype == "ValType" and tok_idx < len(arg_nodes) and isinstance(arg_nodes[tok_idx], BareIdent) and arg_nodes[tok_idx].name in VALTYPE_MAP:
            cpp_args.append(VALTYPE_MAP[arg_nodes[tok_idx].name])
            tok_idx += 1
        elif atype == "ValType":
            cpp_args.append(lhs_vt)
        else:
            if tok_idx >= len(arg_nodes):
                raise ValueError(f"@{fn_name} expected {expected_args}, got {arg_nodes}")
            node = arg_nodes[tok_idx]
            tok_idx += 1
            cpp_args.append(_compile_typed_arg_ast(node, atype, lhs_vt, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions))
    if tok_idx != len(arg_nodes):
        raise ValueError(f"@{fn_name} received extra args: {arg_nodes[tok_idx:]}")
    return f"abi::{fn_name}({', '.join(cpp_args)})"


def _compile_typed_arg_ast(node, atype, lhs_vt, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions):
    if atype == "SsaId":
        return _compile_expr_ast(node, lhs_vt, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions)
    if atype == "uint32_t":
        if isinstance(node, VarRef):
            if node.name not in declared_vars:
                raise ValueError(f"Undeclared uint32_t var '%{node.name}'")
            return f"v_{node.name}"
        if isinstance(node, DollarCall):
            return _compile_host_call_ast(node.name, [], lhs_vt, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions, expected_ret="uint32_t")
        if isinstance(node, HostCall):
            return _compile_host_call_ast(node.name, node.args, lhs_vt, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions, expected_ret="uint32_t")
        if isinstance(node, HashExpr) and node.name == "lookup_u32_raw":
            table_ident, key_node = node.args
            assert isinstance(table_ident, BareIdent) and isinstance(key_node, ModifierKey)
            return gen_lookup_key_expr(f"$modifiers.{key_node.mod_name}", lookup_tables, table_ident.name)
        if isinstance(node, Literal) and node.kind == "int":
            return f"static_cast<uint32_t>({node.raw})"
        raise ValueError(f"Invalid uint32_t argument AST node: {node}")
    if atype == "int32_t":
        if isinstance(node, Literal) and node.kind == "int":
            return f"static_cast<int32_t>({node.raw})"
        raise ValueError(f"Invalid int32_t argument AST node: {node}")
    if atype == "uint64_t":
        if isinstance(node, Literal) and node.kind == "int":
            return f"static_cast<uint64_t>({node.raw})"
        raise ValueError(f"Invalid uint64_t argument AST node: {node}")
    if atype == "float":
        if isinstance(node, Literal) and node.kind in ("float", "int"):
            return f"static_cast<float>({node.raw})"
        raise ValueError(f"Invalid float argument AST node: {node}")
    if atype == "bool":
        if isinstance(node, Literal) and node.kind == "bool":
            return node.raw
        if isinstance(node, Cond):
            return _compile_cond_ast(node, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions)
        if isinstance(node, (DollarCall, HostCall)):
            return _compile_cond_ast(Cond(False, node), src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions)
        raise ValueError(f"Invalid bool argument AST node: {node}")
    if atype == "const char*":
        if isinstance(node, Literal) and node.kind == "string":
            return node.raw
        if isinstance(node, BareIdent):
            return f'"{node.name}"'
        raise ValueError(f"Invalid const char* argument AST node: {node}")
    raise ValueError(f"Unsupported host_functions arg type '{atype}'")


def _compile_cond_ast(cond: Cond, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions):
    atom = cond.atom
    if isinstance(atom, DollarCall):
        expr = _compile_host_call_ast(atom.name, [], "ValType::Void", src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions, expected_ret="bool")
    elif isinstance(atom, HostCall):
        expr = _compile_host_call_ast(atom.name, atom.args, "ValType::Void", src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions, expected_ret="bool")
    else:
        raise ValueError(f"Unknown condition atom AST node: {atom}")
    return f"!({expr})" if cond.negated else expr


def _compile_flags(flags, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions):
    barrier = "false"
    side_eff = "false"
    aux_bool = None
    for fl in flags:
        if fl.name == "barrier":
            barrier = "true"
        elif fl.name == "side_effect":
            side_eff = "true"
        elif fl.name == "aux_bool":
            if fl.aux_bool is True:
                aux_bool = "true"
            elif fl.aux_bool is False:
                aux_bool = "false"
            else:
                aux_bool = _compile_cond_ast(fl.aux_bool, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions)
        else:
            raise ValueError(f"Unknown inline_ops node flag '[{fl.name}]'")
    return barrier, side_eff, aux_bool


def _compile_expr_ast(node, lhs_vt, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions):
    if isinstance(node, SrcRef):
        if node.index >= len(src_types):
            raise ValueError(f"inline_ops references %{node.index}, but src_types only has {len(src_types)} entries")
        return gen_fetch_expr(src_types[node.index], node.index)
    if isinstance(node, VarRef):
        if node.name not in declared_vars:
            raise ValueError(f"inline_ops references undeclared variable '%{node.name}'")
        return f"v_{node.name}"
    if isinstance(node, DollarCall):
        return _compile_host_call_ast(node.name, [], lhs_vt, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions, expected_ret="SsaId")
    if isinstance(node, HashExpr):
        if node.name == "lookup_u32":
            table_ident, key_node = node.args
            assert isinstance(table_ident, BareIdent) and isinstance(key_node, ModifierKey)
            call_expr = gen_lookup_key_expr(f"$modifiers.{key_node.mod_name}", lookup_tables, table_ident.name)
            return f"ctx.sb.emit_const_u32(blk, {call_expr})"
        return _compile_host_call_ast(node.name, node.args or [], lhs_vt, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions, expected_ret="SsaId")
    if isinstance(node, HostCall):
        return _compile_host_call_ast(node.name, node.args, lhs_vt, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions)
    if isinstance(node, LookupCall):
        op_expr = gen_lookup_key_expr(f"$modifiers.{node.key.mod_name}", lookup_tables, node.table)
        arg_exprs = [_compile_expr_ast(a, lhs_vt, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions) for a in node.args]
        barrier, side_eff, _ = _compile_flags(node.flags, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions)
        return f"ctx.emit_op(blk, {op_expr}, {lhs_vt}, {{ {', '.join(arg_exprs)} }}, rule->effects, {barrier}, {side_eff})"
    if isinstance(node, SsaCall):
        if node.op not in canonical_ssa_ops and node.op not in ("Kill", "ControlBarrier"):
            raise ValueError(f"Unknown SsaOp '{node.op}' in inline_ops")
        arg_exprs = [_compile_expr_ast(a, lhs_vt, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions) for a in node.args]
        barrier, side_eff, aux_bool = _compile_flags(node.flags, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions)
        if aux_bool is not None:
            return (f"([&]() {{ SsaNode n{{}}; n.op = tables::SsaOp::{node.op}; n.type = {lhs_vt}; "
                    f"n.args = {{ {', '.join(arg_exprs)} }}; n.effects = rule->effects; "
                    f"n.is_barrier_or_cf = {barrier}; n.is_side_effect = {side_eff}; n.aux_bool = {aux_bool}; "
                    f"return ctx.sb.emit(blk, std::move(n)); }})()")
        return f"ctx.emit_op(blk, tables::SsaOp::{node.op}, {lhs_vt}, {{ {', '.join(arg_exprs)} }}, rule->effects, {barrier}, {side_eff})"
    raise ValueError(f"Unknown inline_ops expression AST node: {node}")


def _compile_stmts(stmts, dest_type, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions, mnemonic):
    lines = []
    for st in stmts:
        if isinstance(st, ReturnStmt):
            if st.value is None:
                lines.append("return true;")
            else:
                expr = _compile_expr_ast(st.value, "ValType::Void", src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions)
                store = gen_store_stmt(dest_type, expr)
                lines.append(f"{store} return true;")
        elif isinstance(st, AssignStmt):
            if st.val_type is not None:
                vt_cpp = VALTYPE_MAP[st.val_type]
            else:
                if st.var not in declared_vars:
                    raise ValueError(f"Reassignment to undeclared '%{st.var}' in {mnemonic}")
                vt_cpp = declared_vars[st.var]
            rhs_cpp = _compile_expr_ast(st.rhs, vt_cpp, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions)
            lines.append(f"v_{st.var} = {rhs_cpp};")
        elif isinstance(st, ExprStmt):
            rhs_cpp = _compile_expr_ast(st.expr, "ValType::Void", src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions)
            if isinstance(st.expr, HostCall):
                lines.append(f"{rhs_cpp};")
            else:
                lines.append(f"(void)({rhs_cpp});")
        elif isinstance(st, IfStmt):
            cond_cpp = _compile_cond_ast(st.cond, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions)
            then_lines = _compile_stmts(st.then_body, dest_type, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions, mnemonic)
            lines.append(f"if ({cond_cpp}) {{ {' '.join(then_lines)} }}")
            if st.else_body is not None:
                else_lines = _compile_stmts(st.else_body, dest_type, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions, mnemonic)
                lines.append(f"else {{ {' '.join(else_lines)} }}")
        elif isinstance(st, ForStmt):
            bound_cpp = _compile_typed_arg_ast(st.bound, "uint32_t", "ValType::Void", src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions)
            body_lines = _compile_stmts(st.body, dest_type, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions, mnemonic)
            lines.append(f"for (v_{st.var} = 0; v_{st.var} < {bound_cpp}; ++v_{st.var}) {{ {' '.join(body_lines)} }}")
        else:
            raise ValueError(f"Unknown statement AST node: {st}")
    return lines


def compile_inline_ops_rule(rule, lookup_tables, canonical_ssa_ops, host_functions):
    ops_list = rule.get("inline_ops", [])
    if not ops_list:
        raise ValueError(f"INLINE_OPS rule {rule['mnemonic']} has empty inline_ops")

    src_types = rule.get("src_types", [])
    dest_type = rule["dest_type"]
    prog = parse_inline_ops(ops_list)

    declared_vars = {}
    _collect_decls(prog.stmts, declared_vars, rule["mnemonic"])

    lines = []
    for vname, vt_cpp in declared_vars.items():
        if vt_cpp == "uint32_t":
            lines.append(f"uint32_t v_{vname} = 0; (void)v_{vname};")
        else:
            lines.append(f"SsaId v_{vname} = kInvalidSsa; (void)v_{vname};")

    lines.extend(_compile_stmts(prog.stmts, dest_type, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions, rule["mnemonic"]))

    if "out" in declared_vars and dest_type != "none":
        st = gen_store_stmt(dest_type, "v_out")
        lines.append(st)
    lines.append("return true;")
    return " ".join(lines)

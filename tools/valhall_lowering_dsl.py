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
    CondBinOp,
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
    parse_cond_str,
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
        return f"fetch_src_f32(ctx, blk, ins, rule, {idx})"
    if src_type == "f32_or_v2f16":
        return f"(is_v2f16 ? fetch_src_v2f16(ctx, blk, ins, rule, {idx}) : fetch_src_f32(ctx, blk, ins, rule, {idx}))"
    if src_type == "u32":
        return f"fetch_src_int32(ctx, blk, ins, rule, {idx}, false)"
    if src_type == "s32_raw":
        return f"fetch_src_int32(ctx, blk, ins, rule, {idx}, true)"
    if src_type == "s32":
        return f"ctx.sb.emit_bitcast(blk, ValType::S32, fetch_src_int32(ctx, blk, ins, rule, {idx}, true))"
    if src_type == "i32_by_var":
        return f'fetch_src_int32(ctx, blk, ins, rule, {idx}, (var.find("s") == 0))'
    if src_type == "u64":
        return f"fetch_src_u64(ctx, blk, ins, rule, {idx}, false)"
    if src_type == "s64":
        return f"fetch_src_u64(ctx, blk, ins, rule, {idx}, true)"
    if src_type == "i64_by_var":
        return f'fetch_src_u64(ctx, blk, ins, rule, {idx}, (var.find("s64") != std::string::npos))'
    if src_type == "v2f16":
        return f"fetch_src_v2f16(ctx, blk, ins, rule, {idx})"
    if src_type == "v2s16":
        return f"fetch_src_v2i16(ctx, blk, ins, rule, {idx}, true)"
    if src_type == "v2u16":
        return f"fetch_src_v2i16(ctx, blk, ins, rule, {idx}, false)"
    if src_type == "v2i16_by_var":
        return f"fetch_src_v2i16(ctx, blk, ins, rule, {idx}, is_signed)"
    if src_type == "v4i8_by_var":
        return f"fetch_src_v4i8(ctx, blk, ins, rule, {idx}, is_signed)"
    if src_type == "v4s8":
        return f"fetch_src_v4i8(ctx, blk, ins, rule, {idx}, true)"
    if src_type == "v4u8":
        return f"fetch_src_v4i8(ctx, blk, ins, rule, {idx}, false)"
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
        return f"store_dest_f32(ctx, blk, ins, rule, {val_expr});"
    if dest_type == "f32_or_v2f16":
        return f"if (is_v2f16) store_dest_v2f16(ctx, blk, ins, rule, {val_expr}); else store_dest_f32(ctx, blk, ins, rule, {val_expr});"
    if dest_type == "u32":
        return f"store_dest_u32(ctx, blk, ins, rule, {val_expr});"
    if dest_type == "s32":
        return f"store_dest_u32(ctx, blk, ins, rule, ctx.sb.emit_bitcast(blk, ValType::U32, {val_expr}));"
    if dest_type == "u64":
        return f"store_dest_u64(ctx, blk, ins, rule, {val_expr});"
    if dest_type == "v2f16":
        return f"store_dest_v2f16(ctx, blk, ins, rule, {val_expr});"
    if dest_type == "v2s16":
        return f"store_dest_v2i16(ctx, blk, ins, rule, {val_expr}, true);"
    if dest_type == "v2u16":
        return f"store_dest_v2i16(ctx, blk, ins, rule, {val_expr}, false);"
    if dest_type == "v2i16_by_var":
        return f"store_dest_v2i16(ctx, blk, ins, rule, {val_expr}, is_signed);"
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
    for atype_raw in expected_args:
        atype = parse_param_spec(atype_raw, 0)[0]
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
            return f'"{node.raw[1:-1]}"'
        if isinstance(node, BareIdent):
            return f'"{node.name}"'
        raise ValueError(f"Invalid const char* argument AST node: {node}")
    raise ValueError(f"Unsupported host_functions arg type '{atype}'")


def _compile_cond_ast(cond, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions):
    if isinstance(cond, CondBinOp):
        l = _compile_cond_ast(cond.left, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions)
        r = _compile_cond_ast(cond.right, src_types, declared_vars, lookup_tables, canonical_ssa_ops, host_functions)
        return f"({l} {cond.op} {r})"
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


# ---------------------------------------------------------------------------
# Unified Parameter Schema ("TYPE" or "TYPE %name") & Operands C++17 Codegen
# ---------------------------------------------------------------------------

def parse_param_spec(spec_str, index):
    s = spec_str.strip()
    m = re.match(r"^(.+?)\s+%([A-Za-z_][A-Za-z0-9_]*)$", s)
    if m:
        return (m.group(1).strip(), m.group(2))
    return (s, f"arg{index}")


def parse_param_list(params):
    return [parse_param_spec(p, i) for i, p in enumerate(params)]


_OPERAND_DSL_TYPE_TO_CPP = {
    "u32": "SsaId",
    "s32": "SsaId",
    "f32": "SsaId",
    "u64": "SsaId",
    "s64": "SsaId",
    "v2f32": "SsaId",
    "v2u32": "SsaId",
    "v2s32": "SsaId",
    "v4f32": "SsaId",
    "v4u32": "SsaId",
    "v4s32": "SsaId",
    "uint32_t": "uint32_t",
    "bool": "bool",
    "ValType": "ValType",
}

_OPERAND_SSA_OP_DEFAULTS = {
    "IAdd64": ("ValType::U64", ["cap:Int64"]),
    "SConvert32To64": ("ValType::S64", ["cap:Int64"]),
    "UConvert32To64": ("ValType::U64", ["cap:Int64"]),
    "BfeS": ("ValType::S32", []),
    "BfeU": ("ValType::U32", []),
    "Shl": ("ValType::U32", []),
    "BitOr": ("ValType::U32", []),
    "IMul": ("ValType::U32", []),
    "BitNot": ("ValType::U32", []),
    "SAbs": ("ValType::S32", ["uses_glsl_std450"]),
    "SNeg": ("ValType::S32", []),
    "UnpackHalf2x16": ("ValType::V2F32", ["uses_glsl_std450"]),
    "PackHalf2x16": ("ValType::U32", ["uses_glsl_std450"]),
    "FAbs": ("ValType::F32", ["uses_glsl_std450"]),
    "FNeg": ("ValType::F32", []),
    "FClamp": ("ValType::F32", ["uses_glsl_std450"]),
    "FMax": ("ValType::F32", ["uses_glsl_std450"]),
    "VecConstruct2": ("ValType::V2F32", []),
    "VecConstruct4": ("ValType::V4U32", []),
}

_OPERAND_EFFECT_STMTS = {
    "uses_sideband": "{nv}.effects.flags |= EffectSet::kFlagUsesSideband;",
    "uses_glsl_std450": "{nv}.effects.flags |= EffectSet::kFlagUsesGlsl450;",
    "cap:Int64": "{nv}.effects.cap_bits |= EffectSet::kCapInt64;",
}


class _OperandsDslCompiler:
    def __init__(self, host_functions, operand_names):
        self.host_functions = host_functions
        self.operand_names = operand_names
        self._node_idx = 0

    def next_nv(self):
        self._node_idx += 1
        return f"_n{self._node_idx}"

    def compile_fn(self, name, ret, params_list, inline_ops, is_static, fn_effects, prepend_ctx=True):
        prog = parse_inline_ops(inline_ops)
        parsed_params = parse_param_list(params_list)
        if prepend_ctx:
            all_params = [
                ("Pass1Context&", "ctx"),
                ("uint32_t", "blk"),
                ("const DecodedValhallInst&", "ins"),
                ("const tables::LoweringRuleSpec*", "rule"),
            ] + parsed_params
        else:
            all_params = parsed_params
        lines = []
        prefix = "static " if is_static else ""
        p_str = ", ".join(f"{t} {n}" for t, n in all_params)
        lines.append(f"{prefix}{ret} {name}({p_str}) {{")
        env_types = {n: t for t, n in all_params}
        declared = {n for _, n in all_params}
        self._emit_stmts(prog.stmts, 1, lines, env_types, declared, fn_effects)
        lines.append("}")
        lines.append("")
        return lines

    def _emit_if(self, st, indent, lines, env_types, declared, fn_effects, prefix_else=False):
        pad = "    " * indent
        cond_cpp = self._emit_cond(st.cond, env_types, declared)
        if prefix_else:
            lines.append(f"{pad}}} else if ({cond_cpp}) {{")
        else:
            lines.append(f"{pad}if ({cond_cpp}) {{")
        self._emit_stmts(st.then_body, indent + 1, lines, dict(env_types), set(declared), fn_effects)
        if st.else_body is not None:
            if len(st.else_body) == 1 and isinstance(st.else_body[0], IfStmt):
                self._emit_if(st.else_body[0], indent, lines, env_types, declared, fn_effects, prefix_else=True)
            else:
                lines.append(f"{pad}}} else {{")
                self._emit_stmts(st.else_body, indent + 1, lines, dict(env_types), set(declared), fn_effects)
                lines.append(f"{pad}}}")
        else:
            lines.append(f"{pad}}}")

    def _emit_stmts(self, stmts, indent, lines, env_types, declared, fn_effects):
        pad = "    " * indent
        for st in stmts:
            if isinstance(st, AssignStmt):
                vt_hint = VALTYPE_MAP.get(st.val_type) if st.val_type else None
                expr_cpp = self._emit_expr(st.rhs, indent, lines, env_types, declared, fn_effects, vt_hint)
                vname = st.var
                if vname in declared:
                    lines.append(f"{pad}{vname} = {expr_cpp};")
                else:
                    cpp_t = _OPERAND_DSL_TYPE_TO_CPP.get(st.val_type or "u32", "SsaId")
                    if st.val_type:
                        env_types[vname] = st.val_type
                    declared.add(vname)
                    lines.append(f"{pad}{cpp_t} {vname} = {expr_cpp};")
            elif isinstance(st, ExprStmt):
                expr_cpp = self._emit_expr(st.expr, indent, lines, env_types, declared, fn_effects, None)
                lines.append(f"{pad}{expr_cpp};")
            elif isinstance(st, ReturnStmt):
                if st.value is None:
                    lines.append(f"{pad}return;")
                else:
                    expr_cpp = self._emit_expr(st.value, indent, lines, env_types, declared, fn_effects, None)
                    lines.append(f"{pad}return {expr_cpp};")
            elif isinstance(st, IfStmt):
                self._emit_if(st, indent, lines, env_types, declared, fn_effects, prefix_else=False)
            elif isinstance(st, ForStmt):
                bound_cpp = self._emit_arg(st.bound, env_types, declared)
                iv = st.var
                lines.append(f"{pad}for (uint32_t {iv} = 0; {iv} < {bound_cpp}; ++{iv}) {{")
                inner_env = dict(env_types)
                inner_decl = set(declared)
                inner_env[iv] = "uint32_t"
                inner_decl.add(iv)
                self._emit_stmts(st.body, indent + 1, lines, inner_env, inner_decl, fn_effects)
                lines.append(f"{pad}}}")
            else:
                raise ValueError(f"Unsupported DSL statement: {st!r}")

    def _emit_cond(self, cond, env_types, declared):
        from dsl_parser import CondBinOp
        if isinstance(cond, CondBinOp):
            return f"({self._emit_cond(cond.left, env_types, declared)} {cond.op} {self._emit_cond(cond.right, env_types, declared)})"
        if isinstance(cond, Cond):
            atom = cond.atom
            if isinstance(atom, VarRef):
                c = atom.name
            elif isinstance(atom, HostCall):
                c = self._emit_host_call(atom, env_types, declared)
            elif isinstance(atom, DollarCall):
                c = f"abi::{atom.name}(ctx, blk, ins, rule)"
            else:
                raise ValueError(f"Unsupported cond atom: {atom!r}")
            return f"!({c})" if cond.negated else c
        raise ValueError(f"Unsupported cond: {cond!r}")

    def _emit_arg(self, arg, env_types, declared):
        if isinstance(arg, SrcRef):
            return f"arg{arg.index}"
        if isinstance(arg, VarRef):
            return arg.name
        if isinstance(arg, Literal):
            if arg.kind == "string":
                return f'"{arg.raw[1:-1]}"'
            return arg.raw
        if isinstance(arg, BareIdent):
            if arg.name in VALTYPE_MAP:
                return VALTYPE_MAP[arg.name]
            return arg.name
        if isinstance(arg, HostCall):
            return self._emit_host_call(arg, env_types, declared)
        if isinstance(arg, DollarCall):
            return f"abi::{arg.name}(ctx, blk, ins, rule)"
        if isinstance(arg, Cond):
            return self._emit_cond(arg, env_types, declared)
        raise ValueError(f"Unsupported DSL arg: {arg!r}")

    def _emit_host_call(self, hc, env_types, declared):
        args_cpp = [self._emit_arg(a, env_types, declared) for a in hc.args]
        if hc.name in self.operand_names:
            return f"{hc.name}({', '.join(['ctx', 'blk', 'ins', 'rule'] + args_cpp)})"
        if hc.name not in self.host_functions:
            raise ValueError(f"Unknown @host function '{hc.name}' in valhall_lowering.json")
        return f"abi::{hc.name}({', '.join(['ctx', 'blk', 'ins', 'rule'] + args_cpp)})"

    def _emit_expr(self, expr, indent, lines, env_types, declared, fn_effects, vt_hint):
        if isinstance(expr, (SrcRef, VarRef, Literal, BareIdent, DollarCall)):
            return self._emit_arg(expr, env_types, declared)
        if isinstance(expr, HostCall):
            return self._emit_host_call(expr, env_types, declared)
        if isinstance(expr, SsaCall):
            pad = "    " * indent
            nv = self.next_nv()
            arg_exprs = [self._emit_arg(a, env_types, declared) for a in expr.args]
            default_vt, op_effs = _OPERAND_SSA_OP_DEFAULTS.get(expr.op, ("ValType::U32", []))
            vt = vt_hint or default_vt
            if vt_hint is None and expr.op in ("VecConstruct2", "FClamp", "FMax", "FAbs", "FNeg"):
                if expr.args and isinstance(expr.args[0], VarRef):
                    at = env_types.get(expr.args[0].name)
                    if at in VALTYPE_MAP:
                        vt = VALTYPE_MAP[at]
            lines.append(f"{pad}SsaNode {nv}{{}};")
            lines.append(f"{pad}{nv}.op = tables::SsaOp::{expr.op};")
            lines.append(f"{pad}{nv}.type = {vt};")
            lines.append(f"{pad}{nv}.args = {{ {', '.join(arg_exprs)} }};")
            effs = list(dict.fromkeys(list(op_effs) + list(fn_effects)))
            for ef in effs:
                if ef in _OPERAND_EFFECT_STMTS:
                    lines.append(f"{pad}{_OPERAND_EFFECT_STMTS[ef].format(nv=nv)}")
            return f"ctx.sb.emit(blk, std::move({nv}))"
        raise ValueError(f"Unsupported DSL expr: {expr!r}")


def compile_operands_cpp(spec):
    host_functions = spec.get("host_functions", {})
    operands = spec.get("operands", [])
    operand_names = {e["name"] for e in operands}
    comp = _OperandsDslCompiler(host_functions, operand_names)

    lines = [
        "// AUTO-GENERATED FILE — DO NOT EDIT.",
        "// Generated from specs/lowering/valhall_lowering.json (operands + inline_ops host_functions) by tools/gen_valhall_tables.py",
        "",
        '#include "valhall_operands.h"',
        '#include "valhall_pass1_handlers.h"',
        "",
        "namespace valhall::compiler {",
        "",
    ]

    for entry in operands:
        if entry["kind"] == "host":
            parsed = parse_param_list(entry.get("params", []))
            all_params = [
                ("Pass1Context&", "ctx"),
                ("uint32_t", "blk"),
                ("const DecodedValhallInst&", "ins"),
                ("const tables::LoweringRuleSpec*", "rule"),
            ] + parsed
            p_decl = ", ".join(f"{t} {n}" for t, n in all_params)
            p_call = ", ".join(n for _, n in all_params)
            lines.append(f"{entry['ret']} {entry['name']}({p_decl}) {{")
            lines.append(f"    return {entry['target']}({p_call});")
            lines.append("}")
            lines.append("")
        elif entry["kind"] == "inline_ops":
            lines.extend(
                comp.compile_fn(
                    entry["name"],
                    entry["ret"],
                    entry.get("params", []),
                    entry["inline_ops"],
                    is_static=False,
                    fn_effects=entry.get("effects", []),
                    prepend_ctx=True,
                )
            )
        else:
            raise ValueError(f"Unknown operand kind {entry.get('kind')!r} for {entry.get('name')!r}")

    lines.append("namespace abi {")
    lines.append("")
    for hname, hdef in host_functions.items():
        if hdef.get("kind") != "inline_ops":
            continue
        lines.extend(
            comp.compile_fn(
                hname,
                hdef["ret"],
                hdef.get("args", []),
                hdef["inline_ops"],
                is_static=False,
                fn_effects=hdef.get("effects", []),
                prepend_ctx=True,
            )
        )
    lines.append("} // namespace abi")
    lines.append("")
    lines.append("} // namespace valhall::compiler")
    return "\n".join(lines) + "\n"

def compile_when_expr(when_str: str, lookup_tables, canonical_ssa_ops, host_functions) -> str:
    cond_ast = parse_cond_str(when_str)
    return _compile_cond_ast(cond_ast, [], set(), lookup_tables, canonical_ssa_ops, host_functions)

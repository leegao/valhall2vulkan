#!/usr/bin/env python3
"""
Build-Time Spec Validator & C++17 Code Generator for Valhall Operands
=====================================================================
Inputs:
  - specs/lowering/valhall_operands.json (schema_version: 2, InlineOps DSL)

Outputs:
  - src/generated/valhall_operands_generated.cpp
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
    ReturnStmt,
    SsaCall,
    VarRef,
    parse_inline_ops,
)

DSL_TYPE_TO_CPP = {
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

DSL_VALTYPE_MAP = {
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
    "bool": "ValType::Bool",
}

SSA_OP_DEFAULTS: Dict[str, Tuple[str, List[str]]] = {
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

EFFECT_STMTS = {
    "uses_sideband": "{nv}.effects.flags |= EffectSet::kFlagUsesSideband;",
    "uses_glsl_std450": "{nv}.effects.flags |= EffectSet::kFlagUsesGlsl450;",
    "cap:Int64": "{nv}.effects.cap_bits |= EffectSet::kCapInt64;",
}


class OperandsDslCompiler:
    def __init__(
        self,
        helpers: Dict[str, Dict[str, Any]],
        entry_abi_names: Set[str],
    ) -> None:
        self.helpers = helpers
        self.entry_abi_names = entry_abi_names
        self._node_idx = 0

    def next_nv(self) -> str:
        self._node_idx += 1
        return f"_n{self._node_idx}"

    def compile_fn(
        self,
        name: str,
        ret: str,
        params: List[Dict[str, str]],
        inline_ops: List[str],
        is_static: bool,
        fn_effects: List[str],
    ) -> List[str]:
        prog = parse_inline_ops(inline_ops)
        lines: List[str] = []
        prefix = "static " if is_static else ""
        p_str = ", ".join(f"{p['type']} {p['name']}" for p in params)
        lines.append(f"{prefix}{ret} {name}({p_str}) {{")

        env_types: Dict[str, str] = {p["name"]: p["type"] for p in params}
        declared: Set[str] = {p["name"] for p in params}
        self._emit_stmts(prog.stmts, 1, lines, env_types, declared, fn_effects)
        lines.append("}")
        lines.append("")
        return lines

    def _var_cpp(self, vname: str, declared: Set[str]) -> str:
        if vname in declared:
            return vname
        return f"v_{vname}"

    def _emit_stmts(
        self,
        stmts: List[Any],
        indent: int,
        lines: List[str],
        env_types: Dict[str, str],
        declared: Set[str],
        fn_effects: List[str],
    ) -> None:
        pad = "    " * indent
        for st in stmts:
            if isinstance(st, AssignStmt):
                vt_hint = DSL_VALTYPE_MAP.get(st.val_type) if st.val_type else None
                expr_cpp = self._emit_expr(st.rhs, indent, lines, env_types, declared, fn_effects, vt_hint)
                vname = st.var
                if vname in declared:
                    lines.append(f"{pad}{vname} = {expr_cpp};")
                else:
                    cpp_t = DSL_TYPE_TO_CPP.get(st.val_type or "u32", "SsaId")
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
                cond_cpp = self._emit_cond(st.cond, env_types, declared)
                lines.append(f"{pad}if ({cond_cpp}) {{")
                self._emit_stmts(st.then_body, indent + 1, lines, dict(env_types), set(declared), fn_effects)
                if st.else_body is not None:
                    if len(st.else_body) == 1 and isinstance(st.else_body[0], IfStmt):
                        # Render clean else if
                        lines.append(f"{pad}}} else {self._emit_if_inline(st.else_body[0], indent, lines, env_types, declared, fn_effects)}")
                    else:
                        lines.append(f"{pad}}} else {{")
                        self._emit_stmts(st.else_body, indent + 1, lines, dict(env_types), set(declared), fn_effects)
                        lines.append(f"{pad}}}")
                else:
                    lines.append(f"{pad}}}")
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

    def _emit_if_inline(
        self,
        st: IfStmt,
        indent: int,
        lines: List[str],
        env_types: Dict[str, str],
        declared: Set[str],
        fn_effects: List[str],
    ) -> str:
        # Emits `if (...) { ... } [else ...]` returning the closing line
        pad = "    " * indent
        cond_cpp = self._emit_cond(st.cond, env_types, declared)
        # Note: caller already prefixed `} else `
        lines[-1] = lines[-1]  # placeholder; caller appends header
        # Actually easier: return header, emit body, handle tail
        # Let's avoid mutating lines[-1] and instead emit cleanly:
        raise NotImplementedError

    def _emit_cond(self, cond: Any, env_types: Dict[str, str], declared: Set[str]) -> str:
        if isinstance(cond, CondBinOp):
            return f"({self._emit_cond(cond.left, env_types, declared)} {cond.op} {self._emit_cond(cond.right, env_types, declared)})"
        if isinstance(cond, Cond):
            atom = cond.atom
            if isinstance(atom, VarRef):
                c = atom.name
            elif isinstance(atom, HostCall):
                c = self._emit_host_call(atom, env_types, declared)
            elif isinstance(atom, DollarCall):
                c = f"operands_host::{atom.name}(ins)"
            else:
                raise ValueError(f"Unsupported cond atom: {atom!r}")
            return f"!({c})" if cond.negated else c
        raise ValueError(f"Unsupported cond: {cond!r}")

    def _emit_arg(self, arg: Any, env_types: Dict[str, str], declared: Set[str]) -> str:
        if isinstance(arg, VarRef):
            return arg.name
        if isinstance(arg, Literal):
            return arg.raw
        if isinstance(arg, BareIdent):
            if arg.name in DSL_VALTYPE_MAP:
                return DSL_VALTYPE_MAP[arg.name]
            return arg.name
        if isinstance(arg, HostCall):
            return self._emit_host_call(arg, env_types, declared)
        if isinstance(arg, Cond):
            return self._emit_cond(arg, env_types, declared)
        raise ValueError(f"Unsupported DSL arg in valhall_operands.json: {arg!r}")

    def _emit_host_call(self, hc: HostCall, env_types: Dict[str, str], declared: Set[str]) -> str:
        args_cpp = [self._emit_arg(a, env_types, declared) for a in hc.args]
        if hc.name in self.entry_abi_names:
            return f"{hc.name}({', '.join(['sb', 'blk_id', 'ins'] + args_cpp)})"
        if hc.name not in self.helpers:
            raise ValueError(f"Unknown @helper '{hc.name}' in valhall_operands.json")
        h = self.helpers[hc.name]
        if h["kind"] == "inline_ops":
            return f"_helper_{hc.name}({', '.join(['sb', 'blk_id', 'ins'] + args_cpp)})"
        # kind == "host" -> operands_host::<name>(...)
        call_args: List[str] = []
        if h.get("takes_sb", False):
            call_args.extend(["sb", "blk_id"])
        if h.get("takes_ins", False):
            call_args.append("ins")
        call_args.extend(args_cpp)
        return f"operands_host::{hc.name}({', '.join(call_args)})"

    def _emit_expr(
        self,
        expr: Any,
        indent: int,
        lines: List[str],
        env_types: Dict[str, str],
        declared: Set[str],
        fn_effects: List[str],
        vt_hint: Optional[str],
    ) -> str:
        if isinstance(expr, (VarRef, Literal, BareIdent)):
            return self._emit_arg(expr, env_types, declared)
        if isinstance(expr, HostCall):
            return self._emit_host_call(expr, env_types, declared)
        if isinstance(expr, SsaCall):
            pad = "    " * indent
            nv = self.next_nv()
            arg_exprs = [self._emit_arg(a, env_types, declared) for a in expr.args]
            default_vt, op_effs = SSA_OP_DEFAULTS.get(expr.op, ("ValType::U32", []))
            vt = vt_hint or default_vt
            # Infer vector/signed ValType from first arg if needed
            if vt_hint is None and expr.op in ("VecConstruct2", "FClamp", "FMax", "FAbs", "FNeg"):
                if expr.args and isinstance(expr.args[0], VarRef):
                    at = env_types.get(expr.args[0].name)
                    if at in DSL_VALTYPE_MAP:
                        vt = DSL_VALTYPE_MAP[at]
            lines.append(f"{pad}SsaNode {nv}{{}};")
            lines.append(f"{pad}{nv}.op = tables::SsaOp::{expr.op};")
            lines.append(f"{pad}{nv}.type = {vt};")
            lines.append(f"{pad}{nv}.args = {{ {', '.join(arg_exprs)} }};")
            effs = list(dict.fromkeys(list(op_effs) + list(fn_effects)))
            for ef in effs:
                if ef in EFFECT_STMTS:
                    lines.append(f"{pad}{EFFECT_STMTS[ef].format(nv=nv)}")
            return f"sb.emit(blk_id, std::move({nv}))"
        raise ValueError(f"Unsupported DSL expr: {expr!r}")


# Override _emit_stmts cleanly to handle else-if without raising
def _patch_compiler() -> None:
    def _emit_if(
        self: OperandsDslCompiler,
        st: IfStmt,
        indent: int,
        lines: List[str],
        env_types: Dict[str, str],
        declared: Set[str],
        fn_effects: List[str],
        prefix_else: bool = False,
    ) -> None:
        pad = "    " * indent
        cond_cpp = self._emit_cond(st.cond, env_types, declared)
        if prefix_else:
            lines.append(f"{pad}}} else if ({cond_cpp}) {{")
        else:
            lines.append(f"{pad}if ({cond_cpp}) {{")
        self._emit_stmts(st.then_body, indent + 1, lines, dict(env_types), set(declared), fn_effects)
        if st.else_body is not None:
            if len(st.else_body) == 1 and isinstance(st.else_body[0], IfStmt):
                _emit_if(self, st.else_body[0], indent, lines, env_types, declared, fn_effects, prefix_else=True)
            else:
                lines.append(f"{pad}}} else {{")
                self._emit_stmts(st.else_body, indent + 1, lines, dict(env_types), set(declared), fn_effects)
                lines.append(f"{pad}}}")
        else:
            lines.append(f"{pad}}}")

    def _emit_stmts(
        self: OperandsDslCompiler,
        stmts: List[Any],
        indent: int,
        lines: List[str],
        env_types: Dict[str, str],
        declared: Set[str],
        fn_effects: List[str],
    ) -> None:
        pad = "    " * indent
        for st in stmts:
            if isinstance(st, AssignStmt):
                vt_hint = DSL_VALTYPE_MAP.get(st.val_type) if st.val_type else None
                expr_cpp = self._emit_expr(st.rhs, indent, lines, env_types, declared, fn_effects, vt_hint)
                vname = st.var
                if vname in declared:
                    lines.append(f"{pad}{vname} = {expr_cpp};")
                else:
                    cpp_t = DSL_TYPE_TO_CPP.get(st.val_type or "u32", "SsaId")
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
                _emit_if(self, st, indent, lines, env_types, declared, fn_effects, prefix_else=False)
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

    OperandsDslCompiler._emit_stmts = _emit_stmts  # type: ignore[method-assign]


_patch_compiler()


def generate_cpp(spec: Dict[str, Any]) -> str:
    if spec.get("schema_version") != 2:
        raise ValueError(f"Expected schema_version == 2, got {spec.get('schema_version')}")

    helpers: Dict[str, Dict[str, Any]] = spec.get("helpers", {})
    entry_abi: List[Dict[str, Any]] = spec.get("entry_abi", [])
    entry_names = {e["name"] for e in entry_abi}
    comp = OperandsDslCompiler(helpers, entry_names)

    lines: List[str] = [
        "// AUTO-GENERATED FILE — DO NOT EDIT.",
        "// Generated from specs/lowering/valhall_operands.json (InlineOps DSL) by tools/gen_valhall_operands.py",
        "",
    ]
    for inc in spec.get("includes", []):
        lines.append(f"#include {inc}")
    lines.append("")
    lines.append(f"namespace {spec['namespace']} {{")
    lines.append("")

    # 1. Emit inline_ops helpers as static functions
    for hname, hdef in helpers.items():
        if hdef["kind"] != "inline_ops":
            continue
        params = [
            {"type": "SsaBuilder&", "name": "sb"},
            {"type": "uint32_t", "name": "blk_id"},
            {"type": "const DecodedValhallInst&", "name": "ins"},
        ] + hdef.get("args", [])
        lines.extend(
            comp.compile_fn(
                f"_helper_{hname}",
                hdef["ret"],
                params,
                hdef["inline_ops"],
                is_static=True,
                fn_effects=hdef.get("effects", []),
            )
        )

    # 2. Emit entry_abi functions
    for entry in entry_abi:
        if entry["kind"] == "host":
            p_decl = ", ".join(f"{p['type']} {p['name']}" for p in entry["params"])
            p_call = ", ".join(p["name"] for p in entry["params"])
            lines.append(f"{entry['ret']} {entry['name']}({p_decl}) {{")
            lines.append(f"    return {entry['target']}({p_call});")
            lines.append("}")
            lines.append("")
        elif entry["kind"] == "inline_ops":
            lines.extend(
                comp.compile_fn(
                    entry["name"],
                    entry["ret"],
                    entry["params"],
                    entry["inline_ops"],
                    is_static=False,
                    fn_effects=entry.get("effects", []),
                )
            )
        else:
            raise ValueError(f"Unknown entry_abi kind '{entry['kind']}' for {entry['name']}")

    lines.append(f"}} // namespace {spec['namespace']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="specs/lowering/valhall_operands.json")
    ap.add_argument("--out", default="src/generated/valhall_operands_generated.cpp")
    ap.add_argument("--check", action="store_true", help="Verify generated output matches existing file")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    with open(args.spec, "r", encoding="utf-8") as f:
        spec = json.load(f)

    cpp_src = generate_cpp(spec)
    n_entry = len(spec.get("entry_abi", []))
    n_inline_helpers = sum(1 for h in spec.get("helpers", {}).values() if h["kind"] == "inline_ops")
    n_host_helpers = sum(1 for h in spec.get("helpers", {}).values() if h["kind"] == "host")

    if args.check:
        if not os.path.isfile(args.out):
            print(f"ERROR: {args.out} does not exist", file=sys.stderr)
            return 1
        with open(args.out, "r", encoding="utf-8") as f:
            existing = f.read()
        if existing != cpp_src:
            print(f"ERROR: {args.out} is out of date relative to {args.spec}", file=sys.stderr)
            return 1
        if args.verbose:
            print(
                f"[gen_valhall_operands] {args.out} is up to date "
                f"({n_entry} entry_abi, {n_inline_helpers} inline_ops helpers, {n_host_helpers} @host helpers)"
            )
        return 0

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(cpp_src)
    if args.verbose:
        print(
            f"[gen_valhall_operands] Generated {args.out} "
            f"({n_entry} entry_abi, {n_inline_helpers} inline_ops helpers, {n_host_helpers} @host helpers)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Loader for specs/lowering/valhall_operands.json (InlineOps DSL) -> CppFunctionDef ASTs
=======================================================================================
Parses the InlineOps DSL strings in specs/lowering/valhall_operands.json using
tools/dsl_parser.py and lowers them into CppFunctionDef ASTs for CppSymbolicEvaluator.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Set

_TOOLS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from dsl_parser import (
    AssignStmt,
    BareIdent,
    Cond,
    CondBinOp,
    DollarCall,
    ExprStmt,
    ForStmt,
    HostCall,
    IfStmt,
    Literal,
    ReturnStmt,
    SsaCall,
    VarRef,
    parse_inline_ops,
)

from tools.verify.cpp_handler_extractor import (
    CppAssign,
    CppBinary,
    CppCall,
    CppExprStmt,
    CppFor,
    CppFunctionDef,
    CppIdent,
    CppIf,
    CppInitList,
    CppLiteral,
    CppMember,
    CppParam,
    CppReturn,
    CppScopedIdent,
    CppUnary,
    CppVarDecl,
)

DSL_VALTYPE_MAP = {
    "u32": "U32",
    "s32": "S32",
    "f32": "F32",
    "u64": "U64",
    "s64": "S64",
    "v2f32": "V2F32",
    "v2u32": "V2U32",
    "v2s32": "V2S32",
    "v4f32": "V4F32",
    "v4u32": "V4U32",
    "v4s32": "V4S32",
    "bool": "Bool",
}

SSA_OP_DEFAULTS = {
    "IAdd64": "U64",
    "SConvert32To64": "S64",
    "UConvert32To64": "U64",
    "BfeS": "S32",
    "BfeU": "U32",
    "Shl": "U32",
    "BitOr": "U32",
    "IMul": "U32",
    "BitNot": "U32",
    "SAbs": "S32",
    "SNeg": "S32",
    "UnpackHalf2x16": "V2F32",
    "PackHalf2x16": "U32",
    "FAbs": "F32",
    "FNeg": "F32",
    "FClamp": "F32",
    "FMax": "F32",
    "VecConstruct2": "V2F32",
    "VecConstruct4": "V4U32",
}


class _DslToCppAst:
    def __init__(self, helpers: Dict[str, Dict[str, Any]], entry_names: Set[str]) -> None:
        self.helpers = helpers
        self.entry_names = entry_names
        self._node_idx = 0

    def convert_fn(
        self,
        name: str,
        ret_type: str,
        params: List[Dict[str, str]],
        inline_ops: List[str],
        source_file: str,
    ) -> CppFunctionDef:
        prog = parse_inline_ops(inline_ops)
        env_types: Dict[str, str] = {p["name"]: p["type"] for p in params}
        declared: Set[str] = {p["name"] for p in params}
        body = self._stmts(prog.stmts, env_types, declared)
        return CppFunctionDef(
            name=name,
            namespace="",
            ret_type=ret_type,
            params=[CppParam(p["type"], p["name"]) for p in params],
            body=body,
            source_file=source_file,
            start_line=0,
            end_line=0,
        )

    def _stmts(self, stmts: List[Any], env_types: Dict[str, str], declared: Set[str]) -> List[Any]:
        out: List[Any] = []
        for st in stmts:
            if isinstance(st, AssignStmt):
                vt_hint = DSL_VALTYPE_MAP.get(st.val_type) if st.val_type else None
                rhs = self._expr(st.rhs, out, env_types, vt_hint)
                if st.var in declared:
                    out.append(CppAssign(CppIdent(st.var), "=", rhs))
                else:
                    if st.val_type:
                        env_types[st.var] = st.val_type
                    declared.add(st.var)
                    out.append(CppVarDecl(st.val_type or "SsaId", st.var, rhs))
            elif isinstance(st, ExprStmt):
                out.append(CppExprStmt(self._expr(st.expr, out, env_types, None)))
            elif isinstance(st, ReturnStmt):
                val = self._expr(st.value, out, env_types, None) if st.value is not None else None
                out.append(CppReturn(val))
            elif isinstance(st, IfStmt):
                cond = self._cond(st.cond)
                then_b = self._stmts(st.then_body, dict(env_types), set(declared))
                else_b = self._stmts(st.else_body, dict(env_types), set(declared)) if st.else_body is not None else None
                out.append(CppIf(cond, then_b, else_b))
            elif isinstance(st, ForStmt):
                iv = st.var
                inner_env = dict(env_types)
                inner_decl = set(declared)
                inner_env[iv] = "uint32_t"
                inner_decl.add(iv)
                out.append(
                    CppFor(
                        init=CppVarDecl("uint32_t", iv, CppLiteral("int", 0)),
                        cond=CppBinary("<", CppIdent(iv), self._arg(st.bound)),
                        step=CppUnary("++", CppIdent(iv), postfix=False),
                        body=self._stmts(st.body, inner_env, inner_decl),
                    )
                )
            else:
                raise ValueError(f"Unsupported DSL stmt in valhall_operands.json: {st!r}")
        return out

    def _cond(self, c: Any) -> Any:
        if isinstance(c, CondBinOp):
            return CppBinary(c.op, self._cond(c.left), self._cond(c.right))
        if isinstance(c, Cond):
            atom = c.atom
            if isinstance(atom, VarRef):
                inner: Any = CppIdent(atom.name)
            elif isinstance(atom, HostCall):
                inner = self._host_call(atom)
            elif isinstance(atom, DollarCall):
                inner = CppCall(CppScopedIdent(["operands_host", atom.name]), [CppIdent("ins")])
            else:
                raise ValueError(f"Unsupported cond atom: {atom!r}")
            return CppUnary("!", inner, postfix=False) if c.negated else inner
        raise ValueError(f"Unsupported cond: {c!r}")

    def _arg(self, a: Any) -> Any:
        if isinstance(a, VarRef):
            return CppIdent(a.name)
        if isinstance(a, Literal):
            if a.kind == "int":
                raw = a.raw.rstrip("uUlL")
                return CppLiteral("int", int(raw, 16) if raw.startswith(("0x", "0X")) else int(raw))
            if a.kind == "float":
                return CppLiteral("float", float(a.raw.rstrip("fF")))
            if a.kind == "string":
                return CppLiteral("string", a.raw[1:-1])
            if a.kind == "bool":
                return CppLiteral("bool", a.raw == "true")
        if isinstance(a, BareIdent):
            if a.name in DSL_VALTYPE_MAP:
                return CppScopedIdent(["ValType", DSL_VALTYPE_MAP[a.name]])
            return CppIdent(a.name)
        if isinstance(a, HostCall):
            return self._host_call(a)
        if isinstance(a, Cond):
            return self._cond(a)
        raise ValueError(f"Unsupported DSL arg: {a!r}")

    def _host_call(self, hc: HostCall) -> CppCall:
        args_ast = [self._arg(a) for a in hc.args]
        if hc.name in self.entry_names:
            return CppCall(CppIdent(hc.name), [CppIdent("sb"), CppIdent("blk_id"), CppIdent("ins")] + args_ast)
        if hc.name not in self.helpers:
            raise ValueError(f"Unknown @helper '{hc.name}' in valhall_operands.json")
        h = self.helpers[hc.name]
        if h["kind"] == "inline_ops":
            return CppCall(CppIdent(f"_helper_{hc.name}"), [CppIdent("sb"), CppIdent("blk_id"), CppIdent("ins")] + args_ast)
        call_args: List[Any] = []
        if h.get("takes_sb", False):
            call_args.extend([CppIdent("sb"), CppIdent("blk_id")])
        if h.get("takes_ins", False):
            call_args.append(CppIdent("ins"))
        call_args.extend(args_ast)
        return CppCall(CppScopedIdent(["operands_host", hc.name]), call_args)

    def _expr(self, expr: Any, out: List[Any], env_types: Dict[str, str], vt_hint: Optional[str]) -> Any:
        if isinstance(expr, (VarRef, Literal, BareIdent)):
            return self._arg(expr)
        if isinstance(expr, HostCall):
            return self._host_call(expr)
        if isinstance(expr, SsaCall):
            self._node_idx += 1
            nv = f"_n{self._node_idx}"
            vt = vt_hint or SSA_OP_DEFAULTS.get(expr.op, "U32")
            if vt_hint is None and expr.op in ("VecConstruct2", "FClamp", "FMax", "FAbs", "FNeg"):
                if expr.args and isinstance(expr.args[0], VarRef):
                    at = env_types.get(expr.args[0].name)
                    if at in DSL_VALTYPE_MAP:
                        vt = DSL_VALTYPE_MAP[at]
            out.append(CppVarDecl("SsaNode", nv, is_struct_zero=True))
            out.append(CppAssign(CppMember(CppIdent(nv), "op"), "=", CppScopedIdent(["tables", "SsaOp", expr.op])))
            out.append(CppAssign(CppMember(CppIdent(nv), "type"), "=", CppScopedIdent(["ValType", vt])))
            out.append(CppAssign(CppMember(CppIdent(nv), "args"), "=", CppInitList([self._arg(a) for a in expr.args])))
            return CppCall(
                CppMember(CppIdent("sb"), "emit"),
                [CppIdent("blk_id"), CppCall(CppScopedIdent(["std", "move"]), [CppIdent(nv)])],
            )
        raise ValueError(f"Unsupported DSL expr: {expr!r}")


def load_operands_spec_json(spec_path: str) -> Dict[str, CppFunctionDef]:
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    if spec.get("schema_version") != 2:
        raise ValueError(f"Expected valhall_operands.json schema_version == 2, got {spec.get('schema_version')}")

    source_file = os.path.basename(spec_path)
    helpers: Dict[str, Dict[str, Any]] = spec.get("helpers", {})
    entry_abi: List[Dict[str, Any]] = spec.get("entry_abi", [])
    entry_names = {e["name"] for e in entry_abi}
    conv = _DslToCppAst(helpers, entry_names)
    funcs: Dict[str, CppFunctionDef] = {}

    for hname, hdef in helpers.items():
        if hdef["kind"] != "inline_ops":
            continue
        params = [
            {"type": "SsaBuilder&", "name": "sb"},
            {"type": "uint32_t", "name": "blk_id"},
            {"type": "const DecodedValhallInst&", "name": "ins"},
        ] + hdef.get("args", [])
        funcs[f"_helper_{hname}"] = conv.convert_fn(
            f"_helper_{hname}",
            hdef["ret"],
            params,
            hdef["inline_ops"],
            source_file,
        )

    for entry in entry_abi:
        if entry["kind"] == "host":
            p_names = [CppIdent(p["name"]) for p in entry["params"]]
            target_parts = entry["target"].split("::")
            funcs[entry["name"]] = CppFunctionDef(
                name=entry["name"],
                namespace="",
                ret_type=entry["ret"],
                params=[CppParam(p["type"], p["name"]) for p in entry["params"]],
                body=[CppReturn(CppCall(CppScopedIdent(target_parts), p_names))],
                source_file=source_file,
                start_line=0,
                end_line=0,
            )
        elif entry["kind"] == "inline_ops":
            funcs[entry["name"]] = conv.convert_fn(
                entry["name"],
                entry["ret"],
                entry["params"],
                entry["inline_ops"],
                source_file,
            )
        else:
            raise ValueError(f"Unknown entry_abi kind '{entry['kind']}' for {entry['name']}")

    return funcs

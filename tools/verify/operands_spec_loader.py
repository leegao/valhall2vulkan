"""
Loader for specs/lowering/valhall_lowering.json (operands + inline_ops host_functions) -> CppFunctionDef ASTs
==============================================================================================================
Parses the InlineOps DSL strings in specs/lowering/valhall_lowering.json["operands"] and
inline_ops entries in specs/lowering/valhall_lowering.json["host_functions"] using
tools/dsl_parser.py and lowers them into CppFunctionDef ASTs for CppSymbolicEvaluator.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

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
    SrcRef,
    SsaCall,
    VarRef,
    parse_inline_ops,
)
from valhall_lowering_dsl import parse_param_list

from tools.verify.clang_ast_evaluator import (
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

PASS1_CTX_PARAMS = [
    ("Pass1Context&", "ctx"),
    ("uint32_t", "blk"),
    ("const DecodedValhallInst&", "ins"),
    ("const tables::LoweringRuleSpec*", "rule"),
]
PASS1_CTX_ARGS = [CppIdent("ctx"), CppIdent("blk"), CppIdent("ins"), CppIdent("rule")]


class _DslToCppAst:
    def __init__(self, host_functions: Dict[str, Dict[str, Any]], operand_names: Set[str]) -> None:
        self.host_functions = host_functions
        self.operand_names = operand_names
        self._node_idx = 0

    def convert_fn(
        self,
        name: str,
        ret_type: str,
        params_list: List[str],
        inline_ops: List[str],
        source_file: str,
        namespace: str = "",
        prepend_ctx: bool = True,
    ) -> CppFunctionDef:
        prog = parse_inline_ops(inline_ops)
        parsed = parse_param_list(params_list)
        all_params = PASS1_CTX_PARAMS + parsed if prepend_ctx else parsed
        env_types: Dict[str, str] = {n: t for t, n in all_params}
        declared: Set[str] = {n for _, n in all_params}
        body = self._stmts(prog.stmts, env_types, declared)
        return CppFunctionDef(
            name=name,
            namespace=namespace,
            ret_type=ret_type,
            params=[CppParam(t, n) for t, n in all_params],
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
                raise ValueError(f"Unsupported DSL stmt in valhall_lowering.json operands: {st!r}")
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
                inner = CppCall(CppScopedIdent(["abi", atom.name]), list(PASS1_CTX_ARGS))
            else:
                raise ValueError(f"Unsupported cond atom: {atom!r}")
            return CppUnary("!", inner, postfix=False) if c.negated else inner
        raise ValueError(f"Unsupported cond: {c!r}")

    def _arg(self, a: Any) -> Any:
        if isinstance(a, SrcRef):
            return CppIdent(f"arg{a.index}")
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
        if isinstance(a, DollarCall):
            return CppCall(CppScopedIdent(["abi", a.name]), list(PASS1_CTX_ARGS))
        if isinstance(a, Cond):
            return self._cond(a)
        raise ValueError(f"Unsupported DSL arg: {a!r}")

    def _host_call(self, hc: HostCall) -> CppCall:
        args_ast = [self._arg(a) for a in hc.args]
        if hc.name in self.operand_names:
            return CppCall(CppIdent(hc.name), list(PASS1_CTX_ARGS) + args_ast)
        if hc.name not in self.host_functions:
            raise ValueError(f"Unknown @host function '{hc.name}' in valhall_lowering.json")
        return CppCall(CppScopedIdent(["abi", hc.name]), list(PASS1_CTX_ARGS) + args_ast)

    def _expr(self, expr: Any, out: List[Any], env_types: Dict[str, str], vt_hint: Optional[str]) -> Any:
        if isinstance(expr, (SrcRef, VarRef, Literal, BareIdent, DollarCall)):
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
                CppMember(CppMember(CppIdent("ctx"), "sb"), "emit"),
                [CppIdent("blk"), CppCall(CppScopedIdent(["std", "move"]), [CppIdent(nv)])],
            )
        raise ValueError(f"Unsupported DSL expr: {expr!r}")


def load_operands_spec_json(spec_path: str) -> Tuple[Dict[str, CppFunctionDef], Dict[str, CppFunctionDef]]:
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    source_file = os.path.basename(spec_path)
    host_functions: Dict[str, Dict[str, Any]] = spec.get("host_functions", {})
    operands: List[Dict[str, Any]] = spec.get("operands", [])
    operand_names = {e["name"] for e in operands}
    conv = _DslToCppAst(host_functions, operand_names)
    operands_funcs: Dict[str, CppFunctionDef] = {}
    inline_abi_funcs: Dict[str, CppFunctionDef] = {}

    for hname, hdef in host_functions.items():
        if hdef.get("kind") != "inline_ops":
            continue
        inline_abi_funcs[hname] = conv.convert_fn(
            hname,
            hdef["ret"],
            hdef.get("args", []),
            hdef["inline_ops"],
            source_file,
            namespace="abi",
            prepend_ctx=True,
        )

    for entry in operands:
        if entry["kind"] == "host":
            parsed = parse_param_list(entry.get("params", []))
            all_params = PASS1_CTX_PARAMS + parsed
            p_names = [CppIdent(n) for _, n in all_params]
            target_parts = entry["target"].split("::")
            operands_funcs[entry["name"]] = CppFunctionDef(
                name=entry["name"],
                namespace="",
                ret_type=entry["ret"],
                params=[CppParam(t, n) for t, n in all_params],
                body=[CppReturn(CppCall(CppScopedIdent(target_parts), p_names))],
                source_file=source_file,
                start_line=0,
                end_line=0,
            )
        elif entry["kind"] == "inline_ops":
            operands_funcs[entry["name"]] = conv.convert_fn(
                entry["name"],
                entry["ret"],
                entry.get("params", []),
                entry["inline_ops"],
                source_file,
                namespace="",
                prepend_ctx=True,
            )
        else:
            raise ValueError(f"Unknown operand kind '{entry['kind']}' for {entry['name']}")

    return operands_funcs, inline_abi_funcs

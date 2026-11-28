"""
Clang AST JSON Loader & Symbolic Evaluator for Valhall Host Handlers
====================================================================
Replaces the ad-hoc Python C++17 lexer/parser with a pure consumer of the
Clang-21 AST JSON (`tools/verify/generated/valhall_host_handlers.json`)
emitted at build time by `clang_ast_dumper`, unified with the declarative
InlineOps DSL ASTs from `specs/lowering/valhall_lowering.json`.

Provides:
  - Clean AST dataclasses (`CppFunctionDef`, `CppVarDecl`, `CppIf`, etc.)
  - `load_clang_ast_json()` to load `valhall_host_handlers.json`
  - `CppHandlerRegistry` & `CppSymbolicEvaluator` to symbolically execute
    host handlers and operand combinators into `SymbolicSsaModule`.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import struct
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# AST Node Dataclasses (shared by Clang AST JSON loader & operands_spec_loader)
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class CppParam:
    type_str: str
    name: str


@dataclasses.dataclass
class CppFunctionDef:
    name: str
    namespace: str  # "", "abi", "Pass1Context"
    ret_type: str
    params: List[CppParam]
    body: List[Any]
    source_file: str = ""
    start_line: int = 0
    end_line: int = 0


@dataclasses.dataclass
class CppVarDecl:
    type_str: str
    name: str
    init: Optional[Any] = None
    is_struct_zero: bool = False


@dataclasses.dataclass
class CppMultiVarDecl:
    decls: List[CppVarDecl]


@dataclasses.dataclass
class CppReturn:
    expr: Optional[Any]


@dataclasses.dataclass
class CppIf:
    cond: Any
    then_body: List[Any]
    else_body: Optional[List[Any]] = None
    init: Optional[Any] = None


@dataclasses.dataclass
class CppFor:
    init: Optional[Any]
    cond: Optional[Any]
    step: Optional[Any]
    body: List[Any]


@dataclasses.dataclass
class CppSwitch:
    expr: Any
    cases: List[Tuple[Optional[Any], List[Any]]]


@dataclasses.dataclass
class CppBreak:
    pass


@dataclasses.dataclass
class CppContinue:
    pass


@dataclasses.dataclass
class CppBlock:
    stmts: List[Any]


@dataclasses.dataclass
class CppExprStmt:
    expr: Any


@dataclasses.dataclass
class CppAssign:
    target: Any
    op: str
    value: Any


@dataclasses.dataclass
class CppBinary:
    op: str
    left: Any
    right: Any


@dataclasses.dataclass
class CppUnary:
    op: str
    operand: Any
    postfix: bool = False


@dataclasses.dataclass
class CppTernary:
    cond: Any
    then_expr: Any
    else_expr: Any


@dataclasses.dataclass
class CppCall:
    callee: Any
    args: List[Any]
    qualified_callee: str = ""


@dataclasses.dataclass
class CppMember:
    obj: Any
    member: str
    is_arrow: bool = False


@dataclasses.dataclass
class CppIndex:
    obj: Any
    index: Any


@dataclasses.dataclass
class CppIdent:
    name: str


@dataclasses.dataclass
class CppScopedIdent:
    parts: List[str]


@dataclasses.dataclass
class CppLiteral:
    kind: str  # "int", "float", "string", "bool", "char"
    value: Any


@dataclasses.dataclass
class CppInitList:
    items: List[Any]


@dataclasses.dataclass
class CppLambda:
    params: List[CppParam]
    body: List[Any]


# ---------------------------------------------------------------------------
# Clang AST JSON -> CppFunctionDef Converter
# ---------------------------------------------------------------------------
def _convert_expr(node: Optional[Dict[str, Any]]) -> Any:
    if node is None:
        return None
    kind = node.get("kind")
    if kind == "literal":
        lk = node.get("lit_kind", "int")
        if lk == "int":
            return CppLiteral("int", int(node.get("int_value", 0)))
        if lk == "float":
            return CppLiteral("float", float(node.get("float_value", 0.0)))
        if lk == "bool":
            return CppLiteral("bool", bool(node.get("bool_value", False)))
        if lk == "string":
            return CppLiteral("string", str(node.get("value", "")))
        if lk == "char":
            return CppLiteral("char", str(node.get("value", "")))
        return CppLiteral(lk, node.get("value"))
    if kind == "ident":
        return CppIdent(node["name"])
    if kind == "scoped_ident":
        return CppScopedIdent(list(node.get("parts", [])))
    if kind == "member":
        return CppMember(
            obj=_convert_expr(node["obj"]),
            member=node["member"],
            is_arrow=bool(node.get("is_arrow", False)),
        )
    if kind == "index":
        return CppIndex(
            obj=_convert_expr(node["obj"]),
            index=_convert_expr(node["index"]),
        )
    if kind == "call":
        return CppCall(
            callee=_convert_expr(node["callee"]),
            args=[_convert_expr(a) for a in node.get("args", [])],
            qualified_callee=node.get("qualified_callee", ""),
        )
    if kind == "binary":
        return CppBinary(
            op=node["op"],
            left=_convert_expr(node["left"]),
            right=_convert_expr(node["right"]),
        )
    if kind == "unary":
        return CppUnary(
            op=node["op"],
            operand=_convert_expr(node["operand"]),
            postfix=bool(node.get("postfix", False)),
        )
    if kind == "assign":
        return CppAssign(
            target=_convert_expr(node["target"]),
            op=node["op"],
            value=_convert_expr(node["value"]),
        )
    if kind == "ternary":
        return CppTernary(
            cond=_convert_expr(node["cond"]),
            then_expr=_convert_expr(node["then_expr"]),
            else_expr=_convert_expr(node["else_expr"]),
        )
    if kind == "cast":
        return CppCall(
            callee=CppIdent("static_cast"),
            args=[_convert_expr(node["expr"])],
            qualified_callee="",
        )
    if kind == "sizeof":
        return CppCall(callee=CppIdent("sizeof"), args=[], qualified_callee="")
    if kind == "lambda":
        return CppLambda(
            params=[CppParam(pr.get("type", ""), pr["name"]) for pr in node.get("params", [])],
            body=[_convert_stmt(s) for s in node.get("body", [])],
        )
    if kind == "init_list":
        items = node.get("items", [])
        ctype = node.get("canonical_type", "")
        if "basic_string" in ctype:
            if len(items) == 1:
                return _convert_expr(items[0])
            if len(items) == 0:
                return CppLiteral("string", "")
        if "vector" in ctype and len(items) == 1 and items[0].get("kind") == "init_list":
            return _convert_expr(items[0])
        return CppInitList([_convert_expr(x) for x in items])
    raise ValueError(f"Unsupported Clang AST JSON expr kind: {kind!r} ({node})")


def _convert_stmt(node: Dict[str, Any]) -> Any:
    kind = node.get("kind")
    if kind == "var_decl":
        tstr = node.get("type", "")
        is_zero = bool(node.get("is_struct_zero", False)) or ("SsaNode" in tstr)
        init = None if is_zero else _convert_expr(node.get("init"))
        return CppVarDecl(
            type_str=tstr,
            name=node["name"],
            init=init,
            is_struct_zero=is_zero,
        )
    if kind == "multi_var_decl":
        return CppMultiVarDecl([_convert_stmt(d) for d in node.get("decls", [])])
    if kind == "return":
        return CppReturn(_convert_expr(node.get("expr")))
    if kind == "if":
        else_b = node.get("else_body")
        init_node = node.get("init")
        return CppIf(
            cond=_convert_expr(node["cond"]),
            then_body=[_convert_stmt(s) for s in node.get("then_body", [])],
            else_body=[_convert_stmt(s) for s in else_b] if else_b is not None else None,
            init=_convert_stmt(init_node) if init_node is not None else None,
        )
    if kind == "while":
        return CppFor(
            init=None,
            cond=_convert_expr(node.get("cond")),
            step=None,
            body=[_convert_stmt(s) for s in node.get("body", [])],
        )
    if kind == "for":
        init_node = node.get("init")
        init = _convert_stmt(init_node) if init_node is not None else None
        return CppFor(
            init=init,
            cond=_convert_expr(node.get("cond")),
            step=_convert_expr(node.get("step")),
            body=[_convert_stmt(s) for s in node.get("body", [])],
        )
    if kind == "range_for":
        var_info = node["var"]
        rng = _convert_expr(node["range"])
        return CppFor(
            init=CppVarDecl(var_info.get("type", ""), var_info["name"], rng),
            cond=None,
            step=None,
            body=[_convert_stmt(s) for s in node.get("body", [])],
        )
    if kind == "assign":
        return CppAssign(
            target=_convert_expr(node["target"]),
            op=node["op"],
            value=_convert_expr(node["value"]),
        )
    if kind == "expr_stmt":
        return CppExprStmt(_convert_expr(node["expr"]))
    if kind == "block":
        return CppBlock([_convert_stmt(s) for s in node.get("stmts", [])])
    if kind == "break":
        return CppBreak()
    if kind == "continue":
        return CppContinue()
    if kind == "null":
        return CppBlock([])
    if kind == "switch":
        cases = []
        for c in node.get("cases", []):
            cval_node = c.get("value")
            cval = _convert_expr(cval_node) if cval_node is not None else None
            cbody = [_convert_stmt(s) for s in c.get("body", [])]
            cases.append((cval, cbody))
        return CppSwitch(expr=_convert_expr(node["expr"]), cases=cases)
    raise ValueError(f"Unsupported Clang AST JSON stmt kind: {kind!r} ({node})")


def load_clang_ast_metadata(ast_json_path: str) -> Tuple[List[CppFunctionDef], Dict[str, Dict[str, str]]]:
    """Load functions and [[verifier::z3(...)]] attributes from clang_ast_dumper's JSON output."""
    with open(ast_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported Clang AST JSON schema_version: {data.get('schema_version')!r}")
    if data.get("unhandled_count", 0) != 0:
        raise ValueError(
            f"Clang AST JSON {ast_json_path} has unhandled_count={data.get('unhandled_count')}; "
            "rebuild with clang_ast_dumper --strict"
        )
    out: List[CppFunctionDef] = []
    for fn in data.get("functions", []):
        short_ns = fn.get("short_namespace", "")
        ns = short_ns if short_ns in ("abi", "Pass1Context") else ""
        params = [CppParam(p.get("type", ""), p.get("name", "")) for p in fn.get("params", [])]
        body = [_convert_stmt(s) for s in fn.get("body", [])]
        out.append(
            CppFunctionDef(
                name=fn["name"],
                namespace=ns,
                ret_type=fn.get("ret_type", "void"),
                params=params,
                body=body,
                source_file=fn.get("source_file", ""),
                start_line=int(fn.get("start_line", 0)),
                end_line=int(fn.get("end_line", 0)),
            )
        )
    vattrs: Dict[str, Dict[str, str]] = {}
    for qname, info in data.get("verifier_attrs", {}).items():
        if isinstance(info, dict) and "expr" in info:
            vattrs[qname] = {
                "receiver": str(info.get("receiver", "")),
                "expr": str(info["expr"]),
            }
    return out, vattrs


def load_clang_ast_json(ast_json_path: str) -> List[CppFunctionDef]:
    """Load functions from clang_ast_dumper's JSON output."""
    fns, _ = load_clang_ast_metadata(ast_json_path)
    return fns


# ---------------------------------------------------------------------------
# Symbolic SSA-IR & Instruction Lowering Context
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class SymbolicSsaNode:
    id: int
    op: str
    val_type: str
    args: List[Any] = dataclasses.field(default_factory=list)
    imm_u32: int = 0
    imm_u32_2: int = 0
    imm_u64: int = 0
    aux_bool: bool = False
    aux_bool2: bool = False
    aux_str: str = ""
    is_barrier_or_cf: bool = False
    is_side_effect: bool = False
    effects: Tuple[str, ...] = ()

    def __str__(self) -> str:
        args_s = ", ".join(f"%{a}" if isinstance(a, int) else repr(a) for a in self.args)
        extras = []
        if self.imm_u32:
            extras.append(f"imm={self.imm_u32}")
        if self.imm_u32_2:
            extras.append(f"imm2={self.imm_u32_2}")
        if self.imm_u64:
            extras.append(f"imm64=0x{self.imm_u64:x}")
        if self.aux_bool:
            extras.append("aux=true")
        if self.aux_str:
            extras.append(f"str={self.aux_str!r}")
        ex_s = f" [{', '.join(extras)}]" if extras else ""
        return f"%{self.id}:{self.val_type} = {self.op}({args_s}){ex_s}"


class SymbolicSsaModule:
    """Symbolic Valhall SSA-IR emitted by evaluating lowered C++ handlers."""

    def __init__(self) -> None:
        self.nodes: List[SymbolicSsaNode] = [SymbolicSsaNode(0, "Nop", "Void")]
        self.reg_state: Dict[int, int] = {}
        self.reg_writes: Dict[int, int] = {}
        self.token_in: int = self._alloc_sym_node("TokenIn", "Token")
        self.token_out: int = self.token_in
        self.branch_cond: Optional[int] = None
        self.uses_sideband: bool = False

    def _alloc_sym_node(self, op: str, val_type: str, **kwargs: Any) -> int:
        nid = len(self.nodes)
        self.nodes.append(SymbolicSsaNode(id=nid, op=op, val_type=val_type, **kwargs))
        return nid

    def emit(self, node: SymbolicSsaNode) -> int:
        nid = len(self.nodes)
        node.id = nid
        self.nodes.append(node)
        return nid

    def current_reg(self, reg: int) -> int:
        reg &= 63
        if reg not in self.reg_state:
            self.reg_state[reg] = self._alloc_sym_node("InputReg", "U32", imm_u32=reg)
        return self.reg_state[reg]

    def set_reg(self, reg: int, val_id: int) -> None:
        reg &= 63
        self.reg_state[reg] = val_id
        self.reg_writes[reg] = val_id

    def current_token(self) -> int:
        return self.token_out

    def set_token(self, tok: int) -> None:
        self.token_out = tok

    def emit_const_u32(self, v: int) -> int:
        return self.emit(SymbolicSsaNode(0, "ConstU32", "U32", imm_u32=int(v) & 0xFFFFFFFF))

    def emit_const_s32(self, v: int) -> int:
        return self.emit(SymbolicSsaNode(0, "ConstS32", "S32", imm_u32=int(v) & 0xFFFFFFFF))

    def emit_const_f32_bits(self, bits: int) -> int:
        return self.emit(SymbolicSsaNode(0, "ConstF32", "F32", imm_u32=int(bits) & 0xFFFFFFFF))

    def emit_const_f32(self, v: float) -> int:
        bits = struct.unpack("<I", struct.pack("<f", float(v)))[0]
        return self.emit_const_f32_bits(bits)

    def set_branch_cond(self, cond: int) -> None:
        self.branch_cond = cond

    def emit_const_u64(self, v: int) -> int:
        return self.emit(SymbolicSsaNode(0, "ConstU64", "U64", imm_u64=int(v) & 0xFFFFFFFFFFFFFFFF))

    def emit_const_bool(self, v: bool) -> int:
        return self.emit(SymbolicSsaNode(0, "ConstBool", "Bool", aux_bool=bool(v)))

    def emit_bitcast(self, vt: str, val: int) -> int:
        return self.emit(SymbolicSsaNode(0, "Bitcast", vt, args=[val]))


@dataclasses.dataclass
class SourceOperandDesc:
    kind: str = "Register"  # Register, FauUniform, SmallConstant, FauSpecial
    reg_num: int = 0
    imm_bits: int = 0
    swizzle_mod: str = ""
    abs: bool = False
    neg: bool = False
    not_mod: bool = False
    word_select: bool = False
    is_64bit: bool = False
    symbolic_name: str = ""
    fau_index64: int = 0


@dataclasses.dataclass
class InstLoweringContext:
    mnemonic: str
    variant: str = ""
    canonical_mnemonic: Optional[str] = None
    canonical_variant: Optional[str] = None
    srcs: List[SourceOperandDesc] = dataclasses.field(default_factory=list)
    modifiers: Dict[str, str] = dataclasses.field(default_factory=dict)
    fields: Dict[str, int] = dataclasses.field(default_factory=dict)
    has_dest: bool = False
    dest_reg: int = 0
    dest_mod: str = ""
    dest_is_64bit: bool = False
    has_imm: bool = False
    imm_signed: int = 0
    imm_unsigned: int = 0
    pc: int = 0
    has_branch_offset: bool = False
    branch_target_pc: int = 0
    sr_read_present: bool = False
    sr_read_base: int = 0
    sr_read_count: int = 0
    sr_write_present: bool = False
    sr_write_base: int = 0
    sr_write_count: int = 0
    stage: str = "Compute"
    arch: int = 11
    rule_effects: Tuple[str, ...] = ()
    rule_op_class: str = ""
    fau_pka_buf_names: Dict[int, str] = dataclasses.field(default_factory=dict)
    frag_varying_addr_to_loc: Dict[int, int] = dataclasses.field(default_factory=dict)
    vert_sout_attr_names: Dict[int, str] = dataclasses.field(default_factory=dict)
    fau_tex_handle_indices: Dict[int, int] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.canonical_mnemonic is None:
            self.canonical_mnemonic = self.mnemonic
        if self.canonical_variant is None:
            self.canonical_variant = self.variant


# ---------------------------------------------------------------------------
# Symbolic Evaluator for Clang AST JSON + InlineOps ASTs
# ---------------------------------------------------------------------------
class _ReturnSignal(Exception):
    def __init__(self, value: Any) -> None:
        self.value = value


class _BreakSignal(Exception):
    pass


class _ContinueSignal(Exception):
    pass


class _DictRef:
    def __init__(self, d: dict, key: Any) -> None:
        self.d = d
        self.key = key

    @property
    def found(self) -> bool:
        return self.key in self.d

    @property
    def second(self) -> Any:
        return self.d[self.key]


class _PointerOut:
    def __init__(self, env: Dict[str, Any], name: str) -> None:
        self.env = env
        self.name = name

    def set(self, val: Any) -> None:
        self.env[self.name] = val


class CppHandlerRegistry:
    """Holds ASTs from valhall_lowering.json (operands + inline_ops) and Clang AST JSON."""

    def __init__(
        self,
        operands_path: str,
        handlers_ast_path: str,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.log = logger or logging.getLogger("clang_ast_evaluator")
        self.operands_funcs: Dict[str, CppFunctionDef] = {}
        self.abi_funcs: Dict[str, CppFunctionDef] = {}
        self.ctx_funcs: Dict[str, CppFunctionDef] = {}
        self.global_handlers: Dict[str, CppFunctionDef] = {}

        from tools.verify.operands_spec_loader import load_operands_spec_json
        op_funcs, inline_abi_funcs = load_operands_spec_json(operands_path)
        self.operands_funcs.update(op_funcs)
        self.abi_funcs.update(inline_abi_funcs)
        self.log.debug(
            "Loaded %d operand functions and %d inline_ops ABI functions from %s",
            len(op_funcs),
            len(inline_abi_funcs),
            os.path.basename(operands_path),
        )

        fns, vattrs = load_clang_ast_metadata(handlers_ast_path)
        self.verifier_attrs: Dict[str, Dict[str, str]] = vattrs
        self.verifier_attr_code: Dict[str, Any] = {
            qname: compile(info["expr"], f"<verifier::z3:{qname}>", "eval")
            for qname, info in vattrs.items()
        }
        self.sb_attr_by_method: Dict[str, str] = {}
        for qname in vattrs:
            if "::SsaBuilder::" in qname:
                self.sb_attr_by_method[qname.split("::")[-1]] = qname
        for fn in fns:
            if fn.namespace == "abi":
                self.abi_funcs[fn.name] = fn
            elif fn.namespace == "Pass1Context":
                self.ctx_funcs[fn.name] = fn
            else:
                self.global_handlers[fn.name] = fn
        self.log.debug(
            "Loaded %d C++ AST functions and %d verifier_attrs from %s",
            len(fns),
            len(self.verifier_attrs),
            os.path.basename(handlers_ast_path),
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "operands_funcs": sorted(self.operands_funcs.keys()),
            "abi_funcs": sorted(self.abi_funcs.keys()),
            "ctx_funcs": sorted(self.ctx_funcs.keys()),
            "global_handlers": sorted(self.global_handlers.keys()),
        }


class CppSymbolicEvaluator:
    """Symbolically executes Clang AST JSON & InlineOps ASTs to emit SymbolicSsaNodes."""

    def __init__(self, registry: CppHandlerRegistry) -> None:
        self.reg = registry

    def _bind_pass1_params(
        self, fn: CppFunctionDef, ssa: SymbolicSsaModule, ins: InstLoweringContext, args: List[Any]
    ) -> Dict[str, Any]:
        env: Dict[str, Any] = {}
        if len(args) == len(fn.params):
            for p, a in zip(fn.params, args):
                if p.name:
                    env[p.name] = a
            return env
        arg_idx = 0
        for p in fn.params:
            if "Pass1Context" in p.type_str:
                env[p.name or "_ctx"] = ("ctx", ssa, ins)
            elif p.name == "blk" or ("uint32_t" in p.type_str and not p.name and arg_idx == 0 and len(fn.params) >= 4):
                env[p.name or "_blk"] = 0
            elif "DecodedValhallInst" in p.type_str:
                env[p.name or "_ins"] = ins
            elif "LoweringRuleSpec" in p.type_str:
                env[p.name or "_rule"] = ("rule", ins)
            else:
                if p.name:
                    env[p.name] = args[arg_idx]
                arg_idx += 1
        return env

    def call_operand_fn(
        self, name: str, ssa: SymbolicSsaModule, ins: InstLoweringContext, args: List[Any]
    ) -> Any:
        fn = self.reg.operands_funcs[name]
        env = self._bind_pass1_params(fn, ssa, ins, args)
        return self._exec_fn(fn, env, ssa, ins)

    # TODO(v17.x): Verify that [[verifier::z3(...)]] short-circuit semantics
    # are equivalent to the actual C++ implementation bodies.
    def _eval_verifier_attr(
        self, qname: str, this_obj: Any, args: List[Any], ssa: SymbolicSsaModule, ins: InstLoweringContext
    ) -> Any:
        vattr = self.reg.verifier_attrs[qname]
        recv = vattr.get("receiver", "")
        if this_obj is None:
            if recv == "SymbolicSsaModule":
                this_obj = ssa
            elif recv == "InstLoweringContext":
                this_obj = ins
        return eval(
            self.reg.verifier_attr_code[qname],
            {"__builtins__": {"len": len, "min": min, "max": max, "bool": bool, "int": int}},
            {"this": this_obj, "args": args, "ssa": ssa, "ins": ins},
        )

    def call_abi_fn(
        self, name: str, ssa: SymbolicSsaModule, ins: InstLoweringContext, args: List[Any]
    ) -> Any:
        qname = f"valhall::compiler::abi::{name}"
        if qname in self.reg.verifier_attr_code:
            eval_args = args[4:] if (len(args) >= 4 and isinstance(args[0], tuple) and args[0][0] == "ctx") else args
            return self._eval_verifier_attr(qname, None, eval_args, ssa, ins)
        fn = self.reg.abi_funcs[name]
        env = self._bind_pass1_params(fn, ssa, ins, args)
        return self._exec_fn(fn, env, ssa, ins)

    def call_global_handler(
        self, name: str, ssa: SymbolicSsaModule, ins: InstLoweringContext
    ) -> Any:
        fn = self.reg.global_handlers[name]
        env = self._bind_pass1_params(fn, ssa, ins, [])
        return self._exec_fn(fn, env, ssa, ins)

    def _call_ctx_method(
        self, name: str, ssa: SymbolicSsaModule, ins: InstLoweringContext, args: List[Any]
    ) -> Any:
        fn = self.reg.ctx_funcs[name]
        env: Dict[str, Any] = {"this_ctx": ("ctx", ssa, ins)}
        for p, a in zip(fn.params, args):
            env[p.name] = a
        return self._exec_fn(fn, env, ssa, ins)

    def _exec_fn(
        self, fn: CppFunctionDef, env: Dict[str, Any], ssa: SymbolicSsaModule, ins: InstLoweringContext
    ) -> Any:
        try:
            self._exec_stmts(fn.body, env, ssa, ins)
        except _ReturnSignal as ret:
            return ret.value
        return None

    def _exec_stmts(
        self, stmts: List[Any], env: Dict[str, Any], ssa: SymbolicSsaModule, ins: InstLoweringContext
    ) -> None:
        for st in stmts:
            self._exec_stmt(st, env, ssa, ins)

    def _exec_stmt(
        self, st: Any, env: Dict[str, Any], ssa: SymbolicSsaModule, ins: InstLoweringContext
    ) -> None:
        if isinstance(st, CppVarDecl):
            if st.is_struct_zero and "SsaNode" in st.type_str:
                env[st.name] = SymbolicSsaNode(id=0, op="Nop", val_type="Void")
            elif "vector" in st.type_str and st.init is not None:
                val = self._eval_expr(st.init, env, ssa, ins)
                env[st.name] = [0] * int(val) if isinstance(val, int) else list(val)
            elif st.init is not None:
                env[st.name] = self._eval_expr(st.init, env, ssa, ins)
            elif "basic_string" in st.type_str or "string" in st.type_str:
                env[st.name] = ""
            elif "vector" in st.type_str:
                env[st.name] = []
            else:
                env[st.name] = 0
        elif isinstance(st, CppMultiVarDecl):
            for d in st.decls:
                self._exec_stmt(d, env, ssa, ins)
        elif isinstance(st, CppReturn):
            val = self._eval_expr(st.expr, env, ssa, ins) if st.expr is not None else None
            raise _ReturnSignal(val)
        elif isinstance(st, CppBreak):
            raise _BreakSignal()
        elif isinstance(st, CppContinue):
            raise _ContinueSignal()
        elif isinstance(st, CppBlock):
            self._exec_stmts(st.stmts, env, ssa, ins)
        elif isinstance(st, CppIf):
            if st.init is not None:
                self._exec_stmt(st.init, env, ssa, ins)
            if self._eval_expr(st.cond, env, ssa, ins):
                self._exec_stmts(st.then_body, env, ssa, ins)
            elif st.else_body is not None:
                self._exec_stmts(st.else_body, env, ssa, ins)
        elif isinstance(st, CppFor):
            if st.cond is None and st.step is None and isinstance(st.init, CppVarDecl):
                seq = self._eval_expr(st.init.init, env, ssa, ins)
                for item in seq:
                    env[st.init.name] = item
                    self._exec_stmts(st.body, env, ssa, ins)
            else:
                if st.init is not None:
                    self._exec_stmt(st.init, env, ssa, ins)
                while st.cond is None or self._eval_expr(st.cond, env, ssa, ins):
                    try:
                        self._exec_stmts(st.body, env, ssa, ins)
                    except _BreakSignal:
                        break
                    if st.step is not None:
                        self._eval_expr(st.step, env, ssa, ins)
        elif isinstance(st, CppSwitch):
            val = self._eval_expr(st.expr, env, ssa, ins)
            start_idx: Optional[int] = None
            default_idx: Optional[int] = None
            for idx, (cval, _cstmts) in enumerate(st.cases):
                if cval is None:
                    default_idx = idx
                elif val == self._eval_expr(cval, env, ssa, ins):
                    start_idx = idx
                    break
            if start_idx is None:
                start_idx = default_idx
            if start_idx is not None:
                for _cval, cstmts in st.cases[start_idx:]:
                    try:
                        self._exec_stmts(cstmts, env, ssa, ins)
                    except _BreakSignal:
                        break
        elif isinstance(st, CppAssign):
            val = self._eval_expr(st.value, env, ssa, ins)
            self._do_assign(st.target, st.op, val, env, ssa, ins)
        elif isinstance(st, CppExprStmt):
            self._eval_expr(st.expr, env, ssa, ins)

    def _do_assign(
        self, target: Any, op: str, val: Any, env: Dict[str, Any], ssa: SymbolicSsaModule, ins: InstLoweringContext
    ) -> None:
        if isinstance(target, CppIdent):
            cur = env.get(target.name, 0)
            env[target.name] = self._apply_assign_op(cur, op, val)
        elif isinstance(target, CppUnary) and target.op == "*":
            ptr = self._eval_expr(target.operand, env, ssa, ins)
            if isinstance(ptr, _PointerOut):
                ptr.set(val)
        elif isinstance(target, CppMember):
            obj = self._eval_expr(target.obj, env, ssa, ins)
            if isinstance(obj, SymbolicSsaNode):
                attr = "val_type" if target.member == "type" else target.member
                if attr == "effects" and op == "|=":
                    pass
                else:
                    cur = getattr(obj, attr, 0)
                    setattr(obj, attr, self._apply_assign_op(cur, op, val))
            elif isinstance(obj, tuple) and len(obj) > 1 and obj[0] == "blk" and target.member == "branch_cond":
                obj[1].branch_cond = val
            elif isinstance(obj, tuple) and len(obj) > 0 and obj[0] == "ctx":
                pass
        elif isinstance(target, CppIndex):
            obj = self._eval_expr(target.obj, env, ssa, ins)
            idx = self._eval_expr(target.index, env, ssa, ins)
            if isinstance(obj, list):
                obj[idx] = val

    @staticmethod
    def _apply_assign_op(cur: Any, op: str, val: Any) -> Any:
        if op == "=":
            return val
        if op == "|=":
            return cur | val
        if op == "&=":
            return cur & val
        if op == "+=":
            return cur + val
        if op == "-=":
            return cur - val
        raise ValueError(f"Unsupported assign op {op}")

    def _eval_expr(
        self, expr: Any, env: Dict[str, Any], ssa: SymbolicSsaModule, ins: InstLoweringContext
    ) -> Any:
        if isinstance(expr, CppLiteral):
            return expr.value
        if isinstance(expr, CppIdent):
            if expr.name == "this" and "this_ctx" in env:
                return env["this_ctx"]
            if expr.name in env:
                return env[expr.name]
            if "this_ctx" in env:
                if expr.name == "sb":
                    return ("sb", ssa)
                if expr.name == "opts":
                    return ("opts", ins)
                if expr.name == "res":
                    return ("res", ssa)
            return expr.name
        if isinstance(expr, CppScopedIdent):
            if expr.parts and expr.parts[-1] == "npos":
                return -1
            if len(expr.parts) >= 2 and expr.parts[-2] == "VaryingNameM":
                _vmap = {"point": 0, "bary": 1, "frag_w": 2, "frag_z": 3}
                if expr.parts[-1] in _vmap:
                    return _vmap[expr.parts[-1]]
            return expr.parts[-1]
        if isinstance(expr, CppInitList):
            return [self._eval_expr(x, env, ssa, ins) for x in expr.items]
        if isinstance(expr, CppLambda):
            return ("lambda", expr, env)
        if isinstance(expr, CppUnary):
            if expr.op == "&" and isinstance(expr.operand, CppIdent):
                return _PointerOut(env, expr.operand.name)
            if expr.op == "*" and isinstance(expr.operand, CppIdent) and expr.operand.name == "this" and "this_ctx" in env:
                return env["this_ctx"]
            if expr.op in ("++", "--") and isinstance(expr.operand, CppIdent):
                cur = env[expr.operand.name]
                nxt = cur + 1 if expr.op == "++" else cur - 1
                env[expr.operand.name] = nxt
                return cur if expr.postfix else nxt
            v = self._eval_expr(expr.operand, env, ssa, ins)
            if expr.op == "!":
                return not bool(v)
            if expr.op == "~":
                return ~int(v)
            if expr.op == "-":
                return -v
            if expr.op == "+":
                return +v
            return v
        if isinstance(expr, CppBinary):
            if expr.op == "&&":
                return bool(self._eval_expr(expr.left, env, ssa, ins)) and bool(self._eval_expr(expr.right, env, ssa, ins))
            if expr.op == "||":
                return bool(self._eval_expr(expr.left, env, ssa, ins)) or bool(self._eval_expr(expr.right, env, ssa, ins))
            l = self._eval_expr(expr.left, env, ssa, ins)
            r = self._eval_expr(expr.right, env, ssa, ins)
            if isinstance(l, _DictRef) and expr.op in ("!=", "=="):
                return (l.found) if expr.op == "!=" else (not l.found)
            if isinstance(r, _DictRef) and expr.op in ("!=", "=="):
                return (r.found) if expr.op == "!=" else (not r.found)
            if expr.op == "+":
                return l + r
            if expr.op == "-":
                if isinstance(l, str) and isinstance(r, str) and len(l) == 1 and len(r) == 1:
                    return ord(l) - ord(r)
                return l - r
            if expr.op == "*":
                return l * r
            if expr.op == "/":
                return l // r if isinstance(l, int) and isinstance(r, int) else l / r
            if expr.op == "%":
                return l % r
            if expr.op == "<<":
                return l << r
            if expr.op == ">>":
                return l >> r
            if expr.op == "&":
                return l & r
            if expr.op == "|":
                return l | r
            if expr.op == "^":
                return l ^ r
            if expr.op == "==":
                return l == r
            if expr.op == "!=":
                return l != r
            if expr.op == "<":
                return l < r
            if expr.op == ">":
                return l > r
            if expr.op == "<=":
                return l <= r
            if expr.op == ">=":
                return l >= r
            raise ValueError(f"Unsupported binary op {expr.op}")
        if isinstance(expr, CppTernary):
            return (
                self._eval_expr(expr.then_expr, env, ssa, ins)
                if self._eval_expr(expr.cond, env, ssa, ins)
                else self._eval_expr(expr.else_expr, env, ssa, ins)
            )
        if isinstance(expr, CppAssign):
            val = self._eval_expr(expr.value, env, ssa, ins)
            self._do_assign(expr.target, expr.op, val, env, ssa, ins)
            return val
        if isinstance(expr, CppIndex):
            obj = self._eval_expr(expr.obj, env, ssa, ins)
            idx = self._eval_expr(expr.index, env, ssa, ins)
            return obj[idx]
        if isinstance(expr, CppMember):
            return self._eval_member(expr, env, ssa, ins)
        if isinstance(expr, CppCall):
            return self._eval_call(expr, env, ssa, ins)
        raise ValueError(f"Unsupported C++ expr AST: {expr}")

    def _eval_member(
        self, expr: CppMember, env: Dict[str, Any], ssa: SymbolicSsaModule, ins: InstLoweringContext
    ) -> Any:
        obj = self._eval_expr(expr.obj, env, ssa, ins)
        m = expr.member
        if isinstance(obj, InstLoweringContext):
            if m == "modifier_strs":
                return obj.modifiers
            if m == "fields":
                return obj.fields
            if m == "sr_write":
                return ("sr_write", obj)
            if m == "sr_read":
                return ("sr_read", obj)
            return getattr(obj, m)
        if isinstance(obj, tuple) and obj[0] in ("sr_write", "sr_read"):
            ci: InstLoweringContext = obj[1]
            prefix = obj[0]
            if m == "present":
                return getattr(ci, f"{prefix}_present")
            if m == "base_reg":
                return getattr(ci, f"{prefix}_base")
            if m == "count":
                return getattr(ci, f"{prefix}_count")
        if isinstance(obj, _DictRef):
            if m == "second":
                return obj.second
        if isinstance(obj, tuple) and obj[0] == "ctx":
            if m == "sb":
                return ("sb", obj[1])
            if m == "opts":
                return ("opts", obj[2])
            if m == "mod":
                return ("mod", obj[1])
            if m == "res":
                return ("res", obj[1])
        if isinstance(obj, SymbolicSsaNode):
            if m == "type":
                return obj.val_type
            if m == "effects":
                return (obj, "effects")
            return getattr(obj, m)
        if isinstance(obj, tuple) and obj[0] == "res" and m == "ssa":
            return ("ssa", obj[1])
        if isinstance(obj, tuple) and obj[0] == "ssa" and m == "nodes":
            return obj[1].nodes
        if isinstance(obj, tuple) and obj[0] == "opts":
            if m == "stage":
                return obj[1].stage
            return getattr(obj[1], m, {})
        if isinstance(obj, tuple) and obj[0] == "rule":
            if m == "effects":
                return obj[1].rule_effects
            if m == "op_class":
                return obj[1].rule_op_class
        if isinstance(obj, tuple) and obj[0] == "mod" and m == "blocks":
            return [("blk", obj[1])]
        if isinstance(obj, tuple) and obj[0] == "blk" and m == "branch_cond":
            return obj[1].branch_cond
        if isinstance(obj, SourceOperandDesc):
            return getattr(obj, m)
        return (obj, m)

    def _eval_call(
        self, expr: CppCall, env: Dict[str, Any], ssa: SymbolicSsaModule, ins: InstLoweringContext
    ) -> Any:
        # Handle method calls: obj.method(args)
        if isinstance(expr.callee, CppMember):
            obj = self._eval_expr(expr.callee.obj, env, ssa, ins)
            m = expr.callee.member
            args = [self._eval_expr(a, env, ssa, ins) for a in expr.args]
            if isinstance(obj, (list, dict, str)) and m in ("size", "empty", "find", "count", "at", "back", "push_back", "pop_back"):
                if m == "size":
                    return len(obj)
                if m == "empty":
                    return len(obj) == 0
                if m == "find":
                    if isinstance(obj, dict):
                        return _DictRef(obj, args[0])
                    return obj.find(args[0])
                if m == "count":
                    return 1 if args[0] in obj else 0
                if m == "at":
                    return obj[args[0]]
                if m == "back":
                    return obj[-1]
                if m == "push_back":
                    obj.append(args[0])
                    return None
                if m == "pop_back":
                    if obj:
                        obj.pop()
                    return None
            if isinstance(obj, dict) and m == "end":
                return ("end", obj)
            if isinstance(obj, tuple) and obj[0] in ("ssa", "mod") and m == "get":
                return obj[1].nodes[int(args[0])]
            qc = expr.qualified_callee
            if not qc and isinstance(obj, tuple) and obj[0] == "sb":
                qc = self.reg.sb_attr_by_method.get(m, "")
            if qc and qc in self.reg.verifier_attr_code:
                this_obj: Any = obj[1] if (isinstance(obj, tuple) and obj[0] == "sb") else obj
                return self._eval_verifier_attr(qc, this_obj, args, ssa, ins)
            if isinstance(obj, tuple) and obj[0] == "ctx":
                return self._call_ctx_method(m, ssa, ins, args)
            if isinstance(obj, tuple) and obj[0] == "blk" and m == "branch_cond":
                return None

        # Qualified callee dispatch from Clang AST JSON
        qc = expr.qualified_callee
        if qc:
            args = [self._eval_expr(a, env, ssa, ins) for a in expr.args]
            if qc in self.reg.verifier_attr_code:
                eval_args = args[4:] if (len(args) >= 4 and isinstance(args[0], tuple) and args[0][0] == "ctx") else args
                return self._eval_verifier_attr(qc, None, eval_args, ssa, ins)
            if qc == "std::move":
                return args[0]
            if qc in ("std::string", "std::__cxx11::basic_string<char>") or qc.startswith("std::__cxx11::basic_string"):
                return str(args[0]) if args else ""
            if qc == "std::to_string":
                return str(args[0])
            if qc == "std::max" or qc.startswith("std::max<"):
                return max(args[0], args[1])
            if qc == "std::min" or qc.startswith("std::min<"):
                return min(args[0], args[1])
            if qc in ("std::strcmp", "strcmp"):
                return 0 if args[0] == args[1] else 1
            if qc.startswith("valhall::compiler::abi::"):
                fn_name = qc.split("::")[-1]
                return self.call_abi_fn(fn_name, ssa, ins, args)
            if qc.startswith("valhall::compiler::"):
                fn_name = qc.split("::")[-1]
                if fn_name in self.reg.operands_funcs:
                    return self.call_operand_fn(fn_name, ssa, ins, args)
                if fn_name in self.reg.abi_funcs:
                    return self.call_abi_fn(fn_name, ssa, ins, args)
                if fn_name in self.reg.global_handlers:
                    fn = self.reg.global_handlers[fn_name]
                    return self._exec_fn(fn, self._bind_pass1_params(fn, ssa, ins, args), ssa, ins)

        # Fallback for DSL-constructed ASTs (from operands_spec_loader.py)
        if isinstance(expr.callee, CppScopedIdent):
            fn_name = "::".join(expr.callee.parts)
            args = [self._eval_expr(a, env, ssa, ins) for a in expr.args]
            if fn_name == "std::move":
                return args[0]
            if fn_name in ("std::string", "std::to_string"):
                return str(args[0])
            if fn_name == "std::max":
                return max(args[0], args[1])
            if fn_name == "std::min":
                return min(args[0], args[1])
            if fn_name == "std::strcmp":
                return 0 if args[0] == args[1] else 1
            if len(expr.callee.parts) >= 2 and expr.callee.parts[-2] == "abi":
                return self.call_abi_fn(expr.callee.parts[-1], ssa, ins, args)

        if isinstance(expr.callee, CppIdent):
            fn_name = expr.callee.name
            args = [self._eval_expr(a, env, ssa, ins) for a in expr.args]
            if fn_name in ("uint32_t", "uint64_t", "int32_t", "int64_t", "uint8_t", "bool", "static_cast"):
                if fn_name == "bool":
                    return bool(args[0])
                if isinstance(args[0], str):
                    return args[0]
                return int(args[0])
            if fn_name == "sizeof":
                return 4
            if fn_name == "variant_to_string":
                return ins.canonical_variant
            if fn_name in env and isinstance(env[fn_name], tuple) and env[fn_name][0] == "lambda":
                _, lam, lam_env = env[fn_name]
                call_env = dict(lam_env)
                for p, a in zip(lam.params, args):
                    call_env[p.name] = a
                try:
                    self._exec_stmts(lam.body, call_env, ssa, ins)
                except _ReturnSignal as ret:
                    return ret.value
                return None
            if "this_ctx" in env and fn_name in self.reg.ctx_funcs:
                return self._call_ctx_method(fn_name, ssa, ins, args)
            if fn_name in self.reg.operands_funcs:
                return self.call_operand_fn(fn_name, ssa, ins, args)
            if fn_name in self.reg.abi_funcs:
                return self.call_abi_fn(fn_name, ssa, ins, args)
            if fn_name in self.reg.global_handlers:
                fn = self.reg.global_handlers[fn_name]
                return self._exec_fn(fn, self._bind_pass1_params(fn, ssa, ins, args), ssa, ins)

        raise ValueError(f"Unsupported C++ call: {expr.callee} (qualified={qc!r})")

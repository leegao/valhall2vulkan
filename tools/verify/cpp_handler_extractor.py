#!/usr/bin/env python3
"""
Continuous C++17 Host Handler & Operand Modeler (Layer 2: GIMPLE Evaluator)
===========================================================================
Consumes GCC GIMPLE ASTs produced by `tools/verify/gcc_gimple_parser.py`
(`g++ -std=c++17 -fdump-tree-gimple`) for:
  - src/valhall_operands.cpp
  - src/valhall_pass1_handlers.cpp

Separates C++ parsing (handled by GCC in `gcc_gimple_parser.py`) from
host-handler & operand SSA modeling (`CppHandlerRegistry` + `CppSymbolicEvaluator`).
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
import struct
from typing import Any, Dict, List, Optional, Tuple

from tools.verify.gcc_gimple_parser import (
    CppEnumMaps,
    GimpleAssign,
    GimpleCall,
    GimpleCondJump,
    GimpleFunctionDef,
    GimpleJump,
    GimpleLabel,
    GimpleReturn,
    GimpleSwitch,
    _split_top_level_commas,
    dump_gimple,
    parse_gimple_functions,
)


# ---------------------------------------------------------------------------
# Symbolic SSA-IR Data Structures (Output of Task 2 Translational Semantics)
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class SymbolicSsaNode:
    id: int
    op: str
    val_type: str
    args: List[int] = dataclasses.field(default_factory=list)
    imm_u32: int = 0
    imm_u32_2: int = 0
    imm_u64: int = 0
    aux_bool: bool = False
    aux_bool2: bool = False
    aux_str: str = ""
    is_barrier_or_cf: bool = False
    is_side_effect: bool = False
    effects: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class SymbolicSsaModule:
    """Symbolic SSA-IR graph emitted by lowering a Valhall instruction."""
    nodes: List[SymbolicSsaNode] = dataclasses.field(default_factory=list)
    reg_writes: Dict[int, int] = dataclasses.field(default_factory=dict)
    token_out: int = 1
    branch_cond: Optional[int] = None
    sideband_reads: List[int] = dataclasses.field(default_factory=list)
    trace_events: List[str] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.nodes:
            self.nodes.append(SymbolicSsaNode(id=0, op="Invalid", val_type="Void"))
            self.nodes.append(SymbolicSsaNode(id=1, op="Nop", val_type="Token"))
        self._reg_ssa: Dict[int, int] = {}

    def current_reg(self, reg: int) -> int:
        reg &= 63
        if reg in self.reg_writes:
            return self.reg_writes[reg]
        if reg not in self._reg_ssa:
            nid = self.emit(SymbolicSsaNode(
                id=0, op="InputReg", val_type="U32", imm_u32=reg, aux_str=f"r{reg}"
            ))
            self._reg_ssa[reg] = nid
        return self._reg_ssa[reg]

    def set_reg(self, reg: int, ssa_id: int) -> None:
        self.reg_writes[reg & 63] = ssa_id

    def emit(self, node: SymbolicSsaNode) -> int:
        nid = len(self.nodes)
        node.id = nid
        self.nodes.append(node)
        return nid

    def emit_const_u32(self, v: int) -> int:
        return self.emit(SymbolicSsaNode(id=0, op="ConstU32", val_type="U32", imm_u32=v & 0xFFFFFFFF))

    def emit_const_s32(self, v: int) -> int:
        return self.emit(SymbolicSsaNode(id=0, op="ConstS32", val_type="S32", imm_u32=v & 0xFFFFFFFF))

    def emit_const_f32_bits(self, bits: int) -> int:
        return self.emit(SymbolicSsaNode(id=0, op="ConstF32", val_type="F32", imm_u32=bits & 0xFFFFFFFF))

    def emit_const_u64(self, v: int) -> int:
        return self.emit(SymbolicSsaNode(id=0, op="ConstU64", val_type="U64", imm_u64=v & 0xFFFFFFFFFFFFFFFF))

    def emit_const_bool(self, v: bool) -> int:
        return self.emit(SymbolicSsaNode(id=0, op="ConstBool", val_type="Bool", aux_bool=bool(v)))

    def emit_bitcast(self, target_vt: str, val_id: int) -> int:
        return self.emit(SymbolicSsaNode(id=0, op="Bitcast", val_type=target_vt, args=[val_id]))


@dataclasses.dataclass
class SourceOperandDesc:
    kind: str = "Register"  # Register, FauUniform, SmallConstant, FauSpecial
    reg_num: int = 0
    imm_bits: int = 0
    is_64bit: bool = False
    word_select: int = 0
    symbolic_name: str = ""
    swizzle_mod: str = ""
    abs: bool = False
    neg: bool = False
    not_mod: bool = False


@dataclasses.dataclass
class InstLoweringContext:
    mnemonic: str
    variant: str = ""
    canonical_mnemonic: str = ""
    canonical_variant: str = ""
    srcs: List[SourceOperandDesc] = dataclasses.field(default_factory=list)
    has_dest: bool = True
    dest_reg: int = 0
    dest_mod: str = ""
    dest_is_64bit: bool = False
    modifiers: Dict[str, str] = dataclasses.field(default_factory=dict)
    fields: Dict[str, int] = dataclasses.field(default_factory=dict)
    has_imm: bool = False
    imm_unsigned: int = 0
    imm_signed: int = 0
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
    rule_effects: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.canonical_mnemonic:
            self.canonical_mnemonic = self.mnemonic
        if not self.canonical_variant:
            self.canonical_variant = self.variant


# ---------------------------------------------------------------------------
# GIMPLE Registry & Symbolic Evaluator
# ---------------------------------------------------------------------------
class _Ref:
    """Models C++ pointer/reference to a variable or object attribute."""
    def __init__(self, container: Any, key: Any) -> None:
        self.container = container
        self.key = key

    def get(self) -> Any:
        if isinstance(self.container, dict):
            return self.container.get(self.key)
        if isinstance(self.container, list):
            return self.container[self.key]
        return getattr(self.container, self.key)

    def set(self, val: Any) -> None:
        if isinstance(self.container, dict):
            self.container[self.key] = val
        elif isinstance(self.container, list):
            self.container[self.key] = val
        else:
            setattr(self.container, self.key, val)


class _ArrIter:
    def __init__(self, arr: List[Any], idx: int) -> None:
        self.arr = arr
        self.idx = idx

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ArrIter) and self.arr is other.arr and self.idx == other.idx

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)


class _MapIter:
    def __init__(self, d: Dict[str, Any], key: Optional[str]) -> None:
        self.d = d
        self.key = key

    @property
    def found(self) -> bool:
        return self.key is not None and self.key in self.d

    @property
    def second(self) -> Any:
        return self.d[self.key] if self.found else None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _MapIter):
            return False
        if not self.found and not other.found:
            return True
        return self.found == other.found and self.key == other.key

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)


class CppHandlerRegistry:
    """Holds all GCC GIMPLE functions extracted from valhall_operands.cpp & valhall_pass1_handlers.cpp."""

    def __init__(
        self,
        operands_path: str,
        handlers_path: str,
        repo_root: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.log = logger or logging.getLogger("cpp_extractor")
        if repo_root is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(operands_path), ".."))
        self.repo_root = repo_root
        self.enums = CppEnumMaps.from_repo(repo_root)

        self.operands_funcs: Dict[str, GimpleFunctionDef] = {}
        self.abi_funcs: Dict[str, GimpleFunctionDef] = {}
        self.ctx_funcs: Dict[str, GimpleFunctionDef] = {}
        self.global_handlers: Dict[str, GimpleFunctionDef] = {}
        self.lambdas: Dict[str, GimpleFunctionDef] = {}

        for path, is_handlers in ((operands_path, False), (handlers_path, True)):
            gimple_txt = dump_gimple(path, repo_root)
            fns = parse_gimple_functions(gimple_txt, source_file=os.path.basename(path))
            for fn in fns:
                if fn.namespace == "lambda":
                    self.lambdas[fn.full_name] = fn
                elif not is_handlers:
                    self.operands_funcs[fn.short_name] = fn
                elif fn.namespace == "abi":
                    self.abi_funcs[fn.short_name] = fn
                elif fn.namespace == "Pass1Context":
                    self.ctx_funcs[fn.short_name] = fn
                else:
                    self.global_handlers[fn.short_name] = fn
            self.log.debug(
                "Extracted %d GIMPLE functions from %s", len(fns), os.path.basename(path)
            )

    def summary(self) -> Dict[str, Any]:
        return {
            "operands_funcs": sorted(self.operands_funcs.keys()),
            "abi_funcs": sorted(self.abi_funcs.keys()),
            "ctx_funcs": sorted(self.ctx_funcs.keys()),
            "global_handlers": sorted(self.global_handlers.keys()),
        }


_BINOP_RE = re.compile(r"^(.+?)\s+(==|!=|<=|>=|<<|>>|\+|-|\*|/|%|&|\||\^|<|>)\s+(.+)$")
_CAST_RE = re.compile(r"^\(([^()]+)\)\s*(.+)$")


class CppSymbolicEvaluator:
    """Executes GCC GIMPLE function IRs to emit SymbolicSsaNodes."""

    def __init__(self, registry: CppHandlerRegistry) -> None:
        self.reg = registry
        self._ssa_op_reverse = {v: k for k, v in registry.enums.ssa_op.items()}
        self._val_type_reverse = {v: k for k, v in registry.enums.val_type.items()}
        self._stage_reverse = {v: k for k, v in registry.enums.shader_stage.items()}
        self._sk_reverse = {v: k for k, v in registry.enums.source_kind.items()}

    def call_operand_fn(
        self, name: str, ssa: SymbolicSsaModule, ins: InstLoweringContext, args: List[Any]
    ) -> Any:
        fn = self.reg.operands_funcs[name]
        env: Dict[str, Any] = {}
        if len(args) == len(fn.params):
            for p, a in zip(fn.params, args):
                env[p.name] = a
        else:
            arg_idx = 0
            for p in fn.params:
                if "SsaBuilder" in p.type_str:
                    env[p.name] = ("sb", ssa)
                elif p.name == "blk_id" or p.name == "blk":
                    env[p.name] = 0
                elif "DecodedValhallInst" in p.type_str:
                    env[p.name] = ins
                else:
                    env[p.name] = args[arg_idx]
                    arg_idx += 1
        return self._exec_gimple(fn, env, ssa, ins)

    def call_abi_fn(
        self, name: str, ssa: SymbolicSsaModule, ins: InstLoweringContext, args: List[Any]
    ) -> Any:
        fn = self.reg.abi_funcs[name]
        env: Dict[str, Any] = {}
        if len(args) == len(fn.params):
            for p, a in zip(fn.params, args):
                if p.name:
                    env[p.name] = a
        else:
            abi_defaults = [("ctx", ssa, ins), 0, ins, ("rule", ins)]
            arg_idx = 0
            for i, p in enumerate(fn.params):
                if i < 4:
                    if p.name:
                        env[p.name] = abi_defaults[i]
                else:
                    val = args[arg_idx]
                    if p.type_str == "ValType" and isinstance(val, str):
                        val = self._val_type_reverse.get(val, val)
                    if p.name:
                        env[p.name] = val
                    arg_idx += 1
        return self._exec_gimple(fn, env, ssa, ins)

    def call_global_handler(
        self, name: str, ssa: SymbolicSsaModule, ins: InstLoweringContext
    ) -> Any:
        fn = self.reg.global_handlers[name]
        env: Dict[str, Any] = {}
        for p in fn.params:
            if "Pass1Context" in p.type_str and p.name:
                env[p.name] = ("ctx", ssa, ins)
            elif p.name in ("blk", "blk_id"):
                env[p.name] = 0
            elif "DecodedValhallInst" in p.type_str and p.name:
                env[p.name] = ins
            elif "LoweringRuleSpec" in p.type_str and p.name:
                env[p.name] = ("rule", ins)
        return self._exec_gimple(fn, env, ssa, ins)

    def _call_ctx_method(
        self, name: str, ssa: SymbolicSsaModule, ins: InstLoweringContext, args: List[Any]
    ) -> Any:
        fn = self.reg.ctx_funcs[name]
        env: Dict[str, Any] = {}
        # First param is `this`
        for p, a in zip(fn.params, [("ctx", ssa, ins)] + list(args)):
            if p.name:
                env[p.name] = a
        return self._exec_gimple(fn, env, ssa, ins)

    # -----------------------------------------------------------------------
    # GIMPLE Flat CFG Execution Engine
    # -----------------------------------------------------------------------
    def _exec_gimple(
        self,
        fn: GimpleFunctionDef,
        env: Dict[str, Any],
        ssa: SymbolicSsaModule,
        ins: InstLoweringContext,
    ) -> Any:
        # Pre-initialize local arrays/structs (e.g. D.1234[N])
        for vname, vtype in fn.local_types.items():
            if vname not in env:
                if "[" in vtype:
                    sz_m = re.search(r"\[(\d+)\]", vtype)
                    env[vname] = [0] * int(sz_m.group(1)) if sz_m else []
                elif "SsaNode" in vtype:
                    env[vname] = SymbolicSsaNode(id=0, op="Nop", val_type="Void")
                elif "._anon_" in vtype or "initializer_list" in vtype:
                    env[vname] = {}
                else:
                    env[vname] = 0

        pc = 0
        stmts = fn.stmts
        n = len(stmts)
        while pc < n:
            st = stmts[pc]
            if isinstance(st, GimpleLabel):
                pc += 1
            elif isinstance(st, GimpleJump):
                pc = fn.labels[st.label]
            elif isinstance(st, GimpleCondJump):
                cond_val = self._eval_expr(st.cond, env, ssa, ins)
                lbl = st.true_label if cond_val else st.false_label
                pc = fn.labels[lbl]
            elif isinstance(st, GimpleSwitch):
                val = self._eval_expr(st.expr, env, ssa, ins)
                if isinstance(val, str) and val in self._sk_reverse:
                    val = self._sk_reverse[val]
                lbl = st.cases.get(int(val), st.default_label)
                pc = fn.labels[lbl] if lbl is not None else pc + 1
            elif isinstance(st, GimpleReturn):
                if st.expr is None:
                    return None
                return self._eval_expr(st.expr, env, ssa, ins)
            elif isinstance(st, GimpleAssign):
                if st.rhs == "{}":
                    if st.lhs in env:
                        cur_v = env[st.lhs]
                        if isinstance(cur_v, SymbolicSsaNode):
                            env[st.lhs] = SymbolicSsaNode(id=0, op="Nop", val_type="Void")
                        elif isinstance(cur_v, list):
                            env[st.lhs] = []
                        else:
                            env[st.lhs] = {}
                    elif st.lhs.endswith(".args") or st.lhs.endswith(".phi_incomings"):
                        self._assign_lhs(st.lhs, [], env, ssa, ins)
                    else:
                        self._assign_lhs(st.lhs, {}, env, ssa, ins)
                else:
                    val = self._eval_rhs(st.rhs, env, ssa, ins)
                    self._assign_lhs(st.lhs, val, env, ssa, ins)
                pc += 1
            elif isinstance(st, GimpleCall):
                res = self._eval_call(st.callee, st.args, env, ssa, ins)
                if st.lhs is not None:
                    self._assign_lhs(st.lhs, res, env, ssa, ins)
                pc += 1
            else:
                pc += 1
        return None

    # -----------------------------------------------------------------------
    # Expression / LValue / Call Evaluators
    # -----------------------------------------------------------------------
    def _eval_rhs(
        self, rhs: str, env: Dict[str, Any], ssa: SymbolicSsaModule, ins: InstLoweringContext
    ) -> Any:
        # Aggregate initializer: {.id=0, .op=94, .type=0, ...} or {"widen", "swz", "lanes"}
        if rhs.startswith("{") and rhs.endswith("}"):
            inner = rhs[1:-1].strip()
            if not inner:
                return SymbolicSsaNode(id=0, op="Nop", val_type="Void")
            items = _split_top_level_commas(inner)
            if items and items[0].startswith("."):
                node = SymbolicSsaNode(id=0, op="Nop", val_type="Void")
                for it in items:
                    k, v = it[1:].split("=", 1)
                    k, v = k.strip(), v.strip()
                    if k == "op":
                        node.op = self.reg.enums.ssa_op.get(int(v), v)
                    elif k == "type":
                        node.val_type = self.reg.enums.val_type.get(int(v), v)
                    elif k in ("imm_u32", "imm_u32_2", "imm_u64"):
                        setattr(node, k, int(v))
                    elif k in ("aux_bool", "aux_bool2", "is_barrier_or_cf", "is_side_effect"):
                        setattr(node, k, bool(int(v)))
                return node
            return [self._eval_expr(it, env, ssa, ins) for it in items]
        return self._eval_expr(rhs, env, ssa, ins)

    def _eval_expr(
        self, expr: str, env: Dict[str, Any], ssa: SymbolicSsaModule, ins: InstLoweringContext
    ) -> Any:
        s = expr.strip()
        # Strip outer cast `(type) inner`
        while True:
            cm = _CAST_RE.match(s)
            if cm and not cm.group(1).strip().startswith("("):
                cast_t = cm.group(1).strip()
                inner = cm.group(2).strip()
                val = self._eval_expr(inner, env, ssa, ins)
                if cast_t in ("uint8_t", "unsigned char") and isinstance(val, int):
                    return val & 0xFF
                if cast_t in ("uint32_t", "unsigned int") and isinstance(val, int):
                    return val & 0xFFFFFFFF
                if cast_t == "bool":
                    return bool(val)
                return val
            break

        # String / char literals
        if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
            return s[1:-1]
        if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
            return ord(s[1:-1])

        # Numeric literals
        if re.fullmatch(r"-?(?:0x[0-9a-fA-F]+|[0-9]+)", s):
            return int(s, 16 if "0x" in s else 10)
        if re.fullmatch(r"-?[0-9]+\.[0-9]+(?:[eE][+-]?[0-9]+)?f?", s):
            return float(s.rstrip("f"))

        # Address-of `&x`
        if s.startswith("&"):
            target = s[1:].strip()
            # If &var, return a _Ref (or value if it's an object/struct)
            if target in env:
                v = env[target]
                if isinstance(v, (dict, list, SymbolicSsaNode, InstLoweringContext)) or (
                    isinstance(v, tuple) and v[0] in ("sb", "ctx", "rule")
                ):
                    return v
                return _Ref(env, target)
            return self._resolve_lvalue_ref(target, env, ssa, ins)

        # Dereference `*x`
        if s.startswith("*"):
            inner = self._eval_expr(s[1:].strip(), env, ssa, ins)
            if isinstance(inner, _Ref):
                return inner.get()
            if isinstance(inner, _ArrIter):
                return inner.arr[inner.idx]
            if isinstance(inner, list):
                return inner[0]
            return inner

        # Unary `~x` or `-x` or `!x`
        if s.startswith("~"):
            v = self._eval_expr(s[1:], env, ssa, ins)
            return (not v) if isinstance(v, bool) else (~int(v))
        if s.startswith("!"):
            return not bool(self._eval_expr(s[1:], env, ssa, ins))
        if s.startswith("-") and not s[1:].isdigit():
            return -self._eval_expr(s[1:], env, ssa, ins)

        # Binary expression
        bm = _BINOP_RE.match(s)
        if bm:
            l = self._eval_expr(bm.group(1), env, ssa, ins)
            op = bm.group(2)
            r = self._eval_expr(bm.group(3), env, ssa, ins)
            if isinstance(l, _MapIter):
                return (l.found) if op == "!=" else (not l.found)
            if isinstance(l, _ArrIter):
                if op == "+":
                    return _ArrIter(l.arr, l.idx + 1)
                if op == "!=":
                    return l.idx != r.idx
                if op == "==":
                    return l.idx == r.idx
            if op == "+":
                return l + r
            if op == "-":
                return l - r
            if op == "*":
                return l * r
            if op == "/":
                return l // r if isinstance(l, int) and isinstance(r, int) else l / r
            if op == "%":
                return l % r
            if op == "<<":
                return l << r
            if op == ">>":
                return l >> r
            if op == "&":
                return l & r
            if op == "|":
                return l | r
            if op == "^":
                return l ^ r
            if op == "==":
                return l == r
            if op == "!=":
                return l != r
            if op == "<":
                return l < r
            if op == ">":
                return l > r
            if op == "<=":
                return l <= r
            if op == ">=":
                return l >= r

        # Array index `a[i]`
        if "[" in s and s.endswith("]"):
            base_s, idx_s = s[:-1].rsplit("[", 1)
            base = self._eval_expr(base_s.strip(), env, ssa, ins)
            idx = self._eval_expr(idx_s.strip(), env, ssa, ins)
            if isinstance(base, str):
                return ord(base[idx])
            return base[idx]

        # Variable lookup (before member access since GIMPLE temps like D.1234 contain `.`)
        if s in env:
            return env[s]

        # Member access `a->b` or `a.b`
        if "->" in s or "." in s:
            return self._eval_member_path(s, env, ssa, ins)
        return s

    @staticmethod
    def _split_base_and_members(expr: str, env: Dict[str, Any]) -> Tuple[str, List[str]]:
        parts = re.split(r"(->|\.)", expr)
        if len(parts) >= 3 and parts[1] == "." and f"{parts[0].strip()}.{parts[2].strip()}" in env:
            return f"{parts[0].strip()}.{parts[2].strip()}", parts[3:]
        return parts[0].strip(), parts[1:]

    def _eval_member_path(
        self, s: str, env: Dict[str, Any], ssa: SymbolicSsaModule, ins: InstLoweringContext
    ) -> Any:
        # Split on -> and . outside brackets
        base_name, rest = self._split_base_and_members(s, env)
        cur = env.get(base_name, base_name)
        if isinstance(cur, _Ref):
            cur = cur.get()
        parts = [base_name] + rest
        i = 1
        while i < len(parts):
            mem = parts[i + 1].strip()
            if isinstance(cur, tuple) and len(cur) > 0 and cur[0] == "ctx":
                if mem == "sb":
                    cur = ("sb", cur[1])
                elif mem == "opts":
                    cur = ("opts", cur[2])
                elif mem == "mod":
                    cur = ("mod", cur[1])
                elif mem == "res":
                    cur = ("res", cur[1])
            elif isinstance(cur, tuple) and len(cur) > 0 and cur[0] == "opts" and mem == "stage":
                cur = self._stage_reverse.get(cur[1].stage, cur[1].stage)
            elif isinstance(cur, tuple) and len(cur) > 0 and cur[0] == "rule" and mem == "effects":
                cur = cur[1].rule_effects
            elif isinstance(cur, tuple) and len(cur) > 0 and cur[0] == "mod" and mem == "blocks":
                cur = [("blk", cur[1])]
            elif isinstance(cur, InstLoweringContext):
                if mem == "modifier_strs":
                    cur = cur.modifiers
                elif mem == "fields":
                    cur = cur.fields
                elif mem == "sr_write":
                    cur = ("sr_write", cur)
                elif mem == "sr_read":
                    cur = ("sr_read", cur)
                elif mem == "canonical_variant":
                    cur = cur.canonical_variant
                elif mem == "variant":
                    cur = cur.variant
                elif mem == "canonical_mnemonic":
                    cur = cur.canonical_mnemonic
                elif mem == "mnemonic":
                    cur = cur.mnemonic
                else:
                    cur = getattr(cur, mem)
            elif isinstance(cur, tuple) and len(cur) > 0 and cur[0] in ("sr_write", "sr_read"):
                ci = cur[1]
                m_attr = "base" if mem == "base_reg" else mem
                cur = getattr(ci, f"{cur[0]}_{m_attr}")
            elif isinstance(cur, SourceOperandDesc):
                val = getattr(cur, mem)
                if mem == "kind" and isinstance(val, str):
                    val = self._sk_reverse.get(val, val)
                cur = val
            elif isinstance(cur, dict):
                # Closure field like __is_signed, __sb, or initializer_list
                if mem.startswith("__") and mem[2:] in cur:
                    cur = cur[mem[2:]]
                else:
                    cur = cur.get(mem, 0)
            elif isinstance(cur, SymbolicSsaNode):
                attr = "val_type" if mem == "type" else mem
                cur = getattr(cur, attr)
            elif isinstance(cur, _MapIter):
                if mem == "second":
                    cur = cur.second
            else:
                cur = getattr(cur, mem, 0)
            i += 2
        return cur

    def _resolve_lvalue_ref(
        self, target: str, env: Dict[str, Any], ssa: SymbolicSsaModule, ins: InstLoweringContext
    ) -> _Ref:
        if target in env:
            return _Ref(env, target)
        if "[" in target and target.endswith("]"):
            base_s, idx_s = target[:-1].rsplit("[", 1)
            base = self._eval_expr(base_s.strip(), env, ssa, ins)
            idx = self._eval_expr(idx_s.strip(), env, ssa, ins)
            return _Ref(base, idx)
        base_name, rest = self._split_base_and_members(target, env)
        parts = [base_name] + rest
        cur = env.get(base_name)
        if isinstance(cur, _Ref):
            cur = cur.get()
        i = 1
        while i + 2 < len(parts):
            mem = parts[i + 1].strip()
            if isinstance(cur, tuple) and cur[0] == "ctx" and mem == "mod":
                cur = ("mod", cur[1])
            elif isinstance(cur, tuple) and len(cur) > 0 and cur[0] == "mod" and mem == "blocks":
                cur = [("blk", cur[1])]
            elif isinstance(cur, dict):
                cur = cur.setdefault(mem, {})
            else:
                cur = getattr(cur, mem)
            i += 2
        last_mem = parts[-1].strip()
        if isinstance(cur, InstLoweringContext) and last_mem == "modifier_strs":
            last_mem = "modifiers"
        if isinstance(cur, tuple) and cur[0] == "mod" and last_mem == "blocks":
            return _Ref({"blocks": [("blk", cur[1])]}, "blocks")
        return _Ref(cur, last_mem)

    def _assign_lhs(
        self, lhs: str, val: Any, env: Dict[str, Any], ssa: SymbolicSsaModule, ins: InstLoweringContext
    ) -> None:
        lhs = lhs.strip()
        # Handle `static const char * const C.2[3] = {...}` — extract variable name
        m_static = re.match(r"^(?:static\s+)?(?:const\s+)?(?:unsigned\s+)?[\w\s:*<>]+\s+([A-Za-z_]\w*\.\d+(?:_\d+)?)\[\d+\]$", lhs)
        if m_static:
            env[m_static.group(1)] = val
            return
        # Handle `(*ptr) = val` or `*ptr = val`
        if lhs.startswith("*"):
            ptr = self._eval_expr(lhs[1:].strip(), env, ssa, ins)
            if isinstance(ptr, _Ref):
                ptr.set(val)
            return
        # Handle `D.1234[0] = val`
        if "[" in lhs and lhs.endswith("]"):
            base_s, idx_s = lhs[:-1].rsplit("[", 1)
            if "[" in base_s:
                # e.g. `(*ctx->mod.blocks)[0].branch_cond`
                pass
            base = self._eval_expr(base_s.strip(), env, ssa, ins)
            idx = self._eval_expr(idx_s.strip(), env, ssa, ins)
            if isinstance(base, list):
                base[idx] = val
            return
        # GIMPLE temp/SSA variables (e.g. D.1234, retval.44, iftmp.1, lo.161_4) contain `.` followed by digits
        if lhs in env or re.fullmatch(r"[A-Za-z_]\w*\.\d+(?:_\d+)?", lhs):
            env[lhs] = val
            return
        # Handle `n.op = 94` / `n.type = 2` / `extract_bits.__sb = sb`
        if "->" in lhs or "." in lhs:
            base_name, rest = self._split_base_and_members(lhs, env)
            parts = [base_name] + rest
            if base_name in env:
                cur = env[base_name]
                if isinstance(cur, _Ref):
                    cur = cur.get()
                for j in range(1, len(parts) - 2, 2):
                    m = parts[j + 1].strip()
                    if isinstance(cur, tuple) and cur[0] == "ctx" and m == "mod":
                        cur = ("mod", cur[1])
                    elif isinstance(cur, dict):
                        cur = cur.setdefault(m, {})
                    else:
                        cur = getattr(cur, m)
                mem = parts[-1].strip()
                if isinstance(cur, tuple) and len(cur) == 2 and cur[0] == "blk":
                    if mem == "branch_cond":
                        cur[1].branch_cond = val
                    return
                if isinstance(cur, SymbolicSsaNode):
                    if mem == "op":
                        cur.op = self.reg.enums.ssa_op.get(int(val), str(val)) if isinstance(val, int) else str(val)
                    elif mem == "type":
                        cur.val_type = self.reg.enums.val_type.get(int(val), str(val)) if isinstance(val, int) else str(val)
                    elif mem == "effects":
                        pass
                    else:
                        setattr(cur, mem, val)
                elif isinstance(cur, dict):
                    cur[mem] = val
                elif isinstance(cur, tuple) and len(cur) > 0 and cur[0] == "blk" and mem == "branch_cond":
                    cur[1].branch_cond = val
            return
        env[lhs] = val

    def _eval_call(
        self,
        callee: str,
        arg_strs: List[str],
        env: Dict[str, Any],
        ssa: SymbolicSsaModule,
        ins: InstLoweringContext,
    ) -> Any:
        # Ignore C++ RAII constructors/destructors/allocators except meaningful ones
        if "SsaNode::SsaNode" in callee:
            if len(arg_strs) >= 2:
                src_v = self._eval_expr(arg_strs[1], env, ssa, ins)
                if isinstance(src_v, _Ref):
                    src_v = src_v.get()
                if isinstance(src_v, SymbolicSsaNode):
                    self._assign_lhs(arg_strs[0].lstrip("&"), dataclasses.replace(src_v, args=list(src_v.args) if isinstance(src_v.args, list) else []), env, ssa, ins)
            return None
        if (
            "::~" in callee
            or callee.startswith("std::allocator<")
            or "_Vector_base" in callee
        ):
            return None

        # std::string / std::__cxx11::basic_string
        if "basic_string<char>::" in callee:
            if callee.rsplit("::", 1)[-1].startswith("basic_string"):
                if len(arg_strs) >= 2:
                    val = self._eval_expr(arg_strs[1], env, ssa, ins)
                    if isinstance(val, _Ref):
                        val = val.get()
                    if isinstance(val, str):
                        self._assign_lhs(arg_strs[0].lstrip("&"), val, env, ssa, ins)
                return None
            if callee.endswith("::empty"):
                obj = self._eval_expr(arg_strs[0], env, ssa, ins)
                if isinstance(obj, _Ref):
                    obj = obj.get()
                return len(obj or "") == 0
            if callee.endswith("::find"):
                obj = self._eval_expr(arg_strs[0], env, ssa, ins)
                if isinstance(obj, _Ref):
                    obj = obj.get()
                sub = self._eval_expr(arg_strs[1], env, ssa, ins)
                if isinstance(sub, _Ref):
                    sub = sub.get()
                pos = (obj or "").find(sub)
                return pos if pos >= 0 else 0xFFFFFFFFFFFFFFFF
            if callee.endswith("::back"):
                obj = self._eval_expr(arg_strs[0], env, ssa, ins)
                if isinstance(obj, _Ref):
                    obj = obj.get()
                return ord(obj[-1]) if obj else 0
            if callee.endswith("::operator[]"):
                obj = self._eval_expr(arg_strs[0], env, ssa, ins)
                if isinstance(obj, _Ref):
                    obj = obj.get()
                idx = self._eval_expr(arg_strs[1], env, ssa, ins)
                return ord(obj[idx]) if obj and 0 <= idx < len(obj) else 0
            if callee.endswith("::operator="):
                val = self._eval_expr(arg_strs[1], env, ssa, ins)
                if isinstance(val, _Ref):
                    val = val.get()
                self._assign_lhs(arg_strs[0].lstrip("&"), val, env, ssa, ins)
                return None

        if "to_string" in callee:
            return str(self._eval_expr(arg_strs[0], env, ssa, ins))

        if "std::operator==" in callee or "std::operator!=" in callee:
            a = self._eval_expr(arg_strs[0], env, ssa, ins)
            b = self._eval_expr(arg_strs[1], env, ssa, ins)
            if isinstance(a, _Ref):
                a = a.get()
            if isinstance(b, _Ref):
                b = b.get()
            if "operator==" in callee:
                return a == b
            return a != b

        if "std::operator+" in callee:
            a = self._eval_expr(arg_strs[0], env, ssa, ins)
            b = self._eval_expr(arg_strs[1], env, ssa, ins)
            if isinstance(a, _Ref):
                a = a.get()
            if isinstance(b, _Ref):
                b = b.get()
            return str(a) + str(b)

        # std::vector
        if callee.startswith("std::vector<"):
            if callee.endswith("::empty"):
                obj = self._eval_expr(arg_strs[0], env, ssa, ins)
                if isinstance(obj, _Ref):
                    obj = obj.get()
                return len(obj) == 0
            if callee.endswith("::size"):
                obj = self._eval_expr(arg_strs[0], env, ssa, ins)
                if isinstance(obj, _Ref):
                    obj = obj.get()
                return len(obj)
            if callee.endswith("::operator[]"):
                obj = self._eval_expr(arg_strs[0], env, ssa, ins)
                if isinstance(obj, _Ref):
                    obj = obj.get()
                idx = self._eval_expr(arg_strs[1], env, ssa, ins)
                if isinstance(obj, list) and obj and isinstance(obj[0], tuple) and len(obj[0]) > 0 and obj[0][0] == "blk":
                    return obj[idx]
                return _Ref(obj, idx) if isinstance(obj, list) else obj[idx]
            if callee.endswith("::vector"):
                # vector(n) or vector(initializer_list, allocator)
                if len(arg_strs) >= 2:
                    if arg_strs[1].isdigit():
                        self._assign_lhs(arg_strs[0].lstrip("&"), [0] * int(arg_strs[1]), env, ssa, ins)
                    else:
                        il = self._eval_expr(arg_strs[1], env, ssa, ins)
                        if isinstance(il, _Ref):
                            il = il.get()
                        if isinstance(il, dict) and "_M_array" in il:
                            arr = il["_M_array"]
                            if isinstance(arr, _Ref):
                                arr = arr.get()
                            ln = il.get("_M_len", len(arr))
                            self._assign_lhs(arg_strs[0].lstrip("&"), list(arr[:ln]), env, ssa, ins)
                        elif isinstance(il, list):
                            self._assign_lhs(arg_strs[0].lstrip("&"), list(il), env, ssa, ins)
                return None
            if callee.endswith("::operator="):
                il = self._eval_expr(arg_strs[1], env, ssa, ins)
                if isinstance(il, dict) and "_M_array" in il:
                    arr = il["_M_array"]
                    if isinstance(arr, _Ref):
                        arr = arr.get()
                    ln = il.get("_M_len", len(arr))
                    val = list(arr[:ln])
                else:
                    val = list(il) if isinstance(il, list) else []
                self._assign_lhs(arg_strs[0].lstrip("&"), val, env, ssa, ins)
                return None

        # std::initializer_list::begin / end
        if "initializer_list" in callee:
            il = self._eval_expr(arg_strs[0], env, ssa, ins)
            if isinstance(il, _Ref):
                il = il.get()
            arr = il.get("_M_array", []) if isinstance(il, dict) else []
            if isinstance(arr, _Ref):
                arr = arr.get()
            ln = il.get("_M_len", len(arr)) if isinstance(il, dict) else len(arr)
            if callee.endswith("::begin"):
                return _ArrIter(arr, 0)
            if callee.endswith("::end"):
                return _ArrIter(arr, ln)

        # std::map & iterators
        if callee.startswith("std::map<"):
            d = self._eval_expr(arg_strs[0], env, ssa, ins)
            if isinstance(d, _Ref):
                d = d.get()
            if callee.endswith("::find"):
                k = self._eval_expr(arg_strs[1], env, ssa, ins)
                if isinstance(k, _Ref):
                    k = k.get()
                return _MapIter(d, k if k in d else None)
            if callee.endswith("::end"):
                return _MapIter(d, None)
            if callee.endswith("::count"):
                k = self._eval_expr(arg_strs[1], env, ssa, ins)
                if isinstance(k, _Ref):
                    k = k.get()
                return 1 if k in d else 0
            if callee.endswith("::at"):
                k = self._eval_expr(arg_strs[1], env, ssa, ins)
                if isinstance(k, _Ref):
                    k = k.get()
                return d[k]

        if "_Rb_tree_const_iterator" in callee:
            if "operator!=" in callee:
                it = self._eval_expr(arg_strs[0], env, ssa, ins)
                if isinstance(it, _Ref):
                    it = it.get()
                return it.found
            if "operator==" in callee:
                it = self._eval_expr(arg_strs[0], env, ssa, ins)
                if isinstance(it, _Ref):
                    it = it.get()
                return not it.found
            if "operator->" in callee:
                it = self._eval_expr(arg_strs[0], env, ssa, ins)
                if isinstance(it, _Ref):
                    it = it.get()
                return it

        if "std::move" in callee:
            v = self._eval_expr(arg_strs[0], env, ssa, ins)
            return v.get() if isinstance(v, _Ref) else v

        if "std::max" in callee or "std::min" in callee:
            a = self._eval_expr(arg_strs[0], env, ssa, ins)
            b = self._eval_expr(arg_strs[1], env, ssa, ins)
            if isinstance(a, _Ref): a = a.get()
            if isinstance(b, _Ref): b = b.get()
            return max(a, b) if "std::max" in callee else min(a, b)
        if "strcmp" in callee:
            a = self._eval_expr(arg_strs[0], env, ssa, ins)
            b = self._eval_expr(arg_strs[1], env, ssa, ins)
            return 0 if a == b else 1
        if "memcpy" in callee:
            src_ref = self._eval_expr(arg_strs[1], env, ssa, ins)
            fval = float(src_ref.get() if isinstance(src_ref, _Ref) else src_ref)
            bits = struct.unpack("<I", struct.pack("<f", fval))[0]
            dst_ref = self._eval_expr(arg_strs[0], env, ssa, ins)
            if isinstance(dst_ref, _Ref):
                dst_ref.set(bits)
            return None

        # SsaBuilder methods
        if "SsaBuilder::" in callee:
            args = [self._eval_expr(a, env, ssa, ins) for a in arg_strs]
            m = callee.rsplit("::", 1)[-1]
            if m == "current_reg":
                return ssa.current_reg(args[2])
            if m == "set_reg":
                ssa.set_reg(args[2], args[3])
                return None
            if m == "current_token":
                return ssa.token_out
            if m == "set_token":
                ssa.token_out = args[2]
                return None
            if m == "emit":
                return ssa.emit(args[2])
            if m == "emit_const_u32":
                return ssa.emit_const_u32(args[2])
            if m == "emit_const_s32":
                return ssa.emit_const_s32(args[2])
            if m == "emit_const_f32_bits":
                return ssa.emit_const_f32_bits(args[2])
            if m == "emit_const_u64":
                return ssa.emit_const_u64(args[2])
            if m == "emit_const_bool":
                return ssa.emit_const_bool(args[2])
            if m == "emit_bitcast":
                vt = self.reg.enums.val_type.get(int(args[2]), str(args[2])) if isinstance(args[2], int) else str(args[2])
                return ssa.emit_bitcast(vt, args[3])

        # Lambda operator()
        if callee in self.reg.lambdas:
            lam_fn = self.reg.lambdas[callee]
            args = [self._eval_expr(a, env, ssa, ins) for a in arg_strs]
            lam_env = {p.name: a for p, a in zip(lam_fn.params, args) if p.name}
            return self._exec_gimple(lam_fn, lam_env, ssa, ins)

        # valhall::variant_to_string
        if "variant_to_string" in callee:
            return ins.canonical_variant

        # valhall::compiler::Pass1Context::*
        if "Pass1Context::" in callee:
            m = callee.rsplit("::", 1)[-1]
            args = [self._eval_expr(a, env, ssa, ins) for a in arg_strs[1:]]
            return self._call_ctx_method(m, ssa, ins, args)

        # valhall::compiler::abi::*
        if "valhall::compiler::abi::" in callee:
            fn_name = callee.rsplit("::", 1)[-1]
            args = [self._eval_expr(a, env, ssa, ins) for a in arg_strs]
            return self.call_abi_fn(fn_name, ssa, ins, args)

        # valhall::compiler::* operand/global helpers
        if "valhall::compiler::" in callee:
            fn_name = callee.rsplit("::", 1)[-1]
            args = [self._eval_expr(a, env, ssa, ins) for a in arg_strs]
            if fn_name in self.reg.operands_funcs:
                return self.call_operand_fn(fn_name, ssa, ins, args)
            if fn_name in self.reg.global_handlers:
                return self.call_global_handler(fn_name, ssa, ins)

        raise ValueError(f"Unsupported GIMPLE call: {callee}({arg_strs})")

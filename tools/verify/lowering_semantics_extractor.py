"""
Automated Valhall Lowering & Host Handler Translational Semantics Extractor
===========================================================================
Continuously extracts the translational semantics (Valhall ISA -> Symbolic
Valhall SSA-IR) for every instruction in specs/kraid/isa-v9-v14.xml (627 specs,
201 unique mnemonics) from:
  1. specs/lowering/valhall_lowering.json (232 rules, 10 lookup_tables, 76 host_functions)
  2. tools/dsl_parser.py & tools/valhall_lowering_dsl.py (LL(1) @INLINE_OPS AST)
  3. src/valhall_operands.cpp & src/valhall_pass1_handlers.cpp (via CppHandlerRegistry)

Produces a `SymbolicSsaModule` + structured `LoweringTrace` for any instruction
configuration, ready for direct Z3 evaluation in Task 3.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

_TOOLS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "specs"))
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
    HashExpr,
    HostCall,
    IfStmt,
    Literal,
    LookupCall,
    ModifierKey,
    Program,
    ReturnStmt,
    SrcRef,
    SsaCall,
    VarRef,
    parse_cond_str,
    parse_inline_ops,
)
from valhall_lowering_dsl import VALTYPE_MAP, parse_ssa_op_spec

from tools.verify.clang_ast_evaluator import (
    CppHandlerRegistry,
    CppSymbolicEvaluator,
    InstLoweringContext,
    SourceOperandDesc,
    SymbolicSsaModule,
    SymbolicSsaNode,
)


# ---------------------------------------------------------------------------
# Structured Logging & Trace Diagnostics
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class LoweringTrace:
    mnemonic: str
    variant: str
    rule_index: int
    rule_kind: str
    ssa_op_spec: str
    xml_index: Optional[int] = None
    steps: List[str] = dataclasses.field(default_factory=list)
    warnings: List[str] = dataclasses.field(default_factory=list)
    emitted_ops: List[str] = dataclasses.field(default_factory=list)

    def format_summary(self) -> str:
        lines = [
            f"[T_lowering] {self.mnemonic}.{self.variant or '<none>'} "
            f"(rule #{self.rule_index}, kind={self.rule_kind}, spec={self.ssa_op_spec})"
        ]
        for s in self.steps:
            lines.append(f"  {s}")
        for w in self.warnings:
            lines.append(f"  WARN: {w}")
        lines.append(f"  Emitted SsaOps ({len(self.emitted_ops)}): {', '.join(self.emitted_ops) or 'none'}")
        return "\n".join(lines)


class LoweringLogger:
    """Structured logger for automated lowering extraction and debugging."""

    def __init__(self, name: str = "lowering_extractor", level: int = logging.INFO) -> None:
        self._log = logging.getLogger(name)
        if not self._log.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
            self._log.addHandler(handler)
        self._log.setLevel(level)
        self.traces: List[LoweringTrace] = []

    def record(self, trace: LoweringTrace) -> None:
        self.traces.append(trace)
        if trace.warnings:
            for w in trace.warnings:
                self._log.debug("%s.%s: %s", trace.mnemonic, trace.variant, w)
        self._log.debug(trace.format_summary())


# ---------------------------------------------------------------------------
# Parsed Rule Representation
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class ParsedLoweringRule:
    rule_index: int
    mnemonic: str
    variants: List[str]
    op_class: str
    kind: str  # DirectSsaOp, DynamicSsaOp, InlineOps, Handler
    target: Any
    src_types: List[str]
    dest_type: str
    effects: Tuple[str, ...]
    arch_range: Tuple[int, int] = (9, 14)
    when_cond: Optional[Any] = None
    when_str: str = ""
    inline_program: Optional[Program] = None


# ---------------------------------------------------------------------------
# Automated Extractor Engine
# ---------------------------------------------------------------------------
class LoweringSemanticsExtractor:
    """Extracts and evaluates Valhall ISA -> Symbolic SSA-IR translational semantics."""

    def __init__(
        self,
        repo_root: Optional[str] = None,
        xml_path: Optional[str] = None,
        lowering_path: Optional[str] = None,
        operands_path: Optional[str] = None,
        handlers_path: Optional[str] = None,
        logger: Optional[LoweringLogger] = None,
    ) -> None:
        if repo_root is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.repo_root = repo_root
        self.xml_path = xml_path or os.path.join(repo_root, "specs/kraid/isa-v9-v14.xml")
        self.lowering_path = lowering_path or os.path.join(repo_root, "specs/lowering/valhall_lowering.json")
        self.operands_path = operands_path or self.lowering_path
        self.handlers_path = handlers_path or os.path.join(repo_root, "tools/verify/generated/valhall_host_handlers.json")
        self.logger = logger or LoweringLogger()

        # 1. Load XML instruction catalog (627 specs)
        self.xml_specs = self._load_xml_specs(self.xml_path)

        # 2. Load valhall_lowering.json & parse all 232 rules + LL(1) DSL ASTs
        with open(self.lowering_path, "r", encoding="utf-8") as f:
            spec_json = json.load(f)
        self.schema_version = spec_json.get("schema_version")
        self.host_functions: Dict[str, Dict[str, Any]] = spec_json.get("host_functions", {})
        self.lookup_tables: Dict[str, Dict[str, Any]] = spec_json.get("lookup_tables", {})
        self.rules: List[ParsedLoweringRule] = []
        for idx, r in enumerate(spec_json["rules"]):
            kind, target = parse_ssa_op_spec(r["ssa_op"])
            prog = parse_inline_ops(r["inline_ops"]) if kind == "InlineOps" else None
            ar_raw = r.get("arch_range", [9, 14])
            ar = (int(ar_raw[0]), int(ar_raw[1]))
            w_str = r.get("when", "")
            w_cond = parse_cond_str(w_str) if w_str else None
            self.rules.append(
                ParsedLoweringRule(
                    rule_index=idx,
                    mnemonic=r["mnemonic"],
                    variants=r.get("variants", ["*"]),
                    op_class=r.get("op_class", ""),
                    kind=kind,
                    target=target,
                    src_types=r.get("src_types", []),
                    dest_type=r.get("dest_type", "none"),
                    effects=tuple(r.get("effects", [])),
                    arch_range=ar,
                    when_cond=w_cond,
                    when_str=w_str,
                    inline_program=prog,
                )
            )

        # 3. Continuously extract C++ operand & host handler ASTs
        self.cpp_registry = CppHandlerRegistry(self.operands_path, self.handlers_path)
        self.cpp_eval = CppSymbolicEvaluator(self.cpp_registry)
        self._validate_completeness()

    @staticmethod
    def _load_xml_specs(xml_path: str) -> List[Dict[str, Any]]:
        root = ET.parse(xml_path).getroot()
        specs: List[Dict[str, Any]] = []
        for elem in root:
            if elem.tag != "instruction":
                continue
            since = int(elem.attrib.get("since", "9"))
            until = int(elem.attrib.get("until", "14"))
            arch_mask = 0
            for a in range(since, until + 1):
                arch_mask |= 1 << a
            if arch_mask == 0:
                continue
            srcs = [c.attrib.get("name", f"src{i}") for i, c in enumerate(elem.findall(".//src"))]
            mods = [m.attrib.get("name", "") for m in elem.findall(".//mod")]
            specs.append(
                {
                    "xml_index": len(specs),
                    "name": elem.attrib["name"],
                    "variant": elem.attrib.get("variant", ""),
                    "since": since,
                    "until": until,
                    "xml_srcs": srcs,
                    "xml_mods": mods,
                }
            )
        return specs

    def _validate_completeness(self) -> None:
        expected_abi = set(self.host_functions.keys())
        missing_abi = expected_abi - set(self.cpp_registry.abi_funcs.keys())
        if missing_abi:
            raise RuntimeError(f"Missing abi:: functions in {self.handlers_path} / {self.operands_path}: {sorted(missing_abi)}")
        for r in self.rules:
            if r.kind == "Handler" and r.target not in self.cpp_registry.global_handlers:
                raise RuntimeError(f"Missing C++ @HANDLER({r.target}) in {self.handlers_path}")
        for x in self.xml_specs:
            if self.match_rule(x["name"], x["variant"]) is None:
                raise RuntimeError(f"XML spec {x['name']}.{x['variant']} has no lowering rule")

    def _eval_when_cond(self, c: Any, ssa: SymbolicSsaModule, ins: InstLoweringContext) -> bool:
        if isinstance(c, CondBinOp):
            l = self._eval_when_cond(c.left, ssa, ins)
            if c.op == "&&":
                return l and self._eval_when_cond(c.right, ssa, ins)
            if c.op == "||":
                return l or self._eval_when_cond(c.right, ssa, ins)
            raise ValueError(f"Unsupported CondBinOp {c.op}")
        atom = c.atom
        if isinstance(atom, DollarCall):
            res = bool(self.cpp_eval.call_abi_fn(atom.name, ssa, ins, []))
        elif isinstance(atom, HostCall):
            sig = self.host_functions[atom.name]
            call_args: List[Any] = []
            for atype_raw, anode in zip(sig["args"], atom.args):
                atype = atype_raw.split()[0] if " %" in atype_raw else atype_raw
                if atype == "const char*":
                    if isinstance(anode, Literal) and anode.kind == "string":
                        call_args.append(anode.raw[1:-1])
                    elif isinstance(anode, BareIdent):
                        call_args.append(anode.name)
                elif atype == "uint32_t" and isinstance(anode, Literal):
                    raw = anode.raw.rstrip("uUlL")
                    call_args.append(int(raw, 16 if raw.startswith("0x") else 10))
                else:
                    raise ValueError(f"Unsupported when arg {anode} for {atype}")
            res = bool(self.cpp_eval.call_abi_fn(atom.name, ssa, ins, call_args))
        else:
            raise ValueError(f"Unknown when Cond atom: {atom}")
        return (not res) if c.negated else res

    def match_rule(
        self,
        mnemonic: str,
        variant: str = "",
        *,
        arch: Optional[int] = None,
        ins: Optional[InstLoweringContext] = None,
        ssa: Optional[SymbolicSsaModule] = None,
    ) -> Optional[ParsedLoweringRule]:
        exact_cands: List[ParsedLoweringRule] = []
        wild_cands: List[ParsedLoweringRule] = []
        for r in self.rules:
            if r.mnemonic != mnemonic:
                continue
            if arch is not None and not (r.arch_range[0] <= arch <= r.arch_range[1]):
                continue
            if variant in r.variants:
                exact_cands.append(r)
            elif "*" in r.variants:
                wild_cands.append(r)
        for r in exact_cands + wild_cands:
            if r.when_cond is not None and ins is not None:
                eval_ssa = ssa if ssa is not None else SymbolicSsaModule()
                if not self._eval_when_cond(r.when_cond, eval_ssa, ins):
                    continue
            return r
        return None

    # -----------------------------------------------------------------------
    # Helpers mirroring valhall_lowering_dsl.py & gen_valhall_tables.py
    # -----------------------------------------------------------------------
    @staticmethod
    def _is_v2f16(ins: InstLoweringContext) -> bool:
        return ins.canonical_variant == "v2f16" or ins.variant == "v2f16"

    @staticmethod
    def _is_signed(ins: InstLoweringContext) -> bool:
        v = ins.canonical_variant
        return v.startswith("s") or v.startswith("v2s") or v.startswith("v4s")

    def _resolve_valtype(self, vt_key: str, ins: InstLoweringContext) -> str:
        if vt_key == "fp_vt":
            return "V2F32" if self._is_v2f16(ins) else "F32"
        if vt_key == "v2i_vt":
            return "V2S32" if self._is_signed(ins) else "V2U32"
        if vt_key in VALTYPE_MAP:
            return VALTYPE_MAP[vt_key].replace("ValType::", "")
        return vt_key

    def _dest_valtype(self, dest_type: str, ins: InstLoweringContext) -> str:
        mapping = {
            "f32": "F32",
            "f32_or_v2f16": "V2F32" if self._is_v2f16(ins) else "F32",
            "u32": "U32",
            "s32": "S32",
            "u64": "U64",
            "v2f16": "V2F32",
            "v2s16": "V2S32",
            "v2u16": "V2U32",
            "v2i16_by_var": "V2S32" if self._is_signed(ins) else "V2U32",
            "none": "Void",
            "sr": "Void",
        }
        if dest_type not in mapping:
            raise ValueError(f"Unsupported dest_type '{dest_type}'")
        return mapping[dest_type]

    def _fetch_src(
        self, src_type: str, idx: int, ssa: SymbolicSsaModule, ins: InstLoweringContext, tr: LoweringTrace
    ) -> int:
        if idx >= len(ins.srcs):
            tr.warnings.append(f"Arity mismatch: rule reads %{idx} ({src_type}) but ins.srcs has {len(ins.srcs)}")
        tr.steps.append(f"fetch_src[{idx}] as {src_type}")
        if src_type == "f32":
            return self.cpp_eval.call_operand_fn("fetch_src_f32", ssa, ins, [idx])
        if src_type == "f32_or_v2f16":
            fn = "fetch_src_v2f16" if self._is_v2f16(ins) else "fetch_src_f32"
            return self.cpp_eval.call_operand_fn(fn, ssa, ins, [idx])
        if src_type == "u32":
            return self.cpp_eval.call_operand_fn("fetch_src_int32", ssa, ins, [idx, False])
        if src_type == "s32_raw":
            return self.cpp_eval.call_operand_fn("fetch_src_int32", ssa, ins, [idx, True])
        if src_type == "s32":
            raw = self.cpp_eval.call_operand_fn("fetch_src_int32", ssa, ins, [idx, True])
            return ssa.emit_bitcast("S32", raw)
        if src_type == "i32_by_var":
            return self.cpp_eval.call_operand_fn(
                "fetch_src_int32", ssa, ins, [idx, ins.canonical_variant.startswith("s")]
            )
        if src_type == "u64":
            return self.cpp_eval.call_operand_fn("fetch_src_u64", ssa, ins, [idx, False])
        if src_type == "s64":
            return self.cpp_eval.call_operand_fn("fetch_src_u64", ssa, ins, [idx, True])
        if src_type == "i64_by_var":
            return self.cpp_eval.call_operand_fn(
                "fetch_src_u64", ssa, ins, [idx, "s64" in ins.canonical_variant]
            )
        if src_type == "v2f16":
            return self.cpp_eval.call_operand_fn("fetch_src_v2f16", ssa, ins, [idx])
        if src_type == "v2s16":
            return self.cpp_eval.call_operand_fn("fetch_src_v2i16", ssa, ins, [idx, True])
        if src_type == "v2u16":
            return self.cpp_eval.call_operand_fn("fetch_src_v2i16", ssa, ins, [idx, False])
        if src_type == "v2i16_by_var":
            return self.cpp_eval.call_operand_fn("fetch_src_v2i16", ssa, ins, [idx, self._is_signed(ins)])
        if src_type == "v4i8_by_var":
            return self.cpp_eval.call_operand_fn("fetch_src_v4i8", ssa, ins, [idx, self._is_signed(ins)])
        if src_type == "v4s8":
            return self.cpp_eval.call_operand_fn("fetch_src_v4i8", ssa, ins, [idx, True])
        if src_type == "v4u8":
            return self.cpp_eval.call_operand_fn("fetch_src_v4i8", ssa, ins, [idx, False])
        raise ValueError(f"Unsupported src_type '{src_type}'")

    def _store_dest(
        self, dest_type: str, val_id: int, ssa: SymbolicSsaModule, ins: InstLoweringContext, tr: LoweringTrace
    ) -> None:
        if dest_type in ("none", "sr"):
            return
        tr.steps.append(f"store_dest as {dest_type} (%{val_id})")
        if dest_type == "f32":
            self.cpp_eval.call_operand_fn("store_dest_f32", ssa, ins, [val_id])
        elif dest_type == "f32_or_v2f16":
            fn = "store_dest_v2f16" if self._is_v2f16(ins) else "store_dest_f32"
            self.cpp_eval.call_operand_fn(fn, ssa, ins, [val_id])
        elif dest_type == "u32":
            self.cpp_eval.call_operand_fn("store_dest_u32", ssa, ins, [val_id])
        elif dest_type == "s32":
            bc = ssa.emit_bitcast("U32", val_id)
            self.cpp_eval.call_operand_fn("store_dest_u32", ssa, ins, [bc])
        elif dest_type == "u64":
            self.cpp_eval.call_operand_fn("store_dest_u64", ssa, ins, [val_id])
        elif dest_type == "v2f16":
            self.cpp_eval.call_operand_fn("store_dest_v2f16", ssa, ins, [val_id])
        elif dest_type == "v2s16":
            self.cpp_eval.call_operand_fn("store_dest_v2i16", ssa, ins, [val_id, True])
        elif dest_type == "v2u16":
            self.cpp_eval.call_operand_fn("store_dest_v2i16", ssa, ins, [val_id, False])
        elif dest_type == "v2i16_by_var":
            self.cpp_eval.call_operand_fn("store_dest_v2i16", ssa, ins, [val_id, self._is_signed(ins)])
        else:
            raise ValueError(f"Unsupported dest_type '{dest_type}'")

    def _lookup_table(self, table_name: str, key_expr: str, ins: InstLoweringContext) -> Any:
        tmap = self.lookup_tables[table_name]
        if key_expr.startswith("$modifiers."):
            mod_name = key_expr[len("$modifiers."):]
            k = ins.modifiers.get(mod_name, "")
        else:
            k = ins.modifiers.get(key_expr, "")
        raw = tmap.get(k, tmap.get("_default"))
        tkind = tmap.get("_kind")
        if tkind == "u32":
            return int(raw, 16) if isinstance(raw, str) and raw.startswith("0x") else int(raw)
        return raw

    # -----------------------------------------------------------------------
    # Lowering Execution (DirectSsaOp / DynamicSsaOp / @INLINE_OPS / @HANDLER)
    # -----------------------------------------------------------------------
    def lower_instruction(
        self, ins: InstLoweringContext, xml_index: Optional[int] = None
    ) -> Tuple[SymbolicSsaModule, LoweringTrace]:
        ssa = SymbolicSsaModule()
        rule = self.match_rule(ins.mnemonic, ins.variant, arch=ins.arch, ins=ins, ssa=ssa)
        if rule is None:
            raise KeyError(f"No lowering rule for {ins.mnemonic}.{ins.variant} (arch={ins.arch})")

        ins.rule_effects = rule.effects
        tr = LoweringTrace(
            mnemonic=ins.mnemonic,
            variant=ins.variant,
            rule_index=rule.rule_index,
            rule_kind=rule.kind,
            ssa_op_spec=str(rule.target),
            xml_index=xml_index,
        )

        if rule.kind == "Handler":
            tr.steps.append(f"call @HANDLER({rule.target})")
            self.cpp_eval.call_global_handler(rule.target, ssa, ins)
        elif rule.kind in ("DirectSsaOp", "DynamicSsaOp"):
            if rule.kind == "DirectSsaOp" and rule.target == "Nop":
                tr.steps.append("emit Nop (no-op rule)")
                tr.emitted_ops = []
                self.logger.record(tr)
                return ssa, tr
            if rule.kind == "DirectSsaOp":
                ssa_op = rule.target
            else:
                _, tname, kexpr = rule.target
                ssa_op = self._lookup_table(tname, kexpr, ins)
            args = [self._fetch_src(st, i, ssa, ins, tr) for i, st in enumerate(rule.src_types)]
            dvt = self._dest_valtype(rule.dest_type, ins)
            nid = ssa.emit(
                SymbolicSsaNode(id=0, op=ssa_op, val_type=dvt, args=args, effects=rule.effects)
            )
            tr.steps.append(f"emit {ssa_op}:{dvt}({args}) -> %{nid}")
            self._store_dest(rule.dest_type, nid, ssa, ins, tr)
        elif rule.kind == "InlineOps":
            assert rule.inline_program is not None
            self._exec_inline_program(rule.inline_program, rule, ssa, ins, tr)
        else:
            raise ValueError(f"Unknown rule kind '{rule.kind}'")

        tr.emitted_ops = [n.op for n in ssa.nodes[2:]]
        self.logger.record(tr)
        return ssa, tr

    def _exec_inline_program(
        self,
        prog: Program,
        rule: ParsedLoweringRule,
        ssa: SymbolicSsaModule,
        ins: InstLoweringContext,
        tr: LoweringTrace,
    ) -> None:
        env: Dict[str, Any] = {}
        var_types: Dict[str, str] = {}
        returned = False

        def exec_stmts(stmts: List[Any]) -> None:
            nonlocal returned
            for st in stmts:
                if returned:
                    return
                if isinstance(st, ReturnStmt):
                    if st.value is not None:
                        val = eval_expr(st.value, "Void")
                        self._store_dest(rule.dest_type, val, ssa, ins, tr)
                    returned = True
                    return
                elif isinstance(st, AssignStmt):
                    if st.val_type is not None:
                        vt = self._resolve_valtype(st.val_type, ins)
                        var_types[st.var] = vt
                    else:
                        vt = var_types.get(st.var, "U32")
                    env[st.var] = eval_expr(st.rhs, vt)
                elif isinstance(st, ExprStmt):
                    eval_expr(st.expr, "Void")
                elif isinstance(st, IfStmt):
                    if eval_cond(st.cond):
                        exec_stmts(st.then_body)
                    elif st.else_body is not None:
                        exec_stmts(st.else_body)
                elif isinstance(st, ForStmt):
                    bound = int(eval_arg(st.bound, "uint32_t", "Void"))
                    for i in range(bound):
                        env[st.var] = i
                        exec_stmts(st.body)
                        if returned:
                            return
                else:
                    raise ValueError(f"Unknown DSL stmt: {st}")

        def eval_cond(c: Any) -> bool:
            if isinstance(c, CondBinOp):
                l = eval_cond(c.left)
                return (l and eval_cond(c.right)) if c.op == "&&" else (l or eval_cond(c.right))
            atom = c.atom
            if isinstance(atom, DollarCall):
                res = bool(self.cpp_eval.call_abi_fn(atom.name, ssa, ins, []))
            elif isinstance(atom, HostCall):
                res = call_host(atom.name, atom.args, "Void")
                res = bool(res)
            else:
                raise ValueError(f"Unknown Cond atom: {atom}")
            return (not res) if c.negated else res

        def call_host(fn_name: str, arg_nodes: List[Any], lhs_vt: str) -> Any:
            sig = self.host_functions[fn_name]
            expected_args = sig["args"]
            call_args: List[Any] = []
            tok_idx = 0
            for atype_raw in expected_args:
                atype = atype_raw.split()[0] if " %" in atype_raw else atype_raw
                if atype == "ValType" and tok_idx < len(arg_nodes) and isinstance(arg_nodes[tok_idx], BareIdent) and arg_nodes[tok_idx].name in VALTYPE_MAP:
                    call_args.append(self._resolve_valtype(arg_nodes[tok_idx].name, ins))
                    tok_idx += 1
                elif atype == "ValType":
                    call_args.append(lhs_vt)
                else:
                    call_args.append(eval_arg(arg_nodes[tok_idx], atype, lhs_vt))
                    tok_idx += 1
            tr.steps.append(f"call abi::{fn_name}({call_args})")
            return self.cpp_eval.call_abi_fn(fn_name, ssa, ins, call_args)

        def eval_arg(node: Any, atype: str, lhs_vt: str) -> Any:
            if atype == "SsaId":
                return eval_expr(node, lhs_vt)
            if atype == "uint32_t":
                if isinstance(node, VarRef):
                    return int(env[node.name])
                if isinstance(node, DollarCall):
                    return int(self.cpp_eval.call_abi_fn(node.name, ssa, ins, []))
                if isinstance(node, HostCall):
                    return int(call_host(node.name, node.args, lhs_vt))
                if isinstance(node, HashExpr) and node.name == "lookup_u32_raw":
                    table_ident, key_node = node.args
                    return int(self._lookup_table(table_ident.name, f"$modifiers.{key_node.mod_name}", ins))
                if isinstance(node, Literal) and node.kind == "int":
                    raw = node.raw.rstrip("uUlL")
                    return int(raw, 16 if raw.startswith("0x") else 10)
            if atype in ("int32_t", "uint64_t") and isinstance(node, Literal):
                raw = node.raw.rstrip("uUlL")
                return int(raw, 16 if raw.startswith("0x") else 10)
            if atype == "float" and isinstance(node, Literal):
                return float(node.raw.rstrip("f"))
            if atype == "bool":
                if isinstance(node, Literal) and node.kind == "bool":
                    return node.raw == "true"
                if isinstance(node, Cond):
                    return eval_cond(node)
                if isinstance(node, (DollarCall, HostCall)):
                    return eval_cond(Cond(False, node))
            if atype == "const char*":
                if isinstance(node, Literal) and node.kind == "string":
                    return node.raw[1:-1]
                if isinstance(node, BareIdent):
                    return node.name
            raise ValueError(f"Unsupported arg {node} for type {atype}")

        def eval_flags(flags: List[Any]) -> Tuple[bool, bool, bool]:
            barrier = False
            side_eff = False
            aux_bool = False
            for fl in flags:
                if fl.name == "barrier":
                    barrier = True
                elif fl.name == "side_effect":
                    side_eff = True
                elif fl.name == "aux_bool":
                    if fl.aux_bool is True:
                        aux_bool = True
                    elif fl.aux_bool is False:
                        aux_bool = False
                    else:
                        aux_bool = eval_cond(fl.aux_bool)
            return barrier, side_eff, aux_bool

        def eval_expr(node: Any, lhs_vt: str) -> Any:
            if isinstance(node, SrcRef):
                st = rule.src_types[node.index] if node.index < len(rule.src_types) else "u32"
                return self._fetch_src(st, node.index, ssa, ins, tr)
            if isinstance(node, VarRef):
                return env[node.name]
            if isinstance(node, DollarCall):
                return call_host(node.name, [], lhs_vt)
            if isinstance(node, HashExpr):
                if node.name == "lookup_u32":
                    table_ident, key_node = node.args
                    val = int(self._lookup_table(table_ident.name, f"$modifiers.{key_node.mod_name}", ins))
                    return ssa.emit_const_u32(val)
                return call_host(node.name, node.args or [], lhs_vt)
            if isinstance(node, HostCall):
                return call_host(node.name, node.args, lhs_vt)
            if isinstance(node, LookupCall):
                op = self._lookup_table(node.table, f"$modifiers.{node.key.mod_name}", ins)
                arg_ids = [eval_expr(a, lhs_vt) for a in node.args]
                barrier, side_eff, _ = eval_flags(node.flags)
                nid = ssa.emit(
                    SymbolicSsaNode(
                        id=0, op=op, val_type=lhs_vt, args=arg_ids,
                        is_barrier_or_cf=barrier, is_side_effect=side_eff, effects=rule.effects
                    )
                )
                tr.steps.append(f"emit {op}:{lhs_vt}({arg_ids}) -> %{nid}")
                return nid
            if isinstance(node, SsaCall):
                arg_ids = [eval_expr(a, lhs_vt) for a in node.args]
                barrier, side_eff, aux_b = eval_flags(node.flags)
                nid = ssa.emit(
                    SymbolicSsaNode(
                        id=0, op=node.op, val_type=lhs_vt, args=arg_ids,
                        aux_bool=aux_b, is_barrier_or_cf=barrier, is_side_effect=side_eff,
                        effects=rule.effects
                    )
                )
                tr.steps.append(f"emit {node.op}:{lhs_vt}({arg_ids}) -> %{nid}")
                return nid
            raise ValueError(f"Unsupported DSL expr: {node}")

        exec_stmts(prog.stmts)
        if not returned and "out" in env and rule.dest_type != "none":
            self._store_dest(rule.dest_type, env["out"], ssa, ins, tr)

    def extract_all_xml_semantics(self) -> List[Dict[str, Any]]:
        """Extract symbolic SSA-IR translational semantics for all 627 XML specs."""
        results: List[Dict[str, Any]] = []
        for x in self.xml_specs:
            srcs = [SourceOperandDesc(reg_num=i) for i in range(len(x["xml_srcs"]))]
            ins = InstLoweringContext(
                mnemonic=x["name"],
                variant=x["variant"],
                srcs=srcs,
                has_dest=True,
                dest_reg=0,
                sr_read_present=True,
                sr_read_base=0,
                sr_read_count=4,
                sr_write_present=True,
                sr_write_base=0,
                sr_write_count=4,
            )
            ssa, tr = self.lower_instruction(ins, xml_index=x["xml_index"])
            results.append(
                {
                    "xml_index": x["xml_index"],
                    "mnemonic": x["name"],
                    "variant": x["variant"],
                    "rule_index": tr.rule_index,
                    "rule_kind": tr.rule_kind,
                    "node_count": len(ssa.nodes) - 2,
                    "emitted_ops": tr.emitted_ops,
                    "warnings": tr.warnings,
                    "ssa_module": ssa,
                    "trace": tr,
                }
            )
        return results

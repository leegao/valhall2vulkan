#!/usr/bin/env python3
"""
Continuous C++17 Host Handler & Operand Semantics Extractor
===========================================================
Parses C++17 source functions directly from:
  - src/valhall_operands.cpp       (get_src_*, get_dest_mod, fetch_src_*, store_dest_*)
  - src/valhall_pass1_handlers.cpp (Pass1Context::*, abi::* 76 host functions, handle_int_idpadd)

Produces:
  1. A structured C++ AST (`CppFunctionDef`) for every function in those files.
  2. A symbolic SSA-IR interpreter (`CppSymbolicEvaluator`) that executes the
     extracted C++ AST against a `LoweringContext` to emit `SymbolicSsaNode`s.

Because the C++ source is parsed dynamically on each run, any edits to
valhall_operands.cpp or valhall_pass1_handlers.cpp are automatically reflected
in the extracted translational semantics without manual synchronization.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
import struct
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# C++17 Tokenizer
# ---------------------------------------------------------------------------
_CPP_TOKEN_SPEC = [
    ("LINE_COMMENT", r"//[^\n]*"),
    ("BLOCK_COMMENT", r"/\*[\s\S]*?\*/"),
    ("PREPROC", r"#[^\n]*"),
    ("WS", r"\s+"),
    ("STRING", r'"(?:\\.|[^"\\])*"'),
    ("CHAR", r"'(?:\\.|[^'\\])*'"),
    ("SCOPE", r"::"),
    ("ARROW", r"->"),
    ("INC", r"\+\+"),
    ("DEC", r"--"),
    ("LE", r"<="),
    ("GE", r">="),
    ("EQ", r"=="),
    ("NE", r"!="),
    ("AND", r"&&"),
    ("OR", r"\|\|"),
    ("SHL", r"<<"),
    ("SHR", r">>"),
    ("OR_EQ", r"\|="),
    ("AND_EQ", r"&="),
    ("ADD_EQ", r"\+="),
    ("SUB_EQ", r"-="),
    ("FLOAT", r"-?[0-9]+\.[0-9]+f?"),
    ("HEX", r"0x[0-9a-fA-F]+(?:ULL|ull|UL|ul|u|U|LL|ll)?"),
    ("INT", r"[0-9]+(?:ULL|ull|UL|ul|u|U|LL|ll)?"),
    ("IDENT", r"[A-Za-z_][A-Za-z0-9_]*"),
    ("LBRACE", r"\{"),
    ("RBRACE", r"\}"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("LBRACK", r"\["),
    ("RBRACK", r"\]"),
    ("SEMI", r";"),
    ("COMMA", r","),
    ("DOT", r"\."),
    ("COLON", r":"),
    ("QMARK", r"\?"),
    ("ASSIGN", r"="),
    ("BANG", r"!"),
    ("TILDE", r"~"),
    ("AMP", r"&"),
    ("PIPE", r"\|"),
    ("CARET", r"\^"),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("STAR", r"\*"),
    ("SLASH", r"/"),
    ("PERCENT", r"%"),
    ("LT", r"<"),
    ("GT", r">"),
]
_CPP_MASTER_RE = re.compile("|".join(f"(?P<{n}>{p})" for n, p in _CPP_TOKEN_SPEC))

_CPP_KEYWORDS = {
    "auto", "bool", "break", "case", "char", "const", "constexpr", "continue",
    "default", "else", "false", "float", "for", "if", "inline", "int",
    "namespace", "return", "sizeof", "static", "struct", "switch", "true",
    "uint8_t", "uint32_t", "uint64_t", "int32_t", "int64_t", "size_t",
    "using", "void",
}


@dataclasses.dataclass(frozen=True)
class CppToken:
    kind: str
    val: str
    line: int


def tokenize_cpp(src: str) -> List[CppToken]:
    tokens: List[CppToken] = []
    pos = 0
    line = 1
    n = len(src)
    while pos < n:
        m = _CPP_MASTER_RE.match(src, pos)
        if not m:
            snippet = src[pos:pos + 32]
            raise SyntaxError(f"Unexpected C++ char at line {line}: {snippet!r}")
        kind = m.lastgroup
        val = m.group()
        if kind not in ("WS", "LINE_COMMENT", "BLOCK_COMMENT", "PREPROC"):
            if kind == "IDENT" and val in _CPP_KEYWORDS:
                kind = f"KW_{val.upper()}"
            tokens.append(CppToken(kind, val, line))
        line += val.count("\n")
        pos = m.end()
    tokens.append(CppToken("EOF", "", line))
    return tokens


# ---------------------------------------------------------------------------
# C++ AST Nodes
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
    source_file: str
    start_line: int
    end_line: int


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
    expr: Optional[Any] = None


@dataclasses.dataclass
class CppIf:
    cond: Any
    then_body: List[Any]
    else_body: Optional[List[Any]] = None


@dataclasses.dataclass
class CppFor:
    init: Optional[Any]
    cond: Optional[Any]
    step: Optional[Any]
    body: List[Any]


@dataclasses.dataclass
class CppSwitch:
    expr: Any
    cases: List[Tuple[Optional[Any], List[Any]]]  # None = default


@dataclasses.dataclass
class CppBreak:
    pass


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
    template_args: List[str] = dataclasses.field(default_factory=list)


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
    kind: str  # "int", "float", "string", "char", "bool"
    value: Any


@dataclasses.dataclass
class CppInitList:
    items: List[Any]


@dataclasses.dataclass
class CppLambda:
    params: List[CppParam]
    body: List[Any]


# ---------------------------------------------------------------------------
# Recursive-Descent C++17 Subset Parser
# ---------------------------------------------------------------------------
_TYPE_START_KWS = {
    "KW_VOID", "KW_BOOL", "KW_CHAR", "KW_INT", "KW_FLOAT",
    "KW_UINT8_T", "KW_UINT32_T", "KW_UINT64_T", "KW_INT32_T", "KW_INT64_T",
    "KW_SIZE_T", "KW_AUTO", "KW_CONST", "KW_CONSTEXPR", "KW_STATIC", "KW_INLINE",
}


class CppParser:
    def __init__(self, tokens: List[CppToken], source_file: str = "") -> None:
        self.tokens = tokens
        self.idx = 0
        self.source_file = source_file

    def peek(self, offset: int = 0) -> CppToken:
        i = min(self.idx + offset, len(self.tokens) - 1)
        return self.tokens[i]

    def advance(self) -> CppToken:
        t = self.tokens[self.idx]
        self.idx += 1
        return t

    def match(self, kind: str, val: Optional[str] = None) -> bool:
        t = self.peek()
        if t.kind == kind and (val is None or t.val == val):
            self.advance()
            return True
        return False

    def expect(self, kind: str, val: Optional[str] = None) -> CppToken:
        t = self.peek()
        if t.kind != kind or (val is not None and t.val != val):
            raise SyntaxError(
                f"{self.source_file}:{t.line}: Expected {kind}({val}), got {t.kind}({t.val!r})"
            )
        return self.advance()

    def _skip_balanced(self, open_k: str, close_k: str) -> None:
        self.expect(open_k)
        depth = 1
        while depth > 0 and self.peek().kind != "EOF":
            t = self.advance()
            if t.kind == open_k:
                depth += 1
            elif t.kind == close_k:
                depth -= 1

    def _is_type_token(self, tok: CppToken) -> bool:
        return tok.kind in _TYPE_START_KWS or tok.kind == "IDENT"

    def _parse_type_str(self) -> str:
        parts: List[str] = []
        while True:
            t = self.peek()
            if t.kind in (
                "KW_CONST", "KW_CONSTEXPR", "KW_STATIC", "KW_INLINE", "KW_AUTO",
                "KW_VOID", "KW_BOOL", "KW_CHAR", "KW_INT", "KW_FLOAT",
                "KW_UINT8_T", "KW_UINT32_T", "KW_UINT64_T", "KW_INT32_T", "KW_INT64_T",
                "KW_SIZE_T",
            ):
                parts.append(self.advance().val)
            elif t.kind == "IDENT":
                nxt = self.peek(1)
                qual = (not parts or parts[-1] in ("const", "constexpr", "static", "inline"))
                if qual and nxt.kind in ("SCOPE", "LT"):
                    parts.append(self.advance().val)
                    if self.match("SCOPE"):
                        parts.append("::")
                    elif self.peek().kind == "LT":
                        parts.append(self._collect_template_str())
                elif parts and parts[-1] == "::":
                    parts.append(self.advance().val)
                    if self.peek().kind == "LT":
                        parts.append(self._collect_template_str())
                    break
                elif qual:
                    parts.append(self.advance().val)
                    break
                else:
                    break
            elif t.kind in ("STAR", "AMP"):
                parts.append(self.advance().val)
            else:
                break
        # Handle trailing * or & after type ident
        while self.peek().kind in ("STAR", "AMP"):
            parts.append(self.advance().val)
        return " ".join(parts).replace(" :: ", "::")

    def _collect_template_str(self) -> str:
        self.expect("LT")
        depth = 1
        items: List[str] = ["<"]
        while depth > 0 and self.peek().kind != "EOF":
            t = self.advance()
            if t.kind == "LT":
                depth += 1
            elif t.kind == "GT":
                depth -= 1
            items.append(t.val)
        return "".join(items)

    def parse_file(self) -> List[CppFunctionDef]:
        funcs: List[CppFunctionDef] = []
        self._parse_namespace_body(funcs, "")
        return funcs

    def _parse_namespace_body(self, funcs: List[CppFunctionDef], ns: str) -> None:
        while self.peek().kind != "EOF":
            t = self.peek()
            if t.kind == "RBRACE":
                return
            if t.kind == "KW_NAMESPACE":
                self.advance()
                if self.peek().kind == "IDENT":
                    ns_name = self.advance().val
                    while self.match("SCOPE"):
                        ns_name += "::" + self.expect("IDENT").val
                else:
                    ns_name = ""
                if self.match("LBRACE"):
                    sub_ns = ns_name if ns_name == "abi" else ns
                    self._parse_namespace_body(funcs, sub_ns)
                    self.expect("RBRACE")
                else:
                    self._skip_until_semi()
                continue
            if t.kind == "KW_USING":
                self._skip_until_semi()
                continue
            # Try parsing a top-level function definition
            saved = self.idx
            fn = self._try_parse_function_def(ns)
            if fn is not None:
                funcs.append(fn)
            else:
                self.idx = saved
                # Skip declaration or statement
                self._skip_top_item()

    def _skip_until_semi(self) -> None:
        while self.peek().kind not in ("SEMI", "EOF"):
            if self.peek().kind == "LBRACE":
                self._skip_balanced("LBRACE", "RBRACE")
            else:
                self.advance()
        self.match("SEMI")

    def _skip_top_item(self) -> None:
        while self.peek().kind not in ("SEMI", "LBRACE", "EOF", "RBRACE"):
            self.advance()
        if self.peek().kind == "LBRACE":
            self._skip_balanced("LBRACE", "RBRACE")
        else:
            self.match("SEMI")

    def _try_parse_function_def(self, ns: str) -> Optional[CppFunctionDef]:
        start_line = self.peek().line
        while self.peek().kind in ("KW_STATIC", "KW_INLINE"):
            self.advance()
        if not self._is_type_token(self.peek()):
            return None
        ret_type = self._parse_type_str()
        if not ret_type or self.peek().kind != "IDENT":
            return None
        fn_ns = ns
        fn_name = self.advance().val
        while self.match("SCOPE"):
            fn_ns = fn_name
            fn_name = self.expect("IDENT").val
        if not self.match("LPAREN"):
            return None
        params: List[CppParam] = []
        if self.peek().kind != "RPAREN":
            while True:
                ptype = self._parse_type_str()
                pname = ""
                if self.peek().kind == "IDENT":
                    pname = self.advance().val
                params.append(CppParam(ptype, pname))
                if not self.match("COMMA"):
                    break
        self.expect("RPAREN")
        # Optional const / trailing
        self.match("KW_CONST")
        if self.peek().kind != "LBRACE":
            return None
        self.advance()
        body = self._parse_block_stmts()
        end_line = self.peek().line
        self.expect("RBRACE")
        return CppFunctionDef(
            name=fn_name,
            namespace=fn_ns,
            ret_type=ret_type,
            params=params,
            body=body,
            source_file=self.source_file,
            start_line=start_line,
            end_line=end_line,
        )

    def _parse_block_stmts(self) -> List[Any]:
        stmts: List[Any] = []
        while self.peek().kind not in ("RBRACE", "EOF"):
            stmts.append(self._parse_stmt())
        return stmts

    def _parse_stmt_or_block(self) -> List[Any]:
        if self.match("LBRACE"):
            stmts = self._parse_block_stmts()
            self.expect("RBRACE")
            return stmts
        return [self._parse_stmt()]

    def _parse_stmt(self) -> Any:
        t = self.peek()
        if t.kind == "LBRACE":
            self.advance()
            stmts = self._parse_block_stmts()
            self.expect("RBRACE")
            return stmts[0] if len(stmts) == 1 else CppIf(CppLiteral("bool", True), stmts, None)
        if t.kind == "KW_RETURN":
            self.advance()
            if self.match("SEMI"):
                return CppReturn(None)
            expr = self._parse_expr()
            self.expect("SEMI")
            return CppReturn(expr)
        if t.kind == "KW_IF":
            self.advance()
            self.expect("LPAREN")
            cond = self._parse_expr()
            self.expect("RPAREN")
            then_b = self._parse_stmt_or_block()
            else_b = None
            if self.match("KW_ELSE"):
                else_b = self._parse_stmt_or_block()
            return CppIf(cond, then_b, else_b)
        if t.kind == "KW_FOR":
            self.advance()
            self.expect("LPAREN")
            # Range-based for? e.g. for (const char* prefix : {"widen", ...}) or for (const auto& s : ...)
            if self._looks_like_range_for():
                vtype = self._parse_type_str()
                vname = self.expect("IDENT").val
                self.expect("COLON")
                rng = self._parse_expr()
                self.expect("RPAREN")
                body = self._parse_stmt_or_block()
                return CppFor(CppVarDecl(vtype, vname, rng), None, None, body)
            init = None
            if not self.match("SEMI"):
                init = self._parse_decl_or_expr()
                self.expect("SEMI")
            cond = None
            if not self.match("SEMI"):
                cond = self._parse_expr()
                self.expect("SEMI")
            step = None
            if self.peek().kind != "RPAREN":
                step = self._parse_expr()
            self.expect("RPAREN")
            body = self._parse_stmt_or_block()
            return CppFor(init, cond, step, body)
        if t.kind == "KW_SWITCH":
            self.advance()
            self.expect("LPAREN")
            expr = self._parse_expr()
            self.expect("RPAREN")
            self.expect("LBRACE")
            cases: List[Tuple[Optional[Any], List[Any]]] = []
            while self.peek().kind not in ("RBRACE", "EOF"):
                if self.match("KW_CASE"):
                    cval = self._parse_expr()
                    self.expect("COLON")
                elif self.match("KW_DEFAULT"):
                    cval = None
                    self.expect("COLON")
                else:
                    # Statement inside current case
                    s = self._parse_stmt()
                    if cases:
                        cases[-1][1].append(s)
                    continue
                cstmts: List[Any] = []
                while self.peek().kind not in ("KW_CASE", "KW_DEFAULT", "RBRACE", "EOF"):
                    if self.peek().kind == "LBRACE":
                        self.advance()
                        cstmts.extend(self._parse_block_stmts())
                        self.expect("RBRACE")
                    else:
                        cstmts.append(self._parse_stmt())
                cases.append((cval, cstmts))
            self.expect("RBRACE")
            return CppSwitch(expr, cases)
        if t.kind == "KW_BREAK":
            self.advance()
            self.expect("SEMI")
            return CppBreak()
        res = self._parse_decl_or_expr()
        self.expect("SEMI")
        return res

    def _looks_like_range_for(self) -> bool:
        saved = self.idx
        depth = 0
        found_colon = False
        while self.peek().kind != "EOF":
            tk = self.advance()
            if tk.kind == "LPAREN":
                depth += 1
            elif tk.kind == "RPAREN":
                if depth == 0:
                    break
                depth -= 1
            elif tk.kind == "SEMI" and depth == 0:
                break
            elif tk.kind == "COLON" and depth == 0:
                found_colon = True
                break
        self.idx = saved
        return found_colon

    def _looks_like_var_decl(self) -> bool:
        t = self.peek()
        if t.kind in _TYPE_START_KWS:
            return True
        if t.kind == "IDENT":
            nxt = self.peek(1)
            if nxt.kind in ("IDENT", "STAR", "AMP"):
                return True
            if nxt.kind == "SCOPE":
                # e.g. std::string var = ...; vs std::memcpy(...)
                saved = self.idx
                self._parse_type_str()
                is_decl = self.peek().kind == "IDENT"
                self.idx = saved
                return is_decl
        return False

    def _parse_decl_or_expr(self) -> Any:
        if self._looks_like_var_decl():
            vtype = self._parse_type_str()
            decls: List[CppVarDecl] = []
            while True:
                vname = self.expect("IDENT").val
                init = None
                is_zero = False
                if self.match("LBRACE"):
                    if self.match("RBRACE"):
                        is_zero = True
                    else:
                        items = [self._parse_expr()]
                        while self.match("COMMA"):
                            items.append(self._parse_expr())
                        self.expect("RBRACE")
                        init = CppInitList(items)
                elif self.match("LPAREN"):
                    args = []
                    if self.peek().kind != "RPAREN":
                        args.append(self._parse_expr())
                        while self.match("COMMA"):
                            args.append(self._parse_expr())
                    self.expect("RPAREN")
                    init = args[0] if len(args) == 1 else CppInitList(args)
                elif self.match("ASSIGN"):
                    init = self._parse_expr()
                decls.append(CppVarDecl(vtype, vname, init, is_zero))
                if not self.match("COMMA"):
                    break
            return decls[0] if len(decls) == 1 else CppMultiVarDecl(decls)
        expr = self._parse_expr()
        return expr if isinstance(expr, CppAssign) else CppExprStmt(expr)

    # Precedence-climbing expression parser
    def _parse_expr(self) -> Any:
        return self._parse_assign()

    def _parse_assign(self) -> Any:
        lhs = self._parse_ternary()
        t = self.peek()
        if t.kind in ("ASSIGN", "OR_EQ", "AND_EQ", "ADD_EQ", "SUB_EQ"):
            op = self.advance().val
            rhs = self._parse_assign()
            return CppAssign(lhs, op, rhs)
        return lhs

    def _parse_ternary(self) -> Any:
        cond = self._parse_or()
        if self.match("QMARK"):
            then_e = self._parse_expr()
            self.expect("COLON")
            else_e = self._parse_ternary()
            return CppTernary(cond, then_e, else_e)
        return cond

    def _parse_or(self) -> Any:
        left = self._parse_and()
        while self.match("OR"):
            left = CppBinary("||", left, self._parse_and())
        return left

    def _parse_and(self) -> Any:
        left = self._parse_bitor()
        while self.match("AND"):
            left = CppBinary("&&", left, self._parse_bitor())
        return left

    def _parse_bitor(self) -> Any:
        left = self._parse_bitxor()
        while self.match("PIPE"):
            left = CppBinary("|", left, self._parse_bitxor())
        return left

    def _parse_bitxor(self) -> Any:
        left = self._parse_bitand()
        while self.match("CARET"):
            left = CppBinary("^", left, self._parse_bitand())
        return left

    def _parse_bitand(self) -> Any:
        left = self._parse_eq()
        while self.match("AMP"):
            left = CppBinary("&", left, self._parse_eq())
        return left

    def _parse_eq(self) -> Any:
        left = self._parse_rel()
        while self.peek().kind in ("EQ", "NE"):
            op = self.advance().val
            left = CppBinary(op, left, self._parse_rel())
        return left

    def _parse_rel(self) -> Any:
        left = self._parse_shift()
        while self.peek().kind in ("LT", "GT", "LE", "GE"):
            op = self.advance().val
            left = CppBinary(op, left, self._parse_shift())
        return left

    def _parse_shift(self) -> Any:
        left = self._parse_add()
        while self.peek().kind in ("SHL", "SHR"):
            op = self.advance().val
            left = CppBinary(op, left, self._parse_add())
        return left

    def _parse_add(self) -> Any:
        left = self._parse_mul()
        while self.peek().kind in ("PLUS", "MINUS"):
            op = self.advance().val
            left = CppBinary(op, left, self._parse_mul())
        return left

    def _parse_mul(self) -> Any:
        left = self._parse_unary()
        while self.peek().kind in ("STAR", "SLASH", "PERCENT"):
            op = self.advance().val
            left = CppBinary(op, left, self._parse_unary())
        return left

    def _parse_unary(self) -> Any:
        t = self.peek()
        if t.kind in ("BANG", "TILDE", "MINUS", "PLUS", "STAR", "AMP", "INC", "DEC"):
            op = self.advance().val
            return CppUnary(op, self._parse_unary())
        if t.kind == "KW_SIZEOF":
            self.advance()
            self.expect("LPAREN")
            tp = self._parse_type_str()
            self.expect("RPAREN")
            return CppCall(CppIdent("sizeof"), [CppLiteral("string", tp)])
        return self._parse_postfix()

    def _parse_postfix(self) -> Any:
        expr = self._parse_primary()
        while True:
            t = self.peek()
            if t.kind == "LPAREN":
                self.advance()
                args = self._parse_arg_list()
                self.expect("RPAREN")
                expr = CppCall(expr, args)
            elif t.kind == "DOT":
                self.advance()
                mem = self.expect("IDENT").val
                expr = CppMember(expr, mem, is_arrow=False)
            elif t.kind == "ARROW":
                self.advance()
                mem = self.expect("IDENT").val
                expr = CppMember(expr, mem, is_arrow=True)
            elif t.kind == "LBRACK":
                self.advance()
                idx_e = self._parse_expr()
                self.expect("RBRACK")
                expr = CppIndex(expr, idx_e)
            elif t.kind == "LT" and isinstance(expr, (CppIdent, CppScopedIdent)):
                saved = self.idx
                try:
                    targs_str = self._collect_template_str()
                    if self.peek().kind == "LPAREN":
                        self.advance()
                        args = self._parse_arg_list()
                        self.expect("RPAREN")
                        expr = CppCall(expr, args, [targs_str])
                        continue
                except SyntaxError:
                    pass
                self.idx = saved
                break
            elif t.kind in ("INC", "DEC"):
                op = self.advance().val
                expr = CppUnary(op, expr, postfix=True)
            else:
                break
        return expr

    def _parse_arg_list(self) -> List[Any]:
        args: List[Any] = []
        if self.peek().kind == "RPAREN":
            return args
        args.append(self._parse_expr())
        while self.match("COMMA"):
            args.append(self._parse_expr())
        return args

    def _parse_primary(self) -> Any:
        t = self.peek()
        if t.kind == "LPAREN":
            self.advance()
            expr = self._parse_expr()
            self.expect("RPAREN")
            return expr
        if t.kind == "LBRACE":
            self.advance()
            items: List[Any] = []
            if self.peek().kind != "RBRACE":
                items.append(self._parse_expr())
                while self.match("COMMA"):
                    if self.peek().kind == "RBRACE":
                        break
                    items.append(self._parse_expr())
            self.expect("RBRACE")
            return CppInitList(items)
        if t.kind == "LBRACK":
            # Lambda: [&](...) -> Ret { ... }
            self._skip_balanced("LBRACK", "RBRACK")
            params: List[CppParam] = []
            if self.match("LPAREN"):
                if self.peek().kind != "RPAREN":
                    while True:
                        pt = self._parse_type_str()
                        pn = self.expect("IDENT").val if self.peek().kind == "IDENT" else ""
                        params.append(CppParam(pt, pn))
                        if not self.match("COMMA"):
                            break
                self.expect("RPAREN")
            if self.match("ARROW"):
                self._parse_type_str()
            self.expect("LBRACE")
            body = self._parse_block_stmts()
            self.expect("RBRACE")
            return CppLambda(params, body)
        if t.kind in ("HEX", "INT"):
            raw = self.advance().val.rstrip("uUlL")
            return CppLiteral("int", int(raw, 16 if raw.startswith("0x") else 10))
        if t.kind == "FLOAT":
            raw = self.advance().val.rstrip("f")
            return CppLiteral("float", float(raw))
        if t.kind == "STRING":
            val = self.advance().val[1:-1]
            return CppLiteral("string", val)
        if t.kind == "CHAR":
            val = self.advance().val[1:-1]
            return CppLiteral("char", val)
        if t.kind == "KW_TRUE":
            self.advance()
            return CppLiteral("bool", True)
        if t.kind == "KW_FALSE":
            self.advance()
            return CppLiteral("bool", False)
        if t.kind in ("KW_UINT32_T", "KW_UINT64_T", "KW_INT32_T", "KW_INT64_T", "KW_BOOL", "KW_FLOAT"):
            # Function-style cast e.g. uint64_t(x)
            name = self.advance().val
            return CppIdent(name)
        if t.kind == "IDENT":
            first = self.advance().val
            if self.peek().kind == "SCOPE":
                parts = [first]
                while self.match("SCOPE"):
                    parts.append(self.expect("IDENT").val)
                return CppScopedIdent(parts)
            return CppIdent(first)
        raise SyntaxError(f"{self.source_file}:{t.line}: Unexpected C++ token {t.kind}({t.val!r})")


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
            # id=0 is kInvalidSsa, id=1 is initial token
            self.nodes.append(SymbolicSsaNode(id=0, op="Invalid", val_type="Void"))
            self.nodes.append(SymbolicSsaNode(id=1, op="Nop", val_type="Token"))
            # Symbolic initial register values r0..r63 are represented as InputReg nodes on demand
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
    stage: str = "Compute"  # Compute, Vertex, Fragment, Pilot
    rule_effects: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.canonical_mnemonic:
            self.canonical_mnemonic = self.mnemonic
        if not self.canonical_variant:
            self.canonical_variant = self.variant


# ---------------------------------------------------------------------------
# Extracted C++ Registry & Symbolic Evaluator
# ---------------------------------------------------------------------------
class _ReturnSignal(Exception):
    def __init__(self, value: Any) -> None:
        self.value = value


class _BreakSignal(Exception):
    pass


class _DictRef:
    """Models C++ std::map / iterator returned by .find(key)."""
    def __init__(self, d: Dict[str, Any], key: str) -> None:
        self.d = d
        self.key = key

    @property
    def found(self) -> bool:
        return self.key in self.d

    @property
    def second(self) -> Any:
        return self.d.get(self.key)


class _PointerOut:
    """Models C++ &var out-parameters (e.g. emit_split_u64(..., &lo, &hi))."""
    def __init__(self, env: Dict[str, Any], name: str) -> None:
        self.env = env
        self.name = name

    def set(self, val: Any) -> None:
        self.env[self.name] = val


class CppHandlerRegistry:
    """Holds all C++17 ASTs extracted from valhall_operands.cpp & valhall_pass1_handlers.cpp."""

    def __init__(
        self,
        operands_path: str,
        handlers_path: str,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.log = logger or logging.getLogger("cpp_extractor")
        self.operands_funcs: Dict[str, CppFunctionDef] = {}
        self.abi_funcs: Dict[str, CppFunctionDef] = {}
        self.ctx_funcs: Dict[str, CppFunctionDef] = {}
        self.global_handlers: Dict[str, CppFunctionDef] = {}

        for path, is_handlers in ((operands_path, False), (handlers_path, True)):
            with open(path, "r", encoding="utf-8") as f:
                src = f.read()
            toks = tokenize_cpp(src)
            fns = CppParser(toks, source_file=os.path.basename(path)).parse_file()
            for fn in fns:
                if not is_handlers:
                    self.operands_funcs[fn.name] = fn
                elif fn.namespace == "abi":
                    self.abi_funcs[fn.name] = fn
                elif fn.namespace == "Pass1Context":
                    self.ctx_funcs[fn.name] = fn
                else:
                    self.global_handlers[fn.name] = fn
            self.log.debug(
                "Extracted %d functions from %s", len(fns), os.path.basename(path)
            )

    def summary(self) -> Dict[str, Any]:
        return {
            "operands_funcs": sorted(self.operands_funcs.keys()),
            "abi_funcs": sorted(self.abi_funcs.keys()),
            "ctx_funcs": sorted(self.ctx_funcs.keys()),
            "global_handlers": sorted(self.global_handlers.keys()),
        }


class CppSymbolicEvaluator:
    """Executes extracted C++17 function ASTs to emit SymbolicSsaNodes."""

    def __init__(self, registry: CppHandlerRegistry) -> None:
        self.reg = registry

    def call_operand_fn(
        self, name: str, ssa: SymbolicSsaModule, ins: InstLoweringContext, args: List[Any]
    ) -> Any:
        fn = self.reg.operands_funcs[name]
        # Operand functions take (SsaBuilder& sb, uint32_t blk_id, const DecodedValhallInst& ins, ...)
        # or (const DecodedValhallInst& ins, ...)
        env: Dict[str, Any] = {}
        if len(args) == len(fn.params):
            for p, a in zip(fn.params, args):
                env[p.name] = a
        else:
            arg_idx = 0
            for p in fn.params:
                if "SsaBuilder" in p.type_str:
                    env[p.name] = ("sb", ssa)
                elif "blk" in p.name:
                    env[p.name] = 0
                elif "DecodedValhallInst" in p.type_str:
                    env[p.name] = ins
                else:
                    env[p.name] = args[arg_idx]
                    arg_idx += 1
        return self._exec_fn(fn, env, ssa, ins)

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
            arg_idx = 0
            for p in fn.params:
                if "Pass1Context" in p.type_str:
                    env[p.name] = ("ctx", ssa, ins)
                elif p.name == "blk" or ("uint32_t" in p.type_str and not p.name and arg_idx == 0 and len(fn.params) >= 4):
                    env[p.name or "_blk"] = 0
                elif "DecodedValhallInst" in p.type_str:
                    env[p.name or "_ins"] = ins
                elif "LoweringRuleSpec" in p.type_str:
                    env[p.name or "_rule"] = ("rule", ins)
                else:
                    env[p.name] = args[arg_idx]
                    arg_idx += 1
        return self._exec_fn(fn, env, ssa, ins)

    def call_global_handler(
        self, name: str, ssa: SymbolicSsaModule, ins: InstLoweringContext
    ) -> Any:
        fn = self.reg.global_handlers[name]
        env: Dict[str, Any] = {}
        for p in fn.params:
            if "Pass1Context" in p.type_str:
                env[p.name] = ("ctx", ssa, ins)
            elif p.name == "blk":
                env[p.name] = 0
            elif "DecodedValhallInst" in p.type_str:
                env[p.name] = ins
            elif "LoweringRuleSpec" in p.type_str:
                env[p.name or "_rule"] = ("rule", ins)
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
        elif isinstance(st, CppIf):
            if self._eval_expr(st.cond, env, ssa, ins):
                self._exec_stmts(st.then_body, env, ssa, ins)
            elif st.else_body is not None:
                self._exec_stmts(st.else_body, env, ssa, ins)
        elif isinstance(st, CppFor):
            if st.cond is None and st.step is None and isinstance(st.init, CppVarDecl):
                # Range-based for
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
            matched = False
            for cval, cstmts in st.cases:
                if not matched:
                    if cval is None or val == self._eval_expr(cval, env, ssa, ins):
                        matched = True
                if matched:
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
            elif isinstance(obj, tuple) and obj[0] == "blk" and target.member == "branch_cond":
                obj[1].branch_cond = val
            elif isinstance(obj, tuple) and obj[0] == "ctx":
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
            if expr.name in env:
                return env[expr.name]
            # Implicit `this->...` inside Pass1Context methods
            if "this_ctx" in env:
                if expr.name == "sb":
                    return ("sb", ssa)
                if expr.name == "opts":
                    return ("opts", ins)
                if expr.name == "res":
                    return ("res", ssa)
            return expr.name
        if isinstance(expr, CppScopedIdent):
            # Enums like ValType::U32, SsaOp::IAdd, SourceKind::Register, ShaderStage::Fragment, spirv::BuiltIn::Position
            return expr.parts[-1]
        if isinstance(expr, CppInitList):
            return [self._eval_expr(x, env, ssa, ins) for x in expr.items]
        if isinstance(expr, CppLambda):
            return ("lambda", expr, env)
        if isinstance(expr, CppUnary):
            if expr.op == "&" and isinstance(expr.operand, CppIdent):
                return _PointerOut(env, expr.operand.name)
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
                # it != map.end()
                return (l.found) if expr.op == "!=" else (not l.found)
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
            if isinstance(obj, str):
                return obj[idx]
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
        if isinstance(obj, tuple) and obj[0] == "opts" and m == "stage":
            return obj[1].stage
        if isinstance(obj, tuple) and obj[0] == "rule" and m == "effects":
            return obj[1].rule_effects
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
            if isinstance(obj, (list, dict, str)) and m in ("size", "empty", "find", "count", "at", "back"):
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
            if isinstance(obj, dict) and m == "end":
                return ("end", obj)
            if isinstance(obj, tuple) and obj[0] == "sb":
                sb_ssa: SymbolicSsaModule = obj[1]
                if m == "current_reg":
                    return sb_ssa.current_reg(args[1])
                if m == "set_reg":
                    sb_ssa.set_reg(args[1], args[2])
                    return None
                if m == "current_token":
                    return sb_ssa.token_out
                if m == "set_token":
                    sb_ssa.token_out = args[1]
                    return None
                if m == "emit":
                    return sb_ssa.emit(args[1])
                if m == "emit_const_u32":
                    return sb_ssa.emit_const_u32(args[1])
                if m == "emit_const_s32":
                    return sb_ssa.emit_const_s32(args[1])
                if m == "emit_const_f32_bits":
                    return sb_ssa.emit_const_f32_bits(args[1])
                if m == "emit_const_u64":
                    return sb_ssa.emit_const_u64(args[1])
                if m == "emit_const_bool":
                    return sb_ssa.emit_const_bool(args[1])
                if m == "emit_bitcast":
                    return sb_ssa.emit_bitcast(args[1], args[2])
            if isinstance(obj, tuple) and obj[0] == "ctx":
                return self._call_ctx_method(m, ssa, ins, args)
            if isinstance(obj, tuple) and obj[0] == "blk" and m == "branch_cond":
                return None

        # Free function or lambda or std::* call
        if isinstance(expr.callee, CppScopedIdent):
            fn_name = "::".join(expr.callee.parts)
            args = [self._eval_expr(a, env, ssa, ins) for a in expr.args]
            if fn_name == "std::move":
                return args[0]
            if fn_name == "std::string":
                return str(args[0])
            if fn_name == "std::to_string":
                return str(args[0])
            if fn_name == "std::max":
                return max(args[0], args[1])
            if fn_name == "std::min":
                return min(args[0], args[1])
            if fn_name == "std::strcmp":
                return 0 if args[0] == args[1] else 1
            if fn_name == "std::memcpy":
                # std::memcpy(&bits, &v, sizeof(float))
                if isinstance(args[0], _PointerOut) and isinstance(args[1], _PointerOut):
                    fval = float(args[1].env[args[1].name])
                    bits = struct.unpack("<I", struct.pack("<f", fval))[0]
                    args[0].set(bits)
                return None

        if isinstance(expr.callee, CppIdent):
            fn_name = expr.callee.name
            args = [self._eval_expr(a, env, ssa, ins) for a in expr.args]
            if fn_name in ("uint32_t", "uint64_t", "int32_t", "int64_t", "bool", "static_cast"):
                if fn_name == "bool":
                    return bool(args[0])
                if isinstance(args[0], str):
                    return args[0]
                return int(args[0])
            if fn_name == "sizeof":
                return 4
            if fn_name == "variant_to_string":
                return ins.canonical_variant
            # Lambda call
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
            # Pass1Context implicit this method call
            if "this_ctx" in env and fn_name in self.reg.ctx_funcs:
                return self._call_ctx_method(fn_name, ssa, ins, args)
            # Operand helper call
            if fn_name in self.reg.operands_funcs:
                return self.call_operand_fn(fn_name, ssa, ins, args)
            # ABI helper call
            if fn_name in self.reg.abi_funcs:
                return self.call_abi_fn(fn_name, ssa, ins, args)

        # Handle static_cast<T>(x) parsed as CppCall(CppIdent("static_cast"), ...)
        raise ValueError(f"Unsupported C++ call: {expr.callee}")

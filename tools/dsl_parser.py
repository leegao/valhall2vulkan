#!/usr/bin/env python3
"""
Strict LL(1) Tokenizer & Recursive-Descent Parser for the Valhall Pass 1 Lowering DSL.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union
import re


# ---------------------------------------------------------------------------
# Lexer / Tokenizer
# ---------------------------------------------------------------------------

KEYWORDS = {"if", "else", "for", "return", "true", "false"}

TOKEN_SPEC = [
    ("WS",         r"\s+"),
    ("DOTDOT",     r"\.\."),
    ("ANDAND",     r"&&"),
    ("OROR",       r"\|\|"),
    ("SRC_REF",    r"%[0-9]+"),
    ("VAR_REF",    r"%[A-Za-z_][A-Za-z0-9_]*"),
    ("FLOAT_LIT",  r"-?[0-9]+\.[0-9]+f?"),
    ("INT_LIT",    r"(?:0x[0-9a-fA-F]+|-?[0-9]+)(?:ULL|ull|u|U)?"),
    ("STRING_LIT", r'"[^"]*"'),
    ("IDENT",      r"[A-Za-z_][A-Za-z0-9_]*"),
    ("LBRACE",     r"\{"),
    ("RBRACE",     r"\}"),
    ("LPAREN",     r"\("),
    ("RPAREN",     r"\)"),
    ("LBRACK",     r"\["),
    ("RBRACK",     r"\]"),
    ("COLON",      r":"),
    ("SEMI",       r";"),
    ("COMMA",      r","),
    ("EQ",         r"="),
    ("BANG",       r"!"),
    ("AT",         r"@"),
    ("HASH",       r"#"),
    ("DOLLAR",     r"\$"),
    ("DOT",        r"\."),
]

MASTER_RE = re.compile("|".join(f"(?P<{name}>{pat})" for name, pat in TOKEN_SPEC))


@dataclass(frozen=True)
class Token:
    kind: str
    val: str
    pos: int


class DslSyntaxError(SyntaxError):
    pass


def tokenize(src: str) -> List[Token]:
    tokens: List[Token] = []
    pos = 0
    n = len(src)
    while pos < n:
        m = MASTER_RE.match(src, pos)
        if not m:
            snippet = src[pos:pos + 24]
            raise DslSyntaxError(f"Unexpected character at offset {pos}: {snippet!r}")
        kind = m.lastgroup
        val = m.group()
        if kind != "WS":
            if kind == "IDENT" and val in KEYWORDS:
                kind = val.upper()
            tokens.append(Token(kind, val, pos))
        pos = m.end()
    tokens.append(Token("EOF", "", n))
    return tokens


# ---------------------------------------------------------------------------
# AST Nodes
# ---------------------------------------------------------------------------

@dataclass
class NodeFlag:
    name: str
    aux_bool: Optional[Union[bool, "Cond"]] = None


@dataclass
class SrcRef:
    index: int


@dataclass
class VarRef:
    name: str


@dataclass
class DollarCall:
    name: str


@dataclass
class ModifierKey:
    mod_name: str


@dataclass
class HashExpr:
    name: str
    args: Optional[List["Arg"]] = None


@dataclass
class LookupCall:
    table: str
    key: ModifierKey
    args: List["Arg"]
    flags: List[NodeFlag] = field(default_factory=list)


@dataclass
class HostCall:
    name: str
    args: List["Arg"]


@dataclass
class SsaCall:
    op: str
    args: List["Arg"]
    flags: List[NodeFlag] = field(default_factory=list)


@dataclass
class Literal:
    kind: str  # "int", "float", "string", "bool"
    raw: str


@dataclass
class BareIdent:
    name: str


@dataclass
class Cond:
    negated: bool
    atom: Union[DollarCall, HostCall, VarRef]


@dataclass
class CondBinOp:
    op: str  # "&&" or "||"
    left: Union["Cond", "CondBinOp"]
    right: Union["Cond", "CondBinOp"]


Expr = Union[SrcRef, VarRef, DollarCall, HashExpr, LookupCall, HostCall, SsaCall]
Arg = Union[Expr, Cond, Literal, BareIdent, ModifierKey]


@dataclass
class AssignStmt:
    var: str
    val_type: Optional[str]
    rhs: Expr


@dataclass
class ExprStmt:
    expr: Union[HostCall, LookupCall, SsaCall]


@dataclass
class IfStmt:
    cond: Union[Cond, CondBinOp]
    then_body: List["Stmt"]
    else_body: Optional[List["Stmt"]] = None


@dataclass
class ForStmt:
    var: str
    bound: Arg
    body: List["Stmt"]


@dataclass
class ReturnStmt:
    value: Optional[Expr] = None


Stmt = Union[AssignStmt, ExprStmt, IfStmt, ForStmt, ReturnStmt]


@dataclass
class Program:
    stmts: List[Stmt]


# ---------------------------------------------------------------------------
# Strict LL(1) Recursive-Descent Parser
# ---------------------------------------------------------------------------

class LL1Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.idx = 0

    def peek(self) -> Token:
        return self.tokens[self.idx]

    def advance(self) -> Token:
        tok = self.tokens[self.idx]
        self.idx += 1
        return tok

    def expect(self, kind: str, val: Optional[str] = None) -> Token:
        tok = self.peek()
        if tok.kind != kind or (val is not None and tok.val != val):
            expected = f"{kind}({val!r})" if val is not None else kind
            raise DslSyntaxError(
                f"Expected {expected}, got {tok.kind}({tok.val!r}) at offset {tok.pos}"
            )
        return self.advance()

    # <program> ::= <stmt_list> EOF
    def parse_program(self) -> Program:
        stmts = self.parse_stmt_list()
        self.expect("EOF")
        return Program(stmts)

    # <stmt_list> ::= <stmt> <stmt_list> | ε
    def parse_stmt_list(self) -> List[Stmt]:
        stmts: List[Stmt] = []
        while self.peek().kind in ("VAR_REF", "AT", "IDENT", "IF", "FOR", "RETURN"):
            stmts.append(self.parse_stmt())
        return stmts

    # <stmt> ::= VAR_REF <type_ann_opt> "=" <expr> ";"
    #          | "@" <at_expr> ";"
    #          | IDENT "(" <arg_list_opt> ")" <node_flag_list> ";"
    #          | "if" "(" <cond> ")" "{" <stmt_list> "}" <else_opt>
    #          | "for" "(" VAR_REF ":" INT_LIT ".." <u32_expr> ")" "{" <stmt_list> "}"
    #          | "return" <expr_opt> ";"
    def parse_stmt(self) -> Stmt:
        tok = self.peek()
        if tok.kind == "VAR_REF":
            var = self.advance().val[1:]
            val_type = self.parse_type_ann_opt()
            self.expect("EQ")
            rhs = self.parse_expr()
            self.expect("SEMI")
            return AssignStmt(var, val_type, rhs)

        if tok.kind == "AT":
            self.advance()
            expr = self.parse_at_expr()
            self.expect("SEMI")
            return ExprStmt(expr)

        if tok.kind == "IDENT":
            op = self.advance().val
            self.expect("LPAREN")
            args = self.parse_arg_list_opt()
            self.expect("RPAREN")
            flags = self.parse_node_flag_list()
            self.expect("SEMI")
            return ExprStmt(SsaCall(op, args, flags))

        if tok.kind == "IF":
            self.advance()
            self.expect("LPAREN")
            cond = self.parse_cond()
            self.expect("RPAREN")
            self.expect("LBRACE")
            then_body = self.parse_stmt_list()
            self.expect("RBRACE")
            else_body = self.parse_else_opt()
            return IfStmt(cond, then_body, else_body)

        if tok.kind == "FOR":
            self.advance()
            self.expect("LPAREN")
            var = self.expect("VAR_REF").val[1:]
            self.expect("COLON")
            start = self.expect("INT_LIT")
            if start.val != "0":
                raise DslSyntaxError(f"for-loop must start at 0, got {start.val!r} at {start.pos}")
            self.expect("DOTDOT")
            bound = self.parse_u32_expr()
            self.expect("RPAREN")
            self.expect("LBRACE")
            body = self.parse_stmt_list()
            self.expect("RBRACE")
            return ForStmt(var, bound, body)

        if tok.kind == "RETURN":
            self.advance()
            val = self.parse_expr_opt()
            self.expect("SEMI")
            return ReturnStmt(val)

        raise DslSyntaxError(f"Unexpected token {tok.kind}({tok.val!r}) in <stmt> at {tok.pos}")

    # <type_ann_opt> ::= ":" IDENT | ε
    def parse_type_ann_opt(self) -> Optional[str]:
        if self.peek().kind == "COLON":
            self.advance()
            return self.expect("IDENT").val
        return None

    # <else_opt> ::= "else" "{" <stmt_list> "}" | ε
    def parse_else_opt(self) -> Optional[List[Stmt]]:
        if self.peek().kind == "ELSE":
            self.advance()
            if self.peek().kind == "IF":
                return [self.parse_stmt()]
            self.expect("LBRACE")
            body = self.parse_stmt_list()
            self.expect("RBRACE")
            return body
        return None

    # <expr_opt> ::= <expr> | ε
    def parse_expr_opt(self) -> Optional[Arg]:
        if self.peek().kind in ("SRC_REF", "VAR_REF", "DOLLAR", "HASH", "AT", "IDENT", "INT_LIT", "FLOAT_LIT", "STRING_LIT", "TRUE", "FALSE"):
            return self.parse_arg()
        return None

    # <expr> ::= SRC_REF | VAR_REF | "$" IDENT | "#" <hash_tail>
    #          | "@" <at_expr> | IDENT "(" <arg_list_opt> ")" <node_flag_list>
    def parse_expr(self) -> Expr:
        tok = self.peek()
        if tok.kind == "SRC_REF":
            return SrcRef(int(self.advance().val[1:]))
        if tok.kind == "VAR_REF":
            return VarRef(self.advance().val[1:])
        if tok.kind == "DOLLAR":
            self.advance()
            return DollarCall(self.expect("IDENT").val)
        if tok.kind == "HASH":
            self.advance()
            return self.parse_hash_tail()
        if tok.kind == "AT":
            self.advance()
            return self.parse_at_expr()
        if tok.kind == "IDENT":
            op = self.advance().val
            self.expect("LPAREN")
            args = self.parse_arg_list_opt()
            self.expect("RPAREN")
            flags = self.parse_node_flag_list()
            return SsaCall(op, args, flags)
        raise DslSyntaxError(f"Unexpected token {tok.kind}({tok.val!r}) in <expr> at {tok.pos}")

    # <at_expr> ::= "LOOKUP" "(" IDENT "," <key_expr> ")" "(" <arg_list_opt> ")" <node_flag_list>
    #             | IDENT "(" <arg_list_opt> ")"
    def parse_at_expr(self) -> Union[LookupCall, HostCall]:
        ident = self.expect("IDENT").val
        if ident == "LOOKUP":
            self.expect("LPAREN")
            table = self.expect("IDENT").val
            self.expect("COMMA")
            key = self.parse_key_expr()
            self.expect("RPAREN")
            self.expect("LPAREN")
            args = self.parse_arg_list_opt()
            self.expect("RPAREN")
            flags = self.parse_node_flag_list()
            return LookupCall(table, key, args, flags)
        self.expect("LPAREN")
        args = self.parse_arg_list_opt()
        self.expect("RPAREN")
        return HostCall(ident, args)

    # <hash_tail> ::= "lookup_u32" "(" IDENT "," <key_expr> ")"
    #               | "lookup_u32_raw" "(" IDENT "," <key_expr> ")"
    #               | IDENT <call_suffix_opt>
    def parse_hash_tail(self) -> HashExpr:
        name = self.expect("IDENT").val
        if name in ("lookup_u32", "lookup_u32_raw"):
            self.expect("LPAREN")
            table = self.expect("IDENT").val
            self.expect("COMMA")
            key = self.parse_key_expr()
            self.expect("RPAREN")
            return HashExpr(name, [BareIdent(table), key])
        args = self.parse_call_suffix_opt()
        return HashExpr(name, args)

    # <call_suffix_opt> ::= "(" <arg_list_opt> ")" | ε
    def parse_call_suffix_opt(self) -> Optional[List[Arg]]:
        if self.peek().kind == "LPAREN":
            self.advance()
            args = self.parse_arg_list_opt()
            self.expect("RPAREN")
            return args
        return None

    # <node_flag_list> ::= <node_flag> <node_flag_list> | ε
    def parse_node_flag_list(self) -> List[NodeFlag]:
        flags: List[NodeFlag] = []
        while self.peek().kind == "LBRACK":
            flags.append(self.parse_node_flag())
        return flags

    # <node_flag> ::= "[" <node_flag_body> "]"
    def parse_node_flag(self) -> NodeFlag:
        self.expect("LBRACK")
        kw = self.expect("IDENT")
        if kw.val in ("barrier", "side_effect"):
            self.expect("RBRACK")
            return NodeFlag(kw.val)
        if kw.val == "aux_bool":
            self.expect("EQ")
            val = self.parse_aux_bool_val()
            self.expect("RBRACK")
            return NodeFlag("aux_bool", val)
        raise DslSyntaxError(f"Unknown node flag {kw.val!r} at {kw.pos}")

    # <aux_bool_val> ::= "true" | "false" | <cond>
    def parse_aux_bool_val(self) -> Union[bool, Cond]:
        tok = self.peek()
        if tok.kind == "TRUE":
            self.advance()
            return True
        if tok.kind == "FALSE":
            self.advance()
            return False
        return self.parse_cond()

    def parse_cond_unary(self) -> Cond:
        neg = False
        if self.peek().kind == "BANG":
            self.advance()
            neg = True
        return Cond(neg, self.parse_cond_atom())

    # <cond> ::= <cond_unary> ( ("&&" | "||") <cond_unary> )*
    def parse_cond(self) -> Union[Cond, CondBinOp]:
        left: Union[Cond, CondBinOp] = self.parse_cond_unary()
        while self.peek().kind in ("ANDAND", "OROR"):
            op = self.advance().val
            right = self.parse_cond_unary()
            left = CondBinOp(op, left, right)
        return left

    # <cond_atom> ::= "$" IDENT | "@" IDENT "(" <arg_list_opt> ")" | VAR_REF
    def parse_cond_atom(self) -> Union[DollarCall, HostCall, VarRef]:
        tok = self.peek()
        if tok.kind == "DOLLAR":
            self.advance()
            return DollarCall(self.expect("IDENT").val)
        if tok.kind == "AT":
            self.advance()
            fn = self.expect("IDENT").val
            self.expect("LPAREN")
            args = self.parse_arg_list_opt()
            self.expect("RPAREN")
            return HostCall(fn, args)
        if tok.kind == "VAR_REF":
            return VarRef(self.advance().val[1:])
        raise DslSyntaxError(f"Expected '$', '@', or '%' in condition, got {tok.kind}({tok.val!r}) at {tok.pos}")

    # <arg_list_opt> ::= <arg> <arg_list_tail> | ε
    def parse_arg_list_opt(self) -> List[Arg]:
        if self.peek().kind == "RPAREN":
            return []
        args = [self.parse_arg()]
        while self.peek().kind == "COMMA":
            self.advance()
            args.append(self.parse_arg())
        return args

    # <arg> ::= SRC_REF | VAR_REF | "$" IDENT | "#" <hash_tail> | "@" <at_expr>
    #         | "!" <cond_atom> | INT_LIT | FLOAT_LIT | STRING_LIT | "true" | "false"
    #         | IDENT <ident_arg_tail>
    def parse_arg(self) -> Arg:
        tok = self.peek()
        if tok.kind == "SRC_REF":
            return SrcRef(int(self.advance().val[1:]))
        if tok.kind == "VAR_REF":
            return VarRef(self.advance().val[1:])
        if tok.kind == "DOLLAR":
            self.advance()
            return DollarCall(self.expect("IDENT").val)
        if tok.kind == "HASH":
            self.advance()
            return self.parse_hash_tail()
        if tok.kind == "AT":
            self.advance()
            return self.parse_at_expr()
        if tok.kind == "BANG":
            self.advance()
            return Cond(True, self.parse_cond_atom())
        if tok.kind == "INT_LIT":
            return Literal("int", self.advance().val)
        if tok.kind == "FLOAT_LIT":
            return Literal("float", self.advance().val)
        if tok.kind == "STRING_LIT":
            return Literal("string", self.advance().val)
        if tok.kind in ("TRUE", "FALSE"):
            return Literal("bool", self.advance().val)
        if tok.kind == "IDENT":
            name = self.advance().val
            if self.peek().kind == "LPAREN":
                self.advance()
                args = self.parse_arg_list_opt()
                self.expect("RPAREN")
                flags = self.parse_node_flag_list()
                return SsaCall(name, args, flags)
            return BareIdent(name)
        raise DslSyntaxError(f"Unexpected token {tok.kind}({tok.val!r}) in <arg> at {tok.pos}")

    # <u32_expr> ::= VAR_REF | "$" IDENT | "@" IDENT "(" <arg_list_opt> ")"
    #              | "#" "lookup_u32_raw" "(" IDENT "," <key_expr> ")" | INT_LIT
    def parse_u32_expr(self) -> Arg:
        tok = self.peek()
        if tok.kind == "VAR_REF":
            return VarRef(self.advance().val[1:])
        if tok.kind == "DOLLAR":
            self.advance()
            return DollarCall(self.expect("IDENT").val)
        if tok.kind == "AT":
            self.advance()
            fn = self.expect("IDENT").val
            self.expect("LPAREN")
            args = self.parse_arg_list_opt()
            self.expect("RPAREN")
            return HostCall(fn, args)
        if tok.kind == "HASH":
            self.advance()
            self.expect("IDENT", "lookup_u32_raw")
            self.expect("LPAREN")
            table = self.expect("IDENT").val
            self.expect("COMMA")
            key = self.parse_key_expr()
            self.expect("RPAREN")
            return HashExpr("lookup_u32_raw", [BareIdent(table), key])
        if tok.kind == "INT_LIT":
            return Literal("int", self.advance().val)
        raise DslSyntaxError(f"Unexpected token {tok.kind}({tok.val!r}) in <u32_expr> at {tok.pos}")

    # <key_expr> ::= "$" "modifiers" "." IDENT
    def parse_key_expr(self) -> ModifierKey:
        self.expect("DOLLAR")
        self.expect("IDENT", "modifiers")
        self.expect("DOT")
        return ModifierKey(self.expect("IDENT").val)


def parse_inline_ops(ops_list: List[str]) -> Program:
    src = " ".join(s.strip() for s in ops_list)
    return LL1Parser(tokenize(src)).parse_program()


if __name__ == "__main__":
    import json
    import os

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec_path = os.path.join(repo_root, "specs", "lowering", "valhall_lowering.json")
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    count = 0
    for rule in spec["rules"]:
        if rule.get("ssa_op") == "@INLINE_OPS":
            prog = parse_inline_ops(rule["inline_ops"])
            assert len(prog.stmts) > 0
            count += 1
    print(f"[OK] LL(1) parser validated {count} @INLINE_OPS rules with 0 syntax errors.")

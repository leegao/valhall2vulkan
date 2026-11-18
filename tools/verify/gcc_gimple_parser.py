#!/usr/bin/env python3
"""
GCC GIMPLE AST Dumper & Parser (Layer 1: C++17 Frontend)
========================================================
Uses `g++ -std=c++17 -fdump-tree-gimple` to parse C++17 source files
(`src/valhall_operands.cpp`, `src/valhall_pass1_handlers.cpp`) using the real
GCC C++ compiler frontend, separating C++ parsing from host-handler modeling.

Caches GIMPLE dumps by (abspath, mtime_ns, size) in the system temp dir so
unchanged files load instantaneously while any C++ edit automatically triggers
re-dumping.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import re
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enum Table Extraction from C++ Headers
# ---------------------------------------------------------------------------
def _parse_enum_from_header(header_path: str, enum_name: str) -> Dict[int, str]:
    with open(header_path, "r", encoding="utf-8") as f:
        txt = f.read()
    m = re.search(rf"enum\s+class\s+{re.escape(enum_name)}[^{{]*\{{([^}}]+)\}};", txt, re.S)
    if not m:
        return {}
    mapping: Dict[int, str] = {}
    cur_val = 0
    for raw_line in m.group(1).splitlines():
        line = raw_line.split("//", 1)[0].strip().rstrip(",")
        if not line:
            continue
        if "=" in line:
            k, v = [p.strip() for p in line.split("=", 1)]
            cur_val = int(v, 16 if v.startswith("0x") else 10)
            mapping[cur_val] = k
        else:
            mapping[cur_val] = line
        cur_val += 1
    return mapping


@dataclasses.dataclass
class CppEnumMaps:
    ssa_op: Dict[int, str]
    val_type: Dict[int, str]
    source_kind: Dict[int, str]
    shader_stage: Dict[int, str]

    @classmethod
    def from_repo(cls, repo_root: str) -> "CppEnumMaps":
        ssa_op = _parse_enum_from_header(
            os.path.join(repo_root, "src/generated/valhall_tables_generated.h"), "SsaOp"
        )
        val_type = _parse_enum_from_header(
            os.path.join(repo_root, "src/include/valhall_ssa.h"), "ValType"
        )
        source_kind = _parse_enum_from_header(
            os.path.join(repo_root, "src/generated/valhall_isa_generated.h"), "SourceKind"
        )
        shader_stage: Dict[int, str] = {}
        for hdr in ("valhall_to_spirv.h", "valhall_pass1.h"):
            p = os.path.join(repo_root, "src/include", hdr)
            if os.path.exists(p):
                shader_stage = _parse_enum_from_header(p, "ShaderStage")
                if shader_stage:
                    break
        return cls(
            ssa_op=ssa_op,
            val_type=val_type,
            source_kind=source_kind,
            shader_stage=shader_stage,
        )


# ---------------------------------------------------------------------------
# GIMPLE AST Data Structures
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class GimpleParam:
    type_str: str
    name: str


@dataclasses.dataclass
class GimpleLabel:
    name: str


@dataclasses.dataclass
class GimpleAssign:
    lhs: str
    rhs: str


@dataclasses.dataclass
class GimpleCall:
    lhs: Optional[str]
    callee: str
    args: List[str]


@dataclasses.dataclass
class GimpleCondJump:
    cond: str
    true_label: str
    false_label: str


@dataclasses.dataclass
class GimpleJump:
    label: str


@dataclasses.dataclass
class GimpleSwitch:
    expr: str
    cases: Dict[int, str]
    default_label: Optional[str]


@dataclasses.dataclass
class GimpleReturn:
    expr: Optional[str] = None


GimpleStmt = Any


@dataclasses.dataclass
class GimpleFunctionDef:
    full_name: str
    short_name: str
    namespace: str  # "", "abi", "Pass1Context", or lambda parent
    ret_type: str
    params: List[GimpleParam]
    local_types: Dict[str, str]
    stmts: List[GimpleStmt]
    labels: Dict[str, int]
    source_file: str


# ---------------------------------------------------------------------------
# GCC GIMPLE Dumper (with mtime/size caching)
# ---------------------------------------------------------------------------
def _find_vulkan_include(repo_root: str) -> Optional[str]:
    candidates = [
        os.path.join(repo_root, "build/_deps/vulkan_headers/include"),
        os.path.join(repo_root, "build-x86-64/_deps/vulkan_headers/include"),
        "/usr/include",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "vulkan", "vulkan.h")):
            return c
    return None


def dump_gimple(cpp_path: str, repo_root: str, cxx: str = "g++") -> str:
    abs_cpp = os.path.abspath(cpp_path)
    st = os.stat(abs_cpp)
    key = hashlib.sha256(
        f"{abs_cpp}:{st.st_mtime_ns}:{st.st_size}:{repo_root}".encode("utf-8")
    ).hexdigest()[:16]
    cache_dir = os.path.join(repo_root, "build")
    os.makedirs(cache_dir, exist_ok=True)
    gimple_path = os.path.join(cache_dir, f"{os.path.basename(abs_cpp)}.{key}.gimple")
    if os.path.exists(gimple_path):
        with open(gimple_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    cmd = [
        cxx,
        "-std=c++17",
        f"-I{os.path.join(repo_root, 'src/include')}",
        f"-I{os.path.join(repo_root, 'src/generated')}",
    ]
    vk_inc = _find_vulkan_include(repo_root)
    if vk_inc:
        cmd.append(f"-I{vk_inc}")
    cmd.extend([
        f"-fdump-tree-gimple={gimple_path}",
        "-c",
        abs_cpp,
        "-o",
        "/dev/null",
    ])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(gimple_path):
        raise RuntimeError(
            f"g++ -fdump-tree-gimple failed for {cpp_path}:\n{proc.stderr}"
        )
    with open(gimple_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


# ---------------------------------------------------------------------------
# GIMPLE Text Parser
# ---------------------------------------------------------------------------
_FN_HDR_RE = re.compile(
    r"^(?P<head>.+?)\s+\((?P<params>.*)\)\s*$"
)
_LABEL_RE = re.compile(r"^<(?P<lbl>[^>]+)>:$")
_IF_GOTO_RE = re.compile(
    r"^if\s+\((?P<cond>.+)\)\s+goto\s+<(?P<t>[^>]+)>;\s+else\s+goto\s+<(?P<f>[^>]+)>;$"
)
_GOTO_RE = re.compile(r"^goto\s+<(?P<lbl>[^>]+)>;$")
_SWITCH_RE = re.compile(r"^switch\s+\((?P<expr>.+?)\)\s+<(?P<body>.+)>$")
_RETURN_RE = re.compile(r"^return(?:\s+(?P<expr>.+))?;$")


def _split_top_level_commas(s: str) -> List[str]:
    parts: List[str] = []
    cur: List[str] = []
    paren = 0
    angle = 0
    brack = 0
    in_str = False
    escape = False
    for ch in s:
        if escape:
            cur.append(ch)
            escape = False
            continue
        if ch == "\\" and in_str:
            cur.append(ch)
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            cur.append(ch)
            continue
        if not in_str:
            if ch == "(":
                paren += 1
            elif ch == ")":
                paren -= 1
            elif ch == "[":
                brack += 1
            elif ch == "]":
                brack -= 1
            elif ch == "<":
                angle += 1
            elif ch == ">" and angle > 0:
                angle -= 1
            elif ch == "," and paren == 0 and angle == 0 and brack == 0:
                parts.append("".join(cur).strip())
                cur = []
                continue
        cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_fn_header(line: str) -> Optional[Tuple[str, str, str, str, List[GimpleParam]]]:
    m = _FN_HDR_RE.match(line)
    if not m:
        return None
    head = m.group("head").strip()
    if "valhall::compiler::" not in head:
        return None
    # Split return type from qualified function name
    # Skip std::template<valhall::compiler::...> helpers
    if "std::" in head.split("valhall::compiler::", 1)[0]:
        return None
    idx = head.find("valhall::compiler::")
    ret_type = head[:idx].strip()
    qual_name = head[idx:].strip()

    # Skip compiler-generated struct/lambda/destructor helpers except lambda operator()
    if "::~" in qual_name or "SsaNode::" in qual_name or "PhiIncoming::" in qual_name:
        return None

    # Parse namespace & short_name
    rel = qual_name[len("valhall::compiler::"):]
    if rel.startswith("abi::"):
        ns = "abi"
        short = rel[len("abi::"):]
    elif rel.startswith("Pass1Context::"):
        ns = "Pass1Context"
        short = rel[len("Pass1Context::"):]
    elif "::<lambda(" in rel and rel.endswith("::operator()"):
        ns = "lambda"
        short = rel
    elif "::" in rel:
        return None
    else:
        ns = ""
        short = rel

    params: List[GimpleParam] = []
    pstr = m.group("params").strip()
    if pstr and pstr != "void":
        for p in _split_top_level_commas(pstr):
            toks = p.rsplit(" ", 1)
            if len(toks) == 2:
                params.append(GimpleParam(toks[0].strip(), toks[1].strip()))
            else:
                params.append(GimpleParam(p, ""))
    return qual_name, short, ns, ret_type, params


def _strip_rso_comment(line: str) -> str:
    if line.endswith("[return slot optimization]"):
        return line[: -len("[return slot optimization]")].strip()
    return line


def _parse_call_rhs(rhs: str) -> Optional[Tuple[str, List[str]]]:
    # Matches `callee (arg1, arg2, ...)` at top level
    # Find the matching '(' for the trailing ')'
    if not rhs.endswith(")"):
        return None
    depth = 0
    in_str = False
    esc = False
    open_idx = -1
    for i in range(len(rhs) - 1, -1, -1):
        ch = rhs[i]
        if in_str:
            if ch == '"' and (i == 0 or rhs[i - 1] != "\\"):
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == ")":
            depth += 1
        elif ch == "(":
            depth -= 1
            if depth == 0:
                open_idx = i
                break
    if open_idx <= 0:
        return None
    callee = rhs[:open_idx].strip()
    # Reject casts like `(int) x` or parenthesized binary expressions
    if not callee or (callee.endswith(("=", "+", "-", "*", "/", "&", "|", "^", "!", "~")) and "operator" not in callee):
        return None
    # Callee must be an identifier / qualified name / member access (spaces allowed inside <...>)
    stripped = []
    adepth = 0
    for ch in callee:
        if ch == "<":
            adepth += 1
        elif ch == ">" and adepth > 0:
            adepth -= 1
        elif adepth == 0:
            stripped.append(ch)
    if any(ch in "".join(stripped) for ch in (" ", "\t")) and "operator" not in callee:
        return None
    args_str = rhs[open_idx + 1 : -1].strip()
    args = _split_top_level_commas(args_str) if args_str else []
    return callee, args


def parse_gimple_functions(gimple_text: str, source_file: str = "") -> List[GimpleFunctionDef]:
    lines = gimple_text.splitlines()
    n = len(lines)
    i = 0
    funcs: List[GimpleFunctionDef] = []

    while i < n:
        line = lines[i]
        if line and not line[0].isspace() and "valhall::compiler::" in line:
            parsed_hdr = _parse_fn_header(line)
            if parsed_hdr is not None and i + 1 < n and lines[i + 1].strip() == "{":
                qual_name, short, ns, ret_type, params = parsed_hdr
                i += 2
                body_lines: List[str] = []
                depth = 1
                while i < n and depth > 0:
                    ln = lines[i]
                    stripped = ln.strip()
                    if stripped == "{":
                        depth += 1
                    elif stripped == "}":
                        depth -= 1
                        if depth == 0:
                            i += 1
                            break
                    body_lines.append(ln)
                    i += 1
                fn = _parse_gimple_body(
                    qual_name, short, ns, ret_type, params, body_lines, source_file
                )
                funcs.append(fn)
                continue
        i += 1
    return funcs


def _parse_gimple_body(
    qual_name: str,
    short: str,
    ns: str,
    ret_type: str,
    params: List[GimpleParam],
    body_lines: List[str],
    source_file: str,
) -> GimpleFunctionDef:
    local_types: Dict[str, str] = {}
    stmts: List[GimpleStmt] = []
    labels: Dict[str, int] = {}

    idx = 0
    m = len(body_lines)
    while idx < m:
        raw = body_lines[idx]
        s = _strip_rso_comment(raw.strip())
        idx += 1
        if not s or s in ("{", "}", "try"):
            continue
        if s.startswith("//"):
            continue
        if s == "finally":
            # Skip cleanup destructor block `{ ... }`
            if idx < m and body_lines[idx].strip() == "{":
                d = 1
                idx += 1
                while idx < m and d > 0:
                    st = body_lines[idx].strip()
                    if st == "{":
                        d += 1
                    elif st == "}":
                        d -= 1
                    idx += 1
            continue

        # Label
        lm = _LABEL_RE.match(s)
        if lm:
            lbl = lm.group("lbl")
            labels[lbl] = len(stmts)
            stmts.append(GimpleLabel(lbl))
            continue

        # if (...) goto <t>; else goto <f>;
        im = _IF_GOTO_RE.match(s)
        if im:
            stmts.append(GimpleCondJump(im.group("cond"), im.group("t"), im.group("f")))
            continue

        # goto <lbl>;
        gm = _GOTO_RE.match(s)
        if gm:
            stmts.append(GimpleJump(gm.group("lbl")))
            continue

        # switch (expr) <default: <d>, case 0: <c0>, ...>
        sm = _SWITCH_RE.match(s)
        if sm:
            cases: Dict[int, str] = {}
            def_lbl: Optional[str] = None
            for part in _split_top_level_commas(sm.group("body")):
                if part.startswith("default:"):
                    def_lbl = part.split("<", 1)[1].rstrip(">")
                elif part.startswith("case "):
                    c_part, l_part = part[5:].split(":", 1)
                    cases[int(c_part.strip())] = l_part.strip().lstrip("<").rstrip(">")
            stmts.append(GimpleSwitch(sm.group("expr"), cases, def_lbl))
            continue

        # return [expr];
        rm = _RETURN_RE.match(s)
        if rm:
            stmts.append(GimpleReturn(rm.group("expr")))
            continue

        # Local declaration (ends with ';' and has no '=')
        if s.endswith(";") and "=" not in s and "(" not in s:
            decl = s[:-1].strip()
            parts = decl.rsplit(" ", 1)
            if len(parts) == 2:
                vtype = parts[0].strip()
                vname = parts[1].strip()
                arr_m = re.match(r"^([^\[]+)(\[.*\])$", vname)
                if arr_m:
                    vname = arr_m.group(1).strip()
                    vtype = vtype + arr_m.group(2)
                local_types[vname] = vtype
            continue

        # Assignment or void call
        if s.endswith(";"):
            stmt_str = s[:-1].strip()
            # Check for `lhs = rhs` (being careful of `==`, `!=`, `<=`, `>=`)
            eq_pos = _find_assign_eq(stmt_str)
            if eq_pos >= 0:
                lhs = stmt_str[:eq_pos].strip()
                rhs = stmt_str[eq_pos + 1 :].strip()
                call = _parse_call_rhs(rhs)
                if call is not None:
                    stmts.append(GimpleCall(lhs, call[0], call[1]))
                else:
                    stmts.append(GimpleAssign(lhs, rhs))
            else:
                call = _parse_call_rhs(stmt_str)
                if call is not None:
                    stmts.append(GimpleCall(None, call[0], call[1]))
            continue

    return GimpleFunctionDef(
        full_name=qual_name,
        short_name=short,
        namespace=ns,
        ret_type=ret_type,
        params=params,
        local_types=local_types,
        stmts=stmts,
        labels=labels,
        source_file=source_file,
    )


def _find_assign_eq(s: str) -> int:
    in_str = False
    paren = 0
    brace = 0
    for i, ch in enumerate(s):
        if ch == '"' and (i == 0 or s[i - 1] != "\\"):
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "(":
            paren += 1
        elif ch == ")":
            paren -= 1
        elif ch == "{":
            brace += 1
        elif ch == "}":
            brace -= 1
        elif ch == "=" and paren == 0 and brace == 0:
            prev_c = s[i - 1] if i > 0 else ""
            next_c = s[i + 1] if i + 1 < len(s) else ""
            if prev_c not in ("=", "!", "<", ">") and next_c != "=":
                if s[max(0, i - 8):i] == "operator":
                    continue
                return i
    return -1

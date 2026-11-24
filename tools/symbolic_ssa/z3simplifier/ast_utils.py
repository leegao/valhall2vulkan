"""
Pure Z3 AST utilities: iterative DAG traversal, memoized bottom-up rewriting,
operator/theory census, and human-readable pretty-printing (`format_z3_ast`).

All traversals are strictly iterative (explicit work stacks) so deep shader
DAGs (10k+ nodes) never overflow Python's recursion limit and preserve DAG
sharing in O(|V|) time.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import z3

from tools.symbolic_ssa.z3utils import dsl


@dataclass(frozen=True)
class AstCensus:
    """Operator and theory census over a set of Z3 AST roots."""

    total_nodes: int
    max_depth: int
    fp_ops: int
    nlin_real_ops: int
    bv_mul_shift_ops: int
    array_select_ops: int
    array_store_ops: int
    bridge_ops: int
    ite_ops: int
    bool_wrapper_ops: int
    int_bv_ops: int = 0


_BRIDGE_NAMES: set[str] = {
    "u_f32_to_bv32",
    "u_bv32_to_f32",
    "u_f16_to_bv16",
    "u_bv16_to_f16",
    "u_f64_to_bv64",
    "u_bv64_to_f64",
    "real2bv32",
    "bv32_to_real",
}


def count_ast_nodes(roots: Sequence[dsl.ExprRef], limit: int = 100000) -> int:
    """Count unique Z3 AST nodes reachable from `roots` (capped at `limit`)."""
    visited: set[int] = set()
    stack: list[dsl.ExprRef] = [r for r in roots if r is not None]
    while stack and len(visited) < limit:
        cur = stack.pop()
        cid = cur.get_id()
        if cid in visited:
            continue
        visited.add(cid)
        if dsl.is_app(cur):
            for i in range(cur.num_args()):
                stack.append(cur.arg(i))
    return len(visited)


def census_ast(roots: Sequence[dsl.ExprRef], limit: int = 100000) -> AstCensus:
    """Compute an operator and theory census over a Z3 DAG."""
    visited: set[int] = set()
    depths: dict[int, int] = {}
    post_order: list[dsl.ExprRef] = []
    stack: list[tuple[dsl.ExprRef, bool]] = [(r, False) for r in roots if r is not None]

    while stack and len(visited) < limit:
        cur, expanded = stack.pop()
        cid = cur.get_id()
        if expanded:
            if cid not in depths:
                if dsl.is_app(cur) and cur.num_args() > 0:
                    depths[cid] = 1 + max(depths.get(cur.arg(i).get_id(), 0) for i in range(cur.num_args()))
                else:
                    depths[cid] = 0
                post_order.append(cur)
            continue
        if cid in visited:
            continue
        visited.add(cid)
        stack.append((cur, True))
        if dsl.is_app(cur):
            for i in range(cur.num_args() - 1, -1, -1):
                ch = cur.arg(i)
                if ch.get_id() not in visited:
                    stack.append((ch, False))

    fp_ops = 0
    nlin_real = 0
    bv_mul_shift = 0
    arr_sel = 0
    arr_st = 0
    bridges = 0
    ites = 0
    bool_wrappers = 0
    int_bv = 0

    for node in post_order:
        if not dsl.is_app(node):
            continue
        decl = node.decl()
        kind = decl.kind()
        name = decl.name()
        sort_kind = node.sort().kind()

        if sort_kind == dsl.Z3SortKind.FLOATING_POINT:
            fp_ops += 1
        elif kind == dsl.Z3OpKind.MUL and sort_kind == dsl.Z3SortKind.REAL:
            non_const = sum(1 for i in range(node.num_args()) if not dsl.is_rational_value(node.arg(i)))
            if non_const >= 2:
                nlin_real += 1
        elif kind == dsl.Z3OpKind.DIV and sort_kind == dsl.Z3SortKind.REAL:
            if not dsl.is_rational_value(node.arg(1)):
                nlin_real += 1
        elif kind in (
            dsl.Z3OpKind.BMUL,
            dsl.Z3OpKind.BSDIV,
            dsl.Z3OpKind.BUDIV,
            dsl.Z3OpKind.BSREM,
            dsl.Z3OpKind.BUREM,
            dsl.Z3OpKind.BSHL,
            dsl.Z3OpKind.BLSHR,
            dsl.Z3OpKind.BASHR,
        ):
            bv_mul_shift += 1
        elif kind == dsl.Z3OpKind.SELECT:
            arr_sel += 1
        elif kind == dsl.Z3OpKind.STORE:
            arr_st += 1
        elif kind == dsl.Z3OpKind.ITE:
            ites += 1
            if (
                node.num_args() == 3
                and dsl.is_bv_value(node.arg(1))
                and dsl.is_bv_value(node.arg(2))
                and dsl.as_long(node.arg(1)) == 1
                and dsl.as_long(node.arg(2)) == 0
            ):
                bool_wrappers += 1
        if name in _BRIDGE_NAMES:
            bridges += 1
        if kind in (dsl.Z3OpKind.BV2INT, dsl.Z3OpKind.INT2BV, dsl.Z3OpKind.TO_INT, dsl.Z3OpKind.TO_REAL):
            int_bv += 1

    max_d = max(depths.values()) if depths else 0
    return AstCensus(
        total_nodes=len(visited),
        max_depth=max_d,
        fp_ops=fp_ops,
        nlin_real_ops=nlin_real,
        bv_mul_shift_ops=bv_mul_shift,
        array_select_ops=arr_sel,
        array_store_ops=arr_st,
        bridge_ops=bridges,
        ite_ops=ites,
        bool_wrapper_ops=bool_wrappers,
        int_bv_ops=int_bv,
    )


def rewrite_dag(
    root: dsl.ExprRef,
    rule_fn: Callable[[dsl.ExprRef], Any],
    memo: dict[int, Any] | None = None,
) -> tuple[dsl.ExprRef, int]:
    """
    Iterative, memoized bottom-up DAG rewriter.

    Visits each unique node once. After rewriting a node's children, rebuilds
    the parent only if any child changed, then invokes `rule_fn(rebuilt)`.
    If `rule_fn` returns a non-None `ExprRef` different from `rebuilt`,
    increments the rewrite counter and memoizes the result.
    """
    cache: dict[int, Any] = memo if memo is not None else {}
    rewrites = 0
    root_id = root.get_id()
    stack: list[tuple[dsl.ExprRef, bool]] = [(root, False)]

    while stack:
        cur, expanded = stack.pop()
        cid = cur.get_id()
        if cid in cache:
            continue
        if not expanded:
            if not dsl.is_app(cur) or cur.num_args() == 0:
                repl = rule_fn(cur)
                if repl is not None and not dsl.expr_eq(repl, cur):
                    rewrites += 1
                    cache[cid] = repl
                else:
                    cache[cid] = cur
                continue
            stack.append((cur, True))
            for i in range(cur.num_args() - 1, -1, -1):
                ch = cur.arg(i)
                if ch.get_id() not in cache:
                    stack.append((ch, False))
        else:
            nargs = cur.num_args()
            new_args: list[Any] = []
            args_changed = False
            for i in range(nargs):
                orig_a = cur.arg(i)
                new_a = cache.get(orig_a.get_id(), orig_a)
                if not dsl.expr_eq(new_a, orig_a):
                    args_changed = True
                new_args.append(new_a)
            rebuilt = rebuild_app(cur, new_args) if args_changed else cur
            repl = rule_fn(rebuilt)
            if repl is not None and not dsl.expr_eq(repl, rebuilt):
                rewrites += 1
                res_node: Any = repl
            else:
                res_node = rebuilt
            if not isinstance(cur, dsl.Z3Expr) and isinstance(res_node, dsl.Z3Expr):
                res_node = dsl.lower(res_node)
            cache[cid] = res_node

    return cache.get(root_id, cache.get(root.get_id(), root)), rewrites


def rebuild_app(node: dsl.ExprRef, new_args: Sequence[Any]) -> dsl.ExprRef:
    """
    Safely rebuild a Z3 application node with rewritten children.

    Handles built-in operators whose `decl()` requires special constructors
    (`Extract`, `ZeroExt`, `SignExt`, `Concat`, `If`, `And`, `Or`, `Not`,
    `Select`, `Store`, etc.) cleanly without Z3 FFI parameter-mismatch errors.
    """
    if not isinstance(node, dsl.Z3Expr):
        node = dsl.view(node)
    args: list[Any] = [a if isinstance(a, dsl.Z3Expr) else dsl.view(a) for a in new_args]
    decl = node.decl()
    kind = decl.kind()

    if kind == dsl.Z3OpKind.EXTRACT:
        hi = decl.params()[0]
        lo = decl.params()[1]
        return dsl.Extract(hi, lo, args[0])
    if kind == dsl.Z3OpKind.ZERO_EXT:
        return dsl.ZeroExt(decl.params()[0], args[0])
    if kind == dsl.Z3OpKind.SIGN_EXT:
        return dsl.SignExt(decl.params()[0], args[0])
    if kind == dsl.Z3OpKind.CONCAT:
        return dsl.Concat(*args)
    if kind == dsl.Z3OpKind.ITE:
        return dsl.If(args[0], args[1], args[2])
    if kind == dsl.Z3OpKind.AND:
        return dsl.And(*args)
    if kind == dsl.Z3OpKind.OR:
        return dsl.Or(*args)
    if kind == dsl.Z3OpKind.NOT:
        return dsl.Not(args[0])
    if kind == dsl.Z3OpKind.EQ:
        return args[0] == args[1]
    if kind == dsl.Z3OpKind.DISTINCT and len(args) == 2:
        return args[0] != args[1]
    if kind == dsl.Z3OpKind.SELECT:
        return dsl.Select(args[0], args[1])
    if kind == dsl.Z3OpKind.STORE:
        return dsl.Store(args[0], args[1], args[2])
    if kind == dsl.Z3OpKind.ADD:
        acc = args[0]
        for a in args[1:]:
            acc = acc + a
        return acc
    if kind == dsl.Z3OpKind.SUB and len(args) == 2:
        return args[0] - args[1]
    if kind == dsl.Z3OpKind.UMINUS and len(args) == 1:
        return -args[0]
    if kind == dsl.Z3OpKind.MUL:
        acc = args[0]
        for a in args[1:]:
            acc = acc * a
        return acc
    if kind == dsl.Z3OpKind.DIV and len(args) == 2:
        return args[0] / args[1]
    if kind == dsl.Z3OpKind.BADD and len(args) == 2:
        return args[0] + args[1]
    if kind == dsl.Z3OpKind.BSUB and len(args) == 2:
        return args[0] - args[1]
    if kind == dsl.Z3OpKind.BMUL and len(args) == 2:
        return args[0] * args[1]
    if kind == dsl.Z3OpKind.BAND and len(args) == 2:
        return args[0] & args[1]
    if kind == dsl.Z3OpKind.BOR and len(args) == 2:
        return args[0] | args[1]
    if kind == dsl.Z3OpKind.BXOR and len(args) == 2:
        return args[0] ^ args[1]
    if kind == dsl.Z3OpKind.BSHL and len(args) == 2:
        return args[0] << args[1]
    if kind == dsl.Z3OpKind.BLSHR and len(args) == 2:
        return dsl.LShR(args[0], args[1])
    if kind == dsl.Z3OpKind.BASHR and len(args) == 2:
        return args[0] >> args[1]
    return decl.rebuild(args)


def format_z3_ast(
    expr: Any,
    *,
    max_depth: int = 12,
    max_lines: int = 120,
    let_bind_shared: bool = True,
) -> str:
    """
    Pretty-print a Z3 AST in readable, indented let-bound S-expression form.

    Shared sub-DAGs referenced >= 2 times (with size >= 3) are hoisted into
    compact `(let ((%t0 ...) (%t1 ...)) ...)` bindings when `let_bind_shared`
    is enabled, keeping 10,000-node shader expressions scannable in `--dump-ast`.
    """
    if expr is None:
        return "<None>"

    # Pass 1: count parent references to identify shared subexpressions.
    ref_count: dict[int, int] = {}
    order: list[dsl.ExprRef] = []
    visited: set[int] = set()
    stack: list[tuple[dsl.ExprRef, bool]] = [(expr, False)]
    while stack:
        cur, expanded = stack.pop()
        cid = cur.get_id()
        if expanded:
            order.append(cur)
            continue
        ref_count[cid] = ref_count.get(cid, 0) + 1
        if cid in visited:
            continue
        visited.add(cid)
        stack.append((cur, True))
        if dsl.is_app(cur):
            for i in range(cur.num_args() - 1, -1, -1):
                stack.append((cur.arg(i), False))

    let_names: dict[int, str] = {}
    let_defs: list[tuple[str, dsl.ExprRef]] = []
    if let_bind_shared:
        for node in order:
            nid = node.get_id()
            if nid == expr.get_id():
                continue
            if ref_count.get(nid, 0) >= 2 and dsl.is_app(node) and node.num_args() > 0:
                lname = f"%t{len(let_defs)}"
                let_names[nid] = lname
                let_defs.append((lname, node))

    def _op_label(n: dsl.ExprRef) -> str:
        if isinstance(n, dsl.Z3BvConst):
            return f"bv{n.size()}(0x{n.as_long():x})"
        if isinstance(n, dsl.Z3RealConst):
            return n.as_decimal(6).rstrip("?")
        if isinstance(n, dsl.Z3IntConst):
            return str(n.as_long())
        if dsl.is_true(n):
            return "true"
        if dsl.is_false(n):
            return "false"
        d = n.decl()
        k = d.kind()
        if k == dsl.Z3OpKind.EXTRACT:
            return f"Extract[{d.params()[0]}:{d.params()[1]}]"
        if k == dsl.Z3OpKind.ZERO_EXT:
            return f"ZeroExt[{d.params()[0]}]"
        if k == dsl.Z3OpKind.SIGN_EXT:
            return f"SignExt[{d.params()[0]}]"
        return d.name()

    lines: list[str] = []

    def _fmt(n: dsl.ExprRef, depth: int, indent: str, allow_let_ref: bool = True) -> str:
        nid = n.get_id()
        if allow_let_ref and nid in let_names:
            return let_names[nid]
        if not dsl.is_app(n) or n.num_args() == 0:
            return _op_label(n)
        if depth >= max_depth:
            return f"({_op_label(n)} ...)"
        lbl = _op_label(n)
        ch_strs = [_fmt(n.arg(i), depth + 1, indent + "  ", True) for i in range(n.num_args())]
        one_line = f"({lbl} {' '.join(ch_strs)})"
        if len(one_line) <= 76 and "\n" not in one_line:
            return one_line
        body = f"\n{indent}  ".join(ch_strs)
        return f"({lbl}\n{indent}  {body})"

    if let_defs:
        lines.append("(let (")
        for lname, lnode in let_defs[: max(1, max_lines // 2)]:
            rendered = _fmt(lnode, 0, "    ", allow_let_ref=False)
            lines.append(f"  ({lname} {rendered})")
        if len(let_defs) > max(1, max_lines // 2):
            lines.append(f"  ; ... ({len(let_defs) - max(1, max_lines // 2)} more shared bindings omitted)")
        lines.append(")")
        body_str = _fmt(expr, 0, "  ", allow_let_ref=True)
        for bl in body_str.splitlines():
            lines.append(f"  {bl}")
        lines.append(")")
    else:
        lines.extend(_fmt(expr, 0, "", allow_let_ref=True).splitlines())

    if len(lines) > max_lines:
        omitted = len(lines) - max_lines
        lines = lines[:max_lines] + [f"  ; ... ({omitted} lines truncated)"]
    return "\n".join(lines)

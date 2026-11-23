#!/usr/bin/env python3
"""
Unique-Call Uninterpreted Function (UF) Pruning & AST Size Helpers.

For any uninterpreted hardware function `hw_*` (such as `hw_tex_*`, `hw_imgq_*`,
`hw_sideband_u32_*`) that appears at most once as a distinct application node
`f(e_1, ..., e_m)` across a set of Z3 formulas (and is unconstrained by
preconditions), replacing `f(e_1, ..., e_m)` with a fresh symbolic constant
`v_f` of sort `f.range()` is equisatisfiable:

    exists f. Phi[f(e)]  <==>  exists v. Phi[v]

A satisfying assignment `v*` lifts soundly back to the constant function
interpretation `{"table": {}, "default": v*}`, guaranteeing bit-exact concrete
replay while pruning arbitrarily deep FP32 argument cones (e.g., texture
coordinate calculations) from Z3 bit-blasting.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import z3

from tools.symbolic_ssa.z3 import dsl

from tools.symbolic_ssa.env import MASK32


def ast_node_count(expr: Any, limit: int = 10000) -> int:
    """Count unique AST nodes in a Z3 expression DAG (capped at `limit`)."""
    visited: Set[int] = set()
    stack: List[z3.ExprRef] = [expr]
    while stack and len(visited) < limit:
        cur = stack.pop()
        cid = cur.get_id()
        if cid in visited:
            continue
        visited.add(cid)
        if z3.is_app(cur):
            for ch in cur.children():
                stack.append(ch)
    return len(visited)


def _collect_hw_uf_apps(
    exprs: Sequence[z3.ExprRef],
) -> Dict[str, Dict[int, z3.ExprRef]]:
    """Map each `hw_*` FuncDecl name to `{expr_id: expr}` of distinct applications."""
    apps_by_name: Dict[str, Dict[int, z3.ExprRef]] = {}
    visited: Set[int] = set()
    stack: List[z3.ExprRef] = [dsl.lower(e) for e in exprs]
    while stack:
        cur = stack.pop()
        cid = cur.get_id()
        if cid in visited:
            continue
        visited.add(cid)
        if z3.is_app(cur):
            decl = cur.decl()
            if decl.arity() >= 1 and decl.kind() == z3.Z3_OP_UNINTERPRETED:
                dname = decl.name()
                if dname.startswith("hw_"):
                    apps_by_name.setdefault(dname, {})[cid] = cur
            for ch in cur.children():
                stack.append(ch)
    return apps_by_name


def prune_unique_hw_ufs(
    exprs: Sequence[z3.ExprRef],
    preconditions: Sequence[z3.BoolRef] = (),
) -> Tuple[List[z3.ExprRef], Dict[str, z3.ExprRef]]:
    """
    Substitute every single-occurrence `hw_*` UF application across `exprs`
    (not mentioned in `preconditions`) with a fresh constant variable.

    Returns `(rewritten_exprs, uf_proxies)` where `uf_proxies` maps each pruned
    UF declaration name `dname` to its fresh Z3 constant `proxy_var`.
    """
    if not exprs:
        return [], {}

    pre_apps = _collect_hw_uf_apps(preconditions) if preconditions else {}
    all_apps = _collect_hw_uf_apps(exprs)

    subst_pairs: List[Tuple[z3.ExprRef, z3.ExprRef]] = []
    uf_proxies: Dict[str, z3.ExprRef] = {}

    for dname, app_map in all_apps.items():
        if dname in pre_apps:
            continue
        if len(app_map) != 1:
            continue
        app_expr = next(iter(app_map.values()))
        rng = app_expr.sort()
        pname = f"__ufproxy__{dname}"
        if rng == z3.BoolSort():
            pvar: z3.ExprRef = z3.Bool(pname)
        elif z3.is_bv_sort(rng):
            pvar = z3.BitVec(pname, rng.size())
        else:
            continue
        subst_pairs.append((app_expr, pvar))
        uf_proxies[dname] = pvar

    if not subst_pairs:
        return list(exprs), {}

    rewritten = [z3.substitute(dsl.lower(e), *subst_pairs) for e in exprs]
    return rewritten, uf_proxies


def lift_uf_proxies_into_overrides(
    model: z3.ModelRef,
    uf_proxies: Dict[str, z3.ExprRef],
    overrides: Dict[str, Any],
) -> None:
    """Lift proxy constant assignments from `model` into constant `uf:<name>` overrides."""
    if not uf_proxies:
        return
    for dname, pvar in uf_proxies.items():
        val = model.eval(pvar, model_completion=True)
        def_val: Optional[int] = None
        if z3.is_bv_value(val):
            def_val = int(val.as_long()) & MASK32
        elif z3.is_true(val):
            def_val = 1
        elif z3.is_false(val):
            def_val = 0
        if def_val is not None:
            overrides[f"uf:{dname}"] = {"table": {}, "default": def_val}
            overrides.pop(pvar.decl().name(), None)

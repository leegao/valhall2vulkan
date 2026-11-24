#!/usr/bin/env python3
"""
Z3-domain dataflow analysis and extensional-equality propagation engine,
operating on lifted `z3.dsl` terms via Python `match`/`case` and
lowering back to `z3.ExprRef` via `.lower()`.

Responsibilities (cleanly separated from local peephole folding):
  1. `EqualityTable` (Union-Find + Congruence): harvests ground truths and
     extensional equalities (`a == b`) from `preconditions` and `guard`,
     preferring concrete constants then smaller ASTs as canonical reps.
  2. `KnownBits` (Forward BitVec Lattice): propagates `(zeros, ones)` bitmasks
     via `match lift(node, depth=1): case ...` across `Z3BvConst`, `Z3OpZeroExt`,
     `Z3OpSignExt`, `Z3OpExtract`, `Z3OpConcat`, `Z3OpBvAnd/Or/Xor/Shl/LShr`,
     and `Z3OpIte`, collapsing fully-determined BitVecs via `Z3BvConst(...).lower()`.
  3. Bounded Path-Sensitive `If` Propagation: refines `EqualityTable` along
     `then`/`else` arms of `Z3OpIte` (bounded to `max_path_depth=2`, `<=512`
     nodes) to collapse redundant guards and `discarded`-guarded fragment outputs.

100% equivalence-preserving (`is_equivalence_preserving = True`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from tools.symbolic_ssa import z3utils as z3

from tools.symbolic_ssa.z3simplifier.ast_utils import count_ast_nodes, rebuild_app
from tools.symbolic_ssa.z3simplifier.context import (
    PassStats,
    SimplificationContext,
    SimplifyQuery,
)
from tools.symbolic_ssa.z3utils.dsl import (
    Z3BoolConst,
    Z3BvConst,
    Z3Node,
    Z3OpAnd,
    Z3OpBvAnd,
    Z3OpBvLShr,
    Z3OpBvOr,
    Z3OpBvShl,
    Z3OpBvSub,
    Z3OpBvXor,
    Z3OpConcat,
    Z3OpDistinct,
    Z3OpEq,
    Z3OpExtract,
    Z3OpIte,
    Z3OpNot,
    Z3OpOr,
    Z3OpRealSub,
    Z3OpSignExt,
    Z3OpZeroExt,
    Z3RealConst,
    lift,
    unwrap,
)
from tools.symbolic_ssa.z3simplifier.pass_base import Z3QueryPass


@dataclass(frozen=True, slots=True)
class KnownBits:
    """Sound known-bits abstract domain element for a `width`-bit BitVector."""

    width: int
    zeros: int
    ones: int

    @property
    def mask(self) -> int:
        return (1 << self.width) - 1

    @property
    def is_const(self) -> bool:
        return ((self.zeros | self.ones) & self.mask) == self.mask

    @classmethod
    def top(cls, width: int) -> "KnownBits":
        return cls(width, 0, 0)

    @classmethod
    def const(cls, val: int, width: int) -> "KnownBits":
        m = (1 << width) - 1
        v = val & m
        return cls(width, (~v) & m, v)

    def join(self, other: "KnownBits") -> "KnownBits":
        if self.width != other.width:
            return KnownBits.top(self.width)
        return KnownBits(self.width, self.zeros & other.zeros, self.ones & other.ones)


def _is_ground_const(e: z3.ExprRef) -> bool:
    return (
        isinstance(e, (z3.BitVecNumRef, z3.RatNumRef, z3.IntNumRef))
        or z3.is_true(e)
        or z3.is_false(e)
    )


def _rep_rank(e: z3.ExprRef) -> Tuple[int, int, int]:
    if _is_ground_const(e):
        return (0, 0, e.get_id())
    if z3.is_const(e):
        return (1, 0, e.get_id())
    nargs = e.num_args() if z3.is_app(e) else 0
    return (2, nargs, e.get_id())


class EqualityTable:
    """Scoped Union-Find table of extensional equalities (`a == b`) and boolean truth facts."""

    def __init__(self, parent: Optional["EqualityTable"] = None) -> None:
        if parent is not None:
            self._uf: Dict[int, int] = dict(parent._uf)
            self._exprs: Dict[int, z3.ExprRef] = dict(parent._exprs)
            self._bool_vals: Dict[int, bool] = dict(parent._bool_vals)
        else:
            self._uf = {}
            self._exprs = {}
            self._bool_vals = {}

    def _find_id(self, eid: int) -> int:
        root = eid
        while self._uf.get(root, root) != root:
            root = self._uf[root]
        cur = eid
        while self._uf.get(cur, cur) != root:
            nxt = self._uf[cur]
            self._uf[cur] = root
            cur = nxt
        return root

    def find_rep(self, expr: z3.ExprRef) -> z3.ExprRef:
        eid = expr.get_id()
        if eid not in self._uf:
            return expr
        return self._exprs[self._find_id(eid)]

    def are_equal(self, a: z3.ExprRef, b: z3.ExprRef) -> bool:
        if z3.expr_eq(a, b):
            return True
        aid, bid = a.get_id(), b.get_id()
        if aid not in self._uf or bid not in self._uf:
            return False
        return self._find_id(aid) == self._find_id(bid)

    def lookup_bool(self, expr: z3.ExprRef) -> Optional[bool]:
        if z3.is_true(expr):
            return True
        if z3.is_false(expr):
            return False
        eid = expr.get_id()
        if eid in self._bool_vals:
            return self._bool_vals[eid]
        rep = self.find_rep(expr)
        if z3.is_true(rep):
            return True
        if z3.is_false(rep):
            return False
        return self._bool_vals.get(rep.get_id())

    def union(self, a: z3.ExprRef, b: z3.ExprRef) -> None:
        if z3.expr_eq(a, b):
            return
        aid, bid = a.get_id(), b.get_id()
        if aid not in self._uf:
            self._uf[aid] = aid
            self._exprs[aid] = a
        if bid not in self._uf:
            self._uf[bid] = bid
            self._exprs[bid] = b
        ra, rb = self._find_id(aid), self._find_id(bid)
        if ra == rb:
            return
        ea, eb = self._exprs[ra], self._exprs[rb]
        if _rep_rank(eb) < _rep_rank(ea):
            ra, rb = rb, ra
            ea, eb = eb, ea
        self._uf[rb] = ra
        ba = self._bool_vals.get(ra)
        bb = self._bool_vals.get(rb)
        if ba is None and bb is not None:
            self._bool_vals[ra] = bb
        elif z3.is_true(ea):
            self._bool_vals[ra] = True
        elif z3.is_false(ea):
            self._bool_vals[ra] = False

    def assume_bool(self, pred: z3.ExprRef, truth: bool) -> None:
        """Harvest boolean truth and nested `And`/`Or`/`Not`/`Eq`/`Distinct` facts via DSL `match`."""
        work: List[Tuple[z3.ExprRef, bool]] = [(pred, truth)]
        seen: Set[Tuple[int, bool]] = set()
        while work and len(seen) < 256:
            cur, val = work.pop()
            key = (cur.get_id(), val)
            if key in seen:
                continue
            seen.add(key)
            if z3.is_true(cur) or z3.is_false(cur):
                continue
            self._bool_vals[cur.get_id()] = val
            self.union(cur, Z3BoolConst(val).lower())

            match lift(cur, depth=1):
                case Z3OpNot(inner):
                    work.append((unwrap(inner), not val))
                case Z3OpAnd(args) if val is True:
                    for a in args:
                        work.append((unwrap(a), True))
                case Z3OpOr(args) if val is False:
                    for a in args:
                        work.append((unwrap(a), False))
                case Z3OpEq(a, b) if val is True:
                    self.union(unwrap(a), unwrap(b))
                case Z3OpDistinct(a, b) if val is False:
                    self.union(unwrap(a), unwrap(b))


def _transfer_known_bits(node: Z3Node, width: int, kb_map: Dict[int, KnownBits]) -> KnownBits:
    """Compute sound `KnownBits` over lifted DSL `Z3Node` using structural `match`/`case`."""
    mask = (1 << width) - 1

    def get_kb(e: object) -> KnownBits:
        raw = unwrap(e)
        if isinstance(raw, z3.BitVecNumRef):
            return KnownBits.const(raw.as_long(), raw.size())
        return kb_map.get(raw.get_id(), KnownBits.top(raw.size()))

    match node:
        case Z3BvConst(val, w):
            return KnownBits.const(val, w)
        case Z3OpZeroExt(k, src):
            sr = unwrap(src)
            sk = get_kb(sr)
            hi_zeros = ((1 << k) - 1) << sr.size()
            return KnownBits(width, (sk.zeros | hi_zeros) & mask, sk.ones & mask)
        case Z3OpSignExt(k, src):
            sr = unwrap(src)
            sk = get_kb(sr)
            sw = sr.size()
            sign_bit = 1 << (sw - 1)
            hi_mask = ((1 << k) - 1) << sw
            if sk.zeros & sign_bit:
                return KnownBits(width, (sk.zeros | hi_mask) & mask, sk.ones & mask)
            if sk.ones & sign_bit:
                return KnownBits(width, sk.zeros & mask, (sk.ones | hi_mask) & mask)
            return KnownBits(width, sk.zeros & mask, sk.ones & mask)
        case Z3OpExtract(_, lo, src):
            sk = get_kb(src)
            return KnownBits(width, (sk.zeros >> lo) & mask, (sk.ones >> lo) & mask)
        case Z3OpConcat(args):
            z_acc, o_acc = 0, 0
            for ch in args:
                cr = unwrap(ch)
                ck = get_kb(cr)
                cw = cr.size()
                z_acc = (z_acc << cw) | ck.zeros
                o_acc = (o_acc << cw) | ck.ones
            return KnownBits(width, z_acc & mask, o_acc & mask)
        case Z3OpBvAnd(lhs, rhs, _):
            ka, kb = get_kb(lhs), get_kb(rhs)
            return KnownBits(width, (ka.zeros | kb.zeros) & mask, (ka.ones & kb.ones) & mask)
        case Z3OpBvOr(lhs, rhs, _):
            ka, kb = get_kb(lhs), get_kb(rhs)
            return KnownBits(width, (ka.zeros & kb.zeros) & mask, (ka.ones | kb.ones) & mask)
        case Z3OpBvXor(lhs, rhs, _):
            ka, kb = get_kb(lhs), get_kb(rhs)
            z = ((ka.zeros & kb.zeros) | (ka.ones & kb.ones)) & mask
            o = ((ka.zeros & kb.ones) | (ka.ones & kb.zeros)) & mask
            return KnownBits(width, z, o)
        case Z3OpBvShl(lhs, Z3BvConst(sh, _), _):
            if sh >= width:
                return KnownBits.const(0, width)
            ka = get_kb(lhs)
            lo_zeros = (1 << sh) - 1
            return KnownBits(width, ((ka.zeros << sh) | lo_zeros) & mask, (ka.ones << sh) & mask)
        case Z3OpBvLShr(lhs, Z3BvConst(sh, _), _):
            if sh >= width:
                return KnownBits.const(0, width)
            ka = get_kb(lhs)
            hi_zeros = ((1 << sh) - 1) << (width - sh)
            return KnownBits(width, ((ka.zeros >> sh) | hi_zeros) & mask, (ka.ones >> sh) & mask)
        case Z3OpIte(_, t_arm, f_arm):
            return get_kb(t_arm).join(get_kb(f_arm))
    return KnownBits.top(width)


def _apply_extensional_rules(lifted: Z3Node, raw: z3.ExprRef, env: EqualityTable) -> Optional[z3.ExprRef]:
    """Apply extensional equality & boolean truth identities via DSL `match`/`case` and `.lower()`."""
    if z3.is_bool(raw):
        bv = env.lookup_bool(raw)
        if bv is not None and not (z3.is_true(raw) or z3.is_false(raw)):
            return Z3BoolConst(bv).lower()

    rep = env.find_rep(raw)
    if not rep.eq(raw):
        return rep

    match lifted:
        case Z3OpEq(a, b) if env.are_equal(unwrap(a), unwrap(b)):
            return Z3BoolConst(True).lower()
        case Z3OpDistinct(a, b) if env.are_equal(unwrap(a), unwrap(b)):
            return Z3BoolConst(False).lower()
        case Z3OpIte(Z3BoolConst(True), t_arm, _):
            return unwrap(t_arm)
        case Z3OpIte(Z3BoolConst(False), _, f_arm):
            return unwrap(f_arm)
        case Z3OpIte(_, t_arm, f_arm) if env.are_equal(unwrap(t_arm), unwrap(f_arm)):
            return unwrap(t_arm)
        case Z3OpBvSub(a, b, w) | Z3OpBvXor(a, b, w) if env.are_equal(unwrap(a), unwrap(b)):
            return Z3BvConst(0, w).lower()
        case Z3OpRealSub(a, b) if env.are_equal(unwrap(a), unwrap(b)):
            return Z3RealConst(0).lower()
    return None


def _rewrite_with_dataflow(
    root: z3.ExprRef,
    eq_table: EqualityTable,
    *,
    max_path_depth: int = 2,
    max_nodes: int = 50000,
) -> Tuple[z3.ExprRef, int]:
    """Propagate `eq_table` and `KnownBits` through `root` using DSL-lifted match rules."""
    rewrites = 0

    def _run_dag(expr: z3.ExprRef, env: EqualityTable) -> z3.ExprRef:
        nonlocal rewrites
        memo: Dict[int, z3.ExprRef] = {}
        kb_map: Dict[int, KnownBits] = {}
        stack: List[Tuple[z3.ExprRef, bool]] = [(expr, False)]

        while stack and len(memo) < max_nodes:
            cur, expanded = stack.pop()
            cid = cur.get_id()
            if cid in memo:
                continue

            if not expanded:
                bv = env.lookup_bool(cur) if z3.is_bool(cur) else None
                if bv is not None and not (z3.is_true(cur) or z3.is_false(cur)):
                    memo[cid] = Z3BoolConst(bv).lower()
                    rewrites += 1
                    continue
                rep = env.find_rep(cur)
                if not rep.eq(cur):
                    memo[cid] = rep
                    if isinstance(rep, z3.BitVecNumRef):
                        kb_map[rep.get_id()] = KnownBits.const(rep.as_long(), rep.size())
                    rewrites += 1
                    continue

                if not z3.is_app(cur) or cur.num_args() == 0:
                    memo[cid] = cur
                    if isinstance(cur, z3.BitVecNumRef):
                        kb_map[cid] = KnownBits.const(cur.as_long(), cur.size())
                    continue

                if cur.decl().kind() == z3.Z3_OP_ITE and cur.num_args() == 3:
                    cond_b = env.lookup_bool(cur.arg(0))
                    if cond_b is True:
                        stack.append((cur, True))
                        if cur.arg(1).get_id() not in memo:
                            stack.append((cur.arg(1), False))
                        continue
                    if cond_b is False:
                        stack.append((cur, True))
                        if cur.arg(2).get_id() not in memo:
                            stack.append((cur.arg(2), False))
                        continue

                stack.append((cur, True))
                for i in range(cur.num_args() - 1, -1, -1):
                    ch = cur.arg(i)
                    if ch.get_id() not in memo:
                        stack.append((ch, False))
            else:
                if cur.decl().kind() == z3.Z3_OP_ITE and cur.num_args() == 3:
                    cond_b = env.lookup_bool(cur.arg(0))
                    if cond_b is True and cur.arg(1).get_id() in memo:
                        memo[cid] = memo[cur.arg(1).get_id()]
                        rewrites += 1
                        continue
                    if cond_b is False and cur.arg(2).get_id() in memo:
                        memo[cid] = memo[cur.arg(2).get_id()]
                        rewrites += 1
                        continue

                nargs = cur.num_args()
                orig_children = [cur.arg(i) for i in range(nargs)]
                new_children = [memo.get(ch.get_id(), ch) for ch in orig_children]
                any_changed = any(not nc.eq(oc) for nc, oc in zip(new_children, orig_children))
                rebuilt = rebuild_app(cur, new_children) if any_changed else cur

                lifted = lift(rebuilt, depth=1)
                simplified = _apply_extensional_rules(lifted, rebuilt, env)
                if simplified is not None and not simplified.eq(rebuilt):
                    rebuilt = simplified
                    lifted = lift(rebuilt, depth=1)
                    rewrites += 1

                if isinstance(rebuilt, z3.BitVecRef) and not isinstance(rebuilt, z3.BitVecNumRef):
                    kb = _transfer_known_bits(lifted, rebuilt.size(), kb_map)
                    if kb.is_const:
                        rebuilt = Z3BvConst(kb.ones, kb.width).lower()
                        kb_map[rebuilt.get_id()] = kb
                        rewrites += 1
                    else:
                        kb_map[rebuilt.get_id()] = kb
                elif isinstance(rebuilt, z3.BitVecNumRef):
                    kb_map[rebuilt.get_id()] = KnownBits.const(rebuilt.as_long(), rebuilt.size())

                memo[cid] = rebuilt

        return memo.get(expr.get_id(), expr)

    def _run_path_sensitive(expr: z3.ExprRef, env: EqualityTable, depth: int) -> z3.ExprRef:
        nonlocal rewrites
        base = _run_dag(expr, env)
        if depth <= 0 or count_ast_nodes([base], limit=513) > 512:
            return base
        match lift(base, depth=1):
            case Z3OpIte(Z3BoolConst(True), t_arm, _):
                rewrites += 1
                return _run_path_sensitive(unwrap(t_arm), env, depth - 1)
            case Z3OpIte(Z3BoolConst(False), _, f_arm):
                rewrites += 1
                return _run_path_sensitive(unwrap(f_arm), env, depth - 1)
            case Z3OpIte(cond, t_arm, f_arm):
                cr, tr, fr = unwrap(cond), unwrap(t_arm), unwrap(f_arm)
                t_env = EqualityTable(env)
                t_env.assume_bool(cr, True)
                new_t = _run_path_sensitive(tr, t_env, depth - 1)

                f_env = EqualityTable(env)
                f_env.assume_bool(cr, False)
                new_f = _run_path_sensitive(fr, f_env, depth - 1)

                if z3.expr_eq(new_t, new_f):
                    rewrites += 1
                    return new_t
                if not new_t.eq(tr) or not new_f.eq(fr):
                    rewrites += 1
                    return Z3OpIte(cr, new_t, new_f).lower()
        return base

    out = _run_path_sensitive(root, eq_table, max_path_depth)
    return out, rewrites


class DataflowPropagatePass(Z3QueryPass):
    """
    Global Z3-domain dataflow pass: propagates precondition & guard equalities
    (`EqualityTable`), BitVec `KnownBits`, and path-sensitive `If` facts across
    `lhs` and `rhs` using lifted `dsl.py` types.
    """

    name = "dataflow_propagate"
    is_equivalence_preserving = True

    def __init__(self, max_path_depth: int = 2) -> None:
        self.max_path_depth = max(0, int(max_path_depth))

    def run(
        self,
        query: SimplifyQuery,
        ctx: SimplificationContext,
    ) -> Tuple[SimplifyQuery, PassStats]:
        t0 = time.perf_counter()
        if query.is_trivially_unsat():
            return query, PassStats(self.name, False, 0, 0.0, "trivial_unsat")

        env = EqualityTable()
        for pre in query.preconditions:
            env.assume_bool(pre, True)
        if query.guard is not None:
            env.assume_bool(query.guard, True)

        new_lhs, r_l = _rewrite_with_dataflow(
            query.lhs, env, max_path_depth=self.max_path_depth, max_nodes=ctx.max_dag_nodes
        )
        new_rhs, r_r = _rewrite_with_dataflow(
            query.rhs, env, max_path_depth=self.max_path_depth, max_nodes=ctx.max_dag_nodes
        )

        if not z3.expr_eq(new_lhs, new_rhs) and env.are_equal(new_lhs, new_rhs):
            rep = env.find_rep(new_lhs)
            new_lhs, new_rhs = rep, rep
            r_l += 1

        total_r = r_l + r_r
        changed = not new_lhs.eq(query.lhs) or not new_rhs.eq(query.rhs)
        out_q = query.with_terms(lhs=new_lhs, rhs=new_rhs) if changed else query
        dt = (time.perf_counter() - t0) * 1000.0
        return out_q, PassStats(
            self.name,
            changed,
            total_r if changed else 0,
            dt,
            f"eq_facts={len(env._uf)} bool_facts={len(env._bool_vals)}",
        )

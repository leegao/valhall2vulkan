#!/usr/bin/env python3
"""
Z3-domain dataflow analysis and extensional-equality propagation engine.

Cleanly separates global/relational dataflow reasoning from local peephole
rewrites (`passes_bv.py`, `passes_real.py`):

  1. Extensional Equality & Congruence (`EqualityTable` / Union-Find):
     Harvests ground truths (`p`, `Not(p)`) and equalities (`a == b`) from
     `preconditions` and `guard`. Maintains equivalence classes whose canonical
     representative prefers concrete constants first, then smaller ASTs.
     Applies extensional identities:
       - `x ~ y` => `x == y -> True`, `x != y -> False`, `x - y -> 0`, `x ^ y -> 0`
       - Congruence: replacing `x` with `rep(x)` normalizes `f(x)` and `f(y)`
         to the same AST node in a single pass.

  2. BitVec Known-Bits Forward Dataflow Lattice (`KnownBits`):
     Propagates per-bit `(zeros, ones)` masks bottom-up through `BitVecVal`,
     `ZeroExt`, `SignExt`, `Concat`, `Extract`, `BvAnd`, `BvOr`, `BvXor`,
     `BvShl`, `LShR`, and `If` (join = bitwise intersection of known zeros/ones).
     Whenever `(zeros | ones) == (1 << width) - 1`, collapses the node directly
     to `BitVecVal(ones, width)` — soundly proving multi-step bit-mask/extract
     identities such as `Extract(63, 32, ZeroExt(32, x)) == 0` or
     `((x << 4) & 0xF) == 0` without ad-hoc peephole chains.

  3. Path-Sensitive Conditional Propagation (`If(c, then_e, else_e)`):
     Refines the dataflow environment along `then_e` with `c = True` (including
     equalities inside `c`) and along `else_e` with `c = False`, collapsing
     redundant nested guards (`If(c, If(c, a, b), d) -> If(c, a, d)`),
     `discarded`-guarded fragment outputs (`guard = Not(discarded)` collapses
     `If(discarded, undef, color) -> color`), and guarded constant substitutions
     (`If(x == 0, x + y, z) -> If(x == 0, y, z)`).

Soundness:
  `DataflowPropagatePass` is **100% equivalence-preserving**
  (`is_equivalence_preserving = True`): every fact is derived strictly from
  `preconditions`, `guard`, dominating `If` conditions, or sound abstract
  interpretation of BitVec semantics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import z3

from tools.symbolic_ssa.z3simplifier.ast_utils import count_ast_nodes, rebuild_app
from tools.symbolic_ssa.z3simplifier.context import (
    PassStats,
    SimplificationContext,
    SimplifyQuery,
)
from tools.symbolic_ssa.z3simplifier.pass_base import Z3QueryPass


@dataclass(frozen=True, slots=True)
class KnownBits:
    """
    Sound known-bits abstract domain element for a `width`-bit BitVector.

    Invariant: `(zeros & ones) == 0` and `(zeros | ones) <= mask`.
      - bit `i` of `zeros` is 1 iff bit `i` of the concrete value is provably 0.
      - bit `i` of `ones`  is 1 iff bit `i` of the concrete value is provably 1.
    """

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
    """Ranking key for canonical equivalence-class representative (smaller is better)."""
    if _is_ground_const(e):
        return (0, 0, e.get_id())
    if z3.is_const(e):
        return (1, 0, e.get_id())
    nargs = e.num_args() if z3.is_app(e) else 0
    return (2, nargs, e.get_id())


class EqualityTable:
    """
    Scoped Union-Find table of extensional equalities (`a == b`) and boolean
    truth assignments (`cond == True/False`).
    """

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
        if a.eq(b):
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
        if a.eq(b):
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
        # Propagate boolean truth if either class had a known truth value
        ba = self._bool_vals.get(ra)
        bb = self._bool_vals.get(rb)
        if ba is None and bb is not None:
            self._bool_vals[ra] = bb
        elif z3.is_true(ea):
            self._bool_vals[ra] = True
        elif z3.is_false(ea):
            self._bool_vals[ra] = False

    def assume_bool(self, pred: z3.ExprRef, truth: bool) -> None:
        """Harvest boolean truth and any nested `And`/`Not`/`Eq` facts from `pred == truth`."""
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
            self.union(cur, z3.BoolVal(val))

            if not z3.is_app(cur):
                continue
            kind = cur.decl().kind()
            if kind == z3.Z3_OP_NOT and cur.num_args() == 1:
                work.append((cur.arg(0), not val))
            elif kind == z3.Z3_OP_AND and val is True:
                for i in range(cur.num_args()):
                    work.append((cur.arg(i), True))
            elif kind == z3.Z3_OP_OR and val is False:
                for i in range(cur.num_args()):
                    work.append((cur.arg(i), False))
            elif kind == z3.Z3_OP_EQ and cur.num_args() == 2 and val is True:
                a, b = cur.arg(0), cur.arg(1)
                self.union(a, b)
            elif kind == z3.Z3_OP_DISTINCT and cur.num_args() == 2 and val is False:
                a, b = cur.arg(0), cur.arg(1)
                self.union(a, b)

    def has_facts(self) -> bool:
        return bool(self._uf or self._bool_vals)


def _transfer_known_bits(node: z3.ExprRef, kb_map: Dict[int, KnownBits]) -> Optional[KnownBits]:
    """Compute sound `KnownBits` for a BitVec expression from its children's known bits."""
    if not isinstance(node, z3.BitVecRef):
        return None
    w = node.size()
    if isinstance(node, z3.BitVecNumRef):
        return KnownBits.const(node.as_long(), w)
    if not z3.is_app(node):
        return KnownBits.top(w)

    kind = node.decl().kind()
    nargs = node.num_args()
    mask = (1 << w) - 1

    def get_kb(e: z3.ExprRef) -> KnownBits:
        if isinstance(e, z3.BitVecNumRef):
            return KnownBits.const(e.as_long(), e.size())
        return kb_map.get(e.get_id(), KnownBits.top(e.size()))

    if kind == z3.Z3_OP_ZERO_EXT and nargs == 1:
        src = node.arg(0)
        sk = get_kb(src)
        k = node.params()[0]
        hi_zeros = ((1 << k) - 1) << src.size()
        return KnownBits(w, (sk.zeros | hi_zeros) & mask, sk.ones & mask)

    if kind == z3.Z3_OP_SIGN_EXT and nargs == 1:
        src = node.arg(0)
        sk = get_kb(src)
        sw = src.size()
        k = node.params()[0]
        sign_bit = 1 << (sw - 1)
        hi_mask = ((1 << k) - 1) << sw
        if sk.zeros & sign_bit:
            return KnownBits(w, (sk.zeros | hi_mask) & mask, sk.ones & mask)
        if sk.ones & sign_bit:
            return KnownBits(w, sk.zeros & mask, (sk.ones | hi_mask) & mask)
        return KnownBits(w, sk.zeros & mask, sk.ones & mask)

    if kind == z3.Z3_OP_EXTRACT and nargs == 1:
        hi, lo = node.params()
        sk = get_kb(node.arg(0))
        return KnownBits(w, (sk.zeros >> lo) & mask, (sk.ones >> lo) & mask)

    if kind == z3.Z3_OP_CONCAT and nargs >= 2:
        z_acc, o_acc = 0, 0
        for i in range(nargs):
            ch = node.arg(i)
            ck = get_kb(ch)
            cw = ch.size()
            z_acc = (z_acc << cw) | ck.zeros
            o_acc = (o_acc << cw) | ck.ones
        return KnownBits(w, z_acc & mask, o_acc & mask)

    if kind == z3.Z3_OP_BAND and nargs == 2:
        ka, kb = get_kb(node.arg(0)), get_kb(node.arg(1))
        return KnownBits(w, (ka.zeros | kb.zeros) & mask, (ka.ones & kb.ones) & mask)

    if kind == z3.Z3_OP_BOR and nargs == 2:
        ka, kb = get_kb(node.arg(0)), get_kb(node.arg(1))
        return KnownBits(w, (ka.zeros & kb.zeros) & mask, (ka.ones | kb.ones) & mask)

    if kind == z3.Z3_OP_BXOR and nargs == 2:
        ka, kb = get_kb(node.arg(0)), get_kb(node.arg(1))
        z = ((ka.zeros & kb.zeros) | (ka.ones & kb.ones)) & mask
        o = ((ka.zeros & kb.ones) | (ka.ones & kb.zeros)) & mask
        return KnownBits(w, z, o)

    if kind == z3.Z3_OP_BSHL and nargs == 2:
        rhs = node.arg(1)
        if isinstance(rhs, z3.BitVecNumRef):
            sh = rhs.as_long()
            if sh >= w:
                return KnownBits.const(0, w)
            ka = get_kb(node.arg(0))
            lo_zeros = (1 << sh) - 1
            return KnownBits(w, ((ka.zeros << sh) | lo_zeros) & mask, (ka.ones << sh) & mask)

    if kind == z3.Z3_OP_BLSHR and nargs == 2:
        rhs = node.arg(1)
        if isinstance(rhs, z3.BitVecNumRef):
            sh = rhs.as_long()
            if sh >= w:
                return KnownBits.const(0, w)
            ka = get_kb(node.arg(0))
            hi_zeros = ((1 << sh) - 1) << (w - sh)
            return KnownBits(w, ((ka.zeros >> sh) | hi_zeros) & mask, (ka.ones >> sh) & mask)

    if kind == z3.Z3_OP_ITE and nargs == 3:
        kt, kf = get_kb(node.arg(1)), get_kb(node.arg(2))
        return kt.join(kf)

    return KnownBits.top(w)


def _rewrite_with_dataflow(
    root: z3.ExprRef,
    eq_table: EqualityTable,
    *,
    max_path_depth: int = 6,
    max_nodes: int = 50000,
) -> Tuple[z3.ExprRef, int]:
    """
    Propagate `eq_table` (extensional equalities + boolean facts) and `KnownBits`
    through `root`.

    Uses a fast iterative bottom-up DAG pass for the main expression body, and
    performs bounded path-sensitive specialization on `If(c, then_e, else_e)`
    nodes up to `max_path_depth` when `c` carries new boolean/equality facts.
    """
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
                # Check if `cur` already resolves via boolean truth or canonical equality rep
                bv = env.lookup_bool(cur) if z3.is_bool(cur) else None
                if bv is not None and not (z3.is_true(cur) or z3.is_false(cur)):
                    memo[cid] = z3.BoolVal(bv)
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

                # Short-circuit dead `If` arm before traversing it
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
                # Short-circuit If when cond was already known
                if cur.decl().kind() == z3.Z3_OP_ITE and cur.num_args() == 3:
                    cond_b = env.lookup_bool(cur.arg(0))
                    if cond_b is True and cur.arg(1).get_id() in memo:
                        res = memo[cur.arg(1).get_id()]
                        memo[cid] = res
                        rewrites += 1
                        continue
                    if cond_b is False and cur.arg(2).get_id() in memo:
                        res = memo[cur.arg(2).get_id()]
                        memo[cid] = res
                        rewrites += 1
                        continue

                nargs = cur.num_args()
                orig_children = [cur.arg(i) for i in range(nargs)]
                new_children = [memo.get(ch.get_id(), ch) for ch in orig_children]
                any_changed = any(not nc.eq(oc) for nc, oc in zip(new_children, orig_children))
                rebuilt = rebuild_app(cur, new_children) if any_changed else cur

                # Apply extensional equality rules on the rebuilt node
                simplified = _apply_extensional_rules(rebuilt, env)
                if simplified is not None and not simplified.eq(rebuilt):
                    rebuilt = simplified
                    rewrites += 1

                # Apply KnownBits lattice transfer & constant collapse
                if isinstance(rebuilt, z3.BitVecRef) and not isinstance(rebuilt, z3.BitVecNumRef):
                    kb = _transfer_known_bits(rebuilt, kb_map)
                    if kb is not None:
                        if kb.is_const:
                            rebuilt = z3.BitVecVal(kb.ones, kb.width)
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
        if not (z3.is_app(base) and base.decl().kind() == z3.Z3_OP_ITE and base.num_args() == 3):
            return base
        cond, t_arm, f_arm = base.arg(0), base.arg(1), base.arg(2)
        if z3.is_true(cond):
            rewrites += 1
            return _run_path_sensitive(t_arm, env, depth - 1)
        if z3.is_false(cond):
            rewrites += 1
            return _run_path_sensitive(f_arm, env, depth - 1)

        t_env = EqualityTable(env)
        t_env.assume_bool(cond, True)
        new_t = _run_path_sensitive(t_arm, t_env, depth - 1)

        f_env = EqualityTable(env)
        f_env.assume_bool(cond, False)
        new_f = _run_path_sensitive(f_arm, f_env, depth - 1)

        if new_t.eq(new_f):
            rewrites += 1
            return new_t
        if not new_t.eq(t_arm) or not new_f.eq(f_arm):
            rewrites += 1
            return z3.If(cond, new_t, new_f)
        return base

    out = _run_path_sensitive(root, eq_table, max_path_depth)
    return out, rewrites


def _apply_extensional_rules(node: z3.ExprRef, env: EqualityTable) -> Optional[z3.ExprRef]:
    """Apply extensional equality & boolean truth identities on a single rebuilt node."""
    if z3.is_bool(node):
        bv = env.lookup_bool(node)
        if bv is not None and not (z3.is_true(node) or z3.is_false(node)):
            return z3.BoolVal(bv)

    rep = env.find_rep(node)
    if not rep.eq(node):
        return rep

    if not z3.is_app(node):
        return None
    kind = node.decl().kind()
    nargs = node.num_args()

    if kind == z3.Z3_OP_EQ and nargs == 2:
        a, b = node.arg(0), node.arg(1)
        if env.are_equal(a, b):
            return z3.BoolVal(True)
    elif kind == z3.Z3_OP_DISTINCT and nargs == 2:
        a, b = node.arg(0), node.arg(1)
        if env.are_equal(a, b):
            return z3.BoolVal(False)
    elif kind == z3.Z3_OP_ITE and nargs == 3:
        c, t_arm, f_arm = node.arg(0), node.arg(1), node.arg(2)
        if z3.is_true(c):
            return t_arm
        if z3.is_false(c):
            return f_arm
        if env.are_equal(t_arm, f_arm):
            return t_arm
    elif kind in (z3.Z3_OP_BSUB, z3.Z3_OP_BXOR) and nargs == 2 and isinstance(node, z3.BitVecRef):
        if env.are_equal(node.arg(0), node.arg(1)):
            return z3.BitVecVal(0, node.size())
    elif kind == z3.Z3_OP_SUB and nargs == 2 and isinstance(node, z3.ArithRef) and node.is_real():
        if env.are_equal(node.arg(0), node.arg(1)):
            return z3.RealVal(0)
    return None


class DataflowPropagatePass(Z3QueryPass):
    """
    Global Z3-domain dataflow pass: propagates precondition & guard equalities
    (`EqualityTable`), BitVec `KnownBits`, and path-sensitive `If` facts across
    `lhs` and `rhs`.

    100% equivalence-preserving (`is_equivalence_preserving = True`).
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

        # If `lhs` and `rhs` belong to the same equivalence class under `env`, discharge immediately
        if not new_lhs.eq(new_rhs) and env.are_equal(new_lhs, new_rhs):
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

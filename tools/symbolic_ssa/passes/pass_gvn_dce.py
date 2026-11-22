#!/usr/bin/env python3
"""
Pass 1.6: Algebraic Normalization, Dominator-Scoped GVN, and DCE.

Uses AlgebraicSimplifierInterpreter (plugged into SsaInterpreter) to discover
semantics-preserving local node-identity rewrites (Copy, Bitcast roundtrips,
LogicalNot involution, trivial Select, VecExtract(VecConstruct), Pack64/Split64,
and integer additive/bitwise identities), followed by dominator-tree scoped
hash-consing GVN and mark-and-sweep DCE.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

from tools.symbolic_ssa.passes.ir_utils import (
    COMMUTATIVE_OPS,
    NEVER_GVN_OPS,
    SIDE_EFFECT_OPS,
    AbstractSsaPass,
    Stratum1Stats,
    compute_idom,
    compute_rpo,
    replace_all_uses,
)


class AlgebraicSimplifierInterpreter(AbstractSsaPass):
    """
    Abstract interpreter over SSA node identities.

    Each node evaluates to its canonical replacement node ID (`int`), recording
    any nontrivial rewrite into `self.repl[nid] = canonical_id`.
    """

    def __init__(self, blocks_by_id: Dict[int, Dict[str, Any]], nodes_by_id: Dict[int, Dict[str, Any]]):
        super().__init__(blocks_by_id, nodes_by_id)
        self.repl: Dict[int, int] = {}
        self._cur_node: Optional[Dict[str, Any]] = None

    def lookup_arg(self, values: Dict[int, Any], aid: int) -> Any:
        return values.get(aid, aid)

    def _const_u32_val(self, nid: Any) -> Any:
        if not isinstance(nid, int):
            return None
        n = self.nodes_by_id.get(nid)
        if n and n["op"] in ("ConstU32", "ConstS32"):
            return int(n.get("imm_u32", 0))
        return None

    def eval_node(self, node: Dict[str, Any], values: Dict[int, Any], state: Any = None, reach: Any = None) -> Any:
        nid = int(node["id"])
        self._cur_node = node
        res = super().eval_node(node, values, state, reach)
        if isinstance(res, int) and res > 0:
            if res != nid:
                self.repl[nid] = res
            return res
        # Preserve tuples for VecConstruct / Split64 / Pack64 propagation
        if isinstance(res, tuple):
            return res
        return nid

    def cast_value(self, op: str, val: Any, val_type: str) -> Any:
        if op == "Copy" and self._cur_node is not None and self._cur_node["op"] == "Copy" and isinstance(val, int):
            return val
        if op == "Bitcast" and isinstance(val, int) and self._cur_node is not None:
            an = self.nodes_by_id.get(val)
            if an and an.get("type") == val_type:
                return val
            if an and an["op"] == "Bitcast" and len(an.get("args", [])) == 1:
                inner_id = self.lookup_arg(self.abs_vals, int(an["args"][0]))
                if isinstance(inner_id, int):
                    inner = self.nodes_by_id.get(inner_id)
                    if inner and inner.get("type") == val_type:
                        return inner_id
        return None

    def pack64(self, lo: Any, hi: Any) -> Any:
        return ("__pack64__", lo, hi)

    def split64(self, val64: Any) -> Tuple[Any, Any]:
        if isinstance(val64, tuple) and len(val64) == 3 and val64[0] == "__pack64__":
            return (val64[1], val64[2])
        return (None, None)

    def logical_not(self, a: Any) -> Any:
        if isinstance(a, int):
            an = self.nodes_by_id.get(a)
            if an and an["op"] == "LogicalNot" and len(an.get("args", [])) == 1:
                return self.lookup_arg(self.abs_vals, int(an["args"][0]))
        return None

    def select(self, cond: Any, t_val: Any, f_val: Any) -> Any:
        if isinstance(t_val, int) and t_val == f_val:
            return t_val
        if isinstance(cond, int):
            cn = self.nodes_by_id.get(cond)
            if cn and cn["op"] in ("ConstBool", "ConstU32", "ConstS32"):
                is_true = (int(cn.get("imm_u32", 0)) != 0) or bool(cn.get("aux_bool", False))
                picked = t_val if is_true else f_val
                if isinstance(picked, int):
                    return picked
        return None

    def int32_binary(self, op: str, a: Any, b: Any) -> Any:
        if not (isinstance(a, int) and isinstance(b, int)):
            return None
        ca = self._const_u32_val(a)
        cb = self._const_u32_val(b)
        if op in ("IAdd", "BitOr", "BitXor"):
            if cb == 0:
                return a
            if ca == 0:
                return b
        elif op == "ISub":
            if cb == 0:
                return a
        elif op == "BitAnd":
            if a == b:
                return a
            if cb == 0xFFFFFFFF:
                return a
            if ca == 0xFFFFFFFF:
                return b
        elif op == "IMul":
            if cb == 1:
                return a
            if ca == 1:
                return b
        return None


def pass_gvn_dce(
    blocks_by_id: Dict[int, Dict[str, Any]],
    nodes_by_id: Dict[int, Dict[str, Any]],
    readonly_load_ids: Set[int],
    stats: Stratum1Stats,
) -> bool:
    """Run algebraic simplification, dominator-scoped GVN, and mark-and-sweep DCE."""
    changed = False

    simp = AlgebraicSimplifierInterpreter(blocks_by_id, nodes_by_id)
    simp.run_rpo()
    if simp.repl:
        replace_all_uses(blocks_by_id, nodes_by_id, simp.repl)
        stats.alg_simplified += len(simp.repl)
        changed = True

    rpo = compute_rpo(blocks_by_id, 0)
    idom = compute_idom(blocks_by_id, rpo)
    dom_children: Dict[int, List[int]] = {b: [] for b in rpo}
    for b in rpo[1:]:
        p = idom.get(b)
        if p is not None and p != b:
            dom_children[p].append(b)

    gvn_repl: Dict[int, int] = {}

    def resolve_gvn(x: int) -> int:
        while x in gvn_repl:
            x = gvn_repl[x]
        return x

    def can_gvn(nid: int, n: Dict[str, Any]) -> bool:
        op = n["op"]
        if op in NEVER_GVN_OPS:
            return False
        if op in ("PsbLoad", "TexSample"):
            return nid in readonly_load_ids
        return True

    def gvn_key(n: Dict[str, Any]) -> Tuple[Any, ...]:
        op = n["op"]
        args = [resolve_gvn(int(a)) for a in n.get("args", [])]
        if op in COMMUTATIVE_OPS and len(args) == 2 and args[0] > args[1]:
            args = [args[1], args[0]]
        return (
            op,
            n.get("type", "U32"),
            tuple(args),
            int(n.get("imm_u32", 0)),
            int(n.get("imm_u32_2", 0)),
            int(n.get("imm_u64", 0)),
            int(n.get("aux_index", 0)),
            bool(n.get("aux_bool", False)),
            bool(n.get("aux_bool2", False)),
            n.get("aux_str", ""),
        )

    def walk_dom(bid: int, table: Dict[Tuple[Any, ...], int]) -> None:
        local_keys: List[Tuple[Any, ...]] = []
        blk = blocks_by_id[bid]
        for nid in blk.get("nodes", []):
            n = nodes_by_id.get(nid)
            if not n or not can_gvn(nid, n):
                continue
            key = gvn_key(n)
            prev = table.get(key)
            if prev is not None and prev != nid:
                gvn_repl[nid] = prev
            else:
                table[key] = nid
                local_keys.append(key)
        for ch in dom_children.get(bid, []):
            walk_dom(ch, table)
        for k in local_keys:
            table.pop(k, None)

    if rpo:
        walk_dom(rpo[0], {})

    if gvn_repl:
        replace_all_uses(blocks_by_id, nodes_by_id, gvn_repl)
        stats.gvn_merged += len(gvn_repl)
        changed = True

    live: Set[int] = set()
    worklist: List[int] = []

    def mark(nid: int) -> None:
        if nid > 0 and nid not in live and nid in nodes_by_id:
            live.add(nid)
            worklist.append(nid)

    for blk in blocks_by_id.values():
        bc = int(blk.get("branch_cond", 0))
        if bc > 0:
            mark(bc)
        for nid in blk.get("nodes", []):
            n = nodes_by_id.get(nid)
            if n and n["op"] in SIDE_EFFECT_OPS:
                mark(nid)

    while worklist:
        cur = worklist.pop()
        n = nodes_by_id[cur]
        for a in n.get("args", []):
            mark(int(a))
        for _, v in n.get("phi", []):
            mark(int(v))

    dead_nids = [nid for nid in list(nodes_by_id.keys()) if nid not in live]
    if dead_nids:
        for nid in dead_nids:
            del nodes_by_id[nid]
        for blk in blocks_by_id.values():
            blk["phis"] = [p for p in blk.get("phis", []) if p in live]
            blk["nodes"] = [x for x in blk.get("nodes", []) if x in live]
        stats.dce_removed += len(dead_nids)
        changed = True

    return changed

"""SSA Dominance and Def-Before-Use Verifier for ValhallSSA modules.

Computes the Cooper-Harvey-Kennedy immediate-dominator tree over reachable CFG
blocks and verifies that every non-Phi operand, block `branch_cond`, and Phi
incoming value is defined and strictly dominates its use site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tools.symbolic_ssa.interpreter import SsaInterpreter


@dataclass
class DominanceViolation:
    """Single SSA dominance or missing-definition violation."""

    use_block: int
    use_node: int
    use_op: str
    arg_id: int
    reason: str


def compute_idom(blocks_by_id: dict[int, dict[str, Any]], rpo: list[int]) -> dict[int, int]:
    """Compute immediate dominators over `rpo` using the iterative Cooper-Harvey-Kennedy algorithm."""
    if not rpo:
        return {}
    rpo_idx = {b: i for i, b in enumerate(rpo)}
    idom: dict[int, int] = {rpo[0]: rpo[0]}

    def intersect(b1: int, b2: int) -> int:
        while b1 != b2:
            while rpo_idx[b1] > rpo_idx[b2]:
                b1 = idom[b1]
            while rpo_idx[b2] > rpo_idx[b1]:
                b2 = idom[b2]
        return b1

    changed = True
    while changed:
        changed = False
        for b in rpo[1:]:
            preds = [int(p) for p in blocks_by_id[b].get("preds", []) if int(p) in idom]
            if not preds:
                continue
            new_idom = preds[0]
            for p in preds[1:]:
                new_idom = intersect(p, new_idom)
            if idom.get(b) != new_idom:
                idom[b] = new_idom
                changed = True
    return idom


def verify_ssa_dominance(
    ssa_or_blocks: dict[str, Any],
    nodes_by_id: dict[int, dict[str, Any]] | None = None,
) -> list[DominanceViolation]:
    """Verify SSA dominance and def-before-use across all reachable blocks.

    Accepts either a full SSA module dict (`{"blocks": ..., "nodes": ...}`) or
    a `(blocks_by_id, nodes_by_id)` pair. Returns an empty list when valid.
    """
    if nodes_by_id is None:
        blocks_by_id = {int(b["id"]): b for b in ssa_or_blocks.get("blocks", [])}
        nodes_map = {int(n["id"]): n for n in ssa_or_blocks.get("nodes", [])}
    else:
        blocks_by_id = ssa_or_blocks
        nodes_map = nodes_by_id

    if not blocks_by_id:
        return []

    entry = 0 if 0 in blocks_by_id else min(blocks_by_id.keys())
    rpo = SsaInterpreter.compute_rpo(blocks_by_id, entry)
    idom = compute_idom(blocks_by_id, rpo)

    def_loc: dict[int, tuple[int, int]] = {}
    for bid, blk in blocks_by_id.items():
        for pid in blk.get("phis", []):
            def_loc[int(pid)] = (bid, -1)
        for idx, nid in enumerate(blk.get("nodes", [])):
            def_loc[int(nid)] = (bid, idx)

    def dominates(a: int, b: int) -> bool:
        cur: int | None = b
        while cur is not None:
            if cur == a:
                return True
            parent = idom.get(cur)
            if parent == cur:
                break
            cur = parent
        return False

    viols: list[DominanceViolation] = []
    for bid in rpo:
        blk = blocks_by_id[bid]
        bc = int(blk.get("branch_cond", 0))
        if bc > 0:
            if bc not in def_loc:
                viols.append(DominanceViolation(bid, -1, "BranchCond", bc, "MISSING_DEF"))
            else:
                db, _ = def_loc[bc]
                if not dominates(db, bid):
                    viols.append(
                        DominanceViolation(bid, -1, "BranchCond", bc, f"NON_DOM(def_blk={db},use_blk={bid})")
                    )

        for idx, nid in enumerate(blk.get("nodes", [])):
            n = nodes_map.get(int(nid))
            if not n or n.get("op") == "Phi":
                continue
            for raw_aid in n.get("args", []):
                aid = int(raw_aid)
                if aid <= 0:
                    continue
                if aid not in def_loc:
                    viols.append(DominanceViolation(bid, int(nid), n["op"], aid, "MISSING_DEF"))
                    continue
                db, di = def_loc[aid]
                if db == bid:
                    if di >= idx:
                        viols.append(
                            DominanceViolation(bid, int(nid), n["op"], aid, f"SAME_BLK_AFTER({di}>={idx})")
                        )
                elif not dominates(db, bid):
                    viols.append(
                        DominanceViolation(bid, int(nid), n["op"], aid, f"NON_DOM(def_blk={db},use_blk={bid})")
                    )

    return viols

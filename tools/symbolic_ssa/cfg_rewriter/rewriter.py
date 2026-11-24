"""
Reusable CFG Rewriting & Phi-Maintenance Engine (`CfgRewriter`).

Encapsulates all low-level structural CFG mutations across `ValhallSsaModule`
basic blocks — edge rewiring, block splicing, chain merging, Phi-to-Select
lowering, unreachable block pruning, and generic dead/trivial Phi cleanup —
so individual optimization passes stay declarative and DRY.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from tools.symbolic_ssa.ir_utils import (
    SIDE_EFFECT_OPS,
    Stratum1Stats,
    block_succs,
    compute_rpo,
    extract_non_marker_nodes,
    insert_before_terminator,
    is_block_pure,
    next_node_id,
    rebuild_preds,
    replace_all_uses,
)

TERMINATOR_FIELDS = (
    "branch_target",
    "fallthrough",
    "is_cond",
    "is_uncond",
    "is_end",
    "is_discard",
    "branch_cond",
    "sel_merge",
    "loop_merge",
    "loop_backedge",
)


class CfgRewriter:
    """
    Stateful CFG transformation helper operating on `(blocks_by_id, nodes_by_id)`.

    Maintains SSA invariants (node `blk` ownership, predecessor lists, Phi
    incoming edge consistency, and dead/trivial Phi elimination) across block
    splicing, chain contraction, branch folding, and tile rewrites.
    """

    def __init__(
        self,
        blocks_by_id: dict[int, dict[str, Any]],
        nodes_by_id: dict[int, dict[str, Any]],
        stats: Stratum1Stats | None = None,
    ):
        self.blocks_by_id = blocks_by_id
        self.nodes_by_id = nodes_by_id
        self.stats = stats if stats is not None else Stratum1Stats()

    def emit_node(
        self,
        bid: int,
        op: str,
        val_type: str,
        args: Sequence[int],
        **extra: Any,
    ) -> int:
        """Allocate and register a new SSA node belonging to block `bid`."""
        nid = next_node_id(self.nodes_by_id)
        node: dict[str, Any] = {
            "id": nid,
            "op": op,
            "type": val_type,
            "blk": bid,
            "args": [int(a) for a in args],
        }
        node.update(extra)
        self.nodes_by_id[nid] = node
        return nid

    def set_unconditional_branch(self, bid: int, target_id: int) -> None:
        """Rewire block `bid` to jump unconditionally to `target_id`."""
        blk = self.blocks_by_id[bid]
        blk["is_cond"] = False
        blk["is_uncond"] = True
        blk["branch_target"] = int(target_id)
        blk["fallthrough"] = -1
        blk["branch_cond"] = 0

    def copy_terminator(self, dst_bid: int, src_blk: dict[str, Any]) -> None:
        """Copy all terminator/CFG metadata fields from `src_blk` onto `dst_bid`."""
        dst = self.blocks_by_id[dst_bid]
        for field in TERMINATOR_FIELDS:
            dst[field] = src_blk.get(field)

    def retarget_edge(self, src_bid: int, old_target: int, new_target: int) -> None:
        """Retarget any outgoing branch/fallthrough edge of `src_bid` from `old_target` to `new_target`."""
        blk = self.blocks_by_id[src_bid]
        if int(blk.get("branch_target", -1)) == old_target:
            blk["branch_target"] = int(new_target)
        if int(blk.get("fallthrough", -1)) == old_target:
            blk["fallthrough"] = int(new_target)

    def retarget_incoming_edges(
        self,
        old_pred: int,
        new_pred: int,
        target_bids: Iterable[int],
    ) -> None:
        """Rewrite predecessor lists and Phi incoming edges from `old_pred` to `new_pred` in `target_bids`."""
        for tid in target_bids:
            target = self.blocks_by_id.get(tid)
            if target is None:
                continue
            target["preds"] = [
                new_pred if int(p) == old_pred else int(p)
                for p in target.get("preds", [])
            ]
            for pid in target.get("phis", []):
                pn = self.nodes_by_id.get(pid)
                if pn and "phi" in pn:
                    pn["phi"] = [
                        [new_pred if int(p) == old_pred else int(p), int(v)]
                        for p, v in pn["phi"]
                    ]

    def lower_merge_phis_to_select(
        self,
        header_id: int,
        merge_id: int,
        cond_id: int,
        true_pred: int,
        false_pred: int,
    ) -> list[int]:
        """
        Lower Phis in `merge_id` arriving from `(true_pred, false_pred)` into
        `Select(cond_id, v_true, v_false)` nodes placed in `header_id`,
        replacing both incoming Phi edges with `[header_id, merged_val]`.
        """
        merge_blk = self.blocks_by_id[merge_id]
        select_ids: list[int] = []

        for pid in merge_blk.get("phis", []):
            pn = self.nodes_by_id.get(pid)
            if not pn:
                continue
            inc_map = {int(p): int(v) for p, v in pn.get("phi", [])}
            if true_pred not in inc_map or false_pred not in inc_map:
                continue

            v_true, v_false = inc_map[true_pred], inc_map[false_pred]
            if pn.get("type") == "Token" or v_true == v_false:
                merged_val = v_true
            else:
                sel_id = self.emit_node(
                    header_id,
                    "Select",
                    pn.get("type", "U32"),
                    [cond_id, v_true, v_false],
                )
                select_ids.append(sel_id)
                merged_val = sel_id

            pn["phi"] = [
                [int(p), int(v)]
                for p, v in pn["phi"]
                if int(p) not in (true_pred, false_pred)
            ] + [[header_id, merged_val]]

        return select_ids

    def splice_blocks_into(
        self,
        header_id: int,
        arm_ids: Sequence[int],
        extra_node_ids: Sequence[int] = (),
        new_target_id: int | None = None,
    ) -> list[int]:
        """
        Pop each block in `arm_ids`, move its non-marker nodes + `extra_node_ids`
        into `header_id` before the terminator, and optionally rewire `header_id`
        unconditionally to `new_target_id`.
        """
        header = self.blocks_by_id[header_id]
        hoisted: list[int] = []
        for arm_id in arm_ids:
            arm_blk = self.blocks_by_id.pop(arm_id)
            moved = extract_non_marker_nodes(arm_blk, self.nodes_by_id)
            moved_set = set(moved)
            for nid in moved:
                self.nodes_by_id[nid]["blk"] = header_id
            for nid in arm_blk.get("nodes", []):
                if nid not in moved_set:
                    self.nodes_by_id.pop(nid, None)
            hoisted.extend(moved)

        combined = hoisted + list(extra_node_ids)
        if combined:
            insert_before_terminator(header, self.nodes_by_id, combined)
        if new_target_id is not None:
            self.set_unconditional_branch(header_id, new_target_id)
        return combined

    def merge_block_into_predecessor(self, pred_id: int, succ_id: int) -> None:
        """
        Contract a single-edge chain `pred_id -> succ_id`: resolve single-incoming
        Phis in `succ_id`, append `succ_id`'s nodes to `pred_id`, copy `succ_id`'s
        terminator, and retarget downstream predecessor/Phi edges to `pred_id`.
        """
        pred = self.blocks_by_id[pred_id]
        succ = self.blocks_by_id.pop(succ_id)

        phi_repl: dict[int, int] = {}
        for pid in succ.get("phis", []):
            pn = self.nodes_by_id.get(pid)
            if pn and pn.get("phi"):
                phi_repl[pid] = int(pn["phi"][0][1])
                del self.nodes_by_id[pid]
        if phi_repl:
            replace_all_uses(self.blocks_by_id, self.nodes_by_id, phi_repl)

        kept_pred_nodes = extract_non_marker_nodes(pred, self.nodes_by_id)
        kept_set = set(kept_pred_nodes)
        for nid in pred.get("nodes", []):
            if nid not in kept_set:
                self.nodes_by_id.pop(nid, None)

        succ_nodes = list(succ.get("nodes", []))
        for nid in succ_nodes:
            if nid in self.nodes_by_id:
                self.nodes_by_id[nid]["blk"] = pred_id
        pred["nodes"] = kept_pred_nodes + succ_nodes

        self.copy_terminator(pred_id, succ)
        self.retarget_incoming_edges(succ_id, pred_id, block_succs(succ))

    def remove_unreachable_blocks(self, entry: int = 0) -> int:
        """Remove all blocks unreachable from `entry` along with their Phis and nodes."""
        reach_set = set(compute_rpo(self.blocks_by_id, entry))
        removed = 0
        for bid in [b for b in list(self.blocks_by_id.keys()) if b not in reach_set]:
            blk = self.blocks_by_id.pop(bid)
            for nid in blk.get("phis", []) + blk.get("nodes", []):
                self.nodes_by_id.pop(nid, None)
            removed += 1
        return removed

    def _referenced_node_ids(self) -> set[int]:
        used: set[int] = set()
        for nid, n in self.nodes_by_id.items():
            for a in n.get("args", []):
                used.add(int(a))
            if "phi" in n:
                for _, v in n["phi"]:
                    iv = int(v)
                    if iv != nid:
                        used.add(iv)
        for blk in self.blocks_by_id.values():
            bc = int(blk.get("branch_cond", 0))
            if bc > 0:
                used.add(bc)
        return used

    def cleanup_phis(self) -> int:
        """
        Generic fixed-point dead & trivial Phi cleanup pass:
        - Prunes incoming Phi edges whose predecessor is no longer in `preds`.
        - Collapses Phis with a single unique non-self incoming value.
        - Removes dead Phis with zero incoming values or zero external uses.
        """
        total_removed = 0
        while True:
            rebuild_preds(self.blocks_by_id, self.nodes_by_id)
            used = self._referenced_node_ids()
            repl: dict[int, int] = {}
            removed_round = 0

            for blk in self.blocks_by_id.values():
                keep_phis: list[int] = []
                for pid in blk.get("phis", []):
                    pn = self.nodes_by_id.get(pid)
                    if pn is None:
                        continue
                    incs = pn.get("phi", [])
                    vals = {int(v) for _, v in incs if int(v) != pid}
                    if len(vals) == 1:
                        repl[pid] = next(iter(vals))
                        del self.nodes_by_id[pid]
                        removed_round += 1
                    elif len(vals) == 0 or pid not in used:
                        del self.nodes_by_id[pid]
                        removed_round += 1
                    else:
                        keep_phis.append(pid)
                blk["phis"] = keep_phis
                blk["nodes"] = [nid for nid in blk.get("nodes", []) if nid in self.nodes_by_id]

            if repl:
                replace_all_uses(self.blocks_by_id, self.nodes_by_id, repl)
            total_removed += removed_round
            if removed_round == 0:
                break

        return total_removed

    def sync_cfg(self) -> int:
        """Rebuild predecessor lists and run generic dead/trivial Phi cleanup."""
        return self.cleanup_phis()

    def is_pure_single_pred_arm(self, arm_id: int, header_id: int, max_arm_nodes: int) -> bool:
        """Return True iff `arm_id` is a pure, Phi-free block with sole predecessor `header_id`."""
        blk = self.blocks_by_id.get(arm_id)
        if blk is None:
            return False
        return (
            blk.get("preds") == [header_id]
            and not blk.get("phis")
            and len(blk.get("nodes", [])) <= max_arm_nodes
            and is_block_pure(blk, self.nodes_by_id)
        )

    def is_pure_kill_block(self, blk_id: int, header_id: int) -> bool:
        """Return True iff `blk_id` has sole predecessor `header_id`, no Phis/succs, and one Kill."""
        blk = self.blocks_by_id.get(blk_id)
        if blk is None or blk.get("preds") != [header_id] or blk.get("phis") or block_succs(blk):
            return False
        kill_count = 0
        for nid in blk.get("nodes", []):
            node = self.nodes_by_id.get(nid)
            if not node:
                continue
            if node["op"] == "Kill":
                kill_count += 1
            elif node["op"] in SIDE_EFFECT_OPS:
                return False
        return kill_count == 1

    def is_empty_forwarding_block(self, blk_id: int) -> bool:
        """Return True iff `blk_id` is an unconditional, Phi-free block with no non-marker instructions."""
        blk = self.blocks_by_id.get(blk_id)
        if blk is None or blk_id == 0:
            return False
        if blk.get("is_cond") or blk.get("is_end") or blk.get("is_discard") or blk.get("is_loop_header"):
            return False
        if blk.get("phis"):
            return False
        succs = block_succs(blk)
        if len(succs) != 1 or succs[0] == blk_id:
            return False
        return len(extract_non_marker_nodes(blk, self.nodes_by_id)) == 0


def pass_cleanup_phis(
    blocks_by_id: dict[int, dict[str, Any]],
    nodes_by_id: dict[int, dict[str, Any]],
    stats: Stratum1Stats | None = None,
) -> bool:
    """Dedicated generic dead & trivial Phi cleanup pass."""
    rewriter = CfgRewriter(blocks_by_id, nodes_by_id, stats)
    removed = rewriter.cleanup_phis()
    return removed > 0

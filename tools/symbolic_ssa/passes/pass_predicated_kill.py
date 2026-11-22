#!/usr/bin/env python3
"""
Pass 1.2: Predicated-Discard Folding.

Folds `if (c) { pure; Kill; } else { C }` into `pure; Kill(c); br C`,
aligning explicit SPIR-V branch-to-OpKill CFGs with Valhall's inline
predicated DISCARD.f32 representation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from tools.symbolic_ssa.passes.ir_utils import (
    SIDE_EFFECT_OPS,
    Stratum1Stats,
    block_succs,
    collapse_trivial_phis,
    extract_non_marker_nodes,
    insert_before_terminator,
    next_node_id,
    rebuild_preds,
)


def pass_predicated_kill(
    blocks_by_id: Dict[int, Dict[str, Any]],
    nodes_by_id: Dict[int, Dict[str, Any]],
    stats: Stratum1Stats,
) -> bool:
    """Fold single-predecessor pure+Kill branch arms into inline predicated Kill nodes."""
    changed = False
    for hid in list(blocks_by_id.keys()):
        if hid not in blocks_by_id:
            continue
        h = blocks_by_id[hid]
        if not h.get("is_cond"):
            continue
        bt = int(h.get("branch_target", -1))
        ft = int(h.get("fallthrough", -1))
        if bt < 0 or ft < 0 or bt == ft or bt == hid or ft == hid:
            continue

        matched_side: Optional[Tuple[int, int, bool]] = None
        for k_bid, c_bid, on_true in ((bt, ft, True), (ft, bt, False)):
            kb = blocks_by_id.get(k_bid)
            if not kb or kb.get("preds") != [hid] or kb.get("phis"):
                continue
            if block_succs(kb):
                continue
            kills = []
            other_se = False
            for nid in kb.get("nodes", []):
                n = nodes_by_id.get(nid)
                if not n:
                    continue
                if n["op"] == "Kill":
                    kills.append(n)
                elif n["op"] in SIDE_EFFECT_OPS:
                    other_se = True
                    break
            if other_se or len(kills) != 1:
                continue
            matched_side = (k_bid, c_bid, on_true)
            break

        if matched_side is None:
            continue

        k_bid, c_bid, on_true = matched_side
        kb = blocks_by_id[k_bid]
        bcond = int(h.get("branch_cond", 0))
        if bcond == 0:
            continue

        new_nodes_to_insert: List[int] = []
        pred_id = bcond
        if not on_true:
            not_id = next_node_id(nodes_by_id)
            nodes_by_id[not_id] = {
                "id": not_id,
                "op": "LogicalNot",
                "type": "Bool",
                "blk": hid,
                "args": [bcond],
            }
            new_nodes_to_insert.append(not_id)
            pred_id = not_id

        for nid in extract_non_marker_nodes(kb, nodes_by_id):
            n = nodes_by_id[nid]
            n["blk"] = hid
            if n["op"] == "Kill":
                orig_args = n.get("args", [])
                tok = orig_args[-1] if orig_args else 0
                if n.get("aux_bool") and len(orig_args) >= 2:
                    and_id = next_node_id(nodes_by_id)
                    nodes_by_id[and_id] = {
                        "id": and_id,
                        "op": "LogicalAnd",
                        "type": "Bool",
                        "blk": hid,
                        "args": [pred_id, orig_args[0]],
                    }
                    new_nodes_to_insert.append(and_id)
                    n["args"] = [and_id, tok]
                else:
                    n["aux_bool"] = True
                    n["args"] = [pred_id, tok]
            new_nodes_to_insert.append(nid)

        insert_before_terminator(h, nodes_by_id, new_nodes_to_insert)
        for nid in kb.get("nodes", []):
            if nid not in new_nodes_to_insert:
                nodes_by_id.pop(nid, None)
        del blocks_by_id[k_bid]

        h["is_cond"] = False
        h["is_uncond"] = True
        h["branch_target"] = c_bid
        h["fallthrough"] = -1
        h["branch_cond"] = 0
        stats.discards_predicated += 1
        changed = True

    if changed:
        rebuild_preds(blocks_by_id, nodes_by_id)
        collapse_trivial_phis(blocks_by_id, nodes_by_id)
    return changed

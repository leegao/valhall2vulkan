#!/usr/bin/env python3
"""
Pass 1.2: Predicated-Discard Folding.

Folds `if (c) { pure; Kill; } else { C }` into `pure; Kill(c); br C`,
aligning explicit SPIR-V branch-to-OpKill CFGs with Valhall's inline
predicated `DISCARD.f32` representation.
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


def _is_pure_kill_block(blk: Dict[str, Any], nodes_by_id: Dict[int, Dict[str, Any]], header_id: int) -> bool:
    """Return True iff `blk` has sole predecessor `header_id`, no Phis/succs, and exactly one Kill side effect."""
    if blk.get("preds") != [header_id] or blk.get("phis") or block_succs(blk):
        return False
    kill_count = 0
    for nid in blk.get("nodes", []):
        node = nodes_by_id.get(nid)
        if not node:
            continue
        if node["op"] == "Kill":
            kill_count += 1
        elif node["op"] in SIDE_EFFECT_OPS:
            return False
    return kill_count == 1


def _find_kill_arm(
    blocks_by_id: Dict[int, Dict[str, Any]],
    nodes_by_id: Dict[int, Dict[str, Any]],
    header_id: int,
) -> Optional[Tuple[int, int, bool]]:
    """Return `(kill_bid, cont_bid, kill_on_true)` if `header_id` branches to a pure Kill arm."""
    header = blocks_by_id.get(header_id)
    if header is None or not header.get("is_cond"):
        return None
    bt, ft, bcond = int(header.get("branch_target", -1)), int(header.get("fallthrough", -1)), int(header.get("branch_cond", 0))
    if bt < 0 or ft < 0 or bt == ft or bt == header_id or ft == header_id or bcond == 0:
        return None

    for kill_bid, cont_bid, on_true in ((bt, ft, True), (ft, bt, False)):
        kb = blocks_by_id.get(kill_bid)
        if kb is not None and _is_pure_kill_block(kb, nodes_by_id, header_id):
            return (kill_bid, cont_bid, on_true)
    return None


def _fold_kill_arm_into_header(
    blocks_by_id: Dict[int, Dict[str, Any]],
    nodes_by_id: Dict[int, Dict[str, Any]],
    header_id: int,
    kill_bid: int,
    cont_bid: int,
    kill_on_true: bool,
) -> None:
    """Hoist pure ops from `kill_bid` into `header_id` with a predicated `Kill` and branch to `cont_bid`."""
    header = blocks_by_id[header_id]
    kill_blk = blocks_by_id.pop(kill_bid)
    bcond = int(header["branch_cond"])

    hoisted: List[int] = []
    pred_id = bcond
    if not kill_on_true:
        not_id = next_node_id(nodes_by_id)
        nodes_by_id[not_id] = {"id": not_id, "op": "LogicalNot", "type": "Bool", "blk": header_id, "args": [bcond]}
        hoisted.append(not_id)
        pred_id = not_id

    for nid in extract_non_marker_nodes(kill_blk, nodes_by_id):
        node = nodes_by_id[nid]
        node["blk"] = header_id
        if node["op"] == "Kill":
            orig_args = node.get("args", [])
            tok = orig_args[-1] if orig_args else 0
            if node.get("aux_bool") and len(orig_args) >= 2:
                and_id = next_node_id(nodes_by_id)
                nodes_by_id[and_id] = {
                    "id": and_id,
                    "op": "LogicalAnd",
                    "type": "Bool",
                    "blk": header_id,
                    "args": [pred_id, orig_args[0]],
                }
                hoisted.append(and_id)
                node["args"] = [and_id, tok]
            else:
                node["aux_bool"] = True
                node["args"] = [pred_id, tok]
        hoisted.append(nid)

    insert_before_terminator(header, nodes_by_id, hoisted)
    for nid in kill_blk.get("nodes", []):
        if nid not in hoisted:
            nodes_by_id.pop(nid, None)

    header["is_cond"] = False
    header["is_uncond"] = True
    header["branch_target"] = cont_bid
    header["fallthrough"] = -1
    header["branch_cond"] = 0


def pass_predicated_kill(
    blocks_by_id: Dict[int, Dict[str, Any]],
    nodes_by_id: Dict[int, Dict[str, Any]],
    stats: Stratum1Stats,
) -> bool:
    """Fold single-predecessor pure+Kill branch arms into inline predicated Kill nodes."""
    changed = False
    for header_id in list(blocks_by_id.keys()):
        match = _find_kill_arm(blocks_by_id, nodes_by_id, header_id)
        if match is None:
            continue
        kill_bid, cont_bid, kill_on_true = match
        _fold_kill_arm_into_header(blocks_by_id, nodes_by_id, header_id, kill_bid, cont_bid, kill_on_true)
        stats.discards_predicated += 1
        changed = True

    if changed:
        rebuild_preds(blocks_by_id, nodes_by_id)
        collapse_trivial_phis(blocks_by_id, nodes_by_id)
    return changed

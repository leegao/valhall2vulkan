#!/usr/bin/env python3
"""
Pass 1.4: Trivial CFG Chain Contraction.

Merges sequential single-edge blocks `A -> B` whenever `outdeg(A) == 1`,
`indeg(B) == 1`, `B != 0` (entry), and `B` is not a loop header, eliminating
artificial sub-block chains introduced by `split_cfg_subblocks`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from tools.symbolic_ssa.passes.ir_utils import (
    Stratum1Stats,
    block_succs,
    collapse_trivial_phis,
    extract_non_marker_nodes,
    rebuild_preds,
    replace_all_uses,
)

_TERMINATOR_FIELDS = (
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


def _find_contractible_successor(
    blocks_by_id: Dict[int, Dict[str, Any]],
    pred_id: int,
) -> Optional[int]:
    """Return the unique successor `succ_id` if `pred_id -> succ_id` can be contracted."""
    pred = blocks_by_id.get(pred_id)
    if pred is None or pred.get("is_cond") or pred.get("is_end") or pred.get("is_discard"):
        return None
    succs = block_succs(pred)
    if len(succs) != 1:
        return None
    succ_id = succs[0]
    if succ_id in (pred_id, 0) or succ_id not in blocks_by_id:
        return None
    succ = blocks_by_id[succ_id]
    if succ.get("preds") != [pred_id] or succ.get("is_loop_header"):
        return None
    return succ_id


def _merge_block_into_predecessor(
    blocks_by_id: Dict[int, Dict[str, Any]],
    nodes_by_id: Dict[int, Dict[str, Any]],
    pred_id: int,
    succ_id: int,
) -> None:
    """Splice `succ_id` into `pred_id`, resolve single-incoming Phis, and retarget successors."""
    pred = blocks_by_id[pred_id]
    succ = blocks_by_id.pop(succ_id)

    # Single-predecessor Phis in `succ` collapse directly to their sole incoming value.
    phi_repl: Dict[int, int] = {}
    for pid in succ.get("phis", []):
        pn = nodes_by_id.get(pid)
        if pn and pn.get("phi"):
            phi_repl[pid] = int(pn["phi"][0][1])
            del nodes_by_id[pid]
    if phi_repl:
        replace_all_uses(blocks_by_id, nodes_by_id, phi_repl)

    # Strip trailing Branch marker from `pred` and append `succ`'s instructions.
    kept_pred_nodes = extract_non_marker_nodes(pred, nodes_by_id)
    for nid in pred.get("nodes", []):
        if nid not in kept_pred_nodes:
            nodes_by_id.pop(nid, None)

    succ_nodes = list(succ.get("nodes", []))
    for nid in succ_nodes:
        if nid in nodes_by_id:
            nodes_by_id[nid]["blk"] = pred_id
    pred["nodes"] = kept_pred_nodes + succ_nodes

    for field in _TERMINATOR_FIELDS:
        pred[field] = succ.get(field)

    # Retarget `succ_id -> pred_id` in all downstream predecessor lists and Phi edges.
    for downstream_id in block_succs(succ):
        downstream = blocks_by_id.get(downstream_id)
        if downstream is None:
            continue
        downstream["preds"] = [pred_id if p == succ_id else p for p in downstream.get("preds", [])]
        for pid in downstream.get("phis", []):
            pn = nodes_by_id.get(pid)
            if pn and "phi" in pn:
                pn["phi"] = [[pred_id if int(p) == succ_id else int(p), int(v)] for p, v in pn["phi"]]


def pass_contract_chains(
    blocks_by_id: Dict[int, Dict[str, Any]],
    nodes_by_id: Dict[int, Dict[str, Any]],
    stats: Stratum1Stats,
) -> bool:
    """Contract trivial unconditional `A -> B` block chains to fixed point."""
    changed = False
    while True:
        merged_any = False
        for pred_id in list(blocks_by_id.keys()):
            succ_id = _find_contractible_successor(blocks_by_id, pred_id)
            if succ_id is None:
                continue
            _merge_block_into_predecessor(blocks_by_id, nodes_by_id, pred_id, succ_id)
            stats.chains_contracted += 1
            merged_any = True
            changed = True
        if not merged_any:
            break

    if changed:
        rebuild_preds(blocks_by_id, nodes_by_id)
        collapse_trivial_phis(blocks_by_id, nodes_by_id)
    return changed

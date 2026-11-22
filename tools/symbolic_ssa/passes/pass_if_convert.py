#!/usr/bin/env python3
"""
Pass 1.3: Pure Acyclic Selection If-Conversion.

Converts pure forward triangles (H -> {A, M}, A -> M) and diamonds
(H -> {T, F}, T -> M, F -> M) into straight-line Select nodes inside H,
canonicalizing SPIR-V control-flow selections and Valhall CSEL instructions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from tools.symbolic_ssa.passes.ir_utils import (
    Stratum1Stats,
    block_succs,
    collapse_trivial_phis,
    compute_rpo,
    extract_non_marker_nodes,
    insert_before_terminator,
    is_block_pure,
    next_node_id,
    rebuild_preds,
)


def pass_if_convert(
    blocks_by_id: Dict[int, Dict[str, Any]],
    nodes_by_id: Dict[int, Dict[str, Any]],
    stats: Stratum1Stats,
    max_arm_nodes: int = 128,
) -> bool:
    """If-convert pure acyclic triangles and diamonds into Select nodes."""
    any_changed = False
    while True:
        changed = False
        rpo = compute_rpo(blocks_by_id, 0)
        rpo_idx = {b: i for i, b in enumerate(rpo)}

        for hid in list(rpo):
            if hid not in blocks_by_id:
                continue
            h = blocks_by_id[hid]
            if not h.get("is_cond"):
                continue
            bt = int(h.get("branch_target", -1))
            ft = int(h.get("fallthrough", -1))
            bcond = int(h.get("branch_cond", 0))
            if bt < 0 or ft < 0 or bt == ft or bcond == 0:
                continue
            if bt not in blocks_by_id or ft not in blocks_by_id:
                continue
            if rpo_idx.get(bt, -1) <= rpo_idx.get(hid, -1) or rpo_idx.get(ft, -1) <= rpo_idx.get(hid, -1):
                continue

            bt_blk = blocks_by_id[bt]
            ft_blk = blocks_by_id[ft]
            bt_succs = block_succs(bt_blk)
            ft_succs = block_succs(ft_blk)

            tri: Optional[Tuple[int, int, bool]] = None
            if (
                bt_blk.get("preds") == [hid]
                and not bt_blk.get("phis")
                and bt_succs == [ft]
                and len(bt_blk.get("nodes", [])) <= max_arm_nodes
                and is_block_pure(bt_blk, nodes_by_id)
                and rpo_idx.get(ft, -1) > rpo_idx.get(bt, -1)
            ):
                tri = (bt, ft, True)
            elif (
                ft_blk.get("preds") == [hid]
                and not ft_blk.get("phis")
                and ft_succs == [bt]
                and len(ft_blk.get("nodes", [])) <= max_arm_nodes
                and is_block_pure(ft_blk, nodes_by_id)
                and rpo_idx.get(bt, -1) > rpo_idx.get(ft, -1)
            ):
                tri = (ft, bt, False)

            if tri is not None:
                arm_id, merge_id, arm_is_true = tri
                arm_blk = blocks_by_id[arm_id]
                merge_blk = blocks_by_id[merge_id]

                moved = extract_non_marker_nodes(arm_blk, nodes_by_id)
                for nid in moved:
                    nodes_by_id[nid]["blk"] = hid

                select_nodes: List[int] = []
                for pid in merge_blk.get("phis", []):
                    pn = nodes_by_id.get(pid)
                    if not pn:
                        continue
                    inc_map = {int(p): int(v) for p, v in pn.get("phi", [])}
                    if hid in inc_map and arm_id in inc_map:
                        v_h = inc_map[hid]
                        v_a = inc_map[arm_id]
                        if pn.get("type") == "Token" or v_h == v_a:
                            merged_v = v_h
                        else:
                            sel_id = next_node_id(nodes_by_id)
                            t_v, f_v = (v_a, v_h) if arm_is_true else (v_h, v_a)
                            nodes_by_id[sel_id] = {
                                "id": sel_id,
                                "op": "Select",
                                "type": pn.get("type", "U32"),
                                "blk": hid,
                                "args": [bcond, t_v, f_v],
                            }
                            select_nodes.append(sel_id)
                            merged_v = sel_id
                        new_phi = [[p, v] for p, v in pn["phi"] if int(p) not in (hid, arm_id)]
                        new_phi.append([hid, merged_v])
                        pn["phi"] = new_phi

                insert_before_terminator(h, nodes_by_id, moved + select_nodes)
                for nid in arm_blk.get("nodes", []):
                    if nid not in moved:
                        nodes_by_id.pop(nid, None)
                del blocks_by_id[arm_id]

                h["is_cond"] = False
                h["is_uncond"] = True
                h["branch_target"] = merge_id
                h["fallthrough"] = -1
                h["branch_cond"] = 0
                stats.triangles_if_converted += 1
                changed = True
                break

            if (
                bt_blk.get("preds") == [hid]
                and ft_blk.get("preds") == [hid]
                and not bt_blk.get("phis")
                and not ft_blk.get("phis")
                and len(bt_succs) == 1
                and bt_succs == ft_succs
                and bt_succs[0] in blocks_by_id
                and len(bt_blk.get("nodes", [])) <= max_arm_nodes
                and len(ft_blk.get("nodes", [])) <= max_arm_nodes
                and is_block_pure(bt_blk, nodes_by_id)
                and is_block_pure(ft_blk, nodes_by_id)
            ):
                merge_id = bt_succs[0]
                if rpo_idx.get(merge_id, -1) <= max(rpo_idx.get(bt, -1), rpo_idx.get(ft, -1)):
                    continue
                merge_blk = blocks_by_id[merge_id]
                moved_t = extract_non_marker_nodes(bt_blk, nodes_by_id)
                moved_f = extract_non_marker_nodes(ft_blk, nodes_by_id)
                for nid in moved_t + moved_f:
                    nodes_by_id[nid]["blk"] = hid

                select_nodes = []
                for pid in merge_blk.get("phis", []):
                    pn = nodes_by_id.get(pid)
                    if not pn:
                        continue
                    inc_map = {int(p): int(v) for p, v in pn.get("phi", [])}
                    if bt in inc_map and ft in inc_map:
                        v_t = inc_map[bt]
                        v_f = inc_map[ft]
                        if pn.get("type") == "Token" or v_t == v_f:
                            merged_v = v_t
                        else:
                            sel_id = next_node_id(nodes_by_id)
                            nodes_by_id[sel_id] = {
                                "id": sel_id,
                                "op": "Select",
                                "type": pn.get("type", "U32"),
                                "blk": hid,
                                "args": [bcond, v_t, v_f],
                            }
                            select_nodes.append(sel_id)
                            merged_v = sel_id
                        new_phi = [[p, v] for p, v in pn["phi"] if int(p) not in (bt, ft)]
                        new_phi.append([hid, merged_v])
                        pn["phi"] = new_phi

                insert_before_terminator(h, nodes_by_id, moved_t + moved_f + select_nodes)
                for nid in bt_blk.get("nodes", []):
                    if nid not in moved_t:
                        nodes_by_id.pop(nid, None)
                for nid in ft_blk.get("nodes", []):
                    if nid not in moved_f:
                        nodes_by_id.pop(nid, None)
                del blocks_by_id[bt]
                del blocks_by_id[ft]

                h["is_cond"] = False
                h["is_uncond"] = True
                h["branch_target"] = merge_id
                h["fallthrough"] = -1
                h["branch_cond"] = 0
                stats.diamonds_if_converted += 1
                changed = True
                break

        if changed:
            any_changed = True
            rebuild_preds(blocks_by_id, nodes_by_id)
            collapse_trivial_phis(blocks_by_id, nodes_by_id)
        else:
            break
    return any_changed

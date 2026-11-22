#!/usr/bin/env python3
"""
Pass 1.3: Pure Acyclic Selection If-Conversion via CFG Tile Rewriter DSL.

Expresses pure forward triangle (`H -> {A, M}, A -> M`) and diamond
(`H -> {T, F}, T -> M, F -> M`) if-conversions as declarative `CfgTileRule`s
executed bottom-up by `apply_cfg_tile_rules`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from tools.symbolic_ssa.passes.cfg_tile_rewriter import (
    CfgRewriteContext,
    CfgTileRule,
    TileMatch,
    apply_cfg_tile_rules,
)
from tools.symbolic_ssa.passes.ir_utils import (
    Stratum1Stats,
    block_succs,
    extract_non_marker_nodes,
    insert_before_terminator,
    is_block_pure,
    next_node_id,
)


def _is_pure_single_pred_arm(
    ctx: CfgRewriteContext,
    arm_id: int,
    header_id: int,
    max_arm_nodes: int,
) -> bool:
    blk = ctx.blocks_by_id.get(arm_id)
    if blk is None:
        return False
    return (
        blk.get("preds") == [header_id]
        and not blk.get("phis")
        and len(blk.get("nodes", [])) <= max_arm_nodes
        and is_block_pure(blk, ctx.nodes_by_id)
    )


def _rewrite_merge_phis_with_select(
    ctx: CfgRewriteContext,
    header_id: int,
    merge_id: int,
    bcond: int,
    true_pred: int,
    false_pred: int,
) -> List[int]:
    """Rewrite Phis in `merge_id` coming from `(true_pred, false_pred)` into Select nodes in `header_id`."""
    merge_blk = ctx.blocks_by_id[merge_id]
    select_ids: List[int] = []

    for pid in merge_blk.get("phis", []):
        pn = ctx.nodes_by_id.get(pid)
        if not pn:
            continue
        inc_map = {int(p): int(v) for p, v in pn.get("phi", [])}
        if true_pred not in inc_map or false_pred not in inc_map:
            continue

        v_true, v_false = inc_map[true_pred], inc_map[false_pred]
        if pn.get("type") == "Token" or v_true == v_false:
            merged_val = v_true
        else:
            sel_id = next_node_id(ctx.nodes_by_id)
            ctx.nodes_by_id[sel_id] = {
                "id": sel_id,
                "op": "Select",
                "type": pn.get("type", "U32"),
                "blk": header_id,
                "args": [bcond, v_true, v_false],
            }
            select_ids.append(sel_id)
            merged_val = sel_id

        pn["phi"] = [
            [p, v] for p, v in pn["phi"] if int(p) not in (true_pred, false_pred)
        ] + [[header_id, merged_val]]

    return select_ids


def _splice_arms_into_header(
    ctx: CfgRewriteContext,
    header_id: int,
    merge_id: int,
    arm_ids: List[int],
    select_ids: List[int],
) -> None:
    """Move non-marker nodes from `arm_ids` + `select_ids` into `header_id` and jump unconditionally to `merge_id`."""
    header = ctx.blocks_by_id[header_id]
    hoisted: List[int] = []
    for arm_id in arm_ids:
        arm_blk = ctx.blocks_by_id.pop(arm_id)
        moved = extract_non_marker_nodes(arm_blk, ctx.nodes_by_id)
        for nid in moved:
            ctx.nodes_by_id[nid]["blk"] = header_id
        for nid in arm_blk.get("nodes", []):
            if nid not in moved:
                ctx.nodes_by_id.pop(nid, None)
        hoisted.extend(moved)

    insert_before_terminator(header, ctx.nodes_by_id, hoisted + select_ids)
    header["is_cond"] = False
    header["is_uncond"] = True
    header["branch_target"] = merge_id
    header["fallthrough"] = -1
    header["branch_cond"] = 0


def make_pure_triangle_rule(max_arm_nodes: int = 128) -> CfgTileRule:
    """Declarative tile rule for pure forward triangles (`H -> {A, M}, A -> M`)."""

    def match(ctx: CfgRewriteContext, hid: int) -> Optional[TileMatch]:
        h = ctx.blocks_by_id[hid]
        if not h.get("is_cond"):
            return None
        bt, ft, bcond = int(h.get("branch_target", -1)), int(h.get("fallthrough", -1)), int(h.get("branch_cond", 0))
        if bt < 0 or ft < 0 or bt == ft or bcond == 0:
            return None
        if ctx.rpo_idx.get(bt, -1) <= ctx.rpo_idx.get(hid, -1) or ctx.rpo_idx.get(ft, -1) <= ctx.rpo_idx.get(hid, -1):
            return None

        for arm_id, merge_id, arm_is_true in ((bt, ft, True), (ft, bt, False)):
            if (
                _is_pure_single_pred_arm(ctx, arm_id, hid, max_arm_nodes)
                and block_succs(ctx.blocks_by_id[arm_id]) == [merge_id]
                and ctx.rpo_idx.get(merge_id, -1) > ctx.rpo_idx.get(arm_id, -1)
            ):
                return TileMatch("pure_triangle", hid, (arm_id, merge_id, bcond, arm_is_true))
        return None

    def rewrite(ctx: CfgRewriteContext, m: TileMatch) -> None:
        arm_id, merge_id, bcond, arm_is_true = m.payload
        t_pred, f_pred = (arm_id, m.header_id) if arm_is_true else (m.header_id, arm_id)
        selects = _rewrite_merge_phis_with_select(ctx, m.header_id, merge_id, bcond, t_pred, f_pred)
        _splice_arms_into_header(ctx, m.header_id, merge_id, [arm_id], selects)
        ctx.stats.triangles_if_converted += 1

    return CfgTileRule(name="pure_triangle", match_fn=match, rewrite_fn=rewrite, priority=2)


def make_pure_diamond_rule(max_arm_nodes: int = 128) -> CfgTileRule:
    """Declarative tile rule for pure forward diamonds (`H -> {T, F}, T -> M, F -> M`)."""

    def match(ctx: CfgRewriteContext, hid: int) -> Optional[TileMatch]:
        h = ctx.blocks_by_id[hid]
        if not h.get("is_cond"):
            return None
        bt, ft, bcond = int(h.get("branch_target", -1)), int(h.get("fallthrough", -1)), int(h.get("branch_cond", 0))
        if bt < 0 or ft < 0 or bt == ft or bcond == 0:
            return None
        if ctx.rpo_idx.get(bt, -1) <= ctx.rpo_idx.get(hid, -1) or ctx.rpo_idx.get(ft, -1) <= ctx.rpo_idx.get(hid, -1):
            return None
        if not (_is_pure_single_pred_arm(ctx, bt, hid, max_arm_nodes) and _is_pure_single_pred_arm(ctx, ft, hid, max_arm_nodes)):
            return None

        bt_succs = block_succs(ctx.blocks_by_id[bt])
        ft_succs = block_succs(ctx.blocks_by_id[ft])
        if len(bt_succs) != 1 or bt_succs != ft_succs:
            return None
        merge_id = bt_succs[0]
        if merge_id not in ctx.blocks_by_id:
            return None
        if ctx.rpo_idx.get(merge_id, -1) <= max(ctx.rpo_idx.get(bt, -1), ctx.rpo_idx.get(ft, -1)):
            return None
        return TileMatch("pure_diamond", hid, (bt, ft, merge_id, bcond))

    def rewrite(ctx: CfgRewriteContext, m: TileMatch) -> None:
        bt, ft, merge_id, bcond = m.payload
        selects = _rewrite_merge_phis_with_select(ctx, m.header_id, merge_id, bcond, bt, ft)
        _splice_arms_into_header(ctx, m.header_id, merge_id, [bt, ft], selects)
        ctx.stats.diamonds_if_converted += 1

    return CfgTileRule(name="pure_diamond", match_fn=match, rewrite_fn=rewrite, priority=1)


def pass_if_convert(
    blocks_by_id: Dict[int, Dict[str, Any]],
    nodes_by_id: Dict[int, Dict[str, Any]],
    stats: Stratum1Stats,
    max_arm_nodes: int = 128,
) -> bool:
    """If-convert pure acyclic triangles and diamonds bottom-up via CFG tile rules."""
    rules = [
        make_pure_triangle_rule(max_arm_nodes=max_arm_nodes),
        make_pure_diamond_rule(max_arm_nodes=max_arm_nodes),
    ]
    return apply_cfg_tile_rules(blocks_by_id, nodes_by_id, stats, rules)

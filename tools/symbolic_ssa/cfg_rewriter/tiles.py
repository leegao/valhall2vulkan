"""
Bottom-Up DP CFG Tile-Rewrite Engine & Standard Tile Library.

Provides `CfgRewriteContext`, `TileMatch`, `CfgTileRule`, `apply_cfg_tile_rules`,
and standard declarative CFG tiles built on `CfgRewriter`:
- Degenerate conditional branch (`H -> {M, M}`)
- Empty jump-threading forwarding block (`H -> E -> T`)
- Pure forward triangle (`H -> {A, M}, A -> M`)
- Pure forward diamond (`H -> {T, F}, T -> M, F -> M`)
- Predicated kill arm (`H -> {K, C}` where `K` is pure + single `Kill`)
- Single-edge unconditional chain contraction (`A -> B`)
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List, Optional, Sequence

from tools.symbolic_ssa.cfg_rewriter.rewriter import CfgRewriter
from tools.symbolic_ssa.ir_utils import (
    Stratum1Stats,
    block_succs,
    compute_rpo,
    extract_non_marker_nodes,
    insert_before_terminator,
)


@dataclasses.dataclass
class CfgRewriteContext:
    """Mutable IR state, `CfgRewriter`, and RPO ordering snapshot passed to CFG tile rules."""

    blocks_by_id: Dict[int, Dict[str, Any]]
    nodes_by_id: Dict[int, Dict[str, Any]]
    stats: Stratum1Stats
    rpo_idx: Dict[int, int]
    rewriter: CfgRewriter


@dataclasses.dataclass
class TileMatch:
    """Matched CFG tile rooted at `header_id` with rule-specific payload."""

    rule_name: str
    header_id: int
    payload: Any


@dataclasses.dataclass(frozen=True)
class CfgTileRule:
    """Declarative CFG tile rewrite rule."""

    name: str
    match_fn: Callable[[CfgRewriteContext, int], Optional[TileMatch]]
    rewrite_fn: Callable[[CfgRewriteContext, TileMatch], None]
    priority: int = 0


def apply_cfg_tile_rules(
    blocks_by_id: Dict[int, Dict[str, Any]],
    nodes_by_id: Dict[int, Dict[str, Any]],
    stats: Stratum1Stats,
    rules: Sequence[CfgTileRule],
) -> bool:
    """
    Apply `rules` bottom-up (reverse-RPO / post-order DP) to fixed point.

    Evaluating headers bottom-up ensures nested tiles (e.g. an inner diamond or
    cascaded if-else-if inside an outer arm) collapse in a single sweep, and
    `CfgRewriter.sync_cfg()` keeps predecessors and Phis normalized after each step.
    """
    ordered_rules = sorted(rules, key=lambda r: -r.priority)
    rewriter = CfgRewriter(blocks_by_id, nodes_by_id, stats)
    any_changed = False

    while True:
        rpo = compute_rpo(blocks_by_id, 0)
        ctx = CfgRewriteContext(
            blocks_by_id=blocks_by_id,
            nodes_by_id=nodes_by_id,
            stats=stats,
            rpo_idx={b: i for i, b in enumerate(rpo)},
            rewriter=rewriter,
        )
        round_changed = False

        for hid in reversed(rpo):
            if hid not in blocks_by_id:
                continue
            for rule in ordered_rules:
                m = rule.match_fn(ctx, hid)
                if m is not None:
                    rule.rewrite_fn(ctx, m)
                    rewriter.sync_cfg()
                    round_changed = True
                    any_changed = True
                    break

        if not round_changed:
            break

    return any_changed


def make_degenerate_branch_rule() -> CfgTileRule:
    """Collapse conditional branches whose true and false targets are identical (`H -> {M, M}`)."""

    def match(ctx: CfgRewriteContext, hid: int) -> Optional[TileMatch]:
        h = ctx.blocks_by_id[hid]
        if not h.get("is_cond"):
            return None
        bt = int(h.get("branch_target", -1))
        ft = int(h.get("fallthrough", -1))
        if bt >= 0 and bt == ft:
            return TileMatch("degenerate_branch", hid, bt)
        return None

    def rewrite(ctx: CfgRewriteContext, m: TileMatch) -> None:
        ctx.rewriter.set_unconditional_branch(m.header_id, m.payload)
        ctx.stats.const_branches_folded += 1

    return CfgTileRule(name="degenerate_branch", match_fn=match, rewrite_fn=rewrite, priority=10)


def make_empty_jump_thread_rule() -> CfgTileRule:
    """
    Thread edges through empty Phi-free forwarding blocks (`H -> E -> T`).

    Only threads when `E` has sole predecessor `H` and `T` either has no Phis
    or does not already have `H` as a predecessor, guaranteeing Phi uniqueness.
    """

    def match(ctx: CfgRewriteContext, hid: int) -> Optional[TileMatch]:
        h = ctx.blocks_by_id[hid]
        for succ_id in block_succs(h):
            if succ_id == hid or not ctx.rewriter.is_empty_forwarding_block(succ_id):
                continue
            empty_blk = ctx.blocks_by_id[succ_id]
            if empty_blk.get("preds") != [hid]:
                continue
            target_id = block_succs(empty_blk)[0]
            if target_id not in ctx.blocks_by_id or target_id == succ_id:
                continue
            target_blk = ctx.blocks_by_id[target_id]
            if target_blk.get("phis") and hid in [int(p) for p in target_blk.get("preds", [])]:
                continue
            return TileMatch("empty_jump_thread", hid, (succ_id, target_id))
        return None

    def rewrite(ctx: CfgRewriteContext, m: TileMatch) -> None:
        empty_id, target_id = m.payload
        empty_blk = ctx.blocks_by_id.pop(empty_id)
        for nid in empty_blk.get("nodes", []):
            ctx.nodes_by_id.pop(nid, None)
        ctx.rewriter.retarget_edge(m.header_id, empty_id, target_id)
        ctx.rewriter.retarget_incoming_edges(empty_id, m.header_id, [target_id])
        ctx.stats.chains_contracted += 1

    return CfgTileRule(name="empty_jump_thread", match_fn=match, rewrite_fn=rewrite, priority=5)


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
                ctx.rewriter.is_pure_single_pred_arm(arm_id, hid, max_arm_nodes)
                and block_succs(ctx.blocks_by_id[arm_id]) == [merge_id]
                and ctx.rpo_idx.get(merge_id, -1) > ctx.rpo_idx.get(arm_id, -1)
            ):
                return TileMatch("pure_triangle", hid, (arm_id, merge_id, bcond, arm_is_true))
        return None

    def rewrite(ctx: CfgRewriteContext, m: TileMatch) -> None:
        arm_id, merge_id, bcond, arm_is_true = m.payload
        t_pred, f_pred = (arm_id, m.header_id) if arm_is_true else (m.header_id, arm_id)
        selects = ctx.rewriter.lower_merge_phis_to_select(m.header_id, merge_id, bcond, t_pred, f_pred)
        ctx.rewriter.splice_blocks_into(m.header_id, [arm_id], selects, new_target_id=merge_id)
        ctx.stats.triangles_if_converted += 1

    return CfgTileRule(name="pure_triangle", match_fn=match, rewrite_fn=rewrite, priority=3)


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
        if not (
            ctx.rewriter.is_pure_single_pred_arm(bt, hid, max_arm_nodes)
            and ctx.rewriter.is_pure_single_pred_arm(ft, hid, max_arm_nodes)
        ):
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
        selects = ctx.rewriter.lower_merge_phis_to_select(m.header_id, merge_id, bcond, bt, ft)
        ctx.rewriter.splice_blocks_into(m.header_id, [bt, ft], selects, new_target_id=merge_id)
        ctx.stats.diamonds_if_converted += 1

    return CfgTileRule(name="pure_diamond", match_fn=match, rewrite_fn=rewrite, priority=2)


def make_predicated_kill_rule() -> CfgTileRule:
    """Declarative tile rule folding `if (c) { pure; Kill; } else { C }` into `pure; Kill(c); br C`."""

    def match(ctx: CfgRewriteContext, hid: int) -> Optional[TileMatch]:
        h = ctx.blocks_by_id[hid]
        if not h.get("is_cond"):
            return None
        bt, ft, bcond = int(h.get("branch_target", -1)), int(h.get("fallthrough", -1)), int(h.get("branch_cond", 0))
        if bt < 0 or ft < 0 or bt == ft or bt == hid or ft == hid or bcond == 0:
            return None
        for kill_bid, cont_bid, on_true in ((bt, ft, True), (ft, bt, False)):
            if ctx.rewriter.is_pure_kill_block(kill_bid, hid):
                return TileMatch("predicated_kill", hid, (kill_bid, cont_bid, bcond, on_true))
        return None

    def rewrite(ctx: CfgRewriteContext, m: TileMatch) -> None:
        kill_bid, cont_bid, bcond, kill_on_true = m.payload
        kill_blk = ctx.blocks_by_id.pop(kill_bid)
        header = ctx.blocks_by_id[m.header_id]

        hoisted: List[int] = []
        pred_id = bcond
        if not kill_on_true:
            pred_id = ctx.rewriter.emit_node(m.header_id, "LogicalNot", "Bool", [bcond])
            hoisted.append(pred_id)

        moved = extract_non_marker_nodes(kill_blk, ctx.nodes_by_id)
        moved_set = set(moved)
        for nid in moved:
            node = ctx.nodes_by_id[nid]
            node["blk"] = m.header_id
            if node["op"] == "Kill":
                orig_args = node.get("args", [])
                tok = orig_args[-1] if orig_args else 0
                if node.get("aux_bool") and len(orig_args) >= 2:
                    and_id = ctx.rewriter.emit_node(m.header_id, "LogicalAnd", "Bool", [pred_id, orig_args[0]])
                    hoisted.append(and_id)
                    node["args"] = [and_id, tok]
                else:
                    node["aux_bool"] = True
                    node["args"] = [pred_id, tok]
            hoisted.append(nid)

        for nid in kill_blk.get("nodes", []):
            if nid not in moved_set:
                ctx.nodes_by_id.pop(nid, None)

        insert_before_terminator(header, ctx.nodes_by_id, hoisted)
        ctx.rewriter.set_unconditional_branch(m.header_id, cont_bid)
        ctx.stats.discards_predicated += 1

    return CfgTileRule(name="predicated_kill", match_fn=match, rewrite_fn=rewrite, priority=4)


def make_chain_contract_rule() -> CfgTileRule:
    """Declarative tile rule contracting trivial unconditional `A -> B` chains."""

    def match(ctx: CfgRewriteContext, hid: int) -> Optional[TileMatch]:
        pred = ctx.blocks_by_id.get(hid)
        if pred is None or pred.get("is_cond") or pred.get("is_end") or pred.get("is_discard"):
            return None
        succs = block_succs(pred)
        if len(succs) != 1:
            return None
        succ_id = succs[0]
        if succ_id in (hid, 0) or succ_id not in ctx.blocks_by_id:
            return None
        succ = ctx.blocks_by_id[succ_id]
        if succ.get("preds") != [hid] or succ.get("is_loop_header"):
            return None
        return TileMatch("chain_contract", hid, succ_id)

    def rewrite(ctx: CfgRewriteContext, m: TileMatch) -> None:
        ctx.rewriter.merge_block_into_predecessor(m.header_id, m.payload)
        ctx.stats.chains_contracted += 1

    return CfgTileRule(name="chain_contract", match_fn=match, rewrite_fn=rewrite, priority=1)

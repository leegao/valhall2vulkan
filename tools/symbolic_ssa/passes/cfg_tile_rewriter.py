#!/usr/bin/env python3
"""
Bottom-Up DP CFG Tile-Rewrite Engine & Python DSL.

Allows structural CFG transformations (such as pure triangle/diamond
if-conversion, predicated-kill folding, or future SESE region rewrites) to be
expressed declaratively as `CfgTileRule` objects and applied bottom-up in
post-order so nested tiles collapse in a single sweep.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List, Optional, Sequence

from tools.symbolic_ssa.passes.ir_utils import (
    Stratum1Stats,
    collapse_trivial_phis,
    compute_rpo,
    rebuild_preds,
)


@dataclasses.dataclass
class CfgRewriteContext:
    """Mutable IR state and RPO ordering snapshot passed to CFG tile rules."""

    blocks_by_id: Dict[int, Dict[str, Any]]
    nodes_by_id: Dict[int, Dict[str, Any]]
    stats: Stratum1Stats
    rpo_idx: Dict[int, int]


@dataclasses.dataclass
class TileMatch:
    """Matched CFG tile rooted at `header_id` with rule-specific payload."""

    rule_name: str
    header_id: int
    payload: Any


@dataclasses.dataclass(frozen=True)
class CfgTileRule:
    """Declarative CFG tile rewrite rule (small Python DSL primitive)."""

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

    Evaluating headers bottom-up ensures inner tiles (e.g. an inner diamond
    inside an outer triangle arm) are rewritten first, exposing the outer tile
    without redundant full-CFG restarts.
    """
    ordered_rules = sorted(rules, key=lambda r: -r.priority)
    any_changed = False

    while True:
        rpo = compute_rpo(blocks_by_id, 0)
        ctx = CfgRewriteContext(
            blocks_by_id=blocks_by_id,
            nodes_by_id=nodes_by_id,
            stats=stats,
            rpo_idx={b: i for i, b in enumerate(rpo)},
        )
        round_changed = False

        # Post-order (bottom-up DP): collapse deepest nested tiles first.
        for hid in reversed(rpo):
            if hid not in blocks_by_id:
                continue
            for rule in ordered_rules:
                m = rule.match_fn(ctx, hid)
                if m is not None:
                    rule.rewrite_fn(ctx, m)
                    rebuild_preds(blocks_by_id, nodes_by_id)
                    collapse_trivial_phis(blocks_by_id, nodes_by_id)
                    round_changed = True
                    any_changed = True
                    break

        if not round_changed:
            break

    return any_changed

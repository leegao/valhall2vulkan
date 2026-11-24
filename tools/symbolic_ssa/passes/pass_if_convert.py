"""
Pass 1.3: Pure Acyclic Selection If-Conversion via `cfg_rewriter` Tile Engine.

Applies degenerate-branch collapse, empty jump-threading, pure forward triangle,
and pure forward diamond tiles bottom-up via `apply_cfg_tile_rules`.
"""

from __future__ import annotations

from typing import Any, Dict

from tools.symbolic_ssa.cfg_rewriter import (
    apply_cfg_tile_rules,
    make_degenerate_branch_rule,
    make_empty_jump_thread_rule,
    make_pure_diamond_rule,
    make_pure_triangle_rule,
)
from tools.symbolic_ssa.passes.ir_utils import Stratum1Stats


def pass_if_convert(
    blocks_by_id: Dict[int, Dict[str, Any]],
    nodes_by_id: Dict[int, Dict[str, Any]],
    stats: Stratum1Stats,
    max_arm_nodes: int = 128,
) -> bool:
    """If-convert pure acyclic selections and simplify trivial forwarding tiles bottom-up."""
    rules = [
        make_degenerate_branch_rule(),
        make_empty_jump_thread_rule(),
        make_pure_triangle_rule(max_arm_nodes=max_arm_nodes),
        make_pure_diamond_rule(max_arm_nodes=max_arm_nodes),
    ]
    return apply_cfg_tile_rules(blocks_by_id, nodes_by_id, stats, rules)

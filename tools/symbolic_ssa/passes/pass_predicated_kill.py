"""
Pass 1.2: Predicated-Discard Folding.

Folds `if (c) { pure; Kill; } else { C }` into `pure; Kill(c); br C` via
`tools.symbolic_ssa.cfg_rewriter.make_predicated_kill_rule`.
"""

from __future__ import annotations

from typing import Any

from tools.symbolic_ssa.cfg_rewriter import (
    apply_cfg_tile_rules,
    make_predicated_kill_rule,
)
from tools.symbolic_ssa.passes.ir_utils import Stratum1Stats


def pass_predicated_kill(
    blocks_by_id: dict[int, dict[str, Any]],
    nodes_by_id: dict[int, dict[str, Any]],
    stats: Stratum1Stats,
) -> bool:
    """Fold single-predecessor pure+Kill branch arms into inline predicated Kill nodes."""
    return apply_cfg_tile_rules(blocks_by_id, nodes_by_id, stats, [make_predicated_kill_rule()])

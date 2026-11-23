#!/usr/bin/env python3
"""
Pass 1.4: Trivial CFG Chain Contraction.

Merges sequential single-edge blocks `A -> B` whenever `outdeg(A) == 1`,
`indeg(B) == 1`, `B != 0` (entry), and `B` is not a loop header via
`tools.symbolic_ssa.cfg_rewriter.make_chain_contract_rule`.
"""

from __future__ import annotations

from typing import Any, Dict

from tools.symbolic_ssa.cfg_rewriter import (
    apply_cfg_tile_rules,
    make_chain_contract_rule,
)
from tools.symbolic_ssa.passes.ir_utils import Stratum1Stats


def pass_contract_chains(
    blocks_by_id: Dict[int, Dict[str, Any]],
    nodes_by_id: Dict[int, Dict[str, Any]],
    stats: Stratum1Stats,
) -> bool:
    """Contract trivial unconditional `A -> B` block chains to fixed point."""
    return apply_cfg_tile_rules(blocks_by_id, nodes_by_id, stats, [make_chain_contract_rule()])

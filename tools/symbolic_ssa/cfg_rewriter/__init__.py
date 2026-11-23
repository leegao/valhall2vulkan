"""Reusable CFG rewriting, Phi maintenance, and bottom-up tile-rewrite engine."""

from tools.symbolic_ssa.cfg_rewriter.rewriter import (
    TERMINATOR_FIELDS,
    CfgRewriter,
    pass_cleanup_phis,
)
from tools.symbolic_ssa.cfg_rewriter.tiles import (
    CfgRewriteContext,
    CfgTileRule,
    TileMatch,
    apply_cfg_tile_rules,
    make_chain_contract_rule,
    make_degenerate_branch_rule,
    make_empty_jump_thread_rule,
    make_predicated_kill_rule,
    make_pure_diamond_rule,
    make_pure_triangle_rule,
)

__all__ = [
    "CfgRewriteContext",
    "CfgRewriter",
    "CfgTileRule",
    "TERMINATOR_FIELDS",
    "TileMatch",
    "apply_cfg_tile_rules",
    "make_chain_contract_rule",
    "make_degenerate_branch_rule",
    "make_empty_jump_thread_rule",
    "make_predicated_kill_rule",
    "make_pure_diamond_rule",
    "make_pure_triangle_rule",
    "pass_cleanup_phis",
]

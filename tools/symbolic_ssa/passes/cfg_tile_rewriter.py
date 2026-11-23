#!/usr/bin/env python3
"""Backward-compatible re-export shim for `tools.symbolic_ssa.cfg_rewriter`."""

from tools.symbolic_ssa.cfg_rewriter import (
    CfgRewriteContext,
    CfgRewriter,
    CfgTileRule,
    TileMatch,
    apply_cfg_tile_rules,
    pass_cleanup_phis,
)

__all__ = [
    "CfgRewriteContext",
    "CfgRewriter",
    "CfgTileRule",
    "TileMatch",
    "apply_cfg_tile_rules",
    "pass_cleanup_phis",
]

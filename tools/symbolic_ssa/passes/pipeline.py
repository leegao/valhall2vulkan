"""Stratum 1 fixed-point canonicalization pipeline orchestrating all passes."""

from __future__ import annotations

import copy
from typing import Any, Dict, Set, Tuple

from tools.symbolic_ssa.cfg_rewriter import pass_cleanup_phis
from tools.symbolic_ssa.passes.ir_utils import Stratum1Stats, compute_rpo
from tools.symbolic_ssa.passes.pass_contract_chains import pass_contract_chains
from tools.symbolic_ssa.passes.pass_gvn_dce import pass_gvn_dce
from tools.symbolic_ssa.passes.pass_if_convert import pass_if_convert
from tools.symbolic_ssa.passes.pass_predicated_kill import pass_predicated_kill
from tools.symbolic_ssa.passes.pass_prune_const_branches import pass_prune_const_branches
from tools.symbolic_ssa.passes.pass_readonly_tokens import pass_readonly_tokens


def canonicalize_ssa_module(
    ssa_json: Dict[str, Any],
    max_rounds: int = 8,
    max_arm_nodes: int = 128,
) -> Tuple[Dict[str, Any], Stratum1Stats]:
    """Run all Stratum 1 canonicalization passes (including generic dead/trivial Phi cleanup) to fixed point."""
    mod = copy.deepcopy(ssa_json)
    blocks_by_id: Dict[int, Dict[str, Any]] = {int(b["id"]): b for b in mod.get("blocks", [])}
    nodes_by_id: Dict[int, Dict[str, Any]] = {int(n["id"]): n for n in mod.get("nodes", [])}

    stats = Stratum1Stats(
        orig_blocks=len(blocks_by_id),
        orig_nodes=len(nodes_by_id),
    )

    readonly_load_ids: Set[int] = set()
    for rnd in range(1, max_rounds + 1):
        stats.rounds = rnd
        c1 = pass_prune_const_branches(blocks_by_id, nodes_by_id, stats)
        c2 = pass_predicated_kill(blocks_by_id, nodes_by_id, stats)
        c3 = pass_if_convert(blocks_by_id, nodes_by_id, stats, max_arm_nodes=max_arm_nodes)
        c4 = pass_contract_chains(blocks_by_id, nodes_by_id, stats)
        c_phi = pass_cleanup_phis(blocks_by_id, nodes_by_id, stats)
        c5, readonly_load_ids = pass_readonly_tokens(blocks_by_id, nodes_by_id, stats)
        c6 = pass_gvn_dce(blocks_by_id, nodes_by_id, readonly_load_ids, stats)
        if not (c1 or c2 or c3 or c4 or c_phi or c5 or c6):
            break

    rpo = compute_rpo(blocks_by_id, 0)
    ordered_blocks = [blocks_by_id[bid] for bid in rpo]
    ordered_nodes = [nodes_by_id[nid] for nid in sorted(nodes_by_id.keys())]

    stats.canon_blocks = len(ordered_blocks)
    stats.canon_nodes = len(ordered_nodes)

    return {"blocks": ordered_blocks, "nodes": ordered_nodes}, stats

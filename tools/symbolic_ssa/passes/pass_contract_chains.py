#!/usr/bin/env python3
"""
Pass 1.4: Trivial CFG Chain Contraction.

Merges sequential single-edge blocks A -> B whenever outdeg(A) == 1,
indeg(B) == 1, B != entry (0), and B is not a loop header, eliminating
artificial sub-block chains introduced by split_cfg_subblocks.
"""

from __future__ import annotations

from typing import Any, Dict

from tools.symbolic_ssa.passes.ir_utils import (
    Stratum1Stats,
    block_succs,
    collapse_trivial_phis,
    extract_non_marker_nodes,
    rebuild_preds,
    replace_all_uses,
)


def pass_contract_chains(
    blocks_by_id: Dict[int, Dict[str, Any]],
    nodes_by_id: Dict[int, Dict[str, Any]],
    stats: Stratum1Stats,
) -> bool:
    """Contract trivial unconditional A -> B block chains to fixed point."""
    changed = False
    while True:
        merged_any = False
        for aid in list(blocks_by_id.keys()):
            if aid not in blocks_by_id:
                continue
            a = blocks_by_id[aid]
            if a.get("is_cond") or a.get("is_end") or a.get("is_discard"):
                continue
            succs = block_succs(a)
            if len(succs) != 1:
                continue
            bid = succs[0]
            if bid == aid or bid == 0 or bid not in blocks_by_id:
                continue
            b = blocks_by_id[bid]
            if b.get("preds") != [aid] or b.get("is_loop_header"):
                continue

            repl: Dict[int, int] = {}
            for pid in b.get("phis", []):
                pn = nodes_by_id.get(pid)
                if pn and pn.get("phi"):
                    repl[pid] = int(pn["phi"][0][1])
                    del nodes_by_id[pid]
            if repl:
                replace_all_uses(blocks_by_id, nodes_by_id, repl)

            a_nodes = extract_non_marker_nodes(a, nodes_by_id)
            for nid in a.get("nodes", []):
                if nid not in a_nodes:
                    nodes_by_id.pop(nid, None)
            b_nodes = list(b.get("nodes", []))
            for nid in b_nodes:
                if nid in nodes_by_id:
                    nodes_by_id[nid]["blk"] = aid
            a["nodes"] = a_nodes + b_nodes

            for field in (
                "branch_target",
                "fallthrough",
                "is_cond",
                "is_uncond",
                "is_end",
                "is_discard",
                "branch_cond",
                "sel_merge",
                "loop_merge",
                "loop_backedge",
            ):
                a[field] = b.get(field)

            for sid in block_succs(b):
                sb = blocks_by_id.get(sid)
                if not sb:
                    continue
                sb["preds"] = [aid if p == bid else p for p in sb.get("preds", [])]
                for pid in sb.get("phis", []):
                    pn = nodes_by_id.get(pid)
                    if pn and "phi" in pn:
                        pn["phi"] = [[aid if int(p) == bid else int(p), int(v)] for p, v in pn["phi"]]

            del blocks_by_id[bid]
            stats.chains_contracted += 1
            merged_any = True
            changed = True
        if not merged_any:
            break

    if changed:
        rebuild_preds(blocks_by_id, nodes_by_id)
        collapse_trivial_phis(blocks_by_id, nodes_by_id)
    return changed

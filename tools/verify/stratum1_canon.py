#!/usr/bin/env python3
"""
Stratum 1: Pre-SMT Structural Canonicalization for ValhallSsaModule JSON.

Implements a fixed-point pipeline of semantics-preserving IR/CFG passes:
  1.1 Reachability & constant/degenerate branch pruning
  1.2 Predicated-discard folding (branch-to-Kill -> predicated Kill)
  1.3 Pure acyclic selection if-conversion (triangles & diamonds -> Select)
  1.4 Trivial CFG chain contraction (eliminates split_cfg_subblocks chains)
  1.5 Read-only Token detachment (immutable UBO/PushConstant/FAU/Tex loads)
  1.6 Algebraic normalization, dominator-scoped GVN (hash-consing), and DCE

Plus canonical CFG isomorphism checking (up to block relabeling and branch
polarity inversion) for evaluating structural alignment between direct_ssa
and valhall_ssa.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, Dict, List, Optional, Set, Tuple

SIDE_EFFECT_OPS: Set[str] = {
    "PsbStore",
    "StoreFragColor",
    "StoreBuiltinOutput",
    "StoreVertVarying",
    "StoreTexel",
    "Kill",
    "AtomicRmw",
    "AtomicExchange",
    "AtomicCmpXchg",
    "ControlBarrier",
}

CONTROL_MARKER_OPS: Set[str] = {
    "Branch",
    "BranchCond",
    "Return",
}

COMMUTATIVE_OPS: Set[str] = {
    "IAdd",
    "IMul",
    "BitAnd",
    "BitOr",
    "BitXor",
    "IAdd64",
    "IMul64",
    "BitAnd64",
    "BitOr64",
    "BitXor64",
    "ICmpEq",
    "ICmpNe",
    "LogicalAnd",
    "LogicalOr",

}

# Ops that are pure and safe to GVN when their inputs match (excludes side effects,
# Phi, and non-detached mutable memory ops).
NEVER_GVN_OPS: Set[str] = SIDE_EFFECT_OPS | CONTROL_MARKER_OPS | {"Phi", "Nop"}


@dataclasses.dataclass
class Stratum1Stats:
    orig_blocks: int = 0
    orig_nodes: int = 0
    canon_blocks: int = 0
    canon_nodes: int = 0
    rounds: int = 0
    unreachable_blocks_removed: int = 0
    const_branches_folded: int = 0
    discards_predicated: int = 0
    triangles_if_converted: int = 0
    diamonds_if_converted: int = 0
    chains_contracted: int = 0
    readonly_tokens_detached: int = 0
    gvn_merged: int = 0
    alg_simplified: int = 0
    dce_removed: int = 0


def _block_succs(blk: Dict[str, Any]) -> List[int]:
    if blk.get("is_end") or blk.get("is_discard"):
        return []
    bt = int(blk.get("branch_target", -1))
    ft = int(blk.get("fallthrough", -1))
    if blk.get("is_cond"):
        res = []
        if bt >= 0:
            res.append(bt)
        if ft >= 0 and ft != bt:
            res.append(ft)
        return res
    if blk.get("is_uncond"):
        return [bt] if bt >= 0 else []
    if bt >= 0:
        return [bt]
    if ft >= 0:
        return [ft]
    return []


def _compute_rpo(blocks_by_id: Dict[int, Dict[str, Any]], entry: int = 0) -> List[int]:
    if entry not in blocks_by_id:
        return []
    visited: Set[int] = set()
    postorder: List[int] = []
    stack: List[Tuple[int, int]] = [(entry, 0)]
    while stack:
        u, step = stack.pop()
        if step == 0:
            if u in visited:
                continue
            visited.add(u)
            stack.append((u, 1))
            for s in reversed(_block_succs(blocks_by_id[u])):
                if s in blocks_by_id and s not in visited:
                    stack.append((s, 0))
        else:
            postorder.append(u)
    return list(reversed(postorder))


def _compute_idom(blocks_by_id: Dict[int, Dict[str, Any]], rpo: List[int]) -> Dict[int, int]:
    """Cooper-Harvey-Kennedy iterative dominator tree."""
    if not rpo:
        return {}
    entry = rpo[0]
    rpo_idx = {b: i for i, b in enumerate(rpo)}
    idom: Dict[int, int] = {entry: entry}

    def intersect(b1: int, b2: int) -> int:
        f1, f2 = b1, b2
        while f1 != f2:
            while rpo_idx[f1] > rpo_idx[f2]:
                f1 = idom[f1]
            while rpo_idx[f2] > rpo_idx[f1]:
                f2 = idom[f2]
        return f1

    changed = True
    while changed:
        changed = False
        for b in rpo[1:]:
            preds = [int(p) for p in blocks_by_id[b].get("preds", []) if int(p) in idom]
            if not preds:
                continue
            new_idom = preds[0]
            for p in preds[1:]:
                new_idom = intersect(p, new_idom)
            if idom.get(b) != new_idom:
                idom[b] = new_idom
                changed = True
    return idom


def _rebuild_preds(blocks_by_id: Dict[int, Dict[str, Any]], nodes_by_id: Dict[int, Dict[str, Any]]) -> None:
    new_preds: Dict[int, List[int]] = {bid: [] for bid in blocks_by_id}
    for bid, blk in blocks_by_id.items():
        for s in _block_succs(blk):
            if s in new_preds and bid not in new_preds[s]:
                new_preds[s].append(bid)
    for bid, blk in blocks_by_id.items():
        blk["preds"] = new_preds[bid]
        pred_set = set(new_preds[bid])
        for pid in blk.get("phis", []):
            pn = nodes_by_id.get(pid)
            if pn and "phi" in pn:
                pn["phi"] = [[int(p), int(v)] for p, v in pn["phi"] if int(p) in pred_set]


def _replace_all_uses(
    blocks_by_id: Dict[int, Dict[str, Any]],
    nodes_by_id: Dict[int, Dict[str, Any]],
    repl: Dict[int, int],
) -> None:
    if not repl:
        return

    def resolve(x: int) -> int:
        seen: Set[int] = set()
        while x in repl and x not in seen:
            seen.add(x)
            x = repl[x]
        return x

    for n in nodes_by_id.values():
        if n.get("args"):
            n["args"] = [resolve(int(a)) for a in n["args"]]
        if "phi" in n:
            n["phi"] = [[int(p), resolve(int(v))] for p, v in n["phi"]]
    for blk in blocks_by_id.values():
        bc = int(blk.get("branch_cond", 0))
        if bc in repl:
            blk["branch_cond"] = resolve(bc)


def _collapse_trivial_phis(
    blocks_by_id: Dict[int, Dict[str, Any]],
    nodes_by_id: Dict[int, Dict[str, Any]],
) -> int:
    """Eliminate Phi nodes with 1 unique non-self incoming value."""
    repl: Dict[int, int] = {}
    for blk in blocks_by_id.values():
        keep_phis: List[int] = []
        for pid in blk.get("phis", []):
            pn = nodes_by_id.get(pid)
            if not pn:
                continue
            incs = pn.get("phi", [])
            vals = {int(v) for _, v in incs if int(v) != pid}
            if len(vals) == 1:
                repl[pid] = next(iter(vals))
                del nodes_by_id[pid]
            else:
                keep_phis.append(pid)
        blk["phis"] = keep_phis
    if repl:
        _replace_all_uses(blocks_by_id, nodes_by_id, repl)
    return len(repl)


def _is_block_pure(blk: Dict[str, Any], nodes_by_id: Dict[int, Dict[str, Any]]) -> bool:
    for nid in blk.get("nodes", []):
        n = nodes_by_id.get(nid)
        if n and n["op"] in SIDE_EFFECT_OPS:
            return False
    return True


def _next_node_id(nodes_by_id: Dict[int, Dict[str, Any]]) -> int:
    return (max(nodes_by_id.keys()) + 1) if nodes_by_id else 1


def _insert_before_terminator(blk: Dict[str, Any], nodes_by_id: Dict[int, Dict[str, Any]], new_ids: List[int]) -> None:
    nodes = blk["nodes"]
    idx = len(nodes)
    while idx > 0 and nodes_by_id.get(nodes[idx - 1], {}).get("op") in CONTROL_MARKER_OPS:
        idx -= 1
    blk["nodes"] = nodes[:idx] + new_ids + nodes[idx:]


def _extract_non_marker_nodes(blk: Dict[str, Any], nodes_by_id: Dict[int, Dict[str, Any]]) -> List[int]:
    return [nid for nid in blk.get("nodes", []) if nodes_by_id.get(nid, {}).get("op") not in CONTROL_MARKER_OPS]


def _pass_prune_and_const_branches(
    blocks_by_id: Dict[int, Dict[str, Any]],
    nodes_by_id: Dict[int, Dict[str, Any]],
    stats: Stratum1Stats,
) -> bool:
    changed = False
    for blk in blocks_by_id.values():
        if blk.get("is_cond"):
            bt = int(blk.get("branch_target", -1))
            ft = int(blk.get("fallthrough", -1))
            bc = int(blk.get("branch_cond", 0))
            cn = nodes_by_id.get(bc)
            taken: Optional[int] = None
            if bt == ft and bt >= 0:
                taken = bt
            elif cn is not None and cn["op"] in ("ConstBool", "ConstU32", "ConstS32"):
                is_true = (int(cn.get("imm_u32", 0)) != 0) or bool(cn.get("aux_bool", False))
                taken = bt if is_true else ft
            if taken is not None and taken >= 0:
                blk["is_cond"] = False
                blk["is_uncond"] = True
                blk["branch_target"] = taken
                blk["fallthrough"] = -1
                blk["branch_cond"] = 0
                stats.const_branches_folded += 1
                changed = True

    rpo = _compute_rpo(blocks_by_id, 0)
    reach_set = set(rpo)
    dead_blks = [bid for bid in list(blocks_by_id.keys()) if bid not in reach_set]
    for bid in dead_blks:
        blk = blocks_by_id.pop(bid)
        for nid in blk.get("phis", []) + blk.get("nodes", []):
            nodes_by_id.pop(nid, None)
        stats.unreachable_blocks_removed += 1
        changed = True

    _rebuild_preds(blocks_by_id, nodes_by_id)
    if _collapse_trivial_phis(blocks_by_id, nodes_by_id) > 0:
        changed = True
    return changed


def _pass_fold_branch_to_discard(
    blocks_by_id: Dict[int, Dict[str, Any]],
    nodes_by_id: Dict[int, Dict[str, Any]],
    stats: Stratum1Stats,
) -> bool:
    """Fold `if (c) { pure; Kill; } else { C }` into `pure; Kill(c); br C`."""
    changed = False
    for hid in list(blocks_by_id.keys()):
        if hid not in blocks_by_id:
            continue
        h = blocks_by_id[hid]
        if not h.get("is_cond"):
            continue
        bt = int(h.get("branch_target", -1))
        ft = int(h.get("fallthrough", -1))
        if bt < 0 or ft < 0 or bt == ft or bt == hid or ft == hid:
            continue

        matched_side: Optional[Tuple[int, int, bool]] = None  # (kill_bid, cont_bid, kill_on_true)
        for k_bid, c_bid, on_true in ((bt, ft, True), (ft, bt, False)):
            kb = blocks_by_id.get(k_bid)
            if not kb or kb.get("preds") != [hid] or kb.get("phis"):
                continue
            if _block_succs(kb):
                continue
            kills = []
            other_se = False
            for nid in kb.get("nodes", []):
                n = nodes_by_id.get(nid)
                if not n:
                    continue
                if n["op"] == "Kill":
                    kills.append(n)
                elif n["op"] in SIDE_EFFECT_OPS:
                    other_se = True
                    break
            if other_se or len(kills) != 1:
                continue
            # Ensure it's an unconditional kill (or we can conjoin its cond)
            matched_side = (k_bid, c_bid, on_true)
            break

        if matched_side is None:
            continue

        k_bid, c_bid, on_true = matched_side
        kb = blocks_by_id[k_bid]
        bcond = int(h.get("branch_cond", 0))
        if bcond == 0:
            continue

        new_nodes_to_insert: List[int] = []
        pred_id = bcond
        if not on_true:
            not_id = _next_node_id(nodes_by_id)
            nodes_by_id[not_id] = {
                "id": not_id,
                "op": "LogicalNot",
                "type": "Bool",
                "blk": hid,
                "args": [bcond],
            }
            new_nodes_to_insert.append(not_id)
            pred_id = not_id

        for nid in _extract_non_marker_nodes(kb, nodes_by_id):
            n = nodes_by_id[nid]
            n["blk"] = hid
            if n["op"] == "Kill":
                orig_args = n.get("args", [])
                tok = orig_args[-1] if orig_args else 0
                if n.get("aux_bool") and len(orig_args) >= 2:
                    and_id = _next_node_id(nodes_by_id)
                    nodes_by_id[and_id] = {
                        "id": and_id,
                        "op": "LogicalAnd",
                        "type": "Bool",
                        "blk": hid,
                        "args": [pred_id, orig_args[0]],
                    }
                    new_nodes_to_insert.append(and_id)
                    n["args"] = [and_id, tok]
                else:
                    n["aux_bool"] = True
                    n["args"] = [pred_id, tok]
            new_nodes_to_insert.append(nid)

        _insert_before_terminator(h, nodes_by_id, new_nodes_to_insert)
        # Drop any marker nodes from kb
        for nid in kb.get("nodes", []):
            if nid not in new_nodes_to_insert:
                nodes_by_id.pop(nid, None)
        del blocks_by_id[k_bid]

        h["is_cond"] = False
        h["is_uncond"] = True
        h["branch_target"] = c_bid
        h["fallthrough"] = -1
        h["branch_cond"] = 0
        stats.discards_predicated += 1
        changed = True

    if changed:
        _rebuild_preds(blocks_by_id, nodes_by_id)
        _collapse_trivial_phis(blocks_by_id, nodes_by_id)
    return changed


def _pass_if_convert_pure_selections(
    blocks_by_id: Dict[int, Dict[str, Any]],
    nodes_by_id: Dict[int, Dict[str, Any]],
    stats: Stratum1Stats,
    max_arm_nodes: int = 128,
) -> bool:
    """If-convert pure acyclic triangles (H->{A,M}, A->M) and diamonds (H->{T,F}, T->M, F->M)."""
    any_changed = False
    while True:
        changed = False
        rpo = _compute_rpo(blocks_by_id, 0)
        rpo_idx = {b: i for i, b in enumerate(rpo)}

        for hid in list(rpo):
            if hid not in blocks_by_id:
                continue
            h = blocks_by_id[hid]
            if not h.get("is_cond"):
                continue
            bt = int(h.get("branch_target", -1))
            ft = int(h.get("fallthrough", -1))
            bcond = int(h.get("branch_cond", 0))
            if bt < 0 or ft < 0 or bt == ft or bcond == 0:
                continue
            if bt not in blocks_by_id or ft not in blocks_by_id:
                continue
            # Require forward edges (acyclic selection)
            if rpo_idx.get(bt, -1) <= rpo_idx.get(hid, -1) or rpo_idx.get(ft, -1) <= rpo_idx.get(hid, -1):
                continue

            bt_blk = blocks_by_id[bt]
            ft_blk = blocks_by_id[ft]
            bt_succs = _block_succs(bt_blk)
            ft_succs = _block_succs(ft_blk)

            # Case 1: Pure Triangle: H -> A -> M and H -> M
            tri: Optional[Tuple[int, int, bool]] = None  # (arm_id, merge_id, arm_is_true)
            if (
                bt_blk.get("preds") == [hid]
                and not bt_blk.get("phis")
                and bt_succs == [ft]
                and len(bt_blk.get("nodes", [])) <= max_arm_nodes
                and _is_block_pure(bt_blk, nodes_by_id)
                and rpo_idx.get(ft, -1) > rpo_idx.get(bt, -1)
            ):
                tri = (bt, ft, True)
            elif (
                ft_blk.get("preds") == [hid]
                and not ft_blk.get("phis")
                and ft_succs == [bt]
                and len(ft_blk.get("nodes", [])) <= max_arm_nodes
                and _is_block_pure(ft_blk, nodes_by_id)
                and rpo_idx.get(bt, -1) > rpo_idx.get(ft, -1)
            ):
                tri = (ft, bt, False)

            if tri is not None:
                arm_id, merge_id, arm_is_true = tri
                arm_blk = blocks_by_id[arm_id]
                merge_blk = blocks_by_id[merge_id]

                moved = _extract_non_marker_nodes(arm_blk, nodes_by_id)
                for nid in moved:
                    nodes_by_id[nid]["blk"] = hid

                select_nodes: List[int] = []
                for pid in merge_blk.get("phis", []):
                    pn = nodes_by_id.get(pid)
                    if not pn:
                        continue
                    inc_map = {int(p): int(v) for p, v in pn.get("phi", [])}
                    if hid in inc_map and arm_id in inc_map:
                        v_h = inc_map[hid]
                        v_a = inc_map[arm_id]
                        if pn.get("type") == "Token" or v_h == v_a:
                            merged_v = v_h
                        else:
                            sel_id = _next_node_id(nodes_by_id)
                            t_v, f_v = (v_a, v_h) if arm_is_true else (v_h, v_a)
                            nodes_by_id[sel_id] = {
                                "id": sel_id,
                                "op": "Select",
                                "type": pn.get("type", "U32"),
                                "blk": hid,
                                "args": [bcond, t_v, f_v],
                            }
                            select_nodes.append(sel_id)
                            merged_v = sel_id
                        new_phi = [[p, v] for p, v in pn["phi"] if int(p) not in (hid, arm_id)]
                        new_phi.append([hid, merged_v])
                        pn["phi"] = new_phi

                _insert_before_terminator(h, nodes_by_id, moved + select_nodes)
                for nid in arm_blk.get("nodes", []):
                    if nid not in moved:
                        nodes_by_id.pop(nid, None)
                del blocks_by_id[arm_id]

                h["is_cond"] = False
                h["is_uncond"] = True
                h["branch_target"] = merge_id
                h["fallthrough"] = -1
                h["branch_cond"] = 0
                stats.triangles_if_converted += 1
                changed = True
                break  # restart RPO cleanly

            # Case 2: Pure Diamond: H -> T -> M and H -> F -> M
            if (
                bt_blk.get("preds") == [hid]
                and ft_blk.get("preds") == [hid]
                and not bt_blk.get("phis")
                and not ft_blk.get("phis")
                and len(bt_succs) == 1
                and bt_succs == ft_succs
                and bt_succs[0] in blocks_by_id
                and len(bt_blk.get("nodes", [])) <= max_arm_nodes
                and len(ft_blk.get("nodes", [])) <= max_arm_nodes
                and _is_block_pure(bt_blk, nodes_by_id)
                and _is_block_pure(ft_blk, nodes_by_id)
            ):
                merge_id = bt_succs[0]
                if rpo_idx.get(merge_id, -1) <= max(rpo_idx.get(bt, -1), rpo_idx.get(ft, -1)):
                    continue
                merge_blk = blocks_by_id[merge_id]
                moved_t = _extract_non_marker_nodes(bt_blk, nodes_by_id)
                moved_f = _extract_non_marker_nodes(ft_blk, nodes_by_id)
                for nid in moved_t + moved_f:
                    nodes_by_id[nid]["blk"] = hid

                select_nodes = []
                for pid in merge_blk.get("phis", []):
                    pn = nodes_by_id.get(pid)
                    if not pn:
                        continue
                    inc_map = {int(p): int(v) for p, v in pn.get("phi", [])}
                    if bt in inc_map and ft in inc_map:
                        v_t = inc_map[bt]
                        v_f = inc_map[ft]
                        if pn.get("type") == "Token" or v_t == v_f:
                            merged_v = v_t
                        else:
                            sel_id = _next_node_id(nodes_by_id)
                            nodes_by_id[sel_id] = {
                                "id": sel_id,
                                "op": "Select",
                                "type": pn.get("type", "U32"),
                                "blk": hid,
                                "args": [bcond, v_t, v_f],
                            }
                            select_nodes.append(sel_id)
                            merged_v = sel_id
                        new_phi = [[p, v] for p, v in pn["phi"] if int(p) not in (bt, ft)]
                        new_phi.append([hid, merged_v])
                        pn["phi"] = new_phi

                _insert_before_terminator(h, nodes_by_id, moved_t + moved_f + select_nodes)
                for nid in bt_blk.get("nodes", []):
                    if nid not in moved_t:
                        nodes_by_id.pop(nid, None)
                for nid in ft_blk.get("nodes", []):
                    if nid not in moved_f:
                        nodes_by_id.pop(nid, None)
                del blocks_by_id[bt]
                del blocks_by_id[ft]

                h["is_cond"] = False
                h["is_uncond"] = True
                h["branch_target"] = merge_id
                h["fallthrough"] = -1
                h["branch_cond"] = 0
                stats.diamonds_if_converted += 1
                changed = True
                break

        if changed:
            any_changed = True
            _rebuild_preds(blocks_by_id, nodes_by_id)
            _collapse_trivial_phis(blocks_by_id, nodes_by_id)
        else:
            break
    return any_changed


def _pass_contract_trivial_chains(
    blocks_by_id: Dict[int, Dict[str, Any]],
    nodes_by_id: Dict[int, Dict[str, Any]],
    stats: Stratum1Stats,
) -> bool:
    """Merge A -> B whenever outdeg(A) == 1, indeg(B) == 1, B != 0, and B is not a loop header."""
    changed = False
    while True:
        merged_any = False
        for aid in list(blocks_by_id.keys()):
            if aid not in blocks_by_id:
                continue
            a = blocks_by_id[aid]
            if a.get("is_cond") or a.get("is_end") or a.get("is_discard"):
                continue
            succs = _block_succs(a)
            if len(succs) != 1:
                continue
            bid = succs[0]
            if bid == aid or bid == 0 or bid not in blocks_by_id:
                continue
            b = blocks_by_id[bid]
            if b.get("preds") != [aid] or b.get("is_loop_header"):
                continue

            # Resolve single-incoming Phis in B
            repl: Dict[int, int] = {}
            for pid in b.get("phis", []):
                pn = nodes_by_id.get(pid)
                if pn and pn.get("phi"):
                    repl[pid] = int(pn["phi"][0][1])
                    del nodes_by_id[pid]
            if repl:
                _replace_all_uses(blocks_by_id, nodes_by_id, repl)

            # Strip trailing Branch marker from A, append B's nodes
            a_nodes = _extract_non_marker_nodes(a, nodes_by_id)
            for nid in a.get("nodes", []):
                if nid not in a_nodes:
                    nodes_by_id.pop(nid, None)
            b_nodes = list(b.get("nodes", []))
            for nid in b_nodes:
                if nid in nodes_by_id:
                    nodes_by_id[nid]["blk"] = aid
            a["nodes"] = a_nodes + b_nodes

            # Transfer B's CFG terminator to A
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

            # Update successors of B: replace pred bid -> aid in preds & Phi incomings
            for sid in _block_succs(b):
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
        _rebuild_preds(blocks_by_id, nodes_by_id)
        _collapse_trivial_phis(blocks_by_id, nodes_by_id)
    return changed


def _infer_buf_key(node: Dict[str, Any], nodes_by_id: Dict[int, Dict[str, Any]]) -> Optional[str]:
    astr = node.get("aux_str", "")
    if astr.startswith("sym:"):
        return astr[4:].split("[", 1)[0]
    if astr.startswith(("ubo:", "ssbo:", "push_const", "img:", "workgroup:")):
        return astr
    if astr:
        return astr
    # Walk additive address chain up to depth 8 to find LoadSidebandU64
    if not node.get("args"):
        return None
    cur_id = int(node["args"][0])
    for _ in range(8):
        cn = nodes_by_id.get(cur_id)
        if not cn:
            return None
        if cn["op"] == "LoadSidebandU64":
            return cn.get("aux_str") or f"sideband_{cn.get('imm_u32', 0)}"
        if cn["op"] in ("IAdd64", "ISub64", "Copy", "Bitcast") and cn.get("args"):
            # Pick the U64/non-const child
            picked = None
            for aid in cn["args"]:
                an = nodes_by_id.get(int(aid))
                if an and an["op"] not in ("ConstU64", "ConstU32", "ConstS32", "SConvert32To64", "UConvert32To64"):
                    picked = int(aid)
                    break
            if picked is None:
                return None
            cur_id = picked
        else:
            return None
    return None


def _pass_detach_readonly_tokens(
    blocks_by_id: Dict[int, Dict[str, Any]],
    nodes_by_id: Dict[int, Dict[str, Any]],
    stats: Stratum1Stats,
) -> Tuple[bool, Set[int]]:
    """Detach Token chains from provably read-only PsbLoad and TexSample ops."""
    mutated_bkeys: Set[str] = set()
    has_unknown_store = False
    mutated_imgs: Set[str] = set()

    for n in nodes_by_id.values():
        op = n["op"]
        if op in ("PsbStore", "AtomicRmw", "AtomicExchange", "AtomicCmpXchg"):
            bk = _infer_buf_key(n, nodes_by_id)
            if bk is None:
                has_unknown_store = True
            else:
                mutated_bkeys.add(bk)
        elif op == "StoreTexel":
            mutated_imgs.add(n.get("aux_str", ""))

    # Find or create canonical root Token in entry block (bb0)
    root_tok = 0
    if 0 in blocks_by_id:
        for nid in blocks_by_id[0].get("nodes", []):
            n = nodes_by_id.get(nid)
            if n and n["op"] == "Nop" and n.get("type") == "Token":
                root_tok = nid
                break
        if root_tok == 0:
            root_tok = _next_node_id(nodes_by_id)
            nodes_by_id[root_tok] = {
                "id": root_tok,
                "op": "Nop",
                "type": "Token",
                "blk": 0,
                "args": [],
            }
            blocks_by_id[0]["nodes"].insert(0, root_tok)

    readonly_load_ids: Set[int] = set()
    for nid, n in nodes_by_id.items():
        op = n["op"]
        if op == "PsbLoad":
            bk = _infer_buf_key(n, nodes_by_id)
            # UBOs and push_const are architecturally immutable in Vulkan/SPIR-V
            is_arch_ro = bk is not None and (bk.startswith("ubo:") or bk == "push_const")
            is_unmutated = (not has_unknown_store) and (bk is not None) and (bk not in mutated_bkeys)
            if is_arch_ro or is_unmutated:
                readonly_load_ids.add(nid)
        elif op == "TexSample":
            tkey = n.get("aux_str", "")
            if tkey not in mutated_imgs:
                readonly_load_ids.add(nid)

    if not readonly_load_ids:
        return False, set()

    # Map TupleExtract(ro_load, imm_u32=1) -> ro_load's incoming token
    tok_repl: Dict[int, int] = {}
    for nid, n in nodes_by_id.items():
        if n["op"] == "TupleExtract" and int(n.get("imm_u32", 0)) == 1 and n.get("args"):
            src = int(n["args"][0])
            if src in readonly_load_ids:
                src_node = nodes_by_id[src]
                if len(src_node.get("args", [])) >= 2:
                    tok_in = int(src_node["args"][-1])
                    if tok_in != nid:
                        tok_repl[nid] = tok_in

    changed = False
    if tok_repl:
        _replace_all_uses(blocks_by_id, nodes_by_id, tok_repl)
        stats.readonly_tokens_detached += len(tok_repl)
        changed = True

    # Rebind each read-only load's own token input to root_tok so identical loads unify under GVN
    for lid in readonly_load_ids:
        ln = nodes_by_id[lid]
        if len(ln.get("args", [])) >= 2 and int(ln["args"][-1]) != root_tok:
            ln["args"][-1] = root_tok
            stats.readonly_tokens_detached += 1
            changed = True

    return changed, readonly_load_ids


def _pass_normalize_gvn_dce(
    blocks_by_id: Dict[int, Dict[str, Any]],
    nodes_by_id: Dict[int, Dict[str, Any]],
    readonly_load_ids: Set[int],
    stats: Stratum1Stats,
) -> bool:
    """Algebraic identity folding, dominator-scoped GVN (hash-consing), and DCE."""
    changed = False

    # 1. Local algebraic / identity / tuple / vector simplifications
    repl: Dict[int, int] = {}
    for nid, n in list(nodes_by_id.items()):
        op = n["op"]
        args = [int(a) for a in n.get("args", [])]
        if op == "Copy" and len(args) == 1:
            repl[nid] = args[0]
        elif op == "Bitcast" and len(args) == 1:
            an = nodes_by_id.get(args[0])
            if an and an.get("type") == n.get("type"):
                repl[nid] = args[0]
            elif an and an["op"] == "Bitcast" and len(an.get("args", [])) == 1:
                inner = nodes_by_id.get(int(an["args"][0]))
                if inner and inner.get("type") == n.get("type"):
                    repl[nid] = int(an["args"][0])
        elif op == "LogicalNot" and len(args) == 1:
            an = nodes_by_id.get(args[0])
            if an and an["op"] == "LogicalNot" and len(an.get("args", [])) == 1:
                repl[nid] = int(an["args"][0])
        elif op == "Select" and len(args) == 3:
            c, t, f = args
            if t == f:
                repl[nid] = t
            else:
                cn = nodes_by_id.get(c)
                if cn and cn["op"] in ("ConstBool", "ConstU32", "ConstS32"):
                    is_true = (int(cn.get("imm_u32", 0)) != 0) or bool(cn.get("aux_bool", False))
                    repl[nid] = t if is_true else f
        elif op == "VecExtract" and len(args) == 1:
            an = nodes_by_id.get(args[0])
            idx = int(n.get("imm_u32", 0))
            if an and an["op"] in ("VecConstruct2", "VecConstruct3", "VecConstruct4") and idx < len(an.get("args", [])):
                repl[nid] = int(an["args"][idx])
        elif op == "TupleExtract" and len(args) == 1:
            an = nodes_by_id.get(args[0])
            idx = int(n.get("imm_u32", 0))
            if an and an["op"] == "Split64" and len(an.get("args", [])) == 1:
                inner = nodes_by_id.get(int(an["args"][0]))
                if inner and inner["op"] == "Pack64" and len(inner.get("args", [])) == 2 and idx in (0, 1):
                    repl[nid] = int(inner["args"][idx])

    if repl:
        _replace_all_uses(blocks_by_id, nodes_by_id, repl)
        stats.alg_simplified += len(repl)
        changed = True

    # 2. Dominator-tree scoped GVN (hash-consing)
    rpo = _compute_rpo(blocks_by_id, 0)
    idom = _compute_idom(blocks_by_id, rpo)
    dom_children: Dict[int, List[int]] = {b: [] for b in rpo}
    for b in rpo[1:]:
        p = idom.get(b)
        if p is not None and p != b:
            dom_children[p].append(b)

    gvn_repl: Dict[int, int] = {}

    def resolve_gvn(x: int) -> int:
        while x in gvn_repl:
            x = gvn_repl[x]
        return x

    def can_gvn(nid: int, n: Dict[str, Any]) -> bool:
        op = n["op"]
        if op in NEVER_GVN_OPS:
            return False
        if op in ("PsbLoad", "TexSample"):
            return nid in readonly_load_ids
        return True

    def gvn_key(n: Dict[str, Any]) -> Tuple[Any, ...]:
        op = n["op"]
        args = [resolve_gvn(int(a)) for a in n.get("args", [])]
        if op in COMMUTATIVE_OPS and len(args) == 2 and args[0] > args[1]:
            args = [args[1], args[0]]
        return (
            op,
            n.get("type", "U32"),
            tuple(args),
            int(n.get("imm_u32", 0)),
            int(n.get("imm_u32_2", 0)),
            int(n.get("imm_u64", 0)),
            int(n.get("aux_index", 0)),
            bool(n.get("aux_bool", False)),
            bool(n.get("aux_bool2", False)),
            n.get("aux_str", ""),
        )

    def walk_dom(bid: int, table: Dict[Tuple[Any, ...], int]) -> None:
        local_keys: List[Tuple[Any, ...]] = []
        blk = blocks_by_id[bid]
        for nid in blk.get("nodes", []):
            n = nodes_by_id.get(nid)
            if not n or not can_gvn(nid, n):
                continue
            key = gvn_key(n)
            prev = table.get(key)
            if prev is not None and prev != nid:
                gvn_repl[nid] = prev
            else:
                table[key] = nid
                local_keys.append(key)
        for ch in dom_children.get(bid, []):
            walk_dom(ch, table)
        for k in local_keys:
            table.pop(k, None)

    if rpo:
        walk_dom(rpo[0], {})

    if gvn_repl:
        _replace_all_uses(blocks_by_id, nodes_by_id, gvn_repl)
        stats.gvn_merged += len(gvn_repl)
        changed = True

    # 3. Dead Code Elimination (mark-and-sweep from side effects & branch_conds)
    live: Set[int] = set()
    worklist: List[int] = []

    def mark(nid: int) -> None:
        if nid > 0 and nid not in live and nid in nodes_by_id:
            live.add(nid)
            worklist.append(nid)

    for blk in blocks_by_id.values():
        bc = int(blk.get("branch_cond", 0))
        if bc > 0:
            mark(bc)
        for nid in blk.get("nodes", []):
            n = nodes_by_id.get(nid)
            if n and n["op"] in SIDE_EFFECT_OPS:
                mark(nid)

    while worklist:
        cur = worklist.pop()
        n = nodes_by_id[cur]
        for a in n.get("args", []):
            mark(int(a))
        for _, v in n.get("phi", []):
            mark(int(v))

    dead_nids = [nid for nid in list(nodes_by_id.keys()) if nid not in live]
    if dead_nids:
        for nid in dead_nids:
            del nodes_by_id[nid]
        for blk in blocks_by_id.values():
            blk["phis"] = [p for p in blk.get("phis", []) if p in live]
            blk["nodes"] = [x for x in blk.get("nodes", []) if x in live]
        stats.dce_removed += len(dead_nids)
        changed = True

    return changed


def canonicalize_ssa_module(
    ssa_json: Dict[str, Any],
    max_rounds: int = 8,
    max_arm_nodes: int = 128,
) -> Tuple[Dict[str, Any], Stratum1Stats]:
    """Run Stratum 1 canonicalization to fixed point on a serialized ValhallSsaModule."""
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
        c1 = _pass_prune_and_const_branches(blocks_by_id, nodes_by_id, stats)
        c2 = _pass_fold_branch_to_discard(blocks_by_id, nodes_by_id, stats)
        c3 = _pass_if_convert_pure_selections(blocks_by_id, nodes_by_id, stats, max_arm_nodes=max_arm_nodes)
        c4 = _pass_contract_trivial_chains(blocks_by_id, nodes_by_id, stats)
        c5, readonly_load_ids = _pass_detach_readonly_tokens(blocks_by_id, nodes_by_id, stats)
        c6 = _pass_normalize_gvn_dce(blocks_by_id, nodes_by_id, readonly_load_ids, stats)
        if not (c1 or c2 or c3 or c4 or c5 or c6):
            break

    # Final RPO ordering of blocks
    rpo = _compute_rpo(blocks_by_id, 0)
    ordered_blocks = [blocks_by_id[bid] for bid in rpo]
    ordered_nodes = [nodes_by_id[nid] for nid in sorted(nodes_by_id.keys())]

    stats.canon_blocks = len(ordered_blocks)
    stats.canon_nodes = len(ordered_nodes)

    return {"blocks": ordered_blocks, "nodes": ordered_nodes}, stats


def cfg_isomorphic(ssa_a: Dict[str, Any], ssa_b: Dict[str, Any]) -> bool:
    """
    Exact directed CFG isomorphism check up to block relabeling and conditional
    branch polarity flip (treating succs as an unordered pair for is_cond blocks).
    """
    ba = {int(b["id"]): b for b in ssa_a.get("blocks", [])}
    bb = {int(b["id"]): b for b in ssa_b.get("blocks", [])}
    if len(ba) != len(bb):
        return False
    if not ba:
        return True
    if 0 not in ba or 0 not in bb:
        return False

    sa = {bid: _block_succs(blk) for bid, blk in ba.items()}
    sb = {bid: _block_succs(blk) for bid, blk in bb.items()}
    pa = {bid: set(int(p) for p in blk.get("preds", [])) for bid, blk in ba.items()}
    pb = {bid: set(int(p) for p in blk.get("preds", [])) for bid, blk in bb.items()}

    # Degree-sequence fast reject
    sig_a = sorted((len(pa[k]), len(sa[k]), bool(ba[k].get("is_cond")), bool(ba[k].get("is_end"))) for k in ba)
    sig_b = sorted((len(pb[k]), len(sb[k]), bool(bb[k].get("is_cond")), bool(bb[k].get("is_end"))) for k in bb)
    if sig_a != sig_b:
        return False

    # Backtracking VF2-style matcher rooted at entry 0 -> 0 (bounded steps for safety)
    rpo_a = _compute_rpo(ba, 0)
    if len(rpo_a) != len(ba):
        return False
    mapping: Dict[int, int] = {0: 0}
    used_b: Set[int] = {0}
    steps = [0]
    MAX_STEPS = 50000

    def compatible(u: int, v: int) -> bool:
        if len(sa[u]) != len(sb[v]) or len(pa[u]) != len(pb[v]):
            return False
        if bool(ba[u].get("is_cond")) != bool(bb[v].get("is_cond")):
            return False
        if bool(ba[u].get("is_end")) != bool(bb[v].get("is_end")):
            return False
        # Check already-mapped predecessors and successors
        for p in pa[u]:
            if p in mapping and mapping[p] not in pb[v]:
                return False
        for s in sa[u]:
            if s in mapping and mapping[s] not in sb[v]:
                return False
        return True

    def backtrack(idx: int) -> bool:
        steps[0] += 1
        if steps[0] > MAX_STEPS:
            return False
        if idx == len(rpo_a):
            return True
        u = rpo_a[idx]
        # Candidate v must be a successor of at least one already-mapped pred of u
        cands: Optional[Set[int]] = None
        for p in pa[u]:
            if p in mapping:
                ps = set(sb[mapping[p]])
                cands = ps if cands is None else (cands & ps)
        if cands is None:
            cands = set(bb.keys())
        for v in cands:
            if v in used_b:
                continue
            if compatible(u, v):
                mapping[u] = v
                used_b.add(v)
                if backtrack(idx + 1):
                    return True
                del mapping[u]
                used_b.remove(v)
        return False

    return compatible(0, 0) and backtrack(1)

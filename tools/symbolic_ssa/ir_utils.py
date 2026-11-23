#!/usr/bin/env python3
"""Shared CFG/SSA graph utilities, Stratum 1 stats, and AbstractSsaPass base."""

from __future__ import annotations

import dataclasses
from typing import Any

from tools.symbolic_ssa.interpreter import SsaInterpreter

SIDE_EFFECT_OPS: set[str] = {
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

CONTROL_MARKER_OPS: set[str] = {
    "Branch",
    "BranchCond",
    "Return",
}

COMMUTATIVE_OPS: set[str] = {
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

NEVER_GVN_OPS: set[str] = SIDE_EFFECT_OPS | CONTROL_MARKER_OPS | {"Phi", "Nop"}


@dataclasses.dataclass
class Stratum1Stats:
    """Telemetry counters collected across Stratum 1 canonicalization rounds."""

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


def block_succs(blk: dict[str, Any]) -> list[int]:
    """Return outgoing successor block IDs in deterministic branch/fallthrough order."""
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


def compute_rpo(blocks_by_id: dict[int, dict[str, Any]], entry: int = 0) -> list[int]:
    """Compute Reverse Post-Order traversal of reachable blocks from entry."""
    if entry not in blocks_by_id:
        return []
    visited: set[int] = set()
    postorder: list[int] = []
    stack: list[tuple[int, int]] = [(entry, 0)]
    while stack:
        u, step = stack.pop()
        if step == 0:
            if u in visited:
                continue
            visited.add(u)
            stack.append((u, 1))
            for s in reversed(block_succs(blocks_by_id[u])):
                if s in blocks_by_id and s not in visited:
                    stack.append((s, 0))
        else:
            postorder.append(u)
    return list(reversed(postorder))


def compute_idom(blocks_by_id: dict[int, dict[str, Any]], rpo: list[int]) -> dict[int, int]:
    """Compute immediate dominators via Cooper-Harvey-Kennedy iterative algorithm."""
    if not rpo:
        return {}
    entry = rpo[0]
    rpo_idx = {b: i for i, b in enumerate(rpo)}
    idom: dict[int, int] = {entry: entry}

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


def rebuild_preds(blocks_by_id: dict[int, dict[str, Any]], nodes_by_id: dict[int, dict[str, Any]]) -> None:
    """Recompute block predecessor lists and prune dead Phi incoming edges."""
    new_preds: dict[int, list[int]] = {bid: [] for bid in blocks_by_id}
    for bid, blk in blocks_by_id.items():
        for s in block_succs(blk):
            if s in new_preds and bid not in new_preds[s]:
                new_preds[s].append(bid)
    for bid, blk in blocks_by_id.items():
        blk["preds"] = new_preds[bid]
        pred_set = set(new_preds[bid])
        for pid in blk.get("phis", []):
            pn = nodes_by_id.get(pid)
            if pn and "phi" in pn:
                pn["phi"] = [[int(p), int(v)] for p, v in pn["phi"] if int(p) in pred_set]


def replace_all_uses(
    blocks_by_id: dict[int, dict[str, Any]],
    nodes_by_id: dict[int, dict[str, Any]],
    repl: dict[int, int],
) -> None:
    """Transitvely rewrite all node argument, Phi incoming, and branch_cond references."""
    if not repl:
        return

    def resolve(x: int) -> int:
        seen: set[int] = set()
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


def collapse_trivial_phis(
    blocks_by_id: dict[int, dict[str, Any]],
    nodes_by_id: dict[int, dict[str, Any]],
) -> int:
    """Eliminate Phi nodes with exactly one unique non-self incoming value."""
    repl: dict[int, int] = {}
    for blk in blocks_by_id.values():
        keep_phis: list[int] = []
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
        replace_all_uses(blocks_by_id, nodes_by_id, repl)
    return len(repl)


def is_block_pure(blk: dict[str, Any], nodes_by_id: dict[int, dict[str, Any]]) -> bool:
    """Return True iff the block contains no side-effecting instructions."""
    for nid in blk.get("nodes", []):
        n = nodes_by_id.get(nid)
        if n and n["op"] in SIDE_EFFECT_OPS:
            return False
    return True


def next_node_id(nodes_by_id: dict[int, dict[str, Any]]) -> int:
    """Allocate the next unused integer SSA node ID."""
    return (max(nodes_by_id.keys()) + 1) if nodes_by_id else 1


def insert_before_terminator(
    blk: dict[str, Any],
    nodes_by_id: dict[int, dict[str, Any]],
    new_ids: list[int],
) -> None:
    """Insert node IDs immediately before trailing control marker nodes."""
    nodes = blk["nodes"]
    idx = len(nodes)
    while idx > 0 and nodes_by_id.get(nodes[idx - 1], {}).get("op") in CONTROL_MARKER_OPS:
        idx -= 1
    blk["nodes"] = nodes[:idx] + new_ids + nodes[idx:]


def extract_non_marker_nodes(blk: dict[str, Any], nodes_by_id: dict[int, dict[str, Any]]) -> list[int]:
    """Return all node IDs in a block excluding trailing control markers."""
    return [nid for nid in blk.get("nodes", []) if nodes_by_id.get(nid, {}).get("op") not in CONTROL_MARKER_OPS]


def cfg_isomorphic(ssa_a: dict[str, Any], ssa_b: dict[str, Any]) -> bool:
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

    sa = {bid: block_succs(blk) for bid, blk in ba.items()}
    sb = {bid: block_succs(blk) for bid, blk in bb.items()}
    pa = {bid: set(int(p) for p in blk.get("preds", [])) for bid, blk in ba.items()}
    pb = {bid: set(int(p) for p in blk.get("preds", [])) for bid, blk in bb.items()}

    sig_a = sorted((len(pa[k]), len(sa[k]), bool(ba[k].get("is_cond")), bool(ba[k].get("is_end"))) for k in ba)
    sig_b = sorted((len(pb[k]), len(sb[k]), bool(bb[k].get("is_cond")), bool(bb[k].get("is_end"))) for k in bb)
    if sig_a != sig_b:
        return False

    rpo_a = compute_rpo(ba, 0)
    if len(rpo_a) != len(ba):
        return False
    mapping: dict[int, int] = {0: 0}
    used_b: set[int] = {0}
    steps = [0]
    max_steps = 50000

    def compatible(u: int, v: int) -> bool:
        if len(sa[u]) != len(sb[v]) or len(pa[u]) != len(pb[v]):
            return False
        if bool(ba[u].get("is_cond")) != bool(bb[v].get("is_cond")):
            return False
        if bool(ba[u].get("is_end")) != bool(bb[v].get("is_end")):
            return False
        for p in pa[u]:
            if p in mapping and mapping[p] not in pb[v]:
                return False
        for s in sa[u]:
            if s in mapping and mapping[s] not in sb[v]:
                return False
        return True

    def backtrack(idx: int) -> bool:
        steps[0] += 1
        if steps[0] > max_steps:
            return False
        if idx == len(rpo_a):
            return True
        u = rpo_a[idx]
        cands: set[int] | None = None
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


class AbstractSsaPass(SsaInterpreter):
    """
    Base class for abstract-interpretation passes over ValhallSsaModule.

    Defaults constants and unhandled domain hooks to `None` (lattice Top) so
    subclasses override only the transfer functions relevant to their analysis.
    """

    def __init__(self, blocks_by_id: dict[int, dict[str, Any]], nodes_by_id: dict[int, dict[str, Any]]):
        self.blocks_by_id = blocks_by_id
        self.nodes_by_id = nodes_by_id
        self.abs_vals: dict[int, Any] = {}

    def default_value(self) -> Any:
        return None

    def const_u32(self, imm: int) -> Any:
        return None

    def const_u64(self, imm: int) -> Any:
        return None

    def const_bool(self, val: bool) -> Any:
        return None

    def icmp(self, op: str, a: Any, b: Any) -> Any:
        return None

    def logical_and(self, a: Any, b: Any) -> Any:
        return None

    def logical_or(self, a: Any, b: Any) -> Any:
        return None

    def logical_not(self, a: Any) -> Any:
        return None

    def select(self, cond: Any, t_val: Any, f_val: Any) -> Any:
        return None

    def fp_is_nan(self, x: Any) -> Any:
        return None

    def fp_is_inf(self, x: Any) -> Any:
        return None

    def fp_cmp(self, op: str, a: Any, b: Any) -> Any:
        return None

    def eval_phi(self, node: dict[str, Any]) -> Any:
        return None

    def run_rpo(self) -> dict[int, Any]:
        """Evaluate every reachable Phi and node in RPO order."""
        rpo = compute_rpo(self.blocks_by_id, 0)
        for bid in rpo:
            blk = self.blocks_by_id[bid]
            for pid in blk.get("phis", []):
                pn = self.nodes_by_id.get(pid)
                if pn is not None:
                    self.abs_vals[pid] = self.eval_phi(pn)
            for nid in blk.get("nodes", []):
                n = self.nodes_by_id.get(nid)
                if n is not None:
                    self.abs_vals[nid] = self.eval_node(n, self.abs_vals, None)
        return self.abs_vals

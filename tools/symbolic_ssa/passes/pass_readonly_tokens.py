#!/usr/bin/env python3
"""
Pass 1.5: Read-Only Token Detachment.

Uses BufferProvenanceInterpreter (an SsaInterpreter abstract interpretation over
the flat buffer-key provenance lattice) to propagate base buffer provenance
through arbitrary additive/subtractive/masking 64-bit address arithmetic, copies,
bitcasts, selects, and Phis without a fixed depth bound. Provably read-only
PsbLoad and TexSample operations are detached from the linear Token chain and
rebound to root_tok so subsequent GVN can unify identical loads.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set, Tuple

from tools.symbolic_ssa.passes.ir_utils import (
    AbstractSsaPass,
    Stratum1Stats,
    next_node_id,
    replace_all_uses,
)


class BufferProvenanceInterpreter(AbstractSsaPass):
    """Flat buffer-provenance lattice (None = Unknown/Top, str = Unique Buffer Key)."""

    def load_sideband_u64(self, imm_u32: int, aux_str: str) -> Any:
        return aux_str or f"sideband_{imm_u32}"

    def cast_value(self, op: str, val: Any, val_type: str) -> Any:
        return val if isinstance(val, str) else None

    def int64_binary(self, op: str, a: Any, b: Any) -> Any:
        if op in ("IAdd64", "BitAnd64"):
            if isinstance(a, str) and b is None:
                return a
            if isinstance(b, str) and a is None:
                return b
            if isinstance(a, str) and a == b:
                return a
        elif op == "ISub64":
            # ptr - offset preserves provenance; ptr - ptr yields a scalar distance.
            if isinstance(a, str) and b is None:
                return a
        return None

    def select(self, cond: Any, t_val: Any, f_val: Any) -> Any:
        if isinstance(t_val, str) and t_val == f_val:
            return t_val
        return None

    def eval_phi(self, node: Dict[str, Any]) -> Any:
        keys: Set[str] = set()
        for _, v in node.get("phi", []):
            vid = int(v)
            if vid == int(node["id"]):
                continue
            kv = self.abs_vals.get(vid)
            if not isinstance(kv, str):
                return None
            keys.add(kv)
        return next(iter(keys)) if len(keys) == 1 else None


def _node_buf_key(node: Dict[str, Any], prov_env: Dict[int, Any]) -> Optional[str]:
    astr = node.get("aux_str", "")
    if astr.startswith("sym:"):
        return astr[4:].split("[", 1)[0]
    if astr.startswith(("ubo:", "ssbo:", "push_const", "img:", "workgroup:")):
        return astr
    if astr:
        return astr
    args = node.get("args", [])
    if not args:
        return None
    kv = prov_env.get(int(args[0]))
    return kv if isinstance(kv, str) else None


def pass_readonly_tokens(
    blocks_by_id: Dict[int, Dict[str, Any]],
    nodes_by_id: Dict[int, Dict[str, Any]],
    stats: Stratum1Stats,
) -> Tuple[bool, Set[int]]:
    """Detach Token chains from provably read-only PsbLoad and TexSample ops."""
    prov_env = BufferProvenanceInterpreter(blocks_by_id, nodes_by_id).run_rpo()

    mutated_bkeys: Set[str] = set()
    has_unknown_store = False
    mutated_imgs: Set[str] = set()

    for n in nodes_by_id.values():
        op = n["op"]
        if op in ("PsbStore", "AtomicRmw", "AtomicExchange", "AtomicCmpXchg"):
            bk = _node_buf_key(n, prov_env)
            if bk is None:
                has_unknown_store = True
            else:
                mutated_bkeys.add(bk)
        elif op == "StoreTexel":
            mutated_imgs.add(n.get("aux_str", ""))

    root_tok = 0
    if 0 in blocks_by_id:
        for nid in blocks_by_id[0].get("nodes", []):
            n = nodes_by_id.get(nid)
            if n and n["op"] == "Nop" and n.get("type") == "Token":
                root_tok = nid
                break
        if root_tok == 0:
            root_tok = next_node_id(nodes_by_id)
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
            bk = _node_buf_key(n, prov_env)
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
        replace_all_uses(blocks_by_id, nodes_by_id, tok_repl)
        stats.readonly_tokens_detached += len(tok_repl)
        changed = True

    for lid in readonly_load_ids:
        ln = nodes_by_id[lid]
        if len(ln.get("args", [])) >= 2 and int(ln["args"][-1]) != root_tok:
            ln["args"][-1] = root_tok
            stats.readonly_tokens_detached += 1
            changed = True

    return changed, readonly_load_ids

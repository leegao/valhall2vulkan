"""
Side-by-Side Concrete Backward-Slice Differ (`--slice-diff <depth>`).

Given a concrete divergence witness `(vec_idx, overrides, output_name)`, walks
`depth` hops backward from the diverging sink node(s) on BOTH `direct_ssa` and
`valhall_ssa`, aligns nodes by hop and shared concrete values, and highlights
the first node where concrete values diverge while inputs still agree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from tools.coverage.shrinker import DivergenceReport
from tools.symbolic_ssa import ConcreteSsaInterpreter
from tools.symbolic_ssa.env import ConcreteShaderEnv, _u32_to_f32

_SINK_OPS = {
    "StoreFragColor",
    "StoreBuiltinOutput",
    "StoreVertVarying",
    "PsbStore",
    "StoreTexel",
    "Kill",
    "AtomicRmw",
    "AtomicExchange",
    "AtomicCmpXchg",
}


@dataclass(frozen=True)
class SliceNodeInfo:
    """Single node in a concrete backward slice."""

    hop: int
    node_id: int
    op: str
    val: Any
    val_fmt: str
    args: tuple[int, ...]
    pc_disasm: str = ""


def _fmt_val(val: Any) -> str:
    if val is None:
        return "None"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, int):
        u = val & 0xFFFFFFFF
        try:
            f = _u32_to_f32(u)
            if math.isnan(f):
                fp_s = "nan"
            elif math.isinf(f):
                fp_s = "+inf" if f > 0 else "-inf"
            else:
                fp_s = f"{f:.4g}"
            return f"0x{u:08x}({fp_s})"
        except (ValueError, OverflowError):
            return f"0x{u:08x}"
    s = str(val)
    return s if len(s) <= 28 else s[:25] + "..."


def _collect_backward_slice(
    ssa_mod: dict[str, Any],
    vals: dict[int, Any],
    *,
    depth: int,
    pc_disasm: dict[str, str] | None = None,
) -> list[SliceNodeInfo]:
    """BFS `depth` hops backward from sink nodes in `ssa_mod`."""
    nodes = {int(n["id"]): n for n in ssa_mod.get("nodes", [])}
    sinks = [int(n["id"]) for n in ssa_mod.get("nodes", []) if n.get("op") in _SINK_OPS]
    if not sinks:
        return []

    visited: dict[int, int] = {}
    queue: list[tuple[int, int]] = [(sid, 0) for sid in sinks]
    for sid in sinks:
        visited[sid] = 0

    ordered: list[SliceNodeInfo] = []
    while queue:
        nid, hop = queue.pop(0)
        n = nodes.get(nid)
        if n is None:
            continue
        args = tuple(int(a) for a in n.get("args", []))
        dis = ""
        if pc_disasm is not None and "orig_pc" in n:
            pc = int(n["orig_pc"])
            dis = f"0x{pc:x}:{pc_disasm.get(str(pc), '')}"
        v = vals.get(nid)
        ordered.append(
            SliceNodeInfo(
                hop=hop,
                node_id=nid,
                op=str(n.get("op", "?")),
                val=v,
                val_fmt=_fmt_val(v),
                args=args,
                pc_disasm=dis,
            )
        )
        if hop < depth:
            for a in args:
                if a not in visited and a in nodes:
                    visited[a] = hop + 1
                    queue.append((a, hop + 1))
    return ordered


def format_slice_diff(
    bundle: dict[str, Any],
    direct_ssa: dict[str, Any],
    valhall_ssa: dict[str, Any],
    rep: DivergenceReport,
    *,
    depth: int = 6,
    fp_range: str = "unconstrained",
) -> list[str]:
    """Return formatted lines comparing N-hop Direct vs Valhall backward slices."""
    cenv = ConcreteShaderEnv(bundle, rep.vec_idx, overrides=rep.shrunk_overrides, fp_range=fp_range)
    d_vals, _ = ConcreteSsaInterpreter(direct_ssa, is_valhall_side=False).execute_one(cenv)
    v_vals, _ = ConcreteSsaInterpreter(valhall_ssa, is_valhall_side=True).execute_one(cenv)
    pc_disasm: dict[str, str] = bundle.get("mbs2_context", {}).get("pc_disasm", {})

    d_slice = _collect_backward_slice(direct_ssa, d_vals, depth=depth)
    v_slice = _collect_backward_slice(valhall_ssa, v_vals, depth=depth, pc_disasm=pc_disasm)

    d_val_set = {s.val for s in d_slice if s.val is not None}
    v_val_set = {s.val for s in v_slice if s.val is not None}
    shared_vals = d_val_set & v_val_set

    lines: list[str] = [
        f"  [SLICE-DIFF depth={depth}] vec={rep.vec_idx} out={rep.output_name} "
        f"(direct={len(d_slice)} nodes, valhall={len(v_slice)} nodes, shared_vals={len(shared_vals)})"
    ]

    max_hop = max([s.hop for s in d_slice] + [s.hop for s in v_slice] + [0])
    for hop in range(max_hop + 1):
        ds = [s for s in d_slice if s.hop == hop]
        vs = [s for s in v_slice if s.hop == hop]
        if not ds and not vs:
            continue
        # Sort so shared values appear first within each hop for easy visual alignment
        ds.sort(key=lambda s: (0 if s.val in shared_vals else 1, s.node_id))
        vs.sort(key=lambda s: (0 if s.val in shared_vals else 1, s.node_id))
        rows = max(len(ds), len(vs))
        for i in range(min(rows, 8)):
            dn = ds[i] if i < len(ds) else None
            vn = vs[i] if i < len(vs) else None
            d_col = f"%{dn.node_id:<3d} {dn.op:<18s} = {dn.val_fmt:<20s}" if dn else f"{'':46s}"
            v_col = f"%{vn.node_id:<3d} {vn.op:<18s} = {vn.val_fmt:<20s}" if vn else f"{'':46s}"
            dis = f" [{vn.pc_disasm}]" if vn and vn.pc_disasm else ""

            marker = ""
            if dn and vn and dn.val is not None and vn.val is not None:
                if dn.val == vn.val:
                    marker = " == MATCH"
                else:
                    # Highlight if Valhall inputs are in shared_vals while output diverged!
                    v_arg_vals = [v_vals.get(a) for a in vn.args if v_vals.get(a) is not None]
                    if v_arg_vals and all(av in shared_vals for av in v_arg_vals):
                        marker = " << FIRST-DIVERGE"
                    else:
                        marker = " != DIFF"
            elif vn and vn.val in shared_vals:
                marker = " == SHARED"
            elif dn and dn.val in shared_vals:
                marker = " == SHARED"

            lines.append(f"    h{hop} | D: {d_col} | V: {v_col}{marker}{dis}")
    return lines

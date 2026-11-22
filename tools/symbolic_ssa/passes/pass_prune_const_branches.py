#!/usr/bin/env python3
"""
Pass 1.1: Reachability & Constant/Degenerate Branch Pruning.

Uses `ConstantPropDomain(SsaDomain)` over a tagged-union flat lattice
(`int | float | bool | uniform_ptr` or `TOP=None`) executed via chaotic
fixpoint iteration (`SsaInterpreter.run_fixpoint`), folding constant/degenerate
conditional branches, pruning unreachable blocks, and collapsing trivial Phis.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Literal, Optional

from tools.symbolic_ssa.domains import ConcreteDomain, SsaDomain
from tools.symbolic_ssa.env import MASK32
from tools.symbolic_ssa.interpreter import SsaInterpreter
from tools.symbolic_ssa.passes.ir_utils import (
    Stratum1Stats,
    collapse_trivial_phis,
    compute_rpo,
    rebuild_preds,
)

ConstKind = Literal["int", "float", "bool", "uniform_ptr"]


@dataclasses.dataclass(frozen=True)
class ConstVal:
    """Tagged-union constant lattice element (`None` represents lattice Top)."""

    kind: ConstKind
    value: Any


def join_const(a: Optional[ConstVal], b: Optional[ConstVal]) -> Optional[ConstVal]:
    """Flat lattice join over tagged ConstVal elements."""
    if a is None or b is None:
        return None
    return a if a == b else None


class ConstantPropDomain(SsaDomain):
    """
    Real SsaDomain over tagged-union constants (`int`, `float`, `bool`, `uniform_ptr`).

    Delegates bit-exact scalar operations to `ConcreteDomain` whenever all
    operands are known constants, and propagates `(buf_key, byte_offset)` for
    architecturally read-only `ubo:*` / `push_const` uniform pointers.
    """

    def __init__(self) -> None:
        self._concrete = ConcreteDomain()

    def default_value(self) -> Optional[ConstVal]:
        return None

    def const_u32(self, imm: int) -> ConstVal:
        return ConstVal("int", imm & MASK32)

    def const_u64(self, imm: int) -> ConstVal:
        return ConstVal("int", imm)

    def const_bool(self, val: bool) -> ConstVal:
        return ConstVal("bool", bool(val))

    def cast_value(self, op: str, val: Any, val_type: str) -> Optional[ConstVal]:
        if not isinstance(val, ConstVal):
            return None
        if op == "Copy":
            return val
        if op == "Bitcast" and val.kind in ("int", "float"):
            if val_type == "F32":
                return ConstVal("float", int(val.value) & MASK32)
            if val_type in ("U32", "S32"):
                return ConstVal("int", int(val.value) & MASK32)
        return None

    def logical_not(self, a: Any) -> Optional[ConstVal]:
        if isinstance(a, ConstVal) and a.kind == "bool":
            return ConstVal("bool", not a.value)
        return None

    def logical_and(self, a: Any, b: Any) -> Optional[ConstVal]:
        if isinstance(a, ConstVal) and a.kind == "bool" and a.value is False:
            return ConstVal("bool", False)
        if isinstance(b, ConstVal) and b.kind == "bool" and b.value is False:
            return ConstVal("bool", False)
        if isinstance(a, ConstVal) and isinstance(b, ConstVal) and a.kind == "bool" and b.kind == "bool":
            return ConstVal("bool", a.value and b.value)
        return None

    def logical_or(self, a: Any, b: Any) -> Optional[ConstVal]:
        if isinstance(a, ConstVal) and a.kind == "bool" and a.value is True:
            return ConstVal("bool", True)
        if isinstance(b, ConstVal) and b.kind == "bool" and b.value is True:
            return ConstVal("bool", True)
        if isinstance(a, ConstVal) and isinstance(b, ConstVal) and a.kind == "bool" and b.kind == "bool":
            return ConstVal("bool", a.value or b.value)
        return None

    def int32_unary(self, op: str, x: Any) -> Optional[ConstVal]:
        if isinstance(x, ConstVal) and x.kind == "int":
            return ConstVal("int", self._concrete.int32_unary(op, x.value))
        return None

    def int32_binary(self, op: str, a: Any, b: Any) -> Optional[ConstVal]:
        if isinstance(a, ConstVal) and isinstance(b, ConstVal) and a.kind == "int" and b.kind == "int":
            return ConstVal("int", self._concrete.int32_binary(op, a.value, b.value))
        return None

    def int64_binary(self, op: str, a: Any, b: Any) -> Optional[ConstVal]:
        if op == "IAdd64":
            if isinstance(a, ConstVal) and a.kind == "uniform_ptr" and isinstance(b, ConstVal) and b.kind == "int":
                bkey, off = a.value
                return ConstVal("uniform_ptr", (bkey, off + int(b.value)))
            if isinstance(b, ConstVal) and b.kind == "uniform_ptr" and isinstance(a, ConstVal) and a.kind == "int":
                bkey, off = b.value
                return ConstVal("uniform_ptr", (bkey, off + int(a.value)))
        if isinstance(a, ConstVal) and isinstance(b, ConstVal) and a.kind == "int" and b.kind == "int":
            return ConstVal("int", self._concrete.int64_binary(op, a.value, b.value))
        return None

    def icmp(self, op: str, a: Any, b: Any) -> Optional[ConstVal]:
        if isinstance(a, ConstVal) and isinstance(b, ConstVal) and a.kind == "int" and b.kind == "int":
            return ConstVal("bool", self._concrete.icmp(op, a.value, b.value))
        return None

    def fp_unary(self, op: str, x: Any) -> Optional[ConstVal]:
        if isinstance(x, ConstVal) and x.kind in ("float", "int"):
            return ConstVal("float", self._concrete.fp_unary(op, x.value))
        return None

    def fp_binary(self, op: str, a: Any, b: Any) -> Optional[ConstVal]:
        if isinstance(a, ConstVal) and isinstance(b, ConstVal) and a.kind in ("float", "int") and b.kind in ("float", "int"):
            return ConstVal("float", self._concrete.fp_binary(op, a.value, b.value))
        return None

    def fp_cmp(self, op: str, a: Any, b: Any) -> Optional[ConstVal]:
        if isinstance(a, ConstVal) and isinstance(b, ConstVal) and a.kind in ("float", "int") and b.kind in ("float", "int"):
            return ConstVal("bool", self._concrete.fp_cmp(op, a.value, b.value))
        return None

    def select(self, cond: Any, t_val: Any, f_val: Any) -> Optional[ConstVal]:
        if isinstance(cond, ConstVal) and cond.kind == "bool":
            return t_val if cond.value else f_val
        return join_const(t_val, f_val)


class ConstantFoldInterpreter(SsaInterpreter):
    """Fixpoint constant-propagation interpreter backed by ConstantPropDomain."""

    def __init__(self, blocks_by_id: Dict[int, Dict[str, Any]], nodes_by_id: Dict[int, Dict[str, Any]]):
        self.blocks_by_id = blocks_by_id
        self.nodes_by_id = nodes_by_id
        self.domain = ConstantPropDomain()
        self.values: Dict[int, Optional[ConstVal]] = {}

    def load_sideband_u64(self, imm_u32: int, aux_str: str) -> Optional[ConstVal]:
        if aux_str and (aux_str.startswith("ubo:") or aux_str == "push_const"):
            return ConstVal("uniform_ptr", (aux_str, 0))
        return None

    def _eval_phi(self, node: Dict[str, Any]) -> Optional[ConstVal]:
        nid = int(node["id"])
        acc: Optional[ConstVal] = None
        seen_any = False
        for _, v in node.get("phi", []):
            vid = int(v)
            if vid == nid or vid not in self.values:
                continue
            cand = self.values[vid]
            if not seen_any:
                acc = cand
                seen_any = True
            else:
                acc = join_const(acc, cand)
            if acc is None:
                return None
        return acc if seen_any else None

    def run(self) -> Dict[int, Optional[ConstVal]]:
        return self.run_fixpoint(
            self.blocks_by_id,
            self.nodes_by_id,
            self.values,
            eval_phi=self._eval_phi,
            join_fn=join_const,
        )


def pass_prune_const_branches(
    blocks_by_id: Dict[int, Dict[str, Any]],
    nodes_by_id: Dict[int, Dict[str, Any]],
    stats: Stratum1Stats,
) -> bool:
    """Fold constant/degenerate conditional branches, prune unreachable blocks, and collapse trivial Phis."""
    changed = False
    const_env = ConstantFoldInterpreter(blocks_by_id, nodes_by_id).run()

    for blk in blocks_by_id.values():
        if not blk.get("is_cond"):
            continue
        bt = int(blk.get("branch_target", -1))
        ft = int(blk.get("fallthrough", -1))
        bc = int(blk.get("branch_cond", 0))
        taken: Optional[int] = None

        if bt == ft and bt >= 0:
            taken = bt
        else:
            cval = const_env.get(bc)
            if isinstance(cval, ConstVal) and cval.kind == "bool":
                taken = bt if cval.value else ft
            elif isinstance(cval, ConstVal) and cval.kind == "int":
                taken = bt if (cval.value != 0) else ft

        if taken is not None and taken >= 0:
            blk["is_cond"] = False
            blk["is_uncond"] = True
            blk["branch_target"] = taken
            blk["fallthrough"] = -1
            blk["branch_cond"] = 0
            stats.const_branches_folded += 1
            changed = True

    reach_set = set(compute_rpo(blocks_by_id, 0))
    for bid in [b for b in list(blocks_by_id.keys()) if b not in reach_set]:
        blk = blocks_by_id.pop(bid)
        for nid in blk.get("phis", []) + blk.get("nodes", []):
            nodes_by_id.pop(nid, None)
        stats.unreachable_blocks_removed += 1
        changed = True

    rebuild_preds(blocks_by_id, nodes_by_id)
    if collapse_trivial_phis(blocks_by_id, nodes_by_id) > 0:
        changed = True
    return changed

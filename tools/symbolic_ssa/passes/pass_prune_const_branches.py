#!/usr/bin/env python3
"""
Pass 1.1: Reachability & Constant/Degenerate Branch Pruning.

Uses ConstantFoldInterpreter (an SsaInterpreter abstract interpretation over
the flat constant lattice) so branches conditioned on constant expressions
fold cleanly before unreachable blocks and trivial Phis are pruned.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from tools.symbolic_ssa.env import MASK32, _s32
from tools.symbolic_ssa.passes.ir_utils import (
    AbstractSsaPass,
    Stratum1Stats,
    collapse_trivial_phis,
    compute_rpo,
    rebuild_preds,
)


class ConstantFoldInterpreter(AbstractSsaPass):
    """Flat constant-propagation lattice (None = Top, int/bool = Known Constant)."""

    def const_u32(self, imm: int) -> Any:
        return imm & MASK32

    def const_bool(self, val: bool) -> Any:
        return bool(val)

    def cast_value(self, op: str, val: Any, val_type: str) -> Any:
        if op == "Copy":
            return val
        if op == "Bitcast" and isinstance(val, int) and not isinstance(val, bool):
            return val
        return None

    def logical_not(self, a: Any) -> Any:
        return (not a) if isinstance(a, bool) else None

    def logical_and(self, a: Any, b: Any) -> Any:
        if a is False or b is False:
            return False
        if a is True and b is True:
            return True
        return None

    def logical_or(self, a: Any, b: Any) -> Any:
        if a is True or b is True:
            return True
        if a is False and b is False:
            return False
        return None

    def int32_binary(self, op: str, a: Any, b: Any) -> Any:
        if not (isinstance(a, int) and isinstance(b, int) and not isinstance(a, bool) and not isinstance(b, bool)):
            return None
        if op == "IAdd":
            return (a + b) & MASK32
        if op == "ISub":
            return (a - b) & MASK32
        if op == "IMul":
            return (a * b) & MASK32
        if op == "BitAnd":
            return (a & b) & MASK32
        if op == "BitOr":
            return (a | b) & MASK32
        if op == "BitXor":
            return (a ^ b) & MASK32
        if op == "Shl":
            return (a << (b & 31)) & MASK32
        if op == "ShrU":
            return (a & MASK32) >> (b & 31)
        if op == "ShrS":
            return (_s32(a) >> (b & 31)) & MASK32
        return None

    def icmp(self, op: str, a: Any, b: Any) -> Any:
        if not (isinstance(a, int) and isinstance(b, int) and not isinstance(a, bool) and not isinstance(b, bool)):
            return None
        if op == "ICmpEq":
            return (a & MASK32) == (b & MASK32)
        if op == "ICmpNe":
            return (a & MASK32) != (b & MASK32)
        if op == "ICmpULt":
            return (a & MASK32) < (b & MASK32)
        if op == "ICmpULe":
            return (a & MASK32) <= (b & MASK32)
        if op == "ICmpUGt":
            return (a & MASK32) > (b & MASK32)
        if op == "ICmpUGe":
            return (a & MASK32) >= (b & MASK32)
        if op == "ICmpSLt":
            return _s32(a) < _s32(b)
        if op == "ICmpSLe":
            return _s32(a) <= _s32(b)
        if op == "ICmpSGt":
            return _s32(a) > _s32(b)
        if op == "ICmpSGe":
            return _s32(a) >= _s32(b)
        return None

    def select(self, cond: Any, t_val: Any, f_val: Any) -> Any:
        if isinstance(cond, bool):
            return t_val if cond else f_val
        if t_val is not None and t_val == f_val:
            return t_val
        return None


def pass_prune_const_branches(
    blocks_by_id: Dict[int, Dict[str, Any]],
    nodes_by_id: Dict[int, Dict[str, Any]],
    stats: Stratum1Stats,
) -> bool:
    """Fold constant/degenerate conditional branches, prune unreachable blocks, and collapse trivial Phis."""
    changed = False
    const_env = ConstantFoldInterpreter(blocks_by_id, nodes_by_id).run_rpo()

    for blk in blocks_by_id.values():
        if blk.get("is_cond"):
            bt = int(blk.get("branch_target", -1))
            ft = int(blk.get("fallthrough", -1))
            bc = int(blk.get("branch_cond", 0))
            taken: Optional[int] = None
            if bt == ft and bt >= 0:
                taken = bt
            else:
                cval = const_env.get(bc)
                if isinstance(cval, bool):
                    taken = bt if cval else ft
                elif isinstance(cval, int):
                    taken = bt if (cval != 0) else ft
                else:
                    cn = nodes_by_id.get(bc)
                    if cn is not None and cn["op"] in ("ConstBool", "ConstU32", "ConstS32"):
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

    rpo = compute_rpo(blocks_by_id, 0)
    reach_set = set(rpo)
    dead_blks = [bid for bid in list(blocks_by_id.keys()) if bid not in reach_set]
    for bid in dead_blks:
        blk = blocks_by_id.pop(bid)
        for nid in blk.get("phis", []) + blk.get("nodes", []):
            nodes_by_id.pop(nid, None)
        stats.unreachable_blocks_removed += 1
        changed = True

    rebuild_preds(blocks_by_id, nodes_by_id)
    if collapse_trivial_phis(blocks_by_id, nodes_by_id) > 0:
        changed = True
    return changed

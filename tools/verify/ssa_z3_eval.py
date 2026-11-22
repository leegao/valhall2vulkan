#!/usr/bin/env python3
"""
Direct Z3 Symbolic Evaluator for `SymbolicSsaModule` (Instruction-Level Lowering).

Delegates all SSA opcode semantics to the unified `tools.symbolic_ssa`
`SsaInterpreter` + `SymbolicDomain` infrastructure so instruction-level miter
verification (`all_paths_miter.py`) shares the exact same denotational semantics
as whole-shader verification and can swap domains seamlessly.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import z3

from tools.verify.clang_ast_evaluator import SymbolicSsaModule, SymbolicSsaNode
from tools.verify.panfrost_z3_model import (
    BV32,
    BV64,
    ValhallZ3State,
    _hw_uf,
    bv32,
    bv64,
)

if TYPE_CHECKING:
    from tools.symbolic_ssa.domains import SsaDomain

Z3Val = Union[z3.ExprRef, Tuple[z3.ExprRef, ...]]


@dataclasses.dataclass
class SsaZ3EvalResult:
    """Symbolic evaluation output of a SymbolicSsaModule."""

    values: Dict[int, Z3Val]
    reg_writes: Dict[int, z3.BitVecRef]
    final_state: ValhallZ3State
    branch_cond: Optional[z3.BoolRef]
    trace_notes: List[str] = dataclasses.field(default_factory=list)


def _vec_len(val_type: str) -> int:
    if val_type.startswith("V2"): return 2
    if val_type.startswith("V3"): return 3
    if val_type.startswith("V4"): return 4
    return 1


def _build_inst_interpreter(
    st: ValhallZ3State,
    raw_srcs: Optional[Sequence[z3.BitVecRef]],
    dst_old: Optional[z3.BitVecRef],
    dest_reg: int,
    domain: Optional["SsaDomain"],
):
    from tools.symbolic_ssa.domains import SymbolicDomain
    from tools.symbolic_ssa.interpreter import SsaInterpreter

    class _InstLevelSsaInterpreter(SsaInterpreter):
        def __init__(self) -> None:
            self.st = st
            self.raw_srcs = raw_srcs
            self.dst_old = dst_old
            self.dest_reg = dest_reg
            self.domain = domain if domain is not None else SymbolicDomain(fp_mode="exact")
            self.notes: List[str] = []

        def cast_value(self, op: str, val: Any, val_type: str) -> Any:
            if val_type == "U64" and isinstance(val, z3.BitVecRef) and val.size() == 32:
                return z3.ZeroExt(32, val)
            if val_type in ("U32", "S32", "F32") and isinstance(val, z3.BitVecRef) and val.size() == 64:
                return z3.Extract(31, 0, val)
            return val

        def load_fau(self, word_idx: int) -> z3.BitVecRef:
            return z3.Select(self.st.fau, z3.BitVecVal(word_idx & 63, 8))

        def load_sideband_u32(self, imm_u32: int, imm_u32_2: int) -> z3.BitVecRef:
            return _hw_uf(f"sideband_u32_{imm_u32}", BV32, BV32)(bv32(imm_u32_2))

        def load_sideband_u64(self, imm_u32: int, aux_str: str) -> z3.BitVecRef:
            return _hw_uf(f"sideband_u64_{imm_u32}", BV32, BV64)(bv32(imm_u32))

        def load_builtin(self, builtin_id: int, val_type: str, aux_str: str = "") -> Z3Val:
            vlen = _vec_len(val_type)
            tag = aux_str or builtin_id
            if vlen > 1:
                return tuple(
                    _hw_uf(f"builtin_{tag}_{i}", BV32, BV32)(z3.ZeroExt(27, self.st.lane_id))
                    for i in range(vlen)
                )
            return _hw_uf(f"builtin_{tag}", BV32, BV32)(z3.ZeroExt(27, self.st.lane_id))

        def load_vert_attr(self, imm_u32: int) -> Tuple[z3.BitVecRef, ...]:
            uf = _hw_uf(f"ssa_LoadVertAttr_{imm_u32}", BV32, BV32)
            return tuple(uf(bv32(c)) for c in range(4))

        def load_frag_varying(self, imm_u32: int) -> Tuple[z3.BitVecRef, ...]:
            uf = _hw_uf(f"ssa_LoadFragVarying_{imm_u32}", BV32, BV32)
            return tuple(uf(bv32(c)) for c in range(4))

        def tex_sample(self, op: str, args: List[Any], imm_u32: int, aux_str: str) -> Tuple[Tuple[z3.BitVecRef, ...], Z3Val]:
            uf = _hw_uf("ssa_TexSample", BV32, BV32, BV32)
            uv = args[0] if args else (bv32(0), bv32(0))
            u0 = self.domain.as_u32(uv[0]) if isinstance(uv, tuple) else self.domain.as_u32(uv)
            u1 = self.domain.as_u32(uv[1]) if isinstance(uv, tuple) and len(uv) > 1 else bv32(0)
            return (tuple(uf(u0, u1) for _ in range(4)), args[-1] if args else bv32(0))

        def subgroup_ballot(self, x: Any) -> z3.BitVecRef:
            return _hw_uf("SubgroupBallot", BV32, z3.BitVecSort(5), BV32)(self.domain.as_u32(x), self.st.lane_id)

        def subgroup_shuffle(self, x: Any, lane: Any) -> z3.BitVecRef:
            return _hw_uf("SubgroupShuffle", BV32, BV32, z3.BitVecSort(5), BV32)(
                self.domain.as_u32(x), self.domain.as_u32(lane), self.st.lane_id
            )

        def kill(self, node: Dict[str, Any], args: List[Any], state: Any, reach: Any) -> z3.BitVecRef:
            cond = self.domain.as_bool(args[0]) if args else z3.BoolVal(True)
            self.st.discarded = z3.Or(self.st.discarded, cond)
            self.notes.append("Kill")
            return bv32(0)

        def psb_load(self, node: Dict[str, Any], args: List[Any], state: Any) -> Tuple[z3.BitVecRef, Z3Val]:
            addr = self.domain.as_u64(args[0])
            eb = int(node.get("imm_u32", 0)) or 4
            if eb == 1:
                val = z3.ZeroExt(24, z3.Select(self.st.mem_global, addr))
            elif eb == 2:
                val = z3.ZeroExt(16, z3.Concat(z3.Select(self.st.mem_global, addr + bv64(1)), z3.Select(self.st.mem_global, addr)))
            else:
                val = z3.Concat(
                    z3.Select(self.st.mem_global, addr + bv64(3)),
                    z3.Select(self.st.mem_global, addr + bv64(2)),
                    z3.Select(self.st.mem_global, addr + bv64(1)),
                    z3.Select(self.st.mem_global, addr),
                )
            return (val, args[1] if len(args) > 1 else bv32(0))

        def psb_store(self, node: Dict[str, Any], args: List[Any], state: Any, reach: Any) -> Z3Val:
            addr = self.domain.as_u64(args[0])
            val = self.domain.as_u32(args[1])
            eb = int(node.get("imm_u32", 0)) or 4
            for j in range(min(eb, 4)):
                self.st.mem_global = z3.Store(self.st.mem_global, addr + bv64(j), z3.Extract(8 * j + 7, 8 * j, val))
            return args[2] if len(args) > 2 else bv32(0)

        def atomic_op(self, node: Dict[str, Any], args: List[Any], state: Any, reach: Any) -> Tuple[z3.BitVecRef, Z3Val]:
            addr = self.domain.as_u64(args[0])
            arg = self.domain.as_u32(args[1])
            uf = _hw_uf(f"ssa_{node['op']}_{node.get('aux_str', '')}", BV64, BV32, BV32)
            return (uf(addr, arg), args[-1] if args else bv32(0))

        def store_frag_color(self, node: Dict[str, Any], args: List[Any], state: Any, reach: Any) -> Z3Val:
            return args[-1] if args else bv32(0)

        def store_builtin_output(self, node: Dict[str, Any], args: List[Any], state: Any, reach: Any) -> Z3Val:
            return args[-1] if args else bv32(0)

        def store_vert_varying(self, node: Dict[str, Any], args: List[Any], state: Any, reach: Any) -> Z3Val:
            return args[-1] if args else bv32(0)

        def eval_sym_node(self, node: SymbolicSsaNode, values: Dict[int, Z3Val]) -> Z3Val:
            op = node.op
            if op == "InputReg":
                r = node.imm_u32 & 63
                if self.raw_srcs is not None and r < len(self.raw_srcs):
                    return self.raw_srcs[r]
                if self.dst_old is not None and r == (self.dest_reg & 63):
                    return self.dst_old
                return z3.Select(self.st.regs, z3.BitVecVal(r, 6))
            if op == "Phi":
                return values[node.args[0]] if node.args else bv32(0)
            if op == "LoadBuiltinInput":
                return self.load_builtin(node.imm_u32, node.val_type, aux_str=node.aux_str)
            if op == "ControlBarrier":
                self.st.barrier_epoch = self.st.barrier_epoch + bv32(1)
                self.notes.append("ControlBarrier")
                return bv32(0)
            if op == "Branch":
                if node.args:
                    self.st.pc = self.domain.as_u64(values[node.args[0]])
                return bv32(0)
            if op in ("BranchCond", "Return"):
                return bv32(0)

            node_dict = {
                "id": node.id,
                "op": node.op,
                "type": node.val_type,
                "args": node.args,
                "imm_u32": node.imm_u32 if isinstance(node.imm_u32, int) else 0,
                "imm_u32_2": node.imm_u32_2 if isinstance(node.imm_u32_2, int) else 0,
                "imm_u64": node.imm_u64 if isinstance(node.imm_u64, int) else 0,
                "aux_bool": node.aux_bool,
                "aux_str": node.aux_str,
            }
            return self.eval_node(node_dict, values, self.st, z3.BoolVal(True))

    return _InstLevelSsaInterpreter()


def eval_ssa_module_z3(
    ssa: SymbolicSsaModule,
    raw_srcs: Optional[Sequence[z3.BitVecRef]] = None,
    dst_old: Optional[z3.BitVecRef] = None,
    dest_reg: int = 0,
    state: Optional[ValhallZ3State] = None,
    branch_offset: int = 0,
    domain: Optional["SsaDomain"] = None,
) -> SsaZ3EvalResult:
    """Evaluate every node of a SymbolicSsaModule using the shared SsaInterpreter + SsaDomain."""
    st = dataclasses.replace(state) if state is not None else ValhallZ3State.symbolic()
    interp = _build_inst_interpreter(
        st=st,
        raw_srcs=raw_srcs,
        dst_old=dst_old,
        dest_reg=dest_reg,
        domain=domain,
    )
    values: Dict[int, Z3Val] = {0: bv32(0), 1: bv32(0)}
    for node in ssa.nodes[2:]:
        values[node.id] = interp.eval_sym_node(node, values)

    reg_writes: Dict[int, z3.BitVecRef] = {}
    for r, sid in ssa.reg_writes.items():
        val = values[sid]
        reg_writes[r] = interp.domain.as_u32(val) if not isinstance(val, z3.BitVecRef) else val

    bcond = interp.domain.as_bool(values[ssa.branch_cond]) if ssa.branch_cond is not None else None
    if bcond is not None:
        st.pc = z3.If(bcond, st.pc + bv64(8) + bv64(branch_offset), st.pc + bv64(8))
    return SsaZ3EvalResult(
        values=values,
        reg_writes=reg_writes,
        final_state=st,
        branch_cond=bcond,
        trace_notes=interp.notes,
    )

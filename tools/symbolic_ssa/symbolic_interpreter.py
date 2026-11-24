"""
Whole-CFG Symbolic Z3 Interpreter (`SymbolicSsaInterpreter`).

Subclasses `SsaInterpreter` with `SymbolicZ3BitvecDomain(fp_mode=...)`, implementing
only the Z3 `BlockEffectState` storage backend and RPO bounded-unroll CFG
evaluation. All opcode, memory, kill, atomic, and output semantics are inherited
from `SsaInterpreter` and `SsaDomain`.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import z3

from tools.symbolic_ssa.domains import Z3SymbolicDomain, make_z3_symbolic_domain
from tools.symbolic_ssa.env import (
    SymbolicShaderEnv,
    Z3Val,
    _z3_and,
    _z3_if,
    _z3_not,
    _z3_or,
)
from tools.symbolic_ssa.env import _dsl_bv32 as bv32
from tools.symbolic_ssa.interpreter import SsaInterpreter
from tools.symbolic_ssa.z3utils import dsl


@dataclasses.dataclass
class BlockEffectState:
    """Path-conditioned side-effect state carried across CFG blocks during symbolic evaluation."""

    discarded: z3.BoolRef
    terminated: z3.BoolRef
    demoted: z3.BoolRef
    frag_colors: dict[int, tuple[z3.BitVecRef, ...]]
    builtin_outputs: dict[int, Z3Val]
    vert_varyings: dict[int, tuple[z3.BitVecRef, ...]]
    const_stores: dict[tuple[str, int], tuple[z3.BoolRef, z3.BitVecRef]]
    sym_buf_stores: dict[str, tuple[z3.BoolRef, z3.BitVecRef, z3.BitVecRef]]
    buf_mems: dict[str, z3.ArrayRef]
    image_mems: dict[str, tuple[z3.ArrayRef, ...]]
    fallback_mem: z3.ArrayRef
    atomic_events: dict[str, list[tuple[z3.BoolRef, str, z3.BitVecRef, z3.BitVecRef, z3.BitVecRef]]]
    output_pcs: dict[str, set[int]]


def _clone_state(st: BlockEffectState) -> BlockEffectState:
    return BlockEffectState(
        discarded=st.discarded,
        terminated=st.terminated,
        demoted=st.demoted,
        frag_colors=dict(st.frag_colors),
        builtin_outputs=dict(st.builtin_outputs),
        vert_varyings=dict(st.vert_varyings),
        const_stores=dict(st.const_stores),
        sym_buf_stores=dict(st.sym_buf_stores),
        buf_mems=dict(st.buf_mems),
        image_mems=dict(st.image_mems),
        fallback_mem=st.fallback_mem,
        atomic_events={k: list(v) for k, v in st.atomic_events.items()},
        output_pcs={k: set(v) for k, v in st.output_pcs.items()},
    )


def _atomic_events_eq(
    a: list[tuple[z3.BoolRef, str, z3.BitVecRef, z3.BitVecRef, z3.BitVecRef]],
    b: list[tuple[z3.BoolRef, str, z3.BitVecRef, z3.BitVecRef, z3.BitVecRef]],
) -> bool:
    """Structural equality for atomic-event lists whose entries may hold DSL nodes."""
    if len(a) != len(b):
        return False
    return all(
        len(ea) == len(eb) and all(dsl.expr_eq(x, y) for x, y in zip(ea, eb))
        for ea, eb in zip(a, b)
    )


def _merge_states(
    cond_states: list[tuple[z3.BoolRef, BlockEffectState]],
    env: SymbolicShaderEnv,
) -> BlockEffectState:
    if len(cond_states) == 1:
        return _clone_state(cond_states[0][1])

    base_st = _clone_state(cond_states[0][1])
    for cond, st in cond_states[1:]:
        base_st.discarded = _z3_if(cond, st.discarded, base_st.discarded)
        base_st.terminated = _z3_if(cond, st.terminated, base_st.terminated)
        base_st.demoted = _z3_if(cond, st.demoted, base_st.demoted)

        for rt in set(base_st.frag_colors) | set(st.frag_colors):
            a = st.frag_colors.get(rt, (bv32(0),) * 4)
            b = base_st.frag_colors.get(rt, (bv32(0),) * 4)
            base_st.frag_colors[rt] = tuple(_z3_if(cond, ai, bi) for ai, bi in zip(a, b))

        for bi in set(base_st.builtin_outputs) | set(st.builtin_outputs):
            if bi in st.builtin_outputs and bi in base_st.builtin_outputs:
                base_st.builtin_outputs[bi] = _z3_if(cond, st.builtin_outputs[bi], base_st.builtin_outputs[bi])
            elif bi in st.builtin_outputs:
                base_st.builtin_outputs[bi] = st.builtin_outputs[bi]

        for loc in set(base_st.vert_varyings) | set(st.vert_varyings):
            a = st.vert_varyings.get(loc, (bv32(0),) * 4)
            b = base_st.vert_varyings.get(loc, (bv32(0),) * 4)
            base_st.vert_varyings[loc] = tuple(_z3_if(cond, ai, bi) for ai, bi in zip(a, b))

        for key in set(base_st.const_stores) | set(st.const_stores):
            bkey, off = key
            def_v = dsl.Select(env.get_buf_mem(bkey), bv32(off))
            aw, av = st.const_stores.get(key, (dsl.BoolVal(False), def_v))
            bw, bv_ = base_st.const_stores.get(key, (dsl.BoolVal(False), def_v))
            base_st.const_stores[key] = (_z3_if(cond, aw, bw), _z3_if(cond, av, bv_))

        for skey in set(base_st.sym_buf_stores) | set(st.sym_buf_stores):
            if skey in st.sym_buf_stores and skey in base_st.sym_buf_stores:
                aw, av, ao = st.sym_buf_stores[skey]
                bw, bv_, bo = base_st.sym_buf_stores[skey]
                base_st.sym_buf_stores[skey] = (_z3_if(cond, aw, bw), _z3_if(cond, av, bv_), _z3_if(cond, ao, bo))
            elif skey in st.sym_buf_stores:
                aw, av, ao = st.sym_buf_stores[skey]
                def_v = env.get_sym_buf_slot(skey, ao)
                base_st.sym_buf_stores[skey] = (_z3_and(cond, aw), _z3_if(cond, av, def_v), ao)
            else:
                bw, bv_, bo = base_st.sym_buf_stores[skey]
                def_v = env.get_sym_buf_slot(skey, bo)
                base_st.sym_buf_stores[skey] = (_z3_and(_z3_not(cond), bw), _z3_if(cond, def_v, bv_), bo)

        for bkey in set(base_st.buf_mems) | set(st.buf_mems):
            a_m = st.buf_mems.get(bkey, env.get_buf_mem(bkey))
            b_m = base_st.buf_mems.get(bkey, env.get_buf_mem(bkey))
            if not a_m.eq(b_m):
                base_st.buf_mems[bkey] = dsl.If(cond, a_m, b_m)

        for ikey in set(base_st.image_mems) | set(st.image_mems):
            a_im = st.image_mems.get(ikey, env.get_img_mem(ikey))
            b_im = base_st.image_mems.get(ikey, env.get_img_mem(ikey))
            base_st.image_mems[ikey] = tuple(
                dsl.If(cond, ac, bc) if not ac.eq(bc) else ac
                for ac, bc in zip(a_im, b_im)
            )

        if not st.fallback_mem.eq(base_st.fallback_mem):
            base_st.fallback_mem = dsl.If(cond, st.fallback_mem, base_st.fallback_mem)

        for akey in set(base_st.atomic_events) | set(st.atomic_events):
            b_evs = base_st.atomic_events.get(akey, [])
            s_evs = st.atomic_events.get(akey, [])
            if _atomic_events_eq(b_evs, s_evs):
                continue
            merged_evs = [(_z3_and(_z3_not(cond), m), op, off, arg, cmp_v) for (m, op, off, arg, cmp_v) in b_evs]
            merged_evs.extend((_z3_and(cond, m), op, off, arg, cmp_v) for (m, op, off, arg, cmp_v) in s_evs)
            base_st.atomic_events[akey] = merged_evs

        for k, pcs in st.output_pcs.items():
            base_st.output_pcs.setdefault(k, set()).update(pcs)
    return base_st


class SymbolicSsaInterpreter(SsaInterpreter):
    """Whole-CFG RPO symbolic Z3 interpreter for ValhallSsaModule."""

    def __init__(
        self,
        ssa_json: dict[str, Any],
        env: SymbolicShaderEnv,
        is_valhall_side: bool,
        fp_mode: str = "exact",
        max_unroll: int = 1,
        cutpoints: dict[int, Any] | None = None,
        domain: str | Z3SymbolicDomain | None = "udomain",
    ):
        self.ssa = ssa_json
        self.env = env
        self._env = env
        self.is_valhall = is_valhall_side
        self.fp_mode = fp_mode
        self.domain: Z3SymbolicDomain = make_z3_symbolic_domain(domain, fp_mode=fp_mode)
        self.max_unroll = max(1, int(max_unroll))
        self.cutpoints: dict[int, Any] = cutpoints if cutpoints is not None else {}
        self.blocks: list[dict[str, Any]] = ssa_json.get("blocks", [])
        self.nodes_by_id: dict[int, dict[str, Any]] = {int(n["id"]): n for n in ssa_json.get("nodes", [])}
        self.values: dict[int, Any] = {0: self.domain.const_u32(0)}
        self.block_reach: dict[int, z3.BoolRef] = {}
        self.iter_branch_decisions: list[tuple[int, int, int, z3.BoolRef, z3.BoolRef]] = []
        self.has_loops: bool = False
        self.max_unroll_depth: int = 0

    def lower_bool_value(self, nid: int) -> z3.BoolRef:
        """Return the raw Z3 Bool expression for SSA node `nid`."""
        return self.domain.lower_z3_bool(self.values.get(nid, self.domain.const_bool(False)))

    # Minimal Z3 BlockEffectState backend hooks
    def _active_mask(self, st: BlockEffectState, reach: z3.BoolRef) -> z3.BoolRef:
        return _z3_and(self.domain.lower_z3_bool(reach), _z3_not(_z3_or(st.terminated, st.demoted)))

    def _resolve_buf_addr(self, addr: Any, hint_key: str) -> tuple[str | None, int | None, z3.BitVecRef]:
        return self.env.resolve_addr(self.domain.lower_z3_bv64(addr), hint_key=hint_key)

    def _read_sym_slot(self, st: BlockEffectState, sym_tag: str, off: Any) -> Any | None:
        off_bv = self.domain.lower_z3_bv32(off)
        bkey_hint = sym_tag[4:].split("[", 1)[0]
        if not self.env.is_shared_sym_bkey(bkey_hint):
            return None
        if sym_tag in st.sym_buf_stores:
            w_cond, w_val, w_off = st.sym_buf_stores[sym_tag]
            raw = _z3_if(_z3_and(w_cond, off_bv == w_off), w_val, self.env.get_sym_buf_slot(sym_tag, off_bv))
        else:
            raw = self.env.get_sym_buf_slot(sym_tag, off_bv)
        return self.domain.lift_z3_bv32(raw)

    def _write_sym_slot(self, st: BlockEffectState, mask: Any, sym_tag: str, off: Any, val: Any, nid: int) -> bool:
        m_b = self.domain.lower_z3_bool(mask)
        off_bv = self.domain.lower_z3_bv32(off)
        val_bv = self.domain.lower_z3_bv32(val)
        bkey_hint = sym_tag[4:].split("[", 1)[0]
        if not self.env.is_shared_sym_bkey(bkey_hint):
            return False
        prev_w, prev_v, prev_o = st.sym_buf_stores.get(
            sym_tag, (dsl.BoolVal(False), self.env.get_sym_buf_slot(sym_tag, off_bv), off_bv)
        )
        st.sym_buf_stores[sym_tag] = (
            _z3_or(prev_w, m_b), _z3_if(m_b, val_bv, prev_v), _z3_if(m_b, off_bv, prev_o)
        )
        st.output_pcs.setdefault(sym_tag, set()).add(nid)
        return True

    def _read_buf(self, st: BlockEffectState, bkey: str | None, const_off: int | None, off: Any, addr: Any) -> Any:
        off_bv = self.domain.lower_z3_bv32(off)
        addr_bv = self.domain.lower_z3_bv64(addr)
        if bkey is None:
            raw = dsl.Select(st.fallback_mem, addr_bv)
        else:
            mem_v = dsl.Select(st.buf_mems.get(bkey, self.env.get_buf_mem(bkey)), off_bv)
            if const_off is not None and (bkey, const_off) in st.const_stores:
                w_cond, w_val = st.const_stores[(bkey, const_off)]
                raw = _z3_if(w_cond, w_val, mem_v)
            else:
                raw = mem_v
        return self.domain.lift_z3_bv32(raw)

    def _write_buf(self, st: BlockEffectState, mask: Any, bkey: str | None, const_off: int | None, off: Any, addr: Any, val: Any, nid: int) -> None:
        m_b = self.domain.lower_z3_bool(mask)
        off_bv = self.domain.lower_z3_bv32(off)
        addr_bv = self.domain.lower_z3_bv64(addr)
        val_bv = self.domain.lower_z3_bv32(val)
        if bkey is None:
            st.fallback_mem = dsl.If(m_b, dsl.Store(st.fallback_mem, addr_bv, val_bv), st.fallback_mem)
            st.output_pcs.setdefault("fallback_mem", set()).add(nid)
            return
        cur_mem = st.buf_mems.get(bkey, self.env.get_buf_mem(bkey))
        st.buf_mems[bkey] = dsl.Store(cur_mem, off_bv, val_bv) if dsl.is_true(m_b) else dsl.If(m_b, dsl.Store(cur_mem, off_bv, val_bv), cur_mem)
        if const_off is not None:
            prev_w, prev_v = st.const_stores.get((bkey, const_off), (dsl.BoolVal(False), dsl.Select(cur_mem, off_bv)))
            st.const_stores[(bkey, const_off)] = (_z3_or(prev_w, m_b), _z3_if(m_b, val_bv, prev_v))
            st.output_pcs.setdefault(f"{bkey}+0x{const_off:x}", set()).add(nid)
        else:
            st.output_pcs.setdefault(f"{bkey}:dyn", set()).add(nid)

    def _extract_img_channel(self, addr: Any) -> tuple[int, dsl.Z3BitVecExpr]:
        addr_bv = self.domain.lower_z3_bv64(addr)
        if isinstance(addr_bv, dsl.Z3OpBvAdd):
            c0, c1 = addr_bv.lhs, addr_bv.rhs
            if dsl.is_bv_value(c1) and dsl.as_long(c1) in (4, 8, 12):
                return (dsl.as_long(c1) // 4, c0)
        return (0, addr_bv)

    def _write_img_channel(self, st: BlockEffectState, mask: Any, ikey: str, xy_addr: Any, ch_idx: int, val: Any, nid: int) -> None:
        m_b = self.domain.lower_z3_bool(mask)
        xy_bv = self.domain.lower_z3_bv64(xy_addr)
        val_bv = self.domain.lower_z3_bv32(val)
        cur_im = list(st.image_mems.get(ikey, self.env.get_img_mem(ikey)))
        cur_im[ch_idx] = (
            dsl.Store(cur_im[ch_idx], xy_bv, val_bv)
            if dsl.is_true(m_b)
            else dsl.If(m_b, dsl.Store(cur_im[ch_idx], xy_bv, val_bv), cur_im[ch_idx])
        )
        st.image_mems[ikey] = tuple(cur_im)
        st.output_pcs.setdefault(f"img:{ikey}", set()).add(nid)

    def _write_img_texel(self, st: BlockEffectState, mask: Any, ikey: str, xy: Any, vec4: tuple[Any, ...], nid: int) -> None:
        m_b = self.domain.lower_z3_bool(mask)
        xy_bv = self.domain.lower_z3_bv64(xy)
        v4 = tuple(self.domain.lower_z3_bv32(v) for v in vec4)
        cur_im = st.image_mems.get(ikey, self.env.get_img_mem(ikey))
        st.image_mems[ikey] = tuple(
            dsl.Store(cur_im[c], xy_bv, v4[c]) if dsl.is_true(m_b) else dsl.If(m_b, dsl.Store(cur_im[c], xy_bv, v4[c]), cur_im[c])
            for c in range(4)
        )
        st.output_pcs.setdefault(f"img:{ikey}", set()).add(nid)

    def _record_kill(self, st: BlockEffectState, eff: Any, is_demote: bool, nid: int) -> None:
        eff_b = self.domain.lower_z3_bool(eff)
        if is_demote:
            st.demoted = _z3_or(st.demoted, eff_b)
        else:
            st.terminated = _z3_or(st.terminated, eff_b)
        st.discarded = _z3_or(st.terminated, st.demoted)
        st.output_pcs.setdefault("discarded", set()).add(nid)

    def _write_frag_color(self, st: BlockEffectState, mask: Any, rt: int, vec4: tuple[Any, ...], nid: int) -> None:
        m_b = self.domain.lower_z3_bool(mask)
        v4 = tuple(self.domain.lower_z3_bv32(v) for v in vec4)
        prev = st.frag_colors.get(rt, (bv32(0),) * 4)
        st.frag_colors[rt] = tuple(_z3_if(m_b, n, p) for n, p in zip(v4, prev))
        st.output_pcs.setdefault(f"frag_color[{rt}]", set()).add(nid)

    def _write_builtin_output(self, st: BlockEffectState, mask: Any, bi: int, payload: Any, nid: int) -> None:
        m_b = self.domain.lower_z3_bool(mask)
        pl = self.domain.lower_z3_val(payload)
        st.builtin_outputs[bi] = _z3_if(m_b, pl, st.builtin_outputs[bi]) if bi in st.builtin_outputs else pl
        st.output_pcs.setdefault(f"builtin_out[{bi}]", set()).add(nid)

    def _write_vert_varying(self, st: BlockEffectState, mask: Any, loc: int, vec4: tuple[Any, ...], nid: int) -> None:
        m_b = self.domain.lower_z3_bool(mask)
        v4 = tuple(self.domain.lower_z3_bv32(v) for v in vec4)
        prev = st.vert_varyings.get(loc, (bv32(0),) * 4)
        st.vert_varyings[loc] = tuple(_z3_if(m_b, n, p) for n, p in zip(v4, prev))
        st.output_pcs.setdefault(f"vert_var[{loc}]", set()).add(nid)

    def _record_atomic(self, st: BlockEffectState, mask: Any, bkey: str | None, off: Any, addr: Any, op_tag: str, arg: Any, cmp_val: Any, new_val: Any, nid: int) -> None:
        m_b = self.domain.lower_z3_bool(mask)
        off_bv = self.domain.lower_z3_bv32(off)
        addr_bv = self.domain.lower_z3_bv64(addr)
        arg_bv = self.domain.lower_z3_bv32(arg)
        cmp_bv = self.domain.lower_z3_bv32(cmp_val)
        new_bv = self.domain.lower_z3_bv32(new_val)
        if bkey is not None:
            cur_mem = st.buf_mems.get(bkey, self.env.get_buf_mem(bkey))
            st.buf_mems[bkey] = dsl.Store(cur_mem, off_bv, new_bv) if dsl.is_true(m_b) else dsl.If(m_b, dsl.Store(cur_mem, off_bv, new_bv), cur_mem)
            st.atomic_events.setdefault(bkey, []).append((m_b, op_tag, off_bv, arg_bv, cmp_bv))
            st.output_pcs.setdefault(f"atomic:{bkey}", set()).add(nid)
            st.output_pcs.setdefault(f"{bkey}:dyn", set()).add(nid)
        else:
            st.fallback_mem = dsl.If(m_b, dsl.Store(st.fallback_mem, addr_bv, new_bv), st.fallback_mem)
            st.atomic_events.setdefault("fallback_mem", []).append((m_b, op_tag, addr_bv, arg_bv, cmp_bv))
            st.output_pcs.setdefault("atomic:fallback_mem", set()).add(nid)
            st.output_pcs.setdefault("fallback_mem", set()).add(nid)

    def _eval_node(self, node: dict[str, Any], st: BlockEffectState, reach: z3.BoolRef) -> Z3Val:
        nid = int(node["id"])
        if nid in self.cutpoints:
            return self.cutpoints[nid]
        return self.eval_node(node, self.values, st, reach)

    def collect_pcs_for_nodes(self, root_nids: set[int], max_visit: int = 2048) -> list[int]:
        pcs: set[int] = set()
        seen = self.run_backward_worklist(
            self.nodes_by_id,
            root_nids,
            on_visit=lambda _nid, n: pcs.add(int(n["orig_pc"])) if "orig_pc" in n and len(pcs) < max_visit else None,
        )
        _ = seen
        return sorted(pcs)

    def evaluate_single_block_cone(self, target_nid: int, st: BlockEffectState) -> Z3Val:
        """Incrementally evaluate only the uncached backward cone of `target_nid` on a 1-block DAG."""
        if target_nid in self.cutpoints:
            return self.cutpoints[target_nid]
        if target_nid in self.values:
            return self.values[target_nid]

        order: list[int] = []
        visited: set[int] = set()
        stack: list[tuple[int, int]] = [(target_nid, 0)]
        while stack:
            nid, step = stack.pop()
            if nid in self.values or nid in self.cutpoints:
                continue
            if step == 0:
                if nid in visited:
                    continue
                visited.add(nid)
                node = self.nodes_by_id.get(nid)
                if not node:
                    continue
                stack.append((nid, 1))
                for aid in reversed(node.get("args", [])):
                    ia = int(aid)
                    if ia not in self.values and ia not in self.cutpoints and ia not in visited:
                        stack.append((ia, 0))
            else:
                order.append(nid)

        reach = dsl.BoolVal(True)
        for nid in order:
            self.values[nid] = self._eval_node(self.nodes_by_id[nid], st, reach)
        return self.values.get(target_nid, bv32(0))

    def invalidate_cone_from(self, root_nid: int, users_map: dict[int, list[int]]) -> None:
        """Invalidate cached symbolic values downstream of a newly installed cutpoint."""
        queue = [root_nid]
        seen: set[int] = set()
        while queue:
            cur = queue.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if cur != 0:
                self.values.pop(cur, None)
            for u in users_map.get(cur, ()):
                if u not in seen:
                    queue.append(u)

    def evaluate(self) -> BlockEffectState:
        """Evaluate all blocks in RPO topological order with bounded loop unrolling."""
        init_state = BlockEffectState(
            discarded=dsl.BoolVal(False),
            terminated=dsl.BoolVal(False),
            demoted=dsl.BoolVal(False),
            frag_colors={},
            builtin_outputs={},
            vert_varyings={},
            const_stores={},
            sym_buf_stores={},
            buf_mems={},
            image_mems={},
            fallback_mem=self.env.fallback_mem,
            atomic_events={},
            output_pcs={},
        )
        self.iter_branch_decisions = []
        if not self.blocks:
            return init_state

        blk_by_id: dict[int, dict[str, Any]] = {int(b["id"]): b for b in self.blocks}
        rpo_ids = self.compute_rpo(blk_by_id, 0)
        rpo_index: dict[int, int] = {bid: idx for idx, bid in enumerate(rpo_ids)}
        rpo_blocks = [blk_by_id[bid] for bid in rpo_ids]

        blk_reach: dict[int, z3.BoolRef] = {}
        blk_out_state: dict[int, BlockEffectState] = {}
        edge_conds: dict[int, list[tuple[int, z3.BoolRef]]] = {int(b["id"]): [] for b in self.blocks}
        exit_states: list[tuple[z3.BoolRef, BlockEffectState]] = []

        def _eval_block_slice(
            block_list: list[dict[str, Any]],
            allow_backedges: bool,
            iter_idx: int,
        ) -> list[tuple[int, int, z3.BoolRef]]:
            backedges: list[tuple[int, int, z3.BoolRef]] = []
            for blk in block_list:
                bid = int(blk["id"])
                b_rpo = rpo_index.get(bid, bid)
                if bid == 0 and not allow_backedges:
                    reach = dsl.BoolVal(True)
                    cur_st = _clone_state(init_state)
                else:
                    inc_edges = [(p, c) for (p, c) in edge_conds.get(bid, []) if p in blk_out_state]
                    if not inc_edges:
                        blk_reach[bid] = dsl.BoolVal(False)
                        blk_out_state[bid] = _clone_state(init_state)
                        continue
                    reach = dsl.BoolVal(False)
                    for _, c in inc_edges:
                        reach = _z3_or(reach, c)
                    cur_st = _merge_states([(c, blk_out_state[p]) for (p, c) in inc_edges], self.env)

                blk_reach[bid] = _z3_or(blk_reach[bid], reach) if bid in blk_reach else reach

                for p in blk.get("preds", []):
                    ip = int(p)
                    if ip in rpo_index and rpo_index[ip] >= b_rpo:
                        self.has_loops = True

                inc_cond_map = {p: c for (p, c) in edge_conds.get(bid, [])}
                for pid in blk.get("phis", []):
                    pnode = self.nodes_by_id.get(int(pid))
                    if not pnode:
                        continue
                    incomings = pnode.get("phi", [])
                    fwd = [(int(p), int(v)) for p, v in incomings if int(p) in blk_out_state and int(v) in self.values]
                    if not fwd:
                        vt = pnode.get("type", "U32")
                        self.values[int(pid)] = self.domain.const_bool(False) if vt == "Bool" else self.domain.const_u32(0)
                        continue
                    if len(fwd) == 1:
                        self.values[int(pid)] = self.values[fwd[0][1]]
                    else:
                        acc = self.values[fwd[0][1]]
                        for pred_b, val_id in fwd[1:]:
                            ec = inc_cond_map.get(pred_b, dsl.BoolVal(False))
                            acc = self.domain.merge_z3(ec, self.values[val_id], acc)
                        self.values[int(pid)] = acc

                for nid in blk.get("nodes", []):
                    node = self.nodes_by_id.get(int(nid))
                    if node is not None:
                        self.values[int(nid)] = self._eval_node(node, cur_st, reach)

                blk_out_state[bid] = cur_st
                cf_reach = _z3_and(reach, _z3_not(cur_st.terminated))

                bt = int(blk.get("branch_target", -1))
                ft = int(blk.get("fallthrough", -1))
                is_cond = bool(blk.get("is_cond", False))
                is_end = bool(blk.get("is_end", False))
                is_discard = bool(blk.get("is_discard", False))

                bt_rpo = rpo_index.get(bt, -1)
                ft_rpo = rpo_index.get(ft, -1)
                if is_cond and bt >= 0 and ft >= 0:
                    bcond_id = int(blk.get("branch_cond", 0))
                    bcond = self.domain.lower_z3_bool(self.domain.as_bool(self.values.get(bcond_id, self.domain.const_bool(False))))
                    self.iter_branch_decisions.append((iter_idx, bid, bcond_id, cf_reach, bcond))
                    t_cond = _z3_and(cf_reach, bcond)
                    f_cond = _z3_and(cf_reach, _z3_not(bcond))
                    if bt_rpo > b_rpo:
                        edge_conds.setdefault(bt, []).append((bid, t_cond))
                    else:
                        self.has_loops = True
                        if not dsl.is_false(t_cond):
                            backedges.append((bid, bt, t_cond))
                    if ft_rpo > b_rpo:
                        edge_conds.setdefault(ft, []).append((bid, f_cond))
                    else:
                        self.has_loops = True
                        if not dsl.is_false(f_cond):
                            backedges.append((bid, ft, f_cond))
                elif bt >= 0 and bt_rpo > b_rpo:
                    edge_conds.setdefault(bt, []).append((bid, cf_reach))
                elif ft >= 0 and ft_rpo > b_rpo:
                    edge_conds.setdefault(ft, []).append((bid, cf_reach))
                elif bt >= 0 and bt_rpo <= b_rpo:
                    self.has_loops = True
                    if not dsl.is_false(cf_reach):
                        backedges.append((bid, bt, cf_reach))

                if is_end or is_discard or (bt < 0 and ft < 0):
                    exit_states.append((reach, cur_st))
            return backedges

        backedges = _eval_block_slice(rpo_blocks, allow_backedges=False, iter_idx=0)
        self.max_unroll_depth = 1 if self.has_loops else 0

        unroll_iter = 1
        while backedges and unroll_iter < self.max_unroll:
            unroll_iter += 1
            self.max_unroll_depth = unroll_iter
            min_hdr_rpo = min(rpo_index.get(hdr, 0) for _, hdr, _ in backedges)
            for src_b, hdr, bcond in backedges:
                edge_conds.setdefault(hdr, []).append((src_b, bcond))
            loop_slice = [b for b in rpo_blocks if rpo_index[int(b["id"])] >= min_hdr_rpo]
            exit_states.clear()
            backedges = _eval_block_slice(loop_slice, allow_backedges=True, iter_idx=unroll_iter - 1)

        self.block_reach = blk_reach
        if not exit_states:
            return blk_out_state.get(len(self.blocks) - 1, init_state)
        return _merge_states(exit_states, self.env)

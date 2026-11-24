"""
Bit-Exact Concrete SSA Interpreter (`ConcreteSsaInterpreter`).

Subclasses `SsaInterpreter` with `ConcreteBitvecDomain()`, implementing only the
concrete `_ConcreteState` storage backend and single-path CFG stepper. All
opcode, memory, kill, atomic, and output semantics are inherited from
`SsaInterpreter` and `SsaDomain`.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

from tools.symbolic_ssa.domains import ConcreteBitvecDomain
from tools.symbolic_ssa.env import ConcreteShaderEnv
from tools.symbolic_ssa.interpreter import SsaInterpreter


@dataclasses.dataclass
class _ConcreteState:
    terminated: bool = False
    demoted: bool = False
    frag_colors: Dict[int, Tuple[int, ...]] = dataclasses.field(default_factory=dict)
    builtin_outputs: Dict[int, Any] = dataclasses.field(default_factory=dict)
    vert_varyings: Dict[int, Tuple[int, ...]] = dataclasses.field(default_factory=dict)
    buf_writes: Dict[Tuple[str, int], int] = dataclasses.field(default_factory=dict)
    sym_buf_writes: Dict[Tuple[str, int], int] = dataclasses.field(default_factory=dict)
    img_writes: Dict[Tuple[str, int], List[int]] = dataclasses.field(default_factory=dict)
    fallback_writes: Dict[int, int] = dataclasses.field(default_factory=dict)
    atomic_events: Dict[str, List[Tuple[str, int, int, int]]] = dataclasses.field(default_factory=dict)


class ConcreteSsaInterpreter(SsaInterpreter):
    """Fast concrete bit-exact interpreter subclassing SsaInterpreter with ConcreteBitvecDomain."""

    def __init__(
        self,
        ssa_json: Dict[str, Any],
        is_valhall_side: bool,
        max_block_steps: int = 256,
        domain: Optional[ConcreteBitvecDomain] = None,
    ):
        self.blocks_by_id: Dict[int, Dict[str, Any]] = {int(b["id"]): b for b in ssa_json.get("blocks", [])}
        self.nodes_by_id: Dict[int, Dict[str, Any]] = {int(n["id"]): n for n in ssa_json.get("nodes", [])}
        self.is_valhall = is_valhall_side
        self.max_block_steps = max_block_steps
        self.domain: ConcreteBitvecDomain = domain if domain is not None else ConcreteBitvecDomain()
        self._env: Optional[ConcreteShaderEnv] = None

    # Minimal concrete _ConcreteState backend hooks
    def _active_mask(self, st: _ConcreteState, reach: Any) -> bool:
        return bool(reach) and not (st.terminated or st.demoted)

    def _resolve_buf_addr(self, addr: int, hint_key: str) -> Tuple[Optional[str], Optional[int], int]:
        assert self._env is not None
        bkey, off = self._env.resolve_addr(addr, hint_key=hint_key)
        return (bkey, off, off)

    def _read_sym_slot(self, st: _ConcreteState, sym_tag: str, off: int) -> Optional[int]:
        assert self._env is not None
        if (sym_tag, off) in st.sym_buf_writes:
            return st.sym_buf_writes[(sym_tag, off)]
        if f"sym_slot:{sym_tag}:{off}" in self._env.overrides or f"sym_slot_default:{sym_tag}" in self._env.overrides:
            return self._env.read_sym_buf_slot(sym_tag, off)
        return None

    def _write_sym_slot(self, st: _ConcreteState, mask: bool, sym_tag: str, off: int, val: int, nid: int) -> bool:
        if mask:
            st.sym_buf_writes[(sym_tag, off)] = val
        # Also fall through to underlying buffer write so concrete buf_writes stays synchronized.
        return False

    def _read_buf(self, st: _ConcreteState, bkey: Optional[str], const_off: Optional[int], off: int, addr: int) -> int:
        assert self._env is not None
        if bkey is not None:
            return st.buf_writes.get((bkey, off), self._env.read_init_buf(bkey, off))
        return st.fallback_writes.get(addr, self._env.read_fallback_mem(addr))

    def _write_buf(self, st: _ConcreteState, mask: bool, bkey: Optional[str], const_off: Optional[int], off: int, addr: int, val: int, nid: int) -> None:
        if not mask:
            return
        if bkey is not None:
            st.buf_writes[(bkey, off)] = val
        else:
            st.fallback_writes[addr] = val

    def _extract_img_channel(self, addr: int) -> Tuple[int, int]:
        if (addr & 0xF) in (4, 8, 12):
            return ((addr & 0xF) // 4, addr - (addr & 0xF))
        return (0, addr)

    def _write_img_channel(self, st: _ConcreteState, mask: bool, ikey: str, xy_addr: int, ch_idx: int, val: int, nid: int) -> None:
        assert self._env is not None
        if not mask:
            return
        cur = st.img_writes.get((ikey, xy_addr))
        if cur is None:
            cur = list(self._env.read_init_img(ikey, xy_addr))
        cur[ch_idx] = val
        st.img_writes[(ikey, xy_addr)] = cur

    def _write_img_texel(self, st: _ConcreteState, mask: bool, ikey: str, xy: int, vec4: Tuple[int, ...], nid: int) -> None:
        if mask:
            st.img_writes[(ikey, xy)] = list(vec4)

    def _record_kill(self, st: _ConcreteState, eff: bool, is_demote: bool, nid: int) -> None:
        if eff:
            if is_demote:
                st.demoted = True
            else:
                st.terminated = True

    def _write_frag_color(self, st: _ConcreteState, mask: bool, rt: int, vec4: Tuple[int, ...], nid: int) -> None:
        if mask:
            st.frag_colors[rt] = vec4

    def _write_builtin_output(self, st: _ConcreteState, mask: bool, bi: int, payload: Any, nid: int) -> None:
        if mask:
            st.builtin_outputs[bi] = payload

    def _write_vert_varying(self, st: _ConcreteState, mask: bool, loc: int, vec4: Tuple[int, ...], nid: int) -> None:
        if mask:
            st.vert_varyings[loc] = vec4

    def _record_atomic(self, st: _ConcreteState, mask: bool, bkey: Optional[str], off: int, addr: int, op_tag: str, arg: int, cmp_val: int, new_val: int, nid: int) -> None:
        if not mask:
            return
        if bkey is not None:
            st.buf_writes[(bkey, off)] = new_val
            st.atomic_events.setdefault(bkey, []).append((op_tag, off, arg, cmp_val))
        else:
            st.fallback_writes[addr] = new_val
            st.atomic_events.setdefault("fallback_mem", []).append((op_tag, addr, arg, cmp_val))

    def evaluate(self, env: ConcreteShaderEnv) -> Tuple[Dict[int, Any], Dict[str, Any]]:
        """Execute the CFG concretely against `env`, returning `(node_values, flat_outputs)`."""
        self._env = env
        vals: Dict[int, Any] = {0: 0}
        st = _ConcreteState()

        if 0 not in self.blocks_by_id:
            return vals, {}

        cur_bid = 0
        prev_bid = -1
        steps = 0
        while cur_bid in self.blocks_by_id and steps < self.max_block_steps:
            steps += 1
            blk = self.blocks_by_id[cur_bid]
            for pid in blk.get("phis", []):
                pn = self.nodes_by_id.get(pid)
                if not pn:
                    continue
                chosen = None
                for pb, vid in pn.get("phi", []):
                    if int(pb) == prev_bid and int(vid) in vals:
                        chosen = vals[int(vid)]
                        break
                if chosen is None:
                    for _, vid in pn.get("phi", []):
                        if int(vid) in vals:
                            chosen = vals[int(vid)]
                            break
                vals[pid] = chosen if chosen is not None else (False if pn.get("type") == "Bool" else 0)

            for nid in blk.get("nodes", []):
                node = self.nodes_by_id.get(nid)
                if node is not None:
                    vals[nid] = self.eval_node(node, vals, st, True)

            if st.terminated or blk.get("is_end") or blk.get("is_discard"):
                break
            prev_bid = cur_bid
            bt = int(blk.get("branch_target", -1))
            ft = int(blk.get("fallthrough", -1))
            if blk.get("is_cond") and bt >= 0 and ft >= 0:
                cond_v = self.domain.as_bool(vals.get(int(blk.get("branch_cond", 0)), False))
                cur_bid = bt if cond_v else ft
            elif bt >= 0:
                cur_bid = bt
            elif ft >= 0:
                cur_bid = ft
            else:
                break

        outs: Dict[str, Any] = {}
        if env.stage == "Fragment" or st.terminated or st.demoted:
            outs["discarded"] = bool(st.terminated or st.demoted)
        if not (st.terminated or st.demoted):
            for rt, col in st.frag_colors.items():
                for c, cname in enumerate(("r", "g", "b", "a")):
                    outs[f"frag_color[{rt}].{cname}"] = col[c]
            for bi, payload in st.builtin_outputs.items():
                if isinstance(payload, tuple):
                    for c, v in enumerate(payload):
                        outs[f"builtin_out[{bi}][{c}]"] = self.domain.as_u32(v)
                else:
                    outs[f"builtin_out[{bi}]"] = self.domain.as_u32(payload)
            for loc, vec in st.vert_varyings.items():
                for c in range(4):
                    outs[f"vert_var[{loc}][{c}]"] = vec[c]
            for (bkey, off), v in st.buf_writes.items():
                if bkey in ("tls_private", "wls_shared") or bkey.startswith("workgroup:"):
                    continue
                outs[f"{bkey}+0x{off:x}"] = v
            for (ikey, xy), vec in st.img_writes.items():
                for c, cname in enumerate(("r", "g", "b", "a")):
                    outs[f"img:{ikey}[0x{xy:x}].{cname}"] = vec[c]
            for akey, evs in st.atomic_events.items():
                if akey in ("tls_private", "wls_shared") or akey.startswith("workgroup:"):
                    continue
                outs[f"atomic:{akey}"] = tuple(evs)
        return vals, outs

    execute_one = evaluate

"""
Demand-Driven Backfill UDomain Type Analysis (`U^bf`).

Implements the backward use-site demand propagation and kill-time backfill
lattice formalized in Section 7 of `SSA-to-Z3-Denotational-Semantics`:

  - Finite-height lattice `DemandKind` (`UNUSED < {BOOL, BV32, BV64, F16X4,
    F32 < F32X2, F64, PTR(B), SYM} < TOP_BITS`, height h <= 3).
  - Monotone backward dataflow worklist (`DemandBackfillAnalysis`) that tags
    every SSA node ID with its inferred demand type `Delta*(v)`, identifies
    pure dead nodes (`Delta*(v) == UNUSED`) for kill-time elision, and tracks
    per-lane vector demands as well as per-offset `(buffer, const_offset)`
    slot demands so mixed-type UBO/SSBO structs do not escalate an entire
    buffer array to `TOP_BITS`.
  - Two-shader joint demand merging (`merge_miter_demands`) guaranteeing
    identical Z3 sorts across both sides of an equivalence miter: purely FP
    shaders emit pure `QF_NRA` with zero bridges, while mixed FP/bit-hack
    miters escalate only the conflicted symbols/slots to `TOP_BITS`.
"""

from __future__ import annotations

import copy
import enum
from dataclasses import dataclass, field
from typing import Any


class DemandKind(enum.Enum):
    """Finite-height demand type lattice L_ty (Section 7.1)."""

    UNUSED = "Unused"
    BOOL = "Bool"
    BV32 = "BV32"
    BV64 = "BV64"
    F16X4 = "F16x4"
    F32 = "F32"
    F32X2 = "F32x2"
    F64 = "F64"
    PTR = "Ptr"
    SYM = "Sym"
    TOP_BITS = "TopBits"


@dataclass(frozen=True, slots=True)
class DemandTag:
    """Lattice element in L_ty, carrying optional pointer provenance base for `PTR`."""

    kind: DemandKind
    ptr_base: str | None = None

    @property
    def label(self) -> str:
        if self.kind is DemandKind.PTR and self.ptr_base:
            return f"Ptr({self.ptr_base})"
        return self.kind.value

    @property
    def is_unused(self) -> bool:
        return self.kind is DemandKind.UNUSED

    @property
    def is_top(self) -> bool:
        return self.kind is DemandKind.TOP_BITS

    @property
    def is_fp(self) -> bool:
        return self.kind in (DemandKind.F16X4, DemandKind.F32, DemandKind.F32X2, DemandKind.F64)


D_UNUSED = DemandTag(DemandKind.UNUSED)
D_BOOL = DemandTag(DemandKind.BOOL)
D_BV32 = DemandTag(DemandKind.BV32)
D_BV64 = DemandTag(DemandKind.BV64)
D_F16X4 = DemandTag(DemandKind.F16X4)
D_F32 = DemandTag(DemandKind.F32)
D_F32X2 = DemandTag(DemandKind.F32X2)
D_F64 = DemandTag(DemandKind.F64)
D_SYM = DemandTag(DemandKind.SYM)
D_TOP = DemandTag(DemandKind.TOP_BITS)


def demand_ptr(base: str | None = None) -> DemandTag:
    return DemandTag(DemandKind.PTR, ptr_base=base)


def demand_join(a: DemandTag, b: DemandTag) -> DemandTag:
    """Monotone join `a |_| b` on the finite-height demand lattice L_ty."""
    if a.kind is DemandKind.UNUSED:
        return b
    if b.kind is DemandKind.UNUSED or a == b:
        return a
    if a.kind is DemandKind.TOP_BITS or b.kind is DemandKind.TOP_BITS:
        return D_TOP
    pair = {a.kind, b.kind}
    if pair == {DemandKind.F32, DemandKind.F32X2}:
        return D_F32X2
    if a.kind is DemandKind.PTR and b.kind is DemandKind.PTR:
        if a.ptr_base is None:
            return b
        if b.ptr_base is None:
            return a
        return a if a.ptr_base == b.ptr_base else D_TOP
    return D_TOP


def demand_leq(a: DemandTag, b: DemandTag) -> bool:
    """Partial order `a <= b` on L_ty."""
    if a.kind is DemandKind.UNUSED or a == b or b.kind is DemandKind.TOP_BITS:
        return True
    if a.kind is DemandKind.F32 and b.kind is DemandKind.F32X2:
        return True
    return a.kind is DemandKind.PTR and b.kind is DemandKind.PTR and (
        a.ptr_base is None or b.ptr_base is None or a.ptr_base == b.ptr_base
    )


def needs_bridge(def_tag: DemandTag, use_tag: DemandTag) -> bool:
    """
    Return True iff an uninterpreted bridge (`u_bv32_to_f32` / `u_f32_to_bv32`)
    is required at a use site demanding `use_tag` from a value backfilled as `def_tag`.

    Directly implements the Bridge Elimination Rule (Section 7.3):
    zero bridges whenever `def_tag` and `use_tag` agree (or `F32X2 -> F32` lane 0).
    """
    if def_tag.is_unused or use_tag.is_unused:
        return False
    if def_tag == use_tag:
        return False
    if def_tag.kind is DemandKind.F32X2 and use_tag.kind is DemandKind.F32:
        return False
    if def_tag.kind is DemandKind.TOP_BITS and use_tag.kind in (DemandKind.BV32, DemandKind.BV64):
        return False
    if def_tag.kind in (DemandKind.BV32, DemandKind.BV64) and use_tag.kind is DemandKind.TOP_BITS:
        return False
    if def_tag.is_fp and use_tag in (D_TOP, D_BV32, D_BV64):
        return True
    return def_tag in (D_TOP, D_BV32, D_BV64) and use_tag.is_fp


_SIDE_EFFECT_OPS = frozenset({
    "PsbStore",
    "Kill",
    "StoreFragColor",
    "StoreBuiltinOutput",
    "StoreVertVarying",
    "StoreTexel",
    "AtomicRmw",
    "AtomicExchange",
    "AtomicCmpXchg",
    "ControlBarrier",
    "Branch",
    "BranchCond",
    "Return",
})

_INT32_UNARY_OPS = frozenset({"BitNot", "SNeg", "SAbs", "Clz", "BitCount", "BitReverse"})
_INT32_BINARY_OPS = frozenset({
    "IAdd", "ISub", "IMul", "BitAnd", "BitOr", "BitXor",
    "Shl", "ShrU", "ShrS", "RotL", "RotR",
    "UDiv", "SDiv", "UMod", "SRem", "SMod",
})
_INT64_BINARY_OPS = frozenset({
    "IAdd64", "ISub64", "IMul64", "BitAnd64", "BitOr64", "BitXor64",
    "Shl64", "ShrU64", "ShrS64",
})
_FP_UNARY_OPS = frozenset({
    "FAbs", "FNeg", "Floor", "Ceil", "Trunc", "RoundEven", "Fract", "Sqrt", "InverseSqrt",
    "Sin", "Cos", "Exp", "Exp2", "Log", "Log2", "Atan", "Asin", "Acos",
    "Sinh", "Cosh", "Tanh", "DPdx", "DPdy", "Fwidth", "FrexpExp", "FrexpSignificand",
})
_FP_BINARY_OPS = frozenset({
    "FAdd", "FSub", "FMul", "FDiv", "FMin", "FMax", "NMin", "NMax", "Pow", "Ldexp", "Atan2",
})
_TEX_OPS = frozenset({
    "TexSample", "TexSampleLod", "TexSampleDref", "TexGather", "TexFetch", "ImageQueryLod",
})


def is_pure_ssa_node(node: dict[str, Any]) -> bool:
    """Return True if an SSA node has no observable side effects and can be elided when unused."""
    op = node.get("op", "")
    if op in _SIDE_EFFECT_OPS:
        return False
    return not (node.get("is_side_effect", False) or node.get("is_barrier_or_cf", False))


@dataclass
class DemandAnnotations:
    """
    Output of `DemandBackfillAnalysis` for a single SSA module.

    Attributes:
      node_types:       `Delta*(v)` inferred demand tag per SSA node ID.
      def_types:        intrinsic output type `tau_def(v)` for fixed-representation ops.
      lane_demands:     per-lane `(nid, lane_idx) -> DemandTag` for vector producers.
      input_demands:    canonical input / per-offset buffer slot key -> `DemandTag`.
      output_demands:   observable output channel key -> `DemandTag`.
      dead_nodes:       pure SSA node IDs with `Delta*(v) == UNUSED`.
      const_load_slots: maps `PsbLoad` / `LoadFau` node ID -> `(bkey, const_off)` when resolved.
    """

    node_types: dict[int, DemandTag] = field(default_factory=dict)
    def_types: dict[int, DemandTag] = field(default_factory=dict)
    lane_demands: dict[tuple[int, int], DemandTag] = field(default_factory=dict)
    input_demands: dict[str, DemandTag] = field(default_factory=dict)
    output_demands: dict[str, DemandTag] = field(default_factory=dict)
    dead_nodes: frozenset[int] = frozenset()
    const_load_slots: dict[int, tuple[str, int]] = field(default_factory=dict)

    def tag_of(self, nid: int) -> DemandTag:
        return self.node_types.get(nid, D_UNUSED)

    def bridge_required(self, src_nid: int, req: DemandTag) -> bool:
        return needs_bridge(self.tag_of(src_nid), req)


def _intrinsic_def_type(node: dict[str, Any]) -> DemandTag:
    """Return the intrinsic definition representation `tau_def(v)`, or `D_UNUSED` if untyped/polymorphic."""
    op = node.get("op", "")
    if op in _FP_UNARY_OPS or op in _FP_BINARY_OPS or op in ("FClamp", "Fma", "FMix", "ConvertSToF", "ConvertUToF", "FConvert64To32"):
        return D_F32
    if op == "FConvert32To64":
        return D_F64
    if op == "PackHalf2x16":
        return D_F16X4
    if op == "UnpackHalf2x16":
        return D_F32
    if op in _INT32_UNARY_OPS or op in _INT32_BINARY_OPS or op in (
        "BitFieldUExtract", "BitFieldSExtract", "BfeU", "BfeS", "BitFieldInsert", "Bfi",
        "IDot4S", "IDot4SU", "IDot4U", "ConvertFToS", "ConvertFToU", "UConvert64To32", "Trunc64To32",
        "SubgroupBallot", "SubgroupShuffle", "LoadSidebandU32", "ImageQuerySize",
    ):
        return D_BV32
    if op in ("BitNot64", "SConvert32To64", "UConvert32To64") or op in _INT64_BINARY_OPS:
        return D_BV64
    if op.startswith(("ICmp", "FCmp")) or op in ("LogicalAnd", "LogicalOr", "LogicalNot", "IsNan", "IsInf", "ConstBool"):
        return D_BOOL
    if op == "LoadSidebandU64":
        aux = node.get("aux_str", "")
        if aux:
            return demand_ptr(aux)
        imm = int(node.get("imm_u32", 0))
        if imm == 6:
            return demand_ptr("tls_private")
        if imm == 7:
            return demand_ptr("wls_shared")
        return D_BV64
    return D_UNUSED


def _resolve_static_buf_slot(
    nid: int,
    nodes_by_id: dict[int, dict[str, Any]],
    env: Any | None,
    is_valhall: bool,
) -> tuple[str, int] | None:
    """
    Statically resolve a `LoadFau` or `PsbLoad` node to `(bkey, const_byte_offset)`
    when possible, enabling per-offset demand granularity for mixed-type structs.
    """
    node = nodes_by_id.get(nid)
    if node is None:
        return None
    op = node.get("op", "")
    if op == "LoadFau" and env is not None:
        w = int(node.get("imm_u32", 0))
        if w in getattr(env, "rloc_by_word", {}):
            bkey, uoff = env.rloc_by_word[w]
            return (bkey, int(uoff))
        if w in getattr(env, "pilot_by_word", {}):
            pf, _is_hi = env.pilot_by_word[w]
            vkind = pf.get("value_kind", "")
            if (vkind == "symbol_value" or pf.get("is_loaded_value", False)) and pf.get("has_byte_offset", False) and not pf.get("is_derived", False):
                return (str(pf["key"]), int(pf["byte_offset"]))
        return None

    if op == "PsbLoad":
        args = node.get("args", [])
        if not args:
            return None
        addr_id = int(args[0])
        base_off = _trace_const_ptr_offset(addr_id, nodes_by_id, env)
        if base_off is not None:
            return base_off
    return None


def _trace_const_int(nid: int, nodes_by_id: dict[int, dict[str, Any]], depth: int = 0) -> int | None:
    if depth > 6 or nid not in nodes_by_id:
        return None
    n = nodes_by_id[nid]
    op = n.get("op", "")
    if op in ("ConstU32", "ConstS32", "ConstF32"):
        return int(n.get("imm_u32", 0)) & 0xFFFFFFFF
    if op == "ConstU64":
        return int(n.get("imm_u64", 0)) & 0xFFFFFFFFFFFFFFFF
    if op in ("Copy", "Bitcast", "UConvert32To64", "SConvert32To64", "UConvert64To32"):
        args = n.get("args", [])
        return _trace_const_int(int(args[0]), nodes_by_id, depth + 1) if args else None
    if op == "IAdd" or op == "IAdd64":
        args = n.get("args", [])
        if len(args) == 2:
            a = _trace_const_int(int(args[0]), nodes_by_id, depth + 1)
            b = _trace_const_int(int(args[1]), nodes_by_id, depth + 1)
            if a is not None and b is not None:
                return a + b
    return None


def _trace_const_ptr_offset(
    nid: int,
    nodes_by_id: dict[int, dict[str, Any]],
    env: Any | None,
    depth: int = 0,
) -> tuple[str, int] | None:
    if depth > 8 or nid not in nodes_by_id:
        return None
    n = nodes_by_id[nid]
    op = n.get("op", "")
    if op == "LoadSidebandU64":
        aux = n.get("aux_str", "")
        if aux:
            return (aux, 0)
        imm = int(n.get("imm_u32", 0))
        if imm == 6:
            return ("tls_private", 0)
        if imm == 7:
            wg = getattr(env, "unique_workgroup_key", None) if env is not None else None
            return (wg or "wls_shared", 0)
        return None
    if op in ("Copy", "Bitcast"):
        args = n.get("args", [])
        return _trace_const_ptr_offset(int(args[0]), nodes_by_id, env, depth + 1) if args else None
    if op == "IAdd64":
        args = n.get("args", [])
        if len(args) == 2:
            p0 = _trace_const_ptr_offset(int(args[0]), nodes_by_id, env, depth + 1)
            c1 = _trace_const_int(int(args[1]), nodes_by_id, depth + 1)
            if p0 is not None and c1 is not None:
                return (p0[0], p0[1] + c1)
            p1 = _trace_const_ptr_offset(int(args[1]), nodes_by_id, env, depth + 1)
            c0 = _trace_const_int(int(args[0]), nodes_by_id, depth + 1)
            if p1 is not None and c0 is not None:
                return (p1[0], p1[1] + c0)
    if op == "Pack64" and env is not None:
        args = n.get("args", [])
        if len(args) == 2:
            lo_n = nodes_by_id.get(int(args[0]))
            hi_n = nodes_by_id.get(int(args[1]))
            if lo_n and hi_n and lo_n.get("op") == "LoadFau" and hi_n.get("op") == "LoadFau":
                w_lo = int(lo_n.get("imm_u32", 0))
                if w_lo in getattr(env, "pilot_by_word", {}):
                    pf, is_hi = env.pilot_by_word[w_lo]
                    if not is_hi and pf.get("key"):
                        poff = int(pf.get("pointer_offset", 0)) if pf.get("has_pointer_offset", False) else 0
                        return (str(pf["key"]), poff)
    return None


class DemandBackfillAnalysis:
    """
    Backward use-site demand dataflow analysis over a `ValhallSsaModule` JSON dict.

    Computes the least fixpoint `Delta* : NodeId -> L_ty` (Section 7.2 & 7.4)
    in at most `3 * |V|` lattice updates, tracking per-lane vector demands and
    per-offset `(buffer, const_offset)` slot demands.
    """

    def __init__(
        self,
        ssa_json: dict[str, Any],
        env: Any | None = None,
        *,
        is_valhall: bool = False,
    ) -> None:
        self.ssa = ssa_json
        self.env = env
        self.is_valhall = is_valhall
        self.nodes_by_id: dict[int, dict[str, Any]] = {
            int(n["id"]): n for n in ssa_json.get("nodes", [])
        }
        self.blocks: list[dict[str, Any]] = ssa_json.get("blocks", [])

    def analyze(self) -> DemandAnnotations:
        delta: dict[int, DemandTag] = {nid: D_UNUSED for nid in self.nodes_by_id}
        def_types: dict[int, DemandTag] = {
            nid: _intrinsic_def_type(n) for nid, n in self.nodes_by_id.items()
        }
        lane_demands: dict[tuple[int, int], DemandTag] = {}
        output_demands: dict[str, DemandTag] = {}
        worklist: list[tuple[int, DemandTag]] = []

        def push(nid: int, req: DemandTag) -> None:
            if nid > 0 and nid in self.nodes_by_id and not req.is_unused:
                worklist.append((nid, req))

        def push_lane(nid: int, lane: int, req: DemandTag) -> None:
            if nid <= 0 or nid not in self.nodes_by_id or req.is_unused:
                return
            key = (nid, lane)
            old = lane_demands.get(key, D_UNUSED)
            new = demand_join(old, req)
            if new != old:
                lane_demands[key] = new
                # Propagate into VecConstruct* producers lane-by-lane.
                prod = self.nodes_by_id[nid]
                pop = prod.get("op", "")
                pargs = prod.get("args", [])
                if pop in ("VecConstruct2", "VecConstruct3", "VecConstruct4") and lane < len(pargs):
                    push(int(pargs[lane]), new)
                elif pop in ("Copy", "Bitcast") and pargs:
                    push_lane(int(pargs[0]), lane, new)
                elif pop == "Select" and len(pargs) >= 3:
                    push(int(pargs[0]), D_BOOL)
                    push_lane(int(pargs[1]), lane, new)
                    push_lane(int(pargs[2]), lane, new)
                elif pop == "Phi":
                    for _, vid in prod.get("phi", []):
                        push_lane(int(vid), lane, new)
                else:
                    push(nid, new)

        # Phase 1: Seed roots from observable side effects and branch conditions.
        for blk in self.blocks:
            if blk.get("is_cond", False):
                bc = int(blk.get("branch_cond", 0))
                if bc > 0:
                    push(bc, D_BOOL)

        for nid, node in self.nodes_by_id.items():
            op = node.get("op", "")
            args = [int(a) for a in node.get("args", [])]
            if op == "StoreFragColor" and args:
                # Seed frag color lanes from their producer's intrinsic type if known,
                # otherwise default to TOP_BITS so untyped stores remain sound.
                rt = int(node.get("imm_u32", 0))
                self._seed_vec_output(args[0], f"frag_color[{rt}]", 4, def_types, output_demands, push_lane)
                if len(args) > 1:
                    push(args[-1], D_SYM)
            elif op == "StoreVertVarying" and args:
                imm32 = int(node.get("imm_u32", 0))
                loc = (
                    self.env.sout_addr_to_loc.get(imm32, imm32)
                    if (self.is_valhall and self.env is not None)
                    else imm32
                )
                self._seed_vec_output(args[0], f"vert_var[{loc}]", 4, def_types, output_demands, push_lane)
                if len(args) > 1:
                    push(args[-1], D_SYM)
            elif op == "StoreBuiltinOutput" and args:
                bi = int(node.get("imm_u32", 0))
                self._seed_vec_output(args[0], f"builtin_out[{bi}]", 4, def_types, output_demands, push_lane)
                if len(args) > 1:
                    push(args[-1], D_SYM)
            elif op == "PsbStore" and len(args) >= 2:
                push(args[0], D_BV64)
                # Preserve exact bits at untyped memory stores unless producer is purely typed.
                push(args[1], D_TOP)
                if len(args) >= 3:
                    push(args[2], D_SYM)
            elif op == "StoreTexel" and len(args) >= 2:
                push(args[0], D_BV32)
                push(args[1], D_TOP)
                if len(args) >= 3:
                    push(args[-1], D_SYM)
            elif op == "Kill":
                if node.get("aux_bool", False) and len(args) >= 2:
                    push(args[0], D_BOOL)
            elif op in ("AtomicRmw", "AtomicExchange", "AtomicCmpXchg"):
                if args:
                    push(args[0], D_BV64)
                if len(args) > 1:
                    push(args[1], D_BV32)
                if op == "AtomicCmpXchg" and len(args) > 2:
                    push(args[2], D_BV32)
            elif op == "ControlBarrier" and args:
                push(args[-1], D_SYM)

        # Phase 2: Monotone backward worklist propagation to fixpoint.
        while worklist:
            nid, req = worklist.pop()
            node = self.nodes_by_id[nid]
            # Join incoming demand with the node's own intrinsic definition type (if any).
            dt = def_types.get(nid, D_UNUSED)
            combined = demand_join(req, dt) if not dt.is_unused else req
            tau_new = demand_join(delta[nid], combined)
            if tau_new == delta[nid]:
                continue
            delta[nid] = tau_new
            self._propagate_operands(node, tau_new, push, push_lane)

        # Phase 3: Collect per-input and per-offset buffer slot demands + dead pure nodes.
        input_demands: dict[str, DemandTag] = {}
        const_load_slots: dict[int, tuple[str, int]] = {}

        def record_input(key: str, tag: DemandTag) -> None:
            if tag.is_unused:
                return
            input_demands[key] = demand_join(input_demands.get(key, D_UNUSED), tag)

        for nid, node in self.nodes_by_id.items():
            op = node.get("op", "")
            nt = delta[nid]
            if op in ("LoadFragVarying", "LoadVertAttr"):
                imm32 = int(node.get("imm_u32", 0))
                loc = (
                    self.env.sinp_addr_to_loc.get(imm32, imm32)
                    if (self.is_valhall and self.env is not None)
                    else imm32
                )
                prefix = "frag_var" if op == "LoadFragVarying" else "vert_attr"
                for c in range(4):
                    lt = lane_demands.get((nid, c), nt)
                    record_input(f"{prefix}_loc{loc}_{c}", lt)
            elif op == "LoadBuiltinInput":
                bi = int(node.get("imm_u32", 0))
                vt = node.get("type", "U32")
                if vt in ("V4F32", "V3U32"):
                    nlanes = 4 if vt == "V4F32" else 3
                    for c in range(nlanes):
                        lt = lane_demands.get((nid, c), nt)
                        record_input(f"bi_{bi}_{c}", lt)
                else:
                    record_input(f"bi_{bi}", nt)
            elif op == "LoadFau":
                slot = _resolve_static_buf_slot(nid, self.nodes_by_id, self.env, self.is_valhall)
                if slot is not None:
                    bkey, uoff = slot
                    const_load_slots[nid] = slot
                    record_input(f"buf_slot:{bkey}:{uoff}", nt)
                else:
                    w = int(node.get("imm_u32", 0))
                    record_input(f"fau_word:{w}", nt)
            elif op == "PsbLoad":
                # PsbLoad returns a (val, token) tuple; lane 0 carries the loaded value.
                val_tag = lane_demands.get((nid, 0), nt)
                slot = _resolve_static_buf_slot(nid, self.nodes_by_id, self.env, self.is_valhall)
                if slot is not None:
                    bkey, uoff = slot
                    const_load_slots[nid] = slot
                    record_input(f"buf_slot:{bkey}:{uoff}", val_tag)

        dead_nodes = frozenset(
            nid for nid, node in self.nodes_by_id.items()
            if delta[nid].is_unused and is_pure_ssa_node(node)
        )

        return DemandAnnotations(
            node_types=delta,
            def_types=def_types,
            lane_demands=lane_demands,
            input_demands=input_demands,
            output_demands=output_demands,
            dead_nodes=dead_nodes,
            const_load_slots=const_load_slots,
        )

    def _infer_producer_hint(
        self,
        nid: int,
        def_types: dict[int, DemandTag],
        depth: int = 0,
    ) -> DemandTag:
        """Peek through transparent ops (`Bitcast`, `Copy`, `Select`, `Phi`) to see if an output is purely FP."""
        if depth > 8 or nid not in self.nodes_by_id:
            return D_TOP
        dt = def_types.get(nid, D_UNUSED)
        if not dt.is_unused:
            return dt
        n = self.nodes_by_id[nid]
        op = n.get("op", "")
        args = [int(a) for a in n.get("args", [])]
        if op == "ConstF32":
            return D_F32
        if op in ("Copy", "Bitcast") and args:
            return self._infer_producer_hint(args[0], def_types, depth + 1)
        if op == "Select" and len(args) >= 3:
            t_h = self._infer_producer_hint(args[1], def_types, depth + 1)
            f_h = self._infer_producer_hint(args[2], def_types, depth + 1)
            return demand_join(t_h, f_h)
        if op == "Phi":
            acc = D_UNUSED
            for _, vid in n.get("phi", []):
                acc = demand_join(acc, self._infer_producer_hint(int(vid), def_types, depth + 1))
            return acc if not acc.is_unused else D_TOP
        return D_TOP

    def _seed_vec_output(
        self,
        vec_nid: int,
        out_prefix: str,
        nlanes: int,
        def_types: dict[int, DemandTag],
        output_demands: dict[str, DemandTag],
        push_lane: Any,
    ) -> None:
        prod = self.nodes_by_id.get(vec_nid)
        if prod is not None and prod.get("op") in ("VecConstruct2", "VecConstruct3", "VecConstruct4"):
            pargs = [int(a) for a in prod.get("args", [])]
            for c in range(min(nlanes, len(pargs))):
                hint = self._infer_producer_hint(pargs[c], def_types)
                tag = D_F32 if hint.kind is DemandKind.F32 else D_TOP
                output_demands[f"{out_prefix}[{c}]"] = demand_join(
                    output_demands.get(f"{out_prefix}[{c}]", D_UNUSED), tag
                )
                push_lane(vec_nid, c, tag)
            return
        hint = self._infer_producer_hint(vec_nid, def_types)
        tag = D_F32 if hint.kind is DemandKind.F32 else D_TOP
        for c in range(nlanes):
            output_demands[f"{out_prefix}[{c}]"] = demand_join(
                output_demands.get(f"{out_prefix}[{c}]", D_UNUSED), tag
            )
            push_lane(vec_nid, c, tag)

    def _propagate_operands(
        self,
        node: dict[str, Any],
        tau_out: DemandTag,
        push: Any,
        push_lane: Any,
    ) -> None:
        op = node.get("op", "")
        args = [int(a) for a in node.get("args", [])]

        # Transparent ops forward demand directly.
        if op in ("Copy", "Bitcast", "Nop"):
            if args:
                push(args[0], tau_out)
            return
        if op == "Select" and len(args) >= 3:
            push(args[0], D_BOOL)
            push(args[1], tau_out)
            push(args[2], tau_out)
            return
        if op == "Phi":
            for _, vid in node.get("phi", []):
                push(int(vid), tau_out)
            return

        # Vector / tuple projection & packing.
        if op == "VecExtract" and args:
            lane = int(node.get("imm_u32", 0))
            push_lane(args[0], lane, tau_out)
            return
        if op in ("VecConstruct2", "VecConstruct3", "VecConstruct4"):
            for a in args:
                push(a, tau_out)
            return
        if op == "VecShuffle" and len(args) >= 2:
            push(args[0], tau_out)
            push(args[1], tau_out)
            return
        if op == "TupleExtract" and args:
            idx = int(node.get("imm_u32", 0))
            push_lane(args[0], idx, tau_out)
            return
        if op == "Pack64" and len(args) >= 2:
            if tau_out.kind is DemandKind.F32X2:
                push(args[0], D_F32)
                push(args[1], D_F32)
            else:
                push(args[0], D_BV32)
                push(args[1], D_BV32)
            return
        if op == "Split64" and args:
            if tau_out.kind is DemandKind.F32:
                push(args[0], D_F32X2)
            else:
                push(args[0], D_BV64)
            return
        if op == "UnpackHalf2x16" and args:
            push(args[0], D_F16X4 if tau_out.is_fp else D_BV32)
            return
        if op == "PackHalf2x16" and args:
            push(args[0], D_F32)
            return

        # Floating-point ops demand F32 operands.
        if op in _FP_UNARY_OPS or op in _FP_BINARY_OPS or op in ("FClamp", "Fma", "FMix") or op.startswith("FCmp") or op in ("IsNan", "IsInf"):
            for a in args:
                push(a, D_F32)
            return

        # Conversions.
        if op in ("ConvertFToS", "ConvertFToU", "FConvert32To64"):
            if args:
                push(args[0], D_F32)
            return
        if op == "FConvert64To32":
            if args:
                push(args[0], D_F64)
            return
        if op in ("ConvertSToF", "ConvertUToF", "SConvert32To64", "UConvert32To64"):
            if args:
                push(args[0], D_BV32)
            return
        if op in ("UConvert64To32", "Trunc64To32"):
            if args:
                push(args[0], D_BV64)
            return

        # Integer & bitfield ops demand BV32/BV64.
        if op in _INT32_UNARY_OPS or op in _INT32_BINARY_OPS or op.startswith("ICmp") or op in (
            "BitFieldUExtract", "BitFieldSExtract", "BfeU", "BfeS", "BitFieldInsert", "Bfi",
            "IDot4S", "IDot4SU", "IDot4U", "SubgroupShuffle",
        ):
            for a in args:
                push(a, D_BV32)
            return
        if op == "SubgroupBallot" and args:
            push(args[0], D_BOOL)
            return
        if op == "BitNot64" or op in _INT64_BINARY_OPS:
            for a in args:
                push(a, D_BV64)
            return

        # Boolean logic.
        if op in ("LogicalAnd", "LogicalOr", "LogicalNot"):
            for a in args:
                push(a, D_BOOL)
            return

        # Memory / texture reads.
        if op == "PsbLoad" and args:
            push(args[0], D_BV64)
            if len(args) > 1:
                push(args[1], D_SYM)
            return
        if op in _TEX_OPS:
            data_args = args[:-1] if len(args) > 1 else args
            for a in data_args:
                push(a, D_TOP)
            if len(args) > 1:
                push(args[-1], D_SYM)
            return
        if op == "ImageQuerySize" and args:
            push(args[0], D_BV32)
            return


def annotate_ssa_demands(
    ssa_json: dict[str, Any],
    annotations: DemandAnnotations,
    *,
    in_place: bool = False,
) -> dict[str, Any]:
    """
    Tag every SSA node in `ssa_json` with its inferred demand type (`demand_type`).

    Returns a shallow-copied module dict (unless `in_place=True`) so callers can
    inspect or dump `node["demand_type"]` without mutating original bundles.
    """
    out = ssa_json if in_place else copy.copy(ssa_json)
    nodes = ssa_json.get("nodes", [])
    if in_place:
        for n in nodes:
            n["demand_type"] = annotations.tag_of(int(n["id"])).label
    else:
        tagged_nodes = []
        for n in nodes:
            cn = dict(n)
            cn["demand_type"] = annotations.tag_of(int(n["id"])).label
            tagged_nodes.append(cn)
        out["nodes"] = tagged_nodes
    return out


def merge_miter_demands(
    ann_a: DemandAnnotations,
    ann_b: DemandAnnotations,
) -> tuple[dict[str, DemandTag], dict[str, DemandTag]]:
    """
    Compute the joint two-shader demand maps `(shared_input_demands, shared_output_demands)`
    per Theorem (2) of Section 7.3:

      Delta_shared(x) = Delta_1*(x) |_| Delta_2*(x)

    Guarantees identical Z3 sorts across both sides of an equivalence miter:
    purely FP shaders emit pure `Real` (`QF_NRA`) with zero bridges, while
    mixed FP/bit-hack miters escalate only the conflicted symbols/slots to `TOP_BITS`.
    """
    shared_in: dict[str, DemandTag] = {}
    for k in set(ann_a.input_demands) | set(ann_b.input_demands):
        ta = ann_a.input_demands.get(k, D_UNUSED)
        tb = ann_b.input_demands.get(k, D_UNUSED)
        shared_in[k] = demand_join(ta, tb)

    shared_out: dict[str, DemandTag] = {}
    for k in set(ann_a.output_demands) | set(ann_b.output_demands):
        ta = ann_a.output_demands.get(k, D_UNUSED)
        tb = ann_b.output_demands.get(k, D_UNUSED)
        shared_out[k] = demand_join(ta, tb)

    return shared_in, shared_out

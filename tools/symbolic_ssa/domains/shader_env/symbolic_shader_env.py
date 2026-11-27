"""
Domain-Aware Symbolic Shader Environment (`SymbolicShaderEnv`).

Initialized with the requested `Z3SymbolicDomain` (e.g., `SymbolicZ3UDomain`,
`SymbolicZ3BitvecDomain`, or any custom domain) so symbolic inputs, FAU loads,
and buffer/image arrays stay aligned with the interpreter's analysis domain
instead of constructing ad-hoc raw Z3 expressions.

Supports:
  - Demand-driven backfill (`shared_demands` / `shared_output_demands`) for
    bridge-free `Real` / `Bool` / `BitVec` inputs and per-offset read-only
    UBO/push-constant slots.
  - Configurable memory granularity (`memory_model="word" | "byte"`) and
    optional cross-binding SSBO aliasing (`allow_ssbo_aliasing=True`).
  - Deterministic symmetric havoc helpers for `ControlBarrier` epochs and
    atomic RMW pre-values (`get_barrier_buf_mem`, `get_atomic_old_val`).
"""

from __future__ import annotations

from typing import Any

from tools.symbolic_ssa import z3utils
from tools.symbolic_ssa.domains.backfill import DemandKind, DemandTag
from tools.symbolic_ssa.domains.shader_env.base import MASK32, MASK64, ShaderEnvBase
from tools.symbolic_ssa.domains.z3_base import Z3SymbolicDomain, make_z3_symbolic_domain
from tools.symbolic_ssa.z3utils import dsl
from tools.symbolic_ssa.z3utils.helpers import (
    BV8,
    BV32,
    BV64,
    Z3Val,
    _dsl_bv32,
    _try_const_u64,
)


def is_readonly_bkey(bkey: str) -> bool:
    """Return True if a buffer key denotes an immutable uniform/push-constant region."""
    b = bkey.lower()
    return b.startswith(("ubo:", "push_const", "uniform:", "ro:"))


def is_shared_mutable_bkey(bkey: str) -> bool:
    """Return True if a buffer key can be concurrently modified by other invocations."""
    b = bkey.lower()
    return b == "wls_shared" or b.startswith(("workgroup:", "ssbo:"))


class SymbolicShaderEnv(ShaderEnvBase):
    """Shared symbolic input & buffer address environment initialized with a `Z3SymbolicDomain`."""

    def __init__(
        self,
        bundle: dict[str, Any],
        domain: str | Z3SymbolicDomain | None = None,
        fp_mode: str = "exact",
        *,
        memory_model: str = "word",
        allow_ssbo_aliasing: bool = False,
        shared_demands: dict[str, DemandTag] | None = None,
        shared_output_demands: dict[str, DemandTag] | None = None,
        strict_fp_edges: bool = False,
    ):
        if memory_model not in ("word", "byte"):
            raise ValueError(f"Unsupported memory_model: {memory_model!r} (expected 'word' or 'byte')")
        self._init_abi_layout(bundle)
        self.domain: Z3SymbolicDomain = make_z3_symbolic_domain(
            domain, fp_mode=fp_mode, strict_fp_edges=strict_fp_edges
        )
        self.memory_model: str = memory_model
        self.allow_ssbo_aliasing: bool = bool(allow_ssbo_aliasing)
        self.shared_demands: dict[str, DemandTag] = dict(shared_demands) if shared_demands else {}
        self.shared_output_demands: dict[str, DemandTag] = (
            dict(shared_output_demands) if shared_output_demands else {}
        )
        self.inputs: dict[str, Any] = {}
        self.init_buf_mem: dict[str, Any] = {}
        self.init_img_mem: dict[str, tuple[Any, ...]] = {}
        self._barrier_buf_mems: dict[tuple[str, int], Any] = {}
        self._atomic_old_ufs: dict[tuple[str, int, int], Any] = {}
        self._ssbo_sym_bases: dict[str, Any] = {}
        self.ssbo_shared_mem: Any | None = None
        self.fallback_mem: Any = dsl.Array("mem_fallback", BV64, BV32)
        self.fau_sym: Any = dsl.Array("fau_sym", BV32, BV32)
        self.lane_id: Any = dsl.BitVec("lane_id", 5)
        self.inputs["lane_id"] = self.lane_id
        self._pilot_executed: bool = False
        self.pilot_exec_fau_words: dict[int, Any] = {}


    def set_shared_demands(
        self,
        input_demands: dict[str, DemandTag],
        output_demands: dict[str, DemandTag] | None = None,
    ) -> None:
        """Install joint two-shader demand maps computed by `merge_miter_demands`."""
        self.shared_demands = dict(input_demands)
        if output_demands is not None:
            self.shared_output_demands = dict(output_demands)

    def read_untyped_slot(self, domain: Z3SymbolicDomain | None, raw_bv32: Any) -> Any:
        """Lift an untyped 32-bit shader input/memory word into `domain` (or `self.domain`)."""
        dom = domain if domain is not None else self.domain
        if hasattr(dom, "lift_z3_bv32"):
            return dom.lift_z3_bv32(raw_bv32)
        return raw_bv32

    def _demand_for(self, key: str) -> DemandTag | None:
        return self.shared_demands.get(key)

    def _make_demanded_scalar(self, sym_name: str, demand_key: str) -> Any:
        """Materialize a scalar input with the precise Z3 sort requested by `shared_demands`."""
        tag = self._demand_for(demand_key)
        if tag is not None and tag.kind is DemandKind.F32 and hasattr(self.domain, "lift_z3_real_f32"):
            rv = dsl.Real(f"{sym_name}_f32")
            self.inputs[f"{sym_name}_f32"] = rv
            return self.domain.lift_z3_real_f32(rv)
        if tag is not None and tag.kind is DemandKind.BOOL:
            bv = dsl.Bool(f"{sym_name}_b")
            self.inputs[f"{sym_name}_b"] = bv
            return self.domain.lift_z3_bool(bv)
        v = dsl.BitVec(sym_name, 32)
        self.inputs[sym_name] = v
        return v

    def get_ro_buf_slot_value(self, bkey: str, const_off: int) -> Any | None:
        """
        Under demand-driven backfill, return a bridge-free scalar `Real`/`Bool`
        symbol for a read-only `(bkey, const_off)` slot when demanded precisely.
        Returns `None` when the slot is mutable, multi-typed (`TOP_BITS`), or
        not in backfill mode.
        """
        if bkey == "ebin_rodata":
            return self.domain.lift_z3_bv32(_dsl_bv32(self.read_ebin_rodata_u32(const_off)))
        if bkey == "panfrost_ebin_rodata":
            return self.domain.lift_z3_bv32(_dsl_bv32(self.read_panfrost_ebin_rodata_u32(const_off)))
        srt_val = self.read_pilot_srt_u32(bkey, const_off)
        if srt_val is not None:
            return self.domain.lift_z3_bv32(_dsl_bv32(srt_val))
        if not self.shared_demands or not is_readonly_bkey(bkey):
            return None
        slot_key = f"buf_slot:{bkey}:{const_off}"
        tag = self._demand_for(slot_key)
        if tag is None:
            return None
        safe = bkey.replace(":", "_")
        if tag.kind is DemandKind.F32 and hasattr(self.domain, "lift_z3_real_f32"):
            sym = f"ro_{safe}_0x{const_off:x}_f32"
            rv = dsl.Real(sym)
            self.inputs[sym] = rv
            return self.domain.lift_z3_real_f32(rv)
        if tag.kind is DemandKind.BOOL:
            sym = f"ro_{safe}_0x{const_off:x}_b"
            bv = dsl.Bool(sym)
            self.inputs[sym] = bv
            return self.domain.lift_z3_bool(bv)
        return None

    def is_shared_sym_bkey(self, bkey: str) -> bool:
        return bkey in getattr(self, "shared_sym_bkeys", set())

    def get_sym_buf_slot(self, sym_tag: str, off32: Any) -> Any:
        if not hasattr(self, "sym_buf_ufs"):
            self.sym_buf_ufs: dict[str, Any] = {}
        if sym_tag not in self.sym_buf_ufs:
            safe = "".join(c if c.isalnum() or c == "_" else "_" for c in sym_tag)
            self.sym_buf_ufs[sym_tag] = dsl.Function(f"sym_slot_{safe}", BV32, BV32)
        return self.sym_buf_ufs[sym_tag](off32)

    def get_buf_mem(self, key: str) -> Any:
        if key == "wls_shared" and self.unique_workgroup_key is not None:
            key = self.unique_workgroup_key
        if key not in self.init_buf_mem:
            if key == "panfrost_ebin_rodata" and self._panfrost_ebin_rodata_bytes:
                if self.memory_model == "byte":
                    arr = dsl.K(BV32, dsl.BitVecVal(0, 8))
                    for i, bval in enumerate(self._panfrost_ebin_rodata_bytes):
                        if bval != 0:
                            arr = dsl.Store(arr, _dsl_bv32(self._panfrost_ebin_rodata_offset + i), dsl.BitVecVal(bval, 8))
                else:
                    arr = dsl.K(BV32, _dsl_bv32(0))
                    n = len(self._panfrost_ebin_rodata_bytes)
                    for i in range(0, n, 4):
                        w = self.read_panfrost_ebin_rodata_u32(self._panfrost_ebin_rodata_offset + i)
                        if w != 0:
                            arr = dsl.Store(arr, _dsl_bv32(self._panfrost_ebin_rodata_offset + i), _dsl_bv32(w))
                self.init_buf_mem[key] = arr
            elif key == "ebin_rodata" and self._ebin_rodata_bytes:
                if self.memory_model == "byte":
                    arr = dsl.K(BV32, dsl.BitVecVal(0, 8))
                    for i, bval in enumerate(self._ebin_rodata_bytes):
                        if bval != 0:
                            arr = dsl.Store(arr, _dsl_bv32(self._ebin_rodata_offset + i), dsl.BitVecVal(bval, 8))
                else:
                    arr = dsl.K(BV32, _dsl_bv32(0))
                    n = len(self._ebin_rodata_bytes)
                    for i in range(0, n, 4):
                        w = self.read_ebin_rodata_u32(self._ebin_rodata_offset + i)
                        if w != 0:
                            arr = dsl.Store(arr, _dsl_bv32(self._ebin_rodata_offset + i), _dsl_bv32(w))
                self.init_buf_mem[key] = arr
            else:
                safe = key.replace(":", "_")
                val_sort = BV8 if self.memory_model == "byte" else BV32
                self.init_buf_mem[key] = dsl.Array(f"mem_{safe}", BV32, val_sort)
        return self.init_buf_mem[key]

    def get_ssbo_shared_mem(self) -> Any:
        """Shared flat 64-bit address space used when `allow_ssbo_aliasing=True`."""
        if self.ssbo_shared_mem is None:
            val_sort = BV8 if self.memory_model == "byte" else BV32
            self.ssbo_shared_mem = dsl.Array("mem_ssbo_shared", BV64, val_sort)
        return self.ssbo_shared_mem

    def get_ssbo_sym_base(self, bkey: str) -> Any:
        """Symbolic 64-bit base address for an SSBO binding when `allow_ssbo_aliasing=True`."""
        if bkey not in self._ssbo_sym_bases:
            safe = bkey.replace(":", "_")
            sym = dsl.BitVec(f"sym_base_{safe}", 64)
            self._ssbo_sym_bases[bkey] = sym
            self.inputs[f"sym_base_{safe}"] = sym
        return self._ssbo_sym_bases[bkey]

    def read_buf_u32(self, mem: Any, off: Any) -> Any:
        """Read a 32-bit word from `mem` at byte offset `off` according to `memory_model`."""
        off_d = off if isinstance(off, dsl.Z3Expr) else dsl.view(off)
        if self.memory_model == "byte":
            w = off_d.size() if hasattr(off_d, "size") else 32
            b0 = dsl.Select(mem, off_d)
            b1 = dsl.Select(mem, off_d + dsl.BitVecVal(1, w))
            b2 = dsl.Select(mem, off_d + dsl.BitVecVal(2, w))
            b3 = dsl.Select(mem, off_d + dsl.BitVecVal(3, w))
            return dsl.Concat(b3, b2, b1, b0)
        return dsl.Select(mem, off_d)

    def write_buf_u32(self, mem: Any, off: Any, val32: Any) -> Any:
        """Write a 32-bit word `val32` to `mem` at byte offset `off` according to `memory_model`."""
        off_d = off if isinstance(off, dsl.Z3Expr) else dsl.view(off)
        val_d = val32 if isinstance(val32, dsl.Z3Expr) else dsl.view(val32)
        if self.memory_model == "byte":
            w = off_d.size() if hasattr(off_d, "size") else 32
            m1 = dsl.Store(mem, off_d, dsl.Extract(7, 0, val_d))
            m2 = dsl.Store(m1, off_d + dsl.BitVecVal(1, w), dsl.Extract(15, 8, val_d))
            m3 = dsl.Store(m2, off_d + dsl.BitVecVal(2, w), dsl.Extract(23, 16, val_d))
            return dsl.Store(m3, off_d + dsl.BitVecVal(3, w), dsl.Extract(31, 24, val_d))
        return dsl.Store(mem, off_d, val_d)

    def get_barrier_buf_mem(self, key: str, epoch: int) -> Any:
        """
        Return the deterministic fresh post-barrier memory array for `(key, epoch)`.
        Shared symmetrically across Direct and Valhall interpreters so equivalent
        shaders see the same havoced post-barrier state while preventing false
        local-store forwarding across `ControlBarrier`.
        """
        if key == "wls_shared" and self.unique_workgroup_key is not None:
            key = self.unique_workgroup_key
        ckey = (key, int(epoch))
        if ckey not in self._barrier_buf_mems:
            safe = key.replace(":", "_")
            val_sort = BV8 if self.memory_model == "byte" else BV32
            idx_sort = BV64 if (self.allow_ssbo_aliasing and key == "ssbo_shared") else BV32
            self._barrier_buf_mems[ckey] = dsl.Array(f"mem_{safe}_bar{epoch}", idx_sort, val_sort)
        return self._barrier_buf_mems[ckey]

    def get_atomic_old_val(self, key: str, seq_idx: int, off: Any) -> Any:
        """
        Return a deterministic fresh havoced pre-atomic value representing arbitrary
        interleaving from other invocations at atomic sequence index `seq_idx`.
        """
        if key == "wls_shared" and self.unique_workgroup_key is not None:
            key = self.unique_workgroup_key
        off_bits = off.size() if hasattr(off, "size") else 32
        ckey = (key, int(seq_idx), int(off_bits))
        if ckey not in self._atomic_old_ufs:
            safe = key.replace(":", "_")
            idx_sort = BV64 if off_bits == 64 else BV32
            self._atomic_old_ufs[ckey] = dsl.Function(f"atomic_old_{safe}_{seq_idx}", idx_sort, BV32)
        return self._atomic_old_ufs[ckey](off)

    def build_preconditions(self) -> list[Any]:
        """Build semantic execution-environment preconditions (e.g. Compute LocalSize bounds)."""
        conds: list[Any] = []
        if self.stage == "Compute" and self.local_size is not None:
            lx, ly, lz = self.local_size
            for c, lim in enumerate((lx, ly, lz)):
                lid_k = f"bi_27_{c}"
                if lid_k in self.inputs:
                    conds.append(dsl.ULT(self.inputs[lid_k], _dsl_bv32(lim)))
                gid_k = f"bi_28_{c}"
                wg_k = f"bi_26_{c}"
                if gid_k in self.inputs and lid_k in self.inputs and wg_k in self.inputs:
                    conds.append(self.inputs[gid_k] == self.inputs[wg_k] * _dsl_bv32(lim) + self.inputs[lid_k])
            if "bi_29" in self.inputs:
                if all(f"bi_27_{c}" in self.inputs for c in range(3)):
                    lidx_expr = (
                        self.inputs["bi_27_2"] * _dsl_bv32(lx * ly)
                        + self.inputs["bi_27_1"] * _dsl_bv32(lx)
                        + self.inputs["bi_27_0"]
                    )
                    conds.append(self.inputs["bi_29"] == lidx_expr)
                else:
                    conds.append(dsl.ULT(self.inputs["bi_29"], _dsl_bv32(lx * ly * lz)))
        if "gl_SamplePosition" in self.init_buf_mem:
            # Mali's `sample` sideband table packs (x, y) as two Q8.8 signed 16-bit halfwords
            # in a single 32-bit word (`h0 = x * 256`, `h1 = y * 256`), matching ConcreteShaderEnv.
            sp_mem = self.init_buf_mem["gl_SamplePosition"]
            sp_word = self.read_buf_u32(sp_mem, _dsl_bv32(0))
            inv_256 = self.domain.const_u32(0x3B800000)
            if "bi_19_0" in self.inputs:
                sx_s32 = self.domain.lift_z3_bv32(dsl.SignExt(16, dsl.Extract(15, 0, sp_word)))
                sx_f32 = self.domain.fp_binary("FMul", self.domain.cast_value("ConvertSToF", sx_s32, "f32"), inv_256)
                conds.append(self.domain.lower_z3_bv32(sx_f32) == self.inputs["bi_19_0"])
            if "bi_19_1" in self.inputs:
                sy_s32 = self.domain.lift_z3_bv32(dsl.SignExt(16, dsl.Extract(31, 16, sp_word)))
                sy_f32 = self.domain.fp_binary("FMul", self.domain.cast_value("ConvertSToF", sy_s32, "f32"), inv_256)
                conds.append(self.domain.lower_z3_bv32(sy_f32) == self.inputs["bi_19_1"])
        return conds

    def get_img_mem(self, key: str) -> tuple[Any, ...]:
        if key not in self.init_img_mem:
            safe = key.replace(":", "_")
            self.init_img_mem[key] = tuple(dsl.Array(f"img_{safe}_c{c}", BV64, BV32) for c in range(4))
        return self.init_img_mem[key]

    def _find_base_region_in_add(self, expr: Any) -> str | None:
        found: set[str] = set()
        stack: list[Any] = [expr]
        visited = 0
        while stack and visited < 64:
            cur = stack.pop()
            visited += 1
            if dsl.is_bv_value(cur) and cur.size() == 64:
                bkey = self.resolve_region_hi32(dsl.as_long(cur) >> 32)
                if bkey is not None:
                    found.add(bkey)
                    if len(found) > 1:
                        return None
            elif dsl.is_app(cur) and cur.decl().kind() == dsl.Z3OpKind.CONCAT and cur.num_args() == 2 and cur.arg(0).size() == 32:
                hi_c = _try_const_u64(z3utils.simplify(cur.arg(0), relift=True))
                if hi_c is not None:
                    bkey = self.resolve_region_hi32(hi_c)
                    if bkey is not None:
                        found.add(bkey)
                        if len(found) > 1:
                            return None
            elif isinstance(cur, dsl.Z3OpBvAdd):
                stack.extend(cur.children())
        return next(iter(found)) if len(found) == 1 else None

    def resolve_addr(
        self,
        addr_expr: Any,
        hint_key: str = "",
    ) -> tuple[str | None, int | None, Any]:
        """Decompose a 64-bit address into (buf_key, const_offset_or_None, offset_bv32)."""
        c = _try_const_u64(addr_expr)
        if c is not None:
            off = c & MASK32
            bkey = self.resolve_region_hi32(c >> 32)
            if bkey is not None:
                return bkey, off, _dsl_bv32(off)
            if hint_key:
                return hint_key, off, _dsl_bv32(off)
            return None, c, _dsl_bv32(off)

        hi = _try_const_u64(z3utils.simplify(dsl.Extract(63, 32, addr_expr), relift=True))
        if hi is not None:
            bkey = self.resolve_region_hi32(hi)
            if bkey is not None:
                lo = z3utils.simplify(dsl.Extract(31, 0, addr_expr), relift=True)
                lo_c = _try_const_u64(lo)
                return bkey, lo_c, lo

        bkey = self._find_base_region_in_add(addr_expr)
        if bkey is not None:
            lo = z3utils.simplify(dsl.Extract(31, 0, addr_expr), relift=True)
            lo_c = _try_const_u64(lo)
            return bkey, lo_c, lo

        if hint_key:
            lo = z3utils.simplify(dsl.Extract(31, 0, addr_expr), relift=True)
            lo_c = _try_const_u64(lo)
            return hint_key, lo_c, lo

        return None, None, dsl.Extract(31, 0, addr_expr)

    def get_builtin(self, builtin_id: int, val_type: str) -> Z3Val:
        if builtin_id in (4424, 4425):
            return _dsl_bv32(0)
        name = f"bi_{builtin_id}"
        if val_type == "V4F32":
            return tuple(self._make_demanded_scalar(f"{name}_{c}", f"{name}_{c}") for c in range(4))
        if val_type == "V3U32":
            lanes = tuple(dsl.BitVec(f"{name}_{c}", 32) for c in range(3))
            for c, lv in enumerate(lanes):
                self.inputs[f"{name}_{c}"] = lv
            return lanes
        if val_type == "Bool":
            v = dsl.Bool(name)
            self.inputs[name] = v
            return v
        return self._make_demanded_scalar(name, name)

    def get_input_vec4(self, kind: str, location: int) -> tuple[Any, ...]:
        return tuple(
            self._make_demanded_scalar(f"{kind}_loc{location}_{c}", f"{kind}_loc{location}_{c}")
            for c in range(4)
        )

    def load_fau_word(self, word_idx: int) -> Any:
        if word_idx in self.fcst_by_word:
            return _dsl_bv32(self.fcst_by_word[word_idx])
        if word_idx in self.viewport_fau_words:
            return _dsl_bv32(self.viewport_fau_words[word_idx])
        if word_idx in self.pilot_by_word:
            pf, is_hi = self.pilot_by_word[word_idx]
            vkind = pf.get("value_kind", "")
            if vkind == "const_int" and pf.get("has_const_value", False):
                cv = int(pf["const_value"])
                return _dsl_bv32((cv >> 32) & MASK32) if is_hi else _dsl_bv32(cv & MASK32)
            if vkind in ("buffer_size", "descriptor_addr", "abi_preload", "unknown"):
                self._ensure_pilot_executed()
                if word_idx in self.pilot_exec_fau_words:
                    return self.pilot_exec_fau_words[word_idx]
                return dsl.Select(self.fau_sym, _dsl_bv32(word_idx))
            if vkind == "symbol_value" or pf.get("is_loaded_value", False):
                if pf.get("has_byte_offset", False) and not pf.get("is_derived", False) and int(pf.get("byte_size", 4)) >= 4:
                    bkey = str(pf["key"])
                    boff = int(pf["byte_offset"])
                    ro_val = self.get_ro_buf_slot_value(bkey, boff)
                    if ro_val is not None:
                        return ro_val
                    return self.read_buf_u32(self.get_buf_mem(bkey), _dsl_bv32(boff))
                self._ensure_pilot_executed()
                if word_idx in self.pilot_exec_fau_words:
                    return self.pilot_exec_fau_words[word_idx]
                return dsl.Select(self.fau_sym, _dsl_bv32(word_idx))
            if pf.get("is_derived", False):
                self._ensure_pilot_executed()
                if word_idx in self.pilot_exec_fau_words:
                    return self.pilot_exec_fau_words[word_idx]
                return dsl.Select(self.fau_sym, _dsl_bv32(word_idx))
            base = self.get_buf_base(pf["key"])
            if pf.get("has_pointer_offset", False):
                base = (base + int(pf.get("pointer_offset", 0))) & MASK64
            if not is_hi:
                return _dsl_bv32(base & MASK32)
            return _dsl_bv32((base >> 32) & MASK32)
        if word_idx in self.rloc_by_word:
            bkey, uoff = self.rloc_by_word[word_idx]
            ro_val = self.get_ro_buf_slot_value(bkey, uoff)
            if ro_val is not None:
                return ro_val
            return self.read_buf_u32(self.get_buf_mem(bkey), _dsl_bv32(uoff))
        self._ensure_pilot_executed()
        if word_idx in self.pilot_exec_fau_words:
            return self.pilot_exec_fau_words[word_idx]
        return dsl.Select(self.fau_sym, _dsl_bv32(word_idx))

    def _ensure_pilot_executed(self) -> None:
        """Lazily execute `bundle['pilot_ssa']` once symbolically to compute derived FAU words."""
        if self._pilot_executed:
            return
        self._pilot_executed = True
        pilot_ssa = self.bundle.get("pilot_ssa")
        if not pilot_ssa:
            return
        from tools.symbolic_ssa.symbolic_interpreter import SymbolicSsaInterpreter
        p_interp = SymbolicSsaInterpreter(pilot_ssa, self, is_valhall_side=True)
        p_state = p_interp.evaluate()
        for (bkey, byte_off), entry in p_state.const_stores.items():
            if bkey == "pilot_fau_0" and (byte_off & 3) == 0:
                val = entry[1] if isinstance(entry, tuple) and len(entry) == 2 else entry
                self.pilot_exec_fau_words[byte_off // 4] = val

"""
Concrete Shader Execution Environment (`ConcreteShaderEnv`).

Deterministic concrete input, FAU, buffer, image, and UF oracle aligned
with `SymbolicShaderEnv`, with zero Z3 dependency.
"""

from __future__ import annotations

import random
from typing import Any

from tools.symbolic_ssa.domains.shader_env.base import (
    _CORNER_U32S,
    MASK32,
    MASK64,
    ShaderEnvBase,
    _det_hash_u32,
    _f32_to_u32,
)


class ConcreteShaderEnv(ShaderEnvBase):
    """Shared concrete input & memory oracle aligned with `SymbolicShaderEnv`."""

    def __init__(
        self,
        bundle: dict[str, Any],
        vec_idx: int,
        seed: int = 0x5A17,
        overrides: dict[str, Any] | None = None,
    ):
        self._init_abi_layout(bundle)
        self.vec_idx = vec_idx
        self.seed = seed
        self.overrides: dict[str, Any] = overrides if overrides is not None else {}
        self.lane_id = int(self.overrides.get("lane_id", vec_idx & 31)) & 31

    def sample_u32(self, name: str) -> int:
        if name in self.overrides:
            v = self.overrides[name]
            return (1 if v else 0) if isinstance(v, bool) else (int(v) & MASK32)
        if self.vec_idx < len(_CORNER_U32S):
            h = _det_hash_u32(name, self.seed, self.vec_idx)
            if (h & 3) == 0:
                return _CORNER_U32S[self.vec_idx]
            return _CORNER_U32S[(self.vec_idx + h) % len(_CORNER_U32S)]
        rng = random.Random(_det_hash_u32(name, self.seed, self.vec_idx))
        if (self.vec_idx & 3) != 3:
            return _f32_to_u32(rng.uniform(-16.0, 16.0))
        return rng.getrandbits(32)

    def read_init_buf(self, bkey: str, off32: int) -> int:
        if bkey == "wls_shared" and self.unique_workgroup_key is not None:
            bkey = self.unique_workgroup_key
        key = f"mem:{bkey}:{off32 & MASK32}"
        if key in self.overrides:
            return int(self.overrides[key]) & MASK32
        def_key = f"mem_default:{bkey}"
        if def_key in self.overrides:
            return int(self.overrides[def_key]) & MASK32
        return self.sample_u32(key)

    def read_sym_buf_slot(self, sym_tag: str, off32: int) -> int:
        o = int(off32) & MASK32
        key = f"sym_slot:{sym_tag}:{o}"
        if key in self.overrides:
            return int(self.overrides[key]) & MASK32
        dkey = f"sym_slot_default:{sym_tag}"
        if dkey in self.overrides:
            return int(self.overrides[dkey]) & MASK32
        return self.sample_u32(key)

    def read_init_img(self, ikey: str, xy64: int) -> tuple[int, int, int, int]:
        xy = int(xy64) & MASK64
        out = []
        for c in range(4):
            key = f"img:{ikey}:c{c}:{xy}"
            dkey = f"img_default:{ikey}:c{c}"
            if key in self.overrides:
                out.append(int(self.overrides[key]) & MASK32)
            elif dkey in self.overrides:
                out.append(int(self.overrides[dkey]) & MASK32)
            else:
                out.append(self.sample_u32(key))
        return (out[0], out[1], out[2], out[3])

    def read_fallback_mem(self, addr64: int) -> int:
        a = int(addr64) & MASK64
        key = f"mem_fallback:{a}"
        if key in self.overrides:
            return int(self.overrides[key]) & MASK32
        if "mem_fallback_default" in self.overrides:
            return int(self.overrides["mem_fallback_default"]) & MASK32
        return self.sample_u32(key)

    def eval_hw_uf(self, uf_name: str, *args: int, fallback_tag: str = "") -> int:
        spec = self.overrides.get(f"uf:{uf_name}")
        if isinstance(spec, dict):
            key = ",".join(str(int(a) & MASK32) for a in args)
            tbl = spec.get("table", {})
            if key in tbl:
                return int(tbl[key]) & MASK32
            if spec.get("default") is not None:
                return int(spec["default"]) & MASK32
        tag = fallback_tag or uf_name
        return _det_hash_u32(tag, self.seed, self.vec_idx, *(int(a) & MASK32 for a in args))

    def resolve_addr(self, addr64: int, hint_key: str = "") -> tuple[str | None, int]:
        region = (addr64 >> 32) << 32
        off = addr64 & MASK32
        if region in self.base_to_buf:
            return self.base_to_buf[region], off
        if hint_key:
            return hint_key, off
        return None, off

    def get_builtin(self, bi: int, vt: str) -> Any:
        if self.stage == "Compute" and self.local_size is not None:
            lx, ly, lz = self.local_size
            lid = (
                self.sample_u32("bi_27_0") % lx,
                self.sample_u32("bi_27_1") % ly,
                self.sample_u32("bi_27_2") % lz,
            )
            wg = (
                self.sample_u32("bi_26_0") & 0xFFFF,
                self.sample_u32("bi_26_1") & 0xFFFF,
                self.sample_u32("bi_26_2") & 0xFFFF,
            )
            if bi == 27:
                return lid
            if bi == 26:
                return wg
            if bi == 28:
                return (
                    (wg[0] * lx + lid[0]) & MASK32,
                    (wg[1] * ly + lid[1]) & MASK32,
                    (wg[2] * lz + lid[2]) & MASK32,
                )
            if bi == 29:
                return (lid[2] * (lx * ly) + lid[1] * lx + lid[0]) & MASK32
        if vt == "V4F32":
            return tuple(self.sample_u32(f"bi_{bi}_{c}") for c in range(4))
        if vt == "V3U32":
            return tuple(self.sample_u32(f"bi_{bi}_{c}") for c in range(3))
        if vt == "Bool":
            return (self.sample_u32(f"bi_{bi}") & 1) != 0
        return self.sample_u32(f"bi_{bi}")

    def get_input_vec4(self, kind: str, loc: int) -> tuple[int, ...]:
        return tuple(self.sample_u32(f"{kind}_loc{loc}_{c}") for c in range(4))

    def _sample_fau_word(self, word_idx: int) -> int:
        fkey = f"fau_sym_{word_idx}"
        if fkey in self.overrides:
            return int(self.overrides[fkey]) & MASK32
        if "fau_sym_default" in self.overrides:
            return int(self.overrides["fau_sym_default"]) & MASK32
        return self.sample_u32(fkey)

    def load_fau_word(self, word_idx: int) -> int:
        if word_idx in self.fcst_by_word:
            return self.fcst_by_word[word_idx]
        if word_idx in self.viewport_fau_words:
            return self.viewport_fau_words[word_idx]
        if word_idx in self.pilot_by_word:
            pf, is_hi = self.pilot_by_word[word_idx]
            vkind = pf.get("value_kind", "")
            if vkind == "const_int" and pf.get("has_const_value", False):
                cv = int(pf["const_value"])
                return ((cv >> 32) & MASK32) if is_hi else (cv & MASK32)
            if vkind in ("buffer_size", "descriptor_addr", "abi_preload", "unknown") or pf.get("is_derived", False):
                return self._sample_fau_word(word_idx)
            if vkind == "symbol_value" or pf.get("is_loaded_value", False):
                if pf.get("has_byte_offset", False):
                    return self.read_init_buf(pf["key"], int(pf["byte_offset"]))
                return self._sample_fau_word(word_idx)
            base = self.get_buf_base(pf["key"])
            if pf.get("has_pointer_offset", False):
                base = (base + int(pf.get("pointer_offset", 0))) & MASK64
            if not is_hi:
                return base & MASK32
            return (base >> 32) & MASK32
        if word_idx in self.rloc_by_word:
            bkey, uoff = self.rloc_by_word[word_idx]
            return self.read_init_buf(bkey, uoff)
        return self._sample_fau_word(word_idx)

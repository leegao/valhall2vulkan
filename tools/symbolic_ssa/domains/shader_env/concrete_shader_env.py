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
        fp_range: str = "unconstrained",
    ):
        self._init_abi_layout(bundle)
        self.vec_idx = vec_idx
        self.seed = seed
        self.overrides: dict[str, Any] = overrides if overrides is not None else {}
        self.fp_range = fp_range
        self.lane_id = int(self.overrides.get("lane_id", vec_idx & 31)) & 31
        self._pilot_executed: bool = False
        self.pilot_exec_fau_words: dict[int, int] = {}


    def sample_u32(self, name: str) -> int:
        if name in self.overrides:
            v = self.overrides[name]
            return (1 if v else 0) if isinstance(v, bool) else (int(v) & MASK32)
        if name in ("bi_4424", "bi_4425"):
            return 0
        if name in ("bi_19_0", "bi_19_1"):
            # Vulkan gl_SamplePosition lies on a subpixel grid in [0.0, 1.0);
            # using 1/256 granularity ensures exact Q8.8 fixed-point round-trip in Mali's sample table.
            h = _det_hash_u32(name, self.seed, self.vec_idx)
            return _f32_to_u32((h & 0xFF) / 256.0)
        if name in ("bi_15_0", "bi_15_1"):
            # Vulkan spec: gl_FragCoord.xy = floor(pixel_xy) + gl_SamplePosition.xy with pixel_xy >= 0.
            h = _det_hash_u32(name, self.seed, self.vec_idx)
            pix = float(h & 0xFF)
            sp_name = "bi_19_0" if name == "bi_15_0" else "bi_19_1"
            import struct as _st
            sp_val = _st.unpack("<f", _st.pack("<I", self.sample_u32(sp_name)))[0]
            return _f32_to_u32(pix + sp_val)
        if self.fp_range == "vulkan":
            rng = random.Random(_det_hash_u32(name, self.seed, self.vec_idx))
            if name in ("bi_42", "bi_43"):
                return rng.randrange(0, 64)
            _VULKAN_NICE_F32 = (-16.0, -4.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 16.0)
            if (self.vec_idx & 3) == 0 or rng.randrange(4) == 0:
                return _f32_to_u32(rng.choice(_VULKAN_NICE_F32))
            return _f32_to_u32(rng.uniform(-16.0, 16.0))
        if self.vec_idx < len(_CORNER_U32S):
            h = _det_hash_u32(name, self.seed, self.vec_idx)
            if (h & 3) == 0:
                return _CORNER_U32S[self.vec_idx]
            return _CORNER_U32S[(self.vec_idx + h) % len(_CORNER_U32S)]
        rng = random.Random(_det_hash_u32(name, self.seed, self.vec_idx))
        if (self.vec_idx & 3) != 3:
            return _f32_to_u32(rng.uniform(-16.0, 16.0))
        return rng.getrandbits(32)

    def _get_image_wh(self, tkey: str) -> tuple[int, int]:
        w = (_det_hash_u32(f"img_w:{tkey}", self.seed, self.vec_idx) & 0x3FF) + 1
        h = (_det_hash_u32(f"img_h:{tkey}", self.seed, self.vec_idx) & 0x3FF) + 1
        return (w, h)

    def read_pilot_srt_u32(self, bkey: str, off32: int) -> int | None:
        if bkey == "pilot_srt_rack_4":
            desc_idx = off32 // 32
            field_off = off32 % 32
            if field_off == 4 and self._pilot_srt_by_rack.get(4, {}).get(desc_idx) is None:
                tex_key = self._pilot_srt_by_rack.get(68, {}).get(desc_idx)
                if tex_key is not None:
                    w, h = self._get_image_wh(tex_key)
                    return (((h - 1) & 0xFFFF) << 16) | ((w - 1) & 0xFFFF)
        return super().read_pilot_srt_u32(bkey, off32)

    def read_init_buf(self, bkey: str, off32: int) -> int:
        if bkey == "gl_mali_non_rmu_buffer" and (int(off32) & 0xFFFFFFFF) in (76, 80, 88):
            return 0
        if bkey == "gl_SamplePosition":
            # Valhall's `sample` sideband table packs each sample's (x, y) as two signed 16-bit Q8.8
            # fixed-point integers in one 32-bit word (`h0 = x * 256`, `h1 = y * 256`), which the
            # shader unpacks via `S16_TO_F32(h0/h1) * (1/256)`.
            import math
            import struct
            ux = self.sample_u32("bi_19_0")
            uy = self.sample_u32("bi_19_1")
            fx = struct.unpack("<f", struct.pack("<I", ux))[0]
            fy = struct.unpack("<f", struct.pack("<I", uy))[0]
            sx = 0 if (math.isnan(fx) or math.isinf(fx)) else max(-32768, min(32767, round(fx * 256.0)))
            sy = 0 if (math.isnan(fy) or math.isinf(fy)) else max(-32768, min(32767, round(fy * 256.0)))
            return ((sy & 0xFFFF) << 16) | (sx & 0xFFFF)
        if bkey == "ebin_rodata":
            return self.read_ebin_rodata_u32(off32)
        srt_val = self.read_pilot_srt_u32(bkey, off32)
        if srt_val is not None:
            return srt_val
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
        base_uf = uf_name[3:] if uf_name.startswith("hw_") else uf_name
        if base_uf.startswith("imgq_") and "_c" in base_uf:
            body, _, comp_s = base_uf.rpartition("_c")
            if comp_s in ("0", "1"):
                parts = body[len("imgq_"):].split("_")
                if len(parts) >= 3 and parts[0] == "tex":
                    tkey = f"tex:{parts[1]}:{parts[2]}"
                    w, h = self._get_image_wh(tkey)
                    raw_lod = int(args[0]) & MASK32 if args else 0
                    if raw_lod & 0x80000000:
                        lod = 0
                    else:
                        lod = min(31, raw_lod)
                    dim = w if comp_s == "0" else h
                    return max(1, dim >> lod)
        norm_args = [int(a) & MASK32 for a in args]
        if base_uf.startswith("tex_texsamplelod_") and len(norm_args) in (3, 4):
            import math as _math
            import struct as _struct
            lod_idx = len(norm_args) - 1
            lod_u = norm_args[lod_idx]
            lod_f = _struct.unpack("<f", _struct.pack("<I", lod_u))[0]
            if _math.isnan(lod_f) or lod_f <= 0.0:
                norm_args[lod_idx] = 0
            elif _math.isinf(lod_f) or lod_f >= 127.99609375:
                norm_args[lod_idx] = 0x7FFF
            else:
                norm_args[lod_idx] = int(_math.trunc(lod_f * 256.0))
        elif base_uf.startswith("tex_texfetch_") and len(norm_args) >= 2:
            norm_args[0] &= 0xFFFF
            norm_args[1] &= 0xFFFF
        tag = fallback_tag or uf_name
        return _det_hash_u32(tag, self.seed, self.vec_idx, *norm_args)

    def resolve_addr(self, addr64: int, hint_key: str = "") -> tuple[str | None, int]:
        off = addr64 & MASK32
        if hint_key.startswith("tex:"):
            return hint_key, off
        bkey = self.resolve_region_hi32(addr64 >> 32)
        if bkey is not None:
            return bkey, off
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
        if bi == 40:
            return self.sample_u32("bi_40") & 0xF
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
                self._ensure_pilot_executed()
                if word_idx in self.pilot_exec_fau_words:
                    return self.pilot_exec_fau_words[word_idx]
                return self._sample_fau_word(word_idx)
            if vkind == "symbol_value" or pf.get("is_loaded_value", False):
                if pf.get("has_byte_offset", False) and int(pf.get("byte_size", 4)) >= 4:
                    return self.read_init_buf(pf["key"], int(pf["byte_offset"]))
                self._ensure_pilot_executed()
                if word_idx in self.pilot_exec_fau_words:
                    return self.pilot_exec_fau_words[word_idx]
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
        self._ensure_pilot_executed()
        if word_idx in self.pilot_exec_fau_words:
            return self.pilot_exec_fau_words[word_idx]
        return self._sample_fau_word(word_idx)

    def _ensure_pilot_executed(self) -> None:
        """Lazily execute `bundle['pilot_ssa']` once to compute all derived FAU words."""
        if self._pilot_executed:
            return
        self._pilot_executed = True
        pilot_ssa = self.bundle.get("pilot_ssa")
        if not pilot_ssa:
            return
        from tools.symbolic_ssa.concrete_interpreter import ConcreteSsaInterpreter
        p_interp = ConcreteSsaInterpreter(pilot_ssa, is_valhall_side=True)
        _, p_outs = p_interp.evaluate(self)
        prefix = "pilot_fau_0+0x"
        for k, v in p_outs.items():
            if k.startswith(prefix):
                byte_off = int(k[len(prefix):], 16)
                if (byte_off & 3) == 0:
                    self.pilot_exec_fau_words[byte_off // 4] = int(v) & MASK32

#!/usr/bin/env python3
"""
Abstract ABI / MBS2 / FAU / VA-region Shader Environment Base (`ShaderEnvBase`)
and pure IEEE-754 / scalar helper utilities (zero Z3 dependency).
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Any, Dict, List, Optional, Tuple

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def _vec_len(vt: str) -> int:
    if vt == "V4F32":
        return 4
    if vt == "V3U32":
        return 3
    if vt == "V2F32":
        return 2
    return 1


def _eval_elementwise(vt: str, args: List[Any], fn: Any) -> Any:
    lens = [len(a) for a in args if isinstance(a, tuple)]
    if not lens:
        return fn(*args)
    n = min(lens)
    out = []
    for i in range(n):
        lane_args = [a[i] if isinstance(a, tuple) else a for a in args]
        out.append(fn(*lane_args))
    return tuple(out)


def _u32_to_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & MASK32))[0]


def _f32_to_u32(f: float) -> int:
    if math.isnan(f):
        return 0x7FC00000
    try:
        return struct.unpack("<I", struct.pack("<f", f))[0]
    except OverflowError:
        return 0x7F800000 if f > 0 else 0xFF800000


def _s32(u: int) -> int:
    u &= MASK32
    return u - 0x100000000 if (u & 0x80000000) else u


def _s64(u: int) -> int:
    u &= MASK64
    return u - 0x10000000000000000 if (u & 0x8000000000000000) else u


def _f16_to_f32_u32(h: int) -> int:
    try:
        f = struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]
    except Exception:
        return 0
    return _f32_to_u32(f)


def _f32_to_f16_u16(u: int) -> int:
    f = _u32_to_f32(u)
    if math.isnan(f):
        return 0x7E00
    try:
        return struct.unpack("<H", struct.pack("<e", f))[0]
    except OverflowError:
        return 0x7C00 if f > 0 else 0xFC00


def _f32_ulp_diff(a_u32: int, b_u32: int) -> Optional[int]:
    """Return exact IEEE-754 float32 ULP distance, or None if either is NaN/Inf."""
    a = a_u32 & MASK32
    b = b_u32 & MASK32
    if a == b:
        return 0
    a_exp = (a >> 23) & 0xFF
    b_exp = (b >> 23) & 0xFF
    if a_exp == 0xFF or b_exp == 0xFF:
        return None
    if (a & 0x7FFFFFFF) == 0 and (b & 0x7FFFFFFF) == 0:
        return 0
    ai = (0x80000000 - a) if (a & 0x80000000) else (0x80000000 + a)
    bi = (0x80000000 - b) if (b & 0x80000000) else (0x80000000 + b)
    return abs(ai - bi)


def _det_hash_u32(tag: str, *ints: int) -> int:
    h = hashlib.blake2b(digest_size=4)
    h.update(tag.encode("utf-8"))
    for x in ints:
        h.update(int(x & MASK64).to_bytes(8, "little"))
    return int.from_bytes(h.digest(), "little")


def _c_u32(v: Any) -> int:
    if isinstance(v, tuple):
        return _c_u32(v[0]) if v else 0
    if isinstance(v, bool):
        return 1 if v else 0
    return int(v) & MASK32


def _c_u64(v: Any) -> int:
    if isinstance(v, tuple):
        return _c_u64(v[0]) if v else 0
    if isinstance(v, bool):
        return 1 if v else 0
    return int(v) & MASK64


_c_elemwise = _eval_elementwise
_c_vec_len = _vec_len


def _c_bool(v: Any) -> bool:
    if isinstance(v, tuple):
        return _c_bool(v[0]) if v else False
    return bool(v)


_CORNER_U32S = [
    0x00000000,  # +0.0 / 0
    0x3F800000,  # +1.0
    0xBF800000,  # -1.0
    0x3F000000,  # +0.5
    0x40000000,  # +2.0
    0xC0000000,  # -2.0
    0x40490FDB,  # pi
    0x3EAAAAAB,  # 1/3
    0x00000001,  # 1 / min subnormal
    0x00000002,  # 2
    0x00000007,  # 7
    0x00000010,  # 16
    0x80000000,  # -0.0 / INT_MIN
    0x7FFFFFFF,  # INT_MAX
    0xFFFFFFFF,  # -1 / UINT_MAX
]


class ShaderEnvBase:
    """Shared ABI / MBS2 / FAU / viewport / VA-region layout for concrete and symbolic envs."""

    def _init_abi_layout(self, bundle: Dict[str, Any]) -> None:
        self.bundle = bundle
        self.stage: str = bundle.get("stage", "Compute")
        self.mbs2: Dict[str, Any] = bundle.get("mbs2_context", {})
        self.direct_bindings: List[Dict[str, Any]] = bundle.get("direct_bindings", [])
        self.buf_base_map: Dict[str, int] = {}
        self.base_to_buf: Dict[int, str] = {}
        self._next_region = 1

        ls_raw = self.mbs2.get("local_size")
        if isinstance(ls_raw, list) and len(ls_raw) == 3 and all(int(x) > 0 for x in ls_raw):
            self.local_size: Optional[Tuple[int, int, int]] = (int(ls_raw[0]), int(ls_raw[1]), int(ls_raw[2]))
        else:
            self.local_size = None

        for b in self.direct_bindings:
            key = b.get("key", "")
            if b.get("kind") in ("UBO", "SSBO", "PushConstant") and key:
                self.get_buf_base(key)

        wg_keys: List[str] = []
        for n in bundle.get("direct_ssa", {}).get("nodes", []):
            astr = n.get("aux_str", "")
            if n.get("op") == "LoadSidebandU64" and astr.startswith("workgroup:"):
                if astr not in wg_keys:
                    wg_keys.append(astr)
        self.unique_workgroup_key: Optional[str] = wg_keys[0] if len(wg_keys) == 1 else None
        if self.unique_workgroup_key is not None:
            self.buf_base_map["wls_shared"] = self.get_buf_base(self.unique_workgroup_key)

        for pf in self.mbs2.get("pilot_fau", []):
            key = pf.get("key", "")
            if pf.get("resolved") and key:
                self.get_buf_base(key)
        for rl in self.mbs2.get("rlocs", []):
            self.get_buf_base(f"ubo:0:{rl.get('binding', 0)}")

        self.fcst_by_word: Dict[int, int] = {
            int(fc["word_idx"]): int(fc["value"]) & MASK32 for fc in self.mbs2.get("fcsts", [])
        }
        self.viewport_fau_words: Dict[int, int] = {}
        self.pilot_by_word: Dict[int, Tuple[Dict[str, Any], bool]] = {}
        for pf in self.mbs2.get("pilot_fau", []):
            vkind = pf.get("value_kind", "")
            is_const = vkind == "const_int" and pf.get("has_const_value", False)
            if is_const or (pf.get("resolved") and pf.get("key")):
                w = int(pf["word_idx"])
                is_hi = (int(pf["fau_offset"]) % 8) != 0
                self.pilot_by_word[w] = (pf, is_hi)
                is_buf_addr = (vkind == "buffer_addr") or (not vkind and not pf.get("is_loaded_value", False))
                if is_buf_addr and not is_hi and int(pf.get("byte_size", 4)) >= 8:
                    if (w + 1) not in self.pilot_by_word:
                        self.pilot_by_word[w + 1] = (pf, True)

        self.rloc_by_word: Dict[int, Tuple[str, int]] = {}
        for rl in self.mbs2.get("rlocs", []):
            sym = rl.get("symbol", "")
            w0 = int(rl["word_idx"])
            nwords = max(1, (int(rl.get("size", 4)) + 3) // 4)
            uoff = int(rl.get("uni_offset", 0))
            if sym == "gl_mali_ViewportTransform":
                for k in range(nwords):
                    off = uoff + k * 4
                    self.viewport_fau_words[w0 + k] = 0x3F800000 if off < 16 else 0x00000000
                continue
            bkey = f"ubo:0:{rl.get('binding', 0)}"
            for k in range(nwords):
                self.rloc_by_word[w0 + k] = (bkey, uoff + k * 4)

        self.sinp_addr_to_loc: Dict[int, int] = {
            int(s["addr"]): int(s["location"]) for s in self.mbs2.get("sinp", []) if int(s.get("location", -1)) >= 0
        }
        self.sout_addr_to_loc: Dict[int, int] = {
            int(s["addr"]): int(s["location"]) for s in self.mbs2.get("sout", []) if int(s.get("location", -1)) >= 0
        }

    def get_buf_base(self, key: str) -> int:
        if key not in self.buf_base_map:
            base = self._next_region << 32
            self._next_region += 1
            self.buf_base_map[key] = base
            self.base_to_buf[base] = key
        return self.buf_base_map[key]

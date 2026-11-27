"""
Abstract ABI / MBS2 / FAU / VA-region Shader Environment Base (`ShaderEnvBase`)
and pure IEEE-754 / scalar helper utilities (zero Z3 dependency).
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Any

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


def _eval_elementwise(vt: str, args: list[Any], fn: Any) -> Any:
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


def _f32_ulp_diff(a_u32: int, b_u32: int) -> int | None:
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

    def _init_abi_layout(self, bundle: dict[str, Any]) -> None:
        self.bundle = bundle
        self.stage: str = bundle.get("stage", "Compute")
        self.mbs2: dict[str, Any] = bundle.get("mbs2_context", {})
        self.direct_bindings: list[dict[str, Any]] = bundle.get("direct_bindings", [])
        self.buf_base_map: dict[str, int] = {}
        self.base_to_buf: dict[int, str] = {}
        self._next_region = 1

        ls_raw = self.mbs2.get("local_size")
        if isinstance(ls_raw, list) and len(ls_raw) == 3 and all(int(x) > 0 for x in ls_raw):
            self.local_size: tuple[int, int, int] | None = (int(ls_raw[0]), int(ls_raw[1]), int(ls_raw[2]))
        else:
            self.local_size = None

        for b in self.direct_bindings:
            key = b.get("key", "")
            if b.get("kind") in ("UBO", "SSBO", "PushConstant") and key:
                self.get_buf_base(key)

        wg_keys: list[str] = []
        for n in bundle.get("direct_ssa", {}).get("nodes", []):
            astr = n.get("aux_str", "")
            if n.get("op") == "LoadSidebandU64" and astr.startswith("workgroup:"):
                if astr not in wg_keys:
                    wg_keys.append(astr)
        self.unique_workgroup_key: str | None = wg_keys[0] if len(wg_keys) == 1 else None
        if self.unique_workgroup_key is not None:
            self.buf_base_map["wls_shared"] = self.get_buf_base(self.unique_workgroup_key)

        direct_keys = {str(b.get("key", "")) for b in self.direct_bindings if b.get("key")}
        has_push_const_only = (
            "push_const" in direct_keys
            or any(
                int(se.get("rack", -1)) == 0 and int(se.get("addr", -1)) == 0 and se.get("key") == "push_const"
                for se in self.mbs2.get("pilot_srt", [])
            )
        ) and ("ubo:0:0" not in direct_keys)

        for pf in self.mbs2.get("pilot_fau", []):
            if has_push_const_only and pf.get("key") == "ubo:0:0":
                pf = dict(pf, key="push_const")
            key = pf.get("key", "")
            if pf.get("resolved") and key:
                self.get_buf_base(key)
        for rl in self.mbs2.get("rlocs", []):
            self.get_buf_base(f"ubo:0:{rl.get('binding', 0)}")

        self.fcst_by_word: dict[int, int] = {
            int(fc["word_idx"]): int(fc["value"]) & MASK32 for fc in self.mbs2.get("fcsts", [])
        }
        self.viewport_fau_words: dict[int, int] = {}
        self.pilot_by_word: dict[int, tuple[dict[str, Any], bool]] = {}
        for pf in self.mbs2.get("pilot_fau", []):
            if has_push_const_only and pf.get("key") == "ubo:0:0":
                pf = dict(pf, key="push_const")
            # Mali Vertex pilot shaders load BaseVertex (byte_offset=76) and BaseInstance
            # (byte_offset=80) from the draw-parameter buffer and add them to r60/r61.
            # Direct SPIR-V uses zero-based gl_VertexIndex/gl_InstanceIndex, so bind these
            # unresolved Vertex draw-param FAU slots to 0.
            if (
                self.stage == "Vertex"
                and not pf.get("resolved")
                and pf.get("kind") == "unknown"
                and pf.get("has_byte_offset")
                and int(pf.get("byte_offset", -1)) in (76, 80)
            ):
                pf = dict(pf, value_kind="const_int", has_const_value=True, const_value=0)
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

        self.rloc_by_word: dict[int, tuple[str, int]] = {}
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
            bkey = str(rl.get("key") or f"ubo:0:{rl.get('binding', 0)}")
            if has_push_const_only and bkey == "ubo:0:0":
                bkey = "push_const"
            for k in range(nwords):
                self.rloc_by_word[w0 + k] = (bkey, uoff + k * 4)

        self.sinp_addr_to_loc: dict[int, int] = {}
        _sinp_entries = sorted(
            (int(s["addr"]), int(s["location"]))
            for s in self.mbs2.get("sinp", [])
            if int(s.get("location", -1)) >= 0
        )
        for _i, (_ad, _loc) in enumerate(_sinp_entries):
            self.sinp_addr_to_loc[_ad] = _loc
            if _i + 1 < len(_sinp_entries):
                _next_ad = _sinp_entries[_i + 1][0]
                for _k in range(1, _next_ad - _ad):
                    self.sinp_addr_to_loc.setdefault(_ad + _k, _loc + _k)
        self.sout_addr_to_loc: dict[int, int] = {
            int(s["addr"]): int(s["location"]) for s in self.mbs2.get("sout", []) if int(s.get("location", -1)) >= 0
        }

        self._pilot_srt_by_rack: dict[int, dict[int, str]] = {}
        for se in self.mbs2.get("pilot_srt", []):
            rk = int(se.get("rack", 0))
            ad = int(se.get("addr", 0))
            ky = str(se.get("key", ""))
            if ky:
                self._pilot_srt_by_rack.setdefault(rk, {})[ad] = ky
                self.get_buf_base(ky)

        ro = self.mbs2.get("ebin_rodata") or {}
        self._ebin_rodata_offset: int = int(ro.get("offset", 0))
        ro_hex = str(ro.get("hex", ""))
        self._ebin_rodata_bytes: bytes = bytes.fromhex(ro_hex) if ro_hex else b""
        if self._ebin_rodata_bytes:
            self.get_buf_base("ebin_rodata")

        pro = self.mbs2.get("panfrost_ebin_rodata") or {}
        self._panfrost_ebin_rodata_offset: int = int(pro.get("offset", 0))
        pro_hex = str(pro.get("hex", ""))
        self._panfrost_ebin_rodata_bytes: bytes = bytes.fromhex(pro_hex) if pro_hex else b""
        if self._panfrost_ebin_rodata_bytes:
            self.get_buf_base("panfrost_ebin_rodata")

    REGION_STRIDE_HI: int = 0x100

    def get_buf_base(self, key: str) -> int:
        if key not in self.buf_base_map:
            base = (self._next_region * self.REGION_STRIDE_HI) << 32
            self._next_region += 1
            self.buf_base_map[key] = base
            self.base_to_buf[base] = key
        return self.buf_base_map[key]

    def resolve_region_hi32(self, hi32: int) -> str | None:
        """Resolve a 64-bit pointer's upper 32 bits to its owning buffer key.

        Tolerates small carry/borrow deltas (within +/-0x40 * 4GB) introduced by
        signed `IADD.s64` / `SHADD.s64` or shifted unsigned 32-bit offsets without
        colliding with adjacent regions spaced by `REGION_STRIDE_HI` (0x100 * 4GB).
        """
        h = int(hi32) & MASK32
        exact = h << 32
        if exact in self.base_to_buf:
            return self.base_to_buf[exact]
        stride = self.REGION_STRIDE_HI
        rounded = (h + (stride // 2)) & ~(stride - 1)
        if rounded > 0 and abs(h - rounded) <= (stride // 4):
            cand = rounded << 32
            if cand in self.base_to_buf:
                return self.base_to_buf[cand]
        return None

    def read_pilot_srt_u32(self, bkey: str, off32: int) -> int | None:
        """Read a 32-bit word from virtual Pilot SRT tables (`pilot_srt_0` / `pilot_srt_rack_<r>`)."""
        if bkey == "pilot_srt_0":
            rack = off32 // 16
            rem = off32 % 16
            rbase = self.get_buf_base(f"pilot_srt_rack_{rack}")
            if rem == 0:
                return rbase & MASK32
            if rem == 4:
                return (rbase >> 32) & MASK32
            return 0
        if bkey.startswith("pilot_srt_rack_"):
            rack = int(bkey[len("pilot_srt_rack_"):])
            desc_idx = off32 // 32
            field_off = off32 % 32
            tkey = self._pilot_srt_by_rack.get(rack, {}).get(desc_idx)
            if tkey is not None:
                if field_off == 4:
                    return 0x10000
                bbase = self.get_buf_base(tkey)
                if field_off == 8:
                    return bbase & MASK32
                if field_off == 12:
                    return (bbase >> 32) & MASK32
            return 0
        return None

    def read_ebin_rodata_byte(self, off32: int) -> int:
        """Read one byte from the embedded EBIN data_section at EBIN-relative byte offset `off32`."""
        rel = (int(off32) & MASK32) - self._ebin_rodata_offset
        if 0 <= rel < len(self._ebin_rodata_bytes):
            return self._ebin_rodata_bytes[rel]
        return 0

    def read_ebin_rodata_u32(self, off32: int) -> int:
        """Read a little-endian 32-bit word from the embedded EBIN data_section at `off32`."""
        base = int(off32) & MASK32
        return (
            self.read_ebin_rodata_byte(base)
            | (self.read_ebin_rodata_byte(base + 1) << 8)
            | (self.read_ebin_rodata_byte(base + 2) << 16)
            | (self.read_ebin_rodata_byte(base + 3) << 24)
        )

    def read_panfrost_ebin_rodata_byte(self, off32: int) -> int:
        """Read one byte from Panfrost's embedded EBIN .rodata at EBIN-relative byte offset `off32`."""
        if not self._panfrost_ebin_rodata_bytes:
            return self.read_ebin_rodata_byte(off32)
        rel = (int(off32) & MASK32) - self._panfrost_ebin_rodata_offset
        if 0 <= rel < len(self._panfrost_ebin_rodata_bytes):
            return self._panfrost_ebin_rodata_bytes[rel]
        return 0

    def read_panfrost_ebin_rodata_u32(self, off32: int) -> int:
        """Read a little-endian 32-bit word from Panfrost's embedded EBIN .rodata at `off32`."""
        base = int(off32) & MASK32
        return (
            self.read_panfrost_ebin_rodata_byte(base)
            | (self.read_panfrost_ebin_rodata_byte(base + 1) << 8)
            | (self.read_panfrost_ebin_rodata_byte(base + 2) << 16)
            | (self.read_panfrost_ebin_rodata_byte(base + 3) << 24)
        )

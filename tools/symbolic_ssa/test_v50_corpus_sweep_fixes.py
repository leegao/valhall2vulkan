"""Checkpoint v50 regression tests for samples/ and spirv.tar.xz corpus sweep fixes.

Covers:
  - BUG-026: V1_split_persp_var_w in pass_gvn_dce.py must not fold explicit user
    gl_FragCoord.w multiplies (aux_bool=True on VecExtract from LD_VAR_SPECIAL.frag_w
    with Register src r61), while preserving split-varying folding (aux_bool=False).
  - BUG-027: TEX_SINGLE.explicit Q8.8 fixed-point LOD conversion in
    valhall_pass1_handlers.cpp + symmetric Q8.8 LOD quantization in ConcreteShaderEnv.
  - BUG-028: gl_mali_non_rmu_buffer recognition for stripped-name rack=0, addr=0x20000
    (binding=216) symbols in equiv_bundle.cpp so LD_PKA.i32 resolves 0x00000002.
"""

from __future__ import annotations

import struct
import unittest

from tools.symbolic_ssa.domains.shader_env.concrete_shader_env import ConcreteShaderEnv
from tools.symbolic_ssa.passes import canonicalize_ssa_module


def _f32_bits(val: float) -> int:
    return struct.unpack("<I", struct.pack("<f", val))[0]


class TestV50CorpusSweepFixes(unittest.TestCase):
    def test_bug026_explicit_frag_w_not_folded(self) -> None:
        """Explicit user gl_FragCoord.w (aux_bool=True) must not be folded by V1_split_persp_var_w."""
        ssa = {
            "blocks": [{"id": 0, "is_end": True, "nodes": [1, 2, 3, 4, 5, 6, 7]}],
            "nodes": [
                {"id": 1, "op": "LoadFragVarying", "type": "V4F32", "blk": 0, "args": [], "imm_u32": 1},
                {"id": 2, "op": "VecExtract", "type": "F32", "blk": 0, "args": [1], "imm_u32": 0},
                {
                    "id": 3,
                    "op": "LoadBuiltinInput",
                    "type": "V4F32",
                    "blk": 0,
                    "args": [],
                    "imm_u32": 15,
                    "aux_str": "gl_FragCoord",
                },
                {
                    "id": 4,
                    "op": "VecExtract",
                    "type": "F32",
                    "blk": 0,
                    "args": [3],
                    "imm_u32": 3,
                    "aux_bool": True,
                },
                {"id": 5, "op": "FMul", "type": "F32", "blk": 0, "args": [2, 4], "orig_pc": 0x68},
                {"id": 6, "op": "VecConstruct4", "type": "V4F32", "blk": 0, "args": [5, 5, 5, 5]},
                {"id": 7, "op": "StoreFragColor", "type": "Token", "blk": 0, "args": [6], "imm_u32": 0},
            ],
        }
        can, _ = canonicalize_ssa_module(ssa)
        ops = [n["op"] for n in can["nodes"]]
        self.assertIn("FMul", ops, "Explicit user gl_FragCoord.w FMul must be preserved")

    def test_bug027_texsamplelod_q8_8_quantization(self) -> None:
        """ConcreteShaderEnv quantizes TexSampleLod FP32 LOD to Mali's Q8.8 fixed-point."""
        env = ConcreteShaderEnv(bundle={}, seed=42, vec_idx=0, fp_range="vulkan")
        u, v = _f32_bits(0.25), _f32_bits(0.75)

        # Direct LOD 3.0 and Q8.8-roundtripped (768 / 256.0) must hash identically.
        h_direct = env.eval_hw_uf("hw_tex_texsamplelod_tex_0_2_a3_c0", u, v, _f32_bits(3.0))
        h_q8_8 = env.eval_hw_uf("hw_tex_texsamplelod_tex_0_2_a3_c0", u, v, _f32_bits(768.0 / 256.0))
        self.assertEqual(h_direct, h_q8_8)

        # Non-dyadic LOD 3.1 truncates to int(3.1 * 256) = 793 -> 793 / 256.0 = 3.09765625.
        h_non_dyadic = env.eval_hw_uf("hw_tex_texsamplelod_tex_0_2_a3_c0", u, v, _f32_bits(3.1))
        h_quant = env.eval_hw_uf("hw_tex_texsamplelod_tex_0_2_a3_c0", u, v, _f32_bits(793.0 / 256.0))
        self.assertEqual(h_non_dyadic, h_quant)

        # Negative and NaN LODs clamp to 0; distinct integer LODs stay distinct.
        h_zero = env.eval_hw_uf("hw_tex_texsamplelod_tex_0_2_a3_c0", u, v, _f32_bits(0.0))
        h_neg = env.eval_hw_uf("hw_tex_texsamplelod_tex_0_2_a3_c0", u, v, _f32_bits(-2.5))
        h_nan = env.eval_hw_uf("hw_tex_texsamplelod_tex_0_2_a3_c0", u, v, 0x7FC00000)
        self.assertEqual(h_zero, h_neg)
        self.assertEqual(h_zero, h_nan)
        self.assertNotEqual(h_zero, h_direct)

    def test_bug028_non_rmu_buffer_zero_draw_params(self) -> None:
        """gl_mali_non_rmu_buffer offsets 76/80/88 return 0 (BaseVertex/BaseInstance/DrawIndex)."""
        env = ConcreteShaderEnv(bundle={}, seed=42, vec_idx=0, fp_range="vulkan")
        for off in (76, 80, 88):
            self.assertEqual(env.read_init_buf("gl_mali_non_rmu_buffer", off), 0)


if __name__ == "__main__":
    unittest.main()

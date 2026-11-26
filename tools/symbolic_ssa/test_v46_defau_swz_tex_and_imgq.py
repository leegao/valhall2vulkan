"""Regression tests for Checkpoint v46 fixes:
  - BUG-016: Default --defau / --opt-ssa / --canonicalize + ViewportTransform RLOC & buf:0:2 BaseVertex/BaseInstance
  - BUG-017: TEX_SINGLE.texel_offset signed 8-bit dx/dy unpacking
  - BUG-018: SWZ.v2i16 / SWZ.v4i8 packed-SIMD destination write-mask (.h1 / .b1..b3) + TexFetch 16-bit coord masking + tex: hint_key resolve_addr precedence
  - BUG-019: ImageQuerySize (hw_imgq_tex_*) + pilot_srt simg + negative LOD clamping in ConcreteShaderEnv
"""

from __future__ import annotations

import glob
import os
import unittest

from tools.coverage.equiv_fuzzer import run_concolic_equiv_fuzz
from tools.symbolic_ssa.domains.shader_env.concrete_shader_env import ConcreteShaderEnv
from tools.symbolic_ssa.passes import canonicalize_ssa_module
from tools.equivalence.spvasm_io import load_equiv_bundle

BIN_PATH = "build/spirv2valhall"


class TestV46DefauSwzTexAndImgq(unittest.TestCase):
    def test_df118580_equiv_exact_under_default_defau_opt(self) -> None:
        """df118580 must pass EQUIV_EXACT across k_init=64 under default --defau --opt-ssa."""
        matches = glob.glob("../spirv_corpus/spirv_samples/df118580*.spv")
        if not matches:
            self.skipTest("spirv_corpus df118580 not present")
        b = load_equiv_bundle(matches[0], BIN_PATH)
        d_ssa, _ = canonicalize_ssa_module(b["direct_ssa"])
        v_ssa, _ = canonicalize_ssa_module(b["valhall_ssa"])
        ef = run_concolic_equiv_fuzz(b, d_ssa, v_ssa, k_init=64)
        self.assertEqual(ef.status, "EQUIV_EXACT")
        self.assertEqual(ef.diverging_vectors, 0)

    def test_comp_unique_03_equiv_exact_after_swz_texel_offset_and_resolve_addr(self) -> None:
        """comp_unique_03_fba18fbe.spv must pass EQUIV_EXACT (0/64+ divs, 7/7 blocks)."""
        spv = "samples/comp_unique_03_fba18fbe.spv"
        if not os.path.exists(spv):
            self.skipTest("comp_unique_03_fba18fbe.spv not present")
        b = load_equiv_bundle(spv, BIN_PATH)
        d_ssa, _ = canonicalize_ssa_module(b["direct_ssa"])
        v_ssa, _ = canonicalize_ssa_module(b["valhall_ssa"])
        ef = run_concolic_equiv_fuzz(b, d_ssa, v_ssa, k_init=64)
        self.assertEqual(ef.status, "EQUIV_EXACT")
        self.assertEqual(ef.diverging_vectors, 0)
        self.assertEqual(ef.blocks_reached, ef.total_blocks)

    def test_hw_imgq_and_texsamplelod_negative_lod_clamping(self) -> None:
        """ConcreteShaderEnv must strip hw_ prefix, parse multi-part imgq names, and clamp negative LOD to 0."""
        cenv = ConcreteShaderEnv({}, seed=42, vec_idx=1)
        w0 = cenv.eval_hw_uf("hw_imgq_tex_0_2_OpImageQuerySizeLod_c0", 0)
        w_neg = cenv.eval_hw_uf("hw_imgq_tex_0_2_OpImageQuerySizeLod_c0", 0xFFFFFFFF)
        self.assertGreaterEqual(w0, 1)
        self.assertLessEqual(w0, 1024)
        self.assertEqual(w0, w_neg)

        t_zero = cenv.eval_hw_uf("hw_tex_texsamplelod_tex:0:2_a3_c0", 0x3F000000, 0x3F000000, 0x00000000)
        t_neg = cenv.eval_hw_uf("hw_tex_texsamplelod_tex:0:2_a3_c0", 0x3F000000, 0x3F000000, 0xBF800000)
        self.assertEqual(t_zero, t_neg)

        f_pos = cenv.eval_hw_uf("hw_tex_texfetch_tex:0:5_a2_c0", 0x0000FC80, 0)
        f_neg = cenv.eval_hw_uf("hw_tex_texfetch_tex:0:5_a2_c0", 0xFFFFFC80, 0)
        self.assertEqual(f_pos, f_neg)

    def test_resolve_addr_tex_hint_precedence(self) -> None:
        """Direct OpImageWrite PsbStore with hint_key='tex:0:6' must not collide with another texture's hi32."""
        cenv = ConcreteShaderEnv({}, seed=42, vec_idx=10)
        hi3 = cenv.get_buf_base("tex:0:3") >> 32
        addr64 = (hi3 << 32) | 0x2F59
        bkey, off = cenv.resolve_addr(addr64, hint_key="tex:0:6")
        self.assertEqual(bkey, "tex:0:6")
        self.assertEqual(off, 0x2F59)


if __name__ == "__main__":
    unittest.main()

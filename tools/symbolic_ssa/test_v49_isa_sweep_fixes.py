"""Regression tests for Checkpoint v49 ISA sweep fixes (129/129 emitted & equiv)."""

from __future__ import annotations

import copy
import unittest

from tools.corpus_synth.isa_target_synth import synthesize_for_isa
from tools.coverage.equiv_fuzzer import run_concolic_equiv_fuzz
from tools.coverage.shrinker import DiffPolicy
from tools.coverage.slice_diff import format_slice_diff
from tools.symbolic_ssa import canonicalize_ssa_module


class TestV49IsaSweepFixes(unittest.TestCase):
    def test_fatan_assist_table_and_mux_equiv(self) -> None:
        for mn in ("FATAN_ASSIST", "FATAN_TABLE", "MUX"):
            res = synthesize_for_isa(mn, mode="spirv", verify_equiv=True, k_init=32, fp_range="vulkan", complexity="branchy")
            self.assertTrue(res.compiled_ok and res.target_emitted, f"{mn} failed emit")
            self.assertEqual(res.diverging_vectors, 0, f"{mn} diverged: {res.equiv_status}")

    def test_cube_projection_equiv_exact(self) -> None:
        for mn in ("CUBEFACE1", "CUBEFACE2", "CUBE_SSEL", "CUBE_TSEL", "FMA_RSCALE"):
            res = synthesize_for_isa(mn, mode="spirv", verify_equiv=True, k_init=32, fp_range="vulkan", complexity="branchy")
            self.assertTrue(res.compiled_ok and res.target_emitted, f"{mn} failed emit")
            self.assertEqual(res.equiv_status, "EQUIV_EXACT", f"{mn} expected EQUIV_EXACT")
            self.assertEqual(res.diverging_vectors, 0)

    def test_tex_gradient_grdesc_equiv_exact(self) -> None:
        res = synthesize_for_isa("TEX_GRADIENT", mode="spirv", verify_equiv=True, k_init=32, fp_range="vulkan", complexity="branchy")
        self.assertTrue(res.compiled_ok and res.target_emitted)
        self.assertEqual(res.equiv_status, "EQUIV_EXACT")
        self.assertEqual(res.diverging_vectors, 0)

    def test_icmp_multi_vertex_output_array_equiv_exact(self) -> None:
        res = synthesize_for_isa("ICMP_MULTI", mode="spirv", verify_equiv=True, k_init=32, fp_range="vulkan", complexity="branchy")
        self.assertTrue(res.compiled_ok and res.target_emitted)
        self.assertEqual(res.equiv_status, "EQUIV_EXACT")
        self.assertEqual(res.diverging_vectors, 0)

    def test_ld_pka_srt_table_descriptor_array_equiv_exact(self) -> None:
        res = synthesize_for_isa("LD_PKA", mode="spirv", verify_equiv=True, k_init=32, fp_range="vulkan", complexity="branchy")
        self.assertTrue(res.compiled_ok and res.target_emitted)
        self.assertEqual(res.equiv_status, "EQUIV_EXACT")
        self.assertEqual(res.diverging_vectors, 0)

    def test_clper_and_shift_xor_emit_and_equiv_exact(self) -> None:
        for mn in ("CLPER", "LSHIFT_XOR", "RSHIFT_XOR"):
            res = synthesize_for_isa(mn, mode="spirv", verify_equiv=True, k_init=32, fp_range="vulkan", complexity="branchy")
            self.assertTrue(res.compiled_ok and res.target_emitted, f"{mn} failed emit")
            self.assertEqual(res.equiv_status, "EQUIV_EXACT", f"{mn} expected EQUIV_EXACT")
            self.assertEqual(res.diverging_vectors, 0)

    def test_branchy_transcendental_ssbo_ulp_and_pow(self) -> None:
        for mn in ("FADD_IMM", "FADD_LSCALE", "FCOS_TABLE", "FLOGD", "FMUL_RSCALE", "FSIN_TABLE"):
            res = synthesize_for_isa(mn, mode="spirv", verify_equiv=True, k_init=32, fp_range="vulkan", complexity="branchy")
            self.assertTrue(res.compiled_ok and res.target_emitted, f"{mn} failed emit")
            self.assertIn(res.equiv_status, ("EQUIV_EXACT", "EQUIV_WITHIN_ULP"), f"{mn} status={res.equiv_status}")
            self.assertEqual(res.diverging_vectors, 0)

    def test_slice_diff_on_synthetic_divergence(self) -> None:
        import json
        import os
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            res = synthesize_for_isa("IADD", mode="spirv", verify_equiv=False, complexity="flat", out_dir=td)
            self.assertTrue(res.compiled_ok and res.spv_path)
            bpath = os.path.join(td, "bundle.json")
            subprocess.run(
                ["./build/spirv2valhall", "--opt-ssa", "--defau", "--dump-equiv-json", bpath, res.spv_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with open(bpath, encoding="utf-8") as f:
                bundle = json.load(f)
        d_ssa, _ = canonicalize_ssa_module(bundle["direct_ssa"])
        v_ssa, _ = canonicalize_ssa_module(copy.deepcopy(bundle["valhall_ssa"]))
        # Inject a deliberate IAdd -> ISub mutation on Valhall side to verify --slice-diff formatting.
        mutated = False
        for n in v_ssa.get("nodes", []):
            if n.get("op") == "IAdd":
                n["op"] = "ISub"
                mutated = True
                break
        self.assertTrue(mutated)
        ef = run_concolic_equiv_fuzz(bundle, d_ssa, v_ssa, k_init=4, policy=DiffPolicy(max_ulp=0))
        self.assertEqual(ef.status, "DIVERGED")
        self.assertIsNotNone(ef.first_divergence)
        lines = format_slice_diff(bundle, d_ssa, v_ssa, ef.first_divergence, depth=4)
        self.assertTrue(any("[SLICE-DIFF" in line for line in lines))


if __name__ == "__main__":
    unittest.main()

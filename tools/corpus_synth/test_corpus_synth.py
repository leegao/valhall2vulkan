#!/usr/bin/env python3
"""Unit tests for tools.corpus_synth greedy Valhall-ISA / SPIR-V corpus synthesizer."""

import os
import tempfile
import unittest

from tools.corpus_synth import (
    OP_CATALOG,
    emit_shader_bytes,
    synthesize_corpus,
)


class TestCorpusSynth(unittest.TestCase):
    def test_emit_valid_spv_header_single_and_multi_block(self) -> None:
        ops = OP_CATALOG[:8]
        b1 = emit_shader_bytes(ops, num_blocks=1, seed=1)
        b3 = emit_shader_bytes(ops, num_blocks=3, seed=2)
        self.assertEqual(b1[:4], b"\x03\x02\x23\x07")
        self.assertEqual(b3[:4], b"\x03\x02\x23\x07")
        self.assertGreater(len(b3), len(b1))

    def test_greedy_spirv_ops_mode_covers_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            res = synthesize_corpus(
                target="spirv-ops",
                num_shaders=5,
                num_blocks=1,
                num_insns=14,
                out_dir=td,
                candidates_per_shader=2,
            )
            self.assertEqual(len(res.shaders), 5)
            self.assertGreaterEqual(len(res.total_unique_features), 45)
            for rec in res.shaders:
                self.assertTrue(os.path.exists(rec.spv_path))

    def test_greedy_valhall_isa_mode_compiles_via_libmali(self) -> None:
        if not os.path.exists("./build/spirv2valhall"):
            self.skipTest("spirv2valhall binary not built")
        res = synthesize_corpus(
            target="valhall-isa",
            num_shaders=3,
            num_blocks=1,
            num_insns=12,
            bin_path="./build/spirv2valhall",
            candidates_per_shader=2,
            verify_equiv=True,
        )
        self.assertEqual(len(res.shaders), 3)
        self.assertGreaterEqual(len(res.total_unique_features), 15)
        for rec in res.shaders:
            self.assertIn(rec.equiv_status, ("EQUIV_EXACT", "EQUIV_WITHIN_ULP", "DIVERGED"))


    def test_valhall_isel_catalog_covers_lowering_spec(self) -> None:
        from tools.corpus_synth.valhall_isel_catalog import (
            VALHALL_ISEL_CATALOG,
            audit_lowering_spec_coverage,
            get_rule_for_mnemonic,
        )

        audit = audit_lowering_spec_coverage("specs/lowering/valhall_lowering.json")
        self.assertEqual(audit["missing_from_catalog"], [])
        self.assertEqual(audit["extra_in_catalog"], [])
        self.assertEqual(len(VALHALL_ISEL_CATALOG), len(audit["spec_total"]))
        self.assertEqual(get_rule_for_mnemonic("HADD").tier, "compute")
        self.assertEqual(get_rule_for_mnemonic("BARRIER").tier, "compute")
        self.assertEqual(get_rule_for_mnemonic("CUBEFACE1").tier, "graphics")
        self.assertEqual(get_rule_for_mnemonic("KABOOM").tier, "internal")


    def test_targeted_isa_synth_emits_and_validates(self) -> None:
        if not os.path.exists("./build/spirv2valhall"):
            self.skipTest("spirv2valhall binary not built")
        from tools.corpus_synth.isa_target_synth import (
            list_synthesizable_mnemonics,
            synthesize_for_isa,
        )

        self.assertGreaterEqual(len(list_synthesizable_mnemonics()), 100)
        for mn, cplx in [
            ("HADD", "minimal"),
            ("RROT", "medium"),
            ("FMA", "branchy"),
            ("BARRIER", "minimal"),
            ("WMASK", "minimal"),
            ("DISCARD", "minimal"),
            ("LD_ATTR_IMM", "minimal"),
        ]:
            res = synthesize_for_isa(
                mn,
                complexity=cplx,
                verify_equiv=True,
                k_init=8,
                fp_range="vulkan",
            )
            self.assertTrue(res.compiled_ok, f"{mn} ({cplx}) failed to compile: {res.error}")
            self.assertTrue(res.target_emitted, f"{mn} ({cplx}) not emitted in EBIN[0]: {res.emitted_mnemonics}")
            self.assertEqual(res.equiv_status, "EQUIV_EXACT", f"{mn} ({cplx}) diverged: {res.divergence_report}")


if __name__ == "__main__":
    unittest.main()

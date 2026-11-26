#!/usr/bin/env python3
"""Unit tests for tools.corpus_synth greedy & targeted Valhall-ISA / field-mutation synthesizer."""

import os
import tempfile
import unittest

from tools.corpus_synth import (
    FRAGMENT_REGISTRY,
    OP_CATALOG,
    RECIPES,
    VARIANT_RECIPES,
    build_coverage_report,
    emit_shader_bytes,
    generate_field_cases_for_spec,
    list_all_mnemonics,
    list_synthesizable_mnemonics,
    load_isa_spec_index,
    run_full_raw_sweep,
    synthesize_corpus,
    synthesize_for_isa,
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

        self.assertGreaterEqual(len(list_synthesizable_mnemonics()), 125)
        self.assertEqual(len(list_all_mnemonics()), 201)
        for mn, cplx in [
            ("HADD", "minimal"),
            ("RROT", "medium"),
            ("FMA", "branchy"),
            ("BARRIER", "minimal"),
            ("WMASK", "minimal"),
            ("DISCARD", "minimal"),
            ("LD_ATTR_IMM", "minimal"),
            ("FABS", "minimal"),
            ("FNEG", "minimal"),
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

    def test_new_v47_spirv_recipes_and_variants_emit_in_ebin0(self) -> None:
        if not os.path.exists("./build/spirv2valhall"):
            self.skipTest("spirv2valhall binary not built")
        for mn in ("ATOM1", "F16_TO_S32", "S8_TO_S32", "V2S8_TO_V2F16", "LD_TEX", "LEA_TEX", "TEX_GRADIENT", "VAR_TEX_GATHER", "LD_VAR", "LD_VAR_FLAT", "LD_ATTR"):
            res = synthesize_for_isa(mn, mode="spirv")
            self.assertTrue(res.compiled_ok, f"{mn} compile error: {res.error}")
            self.assertTrue(res.target_emitted, f"{mn} missed in EBIN[0]: {res.emitted_mnemonics}")
        for (mn, var) in (("IADD", "v2u16"), ("FADD", "v2f16"), ("FADD", "clamp_0_1"), ("IDPADD", "v4u8")):
            self.assertIn((mn, var), VARIANT_RECIPES)
            res = synthesize_for_isa(mn, variant=var, mode="spirv")
            self.assertTrue(res.compiled_ok and res.target_emitted, f"{mn}.{var} failed: {res.error}")

    def test_isa_spec_index_and_field_fragments_complete(self) -> None:
        idx = load_isa_spec_index()
        self.assertEqual(len(idx.variants), 627)
        self.assertEqual(len(idx.all_mnemonics()), 201)
        self.assertEqual(len(idx.enums), 219)
        self.assertEqual(len(FRAGMENT_REGISTRY), 17)

        # Verify 5-regime F_SRC_FAU coverage and no invalid FAU special page 2
        fadd_sp = idx.primary_variant("FADD", "f32")
        self.assertIsNotNone(fadd_sp)
        cases = generate_field_cases_for_spec(fadd_sp, idx)
        fau_cases = [c for c in cases if c.fragment_id == "F_SRC_FAU"]
        self.assertGreaterEqual(len(fau_cases), 6)
        for c in fau_cases:
            ov = c.overrides_dict()
            if "fau_page_index" in ov and isinstance(ov.get("src0"), int) and ov["src0"] >= 0xE0:
                self.assertNotEqual(ov["fau_page_index"], 2)

        # Verify instruction-specific branch points are generated
        for mn, expected_sub in [
            ("FMIN", "nan_propagate"),
            ("CLPER", "lane_op_xor"),
            ("FLUSH", "nan_"),
            ("JUMP_EX", "stack_"),
            ("SEG_ADD", "pnull_"),
        ]:
            sp = idx.primary_variant(mn)
            self.assertIsNotNone(sp)
            bp_ids = [c.case_id for c in generate_field_cases_for_spec(sp, idx) if c.is_branch_point]
            self.assertTrue(any(expected_sub in cid for cid in bp_ids), f"{mn} missing branch point '{expected_sub}': {bp_ids[:10]}")

        rep = build_coverage_report(idx)
        self.assertEqual(rep["total_covered_mnemonics"], 201)
        self.assertEqual(rep["xml_total_specs"], 627)
        self.assertEqual(rep["fragments_count"], 17)
        self.assertEqual(rep["spirv_recipe_mnemonics"], len(RECIPES))

    def test_raw_valhall_all_627_variants_and_field_mutations_pass1(self) -> None:
        if not os.path.exists("./build/spirv2valhall"):
            self.skipTest("spirv2valhall binary not built")

        # All 627 XML specs baseline decode + Pass-1 lowering
        base_res = run_full_raw_sweep(all_variants=True, include_fragments=False, include_branch_points=False)
        self.assertEqual(len(base_res), 627)
        for r in base_res:
            self.assertTrue(r.decoded_ok, f"Decode failed: {r.case_id} ({r.word_hex})")
            self.assertTrue(r.mnemonic_matched, f"Mnemonic mismatch: {r.case_id} got {r.decoded_mnemonic}")
            self.assertTrue(r.pass1_ok, f"Pass-1 failed: {r.case_id}: {r.error}")

        # All 201 primary mnemonics x 17 field fragments + branch points
        mut_res = run_full_raw_sweep(all_variants=False, include_fragments=True, include_branch_points=True, max_per_fragment=4)
        self.assertGreaterEqual(len(mut_res), 4000)
        for r in mut_res:
            self.assertTrue(r.decoded_ok, f"Mut decode failed: {r.case_id} ({r.word_hex})")
            self.assertTrue(r.pass1_ok, f"Mut Pass-1 failed: {r.case_id}: {r.error}")

        # Auto mode seamlessly handles internal-only mnemonics via raw backend
        for mn in ("KABOOM", "JUMP_EX", "FLUSH", "LD_GCLK"):
            res = synthesize_for_isa(mn, mode="auto", mutate_fields=True)
            self.assertEqual(res.mode_used, "raw")
            self.assertTrue(res.compiled_ok and res.target_emitted, f"{mn} auto-raw failed: {res.error}")
            self.assertGreaterEqual(len(res.raw_results), 2)


    def test_v47_1_direct_isa_fixtures_slice_diff_and_spirv_fuzz(self) -> None:
        import json
        import subprocess
        from tools.corpus_synth.isa_target_synth import build_spvasm_for_isa
        from tools.corpus_synth.spv_fuzzer import fuzz_spv_with_spirv_fuzz
        from tools.corpus_synth.valhall_isel_catalog import VALHALL_ISEL_CATALOG
        from tools.spec import load_isa_spec_index as spec_loader
        from tools.verify.spvasm_io import find_spirv_tool

        # 1. Unified tools.spec loader
        idx = spec_loader()
        self.assertEqual(len(idx.variants), 627)

        # 2. All non-emitted catalog mnemonics carry DDK RE notes
        for rule in VALHALL_ISEL_CATALOG:
            if rule.mnemonic not in RECIPES:
                self.assertTrue(rule.ddk_re_note, f"Missing ddk_re_note for {rule.mnemonic}")

        # 3. Intermediate SSA materialization emits pinned outbuf stores
        _, hadd_pin = build_spvasm_for_isa("HADD", materialize_intermediates=True)
        self.assertIn("%pin_p1 = OpAccessChain", hadd_pin)

        if not os.path.exists("./build/spirv2valhall"):
            self.skipTest("spirv2valhall binary not built")

        # 4. --direct-isa on .tsv and .valhall fixtures (with @fau sideband)
        for fixture, min_rows in (
            ("tests/fixtures/direct-isa-sample.tsv", 3),
            ("tests/fixtures/direct-isa-sample.valhall", 2),
        ):
            proc = subprocess.run(
                ["./build/spirv2valhall", "--direct-isa", fixture],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(proc.returncode, 0, f"--direct-isa failed on {fixture}: {proc.stderr}")
            rows = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
            self.assertGreaterEqual(len(rows), min_rows)
            for row in rows:
                self.assertTrue(row["decoded_ok"], f"Decode failed in {fixture}: {row}")
                self.assertTrue(row["pass1_ok"], f"Pass1 failed in {fixture}: {row}")
            self.assertTrue(any(r.get("fau_count", 0) == 2 for r in rows), f"Missing @fau sideband in {fixture}")

        # 5. FMA_RSCALE verified SPIR-V emission + --slice-diff on MKVEC
        fma_r = synthesize_for_isa("FMA_RSCALE", mode="spirv")
        self.assertTrue(fma_r.compiled_ok and fma_r.target_emitted)
        mk = synthesize_for_isa("MKVEC", mode="spirv", verify_equiv=True, k_init=4, fp_range="vulkan", slice_diff_depth=4)
        self.assertTrue(any("[SLICE-DIFF" in line for line in mk.slice_diff_lines))

        # 6. spirv-fuzz integration when prebuilt tool is present
        if find_spirv_tool("spirv-fuzz"):
            with tempfile.TemporaryDirectory() as td:
                seed_res = synthesize_for_isa("HADD", mode="spirv", out_dir=td)
                fuzzed = fuzz_spv_with_spirv_fuzz(
                    seed_res.spv_path, num_variants=1, out_dir=td, verify_equiv=True, k_init=4, fp_range="vulkan"
                )
                self.assertEqual(len(fuzzed), 1)
                self.assertTrue(fuzzed[0].fuzz_ok and fuzzed[0].valid_spv and fuzzed[0].compiled_ok)
                self.assertEqual(fuzzed[0].equiv_status, "EQUIV_EXACT")

if __name__ == "__main__":
    unittest.main()

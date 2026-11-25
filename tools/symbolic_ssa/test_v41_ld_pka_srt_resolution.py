"""Regression tests for Checkpoint v41:
  1. Register-indexed `LD_PKA` SRT descriptor resolution (`srt_pka_buf_names`)
     including cross-block unsealed `Phi` propagation and multi-word
     `LD_PKA.i128` destination-register clobber protection (`last_pka_pc`).
  2. End-to-end `EQUIV_EXACT` regression on `d1488b11883613af.spv` (Vertex
     shader combining `LD_PKA`, `FEXP`, `FLOGD`, and IDVS guardband epilogue).
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import tempfile
import unittest

from tools.coverage.equiv_fuzzer import run_concolic_equiv_fuzz


def _dump_bundle(spv_path: str, extra_flags: list[str] | None = None) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tmp_json = tf.name
    try:
        cmd = ["./build/spirv2valhall", spv_path, "--dump-equiv-json", tmp_json, "--quiet"]
        if extra_flags:
            cmd.extend(extra_flags)
        subprocess.run(cmd, check=True)
        with open(tmp_json, "r", encoding="utf-8") as fh:
            return json.load(fh)
    finally:
        if os.path.exists(tmp_json):
            os.unlink(tmp_json)


class TestV41LdPkaSrtResolution(unittest.TestCase):
    def test_d11a039d_all_165_ld_pka_resolved_including_phi_and_clobber(self) -> None:
        matches = glob.glob("/working_dir/c_e95e66c5dab0535f/spirv_corpus/spirv_samples/d11a039dea528d7c*.spv")
        if not (matches and os.path.exists("./build/spirv2valhall")):
            self.skipTest("spirv2valhall binary or corpus sample not present")
        bundle = _dump_bundle(matches[0])
        pka = [n for n in bundle["valhall_ssa"]["nodes"] if n["op"] == "PsbLoad" and n.get("aux_bool")]
        unres = [n for n in pka if not n.get("aux_str")]
        self.assertEqual(len(pka), 165)
        self.assertEqual(len(unres), 0)

        # Verify multi-word clobbering LD_PKA.i128 at 0x2370 resolved all 4 words.
        w_2370 = [n for n in pka if n.get("orig_pc") == 0x2370]
        self.assertEqual(len(w_2370), 4)
        self.assertTrue(all(n.get("aux_str") for n in w_2370))

    def test_d054488e_multiword_ld_pka_clobber_resolved(self) -> None:
        matches = glob.glob("/working_dir/c_e95e66c5dab0535f/spirv_corpus/spirv_samples/d054488e19bef900*.spv")
        if not (matches and os.path.exists("./build/spirv2valhall")):
            self.skipTest("spirv2valhall binary or corpus sample not present")
        bundle = _dump_bundle(matches[0])
        pka = [n for n in bundle["valhall_ssa"]["nodes"] if n["op"] == "PsbLoad" and n.get("aux_bool")]
        # All 4 words of LD_PKA.i128 @r44:r45:r46:r47, r44^, r45^ at 0x2b58 must resolve to ubo:0:5.
        w_2b58 = [n for n in pka if n.get("orig_pc") == 0x2B58]
        self.assertEqual(len(w_2b58), 4)
        self.assertEqual([n.get("aux_str") for n in w_2b58], ["ubo:0:5"] * 4)

    def test_d1488b11_equiv_exact_after_ld_pka_resolution(self) -> None:
        matches = glob.glob("/working_dir/c_e95e66c5dab0535f/spirv_corpus/spirv_samples/d1488b11883613af*.spv")
        if not (matches and os.path.exists("./build/spirv2valhall")):
            self.skipTest("spirv2valhall binary or corpus sample not present")
        bundle = _dump_bundle(matches[0], ["--opt-ssa", "--defau"])
        res = run_concolic_equiv_fuzz(
            bundle,
            bundle["direct_ssa"],
            bundle["valhall_ssa"],
            k_init=16,
            time_budget_ms=1500,
            domain="udomain",
            auto_shrink=False,
        )
        self.assertEqual(res.status, "EQUIV_EXACT")


if __name__ == "__main__":
    unittest.main()

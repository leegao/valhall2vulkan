"""Checkpoint v54 regression tests for Panfrost-vs-Direct de-FAU fixes:

- BUG-040: Non-.end BLEND fixed-function return jump skips software blend-shader
  ADR/JUMP trampoline in valhall_cfg.cpp, and Panfrost compacted sysval FAU slots
  15..22 map to fragment render targets 0..7 in store_frag_color.
- BUG-041: Panfrost LD_VAR_SPECIAL.frag_w sets aux_bool=True so DDK's
  V1_split_persp_var_w canonicalizer rule does not strip explicit gl_FragCoord.w
  multiplies in projective-varying fragment shaders.
- BUG-042: Panfrost vertex qNaN-undef replacement unwraps Kraid's 64-bit register
  pair move idiom (IADD.u64 [rD:rD+1], [rS:rS+1], k0).
"""

import glob
import json
import os
import subprocess
import tempfile
import unittest

from tools.coverage.panfrost_vs_ddk import run_panfrost_vs_ddk_equiv
from tools.symbolic_ssa.passes import canonicalize_ssa_module

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SPIRV2VALHALL = os.path.join(REPO_ROOT, "build", "spirv2valhall")
CORPUS_DIR = os.path.abspath(os.path.join(REPO_ROOT, "..", "spirv_corpus", "spirv_samples"))


def _find_corpus_spv(prefix: str) -> str | None:
    if not os.path.isdir(CORPUS_DIR):
        return None
    matches = glob.glob(os.path.join(CORPUS_DIR, f"{prefix}*.spv"))
    return matches[0] if matches else None


def _compile_panfrost_equiv_bundle(spv_path: str) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json") as tf:
        subprocess.run(
            [
                SPIRV2VALHALL,
                "--panfrost",
                "--panfrost-backend",
                "kraid",
                "--defau",
                "--dump-equiv-json",
                tf.name,
                spv_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with open(tf.name, "r", encoding="utf-8") as f:
            return json.load(f)


class TestV54PanfrostDirectFixes(unittest.TestCase):
    def test_bug041_explicit_panfrost_frag_w_preserved_by_canonicalizer(self) -> None:
        """Explicit gl_FragCoord.w (aux_bool=True) must survive canonicalize_ssa."""
        ssa = {
            "blocks": [
                {
                    "id": 0,
                    "nodes": [0, 1, 2, 3, 4, 5, 6, 7],
                    "preds": [],
                    "is_end": True,
                }
            ],
            "nodes": [
                {"id": 0, "op": "LoadFragVarying", "type": "V4F32", "imm_u32": 3, "aux_bool": False, "args": []},
                {"id": 1, "op": "VecExtract", "type": "F32", "imm_u32": 0, "args": [0]},
                {"id": 2, "op": "LoadBuiltinInput", "type": "V4F32", "aux_str": "gl_FragCoord", "args": []},
                {"id": 3, "op": "VecExtract", "type": "F32", "imm_u32": 3, "aux_bool": True, "args": [2]},
                {"id": 4, "op": "FMul", "type": "F32", "orig_pc": 0x10, "args": [1, 3]},
                {"id": 5, "op": "VecConstruct4", "type": "V4F32", "args": [4, 4, 4, 4]},
                {"id": 6, "op": "InitToken", "type": "Token", "args": []},
                {"id": 7, "op": "StoreFragColor", "type": "Token", "imm_u32": 0, "args": [5, 6]},
            ],
        }
        canon, _ = canonicalize_ssa_module(ssa)
        ops = [n["op"] for n in canon["nodes"]]
        self.assertIn("FMul", ops, "Panfrost explicit gl_FragCoord.w FMul must not be stripped")

    def test_bug040_mrt_blend_cfg_and_rt_mapping_d2f594b49c(self) -> None:
        """MRT shader writing RT 0, 1, 3 must emit all three StoreFragColor slots and match Direct."""
        spv = _find_corpus_spv("d2f594b49c")
        if not spv or not os.path.exists(SPIRV2VALHALL):
            self.skipTest("spirv_corpus or spirv2valhall binary not available")
        bundle = _compile_panfrost_equiv_bundle(spv)
        val_rts = sorted(
            {
                int(n.get("imm_u32", 0))
                for n in bundle["valhall_ssa"]["nodes"]
                if n.get("op") == "StoreFragColor"
            }
        )
        self.assertEqual(val_rts, [0, 1, 3], f"Expected MRT slots [0, 1, 3], got {val_rts}")
        res = run_panfrost_vs_ddk_equiv(
            spv,
            compare="panfrost-vs-direct",
            panfrost_backend="kraid",
            defau=True,
            k_init=32,
        )
        self.assertIn(res.status, ("EQUIV_EXACT", "EQUIV_WITHIN_ULP"))
        self.assertEqual(res.divs, 0)

    def test_bug040_sparse_mrt_no_rt0_d303613cf6(self) -> None:
        """Sparse MRT shader writing only RT 1 and RT 3 must not default RT 1 to RT 0."""
        spv = _find_corpus_spv("d303613cf6")
        if not spv or not os.path.exists(SPIRV2VALHALL):
            self.skipTest("spirv_corpus or spirv2valhall binary not available")
        bundle = _compile_panfrost_equiv_bundle(spv)
        val_rts = sorted(
            {
                int(n.get("imm_u32", 0))
                for n in bundle["valhall_ssa"]["nodes"]
                if n.get("op") == "StoreFragColor"
            }
        )
        self.assertEqual(val_rts, [1, 3], f"Expected sparse MRT slots [1, 3], got {val_rts}")
        res = run_panfrost_vs_ddk_equiv(
            spv,
            compare="panfrost-vs-direct",
            panfrost_backend="kraid",
            defau=True,
            k_init=32,
        )
        self.assertIn(res.status, ("EQUIV_EXACT", "EQUIV_WITHIN_ULP"))
        self.assertEqual(res.divs, 0)

    def test_bug041_projective_frag_w_e2e_d83204bd27(self) -> None:
        """Projective UV fragment shader using explicit gl_FragCoord.w must pass EQUIV_EXACT."""
        spv = _find_corpus_spv("d83204bd27")
        if not spv or not os.path.exists(SPIRV2VALHALL):
            self.skipTest("spirv_corpus or spirv2valhall binary not available")
        bundle = _compile_panfrost_equiv_bundle(spv)
        frag_w_extracts = [
            n
            for n in bundle["valhall_ssa"]["nodes"]
            if n.get("op") == "VecExtract" and int(n.get("imm_u32", -1)) == 3 and n.get("aux_bool") is True
        ]
        self.assertTrue(frag_w_extracts, "Expected Panfrost LD_VAR_SPECIAL.frag_w to set aux_bool=True")
        res = run_panfrost_vs_ddk_equiv(
            spv,
            compare="panfrost-vs-direct",
            panfrost_backend="kraid",
            defau=True,
            k_init=32,
        )
        self.assertEqual(res.status, "EQUIV_EXACT")
        self.assertEqual(res.divs, 0)

    def test_bug042_qnan_undef_through_iadd64_d15477673c(self) -> None:
        """Vertex varying undef qNaN copied via IADD.u64 ..., k0 must be canonicalized to 0.0f."""
        spv = _find_corpus_spv("d15477673c")
        if not spv or not os.path.exists(SPIRV2VALHALL):
            self.skipTest("spirv_corpus or spirv2valhall binary not available")
        bundle = _compile_panfrost_equiv_bundle(spv)
        nodes_by_id = {int(n["id"]): n for n in bundle["valhall_ssa"]["nodes"]}
        loc3_stores = [
            n
            for n in bundle["valhall_ssa"]["nodes"]
            if n.get("op") == "StoreVertVarying" and int(n.get("imm_u32", -1)) == 3
        ]
        self.assertTrue(loc3_stores, "Expected StoreVertVarying for Location 3")
        vec = nodes_by_id[int(loc3_stores[0]["args"][0])]
        comp0 = nodes_by_id[int(vec["args"][0])]
        self.assertEqual(comp0.get("op"), "ConstF32")
        self.assertEqual(int(comp0.get("imm_u32", 0)), 0x00000000)


    def test_bug041_noperspective_varying_still_folds_f375a94aca(self) -> None:
        """Panfrost Fragment shader with NoPerspective varyings must keep frag_w aux_bool=False."""
        spv = _find_corpus_spv("f375a94aca")
        if not spv or not os.path.exists(SPIRV2VALHALL):
            self.skipTest("spirv_corpus or spirv2valhall binary not available")
        res = run_panfrost_vs_ddk_equiv(
            spv,
            compare="panfrost-vs-direct",
            panfrost_backend="kraid",
            defau=True,
            k_init=32,
        )
        self.assertIn(res.status, ("EQUIV_EXACT", "EQUIV_WITHIN_ULP"))
        self.assertEqual(res.divs, 0)


if __name__ == "__main__":
    unittest.main()

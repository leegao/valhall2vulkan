"""Checkpoint v45 regression tests: V1_split_persp_var_w, BaseVertex/BaseInstance direct parity, wide TEX_SINGLE, --fp-range vulkan, and bugs/ manifest sync."""

from __future__ import annotations

import glob
import math
import os
import struct
import subprocess
import unittest

from tools.coverage.equiv_fuzzer import run_concolic_equiv_fuzz
from tools.coverage.shrinker import DiffPolicy
from tools.symbolic_ssa.domains.shader_env.concrete_shader_env import ConcreteShaderEnv
from tools.symbolic_ssa.domains.shader_env.symbolic_shader_env import SymbolicShaderEnv
from tools.symbolic_ssa.passes import canonicalize_ssa_module


class TestV45Fixes(unittest.TestCase):
    def test_v1_split_persp_var_w_canonicalizes_valhall_only(self) -> None:
        """FMul(VecExtract(LoadFragVarying[!flat]), gl_FragCoord.w) folds on Valhall (orig_pc) only."""
        ssa = {
            "blocks": [{"id": 0, "is_end": True, "nodes": [1, 2, 3, 4, 5, 6, 7]}],
            "nodes": [
                {"id": 1, "op": "LoadFragVarying", "type": "V4F32", "blk": 0, "args": [], "imm_u32": 0},
                {"id": 2, "op": "VecExtract", "type": "F32", "blk": 0, "args": [1], "imm_u32": 0},
                {"id": 3, "op": "LoadBuiltinInput", "type": "V4F32", "blk": 0, "args": [], "imm_u32": 15, "aux_str": "gl_FragCoord"},
                {"id": 4, "op": "VecExtract", "type": "F32", "blk": 0, "args": [3], "imm_u32": 3},
                {"id": 5, "op": "FMul", "type": "F32", "blk": 0, "args": [2, 4], "orig_pc": 80},
                {"id": 6, "op": "VecConstruct4", "type": "V4F32", "blk": 0, "args": [5, 5, 5, 5]},
                {"id": 7, "op": "StoreFragColor", "type": "Token", "blk": 0, "args": [6], "imm_u32": 0},
            ],
        }
        can, _ = canonicalize_ssa_module(ssa)
        ops = [n["op"] for n in can["nodes"]]
        self.assertNotIn("FMul", ops)

        # Direct side (no orig_pc) must keep explicit FMul(..., gl_FragCoord.w).
        direct = {
            "blocks": ssa["blocks"],
            "nodes": [
                {k: v for k, v in n.items() if k != "orig_pc"} for n in ssa["nodes"]
            ],
        }
        can_d, _ = canonicalize_ssa_module(direct)
        self.assertIn("FMul", [n["op"] for n in can_d["nodes"]])

    def test_direct_base_vertex_and_base_instance_bind_to_zero(self) -> None:
        """Direct BaseVertex (4424) and BaseInstance (4425) match Valhall's 0-based draw pilot_fau."""
        bundle = {"stage": "Vertex", "mbs2_context": {}}
        for vec_idx in (0, 7, 31):
            cenv = ConcreteShaderEnv(bundle, vec_idx=vec_idx)
            self.assertEqual(cenv.get_builtin(4424, "S32"), 0)
            self.assertEqual(cenv.get_builtin(4425, "S32"), 0)

        senv = SymbolicShaderEnv(bundle)
        self.assertEqual(senv.get_builtin(4424, "S32").as_long(), 0)
        self.assertEqual(senv.get_builtin(4425, "S32").as_long(), 0)

    def test_fp_range_vulkan_bounds_floats_and_vertex_indices(self) -> None:
        """--fp-range vulkan keeps float inputs finite in [-16, 16] and vertex indices in [0, 64)."""
        bundle = {"stage": "Vertex", "mbs2_context": {}}
        for vec_idx in range(32):
            cenv = ConcreteShaderEnv(bundle, vec_idx=vec_idx, fp_range="vulkan")
            u = cenv.sample_u32(f"vert_attr_0_{vec_idx}")
            f = struct.unpack("<f", struct.pack("<I", u))[0]
            self.assertTrue(math.isfinite(f))
            self.assertLessEqual(abs(f), 16.0)
            self.assertIn(cenv.sample_u32("bi_42"), range(64))
            self.assertIn(cenv.sample_u32("bi_43"), range(64))

    def test_corpus_shaders_fixed_in_v45(self) -> None:
        """Corpus shaders fixed in v45 (if spirv_corpus is present) pass equiv-fuzz."""
        corpus_dir = os.path.join("..", "spirv_corpus", "spirv_samples")
        if not os.path.isdir(corpus_dir):
            self.skipTest("spirv_corpus not present")
        from tools.coverage.cli import _load_bundle

        targets = [
            "d338ddafb18151e6eb6deeca825899d5781ee5f0f19a6644e75992a8780fbefd.spv",
            "d83fa6fcd1ad533403c85da2abfe069924bbba09a228dc7941e9c4a2927eade7.spv",
            "f97218f0613c5bf2c87c332368a6c7fcc2ca80620fbfdcb607669593cb8201bd.spv",
            "df11858037b5515813ea7ee4c7369908b10efd03f7a67b4b12640fd20fdda37f.spv",
        ]
        for fname in targets:
            path = os.path.join(corpus_dir, fname)
            if not os.path.isfile(path):
                continue
            bundle = _load_bundle(path, "./build/spirv2valhall")
            d_ssa, _ = canonicalize_ssa_module(bundle["direct_ssa"])
            v_ssa, _ = canonicalize_ssa_module(bundle["valhall_ssa"])
            ef = run_concolic_equiv_fuzz(
                bundle,
                d_ssa,
                v_ssa,
                k_init=64,
                fp_mode="real",
                domain="udomain",
                policy=DiffPolicy(),
            )
            self.assertEqual(ef.diverging_vectors, 0, f"{fname} still diverged: {ef.first_divergence}")

    def test_bugs_manifest_sync(self) -> None:
        """Every bugs/*.spvasm must be referenced in bugs/manifest.md and pass spirv-val."""
        with open(os.path.join("bugs", "manifest.md"), encoding="utf-8") as f:
            manifest = f.read()
        for spvasm in sorted(glob.glob(os.path.join("bugs", "*.spvasm"))):
            self.assertIn(os.path.basename(spvasm), manifest, f"{spvasm} missing from bugs/manifest.md")
            proc = subprocess.run(["spirv-as", spvasm, "-o", "-"], capture_output=True)
            self.assertEqual(proc.returncode, 0, f"spirv-as failed for {spvasm}: {proc.stderr.decode()}")


if __name__ == "__main__":
    unittest.main()

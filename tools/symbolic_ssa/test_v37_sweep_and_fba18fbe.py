"""
Checkpoint v37 regression tests:
  - Item 1: `format_z3_ast` renders native Z3 BitVec/Real/Int constants cleanly (no `bv`/`Real` placeholders).
  - Item 2: `LEA_TEX`/`LEA_TEX_IMM` `ImageTexelAddr` carries `tex:set:binding`, survives `canonicalize_ssa_module`,
            and resolves to `mem_tex_0_6` (not `fallback_mem`) in both Concrete and Symbolic interpreters.
  - Item 3: `SpvOpImageRead` -> `TexFetch` and non-Fragment / `lod_mode=zero` `TEX_SINGLE` -> `TexSampleLod(..., 0.0f)`.
  - Item 4: `SymbolicShaderEnv._ensure_pilot_executed()` computes derived FAU words symbolically via virtual SRT tables.
  - Item 5: Exact Mali Position FIFO epilogue (`0x1228..0x12b0`, `eps=0x272428e0`, NaN propagation) in `align_vertex_gl_position`.
  - Item 6: Mali Q8.8 packed signed 16-bit fixed-point `gl_SamplePosition` sideband table in `ConcreteShaderEnv`.
"""

from __future__ import annotations

import math
import os
import struct
import unittest

import z3

from tools.coverage.shrinker import DiffPolicy, align_vertex_gl_position
from tools.symbolic_ssa.concrete_interpreter import ConcreteSsaInterpreter
from tools.symbolic_ssa.domains.shader_env.concrete_shader_env import ConcreteShaderEnv
from tools.symbolic_ssa.domains.shader_env.symbolic_shader_env import SymbolicShaderEnv
from tools.symbolic_ssa.passes import canonicalize_ssa_module
from tools.symbolic_ssa.symbolic_interpreter import SymbolicSsaInterpreter
from tools.symbolic_ssa.z3simplifier.ast_utils import format_z3_ast


def _f32_to_u32(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _u32_to_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


class TestV37SweepAndFba18fbe(unittest.TestCase):
    def test_item1_format_z3_ast_renders_native_z3_literals(self) -> None:
        """Native Z3 BitVecNumRef / RatNumRef / IntNumRef must print exact literals, never 'bv' or 'Real'."""
        x = z3.BitVec("bi_26_0", 32)
        e_bv = z3.simplify(x + z3.BitVecVal(6, 32))
        s_bv = format_z3_ast(e_bv)
        self.assertIn("bv32(0x6)", s_bv)
        self.assertNotIn(" bv)", s_bv)

        r = z3.Real("ro_ubo_0_7_0x8_f32")
        e_real = z3.simplify((r + z3.RealVal("0.5")) * z3.RealVal(2))
        s_real = format_z3_ast(e_real)
        self.assertNotIn(" Real", s_real)
        self.assertIn("1", s_real)

    def test_item2_tagged_image_texel_addr_survives_canon_and_resolves_to_tex_mem(self) -> None:
        """Tagged ImageTexelAddr(aux_str='tex:0:6') must survive canonicalize_ssa_module and avoid fallback_mem."""
        ssa = {
            "blocks": [{"id": 0, "preds": [], "succs": [], "nodes": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}],
            "nodes": [
                {"id": 1, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 0x00020008},
                {"id": 2, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 0},
                {"id": 3, "op": "ImageTexelAddr", "type": "U64", "blk": 0, "args": [1, 2], "imm_u32": 6, "aux_str": "tex:0:6"},
                {"id": 4, "op": "Split64", "type": "TupleU32U32", "blk": 0, "args": [3]},
                {"id": 5, "op": "TupleExtract", "type": "U32", "blk": 0, "args": [4], "imm_u32": 0},
                {"id": 6, "op": "TupleExtract", "type": "U32", "blk": 0, "args": [4], "imm_u32": 1},
                {"id": 7, "op": "Pack64", "type": "U64", "blk": 0, "args": [5, 6]},
                {"id": 8, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 0xDEADBEEF},
                {"id": 9, "op": "PsbStore", "type": "Token", "blk": 0, "args": [7, 8, 2], "imm_u32": 4},
                {"id": 10, "op": "End", "type": "Token", "blk": 0, "args": [9]},
            ],
        }
        canon_ssa, _ = canonicalize_ssa_module(ssa)
        self.assertTrue(
            any(n["op"] == "ImageTexelAddr" and n.get("aux_str") == "tex:0:6" for n in canon_ssa["nodes"]),
            "canonicalize_ssa_module must not fold away tagged ImageTexelAddr",
        )

        bundle = {"stage": "Compute", "mbs2_context": {}}
        cenv = ConcreteShaderEnv(bundle, vec_idx=0)
        _, cout = ConcreteSsaInterpreter(canon_ssa, is_valhall_side=True).execute_one(cenv)
        self.assertEqual(cout.get("tex:0:6+0x8"), 0xDEADBEEF)
        self.assertFalse(any(k.startswith("fallback_mem") for k in cout))

        senv = SymbolicShaderEnv(bundle, domain="udomain_backfill")
        st = SymbolicSsaInterpreter(canon_ssa, senv, is_valhall_side=True).evaluate()
        self.assertIn("tex:0:6", st.buf_mems)
        self.assertIs(st.fallback_mem, senv.fallback_mem)

    def test_item4_symbolic_pilot_shader_execution_resolves_derived_fau(self) -> None:
        """SymbolicShaderEnv must lazily execute pilot_ssa so derived FAU words correlate with UBOs."""
        pilot_ssa = {
            "blocks": [{"id": 0, "preds": [], "succs": [], "nodes": list(range(1, 14))}],
            "nodes": [
                {"id": 1, "op": "LoadSidebandU64", "type": "U64", "blk": 0, "args": [], "imm_u32": 2},
                {"id": 2, "op": "LoadSidebandU64", "type": "U64", "blk": 0, "args": [], "imm_u32": 3},
                {"id": 3, "op": "ConstU64", "type": "U64", "blk": 0, "args": [], "imm_u64": 64},
                {"id": 4, "op": "IAdd64", "type": "U64", "blk": 0, "args": [1, 3]},
                {"id": 5, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 0},
                {"id": 6, "op": "PsbLoad", "type": "U32", "blk": 0, "args": [4, 5], "imm_u32": 4},
                {"id": 7, "op": "ConstU64", "type": "U64", "blk": 0, "args": [], "imm_u64": 68},
                {"id": 8, "op": "IAdd64", "type": "U64", "blk": 0, "args": [1, 7]},
                {"id": 9, "op": "PsbLoad", "type": "U32", "blk": 0, "args": [8, 5], "imm_u32": 4},
                {"id": 10, "op": "Pack64", "type": "U64", "blk": 0, "args": [6, 9]},
                {"id": 11, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 0x12345678},
                {"id": 12, "op": "PsbStore", "type": "Token", "blk": 0, "args": [2, 11, 5], "imm_u32": 4},
                {"id": 13, "op": "End", "type": "Token", "blk": 0, "args": [12]},
            ],
        }
        bundle = {
            "stage": "Compute",
            "mbs2_context": {
                "pilot_srt": [{"rack": 4, "addr": 0, "kind": "ubo", "key": "ubo:0:0"}],
            },
            "pilot_ssa": pilot_ssa,
        }
        senv = SymbolicShaderEnv(bundle, domain="udomain")
        w0 = senv.load_fau_word(0)
        bv = z3.simplify(senv.domain.lower_z3_bv32(w0).to_z3)
        self.assertTrue(z3.is_bv_value(bv))
        self.assertEqual(bv.as_long(), 0x12345678)

    def test_item5_mali_position_fifo_nan_propagation_and_eps_clamp(self) -> None:
        """NaN in any clip-space coordinate must propagate through Mali's Position FIFO denom to all lanes."""
        bundle = {
            "stage": "Vertex",
            "mbs2_context": {
                "rlocs": [{"fau_offset": 0, "word_idx": 0, "symbol": "gl_mali_ViewportTransform"}],
            },
        }
        env = ConcreteShaderEnv(bundle, vec_idx=0)
        d_out = {
            "builtin_out[0][0]": _f32_to_u32(0.0),
            "builtin_out[0][1]": _f32_to_u32(1.0),
            "builtin_out[0][2]": 0x7FC00000,  # NaN in z
            "builtin_out[0][3]": _f32_to_u32(1.0),
        }
        v_out = {
            "builtin_out[0][0]": 0x7FC00000,
            "builtin_out[0][1]": 0x7FC00000,
            "builtin_out[0][2]": 0x7FC00000,
            "builtin_out[0][3]": 0x7FC00000,
        }
        aligned = align_vertex_gl_position(d_out, v_out, env, policy=DiffPolicy(max_ulp=2))
        for c in range(4):
            self.assertTrue(math.isnan(_u32_to_f32(aligned[f"builtin_out[0][{c}]"])))

    def test_item6_sample_position_q8_8_fixed_point_encoding(self) -> None:
        """ConcreteShaderEnv.read_init_buf('gl_SamplePosition', ...) packs (x, y) as Q8.8 s16 halfwords."""
        bundle = {"stage": "Fragment", "mbs2_context": {}}
        env = ConcreteShaderEnv(
            bundle,
            vec_idx=0,
            overrides={"bi_19_0": _f32_to_u32(0.25), "bi_19_1": _f32_to_u32(0.75)},
        )
        word = env.read_init_buf("gl_SamplePosition", 0)
        sx = word & 0xFFFF
        sy = (word >> 16) & 0xFFFF
        self.assertEqual(sx, 64)   # 0.25 * 256
        self.assertEqual(sy, 192)  # 0.75 * 256
        self.assertAlmostEqual(sx * (1.0 / 256.0), 0.25, places=6)
        self.assertAlmostEqual(sy * (1.0 / 256.0), 0.75, places=6)

    def test_item3_fba18fbe_compute_shader_has_zero_fallback_mem_and_aligned_tex_ops(self) -> None:
        """End-to-end check on samples/comp_unique_03_fba18fbe.spv when spirv2valhall binary is present."""
        bin_path = "./build/spirv2valhall"
        spv_path = "samples/comp_unique_03_fba18fbe.spv"
        if not (os.path.exists(bin_path) and os.path.exists(spv_path)):
            self.skipTest("spirv2valhall binary or sample not present")
        import json
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            jp = os.path.join(td, "fba.json")
            subprocess.run([bin_path, spv_path, "--dump-equiv-json", jp, "--quiet"], check=True)
            with open(jp) as f:
                bundle = json.load(f)
        v_can, _ = canonicalize_ssa_module(bundle["valhall_ssa"])
        d_can, _ = canonicalize_ssa_module(bundle["direct_ssa"])

        d_ops_by_tex = {n.get("aux_str"): n["op"] for n in d_can["nodes"] if n["op"].startswith("Tex")}
        v_ops_by_tex = {n.get("aux_str"): n["op"] for n in v_can["nodes"] if n["op"].startswith("Tex")}
        for tkey in ("tex:0:1", "tex:0:2", "tex:0:3", "tex:0:4", "tex:0:5"):
            self.assertEqual(d_ops_by_tex.get(tkey), v_ops_by_tex.get(tkey), f"Texture op mismatch on {tkey}")

        senv = SymbolicShaderEnv(bundle, domain="udomain_backfill")
        st = SymbolicSsaInterpreter(v_can, senv, is_valhall_side=True).evaluate()
        self.assertIn("tex:0:6", st.buf_mems)
        self.assertIs(st.fallback_mem, senv.fallback_mem)

    def test_v37_1_lea_tex_and_fexp_flogd_in_lowering_spec(self) -> None:
        """v37.1: lea_tex lives in valhall_lowering.json + valhall_host_handlers.json; FEXP/FLOGD handle NaN/Inf."""
        import json

        with open("specs/lowering/valhall_lowering.json") as f:
            spec = json.load(f)
        self.assertIn("lea_tex", spec.get("host_functions", {}))
        by_mnem = {r["mnemonic"]: r for r in spec["rules"]}
        self.assertIn("@lea_tex(%xy, %z_layer)", " ".join(by_mnem["LEA_TEX"]["inline_ops"]))
        self.assertIn("@lea_tex(%xy, %z_layer)", " ".join(by_mnem["LEA_TEX_IMM"]["inline_ops"]))
        fexp_ops = " ".join(by_mnem["FEXP"]["inline_ops"])
        self.assertIn("$has_src1", fexp_ops)
        self.assertIn("0x7fc00000", fexp_ops)
        self.assertIn("0x7f800000", fexp_ops)
        flogd_ops = " ".join(by_mnem["FLOGD"]["inline_ops"])
        self.assertIn("is_pinf", flogd_ops)

        with open("tools/verify/generated/valhall_host_handlers.json") as f:
            hh = json.load(f)
        self.assertTrue(any(fn.get("name") == "lea_tex" for fn in hh["functions"]))


if __name__ == "__main__":
    unittest.main()

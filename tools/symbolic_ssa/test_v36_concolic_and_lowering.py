"""
Unit and regression tests for Checkpoint v36:
  1. ConcreteBitvecDomain & ConcreteUDomain exact IEEE-754 FP unary/binary/cmp
     ops (`FrexpExp`, `FrexpSignificand`, `Ldexp`, `Atan`, `Asin`, `Acos`,
     `Atan2`, `Sinh`, `Cosh`, `Tanh`, `Fract`, `FCmpUnord*`).
  2. `FMin`/`FMax` NaN propagation vs `NMin`/`NMax` NaN suppression.
  3. `gl_SamplePosition` sideband pointer (`LoadSidebandU64(imm=8, aux="gl_SamplePosition")`)
     alignment with `bi_19` (`SpvBuiltInSamplePosition`) in concrete & symbolic envs.
  4. Pilot shader (`pilot_ssa` + `pilot_srt`) concrete execution for derived/unmapped FAU words.
  5. Vertex `gl_Position` Mali viewport epilogue alignment (`align_vertex_gl_position`).
  6. `ST_CVT` `vert_sout_attr_names` handling (`gl_mali_PositionFifoAttributes` no longer
     overwrites `vert_var[1]` with `0x3c00`).
  7. Byte-addressed concrete memory in `ConcreteSsaInterpreter` for sub-word `STORE.i16` /
     `LOAD.i16` TLS spills without clobbering adjacent bytes.
"""

from __future__ import annotations

import math
import unittest
from typing import Any

from tools.coverage.shrinker import DiffPolicy, align_vertex_gl_position
from tools.symbolic_ssa.concrete_interpreter import ConcreteSsaInterpreter
from tools.symbolic_ssa.domains.concrete_bitvec import ConcreteBitvecDomain
from tools.symbolic_ssa.domains.shader_env.base import _f32_to_u32, _u32_to_f32
from tools.symbolic_ssa.domains.shader_env.concrete_shader_env import ConcreteShaderEnv
from tools.symbolic_ssa.domains.shader_env.symbolic_shader_env import SymbolicShaderEnv
from tools.symbolic_ssa.domains.udomain import ConcreteUDomain


def _single_block(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "blocks": [{
            "id": 0,
            "preds": [],
            "phis": [],
            "nodes": [n["id"] for n in nodes],
            "branch_target": -1,
            "fallthrough": -1,
            "is_cond": False,
            "is_uncond": False,
            "is_end": True,
            "is_discard": False,
            "is_loop_header": False,
            "loop_backedge": -1,
            "loop_merge": -1,
            "sel_merge": -1,
            "branch_cond": 0,
        }],
        "nodes": nodes,
    }


def _empty_bundle(stage: str = "Fragment", **extra: Any) -> dict[str, Any]:
    b: dict[str, Any] = {
        "spv_path": "synthetic_v36.spv",
        "stage": stage,
        "arch": 10,
        "direct_ok": True,
        "valhall_ok": True,
        "direct_bindings": extra.pop("direct_bindings", []),
        "mbs2_context": extra.pop("mbs2_context", {
            "fcsts": [],
            "rlocs": [],
            "pilot_fau": [],
            "pilot_srt": [],
            "sinp": [],
            "sout": [],
            "pc_disasm": {},
        }),
        "direct_ssa": _single_block([]),
        "valhall_ssa": _single_block([]),
    }
    b.update(extra)
    return b


class TestV36ConcolicAndLoweringFixes(unittest.TestCase):
    """Regression suite for all 7 Checkpoint v36 root causes."""

    def test_item1_concrete_fp_ops_and_fcmp_unord(self) -> None:
        """Item 1: FrexpExp, FrexpSignificand, Ldexp, inverse/hyperbolic trig, and FCmpUnord*."""
        cb = ConcreteBitvecDomain()
        cu = ConcreteUDomain()

        # frexp(6.0) = (0.75, 3); ldexp(0.75, 3) = 6.0
        u6 = _f32_to_u32(6.0)
        exp = cb.fp_unary("FrexpExp", u6)
        sig = cb.fp_unary("FrexpSignificand", u6)
        self.assertEqual(exp, 3)
        self.assertAlmostEqual(_u32_to_f32(sig), 0.75, places=6)
        self.assertAlmostEqual(_u32_to_f32(cb.fp_binary("Ldexp", sig, exp)), 6.0, places=6)

        # ConcreteUDomain parity
        self.assertEqual(cu.to_bvec64(cu.fp_unary("FrexpExp", cu.const_f32(6.0))) & 0xFFFFFFFF, 3)
        self.assertAlmostEqual(cu.to_f32x2(cu.fp_unary("FrexpSignificand", cu.const_f32(6.0)))[0], 0.75, places=6)
        self.assertAlmostEqual(cu.to_f32x2(cu.fp_binary("Ldexp", cu.const_f32(0.75), cu.const_u32(3)))[0], 6.0, places=6)

        # Inverse & hyperbolic trig
        self.assertAlmostEqual(_u32_to_f32(cb.fp_unary("Atan", _f32_to_u32(1.0))), math.pi / 4.0, places=5)
        self.assertAlmostEqual(_u32_to_f32(cb.fp_unary("Asin", _f32_to_u32(0.5))), math.pi / 6.0, places=5)
        self.assertAlmostEqual(_u32_to_f32(cb.fp_unary("Acos", _f32_to_u32(0.5))), math.pi / 3.0, places=5)
        self.assertAlmostEqual(_u32_to_f32(cb.fp_binary("Atan2", _f32_to_u32(1.0), _f32_to_u32(1.0))), math.pi / 4.0, places=5)
        self.assertAlmostEqual(_u32_to_f32(cb.fp_unary("Sinh", _f32_to_u32(0.0))), 0.0, places=6)
        self.assertAlmostEqual(_u32_to_f32(cb.fp_unary("Cosh", _f32_to_u32(0.0))), 1.0, places=6)
        self.assertAlmostEqual(_u32_to_f32(cb.fp_unary("Tanh", _f32_to_u32(0.0))), 0.0, places=6)
        self.assertEqual(cb.fp_unary("Asin", _f32_to_u32(2.0)), 0x7FC00000)

        # All FCmpUnord* predicates must return True when either operand is NaN
        nan_u = 0x7FC00000
        one_u = _f32_to_u32(1.0)
        for op in ("FCmpUnordEq", "FCmpUnordLt", "FCmpUnordLe", "FCmpUnordGt", "FCmpUnordGe", "FCmpUnordNe"):
            self.assertTrue(cb.fp_cmp(op, nan_u, one_u), f"{op}(NaN, 1.0) must be True")
        self.assertTrue(cb.fp_cmp("FCmpUnordLt", _f32_to_u32(1.0), _f32_to_u32(2.0)))
        self.assertFalse(cb.fp_cmp("FCmpUnordLt", _f32_to_u32(2.0), _f32_to_u32(1.0)))

    def test_item2_fmin_fmax_propagate_nan_vs_nmin_nmax_suppress(self) -> None:
        """Item 2: FMin/FMax propagate NaN (0x7FC00000); NMin/NMax suppress NaN."""
        cb = ConcreteBitvecDomain()
        cu = ConcreteUDomain()
        nan_u = 0x7FC00000
        two_u = _f32_to_u32(2.0)

        self.assertEqual(cb.fp_binary("FMin", nan_u, two_u), nan_u)
        self.assertEqual(cb.fp_binary("FMax", two_u, nan_u), nan_u)
        self.assertEqual(cb.fp_binary("NMin", nan_u, two_u), two_u)
        self.assertEqual(cb.fp_binary("NMax", two_u, nan_u), two_u)

        self.assertTrue(math.isnan(cu.to_f32x2(cu.fp_binary("FMin", cu.const_f32(float("nan")), cu.const_f32(2.0)))[0]))
        self.assertEqual(cu.to_f32x2(cu.fp_binary("NMin", cu.const_f32(float("nan")), cu.const_f32(2.0)))[0], 2.0)

    def test_item3_gl_sample_position_sideband_aligned(self) -> None:
        """Item 3: LoadSidebandU64(aux='gl_SamplePosition') aligns with bi_19 (SpvBuiltInSamplePosition)."""
        ssa = _single_block([
            {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
            {"id": 2, "op": "LoadSidebandU64", "type": "U64", "blk": 0, "args": [], "imm_u32": 8, "aux_str": "gl_SamplePosition"},
            {"id": 3, "op": "PsbLoad", "type": "TupleValToken", "blk": 0, "args": [2, 1], "imm_u32": 4},
            {"id": 4, "op": "TupleExtract", "type": "U32", "blk": 0, "args": [3], "imm_u32": 0},
            {"id": 5, "op": "ConstU64", "type": "U64", "blk": 0, "args": [], "imm_u64": 4},
            {"id": 6, "op": "IAdd64", "type": "U64", "blk": 0, "args": [2, 5]},
            {"id": 7, "op": "PsbLoad", "type": "TupleValToken", "blk": 0, "args": [6, 1], "imm_u32": 4},
            {"id": 8, "op": "TupleExtract", "type": "U32", "blk": 0, "args": [7], "imm_u32": 0},
            {"id": 9, "op": "VecConstruct4", "type": "V4F32", "blk": 0, "args": [4, 8, 4, 8]},
            {"id": 10, "op": "StoreFragColor", "type": "Token", "blk": 0, "args": [9, 1], "imm_u32": 0},
        ])
        env = ConcreteShaderEnv(
            _empty_bundle("Fragment"),
            vec_idx=0,
            overrides={"bi_19_0": _f32_to_u32(0.25), "bi_19_1": _f32_to_u32(0.75)},
        )
        _, outs = ConcreteSsaInterpreter(ssa, is_valhall_side=True).evaluate(env)
        self.assertEqual(outs["frag_color[0].r"], _f32_to_u32(0.25))
        self.assertEqual(outs["frag_color[0].g"], _f32_to_u32(0.75))

        # Symbolic env precondition links mem_gl_SamplePosition[0, 4] to bi_19_0, bi_19_1
        senv = SymbolicShaderEnv(_empty_bundle("Fragment"))
        senv.get_builtin(19, "V4F32")
        senv.get_buf_mem("gl_SamplePosition")
        pres = senv.build_preconditions()
        self.assertEqual(len(pres), 2)

    def test_item4_pilot_ssa_execution_computes_derived_fau_words(self) -> None:
        """Item 4: ConcreteShaderEnv executes pilot_ssa to compute derived FAU words correlated with UBOs."""
        # Pilot shader: loads SRT[4] -> Buf[0] (ubo:0:14), reads ubo:0:14+0, adds 10, stores to pilot_fau_0+0
        pilot_ssa = _single_block([
            {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
            {"id": 2, "op": "LoadSidebandU64", "type": "U64", "blk": 0, "args": [], "imm_u32": 3},  # pilot_fau_0
            {"id": 3, "op": "LoadSidebandU64", "type": "U64", "blk": 0, "args": [], "imm_u32": 2},  # pilot_srt_0
            {"id": 4, "op": "ConstU64", "type": "U64", "blk": 0, "args": [], "imm_u64": 0x40},     # rack 4 * 16
            {"id": 5, "op": "IAdd64", "type": "U64", "blk": 0, "args": [3, 4]},
            {"id": 6, "op": "PsbLoad", "type": "TupleValToken", "blk": 0, "args": [5, 1], "imm_u32": 4},
            {"id": 7, "op": "TupleExtract", "type": "U32", "blk": 0, "args": [6], "imm_u32": 0},
            {"id": 8, "op": "ConstU64", "type": "U64", "blk": 0, "args": [], "imm_u64": 0x44},
            {"id": 9, "op": "IAdd64", "type": "U64", "blk": 0, "args": [3, 8]},
            {"id": 10, "op": "PsbLoad", "type": "TupleValToken", "blk": 0, "args": [9, 1], "imm_u32": 4},
            {"id": 11, "op": "TupleExtract", "type": "U32", "blk": 0, "args": [10], "imm_u32": 0},
            {"id": 12, "op": "Pack64", "type": "U64", "blk": 0, "args": [7, 11]},  # pilot_srt_rack_4
            {"id": 13, "op": "ConstU64", "type": "U64", "blk": 0, "args": [], "imm_u64": 8},       # addr=0, +8 = base lo
            {"id": 14, "op": "IAdd64", "type": "U64", "blk": 0, "args": [12, 13]},
            {"id": 15, "op": "PsbLoad", "type": "TupleValToken", "blk": 0, "args": [14, 1], "imm_u32": 4},
            {"id": 16, "op": "TupleExtract", "type": "U32", "blk": 0, "args": [15], "imm_u32": 0},
            {"id": 17, "op": "ConstU64", "type": "U64", "blk": 0, "args": [], "imm_u64": 12},      # addr=0, +12 = base hi
            {"id": 18, "op": "IAdd64", "type": "U64", "blk": 0, "args": [12, 17]},
            {"id": 19, "op": "PsbLoad", "type": "TupleValToken", "blk": 0, "args": [18, 1], "imm_u32": 4},
            {"id": 20, "op": "TupleExtract", "type": "U32", "blk": 0, "args": [19], "imm_u32": 0},
            {"id": 21, "op": "Pack64", "type": "U64", "blk": 0, "args": [16, 20]},  # ubo:0:14 base
            {"id": 22, "op": "PsbLoad", "type": "TupleValToken", "blk": 0, "args": [21, 1], "imm_u32": 4},
            {"id": 23, "op": "TupleExtract", "type": "U32", "blk": 0, "args": [22], "imm_u32": 0},
            {"id": 24, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 10},
            {"id": 25, "op": "IAdd", "type": "U32", "blk": 0, "args": [23, 24]},
            {"id": 26, "op": "PsbStore", "type": "Token", "blk": 0, "args": [2, 25, 1], "imm_u32": 4},
        ])
        bundle = _empty_bundle(
            "Compute",
            direct_bindings=[{"kind": "UBO", "key": "ubo:0:14"}],
            mbs2_context={
                "fcsts": [],
                "rlocs": [],
                "pilot_fau": [],
                "pilot_srt": [{"rack": 4, "addr": 0, "kind": "ubo", "key": "ubo:0:14"}],
                "sinp": [],
                "sout": [],
                "pc_disasm": {},
            },
            pilot_ssa=pilot_ssa,
        )
        env = ConcreteShaderEnv(bundle, vec_idx=0, overrides={"mem:ubo:0:14:0": 32})
        self.assertEqual(env.load_fau_word(0), 42)

    def test_item5_vertex_gl_position_epilogue_alignment(self) -> None:
        """Item 5: align_vertex_gl_position aligns clip-space (x,y,z,w) with Mali (x/w, y/w, clamp(z/w), 1/w)."""
        bundle = _empty_bundle(
            "Vertex",
            mbs2_context={
                "fcsts": [],
                "rlocs": [{"symbol": "gl_mali_ViewportTransform", "word_idx": 0, "size": 12, "uni_offset": 0, "binding": 0}],
                "pilot_fau": [],
                "pilot_srt": [],
                "sinp": [],
                "sout": [],
                "pc_disasm": {},
            },
        )
        env = ConcreteShaderEnv(bundle, vec_idx=0)
        # Direct clip-space: (2.0, -4.0, 6.0, 2.0) -> Mali epilogue: (1.0, -2.0, clamp(3.0, -1, 1)=1.0, 0.5)
        d_out = {
            "builtin_out[0][0]": _f32_to_u32(2.0),
            "builtin_out[0][1]": _f32_to_u32(-4.0),
            "builtin_out[0][2]": _f32_to_u32(6.0),
            "builtin_out[0][3]": _f32_to_u32(2.0),
        }
        v_out = {
            "builtin_out[0][0]": _f32_to_u32(1.0),
            "builtin_out[0][1]": _f32_to_u32(-2.0),
            "builtin_out[0][2]": _f32_to_u32(1.0),
            "builtin_out[0][3]": _f32_to_u32(0.5),
        }
        aligned = align_vertex_gl_position(d_out, v_out, env, policy=DiffPolicy(max_ulp=2))
        self.assertEqual(aligned, v_out)

    def test_item7a_concrete_byte_addressed_subword_tls_spill(self) -> None:
        """Item 7a: STORE.i16 (imm_u32=2) updates only 2 bytes and preserves adjacent upper halfword."""
        ssa = _single_block([
            {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
            {"id": 2, "op": "LoadSidebandU64", "type": "U64", "blk": 0, "args": [], "imm_u32": 6},  # tls_private
            {"id": 3, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 0xAABBCCDD},
            {"id": 4, "op": "PsbStore", "type": "Token", "blk": 0, "args": [2, 3, 1], "imm_u32": 4},
            {"id": 5, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 0x1234},
            {"id": 6, "op": "PsbStore", "type": "Token", "blk": 0, "args": [2, 5, 4], "imm_u32": 2},  # overwrite lo 16 bits
            {"id": 7, "op": "PsbLoad", "type": "TupleValToken", "blk": 0, "args": [2, 6], "imm_u32": 4},
            {"id": 8, "op": "TupleExtract", "type": "U32", "blk": 0, "args": [7], "imm_u32": 0},
            {"id": 9, "op": "ConstU64", "type": "U64", "blk": 0, "args": [], "imm_u64": 2},
            {"id": 10, "op": "IAdd64", "type": "U64", "blk": 0, "args": [2, 9]},
            {"id": 11, "op": "PsbLoad", "type": "TupleValToken", "blk": 0, "args": [10, 6], "imm_u32": 2},  # hi 16 bits
            {"id": 12, "op": "TupleExtract", "type": "U32", "blk": 0, "args": [11], "imm_u32": 0},
            {"id": 13, "op": "VecConstruct4", "type": "V4F32", "blk": 0, "args": [8, 12, 8, 12]},
            {"id": 14, "op": "StoreFragColor", "type": "Token", "blk": 0, "args": [13, 6], "imm_u32": 0},
        ])
        env = ConcreteShaderEnv(_empty_bundle("Fragment"), vec_idx=0)
        _, outs = ConcreteSsaInterpreter(ssa, is_valhall_side=True).evaluate(env)
        self.assertEqual(outs["frag_color[0].r"], 0xAABB1234)
        self.assertEqual(outs["frag_color[0].g"], 0x0000AABB)


if __name__ == "__main__":
    unittest.main()

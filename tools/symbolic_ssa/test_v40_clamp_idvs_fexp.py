"""Regression tests for Checkpoint v40:
  1. NaN-suppressing `FClamp` / `.clamp_0_inf` (`NMax`) semantics in
     `ConcreteBitvecDomain` and `valhall_lowering.json` (`apply_f32_clamp`),
     preserving NaN-propagating `FMin`/`FMax` for `.nan_propagate`.
  2. `FEXP` `src1` Q8.24 overflow/underflow thresholds (`2147483648.0` /
     `-2516582400.0`) so normal inputs like `x = 1.0` (`src1 = 2^24`) do not
     falsely saturate to `+Inf` or `0.0`.
  3. Mali IDVS guardband `Select(FCmpOrdLt, FNeg(Bitcast(FDiv)), FDiv)`
     canonicalization in `canonicalize_vertex_viewport_epilogue`.
"""

from __future__ import annotations

import glob
import json
import math
import os
import struct
import subprocess
import tempfile
import unittest

from tools.symbolic_ssa.concrete_interpreter import ConcreteSsaInterpreter
from tools.symbolic_ssa.domains.concrete_bitvec import ConcreteBitvecDomain
from tools.symbolic_ssa.domains.shader_env.concrete_shader_env import ConcreteShaderEnv
from tools.verify import (
    InstLoweringContext,
    LoweringSemanticsExtractor,
    SourceOperandDesc,
)


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _bits_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


class TestV40ClampIdvsFexp(unittest.TestCase):
    def test_nan_suppressing_fclamp_and_clamp_0_inf_lowering(self) -> None:
        dom = ConcreteBitvecDomain()
        nan = 0x7FC00000
        zero = _f32_bits(0.0)
        one = _f32_bits(1.0)
        neg1 = _f32_bits(-1.0)

        # FClamp suppresses NaN to lo (matching Mali .clamp_0_1 / .clamp_m1_1).
        self.assertEqual(dom.fp_clamp(nan, zero, one), zero)
        self.assertEqual(dom.fp_clamp(nan, neg1, one), neg1)
        # FMin / FMax still propagate NaN for .nan_propagate.
        self.assertEqual(dom.fp_binary("FMin", nan, one), nan)
        self.assertEqual(dom.fp_binary("FMax", nan, zero), nan)
        # NMin / NMax suppress NaN.
        self.assertEqual(dom.fp_binary("NMin", nan, one), one)
        self.assertEqual(dom.fp_binary("NMax", nan, zero), zero)

        # Verify apply_f32_clamp emits NMax for clamp_0_inf.
        ext = LoweringSemanticsExtractor()
        ins = InstLoweringContext(
            mnemonic="FADD",
            variant="f32",
            srcs=[SourceOperandDesc(reg_num=0), SourceOperandDesc(reg_num=1)],
            modifiers={"clamp": "clamp_0_inf"},
            has_dest=True,
            dest_reg=2,
        )
        _, tr = ext.lower_instruction(ins)
        self.assertIn("NMax", tr.emitted_ops)

    def test_fexp_q8_24_src1_thresholds_do_not_clamp_normal_inputs(self) -> None:
        ext = LoweringSemanticsExtractor()
        ins = InstLoweringContext(
            mnemonic="FEXP",
            variant="f32",
            srcs=[SourceOperandDesc(reg_num=0), SourceOperandDesc(reg_num=1)],
            has_dest=True,
            dest_reg=2,
        )
        ssa, _ = ext.lower_instruction(ins)
        nodes = [
            {
                "id": n.id,
                "op": n.op,
                "type": n.val_type,
                "blk": 0,
                "args": list(n.args),
                "imm_u32": n.imm_u32,
                "aux_str": n.aux_str,
            }
            for n in ssa.nodes
        ]
        # Lowered constants must include 2^31 (0x4f000000) and -150*2^24 (0xcf160000).
        f32_consts = {n["imm_u32"] for n in nodes if n["op"] == "ConstF32"}
        self.assertIn(_f32_bits(2147483648.0), f32_consts)
        self.assertIn(_f32_bits(-2516582400.0), f32_consts)

        dst_nid = ssa.reg_writes[2]

        def eval_fexp(x: float) -> float:
            q_f = x * float(1 << 24)
            if math.isnan(x):
                q_s32 = 0
                src1_u = 0x7FC00000
            else:
                q_clamped = max(-(1 << 31), min((1 << 31) - 1, round(q_f)))
                q_s32 = q_clamped & 0xFFFFFFFF
                if math.isfinite(q_f):
                    src1_u = _f32_bits(q_f)
                else:
                    src1_u = _f32_bits(float("inf") if q_f > 0 else float("-inf"))

            mod = {
                "blocks": [{"id": 0, "nodes": [n["id"] for n in nodes]}],
                "nodes": [
                    (
                        {
                            "id": n["id"],
                            "op": "ConstU32",
                            "type": "U32",
                            "blk": 0,
                            "args": [],
                            "imm_u32": q_s32 if n["imm_u32"] == 0 else src1_u,
                        }
                        if n["op"] == "InputReg"
                        else n
                    )
                    for n in nodes
                ],
            }
            bundle = {"stage": "Compute", "direct_ssa": mod, "valhall_ssa": mod, "mbs2_context": {}}
            env = ConcreteShaderEnv(bundle, 0)
            vals, _ = ConcreteSsaInterpreter(mod, is_valhall_side=True).evaluate(env)
            return _bits_f32(vals[dst_nid])

        self.assertAlmostEqual(eval_fexp(1.0), 2.0, places=5)
        self.assertAlmostEqual(eval_fexp(-1.0), 0.5, places=5)
        self.assertTrue(math.isinf(eval_fexp(200.0)) and eval_fexp(200.0) > 0)
        self.assertEqual(eval_fexp(-200.0), 0.0)
        self.assertTrue(math.isnan(eval_fexp(float("nan"))))

    def test_idvs_guardband_vertex_epilogue_drops_redundant_screen_space_store(self) -> None:
        matches = glob.glob("/working_dir/c_e95e66c5dab0535f/spirv_corpus/spirv_samples/d09fd0a2a11c5680*.spv")
        bin_path = "./build/spirv2valhall"
        if not (matches and os.path.exists(bin_path)):
            self.skipTest("spirv2valhall binary or sample shader not present")
        spv = matches[0]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            tmp_json = tf.name
        try:
            subprocess.run(
                [bin_path, spv, "--dump-equiv-json", tmp_json, "--quiet", "--opt-ssa", "--defau"],
                check=True,
            )
            with open(tmp_json, "r", encoding="utf-8") as fh:
                bundle = json.load(fh)
        finally:
            if os.path.exists(tmp_json):
                os.unlink(tmp_json)
        pos_stores = [
            n for n in bundle["valhall_ssa"]["nodes"]
            if n["op"] == "StoreBuiltinOutput" and n.get("aux_str") == "gl_Position"
        ]
        # Redundant screen-space guardband store at 0xce0 must be eliminated, leaving 1 clip-space store.
        self.assertEqual(len(pos_stores), 1)


if __name__ == "__main__":
    unittest.main()

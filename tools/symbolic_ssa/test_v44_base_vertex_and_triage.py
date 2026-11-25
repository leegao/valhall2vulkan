"""Checkpoint v44 regression tests: Vertex BaseVertex/BaseInstance FAU zero-binding and triage detectors."""

from __future__ import annotations

import unittest

from tools.coverage.shrinker import DivergenceReport
from tools.coverage.triage import triage_divergence
from tools.symbolic_ssa.concrete_interpreter import ConcreteSsaInterpreter
from tools.symbolic_ssa.env import ConcreteShaderEnv, _f32_to_u32
from tools.symbolic_ssa.passes import canonicalize_ssa_module


class TestV44BaseVertexAndTriage(unittest.TestCase):
    def test_vertex_base_vertex_and_base_instance_pilot_fau_bind_to_zero(self) -> None:
        """Unresolved Vertex pilot_fau slots at byte_offset 76 (BaseVertex) and 80 (BaseInstance) bind to 0."""
        bundle = {
            "stage": "Vertex",
            "mbs2_context": {
                "pilot_fau": [
                    {
                        "fau_offset": 360,
                        "word_idx": 90,
                        "byte_size": 4,
                        "value_kind": "symbol_value",
                        "resolved": False,
                        "kind": "unknown",
                        "has_byte_offset": True,
                        "byte_offset": 80,
                        "key": "",
                    },
                    {
                        "fau_offset": 364,
                        "word_idx": 91,
                        "byte_size": 4,
                        "value_kind": "symbol_value",
                        "resolved": False,
                        "kind": "unknown",
                        "has_byte_offset": True,
                        "byte_offset": 76,
                        "key": "",
                    },
                ],
            },
        }
        ssa = {
            "blocks": [{"id": 0, "is_end": True, "nodes": [1, 2]}],
            "nodes": [
                {"id": 1, "op": "LoadFau", "type": "U32", "blk": 0, "args": [], "imm_u32": 90},
                {"id": 2, "op": "LoadFau", "type": "U32", "blk": 0, "args": [], "imm_u32": 91},
            ],
        }
        cenv = ConcreteShaderEnv(bundle, vec_idx=41)
        vals, _ = ConcreteSsaInterpreter(ssa, is_valhall_side=True).execute_one(cenv)
        self.assertEqual(vals[1], 0)
        self.assertEqual(vals[2], 0)

    def test_triage_detects_extreme_trig_range_and_nan_trig_clamp(self) -> None:
        """triage_divergence flags extreme_trig_range (|x|>50) and nan_trig_clamp on Sin/Cos cones."""
        for inp_bits, expected_sig in [
            (_f32_to_u32(728.0), "extreme_trig_range"),
            (0x7FC00000, "nan_trig_clamp"),
        ]:
            ssa = {
                "blocks": [{"id": 0, "is_end": True, "nodes": [1, 2, 3, 4]}],
                "nodes": [
                    {"id": 1, "op": "ConstF32", "type": "F32", "blk": 0, "args": [], "imm_u32": inp_bits},
                    {"id": 2, "op": "Cos", "type": "F32", "blk": 0, "args": [1]},
                    {"id": 3, "op": "TokenIn", "type": "Token", "blk": 0, "args": []},
                    {"id": 4, "op": "StoreBuiltinOutput", "type": "Token", "blk": 0, "args": [2, 3], "imm_u32": 0},
                ],
            }
            rep = DivergenceReport(
                diverged=True,
                output_name="builtin_out[0][0]",
                direct_val="0x3f800000",
                valhall_val="0xbf800000",
                original_override_count=0,
                shrunk_override_count=0,
                diff_kind="REAL_DIVERGENCE",
                vec_idx=0,
            )
            bundle = {"stage": "Vertex", "mbs2_context": {}}
            d_ssa, _ = canonicalize_ssa_module(ssa)
            tr = triage_divergence(bundle, d_ssa, ssa, rep)
            self.assertEqual(tr.suspect_category, "MODELING_ENV_OR_ABI")
            self.assertTrue(
                any(expected_sig in a for a in tr.anomalies),
                f"Expected {expected_sig} in {tr.anomalies}",
            )


if __name__ == "__main__":
    unittest.main()

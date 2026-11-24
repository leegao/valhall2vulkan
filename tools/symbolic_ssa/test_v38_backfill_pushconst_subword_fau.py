"""
Checkpoint v38 regression tests:
  - Item 1: `DemandBackfillAnalysis` propagates `D_BV32` through `ImageTexelAddr` so `udomain_backfill`
            never collapses symbolic texel coordinates to `bv32(0x0)` (eliminates `tex:0:6+0x0.written` on `fba18fbe`).
  - Item 2: `ShaderEnvBase._init_abi_layout` maps push-constant `rlocs` / `pilot_fau` to `"push_const"`
            instead of `"ubo:0:0"` when `push_const` is the sole binding-0 uniform block (`bcn_bc7.spv`).
  - Item 3: `ConcreteShaderEnv` and `SymbolicShaderEnv` defer sub-word `pilot_fau` (`byte_size < 4`)
            to `_ensure_pilot_executed()` instead of returning raw 32-bit UBO words.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest

import z3

from tools.symbolic_ssa.domains.backfill import (
    DemandBackfillAnalysis,
    DemandKind,
    annotate_ssa_demands,
    merge_miter_demands,
)
from tools.symbolic_ssa.domains.shader_env.concrete_shader_env import ConcreteShaderEnv
from tools.symbolic_ssa.domains.shader_env.symbolic_shader_env import SymbolicShaderEnv
from tools.symbolic_ssa.passes import canonicalize_ssa_module
from tools.symbolic_ssa.symbolic_interpreter import SymbolicSsaInterpreter


class TestV38BackfillPushConstSubwordFau(unittest.TestCase):
    def test_item1_backfill_propagates_demand_through_image_texel_addr(self) -> None:
        """ImageTexelAddr operands must receive BV32 demand, never BOT -> bv32(0x0)."""
        ssa = {
            "blocks": [{"id": 0, "preds": [], "succs": [], "nodes": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}],
            "nodes": [
                {"id": 1, "op": "LoadBuiltinInput", "type": "V3U32", "blk": 0, "args": [], "imm_u32": 26},
                {"id": 2, "op": "VecExtract", "type": "U32", "blk": 0, "args": [1], "imm_u32": 0},
                {"id": 3, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 0},
                {"id": 4, "op": "ImageTexelAddr", "type": "U64", "blk": 0, "args": [2, 3], "imm_u32": 6, "aux_str": "tex:0:6"},
                {"id": 5, "op": "Split64", "type": "TupleU32U32", "blk": 0, "args": [4]},
                {"id": 6, "op": "TupleExtract", "type": "U32", "blk": 0, "args": [5], "imm_u32": 0},
                {"id": 7, "op": "TupleExtract", "type": "U32", "blk": 0, "args": [5], "imm_u32": 1},
                {"id": 8, "op": "Pack64", "type": "U64", "blk": 0, "args": [6, 7]},
                {"id": 9, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 0xCAFEBABE},
                {"id": 10, "op": "PsbStore", "type": "Token", "blk": 0, "args": [8, 9, 3], "imm_u32": 4},
                {"id": 11, "op": "End", "type": "Token", "blk": 0, "args": [10]},
            ],
        }
        bundle = {"stage": "Compute", "mbs2_context": {}}
        env = SymbolicShaderEnv(bundle, domain="udomain_backfill")
        ann = DemandBackfillAnalysis(ssa, env, is_valhall=True).analyze()
        self.assertEqual(ann.node_types[2].kind, DemandKind.BV32)

        shared_in, shared_out = merge_miter_demands(ann, ann)
        env.set_shared_demands(shared_in, shared_out)
        ann_ssa = annotate_ssa_demands(ssa, ann)
        st = SymbolicSsaInterpreter(
            ann_ssa, env, is_valhall_side=True, domain="udomain_backfill", demand_annotations=ann
        ).evaluate()
        self.assertIn("tex:0:6", st.buf_mems)
        self.assertEqual(list(st.const_stores.keys()), [], "Symbolic ImageTexelAddr must not collapse to const offset 0")

    def test_item2_push_const_rlocs_and_pilot_fau_map_to_push_const(self) -> None:
        """Push-constant rlocs/pilot_fau must map to 'push_const' when 'ubo:0:0' is absent from direct_bindings."""
        bundle = {
            "stage": "Compute",
            "direct_bindings": [{"key": "push_const"}, {"key": "ssbo:0:0"}],
            "mbs2_context": {
                "pilot_srt": [{"rack": 0, "addr": 0, "kind": "ubo", "key": "push_const"}],
                "rlocs": [{"symbol": "_1679_registers", "binding": 0, "word_idx": 7, "uni_offset": 4, "size": 12}],
                "pilot_fau": [
                    {
                        "word_idx": 6,
                        "fau_offset": 24,
                        "byte_size": 4,
                        "value_kind": "symbol_value",
                        "resolved": True,
                        "is_loaded_value": True,
                        "has_byte_offset": True,
                        "byte_offset": 8,
                        "is_derived": False,
                        "key": "ubo:0:0",
                    }
                ],
            },
        }
        cenv = ConcreteShaderEnv(
            bundle,
            vec_idx=0,
            overrides={"mem:push_const:4": 0x11, "mem:push_const:8": 0x22, "mem:push_const:12": 0x33},
        )
        self.assertEqual(cenv.rloc_by_word[7], ("push_const", 4))
        self.assertEqual(cenv.rloc_by_word[8], ("push_const", 8))
        self.assertEqual(cenv.rloc_by_word[9], ("push_const", 12))
        self.assertEqual(cenv.load_fau_word(7), 0x11)
        self.assertEqual(cenv.load_fau_word(8), 0x22)
        self.assertEqual(cenv.load_fau_word(9), 0x33)
        self.assertEqual(cenv.load_fau_word(6), 0x22)

    def test_item3_subword_pilot_fau_defers_to_pilot_ssa(self) -> None:
        """pilot_fau entries with byte_size < 4 must execute pilot_ssa rather than reading a full 32-bit UBO word."""
        pilot_ssa = {
            "blocks": [{"id": 0, "preds": [], "succs": [], "nodes": [1, 2, 3, 4, 5, 6, 7]}],
            "nodes": [
                {"id": 1, "op": "LoadSidebandU64", "type": "U64", "blk": 0, "args": [], "imm_u32": 3},
                {"id": 2, "op": "ConstU64", "type": "U64", "blk": 0, "args": [], "imm_u64": 28},
                {"id": 3, "op": "IAdd64", "type": "U64", "blk": 0, "args": [1, 2]},
                {"id": 4, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 0x0000AAAB},
                {"id": 5, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 0},
                {"id": 6, "op": "PsbStore", "type": "Token", "blk": 0, "args": [3, 4, 5], "imm_u32": 4},
                {"id": 7, "op": "End", "type": "Token", "blk": 0, "args": [6]},
            ],
        }
        bundle = {
            "stage": "Fragment",
            "mbs2_context": {
                "pilot_fau": [
                    {
                        "word_idx": 7,
                        "fau_offset": 28,
                        "byte_size": 2,
                        "value_kind": "symbol_value",
                        "resolved": True,
                        "is_loaded_value": True,
                        "has_byte_offset": True,
                        "byte_offset": 2344,
                        "is_derived": False,
                        "key": "ubo:0:6",
                    }
                ],
            },
            "pilot_ssa": pilot_ssa,
        }
        cenv = ConcreteShaderEnv(bundle, vec_idx=0, overrides={"mem:ubo:0:6:2344": 0xAAAB1234})
        self.assertEqual(cenv.load_fau_word(7), 0x0000AAAB)

        senv = SymbolicShaderEnv(bundle, domain="udomain")
        bv = z3.simplify(senv.domain.lower_z3_bv32(senv.load_fau_word(7)).to_z3)
        self.assertTrue(z3.is_bv_value(bv))
        self.assertEqual(bv.as_long(), 0x0000AAAB)

    def test_item1_end_to_end_fba18fbe_has_no_const_zero_tex_store_under_backfill(self) -> None:
        """End-to-end check on samples/comp_unique_03_fba18fbe.spv under udomain_backfill."""
        bin_path = "./build/spirv2valhall"
        spv_path = "samples/comp_unique_03_fba18fbe.spv"
        if not (os.path.exists(bin_path) and os.path.exists(spv_path)):
            self.skipTest("spirv2valhall binary or sample not present")
        with tempfile.TemporaryDirectory() as td:
            jp = os.path.join(td, "fba.json")
            subprocess.run([bin_path, spv_path, "--dump-equiv-json", jp, "--quiet"], check=True)
            with open(jp) as f:
                bundle = json.load(f)
        d_can, _ = canonicalize_ssa_module(bundle["direct_ssa"])
        v_can, _ = canonicalize_ssa_module(bundle["valhall_ssa"])
        env = SymbolicShaderEnv(bundle, domain="udomain_backfill")
        d_ann = DemandBackfillAnalysis(d_can, env, is_valhall=False).analyze()
        v_ann = DemandBackfillAnalysis(v_can, env, is_valhall=True).analyze()
        shared_in, shared_out = merge_miter_demands(d_ann, v_ann)
        env.set_shared_demands(shared_in, shared_out)
        v_ann_ssa = annotate_ssa_demands(v_can, v_ann)
        st = SymbolicSsaInterpreter(
            v_ann_ssa, env, is_valhall_side=True, domain="udomain_backfill", demand_annotations=v_ann
        ).evaluate()
        self.assertIn("tex:0:6", st.buf_mems)
        self.assertNotIn(("tex:0:6", 0), st.const_stores)


if __name__ == "__main__":
    unittest.main()

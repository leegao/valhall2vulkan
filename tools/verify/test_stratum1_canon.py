#!/usr/bin/env python3
"""Unit tests for Stratum 1 pre-SMT structural canonicalization."""

import unittest

from tools.verify.stratum1_canon import canonicalize_ssa_module, cfg_isomorphic


class TestStratum1Canon(unittest.TestCase):
    def test_contract_trivial_chain(self) -> None:
        ssa = {
            "blocks": [
                {
                    "id": 0, "preds": [], "phis": [], "nodes": [1, 2],
                    "branch_target": -1, "fallthrough": 1,
                    "is_cond": False, "is_uncond": False, "is_end": False, "is_discard": False,
                    "is_loop_header": False, "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
                },
                {
                    "id": 1, "preds": [0], "phis": [], "nodes": [3, 4],
                    "branch_target": -1, "fallthrough": -1,
                    "is_cond": False, "is_uncond": False, "is_end": True, "is_discard": False,
                    "is_loop_header": False, "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
                },
            ],
            "nodes": [
                {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                {"id": 2, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 42},
                {"id": 3, "op": "VecConstruct4", "type": "V4F32", "blk": 1, "args": [2, 2, 2, 2]},
                {"id": 4, "op": "StoreFragColor", "type": "Token", "blk": 1, "args": [3, 1], "imm_u32": 0},
            ],
        }
        canon, st = canonicalize_ssa_module(ssa)
        self.assertEqual(len(canon["blocks"]), 1)
        self.assertEqual(st.chains_contracted, 1)
        self.assertTrue(canon["blocks"][0]["is_end"])

    def test_predicated_discard_folding(self) -> None:
        ssa = {
            "blocks": [
                {
                    "id": 0, "preds": [], "phis": [], "nodes": [1, 2, 3],
                    "branch_target": 2, "fallthrough": 1,
                    "is_cond": True, "is_uncond": False, "is_end": False, "is_discard": False,
                    "is_loop_header": False, "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 3,
                },
                {
                    "id": 1, "preds": [0], "phis": [], "nodes": [4, 5],
                    "branch_target": -1, "fallthrough": -1,
                    "is_cond": False, "is_uncond": False, "is_end": True, "is_discard": False,
                    "is_loop_header": False, "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
                },
                {
                    "id": 2, "preds": [0], "phis": [], "nodes": [6],
                    "branch_target": -1, "fallthrough": -1,
                    "is_cond": False, "is_uncond": False, "is_end": False, "is_discard": True,
                    "is_loop_header": False, "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
                },
            ],
            "nodes": [
                {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                {"id": 2, "op": "LoadBuiltinInput", "type": "U32", "blk": 0, "args": [], "imm_u32": 15},
                {"id": 3, "op": "ICmpEq", "type": "Bool", "blk": 0, "args": [2, 2]},
                {"id": 4, "op": "VecConstruct4", "type": "V4F32", "blk": 1, "args": [2, 2, 2, 2]},
                {"id": 5, "op": "StoreFragColor", "type": "Token", "blk": 1, "args": [4, 1], "imm_u32": 0},
                {"id": 6, "op": "Kill", "type": "Token", "blk": 2, "args": [1]},
            ],
        }
        canon, st = canonicalize_ssa_module(ssa)
        self.assertEqual(len(canon["blocks"]), 1)
        self.assertEqual(st.discards_predicated, 1)
        kills = [n for n in canon["nodes"] if n["op"] == "Kill"]
        self.assertEqual(len(kills), 1)
        self.assertTrue(kills[0].get("aux_bool"))
        self.assertEqual(kills[0]["args"][0], 3)

    def test_pure_triangle_if_conversion_and_gvn(self) -> None:
        ssa = {
            "blocks": [
                {
                    "id": 0, "preds": [], "phis": [], "nodes": [1, 2, 3, 4],
                    "branch_target": 1, "fallthrough": 2,
                    "is_cond": True, "is_uncond": False, "is_end": False, "is_discard": False,
                    "is_loop_header": False, "loop_backedge": -1, "loop_merge": -1, "sel_merge": 2, "branch_cond": 4,
                },
                {
                    "id": 1, "preds": [0], "phis": [], "nodes": [5],
                    "branch_target": 2, "fallthrough": -1,
                    "is_cond": False, "is_uncond": True, "is_end": False, "is_discard": False,
                    "is_loop_header": False, "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
                },
                {
                    "id": 2, "preds": [0, 1], "phis": [6], "nodes": [7, 8, 9],
                    "branch_target": -1, "fallthrough": -1,
                    "is_cond": False, "is_uncond": False, "is_end": True, "is_discard": False,
                    "is_loop_header": False, "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
                },
            ],
            "nodes": [
                {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                {"id": 2, "op": "LoadBuiltinInput", "type": "U32", "blk": 0, "args": [], "imm_u32": 15},
                {"id": 3, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 1},
                {"id": 4, "op": "ICmpEq", "type": "Bool", "blk": 0, "args": [2, 3]},
                {"id": 5, "op": "IAdd", "type": "U32", "blk": 1, "args": [2, 3]},
                {"id": 6, "op": "Phi", "type": "U32", "blk": 2, "args": [], "phi": [[0, 2], [1, 5]]},
                {"id": 7, "op": "IAdd", "type": "U32", "blk": 2, "args": [3, 2]},  # commutative duplicate of 5
                {"id": 8, "op": "VecConstruct4", "type": "V4F32", "blk": 2, "args": [6, 7, 6, 7]},
                {"id": 9, "op": "StoreFragColor", "type": "Token", "blk": 2, "args": [8, 1], "imm_u32": 0},
            ],
        }
        canon, st = canonicalize_ssa_module(ssa)
        self.assertEqual(len(canon["blocks"]), 1)
        self.assertEqual(st.triangles_if_converted, 1)
        self.assertGreaterEqual(st.gvn_merged, 1)
        self.assertTrue(any(n["op"] == "Select" for n in canon["nodes"]))
        self.assertTrue(cfg_isomorphic(canon, canon))

    def test_readonly_token_detachment_enables_gvn(self) -> None:
        ssa = {
            "blocks": [
                {
                    "id": 0, "preds": [], "phis": [], "nodes": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                    "branch_target": -1, "fallthrough": -1,
                    "is_cond": False, "is_uncond": False, "is_end": True, "is_discard": False,
                    "is_loop_header": False, "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
                },
            ],
            "nodes": [
                {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                {"id": 2, "op": "LoadSidebandU64", "type": "U64", "blk": 0, "args": [], "aux_str": "ubo:0:0"},
                {"id": 3, "op": "PsbLoad", "type": "TupleValToken", "blk": 0, "args": [2, 1], "imm_u32": 4, "aux_str": "ubo:0:0"},
                {"id": 4, "op": "TupleExtract", "type": "U32", "blk": 0, "args": [3], "imm_u32": 0},
                {"id": 5, "op": "TupleExtract", "type": "Token", "blk": 0, "args": [3], "imm_u32": 1},
                {"id": 6, "op": "PsbLoad", "type": "TupleValToken", "blk": 0, "args": [2, 5], "imm_u32": 4, "aux_str": "ubo:0:0"},
                {"id": 7, "op": "TupleExtract", "type": "U32", "blk": 0, "args": [6], "imm_u32": 0},
                {"id": 8, "op": "TupleExtract", "type": "Token", "blk": 0, "args": [6], "imm_u32": 1},
                {"id": 9, "op": "IAdd", "type": "U32", "blk": 0, "args": [4, 7]},
                {"id": 10, "op": "VecConstruct4", "type": "V4F32", "blk": 0, "args": [9, 9, 9, 9]},
                {"id": 11, "op": "StoreFragColor", "type": "Token", "blk": 0, "args": [10, 8], "imm_u32": 0},
            ],
        }
        canon, st = canonicalize_ssa_module(ssa)
        self.assertGreaterEqual(st.readonly_tokens_detached, 1)
        loads = [n for n in canon["nodes"] if n["op"] == "PsbLoad"]
        self.assertEqual(len(loads), 1)


if __name__ == "__main__":
    unittest.main()

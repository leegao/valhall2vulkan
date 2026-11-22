#!/usr/bin/env python3
"""Unit tests for concrete differential simulation and incremental CEC cutpoint sweeping."""

import unittest

from tools.verify.cec import (
    discover_cutpoint_candidates,
    run_concrete_diff,
    verify_with_cec,
)


def _make_fp_chain_bundle(depth: int = 6, inject_bug: bool = False):
    """Build a synthetic 1-block Fragment shader with a deep FP32 chain."""
    def build_side(bug: bool):
        nodes = [
            {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
            {"id": 2, "op": "LoadFragVarying", "type": "V4F32", "blk": 0, "args": [], "imm_u32": 0},
            {"id": 3, "op": "VecExtract", "type": "F32", "blk": 0, "args": [2], "imm_u32": 0},
            {"id": 4, "op": "ConstF32", "type": "F32", "blk": 0, "args": [], "imm_u32": 0x3F000000},
        ]
        cur = 3
        nid = 5
        for step in range(depth):
            mul_id = nid
            nodes.append({"id": mul_id, "op": "FMul", "type": "F32", "blk": 0, "args": [cur, 4]})
            nid += 1
            add_id = nid
            op = "FSub" if (bug and step == depth - 1) else "FAdd"
            nodes.append({"id": add_id, "op": op, "type": "F32", "blk": 0, "args": [mul_id, cur]})
            nid += 1
            cur = add_id
        vec_id = nid
        nodes.append({"id": vec_id, "op": "VecConstruct4", "type": "V4F32", "blk": 0, "args": [cur, cur, cur, cur]})
        nid += 1
        st_id = nid
        nodes.append({"id": st_id, "op": "StoreFragColor", "type": "Token", "blk": 0, "args": [vec_id, 1], "imm_u32": 0})
        blk = {
            "id": 0, "preds": [], "phis": [], "nodes": [n["id"] for n in nodes],
            "branch_target": -1, "fallthrough": -1,
            "is_cond": False, "is_uncond": False, "is_end": True, "is_discard": False,
            "is_loop_header": False, "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
        }
        return {"blocks": [blk], "nodes": nodes}

    return {
        "stage": "Fragment",
        "direct_ok": True,
        "valhall_ok": True,
        "mbs2_context": {},
        "direct_bindings": [],
        "direct_ssa": build_side(False),
        "valhall_ssa": build_side(inject_bug),
    }


class TestCec(unittest.TestCase):
    def test_concrete_diff_catches_bug_instantly(self) -> None:
        b = _make_fp_chain_bundle(depth=4, inject_bug=True)
        res = run_concrete_diff(b, b["direct_ssa"], b["valhall_ssa"], k=16)
        self.assertEqual(res.status, "MISMATCH")
        self.assertGreater(res.mismatched_outputs, 0)
        self.assertIsNotNone(res.first_mismatch)

    def test_cec_cutpoint_sweeping_is_incremental_and_unsat(self) -> None:
        depth = 8
        b = _make_fp_chain_bundle(depth=depth, inject_bug=False)
        res = verify_with_cec(b, use_stratum1=True, k_sim=16, cone_timeout_ms=500, output_timeout_ms=1000)
        self.assertEqual(res.status, "UNSAT")
        self.assertEqual(res.concrete_status, "EXACT_MATCH")
        self.assertGreaterEqual(res.cutpoints_proved, depth)
        self.assertEqual(res.unsat_outputs, res.checked_outputs)
        # Incremental sweeping evaluates O(N) nodes across both sides, far below O(M*N).
        total_nodes = len(b["direct_ssa"]["nodes"]) + len(b["valhall_ssa"]["nodes"])
        self.assertLessEqual(res.sweep_evals, total_nodes)

    def test_polarity_flipped_fneg_cutpoint_discovery(self) -> None:
        d_ssa = {
            "blocks": [{
                "id": 0, "preds": [], "phis": [], "nodes": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                "branch_target": -1, "fallthrough": -1,
                "is_cond": False, "is_uncond": False, "is_end": True, "is_discard": False,
                "is_loop_header": False, "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
            }],
            "nodes": [
                {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                {"id": 2, "op": "LoadFragVarying", "type": "V4F32", "blk": 0, "args": [], "imm_u32": 0},
                {"id": 3, "op": "VecExtract", "type": "F32", "blk": 0, "args": [2], "imm_u32": 0},
                {"id": 4, "op": "ConstF32", "type": "F32", "blk": 0, "args": [], "imm_u32": 0x40000000},
                {"id": 5, "op": "FNeg", "type": "F32", "blk": 0, "args": [3]},
                {"id": 6, "op": "FMul", "type": "F32", "blk": 0, "args": [5, 4]},
                {"id": 7, "op": "FAbs", "type": "F32", "blk": 0, "args": [6]},
                {"id": 8, "op": "VecConstruct4", "type": "V4F32", "blk": 0, "args": [7, 7, 7, 7]},
                {"id": 9, "op": "StoreFragColor", "type": "Token", "blk": 0, "args": [8, 1], "imm_u32": 0},
            ],
        }
        v_ssa = {
            "blocks": [{
                "id": 0, "preds": [], "phis": [], "nodes": [1, 2, 3, 4, 5, 6, 7, 8],
                "branch_target": -1, "fallthrough": -1,
                "is_cond": False, "is_uncond": False, "is_end": True, "is_discard": False,
                "is_loop_header": False, "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
            }],
            "nodes": [
                {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                {"id": 2, "op": "LoadFragVarying", "type": "V4F32", "blk": 0, "args": [], "imm_u32": 0},
                {"id": 3, "op": "VecExtract", "type": "F32", "blk": 0, "args": [2], "imm_u32": 0},
                {"id": 4, "op": "ConstF32", "type": "F32", "blk": 0, "args": [], "imm_u32": 0x40000000},
                {"id": 5, "op": "FMul", "type": "F32", "blk": 0, "args": [3, 4]},
                {"id": 6, "op": "FAbs", "type": "F32", "blk": 0, "args": [5]},
                {"id": 7, "op": "VecConstruct4", "type": "V4F32", "blk": 0, "args": [6, 6, 6, 6]},
                {"id": 8, "op": "StoreFragColor", "type": "Token", "blk": 0, "args": [7, 1], "imm_u32": 0},
            ],
        }
        b = {
            "stage": "Fragment", "direct_ok": True, "valhall_ok": True,
            "mbs2_context": {}, "direct_bindings": [],
            "direct_ssa": d_ssa, "valhall_ssa": v_ssa,
        }
        cd = run_concrete_diff(b, d_ssa, v_ssa, k=16)
        self.assertEqual(cd.status, "EXACT_MATCH")
        cands = discover_cutpoint_candidates(d_ssa, v_ssa, cd.d_node_traces, cd.v_node_traces)
        self.assertTrue(any(kind == "FNEG" for _, _, kind in cands))
        res = verify_with_cec(b, use_stratum1=True, k_sim=16)
        self.assertEqual(res.status, "UNSAT")


if __name__ == "__main__":
    unittest.main()

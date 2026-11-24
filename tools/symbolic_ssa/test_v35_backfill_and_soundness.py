"""
Unit and integration tests for Checkpoint v35:
  - Demand-Driven Backfill UDomain (`U^bf`) backward dataflow analysis,
    SSA demand tagging, kill-time dead code elision, per-offset UBO struct
    granularity, and bridge-free two-shader miters.
  - External soundness review fixes (Gaps 1–7):
      1. Texture & Image op non-UV arguments and 3D StoreTexel coordinates
      2. Stale `const_stores` invalidation after dynamic stores & multi-offset `sym_buf_stores`
      3. Bounded loop unrolling truncation guards (`BMC_EQUIV(k)` vs full `EQUIV`)
      4. Concurrency (`ControlBarrier` havoc + pre-barrier snapshots, atomic `old_val` havoc, quad derivatives)
      5. Byte-addressed overlapping memory model & cross-binding SSBO aliasing
      6. IEEE-754 edge cases (`ConvertFToS`/`ConvertFToU` trunc-toward-zero + saturation, `strict_fp_edges`)
      7. IEEE sign-bit bridge axioms (`BvAlgebraicPass` + `EmatchingUnifyPass`) and integer div-by-zero guards
"""

from __future__ import annotations

import unittest
from typing import Any

import z3

from tools.symbolic_ssa.domains.backfill import (
    D_BV32,
    D_F32,
    D_F32X2,
    D_TOP,
    D_UNUSED,
    DemandBackfillAnalysis,
    annotate_ssa_demands,
    demand_join,
    demand_leq,
    demand_ptr,
    merge_miter_demands,
    needs_bridge,
)
from tools.symbolic_ssa.domains.shader_env.symbolic_shader_env import SymbolicShaderEnv
from tools.symbolic_ssa.domains.udomain import SymbolicZ3UDomain
from tools.symbolic_ssa.symbolic_interpreter import SymbolicSsaInterpreter
from tools.symbolic_ssa.z3simplifier import (
    BvAlgebraicPass,
    EmatchingUnifyPass,
    SimplificationContext,
    SimplifyQuery,
    build_default_pipeline,
    solve_simplified_miter,
)
from tools.symbolic_ssa.z3utils import dsl
from tools.verify.shader_equiv_verifier import verify_equivalence_bundle


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


def _make_bundle(
    direct_ssa: dict[str, Any],
    valhall_ssa: dict[str, Any],
    *,
    stage: str = "Fragment",
    bindings: list[dict[str, Any]] | None = None,
    rlocs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "spv_path": "synthetic_v35.spv",
        "stage": stage,
        "arch": 10,
        "direct_ok": True,
        "valhall_ok": True,
        "direct_bindings": bindings or [],
        "mbs2_context": {
            "fcsts": [],
            "rlocs": rlocs or [],
            "pilot_fau": [],
            "sinp": [],
            "sout": [],
            "pc_disasm": {},
        },
        "direct_ssa": direct_ssa,
        "valhall_ssa": valhall_ssa,
    }


class TestDemandDrivenBackfillUDomain(unittest.TestCase):
    """Tests for Workstream A: Demand-Driven Backfill UDomain (`U^bf`)."""

    def test_lattice_join_order_and_bridge_rule(self) -> None:
        self.assertEqual(demand_join(D_UNUSED, D_F32), D_F32)
        self.assertEqual(demand_join(D_F32, D_F32X2), D_F32X2)
        self.assertEqual(demand_join(D_F32, D_BV32), D_TOP)
        self.assertEqual(demand_join(demand_ptr("ssbo:0:0"), demand_ptr("ssbo:0:0")), demand_ptr("ssbo:0:0"))
        self.assertEqual(demand_join(demand_ptr("ssbo:0:0"), demand_ptr("ssbo:0:1")), D_TOP)
        self.assertTrue(demand_leq(D_UNUSED, D_F32))
        self.assertTrue(demand_leq(D_F32, D_F32X2))
        self.assertTrue(demand_leq(D_F32X2, D_TOP))
        self.assertFalse(needs_bridge(D_F32, D_F32))
        self.assertFalse(needs_bridge(D_F32X2, D_F32))
        self.assertTrue(needs_bridge(D_TOP, D_F32))
        self.assertTrue(needs_bridge(D_F32, D_TOP))

    def test_worked_bithack_example_tagging_and_dead_elision(self) -> None:
        """Section 7.4 worked example: x used by FAdd and BitXor; dead FMul elided."""
        ssa = _single_block([
            {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
            {"id": 2, "op": "LoadFragVarying", "type": "V4F32", "blk": 0, "args": [], "imm_u32": 0},
            {"id": 3, "op": "VecExtract", "type": "U32", "blk": 0, "args": [2], "imm_u32": 0},
            {"id": 4, "op": "ConstF32", "type": "F32", "blk": 0, "args": [], "imm_u32": 0x40000000},  # 2.0
            {"id": 5, "op": "FMul", "type": "F32", "blk": 0, "args": [3, 4]},  # dead
            {"id": 6, "op": "ConstF32", "type": "F32", "blk": 0, "args": [], "imm_u32": 0x3F800000},  # 1.0
            {"id": 7, "op": "FAdd", "type": "F32", "blk": 0, "args": [3, 6]},  # y = x + 1.0
            {"id": 8, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 0x80000000},
            {"id": 9, "op": "BitXor", "type": "U32", "blk": 0, "args": [3, 8]},  # z = x ^ 0x80000000
            {"id": 10, "op": "FAdd", "type": "F32", "blk": 0, "args": [7, 9]},  # w = y + z
            {"id": 11, "op": "VecConstruct4", "type": "V4F32", "blk": 0, "args": [10, 6, 6, 6]},
            {"id": 12, "op": "StoreFragColor", "type": "Token", "blk": 0, "args": [11, 1], "imm_u32": 0},
        ])
        ann = DemandBackfillAnalysis(ssa).analyze()
        self.assertEqual(ann.tag_of(3), D_TOP)      # x: F32 |_| BV32 = TopBits
        self.assertEqual(ann.tag_of(5), D_UNUSED)   # dead = FMul(x, 2.0)
        self.assertIn(5, ann.dead_nodes)
        self.assertEqual(ann.tag_of(7), D_F32)      # y = FAdd(x, 1.0)
        self.assertEqual(ann.tag_of(9), D_TOP)      # z = BitXor def BV32, use F32
        self.assertEqual(ann.tag_of(10), D_F32)     # w = FAdd(y, z)

        tagged = annotate_ssa_demands(ssa, ann)
        by_id = {n["id"]: n["demand_type"] for n in tagged["nodes"]}
        self.assertEqual(by_id[3], "TopBits")
        self.assertEqual(by_id[5], "Unused")
        self.assertEqual(by_id[7], "F32")
        self.assertEqual(by_id[9], "TopBits")
        self.assertEqual(by_id[10], "F32")

        env = SymbolicShaderEnv(_make_bundle(ssa, ssa), domain="udomain_backfill")
        env.set_shared_demands(ann.input_demands, ann.output_demands)
        interp = SymbolicSsaInterpreter(
            tagged, env, is_valhall_side=False, domain="udomain_backfill", demand_annotations=ann
        )
        interp.evaluate()
        self.assertGreaterEqual(interp.elided_dead_nodes, 1)

    def test_pure_fp_shader_emits_zero_bridges_and_pure_real(self) -> None:
        """Pure FP shader under udomain_backfill must emit zero bridges and pure Real output."""
        direct = _single_block([
            {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
            {"id": 2, "op": "LoadFragVarying", "type": "V4F32", "blk": 0, "args": [], "imm_u32": 0},
            {"id": 3, "op": "VecExtract", "type": "U32", "blk": 0, "args": [2], "imm_u32": 0},
            {"id": 4, "op": "LoadSidebandU64", "type": "U64", "blk": 0, "args": [], "imm_u32": 0, "aux_str": "ubo:0:0"},
            {"id": 5, "op": "PsbLoad", "type": "TupleValToken", "blk": 0, "args": [4, 1], "aux_str": "ubo:0:0"},
            {"id": 6, "op": "TupleExtract", "type": "U32", "blk": 0, "args": [5], "imm_u32": 0},
            {"id": 7, "op": "Bitcast", "type": "F32", "blk": 0, "args": [3]},
            {"id": 8, "op": "Bitcast", "type": "F32", "blk": 0, "args": [6]},
            {"id": 9, "op": "ConstF32", "type": "F32", "blk": 0, "args": [], "imm_u32": 0x3F000000},  # 0.5
            {"id": 10, "op": "Fma", "type": "F32", "blk": 0, "args": [7, 8, 9]},
            {"id": 11, "op": "VecConstruct4", "type": "V4F32", "blk": 0, "args": [10, 9, 9, 9]},
            {"id": 12, "op": "StoreFragColor", "type": "Token", "blk": 0, "args": [11, 1], "imm_u32": 0},
        ])
        valhall = _single_block([
            {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
            {"id": 2, "op": "LoadFragVarying", "type": "V4F32", "blk": 0, "args": [], "imm_u32": 0},
            {"id": 3, "op": "VecExtract", "type": "U32", "blk": 0, "args": [2], "imm_u32": 0},
            {"id": 4, "op": "LoadFau", "type": "U32", "blk": 0, "args": [], "imm_u32": 0},
            {"id": 5, "op": "ConstF32", "type": "F32", "blk": 0, "args": [], "imm_u32": 0x3F000000},
            {"id": 6, "op": "FMul", "type": "F32", "blk": 0, "args": [3, 4]},
            {"id": 7, "op": "FAdd", "type": "F32", "blk": 0, "args": [6, 5]},
            {"id": 8, "op": "VecConstruct4", "type": "V4F32", "blk": 0, "args": [7, 5, 5, 5]},
            {"id": 9, "op": "StoreFragColor", "type": "Token", "blk": 0, "args": [8, 1], "imm_u32": 0},
        ])
        bundle = _make_bundle(
            direct,
            valhall,
            bindings=[{"key": "ubo:0:0", "kind": "UBO"}],
            rlocs=[{"symbol": "u0", "word_idx": 0, "size": 4, "uni_offset": 0, "binding": 0}],
        )
        env = SymbolicShaderEnv(bundle, domain="udomain_backfill")
        d_ann = DemandBackfillAnalysis(direct, env, is_valhall=False).analyze()
        v_ann = DemandBackfillAnalysis(valhall, env, is_valhall=True).analyze()
        s_in, s_out = merge_miter_demands(d_ann, v_ann)
        self.assertEqual(s_in["frag_var_loc0_0"], D_F32)
        self.assertEqual(s_in["buf_slot:ubo:0:0:0"], D_F32)
        self.assertEqual(s_out["frag_color[0][0]"], D_F32)

        env.set_shared_demands(s_in, s_out)
        d_interp = SymbolicSsaInterpreter(direct, env, is_valhall_side=False, domain="udomain_backfill", demand_annotations=d_ann)
        v_interp = SymbolicSsaInterpreter(valhall, env, is_valhall_side=True, domain="udomain_backfill", demand_annotations=v_ann)
        d_st = d_interp.evaluate()
        v_st = v_interp.evaluate()
        self.assertEqual(d_interp.domain.bitcast_materializations, 0)
        self.assertEqual(v_interp.domain.bitcast_materializations, 0)
        self.assertEqual(d_st.frag_colors[0][0].sort().kind(), dsl.Z3SortKind.REAL)
        self.assertEqual(v_st.frag_colors[0][0].sort().kind(), dsl.Z3SortKind.REAL)

        res = verify_equivalence_bundle(bundle, domain="udomain_backfill", timeout_ms=2000)
        self.assertEqual(res.status, "UNSAT")

    def test_per_offset_ubo_struct_granularity(self) -> None:
        """Gap #7b: mixed-type UBO (float at +0x0, uint at +0x4) must not escalate +0x0 to TopBits."""
        ssa = _single_block([
            {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
            {"id": 2, "op": "LoadSidebandU64", "type": "U64", "blk": 0, "args": [], "imm_u32": 0, "aux_str": "ubo:0:0"},
            {"id": 3, "op": "PsbLoad", "type": "TupleValToken", "blk": 0, "args": [2, 1], "aux_str": "ubo:0:0"},
            {"id": 4, "op": "TupleExtract", "type": "U32", "blk": 0, "args": [3], "imm_u32": 0},  # +0x0 (float)
            {"id": 5, "op": "ConstU64", "type": "U64", "blk": 0, "args": [], "imm_u64": 4},
            {"id": 6, "op": "IAdd64", "type": "U64", "blk": 0, "args": [2, 5]},
            {"id": 7, "op": "PsbLoad", "type": "TupleValToken", "blk": 0, "args": [6, 1], "aux_str": "ubo:0:0"},
            {"id": 8, "op": "TupleExtract", "type": "U32", "blk": 0, "args": [7], "imm_u32": 0},  # +0x4 (uint)
            {"id": 9, "op": "ConstF32", "type": "F32", "blk": 0, "args": [], "imm_u32": 0x3F800000},
            {"id": 10, "op": "FAdd", "type": "F32", "blk": 0, "args": [4, 9]},
            {"id": 11, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 7},
            {"id": 12, "op": "IAdd", "type": "U32", "blk": 0, "args": [8, 11]},
            {"id": 13, "op": "VecConstruct4", "type": "V4F32", "blk": 0, "args": [10, 12, 9, 9]},
            {"id": 14, "op": "StoreFragColor", "type": "Token", "blk": 0, "args": [13, 1], "imm_u32": 0},
        ])
        ann = DemandBackfillAnalysis(ssa).analyze()
        self.assertEqual(ann.input_demands.get("buf_slot:ubo:0:0:0"), D_F32)
        self.assertEqual(ann.input_demands.get("buf_slot:ubo:0:0:4"), D_BV32)


class TestSoundnessReviewGaps(unittest.TestCase):
    """Regression tests for all 7 external-review soundness gaps."""

    def test_gap1_tex_sample_lod_and_op_distinguished(self) -> None:
        """Gap #1: TexSampleLod with different LODs or distinct opcodes must return SAT."""
        def make_tex_shader(op: str, lod_hex: int) -> dict[str, Any]:
            return _single_block([
                {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                {"id": 2, "op": "LoadFragVarying", "type": "V4F32", "blk": 0, "args": [], "imm_u32": 0},
                {"id": 3, "op": "VecExtract", "type": "F32", "blk": 0, "args": [2], "imm_u32": 0},
                {"id": 4, "op": "VecExtract", "type": "F32", "blk": 0, "args": [2], "imm_u32": 1},
                {"id": 5, "op": "VecConstruct2", "type": "V2F32", "blk": 0, "args": [3, 4]},
                {"id": 6, "op": "ConstF32", "type": "F32", "blk": 0, "args": [], "imm_u32": lod_hex},
                {"id": 7, "op": op, "type": "TupleValToken", "blk": 0, "args": [5, 6, 1], "aux_str": "tex:0:0"},
                {"id": 8, "op": "TupleExtract", "type": "V4F32", "blk": 0, "args": [7], "imm_u32": 0},
                {"id": 9, "op": "StoreFragColor", "type": "Token", "blk": 0, "args": [8, 1], "imm_u32": 0},
            ])

        # Different LOD (0.0 vs 4.0) -> SAT
        res_lod = verify_equivalence_bundle(
            _make_bundle(make_tex_shader("TexSampleLod", 0x00000000), make_tex_shader("TexSampleLod", 0x40800000)),
            timeout_ms=2000,
        )
        self.assertEqual(res_lod.status, "SAT")

        # Distinct opcodes (TexSampleLod vs TexFetch) -> SAT
        res_op = verify_equivalence_bundle(
            _make_bundle(make_tex_shader("TexSampleLod", 0x00000000), make_tex_shader("TexFetch", 0x00000000)),
            timeout_ms=2000,
        )
        self.assertEqual(res_op.status, "SAT")

        # Identical LOD and op -> UNSAT
        res_eq = verify_equivalence_bundle(
            _make_bundle(make_tex_shader("TexSampleLod", 0x40000000), make_tex_shader("TexSampleLod", 0x40000000)),
            timeout_ms=2000,
        )
        self.assertEqual(res_eq.status, "UNSAT")

    def test_gap1_store_texel_3d_preserves_z(self) -> None:
        """Gap #1: StoreTexel with distinct z coordinates must return SAT."""
        def make_store3d(z_val: int) -> dict[str, Any]:
            return _single_block([
                {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                {"id": 2, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 1},
                {"id": 3, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 2},
                {"id": 4, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": z_val},
                {"id": 5, "op": "VecConstruct3", "type": "V3U32", "blk": 0, "args": [2, 3, 4]},
                {"id": 6, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 0x3F800000},
                {"id": 7, "op": "VecConstruct4", "type": "V4F32", "blk": 0, "args": [6, 6, 6, 6]},
                {"id": 8, "op": "StoreTexel", "type": "Token", "blk": 0, "args": [5, 7, 1], "aux_str": "img:0:0"},
            ])

        res = verify_equivalence_bundle(_make_bundle(make_store3d(0), make_store3d(1), stage="Compute"), timeout_ms=2000)
        self.assertEqual(res.status, "SAT")

    def test_gap2_stale_const_stores_invalidated_by_dynamic_store(self) -> None:
        """Gap #2: Dynamic store to ssbo:0:0+idx must invalidate const_stores[(ssbo:0:0, 0)]."""
        # Direct: store 1 at +0; store 2 at +idx; load +0 and store to frag_color.r
        direct = _single_block([
            {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
            {"id": 2, "op": "LoadSidebandU64", "type": "U64", "blk": 0, "args": [], "imm_u32": 0, "aux_str": "ssbo:0:0"},
            {"id": 3, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 1},
            {"id": 4, "op": "PsbStore", "type": "Token", "blk": 0, "args": [2, 3, 1], "aux_str": "ssbo:0:0"},
            {"id": 5, "op": "LoadBuiltinInput", "type": "U32", "blk": 0, "args": [], "imm_u32": 29},
            {"id": 6, "op": "UConvert32To64", "type": "U64", "blk": 0, "args": [5]},
            {"id": 7, "op": "IAdd64", "type": "U64", "blk": 0, "args": [2, 6]},
            {"id": 8, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 2},
            {"id": 9, "op": "PsbStore", "type": "Token", "blk": 0, "args": [7, 8, 4], "aux_str": "ssbo:0:0"},
            {"id": 10, "op": "PsbLoad", "type": "TupleValToken", "blk": 0, "args": [2, 9], "aux_str": "ssbo:0:0"},
            {"id": 11, "op": "TupleExtract", "type": "U32", "blk": 0, "args": [10], "imm_u32": 0},
            {"id": 12, "op": "VecConstruct4", "type": "V4F32", "blk": 0, "args": [11, 3, 3, 3]},
            {"id": 13, "op": "StoreFragColor", "type": "Token", "blk": 0, "args": [12, 9], "imm_u32": 0},
        ])
        # Buggy shader that ignores the dynamic overwrite and returns constant 1
        buggy = _single_block([
            {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
            {"id": 2, "op": "LoadSidebandU64", "type": "U64", "blk": 0, "args": [], "imm_u32": 0, "aux_str": "ssbo:0:0"},
            {"id": 3, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 1},
            {"id": 4, "op": "PsbStore", "type": "Token", "blk": 0, "args": [2, 3, 1], "aux_str": "ssbo:0:0"},
            {"id": 5, "op": "LoadBuiltinInput", "type": "U32", "blk": 0, "args": [], "imm_u32": 29},
            {"id": 6, "op": "UConvert32To64", "type": "U64", "blk": 0, "args": [5]},
            {"id": 7, "op": "IAdd64", "type": "U64", "blk": 0, "args": [2, 6]},
            {"id": 8, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 2},
            {"id": 9, "op": "PsbStore", "type": "Token", "blk": 0, "args": [7, 8, 4], "aux_str": "ssbo:0:0"},
            {"id": 10, "op": "VecConstruct4", "type": "V4F32", "blk": 0, "args": [3, 3, 3, 3]},
            {"id": 11, "op": "StoreFragColor", "type": "Token", "blk": 0, "args": [10, 9], "imm_u32": 0},
        ])
        res = verify_equivalence_bundle(
            _make_bundle(direct, buggy, bindings=[{"key": "ssbo:0:0", "kind": "SSBO"}]),
            timeout_ms=2000,
        )
        self.assertEqual(res.status, "SAT")
        self.assertTrue(any(m.output_name == "frag_color[0].r" for m in res.mismatches))

    def test_gap2_sym_buf_stores_preserves_multiple_offsets(self) -> None:
        """Gap #2: Two stores to distinct offsets under the same sym_tag must both be checked."""
        direct = _single_block([
            {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
            {"id": 2, "op": "LoadSidebandU64", "type": "U64", "blk": 0, "args": [], "imm_u32": 0, "aux_str": "ssbo:0:0"},
            {"id": 3, "op": "ConstU64", "type": "U64", "blk": 0, "args": [], "imm_u64": 4},
            {"id": 4, "op": "IAdd64", "type": "U64", "blk": 0, "args": [2, 3]},
            {"id": 5, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 11},
            {"id": 6, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 22},
            {"id": 7, "op": "PsbStore", "type": "Token", "blk": 0, "args": [2, 5, 1], "aux_str": "sym:ssbo:0:0[0]"},
            {"id": 8, "op": "PsbStore", "type": "Token", "blk": 0, "args": [4, 6, 7], "aux_str": "sym:ssbo:0:0[0]"},
        ])
        # Missing the first store (only writes 22 at +4)
        missing_first = _single_block([
            {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
            {"id": 2, "op": "LoadSidebandU64", "type": "U64", "blk": 0, "args": [], "imm_u32": 0, "aux_str": "ssbo:0:0"},
            {"id": 3, "op": "ConstU64", "type": "U64", "blk": 0, "args": [], "imm_u64": 4},
            {"id": 4, "op": "IAdd64", "type": "U64", "blk": 0, "args": [2, 3]},
            {"id": 5, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 22},
            {"id": 6, "op": "PsbStore", "type": "Token", "blk": 0, "args": [4, 5, 1], "aux_str": "sym:ssbo:0:0[0]"},
        ])
        res = verify_equivalence_bundle(
            _make_bundle(direct, missing_first, stage="Compute", bindings=[{"key": "ssbo:0:0", "kind": "SSBO"}]),
            timeout_ms=2000,
        )
        self.assertEqual(res.status, "SAT")

    def test_gap3_loop_truncation_guard_distinguishes_bmc_from_full_equiv(self) -> None:
        """Gap #3: Unbounded backedge reports BOUNDED_UNSAT + loop_truncated=True; static exit reports UNSAT."""
        def make_loop(cond_is_sym: bool) -> dict[str, Any]:
            nodes = [
                {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                {"id": 2, "op": "LoadBuiltinInput", "type": "U32", "blk": 0, "args": [], "imm_u32": 29},
                {"id": 3, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 0},
                {"id": 4, "op": "ICmpUGt", "type": "Bool", "blk": 1, "args": [2, 3]}
                if cond_is_sym
                else {"id": 4, "op": "ConstBool", "type": "Bool", "blk": 1, "args": [], "imm_u32": 0},
                {"id": 5, "op": "ConstF32", "type": "F32", "blk": 2, "args": [], "imm_u32": 0x3F800000},
                {"id": 6, "op": "VecConstruct4", "type": "V4F32", "blk": 2, "args": [5, 5, 5, 5]},
                {"id": 7, "op": "StoreFragColor", "type": "Token", "blk": 2, "args": [6, 1], "imm_u32": 0},
            ]
            blocks = [
                {
                    "id": 0, "preds": [], "phis": [], "nodes": [1, 2, 3],
                    "branch_target": 1, "fallthrough": -1, "is_cond": False, "is_uncond": True,
                    "is_end": False, "is_discard": False, "is_loop_header": False,
                    "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
                },
                {
                    "id": 1, "preds": [0, 1], "phis": [], "nodes": [4],
                    "branch_target": 1, "fallthrough": 2, "is_cond": True, "is_uncond": False,
                    "is_end": False, "is_discard": False, "is_loop_header": True,
                    "loop_backedge": 1, "loop_merge": 2, "sel_merge": -1, "branch_cond": 4,
                },
                {
                    "id": 2, "preds": [1], "phis": [], "nodes": [5, 6, 7],
                    "branch_target": -1, "fallthrough": -1, "is_cond": False, "is_uncond": False,
                    "is_end": True, "is_discard": False, "is_loop_header": False,
                    "loop_backedge": -1, "loop_merge": -1, "sel_merge": -1, "branch_cond": 0,
                },
            ]
            return {"blocks": blocks, "nodes": nodes}

        sym_loop = make_loop(cond_is_sym=True)
        res_bmc = verify_equivalence_bundle(_make_bundle(sym_loop, sym_loop), max_unroll=1, timeout_ms=2000)
        self.assertEqual(res_bmc.status, "BOUNDED_UNSAT")
        self.assertTrue(res_bmc.loop_truncated)
        self.assertTrue(any("BMC_EQUIV(k=1)" in n for n in res_bmc.notes))

        static_exit = make_loop(cond_is_sym=False)
        res_full = verify_equivalence_bundle(_make_bundle(static_exit, static_exit), max_unroll=1, timeout_ms=2000)
        self.assertEqual(res_full.status, "UNSAT")
        self.assertFalse(res_full.loop_truncated)

    def test_gap4_control_barrier_havocs_shared_memory_and_checks_pre_barrier_stores(self) -> None:
        """Gap #4: Local store must not forward past ControlBarrier, and dropped pre-barrier store is caught."""
        # store 1 -> ControlBarrier -> load -> store to ssbo:0:0+4
        with_barrier = _single_block([
            {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
            {"id": 2, "op": "LoadSidebandU64", "type": "U64", "blk": 0, "args": [], "imm_u32": 0, "aux_str": "ssbo:0:0"},
            {"id": 3, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 1},
            {"id": 4, "op": "PsbStore", "type": "Token", "blk": 0, "args": [2, 3, 1], "aux_str": "ssbo:0:0"},
            {"id": 5, "op": "ControlBarrier", "type": "Token", "blk": 0, "args": [4]},
            {"id": 6, "op": "PsbLoad", "type": "TupleValToken", "blk": 0, "args": [2, 5], "aux_str": "ssbo:0:0"},
            {"id": 7, "op": "TupleExtract", "type": "U32", "blk": 0, "args": [6], "imm_u32": 0},
            {"id": 8, "op": "ConstU64", "type": "U64", "blk": 0, "args": [], "imm_u64": 4},
            {"id": 9, "op": "IAdd64", "type": "U64", "blk": 0, "args": [2, 8]},
            {"id": 10, "op": "PsbStore", "type": "Token", "blk": 0, "args": [9, 7, 5], "aux_str": "ssbo:0:0"},
        ])
        # Same shader vs itself -> UNSAT (deterministic symmetric barrier havoc)
        res_eq = verify_equivalence_bundle(
            _make_bundle(with_barrier, with_barrier, stage="Compute", bindings=[{"key": "ssbo:0:0", "kind": "SSBO"}]),
            timeout_ms=2000,
        )
        self.assertEqual(res_eq.status, "UNSAT")

        # Incorrectly forwards local constant 1 across ControlBarrier -> SAT!
        forwarded = _single_block([
            {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
            {"id": 2, "op": "LoadSidebandU64", "type": "U64", "blk": 0, "args": [], "imm_u32": 0, "aux_str": "ssbo:0:0"},
            {"id": 3, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 1},
            {"id": 4, "op": "PsbStore", "type": "Token", "blk": 0, "args": [2, 3, 1], "aux_str": "ssbo:0:0"},
            {"id": 5, "op": "ControlBarrier", "type": "Token", "blk": 0, "args": [4]},
            {"id": 8, "op": "ConstU64", "type": "U64", "blk": 0, "args": [], "imm_u64": 4},
            {"id": 9, "op": "IAdd64", "type": "U64", "blk": 0, "args": [2, 8]},
            {"id": 10, "op": "PsbStore", "type": "Token", "blk": 0, "args": [9, 3, 5], "aux_str": "ssbo:0:0"},
        ])
        res_fwd = verify_equivalence_bundle(
            _make_bundle(with_barrier, forwarded, stage="Compute", bindings=[{"key": "ssbo:0:0", "kind": "SSBO"}]),
            timeout_ms=2000,
        )
        self.assertEqual(res_fwd.status, "SAT")

    def test_gap4_atomic_havocs_old_val(self) -> None:
        """Gap #4: AtomicRmw must havoc old_val instead of reading local pre-atomic store."""
        direct = _single_block([
            {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
            {"id": 2, "op": "LoadSidebandU64", "type": "U64", "blk": 0, "args": [], "imm_u32": 0, "aux_str": "ssbo:0:0"},
            {"id": 3, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 42},
            {"id": 4, "op": "PsbStore", "type": "Token", "blk": 0, "args": [2, 3, 1], "aux_str": "ssbo:0:0"},
            {"id": 5, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 1},
            {"id": 6, "op": "AtomicRmw", "type": "TupleValToken", "blk": 0, "args": [2, 5, 4], "aux_str": "aadd"},
            {"id": 7, "op": "TupleExtract", "type": "U32", "blk": 0, "args": [6], "imm_u32": 0},
            {"id": 8, "op": "ConstU64", "type": "U64", "blk": 0, "args": [], "imm_u64": 4},
            {"id": 9, "op": "IAdd64", "type": "U64", "blk": 0, "args": [2, 8]},
            {"id": 10, "op": "PsbStore", "type": "Token", "blk": 0, "args": [9, 7, 4], "aux_str": "ssbo:0:0"},
        ])
        # Assumes old_val == 42 locally -> SAT!
        local_assumed = _single_block([
            {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
            {"id": 2, "op": "LoadSidebandU64", "type": "U64", "blk": 0, "args": [], "imm_u32": 0, "aux_str": "ssbo:0:0"},
            {"id": 3, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 42},
            {"id": 4, "op": "PsbStore", "type": "Token", "blk": 0, "args": [2, 3, 1], "aux_str": "ssbo:0:0"},
            {"id": 5, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 1},
            {"id": 6, "op": "AtomicRmw", "type": "TupleValToken", "blk": 0, "args": [2, 5, 4], "aux_str": "aadd"},
            {"id": 8, "op": "ConstU64", "type": "U64", "blk": 0, "args": [], "imm_u64": 4},
            {"id": 9, "op": "IAdd64", "type": "U64", "blk": 0, "args": [2, 8]},
            {"id": 10, "op": "PsbStore", "type": "Token", "blk": 0, "args": [9, 3, 4], "aux_str": "ssbo:0:0"},
        ])
        res = verify_equivalence_bundle(
            _make_bundle(direct, local_assumed, stage="Compute", bindings=[{"key": "ssbo:0:0", "kind": "SSBO"}]),
            timeout_ms=2000,
        )
        self.assertEqual(res.status, "SAT")

    def test_gap5_byte_memory_overlapping_unaligned_and_ssbo_aliasing(self) -> None:
        """Gap #5: Byte-addressed memory detects off=0/off=2 overlap; allow_ssbo_aliasing detects binding overlap."""
        # Store 0x11223344 at +0, load at +2
        unaligned_read = _single_block([
            {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
            {"id": 2, "op": "LoadSidebandU64", "type": "U64", "blk": 0, "args": [], "imm_u32": 0, "aux_str": "ssbo:0:0"},
            {"id": 3, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 0x11223344},
            {"id": 4, "op": "PsbStore", "type": "Token", "blk": 0, "args": [2, 3, 1], "aux_str": "ssbo:0:0"},
            {"id": 5, "op": "ConstU64", "type": "U64", "blk": 0, "args": [], "imm_u64": 2},
            {"id": 6, "op": "IAdd64", "type": "U64", "blk": 0, "args": [2, 5]},
            {"id": 7, "op": "PsbLoad", "type": "TupleValToken", "blk": 0, "args": [6, 4], "aux_str": "ssbo:0:0"},
            {"id": 8, "op": "TupleExtract", "type": "U32", "blk": 0, "args": [7], "imm_u32": 0},
            {"id": 9, "op": "VecConstruct4", "type": "V4F32", "blk": 0, "args": [8, 3, 3, 3]},
            {"id": 10, "op": "StoreFragColor", "type": "Token", "blk": 0, "args": [9, 4], "imm_u32": 0},
        ])
        # Loads +2 BEFORE storing +0 (assumes +0 and +2 don't overlap)
        reordered = _single_block([
            {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
            {"id": 2, "op": "LoadSidebandU64", "type": "U64", "blk": 0, "args": [], "imm_u32": 0, "aux_str": "ssbo:0:0"},
            {"id": 3, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 0x11223344},
            {"id": 5, "op": "ConstU64", "type": "U64", "blk": 0, "args": [], "imm_u64": 2},
            {"id": 6, "op": "IAdd64", "type": "U64", "blk": 0, "args": [2, 5]},
            {"id": 7, "op": "PsbLoad", "type": "TupleValToken", "blk": 0, "args": [6, 1], "aux_str": "ssbo:0:0"},
            {"id": 8, "op": "TupleExtract", "type": "U32", "blk": 0, "args": [7], "imm_u32": 0},
            {"id": 4, "op": "PsbStore", "type": "Token", "blk": 0, "args": [2, 3, 1], "aux_str": "ssbo:0:0"},
            {"id": 9, "op": "VecConstruct4", "type": "V4F32", "blk": 0, "args": [8, 3, 3, 3]},
            {"id": 10, "op": "StoreFragColor", "type": "Token", "blk": 0, "args": [9, 4], "imm_u32": 0},
        ])
        bundle_unaligned = _make_bundle(unaligned_read, reordered, bindings=[{"key": "ssbo:0:0", "kind": "SSBO"}])
        res_byte = verify_equivalence_bundle(bundle_unaligned, memory_model="byte", timeout_ms=2000)
        self.assertEqual(res_byte.status, "SAT")

        res_word = verify_equivalence_bundle(bundle_unaligned, memory_model="word", timeout_ms=2000)
        self.assertTrue(any("unaligned" in a for a in res_word.soundness_assumptions))

        # Cross-binding SSBO aliasing: store to ssbo:0:0, load from ssbo:0:1 vs reordered
        alias_a = _single_block([
            {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
            {"id": 2, "op": "LoadSidebandU64", "type": "U64", "blk": 0, "args": [], "imm_u32": 0, "aux_str": "ssbo:0:0"},
            {"id": 3, "op": "LoadSidebandU64", "type": "U64", "blk": 0, "args": [], "imm_u32": 0, "aux_str": "ssbo:0:1"},
            {"id": 4, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 99},
            {"id": 5, "op": "PsbStore", "type": "Token", "blk": 0, "args": [2, 4, 1], "aux_str": "ssbo:0:0"},
            {"id": 6, "op": "PsbLoad", "type": "TupleValToken", "blk": 0, "args": [3, 5], "aux_str": "ssbo:0:1"},
            {"id": 7, "op": "TupleExtract", "type": "U32", "blk": 0, "args": [6], "imm_u32": 0},
            {"id": 8, "op": "VecConstruct4", "type": "V4F32", "blk": 0, "args": [7, 4, 4, 4]},
            {"id": 9, "op": "StoreFragColor", "type": "Token", "blk": 0, "args": [8, 5], "imm_u32": 0},
        ])
        alias_b = _single_block([
            {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
            {"id": 2, "op": "LoadSidebandU64", "type": "U64", "blk": 0, "args": [], "imm_u32": 0, "aux_str": "ssbo:0:0"},
            {"id": 3, "op": "LoadSidebandU64", "type": "U64", "blk": 0, "args": [], "imm_u32": 0, "aux_str": "ssbo:0:1"},
            {"id": 4, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 99},
            {"id": 6, "op": "PsbLoad", "type": "TupleValToken", "blk": 0, "args": [3, 1], "aux_str": "ssbo:0:1"},
            {"id": 7, "op": "TupleExtract", "type": "U32", "blk": 0, "args": [6], "imm_u32": 0},
            {"id": 5, "op": "PsbStore", "type": "Token", "blk": 0, "args": [2, 4, 1], "aux_str": "ssbo:0:0"},
            {"id": 8, "op": "VecConstruct4", "type": "V4F32", "blk": 0, "args": [7, 4, 4, 4]},
            {"id": 9, "op": "StoreFragColor", "type": "Token", "blk": 0, "args": [8, 5], "imm_u32": 0},
        ])
        b_alias = _make_bundle(
            alias_a,
            alias_b,
            bindings=[{"key": "ssbo:0:0", "kind": "SSBO"}, {"key": "ssbo:0:1", "kind": "SSBO"}],
        )
        res_alias = verify_equivalence_bundle(b_alias, allow_ssbo_aliasing=True, timeout_ms=2000)
        self.assertEqual(res_alias.status, "SAT")
        res_disjoint = verify_equivalence_bundle(b_alias, allow_ssbo_aliasing=False, timeout_ms=2000)
        self.assertIn("disjoint_ssbo_bindings", res_disjoint.soundness_assumptions)

    def test_gap6_convert_ftos_ftou_trunc_toward_zero_and_strict_fp_edges(self) -> None:
        """Gap #6: udomain_exact ConvertFToS(-1.5) == -1 (not -2) with saturation; strict_fp_edges distinguishes NaN ops."""
        dom_ex = SymbolicZ3UDomain(exact_bitcast=True)
        rx = dsl.Real("x")
        conv_s = dom_ex.lower_z3_bv32(dom_ex.convert("ConvertFToS", dom_ex.lift_z3_real_f32(rx)))
        s = z3.Solver()
        s.add(dsl.lower(rx) == z3.Q(-3, 2))
        s.add(dsl.lower(conv_s) != z3.BitVecVal(0xFFFFFFFF, 32))
        self.assertEqual(s.check(), z3.unsat, "ConvertFToS(-1.5) must truncate toward zero to -1 (0xFFFFFFFF)")

        # Saturation clamp on signed overflow (+3e9 -> 0x7FFFFFFF)
        s2 = z3.Solver()
        s2.add(dsl.lower(rx) == z3.RealVal(3000000000))
        s2.add(dsl.lower(conv_s) != z3.BitVecVal(0x7FFFFFFF, 32))
        self.assertEqual(s2.check(), z3.unsat)

        # Unsigned clamp on negative (-5.0 -> 0)
        conv_u = dom_ex.lower_z3_bv32(dom_ex.convert("ConvertFToU", dom_ex.lift_z3_real_f32(rx)))
        s3 = z3.Solver()
        s3.add(dsl.lower(rx) == z3.RealVal(-5))
        s3.add(dsl.lower(conv_u) != z3.BitVecVal(0, 32))
        self.assertEqual(s3.check(), z3.unsat)

        # strict_fp_edges=True distinguishes FCmpOrdLt vs FCmpUnordLt and FMin vs NMin
        dom_st = SymbolicZ3UDomain(strict_fp_edges=True)
        a = dom_st.fresh_f32("a")
        b = dom_st.fresh_f32("b")
        c_ord = dom_st.lower_z3_bool(dom_st.fp_cmp("FCmpOrdLt", a, b))
        c_unord = dom_st.lower_z3_bool(dom_st.fp_cmp("FCmpUnordLt", a, b))
        s4 = z3.Solver()
        s4.add(dsl.lower(c_ord) != dsl.lower(c_unord))
        self.assertEqual(s4.check(), z3.sat)

        m_fmin = dom_st.to_z3_real_f32(dom_st.fp_binary("FMin", a, b))
        m_nmin = dom_st.to_z3_real_f32(dom_st.fp_binary("NMin", a, b))
        s5 = z3.Solver()
        s5.add(dsl.lower(m_fmin) != dsl.lower(m_nmin))
        self.assertEqual(s5.check(), z3.sat)

    def test_gap7_sign_bit_axioms_and_div_by_zero_guard(self) -> None:
        """Gap #7a & #7c: Sign-bit XOR/AND identities discharge via BvAlgebraicPass & EmatchingUnifyPass; UDiv guards b != 0."""
        u_f2b = dsl.Function("u_f32_to_bv32", dsl.RealSort(), dsl.BitVecSort(32))
        r = dsl.Real("r")
        lhs_neg = u_f2b(r) ^ dsl.BitVecVal(0x80000000, 32)
        rhs_neg = u_f2b(-r)
        lhs_abs = u_f2b(r) & dsl.BitVecVal(0x7FFFFFFF, 32)
        rhs_abs = u_f2b(dsl.If(r < dsl.RealVal(0), -r, r))

        # 1. Local BvAlgebraicPass peephole
        bv_pass = BvAlgebraicPass()
        ctx = SimplificationContext()
        out_neg, rw_neg = bv_pass.rewrite_term(lhs_neg, ctx)
        self.assertGreater(rw_neg, 0)
        self.assertTrue(dsl.expr_eq(out_neg, rhs_neg))

        out_abs, rw_abs = bv_pass.rewrite_term(lhs_abs, ctx)
        self.assertGreater(rw_abs, 0)
        self.assertTrue(dsl.expr_eq(out_abs, rhs_abs))

        # 2. EmatchingUnifyPass (quantified triggered axioms)
        em_pass = EmatchingUnifyPass(include_bridge_axioms=True)
        q_neg = SimplifyQuery(lhs=lhs_neg, rhs=rhs_neg)
        q_neg_out, _ = em_pass.run(q_neg, ctx)
        self.assertTrue(q_neg_out.is_trivially_unsat())

        q_abs = SimplifyQuery(lhs=lhs_abs, rhs=rhs_abs)
        q_abs_out, _ = em_pass.run(q_abs, ctx)
        self.assertTrue(q_abs_out.is_trivially_unsat())

        # 3. Default pipeline + solver
        res = solve_simplified_miter(q_neg, build_default_pipeline(), timeout_ms=2000)
        self.assertEqual(res.status, "UNSAT")

        # 4. Gap #7c: symbolic UDiv records non-zero divisor guard
        div_ssa = _single_block([
            {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
            {"id": 2, "op": "LoadBuiltinInput", "type": "U32", "blk": 0, "args": [], "imm_u32": 29},
            {"id": 3, "op": "LoadBuiltinInput", "type": "U32", "blk": 0, "args": [], "imm_u32": 28},
            {"id": 4, "op": "UDiv", "type": "U32", "blk": 0, "args": [2, 3]},
            {"id": 5, "op": "VecConstruct4", "type": "V4F32", "blk": 0, "args": [4, 4, 4, 4]},
            {"id": 6, "op": "StoreFragColor", "type": "Token", "blk": 0, "args": [5, 1], "imm_u32": 0},
        ])
        res_div = verify_equivalence_bundle(_make_bundle(div_ssa, div_ssa), timeout_ms=2000)
        self.assertEqual(res_div.status, "UNSAT")
        self.assertIn("nonzero_integer_divisors", res_div.soundness_assumptions)


if __name__ == "__main__":
    unittest.main()

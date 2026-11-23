#!/usr/bin/env python3
"""
Unit tests for Checkpoint v31 additions:
  - Modular `tools/symbolic_ssa/domains/` (renames, default FP semantics,
    TupleDomain, PointwiseUnionDomain, MultiSetDomain)
  - `tools/symbolic_ssa/cfg_rewriter/` (CfgRewriter, pass_cleanup_phis, tiles)
  - Pointer `ISub64` and `BitAnd64` alignment masking in ConstantPropDomain
    and BufferProvenanceInterpreter
  - Hoisted `InstructionSsaZ3Interpreter` in `tools/verify/ssa_z3_eval.py`
  - `ConcreteUDomain` and `SymbolicZ3UDomain` (lattice transitions, lazy
    bitcast, Z3 real-arithmetic bit-blasting avoidance, FP bit-hacking
    fallback, pointer ops, and read-only uniform register caching)
"""

from __future__ import annotations

import math
import struct
import time
import unittest

import z3

from tools.symbolic_ssa.cfg_rewriter import (
    CfgRewriter,
    apply_cfg_tile_rules,
    make_chain_contract_rule,
    make_degenerate_branch_rule,
    make_empty_jump_thread_rule,
    make_pure_diamond_rule,
    pass_cleanup_phis,
)
from tools.symbolic_ssa.domains import (
    MULTISET_BOTTOM,
    MULTISET_TOP,
    UNION_BOTTOM,
    UNION_TOP,
    AbstractTopDomain,
    ConcreteBitvecDomain,
    ConcreteDomain,
    ConcreteUDomain,
    MultiSetDomain,
    MultiSetVal,
    PointwiseUnionDomain,
    SsaDomain,
    SymbolicDomain,
    SymbolicZ3BitvecDomain,
    SymbolicZ3UDomain,
    TupleDomain,
    TupleVal,
    UnionVal,
    UVal,
    UValKind,
)
from tools.symbolic_ssa.passes.pass_prune_const_branches import (
    ConstantPropDomain,
    ConstVal,
)
from tools.symbolic_ssa.passes.pass_readonly_tokens import BufferProvenanceInterpreter
from tools.verify.ssa_z3_eval import InstructionSsaZ3Interpreter


def _f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def _bits_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


class _MinimalFpDomain(ConcreteBitvecDomain):
    """Subclass that inherits SsaDomain's default fp_unary/fp_binary implementations."""

    def fp_unary(self, op: str, x: int) -> int:
        if op in ("Sqrt", "Exp2", "Log2"):
            return super().fp_unary(op, x)
        return SsaDomain.fp_unary(self, op, x)

    def fp_binary(self, op: str, a: int, b: int) -> int:
        if op in ("FAdd", "FMul", "FDiv"):
            return super().fp_binary(op, a, b)
        return SsaDomain.fp_binary(self, op, a, b)


class TestDomainsPackage(unittest.TestCase):
    def test_renames_and_backward_aliases(self):
        self.assertIs(ConcreteDomain, ConcreteBitvecDomain)
        self.assertIs(SymbolicDomain, SymbolicZ3BitvecDomain)

    def test_default_fp_semantics_in_base_domain(self):
        d = _MinimalFpDomain()
        self.assertAlmostEqual(_bits_f32(d.fp_unary("FAbs", _f32_bits(-3.25))), 3.25, places=5)
        self.assertAlmostEqual(_bits_f32(d.fp_unary("FNeg", _f32_bits(3.25))), -3.25, places=5)
        self.assertAlmostEqual(_bits_f32(d.fp_unary("Trunc", _f32_bits(-2.7))), -2.0, places=5)
        self.assertAlmostEqual(_bits_f32(d.fp_unary("Floor", _f32_bits(-2.7))), -3.0, places=5)
        self.assertAlmostEqual(_bits_f32(d.fp_unary("Ceil", _f32_bits(2.3))), 3.0, places=5)
        self.assertAlmostEqual(_bits_f32(d.fp_unary("InverseSqrt", _f32_bits(4.0))), 0.5, places=5)
        self.assertAlmostEqual(_bits_f32(d.fp_unary("Exp", _f32_bits(1.0))), math.e, places=4)
        self.assertAlmostEqual(_bits_f32(d.fp_unary("Log", _f32_bits(math.e))), 1.0, places=4)
        self.assertAlmostEqual(_bits_f32(d.fp_binary("FSub", _f32_bits(5.5), _f32_bits(2.0))), 3.5, places=5)
        self.assertAlmostEqual(_bits_f32(d.fp_binary("FMin", _f32_bits(5.5), _f32_bits(2.0))), 2.0, places=5)
        self.assertAlmostEqual(_bits_f32(d.fp_binary("FMax", _f32_bits(5.5), _f32_bits(2.0))), 5.5, places=5)
        self.assertAlmostEqual(_bits_f32(d.fp_binary("Pow", _f32_bits(2.0), _f32_bits(3.0))), 8.0, places=4)
        self.assertAlmostEqual(_bits_f32(d.fp_fma(_f32_bits(2.0), _f32_bits(3.0), _f32_bits(4.0))), 10.0, places=5)

    def test_tuple_domain(self):
        td = TupleDomain(ConcreteBitvecDomain(), AbstractTopDomain())
        a = td.const_u32(10)
        b = td.const_u32(32)
        s = td.int32_binary("IAdd", a, b)
        self.assertIsInstance(s, TupleVal)
        self.assertEqual(s[0], 42)
        self.assertIsNone(s[1])
        self.assertEqual(td.const_int_val(s), 42)
        self.assertEqual(td.join(a, a), a)

    def test_pointwise_union_domain(self):
        ud = PointwiseUnionDomain(
            {"concrete": ConcreteBitvecDomain(), "top": AbstractTopDomain()},
            default_tag="concrete",
        )
        a = ud.const_u32(10)
        b = ud.const_u32(25)
        s = ud.int32_binary("IAdd", a, b)
        self.assertIsInstance(s, UnionVal)
        self.assertEqual(s.tag, "concrete")
        self.assertEqual(s.value, 35)
        # Disjoint tags join to UNION_TOP
        other = UnionVal("top", None)
        self.assertEqual(ud.join(a, other), UNION_TOP)
        # Bottom absorption
        self.assertEqual(ud.int32_binary("IAdd", UNION_BOTTOM, a), UNION_BOTTOM)
        self.assertTrue(ud.leq(UNION_BOTTOM, a))
        self.assertTrue(ud.leq(a, UNION_TOP))

    def test_multiset_domain(self):
        md = MultiSetDomain(ConcreteBitvecDomain(), max_k=4, track_counts=True)
        s1 = md.join(md.const_u32(1), md.const_u32(2))
        s2 = md.join(md.const_u32(10), md.const_u32(20))
        res = md.int32_binary("IAdd", s1, s2)
        self.assertIsInstance(res, MultiSetVal)
        self.assertEqual(set(res.elements()), {11, 12, 21, 22})
        # Widening to MULTISET_TOP beyond max_k=4
        widened = md.join(res, md.const_u32(99))
        self.assertEqual(widened, MULTISET_TOP)
        self.assertTrue(md.leq(MULTISET_BOTTOM, s1))
        self.assertTrue(md.leq(s1, MULTISET_TOP))


class TestCfgRewriterAndTiles(unittest.TestCase):
    def _make_maps(self, blocks, nodes):
        return ({b["id"]: b for b in blocks}, {n["id"]: n for n in nodes})

    def test_cleanup_phis_dead_trivial_and_cascading(self):
        blocks, nodes = self._make_maps(
            blocks=[
                {"id": 0, "preds": [], "nodes": [1, 2], "phis": [], "is_uncond": True, "is_cond": False, "is_end": False, "branch_target": 1, "fallthrough": -1, "branch_cond": -1},
                {"id": 1, "preds": [0, 1], "nodes": [6], "phis": [3, 4, 5], "is_uncond": False, "is_cond": True, "is_end": False, "branch_target": 1, "fallthrough": 2, "branch_cond": 2},
                {"id": 2, "preds": [1], "nodes": [7], "phis": [], "is_uncond": False, "is_cond": False, "is_end": True, "branch_target": -1, "fallthrough": -1, "branch_cond": -1},
            ],
            nodes=[
                {"id": 1, "op": "ConstU32", "type": "U32", "imm_u32": 7, "args": [], "blk": 0},
                {"id": 2, "op": "ConstBool", "type": "Bool", "imm_u32": 0, "args": [], "blk": 0},
                {"id": 3, "op": "Phi", "type": "U32", "args": [], "phi": [[0, 1], [1, 3]], "blk": 1},
                {"id": 4, "op": "Phi", "type": "U32", "args": [], "phi": [[0, 1], [1, 6]], "blk": 1},
                {"id": 5, "op": "Phi", "type": "U32", "args": [], "phi": [[0, 3], [1, 5]], "blk": 1},
                {"id": 6, "op": "IAdd", "type": "U32", "args": [5, 1], "blk": 1},
                {"id": 7, "op": "StoreFragColor", "type": "Void", "args": [6], "blk": 2},
            ],
        )
        changed = pass_cleanup_phis(blocks, nodes)
        self.assertTrue(changed)
        self.assertEqual(blocks[1]["phis"], [])
        self.assertEqual(nodes[6]["args"], [1, 1])

    def test_degenerate_branch_and_empty_jump_threading(self):
        blocks, nodes = self._make_maps(
            blocks=[
                {"id": 0, "preds": [], "nodes": [1], "phis": [], "is_uncond": False, "is_cond": True, "is_end": False, "branch_target": 1, "fallthrough": 1, "branch_cond": 1},
                {"id": 1, "preds": [0], "nodes": [], "phis": [], "is_uncond": True, "is_cond": False, "is_end": False, "branch_target": 2, "fallthrough": -1, "branch_cond": -1},
                {"id": 2, "preds": [1], "nodes": [2], "phis": [], "is_uncond": False, "is_cond": False, "is_end": True, "branch_target": -1, "fallthrough": -1, "branch_cond": -1},
            ],
            nodes=[
                {"id": 1, "op": "ConstBool", "type": "Bool", "imm_u32": 1, "args": [], "blk": 0},
                {"id": 2, "op": "ConstU32", "type": "U32", "imm_u32": 9, "args": [], "blk": 2},
            ],
        )
        from tools.symbolic_ssa.passes import Stratum1Stats
        stats = Stratum1Stats()
        changed = apply_cfg_tile_rules(
            blocks,
            nodes,
            stats,
            [
                make_degenerate_branch_rule(),
                make_empty_jump_thread_rule(),
                make_chain_contract_rule(),
            ],
        )
        self.assertTrue(changed)
        self.assertEqual(len(blocks), 1)
        self.assertTrue(blocks[0]["is_end"])

    def test_pure_diamond_if_conversion(self):
        blocks, nodes = self._make_maps(
            blocks=[
                {"id": 0, "preds": [], "nodes": [1, 2], "phis": [], "is_uncond": False, "is_cond": True, "is_end": False, "branch_target": 1, "fallthrough": 2, "branch_cond": 1},
                {"id": 1, "preds": [0], "nodes": [3], "phis": [], "is_uncond": True, "is_cond": False, "is_end": False, "branch_target": 3, "fallthrough": -1, "branch_cond": -1},
                {"id": 2, "preds": [0], "nodes": [4], "phis": [], "is_uncond": True, "is_cond": False, "is_end": False, "branch_target": 3, "fallthrough": -1, "branch_cond": -1},
                {"id": 3, "preds": [1, 2], "nodes": [6], "phis": [5], "is_uncond": False, "is_cond": False, "is_end": True, "branch_target": -1, "fallthrough": -1, "branch_cond": -1},
            ],
            nodes=[
                {"id": 1, "op": "ConstBool", "type": "Bool", "imm_u32": 1, "args": [], "blk": 0},
                {"id": 2, "op": "ConstU32", "type": "U32", "imm_u32": 10, "args": [], "blk": 0},
                {"id": 3, "op": "IAdd", "type": "U32", "args": [2, 2], "blk": 1},
                {"id": 4, "op": "ISub", "type": "U32", "args": [2, 2], "blk": 2},
                {"id": 5, "op": "Phi", "type": "U32", "args": [], "phi": [[1, 3], [2, 4]], "blk": 3},
                {"id": 6, "op": "StoreFragColor", "type": "Void", "args": [5], "blk": 3},
            ],
        )
        from tools.symbolic_ssa.passes import Stratum1Stats
        stats = Stratum1Stats()
        changed = apply_cfg_tile_rules(blocks, nodes, stats, [make_pure_diamond_rule(max_arm_nodes=4), make_chain_contract_rule()])
        self.assertTrue(changed)
        self.assertEqual(len(blocks), 1)
        ops = [nodes[nid]["op"] for nid in blocks[0]["nodes"]]
        self.assertIn("Select", ops)
        self.assertNotIn("Phi", ops)


class TestPointerArithmeticAndDryInterpreter(unittest.TestCase):
    def test_constant_prop_domain_ptr_sub_and_align_mask(self):
        d = ConstantPropDomain()
        p = ConstVal("uniform_ptr", ("ubo:0:0", 64))
        p2 = d.int64_binary("ISub64", p, ConstVal("int", 16))
        self.assertEqual(p2, ConstVal("uniform_ptr", ("ubo:0:0", 48)))
        delta = d.int64_binary("ISub64", p, p2)
        self.assertEqual(delta, ConstVal("int", 16))
        p_unaligned = ConstVal("uniform_ptr", ("ubo:0:0", 23))
        p_aligned = d.int64_binary("BitAnd64", p_unaligned, ConstVal("int", 0xFFFFFFFFFFFFFFF0))
        self.assertEqual(p_aligned, ConstVal("uniform_ptr", ("ubo:0:0", 16)))

    def test_buffer_provenance_ptr_sub_and_align_mask(self):
        interp = BufferProvenanceInterpreter({}, {})
        self.assertEqual(interp.int64_binary("IAdd64", "ssbo:0:0", None), "ssbo:0:0")
        self.assertEqual(interp.int64_binary("ISub64", "ssbo:0:0", None), "ssbo:0:0")
        # Pointer - pointer yields integer byte delta, NOT a pointer
        self.assertIsNone(interp.int64_binary("ISub64", "ssbo:0:0", "ssbo:0:0"))
        # Pointer & alignment mask preserves buffer provenance
        self.assertEqual(interp.int64_binary("BitAnd64", "ssbo:0:0", None), "ssbo:0:0")
        self.assertEqual(interp.int64_binary("BitAnd64", None, "ssbo:0:0"), "ssbo:0:0")

    def test_hoisted_instruction_ssa_z3_interpreter_inherits_symbolic(self):
        from tools.symbolic_ssa.symbolic_interpreter import SymbolicSsaInterpreter
        self.assertTrue(issubclass(InstructionSsaZ3Interpreter, SymbolicSsaInterpreter))


class TestUDomain(unittest.TestCase):
    def test_concrete_udomain_lattice_lazy_bitcast_and_bithack(self):
        d = ConcreteUDomain()
        u = UVal.unknown()
        self.assertTrue(u.is_unknown)
        self.assertTrue(d.leq(u, d.const_u32(42)))

        slot = d.assign(u, d.const_f32(3.5))
        self.assertEqual(slot.kind, UValKind.F32X2)
        slot2 = d.assign(slot, d.const_u32(7))
        self.assertEqual(slot2.kind, UValKind.BVEC64)

        # Lazy bitcast preserves F32X2 across Bitcast(F32->U32)->Bitcast(U32->F32)
        f = d.const_f32(3.5)
        f_as_u = d.cast_value("Bitcast", f, "U32")
        self.assertEqual(f_as_u.kind, UValKind.F32X2)
        f_back = d.cast_value("Bitcast", f_as_u, "F32")
        self.assertEqual(f_back.kind, UValKind.F32X2)

        # FP bit-hacking: flip sign bit via BitXor -> promotes to BVEC64 -> exact -3.5
        neg_bits = d.int32_binary("BitXor", f_as_u, d.const_u32(0x80000000))
        self.assertEqual(neg_bits.kind, UValKind.BVEC64)
        neg_f = d.fp_binary("FAdd", neg_bits, d.const_f32(0.0))
        self.assertEqual(neg_f.kind, UValKind.F32X2)
        self.assertAlmostEqual(neg_f.data[0], -3.5, places=6)

        # Distinct F16X4 and F64
        h1 = d.const_f16x4(1.0, 2.0, 3.0, 4.0)
        h2 = d.const_f16x4(0.5, 1.5, 2.5, 3.5)
        hs = d.f16x4_binary("FAdd", h1, h2)
        self.assertEqual(hs.kind, UValKind.F16X4)
        self.assertEqual(hs.data, (1.5, 3.5, 5.5, 7.5))

        d1 = d.f64_binary("FMul", d.const_f64(2.5), d.const_f64(4.0))
        self.assertEqual(d1.kind, UValKind.F64)
        self.assertAlmostEqual(d1.data, 10.0, places=9)

    def test_symbolic_udomain_avoids_bitblasting_on_well_formed_fp(self):
        d = SymbolicZ3UDomain(exact_bitcast=False)
        a = d.fresh_f32("a")
        b = d.fresh_f32("b")
        c = d.fresh_f32("c")

        # Simulate Valhall SSA's Bitcast(F32->U32) / Bitcast(U32->F32) wrapping
        prod = d.cast_value("Bitcast", d.fp_binary("FMul", a, b), "U32")
        sum1 = d.fp_binary("FAdd", d.cast_value("Bitcast", prod, "F32"), c)
        fma1 = d.fp_fma(a, b, c)

        self.assertEqual(d.bitcast_materializations, 0)
        self.assertEqual(sum1.kind, UValKind.F32X2)
        self.assertEqual(fma1.kind, UValKind.F32X2)

        t0 = time.monotonic()
        s = z3.Solver()
        s.add(sum1.data[0] != fma1.data[0])
        res = s.check()
        dt = time.monotonic() - t0
        self.assertEqual(res, z3.unsat)
        self.assertLess(dt, 1.0)

    def test_symbolic_udomain_bithack_materializes_on_demand(self):
        d = SymbolicZ3UDomain(exact_bitcast=False)
        x = d.fresh_f32("x")
        x_u32 = d.cast_value("Bitcast", x, "U32")
        self.assertEqual(d.bitcast_materializations, 0)
        flipped = d.int32_binary("BitXor", x_u32, d.const_u32(0x80000000))
        self.assertEqual(flipped.kind, UValKind.BVEC64)
        self.assertGreater(d.bitcast_materializations, 0)

    def test_udomain_pointers_and_readonly_registers(self):
        d = SymbolicZ3UDomain()
        p_ubo = d.make_ptr("ubo:0:0", 0)
        self.assertTrue(p_ubo.is_readonly_ptr)

        p_off = d.int64_binary("IAdd64", p_ubo, d.const_u64(36))
        self.assertEqual(p_off.kind, UValKind.PTR)
        self.assertTrue(p_off.is_readonly_ptr)

        p_aligned = d.int64_binary("BitAnd64", p_off, d.const_u64(0xFFFFFFFFFFFFFFF0))
        self.assertEqual(p_aligned.kind, UValKind.PTR)
        self.assertEqual(d.bv.const_int_val(p_aligned.data.offset), 32)
        self.assertTrue(p_aligned.is_readonly_ptr)

        delta = d.int64_binary("ISub64", p_off, p_aligned)
        self.assertEqual(delta.kind, UValKind.BVEC64)
        self.assertEqual(d.const_int_val(delta), 4)

        r1 = d.deref_ptr(p_aligned)
        r2 = d.deref_ptr(p_aligned)
        self.assertEqual(r1, r2)
        self.assertEqual(r1.kind, UValKind.BVEC64)




class TestUDomainIntegration(unittest.TestCase):
    def _make_fp_equiv_bundle(self):
        # Direct SSA: (a * b) + c -> StoreFragColor
        direct_ssa = {
            "blocks": [{"id": 0, "preds": [], "phis": [], "nodes": [1, 2, 3, 4, 5, 6, 7, 8, 9], "is_end": True}],
            "nodes": [
                {"id": 1, "op": "LoadFragVarying", "type": "Vec4", "imm_u32": 0, "args": []},
                {"id": 2, "op": "VecExtract", "type": "F32", "imm_u32": 0, "args": [1]},
                {"id": 3, "op": "VecExtract", "type": "F32", "imm_u32": 1, "args": [1]},
                {"id": 4, "op": "VecExtract", "type": "F32", "imm_u32": 2, "args": [1]},
                {"id": 5, "op": "FMul", "type": "F32", "args": [2, 3]},
                {"id": 6, "op": "FAdd", "type": "F32", "args": [5, 4]},
                {"id": 7, "op": "VecConstruct4", "type": "Vec4", "args": [6, 6, 6, 6]},
                {"id": 8, "op": "Nop", "type": "U32", "args": []},
                {"id": 9, "op": "StoreFragColor", "type": "Void", "imm_u32": 0, "args": [7, 8]},
            ],
        }
        # Valhall SSA: Fma(a, b, c) wrapped in Bitcast(F32->U32) / Bitcast(U32->F32) pairs
        valhall_ssa = {
            "blocks": [{"id": 0, "preds": [], "phis": [], "nodes": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "is_end": True}],
            "nodes": [
                {"id": 1, "op": "LoadFragVarying", "type": "Vec4", "imm_u32": 0, "args": []},
                {"id": 2, "op": "VecExtract", "type": "U32", "imm_u32": 0, "args": [1]},
                {"id": 3, "op": "VecExtract", "type": "U32", "imm_u32": 1, "args": [1]},
                {"id": 4, "op": "VecExtract", "type": "U32", "imm_u32": 2, "args": [1]},
                {"id": 5, "op": "Bitcast", "type": "F32", "args": [2]},
                {"id": 6, "op": "Bitcast", "type": "F32", "args": [3]},
                {"id": 7, "op": "Bitcast", "type": "F32", "args": [4]},
                {"id": 8, "op": "Fma", "type": "F32", "args": [5, 6, 7]},
                {"id": 9, "op": "Bitcast", "type": "U32", "args": [8]},
                {"id": 10, "op": "VecConstruct4", "type": "Vec4", "args": [9, 9, 9, 9]},
                {"id": 11, "op": "StoreFragColor", "type": "Void", "imm_u32": 0, "args": [10, 9]},
            ],
        }
        return {
            "spv_path": "<synthetic_fp_fma>",
            "stage": "Fragment",
            "direct_ok": True,
            "valhall_ok": True,
            "direct_ssa": direct_ssa,
            "valhall_ssa": valhall_ssa,
            "mbs2_context": {},
        }

    def test_verify_equivalence_bundle_udomain_fma_with_bitcasts(self):
        from tools.verify.shader_equiv_verifier import verify_equivalence_bundle
        bundle = self._make_fp_equiv_bundle()
        res = verify_equivalence_bundle(bundle, fp_mode="real", domain="udomain")
        self.assertEqual(res.status, "UNSAT")

    def test_maximal_reachability_udomain_fp_branch(self):
        from tools.coverage.reachability import solve_maximal_reachability
        ssa = {
            "blocks": [
                {"id": 0, "preds": [], "phis": [], "nodes": [1, 2, 3, 4, 5, 6], "is_cond": True, "branch_cond": 6, "branch_target": 1, "fallthrough": 2},
                {"id": 1, "preds": [0], "phis": [], "nodes": [], "is_end": True},
                {"id": 2, "preds": [0], "phis": [], "nodes": [], "is_end": True},
            ],
            "nodes": [
                {"id": 1, "op": "LoadFragVarying", "type": "Vec4", "imm_u32": 0, "args": []},
                {"id": 2, "op": "VecExtract", "type": "F32", "imm_u32": 0, "args": [1]},
                {"id": 3, "op": "ConstF32", "type": "F32", "imm_u32": 0x3F800000, "args": []},
                {"id": 4, "op": "FAdd", "type": "F32", "args": [2, 3]},
                {"id": 5, "op": "Bitcast", "type": "U32", "args": [4]},
                {"id": 6, "op": "FCmpOrdGt", "type": "Bool", "args": [5, 3]},
            ],
        }
        bundle = {"stage": "Fragment", "direct_ssa": ssa, "valhall_ssa": ssa, "mbs2_context": {}}
        r = solve_maximal_reachability(bundle, ssa, fp_mode="real", domain="udomain")
        self.assertTrue(r.complete)
        self.assertEqual(r.reached_blocks, {0, 1, 2})
        self.assertEqual(len(r.paths), 2)

    def test_all_paths_miter_with_udomain(self):
        from tools.verify.all_paths_miter import check_instruction_miter
        from tools.verify.lowering_semantics_extractor import LoweringSemanticsExtractor
        ext = LoweringSemanticsExtractor()
        r = check_instruction_miter(ext, "IADD", "s32", domain="udomain")
        self.assertEqual(r.status, "UNSAT")


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Unit and regression tests for `tools.symbolic_ssa.z3simplifier` and Checkpoint v32 fixes."""

from __future__ import annotations

import unittest

import z3

from tools.symbolic_ssa.domains.udomain import ConcreteUDomain, SymbolicZ3UDomain, UValKind
from tools.symbolic_ssa.domains.z3_base import make_z3_symbolic_domain
from tools.symbolic_ssa.passes import canonicalize_ssa_module
from tools.symbolic_ssa.z3simplifier import (
    BridgeCancelPass,
    BvAlgebraicPass,
    BvWrapperPeepholePass,
    ConditionedUnificationPass,
    ExactCseUnificationPass,
    InjectiveBridgeUnwrapPass,
    PassPipeline,
    PolynomialNormalizePass,
    PreconditionConstSubPass,
    ReadOnlyArrayScalarizePass,
    RealAlgebraicPass,
    SimplificationContext,
    SimplifyQuery,
    TermPassAdapter,
    TheoryPurifyRealPass,
    build_default_pipeline,
    build_exact_pipeline,
    census_ast,
    format_z3_ast,
    solve_simplified_miter,
)


class TestZ3SimplifierPasses(unittest.TestCase):
    def test_issue1_bv_wrapper_peephole_collapses_nested_bool_wrappers(self) -> None:
        b = z3.Bool("b")
        # Construct 4-layer UDomain v31.2-style BV64 boolean wrapper bloat:
        # If(0 != Extract(31, 0, ZeroExt(32, If(0 != Extract(31, 0, Concat(0, If(b, 1, 0))), 1, 0))), 1, 0) != 0
        w1 = z3.If(b, z3.BitVecVal(1, 32), z3.BitVecVal(0, 32))
        w2 = z3.Extract(31, 0, z3.Concat(z3.BitVecVal(0, 32), w1)) != z3.BitVecVal(0, 32)
        w3 = z3.If(w2, z3.BitVecVal(1, 32), z3.BitVecVal(0, 32))
        w4 = z3.Extract(31, 0, z3.ZeroExt(32, w3)) != z3.BitVecVal(0, 32)

        p = BvWrapperPeepholePass()
        ctx = SimplificationContext()
        out, rw = p.rewrite_term(w4, ctx)
        self.assertGreater(rw, 0)
        self.assertTrue(out.eq(b), f"Expected raw Bool `b`, got {out}")

    def test_issue1_udomain_bool_kind_avoids_bv64_wrapper_churn(self) -> None:
        dom = SymbolicZ3UDomain(exact_bitcast=False)
        x = dom.fresh_f32("x")
        y = dom.fresh_f32("y")
        c1 = dom.fp_cmp("FCmpOrdLt", x, y)
        c2 = dom.logical_not(c1)
        c3 = dom.logical_and(c1, c2)
        self.assertEqual(c1.kind, UValKind.BOOL)
        self.assertEqual(c2.kind, UValKind.BOOL)
        self.assertEqual(c3.kind, UValKind.BOOL)
        raw = dom.lower_z3_bool(c3)
        self.assertNotIn("Extract", str(raw))
        self.assertNotIn("ZeroExt", str(raw))

        # Concrete UDomain BOOL round-trips cleanly
        cdom = ConcreteUDomain()
        cb = cdom.const_bool(True)
        self.assertEqual(cb.kind, UValKind.BOOL)
        self.assertEqual(cdom.to_bvec64(cb), 1)
        self.assertEqual(cdom.to_bvec64(cdom.logical_not(cb)), 0)

    def test_issue2_readonly_array_scalarize_soundness(self) -> None:
        arr_ro = z3.Array("mem_ubo_0_1", z3.BitVecSort(32), z3.BitVecSort(32))
        arr_dyn = z3.Array("mem_ssbo_0_0", z3.BitVecSort(32), z3.BitVecSort(32))
        idx = z3.BitVec("dyn_idx", 32)

        # Read-only array with only constant indices IS scalarized
        q_ro = SimplifyQuery(
            lhs=z3.Select(arr_ro, z3.BitVecVal(40, 32)) + z3.Select(arr_ro, z3.BitVecVal(44, 32)),
            rhs=z3.Select(arr_ro, z3.BitVecVal(44, 32)) + z3.Select(arr_ro, z3.BitVecVal(40, 32)),
        )
        res_ro = PassPipeline([ReadOnlyArrayScalarizePass()]).run(q_ro)
        self.assertEqual(census_ast([res_ro.query.lhs, res_ro.query.rhs]).array_select_ops, 0)
        self.assertFalse(res_ro.query.is_overapprox)

        # Array with a dynamic index MUST NOT be scalarized (soundness guard)
        q_dyn = SimplifyQuery(
            lhs=z3.Select(arr_dyn, z3.BitVecVal(40, 32)),
            rhs=z3.Select(arr_dyn, idx),
        )
        res_dyn = PassPipeline([ReadOnlyArrayScalarizePass()]).run(q_dyn)
        self.assertEqual(census_ast([res_dyn.query.lhs, res_dyn.query.rhs]).array_select_ops, 2)

        # Array with a Store MUST NOT be scalarized
        arr_st = z3.Store(arr_ro, z3.BitVecVal(40, 32), z3.BitVecVal(7, 32))
        q_st = SimplifyQuery(
            lhs=z3.Select(arr_st, z3.BitVecVal(40, 32)),
            rhs=z3.BitVecVal(7, 32),
        )
        res_st = PassPipeline([ReadOnlyArrayScalarizePass()]).run(q_st)
        self.assertGreater(census_ast([res_st.query.lhs, res_st.query.rhs]).array_store_ops, 0)

    def test_issue3_injective_unwrap_and_som_discharges_mvp_dot_product(self) -> None:
        u_f32_to_bv32 = z3.Function("u_f32_to_bv32", z3.RealSort(), z3.BitVecSort(32))
        u_bv32_to_f32 = z3.Function("u_bv32_to_f32", z3.BitVecSort(32), z3.RealSort())
        arr = z3.Array("mem_ubo_0_1", z3.BitVecSort(32), z3.BitVecSort(32))

        v = [u_bv32_to_f32(z3.BitVec(f"v{i}", 32)) for i in range(4)]
        m = [u_bv32_to_f32(z3.Select(arr, z3.BitVecVal(40 + 4 * i, 32))) for i in range(4)]

        # Direct: ((v0*m0 + v1*m1) + v2*m2) + v3*m3 wrapped in Extract(31,0,Concat(0,u_f32_to_bv32(...)))
        p_direct = ((v[0] * m[0] + v[1] * m[1]) + v[2] * m[2]) + v[3] * m[3]
        # Valhall: reassociated FMA chain
        p_valhall = v[3] * m[3] + (v[2] * m[2] + (v[1] * m[1] + v[0] * m[0]))

        lhs = z3.Extract(31, 0, z3.Concat(z3.BitVecVal(0, 32), u_f32_to_bv32(p_direct)))
        rhs = z3.Extract(31, 0, z3.Concat(z3.BitVecVal(0, 32), u_f32_to_bv32(p_valhall)))

        res = solve_simplified_miter(SimplifyQuery(lhs=lhs, rhs=rhs), build_default_pipeline())
        self.assertEqual(res.status, "UNSAT")
        self.assertEqual(res.solver_ms, 0.0, "Expected <1ms polynomial discharge with zero SMT calls")

    def test_issue2_theory_purify_real_routes_to_nlsat_and_guards_spurious_sat(self) -> None:
        u_bv32_to_f32 = z3.Function("u_bv32_to_f32", z3.BitVecSort(32), z3.RealSort())
        x_bv = z3.BitVec("x_bv", 32)
        y_bv = z3.BitVec("y_bv", 32)
        x = u_bv32_to_f32(x_bv)
        y = u_bv32_to_f32(y_bv)

        # Nonlinear algebraic identity: (x + y)^2 == x^2 + 2xy + y^2
        # Disable polynomial_normalize to force TheoryPurifyRealPass -> QF_NRA nlsat
        pipe = PassPipeline([TheoryPurifyRealPass()])
        q_ok = SimplifyQuery(lhs=(x + y) * (x + y), rhs=x * x + z3.RealVal(2) * x * y + y * y)
        res_ok = solve_simplified_miter(q_ok, pipe, timeout_ms=500)
        self.assertEqual(res_ok.status, "UNSAT")
        self.assertEqual(res_ok.simplified_query.pure_theory, "QF_NRA")
        self.assertTrue(res_ok.used_nlsat)

        # Genuine divergence must still return SAT (confirmed on exact_query)
        q_bad = SimplifyQuery(lhs=x * y + z3.RealVal(1), rhs=x * y)
        res_bad = solve_simplified_miter(q_bad, pipe, timeout_ms=500)
        self.assertEqual(res_bad.status, "SAT")
        self.assertTrue(res_bad.refined_from_overapprox)

    def test_issue6_bv_algebraic_and_ssa_imul_pow2_to_shl(self) -> None:
        x = z3.BitVec("idx", 32)
        lhs = (x * z3.BitVecVal(64, 32)) + z3.BitVecVal(16, 32)
        rhs = (x << z3.BitVecVal(6, 32)) + z3.BitVecVal(16, 32)
        res = solve_simplified_miter(SimplifyQuery(lhs=lhs, rhs=rhs), build_exact_pipeline())
        self.assertEqual(res.status, "UNSAT")
        self.assertEqual(res.solver_ms, 0.0)

        # Also test SSA pass_gvn_dce IMul(x, 64) -> Shl(x, 6)
        ssa = {
            "blocks": [{
                "id": 0, "preds": [], "phis": [], "nodes": [1, 2, 3, 4, 5],
                "branch_target": -1, "fallthrough": -1, "is_cond": False, "is_uncond": False,
                "is_end": True, "is_discard": False, "branch_cond": 0,
            }],
            "nodes": [
                {"id": 1, "op": "Nop", "type": "Token", "blk": 0, "args": []},
                {"id": 2, "op": "LoadBuiltinInput", "type": "U32", "blk": 0, "args": [], "imm_u32": 29},
                {"id": 3, "op": "ConstU32", "type": "U32", "blk": 0, "args": [], "imm_u32": 64},
                {"id": 4, "op": "IMul", "type": "U32", "blk": 0, "args": [2, 3]},
                {"id": 5, "op": "StoreFragColor", "type": "Token", "blk": 0, "args": [4, 1], "imm_u32": 0},
            ],
        }
        canon, _ = canonicalize_ssa_module(ssa)
        ops = [n["op"] for n in canon["nodes"]]
        self.assertIn("Shl", ops)
        self.assertNotIn("IMul", ops)

    def test_exact_cse_unification_and_spurious_sat_fallback(self) -> None:
        a = z3.BitVec("a", 32)
        b = z3.BitVec("b", 32)
        # Expensive shared subexpression: ((a * b) ^ (a + b))
        shared = (a * b) ^ (a + b)
        q = SimplifyQuery(lhs=shared + z3.BitVecVal(4, 32), rhs=z3.BitVecVal(4, 32) + shared)
        sim = PassPipeline([ExactCseUnificationPass(min_depth=2)]).run(q)
        self.assertTrue(sim.query.is_overapprox)
        self.assertGreaterEqual(len(sim.query.abstraction_map), 1)
        res = solve_simplified_miter(q, build_default_pipeline())
        self.assertEqual(res.status, "UNSAT")

        # Spurious SAT case: lhs = (a * b) + a, rhs = a * (b + 1).
        # Even if CSE lifted something, exact fallback ensures UNSAT (not false SAT!).
        q_corr = SimplifyQuery(lhs=(a * b) + a, rhs=a * (b + z3.BitVecVal(1, 32)))
        res_corr = solve_simplified_miter(q_corr, build_default_pipeline())
        self.assertEqual(res_corr.status, "UNSAT")

    def test_conditioned_unification_unifies_fma_under_transcendental(self) -> None:
        real_sqrt = z3.Function("real_sqrt", z3.RealSort(), z3.RealSort())
        a, b, c, d = z3.Reals("a b c d")
        # Non-identical subtrees under an uninterpreted `real_sqrt`:
        # Direct:  1 / sqrt((a*b + c) + d*d)
        # Valhall: 1 / sqrt(d*d + (c + b*a))
        sub_l = (a * b + c) + d * d
        sub_r = d * d + (c + b * a)
        q = SimplifyQuery(
            lhs=z3.RealVal(1) / real_sqrt(sub_l),
            rhs=z3.RealVal(1) / real_sqrt(sub_r),
        )
        pipe = PassPipeline([ConditionedUnificationPass(min_depth=2, max_depth=8)])
        sim = pipe.run(q)
        self.assertTrue(sim.query.is_trivially_unsat(), f"Expected conditioned unification to collapse to UNSAT, got {sim.query.lhs} vs {sim.query.rhs}")
        self.assertEqual(len(sim.query.side_obligations), 1)
        self.assertTrue(sim.query.side_obligations[0].discharged)

        # Negative test: genuinely different subtrees must NOT be unified
        q_diff = SimplifyQuery(
            lhs=z3.RealVal(1) / real_sqrt(a * b + c),
            rhs=z3.RealVal(1) / real_sqrt(a * b + d),
        )
        sim_diff = pipe.run(q_diff)
        self.assertEqual(len(sim_diff.query.side_obligations), 0)
        self.assertFalse(sim_diff.query.is_trivially_unsat())

    def test_default_domain_is_udomain_and_format_z3_ast(self) -> None:
        dom = make_z3_symbolic_domain()
        self.assertIsInstance(dom, SymbolicZ3UDomain)
        self.assertFalse(dom.exact_bitcast)

        x, y = z3.Reals("x y")
        shared = x * y + z3.RealVal(1)
        expr = shared * shared + shared
        pretty = format_z3_ast(expr, let_bind_shared=True)
        self.assertIn("(let (", pretty)
        self.assertIn("%t0", pretty)



    def test_v32_1_intbv_constant_folding_and_udomain_avoids_intbv(self) -> None:
        """Constant BV/Int/Real conversions fold cleanly and non-exact udomain emits zero IntBV ops."""
        from tools.symbolic_ssa.domains.udomain import SymbolicZ3UDomain
        from tools.symbolic_ssa.z3simplifier.ast_utils import census_ast

        # Constant-folding through Extract -> ZeroExt -> BVMul and BV2Int -> ToReal -> ToInt -> Int2BV
        x = z3.BitVec("x", 32)
        expr_mul = x * z3.ZeroExt(24, z3.Extract(31, 24, z3.BitVecVal(0x40000000, 32)))
        q1 = SimplifyQuery(lhs=expr_mul, rhs=x << z3.BitVecVal(6, 32))
        res1 = solve_simplified_miter(q1, build_default_pipeline(), timeout_ms=200)
        self.assertEqual(res1.status, "UNSAT")

        intbv_const = z3.Int2BV(z3.ToInt(z3.ToReal(z3.BV2Int(z3.BitVecVal(7, 32), is_signed=False))), 32)
        q2 = SimplifyQuery(lhs=intbv_const, rhs=z3.BitVecVal(7, 32))
        res2 = solve_simplified_miter(q2, build_default_pipeline(), timeout_ms=200)
        self.assertEqual(res2.status, "UNSAT")

        # Non-exact udomain must emit zero IntBV ops on ConvertSToF/ConvertFToS/Floor/Fract
        dom = SymbolicZ3UDomain(exact_bitcast=False)
        ux = dom.lift_z3_bv32(x)
        r1 = dom.convert("ConvertSToF", ux)
        r2 = dom.fp_unary("Floor", r1)
        r3 = dom.fp_unary("Fract", r1)
        b1 = dom.convert("ConvertFToS", r2)
        cen = census_ast([dom.lower_z3_bv32(b1), dom.lower_z3_bv32(r3)])
        self.assertEqual(cen.int_bv_ops, 0)
        self.assertEqual(cen.fp_ops, 0)

        # Constant conversion still folds exactly in non-exact udomain
        c5 = dom.convert("ConvertSToF", dom.const_u32(5))
        c5_back = dom.convert("ConvertFToS", c5)
        self.assertEqual(dom.lower_z3_bv32(c5_back).as_long(), 5)



    def test_v32_2_dataflow_extensional_equalities_and_known_bits(self) -> None:
        """DataflowPropagatePass propagates Union-Find equalities and KnownBits constant collapses."""
        a, b, x = z3.BitVecs("a b x", 32)
        f = z3.Function("opaque_f", z3.BitVecSort(32), z3.BitVecSort(32))
        # Extensional equality: precondition `a == b` => `f(a) == f(b)` and `(a - b) == 0`
        q_eq = SimplifyQuery(lhs=f(a) + (a - b), rhs=f(b), preconditions=(a == b,))
        r_eq = solve_simplified_miter(q_eq, build_default_pipeline(), timeout_ms=200)
        self.assertEqual(r_eq.status, "UNSAT")
        self.assertEqual(r_eq.solver_ms, 0.0)

        # KnownBits: Extract(63, 32, ZeroExt(32, x)) == 0 and ((x << 4) & 0xF) == 0
        hi_bits = z3.Extract(63, 32, z3.ZeroExt(32, x))
        low_mask = (x << z3.BitVecVal(4, 32)) & z3.BitVecVal(0xF, 32)
        q_kb = SimplifyQuery(lhs=hi_bits | low_mask, rhs=z3.BitVecVal(0, 32))
        r_kb = solve_simplified_miter(q_kb, build_default_pipeline(), timeout_ms=200)
        self.assertEqual(r_kb.status, "UNSAT")
        self.assertEqual(r_kb.solver_ms, 0.0)

    def test_v32_2_path_sensitive_if_and_bridge_over_if_unlock_qfnra(self) -> None:
        """Path-sensitive DataflowPropagatePass + bridge-over-If collapse CFG diamonds & unlock QF_NRA."""
        u_f2b = z3.Function("u_f32_to_bv32", z3.RealSort(), z3.BitVecSort(32))
        u_b2f = z3.Function("u_bv32_to_f32", z3.BitVecSort(32), z3.RealSort())
        discarded, cond = z3.Bools("discarded cond")
        r1, r2 = z3.Reals("r1 r2")
        undef = z3.BitVec("undef", 32)

        # Bridge-over-If cancellation: u_bv32_to_f32(If(cond, u_f32_to_bv32(r1), u_f32_to_bv32(r2))) -> If(cond, r1, r2)
        merged_bv = z3.If(cond, u_f2b(r1), u_f2b(r2))
        q_br = SimplifyQuery(lhs=u_b2f(merged_bv), rhs=z3.If(cond, r1, r2))
        r_br = solve_simplified_miter(q_br, build_default_pipeline(), timeout_ms=200)
        self.assertEqual(r_br.status, "UNSAT")
        self.assertEqual(r_br.solver_ms, 0.0)

        # Fragment discarded guard: guard=Not(discarded) collapses If(discarded, undef, u_f2b(...))
        lhs = z3.If(discarded, undef, u_f2b((r1 + r2) * r1))
        rhs = z3.If(discarded, undef, u_f2b(r1 * r1 + r2 * r1))
        q_frag = SimplifyQuery(lhs=lhs, rhs=rhs, guard=z3.Not(discarded))
        r_frag = solve_simplified_miter(q_frag, build_default_pipeline(), timeout_ms=200)
        self.assertEqual(r_frag.status, "UNSAT")
        self.assertEqual(r_frag.solver_ms, 0.0)



    def test_v32_3_autogen_dsl_enums_lower_and_ematching_extensionality(self) -> None:
        """Verify auto-generated DSL enums, .lower() round-trip, UDomain F32X2 select, and E-matching extensionality."""
        from tools.symbolic_ssa.domains.udomain import SymbolicZ3UDomain, UValKind
        from tools.symbolic_ssa.z3simplifier.dsl import (
            BridgeOp,
            BvCmpOp,
            Z3BvConst,
            Z3OpBvAdd,
            Z3OpBvCmp,
            Z3OpExtract,
            Z3OpKind,
            Z3SortKind,
            lift,
        )
        from tools.symbolic_ssa.z3simplifier.passes_ematch import (
            EmatchingUnifyPass,
            build_extensionality_witness_axiom,
        )

        # Auto-generated IntEnum / Enum checks + .lower() round-trip
        self.assertEqual(int(Z3OpKind.EXTRACT), z3.Z3_OP_EXTRACT)
        self.assertEqual(int(Z3SortKind.BV), z3.Z3_BV_SORT)
        self.assertEqual(BridgeOp.F32_TO_BV32.value, "u_f32_to_bv32")
        x = z3.BitVec("x", 32)
        built = Z3OpBvCmp(BvCmpOp.ULT, Z3OpBvAdd(x, Z3BvConst(4, 32), 32), Z3BvConst(16, 32)).lower()
        self.assertTrue(built.eq(z3.ULT(x + z3.BitVecVal(4, 32), z3.BitVecVal(16, 32))))
        self.assertIsInstance(lift(z3.Extract(7, 0, x), depth=1), Z3OpExtract)

        # UDomain.select preserves F32X2 across (F32X2, BVEC64) merges without demoting to BVEC64
        dom = SymbolicZ3UDomain(exact_bitcast=False)
        c = dom.lift_z3_bool(z3.Bool("c"))
        f_arm = dom.fp_binary("FAdd", dom.lift_z3_bv32(x), dom.const_f32(1.0))
        bv_arm = dom.lift_z3_bv32(z3.BitVec("y", 32))
        merged = dom.select(c, f_arm, bv_arm)
        self.assertEqual(merged.kind, UValKind.F32X2)

        # Z3 E-matching Skolem-witness extensionality proof:
        # ForAll([a], apply(fn_f, a) == apply(fn_g, a)) + ext_witness axiom => fn_f == fn_g
        S = z3.DeclareSort("FnSort")
        apply_fn = z3.Function("apply_fn", S, S, S)
        fn_f, fn_g, a = z3.Consts("fn_f fn_g a", S)
        pt_eq = z3.ForAll([a], apply_fn(fn_f, a) == apply_fn(fn_g, a), patterns=[apply_fn(fn_f, a)])
        ext_ax = build_extensionality_witness_axiom(S, apply_fn, fn_f, fn_g)
        q_ext = SimplifyQuery(lhs=fn_f, rhs=fn_g)
        ematch = EmatchingUnifyPass(extra_axioms=(pt_eq, ext_ax), include_bridge_axioms=False)
        out_q, st = ematch.run(q_ext, SimplificationContext())
        self.assertTrue(st.changed)
        self.assertTrue(out_q.is_trivially_unsat())


    def test_v33_lazy_dsl_operators_memoization_and_pattern_matching(self) -> None:
        """Verify autogenerated DSL operator overloads, lazy memoized .to_z3, and match/case."""
        from tools.symbolic_ssa.z3simplifier import dsl

        # 1. BitVec operators stay purely in DSL until .to_z3 is accessed
        x = dsl.BitVec("x", 32)
        y = dsl.BitVec("y", 32)
        expr_bv = ((x + y) ^ (x & dsl.BitVecVal(0xFF, 32))) - (y << 2)
        self.assertIsInstance(expr_bv, dsl.Z3BitVecExpr)
        self.assertIsNone(expr_bv._z3_cache)
        self.assertIsNone(x._z3_cache)
        self.assertEqual(expr_bv.size(), 32)

        # Structural match/case without touching Z3
        match expr_bv:
            case dsl.Z3OpBvSub(lhs=dsl.Z3OpBvXor(lhs=dsl.Z3OpBvAdd(lhs=a, rhs=b)), rhs=dsl.Z3OpBvShl()):
                self.assertTrue(a.eq(x))
                self.assertTrue(b.eq(y))
            case _:
                self.fail(f"Pattern match failed on {expr_bv!r}")
        self.assertIsNone(expr_bv._z3_cache)

        # Lazy + memoized .to_z3 / .lower()
        z1 = expr_bv.to_z3
        z2 = expr_bv.lower()
        self.assertIsInstance(z1, z3.BitVecRef)
        self.assertIs(z1, z2)
        zx, zy = z3.BitVec("x", 32), z3.BitVec("y", 32)
        expected_z = ((zx + zy) ^ (zx & z3.BitVecVal(0xFF, 32))) - (zy << 2)
        self.assertTrue(z1.eq(expected_z))

        # 2. Real / Int / Bool / Array / UF operators
        r = dsl.Real("r")
        r_expr = (r * dsl.RealVal(2)) + dsl.Q(1, 2)
        self.assertIsInstance(r_expr, dsl.Z3ArithExpr)
        self.assertIsNone(r_expr._z3_cache)
        self.assertTrue(r_expr.is_real())

        arr = dsl.Array("mem", z3.BitVecSort(32), z3.BitVecSort(32))
        sel = dsl.Store(arr, dsl.BitVecVal(4, 32), x)[dsl.BitVecVal(4, 32)]
        self.assertIsInstance(sel, dsl.Z3OpSelect)
        self.assertIsNone(sel._z3_cache)
        self.assertTrue(dsl.simplify(sel).eq(zx))

        # Symbolic == / != return Z3BoolExpr; bool() raises TypeError
        eq_expr = (x == y)
        self.assertIsInstance(eq_expr, dsl.Z3OpEq)
        self.assertIsNone(eq_expr._z3_cache)
        with self.assertRaises(TypeError):
            bool(eq_expr)

    def test_v33_dag_eq_and_hash_avoid_exponential_blowup(self) -> None:
        """Comparing two 20-level diamond DAGs via .eq() must stay O(|V|) with zero Z3 lowering."""
        import time
        from tools.symbolic_ssa.z3simplifier import dsl

        def build_diamond(depth: int) -> dsl.Z3BitVecExpr:
            v: dsl.Z3BitVecExpr = dsl.BitVec("x", 32)
            for i in range(depth):
                c = dsl.Bool(f"c{i}")
                v = dsl.If(c, v + dsl.BitVecVal(1, 32), v ^ dsl.BitVecVal(1, 32))
            return v

        d1 = build_diamond(20)
        d2 = build_diamond(20)
        d3 = d1 + dsl.BitVecVal(1, 32)

        t0 = time.perf_counter()
        self.assertTrue(d1.eq(d2))
        self.assertFalse(d1.eq(d3))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.assertLess(elapsed_ms, 50.0, f"DAG .eq() took {elapsed_ms:.1f} ms")
        self.assertIsNone(d1._z3_cache)
        self.assertIsNone(d2._z3_cache)

    def test_v33_zero_z3_churn_trivial_unsat_and_interpreter_lowering(self) -> None:
        """Identical DSL queries discharge UNSAT with zero Z3 lowering; interpreter emits pure DSL."""
        from tools.symbolic_ssa import SharedShaderEnv, SymbolicSsaInterpreter
        from tools.symbolic_ssa.z3simplifier import dsl

        x = dsl.BitVec("x", 32)
        y = dsl.BitVec("y", 32)
        lhs = (x + y) ^ dsl.BitVecVal(0x1234, 32)
        rhs = (x + y) ^ dsl.BitVecVal(0x1234, 32)
        q = SimplifyQuery(lhs=lhs, rhs=rhs)
        pipe = build_default_pipeline()
        res = pipe.run(q, SimplificationContext())
        self.assertTrue(res.query.is_trivially_unsat())
        self.assertIsNone(lhs._z3_cache)
        self.assertIsNone(rhs._z3_cache)

        # End-to-end SymbolicSsaInterpreter + UDomain stays in DSL prior to solving
        ssa = {
            "blocks": [{"id": 0, "preds": [], "phis": [], "nodes": [1, 2, 3, 4], "is_end": True}],
            "nodes": [
                {"id": 1, "op": "LoadFragVarying", "type": "Vec4", "imm_u32": 0, "args": []},
                {"id": 2, "op": "VecExtract", "type": "U32", "imm_u32": 0, "args": [1]},
                {"id": 3, "op": "ConstU32", "type": "U32", "imm_u32": 7, "args": []},
                {"id": 4, "op": "IAdd", "type": "U32", "args": [2, 3]},
            ],
        }
        bundle = {"stage": "Fragment", "direct_ssa": ssa, "valhall_ssa": ssa, "mbs2_context": {}}
        env = SharedShaderEnv(bundle)
        interp = SymbolicSsaInterpreter(ssa, env, is_valhall_side=False, fp_mode="real", domain="udomain")
        interp.evaluate()
        out_bv = interp.domain.lower_z3_bv32(interp.values[4])
        self.assertIsInstance(out_bv, dsl.Z3BitVecExpr)
        self.assertIsNone(out_bv._z3_cache)

if __name__ == "__main__":
    unittest.main()

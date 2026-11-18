#!/usr/bin/env python3
"""
Unit & Formal SMT Property Tests for Frozen Panfrost Reference Semantics (\Psi_pan).
Includes regression tests for all 14 Mesa kraid discrepancies + 3 minor gaps.
"""

import os
import sys
import unittest

import z3

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.verify import (
    PANFROST_SEMANTICS,
    ValhallZ3State,
    VerificationLogger,
    apply_clamp_f32,
    apply_clamp_v2f16,
    apply_src_mod_32,
    apply_swizzle_u32,
    apply_swizzle_u64,
    eval_panfrost_instruction,
    extract_panfrost_spec_catalog,
    verify_instruction_equivalence,
    write_dst_32,
)
from tools.verify.panfrost_z3_model import bv32, bv64


class TestPanfrostZ3Model(unittest.TestCase):
    def test_xml_completeness_627_specs_201_mnemonics(self):
        catalog = extract_panfrost_spec_catalog(os.path.join(REPO_ROOT, "specs/kraid/isa-v9-v14.xml"))
        self.assertEqual(len(catalog), 627)
        mnems = {entry["name"] for entry in catalog}
        self.assertEqual(len(mnems), 201)
        self.assertTrue(mnems.issubset(set(PANFROST_SEMANTICS.keys())))

    def test_all_201_mnemonics_smoke_evaluate(self):
        logger = VerificationLogger()
        x0 = z3.BitVec("x0", 32)
        x1 = z3.BitVec("x1", 32)
        x2 = z3.BitVec("x2", 32)
        x3 = z3.BitVec("x3", 32)
        dst_old = z3.BitVec("dst_old", 32)
        for mnem in sorted(PANFROST_SEMANTICS.keys()):
            res, st, tr = eval_panfrost_instruction(
                mnem,
                "f32" if mnem.startswith("F") else "u32",
                [x0, x1, x2, x3],
                modifiers={},
                dst_old=dst_old,
                logger=logger,
            )
            self.assertEqual(tr.mnemonic, mnem)
            self.assertIsNotNone(st)
        self.assertEqual(len(logger.traces), 201)

    def test_swizzle_involutions_unsat(self):
        x = z3.BitVec("x", 32)
        for swz in ("b3210", "h10", "b2301"):
            double = apply_swizzle_u32(apply_swizzle_u32(x, swz), swz)
            status, cex = verify_instruction_equivalence("SWZ", swz, x, double, {"x": x})
            self.assertEqual(status, "UNSAT", msg=f"{swz} failed involution: {cex}")

    def test_swizzle_concrete_modes(self):
        val = bv32(0x11223344)
        cases = {
            "b0": 0x00000044,
            "b1": 0x00000033,
            "b2": 0x00000022,
            "b3": 0x00000011,
            "b0000": 0x44444444,
            "b3333": 0x11111111,
            "b0011": 0x33334444,
            "b2233": 0x11112222,
            "b0101": 0x33443344,
            "b2323": 0x11221122,
            "b1032": 0x22114433,
            "b3210": 0x44332211,
            "h0": 0x00003344,
            "h1": 0x00001122,
            "h00": 0x33443344,
            "h11": 0x11221122,
            "h10": 0x33441122,
            "b13": 0x00110033,
        }
        for swz, expected in cases.items():
            got = z3.simplify(apply_swizzle_u32(val, swz))
            self.assertEqual(got.as_long(), expected, msg=f"swz={swz}")

        val64 = bv64(0xAABBCCDD11223344)
        self.assertEqual(z3.simplify(apply_swizzle_u64(val64, "w0")).as_long(), 0x0000000011223344)
        self.assertEqual(z3.simplify(apply_swizzle_u64(val64, "w1")).as_long(), 0x00000000AABBCCDD)
        self.assertEqual(z3.simplify(apply_swizzle_u64(val64, "w10")).as_long(), 0x11223344AABBCCDD)

    def test_swar_add_sub_equivalence_unsat(self):
        x = z3.BitVec("x", 32)
        y = z3.BitVec("y", 32)

        pan_add16, _, _ = eval_panfrost_instruction("IADD", "v2u16", [x, y])
        m_lo16 = bv32(0x7FFF7FFF)
        m_hi16 = bv32(0x80008000)
        swar_add16 = ((x & m_lo16) + (y & m_lo16)) ^ ((x ^ y) & m_hi16)
        st1, cex1 = verify_instruction_equivalence("IADD", "v2u16", pan_add16, swar_add16, {"x": x, "y": y})
        self.assertEqual(st1, "UNSAT", msg=cex1)

        pan_sub8, _, _ = eval_panfrost_instruction("ISUB", "v4u8", [x, y])
        m_lo8 = bv32(0x7F7F7F7F)
        m_hi8 = bv32(0x80808080)
        swar_sub8 = ((x | m_hi8) - (y & m_lo8)) ^ ((x ^ (~y)) & m_hi8)
        st2, cex2 = verify_instruction_equivalence("ISUB", "v4u8", pan_sub8, swar_sub8, {"x": x, "y": y})
        self.assertEqual(st2, "UNSAT", msg=cex2)

    def test_miter_catches_scalar_bleed_bug_sat_with_cex(self):
        x = z3.BitVec("x", 32)
        y = z3.BitVec("y", 32)
        pan_add16, _, trace = eval_panfrost_instruction("IADD", "v2u16", [x, y])
        buggy_scalar = x + y
        status, cex = verify_instruction_equivalence("IADD", "v2u16", pan_add16, buggy_scalar, {"x": x, "y": y})
        self.assertEqual(status, "SAT")
        self.assertIn("COUNTEREXAMPLE for IADD.v2u16", cex)
        self.assertIn("Panfrost Expected", cex)
        self.assertIn("StageB:", trace.format_summary())

    def test_ops_rs_edge_cases(self):
        clz_res, _, _ = eval_panfrost_instruction("CLZ", "u32", [bv32(0)], modifiers={"mask": True})
        self.assertEqual(z3.simplify(clz_res).as_long(), 0xFFFFFFFF)

        neg0 = apply_src_mod_32(bv32(0), "f32", neg_flag=True)
        self.assertEqual(z3.simplify(neg0).as_long(), 0x80000000)

        mux_res, _, _ = eval_panfrost_instruction(
            "MUX", "u32", [bv32(0xAAAAAAAA), bv32(0x55555555), bv32(0)], modifiers={"muxf": "int_zero"}
        )
        self.assertEqual(z3.simplify(mux_res).as_long(), 0xAAAAAAAA)

        cf2, _, _ = eval_panfrost_instruction("CUBEFACE2", "", [bv32(0), bv32(0), bv32(0xBF800000)])
        self.assertEqual(z3.simplify(cf2).as_long(), 0xA0000000)

        lscale, _, _ = eval_panfrost_instruction("FADD_LSCALE", "f32", [bv32(0), bv32(0x40C00000)])
        self.assertEqual(z3.simplify(lscale).as_long(), 0x3F400000)

        wd = write_dst_32(bv32(0x0000ABCD), bv32(0x12345678), dest_width="h1")
        self.assertEqual(z3.simplify(wd).as_long(), 0xABCD5678)

        st0 = ValhallZ3State.symbolic("cf")
        _, st1, _ = eval_panfrost_instruction("DISCARD", "", [bv32(1)], state=st0)
        self.assertTrue(z3.is_true(z3.simplify(st1.discarded)))

    # -----------------------------------------------------------------------
    # Regression Tests for All 14 Mesa kraid Discrepancies + 3 Minor Gaps
    # -----------------------------------------------------------------------
    def test_reg_01_flush_canonicalization(self):
        res_nan, _, _ = eval_panfrost_instruction("FLUSH", "f32", [bv32(0x7FC00000)], modifiers={"flush_nan": "flush_nan"})
        self.assertEqual(z3.simplify(res_nan).as_long(), 0x00000000)

        res_qnan, _, _ = eval_panfrost_instruction("FLUSH", "f32", [bv32(0x7F800001)], modifiers={"flush_nan": "quiet_nan"})
        self.assertEqual(z3.simplify(res_qnan).as_long(), 0x7FC00001)

        res_inf, _, _ = eval_panfrost_instruction("FLUSH", "f32", [bv32(0xFF800000)], modifiers={"flush_inf": True})
        self.assertEqual(z3.simplify(res_inf).as_long(), 0xFF7FFFFF)

        res_ftz, _, _ = eval_panfrost_instruction("FLUSH", "f32", [bv32(0x80000001)], modifiers={"ftz": True})
        self.assertEqual(z3.simplify(res_ftz).as_long(), 0x80000000)

    def test_reg_02_mmul_not_integer_imul(self):
        a, b, c = z3.BitVec("a", 32), z3.BitVec("b", 32), z3.BitVec("c", 32)
        mmul_res, _, _ = eval_panfrost_instruction("MMUL", "f32", [a, b, c])
        imul_res, _, _ = eval_panfrost_instruction("IMUL", "u32", [a, b])
        status, _ = verify_instruction_equivalence("MMUL", "f32", mmul_res, imul_res, {"a": a, "b": b})
        self.assertEqual(status, "SAT")

    def test_reg_03_04_shadd_src1_shift_and_unsigned_widen(self):
        # Discrepancy 3: SHADD.u64(1, 2, shift=2) -> 1 + (2 << 2) = 9
        res, _, _ = eval_panfrost_instruction("SHADD", "u64", [bv64(1), bv64(2)], modifiers={"shift": 2})
        self.assertEqual(z3.simplify(res).as_long(), 9)

        # Discrepancy 4: SHADD.u64(0, 0x80000000.w0) -> zero-extended 0x0000000080000000
        res_w0, _, _ = eval_panfrost_instruction(
            "SHADD", "u64", [bv64(0), bv64(0x80000000)], modifiers={"widen1": "w0", "shift": 0}
        )
        self.assertEqual(z3.simplify(res_w0).as_long(), 0x0000000080000000)

    def test_reg_05_clamp_m1_1_on_nan(self):
        self.assertEqual(z3.simplify(apply_clamp_f32(bv32(0x7FC00000), "clamp_m1_1")).as_long(), 0xBF800000)
        self.assertEqual(z3.simplify(apply_clamp_v2f16(bv32(0x7E007E00), "clamp_m1_1")).as_long(), 0xBC00BC00)

    def test_reg_06_fround_modes(self):
        # 1.25 = 0x3FA00000, round_up -> 2.0 = 0x40000000
        r_up, _, _ = eval_panfrost_instruction("FROUND", "f32", [bv32(0x3FA00000)], modifiers={"round": "round_up"})
        self.assertEqual(z3.simplify(r_up).as_long(), 0x40000000)

        # 1.5 = 0x3FC00000, round_na (ties away) -> 2.0 = 0x40000000; -1.5 = 0xBFC00000 -> -2.0 = 0xC0000000
        r_na, _, _ = eval_panfrost_instruction("FROUND", "f32", [bv32(0xBFC00000)], modifiers={"round": "round_na"})
        self.assertEqual(z3.simplify(r_na).as_long(), 0xC0000000)

        with self.assertRaises(ValueError):
            eval_panfrost_instruction("FROUND", "f32", [bv32(0x3FA00000)], modifiers={"round": "bogus"})

    def test_reg_07_08_09_cmp_result_type_nan_total_and_accum(self):
        # Discrepancy 7: .i1 -> 1, .m1 -> 0xFFFFFFFF
        r_i1, _, _ = eval_panfrost_instruction("ICMP", "s32", [bv32(1), bv32(1)], modifiers={"cmpf": "eq", "result_type": "i1"})
        self.assertEqual(z3.simplify(r_i1).as_long(), 1)
        r_m1, _, _ = eval_panfrost_instruction("ICMP", "s32", [bv32(1), bv32(1)], modifiers={"cmpf": "eq", "result_type": "m1"})
        self.assertEqual(z3.simplify(r_m1).as_long(), 0xFFFFFFFF)

        # Discrepancy 8: FCMP.f32.ne.m1(NaN, 1.0) -> True (0xFFFFFFFF); gtlt -> False (0); total(-0.0, +0.0) -> True
        r_ne, _, _ = eval_panfrost_instruction("FCMP", "f32", [bv32(0x7FC00000), bv32(0x3F800000)], modifiers={"cmpf": "ne", "result_type": "m1"})
        self.assertEqual(z3.simplify(r_ne).as_long(), 0xFFFFFFFF)
        r_gtlt, _, _ = eval_panfrost_instruction("FCMP", "f32", [bv32(0x7FC00000), bv32(0x3F800000)], modifiers={"cmpf": "gtlt", "result_type": "m1"})
        self.assertEqual(z3.simplify(r_gtlt).as_long(), 0x00000000)
        r_tot, _, _ = eval_panfrost_instruction("FCMP", "f32", [bv32(0x80000000), bv32(0x00000000)], modifiers={"cmpf": "total", "result_type": "i1"})
        self.assertEqual(z3.simplify(r_tot).as_long(), 1)

        # Discrepancy 9: ICMP_AND boolean accum before res_type
        r_and, _, _ = eval_panfrost_instruction(
            "ICMP_AND", "u32", [bv32(1), bv32(1), bv32(2)], modifiers={"cmpf": "eq", "result_type": "m1"}
        )
        self.assertEqual(z3.simplify(r_and).as_long(), 0xFFFFFFFF)

    def test_reg_10_v2s8_to_v2s16_and_mkvec_v2i8(self):
        res, _, _ = eval_panfrost_instruction("V2S8_TO_V2S16", "", [bv32(0x00020001)], modifiers={"swz0": "b02"})
        self.assertEqual(z3.simplify(res).as_long(), 0x00020001)

    def test_reg_11_fadd_imm_v2f16(self):
        res, _, _ = eval_panfrost_instruction("FADD_IMM", "v2f16", [bv32(0)], modifiers={"imm": 0x40003C00})
        self.assertEqual(z3.simplify(res).as_long(), 0x40003C00)

    def test_reg_12_iaddc_isubb_u32(self):
        sum_r, _, _ = eval_panfrost_instruction("IADDC", "u32", [bv32(0xFFFFFFFF), bv32(0), bv32(1)])
        self.assertEqual(z3.simplify(sum_r).as_long(), 0)
        carry_r, _, _ = eval_panfrost_instruction("IADDC", "u32", [bv32(0xFFFFFFFF), bv32(0), bv32(1)], modifiers={"return_carry": True})
        self.assertEqual(z3.simplify(carry_r).as_long(), 1)
        borrow_r, _, _ = eval_panfrost_instruction("ISUBB", "u32", [bv32(0), bv32(0), bv32(1)], modifiers={"return_carry": True})
        self.assertEqual(z3.simplify(borrow_r).as_long(), 1)

    def test_reg_13_empty_variant_fp_conv_clamp_and_sat(self):
        f16_neg, _, _ = eval_panfrost_instruction("F16_TO_F32", "", [bv32(0x3C00)], modifiers={"neg0": True})
        self.assertEqual(z3.simplify(f16_neg).as_long(), 0xBF800000)

        f32_to_f16_clamped, _, _ = eval_panfrost_instruction("F32_TO_F16", "", [bv32(0x40000000), bv32(0xBF800000)], modifiers={"clamp": "clamp_0_1"})
        self.assertEqual(z3.simplify(f32_to_f16_clamped).as_long(), 0x00003C00)

        nan_to_s32, _, _ = eval_panfrost_instruction("F32_TO_S32", "", [bv32(0x7FC00000)])
        self.assertEqual(z3.simplify(nan_to_s32).as_long(), 0)

        ovf_to_s32, _, _ = eval_panfrost_instruction("F32_TO_S32", "", [bv32(0x60000000)])
        self.assertEqual(z3.simplify(ovf_to_s32).as_long(), 0x7FFFFFFF)

    def test_reg_14_v2f16_h0_swizzle_and_scalar_f16(self):
        mov_v2, _, _ = eval_panfrost_instruction("MOV", "v2f16", [bv32(0x3C003C00)], modifiers={"swz0": "h0"})
        self.assertEqual(z3.simplify(mov_v2).as_long(), 0x3C003C00)

        rcp_f16, _, _ = eval_panfrost_instruction("FRCP", "f16", [bv32(0x00004000)])  # 1/2.0 = 0.5 (0x3800)
        self.assertEqual(z3.simplify(rcp_f16).as_long(), 0x00003800)

        # Scalar f16 h1 swizzle must replicate (not double-widen to f32 in Stage A)
        f16_to_f32_h1, _, _ = eval_panfrost_instruction("F16_TO_F32", "", [bv32(0x3C000000)], modifiers={"lane0": "h1"})
        self.assertEqual(z3.simplify(f16_to_f32_h1).as_long(), 0x3F800000)

        rcp_f16_h1, _, _ = eval_panfrost_instruction("FRCP", "f16", [bv32(0x40000000)], modifiers={"swz0": "h1"})
        self.assertEqual(z3.simplify(rcp_f16_h1).as_long(), 0x00003800)

        # Default XML swizzles when omitted: V2S8_TO_V2S16 defaults swz0="b02", MKVEC.v2i8 defaults swz2="b01"
        v2s8_def, _, _ = eval_panfrost_instruction("V2S8_TO_V2S16", "", [bv32(0x00020001)])
        self.assertEqual(z3.simplify(v2s8_def).as_long(), 0x00020001)

        mkvec_def, _, _ = eval_panfrost_instruction("MKVEC", "v2i8", [bv32(0x11), bv32(0x22), bv32(0x00004433)])
        self.assertEqual(z3.simplify(mkvec_def).as_long(), 0x44332211)

    def test_reg_minor_gaps(self):
        fmax_pnan, _, _ = eval_panfrost_instruction("FMAX", "f32", [bv32(0x7FC00000), bv32(0x3F800000)], modifiers={"propagate_nan": True})
        s = z3.Solver(); s.add(z3.Not(z3.fpIsNaN(z3.fpBVToFP(fmax_pnan, z3.FPSort(8, 24))))); self.assertEqual(s.check(), z3.unsat)

        rcp_sub, _, _ = eval_panfrost_instruction("FRCP", "f32", [bv32(0x00000001)])  # subnormal flushed to +0.0 -> +Inf
        self.assertEqual(z3.simplify(rcp_sub).as_long(), 0x7F800000)


if __name__ == "__main__":
    unittest.main()

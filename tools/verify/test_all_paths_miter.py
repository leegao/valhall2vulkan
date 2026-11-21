#!/usr/bin/env python3
"""
Unit & All-Paths Equivalence Miter Tests (Task 3: Psi_pan vs Eval_Z3(T_lowering)).
"""

import os
import sys
import unittest

import z3

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.verify import (
    LoweringSemanticsExtractor,
    check_instruction_miter,
    eval_ssa_module_z3,
)


class TestAllPathsMiter(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ext = LoweringSemanticsExtractor(repo_root=REPO_ROOT)

    def test_all_627_xml_specs_evaluate_in_z3(self) -> None:
        results = self.ext.extract_all_xml_semantics()
        self.assertEqual(len(results), 627)
        x_srcs = [z3.BitVec(f"r{i}", 32) for i in range(6)]
        dst_old = z3.BitVec("dst_old", 32)
        for r in results:
            ev = eval_ssa_module_z3(
                r["ssa_module"], raw_srcs=x_srcs, dst_old=dst_old, dest_reg=0
            )
            self.assertIsNotNone(ev.final_state)

    def test_sound_instructions_prove_unsat(self) -> None:
        sound_cases = [
            ("IADD", "u32", {}),
            ("IADD", "u64", {}),
            ("HADD", "s32", {}),
            ("ISUB", "u32", {}),
            ("IMUL", "u32", {}),
            ("AND", "i32", {}),
            ("OR", "i32", {}),
            ("XOR", "i32", {}),
            ("FADD", "f32", {}),
            ("FMUL", "f32", {}),
            ("FMA", "f32", {}),
            ("FCMP", "f32", {"cmpf": "ne", "result_type": "i1"}),
            ("ICMP", "u32", {"cmpf": "lt", "result_type": "m1"}),
            ("MOV", "i32", {}),
        ]
        for mnem, var, mods in sound_cases:
            mr = check_instruction_miter(self.ext, mnem, var, modifiers=mods, timeout_ms=4000)
            self.assertEqual(
                mr.status,
                "UNSAT",
                msg=f"Expected UNSAT for {mnem}.{var} {mods}, got {mr.status}: {mr.counterexample}",
            )

    def test_known_unpatched_defects_produce_sat_counterexamples(self) -> None:
        buggy_cases = [
            ("IADD", "v2u16", {}),
            ("ISUB", "v4u8", {}),
            ("IMUL", "v2u16", {}),
            ("HADD", "v2u16", {}),
            ("IADD", "s32", {"saturate": True}),
            ("MKVEC", "v2i8", {}),
            ("CUBEFACE2", "", {}),
            ("MUX", "u32", {"muxf": "int_zero"}),
            ("IDPADD", "v4s8", {}),
            ("FMIN", "f32", {"nan_mode": "nan_suppress"}),  # Psi_pan deterministic -0.0 < +0.0 vs ssa_z3_eval fpMin
        ]
        for mnem, var, mods in buggy_cases:
            mr = check_instruction_miter(self.ext, mnem, var, modifiers=mods, timeout_ms=4000)
            self.assertEqual(
                mr.status,
                "SAT",
                msg=f"Expected SAT counterexample for {mnem}.{var} {mods}, got {mr.status}",
            )
            self.assertIsNotNone(mr.counterexample)
            self.assertIn("COUNTEREXAMPLE", mr.counterexample)



    def test_side_effect_state_equivalence_checks(self) -> None:
        # Sound side-effect specs -> UNSAT
        for mnem, var, mods in [
            ("BRANCH", "", {}),
            ("BRANCH", "", {"combine": "h0"}),
            ("BRANCH", "", {"combine": "h1"}),
            ("BRANCH", "", {"combine": "and"}),
            ("BRANCH", "", {"combine": "lowbits"}),
            ("BARRIER", "", {}),
            ("STORE", "i32", {}),
        ]:
            mr = check_instruction_miter(self.ext, mnem, var, modifiers=mods, timeout_ms=4000)
            self.assertEqual(
                mr.status,
                "UNSAT",
                msg=f"Expected UNSAT for {mnem}.{var} {mods}, got {mr.status}: {mr.counterexample}",
            )

        # Known side-effect divergences -> SAT
        for mnem, var, mods in [
            ("JUMP", "", {}),
            ("DISCARD", "", {}),
        ]:
            mr = check_instruction_miter(self.ext, mnem, var, modifiers=mods, timeout_ms=4000)
            self.assertEqual(
                mr.status,
                "SAT",
                msg=f"Expected SAT for {mnem}.{var} {mods}, got {mr.status}",
            )
            self.assertIsNotNone(mr.counterexample)

if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""
Unit & Integration Tests for Automated Valhall Lowering & Host Handler Extractor.
"""

import os
import shutil
import sys
import tempfile
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.verify import (
    InstLoweringContext,
    LoweringLogger,
    LoweringSemanticsExtractor,
    SourceOperandDesc,
)


class TestLoweringExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.logger = LoweringLogger()
        cls.ext = LoweringSemanticsExtractor(repo_root=REPO_ROOT, logger=cls.logger)

    def test_cpp_handler_and_operand_completeness(self) -> None:
        reg = self.ext.cpp_registry
        self.assertEqual(len(reg.operands_funcs), 22)
        self.assertIn("fetch_src_int32", reg.operands_funcs)
        self.assertIn("store_dest_f32", reg.operands_funcs)
        self.assertTrue(set(self.ext.host_functions.keys()).issubset(set(reg.abi_funcs.keys())))
        self.assertIn("handle_int_idpadd", reg.global_handlers)
        self.assertIn("emit_op", reg.ctx_funcs)

    def test_all_627_xml_specs_extract_cleanly(self) -> None:
        results = self.ext.extract_all_xml_semantics()
        self.assertEqual(len(results), 627)
        mnems = {r["mnemonic"] for r in results}
        self.assertEqual(len(mnems), 201)
        rules_used = {r["rule_index"] for r in results}
        self.assertGreaterEqual(len(rules_used), 200)

        # Verify FADD_LSCALE.f32 (spec 101) arity mismatch warning is surfaced
        lscale = [r for r in results if r["mnemonic"] == "FADD_LSCALE"]
        self.assertEqual(len(lscale), 1)
        self.assertTrue(any("Arity mismatch" in w for w in lscale[0]["warnings"]))

    def test_direct_ssa_op_and_scalar_bleed_capture(self) -> None:
        ins = InstLoweringContext(
            mnemonic="IADD",
            variant="v2u16",
            srcs=[SourceOperandDesc(reg_num=0), SourceOperandDesc(reg_num=1)],
            has_dest=True,
            dest_reg=2,
        )
        ssa, tr = self.ext.lower_instruction(ins)
        self.assertEqual(tr.rule_kind, "DirectSsaOp")
        self.assertIn("IAdd", tr.emitted_ops)
        self.assertIn(2, ssa.reg_writes)

    def test_dynamic_lookup_rule(self) -> None:
        ins_sup = InstLoweringContext(
            mnemonic="FMIN",
            variant="f32",
            srcs=[SourceOperandDesc(reg_num=0), SourceOperandDesc(reg_num=1)],
            modifiers={"nan_mode": "nan_suppress"},
            has_dest=True,
            dest_reg=0,
        )
        _, tr_sup = self.ext.lower_instruction(ins_sup)
        self.assertEqual(tr_sup.rule_kind, "DynamicSsaOp")
        self.assertIn("NMin", tr_sup.emitted_ops)

        ins_prop = InstLoweringContext(
            mnemonic="FMIN",
            variant="f32",
            srcs=[SourceOperandDesc(reg_num=0), SourceOperandDesc(reg_num=1)],
            modifiers={"nan_mode": "nan_propagate"},
            has_dest=True,
            dest_reg=0,
        )
        _, tr_prop = self.ext.lower_instruction(ins_prop)
        self.assertIn("FMin", tr_prop.emitted_ops)

    def test_inline_ops_fcmp_and_mkvec(self) -> None:
        fcmp_ins = InstLoweringContext(
            mnemonic="FCMP",
            variant="f32",
            srcs=[SourceOperandDesc(reg_num=0), SourceOperandDesc(reg_num=1)],
            modifiers={"cmpf": "ne", "result_type": "i1"},
            has_dest=True,
            dest_reg=3,
        )
        ssa_cmp, tr_cmp = self.ext.lower_instruction(fcmp_ins)
        self.assertEqual(tr_cmp.rule_kind, "InlineOps")
        self.assertIn("FCmpUnordNe", tr_cmp.emitted_ops)
        self.assertIn("Select", tr_cmp.emitted_ops)
        self.assertIn(3, ssa_cmp.reg_writes)

        mkvec_ins = InstLoweringContext(
            mnemonic="MKVEC",
            variant="v2i8",
            srcs=[SourceOperandDesc(reg_num=0), SourceOperandDesc(reg_num=1), SourceOperandDesc(reg_num=2)],
            has_dest=True,
            dest_reg=4,
        )
        _, tr_mk = self.ext.lower_instruction(mkvec_ins)
        self.assertIn("Bfi", tr_mk.emitted_ops)

    def test_handler_rule_idpadd(self) -> None:
        ins = InstLoweringContext(
            mnemonic="IDPADD",
            variant="v4s8",
            srcs=[SourceOperandDesc(reg_num=0), SourceOperandDesc(reg_num=1)],
            has_dest=True,
            dest_reg=5,
        )
        ssa, tr = self.ext.lower_instruction(ins)
        self.assertEqual(tr.rule_kind, "Handler")
        self.assertEqual(tr.emitted_ops.count("IMul"), 4)
        self.assertEqual(tr.emitted_ops.count("IAdd"), 4)
        self.assertIn(5, ssa.reg_writes)

    def test_continuous_cpp_extraction_reflects_source_edits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_ops = os.path.join(tmpdir, "valhall_operands.cpp")
            shutil.copy2(os.path.join(REPO_ROOT, "src/valhall_operands.cpp"), tmp_ops)
            with open(tmp_ops, "r", encoding="utf-8") as f:
                src = f.read()
            patched = src.replace('{"widen", "swz", "lanes"}', '{"widen", "swz", "lanes", "lane"}', 1)
            self.assertNotEqual(src, patched)
            with open(tmp_ops, "w", encoding="utf-8") as f:
                f.write(patched)

            ins = InstLoweringContext(
                mnemonic="S16_TO_S32",
                variant="",
                srcs=[SourceOperandDesc(reg_num=0)],
                modifiers={"lane0": "h1"},
                has_dest=True,
                dest_reg=1,
            )
            _, tr_orig = self.ext.lower_instruction(ins)
            self.assertNotIn("BfeS", tr_orig.emitted_ops)

            ext_patched = LoweringSemanticsExtractor(repo_root=REPO_ROOT, operands_path=tmp_ops)
            _, tr_patched = ext_patched.lower_instruction(ins)
            self.assertIn("BfeS", tr_patched.emitted_ops)

    def test_lowering_trace_logging_summary(self) -> None:
        ins = InstLoweringContext(
            mnemonic="FADD_LSCALE",
            variant="f32",
            srcs=[SourceOperandDesc(reg_num=0), SourceOperandDesc(reg_num=1)],
            has_dest=True,
            dest_reg=0,
        )
        _, tr = self.ext.lower_instruction(ins)
        summary = tr.format_summary()
        self.assertIn("[T_lowering] FADD_LSCALE.f32", summary)
        self.assertIn("WARN: Arity mismatch", summary)
        self.assertIn("Emitted SsaOps", summary)


if __name__ == "__main__":
    unittest.main()

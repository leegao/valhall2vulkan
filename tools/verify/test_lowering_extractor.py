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
        self.assertEqual(len(reg.operands_funcs), 15)
        self.assertIn("fetch_src_int32", reg.operands_funcs)
        self.assertIn("store_dest_f32", reg.operands_funcs)
        abi_keys = set(self.ext.host_functions.keys())
        self.assertTrue(abi_keys.issubset(set(reg.abi_funcs.keys())))
        self.assertIn("handle_int_idpadd", reg.global_handlers)
        self.assertIn("emit_op", reg.ctx_funcs)

    def test_all_627_xml_specs_extract_cleanly(self) -> None:
        results = self.ext.extract_all_xml_semantics()
        self.assertEqual(len(results), 627)
        mnems = {r["mnemonic"] for r in results}
        self.assertEqual(len(mnems), 201)
        rules_used = {r["rule_index"] for r in results}
        self.assertGreaterEqual(len(rules_used), 200)

        # Verify FADD_LSCALE.f32 (spec 101) lowers cleanly with 2 f32 inputs and no arity mismatch
        lscale = [r for r in results if r["mnemonic"] == "FADD_LSCALE"]
        self.assertEqual(len(lscale), 1)
        self.assertFalse(any("Arity mismatch" in w for w in lscale[0]["warnings"]))

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
            modifiers={"sem": "nan_suppress"},
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
            modifiers={"sem": "nan_propagate"},
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
        import json
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_ops = os.path.join(tmpdir, "valhall_lowering.json")
            with open(os.path.join(REPO_ROOT, "specs/lowering/valhall_lowering.json"), "r", encoding="utf-8") as f:
                spec = json.load(f)
            fetch_i32 = [e for e in spec["operands"] if e["name"] == "fetch_src_int32"][0]
            fetch_i32["inline_ops"].insert(
                1,
                'if (@src_swz_eq(%src_idx, "h1")) { %val = BitNot(%val); }',
            )
            with open(tmp_ops, "w", encoding="utf-8") as f:
                json.dump(spec, f)

            ins = InstLoweringContext(
                mnemonic="S16_TO_S32",
                variant="",
                srcs=[SourceOperandDesc(reg_num=0, swizzle_mod="h1")],
                has_dest=True,
                dest_reg=1,
            )
            _, tr_orig = self.ext.lower_instruction(ins)
            self.assertNotIn("BitNot", tr_orig.emitted_ops)

            ext_patched = LoweringSemanticsExtractor(repo_root=REPO_ROOT, operands_path=tmp_ops)
            _, tr_patched = ext_patched.lower_instruction(ins)
            self.assertIn("BitNot", tr_patched.emitted_ops)

    def test_operands_spec_codegen_sync(self) -> None:
        import subprocess
        import sys
        res = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "tools/gen_valhall_tables.py"), "--check-operands"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, msg=res.stdout + res.stderr)

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
        self.assertNotIn("WARN: Arity mismatch", summary)
        self.assertIn("FAdd", summary)
        self.assertIn("Emitted SsaOps", summary)



    def test_single_quote_strings_and_unified_abi(self) -> None:
        from tools.dsl_parser import Literal, parse_inline_ops
        prog = parse_inline_ops(["if (@src_swz_eq(0, 'h1')) { return @u32(1); }"])
        self.assertEqual(len(prog.stmts), 1)
        # Verify unified ABI: no operands_host, all host_functions in abi_funcs
        for k, v in self.ext.host_functions.items():
            self.assertNotIn("namespace", v)
            self.assertNotIn("takes_sb", v)
            self.assertNotIn("takes_ins", v)
            self.assertIn(k, self.ext.cpp_registry.abi_funcs)
        # Verify migrated inline_ops host functions work in symbolic evaluation
        for fn in (
            "src0_swz_h1", "raw0", "fp_zero", "split64_lo", "split64_hi",
            "emit_pack_u64", "has_clamp_0_1", "has_clamp_m1_1", "has_clamp_0_inf",
            "src_swz_is_b_rep", "src_swz_b_rep_off", "dest_mod_is_byte", "dest_mod_byte_off",
        ):
            self.assertEqual(self.ext.host_functions[fn].get("kind"), "inline_ops")
            self.assertIn(fn, self.ext.cpp_registry.abi_funcs)


    def test_when_and_arch_range_rule_splitting(self) -> None:
        # Vertex istream -> StoreBuiltinOutput
        ins_ist = InstLoweringContext(
            mnemonic="ST_CHECKED",
            variant="",
            stage="Vertex",
            modifiers={"access": "istream", "memory_width": "v4"},
            srcs=[SourceOperandDesc(reg_num=0)],
            sr_read_present=True,
            sr_read_base=8,
            sr_read_count=4,
        )
        _, tr_ist = self.ext.lower_instruction(ins_ist)
        self.assertIn("StoreBuiltinOutput", tr_ist.emitted_ops)

        # Vertex estream -> StoreVertVarying
        ins_est = InstLoweringContext(
            mnemonic="ST_CHECKED",
            variant="",
            stage="Vertex",
            modifiers={"access": "estream", "memory_width": "v4"},
            srcs=[SourceOperandDesc(reg_num=0)],
            sr_read_present=True,
            sr_read_base=8,
            sr_read_count=4,
        )
        _, tr_est = self.ext.lower_instruction(ins_est)
        self.assertIn("StoreVertVarying", tr_est.emitted_ops)

        # Compute fallback -> PsbStore
        ins_cmp = InstLoweringContext(
            mnemonic="ST_CHECKED",
            variant="",
            stage="Compute",
            modifiers={"access": "none", "memory_width": "v4"},
            srcs=[SourceOperandDesc(reg_num=0)],
            sr_read_present=True,
            sr_read_base=8,
            sr_read_count=2,
        )
        _, tr_cmp = self.ext.lower_instruction(ins_cmp)
        self.assertIn("PsbStore", tr_cmp.emitted_ops)

        # Verify arch_range on all split ST_CHECKED rules
        st_rules = [r for r in self.ext.rules if r.mnemonic == "ST_CHECKED"]
        self.assertEqual(len(st_rules), 3)
        for r in st_rules:
            self.assertEqual(r.arch_range, (9, 14))
        self.assertIsNotNone(self.ext.match_rule("ST_CHECKED", "", arch=11))
        self.assertIsNone(self.ext.match_rule("ST_CHECKED", "", arch=8))


    def test_v29_regenerated_ast_constructs(self) -> None:
        from tools.verify.clang_ast_evaluator import (
            CppAssign,
            CppBinary,
            CppCall,
            CppExprStmt,
            CppFunctionDef,
            CppIdent,
            CppIf,
            CppInitList,
            CppLiteral,
            CppMember,
            CppParam,
            CppReturn,
            CppScopedIdent,
            CppTernary,
            CppVarDecl,
            SymbolicSsaModule,
            SymbolicSsaNode,
        )

        # Simulate regenerated abi::extract AST accessing ctx.res.ssa.nodes.size(),
        # ctx.res.ssa.get(vec).type, rule->op_class, and list.push_back / std::string::npos.
        ssa = SymbolicSsaModule()
        tup_id = ssa.emit(SymbolicSsaNode(0, "Unpack64", "TupleU32x2", args=[]))
        ins = InstLoweringContext(
            mnemonic="TEX_SINGLE",
            modifiers={"write_mask": "gb"},
            rule_op_class="TexSample",
        )

        ctx_res_ssa = CppMember(CppMember(CppIdent("ctx"), "res"), "ssa")
        nodes_size = CppCall(
            callee=CppMember(CppMember(ctx_res_ssa, "nodes"), "size"),
            args=[],
            qualified_callee="std::vector<valhall::compiler::SsaNode>::size",
        )
        get_type = CppMember(
            CppCall(
                callee=CppMember(ctx_res_ssa, "get"),
                args=[CppIdent("vec")],
                qualified_callee="valhall::compiler::ValhallSsaModule::get",
            ),
            "type",
        )
        fn = CppFunctionDef(
            name="_test_v29_extract_ast",
            namespace="abi",
            ret_type="SsaId",
            params=[
                CppParam("Pass1Context&", "ctx"),
                CppParam("uint32_t", "blk"),
                CppParam("const DecodedValhallInst&", "ins"),
                CppParam("const tables::LoweringRuleSpec*", "rule"),
                CppParam("ValType", "vt"),
                CppParam("SsaId", "vec"),
                CppParam("uint32_t", "idx"),
            ],
            body=[
                CppVarDecl("SsaNode", "ex", is_struct_zero=True),
                CppVarDecl(
                    "ValType",
                    "src_ty",
                    CppTernary(
                        cond=CppBinary("<", CppIdent("vec"), nodes_size),
                        then_expr=get_type,
                        else_expr=CppScopedIdent(["ValType", "Void"]),
                    ),
                ),
                CppAssign(
                    CppMember(CppIdent("ex"), "op"),
                    "=",
                    CppTernary(
                        cond=CppBinary("==", CppIdent("src_ty"), CppScopedIdent(["ValType", "TupleU32x2"])),
                        then_expr=CppScopedIdent(["SsaOp", "TupleExtract"]),
                        else_expr=CppScopedIdent(["SsaOp", "VecExtract"]),
                    ),
                ),
                CppAssign(CppMember(CppIdent("ex"), "type"), "=", CppIdent("vt")),
                CppAssign(CppMember(CppIdent("ex"), "imm_u32"), "=", CppIdent("idx")),
                CppExprStmt(
                    CppCall(
                        callee=CppMember(CppMember(CppIdent("ex"), "args"), "push_back"),
                        args=[CppIdent("vec")],
                        qualified_callee="std::vector<unsigned int>::push_back",
                    )
                ),
                CppVarDecl("std::string", "sout_name"),
                CppIf(
                    cond=CppBinary(
                        "!=",
                        CppCall(CppMember(CppIdent("sout_name"), "find"), [CppLiteral("string", "PointSize")]),
                        CppScopedIdent(["std", "string", "npos"]),
                    ),
                    then_body=[CppReturn(CppLiteral("int", 999))],
                ),
                CppVarDecl(
                    "uint32_t",
                    "clamped",
                    CppCall(
                        callee=CppScopedIdent(["std", "min"]),
                        args=[CppLiteral("int", 12), CppLiteral("int", 8)],
                        qualified_callee="std::min<unsigned int>",
                    ),
                ),
                CppAssign(CppMember(CppIdent("ex"), "imm_u32_2"), "=", CppIdent("clamped")),
                CppReturn(
                    CppCall(
                        callee=CppMember(CppMember(CppIdent("ctx"), "sb"), "emit"),
                        args=[CppIdent("blk"), CppIdent("ex")],
                        qualified_callee="valhall::compiler::SsaBuilder::emit",
                    )
                ),
            ],
        )
        self.ext.cpp_registry.abi_funcs[fn.name] = fn
        try:
            out_id = self.ext.cpp_eval.call_abi_fn(fn.name, ssa, ins, ["U32", tup_id, 1])
            node = ssa.nodes[out_id]
            self.assertEqual(node.op, "TupleExtract")
            self.assertEqual(node.args, [tup_id])
            self.assertEqual(node.imm_u32, 1)
            self.assertEqual(node.imm_u32_2, 8)
        finally:
            self.ext.cpp_registry.abi_funcs.pop(fn.name, None)

if __name__ == "__main__":
    unittest.main()

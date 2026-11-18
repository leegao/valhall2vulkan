#!/usr/bin/env python3
"""Unit tests for the static Clang-21 AST dumper and build-time JSON output."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = REPO_ROOT / "tools" / "verify" / "generated" / "valhall_host_handlers.json"


def _collect_kinds(node, out):
    if isinstance(node, dict):
        k = node.get("kind")
        if k:
            out.add(k)
        for v in node.values():
            _collect_kinds(v, out)
    elif isinstance(node, list):
        for v in node:
            _collect_kinds(v, out)


class TestClangAstDumper(unittest.TestCase):
    def test_generated_valhall_host_handlers_json(self):
        json_path = Path(os.environ.get("VALHALL_HOST_HANDLERS_AST_JSON", str(DEFAULT_JSON)))
        self.assertTrue(json_path.is_file(), f"Missing generated AST JSON: {json_path}")

        data = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(data.get("schema_version"), 1)
        self.assertEqual(data.get("unhandled_count"), 0, "Expected zero unhandled AST nodes")
        self.assertGreaterEqual(data.get("function_count", 0), 88)

        funcs = data["functions"]
        by_qname = {f["qualified_name"]: f for f in funcs}

        expected_qnames = [
            "valhall::compiler::Pass1Context::emit_op",
            "valhall::compiler::Pass1Context::emit_stage_preamble",
            "valhall::compiler::Pass1Context::eval_float_cmp",
            "valhall::compiler::abi::has_src0",
            "valhall::compiler::abi::atomic_rmw",
            "valhall::compiler::abi::store_frag_color",
            "valhall::compiler::handle_int_idpadd",
        ]
        for qname in expected_qnames:
            self.assertIn(qname, by_qname, f"Missing function {qname} in AST JSON")
            fn = by_qname[qname]
            self.assertEqual(fn["source_file"], "src/valhall_pass1_handlers.cpp")
            self.assertGreaterEqual(fn["end_line"], fn["start_line"])
            self.assertTrue(len(fn["body"]) > 0, f"Empty body for {qname}")

        self.assertEqual(by_qname["valhall::compiler::Pass1Context::emit_op"]["short_namespace"], "Pass1Context")
        self.assertEqual(by_qname["valhall::compiler::abi::has_src0"]["short_namespace"], "abi")
        self.assertEqual(by_qname["valhall::compiler::handle_int_idpadd"]["short_namespace"], "")

        # Verify resolved canonical types on parameters and return type
        emit_op = by_qname["valhall::compiler::Pass1Context::emit_op"]
        self.assertEqual(emit_op["ret_type"], "SsaId")
        self.assertEqual(emit_op["canonical_ret_type"], "unsigned int")
        self.assertEqual(emit_op["params"][0]["name"], "blk")

    def test_generic_cpp17_constructs_strict(self):
        dumper_bin = os.environ.get("CLANG_AST_DUMPER_BIN")
        if not dumper_bin or not Path(dumper_bin).is_file():
            self.skipTest("CLANG_AST_DUMPER_BIN not set")

        res_dir = os.environ.get(
            "CLANG_RESOURCE_DIR",
            str(REPO_ROOT / "third_party" / "prebuilts" / "llvm-21" / "lib" / "clang" / "21"),
        )
        cpp_src = r"""
#include <algorithm>
#include <cstdint>
#include <vector>

namespace test_ns {
enum class Mode { A, B, C };

uint64_t generic_features(const std::vector<uint32_t>& xs, Mode m) {
    uint64_t acc = 0, bonus = 1;
    if (auto sz = xs.size(); sz > 0) {
        acc += static_cast<uint64_t>(sz);
    }
    for (const auto& v : xs) {
        if (v == 0) continue;
        acc += std::max<uint32_t>(v, 1u);
    }
    uint32_t i = 0;
    while (i < 3) {
        if (i == 2) break;
        ++i;
    }
    do {
        --i;
    } while (i > 0);
    switch (m) {
        case Mode::A:
        case Mode::B:
            acc += 10;
            break;
        default:
            acc += 20;
            break;
    }
    auto fn = [&](uint32_t k) -> uint64_t { return acc + k + sizeof(uint64_t); };
    return (acc > 0) ? fn(static_cast<uint32_t>(bonus)) : 0ULL;
}
}
"""
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src_file = td_path / "generic_test.cpp"
            out_json = td_path / "out.json"
            src_file.write_text(cpp_src, encoding="utf-8")

            cmd = [
                dumper_bin,
                str(src_file),
                "-o", str(out_json),
                "--strict",
                "--",
                "-std=c++17",
                "-resource-dir", res_dir,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertEqual(res.returncode, 0, f"clang_ast_dumper failed:\n{res.stderr}")

            data = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(data["unhandled_count"], 0)
            self.assertEqual(data["function_count"], 1)

            kinds = set()
            _collect_kinds(data["functions"][0]["body"], kinds)
            for expected_kind in (
                "multi_var_decl",
                "if",
                "range_for",
                "continue",
                "while",
                "break",
                "do_while",
                "switch",
                "cast",
                "lambda",
                "sizeof",
                "ternary",
                "call",
                "scoped_ident",
            ):
                self.assertIn(expected_kind, kinds, f"Missing AST kind '{expected_kind}' in {sorted(kinds)}")



    def test_verifier_z3_attributes_extracted(self) -> None:
        json_path = Path(os.environ.get("VALHALL_HOST_HANDLERS_AST_JSON", str(DEFAULT_JSON)))
        data = json.loads(json_path.read_text(encoding="utf-8"))
        vattrs = data.get("verifier_attrs", {})
        self.assertGreaterEqual(len(vattrs), 70)
        self.assertIn("valhall::variant_to_string", vattrs)
        self.assertIn("valhall::compiler::abi::f32", vattrs)
        self.assertIn("valhall::compiler::abi::set_branch_cond", vattrs)
        self.assertIn("valhall::compiler::abi::src_oob", vattrs)
        self.assertIn("valhall::compiler::SsaBuilder::emit_bitcast", vattrs)
        self.assertEqual(
            vattrs["valhall::compiler::SsaBuilder::emit_bitcast"],
            {"receiver": "SymbolicSsaModule", "expr": "this.emit_bitcast(args[1], args[2])"},
        )
        self.assertIn("valhall::compiler::SsaBuilder::current_token", vattrs)
        self.assertIn("valhall::compiler::SsaBuilder::set_token", vattrs)

        dumper_bin = os.environ.get("CLANG_AST_DUMPER_BIN", "")
        if not dumper_bin or not Path(dumper_bin).is_file():
            self.skipTest("CLANG_AST_DUMPER_BIN not set")
        res_dir = os.environ.get(
            "CLANG_RESOURCE_DIR",
            str(REPO_ROOT / "third_party" / "prebuilts" / "llvm-21" / "lib" / "clang" / "21"),
        )

        with tempfile.TemporaryDirectory() as tmp:
            hdr_path = os.path.join(tmp, "custom_hdr.h")
            cpp_path = os.path.join(tmp, "custom_test.cpp")
            out_path = os.path.join(tmp, "custom_out.json")
            with open(hdr_path, "w", encoding="utf-8") as f:
                f.write(
                    'struct CustomBuilder {\n'
                    '    [[verifier::z3("SymbolicSsaModule", "this.emit_const_u32(args[0])")]]\n'
                    '    int custom_emit(int x);\n'
                    '};\n'
                )
            with open(cpp_path, "w", encoding="utf-8") as f:
                f.write(
                    '#include "custom_hdr.h"\n'
                    'int call_it(CustomBuilder& b) { return b.custom_emit(42); }\n'
                )
            cmd = [
                dumper_bin,
                cpp_path,
                "-o", out_path,
                "--root", tmp,
                "--strict",
                "--",
                "-std=c++17",
                "-Wno-unknown-attributes",
                f"-I{tmp}",
            ] + (["-resource-dir", res_dir] if Path(res_dir).is_dir() else [])
            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            with open(out_path, "r", encoding="utf-8") as f:
                cdata = json.load(f)
            cvattrs = cdata.get("verifier_attrs", {})
            self.assertIn("CustomBuilder::custom_emit", cvattrs)
            self.assertEqual(
                cvattrs["CustomBuilder::custom_emit"],
                {"receiver": "SymbolicSsaModule", "expr": "this.emit_const_u32(args[0])"},
            )

if __name__ == "__main__":
    unittest.main()

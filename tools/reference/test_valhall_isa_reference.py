"""Regression & structural verification tests for tools/reference/ Valhall ISA generator."""

from __future__ import annotations

import re
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from tools.reference import csf_abi_map
from tools.reference import fmt_helpers
from tools.reference import gen_valhall_spec_tex as gen
from tools.reference import isa_semantics
from tools.reference import v3_enrichments as v3e


class TestValhallIsaReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.xml_root = ET.parse(gen.XML_PATH).getroot()
        cls.enums, cls.enum_sample_ins = gen.parse_enums(cls.xml_root)
        cls.mnem_specs = gen.parse_instructions(cls.xml_root)

    def test_modular_refactor_boundaries(self) -> None:
        """Semantics live in isa_semantics, formatting in fmt_helpers, Execution Context & GLSL in csf_abi_map."""
        for attr in ("CAT_ORDER", "ISA_SEMANTICS", "INST_EXTRAS", "_SH_META", "_LOP_META", "KEY_ENUMS", "SECONDARY_ENUM_WALKTHROUGHS"):
            self.assertTrue(hasattr(isa_semantics, attr), f"isa_semantics missing {attr}")
        for attr in ("tex_esc", "md_inline_to_tex", "fmt_colored_syntax", "fmt_asm_line", "render_template", "split_notes", "fmt_notes_tex"):
            self.assertTrue(hasattr(fmt_helpers, attr), f"fmt_helpers missing {attr}")
        for attr in ("equiv_glsl_tex", "execution_context_bullets", "render_execution_context_tex", "csf_abi_for_enum"):
            self.assertTrue(hasattr(csf_abi_map, attr), f"csf_abi_map missing {attr}")
        self.assertIn("%%%%%~~~$MNEM_TEX~~~%%%%%", gen.INSTRUCTION_ENTRY_TEMPLATE)
        self.assertIn("%%%%%~~~$EQUIV_GLSL~~~%%%%%", gen.INSTRUCTION_ENTRY_TEMPLATE)
        self.assertIn("%%%%%~~~$EXEC_CONTEXT~~~%%%%%", gen.INSTRUCTION_ENTRY_TEMPLATE)
        self.assertNotIn("Exact Corresponding Valhall", gen.INSTRUCTION_ENTRY_TEMPLATE)
        self.assertIn(r"\clearpage", gen.INSTRUCTION_ENTRY_TEMPLATE)

    def test_all_mnemonics_have_self_contained_behavior_examples_glsl_and_ctx(self) -> None:
        """Every canonical mnemonic has self-contained behavior (no vague Per-lane), >=2 examples, Equiv GLSL, and bulleted Execution Context."""
        self.assertEqual(isa_semantics.missing_semantics(self.mnem_specs), [])
        for mn in sorted(self.mnem_specs):
            beh, _note = isa_semantics.behavior_of(mn)
            self.assertNotIn("Per-lane", beh, f"Vague 'Per-lane' remaining in {mn} behavior")
            self.assertIn(mn, isa_semantics.INST_EXTRAS, f"Missing INST_EXTRAS for {mn}")
            spv, _glsl, examples = isa_semantics.extras_of(mn)
            self.assertTrue(spv, f"Empty SPIR-V op for {mn}")
            self.assertGreaterEqual(len(examples), 2, f"Expected >= 2 worked examples for {mn}")
            equiv_glsl = csf_abi_map.equiv_glsl_tex(mn)
            self.assertTrue(equiv_glsl.strip(), f"Empty Equivalent GLSL for {mn}")
            bullets = csf_abi_map.execution_context_bullets(mn, isa_semantics.category_of(mn))
            self.assertGreaterEqual(len(bullets), 1, f"Empty Execution Context bullets for {mn}")

    def test_all_xml_fields_have_documentation(self) -> None:
        """Every non-reserved <field> across all 627 XML instruction specs must be documented."""
        missing = set()
        for ins in self.xml_root.findall("instruction"):
            for ch in ins:
                if ch.tag == "field":
                    fn = ch.attrib["name"]
                    if v3e.get_field_doc(fn) is None:
                        missing.add(fn)
        self.assertEqual(sorted(missing), [], f"Undocumented XML fields: {sorted(missing)}")

    def test_section_2_key_and_secondary_enum_walkthroughs(self) -> None:
        """All 50 Section 2.X enums have >= 2 Combined Walkthroughs; all 78 Section 2.51.X enums have worked examples."""
        documented = set()
        for ename, _title, _desc, walkthroughs in isa_semantics.KEY_ENUMS:
            documented.add(ename)
            self.assertGreaterEqual(len(walkthroughs), 2, f"Section 2.X enum {ename} needs >= 2 walkthroughs")

        rem_enums = sorted(e for e in self.enums if e.endswith("_m") and e not in documented)
        self.assertGreaterEqual(len(rem_enums), 70)
        for ename in rem_enums:
            walks = isa_semantics.secondary_enum_walkthroughs(ename)
            self.assertGreaterEqual(len(walks), 1, f"Section 2.51.X enum {ename} missing worked example")

    def test_notes_bullet_formatting(self) -> None:
        """Single-item notes stay inline; multi-item notes render as a tightitemize bullet list."""
        single = fmt_helpers.fmt_notes_tex("Exact 32-bit popcount.")
        self.assertIn(r"\textbf{Note:}", single)
        self.assertNotIn(r"\begin{tightitemize}", single)

        multi = fmt_helpers.fmt_notes_tex(
            "64-bit `IADD.sat` is NOT supported (`kraid/ops.rs:2281`). "
            "`v4s8`/`v4u8` only exist on arch <= v10. (`kraid/ops.rs:2250`, `hw_tests.rs:1848`)"
        )
        self.assertIn(r"\textbf{Notes:}", multi)
        self.assertIn(r"\begin{tightitemize}", multi)
        self.assertEqual(multi.count(r"\item "), 3)

    def test_generated_tex_invariants_and_backlinks(self) -> None:
        """End-to-end check: anchors, Equiv GLSL, bulleted Execution Context, Section 1.4 ABI & traces, backlinks, A.2 table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_tex = Path(tmpdir) / "valhall-isa-reference.tex"
            stats = gen.generate(out_tex=out_tex)
            tex = out_tex.read_text()

        n_mns = len(self.mnem_specs)
        self.assertEqual(stats["mnemonics"], n_mns)
        self.assertEqual(re.findall(r"%%%%%~~~\$[A-Z0-9_]+~~~%%%%%", tex), [])
        self.assertNotIn("Uncommon Fields Explained", tex)
        self.assertNotIn(r"\textbf{Inducing GLSL:}", tex)
        self.assertNotIn(r"\textbf{Exact Corresponding Valhall Instruction(s):}", tex)
        self.assertNotIn(r"\textbf{CSF ABI / Descriptor Sideband Consumers:}", tex)
        self.assertEqual(tex.count(r"\textbf{Fields Explained:}"), n_mns)
        self.assertEqual(tex.count(r"\textbf{Equivalent GLSL (Operational Semantics \& Flag Variants):}"), n_mns)
        self.assertGreaterEqual(tex.count(r"\textbf{Execution Context:}"), n_mns)

        # CSF ABI is Section 1.4 inside Chapter 1 (before Chapter 2), NOT Chapter 4
        ch1_pos = tex.index(r"\chapter{Architecture Overview}")
        sec14_pos = tex.index(r"\label{sec:csf-abi}")
        ch2_pos = tex.index(r"\chapter{Enumerations \& Modifiers}")
        ch3_pos = tex.index(r"\chapter{Instruction Reference}")
        app_pos = tex.index(r"\appendix")
        self.assertLess(ch1_pos, sec14_pos)
        self.assertLess(sec14_pos, ch2_pos)
        self.assertNotIn(r"\chapter{CSF Execution Context", tex)

        # Worked traces in Section 1.4.5 contain both Compute and Graphics traces
        self.assertIn("Trace A: End-to-End Compute Shader Execution", tex)
        self.assertIn("Trace B: End-to-End Graphics Execution", tex)

        # Every instruction ends with \clearpage
        ch3_tex = tex[ch3_pos:app_pos]
        self.assertGreaterEqual(ch3_tex.count(r"\clearpage"), n_mns)

        # Every \hyperref[...] target (both enum:* and sec:csf-*) resolves to a defined \label{...}
        defined_labels = set(re.findall(r"\\label\{([a-zA-Z0-9_:-]+)\}", tex))
        referenced_labels = set(re.findall(r"\\hyperref\[([a-zA-Z0-9_:-]+)\]", tex))
        self.assertTrue(referenced_labels, "Expected clickable hyperrefs in generated .tex")
        dangling = referenced_labels - defined_labels
        self.assertEqual(sorted(dangling), [], f"Dangling hyperrefs: {sorted(dangling)}")
        self.assertIn(r"\hyperref[enum:flow_control_m]{\texttt{flow}}", ch3_tex)
        self.assertIn(r"\hyperref[sec:csf-spd]", ch3_tex)
        self.assertIn(r"\hyperref[sec:csf-srt]", ch3_tex)

        # Appendix A.2 summary table appears before A.2.1 deep dive
        a2_table_pos = tex.index(r"\label{tab:a2-ddk-panfrost-summary}")
        a2_1_pos = tex.index(r"\subsubsection*{A.2.1")
        self.assertLess(a2_table_pos, a2_1_pos)


if __name__ == "__main__":
    unittest.main()

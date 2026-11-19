"""Regression & structural verification tests for tools/reference/ Valhall ISA generator."""

from __future__ import annotations

import re
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

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
        """Semantics constants live in isa_semantics, formatting helpers in fmt_helpers, central template in gen."""
        for attr in ("CAT_ORDER", "ISA_SEMANTICS", "INST_EXTRAS", "_SH_META", "_LOP_META", "KEY_ENUMS", "SECONDARY_ENUM_WALKTHROUGHS"):
            self.assertTrue(hasattr(isa_semantics, attr), f"isa_semantics missing {attr}")
        for attr in ("tex_esc", "md_inline_to_tex", "fmt_colored_syntax", "fmt_asm_line", "render_template", "split_notes", "fmt_notes_tex"):
            self.assertTrue(hasattr(fmt_helpers, attr), f"fmt_helpers missing {attr}")
        self.assertIn("%%%%%~~~$MNEM_TEX~~~%%%%%", gen.INSTRUCTION_ENTRY_TEMPLATE)
        self.assertIn(r"\clearpage", gen.INSTRUCTION_ENTRY_TEMPLATE)

    def test_all_mnemonics_have_semantics_and_multiple_examples(self) -> None:
        """Every canonical mnemonic in isa-v9-v14.xml must have semantics and >= 2 worked examples."""
        self.assertEqual(isa_semantics.missing_semantics(self.mnem_specs), [])
        for mn in sorted(self.mnem_specs):
            self.assertIn(mn, isa_semantics.INST_EXTRAS, f"Missing INST_EXTRAS for {mn}")
            spv, glsl, examples = isa_semantics.extras_of(mn)
            self.assertTrue(spv, f"Empty SPIR-V op for {mn}")
            self.assertTrue(glsl, f"Empty GLSL snippet for {mn}")
            self.assertGreaterEqual(len(examples), 2, f"Expected >= 2 worked examples for {mn}")
            for label, asm_tex, walk_md in examples:
                self.assertTrue(label.strip(), f"Empty example label for {mn}")
                self.assertTrue(asm_tex.strip(), f"Empty asm_tex for {mn}")
                self.assertTrue(walk_md.strip(), f"Empty walkthrough for {mn}")

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
        """End-to-end generation check: anchors, Fields Explained, backlinks, page breaks, and A.2 table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_tex = Path(tmpdir) / "valhall-isa-reference.tex"
            stats = gen.generate(out_tex=out_tex)
            tex = out_tex.read_text()

        self.assertEqual(stats["mnemonics"], len(self.mnem_specs))
        self.assertEqual(re.findall(r"%%%%%~~~\$[A-Z0-9_]+~~~%%%%%", tex), [])
        self.assertNotIn("Uncommon Fields Explained", tex)
        self.assertEqual(tex.count(r"\textbf{Fields Explained:}"), len(self.mnem_specs))

        # Every instruction ends with \clearpage
        ch3_start = tex.index(r"\chapter{Instruction Reference}")
        app_start = tex.index(r"\appendix")
        ch3_tex = tex[ch3_start:app_start]
        self.assertGreaterEqual(ch3_tex.count(r"\clearpage"), len(self.mnem_specs))

        # Every \hyperref[enum:...] backlink points to an existing \label{enum:...}
        defined_labels = set(re.findall(r"\\label\{(enum:[a-z0-9_]+)\}", tex))
        referenced_labels = set(re.findall(r"\\hyperref\[(enum:[a-z0-9_]+)\]", tex))
        self.assertTrue(referenced_labels, "Expected clickable enum backlinks in generated .tex")
        dangling = referenced_labels - defined_labels
        self.assertEqual(sorted(dangling), [], f"Dangling enum hyperrefs: {sorted(dangling)}")
        self.assertIn(r"\hyperref[enum:flow_control_m]{\texttt{flow}}", ch3_tex)
        self.assertIn(r"\hyperref[enum:saturate_m]{\texttt{saturate}}", ch3_tex)

        # Appendix A.2 summary table appears before A.2.1 deep dive
        a2_table_pos = tex.index(r"\label{tab:a2-ddk-panfrost-summary}")
        a2_1_pos = tex.index(r"\subsubsection*{A.2.1")
        self.assertLess(a2_table_pos, a2_1_pos)


if __name__ == "__main__":
    unittest.main()

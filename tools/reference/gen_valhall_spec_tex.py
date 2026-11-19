"""Generate tools/reference/generated/valhall-isa-reference.tex from isa-v9-v14.xml.

All ISA semantics, worked examples, and curated enum walkthroughs live in
``isa_semantics.py``; all LaTeX/Markdown/assembly formatting helpers live in
``fmt_helpers.py``; all field and enum-entry descriptions live in
``v3_enrichments.py``; Equivalent GLSL, exact Valhall mappings, and CSF ABI
sideband mappings live in ``csf_abi_map.py``; and static document boilerplate
(including Chapter 4 CSF ABI and Appendix A) lives in ``template.tex``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from . import csf_abi_map
from . import isa_semantics
from . import v3_enrichments as v3e
from .fmt_helpers import (
    fmt_colored_syntax,
    fmt_notes_tex,
    md_inline_to_tex,
    render_enum_entries_table,
    render_field_table_rows,
    render_fields_explained_items,
    render_template,
    render_variant_table_rows,
    render_walkthroughs_tex,
    tex_esc,
)

XML_PATH = Path("specs/kraid/isa-v9-v14.xml")
OUT_TEX = Path("tools/reference/generated/valhall-isa-reference.tex")
TEMPLATE_PATH = Path("tools/reference/template.tex")

# Central LaTeX templates for repeating document units
KEY_ENUM_SECTION_TEMPLATE = r'''\section{%%%%%~~~$TITLE~~~%%%%%}
\label{enum:%%%%%~~~$ENAME~~~%%%%%}
%%%%%~~~$DESC~~~%%%%%
%%%%%~~~$CSF_ABI_NOTE~~~%%%%%
%%%%%~~~$ENTRIES_TABLE~~~%%%%%
%%%%%~~~$WALKTHROUGHS~~~%%%%%
'''

REM_ENUM_SUBSECTION_TEMPLATE = r'''\subsection{%%%%%~~~$PRETTY~~~%%%%% (\texttt{%%%%%~~~$ENAME_TEX~~~%%%%%})}
\label{enum:%%%%%~~~$ENAME~~~%%%%%}
Used by \texttt{%%%%%~~~$SAMPLE_INS~~~%%%%%} and related instructions (%%%%%~~~$VAL_COUNT~~~%%%%% encoding values):
%%%%%~~~$CSF_ABI_NOTE~~~%%%%%
%%%%%~~~$ENTRIES_TABLE~~~%%%%%
%%%%%~~~$WALKTHROUGHS~~~%%%%%
'''

INSTRUCTION_ENTRY_TEMPLATE = r'''\subsection*{%%%%%~~~$MNEM_TEX~~~%%%%%}
\addcontentsline{toc}{subsection}{%%%%%~~~$MNEM_TEX~~~%%%%%}
%%%%%~~~$BEHAVIOR~~~%%%%%
%%%%%~~~$NOTES~~~%%%%%
\codebox{%%%%%~~~$SYNTAX_BODY~~~%%%%%}
\textbf{Closest SPIR-V Op:} \texttt{%%%%%~~~$SPIRV_OP~~~%%%%%}
\par\textbf{Equivalent GLSL (Operational Semantics \& Flag Variants):}
\codebox{%%%%%~~~$EQUIV_GLSL~~~%%%%%}
\textbf{Execution Context:}
\begin{tightitemize}\footnotesize
%%%%%~~~$EXEC_CONTEXT~~~%%%%%
\end{tightitemize}
%%%%%~~~$WORKED_EXAMPLES~~~%%%%%
\begin{center}\footnotesize
\begin{tabularx}{\linewidth}{@{}lllX@{}}
\toprule
Variant & Arch & Class & Opcode Fields \\
\midrule
%%%%%~~~$VARIANT_ROWS~~~%%%%%
\bottomrule
\end{tabularx}
\end{center}
\begin{center}\footnotesize
\begin{tabularx}{0.88\linewidth}{@{}l l X@{}}
\toprule
Field Name & Bits & Type / Fixed Value \\
\midrule
%%%%%~~~$FIELD_ROWS~~~%%%%%
\bottomrule
\end{tabularx}
\end{center}
\textbf{Fields Explained:}
\begin{tightitemize}\footnotesize
%%%%%~~~$FIELDS_EXPLAINED~~~%%%%%
\end{tightitemize}
\clearpage
'''


def parse_enums(root: ET.Element):
    """Parse all <enum> definitions and map each *_m enum to a sample instruction."""
    enums: dict[str, list[tuple[int, str, str, str]]] = defaultdict(list)
    enum_seen: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for e in root.findall("enum"):
        ename = e.attrib["name"]
        since = e.attrib.get("since", "9")
        until = e.attrib.get("until", "14")
        arch_str = f"v{since}" if since == until else f"v{since}--v{until}"
        for v in e.findall("value"):
            vname = v.attrib.get("name", (v.text or "").strip())
            venc = v.attrib.get("value", "")
            if not vname:
                continue
            key = (venc, vname)
            if key not in enum_seen[ename]:
                enum_seen[ename].add(key)
                try:
                    enc_num = int(venc, 0)
                except Exception:
                    enc_num = 999
                enums[ename].append((enc_num, venc, vname, arch_str))
    for ename in enums:
        enums[ename].sort(key=lambda x: (x[0], x[2]))

    enum_sample_ins: dict[str, str] = {}
    for ins_el in root.findall("instruction"):
        mn = ins_el.attrib["name"]
        for ch in ins_el:
            t = ch.attrib.get("type", "")
            if t.endswith("_m") and t not in enum_sample_ins:
                enum_sample_ins[t] = mn

    return enums, enum_sample_ins


def parse_instructions(root: ET.Element) -> dict[str, list[dict]]:
    """Parse all <instruction> specs grouped by canonical mnemonic."""
    mnem_specs: dict[str, list[dict]] = defaultdict(list)
    for ins in root.findall("instruction"):
        full = ins.attrib["name"]
        mnem, var = (full.split(".", 1) if "." in full else (full, ""))
        archs = sorted(int(x) for x in ins.attrib.get("archs", "").split() if x.isdigit())
        arch_str = f"v{archs[0]}" if len(archs) == 1 else (f"v{archs[0]}--v{archs[-1]}" if archs else "v9--v14")

        fields = []
        cls = ""
        opcodes = []
        for child in ins:
            tag = child.tag
            if tag in ("field", "reserved"):
                pos = int(child.attrib["pos"])
                width = int(child.attrib["width"])
                hi = pos + width - 1
                bits_str = f"{pos}" if width == 1 else f"{pos}:{hi}"
                if tag == "reserved":
                    ftype = child.attrib.get("type", "zero")
                    fields.append((pos, hi, bits_str, r"\textit{Reserved}", ftype, ftype))
                else:
                    fname = child.attrib["name"]
                    ftype = child.attrib.get("type", "uint")
                    exact = child.attrib.get("exact")
                    default = child.attrib.get("default")
                    if fname == "instruction_class" and exact:
                        cls = exact
                    if fname.startswith("opcode") and exact:
                        opcodes.append(f"{fname}={exact}")
                    if fname == "datatype" and exact and not var:
                        var = exact
                    meta = ftype
                    if exact is not None:
                        meta = f"exact={exact}"
                    elif default is not None:
                        meta = f"{ftype} (def {default})"
                    fields.append((pos, hi, bits_str, fname, ftype, meta))
        fields.sort(key=lambda x: x[0])

        mnem_specs[mnem].append({
            "full": full,
            "var": var or "(none)",
            "arch": arch_str,
            "cls": cls or "--",
            "opcodes": ", ".join(opcodes) if opcodes else "--",
            "fields": fields,
            "syntax_tex": fmt_colored_syntax(ins),
        })
    return mnem_specs


def build_enum_sections(enums, enum_sample_ins):
    """Build Chapter 2.1--2.50 sections, Section 2.51 summary rows, and 2.51.X subsections."""
    documented_enums: set[str] = set()
    key_enum_blocks: list[str] = []

    for ename, title, desc, walkthroughs in isa_semantics.KEY_ENUMS:
        vals = enums.get(ename, [])
        if not vals:
            continue
        documented_enums.add(ename)
        sample_ins = enum_sample_ins.get(ename, "OP")
        abi_note = csf_abi_map.csf_abi_for_enum(ename)
        abi_tex = f"\\par\\textbf{{CSF ABI Interaction:}} {abi_note}" if abi_note else ""
        key_enum_blocks.append(
            render_template(
                KEY_ENUM_SECTION_TEMPLATE,
                {
                    "TITLE": title,
                    "ENAME": ename,
                    "DESC": md_inline_to_tex(desc),
                    "CSF_ABI_NOTE": abi_tex,
                    "ENTRIES_TABLE": render_enum_entries_table(ename, vals, sample_ins, v3e.describe_enum_entry),
                    "WALKTHROUGHS": render_walkthroughs_tex(
                        walkthroughs,
                        heading_singular="Combined Walkthrough:",
                        heading_plural="Combined Walkthroughs:",
                    ),
                },
                source_name="KEY_ENUM_SECTION_TEMPLATE",
            )
        )

    rem_enums = sorted(e for e in enums if e.endswith("_m") and e not in documented_enums)
    rem_summary_rows: list[str] = []
    rem_subsection_blocks: list[str] = []

    for ename in rem_enums:
        vals = enums[ename]
        vstr = ", ".join(f"{vname}({venc})" for _n, venc, vname, _a in vals[:8])
        if len(vals) > 8:
            vstr += f", ... ({len(vals)} total)"
        rem_summary_rows.append(
            f"\\hyperref[enum:{ename}]{{\\texttt{{{tex_esc(ename)}}}}} & \\texttt{{{tex_esc(vstr)}}} \\\\"
        )

        sample_ins = enum_sample_ins.get(ename, "N/A")
        pretty = (ename[:-2] if ename.endswith("_m") else ename).replace("_", " ").title()
        sec_walks = isa_semantics.secondary_enum_walkthroughs(ename)
        if not sec_walks:
            raise KeyError(f"Missing worked example for secondary enum {ename!r} in isa_semantics.SECONDARY_ENUM_WALKTHROUGHS")
        abi_note = csf_abi_map.csf_abi_for_enum(ename)
        abi_tex = f"\\par\\textbf{{CSF ABI Interaction:}} {abi_note}" if abi_note else ""
        rem_subsection_blocks.append(
            render_template(
                REM_ENUM_SUBSECTION_TEMPLATE,
                {
                    "PRETTY": tex_esc(pretty),
                    "ENAME_TEX": tex_esc(ename),
                    "ENAME": ename,
                    "SAMPLE_INS": tex_esc(sample_ins),
                    "VAL_COUNT": str(len(vals)),
                    "CSF_ABI_NOTE": abi_tex,
                    "ENTRIES_TABLE": render_enum_entries_table(ename, vals, sample_ins, v3e.describe_enum_entry),
                    "WALKTHROUGHS": render_walkthroughs_tex(
                        sec_walks,
                        heading_singular="Worked Example:",
                        heading_plural="Worked Examples:",
                    ),
                },
                source_name="REM_ENUM_SUBSECTION_TEMPLATE",
            )
        )

    ch2_enums = documented_enums | set(rem_enums)
    return (
        "\n".join(key_enum_blocks),
        len(rem_enums),
        "\n".join(rem_summary_rows),
        "\n".join(rem_subsection_blocks),
        ch2_enums,
        len(documented_enums),
    )


def build_instruction_reference(mnem_specs: dict[str, list[dict]], ch2_enums: set[str]) -> str:
    """Build Chapter 3 using the central INSTRUCTION_ENTRY_TEMPLATE."""
    missing_sem = isa_semantics.missing_semantics(mnem_specs)
    if missing_sem:
        raise SystemExit(f"isa_semantics.py is missing entries for: {missing_sem}")

    missing_extras = sorted(set(mnem_specs.keys()) - set(isa_semantics.INST_EXTRAS.keys()))
    if missing_extras:
        raise SystemExit(f"isa_semantics.INST_EXTRAS is missing entries for: {missing_extras}")

    by_cat: dict[str, list[str]] = defaultdict(list)
    for mn in sorted(mnem_specs.keys()):
        by_cat[isa_semantics.category_of(mn)].append(mn)

    out_blocks: list[str] = []
    for cat in isa_semantics.CAT_ORDER:
        mns = by_cat.get(cat, [])
        out_blocks.append(f"\\section{{{tex_esc(cat)} ({len(mns)} mnemonics)}}")
        for mn in mns:
            specs = mnem_specs[mn]
            beh, note = isa_semantics.behavior_of(mn)
            spv_op, _legacy_glsl, examples = isa_semantics.extras_of(mn)
            equiv_glsl_tex = csf_abi_map.equiv_glsl_tex(mn)
            exec_ctx_tex = csf_abi_map.render_execution_context_tex(mn, cat)

            seen_syn: list[str] = []
            for sp in specs:
                if sp["syntax_tex"] not in seen_syn:
                    seen_syn.append(sp["syntax_tex"])
            syn_body = r"\\".join(seen_syn[:8])
            if len(seen_syn) > 8:
                syn_body += r"\\" + tex_esc(f"... ({len(seen_syn) - 8} additional arch/variant forms)")

            rep = specs[0]
            out_blocks.append(
                render_template(
                    INSTRUCTION_ENTRY_TEMPLATE,
                    {
                        "MNEM_TEX": tex_esc(mn),
                        "BEHAVIOR": md_inline_to_tex(beh) if beh else "",
                        "NOTES": fmt_notes_tex(note) if note else "",
                        "SYNTAX_BODY": syn_body,
                        "SPIRV_OP": tex_esc(spv_op),
                        "EQUIV_GLSL": equiv_glsl_tex,
                        "EXEC_CONTEXT": exec_ctx_tex,
                        "WORKED_EXAMPLES": render_walkthroughs_tex(
                            examples,
                            heading_singular="Worked Example:",
                            heading_plural="Worked Examples:",
                        ),
                        "VARIANT_ROWS": render_variant_table_rows(specs),
                        "FIELD_ROWS": render_field_table_rows(rep["fields"], ch2_enums),
                        "FIELDS_EXPLAINED": render_fields_explained_items(
                            rep["fields"], v3e.get_field_doc, ch2_enums
                        ),
                    },
                    source_name="INSTRUCTION_ENTRY_TEMPLATE",
                )
            )
    return "\n".join(out_blocks)


def generate(
    xml_path: Path = XML_PATH,
    template_path: Path = TEMPLATE_PATH,
    out_tex: Path = OUT_TEX,
) -> dict[str, int]:
    """Generate the Valhall ISA Reference .tex file and return summary stats."""
    root = ET.parse(xml_path).getroot()
    enums, enum_sample_ins = parse_enums(root)
    mnem_specs = parse_instructions(root)

    (
        key_enum_tex,
        rem_count,
        rem_summary_tex,
        rem_sub_tex,
        ch2_enums,
        key_count,
    ) = build_enum_sections(enums, enum_sample_ins)
    instruction_ref_tex = build_instruction_reference(mnem_specs, ch2_enums)

    template = template_path.read_text()
    tex = render_template(
        template,
        {
            "KEY_ENUM_SECTIONS": key_enum_tex,
            "REM_ENUM_COUNT": str(rem_count),
            "REM_ENUM_SUMMARY_ROWS": rem_summary_tex,
            "REM_ENUM_SUBSECTIONS": rem_sub_tex,
            "INSTRUCTION_REFERENCE": instruction_ref_tex,
        },
        source_name=str(template_path),
    )

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(tex)
    return {
        "lines": tex.count("\n") + 1,
        "bytes": out_tex.stat().st_size,
        "key_enums": key_count,
        "rem_enums": rem_count,
        "mnemonics": len(mnem_specs),
    }


def main() -> None:
    stats = generate()
    print(
        f"Wrote {OUT_TEX} ({stats['lines']} lines, {stats['bytes'] / 1024:.1f} KiB, "
        f"{stats['key_enums']} detailed + {stats['rem_enums']} secondary enums, "
        f"{stats['mnemonics']} mnemonics)"
    )


if __name__ == "__main__":
    main()

"""Formatting, escaping, and LaTeX rendering helpers for the Valhall ISA reference.

Pure formatting utilities with no ISA semantics data tables so that
isa_semantics.py, v3_enrichments.py, and gen_valhall_spec_tex.py can all share
a single canonical implementation without circular imports.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable, Sequence

_LITERAL_ESC_MAP = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
    "|": r"\textbar{}",
}


def tex_esc_literal(s: str) -> str:
    """Escape all LaTeX special characters literally (including '$')."""
    if not s:
        return ""
    return "".join(_LITERAL_ESC_MAP.get(c, c) for c in s)


def tex_esc(s: str) -> str:
    """Escape prose for LaTeX while preserving inline $math$ spans."""
    if not s:
        return ""
    s = s.replace("±", r"$\pm$").replace("≤", r"$\le$").replace("≥", r"$\ge$")
    s = s.replace("≠", r"$\ne$").replace("→", r"$\to$").replace("×", r"$\times$")
    s = s.replace("–", "--").replace("—", "---").replace("…", r"\ldots{}")
    s = s.replace("π", r"$\pi$").replace("∈", r"$\in$").replace("∉", r"$\notin$")
    s = s.replace("−", "-")
    parts = re.split(r"(\$[^$]+\$)", s)
    out = []
    for i, p in enumerate(parts):
        if i % 2 == 1:
            out.append(p)
        else:
            p = p.replace("\\", r"\textbackslash{}")
            p = p.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
            p = p.replace("_", r"\_").replace("{", r"\{").replace("}", r"\}")
            p = p.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
            p = p.replace("<", r"\textless{}").replace(">", r"\textgreater{}")
            p = p.replace("|", r"\textbar{}")
            out.append(p)
    return "".join(out)


def mnem(s: str) -> str:
    return f"\\mnem{{{tex_esc_literal(s)}}}"


def imod(s: str) -> str:
    return f"\\imod{{{tex_esc_literal(s)}}}"


def dst(s: str) -> str:
    return f"\\dst{{{tex_esc_literal(s)}}}"


def src(s: str) -> str:
    return f"\\src{{{tex_esc_literal(s)}}}"


def imm(s: str) -> str:
    return f"\\imm{{{tex_esc_literal(s)}}}"


def md_inline_to_tex(s: str) -> str:
    """Convert inline Markdown (`code`, **bold**) to LaTeX."""
    if not s:
        return ""
    tokens = re.split(r"(`[^`]+`|\*\*[^*]+\*\*)", s)
    res = []
    for tok in tokens:
        if tok.startswith("`") and tok.endswith("`"):
            res.append(r"\texttt{" + tex_esc(tok[1:-1]) + "}")
        elif tok.startswith("**") and tok.endswith("**"):
            res.append(r"\textbf{" + tex_esc(tok[2:-2]) + "}")
        else:
            res.append(tex_esc(tok))
    return "".join(res)


def fmt_colored_syntax(ins_el) -> str:
    """Render color-coded Valhall syntax line from an XML <instruction> element."""
    full = ins_el.attrib["name"]
    syn_el = ins_el.find("syntax")
    if syn_el is None:
        return r"\mnem{" + tex_esc(full) + "}"
    name_mods = []
    ops_tex = []
    for ch in syn_el:
        if ch.tag == "name":
            for m in ch.findall("modifier"):
                name_mods.append(r"\imod{\{." + tex_esc(m.attrib["name"]) + r"\}}")
        elif ch.tag == "dest":
            dm = "".join(r"\imod{\{." + tex_esc(m.attrib["name"]) + r"\}}" for m in ch.findall("modifier"))
            ops_tex.append(r"\dst{dest}" + dm)
        elif ch.tag == "sr":
            nm = ch.attrib.get("name", "sr")
            sm = "".join(r"\imod{\{." + tex_esc(m.attrib["name"]) + r"\}}" for m in ch.findall("modifier"))
            if "write" in nm or nm in ("sw", "sr_dst"):
                ops_tex.append(r"\dst{@" + tex_esc(nm) + "}" + sm)
            else:
                ops_tex.append(r"\src{@" + tex_esc(nm) + "}" + sm)
        elif ch.tag == "src":
            nm = ch.attrib.get("name", "src")
            sm = "".join(r"\imod{\{." + tex_esc(m.attrib["name"]) + r"\}}" for m in ch.findall("modifier"))
            ops_tex.append(r"\src{" + tex_esc(nm) + "}" + sm)
        else:
            nm = ch.attrib.get("name", ch.tag)
            ops_tex.append(r"\imm{\#" + tex_esc(nm) + "}")
    line = r"\mnem{" + tex_esc(full) + "}" + "".join(name_mods)
    if ops_tex:
        line += " " + ", ".join(ops_tex)
    return line


def fmt_asm_line(
    mnem_full: str,
    mods: str,
    dst_op: str,
    srcs: Sequence[str] | None,
    imms: Sequence[str] | None = None,
) -> str:
    """Format a concrete color-coded Valhall assembly example."""
    parts = [r"\mnem{" + tex_esc(mnem_full) + "}"]
    if mods:
        parts[0] += r"\imod{" + tex_esc(mods) + "}"
    ops = []
    if dst_op:
        if "." in dst_op and not dst_op.startswith("["):
            base, dmod = dst_op.split(".", 1)
            ops.append(r"\dst{" + tex_esc(base) + "}" + r"\imod{." + tex_esc(dmod) + "}")
        else:
            ops.append(r"\dst{" + tex_esc(dst_op) + "}")
    for s in (srcs or []):
        if "." in s and not s.startswith("["):
            base, smod = s.split(".", 1)
            ops.append(r"\src{" + tex_esc(base) + "}" + r"\imod{." + tex_esc(smod) + "}")
        else:
            ops.append(r"\src{" + tex_esc(s) + "}")
    for im in (imms or []):
        ops.append(r"\imm{" + tex_esc(im) + "}")
    if ops:
        return parts[0] + " " + ", ".join(ops)
    return parts[0]


def render_template(template: str, subs: dict[str, str], source_name: str = "template") -> str:
    """Substitute %%%%%~~~$NAME~~~%%%%% anchors in a template string."""
    out = template
    for name, value in subs.items():
        anchor = f"%%%%%~~~${name}~~~%%%%%"
        if anchor not in out:
            raise KeyError(f"Template anchor {anchor!r} not found in {source_name}")
        out = out.replace(anchor, value)
    leftover = re.findall(r"%%%%%~~~\$[A-Z0-9_]+~~~%%%%%", out)
    if leftover:
        raise KeyError(f"Unreplaced template anchors in {source_name}: {leftover}")
    return out


_TRAILING_REFS_RE = re.compile(
    r"\s*\(((?:`[^`]+`(?:,\s*)?)+)\)\s*$"
)


def _split_top_level(text: str, seps: tuple[str, ...] = (". ", "; ")) -> list[str]:
    """Split prose on sentence/clause separators outside parens and backticks."""
    items: list[str] = []
    buf: list[str] = []
    depth = 0
    in_code = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "`":
            in_code = not in_code
            buf.append(ch)
            i += 1
            continue
        if not in_code:
            if ch == "(":
                depth += 1
            elif ch == ")" and depth > 0:
                depth -= 1
            elif depth == 0:
                matched_sep = None
                for sep in seps:
                    if text.startswith(sep, i):
                        # Avoid splitting after common abbreviations like e.g. / i.e. / vs.
                        prefix = "".join(buf).rstrip()
                        if sep == ". " and re.search(r"\b(?:e\.g|i\.e|vs|etc|arch)$", prefix, re.I):
                            continue
                        matched_sep = sep
                        break
                if matched_sep is not None:
                    piece = "".join(buf).strip()
                    if piece:
                        if not piece.endswith((".", "!", "?")):
                            piece += "."
                        items.append(piece)
                    buf = []
                    i += len(matched_sep)
                    continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        items.append(tail)
    return items


def split_notes(note: str | Sequence[str]) -> list[str]:
    """Normalize instruction notes into a list of clear, self-contained items.

    Accepts either a list of strings (already structured) or a prose string with
    multiple sentences/clauses and an optional trailing parenthesized reference
    list such as ``(`kraid/ops.rs:10-20`, `hw_tests.rs:5`)``.
    """
    if not note:
        return []
    if not isinstance(note, str):
        return [item.strip() for item in note if item and item.strip()]

    text = note.strip()
    refs_item: str | None = None
    m = _TRAILING_REFS_RE.search(text)
    if m:
        refs_raw = m.group(1).strip()
        prefix = text[: m.start()].strip()
        if prefix:
            text = prefix
            refs_item = f"**Source references:** {refs_raw}."
        else:
            # The entire note was just references.
            return [f"**Source references:** {refs_raw}."]

    items = _split_top_level(text)
    # Capitalize the leading character of any clause split after a semicolon
    cleaned: list[str] = []
    for item in items:
        s = item.strip()
        if s and s[0].islower() and not s.startswith(("v9", "v10", "v11", "v12", "v13", "v14", "f16", "f32", "s8", "s16", "s32", "s64", "u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "src", "dst")):
            s = s[0].upper() + s[1:]
        cleaned.append(s)
    if refs_item:
        cleaned.append(refs_item)
    return cleaned


def fmt_notes_tex(note: str | Sequence[str]) -> str:
    """Format instruction notes as a bulleted list when multiple items exist."""
    items = split_notes(note)
    if not items:
        return ""
    if len(items) == 1:
        return f"\\par\\textbf{{Note:}} {md_inline_to_tex(items[0])}"
    lines = [r"\par\textbf{Notes:}", r"\begin{tightitemize}"]
    for item in items:
        lines.append(f"  \\item {md_inline_to_tex(item)}")
    lines.append(r"\end{tightitemize}")
    return "\n".join(lines)


def render_walkthroughs_tex(
    walkthroughs: Sequence[tuple[str, str, str]],
    heading_singular: str = "Worked Example:",
    heading_plural: str = "Worked Examples:",
) -> str:
    """Render one or more (label, asm_tex, walkthrough_md) examples cleanly."""
    if not walkthroughs:
        return ""
    if len(walkthroughs) == 1:
        label, asm_tex, walk_md = walkthroughs[0]
        head = f"\\par\\textbf{{{heading_singular}}}"
        if label:
            head += f" \\emph{{{tex_esc(label)}}}"
        return f"{head}\n\\codebox{{{asm_tex}}}\n{md_inline_to_tex(walk_md)}"

    lines = [f"\\par\\textbf{{{heading_plural}}}"]
    for idx, (label, asm_tex, walk_md) in enumerate(walkthroughs, 1):
        title = f"Example {idx} ({tex_esc(label)}):" if label else f"Example {idx}:"
        lines.append(f"\\par\\textbf{{{title}}}")
        lines.append(f"\\codebox{{{asm_tex}}}")
        lines.append(md_inline_to_tex(walk_md))
    return "\n".join(lines)


def render_enum_entries_table(
    ename: str,
    vals: Sequence[tuple[int, str, str, str]],
    sample_ins: str,
    describe_fn: Callable[[str, str, str, str], tuple[str, str]],
) -> str:
    """Render the Chapter 2 longtable for a modifier enum."""
    out = [
        r"\begin{center}\footnotesize",
        r"\begin{longtable}{@{}rll>{\raggedright\arraybackslash}p{0.62\linewidth}@{}}",
        r"\toprule",
        r"Enc & Value & Arch & Description \& Worked Example \\",
        r"\midrule\endhead",
    ]
    for _num, venc, vname, arch_s in vals:
        desc_tex, ex_tex = describe_fn(ename, vname, venc, sample_ins)
        out.append(
            f"\\texttt{{{tex_esc(venc)}}} & \\texttt{{\\imod{{{tex_esc(vname)}}}}} & {tex_esc(arch_s)} & "
            f"{desc_tex} \\newline {{\\scriptsize\\ttfamily {ex_tex}}} \\\\[2pt]"
        )
    out.append(r"\bottomrule")
    out.append(r"\end{longtable}")
    out.append(r"\end{center}")
    return "\n".join(out)


def render_variant_table_rows(specs: Iterable[dict]) -> str:
    """Render table rows for the per-instruction Variant/Arch/Class/Opcodes table."""
    rows = []
    for sp in specs:
        rows.append(
            f"\\texttt{{{tex_esc(sp['var'])}}} & {tex_esc(sp['arch'])} & "
            f"\\texttt{{{tex_esc(sp['cls'])}}} & \\texttt{{{tex_esc(sp['opcodes'])}}} \\\\"
        )
    return "\n".join(rows)


def render_field_table_rows(
    fields: Sequence[tuple[int, int, str, str, str, str]],
    ch2_enums: set[str],
) -> str:
    """Render table rows for the 64-bit encoding layout table with Section 2 backlinks.

    Each tuple in ``fields`` is ``(pos, hi, bits_str, fname, ftype, meta)``.
    When ``ftype`` is one of the Chapter 2 modifier enums, the field name
    hyperlinks directly back to ``\\label{enum:<ftype>}``.
    """
    rows = []
    for _pos, _hi, bits_str, fname, ftype, meta in fields:
        if fname.startswith(r"\textit{"):
            fn_tex = fname
        elif ftype in ch2_enums:
            fn_tex = f"\\hyperref[enum:{ftype}]{{\\texttt{{{tex_esc(fname)}}}}}"
        else:
            fn_tex = f"\\texttt{{{tex_esc(fname)}}}"
        rows.append(f"{fn_tex} & \\texttt{{{tex_esc(bits_str)}}} & \\texttt{{{tex_esc(meta)}}} \\\\")
    return "\n".join(rows)


def render_fields_explained_items(
    fields: Sequence[tuple[int, int, str, str, str, str]],
    get_field_doc_fn: Callable[[str], tuple[str, str, str, str] | None],
    ch2_enums: set[str],
) -> str:
    """Render bulleted explanations for EVERY non-reserved field in an instruction."""
    seen: list[tuple[str, str]] = []
    seen_names: set[str] = set()
    for _pos, _hi, _bits_str, fname, ftype, _meta in fields:
        if fname.startswith(r"\textit{") or fname in seen_names:
            continue
        seen_names.add(fname)
        seen.append((fname, ftype))

    items = []
    for fname, ftype in seen:
        doc = get_field_doc_fn(fname)
        if doc is None:
            raise KeyError(f"Missing field documentation for XML field {fname!r} (type={ftype!r})")
        fdesc, fex, fspv, fglsl = doc
        if ftype in ch2_enums:
            label_tex = f"\\hyperref[enum:{ftype}]{{\\texttt{{{tex_esc(fname)}}}}}"
        else:
            label_tex = f"\\texttt{{{tex_esc(fname)}}}"
        items.append(
            f"  \\item {label_tex}: {md_inline_to_tex(fdesc)} "
            f"\\emph{{Example:}} {{\\scriptsize\\ttfamily {fex}}} "
            f"(\\textbf{{SPIR-V:}} \\texttt{{{tex_esc(fspv)}}}; \\textbf{{GLSL:}} \\texttt{{{tex_esc(fglsl)}}})"
        )
    return "\n".join(items)

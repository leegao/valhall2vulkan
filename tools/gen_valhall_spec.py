#!/usr/bin/env python3
"""
Spec-Driven Code & Semantics Generator for Valhall ISA (v9-v14) and CSF Descriptors (v11-v14)
=============================================================================================
Inputs:
  - specs/valhall/ISA.xml          (Mesa Panfrost Valhall ISA specification: LUTs, FAU pages, descriptions, units)
  - specs/kraid/isa-v9-v14.xml     (Arm Mali ISA v9-v14 specification: 627 instructions, 201 mnemonics, 219 enums)
  - specs/kraid/ops.rs             (Kraid IR opcodes & Foldable operational semantics)
  - specs/kraid/encode_v9.rs       (Kraid IR -> v9-v14 hardware instruction lowering mappings)
  - specs/csf/v11.xml (+ common.xml, v12.xml, v13.xml, v14.xml) (CSF commands, SR register files, GPU descriptors)

Outputs:
  - src/generated/valhall_isa_generated.h
  - src/generated/valhall_isa_generated.cpp
  - src/generated/valhall_csf_descriptors_generated.h
  - src/generated/valhall_csf_descriptors_generated.cpp
  - docs/Formal-Semantics-and-Translation.md
"""

import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict, OrderedDict


def c_ident(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        return "Empty"
    if s[0].isdigit():
        s = "V_" + s
    kw = {
        "default", "switch", "case", "class", "struct", "union", "enum",
        "signed", "unsigned", "int", "float", "double", "bool", "char",
        "void", "return", "break", "continue", "goto", "if", "else",
        "for", "while", "do", "operator", "template", "typename", "private",
        "public", "protected", "virtual", "inline", "static", "const",
        "register", "volatile", "short", "long", "xor", "and", "or", "not",
        "bitand", "bitor", "compl", "not_eq", "and_eq", "or_eq", "xor_eq",
        "true", "false", "auto", "delete", "new", "this", "friend", "using",
        "try", "catch", "throw", "explicit", "export", "mutable", "asm",
        "namespace", "sizeof", "typedef", "extern", "alignas", "alignof",
        "constexpr", "decltype", "noexcept", "nullptr", "static_assert",
        "thread_local"
    }
    if s in kw:
        s = s + "_"
    return s


def pascal_case(name: str) -> str:
    parts = re.split(r"[^a-zA-Z0-9]+", name.strip())
    out = "".join(p[0].upper() + p[1:] for p in parts if p)
    if not out:
        return "Empty"
    if out[0].isdigit():
        out = "V" + out
    return out


def snake_case(name: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name.strip())
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).lower().strip("_")
    if not s:
        return "empty"
    if s[0].isdigit():
        s = "f_" + s
    kw = {
        "default", "switch", "case", "class", "struct", "union", "enum",
        "signed", "unsigned", "int", "float", "double", "bool", "char",
        "void", "return", "break", "continue", "goto", "if", "else",
        "for", "while", "do", "operator", "template", "typename", "private",
        "public", "protected", "virtual", "inline", "static", "const",
        "register", "volatile", "short", "long", "xor", "and", "or", "not"
    }
    if s in kw:
        s = s + "_"
    return s


def parse_int(val_str: str) -> int:
    val_str = val_str.strip()
    if val_str.startswith(("0x", "0X")):
        return int(val_str, 16)
    if val_str.startswith(("0b", "0B")):
        return int(val_str, 2)
    return int(val_str, 10)


def get_arch_mask(elem: ET.Element, default_min=9, default_max=14) -> int:
    since = int(elem.attrib.get("since", default_min))
    until = int(elem.attrib.get("until", default_max))
    mask = 0
    for a in range(since, until + 1):
        mask |= (1 << a)
    return mask


# =============================================================================
# 1. Parse valhall/ISA.xml (LUTs, FAU special pages, instruction descriptions)
# =============================================================================
class ValhallLegacySpec:
    def __init__(self, xml_path: str):
        self.root = ET.parse(xml_path).getroot()
        self.immediates = []  # list of (index, uint32_val, desc)
        self.fau_pages = {}   # page_idx -> list of (idx, name, desc)
        self.instr_docs = {}  # base_mnemonic -> dict(desc, unit, sr_read, sr_write)
        self._parse()

    def _parse(self):
        lut = self.root.find("lut[@name='Immediates']")
        if lut is not None:
            idx = 0
            for c in lut.findall("constant"):
                val = parse_int(c.text)
                desc = c.attrib.get("desc", "")
                self.immediates.append((idx, val, desc))
                idx += 1

        for page_id in (0, 1, 3):
            en = self.root.find(f"enum[@name='FAU special page {page_id}']")
            if en is not None:
                entries = []
                idx = 0
                for child in en:
                    if child.tag == "value":
                        entries.append((idx, (child.text or "").strip(), child.attrib.get("desc", "")))
                        idx += 1
                    elif child.tag == "reserved":
                        idx += 1
                self.fau_pages[page_id] = entries

        def process_ins(el, group_desc="", group_unit=""):
            raw_name = el.attrib.get("name", "")
            base = raw_name.split(".")[0]
            unit = el.attrib.get("unit", group_unit)
            desc_el = el.find("desc")
            desc = (desc_el.text.strip() if desc_el is not None and desc_el.text else group_desc)
            desc = " ".join(desc.split())
            sr_el = el.find("sr")
            sr_read = sr_el is not None and sr_el.attrib.get("read", "false") == "true"
            sr_write = sr_el is not None and sr_el.attrib.get("write", "false") == "true"
            if base not in self.instr_docs or (not self.instr_docs[base]["desc"] and desc):
                self.instr_docs[base] = {
                    "desc": desc,
                    "unit": unit,
                    "sr_read": sr_read,
                    "sr_write": sr_write,
                }

        for el in self.root.findall("ins"):
            process_ins(el)
        for grp in self.root.findall("group"):
            gdesc_el = grp.find("desc")
            gdesc = " ".join(gdesc_el.text.split()) if (gdesc_el is not None and gdesc_el.text) else ""
            gunit = grp.attrib.get("unit", "")
            for el in grp.findall("ins"):
                process_ins(el, gdesc, gunit)


# =============================================================================
# 2. Parse kraid/ops.rs & encode_v9.rs for Operational Semantics Extraction
# =============================================================================
class KraidSemanticsExtractor:
    def __init__(self, ops_rs_path: str, encode_v9_rs_path: str):
        self.op_to_fold = {}     # OpName -> fold body snippet
        self.mnemonic_to_op = {} # ISA Mnemonic -> list of OpName
        self._parse(ops_rs_path, encode_v9_rs_path)

    def _parse(self, ops_rs_path: str, encode_v9_rs_path: str):
        if os.path.exists(ops_rs_path):
            with open(ops_rs_path, "r", encoding="utf-8") as f:
                ops_txt = f.read()
            # Extract impl Foldable / PerCompFoldable blocks
            pattern = re.compile(
                r"impl\s+(?:PerComp)?Foldable\s+for\s+(Op\w+)\s*\{(.*?)\n\}",
                re.DOTALL
            )
            for m in pattern.finditer(ops_txt):
                op_name, body = m.group(1), m.group(2).strip()
                # Condense whitespace
                lines = [ln.strip() for ln in body.splitlines() if ln.strip() and not ln.strip().startswith("//")]
                self.op_to_fold[op_name] = " ".join(lines)

        if os.path.exists(encode_v9_rs_path):
            with open(encode_v9_rs_path, "r", encoding="utf-8") as f:
                enc_txt = f.read()
            # Find impl V9Instr for OpXxx { ... } and extract referenced ISA structs
            pattern = re.compile(
                r"impl\s+V9Instr\s+for\s+(Op\w+)\s*\{(.*?)\n\}",
                re.DOTALL
            )
            for m in pattern.finditer(enc_txt):
                op_name, body = m.group(1), m.group(2)
                # Look for encoder.encode(Xxx { ... }) or Xxx::get_info
                isa_structs = set(re.findall(r"\b([A-Z][A-Za-z0-9_]*)\s*(?:::|\{)", body))
                for st in isa_structs:
                    self.mnemonic_to_op.setdefault(st.upper(), []).append(op_name)


# =============================================================================
# 3. Parse kraid/isa-v9-v14.xml (219 Enums, 627 Instructions, Virtual Expressions)
# =============================================================================
class KraidIsaEnumValue:
    def __init__(self, name: str, value: int, arch_mask: int, canonical: bool, bit_pattern):
        self.name = name
        self.value = value
        self.arch_mask = arch_mask
        self.canonical = canonical
        self.bit_pattern = bit_pattern


class KraidIsaEnum:
    def __init__(self, name: str, arch_mask: int):
        self.name = name
        self.arch_mask = arch_mask
        self.values = OrderedDict()  # name -> KraidIsaEnumValue

    def add_value(self, v: KraidIsaEnumValue):
        if v.name in self.values:
            existing = self.values[v.name]
            existing.arch_mask |= v.arch_mask
        else:
            self.values[v.name] = v


class KraidIsaSpec:
    def __init__(self, xml_path: str):
        self.root = ET.parse(xml_path).getroot()
        self.enums = OrderedDict()  # enum_name -> KraidIsaEnum
        self.instructions = []      # list of dicts
        self.mnemonics = []         # sorted unique mnemonics
        self.variants = []          # sorted unique variants
        self._parse()

    def _parse(self):
        for child in self.root:
            if child.tag == "enum":
                ename = child.attrib["name"]
                earch = get_arch_mask(child)
                if ename not in self.enums:
                    self.enums[ename] = KraidIsaEnum(ename, earch)
                else:
                    self.enums[ename].arch_mask |= earch
                en = self.enums[ename]
                for v in child.findall("value"):
                    vname = v.attrib["name"]
                    vval = parse_int(v.attrib["value"])
                    varch = get_arch_mask(v) & earch
                    canon = v.attrib.get("canonical", "1") != "0"
                    bp = parse_int(v.attrib["bit_pattern"]) if "bit_pattern" in v.attrib else None
                    if varch != 0:
                        en.add_value(KraidIsaEnumValue(vname, vval, varch, canon, bp))

        mn_set = set()
        var_set = set()
        for child in self.root:
            if child.tag == "instruction":
                ins = self._parse_instruction(child)
                if ins["arch_mask"] != 0:
                    self.instructions.append(ins)
                    mn_set.add(ins["name"])
                    if ins["variant"]:
                        var_set.add(ins["variant"])

        self.mnemonics = sorted(mn_set)
        self.variants = sorted(var_set)

    def _resolve_exact(self, type_name, exact_str):
        if type_name and type_name in self.enums:
            en = self.enums[type_name]
            if exact_str in en.values:
                return en.values[exact_str].value
        return parse_int(exact_str)

    def _parse_instruction(self, elem: ET.Element) -> dict:
        name = elem.attrib["name"]
        variant = elem.attrib.get("variant", "")
        alias = elem.attrib.get("alias", "")
        alias_variant = elem.attrib.get("alias_variant", "")
        exec_unit = elem.attrib.get("exec_unit", "alu")
        exec_time = int(elem.attrib.get("exec_time", "1"))
        arch_mask = get_arch_mask(elem)
        since = int(elem.attrib.get("since", 9))
        until = int(elem.attrib.get("until", 14))

        phys_fields = []
        virt_fields = []
        reserved_fields = []
        syntax_items = []

        fixed_mask = 0
        fixed_value = 0

        for c in elem:
            if c.tag == "field":
                fname = c.attrib["name"]
                pos = int(c.attrib["pos"])
                width = int(c.attrib["width"])
                ftype = c.attrib.get("type", "uint")
                fmod = c.attrib.get("modifier", "")
                frestrict = c.attrib.get("restrict", "")
                fdefault = c.attrib.get("default", "")
                fexact_str = c.attrib.get("exact", None)
                fexact = self._resolve_exact(ftype, fexact_str) if fexact_str is not None else None
                fexpr = c.find("expression")

                if fexact is not None:
                    fmask = ((1 << width) - 1) << pos
                    fixed_mask |= fmask
                    fixed_value |= ((fexact & ((1 << width) - 1)) << pos)

                phys_fields.append({
                    "name": fname,
                    "pos": pos,
                    "width": width,
                    "type": ftype,
                    "modifier": fmod,
                    "restrict": frestrict.split() if frestrict else [],
                    "default": fdefault,
                    "exact": fexact,
                    "exact_str": fexact_str,
                    "expr": fexpr,
                })
            elif c.tag == "virtual":
                vname = c.attrib["name"]
                vtype = c.attrib["type"]
                vrestrict = c.attrib.get("restrict", "")
                vdefault = c.attrib.get("default", "")
                vexact_str = c.attrib.get("exact", None)
                vexact = self._resolve_exact(vtype, vexact_str) if vexact_str is not None else None
                vexpr = c.find("expression")
                virt_fields.append({
                    "name": vname,
                    "type": vtype,
                    "restrict": vrestrict.split() if vrestrict else [],
                    "default": vdefault,
                    "exact": vexact,
                    "exact_str": vexact_str,
                    "expr": vexpr,
                })
            elif c.tag == "reserved":
                pos = int(c.attrib["pos"])
                width = int(c.attrib["width"])
                rtype = c.attrib["type"]
                reserved_fields.append({"pos": pos, "width": width, "zero": rtype == "zero"})
            elif c.tag == "syntax":
                for sc in c:
                    mods = [m.attrib["name"] for m in sc.findall("modifier")]
                    syntax_items.append({
                        "kind": sc.tag,
                        "name": sc.attrib.get("name", ""),
                        "direction": sc.attrib.get("direction", ""),
                        "mods": mods,
                    })

        return {
            "name": name,
            "variant": variant,
            "alias": alias,
            "alias_variant": alias_variant,
            "exec_unit": exec_unit,
            "exec_time": exec_time,
            "since": since,
            "until": until,
            "arch_mask": arch_mask,
            "fixed_mask": fixed_mask,
            "fixed_value": fixed_value,
            "specificity": bin(fixed_mask).count("1"),
            "phys_fields": phys_fields,
            "virt_fields": virt_fields,
            "reserved_fields": reserved_fields,
            "syntax": syntax_items,
        }


# =============================================================================
# 4. Semantic Classification & SPIR-V Lowering Rule Mapper for All 201 Mnemonics
# =============================================================================
SEMANTIC_TABLE = {
    # Floating-point Arithmetic
    "FADD": ("FP_ARITH", "dst = clamp(round(mod(src0) + mod(src1)))", "OpFAdd (f32 / v2f16)"),
    "FADD_IMM": ("FP_ARITH", "dst = src0 + bitcast<f32>(imm)", "OpFAdd with immediate constant"),
    "FMUL": ("FP_ARITH", "dst = clamp(round(mod(src0) * mod(src1)))", "OpFMul (f32 / v2f16)"),
    "FMA": ("FP_ARITH", "dst = clamp(round(mod(src0) * mod(src1) + mod(src2)))", "GLSL.std.450 Fma / OpFAdd(OpFMul)"),
    "FMIN": ("FP_ARITH", "dst = clamp(ieee_min(mod(src0), mod(src1)))", "GLSL.std.450 NMin / FMin"),
    "FMAX": ("FP_ARITH", "dst = clamp(ieee_max(mod(src0), mod(src1)))", "GLSL.std.450 NMax / FMax"),
    "FABS": ("FP_ARITH", "dst = |src0|", "GLSL.std.450 FAbs"),
    "FNEG": ("FP_ARITH", "dst = -src0", "OpFNegate"),
    "FROUND": ("FP_ARITH", "dst = round_mode(mod(src0))", "GLSL.std.450 RoundEven / Trunc / Floor / Ceil"),
    "FLUSH": ("FP_ARITH", "dst = flush_subnormals(src0)", "OpFAdd(src0, 0.0)"),
    "HADD": ("FP_ARITH", "dst = (src0 + src1) * 0.5", "Half-add average"),
    # Floating-point Scale / Transcendental / SFU
    "FADD_LSCALE": ("FP_SFU", "dst = ldexp(src0, src2) + src1", "OpFAdd(GLSL.std.450 Ldexp(src0, src2), src1)"),
    "FADD_RSCALE": ("FP_SFU", "dst = ldexp(mod(src0) + mod(src1), src3)", "GLSL.std.450 Ldexp(OpFAdd(src0, src1), src3)"),
    "FMUL_RSCALE": ("FP_SFU", "dst = ldexp(mod(src0) * mod(src1), src3)", "GLSL.std.450 Ldexp(OpFMul(src0, src1), src3)"),
    "FMA_RSCALE": ("FP_SFU", "dst = ldexp(fma(mod(src0), mod(src1), mod(src2)), src3)", "GLSL.std.450 Ldexp(Fma(src0, src1, src2), src3)"),
    "FRCP": ("FP_SFU", "dst = 1.0 / mod(src0)", "OpFDiv(1.0, src0)"),
    "FRCP_APPROX": ("FP_SFU", "dst = approx_rcp(src0)", "OpFDiv(1.0, src0)"),
    "FRSQ": ("FP_SFU", "dst = 1.0 / sqrt(mod(src0))", "GLSL.std.450 InverseSqrt"),
    "FRSQ_APPROX": ("FP_SFU", "dst = approx_rsq(src0)", "GLSL.std.450 InverseSqrt"),
    "FSQRT": ("FP_SFU", "dst = sqrt(mod(src0))", "GLSL.std.450 Sqrt"),
    "FEXPF": ("FP_SFU", "dst = exp2(src0)", "GLSL.std.450 Exp2"),
    "FEXP_TABLE": ("FP_SFU", "dst = exp2_table_lookup(src0)", "GLSL.std.450 Exp2"),
    "FEXP": ("FP_SFU", "dst = exp2(src0)", "GLSL.std.450 Exp2"),
    "FPOW": ("FP_SFU", "dst = pow(src0, src1)", "GLSL.std.450 Pow"),
    "FPOW_SC_DET": ("FP_SFU", "dst = pow_scale_detect(src0, src1)", "GLSL.std.450 Pow helper"),
    "FPOW_SC_APPLY": ("FP_SFU", "dst = pow_scale_apply(src0, src1)", "GLSL.std.450 Pow helper"),
    "FLOG": ("FP_SFU", "dst = log2(src0)", "GLSL.std.450 Log2"),
    "FLOGD": ("FP_SFU", "dst = log2_mantissa_poly(src0)", "GLSL.std.450 Log2"),
    "FLOG_TABLE": ("FP_SFU", "dst = log2_table(src0)", "GLSL.std.450 Log2"),
    "LOGB": ("FP_SFU", "dst = floor(log2(|src0|))", "GLSL.std.450 FrexpStruct exponent"),
    "ILOGB": ("FP_SFU", "dst = int(floor(log2(|src0|)))", "GLSL.std.450 FrexpStruct exponent"),
    "LDEXP": ("FP_SFU", "dst = src0 * 2^(src1)", "GLSL.std.450 Ldexp"),
    "FREXPE": ("FP_SFU", "dst = frexp_exp(src0)", "GLSL.std.450 FrexpStruct.exp"),
    "FREXPM": ("FP_SFU", "dst = frexp_mantissa(src0)", "GLSL.std.450 FrexpStruct.significand"),
    "FSIN_TABLE": ("FP_SFU", "dst = sin(src0 * 2*pi)", "GLSL.std.450 Sin"),
    "FCOS_TABLE": ("FP_SFU", "dst = cos(src0 * 2*pi)", "GLSL.std.450 Cos"),
    "FSINCOS": ("FP_SFU", "dst = sincos(src0)", "GLSL.std.450 Sin / Cos"),
    "FSINCOS_OFFSET": ("FP_SFU", "dst = sincos_phase_offset(src0)", "Phase reduction for Sin/Cos"),
    "FATAN_ASSIST": ("FP_SFU", "dst = atan_assist(src0, src1)", "GLSL.std.450 Atan2"),
    "FATAN_TABLE": ("FP_SFU", "dst = atan_table(src0)", "GLSL.std.450 Atan"),
    "FRCBRT_APPROX_A": ("FP_SFU", "dst = rcbrt_step_a(src0)", "Pow(src0, -1/3)"),
    "FRCBRT_APPROX_B": ("FP_SFU", "dst = rcbrt_step_b(src0)", "Pow(src0, -1/3)"),
    "FRCBRT_APPROX_C": ("FP_SFU", "dst = rcbrt_step_c(src0)", "Pow(src0, -1/3)"),
    "FMUL_SLICE": ("FP_SFU", "dst = slice_mul(src0, src1)", "OpFMul"),
    "FMUL_CSLICE": ("FP_SFU", "dst = cslice_mul(src0, src1)", "OpFMul"),
    "FPCLASS": ("FP_SFU", "dst = classify_fp(src0)", "OpIsNan / OpIsInf"),
    # Cube Map Helpers
    "CUBEFACE1": ("CUBE_HELPER", "dst = max(|x|, |y|, |z|)", "Cube map major axis magnitude"),
    "CUBEFACE2": ("CUBE_HELPER", "dst = cube_face_index(x, y, z)", "Cube map face index 0..5"),
    "CUBE_SSEL": ("CUBE_HELPER", "dst = select_cube_s(z, x, y, face)", "Cube map S coordinate selection"),
    "CUBE_TSEL": ("CUBE_HELPER", "dst = select_cube_t(y, z, face)", "Cube map T coordinate selection"),
    # Integer Arithmetic
    "IADD": ("INT_ARITH", "dst = sat(widen(src0) + widen(src1))", "OpIAdd (with optional saturation)"),
    "IADD_IMM": ("INT_ARITH", "dst = src0 + imm32", "OpIAdd(src0, OpConstant(imm32))"),
    "IADDC": ("INT_ARITH", "dst = src0 + src1 + carry", "OpIAddCarry"),
    "ISUB": ("INT_ARITH", "dst = sat(widen(src0) - widen(src1))", "OpISub (with optional saturation)"),
    "ISUBB": ("INT_ARITH", "dst = src0 - src1 - borrow", "OpISubBorrow"),
    "IMUL": ("INT_ARITH", "dst = low_bits(widen(src0) * widen(src1))", "OpIMul"),
    "IABS": ("INT_ARITH", "dst = |src0|", "GLSL.std.450 SAbs"),
    "IDPADD": ("INT_ARITH", "dst = dot4_i8(src0, src1) + src2", "OpSDot / OpUDot + OpIAdd"),
    "SHADD": ("INT_ARITH", "dst = src0 + (widen(src1) << shift)", "OpIAdd(src0, OpShiftLeftLogical(src1, shift))"),
    "SEG_ADD": ("INT_ARITH", "dst = segment_address_add(src0, src1)", "64-bit pointer + offset OpIAdd"),
    "SEG_SUB": ("INT_ARITH", "dst = segment_address_sub(src0, src1)", "64-bit pointer - offset OpISub"),
    "ADR": ("INT_ARITH", "dst = (pc + 8) + sext(imm32)", "PC-relative address calculation"),
    # Bitwise, Shift & Logic
    "AND": ("BITWISE_SHIFT", "dst = not_result(widen(src0) & not2(src2))", "OpBitwiseAnd (+ optional OpNot)"),
    "AND_IMM": ("BITWISE_SHIFT", "dst = src0 & imm32", "OpBitwiseAnd(src0, imm32)"),
    "OR": ("BITWISE_SHIFT", "dst = not_result(widen(src0) | not2(src2))", "OpBitwiseOr (+ optional OpNot)"),
    "OR_IMM": ("BITWISE_SHIFT", "dst = src0 | imm32", "OpBitwiseOr(src0, imm32)"),
    "XOR": ("BITWISE_SHIFT", "dst = not_result(widen(src0) ^ not2(src2))", "OpBitwiseXor (+ optional OpNot)"),
    "XOR_IMM": ("BITWISE_SHIFT", "dst = src0 ^ imm32", "OpBitwiseXor(src0, imm32)"),
    "NOT": ("BITWISE_SHIFT", "dst = ~widen(src0)", "OpNot"),
    "NOT_OLD": ("BITWISE_SHIFT", "dst = ~widen(src0)", "OpNot"),
    "LSHIFT": ("BITWISE_SHIFT", "dst = not_result(widen(src0) << (src1 & mask))", "OpShiftLeftLogical"),
    "LSHIFT_AND": ("BITWISE_SHIFT", "dst = not_result((widen(src0) << src1) & not2(src2))", "OpBitwiseAnd(OpShiftLeftLogical(src0, src1), src2)"),
    "LSHIFT_OR": ("BITWISE_SHIFT", "dst = not_result((widen(src0) << src1) | not2(src2))", "OpBitwiseOr(OpShiftLeftLogical(src0, src1), src2)"),
    "LSHIFT_XOR": ("BITWISE_SHIFT", "dst = not_result((widen(src0) << src1) ^ not2(src2))", "OpBitwiseXor(OpShiftLeftLogical(src0, src1), src2)"),
    "RSHIFT": ("BITWISE_SHIFT", "dst = not_result(widen(src0) >>u (src1 & mask))", "OpShiftRightLogical"),
    "RSHIFT_AND": ("BITWISE_SHIFT", "dst = not_result((widen(src0) >>u src1) & not2(src2))", "OpBitwiseAnd(OpShiftRightLogical(src0, src1), src2)"),
    "RSHIFT_OR": ("BITWISE_SHIFT", "dst = not_result((widen(src0) >>u src1) | not2(src2))", "OpBitwiseOr(OpShiftRightLogical(src0, src1), src2)"),
    "RSHIFT_XOR": ("BITWISE_SHIFT", "dst = not_result((widen(src0) >>u src1) ^ not2(src2))", "OpBitwiseXor(OpShiftRightLogical(src0, src1), src2)"),
    "ARSHIFT": ("BITWISE_SHIFT", "dst = not_result(widen(src0) >>s (src1 & mask))", "OpShiftRightArithmetic"),
    "ARSHIFT_AND": ("BITWISE_SHIFT", "dst = not_result((widen(src0) >>s src1) & not2(src2))", "OpBitwiseAnd(OpShiftRightArithmetic(src0, src1), src2)"),
    "ARSHIFT_OR": ("BITWISE_SHIFT", "dst = not_result((widen(src0) >>s src1) | not2(src2))", "OpBitwiseOr(OpShiftRightArithmetic(src0, src1), src2)"),
    "ARSHIFT_XOR": ("BITWISE_SHIFT", "dst = not_result((widen(src0) >>s src1) ^ not2(src2))", "OpBitwiseXor(OpShiftRightArithmetic(src0, src1), src2)"),
    "LROT": ("BITWISE_SHIFT", "dst = rotl(src0, src1)", "(src0 << s) | (src0 >> (32 - s))"),
    "LROT_AND": ("BITWISE_SHIFT", "dst = rotl(src0, src1) & not2(src2)", "Rotate-left + OpBitwiseAnd"),
    "LROT_OR": ("BITWISE_SHIFT", "dst = rotl(src0, src1) | not2(src2)", "Rotate-left + OpBitwiseOr"),
    "LROT_XOR": ("BITWISE_SHIFT", "dst = rotl(src0, src1) ^ not2(src2)", "Rotate-left + OpBitwiseXor"),
    "RROT": ("BITWISE_SHIFT", "dst = rotr(src0, src1)", "(src0 >> s) | (src0 << (32 - s))"),
    "RROT_AND": ("BITWISE_SHIFT", "dst = rotr(src0, src1) & not2(src2)", "Rotate-right + OpBitwiseAnd"),
    "RROT_OR": ("BITWISE_SHIFT", "dst = rotr(src0, src1) | not2(src2)", "Rotate-right + OpBitwiseOr"),
    "RROT_XOR": ("BITWISE_SHIFT", "dst = rotr(src0, src1) ^ not2(src2)", "Rotate-right + OpBitwiseXor"),
    "BITREV": ("BITWISE_SHIFT", "dst = reverse_bits(src0)", "OpBitReverse"),
    "CLZ": ("BITWISE_SHIFT", "dst = count_leading_zeros(src0)", "31 - GLSL.std.450 FindUMsb(src0)"),
    "POPCOUNT": ("BITWISE_SHIFT", "dst = count_ones(src0)", "OpBitCount"),
    "MUX": ("BITWISE_SHIFT", "dst = (src0 & src2) | (src1 & ~src2)", "Bitwise multiplex: (src0 & mask) | (src1 & ~mask)"),
    # Comparisons & Conditional Selects
    "FCMP": ("CMP_SELECT", "dst = result_type(cmp_op(mod(src0), mod(src1)))", "OpFOrd* / OpFUnord* -> OpSelect(mask, 1, 0)"),
    "FCMP_AND": ("CMP_SELECT", "dst = result_type(cmp_op(src0, src1)) & src2", "Float compare combined with AND src2"),
    "FCMP_OR": ("CMP_SELECT", "dst = result_type(cmp_op(src0, src1)) | src2", "Float compare combined with OR src2"),
    "ICMP": ("CMP_SELECT", "dst = result_type(cmp_op(widen(src0), widen(src1)))", "OpIEqual / OpSLessThan / OpULessThan -> OpSelect"),
    "ICMP_AND": ("CMP_SELECT", "dst = result_type(cmp_op(src0, src1)) & src2", "Integer compare combined with AND src2"),
    "ICMP_OR": ("CMP_SELECT", "dst = result_type(cmp_op(src0, src1)) | src2", "Integer compare combined with OR src2"),
    "ICMP_MULTI": ("CMP_SELECT", "dst = multi_lane_icmp(src0, src1, src2)", "Multi-component integer comparison"),
    "CSEL": ("CMP_SELECT", "dst = cmp_op(src0, src1) ? src2 : src3", "OpSelect(cmp(src0, src1), src2, src3)"),
    # Data Movement, Swizzle & Vector Construction
    "MOV": ("MOVE_PACK", "dst = src0", "Register / staging copy"),
    "MOV_IMM": ("MOVE_PACK", "dst = imm32", "Constant materialization"),
    "SWZ": ("MOVE_PACK", "dst = swizzle(src0)", "Byte/half/word swizzle extraction"),
    "MKVEC": ("MOVE_PACK", "dst = pack_lanes(src0, src1, src2)", "Bitwise lane packing into 32-bit word"),
    # Type Conversions
    "F16_TO_F32": ("CONVERSION", "dst = f32(f16(src0))", "GLSL.std.450 UnpackHalf2x16"),
    "F32_TO_F16": ("CONVERSION", "dst = pack_f16(round(src0), round(src1))", "GLSL.std.450 PackHalf2x16"),
    "V2F32_TO_V2F16": ("CONVERSION", "dst = pack_v2f16(src0, src1)", "GLSL.std.450 PackHalf2x16"),
    "F32_TO_S32": ("CONVERSION", "dst = s32(round(src0))", "OpConvertFToS"),
    "F32_TO_U32": ("CONVERSION", "dst = u32(round(src0))", "OpConvertFToU"),
    "F16_TO_S32": ("CONVERSION", "dst = s32(round(f16(src0)))", "UnpackHalf2x16 + OpConvertFToS"),
    "F16_TO_U32": ("CONVERSION", "dst = u32(round(f16(src0)))", "UnpackHalf2x16 + OpConvertFToU"),
    "S32_TO_F32": ("CONVERSION", "dst = f32(s32(src0))", "OpConvertSToF"),
    "U32_TO_F32": ("CONVERSION", "dst = f32(u32(src0))", "OpConvertUToF"),
    "S16_TO_F32": ("CONVERSION", "dst = f32(sext16(src0))", "Sign-extend 16->32 + OpConvertSToF"),
    "U16_TO_F32": ("CONVERSION", "dst = f32(zext16(src0))", "Zero-extend 16->32 + OpConvertUToF"),
    "S8_TO_F32": ("CONVERSION", "dst = f32(sext8(src0))", "Sign-extend 8->32 + OpConvertSToF"),
    "U8_TO_F32": ("CONVERSION", "dst = f32(zext8(src0))", "Zero-extend 8->32 + OpConvertUToF"),
    "S16_TO_S32": ("CONVERSION", "dst = sext16(src0)", "OpBitFieldSExtract(src0, offset, 16)"),
    "U16_TO_U32": ("CONVERSION", "dst = zext16(src0)", "OpBitFieldUExtract(src0, offset, 16)"),
    "S8_TO_S32": ("CONVERSION", "dst = sext8(src0)", "OpBitFieldSExtract(src0, offset, 8)"),
    "U8_TO_U32": ("CONVERSION", "dst = zext8(src0)", "OpBitFieldUExtract(src0, offset, 8)"),
    "V2F16_TO_V2S16": ("CONVERSION", "dst = pack_v2s16(s16(f16_lo), s16(f16_hi))", "UnpackHalf2x16 -> OpConvertFToS -> pack"),
    "V2F16_TO_V2U16": ("CONVERSION", "dst = pack_v2u16(u16(f16_lo), u16(f16_hi))", "UnpackHalf2x16 -> OpConvertFToU -> pack"),
    "V2S16_TO_V2F16": ("CONVERSION", "dst = pack_v2f16(f16(s16_lo), f16(s16_hi))", "Extract s16 -> OpConvertSToF -> PackHalf2x16"),
    "V2U16_TO_V2F16": ("CONVERSION", "dst = pack_v2f16(f16(u16_lo), f16(u16_hi))", "Extract u16 -> OpConvertUToF -> PackHalf2x16"),
    "V2S8_TO_V2F16": ("CONVERSION", "dst = pack_v2f16(f16(s8_0), f16(s8_1))", "Extract s8 -> OpConvertSToF -> PackHalf2x16"),
    "V2U8_TO_V2F16": ("CONVERSION", "dst = pack_v2f16(f16(u8_0), f16(u8_1))", "Extract u8 -> OpConvertUToF -> PackHalf2x16"),
    "V2S8_TO_V2S16": ("CONVERSION", "dst = pack_v2s16(sext8(b0), sext8(b1))", "Byte sign-extension to v2s16"),
    "V2U8_TO_V2U16": ("CONVERSION", "dst = pack_v2u16(zext8(b0), zext8(b1))", "Byte zero-extension to v2u16"),
    # Control Flow & Subgroup / Barriers
    "NOP": ("CONTROL_FLOW", "no-op (applies flow control modifier: wait / reconverge / discard / end)", "Flow control action (OpReturn on .end)"),
    "BRANCH": ("CONTROL_FLOW", "if (cond(src0)) pc = (pc + 8) + offset", "OpSelectionMerge / OpLoopMerge + OpBranchConditional"),
    "JUMP": ("CONTROL_FLOW", "if (cond(src0)) pc = target_va", "Indirect or absolute branch"),
    "JUMP_EX": ("CONTROL_FLOW", "exception/blend jump to target_va", "Blend shader / exception call"),
    "BARRIER": ("CONTROL_FLOW", "workgroup execution & memory barrier", "OpControlBarrier(Workgroup, Workgroup, AcquireRelease|WorkgroupMemory)"),
    "DISCARD": ("CONTROL_FLOW", "if (cond) discard_fragment()", "OpDemoteToHelperInvocation / OpKill"),
    "WMASK": ("CONTROL_FLOW", "dst = subgroup_ballot(src0)", "OpSubgroupBallotKHR"),
    "CLPER": ("SUBGROUP", "dst = subgroup_shuffle(src0, lane_op(src1))", "Quad/subgroup derivative shuffle (OpDPdx/OpDPdy/OpGroupNonUniformQuadBroadcast)"),
    "CLPER_OR": ("SUBGROUP", "dst = subgroup_shuffle(src0, lane_op(src1)) | src2", "Quad shuffle + OR"),
    "MMUL": ("MATRIX", "sr_dst = matrix_multiply(sr_src, src0)", "Subgroup matrix multiply-accumulate"),
    # Memory Load / Store / Atomics / Buffer Descriptors
    "LOAD": ("MEMORY_LS", "sr_dst = mem[addr64 + sext(offset)]", "PhysicalStorageBuffer64 / SSBO OpLoad"),
    "LD_CHECKED": ("MEMORY_LS", "sr_dst = (offset < bound) ? mem[addr64 + offset] : 0", "Bounds-checked memory load"),
    "LD_CHECKED_IMM": ("MEMORY_LS", "sr_dst = (imm_offset < bound) ? mem[addr64 + imm_offset] : 0", "Bounds-checked immediate memory load"),
    "LD_MULTI": ("MEMORY_LS", "sr_dst[0..N] = mem[addr64 + offset]", "Multi-register burst load"),
    "LD_PKA": ("MEMORY_LS", "sr_dst = mem[unpack_pka56(ptr) + offset]", "Packaged 56-bit pointer load"),
    "LD_CVT": ("MEMORY_LS", "sr_dst = format_convert(mem[addr64 + offset], fmt)", "Formatted load with conversion"),
    "STORE": ("MEMORY_LS", "mem[addr64 + sext(offset)] = sr_src", "PhysicalStorageBuffer64 / SSBO OpStore (or Position/Varying output on istream/estream)"),
    "ST_CHECKED": ("MEMORY_LS", "if (offset < bound) mem[addr64 + offset] = sr_src", "Bounds-checked memory store"),
    "ST_CHECKED_IMM": ("MEMORY_LS", "if (imm_offset < bound) mem[addr64 + imm_offset] = sr_src", "Bounds-checked immediate memory store"),
    "ST_MULTI": ("MEMORY_LS", "mem[addr64 + offset] = sr_src[0..N]", "Multi-register burst store"),
    "ST_CVT": ("MEMORY_LS", "mem[addr64 + offset] = format_encode(sr_src, fmt)", "Formatted store with pixel conversion"),
    "LEA_BUF": ("MEMORY_LS", "sr_dst = resolve_buffer_va(table, index, offset)", "Compute 64-bit buffer VA from SRT Buffer descriptor"),
    "LEA_BUF_IMM": ("MEMORY_LS", "sr_dst = resolve_buffer_va(table, imm_index, offset)", "Compute 64-bit buffer VA (or Position/Varying FIFO slot)"),
    "LEA_PKA": ("MEMORY_LS", "sr_dst = unpack_pka56(ptr) + offset", "Effective address from packaged pointer"),
    "ATOM": ("MEMORY_ATOMIC", "mem[addr64 + offset] = atom_op(mem[addr64 + offset], sr_src)", "OpAtomicIAdd / SMin / SMax / UMin / UMax / And / Or / Xor / Exchange"),
    "ATOM_RETURN": ("MEMORY_ATOMIC", "sr_dst = atomic_rmw(mem[addr64 + offset], atom_op, sr_src)", "OpAtomic* returning old value"),
    "ATOM1": ("MEMORY_ATOMIC", "mem[addr64 + offset] = atom1_op(mem[addr64 + offset], 1)", "OpAtomicIIncrement / IDecrement"),
    "ATOM1_RETURN": ("MEMORY_ATOMIC", "sr_dst = atomic_rmw(mem[addr64 + offset], atom1_op, 1)", "OpAtomicIIncrement / IDecrement returning old value"),
    "ACMPXCHG": ("MEMORY_ATOMIC", "sr_dst = atomic_cas(mem[addr64 + offset], sr_src[0], sr_src[1])", "OpAtomicCompareExchange"),
    "AXCHG": ("MEMORY_ATOMIC", "sr_dst = atomic_xchg(mem[addr64 + offset], sr_src)", "OpAtomicExchange"),
    # Vertex Attributes & Varyings
    "LD_ATTR": ("VERTEX_VARYING", "sr_dst = fetch_vertex_attr(table, attr_idx, vertex_id, instance_id)", "OpLoad from Input attribute variable (location = attr_idx)"),
    "LD_ATTR_IMM": ("VERTEX_VARYING", "sr_dst = fetch_vertex_attr(table, imm_idx, vertex_id, instance_id)", "OpLoad from Input attribute variable (location = imm_idx)"),
    "LEA_ATTR": ("VERTEX_VARYING", "sr_dst = vertex_attr_address(table, attr_idx, vertex_id, instance_id)", "Effective address of vertex attribute"),
    "LEA_ATTR_IMM": ("VERTEX_VARYING", "sr_dst = vertex_attr_address(table, imm_idx, vertex_id, instance_id)", "Effective address of immediate vertex attribute"),
    "LD_VAR": ("VERTEX_VARYING", "sr_dst = interpolate_varying(index, sample_mode, perspective)", "OpLoad from Fragment Input varying"),
    "LD_VAR_IMM": ("VERTEX_VARYING", "sr_dst = interpolate_varying(imm_index, sample_mode, perspective)", "OpLoad from Fragment Input varying"),
    "LD_VAR_FLAT": ("VERTEX_VARYING", "sr_dst = read_flat_varying(index)", "OpLoad from Flat Fragment Input varying"),
    "LD_VAR_FLAT_IMM": ("VERTEX_VARYING", "sr_dst = read_flat_varying(imm_index)", "OpLoad from Flat Fragment Input varying"),
    "LD_VAR_BUF": ("VERTEX_VARYING", "sr_dst = interpolate_varying_buf(offset, sample_mode)", "OpLoad from Fragment Input varying at byte offset"),
    "LD_VAR_BUF_IMM": ("VERTEX_VARYING", "sr_dst = interpolate_varying_buf(imm_offset, sample_mode)", "OpLoad from Fragment Input varying at byte offset"),
    "LD_VAR_BUF_FLAT": ("VERTEX_VARYING", "sr_dst = read_flat_varying_buf(offset)", "OpLoad from Flat Fragment Input varying at byte offset"),
    "LD_VAR_BUF_FLAT_IMM": ("VERTEX_VARYING", "sr_dst = read_flat_varying_buf(imm_offset)", "OpLoad from Flat Fragment Input varying at byte offset"),
    "LD_VAR_SPECIAL": ("VERTEX_VARYING", "sr_dst = read_special_varying(special_id)", "BuiltIn FragCoord.z / FragCoord.w / PointCoord / FrontFacing"),
    # Texture & Image Operations
    "TEX_SINGLE": ("TEXTURE", "sr_dst = sample_texture(tex_idx, samp_idx, coords, lod_mode)", "OpImageSampleImplicitLod / OpImageSampleExplicitLod"),
    "TEX_GATHER": ("TEXTURE", "sr_dst = texture_gather(tex_idx, samp_idx, coords, comp)", "OpImageGather / OpImageDrefGather"),
    "TEX_GRADIENT": ("TEXTURE", "sr_dst = sample_texture_grad(tex_idx, samp_idx, coords, dPdx, dPdy)", "OpImageSampleExplicitLod Grad"),
    "TEX_FETCH": ("TEXTURE", "sr_dst = texel_fetch(tex_idx, icords, lod_or_sample)", "OpImageFetch"),
    "TEX_DUAL": ("TEXTURE", "sr_dst0, sr_dst1 = dual_texture_sample(tex0, tex1, coords)", "Two OpImageSample* calls"),
    "TEX_WEIGHTS": ("TEXTURE", "sr_dst = bilinear_weights(coords)", "Fractional texel weights"),
    "VAR_TEX_SINGLE": ("TEXTURE", "sr_dst = sample_texture(tex_idx, samp_idx, interpolate_varying(var_idx))", "Fused OpLoad(Input varying) + OpImageSampleImplicitLod"),
    "VAR_TEX_GATHER": ("TEXTURE", "sr_dst = texture_gather(tex_idx, samp_idx, interpolate_varying(var_idx))", "Fused OpLoad(Input varying) + OpImageGather"),
    "VAR_TEX_GRADIENT": ("TEXTURE", "sr_dst = sample_texture_grad(tex_idx, samp_idx, interpolate_varying(var_idx))", "Fused OpLoad(Input varying) + OpImageSampleExplicitLod Grad"),
    "VAR_TEX_DUAL": ("TEXTURE", "sr_dst = dual_sample(interpolate_varying(var_idx))", "Fused varying + dual texture sample"),
    "VAR_TEX_BUF_SINGLE": ("TEXTURE", "sr_dst = sample_texture(tex_idx, samp_idx, interpolate_varying_buf(off))", "Fused varying buffer load + OpImageSampleImplicitLod"),
    "VAR_TEX_BUF_GATHER": ("TEXTURE", "sr_dst = texture_gather(tex_idx, samp_idx, interpolate_varying_buf(off))", "Fused varying buffer load + OpImageGather"),
    "VAR_TEX_BUF_GRADIENT": ("TEXTURE", "sr_dst = sample_texture_grad(tex_idx, samp_idx, interpolate_varying_buf(off))", "Fused varying buffer load + OpImageSampleExplicitLod Grad"),
    "VAR_TEX_BUF_DUAL": ("TEXTURE", "sr_dst = dual_sample(interpolate_varying_buf(off))", "Fused varying buffer load + dual texture sample"),
    "LD_TEX": ("TEXTURE", "sr_dst = image_load(image_idx, coords)", "OpImageRead"),
    "LD_TEX_IMM": ("TEXTURE", "sr_dst = image_load(imm_image_idx, coords)", "OpImageRead"),
    "LEA_TEX": ("TEXTURE", "sr_dst = image_texel_address(image_idx, coords)", "Compute texel memory address from Texture+Plane descriptor"),
    "LEA_TEX_IMM": ("TEXTURE", "sr_dst = image_texel_address(imm_image_idx, coords)", "Compute texel memory address from Texture+Plane descriptor"),
    # Fragment Output, Tile Buffer & System
    "ATEST": ("FRAGMENT_OUT", "coverage_out = update_coverage(coverage_in, alpha, atest_datum)", "Alpha-to-coverage / discard test"),
    "ZS_EMIT": ("FRAGMENT_OUT", "emit_depth_stencil(z, stencil, coverage)", "OpStore BuiltIn FragDepth / FragStencilRefEXT"),
    "BLEND": ("FRAGMENT_OUT", "blend_to_rt(target_rt, sr_src_color, coverage, blend_desc)", "OpStore to Fragment Output location = target_rt"),
    "LD_TILE": ("FRAGMENT_OUT", "sr_dst = read_tilebuffer(rt_index, sample, pixel_fmt)", "Subpass / framebuffer fetch read"),
    "ST_TILE": ("FRAGMENT_OUT", "write_tilebuffer(rt_index, sample, pixel_fmt, sr_src)", "Direct tilebuffer write to Fragment Output"),
    "LD_GCLK": ("SYSTEM", "sr_dst = read_gpu_timestamp_counter()", "OpReadClockKHR"),
    "DOORBELL": ("SYSTEM", "ring_hw_doorbell(src0)", "Hardware doorbell no-op in shader"),
    "EUREKA": ("SYSTEM", "trace_marker()", "Hardware trace marker no-op"),
    "KABOOM": ("SYSTEM", "trap()", "Hardware trap"),
}


# =============================================================================
# 5. Emit C++ Expression Code from XML <expression> AST
# =============================================================================
def compile_xml_expr(expr_el: ET.Element, enums: dict) -> str:
    if "operator" in expr_el.attrib:
        op = expr_el.attrib["operator"]
        children = list(expr_el)
        if op == "if":
            # [cond0, val0, cond1, val1, ..., else_val]
            parts = []
            i = 0
            while i + 1 < len(children):
                c_str = compile_xml_expr(children[i], enums)
                v_str = compile_xml_expr(children[i + 1], enums)
                parts.append(f"(({c_str}) ? ({v_str}) : ")
                i += 2
            else_str = compile_xml_expr(children[-1], enums)
            return "".join(parts) + f"({else_str})" + (")" * len(parts))
        elif op in ("bitwise_and", "bitwise_or", "add"):
            sym = {"bitwise_and": "&", "bitwise_or": "|", "add": "+"}[op]
            sub = [f"({compile_xml_expr(c, enums)})" for c in children]
            return "(" + f" {sym} ".join(sub) + ")"
        elif op in (
            "bitwise_xor", "equal", "not_equal", "less_equal", "less_than",
            "greater_than", "greater_equal", "logical_and", "sub", "rshift", "lshift"
        ):
            sym = {
                "bitwise_xor": "^", "equal": "==", "not_equal": "!=",
                "less_equal": "<=", "less_than": "<", "greater_than": ">",
                "greater_equal": ">=", "logical_and": "&&", "sub": "-",
                "rshift": ">>", "lshift": "<<"
            }[op]
            a = compile_xml_expr(children[0], enums)
            b = compile_xml_expr(children[1], enums)
            return f"(({a}) {sym} ({b}))"
        elif op == "inside":
            target = compile_xml_expr(children[0], enums)
            checks = [f"(({target}) == ({compile_xml_expr(c, enums)}))" for c in children[1:]]
            return "(" + " || ".join(checks) + ")"
        elif op == "countones":
            x = compile_xml_expr(children[0], enums)
            return f"uint32_t(__builtin_popcount({x}))"
        elif op == "mask":
            x = compile_xml_expr(children[0], enums)
            return f"expr_mask_u32({x})"
        elif op == "slice":
            v = compile_xml_expr(children[0], enums)
            hi = compile_xml_expr(children[1], enums)
            lo = compile_xml_expr(children[2], enums)
            return f"expr_slice_u32({v}, {hi}, {lo})"
        else:
            raise ValueError(f"Unknown expression operator: {op}")
    elif "identifier" in expr_el.attrib:
        ident = snake_case(expr_el.attrib["identifier"])
        return f"f_{ident}"
    elif "literal" in expr_el.attrib:
        lit = expr_el.attrib["literal"]
        tname = expr_el.attrib.get("type") or expr_el.attrib.get("enum")
        if tname and tname in enums and lit in enums[tname].values:
            return f"{enums[tname].values[lit].value}u"
        return f"{parse_int(lit)}u"
    else:
        raise ValueError(f"Invalid expression element: {ET.tostring(expr_el)}")


# =============================================================================
# 6. Generate valhall_isa_generated.h and valhall_isa_generated.cpp
# =============================================================================
def generate_isa_files(
    legacy_spec: ValhallLegacySpec,
    kraid_spec: KraidIsaSpec,
    kraid_sem: KraidSemanticsExtractor,
    out_dir: str,
):
    h_path = os.path.join(out_dir, "valhall_isa_generated.h")
    cpp_path = os.path.join(out_dir, "valhall_isa_generated.cpp")

    # Build header
    h_lines = [
        "// AUTO-GENERATED BY tools/gen_valhall_spec.py -- DO NOT EDIT MANUALLY",
        "// Extracted from specs/valhall/ISA.xml and specs/kraid/isa-v9-v14.xml",
        "#pragma once",
        "#include <cstdint>",
        "#include <cstddef>",
        "#include <string>",
        "#include <vector>",
        "#include <map>",
        "",
        "namespace valhall {",
        "",
        "// Architecture version constants (Valhall v9..v14)",
        "constexpr uint8_t kMinValhallArch = 9;",
        "constexpr uint8_t kMaxValhallArch = 14;",
        "",
        "enum class Mnemonic : uint16_t {",
        "    INVALID = 0,",
    ]
    for i, mn in enumerate(kraid_spec.mnemonics, start=1):
        h_lines.append(f"    {c_ident(mn)} = {i},")
    h_lines.append("};")
    h_lines.append("")
    h_lines.append("const char* mnemonic_to_string(Mnemonic m);")
    h_lines.append("")

    h_lines.append("enum class Variant : uint8_t {")
    h_lines.append("    NONE = 0,")
    for i, var in enumerate(kraid_spec.variants, start=1):
        h_lines.append(f"    {c_ident(var.upper())} = {i},")
    h_lines.append("};")
    h_lines.append("const char* variant_to_string(Variant v);")
    h_lines.append("")

    # Emit all 219 enums from isa-v9-v14.xml
    h_lines.append("// ============================================================================")
    h_lines.append(f"// All {len(kraid_spec.enums)} Hardware Enums from isa-v9-v14.xml")
    h_lines.append("// ============================================================================")
    for ename, en in kraid_spec.enums.items():
        cpp_enum_name = pascal_case(ename)
        # Check if values are unique
        seen_vals = set()
        unique_vals = True
        for v in en.values.values():
            if v.value in seen_vals:
                unique_vals = False
                break
            seen_vals.add(v.value)

        h_lines.append(f"enum class {cpp_enum_name} : uint8_t {{")
        if unique_vals:
            for vname, v in en.values.items():
                h_lines.append(f"    {c_ident(vname)} = {v.value},")
        else:
            for idx, (vname, v) in enumerate(en.values.items()):
                h_lines.append(f"    {c_ident(vname)} = {idx}, // encoded={v.value}")
        h_lines.append("};")
        h_lines.append(f"bool try_decode_{snake_case(ename)}(uint8_t raw, uint8_t arch, {cpp_enum_name}* out);")
        h_lines.append(f"const char* to_string({cpp_enum_name} v);")
        h_lines.append("")

    # Emit Small Constant table & FAU Special Page resolvers
    h_lines.extend([
        "struct SmallConstantEntry {",
        "    uint8_t index;",
        "    uint32_t bit_pattern;",
        "    const char* symbolic_name;",
        "    const char* description;",
        "    uint32_t arch_mask;",
        "};",
        "",
        "const SmallConstantEntry* lookup_small_constant(uint8_t idx, uint8_t arch = 11);",
        "const char* lookup_fau_special_name(uint8_t page, uint8_t idx32, uint8_t arch = 11);",
        "",
        "enum class SourceKind : uint8_t {",
        "    Register = 0,",
        "    FauUniform = 1,",
        "    SmallConstant = 2,",
        "    FauSpecial = 3,",
        "};",
        "",
        "struct DecodedSource {",
        "    SourceKind kind = SourceKind::Register;",
        "    uint8_t raw_encoding = 0;",
        "    bool is_64bit = false;",
        "    uint8_t reg_num = 0;           // 0..63 for Register or FAU 32-bit offset",
        "    bool discard_last_use = false; // ^ marker",
        "    uint8_t fau_page = 0;          // 0..3",
        "    uint8_t fau_index64 = 0;       // 64-bit FAU slot index",
        "    bool word_select = false;      // .w0 (false) or .w1 (true)",
        "    uint32_t imm_bits = 0;         // resolved 32-bit immediate for SmallConstant",
        "    std::string symbolic_name;     // e.g. k0, blend_descriptor_0, warp_id",
        "    std::string swizzle_mod;       // e.g. h0, h1, b0, w0, h00",
        "    bool abs = false;",
        "    bool neg = false;",
        "    bool not_mod = false;",
        "};",
        "",
        "bool decode_source_operand(",
        "    uint8_t raw, uint8_t arch, uint8_t fau_page, bool is64, bool fau32,",
        "    DecodedSource* out);",
        "",
        "struct DecodedStaging {",
        "    bool present = false;",
        "    bool is_input = false;",
        "    uint8_t base_reg = 0;",
        "    uint8_t count = 0;",
        "};",
        "",
        "enum class SemanticCategory : uint8_t {",
        "    FP_ARITH = 0,",
        "    FP_SFU = 1,",
        "    CUBE_HELPER = 2,",
        "    INT_ARITH = 3,",
        "    BITWISE_SHIFT = 4,",
        "    CMP_SELECT = 5,",
        "    MOVE_PACK = 6,",
        "    CONVERSION = 7,",
        "    CONTROL_FLOW = 8,",
        "    SUBGROUP = 9,",
        "    MATRIX = 10,",
        "    MEMORY_LS = 11,",
        "    MEMORY_ATOMIC = 12,",
        "    VERTEX_VARYING = 13,",
        "    TEXTURE = 14,",
        "    FRAGMENT_OUT = 15,",
        "    SYSTEM = 16,",
        "};",
        "const char* to_string(SemanticCategory c);",
        "",
        "struct DecodedValhallInst {",
        "    bool valid = false;",
        "    uint64_t raw_word = 0;",
        "    uint64_t pc = 0;",
        "    uint8_t arch = 11;",
        "    uint16_t spec_index = 0;",
        "    Mnemonic mnemonic = Mnemonic::INVALID;",
        "    Variant variant = Variant::NONE;",
        "    Mnemonic canonical_mnemonic = Mnemonic::INVALID;",
        "    Variant canonical_variant = Variant::NONE;",
        "    SemanticCategory category = SemanticCategory::FP_ARITH;",
        "    const char* exec_unit = \"\";",
        "    FlowControlM flow = FlowControlM::none;",
        "    uint8_t fau_page = 0;",
        "    bool has_dest = false;",
        "    bool dest_is_64bit = false;",
        "    uint8_t dest_reg = 0;",
        "    std::string dest_mod;          // e.g. h0, h1, b0..b3",
        "    DecodedStaging sr_write;",
        "    DecodedStaging sr_read;",
        "    std::vector<DecodedSource> srcs;",
        "    bool has_imm = false;",
        "    int64_t imm_signed = 0;",
        "    uint64_t imm_unsigned = 0;",
        "    bool has_branch_offset = false;",
        "    int64_t branch_offset_bytes = 0;",
        "    int64_t branch_target_pc = 0;",
        "    std::map<std::string, uint32_t> fields;",
        "    std::map<std::string, std::string> modifier_strs;",
        "    std::string disassembly;",
        "};",
        "",
        "struct InstructionSemanticsSpec {",
        "    uint16_t spec_index;",
        "    Mnemonic mnemonic;",
        "    Variant variant;",
        "    Mnemonic canonical_mnemonic;",
        "    Variant canonical_variant;",
        "    const char* full_name;",
        "    const char* exec_unit;",
        "    uint8_t since_arch;",
        "    uint8_t until_arch;",
        "    uint64_t fixed_mask;",
        "    uint64_t fixed_value;",
        "    uint8_t specificity;",
        "    SemanticCategory category;",
        "    const char* operational_semantics;",
        "    const char* spirv_lowering;",
        "    const char* isa_xml_description;",
        "};",
        "",
        "constexpr size_t kNumValhallInstructionSpecs = " + str(len(kraid_spec.instructions)) + ";",
        "const InstructionSemanticsSpec* get_all_instruction_specs(size_t* out_count = nullptr);",
        "DecodedValhallInst decode_valhall_instruction(",
        "    uint64_t word, uint8_t arch = 11, uint64_t pc = 0, bool fau32 = false);",
        "std::vector<DecodedValhallInst> decode_valhall_shader(",
        "    const uint8_t* bytes, size_t size_bytes, uint8_t arch = 11, bool fau32 = false);",
        "",
        "} // namespace valhall",
        "",
    ])

    with open(h_path, "w", encoding="utf-8") as f:
        f.write("\n".join(h_lines))

    # Build .cpp implementation
    cpp_lines = [
        "// AUTO-GENERATED BY tools/gen_valhall_spec.py -- DO NOT EDIT MANUALLY",
        "#include \"valhall_isa_generated.h\"",
        "#include <cstdio>",
        "#include <cstring>",
        "#include <sstream>",
        "#include <iomanip>",
        "#include <algorithm>",
        "",
        "namespace valhall {",
        "",
        "static inline uint32_t expr_mask_u32(uint32_t bits) {",
        "    return (bits < 32) ? ((1u << bits) - 1u) : ~0u;",
        "}",
        "static inline uint32_t expr_slice_u32(uint32_t x, uint32_t hi, uint32_t lo) {",
        "    return (x >> lo) & expr_mask_u32(hi + 1u - lo);",
        "}",
        "static inline int32_t sign_ext32(uint32_t val, uint8_t bits) {",
        "    if (bits == 0 || bits >= 32) return static_cast<int32_t>(val);",
        "    uint32_t shift = 32u - bits;",
        "    return static_cast<int32_t>(val << shift) >> shift;",
        "}",
        "",
        "const char* mnemonic_to_string(Mnemonic m) {",
        "    switch (m) {",
        "        case Mnemonic::INVALID: return \"INVALID\";",
    ]
    for mn in kraid_spec.mnemonics:
        cpp_lines.append(f"        case Mnemonic::{c_ident(mn)}: return \"{mn}\";")
    cpp_lines.extend([
        "    }",
        "    return \"INVALID\";",
        "}",
        "",
        "const char* variant_to_string(Variant v) {",
        "    switch (v) {",
        "        case Variant::NONE: return \"\";",
    ])
    for var in kraid_spec.variants:
        cpp_lines.append(f"        case Variant::{c_ident(var.upper())}: return \"{var}\";")
    cpp_lines.extend([
        "    }",
        "    return \"\";",
        "}",
        "",
        "const char* to_string(SemanticCategory c) {",
        "    switch (c) {",
        "        case SemanticCategory::FP_ARITH: return \"FP_ARITH\";",
        "        case SemanticCategory::FP_SFU: return \"FP_SFU\";",
        "        case SemanticCategory::CUBE_HELPER: return \"CUBE_HELPER\";",
        "        case SemanticCategory::INT_ARITH: return \"INT_ARITH\";",
        "        case SemanticCategory::BITWISE_SHIFT: return \"BITWISE_SHIFT\";",
        "        case SemanticCategory::CMP_SELECT: return \"CMP_SELECT\";",
        "        case SemanticCategory::MOVE_PACK: return \"MOVE_PACK\";",
        "        case SemanticCategory::CONVERSION: return \"CONVERSION\";",
        "        case SemanticCategory::CONTROL_FLOW: return \"CONTROL_FLOW\";",
        "        case SemanticCategory::SUBGROUP: return \"SUBGROUP\";",
        "        case SemanticCategory::MATRIX: return \"MATRIX\";",
        "        case SemanticCategory::MEMORY_LS: return \"MEMORY_LS\";",
        "        case SemanticCategory::MEMORY_ATOMIC: return \"MEMORY_ATOMIC\";",
        "        case SemanticCategory::VERTEX_VARYING: return \"VERTEX_VARYING\";",
        "        case SemanticCategory::TEXTURE: return \"TEXTURE\";",
        "        case SemanticCategory::FRAGMENT_OUT: return \"FRAGMENT_OUT\";",
        "        case SemanticCategory::SYSTEM: return \"SYSTEM\";",
        "    }",
        "    return \"UNKNOWN\";",
        "}",
        "",
    ])

    # Emit enum decoders & to_string
    for ename, en in kraid_spec.enums.items():
        cpp_enum_name = pascal_case(ename)
        fn_name = f"try_decode_{snake_case(ename)}"

        # Group values by encoded integer value
        by_val = defaultdict(list)
        for v in en.values.values():
            by_val[v.value].append(v)
        for vlist in by_val.values():
            vlist.sort(key=lambda x: (not x.canonical, x.name))

        cpp_lines.append(f"bool {fn_name}(uint8_t raw, uint8_t arch, {cpp_enum_name}* out) {{")
        cpp_lines.append("    if (arch < 9 || arch > 14) return false;")
        cpp_lines.append("    uint32_t arch_bit = 1u << arch;")
        cpp_lines.append("    switch (raw) {")
        for val_int, vlist in sorted(by_val.items()):
            cpp_lines.append(f"        case {val_int}:")
            for v in vlist:
                cpp_lines.append(
                    f"            if (0x{v.arch_mask:x}u & arch_bit) {{ if (out) *out = {cpp_enum_name}::{c_ident(v.name)}; return true; }}"
                )
            cpp_lines.append("            return false;")
        cpp_lines.append("        default: return false;")
        cpp_lines.append("    }")
        cpp_lines.append("}")
        cpp_lines.append("")

        cpp_lines.append(f"const char* to_string({cpp_enum_name} v) {{")
        cpp_lines.append("    switch (v) {")
        for vname in en.values.keys():
            cpp_lines.append(f"        case {cpp_enum_name}::{c_ident(vname)}: return \"{vname}\";")
        cpp_lines.append("    }")
        cpp_lines.append("    return \"\";")
        cpp_lines.append("}")
        cpp_lines.append("")

    # Emit SmallConstantEntry table (combining kraid small_constant_t and valhall/ISA.xml descriptions)
    sc_enum = kraid_spec.enums.get("small_constant_t")
    legacy_desc_by_idx = {idx: desc for (idx, _, desc) in legacy_spec.immediates}
    cpp_lines.append("static const SmallConstantEntry kSmallConstants[] = {")
    if sc_enum:
        for v in sc_enum.values.values():
            desc = legacy_desc_by_idx.get(v.value, "").replace("\\", "\\\\").replace('"', '\\"')
            bp = v.bit_pattern if v.bit_pattern is not None else 0
            cpp_lines.append(
                f"    {{ {v.value}, 0x{bp:08X}u, \"{v.name}\", \"{desc}\", 0x{v.arch_mask:x}u }},"
            )
    cpp_lines.extend([
        "};",
        "",
        "const SmallConstantEntry* lookup_small_constant(uint8_t idx, uint8_t arch) {",
        "    uint32_t arch_bit = 1u << arch;",
        "    for (const auto& e : kSmallConstants) {",
        "        if (e.index == idx && (e.arch_mask & arch_bit)) return &e;",
        "    }",
        "    for (const auto& e : kSmallConstants) {",
        "        if (e.index == idx) return &e;",
        "    }",
        "    return nullptr;",
        "}",
        "",
        "const char* lookup_fau_special_name(uint8_t page, uint8_t idx32, uint8_t arch) {",
        "    switch (page) {",
        "        case 0: {",
        "            FauSpecialIndexPage0T v;",
        "            if (try_decode_fau_special_index_page0_t(idx32, arch, &v)) return to_string(v);",
        "            break;",
        "        }",
        "        case 1: {",
        "            FauSpecialIndexPage1T v;",
        "            if (try_decode_fau_special_index_page1_t(idx32, arch, &v)) return to_string(v);",
        "            break;",
        "        }",
        "        case 3: {",
        "            FauSpecialIndexPage3T v;",
        "            if (try_decode_fau_special_index_page3_t(idx32, arch, &v)) return to_string(v);",
        "            break;",
        "        }",
        "        default: break;",
        "    }",
        "    return nullptr;",
        "}",
        "",
        "bool decode_source_operand(",
        "    uint8_t raw, uint8_t arch, uint8_t fau_page, bool is64, bool fau32,",
        "    DecodedSource* out)",
        "{",
        "    if (!out) return false;",
        "    *out = DecodedSource{};",
        "    out->raw_encoding = raw;",
        "    out->is_64bit = is64;",
        "    out->fau_page = fau_page;",
        "    uint8_t mode = (raw >> 6) & 0x3u;",
        "    uint8_t mode2 = (raw >> 5) & 0x1u;",
        "    if (mode == 0 || mode == 1) {",
        "        out->kind = SourceKind::Register;",
        "        out->reg_num = raw & 0x3Fu;",
        "        out->discard_last_use = (mode == 1);",
        "        return true;",
        "    } else if (mode == 2) {",
        "        out->kind = SourceKind::FauUniform;",
        "        uint8_t idx = raw & 0x3Fu;",
        "        if (fau32) {",
        "            out->reg_num = static_cast<uint8_t>(64u * fau_page + idx);",
        "            out->fau_index64 = out->reg_num >> 1;",
        "            out->word_select = (idx & 1u) != 0;",
        "        } else {",
        "            out->fau_index64 = static_cast<uint8_t>(32u * fau_page + (idx >> 1));",
        "            out->word_select = !is64 && ((idx & 1u) != 0);",
        "            out->reg_num = static_cast<uint8_t>(out->fau_index64 * 2u + (out->word_select ? 1u : 0u));",
        "        }",
        "        return true;",
        "    } else {",
        "        if (mode2 == 0) {",
        "            uint8_t sc_idx = raw & 0x1Fu;",
        "            const SmallConstantEntry* sc = lookup_small_constant(sc_idx, arch);",
        "            if (!sc) return false;",
        "            out->kind = SourceKind::SmallConstant;",
        "            out->reg_num = sc_idx;",
        "            out->imm_bits = sc->bit_pattern;",
        "            out->symbolic_name = sc->symbolic_name;",
        "            return true;",
        "        } else {",
        "            uint8_t idx32 = (raw & 0x1Fu) >> 1;",
        "            bool ws = (raw & 1u) != 0;",
        "            const char* sp_name = lookup_fau_special_name(fau_page, idx32, arch);",
        "            if (!sp_name) return false;",
        "            out->kind = SourceKind::FauSpecial;",
        "            out->reg_num = idx32;",
        "            out->word_select = !is64 && ws;",
        "            out->symbolic_name = sp_name;",
        "            return true;",
        "        }",
        "    }",
        "}",
        "",
    ])

    # Emit InstructionSemanticsSpec table for all 627 instructions
    cpp_lines.append("static const InstructionSemanticsSpec kInstructionSpecs[] = {")
    for idx, ins in enumerate(kraid_spec.instructions):
        name = ins["name"]
        var = ins["variant"]
        canon_mn = ins["alias"] if ins["alias"] else name
        canon_var = ins["alias_variant"] if ins["alias_variant"] else var
        full_name = f"{name}.{var}" if var else name
        cat_str, op_sem, spv_low = SEMANTIC_TABLE.get(
            name,
            SEMANTIC_TABLE.get(canon_mn, ("FP_ARITH", "dst = op(srcs)", "SPIR-V ALU lowering"))
        )
        # Check if kraid/ops.rs has a Foldable formula for this op
        kraid_ops = kraid_sem.mnemonic_to_op.get(name, []) or kraid_sem.mnemonic_to_op.get(canon_mn, [])
        for kop in kraid_ops:
            if kop in kraid_sem.op_to_fold:
                fold_snip = kraid_sem.op_to_fold[kop][:140].replace("\\", "\\\\").replace('"', '\\"')
                op_sem = f"{op_sem} [kraid::{kop}: {fold_snip}]"
                break
        op_sem_esc = op_sem.replace("\\", "\\\\").replace('"', '\\"')
        spv_low_esc = spv_low.replace("\\", "\\\\").replace('"', '\\"')
        isa_doc = legacy_spec.instr_docs.get(name, legacy_spec.instr_docs.get(canon_mn, {})).get("desc", "")
        isa_doc_esc = isa_doc[:240].replace("\\", "\\\\").replace('"', '\\"')
        var_enum = f"Variant::{c_ident(var.upper())}" if var else "Variant::NONE"
        cvar_enum = f"Variant::{c_ident(canon_var.upper())}" if canon_var else "Variant::NONE"
        cpp_lines.append(
            f"    {{ {idx}, Mnemonic::{c_ident(name)}, {var_enum}, "
            f"Mnemonic::{c_ident(canon_mn)}, {cvar_enum}, "
            f"\"{full_name}\", \"{ins['exec_unit']}\", {ins['since']}, {ins['until']}, "
            f"0x{ins['fixed_mask']:016X}ULL, 0x{ins['fixed_value']:016X}ULL, {ins['specificity']}, "
            f"SemanticCategory::{cat_str}, \"{op_sem_esc}\", \"{spv_low_esc}\", \"{isa_doc_esc}\" }},"
        )
    cpp_lines.extend([
        "};",
        "",
        "const InstructionSemanticsSpec* get_all_instruction_specs(size_t* out_count) {",
        "    if (out_count) *out_count = kNumValhallInstructionSpecs;",
        "    return kInstructionSpecs;",
        "}",
        "",
    ])

    # Emit per-instruction try_decode_and_populate_<idx> functions!
    for idx, ins in enumerate(kraid_spec.instructions):
        cpp_lines.append(
            f"static bool decode_spec_{idx}(uint64_t v, uint8_t arch, uint64_t pc, bool fau32, DecodedValhallInst* out) {{"
        )
        cpp_lines.append(f"    if (arch < {ins['since']} || arch > {ins['until']}) return false;")
        cpp_lines.append(
            f"    if ((v & 0x{ins['fixed_mask']:016X}ULL) != 0x{ins['fixed_value']:016X}ULL) return false;"
        )

        # 1. Extract all physical fields
        phys_by_name = {f["name"]: f for f in ins["phys_fields"]}
        for pf in ins["phys_fields"]:
            fname = snake_case(pf["name"])
            pos = pf["pos"]
            width = pf["width"]
            mask = (1 << width) - 1
            raw_expr = f"uint32_t((v >> {pos}) & 0x{mask:X}ULL)"
            if pf["modifier"].startswith("shr("):
                sh = int(pf["modifier"][4:-1])
                raw_expr = f"({raw_expr} << {sh})"
            elif pf["modifier"].startswith("minus("):
                mn = int(pf["modifier"][6:-1])
                raw_expr = f"({raw_expr} + {mn}u)"
            cpp_lines.append(f"    uint32_t f_{fname} = {raw_expr};")
            cpp_lines.append(f"    (void)f_{fname};")

            ftype = pf["type"]
            if ftype in kraid_spec.enums:
                cpp_en = pascal_case(ftype)
                cpp_lines.append(f"    {cpp_en} e_{fname}{{}};")
                cpp_lines.append(f"    if (!try_decode_{snake_case(ftype)}(uint8_t(f_{fname}), arch, &e_{fname})) return false;")
                if pf["restrict"]:
                    conds = [f"(e_{fname} == {cpp_en}::{c_ident(r)})" for r in pf["restrict"]]
                    cpp_lines.append(f"    if (!({' || '.join(conds)})) return false;")

        # 2. Compute virtual fields
        virt_by_name = {vf["name"]: vf for vf in ins["virt_fields"]}
        for vf in ins["virt_fields"]:
            vname = snake_case(vf["name"])
            if vf["exact"] is not None:
                cpp_lines.append(f"    uint32_t f_{vname} = {vf['exact']}u;")
            else:
                expr_c = compile_xml_expr(vf["expr"], kraid_spec.enums)
                cpp_lines.append(f"    uint32_t f_{vname} = uint32_t({expr_c});")
            cpp_lines.append(f"    (void)f_{vname};")
            vtype = vf["type"]
            if vtype in kraid_spec.enums:
                cpp_en = pascal_case(vtype)
                cpp_lines.append(f"    {cpp_en} e_{vname}{{}};")
                cpp_lines.append(f"    if (!try_decode_{snake_case(vtype)}(uint8_t(f_{vname}), arch, &e_{vname})) return false;")
                if vf["restrict"]:
                    conds = [f"(e_{vname} == {cpp_en}::{c_ident(r)})" for r in vf["restrict"]]
                    cpp_lines.append(f"    if (!({' || '.join(conds)})) return false;")

        # 3. Validate source encodings for all <src> syntax elements before accepting match
        all_fields = {**phys_by_name, **virt_by_name}
        has_fau_page = "fau_page_index" in all_fields
        fau_page_var = "uint8_t(f_fau_page_index)" if has_fau_page else "0u"
        for s_idx, sc in enumerate(ins["syntax"]):
            if sc["kind"] == "src":
                sname = sc["name"]
                sf = all_fields[sname]
                is64 = (sf["type"] == "SourceEncoding64")
                cpp_lines.append(f"    DecodedSource dec_src_{s_idx}{{}};")
                cpp_lines.append(
                    f"    if (!decode_source_operand(uint8_t(f_{snake_case(sname)}), arch, {fau_page_var}, {'true' if is64 else 'false'}, fau32, &dec_src_{s_idx})) return false;"
                )

        # If we reach here, the instruction is 100% valid! Populate *out
        cpp_lines.append("    if (!out) return true;")
        cpp_lines.append("    *out = DecodedValhallInst{};")
        cpp_lines.append("    out->valid = true;")
        cpp_lines.append("    out->raw_word = v;")
        cpp_lines.append("    out->pc = pc;")
        cpp_lines.append("    out->arch = arch;")
        cpp_lines.append(f"    out->spec_index = {idx};")
        cpp_lines.append(f"    out->mnemonic = kInstructionSpecs[{idx}].mnemonic;")
        cpp_lines.append(f"    out->variant = kInstructionSpecs[{idx}].variant;")
        cpp_lines.append(f"    out->canonical_mnemonic = kInstructionSpecs[{idx}].canonical_mnemonic;")
        cpp_lines.append(f"    out->canonical_variant = kInstructionSpecs[{idx}].canonical_variant;")
        cpp_lines.append(f"    out->category = kInstructionSpecs[{idx}].category;")
        cpp_lines.append(f"    out->exec_unit = kInstructionSpecs[{idx}].exec_unit;")
        if "flow" in all_fields:
            cpp_lines.append("    out->flow = e_flow;")
        if has_fau_page:
            cpp_lines.append("    out->fau_page = uint8_t(f_fau_page_index);")

        # Store all decoded fields in out->fields and enum strings in out->modifier_strs
        for fname_raw, fdict in all_fields.items():
            fn = snake_case(fname_raw)
            cpp_lines.append(f"    out->fields[\"{fname_raw}\"] = f_{fn};")
            if fdict["type"] in kraid_spec.enums:
                cpp_lines.append(f"    out->modifier_strs[\"{fname_raw}\"] = to_string(e_{fn});")

        # Build disassembly string and structured operands following <syntax>
        full_name = f"{ins['name']}.{ins['variant']}" if ins["variant"] else ins["name"]
        cpp_lines.append("    std::ostringstream oss;")
        cpp_lines.append(f"    oss << \"{full_name}\";")

        # Name modifiers first
        for sc in ins["syntax"]:
            if sc["kind"] == "name":
                for mname in sc["mods"]:
                    mfn = snake_case(mname)
                    mf = all_fields[mname]
                    if mf["type"] in kraid_spec.enums:
                        en_obj = kraid_spec.enums[mf["type"]]
                        if "none" in en_obj.values:
                            cpp_lines.append(f"    if (std::string(to_string(e_{mfn})) != \"none\") oss << \".\" << to_string(e_{mfn});")
                        else:
                            cpp_lines.append(f"    oss << \".\" << to_string(e_{mfn});")
                    else:
                        cpp_lines.append(f"    oss << \".\" << f_{mfn};")

        cpp_lines.append("    bool first_arg = true;")
        cpp_lines.append("    auto sep = [&]() { if (first_arg) { oss << \" \"; first_arg = false; } else { oss << \", \"; } };")
        cpp_lines.append("    (void)sep; (void)first_arg;")

        for s_idx, sc in enumerate(ins["syntax"]):
            kind = sc["kind"]
            if kind == "name":
                continue
            elif kind == "dest":
                is64 = ins["variant"] in ("i64", "u64", "s64")
                cpp_lines.append("    out->has_dest = true;")
                cpp_lines.append(f"    out->dest_is_64bit = {'true' if is64 else 'false'};")
                cpp_lines.append("    out->dest_reg = uint8_t(f_dest_register_index);")
                cpp_lines.append("    sep();")
                if is64:
                    cpp_lines.append("    oss << \"[r\" << int(out->dest_reg) << \":r\" << int(out->dest_reg + 1) << \"]\";")
                else:
                    cpp_lines.append("    oss << \"r\" << int(out->dest_reg);")
                for mname in sc["mods"]:
                    mfn = snake_case(mname)
                    mf = all_fields[mname]
                    if mf["type"] in kraid_spec.enums:
                        cpp_lines.append(f"    if (std::string(to_string(e_{mfn})) != \"none\") {{ out->dest_mod = to_string(e_{mfn}); oss << \".\" << out->dest_mod; }}")
            elif kind == "staging":
                is_input = (sc["direction"] == "input")
                has_write_sr = "staging_register_write_index" in all_fields
                sr_idx_field = "staging_register_write_index" if (not is_input and has_write_sr) else "staging_register_index"
                if ins["name"].startswith("VAR_TEX"):
                    sr_cnt_field = "sr_write_count"
                else:
                    sr_cnt_field = "sr_write_count" if sr_idx_field == "staging_register_write_index" else "sr_count"
                cnt_f = all_fields.get(sr_cnt_field)
                offset_add = 1 if (cnt_f and cnt_f["type"] in ("sr_write_count_t", "ls_multi_sr_count_m")) else 0
                sr_var = "out->sr_read" if is_input else "out->sr_write"
                cpp_lines.append(f"    {sr_var}.present = true;")
                cpp_lines.append(f"    {sr_var}.is_input = {'true' if is_input else 'false'};")
                cpp_lines.append(f"    {sr_var}.base_reg = uint8_t(f_{snake_case(sr_idx_field)});")
                if cnt_f:
                    cpp_lines.append(f"    {sr_var}.count = uint8_t(f_{snake_case(sr_cnt_field)} + {offset_add});")
                else:
                    cpp_lines.append(f"    {sr_var}.count = 1;")
                cpp_lines.append("    sep();")
                cpp_lines.append(f"    oss << \"@r\" << int({sr_var}.base_reg);")
                cpp_lines.append(f"    for (uint8_t r = 1; r < {sr_var}.count; ++r) oss << \":r\" << int({sr_var}.base_reg + r);")
            elif kind == "src":
                for mname in sc["mods"]:
                    mfn = snake_case(mname)
                    mf = all_fields[mname]
                    if mf["type"] == "abs_m":
                        cpp_lines.append(f"    dec_src_{s_idx}.abs = (f_{mfn} != 0);")
                    elif mf["type"] == "neg_m":
                        cpp_lines.append(f"    dec_src_{s_idx}.neg = (f_{mfn} != 0);")
                    elif mf["type"] == "not_m":
                        cpp_lines.append(f"    dec_src_{s_idx}.not_mod = (f_{mfn} != 0);")
                    elif mf["type"] in kraid_spec.enums:
                        cpp_lines.append(
                            f"    if (std::string(to_string(e_{mfn})) != \"none\") dec_src_{s_idx}.swizzle_mod = to_string(e_{mfn});"
                        )
                cpp_lines.append(f"    out->srcs.push_back(dec_src_{s_idx});")
                cpp_lines.append("    sep();")
                cpp_lines.append(f"    {{ const auto& s = dec_src_{s_idx};")
                cpp_lines.append("      if (s.kind == SourceKind::Register) {")
                cpp_lines.append("          const char* tag = s.discard_last_use ? \"^\" : \"\";")
                cpp_lines.append("          if (s.is_64bit) oss << \"[r\" << int(s.reg_num) << tag << \":r\" << int(s.reg_num + 1) << tag << \"]\";")
                cpp_lines.append("          else oss << \"r\" << int(s.reg_num) << tag;")
                cpp_lines.append("      } else if (s.kind == SourceKind::FauUniform) {")
                cpp_lines.append("          if (fau32) oss << \"u\" << int(s.reg_num);")
                cpp_lines.append("          else if (s.is_64bit) oss << \"u\" << int(s.fau_index64);")
                cpp_lines.append("          else oss << \"u\" << int(s.fau_index64) << (s.word_select ? \".w1\" : \".w0\");")
                cpp_lines.append("      } else if (s.kind == SourceKind::SmallConstant) {")
                cpp_lines.append("          oss << s.symbolic_name;")
                cpp_lines.append("      } else {")
                cpp_lines.append("          if (s.is_64bit) oss << s.symbolic_name;")
                cpp_lines.append("          else oss << s.symbolic_name << (s.word_select ? \".w1\" : \".w0\");")
                cpp_lines.append("      }")
                for mname in sc["mods"]:
                    mfn = snake_case(mname)
                    mf = all_fields[mname]
                    if mf["type"] in kraid_spec.enums:
                        cpp_lines.append(f"      if (std::string(to_string(e_{mfn})) != \"none\") oss << \".\" << to_string(e_{mfn});")
                cpp_lines.append("    }")
            elif kind in ("pc_rel_label_signed", "pc_rel_label_unsigned"):
                oname = sc["name"]
                ofn = snake_case(oname)
                of = all_fields[oname]
                width = of.get("width", 32)
                if of.get("modifier", "").startswith("shr("):
                    width += int(of["modifier"][4:-1])
                cpp_lines.append("    out->has_branch_offset = true;")
                if kind == "pc_rel_label_signed":
                    cpp_lines.append(f"    out->branch_offset_bytes = int64_t(sign_ext32(f_{ofn}, {width}));")
                else:
                    cpp_lines.append(f"    out->branch_offset_bytes = int64_t(f_{ofn});")
                cpp_lines.append("    out->branch_target_pc = int64_t(pc) + out->branch_offset_bytes + 8;")
                cpp_lines.append("    sep();")
                cpp_lines.append("    oss << \"0x\" << std::hex << uint64_t(out->branch_target_pc) << std::dec;")
            else:
                # Immediate (uint, int, float, V2F16, V2I16, V4I8)
                iname = sc["name"]
                ifn = snake_case(iname)
                if_dict = all_fields[iname]
                width = if_dict.get("width", 32)
                if if_dict.get("modifier", "").startswith("shr("):
                    width += int(if_dict["modifier"][4:-1])
                elif if_dict.get("modifier", "").startswith("minus("):
                    width += 1
                cpp_lines.append("    out->has_imm = true;")
                cpp_lines.append(f"    out->imm_unsigned = f_{ifn};")
                if kind == "int" or if_dict["type"] == "int":
                    cpp_lines.append(f"    out->imm_signed = int64_t(sign_ext32(f_{ifn}, {width}));")
                    cpp_lines.append("    sep();")
                    cpp_lines.append("    if (out->imm_signed < 0) oss << \"#-0x\" << std::hex << (-out->imm_signed) << std::dec;")
                    cpp_lines.append("    else oss << \"#0x\" << std::hex << out->imm_signed << std::dec;")
                else:
                    cpp_lines.append(f"    out->imm_signed = int64_t(f_{ifn});")
                    cpp_lines.append("    sep();")
                    cpp_lines.append(f"    oss << \"#0x\" << std::hex << f_{ifn} << std::dec;")

        cpp_lines.append("    out->disassembly = oss.str();")
        cpp_lines.append("    return true;")
        cpp_lines.append("}")
        cpp_lines.append("")

    # Group instructions by instruction_class [56:52] and sort by (specificity DESC, name ASC, spec_index ASC)
    by_class = defaultdict(list)
    fallback_specs = []
    for idx, ins in enumerate(kraid_spec.instructions):
        ic_mask = (0x1F << 52)
        if (ins["fixed_mask"] & ic_mask) == ic_mask:
            ic_val = (ins["fixed_value"] >> 52) & 0x1F
            by_class[ic_val].append((idx, ins))
        else:
            fallback_specs.append((idx, ins))

    for ic_val in by_class:
        by_class[ic_val].sort(key=lambda x: (-x[1]["specificity"], x[1]["name"], x[0]))
    fallback_specs.sort(key=lambda x: (-x[1]["specificity"], x[1]["name"], x[0]))

    cpp_lines.extend([
        "DecodedValhallInst decode_valhall_instruction(",
        "    uint64_t word, uint8_t arch, uint64_t pc, bool fau32)",
        "{",
        "    DecodedValhallInst out{};",
        "    out.raw_word = word;",
        "    out.pc = pc;",
        "    out.arch = arch;",
        "    uint32_t ic = uint32_t((word >> 52) & 0x1F);",
        "    switch (ic) {",
    ])
    for ic_val, spec_list in sorted(by_class.items()):
        cpp_lines.append(f"        case {ic_val}:")
        for idx, _ in spec_list:
            cpp_lines.append(f"            if (decode_spec_{idx}(word, arch, pc, fau32, &out)) return out;")
        cpp_lines.append("            break;")
    cpp_lines.append("        default: break;")
    cpp_lines.append("    }")
    for idx, _ in fallback_specs:
        cpp_lines.append(f"    if (decode_spec_{idx}(word, arch, pc, fau32, &out)) return out;")
    cpp_lines.extend([
        "    return out;",
        "}",
        "",
        "std::vector<DecodedValhallInst> decode_valhall_shader(",
        "    const uint8_t* bytes, size_t size_bytes, uint8_t arch, bool fau32)",
        "{",
        "    std::vector<DecodedValhallInst> res;",
        "    for (size_t off = 0; off + 8 <= size_bytes; off += 8) {",
        "        uint64_t word = 0;",
        "        std::memcpy(&word, bytes + off, sizeof(uint64_t));",
        "        res.push_back(decode_valhall_instruction(word, arch, off, fau32));",
        "    }",
        "    return res;",
        "}",
        "",
        "} // namespace valhall",
        "",
    ])

    with open(cpp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(cpp_lines))


# =============================================================================
# 7. Parse v11.xml (+ common.xml, v12-v14.xml) & Generate CSF Descriptors C++
# =============================================================================
class CsfXmlSpec:
    def __init__(self, xml_path: str):
        self.root = ET.parse(xml_path).getroot()
        self.arch = int(self.root.attrib.get("arch", "11"))
        self.enums = OrderedDict()   # name -> OrderedDict(vname -> int)
        self.structs = OrderedDict() # name -> dict
        self._parse()

    def _parse(self):
        for c in self.root:
            if c.tag == "enum":
                ename = c.attrib["name"]
                vals = OrderedDict()
                for v in c.findall("value"):
                    vals[v.attrib["name"]] = parse_int(v.attrib["value"])
                self.enums[ename] = vals
            elif c.tag == "struct":
                sname = c.attrib["name"]
                size_words = int(c.attrib.get("size", "0"))
                align = int(c.attrib.get("align", "4"))
                fields = []
                max_bit = size_words * 32
                for f in c.findall("field"):
                    fname = f.attrib["name"]
                    fsize = int(f.attrib["size"])
                    start_str = f.attrib["start"]
                    if ":" in start_str:
                        w_str, b_str = start_str.split(":")
                        bit_start = int(w_str) * 32 + int(b_str)
                    else:
                        bit_start = int(start_str)
                    max_bit = max(max_bit, bit_start + fsize)
                    fields.append({
                        "name": fname,
                        "bit_start": bit_start,
                        "bit_size": fsize,
                        "type": f.attrib["type"],
                        "default": f.attrib.get("default", None),
                        "modifier": f.attrib.get("modifier", None),
                    })
                total_words = max(size_words, (max_bit + 31) // 32, 1)
                self.structs[sname] = {
                    "name": sname,
                    "size_words": total_words,
                    "align": align,
                    "fields": fields,
                }


def generate_csf_files(v11_spec: CsfXmlSpec, other_specs: dict, out_dir: str):
    h_path = os.path.join(out_dir, "valhall_csf_descriptors_generated.h")
    cpp_path = os.path.join(out_dir, "valhall_csf_descriptors_generated.cpp")

    h_lines = [
        "// AUTO-GENERATED BY tools/gen_valhall_spec.py -- DO NOT EDIT MANUALLY",
        "// Extracted from specs/csf/v11.xml (+ common.xml, v12.xml, v13.xml, v14.xml)",
        "#pragma once",
        "#include <cstdint>",
        "#include <cstddef>",
        "#include <cstring>",
        "#include <cmath>",
        "#include <array>",
        "#include <string>",
        "#include <ostream>",
        "",
        "namespace valhall::csf {",
        "",
        "// Bitfield extraction & insertion helpers across 32-bit descriptor words",
        "static inline uint64_t extract_bits(const uint32_t* words, uint32_t bit_offset, uint32_t bit_size) {",
        "    if (bit_size == 0) return 0;",
        "    uint32_t word_idx = bit_offset / 32u;",
        "    uint32_t bit_in_word = bit_offset % 32u;",
        "    uint64_t val = uint64_t(words[word_idx]) >> bit_in_word;",
        "    if (bit_in_word + bit_size > 32u) {",
        "        uint32_t rem = (bit_in_word + bit_size) - 32u;",
        "        val |= (uint64_t(words[word_idx + 1]) & ((rem == 64u) ? ~0ULL : ((1ULL << rem) - 1ULL))) << (32u - bit_in_word);",
        "    }",
        "    if (bit_size < 64u) val &= ((1ULL << bit_size) - 1ULL);",
        "    return val;",
        "}",
        "",
        "static inline void insert_bits(uint32_t* words, uint32_t bit_offset, uint32_t bit_size, uint64_t val) {",
        "    if (bit_size == 0 || val == 0) return;",
        "    if (bit_size < 64u) val &= ((1ULL << bit_size) - 1ULL);",
        "    uint32_t word_idx = bit_offset / 32u;",
        "    uint32_t bit_in_word = bit_offset % 32u;",
        "    uint32_t low_bits = std::min(bit_size, 32u - bit_in_word);",
        "    uint32_t low_mask = (low_bits == 32u) ? ~0u : ((1u << low_bits) - 1u);",
        "    words[word_idx] = (words[word_idx] & ~(low_mask << bit_in_word)) | (uint32_t(val & low_mask) << bit_in_word);",
        "    if (bit_in_word + bit_size > 32u) {",
        "        uint32_t high_bits = (bit_in_word + bit_size) - 32u;",
        "        uint32_t high_mask = (high_bits == 32u) ? ~0u : ((1u << high_bits) - 1u);",
        "        words[word_idx + 1] = (words[word_idx + 1] & ~high_mask) | uint32_t((val >> low_bits) & high_mask);",
        "    }",
        "}",
        "",
        "static inline int64_t sign_extend_64(uint64_t val, uint32_t bits) {",
        "    if (bits == 0 || bits >= 64) return static_cast<int64_t>(val);",
        "    uint32_t shift = 64u - bits;",
        "    return static_cast<int64_t>(val << shift) >> shift;",
        "}",
        "",
        "static inline float u32_bits_to_float(uint32_t u) {",
        "    float f = 0.0f;",
        "    std::memcpy(&f, &u, sizeof(float));",
        "    return f;",
        "}",
        "static inline uint32_t float_to_u32_bits(float f) {",
        "    uint32_t u = 0;",
        "    std::memcpy(&u, &f, sizeof(uint32_t));",
        "    return u;",
        "}",
        "",
    ]

    # Emit all enums from v11.xml
    h_lines.append("// ============================================================================")
    h_lines.append(f"// All {len(v11_spec.enums)} CSF & Descriptor Enums from v11.xml")
    h_lines.append("// ============================================================================")
    for ename, vals in v11_spec.enums.items():
        cpp_en = pascal_case(ename)
        h_lines.append(f"enum class {cpp_en} : uint32_t {{")
        seen_idents = set()
        for vname, vval in vals.items():
            cid = c_ident(vname)
            if cid in seen_idents:
                cid = f"{cid}_{vval}"
            seen_idents.add(cid)
            h_lines.append(f"    {cid} = {vval}u,")
        h_lines.append("};")
        h_lines.append(f"const char* to_string({cpp_en} v);")
        h_lines.append("")

    # Also emit v13/v14 IDVS SR and FRAGMENT SR enums so multi-arch CSF ABI (v11..v14) is supported!
    for arch_ver, spec in sorted(other_specs.items()):
        for sr_name in ("IDVS SR", "FRAGMENT SR", "CS Opcode"):
            if sr_name in spec.enums:
                cpp_en = f"V{arch_ver}{pascal_case(sr_name)}"
                h_lines.append(f"enum class {cpp_en} : uint32_t {{")
                for vname, vval in spec.enums[sr_name].items():
                    h_lines.append(f"    {c_ident(vname)} = {vval}u,")
                h_lines.append("};")
                h_lines.append("")

    # Emit helper structs for PixelFormat (22-bit) and ComponentSwizzle (12-bit)
    h_lines.extend([
        "struct ComponentSwizzle {",
        "    Channel r = Channel::R;",
        "    Channel g = Channel::G;",
        "    Channel b = Channel::B;",
        "    Channel a = Channel::A;",
        "    static inline ComponentSwizzle unpack(uint32_t raw12) {",
        "        return {",
        "            static_cast<Channel>((raw12 >> 0) & 0x7u),",
        "            static_cast<Channel>((raw12 >> 3) & 0x7u),",
        "            static_cast<Channel>((raw12 >> 6) & 0x7u),",
        "            static_cast<Channel>((raw12 >> 9) & 0x7u),",
        "        };",
        "    }",
        "    inline uint32_t pack() const {",
        "        return (uint32_t(r) & 0x7u) | ((uint32_t(g) & 0x7u) << 3) |",
        "               ((uint32_t(b) & 0x7u) << 6) | ((uint32_t(a) & 0x7u) << 9);",
        "    }",
        "};",
        "",
        "struct PixelFormat {",
        "    RGBComponentOrder component_order = RGBComponentOrder::RGBA;",
        "    bool srgb = false;",
        "    Format format = Format::RGBA8_UNORM;",
        "    uint32_t raw22 = 0;",
        "    static inline PixelFormat unpack(uint32_t raw) {",
        "        PixelFormat pf;",
        "        pf.raw22 = raw & 0x3FFFFFu;",
        "        pf.component_order = static_cast<RGBComponentOrder>(raw & 0xFFu);",
        "        pf.srgb = ((raw >> 10) & 0x1u) != 0 || ((raw >> 20) & 0x1u) != 0;",
        "        pf.format = static_cast<Format>((raw >> 12) & 0xFFu);",
        "        return pf;",
        "    }",
        "    inline uint32_t pack() const {",
        "        if (raw22 != 0 && format == Format((raw22 >> 12) & 0xFFu)) return raw22;",
        "        return (uint32_t(component_order) & 0xFFu) | (srgb ? (1u << 10) : 0u) | ((uint32_t(format) & 0xFFu) << 12);",
        "    }",
        "};",
        "",
    ])

    # Topologically order structs in v11_spec so nested structs are declared before outer structs
    emitted_structs = set()
    ordered_structs = []

    def visit_struct(sname):
        if sname in emitted_structs or sname not in v11_spec.structs:
            return
        emitted_structs.add(sname)
        st = v11_spec.structs[sname]
        for f in st["fields"]:
            ft = f["type"]
            if ft in v11_spec.structs:
                visit_struct(ft)
        ordered_structs.append(st)

    for sname in v11_spec.structs:
        visit_struct(sname)

    h_lines.append("// ============================================================================")
    h_lines.append(f"// All {len(ordered_structs)} CSF Commands & GPU Memory Descriptors from v11.xml")
    h_lines.append("// ============================================================================")

    for st in ordered_structs:
        sname = st["name"]
        cpp_st = pascal_case(sname)
        swords = st["size_words"]
        sbytes = swords * 4
        h_lines.append(f"struct {cpp_st} {{")
        h_lines.append(f"    static constexpr size_t kWordCount = {swords};")
        h_lines.append(f"    static constexpr size_t kByteSize = {sbytes};")
        h_lines.append(f"    static constexpr size_t kAlignment = {st['align']};")

        # Deduplicate field names if any
        used_fnames = set()
        field_defs = []
        for f in st["fields"]:
            fn = snake_case(f["name"])
            if fn in used_fnames:
                fn = f"{fn}_b{f['bit_start']}"
            used_fnames.add(fn)
            ft = f["type"]
            fsize = f["bit_size"]
            fstart = f["bit_start"]
            fmod = f["modifier"]

            if ft in v11_spec.enums:
                cpp_ft = pascal_case(ft)
                def_val = f"{cpp_ft}::{c_ident(f['default'])}" if (f["default"] and f["default"] in v11_spec.enums[ft]) else f"static_cast<{cpp_en}>(0)"
                def_val = f"{cpp_ft}::{c_ident(f['default'])}" if (f["default"] and f["default"] in v11_spec.enums[ft]) else f"static_cast<{cpp_ft}>(0)"
                h_lines.append(f"    {cpp_ft} {fn} = {def_val};")
            elif ft in v11_spec.structs:
                cpp_ft = pascal_case(ft)
                h_lines.append(f"    {cpp_ft} {fn}{{}};")
            elif ft == "bool":
                bdef = "true" if f["default"] in ("true", "1") else "false"
                h_lines.append(f"    bool {fn} = {bdef};")
            elif ft == "float":
                fdef = f"{float(f['default'])}f" if f["default"] else "0.0f"
                h_lines.append(f"    float {fn} = {fdef};")
            elif ft in ("ulod", "slod"):
                h_lines.append(f"    float {fn} = 0.0f;")
            elif ft == "Pixel Format":
                h_lines.append(f"    PixelFormat {fn}{{}};")
            elif ft == "Component Swizzle":
                h_lines.append(f"    ComponentSwizzle {fn}{{}};")
            elif ft == "int":
                idef = parse_int(f["default"]) if f["default"] else 0
                cpp_t = "int64_t" if fsize > 32 else "int32_t"
                h_lines.append(f"    {cpp_t} {fn} = {idef};")
            else:
                # uint, hex, address, uint/float
                udef = 0
                if f["default"]:
                    try:
                        udef = int(float(f["default"]))
                    except Exception:
                        udef = 0
                cpp_t = "uint64_t" if (fsize > 32 or ft == "address" or (fmod and fmod.startswith("shr("))) else "uint32_t"
                h_lines.append(f"    {cpp_t} {fn} = {udef}u;")

            field_defs.append((fn, f))

        # Generate unpack(const uint32_t* words)
        h_lines.append("")
        h_lines.append(f"    static inline {cpp_st} unpack(const uint32_t* words) {{")
        h_lines.append(f"        {cpp_st} out{{}};")
        for fn, f in field_defs:
            ft = f["type"]
            fstart = f["bit_start"]
            fsize = f["bit_size"]
            fmod = f["modifier"]

            if ft in v11_spec.structs:
                sub_st = pascal_case(ft)
                sub_words = v11_spec.structs[ft]["size_words"]
                if fstart % 32 == 0 and fsize >= sub_words * 32:
                    h_lines.append(f"        out.{fn} = {sub_st}::unpack(words + {fstart // 32});")
                else:
                    h_lines.append(f"        {{ uint32_t tmp[{sub_words}] = {{0}}; tmp[0] = uint32_t(extract_bits(words, {fstart}, {min(fsize, 32)})); out.{fn} = {sub_st}::unpack(tmp); }}")
                continue

            raw_e = f"extract_bits(words, {fstart}, {fsize})"
            if fmod:
                if fmod.startswith("shr("):
                    sh = int(fmod[4:-1])
                    raw_e = f"({raw_e} << {sh})"
                elif fmod.startswith("minus("):
                    mn = int(fmod[6:-1])
                    raw_e = f"({raw_e} + {mn}u)"
                elif fmod == "log2":
                    raw_e = f"(1ULL << ({raw_e}))"

            if ft in v11_spec.enums:
                cpp_ft = pascal_case(ft)
                h_lines.append(f"        out.{fn} = static_cast<{cpp_ft}>({raw_e});")
            elif ft == "bool":
                h_lines.append(f"        out.{fn} = ({raw_e}) != 0;")
            elif ft == "float":
                h_lines.append(f"        out.{fn} = u32_bits_to_float(uint32_t({raw_e}));")
            elif ft == "ulod":
                h_lines.append(f"        out.{fn} = float({raw_e}) / 256.0f;")
            elif ft == "slod":
                h_lines.append(f"        out.{fn} = float(sign_extend_64({raw_e}, {fsize})) / 256.0f;")
            elif ft == "Pixel Format":
                h_lines.append(f"        out.{fn} = PixelFormat::unpack(uint32_t({raw_e}));")
            elif ft == "Component Swizzle":
                h_lines.append(f"        out.{fn} = ComponentSwizzle::unpack(uint32_t({raw_e}));")
            elif ft == "int":
                eff_bits = fsize + (int(fmod[4:-1]) if (fmod and fmod.startswith("shr(")) else 0)
                cpp_t = "int64_t" if fsize > 32 else "int32_t"
                h_lines.append(f"        out.{fn} = static_cast<{cpp_t}>(sign_extend_64({raw_e}, {eff_bits}));")
            else:
                cpp_t = "uint64_t" if (fsize > 32 or ft == "address" or (fmod and fmod.startswith("shr("))) else "uint32_t"
                h_lines.append(f"        out.{fn} = static_cast<{cpp_t}>({raw_e});")
        h_lines.append("        return out;")
        h_lines.append("    }")

        # Convenience unpack from uint64_t for 2-word structs (like CS instructions or FramebufferPointer)
        if swords <= 2:
            h_lines.append(f"    static inline {cpp_st} unpack_u64(uint64_t val64) {{")
            h_lines.append("        uint32_t w[2] = { uint32_t(val64 & 0xFFFFFFFFULL), uint32_t(val64 >> 32) };")
            h_lines.append("        return unpack(w);")
            h_lines.append("    }")

        # Generate pack(uint32_t* words) const
        h_lines.append(f"    inline void pack(uint32_t* words) const {{")
        h_lines.append(f"        std::memset(words, 0, sizeof(uint32_t) * kWordCount);")
        for fn, f in field_defs:
            ft = f["type"]
            fstart = f["bit_start"]
            fsize = f["bit_size"]
            fmod = f["modifier"]

            if ft in v11_spec.structs:
                sub_st = pascal_case(ft)
                sub_words = v11_spec.structs[ft]["size_words"]
                if fstart % 32 == 0 and fsize >= sub_words * 32:
                    if fstart == 0 and len(field_defs) > 1 and all(x[1]["bit_start"] == 0 for x in field_defs):
                        if fn == field_defs[0][0]:
                            h_lines.append(f"        {fn}.pack(words + {fstart // 32});")
                        else:
                            h_lines.append(f"        {{ uint32_t tmp[{sub_words}] = {{0}}, def_tmp[{sub_words}] = {{0}}; {fn}.pack(tmp); {sub_st}{{}}.pack(def_tmp); if (std::memcmp(tmp, def_tmp, sizeof(tmp)) != 0) {{ for (size_t w = 0; w < {sub_words}; ++w) words[{fstart // 32} + w] |= tmp[w]; }} }}")
                    else:
                        h_lines.append(f"        {fn}.pack(words + {fstart // 32});")
                else:
                    h_lines.append(f"        {{ uint32_t tmp[{sub_words}] = {{0}}; {fn}.pack(tmp); insert_bits(words, {fstart}, {min(fsize, 32)}, tmp[0]); }}")
                continue

            val_e = f"uint64_t({fn})"
            if ft == "bool":
                val_e = f"({fn} ? 1ULL : 0ULL)"
            elif ft == "float":
                val_e = f"uint64_t(float_to_u32_bits({fn}))"
            elif ft == "ulod":
                val_e = f"uint64_t(std::lround(std::max(0.0f, {fn}) * 256.0f))"
            elif ft == "slod":
                val_e = f"uint64_t(int64_t(std::lround({fn} * 256.0f)))"
            elif ft == "Pixel Format":
                val_e = f"uint64_t({fn}.pack())"
            elif ft == "Component Swizzle":
                val_e = f"uint64_t({fn}.pack())"

            if fmod:
                if fmod.startswith("shr("):
                    sh = int(fmod[4:-1])
                    val_e = f"(({val_e}) >> {sh})"
                elif fmod.startswith("minus("):
                    mn = int(fmod[6:-1])
                    val_e = f"((({val_e}) >= {mn}ULL) ? (({val_e}) - {mn}ULL) : 0ULL)"
                elif fmod == "log2":
                    val_e = f"((({val_e}) > 0) ? uint64_t(__builtin_ctzll({val_e})) : 0ULL)"

            h_lines.append(f"        insert_bits(words, {fstart}, {fsize}, {val_e});")
        h_lines.append("    }")

        if swords <= 2:
            h_lines.append("    inline uint64_t pack_u64() const {")
            h_lines.append("        uint32_t w[2] = {0, 0};")
            h_lines.append("        pack(w);")
            h_lines.append("        return uint64_t(w[0]) | (uint64_t(w[1]) << 32);")
            h_lines.append("    }")

        h_lines.append("};")
        h_lines.append("")

    h_lines.append("} // namespace valhall::csf")
    h_lines.append("")

    with open(h_path, "w", encoding="utf-8") as f:
        f.write("\n".join(h_lines))

    # Emit valhall_csf_descriptors_generated.cpp (to_string implementations)
    cpp_lines = [
        "// AUTO-GENERATED BY tools/gen_valhall_spec.py -- DO NOT EDIT MANUALLY",
        "#include \"valhall_csf_descriptors_generated.h\"",
        "",
        "namespace valhall::csf {",
        "",
    ]
    for ename, vals in v11_spec.enums.items():
        cpp_en = pascal_case(ename)
        cpp_lines.append(f"const char* to_string({cpp_en} v) {{")
        cpp_lines.append("    switch (v) {")
        seen_vals = set()
        for vname, vval in vals.items():
            if vval in seen_vals:
                continue
            seen_vals.add(vval)
            cpp_lines.append(f"        case {cpp_en}::{c_ident(vname)}: return \"{vname}\";")
        cpp_lines.append("        default: return \"UNKNOWN\";")
        cpp_lines.append("    }")
        cpp_lines.append("}")
        cpp_lines.append("")

    cpp_lines.append("} // namespace valhall::csf")
    cpp_lines.append("")

    with open(cpp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(cpp_lines))


def main():
    parser = argparse.ArgumentParser(description="Generate Valhall ISA (v9-v14) & CSF Descriptor C++ headers/sources")
    parser.add_argument("--valhall-xml", default="specs/valhall/ISA.xml")
    parser.add_argument("--kraid-xml", default="specs/kraid/isa-v9-v14.xml")
    parser.add_argument("--kraid-ops", default="specs/kraid/ops.rs")
    parser.add_argument("--kraid-encode", default="specs/kraid/encode_v9.rs")
    parser.add_argument("--csf-xml", default="specs/csf/v11.xml")
    parser.add_argument("--out-dir", default="src/generated")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[gen_valhall_spec] Parsing {args.valhall_xml} ...")
    legacy_spec = ValhallLegacySpec(args.valhall_xml)
    print(f"[gen_valhall_spec]   -> {len(legacy_spec.immediates)} LUT immediates, {len(legacy_spec.instr_docs)} instruction docs")

    print(f"[gen_valhall_spec] Parsing {args.kraid_xml} ...")
    kraid_spec = KraidIsaSpec(args.kraid_xml)
    print(f"[gen_valhall_spec]   -> {len(kraid_spec.instructions)} instructions, {len(kraid_spec.mnemonics)} mnemonics, {len(kraid_spec.enums)} enums")

    kraid_sem = KraidSemanticsExtractor(args.kraid_ops, args.kraid_encode)
    print(f"[gen_valhall_spec]   -> {len(kraid_sem.op_to_fold)} Foldable ops extracted from kraid/ops.rs")

    print(f"[gen_valhall_spec] Parsing {args.csf_xml} (+ v12..v14) ...")
    v11_spec = CsfXmlSpec(args.csf_xml)
    other_specs = {}
    csf_dir = os.path.dirname(args.csf_xml)
    for ver in (12, 13, 14):
        p = os.path.join(csf_dir, f"v{ver}.xml")
        if os.path.exists(p):
            other_specs[ver] = CsfXmlSpec(p)
    print(f"[gen_valhall_spec]   -> {len(v11_spec.enums)} CSF enums, {len(v11_spec.structs)} CSF/GPU descriptor structs")

    generate_isa_files(legacy_spec, kraid_spec, kraid_sem, args.out_dir)
    generate_csf_files(v11_spec, other_specs, args.out_dir)
    print("[gen_valhall_spec] Successfully generated all C++ headers and sources.")


if __name__ == "__main__":
    main()

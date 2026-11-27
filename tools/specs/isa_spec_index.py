"""
Valhall ISA XML Spec Index (`specs/kraid/isa-v9-v14.xml`).

Provides a single cached, immutable representation of all 219 hardware enums and
627 `(mnemonic, variant, since..until)` instruction definitions from Kraid's
`isa-v9-v14.xml`.

All downstream modules (`isa_field_fragments`, `valhall_raw_synth`,
`isa_target_synth`) consume this index rather than duplicating bit-positions,
enum tables, or fixed opcode encodings.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_ISA_XML = "specs/kraid/isa-v9-v14.xml"

# Structural / fixed-dispatch fields excluded from "fields of interest"
# (still retained in `all_fields` for binary encoding).
STRUCTURAL_FIELD_NAMES: frozenset[str] = frozenset({
    "instruction_class",
    "opcode1",
    "opcode2",
    "dest_register_index",
    "staging_register_index",
    "staging_register_write_index",
    "staging_register_control",
    "staging_register_control_32",
    "staging_register_control_64",
    "compare",
    "memory_width",
})


@dataclasses.dataclass(frozen=True)
class IsaEnum:
    """Hardware modifier or opcode enumeration from `isa-v9-v14.xml`."""

    name: str
    since: int
    until: int
    name_to_val: tuple[tuple[str, int], ...]
    val_to_name: tuple[tuple[int, str], ...]

    def lookup(self, value_name: str) -> int | None:
        for k, v in self.name_to_val:
            if k == value_name:
                return v
        return None

    def name_of(self, value: int) -> str | None:
        for v, k in self.val_to_name:
            if v == value:
                return k
        return None

    def all_names(self) -> tuple[str, ...]:
        return tuple(k for k, _ in self.name_to_val)

    def all_values(self) -> tuple[int, ...]:
        return tuple(v for _, v in self.name_to_val)


@dataclasses.dataclass(frozen=True)
class IsaFieldSpec:
    """Single bitfield definition inside a Valhall 64-bit instruction."""

    name: str
    pos: int
    width: int
    field_type: str
    exact: str | None = None
    default: str | None = None
    restrict: tuple[str, ...] = ()
    modifier: str | None = None
    min_val: int | None = None
    max_val: int | None = None

    @property
    def is_fixed(self) -> bool:
        return self.exact is not None

    @property
    def is_source(self) -> bool:
        return self.field_type in ("SourceEncoding", "SourceEncoding64")

    @property
    def is_source64(self) -> bool:
        return self.field_type == "SourceEncoding64"

    @property
    def is_field_of_interest(self) -> bool:
        if self.is_fixed:
            return False
        if self.name in STRUCTURAL_FIELD_NAMES:
            return False
        if self.name.startswith("encoded_"):
            return False
        return True

    @property
    def bit_mask(self) -> int:
        return ((1 << self.width) - 1) << self.pos


@dataclasses.dataclass(frozen=True)
class IsaVariantSpec:
    """Single `<instruction>` entry from `isa-v9-v14.xml`."""

    spec_id: int
    name: str
    variant: str
    exec_unit: str
    since: int
    until: int
    alias: str | None
    alias_variant: str | None
    all_fields: tuple[IsaFieldSpec, ...]
    fixed_mask: int
    fixed_value: int
    virtual_names: tuple[str, ...]

    @property
    def key(self) -> tuple[str, str]:
        return (self.name, self.variant)

    @property
    def full_name(self) -> str:
        return f"{self.name}.{self.variant}" if self.variant else self.name

    @property
    def canonical_name(self) -> str:
        return self.alias if self.alias else self.name

    @property
    def is_alias(self) -> bool:
        return self.alias is not None

    @property
    def default_arch(self) -> int:
        return max(9, self.since)

    def field_by_name(self, field_name: str) -> IsaFieldSpec | None:
        for f in self.all_fields:
            if f.name == field_name:
                return f
        return None

    def fields_of_interest(self) -> tuple[IsaFieldSpec, ...]:
        return tuple(f for f in self.all_fields if f.is_field_of_interest)

    def variable_field_names(self) -> frozenset[str]:
        return frozenset(f.name for f in self.all_fields if not f.is_fixed)

    def interest_field_names(self) -> frozenset[str]:
        return frozenset(f.name for f in self.all_fields if f.is_field_of_interest)


@dataclasses.dataclass(frozen=True)
class IsaSpecIndex:
    """Indexed view of all enums and instruction variants in `isa-v9-v14.xml`."""

    xml_path: str
    enums: tuple[IsaEnum, ...]
    variants: tuple[IsaVariantSpec, ...]
    _enum_by_name: dict[str, tuple[IsaEnum, ...]] = dataclasses.field(repr=False, compare=False)
    _variants_by_mnemonic: dict[str, tuple[IsaVariantSpec, ...]] = dataclasses.field(repr=False, compare=False)

    def get_enum(self, enum_name: str, arch: int = 11) -> IsaEnum | None:
        """Resolve an enum by name, preferring the version covering `arch`."""
        cands = self._enum_by_name.get(enum_name)
        if not cands:
            return None
        for e in cands:
            if e.since <= arch <= e.until:
                return e
        return cands[0]

    def resolve_field_value(self, field: IsaFieldSpec, raw: str | int, arch: int = 11) -> int:
        """Convert an enum name or integer literal into raw bitfield bits."""
        if isinstance(raw, int):
            return raw
        text = str(raw).strip()
        if text.startswith(("0x", "0X")):
            return int(text, 16)
        if text.lstrip("-").isdigit():
            return int(text, 10)
        enum = self.get_enum(field.field_type, arch=arch)
        if enum is not None:
            val = enum.lookup(text)
            if val is not None:
                return val
        raise KeyError(
            f"Cannot resolve value {raw!r} for field '{field.name}' (type='{field.field_type}', arch={arch})"
        )

    def legal_enum_values(self, field: IsaFieldSpec, arch: int = 11) -> list[tuple[str, int]]:
        """Return legal `(name, int_val)` pairs for an enum field, respecting `restrict`."""
        enum = self.get_enum(field.field_type, arch=arch)
        if enum is None:
            return []
        if field.restrict:
            allowed = set(field.restrict)
            return [(k, v) for k, v in enum.name_to_val if k in allowed]
        return list(enum.name_to_val)

    def all_mnemonics(self) -> list[str]:
        return sorted(self._variants_by_mnemonic.keys())

    def variants_for(self, mnemonic: str) -> tuple[IsaVariantSpec, ...]:
        return self._variants_by_mnemonic.get(mnemonic.upper(), ())

    def primary_variant(self, mnemonic: str, variant: str | None = None) -> IsaVariantSpec | None:
        """Return a matching variant (preferring arch <= 11 when multiple arch ranges exist)."""
        cands = self.variants_for(mnemonic)
        if not cands:
            return None
        if variant is not None:
            matching = [v for v in cands if v.variant == variant]
            if not matching:
                return None
            cands = tuple(matching)
        for v in cands:
            if v.since <= 11 <= v.until:
                return v
        return cands[0]

    def unique_variant_keys(self) -> list[tuple[str, str]]:
        seen: dict[tuple[str, str], None] = {}
        for v in self.variants:
            seen[v.key] = None
        return sorted(seen.keys())


def _resolve_xml_path(xml_path: str | None = None) -> str:
    if xml_path and os.path.isfile(xml_path):
        return os.path.abspath(xml_path)
    repo_root = Path(__file__).resolve().parents[2]
    cand = repo_root / (xml_path or DEFAULT_ISA_XML)
    if cand.is_file():
        return str(cand)
    raise FileNotFoundError(f"Valhall ISA XML spec not found: {xml_path or DEFAULT_ISA_XML}")


@functools.lru_cache(maxsize=4)
def load_isa_spec_index(xml_path: str | None = None) -> IsaSpecIndex:
    """Parse `specs/kraid/isa-v9-v14.xml` into a cached `IsaSpecIndex`."""
    resolved = _resolve_xml_path(xml_path)
    tree = ET.parse(resolved)
    root = tree.getroot()

    enums: list[IsaEnum] = []
    enum_by_name: dict[str, list[IsaEnum]] = {}
    for elem in root.findall("enum"):
        name = elem.attrib["name"]
        since = int(elem.attrib.get("since", "9"))
        until = int(elem.attrib.get("until", "14"))
        n2v: list[tuple[str, int]] = []
        v2n: dict[int, str] = {}
        for val_el in elem.findall("value"):
            vname = val_el.attrib["name"]
            vint = int(val_el.attrib["value"], 0)
            n2v.append((vname, vint))
            if vint not in v2n or val_el.attrib.get("canonical") == "1":
                v2n[vint] = vname
        eobj = IsaEnum(
            name=name,
            since=since,
            until=until,
            name_to_val=tuple(n2v),
            val_to_name=tuple(sorted(v2n.items())),
        )
        enums.append(eobj)
        enum_by_name.setdefault(name, []).append(eobj)

    def _lookup_enum_val(etype: str, vname: str, arch: int) -> int:
        if vname.startswith(("0x", "0X")):
            return int(vname, 16)
        if vname.lstrip("-").isdigit():
            return int(vname, 10)
        cands = enum_by_name.get(etype, [])
        for e in cands:
            if e.since <= arch <= e.until:
                v = e.lookup(vname)
                if v is not None:
                    return v
        for e in cands:
            v = e.lookup(vname)
            if v is not None:
                return v
        raise KeyError(f"Unknown exact value '{vname}' in enum '{etype}'")

    variants: list[IsaVariantSpec] = []
    by_mn: dict[str, list[IsaVariantSpec]] = {}
    for idx, inst in enumerate(root.findall("instruction")):
        name = inst.attrib["name"].upper()
        variant = inst.attrib.get("variant", "")
        exec_unit = inst.attrib.get("exec_unit", "")
        since = int(inst.attrib.get("since", "9"))
        until = int(inst.attrib.get("until", "14"))
        alias = inst.attrib.get("alias")
        if alias:
            alias = alias.upper()
        alias_variant = inst.attrib.get("alias_variant")
        arch = max(9, since)

        fields: list[IsaFieldSpec] = []
        fixed_mask = 0
        fixed_value = 0

        for f_el in inst.findall("field"):
            fname = f_el.attrib["name"]
            pos = int(f_el.attrib["pos"])
            width = int(f_el.attrib["width"])
            ftype = f_el.attrib.get("type", "uint")
            exact = f_el.attrib.get("exact")
            default = f_el.attrib.get("default")
            restrict_raw = f_el.attrib.get("restrict", "").strip()
            restrict = tuple(restrict_raw.split()) if restrict_raw else ()
            modifier = f_el.attrib.get("modifier")
            min_v = int(f_el.attrib["min"], 0) if "min" in f_el.attrib else None
            max_v = int(f_el.attrib["max"], 0) if "max" in f_el.attrib else None

            fspec = IsaFieldSpec(
                name=fname,
                pos=pos,
                width=width,
                field_type=ftype,
                exact=exact,
                default=default,
                restrict=restrict,
                modifier=modifier,
                min_val=min_v,
                max_val=max_v,
            )
            fields.append(fspec)
            if exact is not None:
                vint = _lookup_enum_val(ftype, exact, arch)
                mask = ((1 << width) - 1) << pos
                fixed_mask |= mask
                fixed_value |= (vint & ((1 << width) - 1)) << pos

        for r_el in inst.findall("reserved"):
            rtype = r_el.attrib.get("type", "zero")
            if rtype == "zero":
                pos = int(r_el.attrib["pos"])
                width = int(r_el.attrib["width"])
                mask = ((1 << width) - 1) << pos
                fixed_mask |= mask

        vnames = tuple(v_el.attrib["name"] for v_el in inst.findall("virtual"))

        vspec = IsaVariantSpec(
            spec_id=idx,
            name=name,
            variant=variant,
            exec_unit=exec_unit,
            since=since,
            until=until,
            alias=alias,
            alias_variant=alias_variant,
            all_fields=tuple(fields),
            fixed_mask=fixed_mask,
            fixed_value=fixed_value,
            virtual_names=vnames,
        )
        variants.append(vspec)
        by_mn.setdefault(name, []).append(vspec)

    return IsaSpecIndex(
        xml_path=resolved,
        enums=tuple(enums),
        variants=tuple(variants),
        _enum_by_name={k: tuple(v) for k, v in enum_by_name.items()},
        _variants_by_mnemonic={k: tuple(v) for k, v in by_mn.items()},
    )

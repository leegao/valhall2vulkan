"""
Raw Valhall 64-Bit Instruction Encoder & Pass-1 Lowering Verifier.

Assembles any `IsaVariantSpec` + `FieldMutationCase` directly into valid 64-bit
Valhall machine words, wraps it in a minimal self-contained shader stream
(target instruction + canonical `NOP.end`), and verifies decoding and Pass-1
SSA lowering via `build/spirv2valhall --direct-isa`.

Covers 100% of all 201 mnemonics (including all 34 `internal` ops and any
`compute`/`graphics` ops not directly emitted by `libMali-Gxx` for Vulkan 1.3),
all 627 `(mnemonic, variant)` XML definitions, and all declarative field
mutations & branch points.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import tempfile
from collections.abc import Sequence

from tools.corpus_synth.isa_field_fragments import (
    FieldMutationCase,
    FieldValue,
    generate_field_cases_for_spec,
)
from tools.specs.isa_spec_index import (
    IsaFieldSpec,
    IsaSpecIndex,
    IsaVariantSpec,
    load_isa_spec_index,
)


@dataclasses.dataclass(frozen=True)
class EncodedValhallCase:
    """Encoded 64-bit Valhall instruction stream for a single `FieldMutationCase`."""

    case: FieldMutationCase
    word: int
    stream_words: tuple[int, ...]
    stage: str

    @property
    def stream_bytes(self) -> bytes:
        return b"".join(w.to_bytes(8, "little") for w in self.stream_words)


@dataclasses.dataclass
class RawVerifyResult:
    """Result of decoding and Pass-1 lowering a raw Valhall instruction case."""

    case_id: str
    fragment_id: str
    mnemonic: str
    variant: str
    arch: int
    is_branch_point: bool
    word_hex: str
    decoded_ok: bool
    decoded_mnemonic: str
    decoded_canonical_mnemonic: str
    decoded_variant: str
    decoded_disasm: str
    pass1_ok: bool
    ssa_nodes: int
    ssa_ops: list[str]
    error: str = ""

    @property
    def mnemonic_matched(self) -> bool:
        return self.decoded_ok and (
            self.decoded_mnemonic == self.mnemonic
            or self.decoded_canonical_mnemonic == self.mnemonic
        )


def _infer_stage_for_spec(spec: IsaVariantSpec) -> str:
    mn = spec.name
    if mn.startswith(("LD_VAR", "VAR_TEX", "ATEST", "BLEND", "DISCARD", "ZS_EMIT", "LD_TILE", "ST_TILE")):
        return "fragment"
    if mn.startswith(("LD_ATTR", "LEA_ATTR")):
        return "vertex"
    return "compute"


def _default_field_value(
    field: IsaFieldSpec,
    spec: IsaVariantSpec,
    index: IsaSpecIndex,
    arch: int,
    src_slot_idx: int,
) -> int:
    """Choose a clean, valid default bit value for a non-fixed XML field."""
    if field.default is not None:
        try:
            return index.resolve_field_value(field, field.default, arch=arch)
        except KeyError:
            pass

    if field.is_source:
        # Even-aligned GPRs for SourceEncoding64 (r0, r2, r4), distinct GPRs for 32-bit
        return (src_slot_idx * 2) & 0x3E if field.is_source64 else (src_slot_idx & 0x3F)

    if field.name == "dest_register_index":
        return 8  # even-aligned GPR pair r8/r9
    if field.name == "staging_register_index":
        return 0
    if field.name == "staging_register_write_index":
        return 8
    if field.name == "fau_page_index":
        return 0
    if field.name == "flow":
        enum = index.get_enum(field.field_type, arch=arch)
        if enum and enum.lookup("none") is not None:
            return enum.lookup("none") or 0
        return 0
    if field.name in ("write_mask", "primary_write_mask", "secondary_write_mask"):
        return 0xF & ((1 << field.width) - 1)
    if spec.name == "TEX_DUAL":
        if field.name == "sr_write_count":
            return index.resolve_field_value(field, "write4", arch=arch)
        if field.name == "sr_count":
            return index.resolve_field_value(field, "count2", arch=arch)

    legal = index.legal_enum_values(field, arch=arch)
    if legal:
        for name, val in legal:
            if name == "none":
                return val
        return legal[0][1]

    if field.min_val is not None:
        return field.min_val
    return 0


def _apply_field_modifier_encode(field: IsaFieldSpec, val: int) -> int:
    """Convert a logical integer value into its encoded bitfield representation."""
    if field.modifier == "align(2)":
        return (val // 2) * 2 if val >= 0 else val
    if field.modifier == "align(4)":
        return (val // 4) * 4 if val >= 0 else val
    if field.modifier == "align(8)":
        return (val // 8) * 8 if val >= 0 else val
    if field.modifier == "shr(3)":
        return val >> 3
    return val


def encode_instruction(
    spec: IsaVariantSpec,
    index: IsaSpecIndex,
    overrides: dict[str, FieldValue] | None = None,
    *,
    arch: int | None = None,
) -> int:
    """Encode `spec` with optional `overrides` into a 64-bit Valhall instruction word."""
    eff_arch = spec.default_arch if arch is None else arch
    ov = dict(overrides or {})

    # Handle XML <expression> couplings so mutated encodings stay hardware-legal
    if spec.name in ("ATOM", "ATOM_RETURN") and spec.variant == "i64":
        opc = ov.get("atom_opc")
        if opc in ("aaddu", "aadds") and "sr_count" not in ov:
            sf = spec.field_by_name("sr_count")
            if sf and not sf.is_fixed:
                ov["sr_count"] = "count1"

    word = spec.fixed_value
    src_slot = 0

    for f in spec.all_fields:
        if f.is_fixed:
            continue
        if f.name in ov:
            raw_val = index.resolve_field_value(f, ov[f.name], arch=eff_arch)
            if f.modifier and isinstance(ov[f.name], int):
                raw_val = _apply_field_modifier_encode(f, raw_val)
        else:
            raw_val = _default_field_value(f, spec, index, eff_arch, src_slot)
        if f.is_source:
            src_slot += 1

        mask = (1 << f.width) - 1
        word &= ~(mask << f.pos)
        word |= (raw_val & mask) << f.pos

    return word & 0xFFFFFFFFFFFFFFFF


def canonical_nop_end_word(index: IsaSpecIndex, arch: int = 11) -> int:
    """Return the canonical 64-bit `NOP.end` terminator instruction word."""
    nop_spec = index.primary_variant("NOP")
    if nop_spec is None:
        raise RuntimeError("NOP spec not found in isa-v9-v14.xml")
    return encode_instruction(nop_spec, index, {"flow": "end"}, arch=arch)


def encode_mutation_case(
    case: FieldMutationCase,
    spec: IsaVariantSpec,
    index: IsaSpecIndex,
) -> EncodedValhallCase:
    """Encode a `FieldMutationCase` into a 2-instruction stream (`target; NOP.end`)."""
    w = encode_instruction(spec, index, case.overrides_dict(), arch=case.arch)
    # If the target itself already has flow=end or flow=discard, still append NOP.end for CFG safety
    nop_end = canonical_nop_end_word(index, arch=case.arch)
    return EncodedValhallCase(
        case=case,
        word=w,
        stream_words=(w, nop_end),
        stage=_infer_stage_for_spec(spec),
    )


def verify_raw_valhall_batch(
    encoded_cases: Sequence[EncodedValhallCase],
    bin_path: str = "./build/spirv2valhall",
    timeout: int = 60,
) -> list[RawVerifyResult]:
    """Decode and Pass-1 lower a batch of raw Valhall cases in one `spirv2valhall` invocation."""
    if not encoded_cases:
        return []
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"spirv2valhall binary not found at {bin_path}")

    case_by_id = {ec.case.case_id: ec for ec in encoded_cases}
    fd, tsv_path = tempfile.mkstemp(prefix="valhall_raw_batch_", suffix=".tsv")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for ec in encoded_cases:
                words_csv = ",".join(f"{w:016x}" for w in ec.stream_words)
                fh.write(f"{ec.case.case_id}\t{ec.case.arch}\t{ec.stage}\t{words_csv}\n")

        proc = subprocess.run(
            [bin_path, "--direct-isa", tsv_path],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"spirv2valhall --direct-isa failed (rc={proc.returncode}): {proc.stderr.strip()[:300]}"
            )

        results: list[RawVerifyResult] = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ec = case_by_id[row["case_id"]]
            results.append(
                RawVerifyResult(
                    case_id=row["case_id"],
                    fragment_id=ec.case.fragment_id,
                    mnemonic=ec.case.mnemonic,
                    variant=ec.case.variant,
                    arch=ec.case.arch,
                    is_branch_point=ec.case.is_branch_point,
                    word_hex=f"0x{ec.word:016x}",
                    decoded_ok=bool(row["decoded_ok"]),
                    decoded_mnemonic=row.get("mnemonic", ""),
                    decoded_canonical_mnemonic=row.get("canonical_mnemonic", ""),
                    decoded_variant=row.get("variant", ""),
                    decoded_disasm=row.get("disasm", ""),
                    pass1_ok=bool(row["pass1_ok"]),
                    ssa_nodes=int(row.get("ssa_nodes", 0)),
                    ssa_ops=list(row.get("ssa_ops", [])),
                    error=row.get("error", ""),
                )
            )
        return results
    finally:
        if os.path.exists(tsv_path):
            os.remove(tsv_path)


def synthesize_raw_for_mnemonic(
    mnemonic: str,
    *,
    all_variants: bool = False,
    include_fragments: bool = True,
    include_branch_points: bool = True,
    max_per_fragment: int = 4,
    bin_path: str = "./build/spirv2valhall",
    index: IsaSpecIndex | None = None,
) -> list[RawVerifyResult]:
    """Generate, encode, decode, and Pass-1 verify raw Valhall cases for `mnemonic`."""
    idx = index or load_isa_spec_index()
    specs = idx.variants_for(mnemonic)
    if not specs:
        raise ValueError(f"Unknown Valhall mnemonic '{mnemonic}' (not in isa-v9-v14.xml)")
    selected = list(specs) if all_variants else [idx.primary_variant(mnemonic)]
    encoded: list[EncodedValhallCase] = []
    for sp in selected:
        if sp is None:
            continue
        cases = generate_field_cases_for_spec(
            sp,
            idx,
            include_baseline=True,
            include_fragments=include_fragments,
            include_branch_points=include_branch_points,
            max_per_fragment=max_per_fragment,
        )
        for c in cases:
            encoded.append(encode_mutation_case(c, sp, idx))
    return verify_raw_valhall_batch(encoded, bin_path=bin_path)

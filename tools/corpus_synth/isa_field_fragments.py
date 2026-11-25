"""
Declarative Valhall Cross-Instruction Field Fragments & Branch-Point Generator.

Models the 17 reusable cross-instruction field fragments (`F_FLOW`, `F_SRC_FAU`,
`F_DEST_W`, `F_WIDEN`, `F_SWZ`, `F_LANES`, `F_FP_IN`, `F_BIT_NOT`, `F_FP_OUT`,
`F_SAT`, `F_ROUND`, `F_CMP`, `F_MSG`, `F_LANE_SCALAR`, `F_VEC_FMT`,
`F_TEX_COMMON`, `F_VAR_COMMON`) and the operand/argument/field branch points
specified in `isa-covering.md`.

Purely declarative and decoupled from SPIR-V synthesis and binary encoding:
consumes only `IsaSpecIndex` and emits `FieldMutationCase` records.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Union

from tools.corpus_synth.isa_spec_index import IsaSpecIndex, IsaVariantSpec

FieldValue = Union[int, str]


@dataclasses.dataclass(frozen=True)
class FieldMutationCase:
    """Single declarative field/operand mutation test case for one instruction variant."""

    case_id: str
    fragment_id: str
    mnemonic: str
    variant: str
    arch: int
    field_overrides: tuple[tuple[str, FieldValue], ...]
    rationale: str
    is_branch_point: bool = False

    def overrides_dict(self) -> dict[str, FieldValue]:
        return dict(self.field_overrides)


MutationGenFn = Callable[[IsaVariantSpec, IsaSpecIndex, int], list[FieldMutationCase]]


@dataclasses.dataclass(frozen=True)
class FieldFragment:
    """Reusable cross-instruction field fragment attachable to any matching XML variant."""

    fragment_id: str
    field_names: frozenset[str]
    description: str
    generator: MutationGenFn

    def applies_to(self, spec: IsaVariantSpec) -> bool:
        interest = spec.interest_field_names()
        return bool(interest & self.field_names)


def _mk_case(
    spec: IsaVariantSpec,
    fragment_id: str,
    suffix: str,
    overrides: dict[str, FieldValue],
    rationale: str,
    *,
    arch: int | None = None,
    is_branch_point: bool = False,
) -> FieldMutationCase:
    eff_arch = spec.default_arch if arch is None else arch
    vtag = spec.variant if spec.variant else "none"
    cid = f"{spec.name}.{vtag}@{eff_arch}:{fragment_id}:{suffix}"
    return FieldMutationCase(
        case_id=cid,
        fragment_id=fragment_id,
        mnemonic=spec.name,
        variant=spec.variant,
        arch=eff_arch,
        field_overrides=tuple(sorted(overrides.items())),
        rationale=rationale,
        is_branch_point=is_branch_point,
    )


def _legal_names(spec: IsaVariantSpec, index: IsaSpecIndex, field_name: str, arch: int) -> list[str]:
    f = spec.field_by_name(field_name)
    if f is None or f.is_fixed:
        return []
    return [name for name, _ in index.legal_enum_values(f, arch=arch)]


def _pick_representative(names: Sequence[str], preferred: Sequence[str], max_items: int) -> list[str]:
    picked: list[str] = []
    name_set = set(names)
    for p in preferred:
        if p in name_set and p not in picked:
            picked.append(p)
            if len(picked) >= max_items:
                return picked
    for n in names:
        if n not in picked:
            picked.append(n)
            if len(picked) >= max_items:
                return picked
    return picked


# ---------------------------------------------------------------------------
# Universal SourceEncoding / SourceEncoding64 + fau_page_index (F_SRC_FAU)
# ---------------------------------------------------------------------------

def _gen_src_fau(spec: IsaVariantSpec, index: IsaSpecIndex, max_cases: int) -> list[FieldMutationCase]:
    src_fields = [f for f in spec.all_fields if f.is_source and not f.is_fixed]
    if not src_fields:
        return []

    has_fau_page = spec.field_by_name("fau_page_index") is not None and not spec.field_by_name("fau_page_index").is_fixed
    s0 = src_fields[0]
    arch = spec.default_arch
    cases: list[FieldMutationCase] = []

    # Regime 1: GPR vs GPR + discard_last_use (bit 6: 0x40..0x7F)
    gpr_idx = 0 if s0.is_source64 else 1
    cases.append(_mk_case(
        spec, "F_SRC_FAU", "gpr_discard",
        {s0.name: 0x40 | gpr_idx},
        f"GPR+discard_last_use (0x{0x40 | gpr_idx:02x}) on {s0.name}",
        is_branch_point=True,
    ))
    if len(src_fields) >= 2:
        s1 = src_fields[1]
        same_idx = 2 if (s0.is_source64 or s1.is_source64) else 3
        cases.append(_mk_case(
            spec, "F_SRC_FAU", "gpr_alias_discard",
            {s0.name: same_idx, s1.name: 0x40 | same_idx},
            f"Aliased {s0.name}=={s1.name} (r{same_idx}) with discard_last_use on second read",
            is_branch_point=True,
        ))

    # Regime 2: FAU Uniform (0x80..0xBF) x fau_page_index (0..3), even vs odd
    if has_fau_page:
        for page, u_off, tag in [(0, 0, "p0_even"), (1, 1 if not s0.is_source64 else 2, "p1_word"), (3, 4, "p3_high")]:
            cases.append(_mk_case(
                spec, "F_SRC_FAU", f"fau_u_{tag}",
                {s0.name: 0x80 | u_off, "fau_page_index": page},
                f"FAU uniform page={page} idx={u_off} on {s0.name} ({'64b' if s0.is_source64 else '32b'})",
                is_branch_point=True,
            ))
    else:
        cases.append(_mk_case(
            spec, "F_SRC_FAU", "fau_u_p0",
            {s0.name: 0x80},
            f"FAU uniform page=0 idx=0 on {s0.name}",
            is_branch_point=True,
        ))

    # Regime 3: Small Constants (0xC0..0xDF) — including special bit patterns & arch-dependent slot 21
    if not s0.is_source64:
        for sc_idx, sc_name in [(0, "k0"), (1, "k1n"), (16, "k1f"), (7, "k7FFFFFFF"), (20, "k0f1f")]:
            cases.append(_mk_case(
                spec, "F_SRC_FAU", f"small_const_{sc_name}",
                {s0.name: 0xC0 | sc_idx},
                f"Small constant {sc_name} (0x{0xC0 | sc_idx:02x}) on {s0.name}",
                is_branch_point=True,
            ))
        # Arch-dependent slot 21: k_SPARE (v9..v10) vs kvgf (v11..v14)
        if spec.since <= 10 <= spec.until:
            cases.append(_mk_case(
                spec, "F_SRC_FAU", "small_const_slot21_v9",
                {s0.name: 0xC0 | 21},
                "Arch-dependent small constant slot 21 (v9-v10 k_SPARE = 0x0)",
                arch=max(9, spec.since),
                is_branch_point=True,
            ))
        if spec.since <= 11 <= spec.until:
            cases.append(_mk_case(
                spec, "F_SRC_FAU", "small_const_slot21_v11",
                {s0.name: 0xC0 | 21},
                "Arch-dependent small constant slot 21 (v11-v14 kvgf = 0x272428E0)",
                arch=11,
                is_branch_point=True,
            ))

    # Regime 4: FAU Special (0xE0..0xFF) x fau_page_index (0, 1, 3) — never Page 2
    if has_fau_page:
        if s0.is_source64:
            # 64-bit special pointers on Page 1 (thread_local_pointer=0xE2, workgroup_local_pointer=0xE4, resource_table_pointer=0xE6)
            cases.append(_mk_case(
                spec, "F_SRC_FAU", "fau_spec_p1_tlp64",
                {s0.name: 0xE2, "fau_page_index": 1},
                f"FAU Special Page 1 64-bit thread_local_pointer on {s0.name}",
                is_branch_point=True,
            ))
            cases.append(_mk_case(
                spec, "F_SRC_FAU", "fau_spec_p1_rtp64",
                {s0.name: 0xE6, "fau_page_index": 1},
                f"FAU Special Page 1 64-bit resource_table_pointer on {s0.name}",
                is_branch_point=True,
            ))
        else:
            # Page 0: warp_id (idx=4 -> 0xE4), Page 1: tlp lo/hi (0xE2/0xE3), Page 3: lane_id (0xE2) & pc lo/hi (0xFE/0xFF)
            cases.append(_mk_case(
                spec, "F_SRC_FAU", "fau_spec_p0_warp_id",
                {s0.name: 0xE4, "fau_page_index": 0},
                f"FAU Special Page 0 warp_id on {s0.name}",
                is_branch_point=True,
            ))
            cases.append(_mk_case(
                spec, "F_SRC_FAU", "fau_spec_p1_tlp_hi",
                {s0.name: 0xE3, "fau_page_index": 1},
                f"FAU Special Page 1 thread_local_pointer.hi (word_select=1) on {s0.name}",
                is_branch_point=True,
            ))
            cases.append(_mk_case(
                spec, "F_SRC_FAU", "fau_spec_p3_lane_id",
                {s0.name: 0xE2, "fau_page_index": 3},
                f"FAU Special Page 3 lane_id on {s0.name}",
                is_branch_point=True,
            ))
            cases.append(_mk_case(
                spec, "F_SRC_FAU", "fau_spec_p3_pc_lo",
                {s0.name: 0xFE, "fau_page_index": 3},
                f"FAU Special Page 3 pc.lo on {s0.name}",
                is_branch_point=True,
            ))

    # Regime 5: SourceEncoding64 paired with narrow widen* (subword read widened to 64-bit)
    for sf in src_fields:
        if sf.is_source64:
            idx_num = sf.name[-1] if sf.name[-1].isdigit() else "0"
            wf = spec.field_by_name(f"widen{idx_num}")
            if wf is not None and not wf.is_fixed:
                w_legal = _legal_names(spec, index, wf.name, arch)
                for wval in _pick_representative(w_legal, ("w0", "w1", "h0", "b0"), 2):
                    if wval != "none":
                        cases.append(_mk_case(
                            spec, "F_SRC_FAU", f"src64_{sf.name}_{wval}",
                            {sf.name: 3, wf.name: wval},
                            f"SourceEncoding64 {sf.name} with narrow {wf.name}={wval} (reads odd 32b GPR widened to 64b)",
                            is_branch_point=True,
                        ))

    return cases[:max_cases]


# ---------------------------------------------------------------------------
# Generic & Specialized Reusable Field Fragment Generators
# ---------------------------------------------------------------------------

def _gen_flow(spec: IsaVariantSpec, index: IsaSpecIndex, max_cases: int) -> list[FieldMutationCase]:
    arch = spec.default_arch
    legal = _legal_names(spec, index, "flow", arch)
    picked = _pick_representative(legal, ("wait0", "wait0126", "wait", "reconverge", "discard", "end"), max_cases)
    return [
        _mk_case(spec, "F_FLOW", val, {"flow": val}, f"Scoreboard/warp flow control flow={val}",
                 is_branch_point=(val in ("reconverge", "discard", "end")))
        for val in picked
    ]


def _gen_dest_w(spec: IsaVariantSpec, index: IsaSpecIndex, max_cases: int) -> list[FieldMutationCase]:
    arch = spec.default_arch
    cases: list[FieldMutationCase] = []
    for fname in ("dest_width", "dest_width_narrow", "dest_width_replicate"):
        legal = _legal_names(spec, index, fname, arch)
        for val in _pick_representative(legal, ("h0", "h1", "h01", "none"), max_cases):
            cases.append(_mk_case(
                spec, "F_DEST_W", f"{fname}_{val}", {fname: val},
                f"Writeback width/lane {fname}={val}",
                is_branch_point=(val in ("h0", "h1", "h01")),
            ))
    return cases[:max_cases]


def _gen_widen(spec: IsaVariantSpec, index: IsaSpecIndex, max_cases: int) -> list[FieldMutationCase]:
    arch = spec.default_arch
    cases: list[FieldMutationCase] = []
    for fname in ("widen0", "widen1", "widen2"):
        legal = _legal_names(spec, index, fname, arch)
        if not legal:
            continue
        pref = ("h0", "h1", "b0", "b3", "w0", "w1")
        for val in _pick_representative(legal, pref, 3):
            if val == "none":
                continue
            cases.append(_mk_case(
                spec, "F_WIDEN", f"{fname}_{val}", {fname: val},
                f"Subword widening {fname}={val} on {spec.full_name}",
                is_branch_point=True,
            ))
    return cases[:max_cases]


def _gen_swz(spec: IsaVariantSpec, index: IsaSpecIndex, max_cases: int) -> list[FieldMutationCase]:
    arch = spec.default_arch
    cases: list[FieldMutationCase] = []
    for fname in ("swz0", "swz1", "swz2"):
        legal = _legal_names(spec, index, fname, arch)
        if not legal:
            continue
        for val in _pick_representative(legal, ("h10", "h00", "h11", "b00", "b33", "b23"), 3):
            if val in ("h01", "none"):
                continue
            cases.append(_mk_case(
                spec, "F_SWZ", f"{fname}_{val}", {fname: val},
                f"Packed SIMD swizzle/broadcast {fname}={val}",
            ))
    return cases[:max_cases]


def _gen_lanes(spec: IsaVariantSpec, index: IsaSpecIndex, max_cases: int) -> list[FieldMutationCase]:
    arch = spec.default_arch
    cases: list[FieldMutationCase] = []
    for fname in ("lanes0", "lanes1"):
        legal = _legal_names(spec, index, fname, arch)
        if not legal:
            continue
        for val in _pick_representative(legal, ("b3210", "b0000", "b3333", "b2301", "h10", "h00", "h11"), 3):
            if val in ("b0123", "none", "h01"):
                continue
            cases.append(_mk_case(
                spec, "F_LANES", f"{fname}_{val}", {fname: val},
                f"Byte/halfword lane permutation {fname}={val}",
            ))
    return cases[:max_cases]


def _gen_fp_in(spec: IsaVariantSpec, index: IsaSpecIndex, max_cases: int) -> list[FieldMutationCase]:
    arch = spec.default_arch
    cases: list[FieldMutationCase] = []
    for idx_i in (0, 1, 2):
        af, nf = f"abs{idx_i}", f"neg{idx_i}"
        has_a = "abs" in _legal_names(spec, index, af, arch)
        has_n = "neg" in _legal_names(spec, index, nf, arch)
        if has_a:
            cases.append(_mk_case(spec, "F_FP_IN", f"{af}", {af: "abs"}, f"FP input modifier {af}=abs"))
        if has_n:
            cases.append(_mk_case(spec, "F_FP_IN", f"{nf}", {nf: "neg"}, f"FP input modifier {nf}=neg"))
        if has_a and has_n:
            cases.append(_mk_case(
                spec, "F_FP_IN", f"neg_abs{idx_i}", {af: "abs", nf: "neg"},
                f"Combined FP input modifiers -abs(src{idx_i}) (fetch order: widen/swz -> abs -> neg)",
                is_branch_point=True,
            ))
    return cases[:max_cases]


def _gen_bit_not(spec: IsaVariantSpec, index: IsaSpecIndex, max_cases: int) -> list[FieldMutationCase]:
    arch = spec.default_arch
    cases: list[FieldMutationCase] = []
    has_n2 = "not" in _legal_names(spec, index, "not2", arch)
    has_nr = "not" in _legal_names(spec, index, "not_result", arch)
    if has_n2:
        cases.append(_mk_case(spec, "F_BIT_NOT", "not2", {"not2": "not"}, "Bitwise invert on src2 (not2=not)"))
    if has_nr:
        cases.append(_mk_case(spec, "F_BIT_NOT", "not_result", {"not_result": "not"}, "Bitwise invert on output (not_result=not)"))
    if has_n2 and has_nr:
        cases.append(_mk_case(
            spec, "F_BIT_NOT", "not2_and_result", {"not2": "not", "not_result": "not"},
            "Combined not2=not and not_result=not (De Morgan path)",
            is_branch_point=True,
        ))
    return cases[:max_cases]


def _gen_fp_out(spec: IsaVariantSpec, index: IsaSpecIndex, max_cases: int) -> list[FieldMutationCase]:
    arch = spec.default_arch
    cases: list[FieldMutationCase] = []
    for val in _legal_names(spec, index, "clamp", arch):
        if val != "none":
            cases.append(_mk_case(
                spec, "F_FP_OUT", f"clamp_{val}", {"clamp": val},
                f"Post-op FP clamp={val}",
                is_branch_point=True,
            ))
    if "neg" in _legal_names(spec, index, "neg_result", arch):
        cases.append(_mk_case(spec, "F_FP_OUT", "neg_result", {"neg_result": "neg"}, "Post-op result negation"))
    return cases[:max_cases]


def _gen_sat(spec: IsaVariantSpec, index: IsaSpecIndex, max_cases: int) -> list[FieldMutationCase]:
    arch = spec.default_arch
    if "sat" in _legal_names(spec, index, "saturate", arch):
        return [_mk_case(
            spec, "F_SAT", "sat", {"saturate": "sat"},
            f"Integer saturation saturate=sat on {spec.full_name}",
            is_branch_point=True,
        )]
    return []


def _gen_round(spec: IsaVariantSpec, index: IsaSpecIndex, max_cases: int) -> list[FieldMutationCase]:
    arch = spec.default_arch
    legal = _legal_names(spec, index, "round", arch)
    cases: list[FieldMutationCase] = []
    for val in _pick_representative(legal, ("round_up", "round_down", "round_zero", "even", "up", "down", "zero", "away"), max_cases):
        if val == "none":
            continue
        cases.append(_mk_case(
            spec, "F_ROUND", val, {"round": val},
            f"Rounding mode round={val}",
            is_branch_point=True,
        ))
    return cases[:max_cases]


def _gen_cmp(spec: IsaVariantSpec, index: IsaSpecIndex, max_cases: int) -> list[FieldMutationCase]:
    arch = spec.default_arch
    cases: list[FieldMutationCase] = []
    for val in _pick_representative(_legal_names(spec, index, "cmpf", arch), ("gtlt", "total", "ne", "lt", "le", "eq", "gt", "ge"), 5):
        cases.append(_mk_case(
            spec, "F_CMP", f"cmpf_{val}", {"cmpf": val},
            f"Comparison predicate cmpf={val}",
            is_branch_point=(val in ("gtlt", "total", "ne")),
        ))
    for val in _legal_names(spec, index, "result_type", arch):
        cases.append(_mk_case(
            spec, "F_CMP", f"result_type_{val}", {"result_type": val},
            f"Comparison result encoding result_type={val}",
            is_branch_point=True,
        ))
    for val in _legal_names(spec, index, "combine", arch):
        if val != "none":
            cases.append(_mk_case(
                spec, "F_CMP", f"combine_{val}", {"combine": val},
                f"Branch/jump condition combine={val}",
                is_branch_point=True,
            ))
    return cases[:max_cases]


def _gen_msg(spec: IsaVariantSpec, index: IsaSpecIndex, max_cases: int) -> list[FieldMutationCase]:
    arch = spec.default_arch
    cases: list[FieldMutationCase] = []
    for val in _pick_representative(_legal_names(spec, index, "message_slot_index", arch), ("slot0", "slot1", "slot2", "slot7"), 3):
        cases.append(_mk_case(spec, "F_MSG", f"slot_{val}", {"message_slot_index": val}, f"Async message slot={val}"))
    for val in _legal_names(spec, index, "access", arch):
        if val != "none":
            cases.append(_mk_case(spec, "F_MSG", f"access_{val}", {"access": val}, f"Cache coherency hint access={val}", is_branch_point=True))
    return cases[:max_cases]


def _gen_lane_scalar(spec: IsaVariantSpec, index: IsaSpecIndex, max_cases: int) -> list[FieldMutationCase]:
    arch = spec.default_arch
    cases: list[FieldMutationCase] = []
    for fname in ("lane0", "lane1", "lane"):
        legal = _legal_names(spec, index, fname, arch)
        for val in _pick_representative(legal, ("b0", "b3", "h0", "h1", "w0", "d0"), 3):
            cases.append(_mk_case(spec, "F_LANE_SCALAR", f"{fname}_{val}", {fname: val}, f"Scalar lane selector {fname}={val}"))
    return cases[:max_cases]


def _gen_vec_fmt(spec: IsaVariantSpec, index: IsaSpecIndex, max_cases: int) -> list[FieldMutationCase]:
    arch = spec.default_arch
    cases: list[FieldMutationCase] = []
    for fname in ("vecsize", "register_format", "sr_count"):
        legal = _legal_names(spec, index, fname, arch)
        for val in _pick_representative(legal, ("v1", "v2", "v4", "f32", "f16", "u32", "s32", "count1", "count2", "count4"), 3):
            cases.append(_mk_case(spec, "F_VEC_FMT", f"{fname}_{val}", {fname: val}, f"Vector/format field {fname}={val}"))
    return cases[:max_cases]


def _gen_tex_common(spec: IsaVariantSpec, index: IsaSpecIndex, max_cases: int) -> list[FieldMutationCase]:
    arch = spec.default_arch
    cases: list[FieldMutationCase] = []
    tex_fields = (
        ("lod_mode", ("computed", "explicit", "grdesc")),
        ("dimensionality", ("dim_1d", "dim_2d", "dim_3d", "cube")),
        ("array_enable", ("array",)),
        ("compare_enable", ("shadow",)),
        ("projection_enable", ("proj",)),
        ("wide_indices", ("wide",)),
        ("skip", ("skip",)),
        ("texel_offset", ("offset",)),
        ("register_width", ("f16", "f32")),
    )
    for fname, pref in tex_fields:
        legal = _legal_names(spec, index, fname, arch)
        for val in _pick_representative(legal, pref, 2):
            if val == "none":
                continue
            cases.append(_mk_case(spec, "F_TEX_COMMON", f"{fname}_{val}", {fname: val}, f"Texture mode {fname}={val}", is_branch_point=True))
    return cases[:max_cases]


def _gen_var_common(spec: IsaVariantSpec, index: IsaSpecIndex, max_cases: int) -> list[FieldMutationCase]:
    arch = spec.default_arch
    cases: list[FieldMutationCase] = []
    for fname in ("sample_position", "update", "source_format"):
        legal = _legal_names(spec, index, fname, arch)
        for val in _pick_representative(legal, ("center", "centroid", "sample", "explicit", "store", "retrieve", "f32", "f16"), 2):
            if val == "none":
                continue
            cases.append(_mk_case(spec, "F_VAR_COMMON", f"{fname}_{val}", {fname: val}, f"Varying mode {fname}={val}", is_branch_point=True))
    return cases[:max_cases]


FRAGMENT_REGISTRY: dict[str, FieldFragment] = {
    "F_FLOW": FieldFragment("F_FLOW", frozenset({"flow"}), "Scoreboard wait mask & warp control", _gen_flow),
    "F_SRC_FAU": FieldFragment("F_SRC_FAU", frozenset({"src0", "src1", "src2", "src3", "fau_page_index"}), "5-regime SourceEncoding/SourceEncoding64 + FAU page", _gen_src_fau),
    "F_DEST_W": FieldFragment("F_DEST_W", frozenset({"dest_width", "dest_width_narrow", "dest_width_replicate"}), "Writeback width/lane & narrowing", _gen_dest_w),
    "F_WIDEN": FieldFragment("F_WIDEN", frozenset({"widen0", "widen1", "widen2"}), "Pre-op subword widening (signed/unsigned/f16)", _gen_widen),
    "F_SWZ": FieldFragment("F_SWZ", frozenset({"swz0", "swz1", "swz2"}), "Packed SIMD lane swizzle/broadcast", _gen_swz),
    "F_LANES": FieldFragment("F_LANES", frozenset({"lanes0", "lanes1"}), "Byte/halfword lane permutation", _gen_lanes),
    "F_FP_IN": FieldFragment("F_FP_IN", frozenset({"abs0", "abs1", "abs2", "neg0", "neg1", "neg2"}), "FP input abs/neg modifiers", _gen_fp_in),
    "F_BIT_NOT": FieldFragment("F_BIT_NOT", frozenset({"not2", "not_result"}), "Bitwise inversion on src2 / result", _gen_bit_not),
    "F_FP_OUT": FieldFragment("F_FP_OUT", frozenset({"clamp", "neg_result"}), "Post-op FP clamp & result negation", _gen_fp_out),
    "F_SAT": FieldFragment("F_SAT", frozenset({"saturate"}), "Integer saturation", _gen_sat),
    "F_ROUND": FieldFragment("F_ROUND", frozenset({"round"}), "FP/conversion rounding mode", _gen_round),
    "F_CMP": FieldFragment("F_CMP", frozenset({"cmpf", "result_type", "combine"}), "Comparison predicate, result type, branch combine", _gen_cmp),
    "F_MSG": FieldFragment("F_MSG", frozenset({"message_slot_index", "access"}), "Async message slot & cache coherency hint", _gen_msg),
    "F_LANE_SCALAR": FieldFragment("F_LANE_SCALAR", frozenset({"lane0", "lane1", "lane"}), "Scalar lane selector for conversions/loads", _gen_lane_scalar),
    "F_VEC_FMT": FieldFragment("F_VEC_FMT", frozenset({"vecsize", "register_format", "sr_count"}), "Vector size & register format", _gen_vec_fmt),
    "F_TEX_COMMON": FieldFragment(
        "F_TEX_COMMON",
        frozenset({"dimensionality", "array_enable", "compare_enable", "projection_enable", "lod_mode", "skip", "wide_indices", "write_mask", "register_width", "texel_offset"}),
        "Texture sampling dimensionality/LOD/mask modifiers",
        _gen_tex_common,
    ),
    "F_VAR_COMMON": FieldFragment("F_VAR_COMMON", frozenset({"sample_position", "update", "source_format"}), "Varying interpolation & update modes", _gen_var_common),
}


def applicable_fragments(spec: IsaVariantSpec) -> list[FieldFragment]:
    """Return all reusable field fragments whose fields appear in `spec`."""
    return [frag for frag in FRAGMENT_REGISTRY.values() if frag.applies_to(spec)]


# ---------------------------------------------------------------------------
# Instruction & Field-Specific Branch Points (Section 3.2 + Corner Cases)
# ---------------------------------------------------------------------------

def generate_branch_point_cases(spec: IsaVariantSpec, index: IsaSpecIndex) -> list[FieldMutationCase]:
    """Generate instruction-specific branch-point and corner-case mutations for `spec`."""
    mn = spec.name
    arch = spec.default_arch
    cases: list[FieldMutationCase] = []

    def add(tag: str, overrides: dict[str, FieldValue], rationale: str) -> None:
        # Only emit if every overridden field exists as a variable field in `spec`
        for k, v in overrides.items():
            f = spec.field_by_name(k)
            if f is None or f.is_fixed:
                return
            if isinstance(v, str):
                legal = _legal_names(spec, index, k, arch)
                if legal and v not in legal:
                    return
        cases.append(_mk_case(spec, "BRANCH_POINT", tag, overrides, rationale, is_branch_point=True))

    # FMIN / FMAX: sem (nan_suppress vs nan_propagate) x clamp
    if mn in ("FMIN", "FMAX"):
        for sem in ("nan_suppress", "nan_propagate"):
            add(f"{sem}_clamp01", {"sem": sem, "clamp": "clamp_0_1"}, f"{mn} {sem} interacting with clamp_0_1 on NaN/Inf")
            add(f"{sem}_clamp_m1_1", {"sem": sem, "clamp": "clamp_m1_1"}, f"{mn} {sem} interacting with clamp_m1_1")

    # FADD / FADD_LSCALE: sticky (round-to-odd) & f16 widening
    if mn in ("FADD", "FADD_LSCALE"):
        add("sticky_rto", {"sticky": "sticky"}, f"{mn} sticky=sticky (round-to-odd for double-rounding avoidance)")
    if mn in ("FADD", "FMUL", "FMA") and spec.variant == "f32":
        add("f16_widen_narrow", {"widen0": "h0", "dest_width_narrow": "h1"}, f"{mn}.f32 fp16->fp32 widen0 + f32->f16 narrow writeback")
    if mn in ("FMUL", "F32_TO_F16"):
        add("round_down_coupling", {"round": "round_down"}, f"{mn} round=round_down XML expression coupling")

    # FCMP* / ICMP*: gtlt vs ne vs total x result_type
    if mn in ("FCMP", "FCMP_AND", "FCMP_OR"):
        for cmpf in ("gtlt", "ne", "total"):
            for rtype in ("i1", "m1", "f1", "c"):
                add(f"{cmpf}_{rtype}", {"cmpf": cmpf, "result_type": rtype}, f"{mn} {cmpf} x result_type={rtype} (ordered vs unordered NaN)")
    if mn in ("ICMP", "ICMP_AND", "ICMP_OR", "ICMP_MULTI"):
        for cmpf in ("eq", "ne", "lt", "le"):
            for rtype in ("i1", "m1", "c"):
                add(f"{cmpf}_{rtype}", {"cmpf": cmpf, "result_type": rtype}, f"{mn} {cmpf} x result_type={rtype}")

    # IDPADD / MMUL: mixed signedness (s8*u8 / u8*s8) x saturate
    if mn in ("IDPADD", "MMUL"):
        add("mixed_su_sat", {"unsigned0": "none", "unsigned1": "v4u8", "saturate": "sat"}, f"{mn} mixed signed/unsigned (s8*u8) with saturation")
        add("mixed_us_sat", {"unsigned0": "v4u8", "unsigned1": "none", "saturate": "sat"}, f"{mn} mixed unsigned/signed (u8*s8) with saturation")
        add("mixed_su", {"unsigned0": "none", "unsigned1": "v4u8"}, f"{mn} mixed signed/unsigned (s8*u8)")

    # IADDC / ISUBB: return_carry
    if mn in ("IADDC", "ISUBB"):
        add("return_carry_none", {"return_carry": "none"}, f"{mn} returning 32-bit sum/diff")
        add("return_carry_c", {"return_carry": "c"}, f"{mn} returning carry/borrow bit")

    # CLPER: 5 lane_ops x subgroup sizes x inactive_result sentinels
    if mn == "CLPER":
        for lop in ("none", "xor", "accumulate", "shift", "rotate"):
            add(f"lane_op_{lop}", {"lane_op": lop, "subgroup": "subgroup16", "inactive_result": "zero"}, f"CLPER lane_op={lop}")
        for sg in ("subgroup2", "subgroup4", "subgroup8", "subgroup16"):
            add(f"sg_{sg}", {"subgroup": sg, "lane_op": "xor"}, f"CLPER subgroup={sg}")
        for inact in ("zero", "umax", "smin", "smax", "f1", "inf", "infn"):
            add(f"inactive_{inact}", {"inactive_result": inact, "lane_op": "shift"}, f"CLPER inactive_result={inact}")

    # CLPER_OR: subgroup8/16 x lane_select
    if mn == "CLPER_OR":
        for sg in ("subgroup8", "subgroup16"):
            for lsel in _pick_representative(_legal_names(spec, index, "lane_select", arch), ("lanes04", "lanes8C", "lanesBF"), 2):
                add(f"{sg}_{lsel}", {"subgroup": sg, "lane_select": lsel}, f"CLPER_OR {sg} x {lsel}")

    # ATOM / ATOM_RETURN: .i64 sr_count coupling (aaddu/aadds vs aadd/asmin)
    if mn in ("ATOM", "ATOM_RETURN"):
        for opc in _pick_representative(_legal_names(spec, index, "atom_opc", arch), ("aadd", "aaddu", "aadds", "asmin", "aumax", "aand", "axor"), 5):
            add(f"opc_{opc}", {"atom_opc": opc}, f"{mn}.{spec.variant} atom_opc={opc}")
    if mn in ("ATOM1", "ATOM1_RETURN"):
        for opc in _legal_names(spec, index, "atom1_opc", arch):
            add(f"opc1_{opc}", {"atom1_opc": opc}, f"{mn}.{spec.variant} atom1_opc={opc}")

    # LOAD / LD_PKA: lane x extend (sext vs zext_or_none)
    if mn in ("LOAD", "LD_PKA"):
        for ext_v in _legal_names(spec, index, "extend", arch):
            add(f"extend_{ext_v}", {"extend": ext_v}, f"{mn}.{spec.variant} subword extend={ext_v}")

    # BRANCH / JUMP / JUMP_EX: control-flow & divergence stack branch points
    if mn in ("BRANCH", "JUMP", "JUMP_EX"):
        for nv in _legal_names(spec, index, "not", arch):
            if nv != "none":
                add("inverted_cond", {"not": nv}, f"{mn} inverted predicate not={nv}")
        for cv in _legal_names(spec, index, "conservative", arch):
            if cv != "none":
                add("conservative", {"conservative": cv}, f"{mn} conservative={cv}")
        for sm in _legal_names(spec, index, "stack_mode", arch):
            add(f"stack_{sm}", {"stack_mode": sm}, f"{mn} divergence stack_mode={sm}")
        for dv in _legal_names(spec, index, "diverge", arch):
            if dv != "none":
                add(f"diverge_{dv}", {"diverge": dv}, f"{mn} diverge={dv}")
        for am in _legal_names(spec, index, "address_mode", arch):
            add(f"addr_{am}", {"address_mode": am}, f"{mn} address_mode={am}")

    # FLUSH: FTZ / Inf / NaN flushing modes
    if mn == "FLUSH":
        for ftz in _legal_names(spec, index, "flush_to_zero_mode", arch):
            add(f"ftz_{ftz}", {"flush_to_zero_mode": ftz}, f"FLUSH flush_to_zero_mode={ftz}")
        for inf_m in _legal_names(spec, index, "inf_mode", arch):
            add(f"inf_{inf_m}", {"inf_mode": inf_m}, f"FLUSH inf_mode={inf_m}")
        for nan_m in _legal_names(spec, index, "nan_mode", arch):
            add(f"nan_{nan_m}", {"nan_mode": nan_m}, f"FLUSH nan_mode={nan_m}")

    # Additional algebraic / SFU / system corner cases
    if mn == "CLZ":
        for mv in _legal_names(spec, index, "mask", arch):
            add(f"mask_{mv}", {"mask": mv}, f"CLZ mask={mv} (behavior on 0 / 0xFFFFFFFF)")
    if mn == "MUX":
        for mf in _legal_names(spec, index, "muxf", arch):
            add(f"muxf_{mf}", {"muxf": mf}, f"MUX selector mode muxf={mf}")
    if mn == "LDEXP":
        for iv in _legal_names(spec, index, "inf", arch):
            add(f"inf_{iv}", {"inf": iv}, f"LDEXP infinity handling inf={iv}")
    if mn in ("FREXPE", "FREXPM"):
        for sv in _legal_names(spec, index, "special", arch):
            add(f"special_{sv}", {"special": sv}, f"{mn} special={sv} (sqrt/log exponent adjustment)")
    if mn in ("FMA_RSCALE", "FMUL_RSCALE", "FADD_RSCALE"):
        for sv in _legal_names(spec, index, "special", arch):
            add(f"special_{sv}", {"special": sv}, f"{mn} special={sv}")
    if mn == "FPOW_SC_DET":
        for fv in _legal_names(spec, index, "func", arch):
            add(f"func_{fv}", {"func": fv}, f"FPOW_SC_DET func={fv} (pow vs powr special-case rules)")
    if mn == "FLOG_TABLE":
        for mode in _legal_names(spec, index, "mode", arch):
            for prec in _legal_names(spec, index, "precision", arch):
                add(f"{mode}_{prec}", {"mode": mode, "precision": prec}, f"FLOG_TABLE mode={mode} precision={prec}")
    if mn == "FEXP_TABLE":
        for adj in _legal_names(spec, index, "adj", arch):
            add(f"adj_{adj}", {"adj": adj}, f"FEXP_TABLE adj={adj}")
    if mn == "FSINCOS":
        for op in _legal_names(spec, index, "operation", arch):
            add(f"op_{op}", {"operation": op}, f"FSINCOS operation={op}")
    if mn == "FSINCOS_OFFSET":
        for sc in _legal_names(spec, index, "scale", arch):
            add(f"scale_{sc}", {"scale": sc}, f"FSINCOS_OFFSET scale={sc}")
    if mn in ("SEG_ADD", "SEG_SUB"):
        for pn in _legal_names(spec, index, "preserve_null", arch):
            add(f"pnull_{pn}", {"preserve_null": pn}, f"{mn} preserve_null={pn} (null pointer preservation)")
    if mn in ("LD_CHECKED", "LD_CHECKED_IMM"):
        for om in _legal_names(spec, index, "oob_mode", arch):
            add(f"oob_{om}", {"oob_mode": om}, f"{mn} out-of-bounds mode oob_mode={om}")
    if mn == "LD_GCLK":
        for src in _legal_names(spec, index, "source", arch):
            add(f"src_{src}", {"source": src}, f"LD_GCLK source={src}")
    if mn in ("LD_VAR_FLAT_IMM", "LD_VAR_FLAT", "LD_VAR_BUF_FLAT_IMM", "LD_VAR_BUF_FLAT"):
        for vt in _legal_names(spec, index, "vertex_type", arch):
            add(f"vtype_{vt}", {"vertex_type": vt}, f"{mn} vertex_type={vt}")
    if mn == "LD_VAR_SPECIAL":
        for vn in _legal_names(spec, index, "varying_name", arch):
            add(f"vname_{vn}", {"varying_name": vn}, f"LD_VAR_SPECIAL varying_name={vn}")
    if mn == "SHADD":
        for sh in (0, 1, 2, 3):
            add(f"shift_{sh}", {"shift": sh}, f"SHADD shift={sh} (address stride scaling)")
    if mn == "HADD":
        for rv in _legal_names(spec, index, "round", arch):
            add(f"round_{rv}", {"round": rv}, f"HADD round={rv} (floor vs ceil halving add)")
    if mn in ("TEX_GATHER", "VAR_TEX_GATHER", "VAR_TEX_BUF_GATHER"):
        for gc in _legal_names(spec, index, "gather_component", arch):
            add(f"comp_{gc}", {"gather_component": gc}, f"{mn} gather_component={gc}")

    return cases


def generate_field_cases_for_spec(
    spec: IsaVariantSpec,
    index: IsaSpecIndex,
    *,
    include_baseline: bool = True,
    include_fragments: bool = True,
    include_branch_points: bool = True,
    max_per_fragment: int = 4,
) -> list[FieldMutationCase]:
    """Generate the deduplicated set of field/operand mutation cases for `spec`."""
    cases: list[FieldMutationCase] = []
    seen_sigs: set = set()

    def _append(c: FieldMutationCase) -> None:
        sig = (c.mnemonic, c.variant, c.arch, c.field_overrides)
        if sig not in seen_sigs:
            seen_sigs.add(sig)
            cases.append(c)

    if include_baseline:
        _append(_mk_case(spec, "BASELINE", "default", {}, f"Canonical default encoding for {spec.full_name}"))

    if include_branch_points:
        for c in generate_branch_point_cases(spec, index):
            _append(c)

    if include_fragments:
        for frag in applicable_fragments(spec):
            limit = max(max_per_fragment, 8) if frag.fragment_id == "F_SRC_FAU" else max_per_fragment
            for c in frag.generator(spec, index, limit):
                _append(c)

    return cases

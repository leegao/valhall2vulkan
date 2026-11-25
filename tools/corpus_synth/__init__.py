"""Standalone greedy and targeted Valhall-ISA / SPIR-V-op corpus synthesizer."""

from tools.corpus_synth.greedy_driver import (
    CorpusSynthResult,
    SynthesizedShaderRecord,
    emit_shader_bytes,
    synthesize_corpus,
)
from tools.corpus_synth.isa_field_fragments import (
    FRAGMENT_REGISTRY,
    FieldFragment,
    FieldMutationCase,
    applicable_fragments,
    generate_branch_point_cases,
    generate_field_cases_for_spec,
)
from tools.corpus_synth.isa_recipes import RECIPES, VARIANT_RECIPES
from tools.corpus_synth.isa_spec_index import (
    IsaEnum,
    IsaFieldSpec,
    IsaSpecIndex,
    IsaVariantSpec,
    load_isa_spec_index,
)
from tools.corpus_synth.isa_spv_harness import IsaRecipe
from tools.corpus_synth.isa_target_synth import (
    IsaTargetResult,
    build_coverage_report,
    build_spvasm_for_isa,
    list_all_mnemonics,
    list_mnemonics_by_tier,
    list_synthesizable_mnemonics,
    run_full_raw_sweep,
    synthesize_for_isa,
)
from tools.corpus_synth.op_catalog import CATALOG_BY_NAME, OP_CATALOG, OpSpec
from tools.corpus_synth.oracle import CoverageOracle, OracleResult
from tools.corpus_synth.valhall_isel_catalog import (
    CATALOG_BY_MNEMONIC,
    VALHALL_ISEL_CATALOG,
    ValhallIselRule,
    audit_lowering_spec_coverage,
    get_rule_for_mnemonic,
)
from tools.corpus_synth.valhall_raw_synth import (
    EncodedValhallCase,
    RawVerifyResult,
    encode_instruction,
    encode_mutation_case,
    synthesize_raw_for_mnemonic,
    verify_raw_valhall_batch,
)

__all__ = [
    "CATALOG_BY_MNEMONIC",
    "CATALOG_BY_NAME",
    "FRAGMENT_REGISTRY",
    "OP_CATALOG",
    "RECIPES",
    "VALHALL_ISEL_CATALOG",
    "VARIANT_RECIPES",
    "CorpusSynthResult",
    "CoverageOracle",
    "EncodedValhallCase",
    "FieldFragment",
    "FieldMutationCase",
    "IsaEnum",
    "IsaFieldSpec",
    "IsaRecipe",
    "IsaSpecIndex",
    "IsaTargetResult",
    "IsaVariantSpec",
    "OpSpec",
    "OracleResult",
    "RawVerifyResult",
    "SynthesizedShaderRecord",
    "ValhallIselRule",
    "applicable_fragments",
    "audit_lowering_spec_coverage",
    "build_coverage_report",
    "build_spvasm_for_isa",
    "emit_shader_bytes",
    "encode_instruction",
    "encode_mutation_case",
    "generate_branch_point_cases",
    "generate_field_cases_for_spec",
    "get_rule_for_mnemonic",
    "list_all_mnemonics",
    "list_mnemonics_by_tier",
    "list_synthesizable_mnemonics",
    "load_isa_spec_index",
    "run_full_raw_sweep",
    "synthesize_corpus",
    "synthesize_for_isa",
    "synthesize_raw_for_mnemonic",
    "verify_raw_valhall_batch",
]

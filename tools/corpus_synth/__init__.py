"""Standalone greedy and targeted Valhall-ISA / SPIR-V-op corpus synthesizer."""

from tools.corpus_synth.greedy_driver import (
    CorpusSynthResult,
    SynthesizedShaderRecord,
    emit_shader_bytes,
    synthesize_corpus,
)
from tools.corpus_synth.isa_target_synth import (
    IsaRecipe,
    IsaTargetResult,
    build_spvasm_for_isa,
    list_synthesizable_mnemonics,
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

__all__ = [
    "CATALOG_BY_MNEMONIC",
    "CATALOG_BY_NAME",
    "CorpusSynthResult",
    "CoverageOracle",
    "IsaRecipe",
    "IsaTargetResult",
    "OP_CATALOG",
    "OpSpec",
    "OracleResult",
    "SynthesizedShaderRecord",
    "VALHALL_ISEL_CATALOG",
    "ValhallIselRule",
    "audit_lowering_spec_coverage",
    "build_spvasm_for_isa",
    "emit_shader_bytes",
    "get_rule_for_mnemonic",
    "list_synthesizable_mnemonics",
    "synthesize_corpus",
    "synthesize_for_isa",
]

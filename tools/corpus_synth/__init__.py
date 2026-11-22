"""Standalone greedy Valhall-ISA / SPIR-V-op corpus synthesizer."""

from tools.corpus_synth.greedy_driver import (
    CorpusSynthResult,
    SynthesizedShaderRecord,
    emit_shader_bytes,
    synthesize_corpus,
)
from tools.corpus_synth.op_catalog import CATALOG_BY_NAME, OP_CATALOG, OpSpec
from tools.corpus_synth.oracle import CoverageOracle, OracleResult

__all__ = [
    "CATALOG_BY_NAME",
    "CorpusSynthResult",
    "CoverageOracle",
    "OP_CATALOG",
    "OpSpec",
    "OracleResult",
    "SynthesizedShaderRecord",
    "emit_shader_bytes",
    "synthesize_corpus",
]

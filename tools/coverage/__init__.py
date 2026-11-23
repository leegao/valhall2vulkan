"""Concolic Coverage Synthesis, Maximal Reachability, Shrinking & Equivalence Fuzzing."""

from tools.coverage.equiv_fuzzer import ConcolicEquivFuzzResult, run_concolic_equiv_fuzz
from tools.coverage.reachability import (
    DisjointPathRecord,
    MaximalReachabilityResult,
    solve_maximal_reachability,
)
from tools.coverage.shrinker import (
    DiffPolicy,
    DivergenceReport,
    classify_value_diff,
    compare_concrete_outputs,
    localize_and_shrink_divergence,
    shrink_counterexample_overrides,
)
from tools.coverage.synthesizer import (
    CoverageFuzzResult,
    collect_predicate_nodes,
    extract_concrete_overrides_from_model,
    measure_predicate_coverage,
    synthesize_coverage_vectors,
)
from tools.symbolic_ssa import (
    BlockEffectState,
    ConcreteShaderEnv,
    ConcreteSsaInterpreter,
    ShaderEnvBase,
    SsaInterpreter,
    SymbolicShaderEnv,
    SymbolicSsaInterpreter,
    evaluate_ssa,
)

__all__ = [
    "BlockEffectState",
    "ConcolicEquivFuzzResult",
    "ConcreteShaderEnv",
    "ConcreteSsaInterpreter",
    "CoverageFuzzResult",
    "DiffPolicy",
    "DisjointPathRecord",
    "DivergenceReport",
    "MaximalReachabilityResult",
    "ShaderEnvBase",
    "SsaInterpreter",
    "SymbolicShaderEnv",
    "SymbolicSsaInterpreter",
    "classify_value_diff",
    "collect_predicate_nodes",
    "compare_concrete_outputs",
    "evaluate_ssa",
    "extract_concrete_overrides_from_model",
    "localize_and_shrink_divergence",
    "measure_predicate_coverage",
    "run_concolic_equiv_fuzz",
    "shrink_counterexample_overrides",
    "solve_maximal_reachability",
    "synthesize_coverage_vectors",
]

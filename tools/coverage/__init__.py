"""
Concolic / MaxSMT coverage synthesizer, unified SSA evaluator, disjoint
maximal-reachability solver, and divergence auto-shrinker.
"""

from tools.coverage.evaluator import (
    BlockEffectState,
    ConcreteShaderEnv,
    ConcreteSsaInterpreter,
    DagSsaZ3Evaluator,
    ShaderEnvBase,
    SharedShaderEnv,
    evaluate_ssa,
)
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

__all__ = [
    "BlockEffectState",
    "ConcreteShaderEnv",
    "ConcreteSsaInterpreter",
    "CoverageFuzzResult",
    "DagSsaZ3Evaluator",
    "DisjointPathRecord",
    "DiffPolicy",
    "DivergenceReport",
    "classify_value_diff",
    "MaximalReachabilityResult",
    "ShaderEnvBase",
    "SharedShaderEnv",
    "collect_predicate_nodes",
    "compare_concrete_outputs",
    "evaluate_ssa",
    "extract_concrete_overrides_from_model",
    "localize_and_shrink_divergence",
    "measure_predicate_coverage",
    "shrink_counterexample_overrides",
    "solve_maximal_reachability",
    "synthesize_coverage_vectors",
]

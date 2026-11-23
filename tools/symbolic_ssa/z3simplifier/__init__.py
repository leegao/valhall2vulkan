#!/usr/bin/env python3
"""
General Z3 Term & Predicate Simplification Framework (`tools.symbolic_ssa.z3simplifier`).

Public API:
  - Data models: `SimplifyQuery`, `SideObligation`, `SimplificationContext`,
    `SimplifyResult`, `RewriteStats`, `PassStats`, `MiterSolveResult`, `AstCensus`.
  - Pass interfaces: `Z3TermPass`, `Z3QueryPass`, `TermPassAdapter`,
    `FixedPointPass`, `PassPipeline`.
  - Built-in passes:
      * `BvWrapperPeepholePass` (Issue 1: BV64/Bool wrapper elimination)
      * `BvAlgebraicPass` (Issue 6: BV mul-by-pow2 -> shl, add/shift folding)
      * `BridgeCancelPass` (round-trip `u_f32_to_bv32(u_bv32_to_f32(x))` cancel)
      * `ReadOnlyArrayScalarizePass` (Issue 2: sound read-only array scalarization)
      * `InjectiveBridgeUnwrapPass` (Issue 3: outer `u_f32_to_bv32` unwrapping)
      * `PolynomialNormalizePass` (Issue 3: bounded `som=True` polynomial collapse)
      * `TheoryPurifyRealPass` (Issue 2: `QF_AUFBVNRA -> QF_NRA` purification)
      * `ExactCseUnificationPass` (cross-side exact subexpression lifting)
      * `ConditionedUnificationPass` (WLP-justified near-identical subexpression lifting)
  - Convenience builders:
      * `build_exact_pipeline()` (100% equivalence-preserving passes only)
      * `build_default_pipeline(...)` (exact passes + sound over-approx passes)
      * `simplify_query(...)`, `solve_simplified_miter(...)`, `format_z3_ast(...)`.
"""

from __future__ import annotations

from typing import List, Optional

from tools.symbolic_ssa.z3simplifier.ast_utils import (
    AstCensus,
    census_ast,
    count_ast_nodes,
    format_z3_ast,
    rebuild_app,
    rewrite_dag,
)
from tools.symbolic_ssa.z3simplifier.context import (
    PassStats,
    RewriteStats,
    SideObligation,
    SimplificationContext,
    SimplifyQuery,
    SimplifyResult,
)
from tools.symbolic_ssa.z3simplifier.pass_base import (
    FixedPointPass,
    PassPipeline,
    TermPassAdapter,
    Z3QueryPass,
    Z3TermPass,
)
from tools.symbolic_ssa.z3simplifier.passes_array import (
    ReadOnlyArrayScalarizePass,
    TheoryPurifyRealPass,
)
from tools.symbolic_ssa.z3simplifier.passes_bv import (
    BvAlgebraicPass,
    BvWrapperPeepholePass,
)
from tools.symbolic_ssa.z3simplifier.passes_real import (
    BridgeCancelPass,
    InjectiveBridgeUnwrapPass,
    PolynomialNormalizePass,
    PreconditionConstSubPass,
    RealAlgebraicPass,
)
from tools.symbolic_ssa.z3simplifier.passes_unify import (
    ConditionedUnificationPass,
    ExactCseUnificationPass,
)
from tools.symbolic_ssa.z3simplifier.solver import (
    MiterSolveResult,
    solve_simplified_miter,
)


def build_exact_pipeline(*, max_rounds: int = 3) -> PassPipeline:
    """Build a pipeline containing only 100% equivalence-preserving passes."""
    local_fp = FixedPointPass(
        passes=[
            TermPassAdapter(BvWrapperPeepholePass()),
            TermPassAdapter(BvAlgebraicPass()),
            TermPassAdapter(RealAlgebraicPass()),
            TermPassAdapter(BridgeCancelPass()),
        ],
        name="local_exact_fp",
        max_rounds=max_rounds,
    )
    return PassPipeline(
        passes=[
            PreconditionConstSubPass(),
            local_fp,
            ReadOnlyArrayScalarizePass(),
            PolynomialNormalizePass(),
        ]
    )


def build_default_pipeline(
    *,
    enable_cse: bool = True,
    enable_conditioned_unify: bool = False,
    enable_purify: bool = True,
    max_rounds: int = 3,
) -> PassPipeline:
    """
    Build the standard miter simplification pipeline:
      1. Equivalence-preserving local fixed-point (`BvWrapperPeephole`,
         `BvAlgebraic`, `BridgeCancel`) + `ReadOnlyArrayScalarize`.
      2. `InjectiveBridgeUnwrap` (strips outer `u_f32_to_bv32` on both sides).
      3. `PolynomialNormalize` (collapses FMA/associativity diffs in < 1 ms).
      4. `ExactCseUnification` (lifts shared subexpressions across `lhs`/`rhs`).
      5. Optional `ConditionedUnification` (WLP-discharged near-match lifting).
      6. `TheoryPurifyReal` + final `PolynomialNormalize` (unlocks `QF_NRA` `nlsat`).
    """
    local_fp = FixedPointPass(
        passes=[
            TermPassAdapter(BvWrapperPeepholePass()),
            TermPassAdapter(BvAlgebraicPass()),
            TermPassAdapter(RealAlgebraicPass()),
            TermPassAdapter(BridgeCancelPass()),
        ],
        name="local_exact_fp",
        max_rounds=max_rounds,
    )
    passes: List[Z3QueryPass] = [
        PreconditionConstSubPass(),
        local_fp,
        ReadOnlyArrayScalarizePass(),
        InjectiveBridgeUnwrapPass(),
        PolynomialNormalizePass(),
    ]
    if enable_cse:
        passes.append(ExactCseUnificationPass(min_depth=2))
    if enable_conditioned_unify:
        passes.append(ConditionedUnificationPass(min_depth=2, max_depth=10, max_candidates=12))
    if enable_purify:
        passes.append(TheoryPurifyRealPass())
        passes.append(PolynomialNormalizePass())
    return PassPipeline(passes=passes)


def simplify_query(
    query: SimplifyQuery,
    *,
    ctx: Optional[SimplificationContext] = None,
    pipeline: Optional[PassPipeline] = None,
) -> SimplifyResult:
    """Convenience entry point running `pipeline` (or `build_default_pipeline()`) on `query`."""
    pipe = pipeline if pipeline is not None else build_default_pipeline()
    return pipe.run(query, ctx)


__all__ = [
    "AstCensus",
    "BridgeCancelPass",
    "BvAlgebraicPass",
    "BvWrapperPeepholePass",
    "ConditionedUnificationPass",
    "ExactCseUnificationPass",
    "FixedPointPass",
    "InjectiveBridgeUnwrapPass",
    "MiterSolveResult",
    "PassPipeline",
    "PassStats",
    "PolynomialNormalizePass",
    "PreconditionConstSubPass",
    "RealAlgebraicPass",
    "ReadOnlyArrayScalarizePass",
    "RewriteStats",
    "SideObligation",
    "SimplificationContext",
    "SimplifyQuery",
    "SimplifyResult",
    "TermPassAdapter",
    "TheoryPurifyRealPass",
    "Z3QueryPass",
    "Z3TermPass",
    "build_default_pipeline",
    "build_exact_pipeline",
    "census_ast",
    "count_ast_nodes",
    "format_z3_ast",
    "rebuild_app",
    "rewrite_dag",
    "simplify_query",
    "solve_simplified_miter",
]

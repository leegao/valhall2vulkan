#!/usr/bin/env python3
"""
`tools.symbolic_ssa.z3simplifier` — General Z3 Term & Predicate Simplification Framework.

Public API:
  - `SimplifyQuery`, `SimplificationContext`, `SimplifyResult`, `RewriteStats`, `PassStats`
  - `Z3TermPass`, `Z3QueryPass`, `TermPassAdapter`, `FixedPointPass`, `PassPipeline`
  - `dsl`: typed `match`/`case` AST pattern-matching DSL (`view`, `unwrap`, `Z3Op*`)
  - `DataflowPropagatePass`: Z3-domain dataflow engine (`EqualityTable`, `KnownBits`, path-sensitive `If`)
  - `MergedLocalPeepholePass`: single-walk fused BV + Real + Bridge peephole pass
  - `build_exact_pipeline()`, `build_default_pipeline()`, `simplify_query()`
  - `solve_simplified_miter()`, `MiterSolveResult`
  - `format_z3_ast()`, `census_ast()`, `count_ast_nodes()`
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import z3

from tools.symbolic_ssa.z3simplifier.ast_utils import (
    AstCensus,
    census_ast,
    count_ast_nodes,
    format_z3_ast,
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
from tools.symbolic_ssa.z3simplifier.dataflow import (
    DataflowPropagatePass,
    EqualityTable,
    KnownBits,
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
    _bv_algebraic_rule,
    _bv_wrapper_rule,
)
from tools.symbolic_ssa.z3simplifier.passes_real import (
    BridgeCancelPass,
    InjectiveBridgeUnwrapPass,
    PolynomialNormalizePass,
    PreconditionConstSubPass,
    RealAlgebraicPass,
    _bridge_cancel_rule,
    _real_algebraic_rule,
)
from tools.symbolic_ssa.z3simplifier.passes_ematch import (
    EmatchingUnifyPass,
    build_bridge_ematch_axioms,
    build_extensionality_witness_axiom,
)
from tools.symbolic_ssa.z3simplifier.passes_unify import (
    ConditionedUnificationPass,
    ExactCseUnificationPass,
)
from tools.symbolic_ssa.z3simplifier.solver import (
    MiterSolveResult,
    solve_simplified_miter,
)


def _merged_local_rule(node: z3.ExprRef) -> Optional[z3.ExprRef]:
    """Single-pass fused local peephole rule (BV wrapper + BV algebraic + Bridge + Real algebraic)."""
    r = _bv_wrapper_rule(node)
    if r is not None:
        return r
    r = _bv_algebraic_rule(node)
    if r is not None:
        return r
    r = _bridge_cancel_rule(node)
    if r is not None:
        return r
    return _real_algebraic_rule(node)


class MergedLocalPeepholePass(Z3TermPass):
    """Fuses BV wrapper, BV algebraic, bridge-cancel/If-hoist, and Real algebraic rules into 1 DAG walk."""

    name = "merged_local_peephole"
    is_equivalence_preserving = True

    def rewrite_term(
        self,
        expr: z3.ExprRef,
        ctx: SimplificationContext,
    ) -> Tuple[z3.ExprRef, int]:
        return rewrite_dag(expr, _merged_local_rule)


def build_exact_pipeline() -> PassPipeline:
    """
    Build a strictly equivalence-preserving (`SAT(Q') <=> SAT(Q)`) simplification
    pipeline with zero over-approximation.
    """
    return PassPipeline(
        [
            DataflowPropagatePass(max_path_depth=2),
            FixedPointPass(
                [TermPassAdapter(MergedLocalPeepholePass())],
                name="local_exact_fp",
            ),
            ReadOnlyArrayScalarizePass(),
            PolynomialNormalizePass(),
        ]
    )


def build_default_pipeline(
    *,
    enable_cse_unify: bool = True,
    enable_conditioned_unify: bool = False,
    enable_ematching: bool = False,
    enable_theory_purify: bool = True,
) -> PassPipeline:
    """
    Build the default miter simplification pipeline combining exact passes first
    (preserving `exact_query` checkpoint) followed by UNSAT-sound over-approx
    passes (bridge unwrap, CSE/conditioned unification, and `QF_NRA` purification).
    """
    passes: list[Z3QueryPass] = [
        DataflowPropagatePass(max_path_depth=2),
        FixedPointPass(
            [TermPassAdapter(MergedLocalPeepholePass())],
            name="local_exact_fp",
        ),
        ReadOnlyArrayScalarizePass(),
        InjectiveBridgeUnwrapPass(),
        PolynomialNormalizePass(),
    ]
    if enable_cse_unify:
        passes.append(ExactCseUnificationPass())
    if enable_conditioned_unify:
        passes.append(ConditionedUnificationPass())
    if enable_ematching:
        passes.append(EmatchingUnifyPass())
    if enable_theory_purify:
        passes.append(TheoryPurifyRealPass())
        passes.append(PolynomialNormalizePass())
    return PassPipeline(passes)


def simplify_query(
    lhs: z3.ExprRef,
    rhs: z3.ExprRef,
    *,
    guard: Optional[z3.BoolRef] = None,
    preconditions: Sequence[z3.BoolRef] = (),
    pipeline: Optional[PassPipeline] = None,
    ctx: Optional[SimplificationContext] = None,
) -> SimplifyResult:
    """Convenience entry point to simplify a single `(lhs, rhs)` miter query."""
    q = SimplifyQuery(
        lhs=lhs,
        rhs=rhs,
        guard=guard,
        preconditions=tuple(preconditions),
    )
    pipe = pipeline if pipeline is not None else build_default_pipeline()
    eff_ctx = ctx if ctx is not None else SimplificationContext()
    return pipe.run(q, eff_ctx)


__all__ = [
    "AstCensus",
    "BridgeCancelPass",
    "BvAlgebraicPass",
    "BvWrapperPeepholePass",
    "ConditionedUnificationPass",
    "DataflowPropagatePass",
    "EmatchingUnifyPass",
    "EqualityTable",
    "ExactCseUnificationPass",
    "FixedPointPass",
    "InjectiveBridgeUnwrapPass",
    "KnownBits",
    "MergedLocalPeepholePass",
    "MiterSolveResult",
    "PassPipeline",
    "PassStats",
    "PolynomialNormalizePass",
    "PreconditionConstSubPass",
    "ReadOnlyArrayScalarizePass",
    "RealAlgebraicPass",
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
    "build_bridge_ematch_axioms",
    "build_exact_pipeline",
    "build_extensionality_witness_axiom",
    "census_ast",
    "count_ast_nodes",
    "format_z3_ast",
    "rewrite_dag",
    "simplify_query",
    "solve_simplified_miter",
]

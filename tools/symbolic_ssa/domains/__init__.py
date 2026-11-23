"""Unified Valhall SSA value-domain package."""

from tools.symbolic_ssa.domains.abstract_top import AbstractTopDomain
from tools.symbolic_ssa.domains.base import SsaDomain
from tools.symbolic_ssa.domains.z3_base import Z3SymbolicDomain, make_z3_symbolic_domain
from tools.symbolic_ssa.domains.combinators import (
    MULTISET_BOTTOM,
    MULTISET_TOP,
    UNION_BOTTOM,
    UNION_TOP,
    MultiSetDomain,
    MultiSetVal,
    PointwiseUnionDomain,
    TupleDomain,
    TupleVal,
    UnionVal,
)
from tools.symbolic_ssa.domains.concrete_bitvec import ConcreteBitvecDomain
from tools.symbolic_ssa.domains.symbolic_z3_bitvec import (
    SymbolicZ3BitvecDomain,
    _bv32_to_real,
    _f32_bits_to_real_const,
    _is_real2bv,
    _real_cos,
    _real_exp2,
    _real_log2,
    _real_sin,
    _real_sqrt,
    _real_to_bv32,
)
from tools.symbolic_ssa.domains.udomain import (
    ConcreteUDomain,
    PtrVal,
    SymbolicZ3UDomain,
    SymVal,
    UVal,
    UValKind,
)

# Backward-compatible aliases
ConcreteDomain = ConcreteBitvecDomain
SymbolicDomain = SymbolicZ3BitvecDomain

__all__ = [
    "AbstractTopDomain",
    "ConcreteBitvecDomain",
    "ConcreteDomain",
    "ConcreteUDomain",
    "MULTISET_BOTTOM",
    "MULTISET_TOP",
    "MultiSetDomain",
    "MultiSetVal",
    "PointwiseUnionDomain",
    "PtrVal",
    "SsaDomain",
    "Z3SymbolicDomain",
    "make_z3_symbolic_domain",
    "SymVal",
    "SymbolicDomain",
    "SymbolicZ3BitvecDomain",
    "SymbolicZ3UDomain",
    "TupleDomain",
    "TupleVal",
    "UNION_BOTTOM",
    "UNION_TOP",
    "UVal",
    "UValKind",
    "UnionVal",
]

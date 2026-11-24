"""Unified Valhall SSA value-domain package."""

from tools.symbolic_ssa.domains.abstract_top import AbstractTopDomain
from tools.symbolic_ssa.domains.base import LaneVal, SsaDomain, ValueDomain
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
from tools.symbolic_ssa.domains.shader_env import (
    ConcreteShaderEnv,
    ShaderEnvBase,
    SymbolicShaderEnv,
)
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
from tools.symbolic_ssa.domains.z3_base import Z3SymbolicDomain, make_z3_symbolic_domain

# Backward-compatible aliases
ConcreteDomain = ConcreteBitvecDomain
SymbolicDomain = SymbolicZ3BitvecDomain

__all__ = [
    "MULTISET_BOTTOM",
    "MULTISET_TOP",
    "UNION_BOTTOM",
    "UNION_TOP",
    "AbstractTopDomain",
    "ConcreteBitvecDomain",
    "ConcreteDomain",
    "ConcreteShaderEnv",
    "ConcreteUDomain",
    "LaneVal",
    "MultiSetDomain",
    "MultiSetVal",
    "PointwiseUnionDomain",
    "PtrVal",
    "ShaderEnvBase",
    "SsaDomain",
    "SymVal",
    "SymbolicDomain",
    "SymbolicShaderEnv",
    "SymbolicZ3BitvecDomain",
    "SymbolicZ3UDomain",
    "TupleDomain",
    "TupleVal",
    "UVal",
    "UValKind",
    "UnionVal",
    "ValueDomain",
    "Z3SymbolicDomain",
    "make_z3_symbolic_domain",
]

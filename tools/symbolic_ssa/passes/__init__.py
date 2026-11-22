"""Stratum 1 canonicalization passes over ValhallSsaModule."""

from tools.symbolic_ssa.passes.ir_utils import (
    COMMUTATIVE_OPS,
    CONTROL_MARKER_OPS,
    NEVER_GVN_OPS,
    SIDE_EFFECT_OPS,
    AbstractSsaPass,
    Stratum1Stats,
    cfg_isomorphic,
)
from tools.symbolic_ssa.passes.pass_contract_chains import pass_contract_chains
from tools.symbolic_ssa.passes.pass_gvn_dce import AlgebraicSimplifierInterpreter, pass_gvn_dce
from tools.symbolic_ssa.passes.pass_if_convert import pass_if_convert
from tools.symbolic_ssa.passes.pass_predicated_kill import pass_predicated_kill
from tools.symbolic_ssa.passes.pass_prune_const_branches import (
    ConstantFoldInterpreter,
    pass_prune_const_branches,
)
from tools.symbolic_ssa.passes.pass_readonly_tokens import (
    BufferProvenanceInterpreter,
    pass_readonly_tokens,
)
from tools.symbolic_ssa.passes.pipeline import canonicalize_ssa_module

__all__ = [
    "AbstractSsaPass",
    "AlgebraicSimplifierInterpreter",
    "BufferProvenanceInterpreter",
    "COMMUTATIVE_OPS",
    "CONTROL_MARKER_OPS",
    "ConstantFoldInterpreter",
    "NEVER_GVN_OPS",
    "SIDE_EFFECT_OPS",
    "Stratum1Stats",
    "canonicalize_ssa_module",
    "cfg_isomorphic",
    "pass_contract_chains",
    "pass_gvn_dce",
    "pass_if_convert",
    "pass_predicated_kill",
    "pass_prune_const_branches",
    "pass_readonly_tokens",
]

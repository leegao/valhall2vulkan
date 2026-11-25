"""Stratum 1 canonicalization passes and CFG tile-rewrite DSL over ValhallSsaModule."""

from tools.symbolic_ssa.passes.cfg_tile_rewriter import (
    CfgRewriteContext,
    CfgTileRule,
    TileMatch,
    apply_cfg_tile_rules,
)
from tools.symbolic_ssa.passes.dominance import (
    DominanceViolation,
    compute_idom,
    verify_ssa_dominance,
)
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
from tools.symbolic_ssa.passes.pass_gvn_dce import (
    AlgebraicSimplifierInterpreter,
    pass_gvn_dce,
)
from tools.symbolic_ssa.passes.pass_if_convert import (
    make_pure_diamond_rule,
    make_pure_triangle_rule,
    pass_if_convert,
)
from tools.symbolic_ssa.passes.pass_predicated_kill import pass_predicated_kill
from tools.symbolic_ssa.passes.pass_prune_const_branches import (
    ConstantFoldInterpreter,
    ConstantPropDomain,
    ConstVal,
    join_const,
    pass_prune_const_branches,
)
from tools.symbolic_ssa.passes.pass_readonly_tokens import (
    BufferProvenanceInterpreter,
    pass_readonly_tokens,
)
from tools.symbolic_ssa.passes.pipeline import canonicalize_ssa_module

__all__ = [
    "COMMUTATIVE_OPS",
    "CONTROL_MARKER_OPS",
    "NEVER_GVN_OPS",
    "SIDE_EFFECT_OPS",
    "AbstractSsaPass",
    "AlgebraicSimplifierInterpreter",
    "BufferProvenanceInterpreter",
    "CfgRewriteContext",
    "CfgTileRule",
    "ConstVal",
    "ConstantFoldInterpreter",
    "ConstantPropDomain",
    "DominanceViolation",
    "Stratum1Stats",
    "TileMatch",
    "apply_cfg_tile_rules",
    "canonicalize_ssa_module",
    "cfg_isomorphic",
    "compute_idom",
    "join_const",
    "make_pure_diamond_rule",
    "make_pure_triangle_rule",
    "pass_contract_chains",
    "pass_gvn_dce",
    "pass_if_convert",
    "pass_predicated_kill",
    "pass_prune_const_branches",
    "pass_readonly_tokens",
    "verify_ssa_dominance",
]

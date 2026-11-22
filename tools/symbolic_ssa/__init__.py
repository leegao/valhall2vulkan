"""Unified Valhall SSA domains, concrete/symbolic interpreters, and canonicalization passes."""

from typing import Any, Dict, Optional

from tools.symbolic_ssa.concrete_interpreter import ConcreteSsaInterpreter
from tools.symbolic_ssa.domains import (
    AbstractTopDomain,
    ConcreteDomain,
    SsaDomain,
    SymbolicDomain,
)
from tools.symbolic_ssa.env import (
    BV8,
    BV32,
    BV64,
    FP32_SORT,
    MASK32,
    MASK64,
    REAL_SORT,
    ConcreteShaderEnv,
    ShaderEnvBase,
    SharedShaderEnv,
    Z3Val,
    _CORNER_U32S,
    _as_bool,
    _as_bv32,
    _as_bv64,
    _bv32_to_real,
    _c_bool,
    _c_elemwise,
    _c_u32,
    _c_u64,
    _c_vec_len,
    _det_hash_u32,
    _eval_elementwise,
    _f16_to_f32_u32,
    _f32_bits_to_real_const,
    _f32_to_f16_u16,
    _f32_to_u32,
    _f32_ulp_diff,
    _is_real2bv,
    _real_cos,
    _real_exp2,
    _real_log2,
    _real_sin,
    _real_sqrt,
    _real_to_bv32,
    _s32,
    _s64,
    _try_const_u64,
    _u32_to_f32,
    _vec_len,
    _z3_and,
    _z3_if,
    _z3_not,
    _z3_or,
)
from tools.symbolic_ssa.interpreter import SsaInterpreter
from tools.symbolic_ssa.passes import (
    SIDE_EFFECT_OPS,
    Stratum1Stats,
    canonicalize_ssa_module,
    cfg_isomorphic,
)
from tools.symbolic_ssa.symbolic_interpreter import (
    BlockEffectState,
    DagSsaZ3Evaluator,
    SymbolicSsaInterpreter,
    _clone_state,
    _merge_states,
)
from tools.verify.panfrost_z3_model import (
    _hw_uf,
    bv32,
    bv32_to_fp32,
    bv64,
    f16_bv_to_f32_bv,
    f32_bv_to_f16_bv,
    fp32_to_bv32,
)


def evaluate_ssa(
    bundle: Dict[str, Any],
    ssa_json: Dict[str, Any],
    is_valhall_side: bool = False,
    mode: str = "symbolic",
    fp_mode: str = "exact",
    max_unroll: int = 1,
    vec_idx: int = 0,
    overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    """Unified entrypoint for evaluating a ValhallSsaModule symbolically or concretely."""
    if mode == "concrete":
        cenv = ConcreteShaderEnv(bundle, vec_idx=vec_idx, overrides=overrides)
        return ConcreteSsaInterpreter(ssa_json, is_valhall_side=is_valhall_side).execute_one(cenv)
    env = SharedShaderEnv(bundle)
    ev = SymbolicSsaInterpreter(ssa_json, env, is_valhall_side=is_valhall_side, fp_mode=fp_mode, max_unroll=max_unroll)
    st = ev.evaluate()
    return ev, st, env


__all__ = [
    "AbstractTopDomain",
    "BV8",
    "BV32",
    "BV64",
    "BlockEffectState",
    "ConcreteDomain",
    "ConcreteShaderEnv",
    "ConcreteSsaInterpreter",
    "DagSsaZ3Evaluator",
    "FP32_SORT",
    "MASK32",
    "MASK64",
    "REAL_SORT",
    "SIDE_EFFECT_OPS",
    "ShaderEnvBase",
    "SharedShaderEnv",
    "SsaDomain",
    "SsaInterpreter",
    "Stratum1Stats",
    "SymbolicDomain",
    "SymbolicSsaInterpreter",
    "Z3Val",
    "canonicalize_ssa_module",
    "cfg_isomorphic",
    "evaluate_ssa",
]

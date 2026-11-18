"""
Formal Verification & Panfrost Reference Semantics Package for valhall2spirv.
"""

from .all_paths_miter import (
    MiterResult,
    check_instruction_miter,
    run_all_627_miter,
)
from .cpp_handler_extractor import (
    CppHandlerRegistry,
    CppSymbolicEvaluator,
    InstLoweringContext,
    SourceOperandDesc,
    SymbolicSsaModule,
    SymbolicSsaNode,
)
from .lowering_semantics_extractor import (
    LoweringLogger,
    LoweringSemanticsExtractor,
    LoweringTrace,
)
from .panfrost_z3_model import (
    PANFROST_SEMANTICS,
    ModelTrace,
    ValhallZ3State,
    VerificationLogger,
    apply_clamp_f32,
    apply_clamp_v2f16,
    apply_src_mod_32,
    apply_src_mod_64,
    apply_swizzle_u32,
    apply_swizzle_u64,
    eval_panfrost_instruction,
    extract_panfrost_spec_catalog,
    verify_instruction_equivalence,
    write_dst_32,
    write_dst_64,
)
from .ssa_z3_eval import (
    SsaZ3EvalResult,
    eval_ssa_module_z3,
)

__all__ = [
    "CppHandlerRegistry",
    "CppSymbolicEvaluator",
    "InstLoweringContext",
    "LoweringLogger",
    "LoweringSemanticsExtractor",
    "LoweringTrace",
    "MiterResult",
    "ModelTrace",
    "PANFROST_SEMANTICS",
    "SourceOperandDesc",
    "SsaZ3EvalResult",
    "SymbolicSsaModule",
    "SymbolicSsaNode",
    "ValhallZ3State",
    "VerificationLogger",
    "apply_clamp_f32",
    "apply_clamp_v2f16",
    "apply_src_mod_32",
    "apply_src_mod_64",
    "apply_swizzle_u32",
    "apply_swizzle_u64",
    "check_instruction_miter",
    "eval_panfrost_instruction",
    "eval_ssa_module_z3",
    "extract_panfrost_spec_catalog",
    "run_all_627_miter",
    "verify_instruction_equivalence",
    "write_dst_32",
    "write_dst_64",
]

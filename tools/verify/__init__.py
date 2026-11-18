"""
Formal Verification & Panfrost Reference Semantics Package for valhall2spirv.
"""

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

__all__ = [
    "CppHandlerRegistry",
    "CppSymbolicEvaluator",
    "InstLoweringContext",
    "LoweringLogger",
    "LoweringSemanticsExtractor",
    "LoweringTrace",
    "ModelTrace",
    "PANFROST_SEMANTICS",
    "SourceOperandDesc",
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
    "eval_panfrost_instruction",
    "extract_panfrost_spec_catalog",
    "verify_instruction_equivalence",
    "write_dst_32",
    "write_dst_64",
]

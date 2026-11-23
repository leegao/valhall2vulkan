"""
Formal Verification & Panfrost Reference Semantics Package for valhall2spirv.
"""

from .clang_ast_evaluator import (
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

_LAZY_EXPORTS = {
    "MiterResult": (".all_paths_miter", "MiterResult"),
    "check_instruction_miter": (".all_paths_miter", "check_instruction_miter"),
    "run_all_627_miter": (".all_paths_miter", "run_all_627_miter"),
    "InstructionSsaZ3Interpreter": (".ssa_z3_eval", "InstructionSsaZ3Interpreter"),
    "SsaZ3EvalResult": (".ssa_z3_eval", "SsaZ3EvalResult"),
    "eval_ssa_module_z3": (".ssa_z3_eval", "eval_ssa_module_z3"),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        import importlib
        mod_name, attr = _LAZY_EXPORTS[name]
        mod = importlib.import_module(mod_name, __name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CppHandlerRegistry",
    "CppSymbolicEvaluator",
    "InstLoweringContext",
    "InstructionSsaZ3Interpreter",
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

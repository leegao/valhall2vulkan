"""Shader-level formal equivalence verification (Z3 miter, cutpoint CEC, and SPIR-V I/O)."""

from .cec import CecResult, run_concrete_diff, verify_with_cec
from .shader_equiv_verifier import (
    ShaderEquivResult,
    verify_equivalence_bundle,
    verify_spv_file,
)
from .spvasm_io import (
    assemble_spvasm,
    disassemble_spv,
    find_spirv_tool,
    load_equiv_bundle,
    validate_spv,
)

__all__ = [
    "CecResult",
    "ShaderEquivResult",
    "assemble_spvasm",
    "disassemble_spv",
    "find_spirv_tool",
    "load_equiv_bundle",
    "run_concrete_diff",
    "validate_spv",
    "verify_with_cec",
    "verify_equivalence_bundle",
    "verify_spv_file",
]

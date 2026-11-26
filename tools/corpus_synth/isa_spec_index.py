"""Backwards-compatible re-export of `tools.spec.isa_spec_index`."""

from tools.spec.isa_spec_index import (
    DEFAULT_ISA_XML,
    STRUCTURAL_FIELD_NAMES,
    IsaEnum,
    IsaFieldSpec,
    IsaSpecIndex,
    IsaVariantSpec,
    load_isa_spec_index,
)

__all__ = [
    "DEFAULT_ISA_XML",
    "STRUCTURAL_FIELD_NAMES",
    "IsaEnum",
    "IsaFieldSpec",
    "IsaSpecIndex",
    "IsaVariantSpec",
    "load_isa_spec_index",
]

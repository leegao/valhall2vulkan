#pragma once

#include "valhall_ssa.h"

#include <string>

namespace valhall::compiler {

/// Return the canonical string name of a ValType enum value.
const char* val_type_name(ValType vt);

/// Serialize a ValhallSsaModule into a compact JSON string for external analysis/verification.
std::string serialize_ssa_module_json(const ValhallSsaModule& mod);

/// Format a ValhallSsaModule into a human-readable textual SSA listing.
std::string format_ssa_module_text(const ValhallSsaModule& mod);

} // namespace valhall::compiler

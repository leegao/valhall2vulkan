#pragma once

#include "valhall_ssa.h"

#include <string>

namespace valhall::compiler {

/// Return the canonical string name of a ValType enum value.
const char* val_type_name(ValType vt);

/// Serialize a ValhallSsaModule into a compact JSON string for external analysis/verification.
std::string serialize_ssa_module_json(const ValhallSsaModule& mod);

} // namespace valhall::compiler

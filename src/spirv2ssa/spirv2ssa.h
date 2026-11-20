#pragma once

#include "valhall_ssa.h"
#include "valhall_to_spirv.h"

#include <cstdint>
#include <string>
#include <vector>

namespace valhall::spirv2ssa {

enum class ResourceKind : uint8_t {
    UniformBuffer = 0,
    StorageBuffer,
    PushConstant,
    SampledImage,
    InputAttrOrVarying,
    OutputColorOrVarying,
    BuiltinInput,
    BuiltinOutput,
};

/// Canonical descriptor of a shader interface resource referenced by the lowered ValhallSSA.
struct SpirvResourceBinding {
    ResourceKind kind = ResourceKind::UniformBuffer;
    uint32_t sideband_slot = 0;     // Matches LoadSidebandU64::imm_u32 for buffers
    uint32_t descriptor_set = 0;
    uint32_t binding = 0;
    uint32_t location = 0;
    uint32_t component = 0;
    uint32_t builtin = 0;
    uint32_t spv_var_id = 0;
    std::string name;
    std::string canonical_key;      // e.g. "ssbo:0:1", "ubo:0:4", "push_const", "tex:0:1"
};

struct Spirv2SsaOptions {
    /// If true, emit Nop placeholders for unsupported SPIR-V opcodes instead of failing.
    bool tolerant = true;
    /// Maximum function call inlining depth.
    uint32_t max_inline_depth = 8;
};

struct Spirv2SsaResult {
    bool ok = false;
    std::string error_message;
    ShaderStage stage = ShaderStage::Compute;
    std::string entry_point_name;
    uint32_t local_size_x = 1;
    uint32_t local_size_y = 1;
    uint32_t local_size_z = 1;

    compiler::ValhallSsaModule ssa;
    std::vector<SpirvResourceBinding> bindings;
    std::vector<std::string> unsupported_ops;
};

/// Directly lower a SPIR-V binary module into ValhallSSA without going through Valhall ISA.
Spirv2SsaResult lower_spirv_to_valhall_ssa(
    const std::vector<uint32_t>& spirv_words,
    const Spirv2SsaOptions& opts = {});

} // namespace valhall::spirv2ssa

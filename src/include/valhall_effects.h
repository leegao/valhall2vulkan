#pragma once
#include "valhall_tables_generated.h"
#include "valhall_spirv_builder.h"
#include <cstdint>
#include <set>
#include <string>

namespace valhall::compiler {

struct EffectSet {
    uint32_t cap_bits = 0;
    uint32_t ext_bits = 0;
    uint32_t exec_bits = 0;
    uint32_t flags = 0;

    static constexpr uint32_t kFlagUsesSideband  = 1u << 0;
    static constexpr uint32_t kFlagUsesGlsl450   = 1u << 1;

    static constexpr uint32_t kCapInt64          = 1u << 0;
    static constexpr uint32_t kCapInt16          = 1u << 1;
    static constexpr uint32_t kCapInt8           = 1u << 2;
    static constexpr uint32_t kCapFloat16        = 1u << 3;
    static constexpr uint32_t kCapFloat64        = 1u << 4;
    static constexpr uint32_t kCapPsb            = 1u << 5;
    static constexpr uint32_t kCapGroupNonUnif   = 1u << 6;
    static constexpr uint32_t kCapGroupBallot    = 1u << 7;
    static constexpr uint32_t kCapGroupShuffle   = 1u << 8;
    static constexpr uint32_t kCapSampled1D      = 1u << 9;
    static constexpr uint32_t kCapImage1D        = 1u << 10;

    static constexpr uint32_t kExtPsb            = 1u << 0;
    static constexpr uint32_t kExecDepthRepl     = 1u << 0;

    void join(const tables::EffectMask& m) {
        cap_bits |= m.cap_bits;
        ext_bits |= m.ext_bits;
        exec_bits |= m.exec_bits;
        flags |= m.flags;
    }

    void join(const EffectSet& other) {
        cap_bits |= other.cap_bits;
        ext_bits |= other.ext_bits;
        exec_bits |= other.exec_bits;
        flags |= other.flags;
    }

    bool uses_sideband() const { return (flags & kFlagUsesSideband) != 0; }
    bool uses_glsl_std450() const { return (flags & kFlagUsesGlsl450) != 0; }

    void apply_to_builder(spirv::SpirvBuilder& b) const {
        using namespace spirv;
        if (cap_bits & kCapInt64)        b.add_capability(Capability::Int64);
        if (cap_bits & kCapInt16)        b.add_capability(Capability::Int16);
        if (cap_bits & kCapInt8)         b.add_capability(Capability::Int8);
        if (cap_bits & kCapFloat16)      b.add_capability(Capability::Float16);
        if (cap_bits & kCapFloat64)      b.add_capability(Capability::Float64);
        if (cap_bits & kCapPsb)          b.add_capability(Capability::PhysicalStorageBufferAddresses);
        if (cap_bits & kCapGroupNonUnif) b.add_capability(Capability::GroupNonUniform);
        if (cap_bits & kCapGroupBallot)  b.add_capability(Capability::GroupNonUniformBallot);
        if (cap_bits & kCapGroupShuffle) b.add_capability(Capability::GroupNonUniformShuffle);
        if (cap_bits & kCapSampled1D)    b.add_capability(Capability::Sampled1D);
        if (cap_bits & kCapImage1D)      b.add_capability(Capability::Image1D);
        if (ext_bits & kExtPsb)          b.add_extension("SPV_KHR_physical_storage_buffer");
        if (exec_bits & kExecDepthRepl)  b.add_execution_mode(ExecutionMode::DepthReplacing);
    }
};

} // namespace valhall::compiler

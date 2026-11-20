#pragma once

#include "abstract_domain.h"
#include "transfer.h"

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace valhall::dataflow {

/// Rich lattice value for Valhall pointer, ABI, resource, constant, and symbol dataflow analysis.
struct TrackedVal {
    enum class Kind : uint8_t {
        Bottom = 0,       // Uninitialized / unreachable
        Top,              // Unknown / over-approximated
        ConstInt,         // Exact 64-bit constant integer
        AbiPreload,       // Hardware-preloaded entry GPR (WorkGroupID, GlobalInvocationID, VertexIndex, etc.)
        FauDstBase,       // Pilot r0:r1 = pilot_fau_dst_va (+ optional byte_offset)
        SrtBase,          // Pilot r2:r3 = srt_base_va (+ optional byte_offset)
        SrtRackPtr,       // SRT[rack] = Resource[rack].address (points to 32B descriptor array)
        DescriptorAddr,   // Pointer to a specific 32B hardware descriptor (SRT[rack]->Desc[idx])
        BufferAddr,       // Buffer[idx].address (64-bit BDA of SSBO/UBO/PushConst/VertexBuf)
        BufferSize,       // Buffer[idx].size
        FauUniformRef,    // Direct reference to FAU[+offset]
        SymbolValue,      // Scalar/vector value loaded from a named symbol (optionally derived)
        DataSectionPtr,   // PC-relative pointer into EBIN embedded constant/LUT data_section
        TlsPtr,           // Thread Local Storage base pointer (FauSpecial::tls_ptr + offset)
        WlsPtr,           // Workgroup Local Storage base pointer (FauSpecial::wls_ptr + offset)
        VertexAttrValue,  // Value loaded via LD_ATTR_IMM
        VaryingValue,     // Value loaded via LD_VAR_IMM
        TexSampleValue,   // Value loaded via TEX_SINGLE / VAR_TEX_SINGLE / TEX_FETCH
        BlendDescRef,     // Reference to blend_descriptor_N
    };

    Kind kind = Kind::Bottom;
    int64_t const_val = 0;         // For ConstInt
    int64_t byte_offset = 0;       // Constant byte offset for pointers (or target_pc for DataSectionPtr)
    int64_t extra_offset = 0;      // For DataSectionPtr: offset relative to data_section_offset
    int32_t rack = -1;             // SRT rack index
    int32_t desc_idx = -1;         // Descriptor index within rack (or location / tex_idx / rt_idx)
    bool is_indexed = false;       // True if a dynamic index was added (e.g. SHADD)
    bool masked_56bit = false;     // True if SrtRackPtr passed through 56-bit VA mask
    bool derived = false;          // True if SymbolValue underwent arithmetic derivation
    std::string symbol_name;       // Canonical symbol or member name
    std::string label;             // Human-readable short description

    static TrackedVal make_bottom() { return TrackedVal{}; }
    static TrackedVal make_top() { TrackedVal v; v.kind = Kind::Top; return v; }
    static TrackedVal make_const(int64_t c);
    static TrackedVal make_abi(const std::string& name);
    static TrackedVal make_fau_dst_base(int64_t off = 0);
    static TrackedVal make_srt_base(int64_t off = 0);
    static TrackedVal make_srt_rack(int32_t rack_idx, bool masked = false);
    static TrackedVal make_buffer_addr(int32_t rack_idx, int32_t desc, const std::string& sym, int64_t off = 0, bool indexed = false);
    static TrackedVal make_buffer_size(int32_t rack_idx, int32_t desc, const std::string& sym);
    static TrackedVal make_symbol_value(int32_t rack_idx, int32_t desc, const std::string& sym, bool is_derived = false);
    static TrackedVal make_data_section_ptr(int64_t target_pc, int64_t sec_off, bool indexed = false);
    static TrackedVal make_tls_ptr(int64_t off = 0);
    static TrackedVal make_wls_ptr(int64_t off = 0);

    bool is_bottom() const { return kind == Kind::Bottom; }
    bool is_top() const { return kind == Kind::Top; }
    bool is_pointer() const {
        return kind == Kind::FauDstBase || kind == Kind::SrtBase ||
               kind == Kind::SrtRackPtr || kind == Kind::DescriptorAddr ||
               kind == Kind::BufferAddr || kind == Kind::DataSectionPtr ||
               kind == Kind::TlsPtr || kind == Kind::WlsPtr;
    }

    TrackedVal with_offset(int64_t delta) const;
    TrackedVal as_indexed() const;
    std::string format_short() const;
};

/// Lattice operations (`Bottom <= concrete <= Top`) for `AbstractState<TrackedValDomain>`.
struct TrackedValDomain {
    using Value = TrackedVal;

    static Value bottom() { return TrackedVal::make_bottom(); }
    static Value top() { return TrackedVal::make_top(); }
    static Value join(const Value& a, const Value& b);
    static Value meet(const Value& a, const Value& b);
    static bool leq(const Value& a, const Value& b);
    static bool equals(const Value& a, const Value& b);
};

using PointerState = AbstractState<TrackedValDomain>;

/// Stage-agnostic environment hooks so `src/dataflow` stays decoupled from MBS2 / tools.
struct PointerAnalysisEnv {
    enum class StageRole : uint8_t {
        Unknown = 0,
        Pilot,
        MainCompute,
        Vertex,
        Fragment,
    };

    StageRole role = StageRole::Unknown;
    uint32_t data_section_offset = 0xFFFFFFFFu;

    // Optional pre-populated FAU map (e.g. from Pilot shader + RLOC + FCST)
    std::map<uint32_t, TrackedVal> initial_fau;

    // Optional symbol resolution callbacks
    std::function<std::string(uint8_t rack, uint32_t desc_idx)> lookup_srt_symbol;
    std::function<std::string(uint8_t rack, uint32_t desc_idx, uint32_t byte_off)> lookup_symbol_member;
    std::function<std::string(uint32_t location)> lookup_input_symbol;
    std::function<std::string(uint32_t location)> lookup_output_symbol;
    std::function<std::string(uint32_t addr)> lookup_image_symbol;
    std::function<std::string(uint32_t addr)> lookup_sampler_symbol;
};

/// A single FAU write discovered while analyzing a Pilot shader.
struct PilotFauWrite {
    uint32_t fau_offset = 0;
    uint32_t byte_size = 4;
    TrackedVal value;
};

/// Complete result of running pointer & ABI dataflow analysis on a decoded EBIN.
struct PointerAnalysisResult {
    std::vector<PointerState> pre_states;
    std::vector<PointerState> post_states;
    std::vector<std::vector<std::string>> inst_notes;  // Semantic events per instruction
    std::map<uint32_t, PilotFauWrite> pilot_fau_writes;
    size_t fixpoint_iterations = 0;
};

/// Build the initial hardware ABI entry state for a given shader stage role.
PointerState make_initial_pointer_state(const PointerAnalysisEnv& env);

/// Run the decoupled pointer, constant, ABI, and resource dataflow pass over a Valhall EBIN.
PointerAnalysisResult run_pointer_analysis(
    const std::vector<DecodedValhallInst>& instructions,
    const PointerAnalysisEnv& env);

}  // namespace valhall::dataflow

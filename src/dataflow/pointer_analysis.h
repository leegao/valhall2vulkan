#pragma once

#include "abstract_domain.h"
#include "liveness.h"
#include "tracked_val.h"
#include "transfer.h"
#include "valhall_ssa.h"

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace valhall::dataflow {

/// External metadata environment supplied to the pure SSA pointer analysis pass.
struct PointerAnalysisEnv {
    enum class StageRole : uint8_t {
        Unknown = 0,
        Pilot,
        MainCompute,
        Vertex,
        Fragment,
    };

    StageRole role = StageRole::Unknown;
    uint8_t arch = 10;
    uint32_t data_section_offset = 0;

    /// Sideband struct member indices (matching `ValhallSidebandData` in `valhall_to_spirv.h`):
    ///   member 2 = `srt_base_va`
    ///   member 3 = `pilot_fau_dst_va`
    uint32_t srt_base_sideband_member = 2;
    uint32_t fau_dst_sideband_member = 3;

    bool is_srt_base_va(uint32_t member_idx) const {
        return role == StageRole::Pilot && member_idx == srt_base_sideband_member;
    }

    bool is_fau_dst_va(uint32_t member_idx) const {
        return role == StageRole::Pilot && member_idx == fau_dst_sideband_member;
    }

    /// Optional pre-seeded FAU values (e.g. from Pilot shader analysis or MBS2 RLOCs).
    std::map<uint32_t, TrackedVal> initial_fau;

    /// Optional symbol resolution callbacks (all may be left empty).
    std::function<std::string(uint8_t rack, uint32_t desc_idx)> lookup_srt_symbol;
    std::function<std::string(uint8_t rack, uint32_t desc_idx, uint32_t byte_off)> lookup_symbol_member;
    std::function<std::string(uint32_t location)> lookup_input_symbol;
    std::function<std::string(uint32_t location)> lookup_output_symbol;
    std::function<std::string(uint32_t addr)> lookup_image_symbol;
    std::function<std::string(uint32_t addr)> lookup_sampler_symbol;
};

/// A single Pilot shader write to FAU_0 discovered via SSA `PsbStore` to `FauDstBase + off`.
struct PilotFauWrite {
    uint32_t fau_offset = 0;
    uint32_t byte_size = 4;
    TrackedVal value;
};

/// Output of the pure SSA-level pointer & ABI dataflow pass.
///
/// Uses `TrackedValSetDomain` (`FiniteSetDomain<TrackedValDomain, 4>`) so up to 4
/// distinct concrete values (e.g., across a `Phi`/`Select` of two constants or two
/// pointers) are preserved precisely before widening to `Top`.
struct PointerAnalysisResult {
    bool converged = false;
    uint32_t iterations = 0;

    /// Converged SSA abstract state (`final_state.val(ssa_id)` -> `TrackedValSet`).
    AbstractState<TrackedValSetDomain> final_state;

    /// Backwards SSA liveness result computed over the SSA module (when provided).
    SsaLivenessResult liveness;

    /// Pilot shader FAU writes discovered (keyed by FAU byte offset).
    std::map<uint32_t, PilotFauWrite> pilot_fau_writes;

    /// Per-SSA-node semantic annotations discovered by the pass (indexed by `SsaId`).
    std::vector<std::vector<std::string>> ssa_notes;

    /// Convenience helper: returns the bounded finite set for `id`.
    TrackedValSet val_set(compiler::SsaId id) const {
        return final_state.val(id);
    }

    /// Convenience helper: collapses `final_state.val(id)` into a single summary `TrackedVal`.
    TrackedVal val(compiler::SsaId id) const {
        return collapse_tracked_val_set(final_state.val(id));
    }
};

/// Primary pure SSA-level pointer & ABI dataflow pass over `compiler::ValhallSsaModule`.
PointerAnalysisResult run_ssa_pointer_analysis(
    const compiler::ValhallSsaModule& mod,
    const PointerAnalysisEnv& env,
    const SsaLivenessResult* liveness = nullptr,
    uint32_t max_iterations = 64);

}  // namespace valhall::dataflow

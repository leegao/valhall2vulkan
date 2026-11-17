#pragma once
#include "valhall_pass1.h"
#include "valhall_spirv_builder.h"
#include <vector>

namespace valhall::compiler {

struct TilerStats {
    size_t total_ssa_nodes = 0;
    size_t live_ssa_nodes = 0;
    size_t fusion_tiles_selected = 0;
    size_t canonical_tiles_selected = 0;
    size_t total_dp_cost = 0;
};

struct TilerResult {
    bool ok = false;
    std::string error_message;
    std::vector<uint32_t> spirv_words;
    EffectSet effects;
    TilerStats stats;
};

TilerResult run_pass2_tiler_and_emit(
    Pass1Result& pass1,
    const ValhallTranslateOptions& opts);

} // namespace valhall::compiler

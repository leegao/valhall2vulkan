#pragma once
#include "valhall_ssa.h"
#include "valhall_to_spirv.h"
#include "valhall_cfg.h"
#include <string>
#include <vector>

namespace valhall::compiler {

struct Pass1Result {
    bool ok = false;
    std::string error_message;
    ValhallSsaModule ssa;
    EffectSet preamble_effects;
    // Resource declarations needed in prelude
    bool uses_sideband = false;
    bool is_fragment_stage = false;
    bool is_compute_or_pilot = false;
    uint32_t local_size_x = 1;
    uint32_t local_size_y = 1;
    uint32_t local_size_z = 1;
};

Pass1Result run_pass1_lowering(
    const cfg::ControlFlowGraph& primary_cfg,
    const cfg::ControlFlowGraph& secondary_cfg,
    const ValhallTranslateOptions& opts);

} // namespace valhall::compiler

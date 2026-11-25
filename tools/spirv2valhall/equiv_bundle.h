#pragma once

#include "spirv2valhall.h"
#include "spirv2ssa.h"
#include "valhall_to_spirv.h"

#include <string>
#include <vector>

namespace valhall::spirv2valhall {

/// Build a self-contained JSON equivalence bundle containing:
/// - `valhall_ssa`: ValhallSsaModule lowered via libMali -> Valhall ISA -> valhall_pass1
/// - `direct_ssa`:  ValhallSsaModule lowered directly via src/spirv2ssa/
/// - `mbs2_context`: FAU FCSTs, FAU RLOCs, Pilot FAU bindings, sinp/sout location maps, pc disassembly
/// - `direct_bindings`: Resource bindings table from src/spirv2ssa/
struct EquivBundleOptions {
    bool run_ssa_opt = false;
    bool run_defau = false;
    bool fp_fast_math = false;
    bool enable_fp_tile_rewrites = true;
    bool enable_fp_cpp_rewrites = true;
};

/// Populate MBS2-derived metadata (workgroup size, samplers, FAU texture handle
/// indices, PKA buffer names, and optionally fragment/vertex varying location maps)
/// onto `vopts` for `valhall_pass1` lowering.
/// Set `remap_varyings = false` when downstream consumers remap `sinp` themselves
/// (e.g. `equiv_bundle` JSON consumers) to prevent double-remapping.
void populate_mbs2_translate_options(
    const ExtractResult& extract_res,
    const ExtractedEbin& primary_eb,
    ValhallTranslateOptions* vopts,
    bool remap_varyings = true);

bool build_equivalence_bundle_json(
    const std::string& spv_path,
    const ExtractResult& extract_res,
    std::string* out_json,
    std::string* err = nullptr,
    const EquivBundleOptions& opt = {});

} // namespace valhall::spirv2valhall

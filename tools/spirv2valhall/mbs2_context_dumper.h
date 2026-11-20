#pragma once

#include "isa_enricher.h"
#include "mbs2_types.h"

#include <string>

namespace valhall::spirv2valhall {

/// Format the CSF Execution Context & Descriptor Graph from cmpbe_chunk_MBS2
/// (everything in MBS2 minus raw EBIN bytecode bytes).
std::string format_csf_execution_context(
    const Mbs2Module& mod,
    const PilotAnalysisResult& pilot_info);

}  // namespace valhall::spirv2valhall

#pragma once

#include "valhall_ssa.h"

#include <cstddef>
#include <cstdint>
#include <functional>

namespace valhall::dataflow {

/// Direction of CFG chaotic worklist iteration.
enum class DataflowDirection : uint8_t {
    Forward = 0,   // Seeds blocks 0..N-1; enqueues successors on change
    Backward = 1,  // Seeds blocks N-1..0; enqueues predecessors on change
};

struct WorklistResult {
    bool converged = false;
    uint32_t iterations = 0;
    size_t visits = 0;
};

/// Domain-independent chaotic worklist fixpoint solver over a `ValhallSsaModule` CFG.
///
/// Decoupled from `DomainT` so both forward abstract interpretation (`transfer.h`)
/// and backwards SSA liveness (`liveness.cpp`) share the exact same worklist engine.
///
/// `transfer_block(block_id)` evaluates block `block_id` and returns `true` iff
/// its outgoing dataflow facts changed (triggering re-queuing of successors for
/// `Forward` or predecessors for `Backward`).
WorklistResult run_cfg_worklist(
    const compiler::ValhallSsaModule& mod,
    DataflowDirection dir,
    const std::function<bool(uint32_t block_id)>& transfer_block,
    uint32_t max_iterations = 64);

}  // namespace valhall::dataflow

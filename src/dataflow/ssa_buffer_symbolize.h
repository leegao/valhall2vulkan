#pragma once

#include "pointer_analysis.h"
#include "ssa_opt_utils.h"
#include "valhall_ssa.h"

#include <cstdint>
#include <string>
#include <vector>

namespace valhall::dataflow {

/// A recognized fixed-stride linear affine access to an SSBO or UBO:
///   `addr = base(buf_key) + index_ssa * stride_bytes + field_offset`
/// (with `index_ssa == kInvalidSsa` and `stride_bytes == 0` for constant-offset accesses).
struct SymbolizedBufferAccess {
    compiler::SsaId node_id = compiler::kInvalidSsa;
    bool is_store = false;
    std::string buf_key;
    compiler::SsaId index_ssa = compiler::kInvalidSsa;
    uint32_t stride_bytes = 0;
    int64_t field_offset = 0;
    uint32_t elem_bytes = 4;
    /// Canonical symbolic slot tag written to `node.aux_str`, e.g.
    /// `"sym:ssbo:0:1[stride=16,off=4]"`.
    std::string sym_slot_tag;
};

/// Options for fixed-stride SSBO/UBO symbolization (`run_ssa_symbolize_buffers`).
struct BufferSymbolizeOptions {
    /// Optional precomputed pointer analysis result; if null, the pass runs or
    /// reuses the module's cached `PointerAnalysisResult`.
    const PointerAnalysisResult* ptr_analysis = nullptr;

    /// Annotate matched `PsbLoad` / `PsbStore` nodes with symbolic slot metadata
    /// (`aux_str`, `imm_u32_2 = stride`, `aux_index = field_offset`, `aux_bool = true`).
    bool annotate_ssa_nodes = true;

    /// Forward intra-block stores to subsequent loads of the exact same symbolic
    /// slot `(buf_key, index_ssa, stride, field_offset)` across non-aliasing ops.
    bool forward_store_to_load = true;
};

struct BufferSymbolizeResult {
    SsaPassStats stats;
    std::vector<SymbolizedBufferAccess> accesses;
};

/// Recognizes fixed-stride SSBO/UBO reads and writes, annotates them with
/// symbolic slot metadata for scalar Z3 reasoning (avoiding Z3 `Array` theory),
/// and performs intra-block symbolic store-to-load forwarding.
BufferSymbolizeResult run_ssa_symbolize_buffers(
    compiler::ValhallSsaModule& mod,
    const BufferSymbolizeOptions& opts = {});

}  // namespace valhall::dataflow

#pragma once

#include "pointer_analysis.h"
#include "ssa_opt_utils.h"
#include "valhall_ssa.h"

#include <cstdint>
#include <string>
#include <vector>

namespace valhall::dataflow {

/// Options for TLS spill promotion (`run_ssa_spill_mem2reg`).
struct SpillMem2RegOptions {
    /// Sideband member index for `thread_local_pointer` (default 6 in Valhall ABI).
    uint32_t tls_sideband_member = 6;

    /// Remove promoted dead TLS stores and loads after rewriting uses.
    bool remove_dead_memory_ops = true;
};

/// Options for fixed-size struct SROA (`run_ssa_sroa`).
struct SroaOptions {
    /// Sideband member index for `thread_local_pointer`.
    uint32_t tls_sideband_member = 6;

    /// Maximum struct footprint in bytes eligible for scalar replacement.
    uint32_t max_struct_bytes = 256;

    /// Promote scalarized 4-byte fields to SSA registers via `run_ssa_spill_mem2reg`.
    bool run_mem2reg_after_split = true;
};

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
    /// `sym:ssbo:0:1[stride=16,off=4]`.
    std::string sym_slot_tag;
};

struct BufferSymbolizeOptions {
    /// Optional pointer analysis result to resolve buffer base pointers.
    const PointerAnalysisResult* ptr_analysis = nullptr;

    /// Forward store-to-load values for matching `(buf_key, index_ssa, stride, field_offset)`
    /// within the same basic block (invalidated by any potentially-aliasing store or barrier).
    bool forward_store_to_load = true;

    /// Annotate symbolized `PsbLoad` / `PsbStore` nodes with canonical `sym:` metadata
    /// in `aux_str` / `imm_u32_2` / `aux_index` / `aux_bool` for the Z3 verifier.
    bool annotate_ssa_nodes = true;
};

struct BufferSymbolizeResult {
    SsaPassStats stats;
    std::vector<SymbolizedBufferAccess> accesses;
};

/// Promotes per-thread TLS spill loads and stores (`tls_base + const_off`) into
/// SSA values and Phi nodes using Braun-style CFG SSA construction.
///
/// Any TLS slot that may alias a dynamic/unknown-offset TLS access is left
/// untouched for soundness. Promoted `PsbLoad` value extractions are replaced
/// directly with the reaching SSA value, and token chains are bypassed around
/// eliminated memory ops.
SsaPassStats run_ssa_spill_mem2reg(compiler::ValhallSsaModule& mod,
                                   const SpillMem2RegOptions& opts = {});

/// Scalar Replacement of Aggregates (SROA) for fixed-size structs in TLS/private
/// memory. Splits 64-bit (`Pack64` / `Split64`) and vector-packed struct accesses
/// into aligned 32-bit scalar field accesses `[off, off+4, ...]` when all accesses
/// to the struct range use constant offsets, then promotes each scalar field to
/// SSA registers.
SsaPassStats run_ssa_sroa(compiler::ValhallSsaModule& mod,
                          const SroaOptions& opts = {});

/// Recognizes fixed-stride linear affine SSBO/UBO reads and writes
/// (`base(buf) + idx * stride + field_off`), performs intra-block store-to-load
/// forwarding on identical symbolic slots, and annotates the SSA nodes with
/// canonical symbolic field tags so the Z3 verifier can model them as scalar
/// symbolic values rather than monolithic `Array(BV32, BV32)` maps.
BufferSymbolizeResult run_ssa_symbolize_buffers(compiler::ValhallSsaModule& mod,
                                                const BufferSymbolizeOptions& opts = {});

}  // namespace valhall::dataflow

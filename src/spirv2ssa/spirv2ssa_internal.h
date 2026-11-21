#pragma once

#include "spirv2ssa.h"
#include "spirv_module.h"
#include "valhall_spirv_parser_generated.h"

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace valhall::spirv2ssa {

/// Scalar or flattened-lane representation of a lowered SPIR-V SSA value.
struct LoweredValue {
    uint32_t spv_type_id = 0;
    std::vector<compiler::SsaId> lanes; // 1 for scalar, N for vector/composite

    bool empty() const { return lanes.empty(); }
    compiler::SsaId scalar() const { return lanes.empty() ? compiler::kInvalidSsa : lanes[0]; }
};

enum class PtrKind : uint8_t {
    Unknown = 0,
    LocalVar,   // Function or Private OpVariable (promoted via mem2reg)
    Buffer,     // Uniform (UBO), StorageBuffer (SSBO), PushConstant, PhysicalStorageBuffer, Workgroup
    InputVar,   // Stage input (BuiltIn, Vertex attribute, Fragment varying)
    OutputVar,  // Stage output (BuiltIn, Fragment color, Vertex varying)
    ImageVar,   // UniformConstant image/sampler/sampled-image
};

/// Symbolic pointer tracked across OpVariable and OpAccessChain.
struct PointerRef {
    PtrKind kind = PtrKind::Unknown;
    uint32_t root_var_id = 0;
    uint32_t pointee_type_id = 0;

    // Buffer addressing
    compiler::SsaId base_ptr = compiler::kInvalidSsa;
    compiler::SsaId byte_offset = compiler::kInvalidSsa;
    bool has_const_offset = true;
    uint64_t const_offset = 0;
    uint32_t matrix_stride = 0;
    bool is_row_major = false;
    std::string canonical_key;

    // Input/Output/Local access path (constant member/element indices)
    int32_t builtin = -1;
    int32_t location = -1;
    uint32_t component = 0;
    bool is_flat = false;
    std::vector<uint32_t> const_path;
    compiler::SsaId dynamic_index = compiler::kInvalidSsa;
};

/// Pending SPIR-V OpPhi whose incoming operands are patched after all blocks are lowered.
struct PendingSpvPhi {
    uint32_t blk_id = 0;
    uint32_t result_id = 0;
    uint32_t type_id = 0;
    std::vector<compiler::SsaId> lane_phi_ids;
    std::vector<std::pair<uint32_t, uint32_t>> raw_incomings; // (spv_val_id, pred_label_id)
};

/// Pending mem2reg Phi for a mutable variable slot.
struct PendingMemPhi {
    uint32_t blk_id = 0;
    uint64_t slot_key = 0;
    compiler::SsaId phi_id = compiler::kInvalidSsa;
};

/// Stateful lowering engine converting a parsed SpirvModule into ValhallSsaModule.
class SpirvLowerer {
public:
    SpirvLowerer(const SpirvModule& mod, const Spirv2SsaOptions& opts, Spirv2SsaResult& res);

    bool run();

    // Value & type helpers
    compiler::ValType to_val_type(uint32_t spv_type_id) const;
    uint32_t int_type_width(uint32_t spv_type_id) const;
    uint32_t flat_lane_count(uint32_t spv_type_id) const;
    uint32_t scalar_elem_type(uint32_t spv_type_id) const;
    LoweredValue get_value(uint32_t blk_id, uint32_t spv_id);
    LoweredValue make_zero_value(uint32_t blk_id, uint32_t spv_type_id);
    compiler::SsaId emit_const_int_for_type(uint32_t blk_id, uint32_t spv_type_id, uint64_t val);
    compiler::SsaId emit_icmp_eq_const(uint32_t blk_id, const LoweredValue& lhs, uint64_t lit);
    compiler::SsaId emit_to_i64(uint32_t blk_id, const LoweredValue& val, bool sign_extend);
    compiler::SsaId pack_to_vec4_f32(uint32_t blk_id, const LoweredValue& val, bool fill_alpha_one);
    compiler::SsaId pack_to_vec2_f32(uint32_t blk_id, const LoweredValue& val);
    uint32_t allocate_ssa_block();

    compiler::SsaId emit_op(
        uint32_t blk_id,
        tables::SsaOp op,
        compiler::ValType vt,
        std::vector<compiler::SsaId> args,
        bool barrier = false,
        bool side_eff = false);

    // Memory / resource lowering (spirv2ssa_mem.cpp)
    void init_global_resources(uint32_t entry_blk);
    void register_local_variable(uint32_t blk_id, uint32_t var_id, uint32_t ptr_type_id, uint32_t init_id);
    PointerRef lower_access_chain(uint32_t blk_id, const SpirvInst& inst);
    LoweredValue lower_load(uint32_t blk_id, const SpirvInst& inst);
    LoweredValue lower_load_ptr(uint32_t blk_id, uint32_t res_type, uint32_t res_id, uint32_t ptr_id);
    void lower_store(uint32_t blk_id, const SpirvInst& inst);
    void lower_store_val(uint32_t blk_id, uint32_t ptr_id, const LoweredValue& val);
    void lower_copy_memory(uint32_t blk_id, const SpirvInst& inst);
    LoweredValue lower_atomic(uint32_t blk_id, const SpirvInst& inst);
    void lower_atomic_store(uint32_t blk_id, const SpirvInst& inst);
    LoweredValue lower_image_sample(uint32_t blk_id, const SpirvInst& inst);
    void lower_image_write(uint32_t blk_id, const SpirvInst& inst);
    void flush_stage_outputs(uint32_t blk_id);

    // Mutable slot SSA promotion (mem2reg)
    uint64_t make_slot_key(uint32_t var_id, uint32_t flat_idx) const;
    void write_slot(uint32_t blk_id, uint64_t slot_key, compiler::SsaId val);
    compiler::SsaId read_slot(uint32_t blk_id, uint64_t slot_key, compiler::ValType vt);
    void seal_mem_phis();

    // Function inlining & switch desugaring (spirv2ssa.cpp)
    bool lower_function_call(uint32_t* io_blk_id, const SpirvInst& inst);
    void lower_switch(uint32_t blk_id, const SpirvInst& inst,
                      const std::unordered_map<uint32_t, uint32_t>& lbl_map);
    bool lower_function_body(
        const SpirvFunction& fn,
        uint32_t entry_blk,
        bool is_entry_point,
        uint32_t* out_exit_blk,
        LoweredValue* out_ret_val);

    // ALU / comparison / conversion / composite / GLSL.std.450 lowering (spirv2ssa_ops.cpp)
    bool lower_alu_or_misc_inst(uint32_t blk_id, const SpirvInst& inst);
    bool lower_glsl450_ext_inst(uint32_t blk_id, const SpirvInst& inst);

    void record_unsupported(const std::string& desc);

    const SpirvModule& mod_;
    const Spirv2SsaOptions& opts_;
    Spirv2SsaResult& res_;
    compiler::SsaBuilder sb_;

    // Lowered SSA values & pointers keyed by SPIR-V result ID
    std::unordered_map<uint32_t, LoweredValue> values_;
    std::unordered_map<uint32_t, PointerRef> pointers_;
    std::unordered_map<uint32_t, PointerRef> sampled_images_;

    // Global resource tracking
    uint32_t next_sideband_slot_ = 16;
    uint32_t inline_depth_ = 0;
    std::vector<uint32_t> output_var_ids_;
    std::set<uint32_t> written_output_vars_;
    std::unordered_set<uint64_t> written_output_slots_;

    // Per-block mem2reg state: block_id -> (slot_key -> SsaId)
    std::vector<std::unordered_map<uint64_t, compiler::SsaId>> blk_slot_defs_;
    std::vector<PendingMemPhi> pending_mem_phis_;

    std::unordered_map<uint32_t, uint32_t> local_var_types_;
    std::set<std::string> seen_unsupported_;
};

} // namespace valhall::spirv2ssa

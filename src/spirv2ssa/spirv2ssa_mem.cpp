#include "spirv2ssa_internal.h"

#include <algorithm>
#include <cstring>

namespace valhall::spirv2ssa {

namespace {

using compiler::SsaId;
using compiler::SsaNode;
using compiler::ValType;
using tables::SsaOp;

const char* builtin_canonical_name(uint32_t bi) {
    switch (static_cast<SpvBuiltIn>(bi)) {
        case SpvBuiltInPosition: return "gl_Position";
        case SpvBuiltInPointSize: return "gl_PointSize";
        case SpvBuiltInFragCoord: return "gl_FragCoord";
        case SpvBuiltInFrontFacing: return "gl_FrontFacing";
        case SpvBuiltInFragDepth: return "gl_FragDepth";
        case SpvBuiltInWorkgroupId: return "gl_WorkGroupID";
        case SpvBuiltInLocalInvocationId: return "gl_LocalInvocationID";
        case SpvBuiltInGlobalInvocationId: return "gl_GlobalInvocationID";
        case SpvBuiltInLocalInvocationIndex: return "gl_LocalInvocationIndex";
        case SpvBuiltInSubgroupSize: return "gl_SubgroupSize";
        case SpvBuiltInSubgroupLocalInvocationId: return "gl_SubgroupInvocationID";
        case SpvBuiltInVertexIndex: return "gl_VertexIndex";
        case SpvBuiltInInstanceIndex: return "gl_InstanceIndex";
        default: return spirv_parser::spv_builtin_name(bi);
    }
}

bool is_buffer_storage_class(uint32_t sc) {
    switch (static_cast<SpvStorageClass>(sc)) {
        case SpvStorageClassUniform:
        case SpvStorageClassStorageBuffer:
        case SpvStorageClassPushConstant:
        case SpvStorageClassPhysicalStorageBuffer:
        case SpvStorageClassWorkgroup:
            return true;
        default:
            return false;
    }
}

uint32_t compute_flat_offset(
    const SpirvModule& mod,
    const SpirvLowerer& lowerer,
    uint32_t root_type_id,
    const std::vector<uint32_t>& path) {
    uint32_t cur_type = root_type_id;
    uint32_t flat_off = 0;
    for (uint32_t idx : path) {
        const SpirvType* t = mod.get_type(cur_type);
        if (!t) break;
        if (t->kind == SpirvTypeKind::Vector) {
            flat_off += idx;
            cur_type = t->elem_type;
        } else if (t->kind == SpirvTypeKind::Matrix || t->kind == SpirvTypeKind::Array) {
            flat_off += idx * lowerer.flat_lane_count(t->elem_type);
            cur_type = t->elem_type;
        } else if (t->kind == SpirvTypeKind::Struct) {
            for (uint32_t m = 0; m < idx && m < t->member_types.size(); ++m) {
                flat_off += lowerer.flat_lane_count(t->member_types[m]);
            }
            cur_type = (idx < t->member_types.size()) ? t->member_types[idx] : 0;
        } else {
            break;
        }
    }
    return flat_off;
}

struct BufferLaneLayout {
    uint64_t byte_offset = 0;
    uint32_t scalar_type_id = 0;
};

void collect_buffer_lane_layout(
    const SpirvModule& mod,
    uint32_t type_id,
    uint64_t base_off,
    uint32_t active_mat_stride,
    bool active_row_major,
    std::vector<BufferLaneLayout>& out) {
    const SpirvType* t = mod.get_type(type_id);
    if (!t) {
        out.push_back({base_off, type_id});
        return;
    }
    switch (t->kind) {
        case SpirvTypeKind::Vector: {
            uint32_t esz = mod.type_byte_size(t->elem_type);
            for (uint32_t c = 0; c < std::max<uint32_t>(1u, t->count); ++c) {
                out.push_back({base_off + uint64_t(c) * esz, t->elem_type});
            }
            break;
        }
        case SpirvTypeKind::Matrix: {
            const SpirvType* col_t = mod.get_type(t->elem_type);
            uint32_t nrows = (col_t && col_t->kind == SpirvTypeKind::Vector) ? col_t->count : 4u;
            uint32_t scalar_tid = (col_t && col_t->kind == SpirvTypeKind::Vector) ? col_t->elem_type : t->elem_type;
            uint32_t esz = mod.type_byte_size(scalar_tid);
            uint32_t stride = (active_mat_stride != 0) ? active_mat_stride : (nrows * esz);
            for (uint32_t c = 0; c < t->count; ++c) {
                for (uint32_t r = 0; r < nrows; ++r) {
                    uint64_t off = active_row_major
                                       ? (base_off + uint64_t(r) * stride + uint64_t(c) * esz)
                                       : (base_off + uint64_t(c) * stride + uint64_t(r) * esz);
                    out.push_back({off, scalar_tid});
                }
            }
            break;
        }
        case SpirvTypeKind::Array: {
            uint32_t stride = (t->array_stride != 0) ? t->array_stride : mod.type_byte_size(t->elem_type);
            uint32_t cnt = std::min<uint32_t>(t->count, 64u);
            for (uint32_t e = 0; e < cnt; ++e) {
                collect_buffer_lane_layout(mod, t->elem_type, base_off + uint64_t(e) * stride,
                                           active_mat_stride, active_row_major, out);
            }
            break;
        }
        case SpirvTypeKind::Struct: {
            for (uint32_t m = 0; m < t->member_types.size(); ++m) {
                uint64_t moff = base_off + mod.struct_member_offset(*t, m);
                uint32_t m_stride = 0;
                bool m_row_maj = false;
                if (auto mit = mod.member_decorations.find({t->id, m}); mit != mod.member_decorations.end()) {
                    m_stride = mit->second.array_stride;
                    m_row_maj = mit->second.flat;
                }
                collect_buffer_lane_layout(mod, t->member_types[m], moff, m_stride, m_row_maj, out);
            }
            break;
        }
        default:
            out.push_back({base_off, type_id});
            break;
    }
}

} // namespace

uint64_t SpirvLowerer::make_slot_key(uint32_t var_id, uint32_t flat_idx) const {
    return (uint64_t(var_id) << 32) | uint64_t(flat_idx);
}

void SpirvLowerer::write_slot(uint32_t blk_id, uint64_t slot_key, SsaId val) {
    if (blk_id < blk_slot_defs_.size()) {
        blk_slot_defs_[blk_id][slot_key] = val;
    }
}

SsaId SpirvLowerer::read_slot(uint32_t blk_id, uint64_t slot_key, ValType vt) {
    auto it = blk_slot_defs_[blk_id].find(slot_key);
    if (it != blk_slot_defs_[blk_id].end()) {
        return it->second;
    }

    const auto& blk = res_.ssa.blocks[blk_id];
    if (blk_id == 0 || blk.preds.empty()) {
        SsaId z = (vt == ValType::F32) ? sb_.emit_const_f32_bits(blk_id, 0u)
                : (vt == ValType::Bool) ? sb_.emit_const_bool(blk_id, false)
                : (vt == ValType::U64 || vt == ValType::S64) ? sb_.emit_const_u64(blk_id, 0u)
                : sb_.emit_const_u32(blk_id, 0u);
        blk_slot_defs_[blk_id][slot_key] = z;
        return z;
    }

    if (blk.preds.size() == 1 &&
        blk_id < blk_static_pred_count_.size() &&
        blk_static_pred_count_[blk_id] <= 1u &&
        !blk.is_loop_header &&
        blk.preds[0] != blk_id) {
        SsaId v = read_slot(blk.preds[0], slot_key, vt);
        blk_slot_defs_[blk_id][slot_key] = v;
        return v;
    }

    SsaNode phi{};
    phi.op = SsaOp::Phi;
    phi.type = vt;
    phi.aux_bool = false;
    SsaId pid = sb_.emit(blk_id, std::move(phi));
    blk_slot_defs_[blk_id][slot_key] = pid;
    pending_mem_phis_.push_back({blk_id, slot_key, pid});
    return pid;
}

void SpirvLowerer::seal_mem_phis() {
    size_t idx = 0;
    while (idx < pending_mem_phis_.size()) {
        PendingMemPhi p = pending_mem_phis_[idx++];
        const auto& blk = res_.ssa.blocks[p.blk_id];
        ValType vt = res_.ssa.get(p.phi_id).type;
        std::vector<compiler::PhiIncoming> incomings;
        incomings.reserve(blk.preds.size());
        for (uint32_t pred : blk.preds) {
            SsaId val = read_slot(pred, p.slot_key, vt);
            incomings.push_back({pred, val});
        }
        res_.ssa.get_mut(p.phi_id).phi_incomings = std::move(incomings);
    }
}

void SpirvLowerer::init_global_resources(uint32_t entry_blk) {
    std::vector<uint32_t> var_ids;
    var_ids.reserve(mod_.global_vars.size());
    for (const auto& [vid, _] : mod_.global_vars) {
        var_ids.push_back(vid);
    }
    std::sort(var_ids.begin(), var_ids.end());

    for (uint32_t vid : var_ids) {
        const auto& gv = mod_.global_vars.at(vid);
        const auto sc = static_cast<SpvStorageClass>(gv.storage_class);
        PointerRef ptr{};
        ptr.root_var_id = vid;
        ptr.pointee_type_id = gv.pointee_type_id;
        ptr.builtin = gv.dec.builtin;
        ptr.location = gv.dec.location;
        ptr.component = gv.dec.component;
        ptr.is_flat = gv.dec.flat;

        if (is_buffer_storage_class(gv.storage_class)) {
            ptr.kind = PtrKind::Buffer;
            uint32_t set = (gv.dec.descriptor_set >= 0) ? static_cast<uint32_t>(gv.dec.descriptor_set) : 0u;
            uint32_t binding = (gv.dec.binding >= 0) ? static_cast<uint32_t>(gv.dec.binding) : 0u;
            uint32_t slot = next_sideband_slot_++;

            const SpirvType* pt = mod_.get_type(gv.pointee_type_id);
            bool is_ssbo = (sc == SpvStorageClassStorageBuffer) ||
                           (sc == SpvStorageClassUniform && pt && pt->is_buffer_block);
            ResourceKind rk = ResourceKind::UniformBuffer;
            std::string key;
            if (sc == SpvStorageClassPushConstant) {
                rk = ResourceKind::PushConstant;
                key = "push_const";
            } else if (sc == SpvStorageClassWorkgroup) {
                rk = ResourceKind::StorageBuffer;
                key = "workgroup:" + std::to_string(vid);
            } else if (is_ssbo) {
                rk = ResourceKind::StorageBuffer;
                key = "ssbo:" + std::to_string(set) + ":" + std::to_string(binding);
            } else {
                rk = ResourceKind::UniformBuffer;
                key = "ubo:" + std::to_string(set) + ":" + std::to_string(binding);
            }

            SsaNode base{};
            base.op = SsaOp::LoadSidebandU64;
            base.type = ValType::U64;
            base.imm_u32 = slot;
            base.aux_str = key;
            SsaId base_id = sb_.emit(entry_blk, std::move(base));

            ptr.base_ptr = base_id;
            ptr.byte_offset = sb_.emit_const_u64(entry_blk, 0u);
            ptr.has_const_offset = true;
            ptr.const_offset = 0;
            ptr.canonical_key = key;

            SpirvResourceBinding rb{};
            rb.kind = rk;
            rb.sideband_slot = slot;
            rb.descriptor_set = set;
            rb.binding = binding;
            rb.spv_var_id = vid;
            rb.name = gv.name;
            rb.canonical_key = key;
            res_.bindings.push_back(std::move(rb));
        } else if (sc == SpvStorageClassInput) {
            ptr.kind = PtrKind::InputVar;
            SpirvResourceBinding rb{};
            if (gv.dec.builtin >= 0) {
                rb.kind = ResourceKind::BuiltinInput;
                rb.builtin = static_cast<uint32_t>(gv.dec.builtin);
                rb.canonical_key = builtin_canonical_name(rb.builtin);
            } else {
                rb.kind = ResourceKind::InputAttrOrVarying;
                rb.location = (gv.dec.location >= 0) ? static_cast<uint32_t>(gv.dec.location) : 0u;
                rb.component = gv.dec.component;
                rb.canonical_key = "in_loc:" + std::to_string(rb.location);
            }
            rb.spv_var_id = vid;
            rb.name = gv.name;
            res_.bindings.push_back(std::move(rb));
        } else if (sc == SpvStorageClassOutput) {
            ptr.kind = PtrKind::OutputVar;
            local_var_types_[vid] = gv.pointee_type_id;
            output_var_ids_.push_back(vid);

            SpirvResourceBinding rb{};
            if (gv.dec.builtin >= 0) {
                rb.kind = ResourceKind::BuiltinOutput;
                rb.builtin = static_cast<uint32_t>(gv.dec.builtin);
                rb.canonical_key = builtin_canonical_name(rb.builtin);
            } else {
                rb.kind = ResourceKind::OutputColorOrVarying;
                rb.location = (gv.dec.location >= 0) ? static_cast<uint32_t>(gv.dec.location) : 0u;
                rb.component = gv.dec.component;
                rb.canonical_key = "out_loc:" + std::to_string(rb.location);
            }
            rb.spv_var_id = vid;
            rb.name = gv.name;
            res_.bindings.push_back(std::move(rb));
        } else if (sc == SpvStorageClassUniformConstant) {
            ptr.kind = PtrKind::ImageVar;
            uint32_t set = (gv.dec.descriptor_set >= 0) ? static_cast<uint32_t>(gv.dec.descriptor_set) : 0u;
            uint32_t binding = (gv.dec.binding >= 0) ? static_cast<uint32_t>(gv.dec.binding) : 0u;
            ptr.canonical_key = "tex:" + std::to_string(set) + ":" + std::to_string(binding);

            SpirvResourceBinding rb{};
            rb.kind = ResourceKind::SampledImage;
            rb.descriptor_set = set;
            rb.binding = binding;
            rb.spv_var_id = vid;
            rb.name = gv.name;
            rb.canonical_key = ptr.canonical_key;
            res_.bindings.push_back(std::move(rb));
        } else if (sc == SpvStorageClassPrivate || sc == SpvStorageClassFunction) {
            ptr.kind = PtrKind::LocalVar;
            local_var_types_[vid] = gv.pointee_type_id;
            if (gv.initializer_id != 0) {
                LoweredValue init_val = get_value(entry_blk, gv.initializer_id);
                for (uint32_t i = 0; i < init_val.lanes.size(); ++i) {
                    write_slot(entry_blk, make_slot_key(vid, i), init_val.lanes[i]);
                }
            }
        }
        pointers_[vid] = std::move(ptr);
    }
}

void SpirvLowerer::register_local_variable(uint32_t blk_id, uint32_t var_id, uint32_t ptr_type_id, uint32_t init_id) {
    uint32_t pointee = 0;
    if (const SpirvType* pt = mod_.get_type(ptr_type_id); pt && pt->kind == SpirvTypeKind::Pointer) {
        pointee = pt->pointee_type;
    }
    PointerRef ptr{};
    ptr.kind = PtrKind::LocalVar;
    ptr.root_var_id = var_id;
    ptr.pointee_type_id = pointee;
    pointers_[var_id] = ptr;
    local_var_types_[var_id] = pointee;

    if (init_id != 0) {
        LoweredValue init_val = get_value(blk_id, init_id);
        for (uint32_t i = 0; i < init_val.lanes.size(); ++i) {
            write_slot(blk_id, make_slot_key(var_id, i), init_val.lanes[i]);
        }
    }
}

PointerRef SpirvLowerer::lower_access_chain(uint32_t blk_id, const SpirvInst& inst) {
    PointerRef res{};
    uint32_t result_ptr_type = 0;
    uint32_t base_id = 0;
    bool has_elem_idx = false;
    uint32_t elem_idx_id = 0;
    spirv_parser::SpvWordSpan indexes{};

    const auto ac_op = static_cast<SpvOp>(inst.opcode);
    if (ac_op == SpvOpPtrAccessChain || ac_op == SpvOpInBoundsPtrAccessChain) {
        spirv_parser::SpvPtrAccessChainView v = inst;
        result_ptr_type = v.result_type;
        base_id = v.base;
        has_elem_idx = true;
        elem_idx_id = v.element;
        indexes = v.indexes;
    } else {
        spirv_parser::SpvAccessChainView v = inst;
        result_ptr_type = v.result_type;
        base_id = v.base;
        indexes = v.indexes;
    }

    auto it = pointers_.find(base_id);
    if (it != pointers_.end()) {
        res = it->second;
    } else {
        LoweredValue bv = get_value(blk_id, base_id);
        res.kind = PtrKind::Buffer;
        res.base_ptr = bv.scalar();
        res.byte_offset = sb_.emit_const_u64(blk_id, 0u);
        res.has_const_offset = true;
        res.const_offset = 0;
    }

    uint32_t cur_type = res.pointee_type_id;

    // SPIR-V §3.3.8: OpPtrAccessChain applies the Element index before walking composite Indexes.
    if (has_elem_idx && res.kind == PtrKind::Buffer) {
        uint32_t elem_stride = mod_.type_byte_size(cur_type);
        if (auto dit = mod_.decorations.find(result_ptr_type);
            dit != mod_.decorations.end() && dit->second.array_stride != 0) {
            elem_stride = dit->second.array_stride;
        }
        const SpirvConstant* ec = mod_.get_const(elem_idx_id);
        bool is_ec = (ec != nullptr &&
                      (ec->kind == SpirvConstKind::Scalar32 || ec->kind == SpirvConstKind::Scalar64 ||
                       ec->kind == SpirvConstKind::Null));
        if (is_ec) {
            int64_t sv = (ec->kind == SpirvConstKind::Scalar64)
                             ? static_cast<int64_t>(ec->scalar_bits)
                             : static_cast<int64_t>(static_cast<int32_t>(ec->scalar_bits));
            uint64_t delta = static_cast<uint64_t>(sv * static_cast<int64_t>(elem_stride));
            res.const_offset += delta;
            if (delta != 0) {
                res.byte_offset = emit_op(blk_id, SsaOp::IAdd64, ValType::U64,
                                          {res.byte_offset, sb_.emit_const_u64(blk_id, delta)});
            }
        } else {
            res.has_const_offset = false;
            SsaId i64 = emit_to_i64(blk_id, get_value(blk_id, elem_idx_id), /*sign_extend=*/true);
            SsaId delta = emit_op(blk_id, SsaOp::IMul64, ValType::U64, {i64, sb_.emit_const_u64(blk_id, elem_stride)});
            res.byte_offset = emit_op(blk_id, SsaOp::IAdd64, ValType::U64, {res.byte_offset, delta});
        }
    }

    auto to_signed_i64 = [&](uint32_t idx_id) -> SsaId {
        // SPIR-V §3.3.8: Each index in OpAccessChain is treated as signed.
        return emit_to_i64(blk_id, get_value(blk_id, idx_id), /*sign_extend=*/true);
    };

    for (size_t k = 0; k < indexes.count; ++k) {
        uint32_t idx_id = indexes[k];
        const SpirvType* t = mod_.get_type(cur_type);
        if (!t) break;

        const SpirvConstant* idx_const = mod_.get_const(idx_id);
        bool is_const_idx = (idx_const != nullptr &&
                             (idx_const->kind == SpirvConstKind::Scalar32 ||
                              idx_const->kind == SpirvConstKind::Scalar64 ||
                              idx_const->kind == SpirvConstKind::Null));
        uint32_t c_idx = is_const_idx ? static_cast<uint32_t>(idx_const->scalar_bits) : 0u;

        if (res.kind == PtrKind::Buffer) {
            if (t->kind == SpirvTypeKind::Struct) {
                uint32_t off = mod_.struct_member_offset(*t, c_idx);
                if (off != 0) {
                    res.const_offset += off;
                    SsaId add_c = sb_.emit_const_u64(blk_id, off);
                    res.byte_offset = emit_op(blk_id, SsaOp::IAdd64, ValType::U64, {res.byte_offset, add_c});
                }
                if (auto mit = mod_.member_decorations.find({t->id, c_idx}); mit != mod_.member_decorations.end()) {
                    if (mit->second.array_stride != 0) res.matrix_stride = mit->second.array_stride;
                    res.is_row_major = mit->second.flat;
                }
                cur_type = (c_idx < t->member_types.size()) ? t->member_types[c_idx] : 0;
            } else if (t->kind == SpirvTypeKind::Array || t->kind == SpirvTypeKind::RuntimeArray) {
                const SpirvType* et = mod_.get_type(t->elem_type);
                bool is_desc_array = (t->array_stride == 0 && et && et->kind == SpirvTypeKind::Struct &&
                                      (et->is_block || et->is_buffer_block));
                uint32_t stride = is_desc_array
                                      ? 0x01000000u
                                      : ((t->array_stride != 0) ? t->array_stride : mod_.type_byte_size(t->elem_type));
                if (is_const_idx) {
                    uint64_t delta = uint64_t(c_idx) * uint64_t(stride);
                    res.const_offset += delta;
                    if (delta != 0) {
                        SsaId add_c = sb_.emit_const_u64(blk_id, delta);
                        res.byte_offset = emit_op(blk_id, SsaOp::IAdd64, ValType::U64, {res.byte_offset, add_c});
                    }
                } else {
                    res.has_const_offset = false;
                    SsaId i64 = to_signed_i64(idx_id);
                    SsaId stride_c = sb_.emit_const_u64(blk_id, stride);
                    SsaId delta = emit_op(blk_id, SsaOp::IMul64, ValType::U64, {i64, stride_c});
                    res.byte_offset = emit_op(blk_id, SsaOp::IAdd64, ValType::U64, {res.byte_offset, delta});
                }
                cur_type = t->elem_type;
            } else if (t->kind == SpirvTypeKind::Matrix) {
                uint32_t col_stride = 0;
                if (res.is_row_major) {
                    const SpirvType* col_t = mod_.get_type(t->elem_type);
                    uint32_t scalar_tid = (col_t && col_t->kind == SpirvTypeKind::Vector) ? col_t->elem_type : t->elem_type;
                    col_stride = mod_.type_byte_size(scalar_tid);
                } else {
                    col_stride = (res.matrix_stride != 0) ? res.matrix_stride : mod_.type_byte_size(t->elem_type);
                }
                if (is_const_idx) {
                    uint64_t delta = uint64_t(c_idx) * uint64_t(col_stride);
                    res.const_offset += delta;
                    if (delta != 0) {
                        SsaId add_c = sb_.emit_const_u64(blk_id, delta);
                        res.byte_offset = emit_op(blk_id, SsaOp::IAdd64, ValType::U64, {res.byte_offset, add_c});
                    }
                } else {
                    res.has_const_offset = false;
                    SsaId i64 = to_signed_i64(idx_id);
                    SsaId delta = emit_op(blk_id, SsaOp::IMul64, ValType::U64, {i64, sb_.emit_const_u64(blk_id, col_stride)});
                    res.byte_offset = emit_op(blk_id, SsaOp::IAdd64, ValType::U64, {res.byte_offset, delta});
                }
                cur_type = t->elem_type;
            } else if (t->kind == SpirvTypeKind::Vector) {
                uint32_t elem_sz = (res.is_row_major && res.matrix_stride != 0)
                                       ? res.matrix_stride
                                       : mod_.type_byte_size(t->elem_type);
                if (is_const_idx) {
                    uint64_t delta = uint64_t(c_idx) * uint64_t(elem_sz);
                    res.const_offset += delta;
                    if (delta != 0) {
                        SsaId add_c = sb_.emit_const_u64(blk_id, delta);
                        res.byte_offset = emit_op(blk_id, SsaOp::IAdd64, ValType::U64, {res.byte_offset, add_c});
                    }
                } else {
                    res.has_const_offset = false;
                    SsaId i64 = to_signed_i64(idx_id);
                    SsaId delta = emit_op(blk_id, SsaOp::IMul64, ValType::U64, {i64, sb_.emit_const_u64(blk_id, elem_sz)});
                    res.byte_offset = emit_op(blk_id, SsaOp::IAdd64, ValType::U64, {res.byte_offset, delta});
                }
                cur_type = t->elem_type;
            }
        } else {
            if (t->kind == SpirvTypeKind::Struct) {
                if (c_idx < t->member_builtins.size() && t->member_builtins[c_idx] >= 0) {
                    res.builtin = t->member_builtins[c_idx];
                }
                res.const_path.push_back(c_idx);
                cur_type = (c_idx < t->member_types.size()) ? t->member_types[c_idx] : 0;
            } else if (t->kind == SpirvTypeKind::Vector || t->kind == SpirvTypeKind::Matrix || t->kind == SpirvTypeKind::Array) {
                if (is_const_idx) {
                    res.const_path.push_back(c_idx);
                } else {
                    LoweredValue iv = get_value(blk_id, idx_id);
                    res.dynamic_index = iv.scalar();
                }
                cur_type = t->elem_type;
            }
        }
    }

    if (const SpirvType* rpt = mod_.get_type(result_ptr_type); rpt && rpt->kind == SpirvTypeKind::Pointer) {
        res.pointee_type_id = rpt->pointee_type;
    } else if (cur_type != 0) {
        res.pointee_type_id = cur_type;
    }
    return res;
}

LoweredValue SpirvLowerer::lower_load(uint32_t blk_id, const SpirvInst& inst) {
    spirv_parser::SpvLoadView ld = inst;
    return lower_load_ptr(blk_id, ld.result_type, ld.result_id, ld.pointer);
}

LoweredValue SpirvLowerer::lower_load_ptr(uint32_t blk_id, uint32_t res_type, uint32_t res_id, uint32_t ptr_id) {
    LoweredValue out{};
    out.spv_type_id = res_type;

    auto pit = pointers_.find(ptr_id);
    if (pit == pointers_.end()) {
        return make_zero_value(blk_id, res_type);
    }
    const PointerRef& ptr = pit->second;

    if (ptr.kind == PtrKind::ImageVar) {
        if (res_id != 0) sampled_images_[res_id] = ptr;
        out.lanes.push_back(sb_.emit_const_u32(blk_id, ptr.root_var_id));
        return out;
    }

    if (ptr.kind == PtrKind::Buffer) {
        SsaId base_addr = emit_op(blk_id, SsaOp::IAdd64, ValType::U64, {ptr.base_ptr, ptr.byte_offset});
        std::vector<BufferLaneLayout> layout;
        collect_buffer_lane_layout(mod_, res_type, 0u, ptr.matrix_stride, ptr.is_row_major, layout);

        for (const auto& lane : layout) {
            const SpirvType* rt = mod_.get_type(lane.scalar_type_id);
            ValType elem_vt = to_val_type(lane.scalar_type_id);
            uint32_t elem_bytes = mod_.type_byte_size(lane.scalar_type_id);
            SsaId lane_addr = (lane.byte_offset == 0)
                                  ? base_addr
                                  : emit_op(blk_id, SsaOp::IAdd64, ValType::U64,
                                            {base_addr, sb_.emit_const_u64(blk_id, lane.byte_offset)});
            SsaId tok = sb_.current_token(blk_id);
            SsaNode ld{};
            ld.op = SsaOp::PsbLoad;
            ld.type = ValType::TupleValToken;
            ld.imm_u32 = std::min<uint32_t>(elem_bytes, 4u);
            ld.aux_str = ptr.canonical_key;
            ld.args = {lane_addr, tok};
            ld.is_barrier_or_cf = true;
            ld.is_side_effect = true;
            SsaId tup = sb_.emit(blk_id, std::move(ld));

            SsaNode ev{};
            ev.op = SsaOp::TupleExtract;
            ev.type = ValType::U32;
            ev.imm_u32 = 0;
            ev.args = {tup};
            SsaId raw_u32 = sb_.emit(blk_id, std::move(ev));

            SsaNode et{};
            et.op = SsaOp::TupleExtract;
            et.type = ValType::Token;
            et.imm_u32 = 1;
            et.args = {tup};
            sb_.set_token(blk_id, sb_.emit(blk_id, std::move(et)));

            if (elem_bytes == 8u) {
                SsaId hi_addr = emit_op(blk_id, SsaOp::IAdd64, ValType::U64,
                                        {lane_addr, sb_.emit_const_u64(blk_id, 4u)});
                SsaNode ld_hi{};
                ld_hi.op = SsaOp::PsbLoad;
                ld_hi.type = ValType::TupleValToken;
                ld_hi.imm_u32 = 4u;
                ld_hi.aux_str = ptr.canonical_key;
                ld_hi.args = {hi_addr, sb_.current_token(blk_id)};
                ld_hi.is_barrier_or_cf = true;
                ld_hi.is_side_effect = true;
                SsaId tup_hi = sb_.emit(blk_id, std::move(ld_hi));

                SsaNode ev_hi{}; ev_hi.op = SsaOp::TupleExtract; ev_hi.type = ValType::U32; ev_hi.imm_u32 = 0; ev_hi.args = {tup_hi};
                SsaId hi_u32 = sb_.emit(blk_id, std::move(ev_hi));
                SsaNode et_hi{}; et_hi.op = SsaOp::TupleExtract; et_hi.type = ValType::Token; et_hi.imm_u32 = 1; et_hi.args = {tup_hi};
                sb_.set_token(blk_id, sb_.emit(blk_id, std::move(et_hi)));

                out.lanes.push_back(emit_op(blk_id, SsaOp::Pack64, ValType::U64, {raw_u32, hi_u32}));
            } else if (rt && rt->kind == SpirvTypeKind::Float && rt->width == 16) {
                SsaId v2 = emit_op(blk_id, SsaOp::UnpackHalf2x16, ValType::V2F32, {raw_u32});
                SsaNode ex{}; ex.op = SsaOp::VecExtract; ex.type = ValType::F32; ex.imm_u32 = 0; ex.args = {v2};
                out.lanes.push_back(sb_.emit(blk_id, std::move(ex)));
            } else {
                out.lanes.push_back(sb_.emit_bitcast(blk_id, elem_vt, raw_u32));
            }
        }
        return out;
    }

    if (ptr.kind == PtrKind::InputVar) {
        if (ptr.builtin >= 0) {
            uint32_t bi = static_cast<uint32_t>(ptr.builtin);
            const SpirvVariable* gv = mod_.get_global_var(ptr.root_var_id);
            uint32_t root_tid = gv ? gv->pointee_type_id : res_type;
            ValType bi_vt = to_val_type(root_tid);
            switch (static_cast<SpvBuiltIn>(bi)) {
                case SpvBuiltInFragCoord: bi_vt = ValType::V4F32; break;
                case SpvBuiltInWorkgroupId:
                case SpvBuiltInLocalInvocationId:
                case SpvBuiltInGlobalInvocationId: bi_vt = ValType::V3U32; break;
                case SpvBuiltInVertexIndex:
                case SpvBuiltInInstanceIndex:
                case SpvBuiltInLocalInvocationIndex:
                case SpvBuiltInSubgroupLocalInvocationId:
                case SpvBuiltInSubgroupSize: bi_vt = ValType::U32; break;
                case SpvBuiltInFrontFacing: bi_vt = ValType::Bool; break;
                default: break;
            }

            SsaNode ld{};
            ld.op = SsaOp::LoadBuiltinInput;
            ld.type = bi_vt;
            ld.imm_u32 = bi;
            ld.aux_bool = ptr.is_flat;
            ld.aux_str = builtin_canonical_name(bi);
            SsaId full = sb_.emit(blk_id, std::move(ld));

            uint32_t nlanes = flat_lane_count(res_type);
            uint32_t comp_off = ptr.const_path.empty() ? 0u : ptr.const_path.back();
            ValType lane_vt = to_val_type(scalar_elem_type(res_type));
            if (bi_vt == ValType::V4F32 || bi_vt == ValType::V3U32) {
                for (uint32_t c = 0; c < nlanes; ++c) {
                    SsaNode ex{};
                    ex.op = SsaOp::VecExtract;
                    ex.type = (bi_vt == ValType::V4F32) ? ValType::F32 : ValType::U32;
                    ex.imm_u32 = comp_off + c;
                    ex.args = {full};
                    SsaId ex_id = sb_.emit(blk_id, std::move(ex));
                    out.lanes.push_back(sb_.emit_bitcast(blk_id, lane_vt, ex_id));
                }
            } else {
                out.lanes.push_back(sb_.emit_bitcast(blk_id, lane_vt, full));
            }
            return out;
        }

        uint32_t loc = (ptr.location >= 0) ? static_cast<uint32_t>(ptr.location) : 0u;
        SsaNode ld{};
        ld.op = (res_.stage == ShaderStage::Vertex) ? SsaOp::LoadVertAttr : SsaOp::LoadFragVarying;
        ld.type = ValType::V4F32;
        ld.imm_u32 = loc;
        ld.aux_bool = ptr.is_flat;
        if (ptr.dynamic_index != valhall::compiler::kInvalidSsa) {
            SsaId dyn_idx = sb_.emit_bitcast(blk_id, ValType::U32, ptr.dynamic_index);
            SsaId dyn_loc = (loc != 0u)
                ? emit_op(blk_id, SsaOp::IAdd, ValType::U32, {dyn_idx, sb_.emit_const_u32(blk_id, loc)})
                : dyn_idx;
            ld.args = { dyn_loc };
        }
        ld.is_barrier_or_cf = true;
        SsaId vec4 = sb_.emit(blk_id, std::move(ld));

        uint32_t nlanes = flat_lane_count(res_type);
        uint32_t base_comp = ptr.component + (ptr.const_path.empty() ? 0u : ptr.const_path.back());
        ValType lane_vt = to_val_type(scalar_elem_type(res_type));
        for (uint32_t c = 0; c < nlanes; ++c) {
            SsaNode ex{};
            ex.op = SsaOp::VecExtract;
            ex.type = ValType::F32;
            ex.imm_u32 = std::min<uint32_t>(3u, base_comp + c);
            ex.args = {vec4};
            SsaId f = sb_.emit(blk_id, std::move(ex));
            out.lanes.push_back(sb_.emit_bitcast(blk_id, lane_vt, f));
        }
        return out;
    }

    uint32_t root_tid = local_var_types_.count(ptr.root_var_id) ? local_var_types_[ptr.root_var_id] : res_type;
    uint32_t base_flat = compute_flat_offset(mod_, *this, root_tid, ptr.const_path);
    uint32_t nlanes = flat_lane_count(res_type);
    ValType lane_vt = to_val_type(scalar_elem_type(res_type));

    if (ptr.dynamic_index != compiler::kInvalidSsa) {
        uint32_t dyn_tid = root_tid;
        for (uint32_t c_idx : ptr.const_path) {
            const SpirvType* ct = mod_.get_type(dyn_tid);
            if (!ct) break;
            if (ct->kind == SpirvTypeKind::Struct && c_idx < ct->member_types.size()) {
                dyn_tid = ct->member_types[c_idx];
            } else if (ct->kind == SpirvTypeKind::Array || ct->kind == SpirvTypeKind::Matrix ||
                       ct->kind == SpirvTypeKind::Vector) {
                dyn_tid = ct->elem_type;
            } else {
                break;
            }
        }
        const SpirvType* rt = mod_.get_type(dyn_tid);
        uint32_t total_elems = (rt && (rt->kind == SpirvTypeKind::Array || rt->kind == SpirvTypeKind::Vector ||
                                       rt->kind == SpirvTypeKind::Matrix))
                                   ? std::min<uint32_t>(rt->count, 256u)
                                   : 4u;
        for (uint32_t lane = 0; lane < nlanes; ++lane) {
            SsaId acc = read_slot(blk_id, make_slot_key(ptr.root_var_id, base_flat + lane), lane_vt);
            for (uint32_t e = 1; e < total_elems; ++e) {
                SsaId cmp = emit_op(blk_id, SsaOp::ICmpEq, ValType::Bool,
                                    {ptr.dynamic_index, sb_.emit_const_u32(blk_id, e)});
                SsaId cand = read_slot(blk_id, make_slot_key(ptr.root_var_id, base_flat + e * nlanes + lane), lane_vt);
                acc = emit_op(blk_id, SsaOp::Select, lane_vt, {cmp, cand, acc});
            }
            out.lanes.push_back(acc);
        }
        return out;
    }

    for (uint32_t i = 0; i < nlanes; ++i) {
        out.lanes.push_back(read_slot(blk_id, make_slot_key(ptr.root_var_id, base_flat + i), lane_vt));
    }
    return out;
}

void SpirvLowerer::lower_store(uint32_t blk_id, const SpirvInst& inst) {
    spirv_parser::SpvStoreView st = inst;
    lower_store_val(blk_id, st.pointer, get_value(blk_id, st.object));
}

void SpirvLowerer::lower_store_val(uint32_t blk_id, uint32_t ptr_id, const LoweredValue& val) {
    auto pit = pointers_.find(ptr_id);
    if (pit == pointers_.end()) return;
    const PointerRef& ptr = pit->second;
    if (val.empty()) return;

    if (ptr.kind == PtrKind::Buffer) {
        SsaId base_addr = emit_op(blk_id, SsaOp::IAdd64, ValType::U64, {ptr.base_ptr, ptr.byte_offset});
        std::vector<BufferLaneLayout> layout;
        collect_buffer_lane_layout(mod_, val.spv_type_id, 0u, ptr.matrix_stride, ptr.is_row_major, layout);

        for (size_t i = 0; i < val.lanes.size(); ++i) {
            uint64_t boff = (i < layout.size()) ? layout[i].byte_offset : (uint64_t(i) * 4u);
            uint32_t elem_tid = (i < layout.size()) ? layout[i].scalar_type_id : scalar_elem_type(val.spv_type_id);
            uint32_t elem_bytes = mod_.type_byte_size(elem_tid);
            SsaId lane_addr = (boff == 0)
                                  ? base_addr
                                  : emit_op(blk_id, SsaOp::IAdd64, ValType::U64,
                                            {base_addr, sb_.emit_const_u64(blk_id, boff)});
            SsaId lane_val = val.lanes[i];
            if (elem_bytes == 8u) {
                SsaNode sp{}; sp.op = SsaOp::Split64; sp.type = ValType::TupleU32x2; sp.args = {lane_val};
                SsaId tup = sb_.emit(blk_id, std::move(sp));
                SsaNode e0{}; e0.op = SsaOp::TupleExtract; e0.type = ValType::U32; e0.imm_u32 = 0; e0.args = {tup};
                SsaNode e1{}; e1.op = SsaOp::TupleExtract; e1.type = ValType::U32; e1.imm_u32 = 1; e1.args = {tup};
                SsaId lo = sb_.emit(blk_id, std::move(e0));
                SsaId hi = sb_.emit(blk_id, std::move(e1));

                SsaNode st0{};
                st0.op = SsaOp::PsbStore; st0.type = ValType::Token; st0.imm_u32 = 4u;
                st0.aux_str = ptr.canonical_key; st0.args = {lane_addr, lo, sb_.current_token(blk_id)};
                st0.is_barrier_or_cf = true; st0.is_side_effect = true;
                sb_.set_token(blk_id, sb_.emit(blk_id, std::move(st0)));

                SsaId hi_addr = emit_op(blk_id, SsaOp::IAdd64, ValType::U64,
                                        {lane_addr, sb_.emit_const_u64(blk_id, 4u)});
                SsaNode st1{};
                st1.op = SsaOp::PsbStore; st1.type = ValType::Token; st1.imm_u32 = 4u;
                st1.aux_str = ptr.canonical_key; st1.args = {hi_addr, hi, sb_.current_token(blk_id)};
                st1.is_barrier_or_cf = true; st1.is_side_effect = true;
                sb_.set_token(blk_id, sb_.emit(blk_id, std::move(st1)));
            } else {
                SsaId u32_val = sb_.emit_bitcast(blk_id, ValType::U32, lane_val);
                SsaNode st{};
                st.op = SsaOp::PsbStore; st.type = ValType::Token; st.imm_u32 = std::min<uint32_t>(elem_bytes, 4u);
                st.aux_str = ptr.canonical_key; st.args = {lane_addr, u32_val, sb_.current_token(blk_id)};
                st.is_barrier_or_cf = true; st.is_side_effect = true;
                sb_.set_token(blk_id, sb_.emit(blk_id, std::move(st)));
            }
        }
        return;
    }

    if (ptr.kind == PtrKind::OutputVar) {
        written_output_vars_.insert(ptr.root_var_id);
    }
    uint32_t root_tid = local_var_types_.count(ptr.root_var_id) ? local_var_types_[ptr.root_var_id] : val.spv_type_id;
    uint32_t base_flat = compute_flat_offset(mod_, *this, root_tid, ptr.const_path);

    if (ptr.dynamic_index != compiler::kInvalidSsa) {
        uint32_t dyn_tid = root_tid;
        for (uint32_t c_idx : ptr.const_path) {
            const SpirvType* ct = mod_.get_type(dyn_tid);
            if (!ct) break;
            if (ct->kind == SpirvTypeKind::Struct && c_idx < ct->member_types.size()) {
                dyn_tid = ct->member_types[c_idx];
            } else if (ct->kind == SpirvTypeKind::Array || ct->kind == SpirvTypeKind::Matrix ||
                       ct->kind == SpirvTypeKind::Vector) {
                dyn_tid = ct->elem_type;
            } else {
                break;
            }
        }
        const SpirvType* rt = mod_.get_type(dyn_tid);
        uint32_t total_elems = (rt && (rt->kind == SpirvTypeKind::Array || rt->kind == SpirvTypeKind::Vector ||
                                       rt->kind == SpirvTypeKind::Matrix))
                                   ? std::min<uint32_t>(rt->count, 256u)
                                   : 4u;
        uint32_t nlanes = static_cast<uint32_t>(val.lanes.size());
        for (uint32_t e = 0; e < total_elems; ++e) {
            SsaId cmp = emit_op(blk_id, SsaOp::ICmpEq, ValType::Bool,
                                {ptr.dynamic_index, sb_.emit_const_u32(blk_id, e)});
            for (uint32_t lane = 0; lane < nlanes; ++lane) {
                uint64_t sk = make_slot_key(ptr.root_var_id, base_flat + e * nlanes + lane);
                if (ptr.kind == PtrKind::OutputVar) written_output_slots_.insert(sk);
                ValType vt = res_.ssa.get(val.lanes[lane]).type;
                SsaId prev = read_slot(blk_id, sk, vt);
                SsaId sel = emit_op(blk_id, SsaOp::Select, vt, {cmp, val.lanes[lane], prev});
                write_slot(blk_id, sk, sel);
            }
        }
        return;
    }

    for (uint32_t i = 0; i < val.lanes.size(); ++i) {
        uint64_t sk = make_slot_key(ptr.root_var_id, base_flat + i);
        if (ptr.kind == PtrKind::OutputVar) written_output_slots_.insert(sk);
        write_slot(blk_id, sk, val.lanes[i]);
    }
}

void SpirvLowerer::flush_stage_outputs(uint32_t blk_id) {
    for (uint32_t vid : output_var_ids_) {
        if (!written_output_vars_.count(vid)) continue;
        const SpirvVariable* gv = mod_.get_global_var(vid);
        if (!gv) continue;
        const SpirvType* pt = mod_.get_type(gv->pointee_type_id);
        if (!pt) continue;

        if (pt->kind == SpirvTypeKind::Struct) {
            uint32_t flat_off = 0;
            for (size_t m = 0; m < pt->member_types.size(); ++m) {
                uint32_t mt = pt->member_types[m];
                uint32_t ml = flat_lane_count(mt);
                if (m < pt->member_builtins.size() && pt->member_builtins[m] >= 0) {
                    bool member_written = false;
                    for (uint32_t k = 0; k < ml; ++k) {
                        if (written_output_slots_.count(make_slot_key(vid, flat_off + k))) {
                            member_written = true;
                            break;
                        }
                    }
                    if (!member_written) {
                        flat_off += ml;
                        continue;
                    }
                    uint32_t bi = static_cast<uint32_t>(pt->member_builtins[m]);
                    LoweredValue lv{};
                    lv.spv_type_id = mt;
                    ValType lane_vt = to_val_type(scalar_elem_type(mt));
                    for (uint32_t k = 0; k < ml; ++k) {
                        lv.lanes.push_back(read_slot(blk_id, make_slot_key(vid, flat_off + k), lane_vt));
                    }
                    SsaNode st{};
                    st.op = SsaOp::StoreBuiltinOutput;
                    st.type = ValType::Token;
                    st.imm_u32 = bi;
                    st.aux_str = builtin_canonical_name(bi);
                    SsaId payload = (ml > 1) ? pack_to_vec4_f32(blk_id, lv, true) : lv.scalar();
                    st.args = {payload, sb_.current_token(blk_id)};
                    st.is_barrier_or_cf = true;
                    st.is_side_effect = true;
                    sb_.set_token(blk_id, sb_.emit(blk_id, std::move(st)));
                }
                flat_off += ml;
            }
            continue;
        }

        uint32_t nlanes = flat_lane_count(gv->pointee_type_id);
        ValType lane_vt = to_val_type(scalar_elem_type(gv->pointee_type_id));
        LoweredValue lv{};
        lv.spv_type_id = gv->pointee_type_id;
        for (uint32_t k = 0; k < nlanes; ++k) {
            lv.lanes.push_back(read_slot(blk_id, make_slot_key(vid, k), lane_vt));
        }

        if (gv->dec.builtin >= 0) {
            uint32_t bi = static_cast<uint32_t>(gv->dec.builtin);
            SsaNode st{};
            st.op = SsaOp::StoreBuiltinOutput;
            st.type = ValType::Token;
            st.imm_u32 = bi;
            st.aux_str = builtin_canonical_name(bi);
            SsaId payload = (nlanes > 1) ? pack_to_vec4_f32(blk_id, lv, true) : lv.scalar();
            st.args = {payload, sb_.current_token(blk_id)};
            st.is_barrier_or_cf = true;
            st.is_side_effect = true;
            sb_.set_token(blk_id, sb_.emit(blk_id, std::move(st)));
        } else if (res_.stage == ShaderStage::Fragment) {
            uint32_t rt = (gv->dec.location >= 0) ? static_cast<uint32_t>(gv->dec.location) : 0u;
            SsaId col4 = pack_to_vec4_f32(blk_id, lv, true);
            SsaNode st{};
            st.op = SsaOp::StoreFragColor;
            st.type = ValType::Token;
            st.imm_u32 = rt;
            st.args = {col4, sb_.current_token(blk_id)};
            st.is_barrier_or_cf = true;
            st.is_side_effect = true;
            sb_.set_token(blk_id, sb_.emit(blk_id, std::move(st)));
        } else if (res_.stage == ShaderStage::Vertex) {
            uint32_t loc = (gv->dec.location >= 0) ? static_cast<uint32_t>(gv->dec.location) : 0u;
            if (pt->kind == SpirvTypeKind::Array && pt->count > 0u) {
                uint32_t elem_lanes = flat_lane_count(pt->elem_type);
                if (elem_lanes == 0u) elem_lanes = 1u;
                uint32_t elem_slots = (elem_lanes + 3u) / 4u;
                for (uint32_t e = 0; e < pt->count; ++e) {
                    LoweredValue elv{};
                    elv.spv_type_id = pt->elem_type;
                    for (uint32_t k = 0; k < elem_lanes && (e * elem_lanes + k) < lv.lanes.size(); ++k) {
                        elv.lanes.push_back(lv.lanes[e * elem_lanes + k]);
                    }
                    SsaId vec4 = pack_to_vec4_f32(blk_id, elv, false);
                    SsaNode st{};
                    st.op = SsaOp::StoreVertVarying;
                    st.type = ValType::Token;
                    st.imm_u32 = loc + e * elem_slots;
                    st.args = {vec4, sb_.current_token(blk_id)};
                    st.is_barrier_or_cf = true;
                    st.is_side_effect = true;
                    sb_.set_token(blk_id, sb_.emit(blk_id, std::move(st)));
                }
            } else {
                SsaId vec4 = pack_to_vec4_f32(blk_id, lv, false);
                SsaNode st{};
                st.op = SsaOp::StoreVertVarying;
                st.type = ValType::Token;
                st.imm_u32 = loc;
                st.args = {vec4, sb_.current_token(blk_id)};
                st.is_barrier_or_cf = true;
                st.is_side_effect = true;
                sb_.set_token(blk_id, sb_.emit(blk_id, std::move(st)));
            }
        }
    }
}

void SpirvLowerer::lower_copy_memory(uint32_t blk_id, const SpirvInst& inst) {
    spirv_parser::SpvCopyMemoryView cm = inst;
    auto sit = pointers_.find(cm.source);
    if (sit == pointers_.end()) return;
    LoweredValue loaded = lower_load_ptr(blk_id, sit->second.pointee_type_id, 0u, cm.source);
    lower_store_val(blk_id, cm.target, loaded);
}

void SpirvLowerer::lower_atomic_store(uint32_t blk_id, const SpirvInst& inst) {
    const auto op = static_cast<SpvOp>(inst.opcode);
    if (op == SpvOpAtomicFlagClear) {
        spirv_parser::SpvAtomicFlagClearView fc = inst;
        LoweredValue zero{};
        zero.lanes.push_back(sb_.emit_const_u32(blk_id, 0u));
        lower_store_val(blk_id, fc.pointer, zero);
        return;
    }
    spirv_parser::SpvAtomicStoreView as = inst;
    lower_store_val(blk_id, as.pointer, get_value(blk_id, as.value));
}

LoweredValue SpirvLowerer::lower_atomic(uint32_t blk_id, const SpirvInst& inst) {
    LoweredValue out{};
    const auto op = static_cast<SpvOp>(inst.opcode);

    if (op == SpvOpAtomicLoad) {
        spirv_parser::SpvAtomicLoadView al = inst;
        return lower_load_ptr(blk_id, al.result_type, al.result_id, al.pointer);
    }

    spirv_parser::SpvAtomicHeaderView hdr = inst;
    uint32_t res_type = hdr.result_type;
    uint32_t ptr_id = hdr.pointer;
    out.spv_type_id = res_type;

    auto pit = pointers_.find(ptr_id);
    if (pit == pointers_.end() ||
        (pit->second.kind != PtrKind::Buffer && pit->second.kind != PtrKind::ImageTexel)) {
        return make_zero_value(blk_id, res_type);
    }
    const PointerRef& ptr = pit->second;
    SsaId addr = (ptr.kind == PtrKind::ImageTexel)
                     ? ptr.base_ptr
                     : emit_op(blk_id, SsaOp::IAdd64, ValType::U64, {ptr.base_ptr, ptr.byte_offset});
    SsaId tok = sb_.current_token(blk_id);

    SsaNode at{};
    at.type = ValType::TupleValToken;
    at.is_barrier_or_cf = true;
    at.is_side_effect = true;

    if (op == SpvOpAtomicExchange) {
        spirv_parser::SpvAtomicExchangeView v = inst;
        at.op = SsaOp::AtomicExchange;
        at.args = {addr, get_value(blk_id, v.value).scalar(), tok};
    } else if (op == SpvOpAtomicCompareExchange || op == SpvOpAtomicCompareExchangeWeak) {
        spirv_parser::SpvAtomicCompareExchangeView v = inst;
        at.op = SsaOp::AtomicCmpXchg;
        at.args = {addr, get_value(blk_id, v.value).scalar(), get_value(blk_id, v.comparator).scalar(), tok};
    } else if (op == SpvOpAtomicIIncrement || op == SpvOpAtomicIDecrement) {
        SsaId one = emit_const_int_for_type(blk_id, res_type, 1u);
        at.op = SsaOp::AtomicRmw;
        at.aux_str = (op == SpvOpAtomicIIncrement) ? "aadd" : "asub";
        at.args = {addr, one, tok};
    } else if (op == SpvOpAtomicFlagTestAndSet) {
        SsaId one = sb_.emit_const_u32(blk_id, 1u);
        at.op = SsaOp::AtomicExchange;
        at.args = {addr, one, tok};
    } else {
        spirv_parser::SpvAtomicRmwView v = inst;
        SsaId val = get_value(blk_id, v.value).scalar();
        at.op = SsaOp::AtomicRmw;
        switch (op) {
            case SpvOpAtomicIAdd: at.aux_str = "aadd"; break;
            case SpvOpAtomicISub: at.aux_str = "asub"; break;
            case SpvOpAtomicSMin: at.aux_str = "asmin"; break;
            case SpvOpAtomicUMin: at.aux_str = "aumin"; break;
            case SpvOpAtomicSMax: at.aux_str = "asmax"; break;
            case SpvOpAtomicUMax: at.aux_str = "aumax"; break;
            case SpvOpAtomicAnd: at.aux_str = "aand"; break;
            case SpvOpAtomicOr: at.aux_str = "aor"; break;
            case SpvOpAtomicXor: at.aux_str = "axor"; break;
            case SpvOpAtomicFAddEXT: at.aux_str = "afadd"; break;
            case SpvOpAtomicFMinEXT: at.aux_str = "afmin"; break;
            case SpvOpAtomicFMaxEXT: at.aux_str = "afmax"; break;
            default: at.aux_str = "aadd"; break;
        }
        at.args = {addr, val, tok};
    }

    SsaId tup = sb_.emit(blk_id, std::move(at));
    SsaNode ev{}; ev.op = SsaOp::TupleExtract; ev.type = to_val_type(res_type); ev.imm_u32 = 0; ev.args = {tup};
    SsaNode et{}; et.op = SsaOp::TupleExtract; et.type = ValType::Token; et.imm_u32 = 1; et.args = {tup};
    out.lanes.push_back(sb_.emit(blk_id, std::move(ev)));
    sb_.set_token(blk_id, sb_.emit(blk_id, std::move(et)));
    return out;
}

LoweredValue SpirvLowerer::lower_image_sample(uint32_t blk_id, const SpirvInst& inst) {
    LoweredValue out{};
    spirv_parser::SpvImageOpHeaderView img = inst;
    uint32_t res_type = img.result_type;
    uint32_t img_id = img.image;
    uint32_t coord_id = img.coordinate;
    out.spv_type_id = res_type;

    LoweredValue coord = get_value(blk_id, coord_id);
    SsaId uv = pack_to_vec2_f32(blk_id, coord);

    std::string tex_key = "tex:0:0";
    bool is_cube_dim = false;
    if (auto it = sampled_images_.find(img_id); it != sampled_images_.end()) {
        tex_key = it->second.canonical_key;
        const SpirvType* sit = mod_.get_type(it->second.pointee_type_id);
        if (sit && sit->kind == SpirvTypeKind::SampledImage) {
            sit = mod_.get_type(sit->elem_type);
        }
        if (sit && sit->kind == SpirvTypeKind::Image && sit->image_dim == 3u) {
            is_cube_dim = true;
        }
    }

    const auto op = static_cast<SpvOp>(inst.opcode);
    SsaNode ts{};
    if (op == SpvOpImageFetch || op == SpvOpImageRead) {
        ts.op = SsaOp::TexFetch;
    } else if (op == SpvOpImageGather || op == SpvOpImageDrefGather) {
        ts.op = SsaOp::TexGather;
    } else if (op == SpvOpImageSampleDrefImplicitLod || op == SpvOpImageSampleDrefExplicitLod ||
               op == SpvOpImageSampleProjDrefImplicitLod || op == SpvOpImageSampleProjDrefExplicitLod) {
        ts.op = SsaOp::TexSampleDref;
    } else if (op == SpvOpImageSampleExplicitLod || op == SpvOpImageSampleProjExplicitLod) {
        ts.op = SsaOp::TexSampleLod;
    } else {
        ts.op = SsaOp::TexSample;
    }
    ts.type = ValType::TupleValToken;
    ts.aux_str = tex_key;
    ts.aux_bool = (op == SpvOpImageSampleImplicitLod || op == SpvOpImageSampleDrefImplicitLod ||
                   op == SpvOpImageSampleProjImplicitLod);
    ts.aux_bool2 = (op == SpvOpImageGather || op == SpvOpImageDrefGather);
    if (is_cube_dim && (ts.op == SsaOp::TexSample || ts.op == SsaOp::TexSampleLod)) {
        ts.imm_u32_2 = 1u;
    }
    ts.args = {uv};
    for (size_t ci = 2; ci < coord.lanes.size(); ++ci) {
        ts.args.push_back(coord.lanes[ci]);
    }
    size_t op_start = 5u;
    if (op == SpvOpImageSampleDrefImplicitLod || op == SpvOpImageSampleDrefExplicitLod ||
        op == SpvOpImageSampleProjDrefImplicitLod || op == SpvOpImageSampleProjDrefExplicitLod ||
        op == SpvOpImageGather || op == SpvOpImageDrefGather) {
        if (inst.words.size() > 5u) {
            ts.args.push_back(get_value(blk_id, inst.words[5]).scalar());
        }
        op_start = 6u;
    }
    if (inst.words.size() > op_start + 1u) {
        for (size_t wi = op_start + 1u; wi < inst.words.size(); ++wi) {
            LoweredValue extra = get_value(blk_id, inst.words[wi]);
            for (SsaId el : extra.lanes) ts.args.push_back(el);
        }
    }
    ts.args.push_back(sb_.current_token(blk_id));
    ts.is_barrier_or_cf = true;
    ts.is_side_effect = true;
    SsaId tup = sb_.emit(blk_id, std::move(ts));

    SsaNode ev{}; ev.op = SsaOp::TupleExtract; ev.type = ValType::V4F32; ev.imm_u32 = 0; ev.args = {tup};
    SsaId vec4 = sb_.emit(blk_id, std::move(ev));
    SsaNode et{}; et.op = SsaOp::TupleExtract; et.type = ValType::Token; et.imm_u32 = 1; et.args = {tup};
    sb_.set_token(blk_id, sb_.emit(blk_id, std::move(et)));

    uint32_t nlanes = flat_lane_count(res_type);
    ValType lane_vt = to_val_type(scalar_elem_type(res_type));
    for (uint32_t c = 0; c < nlanes; ++c) {
        SsaNode ex{};
        ex.op = SsaOp::VecExtract;
        ex.type = ValType::F32;
        ex.imm_u32 = std::min<uint32_t>(3u, c);
        ex.args = {vec4};
        SsaId f = sb_.emit(blk_id, std::move(ex));
        out.lanes.push_back(sb_.emit_bitcast(blk_id, lane_vt, f));
    }
    return out;
}

void SpirvLowerer::lower_image_write(uint32_t blk_id, const SpirvInst& inst) {
    spirv_parser::SpvImageWriteView iw = inst;
    uint32_t img_id = iw.image;
    LoweredValue coord = get_value(blk_id, iw.coordinate);
    LoweredValue texel = get_value(blk_id, iw.texel);

    std::string img_key = "img:0:0";
    if (auto it = sampled_images_.find(img_id); it != sampled_images_.end()) {
        img_key = it->second.canonical_key;
    }

    // Represent image stores via PsbStore on a synthetic image buffer key indexed by packed (x,y).
    SsaId x_raw = !coord.lanes.empty() ? coord.lanes[0] : sb_.emit_const_u32(blk_id, 0u);
    SsaId x = emit_op(blk_id, SsaOp::BitAnd, ValType::U32, {sb_.emit_bitcast(blk_id, ValType::U32, x_raw), sb_.emit_const_u32(blk_id, 0xFFFFu)});
    SsaId y = (coord.lanes.size() >= 2) ? coord.lanes[1] : sb_.emit_const_u32(blk_id, 0u);
    SsaId xy = emit_op(blk_id, SsaOp::Pack64, ValType::U64, {x, y});
    for (uint32_t c = 0; c < texel.lanes.size(); ++c) {
        SsaId addr = (c == 0) ? xy : emit_op(blk_id, SsaOp::IAdd64, ValType::U64, {xy, sb_.emit_const_u64(blk_id, uint64_t(c) * 4u)});
        SsaNode st{};
        st.op = SsaOp::PsbStore;
        st.type = ValType::Token;
        st.imm_u32 = 4u;
        st.aux_str = img_key;
        st.args = {addr, sb_.emit_bitcast(blk_id, ValType::U32, texel.lanes[c]), sb_.current_token(blk_id)};
        st.is_barrier_or_cf = true;
        st.is_side_effect = true;
        sb_.set_token(blk_id, sb_.emit(blk_id, std::move(st)));
    }
}

} // namespace valhall::spirv2ssa

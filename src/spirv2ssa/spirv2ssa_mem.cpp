#include "spirv2ssa_internal.h"

#include <algorithm>
#include <cstring>

namespace valhall::spirv2ssa {

namespace {

using compiler::SsaId;
using compiler::SsaNode;
using compiler::ValType;
using tables::SsaOp;

constexpr uint32_t kScUniformConstant = 0;
constexpr uint32_t kScInput = 1;
constexpr uint32_t kScUniform = 2;
constexpr uint32_t kScOutput = 3;
constexpr uint32_t kScWorkgroup = 4;
constexpr uint32_t kScPrivate = 6;
constexpr uint32_t kScFunction = 7;
constexpr uint32_t kScPushConstant = 9;
constexpr uint32_t kScStorageBuffer = 12;
constexpr uint32_t kScPhysicalStorageBuffer = 5349;

const char* builtin_canonical_name(uint32_t bi) {
    switch (bi) {
        case 0: return "gl_Position";
        case 1: return "gl_PointSize";
        case 15: return "gl_FragCoord";
        case 17: return "gl_FrontFacing";
        case 22: return "gl_FragDepth";
        case 26: return "gl_WorkGroupID";
        case 27: return "gl_LocalInvocationID";
        case 28: return "gl_GlobalInvocationID";
        case 29: return "gl_LocalInvocationIndex";
        case 36: return "gl_SubgroupSize";
        case 41: return "gl_SubgroupInvocationID";
        case 42: return "gl_VertexIndex";
        case 43: return "gl_InstanceIndex";
        default: return "gl_BuiltIn";
    }
}

bool is_buffer_storage_class(uint32_t sc) {
    return sc == kScUniform || sc == kScStorageBuffer || sc == kScPushConstant ||
           sc == kScPhysicalStorageBuffer || sc == kScWorkgroup;
}

/// Compute flat lane index offset inside a composite type along a constant index path.
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

    // Single forward predecessor in RPO: inherit directly without creating a Phi.
    if (blk.preds.size() == 1 && blk.preds[0] < blk_id) {
        SsaId v = read_slot(blk.preds[0], slot_key, vt);
        blk_slot_defs_[blk_id][slot_key] = v;
        return v;
    }

    // Join block or loop header: create a deferred mem2reg Phi node and record it.
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
    // Iterative resolution because reading a predecessor slot may insert new Phis upstream.
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
    // Sort global variables deterministically by ID so sideband slot assignment is stable.
    std::vector<uint32_t> var_ids;
    var_ids.reserve(mod_.global_vars.size());
    for (const auto& [vid, _] : mod_.global_vars) {
        var_ids.push_back(vid);
    }
    std::sort(var_ids.begin(), var_ids.end());

    for (uint32_t vid : var_ids) {
        const auto& gv = mod_.global_vars.at(vid);
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
            bool is_ssbo = (gv.storage_class == kScStorageBuffer) ||
                           (gv.storage_class == kScUniform && pt && pt->is_buffer_block);
            ResourceKind rk = ResourceKind::UniformBuffer;
            std::string key;
            if (gv.storage_class == kScPushConstant) {
                rk = ResourceKind::PushConstant;
                key = "push_const";
            } else if (gv.storage_class == kScWorkgroup) {
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
        } else if (gv.storage_class == kScInput) {
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
        } else if (gv.storage_class == kScOutput) {
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
        } else if (gv.storage_class == kScUniformConstant) {
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
        } else if (gv.storage_class == kScPrivate || gv.storage_class == kScFunction) {
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
    // OpAccessChain / OpInBoundsAccessChain: [hdr, result_type, result_id, base_ptr, idx0, ...]
    PointerRef res{};
    if (inst.words.size() < 4) return res;
    uint32_t result_ptr_type = inst.words[1];
    uint32_t base_id = inst.words[3];

    auto it = pointers_.find(base_id);
    if (it != pointers_.end()) {
        res = it->second;
    } else {
        // PhysicalStorageBuffer pointer constructed from u64 value via OpConvertUToPtr/OpBitcast
        LoweredValue bv = get_value(blk_id, base_id);
        res.kind = PtrKind::Buffer;
        res.base_ptr = bv.scalar();
        res.byte_offset = sb_.emit_const_u64(blk_id, 0u);
        res.has_const_offset = true;
        res.const_offset = 0;
    }

    uint32_t cur_type = res.pointee_type_id;
    for (size_t k = 4; k < inst.words.size(); ++k) {
        uint32_t idx_id = inst.words[k];
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
                cur_type = (c_idx < t->member_types.size()) ? t->member_types[c_idx] : 0;
            } else if (t->kind == SpirvTypeKind::Array || t->kind == SpirvTypeKind::RuntimeArray) {
                uint32_t stride = (t->array_stride != 0) ? t->array_stride : mod_.type_byte_size(t->elem_type);
                if (is_const_idx) {
                    uint64_t delta = uint64_t(c_idx) * uint64_t(stride);
                    res.const_offset += delta;
                    if (delta != 0) {
                        SsaId add_c = sb_.emit_const_u64(blk_id, delta);
                        res.byte_offset = emit_op(blk_id, SsaOp::IAdd64, ValType::U64, {res.byte_offset, add_c});
                    }
                } else {
                    res.has_const_offset = false;
                    LoweredValue iv = get_value(blk_id, idx_id);
                    SsaId i32 = iv.scalar();
                    const SpirvType* itype = mod_.get_type(iv.spv_type_id);
                    SsaId i64 = (itype && itype->width == 64)
                                    ? i32
                                    : emit_op(blk_id,
                                              (itype && itype->is_signed) ? SsaOp::SConvert32To64 : SsaOp::UConvert32To64,
                                              ValType::U64, {i32});
                    SsaId stride_c = sb_.emit_const_u64(blk_id, stride);
                    SsaId delta = emit_op(blk_id, SsaOp::IMul64, ValType::U64, {i64, stride_c});
                    res.byte_offset = emit_op(blk_id, SsaOp::IAdd64, ValType::U64, {res.byte_offset, delta});
                }
                cur_type = t->elem_type;
            } else if (t->kind == SpirvTypeKind::Vector || t->kind == SpirvTypeKind::Matrix) {
                uint32_t elem_sz = mod_.type_byte_size(t->elem_type);
                if (is_const_idx) {
                    uint64_t delta = uint64_t(c_idx) * uint64_t(elem_sz);
                    res.const_offset += delta;
                    if (delta != 0) {
                        SsaId add_c = sb_.emit_const_u64(blk_id, delta);
                        res.byte_offset = emit_op(blk_id, SsaOp::IAdd64, ValType::U64, {res.byte_offset, add_c});
                    }
                } else {
                    res.has_const_offset = false;
                    LoweredValue iv = get_value(blk_id, idx_id);
                    SsaId i64 = emit_op(blk_id, SsaOp::UConvert32To64, ValType::U64, {iv.scalar()});
                    SsaId delta = emit_op(blk_id, SsaOp::IMul64, ValType::U64, {i64, sb_.emit_const_u64(blk_id, elem_sz)});
                    res.byte_offset = emit_op(blk_id, SsaOp::IAdd64, ValType::U64, {res.byte_offset, delta});
                }
                cur_type = t->elem_type;
            }
        } else {
            // LocalVar / InputVar / OutputVar
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
    // OpLoad: [hdr, result_type, result_id, pointer, (memory_operands...)]
    LoweredValue out{};
    if (inst.words.size() < 4) return out;
    uint32_t res_type = inst.words[1];
    uint32_t ptr_id = inst.words[3];
    out.spv_type_id = res_type;

    auto pit = pointers_.find(ptr_id);
    if (pit == pointers_.end()) {
        return make_zero_value(blk_id, res_type);
    }
    const PointerRef& ptr = pit->second;

    if (ptr.kind == PtrKind::ImageVar) {
        // Loading a SampledImage/Image variable yields a handle tracked via sampled_images_.
        sampled_images_[inst.words[2]] = ptr;
        out.lanes.push_back(sb_.emit_const_u32(blk_id, ptr.root_var_id));
        return out;
    }

    if (ptr.kind == PtrKind::Buffer) {
        SsaId base_addr = emit_op(blk_id, SsaOp::IAdd64, ValType::U64, {ptr.base_ptr, ptr.byte_offset});
        const SpirvType* rt = mod_.get_type(res_type);
        uint32_t nlanes = flat_lane_count(res_type);
        uint32_t elem_tid = scalar_elem_type(res_type);
        ValType elem_vt = to_val_type(elem_tid);
        uint32_t elem_bytes = mod_.type_byte_size(elem_tid);

        // Emit one PsbLoad per scalar lane (matching Valhall's 32-bit word PsbLoad model).
        for (uint32_t i = 0; i < nlanes; ++i) {
            SsaId lane_addr = (i == 0)
                                  ? base_addr
                                  : emit_op(blk_id, SsaOp::IAdd64, ValType::U64,
                                            {base_addr, sb_.emit_const_u64(blk_id, uint64_t(i) * elem_bytes)});
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
            ev.type = (elem_vt == ValType::U64 || elem_vt == ValType::S64) ? ValType::U32 : ValType::U32;
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
                // 64-bit load: combine lo32 and hi32 via Pack64.
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

                SsaNode ev_hi{};
                ev_hi.op = SsaOp::TupleExtract;
                ev_hi.type = ValType::U32;
                ev_hi.imm_u32 = 0;
                ev_hi.args = {tup_hi};
                SsaId hi_u32 = sb_.emit(blk_id, std::move(ev_hi));

                SsaNode et_hi{};
                et_hi.op = SsaOp::TupleExtract;
                et_hi.type = ValType::Token;
                et_hi.imm_u32 = 1;
                et_hi.args = {tup_hi};
                sb_.set_token(blk_id, sb_.emit(blk_id, std::move(et_hi)));

                SsaId packed = emit_op(blk_id, SsaOp::Pack64, ValType::U64, {raw_u32, hi_u32});
                out.lanes.push_back(packed);
            } else if (rt && rt->kind == SpirvTypeKind::Float && rt->width == 16) {
                // FP16 load -> unpack low half to F32.
                SsaId v2 = emit_op(blk_id, SsaOp::UnpackHalf2x16, ValType::V2F32, {raw_u32});
                SsaNode ex{};
                ex.op = SsaOp::VecExtract;
                ex.type = ValType::F32;
                ex.imm_u32 = 0;
                ex.args = {v2};
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
            if (bi == 15) bi_vt = ValType::V4F32;       // FragCoord
            else if (bi == 26 || bi == 27 || bi == 28) bi_vt = ValType::V3U32;
            else if (bi == 42 || bi == 43 || bi == 29 || bi == 41 || bi == 36) bi_vt = ValType::U32;
            else if (bi == 17) bi_vt = ValType::Bool;

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

        // Location-based Vertex Attribute or Fragment Varying
        uint32_t loc = (ptr.location >= 0) ? static_cast<uint32_t>(ptr.location) : 0u;
        SsaNode ld{};
        ld.op = (res_.stage == ShaderStage::Vertex) ? SsaOp::LoadVertAttr : SsaOp::LoadFragVarying;
        ld.type = ValType::V4F32;
        ld.imm_u32 = loc;
        ld.aux_bool = ptr.is_flat;
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

    // LocalVar or OutputVar (read-back of previously stored output)
    uint32_t root_tid = local_var_types_.count(ptr.root_var_id) ? local_var_types_[ptr.root_var_id] : res_type;
    uint32_t base_flat = compute_flat_offset(mod_, *this, root_tid, ptr.const_path);
    uint32_t nlanes = flat_lane_count(res_type);
    ValType lane_vt = to_val_type(scalar_elem_type(res_type));

    if (ptr.dynamic_index != compiler::kInvalidSsa) {
        // Select-mux across array elements for dynamic local array indexing.
        const SpirvType* rt = mod_.get_type(root_tid);
        uint32_t total_elems = (rt && (rt->kind == SpirvTypeKind::Array || rt->kind == SpirvTypeKind::Vector))
                                   ? std::min<uint32_t>(rt->count, 32u)
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
    // OpStore: [hdr, pointer, object, (memory_operands...)]
    if (inst.words.size() < 3) return;
    uint32_t ptr_id = inst.words[1];
    uint32_t val_id = inst.words[2];

    auto pit = pointers_.find(ptr_id);
    if (pit == pointers_.end()) return;
    const PointerRef& ptr = pit->second;
    LoweredValue val = get_value(blk_id, val_id);
    if (val.empty()) return;

    if (ptr.kind == PtrKind::Buffer) {
        SsaId base_addr = emit_op(blk_id, SsaOp::IAdd64, ValType::U64, {ptr.base_ptr, ptr.byte_offset});
        uint32_t elem_tid = scalar_elem_type(val.spv_type_id);
        uint32_t elem_bytes = mod_.type_byte_size(elem_tid);

        for (uint32_t i = 0; i < val.lanes.size(); ++i) {
            SsaId lane_addr = (i == 0)
                                  ? base_addr
                                  : emit_op(blk_id, SsaOp::IAdd64, ValType::U64,
                                            {base_addr, sb_.emit_const_u64(blk_id, uint64_t(i) * elem_bytes)});
            SsaId lane_val = val.lanes[i];
            if (elem_bytes == 8u) {
                SsaNode sp{};
                sp.op = SsaOp::Split64;
                sp.type = ValType::TupleU32x2;
                sp.args = {lane_val};
                SsaId tup = sb_.emit(blk_id, std::move(sp));
                SsaNode e0{}; e0.op = SsaOp::TupleExtract; e0.type = ValType::U32; e0.imm_u32 = 0; e0.args = {tup};
                SsaNode e1{}; e1.op = SsaOp::TupleExtract; e1.type = ValType::U32; e1.imm_u32 = 1; e1.args = {tup};
                SsaId lo = sb_.emit(blk_id, std::move(e0));
                SsaId hi = sb_.emit(blk_id, std::move(e1));

                SsaNode st0{};
                st0.op = SsaOp::PsbStore;
                st0.type = ValType::Token;
                st0.imm_u32 = 4u;
                st0.aux_str = ptr.canonical_key;
                st0.args = {lane_addr, lo, sb_.current_token(blk_id)};
                st0.is_barrier_or_cf = true;
                st0.is_side_effect = true;
                sb_.set_token(blk_id, sb_.emit(blk_id, std::move(st0)));

                SsaId hi_addr = emit_op(blk_id, SsaOp::IAdd64, ValType::U64,
                                        {lane_addr, sb_.emit_const_u64(blk_id, 4u)});
                SsaNode st1{};
                st1.op = SsaOp::PsbStore;
                st1.type = ValType::Token;
                st1.imm_u32 = 4u;
                st1.aux_str = ptr.canonical_key;
                st1.args = {hi_addr, hi, sb_.current_token(blk_id)};
                st1.is_barrier_or_cf = true;
                st1.is_side_effect = true;
                sb_.set_token(blk_id, sb_.emit(blk_id, std::move(st1)));
            } else {
                SsaId u32_val = sb_.emit_bitcast(blk_id, ValType::U32, lane_val);
                SsaNode st{};
                st.op = SsaOp::PsbStore;
                st.type = ValType::Token;
                st.imm_u32 = std::min<uint32_t>(elem_bytes, 4u);
                st.aux_str = ptr.canonical_key;
                st.args = {lane_addr, u32_val, sb_.current_token(blk_id)};
                st.is_barrier_or_cf = true;
                st.is_side_effect = true;
                sb_.set_token(blk_id, sb_.emit(blk_id, std::move(st)));
            }
        }
        return;
    }

    // LocalVar or OutputVar: update SSA mem2reg slots.
    if (ptr.kind == PtrKind::OutputVar) {
        written_output_vars_.insert(ptr.root_var_id);
    }
    uint32_t root_tid = local_var_types_.count(ptr.root_var_id) ? local_var_types_[ptr.root_var_id] : val.spv_type_id;
    uint32_t base_flat = compute_flat_offset(mod_, *this, root_tid, ptr.const_path);

    if (ptr.dynamic_index != compiler::kInvalidSsa) {
        const SpirvType* rt = mod_.get_type(root_tid);
        uint32_t total_elems = (rt && (rt->kind == SpirvTypeKind::Array || rt->kind == SpirvTypeKind::Vector))
                                   ? std::min<uint32_t>(rt->count, 32u)
                                   : 4u;
        uint32_t nlanes = static_cast<uint32_t>(val.lanes.size());
        for (uint32_t e = 0; e < total_elems; ++e) {
            SsaId cmp = emit_op(blk_id, SsaOp::ICmpEq, ValType::Bool,
                                {ptr.dynamic_index, sb_.emit_const_u32(blk_id, e)});
            for (uint32_t lane = 0; lane < nlanes; ++lane) {
                uint64_t sk = make_slot_key(ptr.root_var_id, base_flat + e * nlanes + lane);
                ValType vt = res_.ssa.get(val.lanes[lane]).type;
                SsaId prev = read_slot(blk_id, sk, vt);
                SsaId sel = emit_op(blk_id, SsaOp::Select, vt, {cmp, val.lanes[lane], prev});
                write_slot(blk_id, sk, sel);
            }
        }
        return;
    }

    for (uint32_t i = 0; i < val.lanes.size(); ++i) {
        write_slot(blk_id, make_slot_key(ptr.root_var_id, base_flat + i), val.lanes[i]);
    }
}

void SpirvLowerer::flush_stage_outputs(uint32_t blk_id) {
    for (uint32_t vid : output_var_ids_) {
        if (!written_output_vars_.count(vid)) continue;
        const SpirvVariable* gv = mod_.get_global_var(vid);
        if (!gv) continue;
        const SpirvType* pt = mod_.get_type(gv->pointee_type_id);
        if (!pt) continue;

        // Handle gl_PerVertex struct outputs (e.g. gl_Position at member 0).
        if (pt->kind == SpirvTypeKind::Struct) {
            uint32_t flat_off = 0;
            for (size_t m = 0; m < pt->member_types.size(); ++m) {
                uint32_t mt = pt->member_types[m];
                uint32_t ml = flat_lane_count(mt);
                if (m < pt->member_builtins.size() && pt->member_builtins[m] >= 0) {
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

LoweredValue SpirvLowerer::lower_atomic(uint32_t blk_id, const SpirvInst& inst) {
    // OpAtomicIAdd(234), OpAtomicISub(235), OpAtomicSMin(236), OpAtomicUMin(237),
    // OpAtomicSMax(238), OpAtomicUMax(239), OpAtomicAnd(240), OpAtomicOr(241),
    // OpAtomicXor(242), OpAtomicExchange(229), OpAtomicCompareExchange(230)
    LoweredValue out{};
    if (inst.words.size() < 7) return out;
    uint32_t res_type = inst.words[1];
    uint32_t ptr_id = inst.words[3];
    out.spv_type_id = res_type;

    auto pit = pointers_.find(ptr_id);
    if (pit == pointers_.end() || pit->second.kind != PtrKind::Buffer) {
        return make_zero_value(blk_id, res_type);
    }
    const PointerRef& ptr = pit->second;
    SsaId addr = emit_op(blk_id, SsaOp::IAdd64, ValType::U64, {ptr.base_ptr, ptr.byte_offset});
    SsaId tok = sb_.current_token(blk_id);

    SsaNode at{};
    at.type = ValType::TupleValToken;
    at.is_barrier_or_cf = true;
    at.is_side_effect = true;

    if (inst.opcode == 229) { // OpAtomicExchange
        SsaId val = get_value(blk_id, inst.words[6]).scalar();
        at.op = SsaOp::AtomicExchange;
        at.args = {addr, val, tok};
    } else if (inst.opcode == 230 && inst.words.size() >= 9) { // OpAtomicCompareExchange
        SsaId val = get_value(blk_id, inst.words[7]).scalar();
        SsaId cmp = get_value(blk_id, inst.words[8]).scalar();
        at.op = SsaOp::AtomicCmpXchg;
        at.args = {addr, val, cmp, tok};
    } else {
        SsaId val = get_value(blk_id, inst.words[6]).scalar();
        at.op = SsaOp::AtomicRmw;
        switch (inst.opcode) {
            case 234: at.aux_str = "aadd"; break;
            case 235: at.aux_str = "asub"; break;
            case 236: at.aux_str = "asmin"; break;
            case 237: at.aux_str = "aumin"; break;
            case 238: at.aux_str = "asmax"; break;
            case 239: at.aux_str = "aumax"; break;
            case 240: at.aux_str = "aand"; break;
            case 241: at.aux_str = "aor"; break;
            case 242: at.aux_str = "axor"; break;
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
    // OpImageSampleImplicitLod(87), OpImageSampleExplicitLod(88), OpImageFetch(95), OpImageGather(96)
    LoweredValue out{};
    if (inst.words.size() < 5) return out;
    uint32_t res_type = inst.words[1];
    uint32_t img_id = inst.words[3];
    uint32_t coord_id = inst.words[4];
    out.spv_type_id = res_type;

    LoweredValue coord = get_value(blk_id, coord_id);
    SsaId uv = pack_to_vec2_f32(blk_id, coord);

    std::string tex_key = "tex:0:0";
    if (auto it = sampled_images_.find(img_id); it != sampled_images_.end()) {
        tex_key = it->second.canonical_key;
    }

    SsaNode ts{};
    ts.op = SsaOp::TexSample;
    ts.type = ValType::TupleValToken;
    ts.aux_str = tex_key;
    ts.aux_bool = (inst.opcode == 87);  // implicit LOD
    ts.aux_bool2 = (inst.opcode == 96); // gather
    ts.args = {uv, sb_.current_token(blk_id)};
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

} // namespace valhall::spirv2ssa

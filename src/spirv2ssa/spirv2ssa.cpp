#include "spirv2ssa_internal.h"

#include <algorithm>
#include <functional>

namespace valhall::spirv2ssa {

namespace {

using compiler::SsaBlock;
using compiler::SsaId;
using compiler::SsaNode;
using compiler::ValType;
using tables::SsaOp;

constexpr uint16_t kOpNop = 0;
constexpr uint16_t kOpExtInst = 12;
constexpr uint16_t kOpVariable = 59;
constexpr uint16_t kOpLoad = 61;
constexpr uint16_t kOpStore = 62;
constexpr uint16_t kOpAccessChain = 65;
constexpr uint16_t kOpInBoundsAccessChain = 66;
constexpr uint16_t kOpImageSampleImplicitLod = 87;
constexpr uint16_t kOpImageSampleExplicitLod = 88;
constexpr uint16_t kOpImageFetch = 95;
constexpr uint16_t kOpImageGather = 96;
constexpr uint16_t kOpControlBarrier = 224;
constexpr uint16_t kOpMemoryBarrier = 225;
constexpr uint16_t kOpAtomicExchange = 229;
constexpr uint16_t kOpAtomicCompareExchange = 230;
constexpr uint16_t kOpAtomicIAdd = 234;
constexpr uint16_t kOpAtomicXor = 242;
constexpr uint16_t kOpPhi = 245;
constexpr uint16_t kOpLoopMerge = 246;
constexpr uint16_t kOpSelectionMerge = 247;
constexpr uint16_t kOpBranch = 249;
constexpr uint16_t kOpBranchConditional = 250;
constexpr uint16_t kOpSwitch = 251;
constexpr uint16_t kOpKill = 252;
constexpr uint16_t kOpReturn = 253;
constexpr uint16_t kOpReturnValue = 254;
constexpr uint16_t kOpUnreachable = 255;
constexpr uint16_t kOpDemoteToHelperInvocation = 5380;

ShaderStage execution_model_to_stage(uint32_t em) {
    switch (em) {
        case 0: return ShaderStage::Vertex;
        case 4: return ShaderStage::Fragment;
        case 5: return ShaderStage::Compute;
        default: return ShaderStage::Compute;
    }
}

std::vector<uint32_t> block_succ_labels(const SpirvBasicBlock& blk) {
    std::vector<uint32_t> succs;
    if (blk.insts.empty()) return succs;
    const auto& term = blk.insts.back();
    if (term.opcode == kOpBranch && term.words.size() >= 2) {
        succs.push_back(term.words[1]);
    } else if (term.opcode == kOpBranchConditional && term.words.size() >= 4) {
        succs.push_back(term.words[2]);
        if (term.words[3] != term.words[2]) succs.push_back(term.words[3]);
    } else if (term.opcode == kOpSwitch && term.words.size() >= 3) {
        succs.push_back(term.words[2]);
        for (size_t k = 4; k < term.words.size(); k += 2) {
            succs.push_back(term.words[k]);
        }
    }
    return succs;
}

} // namespace

SpirvLowerer::SpirvLowerer(const SpirvModule& mod, const Spirv2SsaOptions& opts, Spirv2SsaResult& res)
    : mod_(mod), opts_(opts), res_(res), sb_(res.ssa) {}

ValType SpirvLowerer::to_val_type(uint32_t spv_type_id) const {
    const SpirvType* t = mod_.get_type(spv_type_id);
    if (!t) return ValType::U32;
    switch (t->kind) {
        case SpirvTypeKind::Void: return ValType::Void;
        case SpirvTypeKind::Bool: return ValType::Bool;
        case SpirvTypeKind::Int:
            if (t->width == 64) return t->is_signed ? ValType::S64 : ValType::U64;
            return t->is_signed ? ValType::S32 : ValType::U32;
        case SpirvTypeKind::Float:
            return ValType::F32;
        case SpirvTypeKind::Pointer:
            return ValType::U64;
        case SpirvTypeKind::Vector: {
            ValType elem = to_val_type(t->elem_type);
            if (t->count == 2) {
                if (elem == ValType::F32) return ValType::V2F32;
                if (elem == ValType::S32) return ValType::V2S32;
                if (elem == ValType::Bool) return ValType::V2Bool;
                return ValType::V2U32;
            }
            if (t->count == 3) return ValType::V3U32;
            if (t->count == 4) {
                if (elem == ValType::F32) return ValType::V4F32;
                if (elem == ValType::S32) return ValType::V4S32;
                return ValType::V4U32;
            }
            return elem;
        }
        default:
            return ValType::U32;
    }
}

uint32_t SpirvLowerer::flat_lane_count(uint32_t spv_type_id) const {
    const SpirvType* t = mod_.get_type(spv_type_id);
    if (!t) return 1u;
    switch (t->kind) {
        case SpirvTypeKind::Void: return 0u;
        case SpirvTypeKind::Vector: return std::max<uint32_t>(1u, t->count);
        case SpirvTypeKind::Matrix: return std::max<uint32_t>(1u, t->count) * flat_lane_count(t->elem_type);
        case SpirvTypeKind::Array: return std::min<uint32_t>(t->count, 64u) * flat_lane_count(t->elem_type);
        case SpirvTypeKind::Struct: {
            uint32_t sum = 0;
            for (uint32_t mt : t->member_types) sum += flat_lane_count(mt);
            return std::max<uint32_t>(1u, sum);
        }
        default: return 1u;
    }
}

uint32_t SpirvLowerer::scalar_elem_type(uint32_t spv_type_id) const {
    const SpirvType* t = mod_.get_type(spv_type_id);
    while (t) {
        if (t->kind == SpirvTypeKind::Vector || t->kind == SpirvTypeKind::Matrix ||
            t->kind == SpirvTypeKind::Array || t->kind == SpirvTypeKind::RuntimeArray) {
            spv_type_id = t->elem_type;
            t = mod_.get_type(spv_type_id);
        } else if (t->kind == SpirvTypeKind::Struct && !t->member_types.empty()) {
            spv_type_id = t->member_types[0];
            t = mod_.get_type(spv_type_id);
        } else {
            break;
        }
    }
    return spv_type_id;
}

SsaId SpirvLowerer::emit_op(
    uint32_t blk_id,
    SsaOp op,
    ValType vt,
    std::vector<SsaId> args,
    bool barrier,
    bool side_eff) {
    SsaNode n{};
    n.op = op;
    n.type = vt;
    n.args = std::move(args);
    n.is_barrier_or_cf = barrier;
    n.is_side_effect = side_eff;
    return sb_.emit(blk_id, std::move(n));
}

LoweredValue SpirvLowerer::make_zero_value(uint32_t blk_id, uint32_t spv_type_id) {
    LoweredValue lv{};
    lv.spv_type_id = spv_type_id;
    uint32_t n = flat_lane_count(spv_type_id);
    ValType vt = to_val_type(scalar_elem_type(spv_type_id));
    for (uint32_t i = 0; i < n; ++i) {
        if (vt == ValType::F32) lv.lanes.push_back(sb_.emit_const_f32_bits(blk_id, 0u));
        else if (vt == ValType::Bool) lv.lanes.push_back(sb_.emit_const_bool(blk_id, false));
        else if (vt == ValType::U64 || vt == ValType::S64) lv.lanes.push_back(sb_.emit_const_u64(blk_id, 0u));
        else if (vt == ValType::S32) lv.lanes.push_back(sb_.emit_const_s32(blk_id, 0));
        else lv.lanes.push_back(sb_.emit_const_u32(blk_id, 0u));
    }
    return lv;
}

LoweredValue SpirvLowerer::get_value(uint32_t blk_id, uint32_t spv_id) {
    if (auto it = values_.find(spv_id); it != values_.end()) {
        return it->second;
    }
    if (const SpirvConstant* c = mod_.get_const(spv_id)) {
        LoweredValue lv{};
        lv.spv_type_id = c->type_id;
        if (c->kind == SpirvConstKind::Null || c->kind == SpirvConstKind::Undef) {
            return make_zero_value(blk_id, c->type_id);
        }
        if (c->kind == SpirvConstKind::Bool) {
            lv.lanes.push_back(sb_.emit_const_bool(blk_id, c->scalar_bits != 0));
            return lv;
        }
        if (c->kind == SpirvConstKind::Scalar32) {
            const SpirvType* t = mod_.get_type(c->type_id);
            if (t && t->kind == SpirvTypeKind::Float) {
                lv.lanes.push_back(sb_.emit_const_f32_bits(blk_id, static_cast<uint32_t>(c->scalar_bits)));
            } else if (t && t->kind == SpirvTypeKind::Int && t->is_signed) {
                lv.lanes.push_back(sb_.emit_const_s32(blk_id, static_cast<int32_t>(c->scalar_bits)));
            } else {
                lv.lanes.push_back(sb_.emit_const_u32(blk_id, static_cast<uint32_t>(c->scalar_bits)));
            }
            return lv;
        }
        if (c->kind == SpirvConstKind::Scalar64) {
            lv.lanes.push_back(sb_.emit_const_u64(blk_id, c->scalar_bits));
            return lv;
        }
        if (c->kind == SpirvConstKind::Composite) {
            for (uint32_t cid : c->constituents) {
                LoweredValue sub = get_value(blk_id, cid);
                lv.lanes.insert(lv.lanes.end(), sub.lanes.begin(), sub.lanes.end());
            }
            return lv;
        }
    }
    return make_zero_value(blk_id, 0u);
}

SsaId SpirvLowerer::pack_to_vec4_f32(uint32_t blk_id, const LoweredValue& val, bool fill_alpha_one) {
    std::vector<SsaId> c(4, compiler::kInvalidSsa);
    for (uint32_t i = 0; i < 4; ++i) {
        if (i < val.lanes.size()) {
            c[i] = sb_.emit_bitcast(blk_id, ValType::F32, val.lanes[i]);
        } else {
            c[i] = sb_.emit_const_f32_bits(blk_id, (i == 3 && fill_alpha_one) ? 0x3F800000u : 0u);
        }
    }
    return emit_op(blk_id, SsaOp::VecConstruct4, ValType::V4F32, std::move(c));
}

SsaId SpirvLowerer::pack_to_vec2_f32(uint32_t blk_id, const LoweredValue& val) {
    SsaId x = !val.lanes.empty() ? sb_.emit_bitcast(blk_id, ValType::F32, val.lanes[0])
                                 : sb_.emit_const_f32_bits(blk_id, 0u);
    SsaId y = (val.lanes.size() >= 2) ? sb_.emit_bitcast(blk_id, ValType::F32, val.lanes[1])
                                      : sb_.emit_const_f32_bits(blk_id, 0u);
    return emit_op(blk_id, SsaOp::VecConstruct2, ValType::V2F32, {x, y});
}

void SpirvLowerer::record_unsupported(const std::string& desc) {
    if (seen_unsupported_.insert(desc).second) {
        res_.unsupported_ops.push_back(desc);
    }
}

bool SpirvLowerer::run() {
    res_.stage = execution_model_to_stage(mod_.entry_point.execution_model);
    res_.entry_point_name = mod_.entry_point.name;
    res_.local_size_x = mod_.entry_point.local_size_x;
    res_.local_size_y = mod_.entry_point.local_size_y;
    res_.local_size_z = mod_.entry_point.local_size_z;

    const SpirvFunction* entry_fn = mod_.get_function(mod_.entry_point.func_id);
    if (!entry_fn || entry_fn->blocks.empty()) {
        res_.error_message = "Entry point function not found or has no basic blocks";
        return false;
    }

    // Build label -> SpirvBasicBlock map and compute Reverse Post-Order (RPO) from entry block.
    std::unordered_map<uint32_t, const SpirvBasicBlock*> lbl_map;
    for (const auto& b : entry_fn->blocks) {
        lbl_map[b.label_id] = &b;
    }

    std::set<uint32_t> visited;
    std::vector<const SpirvBasicBlock*> postorder;
    std::function<void(uint32_t)> dfs = [&](uint32_t lbl) {
        if (!visited.insert(lbl).second) return;
        auto it = lbl_map.find(lbl);
        if (it == lbl_map.end()) return;
        for (uint32_t succ : block_succ_labels(*it->second)) {
            dfs(succ);
        }
        postorder.push_back(it->second);
    };
    dfs(entry_fn->blocks[0].label_id);
    rpo_blocks_.assign(postorder.rbegin(), postorder.rend());

    uint32_t num_blks = static_cast<uint32_t>(rpo_blocks_.size());
    res_.ssa.blocks.resize(num_blks);
    res_.ssa.block_in_token.assign(num_blks, compiler::kInvalidSsa);
    res_.ssa.block_out_token.assign(num_blks, compiler::kInvalidSsa);
    blk_slot_defs_.resize(num_blks);

    for (uint32_t b = 0; b < num_blks; ++b) {
        res_.ssa.blocks[b].id = b;
        label_to_blk_[rpo_blocks_[b]->label_id] = b;
    }

    // Wire CFG edges and structured merge metadata before instruction lowering.
    for (uint32_t b = 0; b < num_blks; ++b) {
        const auto& spv_blk = *rpo_blocks_[b];
        auto& ssa_blk = res_.ssa.blocks[b];
        for (const auto& inst : spv_blk.insts) {
            if (inst.opcode == kOpSelectionMerge && inst.words.size() >= 2) {
                if (auto it = label_to_blk_.find(inst.words[1]); it != label_to_blk_.end()) {
                    ssa_blk.selection_merge_block = static_cast<int32_t>(it->second);
                }
            } else if (inst.opcode == kOpLoopMerge && inst.words.size() >= 3) {
                ssa_blk.is_loop_header = true;
                if (auto it = label_to_blk_.find(inst.words[1]); it != label_to_blk_.end()) {
                    ssa_blk.loop_merge_block = static_cast<int32_t>(it->second);
                }
                if (auto it = label_to_blk_.find(inst.words[2]); it != label_to_blk_.end()) {
                    ssa_blk.loop_backedge_block = static_cast<int32_t>(it->second);
                }
            }
        }
        if (spv_blk.insts.empty()) continue;
        const auto& term = spv_blk.insts.back();
        if (term.opcode == kOpBranch && term.words.size() >= 2) {
            if (auto it = label_to_blk_.find(term.words[1]); it != label_to_blk_.end()) {
                ssa_blk.is_unconditional_branch = true;
                ssa_blk.branch_target_block = static_cast<int32_t>(it->second);
                res_.ssa.blocks[it->second].preds.push_back(b);
            }
        } else if (term.opcode == kOpBranchConditional && term.words.size() >= 4) {
            auto it_t = label_to_blk_.find(term.words[2]);
            auto it_f = label_to_blk_.find(term.words[3]);
            ssa_blk.is_conditional_branch = true;
            if (it_t != label_to_blk_.end()) {
                ssa_blk.branch_target_block = static_cast<int32_t>(it_t->second);
                res_.ssa.blocks[it_t->second].preds.push_back(b);
            }
            if (it_f != label_to_blk_.end()) {
                ssa_blk.fallthrough_block = static_cast<int32_t>(it_f->second);
                if (it_t == label_to_blk_.end() || it_f->second != it_t->second) {
                    res_.ssa.blocks[it_f->second].preds.push_back(b);
                }
            }
        } else if (term.opcode == kOpReturn || term.opcode == kOpReturnValue) {
            ssa_blk.is_terminated_by_end = true;
        } else if (term.opcode == kOpKill || term.opcode == kOpUnreachable) {
            ssa_blk.is_terminated_by_discard = (term.opcode == kOpKill);
            ssa_blk.is_terminated_by_end = (term.opcode == kOpUnreachable);
        }
    }

    // Initialize entry block token and global resource bindings.
    SsaNode init_tok{};
    init_tok.op = SsaOp::Nop;
    init_tok.type = ValType::Token;
    SsaId tok0 = sb_.emit(0, std::move(init_tok));
    res_.ssa.block_in_token[0] = tok0;
    res_.ssa.block_out_token[0] = tok0;
    init_global_resources(0);

    // Lower blocks in RPO order.
    for (uint32_t b = 0; b < num_blks; ++b) {
        if (b > 0) {
            const auto& blk = res_.ssa.blocks[b];
            if (blk.preds.size() == 1 && blk.preds[0] < b) {
                res_.ssa.block_in_token[b] = res_.ssa.block_out_token[blk.preds[0]];
                res_.ssa.block_out_token[b] = res_.ssa.block_out_token[blk.preds[0]];
            } else {
                SsaNode tphi{};
                tphi.op = SsaOp::Phi;
                tphi.type = ValType::Token;
                tphi.aux_bool = true;
                SsaId tpid = sb_.emit(b, std::move(tphi));
                res_.ssa.block_in_token[b] = tpid;
                res_.ssa.block_out_token[b] = tpid;
            }
        }

        const auto& spv_blk = *rpo_blocks_[b];
        for (const auto& inst : spv_blk.insts) {
            switch (inst.opcode) {
                case kOpNop:
                case kOpSelectionMerge:
                case kOpLoopMerge:
                case kOpMemoryBarrier:
                    break;

                case kOpVariable:
                    if (inst.words.size() >= 4) {
                        uint32_t init_id = (inst.words.size() >= 5) ? inst.words[4] : 0u;
                        register_local_variable(b, inst.words[2], inst.words[1], init_id);
                    }
                    break;

                case kOpPhi: {
                    if (inst.words.size() < 3) break;
                    uint32_t res_type = inst.words[1];
                    uint32_t res_id = inst.words[2];
                    uint32_t nlanes = flat_lane_count(res_type);
                    ValType lane_vt = to_val_type(scalar_elem_type(res_type));
                    PendingSpvPhi p{};
                    p.blk_id = b;
                    p.result_id = res_id;
                    p.type_id = res_type;
                    LoweredValue lv{};
                    lv.spv_type_id = res_type;
                    for (uint32_t i = 0; i < nlanes; ++i) {
                        SsaNode phi{};
                        phi.op = SsaOp::Phi;
                        phi.type = lane_vt;
                        SsaId pid = sb_.emit(b, std::move(phi));
                        p.lane_phi_ids.push_back(pid);
                        lv.lanes.push_back(pid);
                    }
                    for (size_t k = 3; k + 1 < inst.words.size(); k += 2) {
                        p.raw_incomings.push_back({inst.words[k], inst.words[k + 1]});
                    }
                    values_[res_id] = std::move(lv);
                    pending_spv_phis_.push_back(std::move(p));
                    break;
                }

                case kOpAccessChain:
                case kOpInBoundsAccessChain:
                    if (inst.words.size() >= 4) {
                        pointers_[inst.words[2]] = lower_access_chain(b, inst);
                    }
                    break;

                case kOpLoad:
                    if (inst.words.size() >= 4) {
                        values_[inst.words[2]] = lower_load(b, inst);
                    }
                    break;

                case kOpStore:
                    lower_store(b, inst);
                    break;

                case kOpImageSampleImplicitLod:
                case kOpImageSampleExplicitLod:
                case kOpImageFetch:
                case kOpImageGather:
                    if (inst.words.size() >= 5) {
                        values_[inst.words[2]] = lower_image_sample(b, inst);
                    }
                    break;

                case kOpControlBarrier: {
                    SsaNode cb{};
                    cb.op = SsaOp::ControlBarrier;
                    cb.type = ValType::Token;
                    cb.args = {sb_.current_token(b)};
                    cb.is_barrier_or_cf = true;
                    cb.is_side_effect = true;
                    sb_.set_token(b, sb_.emit(b, std::move(cb)));
                    break;
                }

                case kOpExtInst:
                    if (!lower_glsl450_ext_inst(b, inst)) {
                        res_.error_message = "Unsupported GLSL.std.450 instruction";
                        return false;
                    }
                    break;

                case kOpBranch:
                    break;

                case kOpBranchConditional:
                    if (inst.words.size() >= 4) {
                        LoweredValue cond = get_value(b, inst.words[1]);
                        res_.ssa.blocks[b].branch_cond = cond.scalar();
                    }
                    break;

                case kOpSwitch:
                    // Record unsupported for multi-way switch (handled conservatively if tolerant)
                    record_unsupported("OpSwitch");
                    break;

                case kOpKill:
                case kOpDemoteToHelperInvocation: {
                    SsaNode k{};
                    k.op = SsaOp::Kill;
                    k.type = ValType::Void;
                    k.aux_bool = true;
                    k.args = {sb_.emit_const_bool(b, true), sb_.current_token(b)};
                    k.is_barrier_or_cf = true;
                    k.is_side_effect = true;
                    sb_.emit(b, std::move(k));
                    break;
                }

                case kOpReturn:
                case kOpReturnValue:
                    flush_stage_outputs(b);
                    break;

                case kOpUnreachable:
                    break;

                default:
                    if ((inst.opcode >= kOpAtomicExchange && inst.opcode <= kOpAtomicXor) &&
                        inst.opcode != 231 && inst.opcode != 232 && inst.opcode != 233) {
                        values_[inst.words[2]] = lower_atomic(b, inst);
                    } else if (!lower_alu_or_misc_inst(b, inst)) {
                        res_.error_message = "Unsupported SPIR-V opcode " + std::to_string(inst.opcode);
                        return false;
                    }
                    break;
            }
        }
    }

    // Patch deferred SPIR-V OpPhi incoming operands.
    for (const auto& p : pending_spv_phis_) {
        for (uint32_t lane = 0; lane < p.lane_phi_ids.size(); ++lane) {
            auto& phi_node = res_.ssa.get_mut(p.lane_phi_ids[lane]);
            phi_node.phi_incomings.clear();
            for (const auto& [val_id, pred_lbl] : p.raw_incomings) {
                auto bit = label_to_blk_.find(pred_lbl);
                if (bit == label_to_blk_.end()) continue;
                LoweredValue lv = get_value(bit->second, val_id);
                SsaId sid = (lane < lv.lanes.size()) ? lv.lanes[lane] : sb_.emit_const_u32(bit->second, 0u);
                phi_node.phi_incomings.push_back({bit->second, sid});
            }
        }
    }

    // Patch token Phis and mem2reg Phis.
    for (uint32_t b = 1; b < num_blks; ++b) {
        const auto& blk = res_.ssa.blocks[b];
        SsaId tpid = res_.ssa.block_in_token[b];
        if (tpid != compiler::kInvalidSsa && res_.ssa.get(tpid).op == SsaOp::Phi && res_.ssa.get(tpid).aux_bool) {
            auto& tphi = res_.ssa.get_mut(tpid);
            tphi.phi_incomings.clear();
            for (uint32_t pred : blk.preds) {
                tphi.phi_incomings.push_back({pred, res_.ssa.block_out_token[pred]});
            }
        }
    }
    seal_mem_phis();

    res_.ssa.recompute_use_counts();
    res_.ok = true;
    return true;
}

Spirv2SsaResult lower_spirv_to_valhall_ssa(
    const std::vector<uint32_t>& spirv_words,
    const Spirv2SsaOptions& opts) {
    Spirv2SsaResult res;
    SpirvModule mod;
    if (!parse_spirv_module(spirv_words, &mod, &res.error_message)) {
        res.ok = false;
        return res;
    }
    SpirvLowerer lowerer(mod, opts, res);
    lowerer.run();
    return res;
}

} // namespace valhall::spirv2ssa

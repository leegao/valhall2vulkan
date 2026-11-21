#include "spirv2ssa_internal.h"
#include "ssa_opt_pipeline.h"

#include <algorithm>
#include <functional>

namespace valhall::spirv2ssa {

namespace {

using compiler::SsaBlock;
using compiler::SsaId;
using compiler::SsaNode;
using compiler::ValType;
using tables::SsaOp;

ShaderStage execution_model_to_stage(uint32_t em) {
    switch (static_cast<SpvExecutionModel>(em)) {
        case SpvExecutionModelVertex: return ShaderStage::Vertex;
        case SpvExecutionModelFragment: return ShaderStage::Fragment;
        case SpvExecutionModelGLCompute: return ShaderStage::Compute;
        default: return ShaderStage::Compute;
    }
}

std::vector<uint32_t> block_succ_labels(
    const SpirvBasicBlock& blk,
    const std::unordered_map<uint32_t, uint32_t>& id_to_width) {
    std::vector<uint32_t> succs;
    if (blk.insts.empty()) return succs;
    const auto& term = blk.insts.back();
    const auto op = static_cast<SpvOp>(term.opcode);
    if (op == SpvOpBranch) {
        spirv_parser::SpvBranchView v = term;
        succs.push_back(v.target_label);
    } else if (op == SpvOpBranchConditional) {
        spirv_parser::SpvBranchConditionalView v = term;
        succs.push_back(v.true_label);
        if (v.false_label != v.true_label) succs.push_back(v.false_label);
    } else if (op == SpvOpSwitch) {
        spirv_parser::SpvSwitchView v = term;
        succs.push_back(v.default_val);
        uint32_t sel_w = 32u;
        if (auto it = id_to_width.find(v.selector); it != id_to_width.end()) sel_w = it->second;
        for (const auto& [lit, lbl] : v.decode_cases(sel_w)) {
            (void)lit;
            if (std::find(succs.begin(), succs.end(), lbl) == succs.end()) {
                succs.push_back(lbl);
            }
        }
    }
    return succs;
}

/// Returns true for all value-producing SPIR-V 1.6 §3.3.18 atomic instructions
/// (void-returning OpAtomicStore and OpAtomicFlagClear are handled separately).
bool is_atomic_opcode(SpvOp op) {
    switch (op) {
        case SpvOpAtomicLoad:
        case SpvOpAtomicExchange:
        case SpvOpAtomicCompareExchange:
        case SpvOpAtomicCompareExchangeWeak:
        case SpvOpAtomicIIncrement:
        case SpvOpAtomicIDecrement:
        case SpvOpAtomicIAdd:
        case SpvOpAtomicISub:
        case SpvOpAtomicSMin:
        case SpvOpAtomicUMin:
        case SpvOpAtomicSMax:
        case SpvOpAtomicUMax:
        case SpvOpAtomicAnd:
        case SpvOpAtomicOr:
        case SpvOpAtomicXor:
        case SpvOpAtomicFlagTestAndSet:
        case SpvOpAtomicFMinEXT:
        case SpvOpAtomicFMaxEXT:
        case SpvOpAtomicFAddEXT:
            return true;
        default:
            return false;
    }
}

} // namespace

SpirvLowerer::SpirvLowerer(const SpirvModule& mod, const Spirv2SsaOptions& opts, Spirv2SsaResult& res)
    : mod_(mod), opts_(opts), res_(res), sb_(res.ssa) {}

uint32_t SpirvLowerer::allocate_ssa_block() {
    uint32_t bid = static_cast<uint32_t>(res_.ssa.blocks.size());
    SsaBlock blk{};
    blk.id = bid;
    res_.ssa.blocks.push_back(std::move(blk));
    res_.ssa.block_in_token.push_back(compiler::kInvalidSsa);
    res_.ssa.block_out_token.push_back(compiler::kInvalidSsa);
    blk_slot_defs_.emplace_back();
    return bid;
}

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

uint32_t SpirvLowerer::int_type_width(uint32_t spv_type_id) const {
    const SpirvType* t = mod_.get_type(spv_type_id);
    if (t && t->kind == SpirvTypeKind::Int && t->width > 0) return t->width;
    return 32u;
}

SsaId SpirvLowerer::emit_const_int_for_type(uint32_t blk_id, uint32_t spv_type_id, uint64_t val) {
    const SpirvType* t = mod_.get_type(spv_type_id);
    if (t && t->kind == SpirvTypeKind::Int) {
        if (t->width == 64u) return sb_.emit_const_u64(blk_id, val);
        if (t->is_signed) return sb_.emit_const_s32(blk_id, static_cast<int32_t>(val));
    }
    return sb_.emit_const_u32(blk_id, static_cast<uint32_t>(val));
}

SsaId SpirvLowerer::emit_icmp_eq_const(uint32_t blk_id, const LoweredValue& lhs, uint64_t lit) {
    SsaId rhs = emit_const_int_for_type(blk_id, lhs.spv_type_id, lit);
    return emit_op(blk_id, SsaOp::ICmpEq, ValType::Bool, {lhs.scalar(), rhs});
}

SsaId SpirvLowerer::emit_to_i64(uint32_t blk_id, const LoweredValue& val, bool sign_extend) {
    if (int_type_width(val.spv_type_id) == 64u) return val.scalar();
    return emit_op(blk_id, sign_extend ? SsaOp::SConvert32To64 : SsaOp::UConvert32To64,
                   ValType::U64, {val.scalar()});
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

void SpirvLowerer::lower_switch(
    uint32_t blk_id,
    const SpirvInst& inst,
    const std::unordered_map<uint32_t, uint32_t>& lbl_map) {
    spirv_parser::SpvSwitchView sw = inst;
    LoweredValue sel = get_value(blk_id, sw.selector);
    auto dit = lbl_map.find(sw.default_val);
    int32_t default_blk = (dit != lbl_map.end()) ? static_cast<int32_t>(dit->second) : -1;

    std::vector<std::pair<uint64_t, uint32_t>> cases;
    for (const auto& [lit, lbl] : sw.decode_cases(int_type_width(sel.spv_type_id))) {
        if (auto tit = lbl_map.find(lbl); tit != lbl_map.end()) {
            cases.emplace_back(lit, tit->second);
        }
    }

    if (cases.empty()) {
        if (default_blk >= 0) {
            res_.ssa.blocks[blk_id].is_unconditional_branch = true;
            res_.ssa.blocks[blk_id].branch_target_block = default_blk;
            res_.ssa.blocks[default_blk].preds.push_back(blk_id);
        }
        return;
    }

    uint32_t cur_blk = blk_id;
    for (size_t i = 0; i < cases.size(); ++i) {
        uint64_t lit = cases[i].first;
        uint32_t target_blk = cases[i].second;
        SsaId cmp = emit_icmp_eq_const(cur_blk, sel, lit);
        res_.ssa.blocks[cur_blk].is_conditional_branch = true;
        res_.ssa.blocks[cur_blk].branch_cond = cmp;
        res_.ssa.blocks[cur_blk].branch_target_block = static_cast<int32_t>(target_blk);
        res_.ssa.blocks[target_blk].preds.push_back(cur_blk);

        if (i + 1 < cases.size()) {
            uint32_t next_test_blk = allocate_ssa_block();
            res_.ssa.blocks[cur_blk].fallthrough_block = static_cast<int32_t>(next_test_blk);
            res_.ssa.blocks[next_test_blk].preds.push_back(cur_blk);
            res_.ssa.block_in_token[next_test_blk] = res_.ssa.block_out_token[cur_blk];
            res_.ssa.block_out_token[next_test_blk] = res_.ssa.block_out_token[cur_blk];
            cur_blk = next_test_blk;
        } else if (default_blk >= 0) {
            res_.ssa.blocks[cur_blk].fallthrough_block = default_blk;
            res_.ssa.blocks[default_blk].preds.push_back(cur_blk);
        }
    }
}

bool SpirvLowerer::lower_function_call(uint32_t* io_blk_id, const SpirvInst& inst) {
    spirv_parser::SpvFunctionCallView call = inst;
    uint32_t res_type = call.result_type;
    uint32_t res_id = call.result_id;
    uint32_t callee_id = call.function;

    const SpirvFunction* callee = mod_.get_function(callee_id);
    if (!callee || callee->blocks.empty() || inline_depth_ >= opts_.max_inline_depth) {
        record_unsupported("OpFunctionCall(depth/missing)");
        if (!opts_.tolerant) return false;
        values_[res_id] = make_zero_value(*io_blk_id, res_type);
        return true;
    }

    // Bind callee parameters (both values and pointer arguments) in a scoped snapshot.
    auto saved_values = values_;
    auto saved_pointers = pointers_;
    for (size_t i = 0; i < callee->params.size() && i < call.argument_0.count; ++i) {
        uint32_t param_id = callee->params[i].first;
        uint32_t arg_id = call.argument_0[i];
        if (auto pit = saved_pointers.find(arg_id); pit != saved_pointers.end()) {
            pointers_[param_id] = pit->second;
        } else {
            values_[param_id] = get_value(*io_blk_id, arg_id);
        }
    }

    ++inline_depth_;
    uint32_t exit_blk = *io_blk_id;
    LoweredValue ret_val{};
    bool ok = lower_function_body(*callee, *io_blk_id, /*is_entry_point=*/false, &exit_blk, &ret_val);
    --inline_depth_;

    // Restore caller scope and publish call result
    values_ = std::move(saved_values);
    pointers_ = std::move(saved_pointers);
    if (res_type != 0 && flat_lane_count(res_type) > 0) {
        values_[res_id] = ret_val.empty() ? make_zero_value(exit_blk, res_type) : std::move(ret_val);
    }
    *io_blk_id = exit_blk;
    return ok;
}

bool SpirvLowerer::lower_function_body(
    const SpirvFunction& fn,
    uint32_t entry_blk,
    bool is_entry_point,
    uint32_t* out_exit_blk,
    LoweredValue* out_ret_val) {
    std::unordered_map<uint32_t, const SpirvBasicBlock*> lbl_to_spv;
    std::unordered_map<uint32_t, uint32_t> id_to_width;
    for (const auto& [cid, c] : mod_.constants) {
        id_to_width[cid] = int_type_width(c.type_id);
    }
    for (const auto& [pid, pty] : fn.params) {
        id_to_width[pid] = int_type_width(pty);
    }
    for (const auto& b : fn.blocks) {
        lbl_to_spv[b.label_id] = &b;
        for (const auto& ins : b.insts) {
            spirv_parser::DecodedSpvInst dec{};
            if (spirv_parser::decode_spv_instruction(ins.words.data(), ins.words.size(), &dec) &&
                dec.result_id != 0 && dec.result_type != 0) {
                id_to_width[dec.result_id] = int_type_width(dec.result_type);
            }
        }
    }

    std::set<uint32_t> visited;
    std::vector<const SpirvBasicBlock*> postorder;
    std::function<void(uint32_t)> dfs = [&](uint32_t lbl) {
        if (!visited.insert(lbl).second) return;
        auto it = lbl_to_spv.find(lbl);
        if (it == lbl_to_spv.end()) return;
        for (uint32_t succ : block_succ_labels(*it->second, id_to_width)) {
            dfs(succ);
        }
        postorder.push_back(it->second);
    };
    dfs(fn.blocks[0].label_id);
    std::vector<const SpirvBasicBlock*> rpo(postorder.rbegin(), postorder.rend());

    std::unordered_map<uint32_t, uint32_t> local_lbl_to_blk;
    std::vector<uint32_t> ssa_blks;
    ssa_blks.reserve(rpo.size());
    for (size_t i = 0; i < rpo.size(); ++i) {
        uint32_t bid = (i == 0) ? entry_blk : allocate_ssa_block();
        local_lbl_to_blk[rpo[i]->label_id] = bid;
        ssa_blks.push_back(bid);
    }

    // Wire static CFG edges (except OpSwitch which allocates test blocks dynamically during lowering)
    for (size_t i = 0; i < rpo.size(); ++i) {
        const auto& spv_blk = *rpo[i];
        uint32_t b = ssa_blks[i];
        auto& ssa_blk = res_.ssa.blocks[b];
        for (const auto& inst : spv_blk.insts) {
            const auto op = static_cast<SpvOp>(inst.opcode);
            if (op == SpvOpSelectionMerge) {
                spirv_parser::SpvSelectionMergeView sm = inst;
                if (auto it = local_lbl_to_blk.find(sm.merge_block); it != local_lbl_to_blk.end()) {
                    ssa_blk.selection_merge_block = static_cast<int32_t>(it->second);
                }
            } else if (op == SpvOpLoopMerge) {
                spirv_parser::SpvLoopMergeView lm = inst;
                ssa_blk.is_loop_header = true;
                if (auto it = local_lbl_to_blk.find(lm.merge_block); it != local_lbl_to_blk.end()) {
                    ssa_blk.loop_merge_block = static_cast<int32_t>(it->second);
                }
                if (auto it = local_lbl_to_blk.find(lm.continue_target); it != local_lbl_to_blk.end()) {
                    ssa_blk.loop_backedge_block = static_cast<int32_t>(it->second);
                }
            }
        }
        if (spv_blk.insts.empty()) continue;
        const auto& term = spv_blk.insts.back();
        const auto top = static_cast<SpvOp>(term.opcode);
        if (top == SpvOpBranch) {
            spirv_parser::SpvBranchView br = term;
            if (auto it = local_lbl_to_blk.find(br.target_label); it != local_lbl_to_blk.end()) {
                ssa_blk.is_unconditional_branch = true;
                ssa_blk.branch_target_block = static_cast<int32_t>(it->second);
                res_.ssa.blocks[it->second].preds.push_back(b);
            }
        } else if (top == SpvOpBranchConditional) {
            spirv_parser::SpvBranchConditionalView bc = term;
            auto it_t = local_lbl_to_blk.find(bc.true_label);
            auto it_f = local_lbl_to_blk.find(bc.false_label);
            ssa_blk.is_conditional_branch = true;
            if (it_t != local_lbl_to_blk.end()) {
                ssa_blk.branch_target_block = static_cast<int32_t>(it_t->second);
                res_.ssa.blocks[it_t->second].preds.push_back(b);
            }
            if (it_f != local_lbl_to_blk.end()) {
                ssa_blk.fallthrough_block = static_cast<int32_t>(it_f->second);
                if (it_t == local_lbl_to_blk.end() || it_f->second != it_t->second) {
                    res_.ssa.blocks[it_f->second].preds.push_back(b);
                }
            }
        } else if (top == SpvOpReturn || top == SpvOpReturnValue) {
            if (is_entry_point) ssa_blk.is_terminated_by_end = true;
        } else if (top == SpvOpKill || top == SpvOpTerminateInvocation || top == SpvOpUnreachable) {
            ssa_blk.is_terminated_by_discard = (top == SpvOpKill || top == SpvOpTerminateInvocation);
            ssa_blk.is_terminated_by_end = (top == SpvOpUnreachable);
        }
    }

    std::vector<PendingSpvPhi> local_phis;
    std::vector<std::pair<uint32_t, LoweredValue>> return_sites;

    for (size_t i = 0; i < rpo.size(); ++i) {
        uint32_t cur_b = ssa_blks[i];
        if (i > 0) {
            const auto& blk = res_.ssa.blocks[cur_b];
            if (blk.preds.size() == 1 && blk.preds[0] < cur_b) {
                res_.ssa.block_in_token[cur_b] = res_.ssa.block_out_token[blk.preds[0]];
                res_.ssa.block_out_token[cur_b] = res_.ssa.block_out_token[blk.preds[0]];
            } else {
                SsaNode tphi{};
                tphi.op = SsaOp::Phi;
                tphi.type = ValType::Token;
                tphi.aux_bool = true;
                SsaId tpid = sb_.emit(cur_b, std::move(tphi));
                res_.ssa.block_in_token[cur_b] = tpid;
                res_.ssa.block_out_token[cur_b] = tpid;
            }
        }

        const auto& spv_blk = *rpo[i];
        for (const auto& inst : spv_blk.insts) {
            const auto op = static_cast<SpvOp>(inst.opcode);
            switch (op) {
                case SpvOpNop:
                case SpvOpSelectionMerge:
                case SpvOpLoopMerge:
                case SpvOpMemoryBarrier:
                case SpvOpLine:
                case SpvOpNoLine:
                case SpvOpBranch:
                case SpvOpUnreachable:
                    break;

                case SpvOpVariable: {
                    spirv_parser::SpvVariableView var = inst;
                    register_local_variable(cur_b, var.result_id, var.result_type,
                                            var.has_initializer ? var.initializer : 0u);
                    break;
                }

                case SpvOpPhi: {
                    spirv_parser::SpvPhiView pv = inst;
                    uint32_t nlanes = flat_lane_count(pv.result_type);
                    ValType lane_vt = to_val_type(scalar_elem_type(pv.result_type));
                    PendingSpvPhi p{};
                    p.blk_id = cur_b;
                    p.result_id = pv.result_id;
                    p.type_id = pv.result_type;
                    LoweredValue lv{};
                    lv.spv_type_id = pv.result_type;
                    for (uint32_t l = 0; l < nlanes; ++l) {
                        SsaNode phi{};
                        phi.op = SsaOp::Phi;
                        phi.type = lane_vt;
                        SsaId pid = sb_.emit(cur_b, std::move(phi));
                        p.lane_phi_ids.push_back(pid);
                        lv.lanes.push_back(pid);
                    }
                    for (size_t k = 0; k < pv.incoming_count(); ++k) {
                        p.raw_incomings.push_back({pv.incoming_value(k), pv.incoming_parent(k)});
                    }
                    values_[pv.result_id] = std::move(lv);
                    local_phis.push_back(std::move(p));
                    break;
                }

                case SpvOpAccessChain:
                case SpvOpInBoundsAccessChain:
                case SpvOpPtrAccessChain:
                case SpvOpInBoundsPtrAccessChain: {
                    spirv_parser::DecodedSpvInst dec{};
                    if (spirv_parser::decode_spv_instruction(inst.words.data(), inst.words.size(), &dec) &&
                        dec.result_id != 0) {
                        pointers_[dec.result_id] = lower_access_chain(cur_b, inst);
                    }
                    break;
                }

                case SpvOpLoad: {
                    spirv_parser::SpvLoadView ld = inst;
                    values_[ld.result_id] = lower_load(cur_b, inst);
                    break;
                }

                case SpvOpStore:
                    lower_store(cur_b, inst);
                    break;

                case SpvOpCopyMemory:
                    lower_copy_memory(cur_b, inst);
                    break;

                case SpvOpAtomicStore:
                case SpvOpAtomicFlagClear:
                    lower_atomic_store(cur_b, inst);
                    break;

                case SpvOpFunctionCall:
                    if (!lower_function_call(&cur_b, inst)) return false;
                    break;

                case SpvOpImageSampleImplicitLod:
                case SpvOpImageSampleExplicitLod:
                case SpvOpImageSampleDrefImplicitLod:
                case SpvOpImageSampleDrefExplicitLod:
                case SpvOpImageSampleProjImplicitLod:
                case SpvOpImageSampleProjExplicitLod:
                case SpvOpImageFetch:
                case SpvOpImageGather:
                case SpvOpImageDrefGather:
                case SpvOpImageRead: {
                    spirv_parser::DecodedSpvInst dec{};
                    if (spirv_parser::decode_spv_instruction(inst.words.data(), inst.words.size(), &dec) &&
                        dec.result_id != 0) {
                        values_[dec.result_id] = lower_image_sample(cur_b, inst);
                    }
                    break;
                }

                case SpvOpImageWrite:
                    lower_image_write(cur_b, inst);
                    break;

                case SpvOpControlBarrier: {
                    SsaNode cb{};
                    cb.op = SsaOp::ControlBarrier;
                    cb.type = ValType::Token;
                    cb.args = {sb_.current_token(cur_b)};
                    cb.is_barrier_or_cf = true;
                    cb.is_side_effect = true;
                    sb_.set_token(cur_b, sb_.emit(cur_b, std::move(cb)));
                    break;
                }

                case SpvOpExtInst:
                    if (!lower_glsl450_ext_inst(cur_b, inst)) {
                        res_.error_message = "Unsupported GLSL.std.450 instruction";
                        return false;
                    }
                    break;

                case SpvOpBranchConditional: {
                    spirv_parser::SpvBranchConditionalView bc = inst;
                    LoweredValue cond = get_value(cur_b, bc.condition);
                    res_.ssa.blocks[cur_b].branch_cond = cond.scalar();
                    break;
                }

                case SpvOpSwitch:
                    lower_switch(cur_b, inst, local_lbl_to_blk);
                    break;

                case SpvOpKill:
                case SpvOpTerminateInvocation:
                case SpvOpDemoteToHelperInvocation: {
                    SsaNode k{};
                    k.op = SsaOp::Kill;
                    k.type = ValType::Void;
                    k.aux_bool = true;
                    k.aux_bool2 = (op == SpvOpDemoteToHelperInvocation);
                    k.args = {sb_.emit_const_bool(cur_b, true), sb_.current_token(cur_b)};
                    k.is_barrier_or_cf = true;
                    k.is_side_effect = true;
                    sb_.emit(cur_b, std::move(k));
                    break;
                }

                case SpvOpReturn:
                    if (is_entry_point) {
                        flush_stage_outputs(cur_b);
                    } else {
                        return_sites.push_back({cur_b, LoweredValue{}});
                    }
                    break;

                case SpvOpReturnValue: {
                    if (is_entry_point) {
                        flush_stage_outputs(cur_b);
                    } else {
                        spirv_parser::SpvReturnValueView rv = inst;
                        return_sites.push_back({cur_b, get_value(cur_b, rv.value)});
                    }
                    break;
                }

                default:
                    if (is_atomic_opcode(op)) {
                        spirv_parser::DecodedSpvInst dec{};
                        if (spirv_parser::decode_spv_instruction(inst.words.data(), inst.words.size(), &dec) &&
                            dec.result_id != 0) {
                            values_[dec.result_id] = lower_atomic(cur_b, inst);
                        }
                    } else if (!lower_alu_or_misc_inst(cur_b, inst)) {
                        res_.error_message = std::string("Unsupported SPIR-V opcode ") +
                                             spirv_parser::spv_op_name(inst.opcode);
                        return false;
                    }
                    break;
            }
        }
    }

    for (const auto& p : local_phis) {
        for (uint32_t lane = 0; lane < p.lane_phi_ids.size(); ++lane) {
            auto& phi_node = res_.ssa.get_mut(p.lane_phi_ids[lane]);
            phi_node.phi_incomings.clear();
            for (const auto& [val_id, pred_lbl] : p.raw_incomings) {
                auto bit = local_lbl_to_blk.find(pred_lbl);
                if (bit == local_lbl_to_blk.end()) continue;
                LoweredValue lv = get_value(bit->second, val_id);
                SsaId sid = (lane < lv.lanes.size()) ? lv.lanes[lane] : sb_.emit_const_u32(bit->second, 0u);
                phi_node.phi_incomings.push_back({bit->second, sid});
            }
        }
    }

    if (!is_entry_point && out_exit_blk) {
        if (return_sites.size() == 1) {
            *out_exit_blk = return_sites[0].first;
            if (out_ret_val) *out_ret_val = return_sites[0].second;
        } else if (return_sites.size() > 1) {
            uint32_t merge_b = allocate_ssa_block();
            for (const auto& [rb, _] : return_sites) {
                res_.ssa.blocks[rb].is_unconditional_branch = true;
                res_.ssa.blocks[rb].branch_target_block = static_cast<int32_t>(merge_b);
                res_.ssa.blocks[merge_b].preds.push_back(rb);
            }
            SsaNode tphi{};
            tphi.op = SsaOp::Phi;
            tphi.type = ValType::Token;
            tphi.aux_bool = true;
            for (const auto& [rb, _] : return_sites) {
                tphi.phi_incomings.push_back({rb, res_.ssa.block_out_token[rb]});
            }
            SsaId tpid = sb_.emit(merge_b, std::move(tphi));
            res_.ssa.block_in_token[merge_b] = tpid;
            res_.ssa.block_out_token[merge_b] = tpid;

            if (out_ret_val && !return_sites[0].second.empty()) {
                LoweredValue merged{};
                merged.spv_type_id = return_sites[0].second.spv_type_id;
                uint32_t nl = static_cast<uint32_t>(return_sites[0].second.lanes.size());
                for (uint32_t l = 0; l < nl; ++l) {
                    SsaNode rphi{};
                    rphi.op = SsaOp::Phi;
                    rphi.type = res_.ssa.get(return_sites[0].second.lanes[l]).type;
                    for (const auto& [rb, rv] : return_sites) {
                        rphi.phi_incomings.push_back({rb, (l < rv.lanes.size()) ? rv.lanes[l] : sb_.emit_const_u32(rb, 0u)});
                    }
                    merged.lanes.push_back(sb_.emit(merge_b, std::move(rphi)));
                }
                *out_ret_val = std::move(merged);
            }
            *out_exit_blk = merge_b;
        }
    }
    return true;
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

    uint32_t entry_blk = allocate_ssa_block();
    SsaNode init_tok{};
    init_tok.op = SsaOp::Nop;
    init_tok.type = ValType::Token;
    SsaId tok0 = sb_.emit(entry_blk, std::move(init_tok));
    res_.ssa.block_in_token[entry_blk] = tok0;
    res_.ssa.block_out_token[entry_blk] = tok0;
    init_global_resources(entry_blk);

    if (!lower_function_body(*entry_fn, entry_blk, /*is_entry_point=*/true, nullptr, nullptr)) {
        return false;
    }

    for (uint32_t b = 1; b < res_.ssa.blocks.size(); ++b) {
        const auto& blk = res_.ssa.blocks[b];
        SsaId tpid = res_.ssa.block_in_token[b];
        if (tpid != compiler::kInvalidSsa && res_.ssa.get(tpid).op == SsaOp::Phi &&
            res_.ssa.get(tpid).aux_bool && res_.ssa.get(tpid).phi_incomings.empty()) {
            auto& tphi = res_.ssa.get_mut(tpid);
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
    if (res.ok && opts.run_ssa_opt) {
        dataflow::SsaOptPipelineOptions popts;
        popts.fp_mode = opts.ssa_opt_fp_fast_math ? dataflow::FpOptMode::FastMath
                                                  : dataflow::FpOptMode::StrictIEEE;
        popts.enable_fp_tile_rewrites = opts.ssa_opt_fp_tile_rewrites;
        popts.enable_fp_cpp_rewrites = opts.ssa_opt_fp_cpp_rewrites;
        dataflow::run_ssa_opt_pipeline(res.ssa, popts);
    }
    return res;
}

} // namespace valhall::spirv2ssa

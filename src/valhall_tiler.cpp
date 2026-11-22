#include "valhall_tiler.h"
#include <algorithm>
#include <limits>
#include <map>
#include <unordered_map>
#include <unordered_set>

namespace valhall::compiler {

using namespace valhall::spirv;
using tables::SsaOp;
using tables::TileEmitKind;

namespace {

struct DpChoice {
    bool is_fusion = false;
    uint32_t cost = 1000000u;
    const tables::FusionTileSpec* fusion = nullptr;
    const tables::CanonicalTileSpec* canon = nullptr;
    std::vector<SsaId> leaf_args; // Captured leaf SsaIds in leaf_slot order
    tables::EffectMask effects{};
};

static bool match_fusion_pattern(
    const ValhallSsaModule& mod,
    const tables::FusionPatternNode* pnodes,
    uint16_t pnode_idx,
    SsaId ssa_id,
    uint32_t root_block_id,
    bool is_root,
    std::vector<SsaId>& captured_leaves,
    std::unordered_map<SsaId, uint32_t>& internal_uses) {
    if (ssa_id == kInvalidSsa) return false;
    const auto& pn = pnodes[pnode_idx];
    if (pn.is_leaf) {
        size_t slot = static_cast<size_t>(pn.leaf_slot);
        if (slot >= captured_leaves.size()) captured_leaves.resize(slot + 1, kInvalidSsa);
        if (captured_leaves[slot] != kInvalidSsa && captured_leaves[slot] != ssa_id) {
            return false; // Non-linear pattern variable mismatch
        }
        captured_leaves[slot] = ssa_id;
        return true;
    }

    const auto& sn = mod.get(ssa_id);
    if (sn.op != pn.ssa_op) return false;
    if (sn.block_id != root_block_id) return false;
    if (!is_root) {
        // Never fuse across barrier/CF/side-effecting interior nodes
        if (sn.is_barrier_or_cf || sn.is_side_effect) return false;
        internal_uses[ssa_id]++;
    }
    if (pn.has_imm && sn.imm_u32 != pn.imm_val) return false;
    if (pn.has_imm_offset && sn.imm_u32 != pn.imm_offset) return false;
    if (pn.has_imm_width && sn.imm_u32_2 != pn.imm_width) return false;
    if (sn.args.size() < pn.num_children) return false;

    if (pn.same_type_as_inner && pn.num_children == 1) {
        SsaId inner = sn.args[0];
        if (inner == kInvalidSsa) return false;
        const auto& inn = mod.get(inner);
        if (inn.args.empty()) return false;
        if (sn.type != mod.get(inn.args[0]).type) return false;
    }

    for (uint8_t c = 0; c < pn.num_children; ++c) {
        if (pn.children[c] < 0) return false;
        if (!match_fusion_pattern(mod, pnodes, static_cast<uint16_t>(pn.children[c]),
                                  sn.args[c], root_block_id, false,
                                  captured_leaves, internal_uses)) {
            return false;
        }
    }
    return true;
}

struct EmitContext {
    SpirvBuilder b;
    const ValhallSsaModule& mod;
    const ValhallTranslateOptions& opts;
    const std::vector<DpChoice>& choices;
    const std::vector<bool>& is_live;
    std::vector<uint32_t> spv_ids;

    uint32_t sideband_struct_t = 0;
    uint32_t sideband_var_id = 0;
    std::map<uint32_t, uint32_t> vert_attr_inputs_;
    std::map<uint32_t, uint32_t> vert_varying_outputs_;
    std::map<uint32_t, uint32_t> frag_varying_inputs_;
    std::map<uint32_t, uint32_t> frag_color_outputs_;
    std::map<uint32_t, uint32_t> sampled_image_vars_;
    std::map<uint32_t, uint32_t> sampled_image_types_;

    EmitContext(ExecutionModel model, const ValhallSsaModule& m, const ValhallTranslateOptions& o,
                const std::vector<DpChoice>& ch, const std::vector<bool>& live)
        : b(model, true), mod(m), opts(o), choices(ch), is_live(live), spv_ids(m.nodes.size(), 0u) {}

    uint32_t to_spv_type(ValType vt) {
        switch (vt) {
            case ValType::Void: return b.type_void();
            case ValType::Bool: return b.type_bool();
            case ValType::U32: return b.type_u32();
            case ValType::S32: return b.type_s32();
            case ValType::F32: return b.type_f32();
            case ValType::U64: return b.type_u64();
            case ValType::S64: return b.type_s64();
            case ValType::V2F32: return b.type_vector(b.type_f32(), 2);
            case ValType::V2U32: return b.type_vector(b.type_u32(), 2);
            case ValType::V2S32: return b.type_vector(b.type_s32(), 2);
            case ValType::V4F32: return b.type_vector(b.type_f32(), 4);
            case ValType::V4U32: return b.type_vector(b.type_u32(), 4);
            case ValType::V4S32: return b.type_vector(b.type_s32(), 4);
            case ValType::V3U32: return b.type_vector(b.type_u32(), 3);
            case ValType::V2Bool: return b.type_vector(b.type_bool(), 2);
            default: return b.type_u32();
        }
    }

    void declare_sideband() {
        uint32_t arr_fau_t = b.type_array(b.type_u32(), b.const_u32(64), 4);
        uint32_t arr_blend_t = b.type_array(b.type_u64(), b.const_u32(4), 8);
        uint32_t v4u32_t = b.type_vector(b.type_u32(), 4);
        sideband_struct_t = b.type_struct(
            { arr_fau_t, b.type_u64(), b.type_u64(), b.type_u64(), arr_blend_t, v4u32_t, v4u32_t },
            "ValhallSidebandBuffer");
        b.decorate(sideband_struct_t, Decoration::Block);
        b.member_decorate(sideband_struct_t, 0, Decoration::Offset, { 0 });
        b.member_decorate(sideband_struct_t, 1, Decoration::Offset, { 256 });
        b.member_decorate(sideband_struct_t, 2, Decoration::Offset, { 264 });
        b.member_decorate(sideband_struct_t, 3, Decoration::Offset, { 272 });
        b.member_decorate(sideband_struct_t, 4, Decoration::Offset, { 280 });
        b.member_decorate(sideband_struct_t, 5, Decoration::Offset, { 312 });
        b.member_decorate(sideband_struct_t, 6, Decoration::Offset, { 328 });

        uint32_t sb_ptr_t = b.type_pointer(StorageClass::StorageBuffer, sideband_struct_t);
        std::string sb_name = opts.symbol_bindings.sideband_var_name.empty() ? "u_sideband" : opts.symbol_bindings.sideband_var_name;
        sideband_var_id = b.add_global_var(sb_ptr_t, StorageClass::StorageBuffer, sb_name, false);
        b.decorate(sideband_var_id, Decoration::DescriptorSet, { opts.sideband_set });
        b.decorate(sideband_var_id, Decoration::Binding, { opts.sideband_binding });
        b.decorate(sideband_var_id, Decoration::NonWritable);
    }

    uint32_t get_or_create_vert_attr(uint32_t loc) {
        auto it = vert_attr_inputs_.find(loc);
        if (it != vert_attr_inputs_.end()) return it->second;
        uint32_t v4f = b.type_vector(b.type_f32(), 4);
        uint32_t ptr_t = b.type_pointer(StorageClass::Input, v4f);
        auto nit = opts.symbol_bindings.vertex_input_names.find(loc);
        std::string vname = (nit != opts.symbol_bindings.vertex_input_names.end() && !nit->second.empty()) ? nit->second : ("in_attr_" + std::to_string(loc));
        uint32_t var_id = b.add_global_var(ptr_t, StorageClass::Input, vname, true);
        b.decorate(var_id, Decoration::Location, { loc });
        vert_attr_inputs_[loc] = var_id;
        return var_id;
    }

    uint32_t get_or_create_vert_varying(uint32_t loc) {
        auto it = vert_varying_outputs_.find(loc);
        if (it != vert_varying_outputs_.end()) return it->second;
        uint32_t v4f = b.type_vector(b.type_f32(), 4);
        uint32_t ptr_t = b.type_pointer(StorageClass::Output, v4f);
        uint32_t var_id = b.add_global_var(ptr_t, StorageClass::Output, "out_var_" + std::to_string(loc), true);
        b.decorate(var_id, Decoration::Location, { loc });
        vert_varying_outputs_[loc] = var_id;
        return var_id;
    }

    uint32_t get_or_create_frag_varying(uint32_t loc, bool flat) {
        auto it = frag_varying_inputs_.find(loc);
        if (it != frag_varying_inputs_.end()) return it->second;
        uint32_t v4f = b.type_vector(b.type_f32(), 4);
        uint32_t ptr_t = b.type_pointer(StorageClass::Input, v4f);
        auto nit = opts.symbol_bindings.varying_input_names.find(loc);
        std::string vname = (nit != opts.symbol_bindings.varying_input_names.end() && !nit->second.empty()) ? nit->second : ("in_var_" + std::to_string(loc));
        uint32_t var_id = b.add_global_var(ptr_t, StorageClass::Input, vname, true);
        b.decorate(var_id, Decoration::Location, { loc });
        if (flat) b.decorate(var_id, Decoration::Flat);
        frag_varying_inputs_[loc] = var_id;
        return var_id;
    }

    uint32_t get_or_create_frag_color_output(uint32_t loc) {
        auto it = frag_color_outputs_.find(loc);
        if (it != frag_color_outputs_.end()) return it->second;
        uint32_t v4f = b.type_vector(b.type_f32(), 4);
        uint32_t ptr_t = b.type_pointer(StorageClass::Output, v4f);
        auto nit = opts.symbol_bindings.color_output_names.find(loc);
        std::string vname = (nit != opts.symbol_bindings.color_output_names.end() && !nit->second.empty()) ? nit->second : ("out_color_" + std::to_string(loc));
        uint32_t var_id = b.add_global_var(ptr_t, StorageClass::Output, vname, true);
        b.decorate(var_id, Decoration::Location, { loc });
        frag_color_outputs_[loc] = var_id;
        return var_id;
    }

    uint32_t get_or_create_sampled_image(uint32_t idx, uint32_t* out_si_type) {
        auto it = sampled_image_vars_.find(idx);
        if (it != sampled_image_vars_.end()) {
            if (out_si_type) *out_si_type = sampled_image_types_[idx];
            return it->second;
        }
        uint32_t d_set = 0, binding = 1 + idx;
        Dim dim = Dim::Dim2D;
        bool arrayed = false, shadow = false;
        for (const auto& s : opts.samplers) {
            if (s.tex_index == idx) {
                d_set = s.descriptor_set; binding = s.binding;
                dim = s.dim; arrayed = s.arrayed; shadow = s.shadow;
                break;
            }
        }
        uint32_t img_t = b.type_image(b.type_f32(), dim, shadow ? 1u : 0u, arrayed ? 1u : 0u, 0u, 1u, 0u);
        uint32_t si_t = b.type_sampled_image(img_t);
        uint32_t ptr_t = b.type_pointer(StorageClass::UniformConstant, si_t);
        uint32_t var_id = b.add_global_var(ptr_t, StorageClass::UniformConstant, "u_sampler_" + std::to_string(idx), false);
        b.decorate(var_id, Decoration::DescriptorSet, { d_set });
        b.decorate(var_id, Decoration::Binding, { binding });
        sampled_image_vars_[idx] = var_id;
        sampled_image_types_[idx] = si_t;
        if (out_si_type) *out_si_type = si_t;
        return var_id;
    }

    uint32_t get_spv(SsaId id) {
        if (id == kInvalidSsa) return 0;
        if (spv_ids[id] != 0) return spv_ids[id];
        return emit_node(id);
    }

    uint32_t emit_node(SsaId id) {
        if (spv_ids[id] != 0) return spv_ids[id];
        const auto& n = mod.get(id);
        const auto& ch = choices[id];

        if (ch.is_fusion) {
            TileEmitKind ek = ch.fusion->emit_kind;
            switch (ek) {
                case TileEmitKind::ForwardArg0:
                case TileEmitKind::ForwardInnerArg0: {
                    uint32_t v = get_spv(ch.leaf_args[0]);
                    spv_ids[id] = v;
                    return v;
                }
                case TileEmitKind::ForwardArg1: {
                    uint32_t v = get_spv(ch.leaf_args[1]);
                    spv_ids[id] = v;
                    return v;
                }
                case TileEmitKind::ForwardArg0_Lo: {
                    uint32_t v = get_spv(ch.leaf_args[0]);
                    spv_ids[id] = v;
                    return v;
                }
                case TileEmitKind::ForwardArg0_Hi: {
                    uint32_t v = get_spv(ch.leaf_args[1]);
                    spv_ids[id] = v;
                    return v;
                }
                case TileEmitKind::GlslFma: {
                    uint32_t rt = to_spv_type(n.type);
                    uint32_t a = get_spv(ch.leaf_args[0]);
                    uint32_t b_v = get_spv(ch.leaf_args[1]);
                    uint32_t c = get_spv(ch.leaf_args[2]);
                    spv_ids[id] = b.emit_glsl_ext(GlslStd450::Fma, rt, { a, b_v, c });
                    return spv_ids[id];
                }
                case TileEmitKind::GlslFmaCommuted: {
                    uint32_t rt = to_spv_type(n.type);
                    uint32_t c = get_spv(ch.leaf_args[0]);
                    uint32_t a = get_spv(ch.leaf_args[1]);
                    uint32_t b_v = get_spv(ch.leaf_args[2]);
                    spv_ids[id] = b.emit_glsl_ext(GlslStd450::Fma, rt, { a, b_v, c });
                    return spv_ids[id];
                }
                case TileEmitKind::OpFSub: {
                    uint32_t rt = to_spv_type(n.type);
                    uint32_t a = get_spv(ch.leaf_args[0]);
                    uint32_t b_v = get_spv(ch.leaf_args[1]);
                    spv_ids[id] = b.op_FSub(rt, a, b_v);
                    return spv_ids[id];
                }
                case TileEmitKind::GlslFClamp: {
                    uint32_t rt = to_spv_type(n.type);
                    uint32_t x = get_spv(ch.leaf_args[0]);
                    uint32_t lo = get_spv(ch.leaf_args[1]);
                    uint32_t hi = get_spv(ch.leaf_args[2]);
                    spv_ids[id] = b.emit_glsl_ext(GlslStd450::FClamp, rt, { x, lo, hi });
                    return spv_ids[id];
                }
                case TileEmitKind::GlslFAbs: {
                    uint32_t rt = to_spv_type(n.type);
                    spv_ids[id] = b.emit_glsl_ext(GlslStd450::FAbs, rt, { get_spv(ch.leaf_args[0]) });
                    return spv_ids[id];
                }
                case TileEmitKind::OpFNegate: {
                    uint32_t rt = to_spv_type(n.type);
                    spv_ids[id] = b.op_FNegate(rt, get_spv(ch.leaf_args[0]));
                    return spv_ids[id];
                }
                case TileEmitKind::OpFMul: {
                    uint32_t rt = to_spv_type(n.type);
                    spv_ids[id] = b.op_FMul(rt, get_spv(ch.leaf_args[0]), get_spv(ch.leaf_args[1]));
                    return spv_ids[id];
                }
                case TileEmitKind::OpFAdd: {
                    uint32_t rt = to_spv_type(n.type);
                    spv_ids[id] = b.op_FAdd(rt, get_spv(ch.leaf_args[0]), get_spv(ch.leaf_args[1]));
                    return spv_ids[id];
                }
                case TileEmitKind::OpFDiv: {
                    uint32_t rt = to_spv_type(n.type);
                    spv_ids[id] = b.op_FDiv(rt, get_spv(ch.leaf_args[0]), get_spv(ch.leaf_args[1]));
                    return spv_ids[id];
                }
                case TileEmitKind::OpFDivSwapped: {
                    uint32_t rt = to_spv_type(n.type);
                    spv_ids[id] = b.op_FDiv(rt, get_spv(ch.leaf_args[1]), get_spv(ch.leaf_args[0]));
                    return spv_ids[id];
                }
                case TileEmitKind::GlslSqrt: {
                    uint32_t rt = to_spv_type(n.type);
                    spv_ids[id] = b.emit_glsl_ext(GlslStd450::Sqrt, rt, { get_spv(ch.leaf_args[0]) });
                    return spv_ids[id];
                }
                case TileEmitKind::GlslPow: {
                    uint32_t rt = to_spv_type(n.type);
                    spv_ids[id] = b.emit_glsl_ext(GlslStd450::Pow, rt, { get_spv(ch.leaf_args[0]), get_spv(ch.leaf_args[1]) });
                    return spv_ids[id];
                }
                case TileEmitKind::GlslPowSwapped: {
                    uint32_t rt = to_spv_type(n.type);
                    spv_ids[id] = b.emit_glsl_ext(GlslStd450::Pow, rt, { get_spv(ch.leaf_args[1]), get_spv(ch.leaf_args[0]) });
                    return spv_ids[id];
                }
                default:
                    break;
            }
        }

        // Canonical emission
        uint32_t rt = to_spv_type(n.type);
        switch (n.op) {
            case SsaOp::Nop:
                spv_ids[id] = 0;
                return 0;
            case SsaOp::Copy:
                spv_ids[id] = get_spv(n.args[0]);
                return spv_ids[id];
            case SsaOp::ConstU32:
                spv_ids[id] = b.const_u32(n.imm_u32);
                return spv_ids[id];
            case SsaOp::ConstS32:
                spv_ids[id] = b.const_s32(static_cast<int32_t>(n.imm_u32));
                return spv_ids[id];
            case SsaOp::ConstF32:
                spv_ids[id] = b.const_f32_bits(n.imm_u32);
                return spv_ids[id];
            case SsaOp::ConstU64:
                spv_ids[id] = b.const_u64(n.imm_u64);
                return spv_ids[id];
            case SsaOp::ConstBool:
                spv_ids[id] = b.const_bool(n.imm_u32 != 0);
                return spv_ids[id];

            case SsaOp::LoadFau: {
                uint32_t ptr_u32 = b.type_pointer(StorageClass::StorageBuffer, b.type_u32());
                uint32_t word_idx = n.imm_u32 & 63u;
                uint32_t ac = b.op_AccessChain(ptr_u32, sideband_var_id, { b.const_u32(0), b.const_u32(word_idx) });
                spv_ids[id] = b.emit_load(b.type_u32(), ac);
                auto fit = opts.symbol_bindings.fau_word_names.find(word_idx * 4u);
                if (fit != opts.symbol_bindings.fau_word_names.end() && !fit->second.empty()) {
                    b.set_name(spv_ids[id], fit->second);
                }
                return spv_ids[id];
            }
            case SsaOp::LoadSidebandU64: {
                uint32_t ptr_u64 = b.type_pointer(StorageClass::StorageBuffer, b.type_u64());
                uint32_t ac = b.op_AccessChain(ptr_u64, sideband_var_id, { b.const_u32(n.imm_u32) });
                spv_ids[id] = b.emit_load(b.type_u64(), ac);
                return spv_ids[id];
            }
            case SsaOp::LoadSidebandU32: {
                uint32_t ptr_u32 = b.type_pointer(StorageClass::StorageBuffer, b.type_u32());
                uint32_t ac = b.op_AccessChain(ptr_u32, sideband_var_id, { b.const_u32(n.imm_u32), b.const_u32(n.imm_u32_2) });
                spv_ids[id] = b.emit_load(b.type_u32(), ac);
                return spv_ids[id];
            }
            case SsaOp::LoadBuiltinInput: {
                uint32_t var_id = b.get_or_create_builtin_input(static_cast<BuiltIn>(n.imm_u32), rt, n.aux_str, n.aux_bool);
                spv_ids[id] = b.emit_load(rt, var_id);
                return spv_ids[id];
            }
            case SsaOp::LoadVertAttr: {
                uint32_t var_id = get_or_create_vert_attr(n.imm_u32);
                spv_ids[id] = b.emit_load(rt, var_id);
                return spv_ids[id];
            }
            case SsaOp::LoadFragVarying: {
                uint32_t var_id = get_or_create_frag_varying(n.imm_u32, n.aux_bool);
                spv_ids[id] = b.emit_load(rt, var_id);
                return spv_ids[id];
            }

            case SsaOp::Pack64:
            case SsaOp::ImageTexelAddr: {
                uint32_t v2u = b.type_vector(b.type_u32(), 2);
                uint32_t vec = b.op_CompositeConstruct(v2u, { get_spv(n.args[0]), get_spv(n.args[1]) });
                spv_ids[id] = b.op_Bitcast(b.type_u64(), vec);
                return spv_ids[id];
            }
            case SsaOp::Split64: {
                uint32_t v2u = b.type_vector(b.type_u32(), 2);
                spv_ids[id] = b.op_Bitcast(v2u, get_spv(n.args[0]));
                return spv_ids[id];
            }
            case SsaOp::TupleExtract: {
                SsaId tup_id = n.args[0];
                const auto& tup_n = mod.get(tup_id);
                if (tup_n.op == SsaOp::Split64) {
                    uint32_t vec = get_spv(tup_id);
                    spv_ids[id] = b.op_CompositeExtract(b.type_u32(), vec, { n.imm_u32 });
                } else {
                    // TupleValToken: index 0 is value, index 1 is token
                    uint32_t v = get_spv(tup_id);
                    spv_ids[id] = (n.imm_u32 == 0) ? v : 0u;
                }
                return spv_ids[id];
            }
            case SsaOp::Bfi:
                spv_ids[id] = b.op_BitFieldInsert(rt, get_spv(n.args[0]), get_spv(n.args[1]), get_spv(n.args[2]), get_spv(n.args[3]));
                return spv_ids[id];
            case SsaOp::BfeU:
                spv_ids[id] = b.op_BitFieldUExtract(rt, get_spv(n.args[0]), get_spv(n.args[1]), get_spv(n.args[2]));
                return spv_ids[id];
            case SsaOp::BfeS:
                spv_ids[id] = b.op_BitFieldSExtract(rt, get_spv(n.args[0]), get_spv(n.args[1]), get_spv(n.args[2]));
                return spv_ids[id];
            case SsaOp::UnpackHalf2x16:
                spv_ids[id] = b.emit_glsl_ext(GlslStd450::UnpackHalf2x16, rt, { get_spv(n.args[0]) });
                return spv_ids[id];
            case SsaOp::PackHalf2x16:
                spv_ids[id] = b.emit_glsl_ext(GlslStd450::PackHalf2x16, rt, { get_spv(n.args[0]) });
                return spv_ids[id];
            case SsaOp::Bitcast:
                spv_ids[id] = b.op_Bitcast(rt, get_spv(n.args[0]));
                return spv_ids[id];
            case SsaOp::VecConstruct2:
                spv_ids[id] = b.op_CompositeConstruct(rt, { get_spv(n.args[0]), get_spv(n.args[1]) });
                return spv_ids[id];
            case SsaOp::VecConstruct4:
                spv_ids[id] = b.op_CompositeConstruct(rt, { get_spv(n.args[0]), get_spv(n.args[1]), get_spv(n.args[2]), get_spv(n.args[3]) });
                return spv_ids[id];
            case SsaOp::VecExtract:
                spv_ids[id] = b.op_CompositeExtract(rt, get_spv(n.args[0]), { n.imm_u32 });
                return spv_ids[id];
            case SsaOp::VecShuffle:
                spv_ids[id] = b.op_VectorShuffle(rt, get_spv(n.args[0]), get_spv(n.args[1]), { n.imm_u32, n.imm_u32_2 });
                return spv_ids[id];

            case SsaOp::FAdd: spv_ids[id] = b.op_FAdd(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::FSub: spv_ids[id] = b.op_FSub(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::FMul: spv_ids[id] = b.op_FMul(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::FDiv: spv_ids[id] = b.op_FDiv(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::FNeg: spv_ids[id] = b.op_FNegate(rt, get_spv(n.args[0])); return spv_ids[id];
            case SsaOp::FAbs: spv_ids[id] = b.emit_glsl_ext(GlslStd450::FAbs, rt, { get_spv(n.args[0]) }); return spv_ids[id];
            case SsaOp::Fma:  spv_ids[id] = b.emit_glsl_ext(GlslStd450::Fma, rt, { get_spv(n.args[0]), get_spv(n.args[1]), get_spv(n.args[2]) }); return spv_ids[id];
            case SsaOp::FMin: spv_ids[id] = b.emit_glsl_ext(GlslStd450::FMin, rt, { get_spv(n.args[0]), get_spv(n.args[1]) }); return spv_ids[id];
            case SsaOp::FMax: spv_ids[id] = b.emit_glsl_ext(GlslStd450::FMax, rt, { get_spv(n.args[0]), get_spv(n.args[1]) }); return spv_ids[id];
            case SsaOp::NMin: spv_ids[id] = b.emit_glsl_ext(GlslStd450::NMin, rt, { get_spv(n.args[0]), get_spv(n.args[1]) }); return spv_ids[id];
            case SsaOp::NMax: spv_ids[id] = b.emit_glsl_ext(GlslStd450::NMax, rt, { get_spv(n.args[0]), get_spv(n.args[1]) }); return spv_ids[id];
            case SsaOp::FClamp: spv_ids[id] = b.emit_glsl_ext(GlslStd450::FClamp, rt, { get_spv(n.args[0]), get_spv(n.args[1]), get_spv(n.args[2]) }); return spv_ids[id];
            case SsaOp::RoundEven: spv_ids[id] = b.emit_glsl_ext(GlslStd450::RoundEven, rt, { get_spv(n.args[0]) }); return spv_ids[id];
            case SsaOp::Floor: spv_ids[id] = b.emit_glsl_ext(GlslStd450::Floor, rt, { get_spv(n.args[0]) }); return spv_ids[id];
            case SsaOp::Ceil: spv_ids[id] = b.emit_glsl_ext(GlslStd450::Ceil, rt, { get_spv(n.args[0]) }); return spv_ids[id];
            case SsaOp::Trunc: spv_ids[id] = b.emit_glsl_ext(GlslStd450::Trunc, rt, { get_spv(n.args[0]) }); return spv_ids[id];
            case SsaOp::Ldexp: spv_ids[id] = b.emit_glsl_ext(GlslStd450::Ldexp, rt, { get_spv(n.args[0]), get_spv(n.args[1]) }); return spv_ids[id];
            case SsaOp::FrexpExp: {
                uint32_t st_t = b.type_struct({ b.type_f32(), b.type_s32() });
                uint32_t fr = b.emit_glsl_ext(GlslStd450::Frexp, st_t, { get_spv(n.args[0]) });
                spv_ids[id] = b.op_CompositeExtract(b.type_s32(), fr, { 1u });
                return spv_ids[id];
            }
            case SsaOp::FrexpSignificand: {
                uint32_t st_t = b.type_struct({ b.type_f32(), b.type_s32() });
                uint32_t fr = b.emit_glsl_ext(GlslStd450::Frexp, st_t, { get_spv(n.args[0]) });
                spv_ids[id] = b.op_CompositeExtract(b.type_f32(), fr, { 0u });
                return spv_ids[id];
            }
            case SsaOp::Exp: spv_ids[id] = b.emit_glsl_ext(GlslStd450::Exp, rt, { get_spv(n.args[0]) }); return spv_ids[id];
            case SsaOp::Exp2: spv_ids[id] = b.emit_glsl_ext(GlslStd450::Exp2, rt, { get_spv(n.args[0]) }); return spv_ids[id];
            case SsaOp::Log: spv_ids[id] = b.emit_glsl_ext(GlslStd450::Log, rt, { get_spv(n.args[0]) }); return spv_ids[id];
            case SsaOp::Log2: spv_ids[id] = b.emit_glsl_ext(GlslStd450::Log2, rt, { get_spv(n.args[0]) }); return spv_ids[id];
            case SsaOp::Pow: spv_ids[id] = b.emit_glsl_ext(GlslStd450::Pow, rt, { get_spv(n.args[0]), get_spv(n.args[1]) }); return spv_ids[id];
            case SsaOp::Sqrt: spv_ids[id] = b.emit_glsl_ext(GlslStd450::Sqrt, rt, { get_spv(n.args[0]) }); return spv_ids[id];
            case SsaOp::InverseSqrt: spv_ids[id] = b.emit_glsl_ext(GlslStd450::InverseSqrt, rt, { get_spv(n.args[0]) }); return spv_ids[id];
            case SsaOp::Sin: spv_ids[id] = b.emit_glsl_ext(GlslStd450::Sin, rt, { get_spv(n.args[0]) }); return spv_ids[id];
            case SsaOp::Cos: spv_ids[id] = b.emit_glsl_ext(GlslStd450::Cos, rt, { get_spv(n.args[0]) }); return spv_ids[id];
            case SsaOp::Atan: spv_ids[id] = b.emit_glsl_ext(GlslStd450::Atan, rt, { get_spv(n.args[0]) }); return spv_ids[id];
            case SsaOp::IsNan: spv_ids[id] = b.op_IsNan(rt, get_spv(n.args[0])); return spv_ids[id];
            case SsaOp::IsInf: spv_ids[id] = b.op_IsInf(rt, get_spv(n.args[0])); return spv_ids[id];

            case SsaOp::IAdd:
            case SsaOp::IAdd64: spv_ids[id] = b.op_IAdd(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::ISub:
            case SsaOp::ISub64: spv_ids[id] = b.op_ISub(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::IMul:
            case SsaOp::IMul64: spv_ids[id] = b.op_IMul(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::SAbs: spv_ids[id] = b.emit_glsl_ext(GlslStd450::SAbs, rt, { get_spv(n.args[0]) }); return spv_ids[id];
            case SsaOp::SNeg: spv_ids[id] = b.op_SNegate(rt, get_spv(n.args[0])); return spv_ids[id];

            case SsaOp::BitAnd:
            case SsaOp::BitAnd64: spv_ids[id] = b.op_BitwiseAnd(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::BitOr:
            case SsaOp::BitOr64:  spv_ids[id] = b.op_BitwiseOr(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::BitXor:
            case SsaOp::BitXor64: spv_ids[id] = b.op_BitwiseXor(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::BitNot:
            case SsaOp::BitNot64: spv_ids[id] = b.op_Not(rt, get_spv(n.args[0])); return spv_ids[id];
            case SsaOp::Shl:
            case SsaOp::Shl64:    spv_ids[id] = b.op_ShiftLeftLogical(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::ShrU:
            case SsaOp::ShrU64:   spv_ids[id] = b.op_ShiftRightLogical(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::ShrS:
            case SsaOp::ShrS64:   spv_ids[id] = b.op_ShiftRightArithmetic(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::RotL:
            case SsaOp::RotR: {
                uint32_t s0 = get_spv(n.args[0]);
                uint32_t sh = get_spv(n.args[1]);
                uint32_t inv_sh = b.op_ISub(b.type_u32(), b.const_u32(32u), sh);
                uint32_t p0 = (n.op == SsaOp::RotR) ? b.op_ShiftRightLogical(b.type_u32(), s0, sh) : b.op_ShiftLeftLogical(b.type_u32(), s0, sh);
                uint32_t p1 = (n.op == SsaOp::RotR) ? b.op_ShiftLeftLogical(b.type_u32(), s0, inv_sh) : b.op_ShiftRightLogical(b.type_u32(), s0, inv_sh);
                spv_ids[id] = b.op_BitwiseOr(b.type_u32(), p0, p1);
                return spv_ids[id];
            }
            case SsaOp::BitReverse: spv_ids[id] = b.op_BitReverse(rt, get_spv(n.args[0])); return spv_ids[id];
            case SsaOp::BitCount: spv_ids[id] = b.op_BitCount(rt, get_spv(n.args[0])); return spv_ids[id];
            case SsaOp::Clz: {
                uint32_t s0 = get_spv(n.args[0]);
                uint32_t msb = b.emit_glsl_ext(GlslStd450::FindUMsb, b.type_s32(), { s0 });
                uint32_t msb_u = b.op_Bitcast(b.type_u32(), msb);
                uint32_t is_z = b.op_IEqual(b.type_bool(), s0, b.const_u32(0u));
                uint32_t sub31 = b.op_ISub(b.type_u32(), b.const_u32(31u), msb_u);
                spv_ids[id] = b.op_Select(b.type_u32(), is_z, b.const_u32(32u), sub31);
                return spv_ids[id];
            }

            case SsaOp::FCmpOrdEq: spv_ids[id] = b.op_FOrdEqual(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::FCmpOrdGt: spv_ids[id] = b.op_FOrdGreaterThan(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::FCmpOrdGe: spv_ids[id] = b.op_FOrdGreaterThanEqual(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::FCmpOrdNe: spv_ids[id] = b.op_FOrdNotEqual(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::FCmpOrdLt: spv_ids[id] = b.op_FOrdLessThan(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::FCmpOrdLe: spv_ids[id] = b.op_FOrdLessThanEqual(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::FCmpUnordNe: spv_ids[id] = b.op_FUnordNotEqual(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::FCmpTotal: {
                uint32_t na = b.op_IsNan(b.type_bool(), get_spv(n.args[0]));
                uint32_t nb = b.op_IsNan(b.type_bool(), get_spv(n.args[1]));
                spv_ids[id] = b.op_LogicalNot(b.type_bool(), b.op_LogicalOr(b.type_bool(), na, nb));
                return spv_ids[id];
            }
            case SsaOp::ICmpEq: spv_ids[id] = b.op_IEqual(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::ICmpNe: spv_ids[id] = b.op_INotEqual(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::ICmpSGt: spv_ids[id] = b.op_SGreaterThan(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::ICmpSGe: spv_ids[id] = b.op_SGreaterThanEqual(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::ICmpSLt: spv_ids[id] = b.op_SLessThan(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::ICmpSLe: spv_ids[id] = b.op_SLessThanEqual(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::ICmpUGt: spv_ids[id] = b.op_UGreaterThan(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::ICmpUGe: spv_ids[id] = b.op_UGreaterThanEqual(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::ICmpULt: spv_ids[id] = b.op_ULessThan(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::ICmpULe: spv_ids[id] = b.op_ULessThanEqual(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::LogicalAnd: spv_ids[id] = b.op_LogicalAnd(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::LogicalOr: spv_ids[id] = b.op_LogicalOr(rt, get_spv(n.args[0]), get_spv(n.args[1])); return spv_ids[id];
            case SsaOp::LogicalNot: spv_ids[id] = b.op_LogicalNot(rt, get_spv(n.args[0])); return spv_ids[id];
            case SsaOp::Select: spv_ids[id] = b.op_Select(rt, get_spv(n.args[0]), get_spv(n.args[1]), get_spv(n.args[2])); return spv_ids[id];

            case SsaOp::ConvertFToS: spv_ids[id] = b.op_ConvertFToS(rt, get_spv(n.args[0])); return spv_ids[id];
            case SsaOp::ConvertFToU: spv_ids[id] = b.op_ConvertFToU(rt, get_spv(n.args[0])); return spv_ids[id];
            case SsaOp::ConvertSToF: spv_ids[id] = b.op_ConvertSToF(rt, get_spv(n.args[0])); return spv_ids[id];
            case SsaOp::ConvertUToF: spv_ids[id] = b.op_ConvertUToF(rt, get_spv(n.args[0])); return spv_ids[id];
            case SsaOp::UConvert32To64: spv_ids[id] = b.op_UConvert(b.type_u64(), get_spv(n.args[0])); return spv_ids[id];
            case SsaOp::SConvert32To64: spv_ids[id] = b.op_SConvert(b.type_s64(), get_spv(n.args[0])); return spv_ids[id];
            case SsaOp::UConvert64To32: spv_ids[id] = b.op_UConvert(b.type_u32(), get_spv(n.args[0])); return spv_ids[id];

            case SsaOp::SubgroupBallot:
                spv_ids[id] = b.op_GroupNonUniformBallot(rt, get_spv(n.args[0]), get_spv(n.args[1]));
                return spv_ids[id];
            case SsaOp::SubgroupShuffle:
                spv_ids[id] = b.op_GroupNonUniformShuffle(rt, get_spv(n.args[0]), get_spv(n.args[1]), get_spv(n.args[2]));
                return spv_ids[id];

            case SsaOp::PsbLoad: {
                uint32_t addr = get_spv(n.args[0]);
                uint32_t elem_bytes = n.imm_u32;
                if (n.aux_bool) {
                    // LD_PKA guard: if addr >= 0x10000 load else 0
                    uint32_t res_var = b.add_local_var(b.type_u32(), "pka_tmp");
                    b.emit_store(res_var, b.const_u32(0u));
                    uint32_t ok_lbl = b.alloc_id(), merge_lbl = b.alloc_id();
                    uint32_t is_valid = b.op_UGreaterThanEqual(b.type_bool(), addr, b.const_u64(0x10000ULL));
                    b.op_SelectionMerge(merge_lbl, 0);
                    b.op_BranchConditional(is_valid, ok_lbl, merge_lbl);
                    b.emit_label(ok_lbl);
                    uint32_t p32 = b.type_pointer(StorageClass::PhysicalStorageBuffer, b.type_u32());
                    b.emit_store(res_var, b.emit_load(b.type_u32(), b.op_ConvertUToPtr(p32, addr), 4u));
                    b.op_Branch(merge_lbl);
                    b.emit_label(merge_lbl);
                    spv_ids[id] = b.emit_load(b.type_u32(), res_var);
                    return spv_ids[id];
                }
                if (elem_bytes == 1) {
                    uint32_t p8 = b.type_pointer(StorageClass::PhysicalStorageBuffer, b.type_u8());
                    spv_ids[id] = b.op_UConvert(b.type_u32(), b.emit_load(b.type_u8(), b.op_ConvertUToPtr(p8, addr), 1u));
                } else if (elem_bytes == 2) {
                    uint32_t p16 = b.type_pointer(StorageClass::PhysicalStorageBuffer, b.type_u16());
                    spv_ids[id] = b.op_UConvert(b.type_u32(), b.emit_load(b.type_u16(), b.op_ConvertUToPtr(p16, addr), 2u));
                } else {
                    uint32_t p32 = b.type_pointer(StorageClass::PhysicalStorageBuffer, b.type_u32());
                    spv_ids[id] = b.emit_load(b.type_u32(), b.op_ConvertUToPtr(p32, addr), 4u);
                }
                return spv_ids[id];
            }

            case SsaOp::PsbStore: {
                uint32_t addr = get_spv(n.args[0]);
                uint32_t val = get_spv(n.args[1]);
                uint32_t elem_bytes = n.imm_u32;
                if (elem_bytes == 1) {
                    uint32_t p8 = b.type_pointer(StorageClass::PhysicalStorageBuffer, b.type_u8());
                    b.emit_store(b.op_ConvertUToPtr(p8, addr), b.op_UConvert(b.type_u8(), val), 1u);
                } else if (elem_bytes == 2) {
                    uint32_t p16 = b.type_pointer(StorageClass::PhysicalStorageBuffer, b.type_u16());
                    b.emit_store(b.op_ConvertUToPtr(p16, addr), b.op_UConvert(b.type_u16(), val), 2u);
                } else {
                    uint32_t p32 = b.type_pointer(StorageClass::PhysicalStorageBuffer, b.type_u32());
                    b.emit_store(b.op_ConvertUToPtr(p32, addr), val, 4u);
                }
                spv_ids[id] = 0;
                return 0;
            }

            case SsaOp::AtomicRmw: {
                uint32_t addr = get_spv(n.args[0]);
                uint32_t arg = get_spv(n.args[1]);
                uint32_t val_t = n.aux_bool ? b.type_u64() : b.type_u32();
                uint32_t ptr = b.op_ConvertUToPtr(b.type_pointer(StorageClass::PhysicalStorageBuffer, val_t), addr);
                uint32_t sc = b.const_u32(1u), sem = b.const_u32(0u);
                const std::string& aop = n.aux_str;
                if (aop == "asmin") spv_ids[id] = b.op_AtomicSMin(val_t, ptr, sc, sem, arg);
                else if (aop == "aumin") spv_ids[id] = b.op_AtomicUMin(val_t, ptr, sc, sem, arg);
                else if (aop == "asmax") spv_ids[id] = b.op_AtomicSMax(val_t, ptr, sc, sem, arg);
                else if (aop == "aumax") spv_ids[id] = b.op_AtomicUMax(val_t, ptr, sc, sem, arg);
                else if (aop == "aand") spv_ids[id] = b.op_AtomicAnd(val_t, ptr, sc, sem, arg);
                else if (aop == "aor") spv_ids[id] = b.op_AtomicOr(val_t, ptr, sc, sem, arg);
                else if (aop == "axor") spv_ids[id] = b.op_AtomicXor(val_t, ptr, sc, sem, arg);
                else spv_ids[id] = b.op_AtomicIAdd(val_t, ptr, sc, sem, arg);
                return spv_ids[id];
            }

            case SsaOp::AtomicExchange: {
                uint32_t ptr = b.op_ConvertUToPtr(b.type_pointer(StorageClass::PhysicalStorageBuffer, b.type_u32()), get_spv(n.args[0]));
                spv_ids[id] = b.op_AtomicExchange(b.type_u32(), ptr, b.const_u32(1u), b.const_u32(0u), get_spv(n.args[1]));
                return spv_ids[id];
            }

            case SsaOp::AtomicCmpXchg: {
                uint32_t ptr = b.op_ConvertUToPtr(b.type_pointer(StorageClass::PhysicalStorageBuffer, b.type_u32()), get_spv(n.args[0]));
                spv_ids[id] = b.op_AtomicCompareExchange(b.type_u32(), ptr, b.const_u32(1u), b.const_u32(0u), b.const_u32(0u),
                                                         get_spv(n.args[1]), get_spv(n.args[2]));
                return spv_ids[id];
            }

            case SsaOp::TexSample:
            case SsaOp::TexSampleLod:
            case SsaOp::TexSampleDref:
            case SsaOp::TexGather:
            case SsaOp::TexFetch:
            case SsaOp::ImageQueryLod: {
                uint32_t si_t = 0;
                uint32_t samp_var = get_or_create_sampled_image(n.imm_u32, &si_t);
                uint32_t si_val = b.emit_load(si_t, samp_var);
                uint32_t uv = get_spv(n.args[0]);
                uint32_t v4f = b.type_vector(b.type_f32(), 4);
                if (n.aux_bool2) {
                    spv_ids[id] = b.op_ImageGather(v4f, si_val, uv, b.const_u32(n.imm_u32_2));
                } else if (n.aux_bool) {
                    spv_ids[id] = b.op_ImageSampleImplicitLod(v4f, si_val, uv);
                } else {
                    spv_ids[id] = b.op_ImageSampleExplicitLod(v4f, si_val, uv, ImageOperandsMask::Lod, { b.const_f32(0.0f) });
                }
                return spv_ids[id];
            }

            case SsaOp::StoreBuiltinOutput: {
                uint32_t val = get_spv(n.args[0]);
                uint32_t vt = (n.imm_u32 == static_cast<uint32_t>(BuiltIn::Position)) ? b.type_vector(b.type_f32(), 4) : b.type_f32();
                uint32_t out_var = b.get_or_create_builtin_output(static_cast<BuiltIn>(n.imm_u32), vt, n.aux_str);
                b.emit_store(out_var, val);
                return 0;
            }
            case SsaOp::StoreVertVarying: {
                uint32_t out_var = get_or_create_vert_varying(n.imm_u32);
                b.emit_store(out_var, get_spv(n.args[0]));
                return 0;
            }
            case SsaOp::StoreFragColor: {
                uint32_t out_var = get_or_create_frag_color_output(n.imm_u32);
                b.emit_store(out_var, get_spv(n.args[0]));
                return 0;
            }
            case SsaOp::ControlBarrier: {
                b.op_ControlBarrier(get_spv(n.args[0]), get_spv(n.args[1]), get_spv(n.args[2]));
                return 0;
            }
            case SsaOp::Kill: {
                if (n.aux_bool) {
                    uint32_t is_zero = get_spv(n.args[0]);
                    uint32_t kill_lbl = b.alloc_id(), cont_lbl = b.alloc_id();
                    b.op_SelectionMerge(cont_lbl, 0);
                    b.op_BranchConditional(is_zero, kill_lbl, cont_lbl);
                    b.emit_label(kill_lbl);
                    b.op_Kill();
                    b.emit_label(cont_lbl);
                } else {
                    b.op_Kill();
                }
                return 0;
            }
            default:
                return 0;
        }
    }
};

} // namespace

TilerResult run_pass2_tiler_and_emit(
    Pass1Result& pass1,
    const ValhallTranslateOptions& opts) {
    TilerResult res;
    if (!pass1.ok) {
        res.error_message = pass1.error_message;
        return res;
    }

    auto& mod = pass1.ssa;
    mod.recompute_use_counts();
    res.stats.total_ssa_nodes = (mod.nodes.size() > 1) ? (mod.nodes.size() - 1) : 0;

    // 1. Bounded DP DAG Cover (K <= 5) in topological order
    size_t num_fusion = 0, num_pnodes = 0;
    const auto* fusion_tiles = tables::get_fusion_tiles(&num_fusion);
    const auto* pnodes = tables::get_fusion_pattern_nodes(&num_pnodes);

    std::vector<DpChoice> choices(mod.nodes.size());
    for (size_t i = 1; i < mod.nodes.size(); ++i) {
        const auto& n = mod.nodes[i];
        DpChoice best{};
        const auto* ctile = tables::lookup_canonical_tile(n.op);
        uint32_t base_c = ctile ? ctile->cost : 1u;
        uint32_t arg_sum = 0;
        for (SsaId a : n.args) {
            if (a > 0 && a < choices.size()) arg_sum += choices[a].cost;
        }
        best.is_fusion = false;
        best.cost = base_c + arg_sum;
        best.canon = ctile;
        best.effects = n.effects;
        if (ctile) {
            best.effects.cap_bits |= ctile->effects.cap_bits;
            best.effects.ext_bits |= ctile->effects.ext_bits;
            best.effects.exec_bits |= ctile->effects.exec_bits;
            best.effects.flags |= ctile->effects.flags;
        }

        // Try fusion tiles
        for (size_t ft_idx = 0; ft_idx < num_fusion; ++ft_idx) {
            const auto& ft = fusion_tiles[ft_idx];
            if (ft.requires_fast_math && !opts.enable_fast_math_tiles) {
                continue;
            }
            std::vector<SsaId> leaves;
            std::unordered_map<SsaId, uint32_t> internal_uses;
            if (!match_fusion_pattern(mod, pnodes, ft.root_node, static_cast<SsaId>(i),
                                      n.block_id, true, leaves, internal_uses)) {
                continue;
            }
            uint32_t total_c = ft.cost;
            for (SsaId lf : leaves) {
                if (lf > 0 && lf < choices.size()) total_c += choices[lf].cost;
            }
            // Multi-use interior node penalty: if an interior node w is used outside the tile, add BestCost[w]
            for (const auto& kv : internal_uses) {
                if (mod.use_counts[kv.first] > kv.second) {
                    total_c += choices[kv.first].cost;
                }
            }
            if (total_c < best.cost) {
                best.is_fusion = true;
                best.cost = total_c;
                best.fusion = &ft;
                best.canon = nullptr;
                best.leaf_args = std::move(leaves);
                best.effects = ft.effects;
            }
        }
        choices[i] = std::move(best);
    }

    // 2. Backward Liveness Pass from side-effecting roots, Phis, and branch conditions
    std::vector<bool> is_live(mod.nodes.size(), false);
    std::vector<SsaId> worklist;
    auto mark_live = [&](SsaId id) {
        if (id > 0 && id < is_live.size() && !is_live[id]) {
            is_live[id] = true;
            worklist.push_back(id);
        }
    };

    for (size_t i = 1; i < mod.nodes.size(); ++i) {
        const auto& n = mod.nodes[i];
        if (n.is_side_effect || n.is_barrier_or_cf) {
            mark_live(static_cast<SsaId>(i));
        }
    }
    for (const auto& blk : mod.blocks) {
        if (blk.branch_cond != kInvalidSsa) mark_live(blk.branch_cond);
    }

    while (!worklist.empty()) {
        SsaId u = worklist.back();
        worklist.pop_back();
        const auto& n = mod.nodes[u];
        const auto& ch = choices[u];
        if (n.op == SsaOp::Phi) {
            for (const auto& inc : n.phi_incomings) mark_live(inc.value);
        } else if (ch.is_fusion) {
            for (SsaId lf : ch.leaf_args) mark_live(lf);
        } else {
            for (SsaId a : n.args) mark_live(a);
        }
    }

    // 3. Post-Tiling EffectSet Lattice Join over surviving live tiles + stage preamble
    EffectSet final_effects = pass1.preamble_effects;
    if (pass1.uses_sideband) final_effects.flags |= EffectSet::kFlagUsesSideband;
    for (size_t i = 1; i < mod.nodes.size(); ++i) {
        if (!is_live[i]) continue;
        res.stats.live_ssa_nodes++;
        const auto& ch = choices[i];
        final_effects.join(ch.effects);
        res.stats.total_dp_cost += ch.cost;
        if (ch.is_fusion) res.stats.fusion_tiles_selected++;
        else res.stats.canonical_tiles_selected++;
    }
    res.effects = final_effects;

    // 4. Single-Pass Prelude + Structured CFG + OpPhi + Tile Emission
    ExecutionModel model = ExecutionModel::GLCompute;
    if (opts.stage == ShaderStage::Vertex) model = ExecutionModel::Vertex;
    else if (opts.stage == ShaderStage::Fragment) model = ExecutionModel::Fragment;

    EmitContext ctx(model, mod, opts, choices, is_live);
    auto& b = ctx.b;

    if (model == ExecutionModel::GLCompute) {
        b.add_execution_mode(ExecutionMode::LocalSize, { pass1.local_size_x, pass1.local_size_y, pass1.local_size_z });
    } else if (model == ExecutionModel::Fragment) {
        b.add_execution_mode(ExecutionMode::OriginUpperLeft);
    }
    final_effects.apply_to_builder(b);

    // Always declare sideband if used (or for uniform buffer layout compatibility)
    if (final_effects.uses_sideband()) {
        ctx.declare_sideband();
    }

    b.begin_main_function();

    if (mod.blocks.empty()) {
        b.op_Return();
        b.end_main_function();
        res.spirv_words = b.assemble();
        res.ok = true;
        return res;
    }

    std::vector<uint32_t> blk_labels(mod.blocks.size());
    for (size_t i = 0; i < mod.blocks.size(); ++i) {
        blk_labels[i] = b.alloc_id();
    }
    b.op_Branch(blk_labels[0]);

    for (size_t bi = 0; bi < mod.blocks.size(); ++bi) {
        const auto& blk = mod.blocks[bi];
        b.emit_label(blk_labels[bi]);

        // Emit live OpPhis first at block start
        for (SsaId phi_id : blk.phis) {
            if (!is_live[phi_id]) continue;
            const auto& pn = mod.get(phi_id);
            if (pn.type == ValType::Token) continue;
            uint32_t spv_phi = b.emit_phi(ctx.to_spv_type(pn.type), pn.phi_incomings.size());
            ctx.spv_ids[phi_id] = spv_phi;
        }

        // Emit live non-Phi nodes in topological order
        for (SsaId nid : blk.nodes) {
            if (!is_live[nid]) continue;
            ctx.emit_node(nid);
        }

        // Emit block terminator
        if (blk.is_terminated_by_discard) {
            continue;
        }
        if (blk.is_unconditional_branch && blk.branch_target_block >= 0 &&
            static_cast<size_t>(blk.branch_target_block) < blk_labels.size()) {
            b.op_Branch(blk_labels[blk.branch_target_block]);
            continue;
        }
        if (blk.is_conditional_branch && blk.branch_target_block >= 0 &&
            static_cast<size_t>(blk.branch_target_block) < blk_labels.size() &&
            blk.fallthrough_block >= 0 &&
            static_cast<size_t>(blk.fallthrough_block) < blk_labels.size()) {
            uint32_t cond = (blk.branch_cond != kInvalidSsa) ? ctx.get_spv(blk.branch_cond) : b.const_bool(true);
            uint32_t tgt_lbl = blk_labels[blk.branch_target_block];
            uint32_t fall_lbl = blk_labels[blk.fallthrough_block];
            if (blk.is_loop_header && blk.loop_backedge_block >= 0 && blk.loop_merge_block >= 0) {
                b.op_LoopMerge(blk_labels[blk.loop_merge_block], blk_labels[blk.loop_backedge_block], 0);
            } else if (blk.selection_merge_block >= 0 &&
                       static_cast<size_t>(blk.selection_merge_block) < blk_labels.size()) {
                b.op_SelectionMerge(blk_labels[blk.selection_merge_block], 0);
            }
            b.op_BranchConditional(cond, tgt_lbl, fall_lbl);
            continue;
        }
        if (blk.is_terminated_by_end) {
            b.op_Return();
            continue;
        }
        if (blk.fallthrough_block >= 0 && static_cast<size_t>(blk.fallthrough_block) < blk_labels.size()) {
            b.op_Branch(blk_labels[blk.fallthrough_block]);
        } else {
            b.op_Return();
        }
    }

    // Patch all OpPhi incomings now that every block's values have been emitted
    for (size_t bi = 0; bi < mod.blocks.size(); ++bi) {
        const auto& blk = mod.blocks[bi];
        for (SsaId phi_id : blk.phis) {
            if (!is_live[phi_id]) continue;
            const auto& pn = mod.get(phi_id);
            if (pn.type == ValType::Token) continue;
            std::vector<std::pair<uint32_t, uint32_t>> incomings;
            incomings.reserve(pn.phi_incomings.size());
            for (const auto& inc : pn.phi_incomings) {
                uint32_t val_spv = ctx.get_spv(inc.value);
                uint32_t pred_lbl = blk_labels[inc.pred_block];
                incomings.emplace_back(val_spv, pred_lbl);
            }
            b.patch_phi(ctx.spv_ids[phi_id], incomings);
        }
    }

    b.end_main_function();
    if (b.has_errors()) {
        res.ok = false;
        res.error_message = b.errors().front();
        return res;
    }
    res.spirv_words = b.assemble();
    res.ok = true;
    return res;
}

} // namespace valhall::compiler

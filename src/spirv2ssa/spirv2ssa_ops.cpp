#include "spirv2ssa_internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace valhall::spirv2ssa {

namespace {

using compiler::SsaId;
using compiler::SsaNode;
using compiler::ValType;
using tables::SsaOp;

uint32_t f32_to_bits(float f) {
    uint32_t b = 0;
    std::memcpy(&b, &f, sizeof(float));
    return b;
}

} // namespace

bool SpirvLowerer::lower_glsl450_ext_inst(uint32_t blk_id, const SpirvInst& inst) {
    // OpExtInst: [hdr, result_type, result_id, set_id, ext_opcode, arg0, ...]
    if (inst.words.size() < 5) return false;
    uint32_t res_type = inst.words[1];
    uint32_t res_id = inst.words[2];
    uint32_t ext_op = inst.words[4];

    uint32_t nlanes = flat_lane_count(res_type);
    ValType elem_vt = to_val_type(scalar_elem_type(res_type));
    LoweredValue out{};
    out.spv_type_id = res_type;

    auto arg_val = [&](size_t idx) -> LoweredValue {
        return (5 + idx < inst.words.size()) ? get_value(blk_id, inst.words[5 + idx]) : make_zero_value(blk_id, res_type);
    };
    auto lane_of = [](const LoweredValue& v, uint32_t i) -> SsaId {
        if (v.lanes.empty()) return compiler::kInvalidSsa;
        return (i < v.lanes.size()) ? v.lanes[i] : v.lanes[0];
    };

    auto emit_unary = [&](SsaOp op) {
        LoweredValue a = arg_val(0);
        for (uint32_t i = 0; i < nlanes; ++i) {
            out.lanes.push_back(emit_op(blk_id, op, elem_vt, {lane_of(a, i)}));
        }
    };
    auto emit_binary = [&](SsaOp op) {
        LoweredValue a = arg_val(0), b = arg_val(1);
        for (uint32_t i = 0; i < nlanes; ++i) {
            out.lanes.push_back(emit_op(blk_id, op, elem_vt, {lane_of(a, i), lane_of(b, i)}));
        }
    };

    switch (ext_op) {
        case 1: // Round
        case 2: // RoundEven
            emit_unary(SsaOp::RoundEven);
            break;
        case 3: // Trunc
            emit_unary(SsaOp::Trunc);
            break;
        case 4: // FAbs
            emit_unary(SsaOp::FAbs);
            break;
        case 5: // SAbs
            emit_unary(SsaOp::SAbs);
            break;
        case 6: { // FSign
            LoweredValue a = arg_val(0);
            SsaId z = sb_.emit_const_f32_bits(blk_id, 0u);
            SsaId p1 = sb_.emit_const_f32_bits(blk_id, f32_to_bits(1.0f));
            SsaId m1 = sb_.emit_const_f32_bits(blk_id, f32_to_bits(-1.0f));
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId x = lane_of(a, i);
                SsaId gt = emit_op(blk_id, SsaOp::FCmpOrdGt, ValType::Bool, {x, z});
                SsaId lt = emit_op(blk_id, SsaOp::FCmpOrdLt, ValType::Bool, {x, z});
                SsaId neg_or_z = emit_op(blk_id, SsaOp::Select, ValType::F32, {lt, m1, z});
                out.lanes.push_back(emit_op(blk_id, SsaOp::Select, ValType::F32, {gt, p1, neg_or_z}));
            }
            break;
        }
        case 7: { // SSign
            LoweredValue a = arg_val(0);
            SsaId z = sb_.emit_const_s32(blk_id, 0);
            SsaId p1 = sb_.emit_const_s32(blk_id, 1);
            SsaId m1 = sb_.emit_const_s32(blk_id, -1);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId x = lane_of(a, i);
                SsaId gt = emit_op(blk_id, SsaOp::ICmpSGt, ValType::Bool, {x, z});
                SsaId lt = emit_op(blk_id, SsaOp::ICmpSLt, ValType::Bool, {x, z});
                SsaId neg_or_z = emit_op(blk_id, SsaOp::Select, ValType::S32, {lt, m1, z});
                out.lanes.push_back(emit_op(blk_id, SsaOp::Select, ValType::S32, {gt, p1, neg_or_z}));
            }
            break;
        }
        case 8: emit_unary(SsaOp::Floor); break;
        case 9: emit_unary(SsaOp::Ceil); break;
        case 10: { // Fract: x - floor(x)
            LoweredValue a = arg_val(0);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId x = lane_of(a, i);
                SsaId fl = emit_op(blk_id, SsaOp::Floor, ValType::F32, {x});
                out.lanes.push_back(emit_op(blk_id, SsaOp::FSub, ValType::F32, {x, fl}));
            }
            break;
        }
        case 13: emit_unary(SsaOp::Sin); break;
        case 14: emit_unary(SsaOp::Cos); break;
        case 15: { // Tan = Sin / Cos
            LoweredValue a = arg_val(0);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId x = lane_of(a, i);
                SsaId s = emit_op(blk_id, SsaOp::Sin, ValType::F32, {x});
                SsaId c = emit_op(blk_id, SsaOp::Cos, ValType::F32, {x});
                out.lanes.push_back(emit_op(blk_id, SsaOp::FDiv, ValType::F32, {s, c}));
            }
            break;
        }
        case 18: emit_unary(SsaOp::Atan); break;
        case 26: emit_binary(SsaOp::Pow); break;
        case 27: emit_unary(SsaOp::Exp); break;
        case 28: emit_unary(SsaOp::Log); break;
        case 29: emit_unary(SsaOp::Exp2); break;
        case 30: emit_unary(SsaOp::Log2); break;
        case 31: emit_unary(SsaOp::Sqrt); break;
        case 32: emit_unary(SsaOp::InverseSqrt); break;
        case 37: emit_binary(SsaOp::FMin); break;
        case 38: { // UMin
            LoweredValue a = arg_val(0), b = arg_val(1);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId c = emit_op(blk_id, SsaOp::ICmpULt, ValType::Bool, {lane_of(a, i), lane_of(b, i)});
                out.lanes.push_back(emit_op(blk_id, SsaOp::Select, elem_vt, {c, lane_of(a, i), lane_of(b, i)}));
            }
            break;
        }
        case 39: { // SMin
            LoweredValue a = arg_val(0), b = arg_val(1);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId c = emit_op(blk_id, SsaOp::ICmpSLt, ValType::Bool, {lane_of(a, i), lane_of(b, i)});
                out.lanes.push_back(emit_op(blk_id, SsaOp::Select, elem_vt, {c, lane_of(a, i), lane_of(b, i)}));
            }
            break;
        }
        case 40: emit_binary(SsaOp::FMax); break;
        case 41: { // UMax
            LoweredValue a = arg_val(0), b = arg_val(1);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId c = emit_op(blk_id, SsaOp::ICmpUGt, ValType::Bool, {lane_of(a, i), lane_of(b, i)});
                out.lanes.push_back(emit_op(blk_id, SsaOp::Select, elem_vt, {c, lane_of(a, i), lane_of(b, i)}));
            }
            break;
        }
        case 42: { // SMax
            LoweredValue a = arg_val(0), b = arg_val(1);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId c = emit_op(blk_id, SsaOp::ICmpSGt, ValType::Bool, {lane_of(a, i), lane_of(b, i)});
                out.lanes.push_back(emit_op(blk_id, SsaOp::Select, elem_vt, {c, lane_of(a, i), lane_of(b, i)}));
            }
            break;
        }
        case 43: // FClamp
        case 81: { // NClamp
            LoweredValue x = arg_val(0), lo = arg_val(1), hi = arg_val(2);
            for (uint32_t i = 0; i < nlanes; ++i) {
                out.lanes.push_back(emit_op(blk_id, SsaOp::FClamp, ValType::F32,
                                            {lane_of(x, i), lane_of(lo, i), lane_of(hi, i)}));
            }
            break;
        }
        case 44: { // UClamp
            LoweredValue x = arg_val(0), lo = arg_val(1), hi = arg_val(2);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId c1 = emit_op(blk_id, SsaOp::ICmpUGt, ValType::Bool, {lane_of(x, i), lane_of(lo, i)});
                SsaId mx = emit_op(blk_id, SsaOp::Select, elem_vt, {c1, lane_of(x, i), lane_of(lo, i)});
                SsaId c2 = emit_op(blk_id, SsaOp::ICmpULt, ValType::Bool, {mx, lane_of(hi, i)});
                out.lanes.push_back(emit_op(blk_id, SsaOp::Select, elem_vt, {c2, mx, lane_of(hi, i)}));
            }
            break;
        }
        case 45: { // SClamp
            LoweredValue x = arg_val(0), lo = arg_val(1), hi = arg_val(2);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId c1 = emit_op(blk_id, SsaOp::ICmpSGt, ValType::Bool, {lane_of(x, i), lane_of(lo, i)});
                SsaId mx = emit_op(blk_id, SsaOp::Select, elem_vt, {c1, lane_of(x, i), lane_of(lo, i)});
                SsaId c2 = emit_op(blk_id, SsaOp::ICmpSLt, ValType::Bool, {mx, lane_of(hi, i)});
                out.lanes.push_back(emit_op(blk_id, SsaOp::Select, elem_vt, {c2, mx, lane_of(hi, i)}));
            }
            break;
        }
        case 46: { // FMix: x * (1 - a) + y * a = x + a * (y - x)
            LoweredValue x = arg_val(0), y = arg_val(1), a = arg_val(2);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId d = emit_op(blk_id, SsaOp::FSub, ValType::F32, {lane_of(y, i), lane_of(x, i)});
                out.lanes.push_back(emit_op(blk_id, SsaOp::Fma, ValType::F32, {lane_of(a, i), d, lane_of(x, i)}));
            }
            break;
        }
        case 48: { // Step: (x < edge) ? 0.0 : 1.0
            LoweredValue edge = arg_val(0), x = arg_val(1);
            SsaId z = sb_.emit_const_f32_bits(blk_id, 0u);
            SsaId one = sb_.emit_const_f32_bits(blk_id, f32_to_bits(1.0f));
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId lt = emit_op(blk_id, SsaOp::FCmpOrdLt, ValType::Bool, {lane_of(x, i), lane_of(edge, i)});
                out.lanes.push_back(emit_op(blk_id, SsaOp::Select, ValType::F32, {lt, z, one}));
            }
            break;
        }
        case 50: { // Fma
            LoweredValue a = arg_val(0), b = arg_val(1), c = arg_val(2);
            for (uint32_t i = 0; i < nlanes; ++i) {
                out.lanes.push_back(emit_op(blk_id, SsaOp::Fma, ValType::F32,
                                            {lane_of(a, i), lane_of(b, i), lane_of(c, i)}));
            }
            break;
        }
        case 53: emit_binary(SsaOp::Ldexp); break;
        case 58: { // PackHalf2x16
            LoweredValue v = arg_val(0);
            SsaId v2 = pack_to_vec2_f32(blk_id, v);
            out.lanes.push_back(emit_op(blk_id, SsaOp::PackHalf2x16, ValType::U32, {v2}));
            break;
        }
        case 62: { // UnpackHalf2x16
            LoweredValue u = arg_val(0);
            SsaId v2 = emit_op(blk_id, SsaOp::UnpackHalf2x16, ValType::V2F32, {u.scalar()});
            for (uint32_t c = 0; c < 2; ++c) {
                SsaNode ex{}; ex.op = SsaOp::VecExtract; ex.type = ValType::F32; ex.imm_u32 = c; ex.args = {v2};
                out.lanes.push_back(sb_.emit(blk_id, std::move(ex)));
            }
            break;
        }
        case 66: { // Length
            LoweredValue v = arg_val(0);
            SsaId acc = emit_op(blk_id, SsaOp::FMul, ValType::F32, {v.lanes[0], v.lanes[0]});
            for (size_t i = 1; i < v.lanes.size(); ++i) {
                acc = emit_op(blk_id, SsaOp::Fma, ValType::F32, {v.lanes[i], v.lanes[i], acc});
            }
            out.lanes.push_back(emit_op(blk_id, SsaOp::Sqrt, ValType::F32, {acc}));
            break;
        }
        case 67: { // Distance
            LoweredValue a = arg_val(0), b = arg_val(1);
            SsaId d0 = emit_op(blk_id, SsaOp::FSub, ValType::F32, {a.lanes[0], b.lanes[0]});
            SsaId acc = emit_op(blk_id, SsaOp::FMul, ValType::F32, {d0, d0});
            for (size_t i = 1; i < a.lanes.size(); ++i) {
                SsaId di = emit_op(blk_id, SsaOp::FSub, ValType::F32, {a.lanes[i], b.lanes[i]});
                acc = emit_op(blk_id, SsaOp::Fma, ValType::F32, {di, di, acc});
            }
            out.lanes.push_back(emit_op(blk_id, SsaOp::Sqrt, ValType::F32, {acc}));
            break;
        }
        case 69: { // Normalize
            LoweredValue v = arg_val(0);
            SsaId acc = emit_op(blk_id, SsaOp::FMul, ValType::F32, {v.lanes[0], v.lanes[0]});
            for (size_t i = 1; i < v.lanes.size(); ++i) {
                acc = emit_op(blk_id, SsaOp::Fma, ValType::F32, {v.lanes[i], v.lanes[i], acc});
            }
            SsaId inv_len = emit_op(blk_id, SsaOp::InverseSqrt, ValType::F32, {acc});
            for (size_t i = 0; i < v.lanes.size(); ++i) {
                out.lanes.push_back(emit_op(blk_id, SsaOp::FMul, ValType::F32, {v.lanes[i], inv_len}));
            }
            break;
        }
        case 75: { // FindUMsb: (x == 0) ? -1 : (31 - clz(x))
            LoweredValue a = arg_val(0);
            SsaId z = sb_.emit_const_u32(blk_id, 0u);
            SsaId c31 = sb_.emit_const_u32(blk_id, 31u);
            SsaId m1 = sb_.emit_const_u32(blk_id, 0xFFFFFFFFu);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId x = lane_of(a, i);
                SsaId is_z = emit_op(blk_id, SsaOp::ICmpEq, ValType::Bool, {x, z});
                SsaId clz = emit_op(blk_id, SsaOp::Clz, ValType::U32, {x});
                SsaId msb = emit_op(blk_id, SsaOp::ISub, ValType::U32, {c31, clz});
                out.lanes.push_back(emit_op(blk_id, SsaOp::Select, elem_vt, {is_z, m1, msb}));
            }
            break;
        }
        case 79: emit_binary(SsaOp::NMin); break;
        case 80: emit_binary(SsaOp::NMax); break;
        default:
            record_unsupported("GLSL.std.450#" + std::to_string(ext_op));
            if (!opts_.tolerant) return false;
            out = make_zero_value(blk_id, res_type);
            break;
    }

    values_[res_id] = std::move(out);
    return true;
}

bool SpirvLowerer::lower_alu_or_misc_inst(uint32_t blk_id, const SpirvInst& inst) {
    uint16_t op = inst.opcode;
    if (inst.words.size() < 3) return true;
    uint32_t res_type = inst.words[1];
    uint32_t res_id = inst.words[2];

    uint32_t nlanes = flat_lane_count(res_type);
    uint32_t elem_tid = scalar_elem_type(res_type);
    ValType elem_vt = to_val_type(elem_tid);
    const SpirvType* et = mod_.get_type(elem_tid);
    bool is_64 = (et && et->width == 64);

    auto arg_v = [&](size_t word_idx) -> LoweredValue {
        return (word_idx < inst.words.size()) ? get_value(blk_id, inst.words[word_idx]) : make_zero_value(blk_id, res_type);
    };
    auto lane_of = [](const LoweredValue& v, uint32_t i) -> SsaId {
        if (v.lanes.empty()) return compiler::kInvalidSsa;
        return (i < v.lanes.size()) ? v.lanes[i] : v.lanes[0];
    };

    LoweredValue out{};
    out.spv_type_id = res_type;

    auto emit_unary = [&](SsaOp sop) {
        LoweredValue a = arg_v(3);
        for (uint32_t i = 0; i < nlanes; ++i) {
            out.lanes.push_back(emit_op(blk_id, sop, elem_vt, {lane_of(a, i)}));
        }
    };
    auto emit_binary = [&](SsaOp sop) {
        LoweredValue a = arg_v(3), b = arg_v(4);
        for (uint32_t i = 0; i < nlanes; ++i) {
            out.lanes.push_back(emit_op(blk_id, sop, elem_vt, {lane_of(a, i), lane_of(b, i)}));
        }
    };
    auto emit_cmp = [&](SsaOp sop) {
        LoweredValue a = arg_v(3), b = arg_v(4);
        for (uint32_t i = 0; i < nlanes; ++i) {
            out.lanes.push_back(emit_op(blk_id, sop, ValType::Bool, {lane_of(a, i), lane_of(b, i)}));
        }
    };

    switch (op) {
        // Composite & Vector operations
        case 79: { // OpVectorShuffle: [hdr, type, id, vec1, vec2, comp0, ...]
            LoweredValue v1 = arg_v(3), v2 = arg_v(4);
            std::vector<SsaId> combined = v1.lanes;
            combined.insert(combined.end(), v2.lanes.begin(), v2.lanes.end());
            for (size_t k = 5; k < inst.words.size(); ++k) {
                uint32_t c = inst.words[k];
                if (c < combined.size()) {
                    out.lanes.push_back(combined[c]);
                } else {
                    out.lanes.push_back(sb_.emit_const_u32(blk_id, 0u));
                }
            }
            break;
        }
        case 80: { // OpCompositeConstruct
            for (size_t k = 3; k < inst.words.size(); ++k) {
                LoweredValue cv = get_value(blk_id, inst.words[k]);
                out.lanes.insert(out.lanes.end(), cv.lanes.begin(), cv.lanes.end());
            }
            break;
        }
        case 81: { // OpCompositeExtract
            LoweredValue comp = arg_v(3);
            uint32_t cur_t = comp.spv_type_id;
            uint32_t flat_off = 0;
            for (size_t k = 4; k < inst.words.size(); ++k) {
                uint32_t idx = inst.words[k];
                const SpirvType* ct = mod_.get_type(cur_t);
                if (!ct) break;
                if (ct->kind == SpirvTypeKind::Vector) {
                    flat_off += idx;
                    cur_t = ct->elem_type;
                } else if (ct->kind == SpirvTypeKind::Matrix || ct->kind == SpirvTypeKind::Array) {
                    flat_off += idx * flat_lane_count(ct->elem_type);
                    cur_t = ct->elem_type;
                } else if (ct->kind == SpirvTypeKind::Struct) {
                    for (uint32_t m = 0; m < idx && m < ct->member_types.size(); ++m) {
                        flat_off += flat_lane_count(ct->member_types[m]);
                    }
                    cur_t = (idx < ct->member_types.size()) ? ct->member_types[idx] : 0;
                }
            }
            for (uint32_t i = 0; i < nlanes; ++i) {
                out.lanes.push_back((flat_off + i < comp.lanes.size()) ? comp.lanes[flat_off + i]
                                                                       : sb_.emit_const_u32(blk_id, 0u));
            }
            break;
        }
        case 82: { // OpCompositeInsert: [hdr, type, id, obj, composite, idx0, ...]
            LoweredValue obj = arg_v(3);
            LoweredValue comp = arg_v(4);
            out.lanes = comp.lanes;
            uint32_t cur_t = comp.spv_type_id;
            uint32_t flat_off = 0;
            for (size_t k = 5; k < inst.words.size(); ++k) {
                uint32_t idx = inst.words[k];
                const SpirvType* ct = mod_.get_type(cur_t);
                if (!ct) break;
                if (ct->kind == SpirvTypeKind::Vector) {
                    flat_off += idx;
                    cur_t = ct->elem_type;
                } else if (ct->kind == SpirvTypeKind::Matrix || ct->kind == SpirvTypeKind::Array) {
                    flat_off += idx * flat_lane_count(ct->elem_type);
                    cur_t = ct->elem_type;
                } else if (ct->kind == SpirvTypeKind::Struct) {
                    for (uint32_t m = 0; m < idx && m < ct->member_types.size(); ++m) {
                        flat_off += flat_lane_count(ct->member_types[m]);
                    }
                    cur_t = (idx < ct->member_types.size()) ? ct->member_types[idx] : 0;
                }
            }
            for (size_t i = 0; i < obj.lanes.size() && flat_off + i < out.lanes.size(); ++i) {
                out.lanes[flat_off + i] = obj.lanes[i];
            }
            break;
        }
        case 83: { // OpCopyObject
            out = arg_v(3);
            out.spv_type_id = res_type;
            break;
        }
        case 86: { // OpSampledImage
            uint32_t img_id = inst.words[3];
            if (auto it = sampled_images_.find(img_id); it != sampled_images_.end()) {
                sampled_images_[res_id] = it->second;
            }
            out.lanes.push_back(sb_.emit_const_u32(blk_id, img_id));
            break;
        }

        // Conversions
        case 109: emit_unary(SsaOp::ConvertFToU); break;
        case 110: emit_unary(SsaOp::ConvertFToS); break;
        case 111: emit_unary(SsaOp::ConvertSToF); break;
        case 112: emit_unary(SsaOp::ConvertUToF); break;
        case 113: { // OpUConvert
            LoweredValue a = arg_v(3);
            const SpirvType* src_et = mod_.get_type(scalar_elem_type(a.spv_type_id));
            uint32_t sw = src_et ? src_et->width : 32u;
            uint32_t dw = et ? et->width : 32u;
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId x = lane_of(a, i);
                if (sw <= 32 && dw == 64) {
                    out.lanes.push_back(emit_op(blk_id, SsaOp::UConvert32To64, ValType::U64, {x}));
                } else if (sw == 64 && dw <= 32) {
                    out.lanes.push_back(emit_op(blk_id, SsaOp::UConvert64To32, ValType::U32, {x}));
                } else if (dw < 32) {
                    out.lanes.push_back(emit_op(blk_id, SsaOp::BfeU, ValType::U32,
                                                {x, sb_.emit_const_u32(blk_id, 0u), sb_.emit_const_u32(blk_id, dw)}));
                } else {
                    out.lanes.push_back(sb_.emit_bitcast(blk_id, elem_vt, x));
                }
            }
            break;
        }
        case 114: { // OpSConvert
            LoweredValue a = arg_v(3);
            const SpirvType* src_et = mod_.get_type(scalar_elem_type(a.spv_type_id));
            uint32_t sw = src_et ? src_et->width : 32u;
            uint32_t dw = et ? et->width : 32u;
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId x = lane_of(a, i);
                if (sw <= 32 && dw == 64) {
                    out.lanes.push_back(emit_op(blk_id, SsaOp::SConvert32To64, ValType::S64, {x}));
                } else if (sw == 64 && dw <= 32) {
                    out.lanes.push_back(emit_op(blk_id, SsaOp::UConvert64To32, ValType::S32, {x}));
                } else if (sw < 32 && dw == 32) {
                    out.lanes.push_back(emit_op(blk_id, SsaOp::BfeS, ValType::S32,
                                                {x, sb_.emit_const_u32(blk_id, 0u), sb_.emit_const_u32(blk_id, sw)}));
                } else {
                    out.lanes.push_back(sb_.emit_bitcast(blk_id, elem_vt, x));
                }
            }
            break;
        }
        case 115: { // OpFConvert (FP16 <-> FP32 represented via PackHalf2x16/UnpackHalf2x16 roundtrip)
            LoweredValue a = arg_v(3);
            for (uint32_t i = 0; i < nlanes; ++i) {
                out.lanes.push_back(sb_.emit_bitcast(blk_id, ValType::F32, lane_of(a, i)));
            }
            break;
        }
        case 121: // OpConvertPtrToU
        case 122: // OpConvertUToPtr
        case 124: { // OpBitcast
            LoweredValue a = arg_v(3);
            for (uint32_t i = 0; i < nlanes; ++i) {
                out.lanes.push_back(sb_.emit_bitcast(blk_id, elem_vt, lane_of(a, i)));
            }
            break;
        }

        // Arithmetic
        case 126: emit_unary(SsaOp::SNeg); break;
        case 127: emit_unary(SsaOp::FNeg); break;
        case 128: emit_binary(is_64 ? SsaOp::IAdd64 : SsaOp::IAdd); break;
        case 129: emit_binary(SsaOp::FAdd); break;
        case 130: emit_binary(is_64 ? SsaOp::ISub64 : SsaOp::ISub); break;
        case 131: emit_binary(SsaOp::FSub); break;
        case 132: emit_binary(is_64 ? SsaOp::IMul64 : SsaOp::IMul); break;
        case 133: emit_binary(SsaOp::FMul); break;
        case 136: emit_binary(SsaOp::FDiv); break;
        case 142: { // OpVectorTimesScalar
            LoweredValue v = arg_v(3), s = arg_v(4);
            for (uint32_t i = 0; i < nlanes; ++i) {
                out.lanes.push_back(emit_op(blk_id, SsaOp::FMul, ValType::F32, {lane_of(v, i), s.scalar()}));
            }
            break;
        }
        case 145: { // OpMatrixTimesVector
            LoweredValue m = arg_v(3), v = arg_v(4);
            const SpirvType* mt = mod_.get_type(m.spv_type_id);
            uint32_t ncols = mt ? mt->count : 4u;
            uint32_t nrows = nlanes;
            for (uint32_t r = 0; r < nrows; ++r) {
                SsaId acc = emit_op(blk_id, SsaOp::FMul, ValType::F32, {lane_of(m, r), lane_of(v, 0)});
                for (uint32_t c = 1; c < ncols; ++c) {
                    acc = emit_op(blk_id, SsaOp::Fma, ValType::F32,
                                  {lane_of(m, c * nrows + r), lane_of(v, c), acc});
                }
                out.lanes.push_back(acc);
            }
            break;
        }
        case 148: { // OpDot
            LoweredValue a = arg_v(3), b = arg_v(4);
            SsaId acc = emit_op(blk_id, SsaOp::FMul, ValType::F32, {lane_of(a, 0), lane_of(b, 0)});
            for (size_t i = 1; i < a.lanes.size(); ++i) {
                SsaId prod = emit_op(blk_id, SsaOp::FMul, ValType::F32, {lane_of(a, i), lane_of(b, i)});
                acc = emit_op(blk_id, SsaOp::FAdd, ValType::F32, {acc, prod});
            }
            out.lanes.push_back(acc);
            break;
        }

        // Relational & Logical
        case 154: { // OpAny
            LoweredValue a = arg_v(3);
            SsaId acc = lane_of(a, 0);
            for (size_t i = 1; i < a.lanes.size(); ++i) {
                acc = emit_op(blk_id, SsaOp::LogicalOr, ValType::Bool, {acc, a.lanes[i]});
            }
            out.lanes.push_back(acc);
            break;
        }
        case 155: { // OpAll
            LoweredValue a = arg_v(3);
            SsaId acc = lane_of(a, 0);
            for (size_t i = 1; i < a.lanes.size(); ++i) {
                acc = emit_op(blk_id, SsaOp::LogicalAnd, ValType::Bool, {acc, a.lanes[i]});
            }
            out.lanes.push_back(acc);
            break;
        }
        case 156: emit_unary(SsaOp::IsNan); break;
        case 157: emit_unary(SsaOp::IsInf); break;
        case 164: { // OpLogicalEqual
            LoweredValue a = arg_v(3), b = arg_v(4);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId na = emit_op(blk_id, SsaOp::LogicalNot, ValType::Bool, {lane_of(a, i)});
                SsaId nb = emit_op(blk_id, SsaOp::LogicalNot, ValType::Bool, {lane_of(b, i)});
                SsaId both_t = emit_op(blk_id, SsaOp::LogicalAnd, ValType::Bool, {lane_of(a, i), lane_of(b, i)});
                SsaId both_f = emit_op(blk_id, SsaOp::LogicalAnd, ValType::Bool, {na, nb});
                out.lanes.push_back(emit_op(blk_id, SsaOp::LogicalOr, ValType::Bool, {both_t, both_f}));
            }
            break;
        }
        case 165: { // OpLogicalNotEqual
            LoweredValue a = arg_v(3), b = arg_v(4);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId na = emit_op(blk_id, SsaOp::LogicalNot, ValType::Bool, {lane_of(a, i)});
                SsaId nb = emit_op(blk_id, SsaOp::LogicalNot, ValType::Bool, {lane_of(b, i)});
                SsaId tf = emit_op(blk_id, SsaOp::LogicalAnd, ValType::Bool, {lane_of(a, i), nb});
                SsaId ft = emit_op(blk_id, SsaOp::LogicalAnd, ValType::Bool, {na, lane_of(b, i)});
                out.lanes.push_back(emit_op(blk_id, SsaOp::LogicalOr, ValType::Bool, {tf, ft}));
            }
            break;
        }
        case 166: emit_binary(SsaOp::LogicalOr); break;
        case 167: emit_binary(SsaOp::LogicalAnd); break;
        case 168: emit_unary(SsaOp::LogicalNot); break;
        case 169: { // OpSelect
            LoweredValue c = arg_v(3), t = arg_v(4), f = arg_v(5);
            for (uint32_t i = 0; i < nlanes; ++i) {
                out.lanes.push_back(emit_op(blk_id, SsaOp::Select, elem_vt,
                                            {lane_of(c, i), lane_of(t, i), lane_of(f, i)}));
            }
            break;
        }

        // Integer & Float Comparisons
        case 170: emit_cmp(SsaOp::ICmpEq); break;
        case 171: emit_cmp(SsaOp::ICmpNe); break;
        case 172: emit_cmp(SsaOp::ICmpUGt); break;
        case 173: emit_cmp(SsaOp::ICmpSGt); break;
        case 174: emit_cmp(SsaOp::ICmpUGe); break;
        case 175: emit_cmp(SsaOp::ICmpSGe); break;
        case 176: emit_cmp(SsaOp::ICmpULt); break;
        case 177: emit_cmp(SsaOp::ICmpSLt); break;
        case 178: emit_cmp(SsaOp::ICmpULe); break;
        case 179: emit_cmp(SsaOp::ICmpSLe); break;
        case 180: emit_cmp(SsaOp::FCmpOrdEq); break;
        case 182: emit_cmp(SsaOp::FCmpOrdNe); break;
        case 183: emit_cmp(SsaOp::FCmpUnordNe); break;
        case 184: emit_cmp(SsaOp::FCmpOrdLt); break;
        case 186: emit_cmp(SsaOp::FCmpOrdGt); break;
        case 188: emit_cmp(SsaOp::FCmpOrdLe); break;
        case 190: emit_cmp(SsaOp::FCmpOrdGe); break;
        case 181: // OpFUnordEqual
        case 185: // OpFUnordLessThan
        case 187: // OpFUnordGreaterThan
        case 189: // OpFUnordLessThanEqual
        case 191: { // OpFUnordGreaterThanEqual
            SsaOp ord_op = (op == 181) ? SsaOp::FCmpOrdEq
                         : (op == 185) ? SsaOp::FCmpOrdLt
                         : (op == 187) ? SsaOp::FCmpOrdGt
                         : (op == 189) ? SsaOp::FCmpOrdLe
                                       : SsaOp::FCmpOrdGe;
            LoweredValue a = arg_v(3), b = arg_v(4);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId la = lane_of(a, i), lb = lane_of(b, i);
                SsaId cmp = emit_op(blk_id, ord_op, ValType::Bool, {la, lb});
                SsaId na = emit_op(blk_id, SsaOp::IsNan, ValType::Bool, {la});
                SsaId nb = emit_op(blk_id, SsaOp::IsNan, ValType::Bool, {lb});
                SsaId any_nan = emit_op(blk_id, SsaOp::LogicalOr, ValType::Bool, {na, nb});
                out.lanes.push_back(emit_op(blk_id, SsaOp::LogicalOr, ValType::Bool, {cmp, any_nan}));
            }
            break;
        }

        // Bitwise & Shifts
        case 194: emit_binary(is_64 ? SsaOp::ShrU64 : SsaOp::ShrU); break;
        case 195: emit_binary(is_64 ? SsaOp::ShrS64 : SsaOp::ShrS); break;
        case 196: emit_binary(is_64 ? SsaOp::Shl64 : SsaOp::Shl); break;
        case 197: emit_binary(is_64 ? SsaOp::BitOr64 : SsaOp::BitOr); break;
        case 198: emit_binary(is_64 ? SsaOp::BitXor64 : SsaOp::BitXor); break;
        case 199: emit_binary(is_64 ? SsaOp::BitAnd64 : SsaOp::BitAnd); break;
        case 200: emit_unary(is_64 ? SsaOp::BitNot64 : SsaOp::BitNot); break;
        case 201: { // OpBitFieldInsert
            LoweredValue base = arg_v(3), ins_v = arg_v(4), off = arg_v(5), cnt = arg_v(6);
            for (uint32_t i = 0; i < nlanes; ++i) {
                out.lanes.push_back(emit_op(blk_id, SsaOp::Bfi, elem_vt,
                                            {lane_of(base, i), lane_of(ins_v, i), lane_of(off, i), lane_of(cnt, i)}));
            }
            break;
        }
        case 202: { // OpBitFieldSExtract
            LoweredValue base = arg_v(3), off = arg_v(4), cnt = arg_v(5);
            for (uint32_t i = 0; i < nlanes; ++i) {
                out.lanes.push_back(emit_op(blk_id, SsaOp::BfeS, elem_vt,
                                            {lane_of(base, i), lane_of(off, i), lane_of(cnt, i)}));
            }
            break;
        }
        case 203: { // OpBitFieldUExtract
            LoweredValue base = arg_v(3), off = arg_v(4), cnt = arg_v(5);
            for (uint32_t i = 0; i < nlanes; ++i) {
                out.lanes.push_back(emit_op(blk_id, SsaOp::BfeU, elem_vt,
                                            {lane_of(base, i), lane_of(off, i), lane_of(cnt, i)}));
            }
            break;
        }
        case 204: emit_unary(SsaOp::BitReverse); break;
        case 205: emit_unary(SsaOp::BitCount); break;

        default:
            record_unsupported("SpvOp#" + std::to_string(op));
            if (!opts_.tolerant) return false;
            out = make_zero_value(blk_id, res_type);
            break;
    }

    values_[res_id] = std::move(out);
    return true;
}

} // namespace valhall::spirv2ssa

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

// GLSL.std.450 extended instruction opcodes (Khronos GLSL.std.450 spec).
enum Glsl450Op : uint32_t {
    kGlslRound = 1,
    kGlslRoundEven = 2,
    kGlslTrunc = 3,
    kGlslFAbs = 4,
    kGlslSAbs = 5,
    kGlslFSign = 6,
    kGlslSSign = 7,
    kGlslFloor = 8,
    kGlslCeil = 9,
    kGlslFract = 10,
    kGlslRadians = 11,
    kGlslDegrees = 12,
    kGlslSin = 13,
    kGlslCos = 14,
    kGlslTan = 15,
    kGlslAsin = 16,
    kGlslAcos = 17,
    kGlslAtan = 18,
    kGlslSinh = 19,
    kGlslCosh = 20,
    kGlslTanh = 21,
    kGlslAtan2 = 25,
    kGlslPow = 26,
    kGlslExp = 27,
    kGlslLog = 28,
    kGlslExp2 = 29,
    kGlslLog2 = 30,
    kGlslSqrt = 31,
    kGlslInverseSqrt = 32,
    kGlslFMin = 37,
    kGlslUMin = 38,
    kGlslSMin = 39,
    kGlslFMax = 40,
    kGlslUMax = 41,
    kGlslSMax = 42,
    kGlslFClamp = 43,
    kGlslUClamp = 44,
    kGlslSClamp = 45,
    kGlslFMix = 46,
    kGlslStep = 48,
    kGlslSmoothStep = 49,
    kGlslFma = 50,
    kGlslFrexp = 51,
    kGlslFrexpStruct = 52,
    kGlslLdexp = 53,
    kGlslPackSnorm4x8 = 54,
    kGlslPackUnorm4x8 = 55,
    kGlslPackSnorm2x16 = 56,
    kGlslPackUnorm2x16 = 57,
    kGlslPackHalf2x16 = 58,
    kGlslUnpackSnorm2x16 = 60,
    kGlslUnpackUnorm2x16 = 61,
    kGlslUnpackHalf2x16 = 62,
    kGlslUnpackSnorm4x8 = 63,
    kGlslUnpackUnorm4x8 = 64,
    kGlslLength = 66,
    kGlslDistance = 67,
    kGlslCross = 68,
    kGlslNormalize = 69,
    kGlslFaceForward = 70,
    kGlslReflect = 71,
    kGlslRefract = 72,
    kGlslFindILsb = 73,
    kGlslFindSMsb = 74,
    kGlslFindUMsb = 75,
    kGlslNMin = 79,
    kGlslNMax = 80,
    kGlslNClamp = 81,
};

} // namespace

bool SpirvLowerer::lower_glsl450_ext_inst(uint32_t blk_id, const SpirvInst& inst) {
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
        case kGlslRound:
        case kGlslRoundEven: emit_unary(SsaOp::RoundEven); break;
        case kGlslTrunc: emit_unary(SsaOp::Trunc); break;
        case kGlslFAbs: emit_unary(SsaOp::FAbs); break;
        case kGlslSAbs: {
            LoweredValue a = arg_val(0);
            const SpirvType* src_et = mod_.get_type(scalar_elem_type(a.spv_type_id));
            const SpirvType* dst_et = mod_.get_type(scalar_elem_type(res_type));
            uint32_t sw = src_et ? src_et->width : 32u;
            uint32_t dw = dst_et ? dst_et->width : 32u;
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId x = lane_of(a, i);
                if (sw < 32u) {
                    x = emit_op(blk_id, SsaOp::BfeS, ValType::S32, {sb_.emit_bitcast(blk_id, ValType::U32, x), sb_.emit_const_u32(blk_id, 0u), sb_.emit_const_u32(blk_id, sw)});
                }
                SsaId r = emit_op(blk_id, SsaOp::SAbs, elem_vt, {x});
                if (dw < 32u) {
                    r = emit_op(blk_id, SsaOp::BfeU, elem_vt, {sb_.emit_bitcast(blk_id, ValType::U32, r), sb_.emit_const_u32(blk_id, 0u), sb_.emit_const_u32(blk_id, dw)});
                }
                out.lanes.push_back(r);
            }
            break;
        }
        case kGlslFSign: {
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
        case kGlslSSign: {
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
        case kGlslFloor: emit_unary(SsaOp::Floor); break;
        case kGlslCeil: emit_unary(SsaOp::Ceil); break;
        case kGlslFract: {
            LoweredValue a = arg_val(0);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId x = lane_of(a, i);
                SsaId fl = emit_op(blk_id, SsaOp::Floor, ValType::F32, {x});
                out.lanes.push_back(emit_op(blk_id, SsaOp::FSub, ValType::F32, {x, fl}));
            }
            break;
        }
        case kGlslRadians: {
            LoweredValue a = arg_val(0);
            SsaId k = sb_.emit_const_f32_bits(blk_id, f32_to_bits(0.017453292519943295f));
            for (uint32_t i = 0; i < nlanes; ++i) {
                out.lanes.push_back(emit_op(blk_id, SsaOp::FMul, ValType::F32, {lane_of(a, i), k}));
            }
            break;
        }
        case kGlslDegrees: {
            LoweredValue a = arg_val(0);
            SsaId k = sb_.emit_const_f32_bits(blk_id, f32_to_bits(57.29577951308232f));
            for (uint32_t i = 0; i < nlanes; ++i) {
                out.lanes.push_back(emit_op(blk_id, SsaOp::FMul, ValType::F32, {lane_of(a, i), k}));
            }
            break;
        }
        case kGlslSin: emit_unary(SsaOp::Sin); break;
        case kGlslCos: emit_unary(SsaOp::Cos); break;
        case kGlslTan: {
            LoweredValue a = arg_val(0);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId x = lane_of(a, i);
                SsaId s = emit_op(blk_id, SsaOp::Sin, ValType::F32, {x});
                SsaId c = emit_op(blk_id, SsaOp::Cos, ValType::F32, {x});
                out.lanes.push_back(emit_op(blk_id, SsaOp::FDiv, ValType::F32, {s, c}));
            }
            break;
        }
        case kGlslAsin: emit_unary(SsaOp::Asin); break;
        case kGlslAcos: emit_unary(SsaOp::Acos); break;
        case kGlslAtan: emit_unary(SsaOp::Atan); break;
        case kGlslSinh: emit_unary(SsaOp::Sinh); break;
        case kGlslCosh: emit_unary(SsaOp::Cosh); break;
        case kGlslTanh: emit_unary(SsaOp::Tanh); break;
        case kGlslAtan2: emit_binary(SsaOp::Atan2); break;
        case kGlslPow: emit_binary(SsaOp::Pow); break;
        case kGlslExp: emit_unary(SsaOp::Exp); break;
        case kGlslLog: emit_unary(SsaOp::Log); break;
        case kGlslExp2: emit_unary(SsaOp::Exp2); break;
        case kGlslLog2: emit_unary(SsaOp::Log2); break;
        case kGlslSqrt: emit_unary(SsaOp::Sqrt); break;
        case kGlslInverseSqrt: emit_unary(SsaOp::InverseSqrt); break;
        case kGlslFMin: emit_binary(SsaOp::NMin); break;
        case kGlslUMin: {
            LoweredValue a = arg_val(0), b = arg_val(1);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId c = emit_op(blk_id, SsaOp::ICmpULt, ValType::Bool, {lane_of(a, i), lane_of(b, i)});
                out.lanes.push_back(emit_op(blk_id, SsaOp::Select, elem_vt, {c, lane_of(a, i), lane_of(b, i)}));
            }
            break;
        }
        case kGlslSMin: {
            LoweredValue a = arg_val(0), b = arg_val(1);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId c = emit_op(blk_id, SsaOp::ICmpSLt, ValType::Bool, {lane_of(a, i), lane_of(b, i)});
                out.lanes.push_back(emit_op(blk_id, SsaOp::Select, elem_vt, {c, lane_of(a, i), lane_of(b, i)}));
            }
            break;
        }
        case kGlslFMax: emit_binary(SsaOp::NMax); break;
        case kGlslUMax: {
            LoweredValue a = arg_val(0), b = arg_val(1);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId c = emit_op(blk_id, SsaOp::ICmpUGt, ValType::Bool, {lane_of(a, i), lane_of(b, i)});
                out.lanes.push_back(emit_op(blk_id, SsaOp::Select, elem_vt, {c, lane_of(a, i), lane_of(b, i)}));
            }
            break;
        }
        case kGlslSMax: {
            LoweredValue a = arg_val(0), b = arg_val(1);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId c = emit_op(blk_id, SsaOp::ICmpSGt, ValType::Bool, {lane_of(a, i), lane_of(b, i)});
                out.lanes.push_back(emit_op(blk_id, SsaOp::Select, elem_vt, {c, lane_of(a, i), lane_of(b, i)}));
            }
            break;
        }
        case kGlslFClamp:
        case kGlslNClamp: {
            LoweredValue x = arg_val(0), lo = arg_val(1), hi = arg_val(2);
            for (uint32_t i = 0; i < nlanes; ++i) {
                out.lanes.push_back(emit_op(blk_id, SsaOp::FClamp, ValType::F32,
                                            {lane_of(x, i), lane_of(lo, i), lane_of(hi, i)}));
            }
            break;
        }
        case kGlslUClamp: {
            LoweredValue x = arg_val(0), lo = arg_val(1), hi = arg_val(2);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId c1 = emit_op(blk_id, SsaOp::ICmpUGt, ValType::Bool, {lane_of(x, i), lane_of(lo, i)});
                SsaId mx = emit_op(blk_id, SsaOp::Select, elem_vt, {c1, lane_of(x, i), lane_of(lo, i)});
                SsaId c2 = emit_op(blk_id, SsaOp::ICmpULt, ValType::Bool, {mx, lane_of(hi, i)});
                out.lanes.push_back(emit_op(blk_id, SsaOp::Select, elem_vt, {c2, mx, lane_of(hi, i)}));
            }
            break;
        }
        case kGlslSClamp: {
            LoweredValue x = arg_val(0), lo = arg_val(1), hi = arg_val(2);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId c1 = emit_op(blk_id, SsaOp::ICmpSGt, ValType::Bool, {lane_of(x, i), lane_of(lo, i)});
                SsaId mx = emit_op(blk_id, SsaOp::Select, elem_vt, {c1, lane_of(x, i), lane_of(lo, i)});
                SsaId c2 = emit_op(blk_id, SsaOp::ICmpSLt, ValType::Bool, {mx, lane_of(hi, i)});
                out.lanes.push_back(emit_op(blk_id, SsaOp::Select, elem_vt, {c2, mx, lane_of(hi, i)}));
            }
            break;
        }
        case kGlslFMix: {
            LoweredValue x = arg_val(0), y = arg_val(1), a = arg_val(2);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId d = emit_op(blk_id, SsaOp::FSub, ValType::F32, {lane_of(y, i), lane_of(x, i)});
                out.lanes.push_back(emit_op(blk_id, SsaOp::Fma, ValType::F32, {lane_of(a, i), d, lane_of(x, i)}));
            }
            break;
        }
        case kGlslStep: {
            LoweredValue edge = arg_val(0), x = arg_val(1);
            SsaId z = sb_.emit_const_f32_bits(blk_id, 0u);
            SsaId one = sb_.emit_const_f32_bits(blk_id, f32_to_bits(1.0f));
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId lt = emit_op(blk_id, SsaOp::FCmpOrdLt, ValType::Bool, {lane_of(x, i), lane_of(edge, i)});
                out.lanes.push_back(emit_op(blk_id, SsaOp::Select, ValType::F32, {lt, z, one}));
            }
            break;
        }
        case kGlslSmoothStep: {
            LoweredValue e0 = arg_val(0), e1 = arg_val(1), x = arg_val(2);
            SsaId z = sb_.emit_const_f32_bits(blk_id, 0u);
            SsaId one = sb_.emit_const_f32_bits(blk_id, f32_to_bits(1.0f));
            SsaId two = sb_.emit_const_f32_bits(blk_id, f32_to_bits(2.0f));
            SsaId three = sb_.emit_const_f32_bits(blk_id, f32_to_bits(3.0f));
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId num = emit_op(blk_id, SsaOp::FSub, ValType::F32, {lane_of(x, i), lane_of(e0, i)});
                SsaId den = emit_op(blk_id, SsaOp::FSub, ValType::F32, {lane_of(e1, i), lane_of(e0, i)});
                SsaId q = emit_op(blk_id, SsaOp::FDiv, ValType::F32, {num, den});
                SsaId t = emit_op(blk_id, SsaOp::FClamp, ValType::F32, {q, z, one});
                SsaId t2 = emit_op(blk_id, SsaOp::FMul, ValType::F32, {t, t});
                SsaId two_t = emit_op(blk_id, SsaOp::FMul, ValType::F32, {two, t});
                SsaId term = emit_op(blk_id, SsaOp::FSub, ValType::F32, {three, two_t});
                out.lanes.push_back(emit_op(blk_id, SsaOp::FMul, ValType::F32, {t2, term}));
            }
            break;
        }
        case kGlslFma: {
            LoweredValue a = arg_val(0), b = arg_val(1), c = arg_val(2);
            for (uint32_t i = 0; i < nlanes; ++i) {
                out.lanes.push_back(emit_op(blk_id, SsaOp::Fma, ValType::F32,
                                            {lane_of(a, i), lane_of(b, i), lane_of(c, i)}));
            }
            break;
        }
        case kGlslFrexpStruct: {
            LoweredValue x = arg_val(0);
            for (size_t i = 0; i < x.lanes.size(); ++i) {
                out.lanes.push_back(emit_op(blk_id, SsaOp::FrexpSignificand, ValType::F32, {x.lanes[i]}));
            }
            for (size_t i = 0; i < x.lanes.size(); ++i) {
                out.lanes.push_back(emit_op(blk_id, SsaOp::FrexpExp, ValType::S32, {x.lanes[i]}));
            }
            break;
        }
        case kGlslLdexp: emit_binary(SsaOp::Ldexp); break;
        case kGlslPackHalf2x16: {
            LoweredValue v = arg_val(0);
            SsaId v2 = pack_to_vec2_f32(blk_id, v);
            out.lanes.push_back(emit_op(blk_id, SsaOp::PackHalf2x16, ValType::U32, {v2}));
            break;
        }
        case kGlslUnpackHalf2x16: {
            LoweredValue u = arg_val(0);
            SsaId v2 = emit_op(blk_id, SsaOp::UnpackHalf2x16, ValType::V2F32, {u.scalar()});
            for (uint32_t c = 0; c < 2; ++c) {
                SsaNode ex{}; ex.op = SsaOp::VecExtract; ex.type = ValType::F32; ex.imm_u32 = c; ex.args = {v2};
                out.lanes.push_back(sb_.emit(blk_id, std::move(ex)));
            }
            break;
        }
        case kGlslPackUnorm4x8:
        case kGlslPackSnorm4x8: {
            LoweredValue v = arg_val(0);
            bool is_signed = (ext_op == kGlslPackSnorm4x8);
            SsaId lo = sb_.emit_const_f32_bits(blk_id, is_signed ? f32_to_bits(-1.0f) : 0u);
            SsaId hi = sb_.emit_const_f32_bits(blk_id, f32_to_bits(1.0f));
            SsaId scale = sb_.emit_const_f32_bits(blk_id, f32_to_bits(is_signed ? 127.0f : 255.0f));
            SsaId acc = sb_.emit_const_u32(blk_id, 0u);
            for (uint32_t c = 0; c < 4; ++c) {
                SsaId cl = emit_op(blk_id, SsaOp::FClamp, ValType::F32, {lane_of(v, c), lo, hi});
                SsaId sc = emit_op(blk_id, SsaOp::FMul, ValType::F32, {cl, scale});
                SsaId rd = emit_op(blk_id, SsaOp::RoundEven, ValType::F32, {sc});
                SsaId iv = emit_op(blk_id, is_signed ? SsaOp::ConvertFToS : SsaOp::ConvertFToU, ValType::U32, {rd});
                SsaId byte = emit_op(blk_id, SsaOp::BitAnd, ValType::U32, {iv, sb_.emit_const_u32(blk_id, 0xFFu)});
                SsaId sh = emit_op(blk_id, SsaOp::Shl, ValType::U32, {byte, sb_.emit_const_u32(blk_id, c * 8u)});
                acc = emit_op(blk_id, SsaOp::BitOr, ValType::U32, {acc, sh});
            }
            out.lanes.push_back(acc);
            break;
        }
        case kGlslUnpackUnorm4x8:
        case kGlslUnpackSnorm4x8: {
            LoweredValue u = arg_val(0);
            bool is_signed = (ext_op == kGlslUnpackSnorm4x8);
            SsaId inv = sb_.emit_const_f32_bits(blk_id, f32_to_bits(is_signed ? (1.0f / 127.0f) : (1.0f / 255.0f)));
            SsaId m1 = sb_.emit_const_f32_bits(blk_id, f32_to_bits(-1.0f));
            for (uint32_t c = 0; c < 4; ++c) {
                SsaId b = emit_op(blk_id, is_signed ? SsaOp::BfeS : SsaOp::BfeU,
                                  is_signed ? ValType::S32 : ValType::U32,
                                  {u.scalar(), sb_.emit_const_u32(blk_id, c * 8u), sb_.emit_const_u32(blk_id, 8u)});
                SsaId fv = emit_op(blk_id, is_signed ? SsaOp::ConvertSToF : SsaOp::ConvertUToF, ValType::F32, {b});
                SsaId scaled = emit_op(blk_id, SsaOp::FMul, ValType::F32, {fv, inv});
                out.lanes.push_back(is_signed ? emit_op(blk_id, SsaOp::FMax, ValType::F32, {scaled, m1}) : scaled);
            }
            break;
        }
        case kGlslPackUnorm2x16:
        case kGlslPackSnorm2x16: {
            LoweredValue v = arg_val(0);
            bool is_signed = (ext_op == kGlslPackSnorm2x16);
            SsaId lo = sb_.emit_const_f32_bits(blk_id, is_signed ? f32_to_bits(-1.0f) : 0u);
            SsaId hi = sb_.emit_const_f32_bits(blk_id, f32_to_bits(1.0f));
            SsaId scale = sb_.emit_const_f32_bits(blk_id, f32_to_bits(is_signed ? 32767.0f : 65535.0f));
            SsaId acc = sb_.emit_const_u32(blk_id, 0u);
            for (uint32_t c = 0; c < 2; ++c) {
                SsaId cl = emit_op(blk_id, SsaOp::FClamp, ValType::F32, {lane_of(v, c), lo, hi});
                SsaId sc = emit_op(blk_id, SsaOp::FMul, ValType::F32, {cl, scale});
                SsaId rd = emit_op(blk_id, SsaOp::RoundEven, ValType::F32, {sc});
                SsaId iv = emit_op(blk_id, is_signed ? SsaOp::ConvertFToS : SsaOp::ConvertFToU, ValType::U32, {rd});
                SsaId half = emit_op(blk_id, SsaOp::BitAnd, ValType::U32, {iv, sb_.emit_const_u32(blk_id, 0xFFFFu)});
                SsaId sh = emit_op(blk_id, SsaOp::Shl, ValType::U32, {half, sb_.emit_const_u32(blk_id, c * 16u)});
                acc = emit_op(blk_id, SsaOp::BitOr, ValType::U32, {acc, sh});
            }
            out.lanes.push_back(acc);
            break;
        }
        case kGlslUnpackUnorm2x16:
        case kGlslUnpackSnorm2x16: {
            LoweredValue u = arg_val(0);
            bool is_signed = (ext_op == kGlslUnpackSnorm2x16);
            SsaId inv = sb_.emit_const_f32_bits(blk_id, f32_to_bits(is_signed ? (1.0f / 32767.0f) : (1.0f / 65535.0f)));
            SsaId m1 = sb_.emit_const_f32_bits(blk_id, f32_to_bits(-1.0f));
            for (uint32_t c = 0; c < 2; ++c) {
                SsaId h = emit_op(blk_id, is_signed ? SsaOp::BfeS : SsaOp::BfeU,
                                  is_signed ? ValType::S32 : ValType::U32,
                                  {u.scalar(), sb_.emit_const_u32(blk_id, c * 16u), sb_.emit_const_u32(blk_id, 16u)});
                SsaId fv = emit_op(blk_id, is_signed ? SsaOp::ConvertSToF : SsaOp::ConvertUToF, ValType::F32, {h});
                SsaId scaled = emit_op(blk_id, SsaOp::FMul, ValType::F32, {fv, inv});
                out.lanes.push_back(is_signed ? emit_op(blk_id, SsaOp::FMax, ValType::F32, {scaled, m1}) : scaled);
            }
            break;
        }
        case kGlslLength: {
            LoweredValue v = arg_val(0);
            SsaId acc = emit_op(blk_id, SsaOp::FMul, ValType::F32, {v.lanes[0], v.lanes[0]});
            for (size_t i = 1; i < v.lanes.size(); ++i) {
                acc = emit_op(blk_id, SsaOp::Fma, ValType::F32, {v.lanes[i], v.lanes[i], acc});
            }
            out.lanes.push_back(emit_op(blk_id, SsaOp::Sqrt, ValType::F32, {acc}));
            break;
        }
        case kGlslDistance: {
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
        case kGlslCross: {
            LoweredValue a = arg_val(0), b = arg_val(1);
            auto sub_prod = [&](uint32_t i, uint32_t j) {
                SsaId p1 = emit_op(blk_id, SsaOp::FMul, ValType::F32, {lane_of(a, i), lane_of(b, j)});
                SsaId p2 = emit_op(blk_id, SsaOp::FMul, ValType::F32, {lane_of(a, j), lane_of(b, i)});
                return emit_op(blk_id, SsaOp::FSub, ValType::F32, {p1, p2});
            };
            out.lanes.push_back(sub_prod(1, 2));
            out.lanes.push_back(sub_prod(2, 0));
            out.lanes.push_back(sub_prod(0, 1));
            break;
        }
        case kGlslNormalize: {
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
        case kGlslReflect: {
            LoweredValue iv = arg_val(0), nv = arg_val(1);
            SsaId dot = emit_op(blk_id, SsaOp::FMul, ValType::F32, {lane_of(nv, 0), lane_of(iv, 0)});
            for (uint32_t i = 1; i < nlanes; ++i) {
                dot = emit_op(blk_id, SsaOp::Fma, ValType::F32, {lane_of(nv, i), lane_of(iv, i), dot});
            }
            SsaId two_dot = emit_op(blk_id, SsaOp::FMul, ValType::F32, {sb_.emit_const_f32_bits(blk_id, f32_to_bits(2.0f)), dot});
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId scaled = emit_op(blk_id, SsaOp::FMul, ValType::F32, {two_dot, lane_of(nv, i)});
                out.lanes.push_back(emit_op(blk_id, SsaOp::FSub, ValType::F32, {lane_of(iv, i), scaled}));
            }
            break;
        }
        case kGlslFaceForward: {
            // FaceForward(N, I, Nref): returns N if dot(Nref, I) < 0, else -N.
            LoweredValue nv = arg_val(0), iv = arg_val(1), nref = arg_val(2);
            SsaId dot = emit_op(blk_id, SsaOp::FMul, ValType::F32, {lane_of(nref, 0), lane_of(iv, 0)});
            for (uint32_t i = 1; i < nlanes; ++i) {
                dot = emit_op(blk_id, SsaOp::Fma, ValType::F32, {lane_of(nref, i), lane_of(iv, i), dot});
            }
            SsaId lt = emit_op(blk_id, SsaOp::FCmpOrdLt, ValType::Bool, {dot, sb_.emit_const_f32_bits(blk_id, 0u)});
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId neg_n = emit_op(blk_id, SsaOp::FNeg, ValType::F32, {lane_of(nv, i)});
                out.lanes.push_back(emit_op(blk_id, SsaOp::Select, ValType::F32, {lt, lane_of(nv, i), neg_n}));
            }
            break;
        }
        case kGlslRefract: {
            // Refract(I, N, eta): k = 1 - eta^2 * (1 - dot(N,I)^2); k < 0 ? 0 : eta*I - (eta*dot(N,I) + sqrt(k))*N
            LoweredValue iv = arg_val(0), nv = arg_val(1), eta_v = arg_val(2);
            SsaId eta = eta_v.scalar();
            SsaId one = sb_.emit_const_f32_bits(blk_id, f32_to_bits(1.0f));
            SsaId zero = sb_.emit_const_f32_bits(blk_id, 0u);
            SsaId dot = emit_op(blk_id, SsaOp::FMul, ValType::F32, {lane_of(nv, 0), lane_of(iv, 0)});
            for (uint32_t i = 1; i < nlanes; ++i) {
                dot = emit_op(blk_id, SsaOp::Fma, ValType::F32, {lane_of(nv, i), lane_of(iv, i), dot});
            }
            SsaId dot2 = emit_op(blk_id, SsaOp::FMul, ValType::F32, {dot, dot});
            SsaId om_d2 = emit_op(blk_id, SsaOp::FSub, ValType::F32, {one, dot2});
            SsaId eta2 = emit_op(blk_id, SsaOp::FMul, ValType::F32, {eta, eta});
            SsaId k = emit_op(blk_id, SsaOp::FSub, ValType::F32, {one, emit_op(blk_id, SsaOp::FMul, ValType::F32, {eta2, om_d2})});
            SsaId k_lt = emit_op(blk_id, SsaOp::FCmpOrdLt, ValType::Bool, {k, zero});
            SsaId sk = emit_op(blk_id, SsaOp::Sqrt, ValType::F32, {emit_op(blk_id, SsaOp::FMax, ValType::F32, {k, zero})});
            SsaId scale_n = emit_op(blk_id, SsaOp::Fma, ValType::F32, {eta, dot, sk});
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId t1 = emit_op(blk_id, SsaOp::FMul, ValType::F32, {eta, lane_of(iv, i)});
                SsaId t2 = emit_op(blk_id, SsaOp::FMul, ValType::F32, {scale_n, lane_of(nv, i)});
                SsaId r = emit_op(blk_id, SsaOp::FSub, ValType::F32, {t1, t2});
                out.lanes.push_back(emit_op(blk_id, SsaOp::Select, ValType::F32, {k_lt, zero, r}));
            }
            break;
        }
        case kGlslFindILsb: {
            LoweredValue a = arg_val(0);
            SsaId z = sb_.emit_const_u32(blk_id, 0u);
            SsaId m1 = sb_.emit_const_u32(blk_id, 0xFFFFFFFFu);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId x = lane_of(a, i);
                SsaId is_z = emit_op(blk_id, SsaOp::ICmpEq, ValType::Bool, {x, z});
                SsaId rev = emit_op(blk_id, SsaOp::BitReverse, ValType::U32, {x});
                SsaId clz = emit_op(blk_id, SsaOp::Clz, ValType::U32, {rev});
                out.lanes.push_back(emit_op(blk_id, SsaOp::Select, elem_vt, {is_z, m1, clz}));
            }
            break;
        }
        case kGlslFindUMsb: {
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
        case kGlslFindSMsb: {
            LoweredValue a = arg_val(0);
            SsaId z = sb_.emit_const_s32(blk_id, 0);
            SsaId m1 = sb_.emit_const_s32(blk_id, -1);
            SsaId c31 = sb_.emit_const_u32(blk_id, 31u);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId x = lane_of(a, i);
                SsaId is_neg = emit_op(blk_id, SsaOp::ICmpSLt, ValType::Bool, {x, z});
                SsaId nx = emit_op(blk_id, SsaOp::BitNot, ValType::U32, {x});
                SsaId val = emit_op(blk_id, SsaOp::Select, ValType::U32, {is_neg, nx, x});
                SsaId is_z = emit_op(blk_id, SsaOp::ICmpEq, ValType::Bool, {val, sb_.emit_const_u32(blk_id, 0u)});
                SsaId clz = emit_op(blk_id, SsaOp::Clz, ValType::U32, {val});
                SsaId msb = emit_op(blk_id, SsaOp::ISub, ValType::U32, {c31, clz});
                out.lanes.push_back(emit_op(blk_id, SsaOp::Select, elem_vt, {is_z, m1, msb}));
            }
            break;
        }
        case kGlslNMin: emit_binary(SsaOp::NMin); break;
        case kGlslNMax: emit_binary(SsaOp::NMax); break;
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
    const auto op = static_cast<SpvOp>(inst.opcode);
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
        case SpvOpVectorExtractDynamic: {
            LoweredValue vec = arg_v(3), idx = arg_v(4);
            SsaId acc = lane_of(vec, 0);
            for (uint32_t i = 1; i < vec.lanes.size(); ++i) {
                SsaId cmp = emit_op(blk_id, SsaOp::ICmpEq, ValType::Bool, {idx.scalar(), sb_.emit_const_u32(blk_id, i)});
                acc = emit_op(blk_id, SsaOp::Select, elem_vt, {cmp, vec.lanes[i], acc});
            }
            out.lanes.push_back(acc);
            break;
        }
        case SpvOpVectorInsertDynamic: {
            LoweredValue vec = arg_v(3), comp = arg_v(4), idx = arg_v(5);
            for (uint32_t i = 0; i < vec.lanes.size(); ++i) {
                SsaId cmp = emit_op(blk_id, SsaOp::ICmpEq, ValType::Bool, {idx.scalar(), sb_.emit_const_u32(blk_id, i)});
                out.lanes.push_back(emit_op(blk_id, SsaOp::Select, elem_vt, {cmp, comp.scalar(), vec.lanes[i]}));
            }
            break;
        }
        case SpvOpVectorShuffle: {
            LoweredValue v1 = arg_v(3), v2 = arg_v(4);
            std::vector<SsaId> combined = v1.lanes;
            combined.insert(combined.end(), v2.lanes.begin(), v2.lanes.end());
            for (size_t k = 5; k < inst.words.size(); ++k) {
                uint32_t c = inst.words[k];
                out.lanes.push_back((c < combined.size()) ? combined[c] : sb_.emit_const_u32(blk_id, 0u));
            }
            break;
        }
        case SpvOpCompositeConstruct: {
            for (size_t k = 3; k < inst.words.size(); ++k) {
                LoweredValue cv = get_value(blk_id, inst.words[k]);
                out.lanes.insert(out.lanes.end(), cv.lanes.begin(), cv.lanes.end());
            }
            break;
        }
        case SpvOpCompositeExtract: {
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
        case SpvOpCompositeInsert: {
            LoweredValue obj = arg_v(3), comp = arg_v(4);
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
        case SpvOpCopyObject:
        case SpvOpCopyLogical:
            out = arg_v(3);
            out.spv_type_id = res_type;
            break;
        case SpvOpTranspose: {
            LoweredValue m = arg_v(3);
            const SpirvType* rt = mod_.get_type(res_type);
            uint32_t dst_cols = rt ? rt->count : 4u;
            uint32_t dst_rows = rt ? flat_lane_count(rt->elem_type) : 4u;
            for (uint32_t c = 0; c < dst_cols; ++c) {
                for (uint32_t r = 0; r < dst_rows; ++r) {
                    out.lanes.push_back(lane_of(m, r * dst_cols + c));
                }
            }
            break;
        }
        case SpvOpSampledImage:
        case SpvOpImage: {
            uint32_t img_id = inst.words[3];
            if (auto it = sampled_images_.find(img_id); it != sampled_images_.end()) {
                sampled_images_[res_id] = it->second;
            }
            out.lanes.push_back(sb_.emit_const_u32(blk_id, img_id));
            break;
        }

        // Conversions
        case SpvOpConvertFToU: emit_unary(SsaOp::ConvertFToU); break;
        case SpvOpConvertFToS: emit_unary(SsaOp::ConvertFToS); break;
        case SpvOpConvertSToF: {
            LoweredValue a = arg_v(3);
            const SpirvType* src_et = mod_.get_type(scalar_elem_type(a.spv_type_id));
            uint32_t sw = src_et ? src_et->width : 32u;
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId x = lane_of(a, i);
                if (sw < 32u) {
                    x = emit_op(blk_id, SsaOp::BfeS, ValType::S32, {sb_.emit_bitcast(blk_id, ValType::U32, x), sb_.emit_const_u32(blk_id, 0u), sb_.emit_const_u32(blk_id, sw)});
                }
                out.lanes.push_back(emit_op(blk_id, SsaOp::ConvertSToF, elem_vt, {x}));
            }
            break;
        }
        case SpvOpConvertUToF: emit_unary(SsaOp::ConvertUToF); break;
        case SpvOpUConvert: {
            LoweredValue a = arg_v(3);
            const SpirvType* src_et = mod_.get_type(scalar_elem_type(a.spv_type_id));
            uint32_t sw = src_et ? src_et->width : 32u;
            uint32_t dw = et ? et->width : 32u;
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId x = lane_of(a, i);
                if (sw == 64 && dw <= 32) {
                    x = emit_op(blk_id, SsaOp::UConvert64To32, ValType::U32, {x});
                    if (dw < 32) {
                        x = emit_op(blk_id, SsaOp::BfeU, ValType::U32,
                                    {x, sb_.emit_const_u32(blk_id, 0u), sb_.emit_const_u32(blk_id, dw)});
                    }
                    out.lanes.push_back(x);
                } else if (sw <= 32 && dw == 64) {
                    if (sw < 32) {
                        x = emit_op(blk_id, SsaOp::BfeU, ValType::U32,
                                    {x, sb_.emit_const_u32(blk_id, 0u), sb_.emit_const_u32(blk_id, sw)});
                    }
                    out.lanes.push_back(emit_op(blk_id, SsaOp::UConvert32To64, ValType::U64, {x}));
                } else {
                    uint32_t keep = std::min(sw, dw);
                    if (keep < 32) {
                        out.lanes.push_back(emit_op(blk_id, SsaOp::BfeU, ValType::U32,
                                                    {x, sb_.emit_const_u32(blk_id, 0u), sb_.emit_const_u32(blk_id, keep)}));
                    } else {
                        out.lanes.push_back(sb_.emit_bitcast(blk_id, elem_vt, x));
                    }
                }
            }
            break;
        }
        case SpvOpSConvert: {
            LoweredValue a = arg_v(3);
            const SpirvType* src_et = mod_.get_type(scalar_elem_type(a.spv_type_id));
            uint32_t sw = src_et ? src_et->width : 32u;
            uint32_t dw = et ? et->width : 32u;
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId x = lane_of(a, i);
                if (sw == 64 && dw <= 32) {
                    x = emit_op(blk_id, SsaOp::UConvert64To32, ValType::S32, {x});
                    if (dw < 32) {
                        x = emit_op(blk_id, SsaOp::BfeS, ValType::S32,
                                    {x, sb_.emit_const_u32(blk_id, 0u), sb_.emit_const_u32(blk_id, dw)});
                    }
                    out.lanes.push_back(x);
                } else if (sw <= 32 && dw == 64) {
                    if (sw < 32) {
                        x = emit_op(blk_id, SsaOp::BfeS, ValType::S32,
                                    {x, sb_.emit_const_u32(blk_id, 0u), sb_.emit_const_u32(blk_id, sw)});
                    }
                    out.lanes.push_back(emit_op(blk_id, SsaOp::SConvert32To64, ValType::S64, {x}));
                } else {
                    uint32_t keep = std::min(sw, dw);
                    if (keep < 32) {
                        out.lanes.push_back(emit_op(blk_id, SsaOp::BfeS, ValType::S32,
                                                    {x, sb_.emit_const_u32(blk_id, 0u), sb_.emit_const_u32(blk_id, keep)}));
                    } else {
                        out.lanes.push_back(sb_.emit_bitcast(blk_id, elem_vt, x));
                    }
                }
            }
            break;
        }
        case SpvOpFConvert: {
            // SPIR-V §3.3.11: Numeric floating-point width conversion (f16 <-> f32).
            LoweredValue a = arg_v(3);
            const SpirvType* src_et = mod_.get_type(scalar_elem_type(a.spv_type_id));
            uint32_t sw = src_et ? src_et->width : 32u;
            uint32_t dw = et ? et->width : 32u;
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId x = lane_of(a, i);
                if (sw == 16 && dw == 32) {
                    SsaId v2 = emit_op(blk_id, SsaOp::UnpackHalf2x16, ValType::V2F32, {sb_.emit_bitcast(blk_id, ValType::U32, x)});
                    SsaNode ex{}; ex.op = SsaOp::VecExtract; ex.type = ValType::F32; ex.imm_u32 = 0; ex.args = {v2};
                    out.lanes.push_back(sb_.emit(blk_id, std::move(ex)));
                } else if (sw == 32 && dw == 16) {
                    SsaId z = sb_.emit_const_f32_bits(blk_id, 0u);
                    SsaId v2 = emit_op(blk_id, SsaOp::VecConstruct2, ValType::V2F32, {sb_.emit_bitcast(blk_id, ValType::F32, x), z});
                    SsaId pk = emit_op(blk_id, SsaOp::PackHalf2x16, ValType::U32, {v2});
                    out.lanes.push_back(emit_op(blk_id, SsaOp::BitAnd, ValType::U32, {pk, sb_.emit_const_u32(blk_id, 0xFFFFu)}));
                } else {
                    out.lanes.push_back(sb_.emit_bitcast(blk_id, ValType::F32, x));
                }
            }
            break;
        }
        case SpvOpQuantizeToF16: {
            // SPIR-V §3.3.11: Quantize f32 to f16 representable value (flushes denorms to 0).
            LoweredValue a = arg_v(3);
            SsaId z = sb_.emit_const_f32_bits(blk_id, 0u);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId v2 = emit_op(blk_id, SsaOp::VecConstruct2, ValType::V2F32, {lane_of(a, i), z});
                SsaId pk = emit_op(blk_id, SsaOp::PackHalf2x16, ValType::U32, {v2});
                SsaId up = emit_op(blk_id, SsaOp::UnpackHalf2x16, ValType::V2F32, {pk});
                SsaNode ex{}; ex.op = SsaOp::VecExtract; ex.type = ValType::F32; ex.imm_u32 = 0; ex.args = {up};
                out.lanes.push_back(sb_.emit(blk_id, std::move(ex)));
            }
            break;
        }
        case SpvOpConvertPtrToU:
        case SpvOpConvertUToPtr:
        case SpvOpBitcast: {
            // SPIR-V §3.3.11: Support both same-count and cross-component-count bitcasts (e.g. uvec2 <-> u64).
            LoweredValue a = arg_v(3);
            const SpirvType* src_et = mod_.get_type(scalar_elem_type(a.spv_type_id));
            uint32_t sw = src_et ? src_et->width : 32u;
            uint32_t dw = et ? et->width : 32u;
            if (sw == 32 && dw == 64 && a.lanes.size() == nlanes * 2) {
                for (uint32_t i = 0; i < nlanes; ++i) {
                    SsaId lo = sb_.emit_bitcast(blk_id, ValType::U32, a.lanes[2 * i]);
                    SsaId hi = sb_.emit_bitcast(blk_id, ValType::U32, a.lanes[2 * i + 1]);
                    out.lanes.push_back(emit_op(blk_id, SsaOp::Pack64, elem_vt, {lo, hi}));
                }
            } else if (sw == 64 && dw == 32 && nlanes == a.lanes.size() * 2) {
                for (size_t i = 0; i < a.lanes.size(); ++i) {
                    SsaNode sp{}; sp.op = SsaOp::Split64; sp.type = ValType::TupleU32x2; sp.args = {a.lanes[i]};
                    SsaId tup = sb_.emit(blk_id, std::move(sp));
                    SsaNode e0{}; e0.op = SsaOp::TupleExtract; e0.type = ValType::U32; e0.imm_u32 = 0; e0.args = {tup};
                    SsaNode e1{}; e1.op = SsaOp::TupleExtract; e1.type = ValType::U32; e1.imm_u32 = 1; e1.args = {tup};
                    out.lanes.push_back(sb_.emit_bitcast(blk_id, elem_vt, sb_.emit(blk_id, std::move(e0))));
                    out.lanes.push_back(sb_.emit_bitcast(blk_id, elem_vt, sb_.emit(blk_id, std::move(e1))));
                }
            } else if (sw == 32 && dw == 16 && nlanes == a.lanes.size() * 2) {
                bool dst_f16 = (et && et->kind == SpirvTypeKind::Float);
                for (size_t i = 0; i < a.lanes.size(); ++i) {
                    SsaId u = sb_.emit_bitcast(blk_id, ValType::U32, a.lanes[i]);
                    if (dst_f16) {
                        SsaId v2 = emit_op(blk_id, SsaOp::UnpackHalf2x16, ValType::V2F32, {u});
                        SsaNode e0{}; e0.op = SsaOp::VecExtract; e0.type = ValType::F32; e0.imm_u32 = 0; e0.args = {v2};
                        SsaNode e1{}; e1.op = SsaOp::VecExtract; e1.type = ValType::F32; e1.imm_u32 = 1; e1.args = {v2};
                        out.lanes.push_back(sb_.emit(blk_id, std::move(e0)));
                        out.lanes.push_back(sb_.emit(blk_id, std::move(e1)));
                    } else {
                        out.lanes.push_back(emit_op(blk_id, SsaOp::BfeU, ValType::U32, {u, sb_.emit_const_u32(blk_id, 0u), sb_.emit_const_u32(blk_id, 16u)}));
                        out.lanes.push_back(emit_op(blk_id, SsaOp::BfeU, ValType::U32, {u, sb_.emit_const_u32(blk_id, 16u), sb_.emit_const_u32(blk_id, 16u)}));
                    }
                }
            } else if (sw == 16 && dw == 32 && a.lanes.size() == nlanes * 2) {
                bool src_f16 = (src_et && src_et->kind == SpirvTypeKind::Float);
                for (uint32_t i = 0; i < nlanes; ++i) {
                    if (src_f16) {
                        SsaId v2 = emit_op(blk_id, SsaOp::VecConstruct2, ValType::V2F32, {a.lanes[2 * i], a.lanes[2 * i + 1]});
                        SsaId pk = emit_op(blk_id, SsaOp::PackHalf2x16, ValType::U32, {v2});
                        out.lanes.push_back(sb_.emit_bitcast(blk_id, elem_vt, pk));
                    } else {
                        SsaId lo = emit_op(blk_id, SsaOp::BitAnd, ValType::U32, {sb_.emit_bitcast(blk_id, ValType::U32, a.lanes[2 * i]), sb_.emit_const_u32(blk_id, 0xFFFFu)});
                        SsaId hi = emit_op(blk_id, SsaOp::Shl, ValType::U32, {sb_.emit_bitcast(blk_id, ValType::U32, a.lanes[2 * i + 1]), sb_.emit_const_u32(blk_id, 16u)});
                        out.lanes.push_back(sb_.emit_bitcast(blk_id, elem_vt, emit_op(blk_id, SsaOp::BitOr, ValType::U32, {lo, hi})));
                    }
                }
            } else if (sw == 32 && dw == 8 && nlanes == a.lanes.size() * 4) {
                for (size_t i = 0; i < a.lanes.size(); ++i) {
                    SsaId u = sb_.emit_bitcast(blk_id, ValType::U32, a.lanes[i]);
                    for (uint32_t b = 0; b < 4; ++b) {
                        out.lanes.push_back(emit_op(blk_id, SsaOp::BfeU, ValType::U32, {u, sb_.emit_const_u32(blk_id, b * 8u), sb_.emit_const_u32(blk_id, 8u)}));
                    }
                }
            } else if (sw == 8 && dw == 32 && a.lanes.size() == nlanes * 4) {
                for (uint32_t i = 0; i < nlanes; ++i) {
                    SsaId acc = emit_op(blk_id, SsaOp::BitAnd, ValType::U32, {sb_.emit_bitcast(blk_id, ValType::U32, a.lanes[4 * i]), sb_.emit_const_u32(blk_id, 0xFFu)});
                    for (uint32_t b = 1; b < 4; ++b) {
                        SsaId lane = emit_op(blk_id, SsaOp::BitAnd, ValType::U32, {sb_.emit_bitcast(blk_id, ValType::U32, a.lanes[4 * i + b]), sb_.emit_const_u32(blk_id, 0xFFu)});
                        SsaId sh = emit_op(blk_id, SsaOp::Shl, ValType::U32, {lane, sb_.emit_const_u32(blk_id, b * 8u)});
                        acc = emit_op(blk_id, SsaOp::BitOr, ValType::U32, {acc, sh});
                    }
                    out.lanes.push_back(sb_.emit_bitcast(blk_id, elem_vt, acc));
                }
            } else {
                for (uint32_t i = 0; i < nlanes; ++i) {
                    out.lanes.push_back(sb_.emit_bitcast(blk_id, elem_vt, lane_of(a, i)));
                }
            }
            break;
        }

        // Arithmetic
        case SpvOpSNegate: emit_unary(SsaOp::SNeg); break;
        case SpvOpFNegate: emit_unary(SsaOp::FNeg); break;
        case SpvOpIAdd: {
            emit_binary(is_64 ? SsaOp::IAdd64 : SsaOp::IAdd);
            uint32_t dw = et ? et->width : 32u;
            if (!is_64 && dw < 32u) {
                SsaOp mask_op = (et && et->is_signed) ? SsaOp::BfeS : SsaOp::BfeU;
                for (auto& lid : out.lanes) {
                    lid = emit_op(blk_id, mask_op, elem_vt, {sb_.emit_bitcast(blk_id, ValType::U32, lid), sb_.emit_const_u32(blk_id, 0u), sb_.emit_const_u32(blk_id, dw)});
                }
            }
            break;
        }
        case SpvOpFAdd: emit_binary(SsaOp::FAdd); break;
        case SpvOpISub: emit_binary(is_64 ? SsaOp::ISub64 : SsaOp::ISub); break;
        case SpvOpFSub: emit_binary(SsaOp::FSub); break;
        case SpvOpIMul: emit_binary(is_64 ? SsaOp::IMul64 : SsaOp::IMul); break;
        case SpvOpFMul: emit_binary(SsaOp::FMul); break;
        case SpvOpFDiv: emit_binary(SsaOp::FDiv); break;
        case SpvOpUDiv: emit_binary(SsaOp::UDiv); break;
        case SpvOpSDiv: emit_binary(SsaOp::SDiv); break;
        case SpvOpUMod: emit_binary(SsaOp::UMod); break;
        case SpvOpSRem: emit_binary(SsaOp::SRem); break;
        case SpvOpSMod: emit_binary(SsaOp::SMod); break;
        case SpvOpFRem:
        case SpvOpFMod: {
            LoweredValue a = arg_v(3), b = arg_v(4);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId q = emit_op(blk_id, SsaOp::FDiv, ValType::F32, {lane_of(a, i), lane_of(b, i)});
                SsaId r = emit_op(blk_id, (op == SpvOpFMod) ? SsaOp::Floor : SsaOp::Trunc, ValType::F32, {q});
                SsaId prod = emit_op(blk_id, SsaOp::FMul, ValType::F32, {r, lane_of(b, i)});
                out.lanes.push_back(emit_op(blk_id, SsaOp::FSub, ValType::F32, {lane_of(a, i), prod}));
            }
            break;
        }
        case SpvOpIAddCarry: {
            LoweredValue a = arg_v(3), b = arg_v(4);
            for (size_t i = 0; i < a.lanes.size(); ++i) {
                SsaId s = emit_op(blk_id, SsaOp::IAdd, ValType::U32, {a.lanes[i], b.lanes[i]});
                SsaId c = emit_op(blk_id, SsaOp::ICmpULt, ValType::Bool, {s, a.lanes[i]});
                SsaId cu = emit_op(blk_id, SsaOp::Select, ValType::U32, {c, sb_.emit_const_u32(blk_id, 1u), sb_.emit_const_u32(blk_id, 0u)});
                out.lanes.push_back(s);
                out.lanes.push_back(cu);
            }
            break;
        }
        case SpvOpISubBorrow: {
            LoweredValue a = arg_v(3), b = arg_v(4);
            for (size_t i = 0; i < a.lanes.size(); ++i) {
                SsaId d = emit_op(blk_id, SsaOp::ISub, ValType::U32, {a.lanes[i], b.lanes[i]});
                SsaId br = emit_op(blk_id, SsaOp::ICmpULt, ValType::Bool, {a.lanes[i], b.lanes[i]});
                SsaId bu = emit_op(blk_id, SsaOp::Select, ValType::U32, {br, sb_.emit_const_u32(blk_id, 1u), sb_.emit_const_u32(blk_id, 0u)});
                out.lanes.push_back(d);
                out.lanes.push_back(bu);
            }
            break;
        }
        case SpvOpUMulExtended:
        case SpvOpSMulExtended: {
            LoweredValue a = arg_v(3), b = arg_v(4);
            SsaOp ext = (op == SpvOpSMulExtended) ? SsaOp::SConvert32To64 : SsaOp::UConvert32To64;
            for (size_t i = 0; i < a.lanes.size(); ++i) {
                SsaId a64 = emit_op(blk_id, ext, ValType::U64, {a.lanes[i]});
                SsaId b64 = emit_op(blk_id, ext, ValType::U64, {b.lanes[i]});
                SsaId p64 = emit_op(blk_id, SsaOp::IMul64, ValType::U64, {a64, b64});
                SsaNode sp{}; sp.op = SsaOp::Split64; sp.type = ValType::TupleU32x2; sp.args = {p64};
                SsaId tup = sb_.emit(blk_id, std::move(sp));
                SsaNode e0{}; e0.op = SsaOp::TupleExtract; e0.type = ValType::U32; e0.imm_u32 = 0; e0.args = {tup};
                SsaNode e1{}; e1.op = SsaOp::TupleExtract; e1.type = ValType::U32; e1.imm_u32 = 1; e1.args = {tup};
                out.lanes.push_back(sb_.emit(blk_id, std::move(e0)));
                out.lanes.push_back(sb_.emit(blk_id, std::move(e1)));
            }
            break;
        }
        case SpvOpVectorTimesScalar:
        case SpvOpMatrixTimesScalar: {
            LoweredValue v = arg_v(3), s = arg_v(4);
            for (uint32_t i = 0; i < nlanes; ++i) {
                out.lanes.push_back(emit_op(blk_id, SsaOp::FMul, ValType::F32, {lane_of(v, i), s.scalar()}));
            }
            break;
        }
        case SpvOpVectorTimesMatrix: {
            LoweredValue v = arg_v(3), m = arg_v(4);
            const SpirvType* mt = mod_.get_type(m.spv_type_id);
            uint32_t ncols = mt ? mt->count : nlanes;
            uint32_t nrows = static_cast<uint32_t>(v.lanes.size());
            for (uint32_t c = 0; c < ncols; ++c) {
                SsaId acc = emit_op(blk_id, SsaOp::FMul, ValType::F32, {lane_of(v, 0), lane_of(m, c * nrows)});
                for (uint32_t r = 1; r < nrows; ++r) {
                    acc = emit_op(blk_id, SsaOp::Fma, ValType::F32, {lane_of(v, r), lane_of(m, c * nrows + r), acc});
                }
                out.lanes.push_back(acc);
            }
            break;
        }
        case SpvOpMatrixTimesVector: {
            LoweredValue m = arg_v(3), v = arg_v(4);
            const SpirvType* mt = mod_.get_type(m.spv_type_id);
            uint32_t ncols = mt ? mt->count : 4u;
            uint32_t nrows = nlanes;
            for (uint32_t r = 0; r < nrows; ++r) {
                SsaId acc = emit_op(blk_id, SsaOp::FMul, ValType::F32, {lane_of(m, r), lane_of(v, 0)});
                for (uint32_t c = 1; c < ncols; ++c) {
                    acc = emit_op(blk_id, SsaOp::Fma, ValType::F32, {lane_of(m, c * nrows + r), lane_of(v, c), acc});
                }
                out.lanes.push_back(acc);
            }
            break;
        }
        case SpvOpMatrixTimesMatrix: {
            LoweredValue a = arg_v(3), b = arg_v(4);
            const SpirvType* at = mod_.get_type(a.spv_type_id);
            const SpirvType* bt = mod_.get_type(b.spv_type_id);
            uint32_t kdim = at ? at->count : 4u;
            uint32_t rows = at ? flat_lane_count(at->elem_type) : 4u;
            uint32_t cols = bt ? bt->count : 4u;
            for (uint32_t c = 0; c < cols; ++c) {
                for (uint32_t r = 0; r < rows; ++r) {
                    SsaId acc = emit_op(blk_id, SsaOp::FMul, ValType::F32, {lane_of(a, r), lane_of(b, c * kdim)});
                    for (uint32_t k = 1; k < kdim; ++k) {
                        acc = emit_op(blk_id, SsaOp::Fma, ValType::F32, {lane_of(a, k * rows + r), lane_of(b, c * kdim + k), acc});
                    }
                    out.lanes.push_back(acc);
                }
            }
            break;
        }
        case SpvOpOuterProduct: {
            LoweredValue a = arg_v(3), b = arg_v(4);
            for (size_t c = 0; c < b.lanes.size(); ++c) {
                for (size_t r = 0; r < a.lanes.size(); ++r) {
                    out.lanes.push_back(emit_op(blk_id, SsaOp::FMul, ValType::F32, {a.lanes[r], b.lanes[c]}));
                }
            }
            break;
        }
        case SpvOpDot: {
            LoweredValue a = arg_v(3), b = arg_v(4);
            SsaId acc = emit_op(blk_id, SsaOp::FMul, ValType::F32, {lane_of(a, 0), lane_of(b, 0)});
            for (size_t i = 1; i < a.lanes.size(); ++i) {
                SsaId prod = emit_op(blk_id, SsaOp::FMul, ValType::F32, {lane_of(a, i), lane_of(b, i)});
                acc = emit_op(blk_id, SsaOp::FAdd, ValType::F32, {acc, prod});
            }
            out.lanes.push_back(acc);
            break;
        }
        case SpvOpSDot: emit_binary(SsaOp::IDot4S); break;
        case SpvOpUDot: emit_binary(SsaOp::IDot4U); break;
        case SpvOpSUDot: emit_binary(SsaOp::IDot4SU); break;
        case SpvOpSDotAccSat:
        case SpvOpUDotAccSat:
        case SpvOpSUDotAccSat: {
            LoweredValue a = arg_v(3), b = arg_v(4), acc = arg_v(5);
            SsaOp dot_op = (op == SpvOpSDotAccSat) ? SsaOp::IDot4S : (op == SpvOpUDotAccSat ? SsaOp::IDot4U : SsaOp::IDot4SU);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId d = emit_op(blk_id, dot_op, elem_vt, {lane_of(a, i), lane_of(b, i)});
                out.lanes.push_back(emit_op(blk_id, SsaOp::IAdd, elem_vt, {d, lane_of(acc, i)}));
            }
            break;
        }

        // Relational & Logical
        case SpvOpAny: {
            LoweredValue a = arg_v(3);
            SsaId acc = lane_of(a, 0);
            for (size_t i = 1; i < a.lanes.size(); ++i) {
                acc = emit_op(blk_id, SsaOp::LogicalOr, ValType::Bool, {acc, a.lanes[i]});
            }
            out.lanes.push_back(acc);
            break;
        }
        case SpvOpAll: {
            LoweredValue a = arg_v(3);
            SsaId acc = lane_of(a, 0);
            for (size_t i = 1; i < a.lanes.size(); ++i) {
                acc = emit_op(blk_id, SsaOp::LogicalAnd, ValType::Bool, {acc, a.lanes[i]});
            }
            out.lanes.push_back(acc);
            break;
        }
        case SpvOpIsNan: emit_unary(SsaOp::IsNan); break;
        case SpvOpIsInf: emit_unary(SsaOp::IsInf); break;
        case SpvOpIsFinite: {
            LoweredValue a = arg_v(3);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId x = lane_of(a, i);
                SsaId bad = emit_op(blk_id, SsaOp::LogicalOr, ValType::Bool, {
                    emit_op(blk_id, SsaOp::IsNan, ValType::Bool, {x}),
                    emit_op(blk_id, SsaOp::IsInf, ValType::Bool, {x})
                });
                out.lanes.push_back(emit_op(blk_id, SsaOp::LogicalNot, ValType::Bool, {bad}));
            }
            break;
        }
        case SpvOpIsNormal: {
            // Normal IEEE-754 f32: exponent in [1, 254] (bits 23..30 neither 0 nor 0xFF).
            LoweredValue a = arg_v(3);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId bits = sb_.emit_bitcast(blk_id, ValType::U32, lane_of(a, i));
                SsaId exp = emit_op(blk_id, SsaOp::BfeU, ValType::U32, {bits, sb_.emit_const_u32(blk_id, 23u), sb_.emit_const_u32(blk_id, 8u)});
                SsaId nz = emit_op(blk_id, SsaOp::ICmpNe, ValType::Bool, {exp, sb_.emit_const_u32(blk_id, 0u)});
                SsaId nff = emit_op(blk_id, SsaOp::ICmpNe, ValType::Bool, {exp, sb_.emit_const_u32(blk_id, 0xFFu)});
                out.lanes.push_back(emit_op(blk_id, SsaOp::LogicalAnd, ValType::Bool, {nz, nff}));
            }
            break;
        }
        case SpvOpSignBitSet: {
            LoweredValue a = arg_v(3);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId bits = sb_.emit_bitcast(blk_id, ValType::U32, lane_of(a, i));
                SsaId sbit = emit_op(blk_id, SsaOp::ShrU, ValType::U32, {bits, sb_.emit_const_u32(blk_id, 31u)});
                out.lanes.push_back(emit_op(blk_id, SsaOp::ICmpNe, ValType::Bool, {sbit, sb_.emit_const_u32(blk_id, 0u)}));
            }
            break;
        }
        case SpvOpOrdered: {
            LoweredValue a = arg_v(3), b = arg_v(4);
            for (uint32_t i = 0; i < nlanes; ++i) {
                SsaId any_nan = emit_op(blk_id, SsaOp::LogicalOr, ValType::Bool, {
                    emit_op(blk_id, SsaOp::IsNan, ValType::Bool, {lane_of(a, i)}),
                    emit_op(blk_id, SsaOp::IsNan, ValType::Bool, {lane_of(b, i)})
                });
                out.lanes.push_back(emit_op(blk_id, SsaOp::LogicalNot, ValType::Bool, {any_nan}));
            }
            break;
        }
        case SpvOpUnordered: {
            LoweredValue a = arg_v(3), b = arg_v(4);
            for (uint32_t i = 0; i < nlanes; ++i) {
                out.lanes.push_back(emit_op(blk_id, SsaOp::LogicalOr, ValType::Bool, {
                    emit_op(blk_id, SsaOp::IsNan, ValType::Bool, {lane_of(a, i)}),
                    emit_op(blk_id, SsaOp::IsNan, ValType::Bool, {lane_of(b, i)})
                }));
            }
            break;
        }
        case SpvOpLogicalEqual: {
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
        case SpvOpLogicalNotEqual: {
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
        case SpvOpLogicalOr: emit_binary(SsaOp::LogicalOr); break;
        case SpvOpLogicalAnd: emit_binary(SsaOp::LogicalAnd); break;
        case SpvOpLogicalNot: emit_unary(SsaOp::LogicalNot); break;
        case SpvOpSelect: {
            LoweredValue c = arg_v(3), t = arg_v(4), f = arg_v(5);
            for (uint32_t i = 0; i < nlanes; ++i) {
                out.lanes.push_back(emit_op(blk_id, SsaOp::Select, elem_vt,
                                            {lane_of(c, i), lane_of(t, i), lane_of(f, i)}));
            }
            break;
        }

        // Integer & Float Comparisons
        case SpvOpIEqual: emit_cmp(SsaOp::ICmpEq); break;
        case SpvOpINotEqual: emit_cmp(SsaOp::ICmpNe); break;
        case SpvOpUGreaterThan: emit_cmp(SsaOp::ICmpUGt); break;
        case SpvOpSGreaterThan: emit_cmp(SsaOp::ICmpSGt); break;
        case SpvOpUGreaterThanEqual: emit_cmp(SsaOp::ICmpUGe); break;
        case SpvOpSGreaterThanEqual: emit_cmp(SsaOp::ICmpSGe); break;
        case SpvOpULessThan: emit_cmp(SsaOp::ICmpULt); break;
        case SpvOpSLessThan: emit_cmp(SsaOp::ICmpSLt); break;
        case SpvOpULessThanEqual: emit_cmp(SsaOp::ICmpULe); break;
        case SpvOpSLessThanEqual: emit_cmp(SsaOp::ICmpSLe); break;
        case SpvOpFOrdEqual: emit_cmp(SsaOp::FCmpOrdEq); break;
        case SpvOpFOrdNotEqual: emit_cmp(SsaOp::FCmpOrdNe); break;
        case SpvOpFUnordNotEqual: emit_cmp(SsaOp::FCmpUnordNe); break;
        case SpvOpFOrdLessThan: emit_cmp(SsaOp::FCmpOrdLt); break;
        case SpvOpFOrdGreaterThan: emit_cmp(SsaOp::FCmpOrdGt); break;
        case SpvOpFOrdLessThanEqual: emit_cmp(SsaOp::FCmpOrdLe); break;
        case SpvOpFOrdGreaterThanEqual: emit_cmp(SsaOp::FCmpOrdGe); break;
        case SpvOpFUnordEqual:
        case SpvOpFUnordLessThan:
        case SpvOpFUnordGreaterThan:
        case SpvOpFUnordLessThanEqual:
        case SpvOpFUnordGreaterThanEqual: {
            SsaOp ord_op = (op == SpvOpFUnordEqual) ? SsaOp::FCmpOrdEq
                         : (op == SpvOpFUnordLessThan) ? SsaOp::FCmpOrdLt
                         : (op == SpvOpFUnordGreaterThan) ? SsaOp::FCmpOrdGt
                         : (op == SpvOpFUnordLessThanEqual) ? SsaOp::FCmpOrdLe
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
        case SpvOpShiftRightLogical: emit_binary(is_64 ? SsaOp::ShrU64 : SsaOp::ShrU); break;
        case SpvOpShiftRightArithmetic: emit_binary(is_64 ? SsaOp::ShrS64 : SsaOp::ShrS); break;
        case SpvOpShiftLeftLogical: emit_binary(is_64 ? SsaOp::Shl64 : SsaOp::Shl); break;
        case SpvOpBitwiseOr: emit_binary(is_64 ? SsaOp::BitOr64 : SsaOp::BitOr); break;
        case SpvOpBitwiseXor: emit_binary(is_64 ? SsaOp::BitXor64 : SsaOp::BitXor); break;
        case SpvOpBitwiseAnd: emit_binary(is_64 ? SsaOp::BitAnd64 : SsaOp::BitAnd); break;
        case SpvOpNot: emit_unary(is_64 ? SsaOp::BitNot64 : SsaOp::BitNot); break;
        case SpvOpBitFieldInsert: {
            LoweredValue base = arg_v(3), ins_v = arg_v(4), off = arg_v(5), cnt = arg_v(6);
            for (uint32_t i = 0; i < nlanes; ++i) {
                out.lanes.push_back(emit_op(blk_id, SsaOp::Bfi, elem_vt,
                                            {lane_of(base, i), lane_of(ins_v, i), lane_of(off, i), lane_of(cnt, i)}));
            }
            break;
        }
        case SpvOpBitFieldSExtract: {
            LoweredValue base = arg_v(3), off = arg_v(4), cnt = arg_v(5);
            for (uint32_t i = 0; i < nlanes; ++i) {
                out.lanes.push_back(emit_op(blk_id, SsaOp::BfeS, elem_vt,
                                            {lane_of(base, i), lane_of(off, i), lane_of(cnt, i)}));
            }
            break;
        }
        case SpvOpBitFieldUExtract: {
            LoweredValue base = arg_v(3), off = arg_v(4), cnt = arg_v(5);
            for (uint32_t i = 0; i < nlanes; ++i) {
                out.lanes.push_back(emit_op(blk_id, SsaOp::BfeU, elem_vt,
                                            {lane_of(base, i), lane_of(off, i), lane_of(cnt, i)}));
            }
            break;
        }
        case SpvOpBitReverse: emit_unary(SsaOp::BitReverse); break;
        case SpvOpBitCount: emit_unary(SsaOp::BitCount); break;

        // Fragment derivatives
        case SpvOpDPdx:
        case SpvOpDPdxFine:
        case SpvOpDPdxCoarse: emit_unary(SsaOp::DPdx); break;
        case SpvOpDPdy:
        case SpvOpDPdyFine:
        case SpvOpDPdyCoarse: emit_unary(SsaOp::DPdy); break;
        case SpvOpFwidth:
        case SpvOpFwidthFine:
        case SpvOpFwidthCoarse: emit_unary(SsaOp::Fwidth); break;

        // Image metadata queries
        case SpvOpImageQueryLod: {
            uint32_t img_id = inst.words[3];
            uint32_t coord_id = inst.words[4];
            std::string ikey = "tex:0:0";
            if (auto it = sampled_images_.find(img_id); it != sampled_images_.end() && !it->second.canonical_key.empty()) {
                ikey = it->second.canonical_key;
            }
            LoweredValue coord = get_value(blk_id, coord_id);
            SsaId u = coord.lanes.empty() ? sb_.emit_const_f32_bits(blk_id, 0u) : sb_.emit_bitcast(blk_id, ValType::F32, coord.lanes[0]);
            SsaId v = (coord.lanes.size() >= 2) ? sb_.emit_bitcast(blk_id, ValType::F32, coord.lanes[1]) : sb_.emit_const_f32_bits(blk_id, 0u);
            SsaId uv = emit_op(blk_id, SsaOp::VecConstruct2, ValType::V2F32, {u, v});
            SsaNode q{};
            q.op = SsaOp::ImageQueryLod;
            q.type = ValType::TupleValToken;
            q.aux_str = ikey;
            q.args = {uv, sb_.current_token(blk_id)};
            q.is_barrier_or_cf = true;
            q.is_side_effect = true;
            SsaId tup = sb_.emit(blk_id, std::move(q));
            SsaNode ev{}; ev.op = SsaOp::TupleExtract; ev.type = ValType::V4F32; ev.imm_u32 = 0; ev.args = {tup};
            SsaId vec4 = sb_.emit(blk_id, std::move(ev));
            SsaNode et{}; et.op = SsaOp::TupleExtract; et.type = ValType::Token; et.imm_u32 = 1; et.args = {tup};
            sb_.set_token(blk_id, sb_.emit(blk_id, std::move(et)));
            for (uint32_t c = 0; c < nlanes; ++c) {
                SsaNode ex{};
                ex.op = SsaOp::VecExtract;
                ex.type = ValType::F32;
                ex.imm_u32 = std::min<uint32_t>(3u, c);
                ex.args = {vec4};
                out.lanes.push_back(sb_.emit_bitcast(blk_id, elem_vt, sb_.emit(blk_id, std::move(ex))));
            }
            break;
        }

        case SpvOpImageQuerySize:
        case SpvOpImageQuerySizeLod:
        case SpvOpImageQueryLevels:
        case SpvOpImageQuerySamples: {
            uint32_t img_id = inst.words[3];
            std::string ikey = "img_" + std::to_string(img_id);
            if (auto it = sampled_images_.find(img_id); it != sampled_images_.end() && !it->second.canonical_key.empty()) {
                ikey = it->second.canonical_key;
            }
            SsaId lod = (op == SpvOpImageQuerySizeLod && inst.words.size() >= 5)
                            ? get_value(blk_id, inst.words[4]).scalar()
                            : sb_.emit_const_u32(blk_id, 0u);
            for (uint32_t c = 0; c < nlanes; ++c) {
                SsaNode q{};
                q.op = SsaOp::ImageQuerySize;
                q.type = elem_vt;
                q.imm_u32 = c;
                q.aux_str = ikey + ":" + spirv_parser::spv_op_name(inst.opcode);
                q.args = {lod};
                out.lanes.push_back(sb_.emit(blk_id, std::move(q)));
            }
            break;
        }

        // Subgroup operations (SPIR-V §3.3.24)
        case SpvOpGroupNonUniformElect: {
            SsaNode lid{};
            lid.op = SsaOp::LoadBuiltinInput;
            lid.type = ValType::U32;
            lid.imm_u32 = static_cast<uint32_t>(SpvBuiltInSubgroupLocalInvocationId);
            lid.aux_str = "gl_SubgroupInvocationID";
            SsaId id_v = sb_.emit(blk_id, std::move(lid));
            out.lanes.push_back(emit_op(blk_id, SsaOp::ICmpEq, ValType::Bool, {id_v, sb_.emit_const_u32(blk_id, 0u)}));
            break;
        }
        case SpvOpGroupNonUniformAll:
        case SpvOpGroupNonUniformAny: {
            LoweredValue pred = arg_v(4);
            SsaId b = emit_op(blk_id, SsaOp::SubgroupBallot, ValType::U32, {pred.scalar()});
            if (op == SpvOpGroupNonUniformAny) {
                out.lanes.push_back(emit_op(blk_id, SsaOp::ICmpNe, ValType::Bool, {b, sb_.emit_const_u32(blk_id, 0u)}));
            } else {
                out.lanes.push_back(emit_op(blk_id, SsaOp::ICmpEq, ValType::Bool, {b, sb_.emit_const_u32(blk_id, 0xFFFFu)}));
            }
            break;
        }
        case SpvOpGroupNonUniformAllEqual: {
            LoweredValue v = arg_v(4);
            SsaId lane0 = sb_.emit_const_u32(blk_id, 0u);
            SsaId acc = sb_.emit_const_bool(blk_id, true);
            for (size_t i = 0; i < v.lanes.size(); ++i) {
                SsaId v0 = emit_op(blk_id, SsaOp::SubgroupShuffle, res_.ssa.get(v.lanes[i]).type, {v.lanes[i], lane0});
                SsaId eq = emit_op(blk_id, (res_.ssa.get(v.lanes[i]).type == ValType::F32) ? SsaOp::FCmpOrdEq : SsaOp::ICmpEq,
                                   ValType::Bool, {v.lanes[i], v0});
                acc = emit_op(blk_id, SsaOp::LogicalAnd, ValType::Bool, {acc, eq});
            }
            SsaId b = emit_op(blk_id, SsaOp::SubgroupBallot, ValType::U32, {acc});
            out.lanes.push_back(emit_op(blk_id, SsaOp::ICmpEq, ValType::Bool, {b, sb_.emit_const_u32(blk_id, 0xFFFFu)}));
            break;
        }
        case SpvOpGroupNonUniformBallot: {
            LoweredValue pred = arg_v(4);
            SsaId b = emit_op(blk_id, SsaOp::SubgroupBallot, ValType::U32, {pred.scalar()});
            out.lanes.push_back(b);
            for (uint32_t i = 1; i < nlanes; ++i) out.lanes.push_back(sb_.emit_const_u32(blk_id, 0u));
            break;
        }
        case SpvOpGroupNonUniformBallotBitCount: {
            LoweredValue mask = arg_v(5);
            out.lanes.push_back(emit_op(blk_id, SsaOp::BitCount, elem_vt, {mask.scalar()}));
            break;
        }
        case SpvOpGroupNonUniformBallotFindLSB: {
            LoweredValue mask = arg_v(4);
            SsaId rev = emit_op(blk_id, SsaOp::BitReverse, ValType::U32, {mask.scalar()});
            out.lanes.push_back(emit_op(blk_id, SsaOp::Clz, elem_vt, {rev}));
            break;
        }
        case SpvOpGroupNonUniformBallotFindMSB: {
            LoweredValue mask = arg_v(4);
            SsaId clz = emit_op(blk_id, SsaOp::Clz, ValType::U32, {mask.scalar()});
            out.lanes.push_back(emit_op(blk_id, SsaOp::ISub, elem_vt, {sb_.emit_const_u32(blk_id, 31u), clz}));
            break;
        }
        case SpvOpGroupNonUniformBroadcast:
        case SpvOpGroupNonUniformShuffle: {
            LoweredValue v = arg_v(4), idx = arg_v(5);
            for (uint32_t i = 0; i < nlanes; ++i) {
                out.lanes.push_back(emit_op(blk_id, SsaOp::SubgroupShuffle, elem_vt, {lane_of(v, i), idx.scalar()}));
            }
            break;
        }
        case SpvOpGroupNonUniformBroadcastFirst: {
            LoweredValue v = arg_v(4);
            SsaId z = sb_.emit_const_u32(blk_id, 0u);
            for (uint32_t i = 0; i < nlanes; ++i) {
                out.lanes.push_back(emit_op(blk_id, SsaOp::SubgroupShuffle, elem_vt, {lane_of(v, i), z}));
            }
            break;
        }
        case SpvOpGroupNonUniformShuffleXor:
        case SpvOpGroupNonUniformShuffleUp:
        case SpvOpGroupNonUniformShuffleDown: {
            LoweredValue v = arg_v(4), delta = arg_v(5);
            SsaNode lid{};
            lid.op = SsaOp::LoadBuiltinInput;
            lid.type = ValType::U32;
            lid.imm_u32 = static_cast<uint32_t>(SpvBuiltInSubgroupLocalInvocationId);
            lid.aux_str = "gl_SubgroupInvocationID";
            SsaId id_v = sb_.emit(blk_id, std::move(lid));
            SsaOp idx_op = (op == SpvOpGroupNonUniformShuffleXor) ? SsaOp::BitXor
                         : (op == SpvOpGroupNonUniformShuffleUp)  ? SsaOp::ISub
                                                                  : SsaOp::IAdd;
            SsaId src_lane = emit_op(blk_id, idx_op, ValType::U32, {id_v, delta.scalar()});
            for (uint32_t i = 0; i < nlanes; ++i) {
                out.lanes.push_back(emit_op(blk_id, SsaOp::SubgroupShuffle, elem_vt, {lane_of(v, i), src_lane}));
            }
            break;
        }
        case SpvOpGroupNonUniformIAdd:
        case SpvOpGroupNonUniformFAdd:
        case SpvOpGroupNonUniformSMin:
        case SpvOpGroupNonUniformUMin:
        case SpvOpGroupNonUniformFMin:
        case SpvOpGroupNonUniformSMax:
        case SpvOpGroupNonUniformUMax:
        case SpvOpGroupNonUniformFMax:
        case SpvOpGroupNonUniformBitwiseAnd:
        case SpvOpGroupNonUniformBitwiseOr:
        case SpvOpGroupNonUniformBitwiseXor: {
            LoweredValue v = arg_v(5);
            for (uint32_t i = 0; i < nlanes; ++i) {
                out.lanes.push_back(emit_op(blk_id, SsaOp::SubgroupShuffle, elem_vt,
                                            {lane_of(v, i), sb_.emit_const_u32(blk_id, inst.opcode)}));
            }
            break;
        }

        default:
            record_unsupported(spirv_parser::spv_op_name(inst.opcode));
            if (!opts_.tolerant) return false;
            out = make_zero_value(blk_id, res_type);
            break;
    }

    values_[res_id] = std::move(out);
    return true;
}

} // namespace valhall::spirv2ssa

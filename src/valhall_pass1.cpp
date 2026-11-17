#include "valhall_pass1.h"
#include "valhall_operands.h"
#include "valhall_tables_generated.h"
#include <algorithm>
#include <cstring>
#include <sstream>

namespace valhall::compiler {

using tables::OpClass;
using tables::SsaOp;

namespace {

struct Pass1Ctx {
    ValhallSsaModule& mod;
    SsaBuilder sb;
    const ValhallTranslateOptions& opts;
    Pass1Result& res;

    Pass1Ctx(ValhallSsaModule& m, const ValhallTranslateOptions& o, Pass1Result& r)
        : mod(m), sb(m), opts(o), res(r) {}

    SsaId emit_op(uint32_t blk, SsaOp op, ValType vt, std::vector<SsaId> args,
                  tables::EffectMask eff = {}, bool barrier = false, bool side_eff = false) {
        SsaNode n{};
        n.op = op;
        n.type = vt;
        n.args = std::move(args);
        n.effects = eff;
        n.is_barrier_or_cf = barrier;
        n.is_side_effect = side_eff;
        return sb.emit(blk, std::move(n));
    }

    SsaId emit_load_fau(uint32_t blk, uint32_t word_idx) {
        SsaNode n{};
        n.op = SsaOp::LoadFau;
        n.type = ValType::U32;
        n.imm_u32 = word_idx & 63u;
        n.effects.flags |= EffectSet::kFlagUsesSideband;
        res.uses_sideband = true;
        return sb.emit(blk, std::move(n));
    }

    SsaId emit_load_sideband_u64(uint32_t blk, uint32_t member_idx) {
        SsaNode n{};
        n.op = SsaOp::LoadSidebandU64;
        n.type = ValType::U64;
        n.imm_u32 = member_idx;
        n.effects.flags |= EffectSet::kFlagUsesSideband;
        n.effects.cap_bits |= EffectSet::kCapInt64;
        res.uses_sideband = true;
        return sb.emit(blk, std::move(n));
    }

    SsaId emit_builtin_input(uint32_t blk, spirv::BuiltIn bi, ValType vt, const char* name, bool flat = false) {
        SsaNode n{};
        n.op = SsaOp::LoadBuiltinInput;
        n.type = vt;
        n.imm_u32 = static_cast<uint32_t>(bi);
        n.aux_bool = flat;
        n.aux_str = name;
        return sb.emit(blk, std::move(n));
    }

    void emit_split64(uint32_t blk, SsaId val_u64, SsaId* lo, SsaId* hi) {
        SsaNode sp{};
        sp.op = SsaOp::Split64;
        sp.type = ValType::TupleU32x2;
        sp.args = { val_u64 };
        sp.effects.cap_bits |= EffectSet::kCapInt64;
        SsaId tup = sb.emit(blk, std::move(sp));

        SsaNode e0{}; e0.op = SsaOp::TupleExtract; e0.type = ValType::U32; e0.imm_u32 = 0; e0.args = { tup };
        SsaNode e1{}; e1.op = SsaOp::TupleExtract; e1.type = ValType::U32; e1.imm_u32 = 1; e1.args = { tup };
        *lo = sb.emit(blk, std::move(e0));
        *hi = sb.emit(blk, std::move(e1));
    }

    void emit_stage_preamble(uint32_t blk, const std::vector<DecodedValhallInst>& insts) {
        if (opts.stage == ShaderStage::Pilot) {
            SsaId fau_va = emit_load_sideband_u64(blk, 3);
            SsaId srt_va = emit_load_sideband_u64(blk, 2);
            SsaId f0 = 0, f1 = 0, s0 = 0, s1 = 0;
            emit_split64(blk, fau_va, &f0, &f1);
            emit_split64(blk, srt_va, &s0, &s1);
            sb.set_reg(blk, 0, f0);
            sb.set_reg(blk, 1, f1);
            sb.set_reg(blk, 2, s0);
            sb.set_reg(blk, 3, s1);
        } else if (opts.stage == ShaderStage::Compute) {
            bool reads_wg_regs = false;
            for (const auto& ins : insts) {
                for (const auto& s : ins.srcs) {
                    if (s.kind == SourceKind::Register && s.reg_num >= 55 && s.reg_num <= 57) {
                        reads_wg_regs = true;
                    }
                }
            }
            SsaId wgid = emit_builtin_input(blk, spirv::BuiltIn::WorkgroupId, ValType::V3U32, "gl_WorkGroupID");
            for (uint32_t c = 0; c < 3; ++c) {
                SsaNode ex{}; ex.op = SsaOp::VecExtract; ex.type = ValType::U32; ex.imm_u32 = c; ex.args = { wgid };
                sb.set_reg(blk, 55 + c, sb.emit(blk, std::move(ex)));
            }
            auto bi = reads_wg_regs ? spirv::BuiltIn::LocalInvocationId : spirv::BuiltIn::GlobalInvocationId;
            const char* bname = reads_wg_regs ? "gl_LocalInvocationID" : "gl_GlobalInvocationID";
            SsaId tid = emit_builtin_input(blk, bi, ValType::V3U32, bname);
            for (uint32_t c = 0; c < 3; ++c) {
                SsaNode ex{}; ex.op = SsaOp::VecExtract; ex.type = ValType::U32; ex.imm_u32 = c; ex.args = { tid };
                sb.set_reg(blk, 60 + c, sb.emit(blk, std::move(ex)));
            }
        } else if (opts.stage == ShaderStage::Vertex) {
            SsaId vi = emit_builtin_input(blk, spirv::BuiltIn::VertexIndex, ValType::U32, "gl_VertexIndex");
            SsaId ii = emit_builtin_input(blk, spirv::BuiltIn::InstanceIndex, ValType::U32, "gl_InstanceIndex");
            sb.set_reg(blk, 59, vi);
            sb.set_reg(blk, 60, vi);
            sb.set_reg(blk, 61, ii);
        } else if (opts.stage == ShaderStage::Fragment) {
            SsaId fc = emit_builtin_input(blk, spirv::BuiltIn::FragCoord, ValType::V4F32, "gl_FragCoord");
            SsaNode ex{}; ex.op = SsaOp::VecExtract; ex.type = ValType::F32; ex.imm_u32 = 2; ex.args = { fc };
            SsaId fz = sb.emit(blk, std::move(ex));
            sb.set_reg(blk, 59, sb.emit_bitcast(blk, ValType::U32, fz));
            sb.set_reg(blk, 60, sb.emit_const_u32(blk, 1u));
        }
    }

    SsaId eval_float_cmp(uint32_t blk, const std::string& cmpf, SsaId a, SsaId b) {
        SsaOp op = SsaOp::FCmpOrdEq;
        if (cmpf == "gt") op = SsaOp::FCmpOrdGt;
        else if (cmpf == "ge") op = SsaOp::FCmpOrdGe;
        else if (cmpf == "ne") op = SsaOp::FCmpUnordNe; // Corner case: Valhall .ne is unordered!
        else if (cmpf == "lt") op = SsaOp::FCmpOrdLt;
        else if (cmpf == "le") op = SsaOp::FCmpOrdLe;
        else if (cmpf == "gtlt") op = SsaOp::FCmpOrdNe; // Corner case: Valhall .gtlt is ordered not-equal!
        else if (cmpf == "total") op = SsaOp::FCmpTotal;
        return emit_op(blk, op, ValType::Bool, { a, b });
    }

    SsaId eval_int_cmp(uint32_t blk, const std::string& cmpf, SsaId a, SsaId b, bool is_signed) {
        SsaOp op = SsaOp::ICmpEq;
        if (cmpf == "ne") op = SsaOp::ICmpNe;
        else if (is_signed) {
            a = sb.emit_bitcast(blk, ValType::S32, a);
            b = sb.emit_bitcast(blk, ValType::S32, b);
            if (cmpf == "gt") op = SsaOp::ICmpSGt;
            else if (cmpf == "ge") op = SsaOp::ICmpSGe;
            else if (cmpf == "lt") op = SsaOp::ICmpSLt;
            else if (cmpf == "le") op = SsaOp::ICmpSLe;
        } else {
            if (cmpf == "gt") op = SsaOp::ICmpUGt;
            else if (cmpf == "ge") op = SsaOp::ICmpUGe;
            else if (cmpf == "lt") op = SsaOp::ICmpULt;
            else if (cmpf == "le") op = SsaOp::ICmpULe;
        }
        return emit_op(blk, op, ValType::Bool, { a, b });
    }

    SsaId format_cmp_result(uint32_t blk, const DecodedValhallInst& ins, SsaId cond_bool) {
        std::string rtype = "m1";
        auto it = ins.modifier_strs.find("result_type");
        if (it != ins.modifier_strs.end()) rtype = it->second;
        uint32_t tval = 0xFFFFFFFFu;
        if (rtype == "i1") tval = 1u;
        else if (rtype == "f1") tval = 0x3F800000u;
        return emit_op(blk, SsaOp::Select, ValType::U32,
                       { cond_bool, sb.emit_const_u32(blk, tval), sb.emit_const_u32(blk, 0u) });
    }

    SsaId eval_branch_cond(uint32_t blk, const DecodedValhallInst& ins) {
        if (ins.srcs.empty()) return sb.emit_const_bool(blk, true);
        SsaId s0 = fetch_src_int32(sb, blk, ins, 0, false);
        SsaId cond = emit_op(blk, SsaOp::ICmpNe, ValType::Bool, { s0, sb.emit_const_u32(blk, 0u) });

        std::string comb = "none";
        auto it_c = ins.modifier_strs.find("combine");
        if (it_c != ins.modifier_strs.end()) comb = it_c->second;
        if (ins.srcs.size() >= 2 && comb != "none") {
            SsaId s1 = fetch_src_int32(sb, blk, ins, 1, false);
            if (comb == "eq") {
                cond = emit_op(blk, SsaOp::ICmpEq, ValType::Bool, { s0, s1 });
            } else {
                SsaId c1 = emit_op(blk, SsaOp::ICmpNe, ValType::Bool, { s1, sb.emit_const_u32(blk, 0u) });
                cond = emit_op(blk, (comb == "and") ? SsaOp::LogicalAnd : SsaOp::LogicalOr, ValType::Bool, { cond, c1 });
            }
        }
        auto it_not = ins.fields.find("not");
        if (it_not != ins.fields.end() && it_not->second != 0) {
            cond = emit_op(blk, SsaOp::LogicalNot, ValType::Bool, { cond });
        }
        return cond;
    }

    void lower_instruction(uint32_t blk, const DecodedValhallInst& ins) {
        const auto* rule = tables::lookup_lowering_rule(ins.spec_index);
        if (!rule) {
            res.error_message = "Missing lowering rule for spec_index=" + std::to_string(ins.spec_index);
            return;
        }
        std::string var = variant_to_string(ins.canonical_variant);
        bool is_v2f16 = (ins.canonical_variant == Variant::V2F16 || ins.variant == Variant::V2F16 || var == "v2f16");
        ValType f_vt = is_v2f16 ? ValType::V2F32 : ValType::F32;

        auto fetch_fp = [&](size_t idx) -> SsaId {
            return is_v2f16 ? fetch_src_v2f16(sb, blk, ins, idx) : fetch_src_f32(sb, blk, ins, idx);
        };
        auto store_fp = [&](SsaId val) {
            if (is_v2f16) store_dest_v2f16(sb, blk, ins, val);
            else store_dest_f32(sb, blk, ins, val);
        };
        auto make_v2f_const = [&](float c) -> SsaId {
            uint32_t bits = 0; std::memcpy(&bits, &c, sizeof(float));
            SsaId s = sb.emit_const_f32_bits(blk, bits);
            return emit_op(blk, SsaOp::VecConstruct2, ValType::V2F32, { s, s });
        };

        switch (rule->op_class) {
            case OpClass::CfNop:
            case OpClass::MemNop:
                return;

            case OpClass::CfBranch:
                mod.blocks[blk].branch_cond = eval_branch_cond(blk, ins);
                return;

            case OpClass::CfJump:
                return;

            case OpClass::SysBarrier: {
                SsaNode n{};
                n.op = SsaOp::ControlBarrier;
                n.type = ValType::Token;
                n.is_barrier_or_cf = true;
                n.is_side_effect = true;
                n.args = { sb.emit_const_u32(blk, 2u), sb.emit_const_u32(blk, 2u),
                           sb.emit_const_u32(blk, 0x108u), sb.current_token(blk) };
                sb.set_token(blk, sb.emit(blk, std::move(n)));
                return;
            }

            case OpClass::SysDiscard:
            case OpClass::SysKaboom: {
                SsaNode n{};
                n.op = SsaOp::Kill;
                n.type = ValType::Void;
                n.is_barrier_or_cf = true;
                n.is_side_effect = true;
                n.args = { sb.current_token(blk) };
                sb.emit(blk, std::move(n));
                return;
            }

            case OpClass::AluAdr: {
                SsaId pc_base = emit_load_sideband_u64(blk, 1);
                int64_t tgt_pc = ins.has_branch_offset ? ins.branch_target_pc : (int64_t(ins.pc) + 8 + ins.imm_signed);
                SsaId full_addr = emit_op(blk, SsaOp::IAdd64, ValType::U64,
                                          { pc_base, sb.emit_const_u64(blk, static_cast<uint64_t>(tgt_pc)) },
                                          rule->effects);
                SsaId lo = 0, hi = 0;
                emit_split64(blk, full_addr, &lo, &hi);
                store_dest_u32(sb, blk, ins, lo);
                return;
            }

            case OpClass::AluMov: {
                store_dest_u32(sb, blk, ins, fetch_src_int32(sb, blk, ins, 0, false));
                return;
            }

            case OpClass::AluMovImm: {
                store_dest_u32(sb, blk, ins, sb.emit_const_u32(blk, static_cast<uint32_t>(ins.imm_signed)));
                return;
            }

            case OpClass::AluVecPack: {
                SsaId s0 = fetch_src_int32(sb, blk, ins, 0, false);
                if (ins.srcs.size() >= 2) {
                    SsaId s1 = fetch_src_int32(sb, blk, ins, 1, false);
                    SsaNode bfi{};
                    bfi.op = SsaOp::Bfi; bfi.type = ValType::U32; bfi.imm_u32 = 16; bfi.imm_u32_2 = 16;
                    bfi.args = { s0, s1, sb.emit_const_u32(blk, 16u), sb.emit_const_u32(blk, 16u) };
                    s0 = sb.emit(blk, std::move(bfi));
                }
                store_dest_u32(sb, blk, ins, s0);
                return;
            }

            case OpClass::FpUnary: {
                SsaId s0 = fetch_fp(0);
                store_fp(emit_op(blk, rule->ssa_op, f_vt, { s0 }, rule->effects));
                return;
            }

            case OpClass::FpBinary: {
                SsaId s0 = fetch_fp(0);
                SsaId s1 = fetch_fp(1);
                SsaId r = emit_op(blk, rule->ssa_op, f_vt, { s0, s1 }, rule->effects);
                if ((ins.canonical_mnemonic == Mnemonic::FMUL_RSCALE || ins.mnemonic == Mnemonic::FMUL_RSCALE) &&
                    ins.srcs.size() >= 3 && !is_v2f16) {
                    SsaId sc = sb.emit_bitcast(blk, ValType::S32, fetch_src_int32(sb, blk, ins, 2, true));
                    r = emit_op(blk, SsaOp::Ldexp, ValType::F32, { r, sc }, rule->effects);
                }
                store_fp(r);
                return;
            }

            case OpClass::FpImm: {
                SsaId s0 = fetch_src_f32(sb, blk, ins, 0);
                SsaId s1 = sb.emit_const_f32_bits(blk, static_cast<uint32_t>(ins.imm_unsigned));
                store_dest_f32(sb, blk, ins, emit_op(blk, SsaOp::FAdd, ValType::F32, { s0, s1 }, rule->effects));
                return;
            }

            case OpClass::FpFma: {
                SsaId s0 = fetch_fp(0);
                SsaId s1 = fetch_fp(1);
                SsaId s2 = (ins.srcs.size() >= 3) ? fetch_fp(2)
                                                  : (is_v2f16 ? make_v2f_const(0.0f) : sb.emit_const_f32_bits(blk, 0u));
                store_fp(emit_op(blk, SsaOp::Fma, f_vt, { s0, s1, s2 }, rule->effects));
                return;
            }

            case OpClass::FpMinmax: {
                // Corner case: nan_suppress (NMin/NMax) vs nan_propagate (FMin/FMax)
                bool is_min = (ins.canonical_mnemonic == Mnemonic::FMIN || ins.mnemonic == Mnemonic::FMIN);
                bool nan_prop = false;
                auto it_nan = ins.modifier_strs.find("nan_mode");
                if (it_nan != ins.modifier_strs.end() && it_nan->second.find("propagate") != std::string::npos) {
                    nan_prop = true;
                }
                SsaOp op = is_min ? (nan_prop ? SsaOp::FMin : SsaOp::NMin)
                                  : (nan_prop ? SsaOp::FMax : SsaOp::NMax);
                store_fp(emit_op(blk, op, f_vt, { fetch_fp(0), fetch_fp(1) }, rule->effects));
                return;
            }

            case OpClass::FpRound: {
                std::string rmode = "rte";
                auto it = ins.modifier_strs.find("round");
                if (it != ins.modifier_strs.end()) rmode = it->second;
                SsaOp op = SsaOp::RoundEven;
                if (rmode == "rtn" || rmode == "round_neg_inf") op = SsaOp::Floor;
                else if (rmode == "rtp" || rmode == "round_pos_inf") op = SsaOp::Ceil;
                else if (rmode == "rtz" || rmode == "round_zero") op = SsaOp::Trunc;
                store_fp(emit_op(blk, op, f_vt, { fetch_fp(0) }, rule->effects));
                return;
            }

            case OpClass::FpFlush: {
                SsaId z = is_v2f16 ? make_v2f_const(0.0f) : sb.emit_const_f32_bits(blk, 0u);
                store_fp(emit_op(blk, SsaOp::FAdd, f_vt, { fetch_fp(0), z }, rule->effects));
                return;
            }

            case OpClass::FpLscale: {
                SsaId s0 = fetch_src_f32(sb, blk, ins, 0);
                SsaId s1 = fetch_src_f32(sb, blk, ins, 1);
                SsaId sc = sb.emit_bitcast(blk, ValType::S32, fetch_src_int32(sb, blk, ins, 2, true));
                SsaId l = emit_op(blk, SsaOp::Ldexp, ValType::F32, { s0, sc }, rule->effects);
                store_dest_f32(sb, blk, ins, emit_op(blk, SsaOp::FAdd, ValType::F32, { l, s1 }));
                return;
            }

            case OpClass::FpRscale: {
                SsaId s0 = fetch_src_f32(sb, blk, ins, 0);
                SsaId s1 = fetch_src_f32(sb, blk, ins, 1);
                SsaId val = kInvalidSsa;
                size_t sc_idx = 2;
                if (ins.canonical_mnemonic == Mnemonic::FMA_RSCALE && ins.srcs.size() >= 4) {
                    SsaId s2 = fetch_src_f32(sb, blk, ins, 2);
                    val = emit_op(blk, SsaOp::Fma, ValType::F32, { s0, s1, s2 }, rule->effects);
                    sc_idx = 3;
                } else if (ins.mnemonic == Mnemonic::FADD_RSCALE) {
                    val = emit_op(blk, SsaOp::FAdd, ValType::F32, { s0, s1 });
                } else {
                    val = emit_op(blk, SsaOp::FMul, ValType::F32, { s0, s1 });
                }
                if (sc_idx < ins.srcs.size()) {
                    SsaId sc = sb.emit_bitcast(blk, ValType::S32, fetch_src_int32(sb, blk, ins, sc_idx, true));
                    val = emit_op(blk, SsaOp::Ldexp, ValType::F32, { val, sc }, rule->effects);
                }
                store_dest_f32(sb, blk, ins, val);
                return;
            }

            case OpClass::FpRcp: {
                uint32_t one_bits = 0x3f800000u;
                SsaId one = sb.emit_const_f32_bits(blk, one_bits);
                store_dest_f32(sb, blk, ins, emit_op(blk, SsaOp::FDiv, ValType::F32, { one, fetch_src_f32(sb, blk, ins, 0) }));
                return;
            }

            case OpClass::FpSfu1: {
                store_dest_f32(sb, blk, ins, emit_op(blk, rule->ssa_op, ValType::F32, { fetch_src_f32(sb, blk, ins, 0) }, rule->effects));
                return;
            }

            case OpClass::FpTrigTurn: {
                // Corner case: FSIN_TABLE / FCOS_TABLE input is in turns -> multiply by 2*pi
                uint32_t tau_bits = 0;
                float tau = 6.283185307179586f;
                std::memcpy(&tau_bits, &tau, sizeof(float));
                SsaId arg = emit_op(blk, SsaOp::FMul, ValType::F32,
                                    { fetch_src_f32(sb, blk, ins, 0), sb.emit_const_f32_bits(blk, tau_bits) });
                store_dest_f32(sb, blk, ins, emit_op(blk, rule->ssa_op, ValType::F32, { arg }, rule->effects));
                return;
            }

            case OpClass::FpSfu2: {
                SsaId s0 = fetch_src_f32(sb, blk, ins, 0);
                SsaId s1 = (ins.srcs.size() >= 2) ? fetch_src_f32(sb, blk, ins, 1) : sb.emit_const_f32_bits(blk, 0x3f800000u);
                store_dest_f32(sb, blk, ins, emit_op(blk, rule->ssa_op, ValType::F32, { s0, s1 }, rule->effects));
                return;
            }

            case OpClass::FpLdexp: {
                SsaId s0 = fetch_src_f32(sb, blk, ins, 0);
                SsaId s1 = sb.emit_bitcast(blk, ValType::S32, fetch_src_int32(sb, blk, ins, 1, true));
                store_dest_f32(sb, blk, ins, emit_op(blk, SsaOp::Ldexp, ValType::F32, { s0, s1 }, rule->effects));
                return;
            }

            case OpClass::FpFrexp: {
                SsaId s0 = fetch_src_f32(sb, blk, ins, 0);
                if (ins.canonical_mnemonic == Mnemonic::FREXPM) {
                    store_dest_f32(sb, blk, ins, emit_op(blk, SsaOp::FrexpSignificand, ValType::F32, { s0 }, rule->effects));
                } else {
                    SsaId e = emit_op(blk, SsaOp::FrexpExp, ValType::S32, { s0 }, rule->effects);
                    if (ins.canonical_mnemonic == Mnemonic::ILOGB || ins.canonical_mnemonic == Mnemonic::LOGB) {
                        e = emit_op(blk, SsaOp::ISub, ValType::S32, { e, sb.emit_const_s32(blk, 1) });
                    }
                    store_dest_u32(sb, blk, ins, sb.emit_bitcast(blk, ValType::U32, e));
                }
                return;
            }

            case OpClass::FpRcbrt: {
                float third = -1.0f / 3.0f;
                uint32_t tb = 0; std::memcpy(&tb, &third, sizeof(float));
                store_dest_f32(sb, blk, ins, emit_op(blk, SsaOp::Pow, ValType::F32,
                                                     { fetch_src_f32(sb, blk, ins, 0), sb.emit_const_f32_bits(blk, tb) },
                                                     rule->effects));
                return;
            }

            case OpClass::FpClass: {
                SsaId s0 = fetch_src_f32(sb, blk, ins, 0);
                SsaId is_nan = emit_op(blk, SsaOp::IsNan, ValType::Bool, { s0 });
                SsaId is_inf = emit_op(blk, SsaOp::IsInf, ValType::Bool, { s0 });
                SsaId inner = emit_op(blk, SsaOp::Select, ValType::U32,
                                      { is_inf, sb.emit_const_u32(blk, 2u), sb.emit_const_u32(blk, 0u) });
                SsaId out = emit_op(blk, SsaOp::Select, ValType::U32,
                                    { is_nan, sb.emit_const_u32(blk, 1u), inner });
                store_dest_u32(sb, blk, ins, out);
                return;
            }

            case OpClass::FpCube: {
                SsaId x = fetch_src_f32(sb, blk, ins, 0);
                SsaId y = (ins.srcs.size() >= 2) ? fetch_src_f32(sb, blk, ins, 1) : sb.emit_const_f32_bits(blk, 0u);
                SsaId z = (ins.srcs.size() >= 3) ? fetch_src_f32(sb, blk, ins, 2) : sb.emit_const_f32_bits(blk, 0u);
                if (ins.canonical_mnemonic == Mnemonic::CUBE_SSEL) { store_dest_f32(sb, blk, ins, x); return; }
                if (ins.canonical_mnemonic == Mnemonic::CUBE_TSEL) { store_dest_f32(sb, blk, ins, y); return; }
                SsaId ax = emit_op(blk, SsaOp::FAbs, ValType::F32, { x }, rule->effects);
                SsaId ay = emit_op(blk, SsaOp::FAbs, ValType::F32, { y }, rule->effects);
                SsaId az = emit_op(blk, SsaOp::FAbs, ValType::F32, { z }, rule->effects);
                SsaId ma = emit_op(blk, SsaOp::FMax, ValType::F32,
                                   { ax, emit_op(blk, SsaOp::FMax, ValType::F32, { ay, az }, rule->effects) }, rule->effects);
                if (ins.canonical_mnemonic == Mnemonic::CUBEFACE2) {
                    store_dest_f32(sb, blk, ins, ma);
                    return;
                }
                SsaId zero = sb.emit_const_f32_bits(blk, 0u);
                SsaId is_x = emit_op(blk, SsaOp::LogicalAnd, ValType::Bool,
                                     { emit_op(blk, SsaOp::FCmpOrdGe, ValType::Bool, { ax, ay }),
                                       emit_op(blk, SsaOp::FCmpOrdGe, ValType::Bool, { ax, az }) });
                SsaId is_y = emit_op(blk, SsaOp::FCmpOrdGe, ValType::Bool, { ay, az });
                SsaId fx = emit_op(blk, SsaOp::Select, ValType::U32,
                                   { emit_op(blk, SsaOp::FCmpOrdLt, ValType::Bool, { x, zero }),
                                     sb.emit_const_u32(blk, 1u), sb.emit_const_u32(blk, 0u) });
                SsaId fy = emit_op(blk, SsaOp::Select, ValType::U32,
                                   { emit_op(blk, SsaOp::FCmpOrdLt, ValType::Bool, { y, zero }),
                                     sb.emit_const_u32(blk, 3u), sb.emit_const_u32(blk, 2u) });
                SsaId fz = emit_op(blk, SsaOp::Select, ValType::U32,
                                   { emit_op(blk, SsaOp::FCmpOrdLt, ValType::Bool, { z, zero }),
                                     sb.emit_const_u32(blk, 5u), sb.emit_const_u32(blk, 4u) });
                SsaId fy_z = emit_op(blk, SsaOp::Select, ValType::U32, { is_y, fy, fz });
                store_dest_u32(sb, blk, ins, emit_op(blk, SsaOp::Select, ValType::U32, { is_x, fx, fy_z }));
                return;
            }

            case OpClass::IntBinary: {
                bool is_s = (var.find("s") == 0);
                SsaId s0 = fetch_src_int32(sb, blk, ins, 0, is_s);
                SsaId s1 = fetch_src_int32(sb, blk, ins, 1, is_s);
                store_dest_u32(sb, blk, ins, emit_op(blk, rule->ssa_op, ValType::U32, { s0, s1 }));
                return;
            }

            case OpClass::IntBinary64: {
                bool is_s = (var.find("s64") != std::string::npos);
                SsaId s0 = fetch_src_u64(sb, blk, ins, 0, is_s);
                SsaId s1 = fetch_src_u64(sb, blk, ins, 1, is_s);
                store_dest_u64(sb, blk, ins, emit_op(blk, rule->ssa_op, ValType::U64, { s0, s1 }, rule->effects));
                return;
            }

            case OpClass::IntImm: {
                if (ins.dest_is_64bit) {
                    SsaId s0 = fetch_src_u64(sb, blk, ins, 0, false);
                    SsaId imm = sb.emit_const_u64(blk, static_cast<uint64_t>(ins.imm_signed));
                    store_dest_u64(sb, blk, ins, emit_op(blk, SsaOp::IAdd64, ValType::U64, { s0, imm }, rule->effects));
                } else {
                    SsaId s0 = fetch_src_int32(sb, blk, ins, 0, false);
                    SsaId imm = sb.emit_const_u32(blk, static_cast<uint32_t>(ins.imm_signed));
                    store_dest_u32(sb, blk, ins, emit_op(blk, SsaOp::IAdd, ValType::U32, { s0, imm }));
                }
                return;
            }

            case OpClass::IntTernary: {
                SsaId s0 = fetch_src_int32(sb, blk, ins, 0, false);
                SsaId s1 = fetch_src_int32(sb, blk, ins, 1, false);
                SsaId s2 = (ins.srcs.size() >= 3) ? fetch_src_int32(sb, blk, ins, 2, false) : sb.emit_const_u32(blk, 0u);
                SsaId t0 = emit_op(blk, rule->ssa_op, ValType::U32, { s0, s1 });
                store_dest_u32(sb, blk, ins, emit_op(blk, rule->ssa_op, ValType::U32, { t0, s2 }));
                return;
            }

            case OpClass::IntAbs: {
                SsaId s0 = sb.emit_bitcast(blk, ValType::S32, fetch_src_int32(sb, blk, ins, 0, true));
                SsaId ab = emit_op(blk, SsaOp::SAbs, ValType::S32, { s0 }, rule->effects);
                store_dest_u32(sb, blk, ins, sb.emit_bitcast(blk, ValType::U32, ab));
                return;
            }

            case OpClass::IntHadd: {
                bool is_s = (var.find("s") == 0);
                SsaId s0 = fetch_src_int32(sb, blk, ins, 0, is_s);
                SsaId s1 = fetch_src_int32(sb, blk, ins, 1, is_s);
                SsaId band = emit_op(blk, SsaOp::BitAnd, ValType::U32, { s0, s1 });
                SsaId bxor = emit_op(blk, SsaOp::BitXor, ValType::U32, { s0, s1 });
                SsaId half = is_s
                    ? sb.emit_bitcast(blk, ValType::U32,
                                      emit_op(blk, SsaOp::ShrS, ValType::S32,
                                              { sb.emit_bitcast(blk, ValType::S32, bxor), sb.emit_const_u32(blk, 1u) }))
                    : emit_op(blk, SsaOp::ShrU, ValType::U32, { bxor, sb.emit_const_u32(blk, 1u) });
                store_dest_u32(sb, blk, ins, emit_op(blk, SsaOp::IAdd, ValType::U32, { band, half }));
                return;
            }

            case OpClass::IntIdpadd: {
                // Corner case: v14 unsigned0 / unsigned1 on IDPADD.v4s8
                bool s0_signed = (var.find("s") == 0);
                bool s1_signed = s0_signed;
                if (ins.fields.count("unsigned0") && ins.fields.at("unsigned0") != 0) s0_signed = false;
                if (ins.fields.count("unsigned1") && ins.fields.at("unsigned1") != 0) s1_signed = false;
                SsaId a4 = fetch_src_v4i8(sb, blk, ins, 0, s0_signed);
                SsaId b4 = fetch_src_v4i8(sb, blk, ins, 1, s1_signed);
                if (s0_signed != s1_signed) {
                    // Cast both to S32 lanes for dot product
                    if (!s0_signed) a4 = sb.emit_bitcast(blk, ValType::V4S32, a4);
                    if (!s1_signed) b4 = sb.emit_bitcast(blk, ValType::V4S32, b4);
                }
                bool acc_s = (s0_signed || s1_signed);
                SsaId acc = (ins.srcs.size() >= 3) ? fetch_src_int32(sb, blk, ins, 2, acc_s) : sb.emit_const_u32(blk, 0u);
                if (acc_s) acc = sb.emit_bitcast(blk, ValType::S32, acc);
                ValType elem_t = acc_s ? ValType::S32 : ValType::U32;
                for (uint32_t i = 0; i < 4; ++i) {
                    SsaNode ea{}; ea.op = SsaOp::VecExtract; ea.type = elem_t; ea.imm_u32 = i; ea.args = { a4 };
                    SsaNode eb{}; eb.op = SsaOp::VecExtract; eb.type = elem_t; eb.imm_u32 = i; eb.args = { b4 };
                    SsaId prod = emit_op(blk, SsaOp::IMul, elem_t, { sb.emit(blk, std::move(ea)), sb.emit(blk, std::move(eb)) });
                    acc = emit_op(blk, SsaOp::IAdd, elem_t, { acc, prod });
                }
                store_dest_u32(sb, blk, ins, acc_s ? sb.emit_bitcast(blk, ValType::U32, acc) : acc);
                return;
            }

            case OpClass::IntShadd64: {
                bool is_s = (var.find("s64") != std::string::npos);
                SsaId base64 = fetch_src_u64(sb, blk, ins, 0, false);
                SsaId idx64 = fetch_src_u64(sb, blk, ins, 1, is_s);
                SsaId sh_amt = sb.emit_const_u64(blk, static_cast<uint64_t>(ins.imm_unsigned & 63u));
                SsaId shifted = emit_op(blk, SsaOp::Shl64, ValType::U64, { idx64, sh_amt }, rule->effects);
                store_dest_u64(sb, blk, ins, emit_op(blk, SsaOp::IAdd64, ValType::U64, { base64, shifted }, rule->effects));
                return;
            }

            case OpClass::IntSeg: {
                SsaId s0 = fetch_src_u64(sb, blk, ins, 0, false);
                SsaId s1 = fetch_src_u64(sb, blk, ins, 1, true);
                store_dest_u64(sb, blk, ins, emit_op(blk, rule->ssa_op, ValType::U64, { s0, s1 }, rule->effects));
                return;
            }

            case OpClass::BitNot: {
                store_dest_u32(sb, blk, ins, emit_op(blk, SsaOp::BitNot, ValType::U32, { fetch_src_int32(sb, blk, ins, 0, false) }));
                return;
            }
            case OpClass::BitNot64: {
                store_dest_u64(sb, blk, ins, emit_op(blk, SsaOp::BitNot64, ValType::U64, { fetch_src_u64(sb, blk, ins, 0, false) }, rule->effects));
                return;
            }
            case OpClass::BitLogic: {
                SsaId s0 = fetch_src_int32(sb, blk, ins, 0, false);
                SsaId s1 = fetch_src_int32(sb, blk, ins, 1, false);
                SsaId r = emit_op(blk, rule->ssa_op, ValType::U32, { s0, s1 });
                auto it_nr = ins.fields.find("not_result");
                if (it_nr != ins.fields.end() && it_nr->second != 0) r = emit_op(blk, SsaOp::BitNot, ValType::U32, { r });
                store_dest_u32(sb, blk, ins, r);
                return;
            }
            case OpClass::BitLogic64: {
                SsaId s0 = fetch_src_u64(sb, blk, ins, 0, false);
                SsaId s1 = fetch_src_u64(sb, blk, ins, 1, false);
                SsaId r = emit_op(blk, rule->ssa_op, ValType::U64, { s0, s1 }, rule->effects);
                auto it_nr = ins.fields.find("not_result");
                if (it_nr != ins.fields.end() && it_nr->second != 0) r = emit_op(blk, SsaOp::BitNot64, ValType::U64, { r }, rule->effects);
                store_dest_u64(sb, blk, ins, r);
                return;
            }
            case OpClass::BitImm: {
                SsaId s0 = fetch_src_int32(sb, blk, ins, 0, false);
                SsaId imm = sb.emit_const_u32(blk, static_cast<uint32_t>(ins.imm_unsigned));
                store_dest_u32(sb, blk, ins, emit_op(blk, rule->ssa_op, ValType::U32, { s0, imm }));
                return;
            }
            case OpClass::BitShift: {
                SsaId s0 = fetch_src_int32(sb, blk, ins, 0, false);
                SsaId sh = emit_op(blk, SsaOp::BitAnd, ValType::U32,
                                   { fetch_src_int32(sb, blk, ins, 1, false), sb.emit_const_u32(blk, 31u) });
                SsaId r = kInvalidSsa;
                if (rule->ssa_op == SsaOp::ShrS) {
                    SsaId s0s = sb.emit_bitcast(blk, ValType::S32, s0);
                    r = sb.emit_bitcast(blk, ValType::U32, emit_op(blk, SsaOp::ShrS, ValType::S32, { s0s, sh }));
                } else {
                    r = emit_op(blk, rule->ssa_op, ValType::U32, { s0, sh });
                }
                auto it_nr = ins.fields.find("not_result");
                if (it_nr != ins.fields.end() && it_nr->second != 0) r = emit_op(blk, SsaOp::BitNot, ValType::U32, { r });
                store_dest_u32(sb, blk, ins, r);
                return;
            }
            case OpClass::BitShift64: {
                SsaId s0 = fetch_src_u64(sb, blk, ins, 0, false);
                SsaId sh = emit_op(blk, SsaOp::BitAnd64, ValType::U64,
                                   { fetch_src_u64(sb, blk, ins, 1, false), sb.emit_const_u64(blk, 63ULL) }, rule->effects);
                SsaId r = emit_op(blk, rule->ssa_op, ValType::U64, { s0, sh }, rule->effects);
                auto it_nr = ins.fields.find("not_result");
                if (it_nr != ins.fields.end() && it_nr->second != 0) r = emit_op(blk, SsaOp::BitNot64, ValType::U64, { r }, rule->effects);
                store_dest_u64(sb, blk, ins, r);
                return;
            }
            case OpClass::BitShiftLogic: {
                SsaId s0 = fetch_src_int32(sb, blk, ins, 0, false);
                SsaId sh = emit_op(blk, SsaOp::BitAnd, ValType::U32,
                                   { fetch_src_int32(sb, blk, ins, 1, false), sb.emit_const_u32(blk, 31u) });
                SsaId shifted = kInvalidSsa;
                if (ins.canonical_mnemonic == Mnemonic::ARSHIFT_AND || ins.canonical_mnemonic == Mnemonic::ARSHIFT_OR ||
                    ins.canonical_mnemonic == Mnemonic::ARSHIFT_XOR) {
                    shifted = sb.emit_bitcast(blk, ValType::U32,
                                              emit_op(blk, SsaOp::ShrS, ValType::S32,
                                                      { sb.emit_bitcast(blk, ValType::S32, s0), sh }));
                } else if (ins.canonical_mnemonic == Mnemonic::RSHIFT_AND || ins.canonical_mnemonic == Mnemonic::RSHIFT_OR ||
                           ins.canonical_mnemonic == Mnemonic::RSHIFT_XOR) {
                    shifted = emit_op(blk, SsaOp::ShrU, ValType::U32, { s0, sh });
                } else {
                    shifted = emit_op(blk, SsaOp::Shl, ValType::U32, { s0, sh });
                }
                SsaId r = shifted;
                if (ins.srcs.size() >= 3) {
                    r = emit_op(blk, rule->ssa_op, ValType::U32, { shifted, fetch_src_int32(sb, blk, ins, 2, false) });
                }
                auto it_nr = ins.fields.find("not_result");
                if (it_nr != ins.fields.end() && it_nr->second != 0) r = emit_op(blk, SsaOp::BitNot, ValType::U32, { r });
                store_dest_u32(sb, blk, ins, r);
                return;
            }
            case OpClass::BitShiftLogic64: {
                SsaId s0 = fetch_src_u64(sb, blk, ins, 0, false);
                SsaId sh = emit_op(blk, SsaOp::BitAnd64, ValType::U64,
                                   { fetch_src_u64(sb, blk, ins, 1, false), sb.emit_const_u64(blk, 63ULL) }, rule->effects);
                SsaOp sop = SsaOp::Shl64;
                if (ins.canonical_mnemonic == Mnemonic::ARSHIFT_AND || ins.canonical_mnemonic == Mnemonic::ARSHIFT_OR ||
                    ins.canonical_mnemonic == Mnemonic::ARSHIFT_XOR) sop = SsaOp::ShrS64;
                else if (ins.canonical_mnemonic == Mnemonic::RSHIFT_AND || ins.canonical_mnemonic == Mnemonic::RSHIFT_OR ||
                         ins.canonical_mnemonic == Mnemonic::RSHIFT_XOR) sop = SsaOp::ShrU64;
                SsaId r = emit_op(blk, sop, ValType::U64, { s0, sh }, rule->effects);
                if (ins.srcs.size() >= 3) {
                    r = emit_op(blk, rule->ssa_op, ValType::U64, { r, fetch_src_u64(sb, blk, ins, 2, false) }, rule->effects);
                }
                auto it_nr = ins.fields.find("not_result");
                if (it_nr != ins.fields.end() && it_nr->second != 0) r = emit_op(blk, SsaOp::BitNot64, ValType::U64, { r }, rule->effects);
                store_dest_u64(sb, blk, ins, r);
                return;
            }
            case OpClass::BitRotLogic: {
                SsaId s0 = fetch_src_int32(sb, blk, ins, 0, false);
                SsaId sh = emit_op(blk, SsaOp::BitAnd, ValType::U32,
                                   { fetch_src_int32(sb, blk, ins, 1, false), sb.emit_const_u32(blk, 31u) });
                bool is_r = (ins.canonical_mnemonic == Mnemonic::RROT_AND || ins.canonical_mnemonic == Mnemonic::RROT_OR ||
                             ins.canonical_mnemonic == Mnemonic::RROT_XOR || ins.mnemonic == Mnemonic::RROT);
                SsaId rot = emit_op(blk, is_r ? SsaOp::RotR : SsaOp::RotL, ValType::U32, { s0, sh });
                SsaId r = rot;
                if (ins.srcs.size() >= 3) {
                    r = emit_op(blk, rule->ssa_op, ValType::U32, { rot, fetch_src_int32(sb, blk, ins, 2, false) });
                }
                store_dest_u32(sb, blk, ins, r);
                return;
            }
            case OpClass::BitRotLogic64: {
                SsaId s0 = fetch_src_u64(sb, blk, ins, 0, false);
                SsaId sh = emit_op(blk, SsaOp::BitAnd64, ValType::U64,
                                   { fetch_src_u64(sb, blk, ins, 1, false), sb.emit_const_u64(blk, 63ULL) }, rule->effects);
                bool is_r = (ins.canonical_mnemonic == Mnemonic::RROT_AND || ins.canonical_mnemonic == Mnemonic::RROT_OR ||
                             ins.canonical_mnemonic == Mnemonic::RROT_XOR || ins.mnemonic == Mnemonic::RROT);
                SsaId inv_sh = emit_op(blk, SsaOp::ISub64, ValType::U64, { sb.emit_const_u64(blk, 64ULL), sh }, rule->effects);
                SsaId p0 = emit_op(blk, is_r ? SsaOp::ShrU64 : SsaOp::Shl64, ValType::U64, { s0, sh }, rule->effects);
                SsaId p1 = emit_op(blk, is_r ? SsaOp::Shl64 : SsaOp::ShrU64, ValType::U64, { s0, inv_sh }, rule->effects);
                SsaId r = emit_op(blk, SsaOp::BitOr64, ValType::U64, { p0, p1 }, rule->effects);
                if (ins.srcs.size() >= 3) {
                    r = emit_op(blk, rule->ssa_op, ValType::U64, { r, fetch_src_u64(sb, blk, ins, 2, false) }, rule->effects);
                }
                store_dest_u64(sb, blk, ins, r);
                return;
            }
            case OpClass::BitUnary: {
                store_dest_u32(sb, blk, ins, emit_op(blk, rule->ssa_op, ValType::U32, { fetch_src_int32(sb, blk, ins, 0, false) }));
                return;
            }
            case OpClass::BitClz: {
                store_dest_u32(sb, blk, ins, emit_op(blk, SsaOp::Clz, ValType::U32, { fetch_src_int32(sb, blk, ins, 0, false) }, rule->effects));
                return;
            }
            case OpClass::BitMux: {
                SsaId a = fetch_src_int32(sb, blk, ins, 0, false);
                SsaId c = fetch_src_int32(sb, blk, ins, 1, false);
                SsaId m = fetch_src_int32(sb, blk, ins, 2, false);
                SsaId nm = emit_op(blk, SsaOp::BitNot, ValType::U32, { m });
                SsaId p0 = emit_op(blk, SsaOp::BitAnd, ValType::U32, { a, m });
                SsaId p1 = emit_op(blk, SsaOp::BitAnd, ValType::U32, { c, nm });
                store_dest_u32(sb, blk, ins, emit_op(blk, SsaOp::BitOr, ValType::U32, { p0, p1 }));
                return;
            }

            case OpClass::CmpFloat: {
                SsaId s0 = fetch_src_f32(sb, blk, ins, 0);
                SsaId s1 = fetch_src_f32(sb, blk, ins, 1);
                std::string cmpf = "eq";
                auto it = ins.modifier_strs.find("cmpf");
                if (it != ins.modifier_strs.end()) cmpf = it->second;
                SsaId mask = format_cmp_result(blk, ins, eval_float_cmp(blk, cmpf, s0, s1));
                if (ins.srcs.size() >= 3 && ins.mnemonic != Mnemonic::FCMP) {
                    mask = emit_op(blk, rule->ssa_op, ValType::U32, { mask, fetch_src_int32(sb, blk, ins, 2, false) });
                }
                store_dest_u32(sb, blk, ins, mask);
                return;
            }

            case OpClass::CmpInt: {
                bool is_s = (var.find("s") == 0);
                SsaId s0 = fetch_src_int32(sb, blk, ins, 0, is_s);
                SsaId s1 = fetch_src_int32(sb, blk, ins, 1, is_s);
                std::string cmpf = "eq";
                auto it = ins.modifier_strs.find("cmpf");
                if (it != ins.modifier_strs.end()) cmpf = it->second;
                SsaId mask = format_cmp_result(blk, ins, eval_int_cmp(blk, cmpf, s0, s1, is_s));
                if (ins.srcs.size() >= 3 && ins.mnemonic != Mnemonic::ICMP && ins.canonical_mnemonic != Mnemonic::ICMP_MULTI) {
                    mask = emit_op(blk, rule->ssa_op, ValType::U32, { mask, fetch_src_int32(sb, blk, ins, 2, false) });
                }
                store_dest_u32(sb, blk, ins, mask);
                return;
            }

            case OpClass::CmpCsel: {
                std::string cmpf = "ne";
                auto it = ins.modifier_strs.find("cmpf");
                if (it != ins.modifier_strs.end()) cmpf = it->second;
                if (var == "v2u16" || var == "v2s16") {
                    bool is_s = (var == "v2s16");
                    ValType elem_t = is_s ? ValType::S32 : ValType::U32;
                    ValType v2t = is_s ? ValType::V2S32 : ValType::V2U32;
                    SsaId a2 = fetch_src_v2i16(sb, blk, ins, 0, is_s);
                    SsaId c2 = fetch_src_v2i16(sb, blk, ins, 1, is_s);
                    SsaId t2 = fetch_src_v2i16(sb, blk, ins, 2, is_s);
                    SsaId f2 = fetch_src_v2i16(sb, blk, ins, 3, is_s);
                    auto ext = [&](SsaId v, uint32_t idx) {
                        SsaNode ex{}; ex.op = SsaOp::VecExtract; ex.type = elem_t; ex.imm_u32 = idx; ex.args = { v };
                        SsaId e = sb.emit(blk, std::move(ex));
                        return is_s ? sb.emit_bitcast(blk, ValType::U32, e) : e;
                    };
                    SsaId cond0 = eval_int_cmp(blk, cmpf, ext(a2, 0), ext(c2, 0), is_s);
                    SsaId cond1 = eval_int_cmp(blk, cmpf, ext(a2, 1), ext(c2, 1), is_s);
                    SsaId cond2 = emit_op(blk, SsaOp::VecConstruct2, ValType::V2Bool, { cond0, cond1 });
                    SsaId res2 = emit_op(blk, SsaOp::Select, v2t, { cond2, t2, f2 });
                    store_dest_v2i16(sb, blk, ins, res2, is_s);
                    return;
                }
                bool is_float = (var.find("f") == 0 || var == "v2f16");
                bool is_s = (var.find("s") == 0);
                SsaId cond = is_float
                    ? eval_float_cmp(blk, cmpf, fetch_src_f32(sb, blk, ins, 0), fetch_src_f32(sb, blk, ins, 1))
                    : eval_int_cmp(blk, cmpf, fetch_src_int32(sb, blk, ins, 0, is_s), fetch_src_int32(sb, blk, ins, 1, is_s), is_s);
                SsaId tv = fetch_src_int32(sb, blk, ins, 2, false);
                SsaId fv = fetch_src_int32(sb, blk, ins, 3, false);
                store_dest_u32(sb, blk, ins, emit_op(blk, SsaOp::Select, ValType::U32, { cond, tv, fv }));
                return;
            }

            case OpClass::CvtF16ToF32: {
                SsaId raw = fetch_raw_src_u32(sb, blk, ins, 0);
                std::string swz = get_src_swz(ins, 0);
                SsaId unp = emit_op(blk, SsaOp::UnpackHalf2x16, ValType::V2F32, { raw }, rule->effects);
                SsaNode ex{}; ex.op = SsaOp::VecExtract; ex.type = ValType::F32; ex.imm_u32 = (swz == "h1") ? 1u : 0u; ex.args = { unp };
                store_dest_f32(sb, blk, ins, sb.emit(blk, std::move(ex)));
                return;
            }
            case OpClass::CvtF32ToF16: {
                SsaId s0 = fetch_src_f32(sb, blk, ins, 0);
                SsaId s1 = (ins.srcs.size() >= 2) ? fetch_src_f32(sb, blk, ins, 1) : s0;
                SsaId v2 = emit_op(blk, SsaOp::VecConstruct2, ValType::V2F32, { s0, s1 });
                store_dest_u32(sb, blk, ins, emit_op(blk, SsaOp::PackHalf2x16, ValType::U32, { v2 }, rule->effects));
                return;
            }
            case OpClass::CvtF32ToS32: {
                SsaId s32 = emit_op(blk, SsaOp::ConvertFToS, ValType::S32, { fetch_src_f32(sb, blk, ins, 0) });
                store_dest_u32(sb, blk, ins, sb.emit_bitcast(blk, ValType::U32, s32));
                return;
            }
            case OpClass::CvtF32ToU32: {
                store_dest_u32(sb, blk, ins, emit_op(blk, SsaOp::ConvertFToU, ValType::U32, { fetch_src_f32(sb, blk, ins, 0) }));
                return;
            }
            case OpClass::CvtF16ToS32:
            case OpClass::CvtF16ToU32: {
                SsaId raw = fetch_raw_src_u32(sb, blk, ins, 0);
                std::string swz = get_src_swz(ins, 0);
                SsaId unp = emit_op(blk, SsaOp::UnpackHalf2x16, ValType::V2F32, { raw }, rule->effects);
                SsaNode ex{}; ex.op = SsaOp::VecExtract; ex.type = ValType::F32; ex.imm_u32 = (swz == "h1") ? 1u : 0u; ex.args = { unp };
                SsaId fval = sb.emit(blk, std::move(ex));
                if (rule->op_class == OpClass::CvtF16ToS32) {
                    SsaId s32 = emit_op(blk, SsaOp::ConvertFToS, ValType::S32, { fval });
                    store_dest_u32(sb, blk, ins, sb.emit_bitcast(blk, ValType::U32, s32));
                } else {
                    store_dest_u32(sb, blk, ins, emit_op(blk, SsaOp::ConvertFToU, ValType::U32, { fval }));
                }
                return;
            }
            case OpClass::CvtS32ToF32: {
                SsaId s32 = sb.emit_bitcast(blk, ValType::S32, fetch_src_int32(sb, blk, ins, 0, true));
                store_dest_f32(sb, blk, ins, emit_op(blk, SsaOp::ConvertSToF, ValType::F32, { s32 }));
                return;
            }
            case OpClass::CvtU32ToF32: {
                store_dest_f32(sb, blk, ins, emit_op(blk, SsaOp::ConvertUToF, ValType::F32, { fetch_src_int32(sb, blk, ins, 0, false) }));
                return;
            }
            case OpClass::CvtIntExtS: {
                store_dest_u32(sb, blk, ins, fetch_src_int32(sb, blk, ins, 0, true));
                return;
            }
            case OpClass::CvtIntExtU: {
                store_dest_u32(sb, blk, ins, fetch_src_int32(sb, blk, ins, 0, false));
                return;
            }
            case OpClass::CvtV2f16ToV2i16: {
                bool is_s = (rule->ssa_op == SsaOp::ConvertFToS);
                SsaId fv = fetch_src_v2f16(sb, blk, ins, 0);
                SsaId iv = emit_op(blk, rule->ssa_op, is_s ? ValType::V2S32 : ValType::V2U32, { fv }, rule->effects);
                store_dest_v2i16(sb, blk, ins, iv, is_s);
                return;
            }
            case OpClass::CvtV2i16ToV2f16: {
                bool is_s = (rule->ssa_op == SsaOp::ConvertSToF);
                SsaId iv = fetch_src_v2i16(sb, blk, ins, 0, is_s);
                store_dest_v2f16(sb, blk, ins, emit_op(blk, rule->ssa_op, ValType::V2F32, { iv }, rule->effects));
                return;
            }
            case OpClass::CvtV2i8ToV2f16: {
                bool is_s = (rule->ssa_op == SsaOp::ConvertSToF);
                SsaId v4 = fetch_src_v4i8(sb, blk, ins, 0, is_s);
                ValType v2t = is_s ? ValType::V2S32 : ValType::V2U32;
                SsaNode sh{}; sh.op = SsaOp::VecShuffle; sh.type = v2t; sh.args = { v4, v4 }; sh.imm_u32 = 0; sh.imm_u32_2 = 1;
                SsaId iv = sb.emit(blk, std::move(sh));
                store_dest_v2f16(sb, blk, ins, emit_op(blk, rule->ssa_op, ValType::V2F32, { iv }, rule->effects));
                return;
            }
            case OpClass::CvtV2i8ToV2i16: {
                bool is_s = (ins.canonical_mnemonic == Mnemonic::V2S8_TO_V2S16);
                SsaId v4 = fetch_src_v4i8(sb, blk, ins, 0, is_s);
                ValType v2t = is_s ? ValType::V2S32 : ValType::V2U32;
                SsaNode sh{}; sh.op = SsaOp::VecShuffle; sh.type = v2t; sh.args = { v4, v4 }; sh.imm_u32 = 0; sh.imm_u32_2 = 1;
                store_dest_v2i16(sb, blk, ins, sb.emit(blk, std::move(sh)), is_s);
                return;
            }

            case OpClass::SubgroupWmask: {
                SsaId s0 = fetch_src_int32(sb, blk, ins, 0, false);
                SsaId cond = emit_op(blk, SsaOp::ICmpNe, ValType::Bool, { s0, sb.emit_const_u32(blk, 0u) });
                SsaId ballot = emit_op(blk, SsaOp::SubgroupBallot, ValType::V4U32,
                                       { sb.emit_const_u32(blk, 3u), cond }, rule->effects, true, true);
                SsaNode ex{}; ex.op = SsaOp::VecExtract; ex.type = ValType::U32; ex.imm_u32 = 0; ex.args = { ballot };
                store_dest_u32(sb, blk, ins, sb.emit(blk, std::move(ex)));
                return;
            }

            case OpClass::SubgroupClper: {
                SsaId s0 = fetch_src_int32(sb, blk, ins, 0, false);
                SsaId s1 = fetch_src_int32(sb, blk, ins, 1, false);
                SsaId lane = emit_op(blk, SsaOp::BitAnd, ValType::U32, { s1, sb.emit_const_u32(blk, 31u) });
                SsaId shuf = emit_op(blk, SsaOp::SubgroupShuffle, ValType::U32,
                                     { sb.emit_const_u32(blk, 3u), s0, lane }, rule->effects, true, true);
                store_dest_u32(sb, blk, ins, shuf);
                return;
            }

            case OpClass::MemLea: {
                uint8_t base_r = ins.sr_write.present ? ins.sr_write.base_reg : 0;
                uint32_t idx = 0;
                for (const char* fn : {"attribute_index", "buffer_index", "texture_index", "index"}) {
                    auto it = ins.fields.find(fn);
                    if (it != ins.fields.end()) { idx = it->second; break; }
                }
                sb.set_reg(blk, base_r, sb.emit_const_u32(blk, idx * 16u));
                sb.set_reg(blk, base_r + 1, sb.emit_const_u32(blk, 0xEEEEEEEEu));
                return;
            }

            case OpClass::MemLoad:
            case OpClass::MemLoadPka: {
                SsaId base_addr = fetch_src_u64(sb, blk, ins, 0, false);
                SsaId byte_addr = base_addr;
                if (ins.has_imm && ins.imm_signed != 0) {
                    byte_addr = emit_op(blk, SsaOp::IAdd64, ValType::U64,
                                        { base_addr, sb.emit_const_u64(blk, static_cast<uint64_t>(ins.imm_signed)) },
                                        rule->effects);
                }
                uint8_t sr = ins.sr_write.present ? ins.sr_write.base_reg : 0;
                uint8_t cnt = ins.sr_write.present ? std::max<uint8_t>(1, ins.sr_write.count) : 1;
                std::string mwidth = "i32";
                auto it_mw = ins.modifier_strs.find("memory_width");
                if (it_mw != ins.modifier_strs.end()) mwidth = it_mw->second;

                uint32_t elem_bytes = 4;
                if (mwidth == "i8") elem_bytes = 1;
                else if (mwidth == "i16") elem_bytes = 2;

                SsaId tok = sb.current_token(blk);
                if (elem_bytes < 4) {
                    SsaNode ld{};
                    ld.op = SsaOp::PsbLoad; ld.type = ValType::TupleValToken;
                    ld.imm_u32 = elem_bytes; ld.aux_bool = (rule->op_class == OpClass::MemLoadPka);
                    ld.args = { byte_addr, tok }; ld.effects = rule->effects;
                    ld.is_barrier_or_cf = true; ld.is_side_effect = true;
                    SsaId tup = sb.emit(blk, std::move(ld));
                    SsaNode ev{}; ev.op = SsaOp::TupleExtract; ev.type = ValType::U32; ev.imm_u32 = 0; ev.args = { tup };
                    SsaNode et{}; et.op = SsaOp::TupleExtract; et.type = ValType::Token; et.imm_u32 = 1; et.args = { tup };
                    sb.set_reg(blk, sr, sb.emit(blk, std::move(ev)));
                    sb.set_token(blk, sb.emit(blk, std::move(et)));
                    return;
                }
                for (uint8_t w = 0; w < cnt; ++w) {
                    SsaId w_addr = (w == 0) ? byte_addr
                                            : emit_op(blk, SsaOp::IAdd64, ValType::U64,
                                                      { byte_addr, sb.emit_const_u64(blk, uint64_t(w) * 4ULL) },
                                                      rule->effects);
                    SsaNode ld{};
                    ld.op = SsaOp::PsbLoad; ld.type = ValType::TupleValToken;
                    ld.imm_u32 = 4u; ld.aux_bool = (rule->op_class == OpClass::MemLoadPka);
                    ld.args = { w_addr, tok }; ld.effects = rule->effects;
                    ld.is_barrier_or_cf = true; ld.is_side_effect = true;
                    SsaId tup = sb.emit(blk, std::move(ld));
                    SsaNode ev{}; ev.op = SsaOp::TupleExtract; ev.type = ValType::U32; ev.imm_u32 = 0; ev.args = { tup };
                    SsaNode et{}; et.op = SsaOp::TupleExtract; et.type = ValType::Token; et.imm_u32 = 1; et.args = { tup };
                    sb.set_reg(blk, sr + w, sb.emit(blk, std::move(ev)));
                    tok = sb.emit(blk, std::move(et));
                }
                sb.set_token(blk, tok);
                return;
            }

            case OpClass::MemStore: {
                std::string access = "none";
                auto it_acc = ins.modifier_strs.find("access");
                if (it_acc != ins.modifier_strs.end()) access = it_acc->second;
                uint8_t sr = ins.sr_read.present ? ins.sr_read.base_reg : 0;
                uint8_t cnt = ins.sr_read.present ? std::max<uint8_t>(1, ins.sr_read.count) : 1;

                if (opts.stage == ShaderStage::Vertex && access == "istream") {
                    SsaId r8_f  = sb.emit_bitcast(blk, ValType::F32, sb.current_reg(blk, sr + 0));
                    SsaId r9_f  = sb.emit_bitcast(blk, ValType::F32, sb.current_reg(blk, sr + 1));
                    SsaId r10_f = sb.emit_bitcast(blk, ValType::F32, sb.current_reg(blk, sr + 2));
                    SsaId r11_f = sb.emit_bitcast(blk, ValType::F32, sb.current_reg(blk, sr + 3));

                    SsaId sx = sb.emit_bitcast(blk, ValType::F32, emit_load_fau(blk, 0));
                    SsaId sy = sb.emit_bitcast(blk, ValType::F32, emit_load_fau(blk, 1));
                    SsaId sz = sb.emit_bitcast(blk, ValType::F32, emit_load_fau(blk, 2));
                    SsaId oz = sb.emit_bitcast(blk, ValType::F32, emit_load_fau(blk, 3));
                    SsaId ox = sb.emit_bitcast(blk, ValType::F32, emit_load_fau(blk, 4));
                    SsaId oy = sb.emit_bitcast(blk, ValType::F32, emit_load_fau(blk, 5));

                    SsaId has_vp = emit_op(blk, SsaOp::FCmpOrdNe, ValType::Bool, { sx, sb.emit_const_f32_bits(blk, 0u) });
                    SsaId wc = emit_op(blk, SsaOp::FDiv, ValType::F32, { sb.emit_const_f32_bits(blk, 0x3f800000u), r11_f });
                    SsaId dx = emit_op(blk, SsaOp::FSub, ValType::F32, { r8_f, ox });
                    SsaId dy = emit_op(blk, SsaOp::FSub, ValType::F32, { r9_f, oy });
                    SsaId dz = emit_op(blk, SsaOp::FSub, ValType::F32, { r10_f, oz });
                    SsaId xc = emit_op(blk, SsaOp::FMul, ValType::F32, { emit_op(blk, SsaOp::FDiv, ValType::F32, { dx, sx }), wc });
                    SsaId yc = emit_op(blk, SsaOp::FMul, ValType::F32, { emit_op(blk, SsaOp::FDiv, ValType::F32, { dy, sy }), wc });
                    SsaId zc = emit_op(blk, SsaOp::FMul, ValType::F32, { emit_op(blk, SsaOp::FDiv, ValType::F32, { dz, sz }), wc });

                    SsaId fx = emit_op(blk, SsaOp::Select, ValType::F32, { has_vp, xc, r8_f });
                    SsaId fy = emit_op(blk, SsaOp::Select, ValType::F32, { has_vp, yc, r9_f });
                    SsaId fz = emit_op(blk, SsaOp::Select, ValType::F32, { has_vp, zc, r10_f });
                    SsaId fw = emit_op(blk, SsaOp::Select, ValType::F32, { has_vp, wc, r11_f });
                    SsaId pos_vec = emit_op(blk, SsaOp::VecConstruct4, ValType::V4F32, { fx, fy, fz, fw });

                    SsaNode st{};
                    st.op = SsaOp::StoreBuiltinOutput; st.type = ValType::Token;
                    st.imm_u32 = static_cast<uint32_t>(spirv::BuiltIn::Position);
                    st.aux_str = "gl_Position";
                    st.args = { pos_vec, sb.current_token(blk) };
                    st.is_barrier_or_cf = true; st.is_side_effect = true;
                    sb.set_token(blk, sb.emit(blk, std::move(st)));
                    return;
                }
                if (opts.stage == ShaderStage::Vertex && access == "estream") {
                    uint32_t var_loc = static_cast<uint32_t>(std::max<int64_t>(0, ins.imm_signed) / 16);
                    std::vector<SsaId> comps(4);
                    for (uint8_t c = 0; c < 4; ++c) {
                        if (c < cnt) comps[c] = sb.emit_bitcast(blk, ValType::F32, sb.current_reg(blk, sr + c));
                        else comps[c] = sb.emit_const_f32_bits(blk, (c == 3) ? 0x3f800000u : 0u);
                    }
                    SsaId vec = emit_op(blk, SsaOp::VecConstruct4, ValType::V4F32, std::move(comps));
                    SsaNode st{};
                    st.op = SsaOp::StoreVertVarying; st.type = ValType::Token;
                    st.imm_u32 = var_loc;
                    st.args = { vec, sb.current_token(blk) };
                    st.is_barrier_or_cf = true; st.is_side_effect = true;
                    sb.set_token(blk, sb.emit(blk, std::move(st)));
                    return;
                }

                SsaId base_addr = fetch_src_u64(sb, blk, ins, 0, false);
                SsaId byte_addr = base_addr;
                if (ins.has_imm && ins.imm_signed != 0) {
                    byte_addr = emit_op(blk, SsaOp::IAdd64, ValType::U64,
                                        { base_addr, sb.emit_const_u64(blk, static_cast<uint64_t>(ins.imm_signed)) },
                                        rule->effects);
                }
                std::string mwidth = "i32";
                auto it_mw = ins.modifier_strs.find("memory_width");
                if (it_mw != ins.modifier_strs.end()) mwidth = it_mw->second;
                uint32_t elem_bytes = (mwidth == "i8") ? 1u : (mwidth == "i16") ? 2u : 4u;

                SsaId tok = sb.current_token(blk);
                if (elem_bytes < 4) {
                    SsaNode st{};
                    st.op = SsaOp::PsbStore; st.type = ValType::Token; st.imm_u32 = elem_bytes;
                    st.args = { byte_addr, sb.current_reg(blk, sr), tok };
                    st.effects = rule->effects; st.is_barrier_or_cf = true; st.is_side_effect = true;
                    sb.set_token(blk, sb.emit(blk, std::move(st)));
                    return;
                }
                for (uint8_t w = 0; w < cnt; ++w) {
                    SsaId w_addr = (w == 0) ? byte_addr
                                            : emit_op(blk, SsaOp::IAdd64, ValType::U64,
                                                      { byte_addr, sb.emit_const_u64(blk, uint64_t(w) * 4ULL) },
                                                      rule->effects);
                    SsaNode st{};
                    st.op = SsaOp::PsbStore; st.type = ValType::Token; st.imm_u32 = 4u;
                    st.args = { w_addr, sb.current_reg(blk, sr + w), tok };
                    st.effects = rule->effects; st.is_barrier_or_cf = true; st.is_side_effect = true;
                    tok = sb.emit(blk, std::move(st));
                }
                sb.set_token(blk, tok);
                return;
            }

            case OpClass::MemAtomic:
            case OpClass::MemAtom1:
            case OpClass::MemAcmpxchg:
            case OpClass::MemAxchg: {
                SsaId base_addr = fetch_src_u64(sb, blk, ins, 0, false);
                SsaId byte_addr = base_addr;
                if (ins.has_imm && ins.imm_signed != 0) {
                    byte_addr = emit_op(blk, SsaOp::IAdd64, ValType::U64,
                                        { base_addr, sb.emit_const_u64(blk, static_cast<uint64_t>(ins.imm_signed)) },
                                        rule->effects);
                }
                bool is_64 = (var.find("64") != std::string::npos) || ins.dest_is_64bit ||
                             (ins.sr_write.present && ins.sr_write.count >= 2) ||
                             (ins.sr_read.present && ins.sr_read.count >= 2 && ins.canonical_mnemonic != Mnemonic::ACMPXCHG);
                uint8_t sr = ins.sr_read.present ? ins.sr_read.base_reg : 0;
                SsaId tok = sb.current_token(blk);

                if (rule->op_class == OpClass::MemAtom1) {
                    std::string aop = "ainc";
                    auto it = ins.modifier_strs.find("atom1_opc");
                    if (it != ins.modifier_strs.end()) aop = it->second;
                    SsaId delta = sb.emit_const_u32(blk, (aop == "adec") ? 0xFFFFFFFFu : 1u);
                    SsaNode at{};
                    at.op = SsaOp::AtomicRmw; at.type = ValType::TupleValToken;
                    at.aux_str = "aadd"; at.aux_bool = false;
                    at.args = { byte_addr, delta, tok };
                    at.effects = rule->effects; at.is_barrier_or_cf = true; at.is_side_effect = true;
                    SsaId tup = sb.emit(blk, std::move(at));
                    SsaNode ev{}; ev.op = SsaOp::TupleExtract; ev.type = ValType::U32; ev.imm_u32 = 0; ev.args = { tup };
                    SsaNode et{}; et.op = SsaOp::TupleExtract; et.type = ValType::Token; et.imm_u32 = 1; et.args = { tup };
                    SsaId old_v = sb.emit(blk, std::move(ev));
                    sb.set_token(blk, sb.emit(blk, std::move(et)));
                    if (ins.sr_write.present) sb.set_reg(blk, ins.sr_write.base_reg, old_v);
                    return;
                }
                if (rule->op_class == OpClass::MemAxchg) {
                    SsaNode at{};
                    at.op = SsaOp::AtomicExchange; at.type = ValType::TupleValToken;
                    at.args = { byte_addr, sb.current_reg(blk, sr), tok };
                    at.effects = rule->effects; at.is_barrier_or_cf = true; at.is_side_effect = true;
                    SsaId tup = sb.emit(blk, std::move(at));
                    SsaNode ev{}; ev.op = SsaOp::TupleExtract; ev.type = ValType::U32; ev.imm_u32 = 0; ev.args = { tup };
                    SsaNode et{}; et.op = SsaOp::TupleExtract; et.type = ValType::Token; et.imm_u32 = 1; et.args = { tup };
                    SsaId old_v = sb.emit(blk, std::move(ev));
                    sb.set_token(blk, sb.emit(blk, std::move(et)));
                    if (ins.sr_write.present) sb.set_reg(blk, ins.sr_write.base_reg, old_v);
                    return;
                }
                if (rule->op_class == OpClass::MemAcmpxchg) {
                    SsaId cmp_v = sb.current_reg(blk, sr);
                    SsaId xchg_v = sb.current_reg(blk, sr + 1);
                    SsaNode at{};
                    at.op = SsaOp::AtomicCmpXchg; at.type = ValType::TupleValToken;
                    at.args = { byte_addr, xchg_v, cmp_v, tok };
                    at.effects = rule->effects; at.is_barrier_or_cf = true; at.is_side_effect = true;
                    SsaId tup = sb.emit(blk, std::move(at));
                    SsaNode ev{}; ev.op = SsaOp::TupleExtract; ev.type = ValType::U32; ev.imm_u32 = 0; ev.args = { tup };
                    SsaNode et{}; et.op = SsaOp::TupleExtract; et.type = ValType::Token; et.imm_u32 = 1; et.args = { tup };
                    SsaId old_v = sb.emit(blk, std::move(ev));
                    sb.set_token(blk, sb.emit(blk, std::move(et)));
                    if (ins.sr_write.present) sb.set_reg(blk, ins.sr_write.base_reg, old_v);
                    return;
                }

                // General ATOM / ATOM_RETURN
                SsaId arg = is_64
                    ? emit_op(blk, SsaOp::Pack64, ValType::U64, { sb.current_reg(blk, sr), sb.current_reg(blk, sr + 1) }, rule->effects)
                    : sb.current_reg(blk, sr);
                std::string aop = "aadd";
                auto it = ins.modifier_strs.find("atom_opc");
                if (it != ins.modifier_strs.end()) aop = it->second;
                SsaNode at{};
                at.op = SsaOp::AtomicRmw; at.type = ValType::TupleValToken;
                at.aux_str = aop; at.aux_bool = is_64;
                at.args = { byte_addr, arg, tok };
                at.effects = rule->effects; at.is_barrier_or_cf = true; at.is_side_effect = true;
                SsaId tup = sb.emit(blk, std::move(at));
                SsaNode ev{}; ev.op = SsaOp::TupleExtract; ev.type = is_64 ? ValType::U64 : ValType::U32; ev.imm_u32 = 0; ev.args = { tup };
                SsaNode et{}; et.op = SsaOp::TupleExtract; et.type = ValType::Token; et.imm_u32 = 1; et.args = { tup };
                SsaId old_v = sb.emit(blk, std::move(ev));
                sb.set_token(blk, sb.emit(blk, std::move(et)));
                if (ins.sr_write.present) {
                    if (is_64) {
                        SsaId lo = 0, hi = 0;
                        emit_split64(blk, old_v, &lo, &hi);
                        sb.set_reg(blk, ins.sr_write.base_reg, lo);
                        sb.set_reg(blk, ins.sr_write.base_reg + 1, hi);
                    } else {
                        sb.set_reg(blk, ins.sr_write.base_reg, old_v);
                    }
                }
                return;
            }

            case OpClass::VertAttr: {
                uint32_t attr_idx = 0;
                auto it = ins.fields.find("attribute_index");
                if (it != ins.fields.end()) attr_idx = it->second;
                SsaNode ld{};
                ld.op = SsaOp::LoadVertAttr; ld.type = ValType::V4F32; ld.imm_u32 = attr_idx;
                ld.is_barrier_or_cf = true;
                SsaId vec = sb.emit(blk, std::move(ld));
                uint8_t base_r = ins.sr_write.present ? ins.sr_write.base_reg : 0;
                uint8_t cnt = ins.sr_write.present ? std::max<uint8_t>(1, ins.sr_write.count) : 4;
                for (uint8_t c = 0; c < cnt && c < 4; ++c) {
                    SsaNode ex{}; ex.op = SsaOp::VecExtract; ex.type = ValType::F32; ex.imm_u32 = c; ex.args = { vec };
                    sb.set_reg(blk, base_r + c, sb.emit_bitcast(blk, ValType::U32, sb.emit(blk, std::move(ex))));
                }
                return;
            }

            case OpClass::FragVar: {
                bool is_flat = (ins.canonical_mnemonic == Mnemonic::LD_VAR_FLAT_IMM ||
                                ins.canonical_mnemonic == Mnemonic::LD_VAR_FLAT);
                uint32_t var_loc = 0;
                auto it = ins.fields.find("index");
                if (it != ins.fields.end()) var_loc = it->second / 16u;
                SsaNode ld{};
                ld.op = SsaOp::LoadFragVarying; ld.type = ValType::V4F32;
                ld.imm_u32 = var_loc; ld.aux_bool = is_flat;
                ld.is_barrier_or_cf = true;
                SsaId vec = sb.emit(blk, std::move(ld));
                uint8_t base_r = ins.sr_write.present ? ins.sr_write.base_reg : 0;
                uint8_t cnt = ins.sr_write.present ? std::max<uint8_t>(1, ins.sr_write.count) : 4;
                for (uint8_t c = 0; c < cnt && c < 4; ++c) {
                    SsaNode ex{}; ex.op = SsaOp::VecExtract; ex.type = ValType::F32; ex.imm_u32 = c; ex.args = { vec };
                    sb.set_reg(blk, base_r + c, sb.emit_bitcast(blk, ValType::U32, sb.emit(blk, std::move(ex))));
                }
                return;
            }

            case OpClass::TexVarSample: {
                SsaNode ld{};
                ld.op = SsaOp::LoadFragVarying; ld.type = ValType::V4F32; ld.imm_u32 = 0; ld.aux_bool = false;
                ld.is_barrier_or_cf = true;
                SsaId uv4 = sb.emit(blk, std::move(ld));
                SsaNode exu{}; exu.op = SsaOp::VecExtract; exu.type = ValType::F32; exu.imm_u32 = 0; exu.args = { uv4 };
                SsaNode exv{}; exv.op = SsaOp::VecExtract; exv.type = ValType::F32; exv.imm_u32 = 1; exv.args = { uv4 };
                SsaId uv = emit_op(blk, SsaOp::VecConstruct2, ValType::V2F32,
                                   { sb.emit(blk, std::move(exu)), sb.emit(blk, std::move(exv)) });

                SsaNode ts{};
                ts.op = SsaOp::TexSample; ts.type = ValType::TupleValToken;
                ts.imm_u32 = 0; ts.aux_bool = (opts.stage == ShaderStage::Fragment); ts.aux_bool2 = false;
                ts.args = { uv, sb.current_token(blk) };
                ts.is_barrier_or_cf = true; ts.is_side_effect = true;
                SsaId tup = sb.emit(blk, std::move(ts));
                SsaNode ev{}; ev.op = SsaOp::TupleExtract; ev.type = ValType::V4F32; ev.imm_u32 = 0; ev.args = { tup };
                SsaNode et{}; et.op = SsaOp::TupleExtract; et.type = ValType::Token; et.imm_u32 = 1; et.args = { tup };
                SsaId sampled = sb.emit(blk, std::move(ev));
                sb.set_token(blk, sb.emit(blk, std::move(et)));

                uint8_t base_r = ins.sr_write.present ? ins.sr_write.base_reg : 0;
                uint8_t cnt = ins.sr_write.present ? std::max<uint8_t>(1, ins.sr_write.count) : 4;
                for (uint8_t c = 0; c < cnt && c < 4; ++c) {
                    SsaNode ex{}; ex.op = SsaOp::VecExtract; ex.type = ValType::F32; ex.imm_u32 = c; ex.args = { sampled };
                    sb.set_reg(blk, base_r + c, sb.emit_bitcast(blk, ValType::U32, sb.emit(blk, std::move(ex))));
                }
                return;
            }

            case OpClass::TexSample: {
                uint8_t coord_r = ins.sr_read.present ? ins.sr_read.base_reg : 0;
                SsaId u = sb.emit_bitcast(blk, ValType::F32, sb.current_reg(blk, coord_r));
                SsaId v = sb.emit_bitcast(blk, ValType::F32, sb.current_reg(blk, coord_r + 1));
                SsaId uv = emit_op(blk, SsaOp::VecConstruct2, ValType::V2F32, { u, v });
                bool is_gather = (ins.canonical_mnemonic == Mnemonic::TEX_GATHER || ins.mnemonic == Mnemonic::TEX_GATHER);
                uint32_t gcomp = 0;
                if (is_gather) {
                    auto it_gc = ins.fields.find("gather_component");
                    if (it_gc != ins.fields.end()) gcomp = it_gc->second & 3u;
                }
                SsaNode ts{};
                ts.op = SsaOp::TexSample; ts.type = ValType::TupleValToken;
                ts.imm_u32 = 0; ts.imm_u32_2 = gcomp;
                ts.aux_bool = (opts.stage == ShaderStage::Fragment && !is_gather);
                ts.aux_bool2 = is_gather;
                ts.args = { uv, sb.current_token(blk) };
                ts.is_barrier_or_cf = true; ts.is_side_effect = true;
                SsaId tup = sb.emit(blk, std::move(ts));
                SsaNode ev{}; ev.op = SsaOp::TupleExtract; ev.type = ValType::V4F32; ev.imm_u32 = 0; ev.args = { tup };
                SsaNode et{}; et.op = SsaOp::TupleExtract; et.type = ValType::Token; et.imm_u32 = 1; et.args = { tup };
                SsaId sampled = sb.emit(blk, std::move(ev));
                sb.set_token(blk, sb.emit(blk, std::move(et)));

                uint8_t base_r = ins.sr_write.present ? ins.sr_write.base_reg : 0;
                uint8_t cnt = ins.sr_write.present ? std::max<uint8_t>(1, ins.sr_write.count) : 4;
                for (uint8_t c = 0; c < cnt && c < 4; ++c) {
                    SsaNode ex{}; ex.op = SsaOp::VecExtract; ex.type = ValType::F32; ex.imm_u32 = c; ex.args = { sampled };
                    sb.set_reg(blk, base_r + c, sb.emit_bitcast(blk, ValType::U32, sb.emit(blk, std::move(ex))));
                }
                return;
            }

            case OpClass::FragAtest: {
                SsaId cov = (ins.srcs.size() >= 1) ? fetch_src_int32(sb, blk, ins, 0, false) : sb.emit_const_u32(blk, 1u);
                if (ins.has_dest) store_dest_u32(sb, blk, ins, cov);
                // Corner case: ATEST outside Fragment writes dest, skips kill
                if (opts.stage == ShaderStage::Fragment && !ins.srcs.empty()) {
                    SsaId is_zero = emit_op(blk, SsaOp::ICmpEq, ValType::Bool, { cov, sb.emit_const_u32(blk, 0u) });
                    SsaNode k{};
                    k.op = SsaOp::Kill; k.type = ValType::Void; k.aux_bool = true; // conditional kill on is_zero
                    k.args = { is_zero, sb.current_token(blk) };
                    k.is_barrier_or_cf = true; k.is_side_effect = true;
                    sb.emit(blk, std::move(k));
                }
                return;
            }

            case OpClass::FragZsEmit: {
                uint8_t sr = ins.sr_read.present ? ins.sr_read.base_reg : 0;
                SsaId z_f32 = sb.emit_bitcast(blk, ValType::F32, sb.current_reg(blk, sr));
                SsaNode st{};
                st.op = SsaOp::StoreBuiltinOutput; st.type = ValType::Token;
                st.imm_u32 = static_cast<uint32_t>(spirv::BuiltIn::FragDepth);
                st.aux_str = "gl_FragDepth";
                st.args = { z_f32, sb.current_token(blk) };
                st.effects = rule->effects; st.is_barrier_or_cf = true; st.is_side_effect = true;
                sb.set_token(blk, sb.emit(blk, std::move(st)));
                return;
            }

            case OpClass::FragBlend: {
                uint8_t sr = ins.sr_read.present ? ins.sr_read.base_reg : 0;
                uint32_t rt_loc = 0;
                if (!ins.srcs.empty() && ins.srcs[0].symbolic_name.find("blend_descriptor_") == 0) {
                    char c = ins.srcs[0].symbolic_name.back();
                    if (c >= '0' && c <= '7') rt_loc = static_cast<uint32_t>(c - '0');
                }
                SsaId col = emit_op(blk, SsaOp::VecConstruct4, ValType::V4F32, {
                    sb.emit_bitcast(blk, ValType::F32, sb.current_reg(blk, sr + 0)),
                    sb.emit_bitcast(blk, ValType::F32, sb.current_reg(blk, sr + 1)),
                    sb.emit_bitcast(blk, ValType::F32, sb.current_reg(blk, sr + 2)),
                    sb.emit_bitcast(blk, ValType::F32, sb.current_reg(blk, sr + 3)),
                });
                SsaNode st{};
                st.op = SsaOp::StoreFragColor; st.type = ValType::Token; st.imm_u32 = rt_loc;
                st.args = { col, sb.current_token(blk) };
                st.is_barrier_or_cf = true; st.is_side_effect = true;
                sb.set_token(blk, sb.emit(blk, std::move(st)));
                return;
            }

            case OpClass::SysGclk: {
                store_dest_u64(sb, blk, ins, sb.emit_const_u64(blk, 0ULL));
                return;
            }
        }
    }
};

} // namespace

Pass1Result run_pass1_lowering(
    const cfg::ControlFlowGraph& primary_cfg,
    const cfg::ControlFlowGraph& secondary_cfg,
    const ValhallTranslateOptions& opts) {
    Pass1Result res;
    res.is_fragment_stage = (opts.stage == ShaderStage::Fragment);
    res.is_compute_or_pilot = (opts.stage == ShaderStage::Compute || opts.stage == ShaderStage::Pilot);
    res.local_size_x = (opts.stage == ShaderStage::Pilot) ? 1u : std::max(1u, opts.local_size_x);
    res.local_size_y = (opts.stage == ShaderStage::Pilot) ? 1u : std::max(1u, opts.local_size_y);
    res.local_size_z = (opts.stage == ShaderStage::Pilot) ? 1u : std::max(1u, opts.local_size_z);

    // Sub-block splitting for primary (and optional secondary Varying EBIN)
    SplitCfgResult pri = split_cfg_subblocks(primary_cfg);
    SplitCfgResult sec;
    bool has_secondary = (opts.stage == ShaderStage::Vertex && !secondary_cfg.blocks.empty());
    if (has_secondary) {
        sec = split_cfg_subblocks(secondary_cfg);
    }

    if (pri.blocks.empty()) {
        res.ok = true;
        return res;
    }

    uint32_t sec_offset = static_cast<uint32_t>(pri.blocks.size());
    std::vector<std::vector<DecodedValhallInst>> all_insts = std::move(pri.block_insts);
    res.ssa.blocks = std::move(pri.blocks);

    if (has_secondary && !sec.blocks.empty()) {
        // Redirect primary terminal ends to secondary entry block (sec_offset)
        for (uint32_t i = 0; i < sec_offset; ++i) {
            if (res.ssa.blocks[i].is_terminated_by_end) {
                res.ssa.blocks[i].is_terminated_by_end = false;
                res.ssa.blocks[i].fallthrough_block = static_cast<int32_t>(sec_offset);
                sec.blocks[0].preds.push_back(i);
            }
        }
        // Shift secondary block IDs and edges by sec_offset
        for (size_t j = 0; j < sec.blocks.size(); ++j) {
            auto sb_blk = sec.blocks[j];
            sb_blk.id += sec_offset;
            for (auto& p : sb_blk.preds) {
                if (j > 0 || p < sec_offset) {
                    // preds from primary were already absolute; preds from sec need +sec_offset
                }
            }
            if (j > 0) {
                for (auto& p : sb_blk.preds) p += sec_offset;
            } else {
                // sec.blocks[0].preds might have primary preds at the back; only shift internal sec preds if any
                for (size_t k = 0; k < sb_blk.preds.size(); ++k) {
                    if (sb_blk.preds[k] >= sec_offset) continue; // already primary
                    // Wait: sec.blocks[0] could have a backedge from sec.blocks[m] before we appended!
                    // Handle cleanly: only primary preds were pushed above; original sec preds were < sec.blocks.size().
                }
            }
            if (sb_blk.branch_target_block >= 0) sb_blk.branch_target_block += static_cast<int32_t>(sec_offset);
            if (sb_blk.fallthrough_block >= 0) sb_blk.fallthrough_block += static_cast<int32_t>(sec_offset);
            if (sb_blk.loop_backedge_block >= 0) sb_blk.loop_backedge_block += static_cast<int32_t>(sec_offset);
            if (sb_blk.loop_merge_block >= 0) sb_blk.loop_merge_block += static_cast<int32_t>(sec_offset);
            if (sb_blk.selection_merge_block >= 0) sb_blk.selection_merge_block += static_cast<int32_t>(sec_offset);
            res.ssa.blocks.push_back(std::move(sb_blk));
            all_insts.push_back(std::move(sec.block_insts[j]));
        }
        // Fix secondary[0] preds cleanly: recompute preds from scratch for entire module
        for (auto& b : res.ssa.blocks) b.preds.clear();
        for (uint32_t i = 0; i < res.ssa.blocks.size(); ++i) {
            const auto& b = res.ssa.blocks[i];
            if (b.branch_target_block >= 0) res.ssa.blocks[b.branch_target_block].preds.push_back(i);
            if (b.fallthrough_block >= 0 && b.fallthrough_block != b.branch_target_block) {
                res.ssa.blocks[b.fallthrough_block].preds.push_back(i);
            }
        }
    }

    size_t num_blks = res.ssa.blocks.size();
    res.ssa.block_in_regs.resize(num_blks);
    res.ssa.block_out_regs.resize(num_blks);
    res.ssa.block_in_token.assign(num_blks, kInvalidSsa);
    res.ssa.block_out_token.assign(num_blks, kInvalidSsa);

    Pass1Ctx ctx(res.ssa, opts, res);

    // Initialize block 0 registers to 0 and initial effect token
    SsaId zero_u32 = ctx.sb.emit_const_u32(0, 0u);
    for (uint32_t r = 0; r < 64; ++r) {
        res.ssa.block_in_regs[0][r] = zero_u32;
        res.ssa.block_out_regs[0][r] = zero_u32;
    }
    SsaNode init_tok{};
    init_tok.op = SsaOp::Nop;
    init_tok.type = ValType::Token;
    SsaId tok0 = ctx.sb.emit(0, std::move(init_tok));
    res.ssa.block_in_token[0] = tok0;
    res.ssa.block_out_token[0] = tok0;

    // Emit stage preamble in block 0
    ctx.emit_stage_preamble(0, primary_cfg.instructions);

    // Lower all blocks in program order
    for (uint32_t b = 0; b < num_blks; ++b) {
        if (b > 0) {
            const auto& blk = res.ssa.blocks[b];
            bool single_forward_pred = (blk.preds.size() == 1 && blk.preds[0] < b);
            if (single_forward_pred) {
                uint32_t p = blk.preds[0];
                res.ssa.block_in_regs[b] = res.ssa.block_out_regs[p];
                res.ssa.block_out_regs[b] = res.ssa.block_out_regs[p];
                res.ssa.block_in_token[b] = res.ssa.block_out_token[p];
                res.ssa.block_out_token[b] = res.ssa.block_out_token[p];
            } else {
                // Allocate Phi placeholders for r0..r63 and effect_token; trivial ones are collapsed by seal_all_phis()
                for (uint32_t r = 0; r < 64; ++r) {
                    SsaNode phi{};
                    phi.op = SsaOp::Phi;
                    phi.type = ValType::U32;
                    phi.aux_index = r;
                    phi.aux_bool = false;
                    SsaId pid = ctx.sb.emit(b, std::move(phi));
                    res.ssa.block_in_regs[b][r] = pid;
                    res.ssa.block_out_regs[b][r] = pid;
                }
                SsaNode tphi{};
                tphi.op = SsaOp::Phi;
                tphi.type = ValType::Token;
                tphi.aux_bool = true;
                SsaId tpid = ctx.sb.emit(b, std::move(tphi));
                res.ssa.block_in_token[b] = tpid;
                res.ssa.block_out_token[b] = tpid;
            }
        }

        // Corner case: Vertex Position+Varying EBIN fusion reloads r59..r61 at Varying EBIN entry
        if (has_secondary && b == sec_offset) {
            SsaId vi = ctx.emit_builtin_input(b, spirv::BuiltIn::VertexIndex, ValType::U32, "gl_VertexIndex");
            SsaId ii = ctx.emit_builtin_input(b, spirv::BuiltIn::InstanceIndex, ValType::U32, "gl_InstanceIndex");
            ctx.sb.set_reg(b, 59, vi);
            ctx.sb.set_reg(b, 60, vi);
            ctx.sb.set_reg(b, 61, ii);
        }

        for (const auto& ins : all_insts[b]) {
            ctx.lower_instruction(b, ins);
            if (!res.error_message.empty()) {
                return res;
            }
        }
    }

    ctx.sb.seal_all_phis();
    res.ok = true;
    return res;
}

} // namespace valhall::compiler

#include "valhall_ssa_json.h"

#include <sstream>

namespace valhall::compiler {

namespace {

void write_json_escaped(std::ostringstream& os, const std::string& s) {
    os << '"';
    for (char c : s) {
        switch (c) {
            case '"': os << "\\\""; break;
            case '\\': os << "\\\\"; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20u) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(c)));
                    os << buf;
                } else {
                    os << c;
                }
                break;
        }
    }
    os << '"';
}

} // namespace

const char* val_type_name(ValType vt) {
    switch (vt) {
        case ValType::Void: return "Void";
        case ValType::Bool: return "Bool";
        case ValType::U32: return "U32";
        case ValType::S32: return "S32";
        case ValType::F32: return "F32";
        case ValType::U64: return "U64";
        case ValType::S64: return "S64";
        case ValType::V2F32: return "V2F32";
        case ValType::V2U32: return "V2U32";
        case ValType::V2S32: return "V2S32";
        case ValType::V4F32: return "V4F32";
        case ValType::V4U32: return "V4U32";
        case ValType::V4S32: return "V4S32";
        case ValType::V3U32: return "V3U32";
        case ValType::V2Bool: return "V2Bool";
        case ValType::TupleU32x2: return "TupleU32x2";
        case ValType::TupleValToken: return "TupleValToken";
        case ValType::Token: return "Token";
    }
    return "Void";
}

const char* ssa_op_name(tables::SsaOp op) {
    using tables::SsaOp;
    switch (op) {
        case SsaOp::Atan: return "Atan";
        case SsaOp::AtomicCmpXchg: return "AtomicCmpXchg";
        case SsaOp::AtomicExchange: return "AtomicExchange";
        case SsaOp::AtomicRmw: return "AtomicRmw";
        case SsaOp::BfeS: return "BfeS";
        case SsaOp::BfeU: return "BfeU";
        case SsaOp::Bfi: return "Bfi";
        case SsaOp::BitAnd: return "BitAnd";
        case SsaOp::BitAnd64: return "BitAnd64";
        case SsaOp::BitCount: return "BitCount";
        case SsaOp::BitNot: return "BitNot";
        case SsaOp::BitNot64: return "BitNot64";
        case SsaOp::BitOr: return "BitOr";
        case SsaOp::BitOr64: return "BitOr64";
        case SsaOp::BitReverse: return "BitReverse";
        case SsaOp::BitXor: return "BitXor";
        case SsaOp::BitXor64: return "BitXor64";
        case SsaOp::Bitcast: return "Bitcast";
        case SsaOp::Branch: return "Branch";
        case SsaOp::BranchCond: return "BranchCond";
        case SsaOp::Ceil: return "Ceil";
        case SsaOp::Clz: return "Clz";
        case SsaOp::ConstBool: return "ConstBool";
        case SsaOp::ConstF32: return "ConstF32";
        case SsaOp::ConstS32: return "ConstS32";
        case SsaOp::ConstU32: return "ConstU32";
        case SsaOp::ConstU64: return "ConstU64";
        case SsaOp::ControlBarrier: return "ControlBarrier";
        case SsaOp::ConvertFToS: return "ConvertFToS";
        case SsaOp::ConvertFToU: return "ConvertFToU";
        case SsaOp::ConvertSToF: return "ConvertSToF";
        case SsaOp::ConvertUToF: return "ConvertUToF";
        case SsaOp::Copy: return "Copy";
        case SsaOp::Cos: return "Cos";
        case SsaOp::Exp: return "Exp";
        case SsaOp::Exp2: return "Exp2";
        case SsaOp::FAbs: return "FAbs";
        case SsaOp::FAdd: return "FAdd";
        case SsaOp::FClamp: return "FClamp";
        case SsaOp::FCmpOrdEq: return "FCmpOrdEq";
        case SsaOp::FCmpOrdGe: return "FCmpOrdGe";
        case SsaOp::FCmpOrdGt: return "FCmpOrdGt";
        case SsaOp::FCmpOrdLe: return "FCmpOrdLe";
        case SsaOp::FCmpOrdLt: return "FCmpOrdLt";
        case SsaOp::FCmpOrdNe: return "FCmpOrdNe";
        case SsaOp::FCmpTotal: return "FCmpTotal";
        case SsaOp::FCmpUnordNe: return "FCmpUnordNe";
        case SsaOp::FDiv: return "FDiv";
        case SsaOp::FMax: return "FMax";
        case SsaOp::FMin: return "FMin";
        case SsaOp::FMul: return "FMul";
        case SsaOp::FNeg: return "FNeg";
        case SsaOp::FSub: return "FSub";
        case SsaOp::Floor: return "Floor";
        case SsaOp::Fma: return "Fma";
        case SsaOp::FrexpExp: return "FrexpExp";
        case SsaOp::FrexpSignificand: return "FrexpSignificand";
        case SsaOp::IAdd: return "IAdd";
        case SsaOp::IAdd64: return "IAdd64";
        case SsaOp::ICmpEq: return "ICmpEq";
        case SsaOp::ICmpNe: return "ICmpNe";
        case SsaOp::ICmpSGe: return "ICmpSGe";
        case SsaOp::ICmpSGt: return "ICmpSGt";
        case SsaOp::ICmpSLe: return "ICmpSLe";
        case SsaOp::ICmpSLt: return "ICmpSLt";
        case SsaOp::ICmpUGe: return "ICmpUGe";
        case SsaOp::ICmpUGt: return "ICmpUGt";
        case SsaOp::ICmpULe: return "ICmpULe";
        case SsaOp::ICmpULt: return "ICmpULt";
        case SsaOp::IDot4S: return "IDot4S";
        case SsaOp::IDot4SU: return "IDot4SU";
        case SsaOp::IDot4U: return "IDot4U";
        case SsaOp::IMul: return "IMul";
        case SsaOp::IMul64: return "IMul64";
        case SsaOp::ISub: return "ISub";
        case SsaOp::ISub64: return "ISub64";
        case SsaOp::InverseSqrt: return "InverseSqrt";
        case SsaOp::IsInf: return "IsInf";
        case SsaOp::IsNan: return "IsNan";
        case SsaOp::Kill: return "Kill";
        case SsaOp::Ldexp: return "Ldexp";
        case SsaOp::LoadBuiltinInput: return "LoadBuiltinInput";
        case SsaOp::LoadFau: return "LoadFau";
        case SsaOp::LoadFragVarying: return "LoadFragVarying";
        case SsaOp::LoadSidebandU32: return "LoadSidebandU32";
        case SsaOp::LoadSidebandU64: return "LoadSidebandU64";
        case SsaOp::LoadVertAttr: return "LoadVertAttr";
        case SsaOp::Log: return "Log";
        case SsaOp::Log2: return "Log2";
        case SsaOp::LogicalAnd: return "LogicalAnd";
        case SsaOp::LogicalNot: return "LogicalNot";
        case SsaOp::LogicalOr: return "LogicalOr";
        case SsaOp::NMax: return "NMax";
        case SsaOp::NMin: return "NMin";
        case SsaOp::Nop: return "Nop";
        case SsaOp::Pack64: return "Pack64";
        case SsaOp::PackHalf2x16: return "PackHalf2x16";
        case SsaOp::Phi: return "Phi";
        case SsaOp::Pow: return "Pow";
        case SsaOp::PsbLoad: return "PsbLoad";
        case SsaOp::PsbStore: return "PsbStore";
        case SsaOp::Return: return "Return";
        case SsaOp::RotL: return "RotL";
        case SsaOp::RotR: return "RotR";
        case SsaOp::RoundEven: return "RoundEven";
        case SsaOp::SAbs: return "SAbs";
        case SsaOp::SConvert32To64: return "SConvert32To64";
        case SsaOp::SNeg: return "SNeg";
        case SsaOp::Select: return "Select";
        case SsaOp::Shl: return "Shl";
        case SsaOp::Shl64: return "Shl64";
        case SsaOp::ShrS: return "ShrS";
        case SsaOp::ShrS64: return "ShrS64";
        case SsaOp::ShrU: return "ShrU";
        case SsaOp::ShrU64: return "ShrU64";
        case SsaOp::Sin: return "Sin";
        case SsaOp::Split64: return "Split64";
        case SsaOp::Sqrt: return "Sqrt";
        case SsaOp::StoreBuiltinOutput: return "StoreBuiltinOutput";
        case SsaOp::StoreFragColor: return "StoreFragColor";
        case SsaOp::StoreVertVarying: return "StoreVertVarying";
        case SsaOp::SubgroupBallot: return "SubgroupBallot";
        case SsaOp::SubgroupShuffle: return "SubgroupShuffle";
        case SsaOp::TexSample: return "TexSample";
        case SsaOp::Trunc: return "Trunc";
        case SsaOp::TupleExtract: return "TupleExtract";
        case SsaOp::UConvert32To64: return "UConvert32To64";
        case SsaOp::UConvert64To32: return "UConvert64To32";
        case SsaOp::UnpackHalf2x16: return "UnpackHalf2x16";
        case SsaOp::VecConstruct2: return "VecConstruct2";
        case SsaOp::VecConstruct3: return "VecConstruct3";
        case SsaOp::VecConstruct4: return "VecConstruct4";
        case SsaOp::VecExtract: return "VecExtract";
        case SsaOp::VecShuffle: return "VecShuffle";
    }
    return "Nop";
}

std::string serialize_ssa_module_json(const ValhallSsaModule& mod) {
    std::ostringstream os;
    os << "{\"blocks\":[";
    for (size_t b = 0; b < mod.blocks.size(); ++b) {
        const auto& blk = mod.blocks[b];
        if (b > 0) os << ',';
        os << "{\"id\":" << blk.id
           << ",\"preds\":[";
        for (size_t i = 0; i < blk.preds.size(); ++i) {
            if (i > 0) os << ',';
            os << blk.preds[i];
        }
        os << "],\"phis\":[";
        for (size_t i = 0; i < blk.phis.size(); ++i) {
            if (i > 0) os << ',';
            os << blk.phis[i];
        }
        os << "],\"nodes\":[";
        for (size_t i = 0; i < blk.nodes.size(); ++i) {
            if (i > 0) os << ',';
            os << blk.nodes[i];
        }
        os << "],\"branch_target\":" << blk.branch_target_block
           << ",\"fallthrough\":" << blk.fallthrough_block
           << ",\"is_cond\":" << (blk.is_conditional_branch ? "true" : "false")
           << ",\"is_uncond\":" << (blk.is_unconditional_branch ? "true" : "false")
           << ",\"is_end\":" << (blk.is_terminated_by_end ? "true" : "false")
           << ",\"is_discard\":" << (blk.is_terminated_by_discard ? "true" : "false")
           << ",\"is_loop_header\":" << (blk.is_loop_header ? "true" : "false")
           << ",\"loop_backedge\":" << blk.loop_backedge_block
           << ",\"loop_merge\":" << blk.loop_merge_block
           << ",\"sel_merge\":" << blk.selection_merge_block
           << ",\"branch_cond\":" << blk.branch_cond
           << '}';
    }
    os << "],\"nodes\":[";
    for (size_t i = 1; i < mod.nodes.size(); ++i) {
        const auto& n = mod.nodes[i];
        if (i > 1) os << ',';
        os << "{\"id\":" << n.id
           << ",\"op\":\"" << ssa_op_name(n.op) << '"'
           << ",\"type\":\"" << val_type_name(n.type) << '"'
           << ",\"blk\":" << n.block_id
           << ",\"args\":[";
        for (size_t a = 0; a < n.args.size(); ++a) {
            if (a > 0) os << ',';
            os << n.args[a];
        }
        os << ']';
        if (!n.phi_incomings.empty()) {
            os << ",\"phi\":[";
            for (size_t p = 0; p < n.phi_incomings.size(); ++p) {
                if (p > 0) os << ',';
                os << '[' << n.phi_incomings[p].pred_block << ',' << n.phi_incomings[p].value << ']';
            }
            os << ']';
        }
        if (n.imm_u32 != 0) os << ",\"imm_u32\":" << n.imm_u32;
        if (n.imm_u32_2 != 0) os << ",\"imm_u32_2\":" << n.imm_u32_2;
        if (n.imm_u64 != 0) os << ",\"imm_u64\":" << n.imm_u64;
        if (n.aux_index != 0) os << ",\"aux_index\":" << n.aux_index;
        if (n.aux_bool) os << ",\"aux_bool\":true";
        if (n.aux_bool2) os << ",\"aux_bool2\":true";
        if (!n.aux_str.empty()) {
            os << ",\"aux_str\":";
            write_json_escaped(os, n.aux_str);
        }
        if (n.orig_inst_idx >= 0) {
            os << ",\"orig_pc\":" << n.orig_pc;
        }
        os << '}';
    }
    os << "]}";
    return os.str();
}

} // namespace valhall::compiler

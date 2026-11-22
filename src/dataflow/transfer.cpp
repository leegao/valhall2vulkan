#include "transfer.h"

namespace valhall::dataflow {

namespace {

inline compiler::SsaId arg_or_invalid(const compiler::SsaNode& node, size_t idx) {
    return (idx < node.args.size()) ? node.args[idx] : compiler::kInvalidSsa;
}

}  // namespace

SsaOpKind classify_ssa_op(tables::SsaOp op) {
    using tables::SsaOp;
    switch (op) {
        case SsaOp::ConstBool:
        case SsaOp::ConstU32:
        case SsaOp::ConstS32:
        case SsaOp::ConstF32:
        case SsaOp::ConstU64:
            return SsaOpKind::Const;

        case SsaOp::Copy:
        case SsaOp::Bitcast:
            return SsaOpKind::Copy;

        case SsaOp::Phi:    return SsaOpKind::Phi;
        case SsaOp::Select: return SsaOpKind::Select;

        case SsaOp::IAdd:
        case SsaOp::IAdd64:
            return SsaOpKind::IntAdd;

        case SsaOp::ISub:
        case SsaOp::ISub64:
            return SsaOpKind::IntSub;

        case SsaOp::IMul:
        case SsaOp::IMul64:
            return SsaOpKind::IntMul;

        case SsaOp::BitAnd:
        case SsaOp::BitAnd64:
            return SsaOpKind::BitAnd;

        case SsaOp::BitOr:
        case SsaOp::BitOr64:
            return SsaOpKind::BitOr;

        case SsaOp::BitXor:
        case SsaOp::BitXor64:
            return SsaOpKind::BitXor;

        case SsaOp::BitNot:
        case SsaOp::BitNot64:
            return SsaOpKind::BitNot;

        case SsaOp::Shl:
        case SsaOp::Shl64:
        case SsaOp::ShrU:
        case SsaOp::ShrU64:
        case SsaOp::ShrS:
        case SsaOp::ShrS64:
        case SsaOp::RotL:
        case SsaOp::RotR:
            return SsaOpKind::Shift;

        case SsaOp::Pack64:       return SsaOpKind::Pack64;
        case SsaOp::Split64:      return SsaOpKind::Split64;
        case SsaOp::TupleExtract: return SsaOpKind::TupleExtract;

        case SsaOp::UConvert32To64:
        case SsaOp::SConvert32To64:
        case SsaOp::UConvert64To32:
        case SsaOp::ConvertFToS:
        case SsaOp::ConvertFToU:
        case SsaOp::ConvertSToF:
        case SsaOp::ConvertUToF:
            return SsaOpKind::Convert;

        case SsaOp::VecConstruct2:
        case SsaOp::VecConstruct3:
        case SsaOp::VecConstruct4:
        case SsaOp::VecExtract:
        case SsaOp::VecShuffle:
        case SsaOp::PackHalf2x16:
        case SsaOp::UnpackHalf2x16:
            return SsaOpKind::Vec;

        case SsaOp::LoadFau:          return SsaOpKind::LoadFau;
        case SsaOp::LoadBuiltinInput: return SsaOpKind::LoadBuiltin;

        case SsaOp::LoadSidebandU32:
        case SsaOp::LoadSidebandU64:
            return SsaOpKind::LoadSideband;

        case SsaOp::PsbLoad:          return SsaOpKind::PsbLoad;
        case SsaOp::PsbStore:         return SsaOpKind::PsbStore;
        case SsaOp::LoadVertAttr:     return SsaOpKind::LoadVertAttr;
        case SsaOp::LoadFragVarying:  return SsaOpKind::LoadFragVarying;
        case SsaOp::TexSample:
        case SsaOp::ImageQueryLod:    return SsaOpKind::TexSample;

        case SsaOp::StoreFragColor:
        case SsaOp::StoreVertVarying:
        case SsaOp::StoreBuiltinOutput:
            return SsaOpKind::StoreOutput;

        case SsaOp::AtomicRmw:
        case SsaOp::AtomicExchange:
        case SsaOp::AtomicCmpXchg:
            return SsaOpKind::Atomic;

        default:
            return SsaOpKind::Other;
    }
}

bool is_ssa_op_64bit(tables::SsaOp op) {
    using tables::SsaOp;
    switch (op) {
        case SsaOp::IAdd64:
        case SsaOp::ISub64:
        case SsaOp::IMul64:
        case SsaOp::BitAnd64:
        case SsaOp::BitOr64:
        case SsaOp::BitXor64:
        case SsaOp::BitNot64:
        case SsaOp::Shl64:
        case SsaOp::ShrU64:
        case SsaOp::ShrS64:
        case SsaOp::LoadSidebandU64:
            return true;
        default:
            return false;
    }
}

ShiftKind classify_ssa_shift_kind(tables::SsaOp op) {
    using tables::SsaOp;
    switch (op) {
        case SsaOp::ShrU:
        case SsaOp::ShrU64:
            return ShiftKind::ShrU;
        case SsaOp::ShrS:
        case SsaOp::ShrS64:
            return ShiftKind::ShrS;
        case SsaOp::RotL:
            return ShiftKind::RotL;
        case SsaOp::RotR:
            return ShiftKind::RotR;
        default:
            return ShiftKind::Shl;
    }
}

ConvertKind classify_ssa_convert_kind(tables::SsaOp op) {
    using tables::SsaOp;
    switch (op) {
        case SsaOp::UConvert32To64: return ConvertKind::U32ToU64;
        case SsaOp::SConvert32To64: return ConvertKind::S32ToS64;
        case SsaOp::UConvert64To32: return ConvertKind::U64ToU32;
        case SsaOp::ConvertFToS:    return ConvertKind::FloatToSInt;
        case SsaOp::ConvertFToU:    return ConvertKind::FloatToUInt;
        case SsaOp::ConvertSToF:    return ConvertKind::SIntToFloat;
        case SsaOp::ConvertUToF:    return ConvertKind::UIntToFloat;
        case SsaOp::Bitcast:        return ConvertKind::Bitcast;
        default:                    return ConvertKind::Other;
    }
}

void dispatch_ssa_transfer(const compiler::SsaNode& node, SsaTransferVisitorBase& visitor) {
    const compiler::SsaId dst = node.id;
    const SsaOpKind kind = classify_ssa_op(node.op);
    const bool is64 = is_ssa_op_64bit(node.op);

    switch (kind) {
        case SsaOpKind::Const:
            visitor.on_const(dst, node);
            return;
        case SsaOpKind::Copy:
            visitor.on_copy(dst, arg_or_invalid(node, 0), node);
            return;
        case SsaOpKind::Phi:
            visitor.on_phi(dst, node);
            return;
        case SsaOpKind::Select:
            visitor.on_select(dst, arg_or_invalid(node, 0),
                              arg_or_invalid(node, 1), arg_or_invalid(node, 2), node);
            return;
        case SsaOpKind::IntAdd:
            visitor.on_iadd(dst, arg_or_invalid(node, 0), arg_or_invalid(node, 1), is64, node);
            return;
        case SsaOpKind::IntSub:
            visitor.on_isub(dst, arg_or_invalid(node, 0), arg_or_invalid(node, 1), is64, node);
            return;
        case SsaOpKind::IntMul:
            visitor.on_imul(dst, arg_or_invalid(node, 0), arg_or_invalid(node, 1), is64, node);
            return;
        case SsaOpKind::BitAnd:
            visitor.on_and(dst, arg_or_invalid(node, 0), arg_or_invalid(node, 1), is64, node);
            return;
        case SsaOpKind::BitOr:
            visitor.on_or(dst, arg_or_invalid(node, 0), arg_or_invalid(node, 1), is64, node);
            return;
        case SsaOpKind::BitXor:
            visitor.on_xor(dst, arg_or_invalid(node, 0), arg_or_invalid(node, 1), is64, node);
            return;
        case SsaOpKind::BitNot:
            visitor.on_not(dst, arg_or_invalid(node, 0), is64, node);
            return;
        case SsaOpKind::Shift:
            visitor.on_shift(dst, arg_or_invalid(node, 0), arg_or_invalid(node, 1),
                             classify_ssa_shift_kind(node.op), is64, node);
            return;
        case SsaOpKind::Pack64:
            visitor.on_pack64(dst, arg_or_invalid(node, 0), arg_or_invalid(node, 1), node);
            return;
        case SsaOpKind::Split64:
            visitor.on_split64(dst, arg_or_invalid(node, 0), node);
            return;
        case SsaOpKind::TupleExtract:
            visitor.on_tuple_extract(dst, arg_or_invalid(node, 0), node.imm_u32, node);
            return;
        case SsaOpKind::Convert:
            visitor.on_convert(dst, arg_or_invalid(node, 0), node);
            return;
        case SsaOpKind::Vec:
            visitor.on_vec(dst, node);
            return;
        case SsaOpKind::LoadFau:
            visitor.on_load_fau(dst, node.imm_u32, node);
            return;
        case SsaOpKind::LoadBuiltin:
            visitor.on_load_builtin(dst, node.imm_u32, node);
            return;
        case SsaOpKind::LoadSideband:
            visitor.on_load_sideband(dst, node.imm_u32, is64, node);
            return;
        case SsaOpKind::PsbLoad:
            visitor.on_psb_load(dst, arg_or_invalid(node, 0), node.imm_u32, node);
            return;
        case SsaOpKind::PsbStore:
            visitor.on_psb_store(arg_or_invalid(node, 0), arg_or_invalid(node, 1), node.imm_u32, node);
            return;
        case SsaOpKind::LoadVertAttr:
            visitor.on_load_vert_attr(dst, node);
            return;
        case SsaOpKind::LoadFragVarying:
            visitor.on_load_frag_varying(dst, node);
            return;
        case SsaOpKind::TexSample:
            visitor.on_tex_sample(dst, node);
            return;
        case SsaOpKind::StoreOutput:
            visitor.on_store_output(node);
            return;
        case SsaOpKind::Atomic:
            visitor.on_atomic(dst, node);
            return;
        case SsaOpKind::Other:
        default:
            visitor.on_other(dst, node);
            return;
    }
}

}  // namespace valhall::dataflow

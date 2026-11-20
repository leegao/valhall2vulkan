#include "transfer.h"

namespace valhall::dataflow {

namespace {

inline compiler::SsaId arg_or_invalid(const compiler::SsaNode& node, size_t idx) {
    return (idx < node.args.size()) ? node.args[idx] : compiler::kInvalidSsa;
}

}  // namespace

void dispatch_ssa_transfer(const compiler::SsaNode& node, SsaTransferVisitorBase& visitor) {
    using tables::SsaOp;
    const compiler::SsaId dst = node.id;

    switch (node.op) {
        // --- Constants ---
        case SsaOp::ConstBool:
        case SsaOp::ConstU32:
        case SsaOp::ConstS32:
        case SsaOp::ConstF32:
        case SsaOp::ConstU64:
            visitor.on_const(dst, node);
            return;

        // --- Moves / Aliases / Phi / Select ---
        case SsaOp::Copy:
        case SsaOp::Bitcast:
            visitor.on_copy(dst, arg_or_invalid(node, 0), node);
            return;

        case SsaOp::Phi:
            visitor.on_phi(dst, node);
            return;

        case SsaOp::Select:
            visitor.on_select(dst, arg_or_invalid(node, 0),
                              arg_or_invalid(node, 1), arg_or_invalid(node, 2), node);
            return;

        // --- Integer arithmetic ---
        case SsaOp::IAdd:
            visitor.on_iadd(dst, arg_or_invalid(node, 0), arg_or_invalid(node, 1), false, node);
            return;
        case SsaOp::IAdd64:
            visitor.on_iadd(dst, arg_or_invalid(node, 0), arg_or_invalid(node, 1), true, node);
            return;

        case SsaOp::ISub:
            visitor.on_isub(dst, arg_or_invalid(node, 0), arg_or_invalid(node, 1), false, node);
            return;
        case SsaOp::ISub64:
            visitor.on_isub(dst, arg_or_invalid(node, 0), arg_or_invalid(node, 1), true, node);
            return;

        case SsaOp::IMul:
            visitor.on_imul(dst, arg_or_invalid(node, 0), arg_or_invalid(node, 1), false, node);
            return;
        case SsaOp::IMul64:
            visitor.on_imul(dst, arg_or_invalid(node, 0), arg_or_invalid(node, 1), true, node);
            return;

        // --- Bitwise logic ---
        case SsaOp::BitAnd:
            visitor.on_and(dst, arg_or_invalid(node, 0), arg_or_invalid(node, 1), false, node);
            return;
        case SsaOp::BitAnd64:
            visitor.on_and(dst, arg_or_invalid(node, 0), arg_or_invalid(node, 1), true, node);
            return;

        case SsaOp::BitOr:
            visitor.on_or(dst, arg_or_invalid(node, 0), arg_or_invalid(node, 1), false, node);
            return;
        case SsaOp::BitOr64:
            visitor.on_or(dst, arg_or_invalid(node, 0), arg_or_invalid(node, 1), true, node);
            return;

        case SsaOp::BitXor:
            visitor.on_xor(dst, arg_or_invalid(node, 0), arg_or_invalid(node, 1), false, node);
            return;
        case SsaOp::BitXor64:
            visitor.on_xor(dst, arg_or_invalid(node, 0), arg_or_invalid(node, 1), true, node);
            return;

        case SsaOp::BitNot:
            visitor.on_not(dst, arg_or_invalid(node, 0), false, node);
            return;
        case SsaOp::BitNot64:
            visitor.on_not(dst, arg_or_invalid(node, 0), true, node);
            return;

        // --- Shifts & rotates ---
        case SsaOp::Shl:
            visitor.on_shift(dst, arg_or_invalid(node, 0), arg_or_invalid(node, 1), ShiftKind::Shl, false, node);
            return;
        case SsaOp::Shl64:
            visitor.on_shift(dst, arg_or_invalid(node, 0), arg_or_invalid(node, 1), ShiftKind::Shl, true, node);
            return;
        case SsaOp::ShrU:
            visitor.on_shift(dst, arg_or_invalid(node, 0), arg_or_invalid(node, 1), ShiftKind::ShrU, false, node);
            return;
        case SsaOp::ShrU64:
            visitor.on_shift(dst, arg_or_invalid(node, 0), arg_or_invalid(node, 1), ShiftKind::ShrU, true, node);
            return;
        case SsaOp::ShrS:
            visitor.on_shift(dst, arg_or_invalid(node, 0), arg_or_invalid(node, 1), ShiftKind::ShrS, false, node);
            return;
        case SsaOp::ShrS64:
            visitor.on_shift(dst, arg_or_invalid(node, 0), arg_or_invalid(node, 1), ShiftKind::ShrS, true, node);
            return;
        case SsaOp::RotL:
            visitor.on_shift(dst, arg_or_invalid(node, 0), arg_or_invalid(node, 1), ShiftKind::RotL, false, node);
            return;
        case SsaOp::RotR:
            visitor.on_shift(dst, arg_or_invalid(node, 0), arg_or_invalid(node, 1), ShiftKind::RotR, false, node);
            return;

        // --- 64-bit packing / splitting / tuples / conversions / vectors ---
        case SsaOp::Pack64:
            visitor.on_pack64(dst, arg_or_invalid(node, 0), arg_or_invalid(node, 1), node);
            return;

        case SsaOp::Split64:
            visitor.on_split64(dst, arg_or_invalid(node, 0), node);
            return;

        case SsaOp::TupleExtract:
            visitor.on_tuple_extract(dst, arg_or_invalid(node, 0), node.imm_u32, node);
            return;

        case SsaOp::UConvert32To64:
        case SsaOp::SConvert32To64:
        case SsaOp::UConvert64To32:
        case SsaOp::ConvertFToS:
        case SsaOp::ConvertFToU:
        case SsaOp::ConvertSToF:
        case SsaOp::ConvertUToF:
            visitor.on_convert(dst, arg_or_invalid(node, 0), node);
            return;

        case SsaOp::VecConstruct2:
        case SsaOp::VecConstruct3:
        case SsaOp::VecConstruct4:
        case SsaOp::VecExtract:
        case SsaOp::VecShuffle:
        case SsaOp::PackHalf2x16:
        case SsaOp::UnpackHalf2x16:
            visitor.on_vec(dst, node);
            return;

        // --- Shader inputs, FAU uniforms, sideband, memory, textures, outputs, atomics ---
        case SsaOp::LoadFau:
            visitor.on_load_fau(dst, node.imm_u32, node);
            return;

        case SsaOp::LoadBuiltinInput:
            visitor.on_load_builtin(dst, node.imm_u32, node);
            return;

        case SsaOp::LoadSidebandU32:
            visitor.on_load_sideband(dst, node.imm_u32, false, node);
            return;
        case SsaOp::LoadSidebandU64:
            visitor.on_load_sideband(dst, node.imm_u32, true, node);
            return;

        case SsaOp::PsbLoad:
            visitor.on_psb_load(dst, arg_or_invalid(node, 0), node.imm_u32, node);
            return;

        case SsaOp::PsbStore:
            visitor.on_psb_store(arg_or_invalid(node, 0), arg_or_invalid(node, 1), node.imm_u32, node);
            return;

        case SsaOp::LoadVertAttr:
            visitor.on_load_vert_attr(dst, node);
            return;

        case SsaOp::LoadFragVarying:
            visitor.on_load_frag_varying(dst, node);
            return;

        case SsaOp::TexSample:
            visitor.on_tex_sample(dst, node);
            return;

        case SsaOp::StoreFragColor:
        case SsaOp::StoreVertVarying:
        case SsaOp::StoreBuiltinOutput:
            visitor.on_store_output(node);
            return;

        case SsaOp::AtomicRmw:
        case SsaOp::AtomicExchange:
        case SsaOp::AtomicCmpXchg:
            visitor.on_atomic(dst, node);
            return;

        default:
            visitor.on_other(dst, node);
            return;
    }
}

}  // namespace valhall::dataflow

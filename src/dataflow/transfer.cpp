#include "transfer.h"

#include <cstdlib>

namespace valhall::dataflow {

AbstractOperand AbstractOperand::from_decoded(const DecodedSource& s) {
    AbstractOperand op;
    op.is_64bit = s.is_64bit;
    op.reg_num = s.reg_num;
    op.discard_last_use = s.discard_last_use;
    op.fau_page = s.fau_page;
    op.symbolic_name = s.symbolic_name;
    op.swizzle_mod = s.swizzle_mod;
    op.abs = s.abs;
    op.neg = s.neg;
    op.not_mod = s.not_mod;

    switch (s.kind) {
        case SourceKind::Register:
            op.kind = Kind::Register;
            break;
        case SourceKind::FauUniform:
            op.kind = Kind::FauUniform;
            op.fau_byte_offset = static_cast<uint32_t>(s.fau_index64) * 8u + (s.word_select ? 4u : 0u);
            break;
        case SourceKind::SmallConstant: {
            op.kind = Kind::SmallConstant;
            uint32_t bits = s.imm_bits;
            // Apply common byte/half swizzle extraction for constant evaluation (e.g. k03020100.b2 -> 2)
            if (s.swizzle_mod == "b0") bits = bits & 0xFFu;
            else if (s.swizzle_mod == "b1") bits = (bits >> 8) & 0xFFu;
            else if (s.swizzle_mod == "b2") bits = (bits >> 16) & 0xFFu;
            else if (s.swizzle_mod == "b3") bits = (bits >> 24) & 0xFFu;
            else if (s.swizzle_mod == "h0") bits = bits & 0xFFFFu;
            else if (s.swizzle_mod == "h1") bits = (bits >> 16) & 0xFFFFu;
            if (s.not_mod) bits = ~bits;
            op.imm_u64 = bits;
            op.imm_s64 = static_cast<int32_t>(bits);
            break;
        }
        case SourceKind::FauSpecial:
            op.kind = Kind::FauSpecial;
            break;
    }
    return op;
}

AbstractOperand AbstractOperand::from_imm(uint64_t u64, int64_t s64) {
    AbstractOperand op;
    op.kind = Kind::Immediate;
    op.imm_u64 = u64;
    op.imm_s64 = s64;
    return op;
}

AbstractOperand AbstractOperand::from_reg(uint8_t reg, bool is64) {
    AbstractOperand op;
    op.kind = Kind::Register;
    op.reg_num = reg;
    op.is_64bit = is64;
    return op;
}

AbstractDest AbstractDest::from_alu(const DecodedValhallInst& insn) {
    AbstractDest d;
    d.present = insn.has_dest;
    d.base_reg = insn.dest_reg;
    d.is_64bit = insn.dest_is_64bit;
    d.reg_count = insn.dest_is_64bit ? 2u : 1u;
    return d;
}

AbstractDest AbstractDest::from_staging(const DecodedStaging& sr) {
    AbstractDest d;
    d.present = sr.present;
    d.base_reg = sr.base_reg;
    d.reg_count = sr.count;
    d.is_64bit = (sr.count >= 2);
    return d;
}

namespace {

AbstractOperand get_src(const DecodedValhallInst& insn, size_t idx) {
    if (idx < insn.srcs.size()) return AbstractOperand::from_decoded(insn.srcs[idx]);
    return AbstractOperand{};
}

int64_t extract_offset(const DecodedValhallInst& insn) {
    if (insn.has_imm) return insn.imm_signed;
    auto it = insn.fields.find("offset");
    if (it != insn.fields.end()) return static_cast<int64_t>(it->second);
    return 0;
}

uint32_t extract_field(const DecodedValhallInst& insn, const char* name, uint32_t fallback = 0) {
    auto it = insn.fields.find(name);
    return (it != insn.fields.end()) ? it->second : fallback;
}

// Valhall AND/OR/XOR encode the second logical operand as src2 (index 1 or 2 depending on shift-lop encoding).
AbstractOperand get_bitwise_rhs(const DecodedValhallInst& insn) {
    if (insn.has_imm) {
        return AbstractOperand::from_imm(insn.imm_unsigned, insn.imm_signed);
    }
    if (insn.srcs.size() >= 3) return get_src(insn, 2);
    if (insn.srcs.size() >= 2) return get_src(insn, 1);
    return AbstractOperand{};
}

}  // namespace

void dispatch_valhall_transfer(const DecodedValhallInst& insn, TransferVisitorBase& visitor) {
    visitor.on_inst_begin(insn);
    if (!insn.valid) {
        visitor.on_inst_end(insn);
        return;
    }

    const Mnemonic m = insn.mnemonic;
    const Mnemonic cm = insn.canonical_mnemonic;
    const AbstractDest alu_dst = AbstractDest::from_alu(insn);
    const AbstractDest sr_dst = AbstractDest::from_staging(insn.sr_write);

    // 1. PC-relative ADR (note: canonical_mnemonic is IADD_IMM, so match exact mnemonic first!)
    if (m == Mnemonic::ADR) {
        int64_t target = insn.has_branch_offset
                             ? insn.branch_target_pc
                             : (static_cast<int64_t>(insn.pc + 8) + insn.imm_signed);
        visitor.on_adr(alu_dst, target, insn);
    }
    // 2. Moves & Immediate loads
    else if (m == Mnemonic::MOV || m == Mnemonic::MOV_IMM || cm == Mnemonic::MOV ||
             cm == Mnemonic::MOV_IMM) {
        AbstractOperand src = insn.has_imm
                                  ? AbstractOperand::from_imm(insn.imm_unsigned, insn.imm_signed)
                                  : get_src(insn, 0);
        visitor.on_mov(alu_dst, src, insn);
    }
    // 3. Bitwise AND group (AND, AND_IMM, LSHIFT_AND, RSHIFT_AND, ARSHIFT_AND)
    else if (m == Mnemonic::AND || m == Mnemonic::AND_IMM ||
             cm == Mnemonic::AND || cm == Mnemonic::AND_IMM ||
             cm == Mnemonic::LSHIFT_AND || cm == Mnemonic::RSHIFT_AND || cm == Mnemonic::ARSHIFT_AND) {
        visitor.on_and(alu_dst, get_src(insn, 0), get_bitwise_rhs(insn), insn);
    }
    // 4. Bitwise NOT
    else if (m == Mnemonic::NOT || m == Mnemonic::NOT_OLD) {
        visitor.on_not(alu_dst, get_src(insn, 0), insn);
    }
    // 5. Shifts (LSHIFT, RSHIFT, ARSHIFT, ROTATE — check before OR/XOR since canonical is *_OR)
    else if (m == Mnemonic::LSHIFT || m == Mnemonic::RSHIFT || m == Mnemonic::ARSHIFT) {
        ShiftKind sk = (m == Mnemonic::LSHIFT)
                           ? ShiftKind::LogicalLeft
                           : (m == Mnemonic::ARSHIFT ? ShiftKind::ArithmeticRight : ShiftKind::LogicalRight);
        AbstractOperand amt = insn.has_imm
                                  ? AbstractOperand::from_imm(insn.imm_unsigned, insn.imm_signed)
                                  : get_src(insn, 1);
        visitor.on_shift(alu_dst, get_src(insn, 0), amt, sk, insn);
    }
    // 6. Bitwise OR group (OR, OR_IMM, LSHIFT_OR, RSHIFT_OR, ARSHIFT_OR)
    else if (m == Mnemonic::OR || m == Mnemonic::OR_IMM ||
             cm == Mnemonic::OR || cm == Mnemonic::OR_IMM ||
             cm == Mnemonic::LSHIFT_OR || cm == Mnemonic::RSHIFT_OR || cm == Mnemonic::ARSHIFT_OR) {
        visitor.on_or(alu_dst, get_src(insn, 0), get_bitwise_rhs(insn), insn);
    }
    // 7. Bitwise XOR group (XOR, XOR_IMM, LSHIFT_XOR, RSHIFT_XOR, ARSHIFT_XOR)
    else if (m == Mnemonic::XOR || m == Mnemonic::XOR_IMM ||
             cm == Mnemonic::XOR || cm == Mnemonic::XOR_IMM ||
             cm == Mnemonic::LSHIFT_XOR || cm == Mnemonic::RSHIFT_XOR || cm == Mnemonic::ARSHIFT_XOR) {
        visitor.on_xor(alu_dst, get_src(insn, 0), get_bitwise_rhs(insn), insn);
    }
    // 8. Pointer / Scaled-Add (SHADD)
    else if (m == Mnemonic::SHADD || cm == Mnemonic::SHADD) {
        uint32_t sh = extract_field(insn, "shift", 0);
        visitor.on_shadd(alu_dst, get_src(insn, 0), get_src(insn, 1), sh, insn);
    }
    // 9. Integer Add group (IADD, IADD_IMM, IADDC, SEG_ADD)
    else if (m == Mnemonic::IADD || m == Mnemonic::IADD_IMM || m == Mnemonic::IADDC || m == Mnemonic::SEG_ADD ||
             cm == Mnemonic::IADD || cm == Mnemonic::IADD_IMM || cm == Mnemonic::SEG_ADD) {
        AbstractOperand rhs = insn.has_imm
                                  ? AbstractOperand::from_imm(insn.imm_unsigned, insn.imm_signed)
                                  : get_src(insn, 1);
        visitor.on_iadd(alu_dst, get_src(insn, 0), rhs, insn);
    }
    // 10. Integer Sub group (ISUB, ISUBB, SEG_SUB)
    else if (m == Mnemonic::ISUB || m == Mnemonic::ISUBB || m == Mnemonic::SEG_SUB ||
             cm == Mnemonic::ISUB || cm == Mnemonic::SEG_SUB) {
        visitor.on_isub(alu_dst, get_src(insn, 0), get_src(insn, 1), insn);
    }
    // 11. Integer Mul group (IMUL, IDPADD)
    else if (m == Mnemonic::IMUL || m == Mnemonic::IDPADD || cm == Mnemonic::IMUL || cm == Mnemonic::IDPADD) {
        visitor.on_imul(alu_dst, get_src(insn, 0), get_src(insn, 1), insn);
    }
    // 12. Integer Unary group (IABS, CLZ, POPCOUNT, BITREV)
    else if (m == Mnemonic::IABS || m == Mnemonic::CLZ || m == Mnemonic::POPCOUNT || m == Mnemonic::BITREV) {
        visitor.on_iunary(alu_dst, get_src(insn, 0), insn);
    }
    // 13. Conditional Select (CSEL, MUX)
    else if (m == Mnemonic::CSEL || cm == Mnemonic::CSEL || m == Mnemonic::MUX || cm == Mnemonic::MUX) {
        visitor.on_csel(alu_dst, get_src(insn, 0), get_src(insn, 1), get_src(insn, 2), get_src(insn, 3), insn);
    }
    // 14. Comparisons (ICMP, FCMP, ICMP_OR, FCMP_OR, ICMP_AND, FCMP_AND)
    else if (insn.category == SemanticCategory::CMP_SELECT) {
        visitor.on_cmp(alu_dst, get_src(insn, 0), get_src(insn, 1), insn);
    }
    // 15. Floating-point ALU & SFU
    else if (m == Mnemonic::FMA || cm == Mnemonic::FMA) {
        visitor.on_fma(alu_dst, get_src(insn, 0), get_src(insn, 1), get_src(insn, 2), insn);
    } else if (m == Mnemonic::FMUL || cm == Mnemonic::FMUL) {
        visitor.on_fmul(alu_dst, get_src(insn, 0), get_src(insn, 1), insn);
    } else if (m == Mnemonic::FADD || m == Mnemonic::FADD_IMM || cm == Mnemonic::FADD || cm == Mnemonic::FADD_IMM) {
        AbstractOperand rhs = insn.has_imm
                                  ? AbstractOperand::from_imm(insn.imm_unsigned, insn.imm_signed)
                                  : get_src(insn, 1);
        visitor.on_fadd(alu_dst, get_src(insn, 0), rhs, insn);
    } else if (m == Mnemonic::FMIN || m == Mnemonic::FMAX) {
        visitor.on_fminmax(alu_dst, get_src(insn, 0), get_src(insn, 1), insn);
    } else if (insn.category == SemanticCategory::FP_ARITH || insn.category == SemanticCategory::FP_SFU) {
        visitor.on_funary(alu_dst, get_src(insn, 0), insn);
    }
    // 16. Conversions
    else if (insn.category == SemanticCategory::CONVERSION) {
        visitor.on_convert(alu_dst, get_src(insn, 0), insn);
    }
    // 17. Memory Loads
    else if (cm == Mnemonic::LOAD || cm == Mnemonic::LD_CHECKED || cm == Mnemonic::LD_MULTI ||
             m == Mnemonic::LOAD || m == Mnemonic::LD_CHECKED || m == Mnemonic::LD_MULTI) {
        visitor.on_load(sr_dst, get_src(insn, 0), extract_offset(insn), insn);
    }
    // 18. Memory Stores
    else if (cm == Mnemonic::STORE || cm == Mnemonic::ST_CHECKED || cm == Mnemonic::ST_MULTI ||
             m == Mnemonic::STORE || m == Mnemonic::ST_CHECKED || m == Mnemonic::ST_MULTI) {
        visitor.on_store(insn.sr_read, get_src(insn, 0), extract_offset(insn), insn);
    }
    // 19. Memory Atomics
    else if (insn.category == SemanticCategory::MEMORY_ATOMIC) {
        visitor.on_atomic(sr_dst, insn.sr_read, get_src(insn, 0), extract_offset(insn), insn);
    }
    // 20. LEA_BUF_IMM / Vertex / Varying / Texture / Blend
    else if (m == Mnemonic::LEA_BUF_IMM || m == Mnemonic::LEA_ATTR_IMM || m == Mnemonic::LEA_TEX_IMM) {
        visitor.on_lea(sr_dst.present ? sr_dst : alu_dst, get_src(insn, 0),
                       extract_field(insn, "index", 0), extract_field(insn, "table", 0), insn);
    } else if (m == Mnemonic::LD_ATTR_IMM || cm == Mnemonic::LD_ATTR_IMM) {
        visitor.on_ld_attr(sr_dst, extract_field(insn, "index", 0), extract_field(insn, "table", 0), insn);
    } else if (m == Mnemonic::LD_VAR_IMM || cm == Mnemonic::LD_VAR_IMM) {
        visitor.on_ld_var(sr_dst, extract_field(insn, "index", 0), extract_field(insn, "table", 0), insn);
    } else if (insn.category == SemanticCategory::TEXTURE) {
        visitor.on_tex(sr_dst, extract_field(insn, "texture_index", 0), extract_field(insn, "sampler_index", 0), insn);
    } else if (m == Mnemonic::BLEND || cm == Mnemonic::BLEND) {
        int32_t rt_idx = -1;
        for (const auto& s : insn.srcs) {
            if (s.kind == SourceKind::FauSpecial && s.symbolic_name.rfind("blend_descriptor_", 0) == 0) {
                rt_idx = std::atoi(s.symbolic_name.c_str() + 17);
                break;
            }
        }
        visitor.on_blend(insn.sr_read, rt_idx, insn);
    }
    // 21. Control Flow
    else if (insn.category == SemanticCategory::CONTROL_FLOW) {
        visitor.on_branch(get_src(insn, 0), insn.branch_target_pc, insn);
    }
    // 22. NOP / Barrier / System
    else if (m == Mnemonic::NOP || m == Mnemonic::BARRIER || insn.category == SemanticCategory::SYSTEM) {
        visitor.on_nop_or_barrier(insn);
    }
    // 23. Fallback
    else {
        visitor.on_unhandled(sr_dst.present ? sr_dst : alu_dst, insn);
    }

    // Fire ISA live-range kills (`^` discard_last_use) for consumed registers that were not rewritten
    for (const auto& s : insn.srcs) {
        if (s.kind == SourceKind::Register && s.discard_last_use) {
            auto written = [&](uint8_t r) {
                if (alu_dst.present && r >= alu_dst.base_reg && r < alu_dst.base_reg + alu_dst.reg_count) return true;
                if (sr_dst.present && r >= sr_dst.base_reg && r < sr_dst.base_reg + sr_dst.reg_count) return true;
                return false;
            };
            if (!written(s.reg_num)) visitor.on_reg_killed(s.reg_num, insn);
            if (s.is_64bit && s.reg_num + 1u < 64u && !written(static_cast<uint8_t>(s.reg_num + 1u))) {
                visitor.on_reg_killed(static_cast<uint8_t>(s.reg_num + 1u), insn);
            }
        }
    }

    visitor.on_inst_end(insn);
}

}  // namespace valhall::dataflow

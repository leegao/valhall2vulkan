#include "pointer_analysis.h"

#include <cstdio>
#include <sstream>

namespace valhall::dataflow {

namespace {

std::string hex_str(int64_t v) {
    char buf[32];
    if (v < 0) std::snprintf(buf, sizeof(buf), "-0x%llx", static_cast<unsigned long long>(-v));
    else std::snprintf(buf, sizeof(buf), "0x%llx", static_cast<unsigned long long>(v));
    return buf;
}

std::string format_fau_slot(uint32_t off, uint32_t sz) {
    uint32_t slot = off / 8u;
    uint32_t lane = (off % 8u) / 4u;
    if (sz >= 8 && lane == 0) return "u" + std::to_string(slot);
    return "u" + std::to_string(slot) + ".w" + std::to_string(lane);
}

}  // namespace

TrackedVal TrackedVal::make_const(int64_t c) {
    TrackedVal v;
    v.kind = Kind::ConstInt;
    v.const_val = c;
    v.label = hex_str(c);
    return v;
}

TrackedVal TrackedVal::make_abi(const std::string& name) {
    TrackedVal v;
    v.kind = Kind::AbiPreload;
    v.symbol_name = name;
    v.label = name;
    return v;
}

TrackedVal TrackedVal::make_fau_dst_base(int64_t off) {
    TrackedVal v;
    v.kind = Kind::FauDstBase;
    v.byte_offset = off;
    v.symbol_name = "FAU_0";
    v.label = (off == 0) ? "pilot_fau_dst_va (FAU_0)" : ("FAU_0+" + hex_str(off));
    return v;
}

TrackedVal TrackedVal::make_srt_base(int64_t off) {
    TrackedVal v;
    v.kind = Kind::SrtBase;
    v.byte_offset = off;
    v.symbol_name = "SRT_0";
    v.label = (off == 0) ? "srt_base_va (SRT_0)" : ("SRT_0+" + hex_str(off));
    return v;
}

TrackedVal TrackedVal::make_srt_rack(int32_t rack_idx, bool masked) {
    TrackedVal v;
    v.kind = Kind::SrtRackPtr;
    v.rack = rack_idx;
    v.masked_56bit = masked;
    v.symbol_name = "SRT[" + std::to_string(rack_idx) + "]";
    v.label = v.symbol_name;
    return v;
}

TrackedVal TrackedVal::make_buffer_addr(int32_t rack_idx, int32_t desc, const std::string& sym, int64_t off, bool indexed) {
    TrackedVal v;
    v.kind = Kind::BufferAddr;
    v.rack = rack_idx;
    v.desc_idx = desc;
    v.byte_offset = off;
    v.is_indexed = indexed;
    v.symbol_name = sym;
    if (indexed) {
        v.label = "&" + sym + "[...]";
    } else if (rack_idx >= 0 && desc >= 0 && off == 0) {
        v.label = "&" + sym + " (SRT[" + std::to_string(rack_idx) + "]->Buf[" + std::to_string(desc) + "].addr)";
    } else if (off != 0) {
        v.label = "&" + sym + "+" + hex_str(off);
    } else {
        v.label = "&" + sym;
    }
    return v;
}

TrackedVal TrackedVal::make_buffer_size(int32_t rack_idx, int32_t desc, const std::string& sym) {
    TrackedVal v;
    v.kind = Kind::BufferSize;
    v.rack = rack_idx;
    v.desc_idx = desc;
    v.symbol_name = sym;
    v.label = "sizeof(" + sym + ")";
    return v;
}

TrackedVal TrackedVal::make_symbol_value(int32_t rack_idx, int32_t desc, const std::string& sym, bool is_derived) {
    TrackedVal v;
    v.kind = Kind::SymbolValue;
    v.rack = rack_idx;
    v.desc_idx = desc;
    v.symbol_name = sym;
    v.derived = is_derived;
    v.label = is_derived ? ("derived(" + sym + ")") : sym;
    return v;
}

TrackedVal TrackedVal::make_data_section_ptr(int64_t target_pc, int64_t sec_off, bool indexed) {
    TrackedVal v;
    v.kind = Kind::DataSectionPtr;
    v.byte_offset = target_pc;
    v.extra_offset = sec_off;
    v.is_indexed = indexed;
    v.symbol_name = "data_section";
    v.label = "data_section+" + hex_str(sec_off) + " (LUT @ " + hex_str(target_pc) + ")";
    return v;
}

TrackedVal TrackedVal::make_tls_ptr(int64_t off) {
    TrackedVal v;
    v.kind = Kind::TlsPtr;
    v.byte_offset = off;
    v.symbol_name = "tls_ptr";
    v.label = (off == 0) ? "tls_ptr" : ("tls_ptr+" + hex_str(off));
    return v;
}

TrackedVal TrackedVal::make_wls_ptr(int64_t off) {
    TrackedVal v;
    v.kind = Kind::WlsPtr;
    v.byte_offset = off;
    v.symbol_name = "wls_ptr";
    v.label = (off == 0) ? "wls_ptr" : ("wls_ptr+" + hex_str(off));
    return v;
}

TrackedVal TrackedVal::with_offset(int64_t delta) const {
    if (delta == 0) return *this;
    TrackedVal out = *this;
    if (kind == Kind::FauDstBase) return make_fau_dst_base(byte_offset + delta);
    if (kind == Kind::SrtBase) return make_srt_base(byte_offset + delta);
    if (kind == Kind::BufferAddr) return make_buffer_addr(rack, desc_idx, symbol_name, byte_offset + delta, is_indexed);
    if (kind == Kind::TlsPtr) return make_tls_ptr(byte_offset + delta);
    if (kind == Kind::WlsPtr) return make_wls_ptr(byte_offset + delta);
    if (kind == Kind::DataSectionPtr) return make_data_section_ptr(byte_offset + delta, extra_offset + delta, is_indexed);
    out.byte_offset += delta;
    return out;
}

TrackedVal TrackedVal::as_indexed() const {
    TrackedVal out = *this;
    if (kind == Kind::BufferAddr) return make_buffer_addr(rack, desc_idx, symbol_name, byte_offset, true);
    if (kind == Kind::DataSectionPtr) return make_data_section_ptr(byte_offset, extra_offset, true);
    out.is_indexed = true;
    return out;
}

std::string TrackedVal::format_short() const {
    if (!label.empty()) return label;
    if (!symbol_name.empty()) return symbol_name;
    switch (kind) {
        case Kind::Bottom: return "⊥";
        case Kind::Top: return "⊤";
        case Kind::ConstInt: return hex_str(const_val);
        default: return "?";
    }
}

// ============================================================================
// Lattice operations: Bottom <= concrete <= Top
// ============================================================================
bool TrackedValDomain::equals(const Value& a, const Value& b) {
    if (a.kind != b.kind) return false;
    if (a.kind == TrackedVal::Kind::Bottom || a.kind == TrackedVal::Kind::Top) return true;
    return a.const_val == b.const_val &&
           a.byte_offset == b.byte_offset &&
           a.extra_offset == b.extra_offset &&
           a.rack == b.rack &&
           a.desc_idx == b.desc_idx &&
           a.is_indexed == b.is_indexed &&
           a.masked_56bit == b.masked_56bit &&
           a.derived == b.derived &&
           a.symbol_name == b.symbol_name &&
           a.label == b.label;
}

bool TrackedValDomain::leq(const Value& a, const Value& b) {
    if (a.is_bottom() || b.is_top()) return true;
    if (b.is_bottom() || a.is_top()) return false;
    if (equals(a, b)) return true;
    // Pointer specialization: fixed-offset BufferAddr <= dynamically-indexed BufferAddr of same buffer
    if (a.kind == TrackedVal::Kind::BufferAddr && b.kind == TrackedVal::Kind::BufferAddr &&
        a.rack == b.rack && a.desc_idx == b.desc_idx && a.symbol_name == b.symbol_name &&
        !a.is_indexed && b.is_indexed) {
        return true;
    }
    // SymbolValue: direct <= derived of same symbol
    if (a.kind == TrackedVal::Kind::SymbolValue && b.kind == TrackedVal::Kind::SymbolValue &&
        a.symbol_name == b.symbol_name && !a.derived && b.derived) {
        return true;
    }
    return false;
}

TrackedVal TrackedValDomain::join(const Value& a, const Value& b) {
    if (a.is_bottom()) return b;
    if (b.is_bottom()) return a;
    if (a.is_top() || b.is_top()) return top();
    if (equals(a, b)) return a;

    // Same buffer with different offsets/indexing -> generalize to indexed BufferAddr
    if (a.kind == TrackedVal::Kind::BufferAddr && b.kind == TrackedVal::Kind::BufferAddr &&
        a.rack == b.rack && a.desc_idx == b.desc_idx && a.symbol_name == b.symbol_name) {
        return TrackedVal::make_buffer_addr(a.rack, a.desc_idx, a.symbol_name, 0, true);
    }
    // Same SRT rack with differing 56-bit mask flag -> preserve rack pointer
    if (a.kind == TrackedVal::Kind::SrtRackPtr && b.kind == TrackedVal::Kind::SrtRackPtr &&
        a.rack == b.rack) {
        return TrackedVal::make_srt_rack(a.rack, a.masked_56bit && b.masked_56bit);
    }
    // Same symbol value with differing derived flag -> derived
    if (a.kind == TrackedVal::Kind::SymbolValue && b.kind == TrackedVal::Kind::SymbolValue &&
        a.symbol_name == b.symbol_name) {
        return TrackedVal::make_symbol_value(a.rack, a.desc_idx, a.symbol_name, true);
    }
    // Same data section pointer with different indexing -> indexed
    if (a.kind == TrackedVal::Kind::DataSectionPtr && b.kind == TrackedVal::Kind::DataSectionPtr &&
        a.byte_offset == b.byte_offset) {
        return TrackedVal::make_data_section_ptr(a.byte_offset, a.extra_offset, true);
    }
    return top();
}

TrackedVal TrackedValDomain::meet(const Value& a, const Value& b) {
    if (a.is_top()) return b;
    if (b.is_top()) return a;
    if (a.is_bottom() || b.is_bottom()) return bottom();
    if (equals(a, b)) return a;
    if (leq(a, b)) return a;
    if (leq(b, a)) return b;
    return bottom();
}

// ============================================================================
// Initial ABI Entry State
// ============================================================================
PointerState make_initial_pointer_state(const PointerAnalysisEnv& env) {
    PointerState st;
    for (const auto& kv : env.initial_fau) {
        st.set_fau(kv.first, kv.second);
    }
    switch (env.role) {
        case PointerAnalysisEnv::StageRole::Pilot:
            st.set_reg_pair(0, TrackedVal::make_fau_dst_base(0));
            st.set_reg_pair(2, TrackedVal::make_srt_base(0));
            break;
        case PointerAnalysisEnv::StageRole::MainCompute:
            st.set_reg(55, TrackedVal::make_abi("WorkGroupID.x"));
            st.set_reg(56, TrackedVal::make_abi("WorkGroupID.y"));
            st.set_reg(57, TrackedVal::make_abi("WorkGroupID.z"));
            st.set_reg(60, TrackedVal::make_abi("GlobalInvocationID.x"));
            st.set_reg(61, TrackedVal::make_abi("GlobalInvocationID.y"));
            st.set_reg(62, TrackedVal::make_abi("GlobalInvocationID.z"));
            break;
        case PointerAnalysisEnv::StageRole::Vertex:
            st.set_reg(59, TrackedVal::make_abi("VertexStreamIndex"));
            st.set_reg(60, TrackedVal::make_abi("VertexIndex"));
            st.set_reg(61, TrackedVal::make_abi("InstanceIndex"));
            break;
        case PointerAnalysisEnv::StageRole::Fragment:
            st.set_reg(59, TrackedVal::make_abi("FragCoord.z"));
            st.set_reg(60, TrackedVal::make_abi("sample_coverage_mask"));
            st.set_reg(61, TrackedVal::make_abi("varying_barycentric"));
            break;
        default:
            break;
    }
    return st;
}

// ============================================================================
// Pointer & Resource Transfer Visitor
// ============================================================================
namespace {

class PointerTransferVisitor : public DomainTransferVisitor<TrackedValDomain> {
public:
    PointerTransferVisitor(const PointerAnalysisEnv& env,
                           std::map<uint64_t, std::vector<std::string>>* notes_by_pc,
                           std::map<uint32_t, PilotFauWrite>* pilot_writes)
        : DomainTransferVisitor<TrackedValDomain>(nullptr),
          env_(env),
          notes_by_pc_(notes_by_pc),
          pilot_writes_(pilot_writes) {}

    TrackedVal eval_operand(const AbstractOperand& op) const override {
        switch (op.kind) {
            case AbstractOperand::Kind::Immediate:
            case AbstractOperand::Kind::SmallConstant:
                return TrackedVal::make_const(op.imm_s64);
            case AbstractOperand::Kind::FauSpecial:
                if (op.symbolic_name == "tls_ptr") return TrackedVal::make_tls_ptr(0);
                if (op.symbolic_name == "wls_ptr") return TrackedVal::make_wls_ptr(0);
                if (op.symbolic_name.rfind("blend_descriptor_", 0) == 0) {
                    TrackedVal v;
                    v.kind = TrackedVal::Kind::BlendDescRef;
                    v.desc_idx = std::atoi(op.symbolic_name.c_str() + 17);
                    v.symbol_name = op.symbolic_name;
                    v.label = op.symbolic_name;
                    return v;
                }
                return TrackedVal::make_top();
            case AbstractOperand::Kind::Register:
                return state_ ? state_->get_reg(op.reg_num) : TrackedVal::make_top();
            case AbstractOperand::Kind::FauUniform:
                return state_ ? state_->get_fau(op.fau_byte_offset) : TrackedVal::make_top();
            default:
                return TrackedVal::make_top();
        }
    }

    void on_inst_begin(const DecodedValhallInst& insn) override {
        cur_pc_ = insn.pc;
        cur_notes_.clear();
    }

    void on_inst_end(const DecodedValhallInst& /*insn*/) override {
        if (notes_by_pc_ && !cur_notes_.empty()) {
            (*notes_by_pc_)[cur_pc_] = cur_notes_;
        }
    }

    void on_mov(const AbstractDest& dst, const AbstractOperand& src, const DecodedValhallInst& /*insn*/) override {
        // Preserve DataSectionPtr across `MOV.i32 r61, pc.w1` (hi32 of ADR pair)
        if (src.kind == AbstractOperand::Kind::FauSpecial && src.symbolic_name == "pc" &&
            dst.present && state_->get_reg(dst.base_reg).kind == TrackedVal::Kind::DataSectionPtr) {
            return;
        }
        write_dest(dst, eval_operand(src));
    }

    void on_and(const AbstractDest& dst, const AbstractOperand& lhs, const AbstractOperand& rhs, const DecodedValhallInst& insn) override {
        TrackedVal lv = eval_operand(lhs);
        TrackedVal rv = eval_operand(rhs);
        // 56-bit pointer mask (`AND_IMM r_hi, r_hi, #0xffffff` or `AND r_hi, r_hi, r_mask`)
        if (lv.kind == TrackedVal::Kind::SrtRackPtr &&
            ((rv.kind == TrackedVal::Kind::ConstInt && (rv.const_val == 0xffffff || rv.const_val == 0xffffffff)) ||
             (lhs.is_reg() && dst.present && lhs.reg_num == dst.base_reg))) {
            TrackedVal masked = TrackedVal::make_srt_rack(lv.rack, true);
            write_dest(dst, masked);
            if (dst.base_reg > 0 && state_->get_reg(static_cast<uint8_t>(dst.base_reg - 1)).kind == TrackedVal::Kind::SrtRackPtr) {
                state_->set_reg(static_cast<uint8_t>(dst.base_reg - 1), masked);
            }
            cur_notes_.push_back("mask 56-bit VA (" + masked.label + ")");
            return;
        }
        if (lv.kind == TrackedVal::Kind::ConstInt && rv.kind == TrackedVal::Kind::ConstInt) {
            write_dest(dst, TrackedVal::make_const(lv.const_val & rv.const_val));
            return;
        }
        if (propagate_derived_symbol(dst, lv, rv)) return;
        DomainTransferVisitor<TrackedValDomain>::on_and(dst, lhs, rhs, insn);
    }

    void on_or(const AbstractDest& dst, const AbstractOperand& lhs, const AbstractOperand& rhs, const DecodedValhallInst& insn) override {
        TrackedVal lv = eval_operand(lhs);
        TrackedVal rv = eval_operand(rhs);
        if (lv.kind == TrackedVal::Kind::ConstInt && rv.kind == TrackedVal::Kind::ConstInt) {
            write_dest(dst, TrackedVal::make_const(lv.const_val | rv.const_val));
            return;
        }
        if (propagate_derived_symbol(dst, lv, rv)) return;
        DomainTransferVisitor<TrackedValDomain>::on_or(dst, lhs, rhs, insn);
    }

    void on_xor(const AbstractDest& dst, const AbstractOperand& lhs, const AbstractOperand& rhs, const DecodedValhallInst& insn) override {
        TrackedVal lv = eval_operand(lhs);
        TrackedVal rv = eval_operand(rhs);
        if (lv.kind == TrackedVal::Kind::ConstInt && rv.kind == TrackedVal::Kind::ConstInt) {
            write_dest(dst, TrackedVal::make_const(lv.const_val ^ rv.const_val));
            return;
        }
        if (propagate_derived_symbol(dst, lv, rv)) return;
        DomainTransferVisitor<TrackedValDomain>::on_xor(dst, lhs, rhs, insn);
    }

    void on_shift(const AbstractDest& dst, const AbstractOperand& lhs, const AbstractOperand& amount, ShiftKind kind, const DecodedValhallInst& insn) override {
        TrackedVal lv = eval_operand(lhs);
        TrackedVal av = eval_operand(amount);
        if (lv.kind == TrackedVal::Kind::ConstInt && av.kind == TrackedVal::Kind::ConstInt) {
            uint64_t u = static_cast<uint64_t>(lv.const_val);
            uint32_t sh = static_cast<uint32_t>(av.const_val) & 63u;
            int64_t out = 0;
            if (kind == ShiftKind::LogicalLeft) out = static_cast<int64_t>(u << sh);
            else if (kind == ShiftKind::LogicalRight) out = static_cast<int64_t>(u >> sh);
            else out = lv.const_val >> sh;
            write_dest(dst, TrackedVal::make_const(out));
            return;
        }
        if (propagate_derived_symbol(dst, lv, av)) return;
        DomainTransferVisitor<TrackedValDomain>::on_shift(dst, lhs, amount, kind, insn);
    }

    void on_iadd(const AbstractDest& dst, const AbstractOperand& lhs, const AbstractOperand& rhs, const DecodedValhallInst& insn) override {
        TrackedVal lv = eval_operand(lhs);
        TrackedVal rv = eval_operand(rhs);
        if (lv.kind == TrackedVal::Kind::ConstInt && rv.kind == TrackedVal::Kind::ConstInt) {
            write_dest(dst, TrackedVal::make_const(lv.const_val + rv.const_val));
            return;
        }
        if (lv.is_pointer() && rv.kind == TrackedVal::Kind::ConstInt) {
            write_dest(dst, lv.with_offset(rv.const_val));
            return;
        }
        if (rv.is_pointer() && lv.kind == TrackedVal::Kind::ConstInt) {
            write_dest(dst, rv.with_offset(lv.const_val));
            return;
        }
        if (lv.is_pointer()) {
            write_dest(dst, lv.as_indexed());
            return;
        }
        if (propagate_derived_symbol(dst, lv, rv)) return;
        DomainTransferVisitor<TrackedValDomain>::on_iadd(dst, lhs, rhs, insn);
    }

    void on_isub(const AbstractDest& dst, const AbstractOperand& lhs, const AbstractOperand& rhs, const DecodedValhallInst& insn) override {
        TrackedVal lv = eval_operand(lhs);
        TrackedVal rv = eval_operand(rhs);
        if (lv.kind == TrackedVal::Kind::ConstInt && rv.kind == TrackedVal::Kind::ConstInt) {
            write_dest(dst, TrackedVal::make_const(lv.const_val - rv.const_val));
            return;
        }
        if (lv.is_pointer() && rv.kind == TrackedVal::Kind::ConstInt) {
            write_dest(dst, lv.with_offset(-rv.const_val));
            return;
        }
        if (propagate_derived_symbol(dst, lv, rv)) return;
        DomainTransferVisitor<TrackedValDomain>::on_isub(dst, lhs, rhs, insn);
    }

    void on_imul(const AbstractDest& dst, const AbstractOperand& lhs, const AbstractOperand& rhs, const DecodedValhallInst& /*insn*/) override {
        TrackedVal lv = eval_operand(lhs);
        TrackedVal rv = eval_operand(rhs);
        if (lv.kind == TrackedVal::Kind::ConstInt && rv.kind == TrackedVal::Kind::ConstInt) {
            write_dest(dst, TrackedVal::make_const(lv.const_val * rv.const_val));
            return;
        }
        if (propagate_derived_symbol(dst, lv, rv)) return;
        write_dest(dst, TrackedVal::make_top());
    }

    void on_shadd(const AbstractDest& dst, const AbstractOperand& base, const AbstractOperand& index, uint32_t shift, const DecodedValhallInst& /*insn*/) override {
        TrackedVal bv = eval_operand(base);
        TrackedVal iv = eval_operand(index);
        if (bv.is_pointer()) {
            if (iv.kind == TrackedVal::Kind::ConstInt) {
                TrackedVal res = bv.with_offset(iv.const_val << shift);
                write_dest(dst, res);
                if (base.is_reg()) cur_notes_.push_back("ptr -> " + res.format_short());
                return;
            }
            TrackedVal res = bv.as_indexed();
            write_dest(dst, res);
            if (base.is_reg()) cur_notes_.push_back("ptr -> " + res.format_short());
            return;
        }
        write_dest(dst, TrackedVal::make_top());
    }

    void on_csel(const AbstractDest& dst, const AbstractOperand& cond0, const AbstractOperand& /*cond1*/,
                 const AbstractOperand& true_val, const AbstractOperand& false_val, const DecodedValhallInst& /*insn*/) override {
        TrackedVal tv = eval_operand(true_val);
        TrackedVal fv = eval_operand(false_val);
        TrackedVal cv = eval_operand(cond0);
        // Pilot shader bounds-check pattern: `CSEL.u32.gt r_hi, buf_size, #3, r_hi, k80002000`
        if (tv.kind == TrackedVal::Kind::BufferAddr && true_val.is_reg() && dst.present &&
            true_val.reg_num == dst.base_reg) {
            write_dest(dst, tv);
            cur_notes_.push_back("bounds-check &" + tv.symbol_name);
            return;
        }
        if (cv.kind == TrackedVal::Kind::SymbolValue) {
            write_dest(dst, TrackedVal::make_symbol_value(cv.rack, cv.desc_idx, cv.symbol_name, true));
            return;
        }
        write_dest(dst, TrackedValDomain::join(tv, fv));
    }

    void on_adr(const AbstractDest& dst, int64_t target_pc, const DecodedValhallInst& /*insn*/) override {
        if (env_.data_section_offset != 0xFFFFFFFFu &&
            target_pc >= static_cast<int64_t>(env_.data_section_offset)) {
            int64_t sec_off = target_pc - static_cast<int64_t>(env_.data_section_offset);
            TrackedVal dv = TrackedVal::make_data_section_ptr(target_pc, sec_off, false);
            // ADR writes the low 32-bit half; reserve the 64-bit pair for subsequent `MOV r_hi, pc.w1`
            if (dst.present) {
                state_->set_reg_pair(dst.base_reg, dv);
            }
            cur_notes_.push_back(dv.label);
            return;
        }
        write_dest(dst, TrackedVal::make_top());
    }

    void on_load(const AbstractDest& dst, const AbstractOperand& base, int64_t offset, const DecodedValhallInst& /*insn*/) override {
        TrackedVal bv = eval_operand(base);
        if (bv.kind == TrackedVal::Kind::SrtBase && (bv.byte_offset + offset) >= 0) {
            int32_t rack = static_cast<int32_t>((bv.byte_offset + offset) / 16);
            TrackedVal rv = TrackedVal::make_srt_rack(rack, false);
            write_dest(dst, rv);
            cur_notes_.push_back("chase SRT[" + std::to_string(rack) + "] (Resource[" + std::to_string(rack) + "].address)");
            return;
        }
        if (bv.kind == TrackedVal::Kind::SrtRackPtr && offset >= 0) {
            int32_t desc_idx = static_cast<int32_t>(offset / 32);
            int32_t field_off = static_cast<int32_t>(offset % 32);
            std::string sym = env_.lookup_srt_symbol
                                  ? env_.lookup_srt_symbol(static_cast<uint8_t>(bv.rack), static_cast<uint32_t>(desc_idx))
                                  : "";
            if (sym.empty()) sym = "Buf[" + std::to_string(desc_idx) + "]";
            const char* fname = (field_off == 4) ? "size" : (field_off == 8 ? "address" : "field");
            cur_notes_.push_back("chase SRT[" + std::to_string(bv.rack) + "]->Buf[" +
                                 std::to_string(desc_idx) + "]." + fname + " (" + sym + ")");
            if (field_off == 4) {
                write_dest(dst, TrackedVal::make_buffer_size(bv.rack, desc_idx, sym));
            } else {
                write_dest(dst, TrackedVal::make_buffer_addr(bv.rack, desc_idx, sym, 0, false));
            }
            return;
        }
        if (bv.kind == TrackedVal::Kind::BufferAddr) {
            int64_t eff_off = bv.byte_offset + offset;
            std::string member;
            if (!bv.is_indexed && bv.rack >= 0 && bv.desc_idx >= 0 && eff_off >= 0 && env_.lookup_symbol_member) {
                member = env_.lookup_symbol_member(static_cast<uint8_t>(bv.rack),
                                                   static_cast<uint32_t>(bv.desc_idx),
                                                   static_cast<uint32_t>(eff_off));
            } else if (bv.rack >= 0 && bv.desc_idx >= 0 && eff_off >= 0 && env_.lookup_symbol_member) {
                member = env_.lookup_symbol_member(static_cast<uint8_t>(bv.rack),
                                                   static_cast<uint32_t>(bv.desc_idx),
                                                   static_cast<uint32_t>(eff_off));
            }
            if (member.empty()) member = bv.format_short();
            cur_notes_.push_back("load " + member);
            write_dest(dst, TrackedVal::make_symbol_value(bv.rack, bv.desc_idx, member, false));
            return;
        }
        if (bv.kind == TrackedVal::Kind::DataSectionPtr) {
            cur_notes_.push_back("load " + bv.label);
            write_dest(dst, TrackedVal::make_symbol_value(-1, -1, bv.label, false));
            return;
        }
        if (bv.kind == TrackedVal::Kind::TlsPtr || bv.kind == TrackedVal::Kind::WlsPtr) {
            TrackedVal ptr = bv.with_offset(offset);
            cur_notes_.push_back("load " + ptr.label);
            write_dest(dst, state_->get_sym_mem(bv.symbol_name, ptr.byte_offset));
            return;
        }
        write_dest(dst, TrackedVal::make_top());
    }

    void on_store(const DecodedStaging& src, const AbstractOperand& base, int64_t offset, const DecodedValhallInst& /*insn*/) override {
        TrackedVal bv = eval_operand(base);
        TrackedVal sv = (src.present && state_) ? state_->get_reg(src.base_reg) : TrackedVal::make_top();
        if (bv.kind == TrackedVal::Kind::FauDstBase && (bv.byte_offset + offset) >= 0) {
            uint32_t fau_off = static_cast<uint32_t>(bv.byte_offset + offset);
            uint32_t byte_sz = static_cast<uint32_t>(src.count) * 4u;
            state_->set_fau(fau_off, sv);
            if (pilot_writes_) {
                (*pilot_writes_)[fau_off] = PilotFauWrite{fau_off, byte_sz, sv};
            }
            std::string slot = format_fau_slot(fau_off, byte_sz);
            if (!sv.is_bottom() && !sv.is_top()) {
                cur_notes_.push_back("FAU[+" + hex_str(fau_off) + "] (" + slot + ") = " + sv.format_short());
            } else {
                cur_notes_.push_back("FAU[+" + hex_str(fau_off) + "] (" + slot + ")");
            }
            return;
        }
        if (bv.kind == TrackedVal::Kind::BufferAddr) {
            cur_notes_.push_back("store " + bv.format_short());
            state_->set_sym_mem(bv.symbol_name, bv.byte_offset + offset, sv);
            return;
        }
        if (bv.kind == TrackedVal::Kind::TlsPtr || bv.kind == TrackedVal::Kind::WlsPtr) {
            TrackedVal ptr = bv.with_offset(offset);
            cur_notes_.push_back("store " + ptr.label);
            state_->set_sym_mem(bv.symbol_name, ptr.byte_offset, sv);
        }
    }

    void on_ld_attr(const AbstractDest& dst, uint32_t index, uint32_t /*table*/, const DecodedValhallInst& /*insn*/) override {
        std::string sym = env_.lookup_input_symbol ? env_.lookup_input_symbol(index) : "";
        if (!sym.empty()) {
            cur_notes_.push_back("attr[" + std::to_string(index) + "] = " + sym);
        }
        TrackedVal v;
        v.kind = TrackedVal::Kind::VertexAttrValue;
        v.desc_idx = static_cast<int32_t>(index);
        v.symbol_name = sym;
        v.label = sym.empty() ? ("attr[" + std::to_string(index) + "]") : sym;
        write_dest(dst, v);
    }

    void on_ld_var(const AbstractDest& dst, uint32_t index, uint32_t /*table*/, const DecodedValhallInst& /*insn*/) override {
        std::string sym = env_.lookup_input_symbol ? env_.lookup_input_symbol(index) : "";
        if (!sym.empty()) {
            cur_notes_.push_back("varying[" + std::to_string(index) + "] = " + sym);
        }
        TrackedVal v;
        v.kind = TrackedVal::Kind::VaryingValue;
        v.desc_idx = static_cast<int32_t>(index);
        v.symbol_name = sym;
        v.label = sym.empty() ? ("varying[" + std::to_string(index) + "]") : sym;
        write_dest(dst, v);
    }

    void on_tex(const AbstractDest& dst, uint32_t tex_idx, uint32_t smp_idx, const DecodedValhallInst& /*insn*/) override {
        std::string tsym = env_.lookup_image_symbol ? env_.lookup_image_symbol(tex_idx) : "";
        std::string ssym = env_.lookup_sampler_symbol ? env_.lookup_sampler_symbol(smp_idx) : "";
        if (!tsym.empty()) cur_notes_.push_back("tex[" + std::to_string(tex_idx) + "] = " + tsym);
        if (!ssym.empty()) cur_notes_.push_back("smp[" + std::to_string(smp_idx) + "] = " + ssym);
        TrackedVal v;
        v.kind = TrackedVal::Kind::TexSampleValue;
        v.desc_idx = static_cast<int32_t>(tex_idx);
        v.symbol_name = tsym;
        v.label = tsym.empty() ? ("tex[" + std::to_string(tex_idx) + "]") : ("sample(" + tsym + ")");
        write_dest(dst, v);
    }

    void on_blend(const DecodedStaging& /*src*/, int32_t rt_idx, const DecodedValhallInst& /*insn*/) override {
        if (rt_idx >= 0 && env_.lookup_output_symbol) {
            std::string osym = env_.lookup_output_symbol(static_cast<uint32_t>(rt_idx));
            if (!osym.empty()) {
                cur_notes_.push_back("RT" + std::to_string(rt_idx) + " -> " + osym);
            }
        }
    }

private:
    bool propagate_derived_symbol(const AbstractDest& dst, const TrackedVal& a, const TrackedVal& b) {
        if (a.kind == TrackedVal::Kind::SymbolValue) {
            write_dest(dst, TrackedVal::make_symbol_value(a.rack, a.desc_idx, a.symbol_name, true));
            return true;
        }
        if (b.kind == TrackedVal::Kind::SymbolValue) {
            write_dest(dst, TrackedVal::make_symbol_value(b.rack, b.desc_idx, b.symbol_name, true));
            return true;
        }
        return false;
    }

    const PointerAnalysisEnv& env_;
    std::map<uint64_t, std::vector<std::string>>* notes_by_pc_ = nullptr;
    std::map<uint32_t, PilotFauWrite>* pilot_writes_ = nullptr;
    uint64_t cur_pc_ = 0;
    std::vector<std::string> cur_notes_;
};

}  // namespace

PointerAnalysisResult run_pointer_analysis(
    const std::vector<DecodedValhallInst>& instructions,
    const PointerAnalysisEnv& env) {
    PointerAnalysisResult out;
    out.inst_notes.resize(instructions.size());
    if (instructions.empty()) return out;

    PointerState init = make_initial_pointer_state(env);
    std::map<uint64_t, std::vector<std::string>> notes_by_pc;
    PointerTransferVisitor visitor(env, &notes_by_pc, &out.pilot_fau_writes);

    auto fp = run_forward_dataflow<TrackedValDomain>(instructions, init, visitor);
    out.pre_states = std::move(fp.pre_states);
    out.post_states = std::move(fp.post_states);
    out.fixpoint_iterations = fp.iterations;

    for (size_t i = 0; i < instructions.size(); ++i) {
        auto it = notes_by_pc.find(instructions[i].pc);
        if (it != notes_by_pc.end()) {
            out.inst_notes[i] = std::move(it->second);
        }
    }
    return out;
}

}  // namespace valhall::dataflow

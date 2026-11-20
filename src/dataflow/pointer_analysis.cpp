#include "pointer_analysis.h"

#include <algorithm>
#include <cstdio>

namespace valhall::dataflow {

namespace {

std::string hex_off(int64_t off) {
    char buf[32];
    if (off < 0) {
        std::snprintf(buf, sizeof(buf), "-0x%llx", static_cast<unsigned long long>(-off));
    } else {
        std::snprintf(buf, sizeof(buf), "+0x%llx", static_cast<unsigned long long>(off));
    }
    return buf;
}

/// Add constant byte offset `delta` via `SymbolicMemoryLoc::with_added_offset`.
TrackedVal add_pointer_offset(const TrackedVal& base, int64_t delta, const PointerAnalysisEnv& env) {
    if (!base.is_pointer()) return TrackedVal::make_top();
    SymbolicMemoryLoc loc = base.to_memory_loc().with_added_offset(delta);
    TrackedVal out = base.with_memory_loc(loc);
    if (out.kind == TrackedVal::Kind::SrtRackPtr && loc.offset_known && loc.byte_offset >= 0 &&
        (loc.byte_offset % 32) == 0 && out.rack >= 0) {
        int32_t desc = static_cast<int32_t>(loc.byte_offset / 32);
        std::string sym = env.lookup_srt_symbol
                              ? env.lookup_srt_symbol(static_cast<uint8_t>(out.rack), static_cast<uint32_t>(desc))
                              : "";
        return TrackedVal::make_descriptor(out.rack, desc, std::move(sym), 0, true);
    }
    return out;
}

/// Widen a pointer's offset to unknown via `SymbolicMemoryLoc::with_unknown_offset`.
TrackedVal add_unknown_offset(const TrackedVal& base) {
    if (!base.is_pointer()) return TrackedVal::make_top();
    return base.with_memory_loc(base.to_memory_loc().with_unknown_offset());
}

/// Mask a pointer's offset via `SymbolicMemoryLoc::with_masked_offset`.
TrackedVal mask_pointer_offset(const TrackedVal& base, int64_t mask) {
    if (!base.is_pointer()) return TrackedVal::make_top();
    return base.with_memory_loc(base.to_memory_loc().with_masked_offset(mask));
}

/// Pointwise dereference of a single `TrackedVal` pointer in `PsbLoad(addr, elem_bytes)`.
TrackedVal deref_pointer_single(const TrackedVal& addr, uint32_t elem_bytes, const PointerAnalysisEnv& env) {
    switch (addr.kind) {
        case TrackedVal::Kind::SrtBase: {
            if (!addr.offset_known || addr.byte_offset < 0) {
                return TrackedVal::make_srt_rack(-1, 0, false);
            }
            int32_t rack = static_cast<int32_t>(addr.byte_offset / 16);
            return TrackedVal::make_srt_rack(rack, 0, true);
        }
        case TrackedVal::Kind::SrtRackPtr: {
            if (!addr.offset_known || addr.byte_offset < 0 || addr.rack < 0) {
                return TrackedVal::make_top();
            }
            int32_t desc = static_cast<int32_t>(addr.byte_offset / 32);
            int64_t field = addr.byte_offset % 32;
            std::string sym = env.lookup_srt_symbol
                                  ? env.lookup_srt_symbol(static_cast<uint8_t>(addr.rack), static_cast<uint32_t>(desc))
                                  : "";
            if (field == 4 && elem_bytes == 4) {
                return TrackedVal::make_buffer_size(addr.rack, desc, std::move(sym));
            }
            if (field == 8 || elem_bytes == 8) {
                return TrackedVal::make_buffer_addr(addr.rack, desc, std::move(sym), 0, false, true);
            }
            return TrackedVal::make_descriptor(addr.rack, desc, std::move(sym), field, true);
        }
        case TrackedVal::Kind::DescriptorAddr: {
            std::string sym = addr.symbol_name;
            if (sym.empty() && env.lookup_srt_symbol && addr.rack >= 0 && addr.desc_idx >= 0) {
                sym = env.lookup_srt_symbol(static_cast<uint8_t>(addr.rack), static_cast<uint32_t>(addr.desc_idx));
            }
            if (addr.offset_known && addr.byte_offset == 4 && elem_bytes == 4) {
                return TrackedVal::make_buffer_size(addr.rack, addr.desc_idx, std::move(sym));
            }
            return TrackedVal::make_buffer_addr(addr.rack, addr.desc_idx, std::move(sym), 0, false, true);
        }
        case TrackedVal::Kind::BufferAddr: {
            std::string mem_sym = addr.symbol_name;
            if (addr.offset_known && addr.byte_offset >= 0 && env.lookup_symbol_member &&
                addr.rack >= 0 && addr.desc_idx >= 0) {
                std::string m = env.lookup_symbol_member(
                    static_cast<uint8_t>(addr.rack),
                    static_cast<uint32_t>(addr.desc_idx),
                    static_cast<uint32_t>(addr.byte_offset));
                if (!m.empty()) mem_sym = std::move(m);
            }
            return TrackedVal::make_symbol_value(addr.rack, addr.desc_idx, std::move(mem_sym), false);
        }
        case TrackedVal::Kind::FauDstBase: {
            if (addr.offset_known && addr.byte_offset >= 0) {
                return TrackedVal::make_fau_uniform(static_cast<uint32_t>(addr.byte_offset));
            }
            return TrackedVal::make_top();
        }
        case TrackedVal::Kind::DataSectionPtr: {
            std::string lbl = ".data";
            if (addr.offset_known) lbl += hex_off(addr.byte_offset);
            return TrackedVal::make_symbol_value(-1, -1, std::move(lbl), false);
        }
        default:
            return TrackedVal::make_top();
    }
}

void append_unique_note(std::vector<std::string>& notes, const std::string& s) {
    if (s.empty()) return;
    if (std::find(notes.begin(), notes.end(), s) == notes.end()) {
        notes.push_back(s);
    }
}

/// SSA Pointer Transfer Visitor operating over `TrackedValSetDomain`
/// (`FiniteSetDomain<TrackedValDomain, 4>`).
///
/// Uses `TrackedValSetDomain::lift_unary` / `lift_binary` so pointwise pointer
/// rules remain concise and readable while automatically handling bounded sets
/// of up to 4 distinct concrete values.
class SsaPointerTransferVisitor : public SsaDomainTransferVisitor<TrackedValSetDomain> {
public:
    using SetDom = TrackedValSetDomain;

    SsaPointerTransferVisitor(
        const PointerAnalysisEnv& env,
        std::map<uint32_t, PilotFauWrite>* pilot_writes,
        std::vector<std::vector<std::string>>* ssa_notes)
        : env_(env), pilot_writes_(pilot_writes), ssa_notes_(ssa_notes) {}

    void on_const(compiler::SsaId dst, const compiler::SsaNode& node) override {
        using tables::SsaOp;
        switch (node.op) {
            case SsaOp::ConstU64:
                assign(dst, SetDom::lift_value(TrackedVal::make_const(static_cast<int64_t>(node.imm_u64))));
                break;
            case SsaOp::ConstS32:
                assign(dst, SetDom::lift_value(TrackedVal::make_const(static_cast<int64_t>(static_cast<int32_t>(node.imm_u32)))));
                break;
            case SsaOp::ConstU32:
            case SsaOp::ConstBool:
            case SsaOp::ConstF32:
                // Note: ConstF32 stores raw IEEE-754 bits in imm_u32 (relevant for Bitcast/Mov)
                assign(dst, SetDom::lift_value(TrackedVal::make_const(static_cast<int64_t>(node.imm_u32))));
                break;
            default:
                assign(dst, SetDom::top());
                break;
        }
    }

    void on_load_sideband(compiler::SsaId dst, uint32_t member_idx, bool /*is_64bit*/,
                          const compiler::SsaNode&) override {
        if (env_.is_fau_dst_va(member_idx)) {
            assign(dst, SetDom::lift_value(TrackedVal::make_fau_dst(0, true)));
            return;
        }
        if (env_.is_srt_base_va(member_idx)) {
            assign(dst, SetDom::lift_value(TrackedVal::make_srt_base(0, true)));
            return;
        }
        assign(dst, SetDom::top());
    }

    void on_load_builtin(compiler::SsaId dst, uint32_t /*builtin_id*/,
                         const compiler::SsaNode& node) override {
        std::string lbl = node.aux_str.empty() ? "gl_BuiltIn" : node.aux_str;
        assign(dst, SetDom::lift_value(TrackedVal::make_abi(std::move(lbl))));
    }

    void on_load_fau(compiler::SsaId dst, uint32_t fau_word_idx,
                     const compiler::SsaNode&) override {
        uint32_t byte_off = fau_word_idx * 4u;
        if (state_) {
            TrackedValSet fv = state_->fau(byte_off);
            if (!fv.is_bottom()) {
                assign(dst, fv);
                return;
            }
            if ((byte_off & 4u) != 0) {
                TrackedValSet lo = state_->fau(byte_off & ~7u);
                if (!lo.is_bottom()) {
                    assign(dst, SetDom::lift_unary(lo, [](const TrackedVal& v) {
                        if (v.is_pointer()) {
                            TrackedVal hi = v;
                            hi.is_hi_half = true;
                            return hi;
                        }
                        return TrackedVal::make_top();
                    }));
                    return;
                }
            }
        }
        assign(dst, SetDom::lift_value(TrackedVal::make_fau_uniform(byte_off)));
    }

    void on_split64(compiler::SsaId dst, compiler::SsaId src64,
                    const compiler::SsaNode&) override {
        assign(dst, eval(src64));
    }

    void on_tuple_extract(compiler::SsaId dst, compiler::SsaId tuple_val, uint32_t elem_idx,
                          const compiler::SsaNode& node) override {
        if (node.type == compiler::ValType::Token) return;
        assign(dst, SetDom::lift_unary(eval(tuple_val), [elem_idx](const TrackedVal& tv) {
            if (elem_idx == 0) {
                TrackedVal lo = tv;
                lo.is_hi_half = false;
                return lo;
            }
            if (elem_idx == 1) {
                if (tv.kind == TrackedVal::Kind::ConstInt) {
                    return TrackedVal::make_const(static_cast<int64_t>(static_cast<uint64_t>(tv.const_val) >> 32));
                }
                TrackedVal hi = tv;
                hi.is_hi_half = true;
                return hi;
            }
            return TrackedVal::make_top();
        }));
    }

    void on_pack64(compiler::SsaId dst, compiler::SsaId lo, compiler::SsaId hi,
                   const compiler::SsaNode&) override {
        TrackedValSet slo = eval(lo);
        TrackedValSet shi = eval(hi);
        // Fast path: if lo is a pointer/FAU/symbol set and hi is Top/unknown, still recover lo's pointer!
        if (!slo.is_bottom() && !slo.is_top() && shi.is_top()) {
            assign(dst, SetDom::lift_unary(slo, [](const TrackedVal& vlo) {
                if (vlo.is_pointer() || vlo.kind == TrackedVal::Kind::FauUniformRef ||
                    vlo.kind == TrackedVal::Kind::SymbolValue) {
                    TrackedVal out = vlo;
                    out.is_hi_half = false;
                    return out;
                }
                return TrackedVal::make_top();
            }));
            return;
        }
        assign(dst, SetDom::lift_binary(slo, shi, [](const TrackedVal& vlo, const TrackedVal& vhi) {
            if (vlo.kind == TrackedVal::Kind::ConstInt && vhi.kind == TrackedVal::Kind::ConstInt) {
                uint64_t packed = (static_cast<uint64_t>(static_cast<uint32_t>(vhi.const_val)) << 32) |
                                  static_cast<uint64_t>(static_cast<uint32_t>(vlo.const_val));
                return TrackedVal::make_const(static_cast<int64_t>(packed));
            }
            if (vlo.is_pointer() || vlo.kind == TrackedVal::Kind::FauUniformRef ||
                vlo.kind == TrackedVal::Kind::SymbolValue) {
                TrackedVal out = vlo;
                out.is_hi_half = false;
                return out;
            }
            if (vhi.is_pointer()) {
                TrackedVal out = vhi;
                out.is_hi_half = false;
                return out;
            }
            return TrackedVal::make_top();
        }));
    }

    void on_vec(compiler::SsaId dst, const compiler::SsaNode& node) override {
        if (node.op == tables::SsaOp::VecExtract && !node.args.empty()) {
            uint32_t comp = node.imm_u32;
            assign(dst, SetDom::lift_unary(eval(node.args[0]), [comp](const TrackedVal& base) {
                if (base.kind == TrackedVal::Kind::AbiPreload) {
                    static const char kComp[] = "xyzw";
                    std::string lbl = base.label;
                    if (comp < 4) {
                        lbl.push_back('.');
                        lbl.push_back(kComp[comp]);
                    }
                    return TrackedVal::make_abi(std::move(lbl));
                }
                return base;
            }));
            return;
        }
        assign(dst, SetDom::top());
    }

    void on_convert(compiler::SsaId dst, compiler::SsaId src,
                    const compiler::SsaNode& node) override {
        if (node.op == tables::SsaOp::UConvert32To64 || node.op == tables::SsaOp::SConvert32To64 ||
            node.op == tables::SsaOp::UConvert64To32) {
            auto op = node.op;
            assign(dst, SetDom::lift_unary(eval(src), [op](const TrackedVal& v) {
                if (v.kind == TrackedVal::Kind::ConstInt) {
                    if (op == tables::SsaOp::SConvert32To64) {
                        return TrackedVal::make_const(static_cast<int64_t>(static_cast<int32_t>(v.const_val)));
                    }
                    return TrackedVal::make_const(static_cast<int64_t>(static_cast<uint32_t>(v.const_val)));
                }
                return v;
            }));
            return;
        }
        assign(dst, SetDom::top());
    }

    void on_iadd(compiler::SsaId dst, compiler::SsaId lhs, compiler::SsaId rhs,
                 bool is_64bit, const compiler::SsaNode& node) override {
        TrackedValSet sa = eval(lhs);
        TrackedValSet sb = eval(rhs);
        // Handle pointer + unknown (Top) by widening the pointer's offset rather than dropping to Top
        if (!sa.is_bottom() && !sa.is_top() && sb.is_top()) {
            TrackedValSet out = SetDom::lift_unary(sa, [](const TrackedVal& a) {
                if (a.is_pointer()) return add_unknown_offset(a);
                if (a.kind == TrackedVal::Kind::SymbolValue) { TrackedVal s = a; s.derived = true; return s; }
                return TrackedVal::make_top();
            });
            assign(dst, out);
            return;
        }
        if (!sb.is_bottom() && !sb.is_top() && sa.is_top()) {
            TrackedValSet out = SetDom::lift_unary(sb, [](const TrackedVal& b) {
                if (b.is_pointer()) return add_unknown_offset(b);
                if (b.kind == TrackedVal::Kind::SymbolValue) { TrackedVal s = b; s.derived = true; return s; }
                return TrackedVal::make_top();
            });
            assign(dst, out);
            return;
        }

        if (!sa.is_bottom() && !sb.is_bottom() && !sa.is_top() && !sb.is_top()) {
            TrackedValSet out = SetDom::lift_binary(sa, sb, [&](const TrackedVal& a, const TrackedVal& b) {
                if (a.is_pointer() && b.kind == TrackedVal::Kind::ConstInt) {
                    return add_pointer_offset(a, b.const_val, env_);
                }
                if (b.is_pointer() && a.kind == TrackedVal::Kind::ConstInt) {
                    return add_pointer_offset(b, a.const_val, env_);
                }
                if (a.is_pointer()) return add_unknown_offset(a);
                if (b.is_pointer()) return add_unknown_offset(b);
                if (a.kind == TrackedVal::Kind::ConstInt && b.kind == TrackedVal::Kind::ConstInt) {
                    return TrackedVal::make_const(a.const_val + b.const_val);
                }
                if (a.kind == TrackedVal::Kind::SymbolValue || b.kind == TrackedVal::Kind::SymbolValue) {
                    TrackedVal s = (a.kind == TrackedVal::Kind::SymbolValue) ? a : b;
                    s.derived = true;
                    return s;
                }
                return TrackedVal::make_top();
            });
            assign(dst, out);
            TrackedVal single;
            if (out.try_as_singleton(&single) && single.is_pointer()) {
                record_note(dst, "ptr " + single.format_short());
            }
            return;
        }
        SsaDomainTransferVisitor<TrackedValSetDomain>::on_iadd(dst, lhs, rhs, is_64bit, node);
    }

    void on_isub(compiler::SsaId dst, compiler::SsaId lhs, compiler::SsaId rhs,
                 bool is_64bit, const compiler::SsaNode& node) override {
        if (lhs != compiler::kInvalidSsa && lhs == rhs) {
            assign(dst, SetDom::make_const_u64(0));
            return;
        }
        TrackedValSet sa = eval(lhs);
        TrackedValSet sb = eval(rhs);
        if (!sa.is_bottom() && !sb.is_bottom() && !sa.is_top() && !sb.is_top()) {
            assign(dst, SetDom::lift_binary(sa, sb, [&](const TrackedVal& a, const TrackedVal& b) {
                if (a.is_pointer() && b.kind == TrackedVal::Kind::ConstInt) {
                    return add_pointer_offset(a, -b.const_val, env_);
                }
                if (a.kind == TrackedVal::Kind::ConstInt && b.kind == TrackedVal::Kind::ConstInt) {
                    return TrackedVal::make_const(a.const_val - b.const_val);
                }
                if (a.kind == TrackedVal::Kind::SymbolValue) {
                    TrackedVal s = a; s.derived = true; return s;
                }
                return TrackedVal::make_top();
            }));
            return;
        }
        SsaDomainTransferVisitor<TrackedValSetDomain>::on_isub(dst, lhs, rhs, is_64bit, node);
    }

    void on_imul(compiler::SsaId dst, compiler::SsaId lhs, compiler::SsaId rhs,
                 bool is_64bit, const compiler::SsaNode& node) override {
        TrackedValSet sa = eval(lhs);
        TrackedValSet sb = eval(rhs);
        if (!sa.is_bottom() && !sb.is_bottom() && !sa.is_top() && !sb.is_top()) {
            assign(dst, SetDom::lift_binary(sa, sb, [](const TrackedVal& a, const TrackedVal& b) {
                if (a.kind == TrackedVal::Kind::ConstInt && b.kind == TrackedVal::Kind::ConstInt) {
                    return TrackedVal::make_const(a.const_val * b.const_val);
                }
                if ((a.kind == TrackedVal::Kind::ConstInt && a.const_val == 0) ||
                    (b.kind == TrackedVal::Kind::ConstInt && b.const_val == 0)) {
                    return TrackedVal::make_const(0);
                }
                if (b.kind == TrackedVal::Kind::ConstInt && b.const_val == 1) return a;
                if (a.kind == TrackedVal::Kind::ConstInt && a.const_val == 1) return b;
                if (a.kind == TrackedVal::Kind::SymbolValue || b.kind == TrackedVal::Kind::SymbolValue) {
                    TrackedVal s = (a.kind == TrackedVal::Kind::SymbolValue) ? a : b;
                    s.derived = true;
                    return s;
                }
                return TrackedVal::make_top();
            }));
            return;
        }
        SsaDomainTransferVisitor<TrackedValSetDomain>::on_imul(dst, lhs, rhs, is_64bit, node);
    }

    void on_and(compiler::SsaId dst, compiler::SsaId lhs, compiler::SsaId rhs,
                bool is_64bit, const compiler::SsaNode& node) override {
        if (lhs != compiler::kInvalidSsa && lhs == rhs) {
            assign(dst, eval(lhs));
            return;
        }
        TrackedValSet sa = eval(lhs);
        TrackedValSet sb = eval(rhs);
        if (!sa.is_bottom() && !sb.is_bottom() && !sa.is_top() && !sb.is_top()) {
            assign(dst, SetDom::lift_binary(sa, sb, [](const TrackedVal& a, const TrackedVal& b) {
                if (a.kind == TrackedVal::Kind::ConstInt && b.kind == TrackedVal::Kind::ConstInt) {
                    return TrackedVal::make_const(a.const_val & b.const_val);
                }
                if ((a.kind == TrackedVal::Kind::ConstInt && a.const_val == 0) ||
                    (b.kind == TrackedVal::Kind::ConstInt && b.const_val == 0)) {
                    return TrackedVal::make_const(0);
                }
                if (a.is_pointer() && b.kind == TrackedVal::Kind::ConstInt && b.const_val != 0) {
                    return mask_pointer_offset(a, b.const_val);
                }
                if (b.is_pointer() && a.kind == TrackedVal::Kind::ConstInt && a.const_val != 0) {
                    return mask_pointer_offset(b, a.const_val);
                }
                return TrackedVal::make_top();
            }));
            return;
        }
        SsaDomainTransferVisitor<TrackedValSetDomain>::on_and(dst, lhs, rhs, is_64bit, node);
    }

    void on_psb_load(compiler::SsaId dst, compiler::SsaId addr, uint32_t elem_bytes,
                     const compiler::SsaNode&) override {
        TrackedValSet addr_set = eval(addr);
        // Soundness rule: pointer dereference over a non-singleton set of addresses widens to TOP.
        TrackedVal single_addr;
        if (!addr_set.try_as_singleton(&single_addr)) {
            assign(dst, addr_set.is_bottom() ? SetDom::bottom() : SetDom::top());
            return;
        }
        TrackedVal loaded = deref_pointer_single(single_addr, elem_bytes, env_);
        assign(dst, SetDom::lift_value(loaded));
        if (!single_addr.is_bottom() && !single_addr.is_top()) {
            record_note(dst, "load " + single_addr.format_short() + " -> " + loaded.format_short());
        }
    }

    void on_psb_store(compiler::SsaId addr, compiler::SsaId val, uint32_t elem_bytes,
                      const compiler::SsaNode& node) override {
        TrackedValSet addr_set = eval(addr);
        TrackedValSet val_set = eval(val);
        TrackedVal single_addr;
        if (!addr_set.try_as_singleton(&single_addr)) {
            return;
        }
        if (single_addr.kind == TrackedVal::Kind::FauDstBase &&
            single_addr.offset_known && single_addr.byte_offset >= 0) {
            uint32_t fau_off = static_cast<uint32_t>(single_addr.byte_offset);
            if (state_) {
                state_->join_fau(fau_off, val_set);
            }
            TrackedVal collapsed = collapse_tracked_val_set(val_set);
            if (pilot_writes_ && !collapsed.is_bottom() && !collapsed.is_top()) {
                auto it = pilot_writes_->find(fau_off);
                if (it == pilot_writes_->end()) {
                    (*pilot_writes_)[fau_off] = PilotFauWrite{fau_off, elem_bytes, collapsed};
                } else {
                    it->second.value = TrackedValDomain::join(it->second.value, collapsed);
                    it->second.byte_size = std::max(it->second.byte_size, elem_bytes);
                }
            }
            record_note(node.id, "pilot store FAU+0x" + hex_off(fau_off).substr(3) + " <- " +
                                     format_tracked_val_set(val_set));
        } else if (single_addr.is_pointer()) {
            if (state_) {
                state_->join_sym(single_addr.to_memory_loc(), val_set);
            }
            record_note(node.id, "store " + single_addr.format_short());
        }
    }

    void on_load_vert_attr(compiler::SsaId dst, const compiler::SsaNode& node) override {
        uint32_t loc = node.imm_u32;
        std::string sym = env_.lookup_input_symbol ? env_.lookup_input_symbol(loc) : "";
        TrackedVal v = TrackedVal::make_vertex_attr(loc, sym);
        assign(dst, SetDom::lift_value(v));
        record_note(dst, v.format_short());
    }

    void on_load_frag_varying(compiler::SsaId dst, const compiler::SsaNode& node) override {
        uint32_t loc = node.imm_u32;
        std::string sym = env_.lookup_input_symbol ? env_.lookup_input_symbol(loc) : "";
        TrackedVal v = TrackedVal::make_varying(loc, sym);
        assign(dst, SetDom::lift_value(v));
        record_note(dst, v.format_short());
    }

    void on_tex_sample(compiler::SsaId dst, const compiler::SsaNode& node) override {
        uint32_t img = node.imm_u32;
        uint32_t smp = node.imm_u32_2;
        std::string sym = env_.lookup_image_symbol ? env_.lookup_image_symbol(img) : "";
        TrackedVal v = TrackedVal::make_tex_sample(img, smp, sym);
        assign(dst, SetDom::lift_value(v));
        record_note(dst, v.format_short());
    }

    void on_store_output(const compiler::SsaNode& node) override {
        if (node.op == tables::SsaOp::StoreFragColor || node.op == tables::SsaOp::StoreVertVarying) {
            uint32_t loc = node.imm_u32;
            std::string sym = env_.lookup_output_symbol ? env_.lookup_output_symbol(loc) : "";
            TrackedVal v = TrackedVal::make_blend_desc(loc, sym);
            record_note(node.id, "out " + v.format_short());
        }
    }

private:
    void record_note(compiler::SsaId id, const std::string& note) {
        if (ssa_notes_ && id != compiler::kInvalidSsa && id < ssa_notes_->size()) {
            append_unique_note((*ssa_notes_)[id], note);
        }
    }

    const PointerAnalysisEnv& env_;
    std::map<uint32_t, PilotFauWrite>* pilot_writes_ = nullptr;
    std::vector<std::vector<std::string>>* ssa_notes_ = nullptr;
};

}  // namespace

PointerAnalysisResult run_ssa_pointer_analysis(
    const compiler::ValhallSsaModule& mod,
    const PointerAnalysisEnv& env,
    const SsaLivenessResult* liveness,
    uint32_t max_iterations) {
    PointerAnalysisResult out;
    out.ssa_notes.resize(mod.nodes.size());

    AbstractState<TrackedValSetDomain> entry(mod.nodes.size());
    for (const auto& kv : env.initial_fau) {
        entry.set_fau(kv.first, TrackedValSetDomain::lift_value(kv.second));
    }

    SsaPointerTransferVisitor visitor(env, &out.pilot_fau_writes, &out.ssa_notes);
    SsaDataflowResult<TrackedValSetDomain> df =
        run_ssa_forward_dataflow<TrackedValSetDomain>(mod, entry, visitor, liveness, max_iterations);

    out.converged = df.converged;
    out.iterations = df.iterations;
    out.final_state = std::move(df.final_state);
    if (liveness) {
        out.liveness = *liveness;
    }
    return out;
}

}  // namespace valhall::dataflow

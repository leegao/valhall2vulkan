#include "pointer_analysis.h"

#include <algorithm>
#include <cstdio>
#include <set>
#include <sstream>

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

bool kind_has_offset(TrackedVal::Kind k) {
    switch (k) {
        case TrackedVal::Kind::FauDstBase:
        case TrackedVal::Kind::SrtBase:
        case TrackedVal::Kind::SrtRackPtr:
        case TrackedVal::Kind::DescriptorAddr:
        case TrackedVal::Kind::BufferAddr:
        case TrackedVal::Kind::DataSectionPtr:
        case TrackedVal::Kind::TlsPtr:
        case TrackedVal::Kind::WlsPtr:
            return true;
        default:
            return false;
    }
}

bool same_base_identity(const TrackedVal& a, const TrackedVal& b) {
    if (a.kind != b.kind) return false;
    if (a.is_hi_half != b.is_hi_half) return false;
    switch (a.kind) {
        case TrackedVal::Kind::FauDstBase:
        case TrackedVal::Kind::SrtBase:
        case TrackedVal::Kind::DataSectionPtr:
        case TrackedVal::Kind::TlsPtr:
        case TrackedVal::Kind::WlsPtr:
            return true;
        case TrackedVal::Kind::SrtRackPtr:
            return a.rack == b.rack;
        case TrackedVal::Kind::DescriptorAddr:
        case TrackedVal::Kind::BufferAddr:
        case TrackedVal::Kind::BufferSize:
            return a.rack == b.rack && a.desc_idx == b.desc_idx && a.symbol_name == b.symbol_name;
        case TrackedVal::Kind::SymbolValue:
            return a.rack == b.rack && a.desc_idx == b.desc_idx && a.symbol_name == b.symbol_name;
        case TrackedVal::Kind::FauUniformRef:
            return a.fau_offset == b.fau_offset && a.symbol_name == b.symbol_name;
        case TrackedVal::Kind::AbiPreload:
            return a.label == b.label;
        case TrackedVal::Kind::VertexAttrValue:
        case TrackedVal::Kind::VaryingValue:
        case TrackedVal::Kind::BlendDescRef:
            return a.location == b.location && a.symbol_name == b.symbol_name;
        case TrackedVal::Kind::TexSampleValue:
            return a.location == b.location && a.sampler_idx == b.sampler_idx && a.symbol_name == b.symbol_name;
        default:
            return false;
    }
}

/// Add constant byte offset `delta` to any pointer-like `TrackedVal`.
TrackedVal add_pointer_offset(const TrackedVal& base, int64_t delta, const PointerAnalysisEnv& env) {
    if (!base.is_pointer()) return TrackedVal::make_top();
    TrackedVal out = base;
    if (out.offset_known) {
        out.byte_offset += delta;
    }
    if (out.kind == TrackedVal::Kind::SrtRackPtr && out.offset_known && out.byte_offset >= 0 &&
        (out.byte_offset % 32) == 0 && out.rack >= 0) {
        int32_t desc = static_cast<int32_t>(out.byte_offset / 32);
        std::string sym = env.lookup_srt_symbol
                              ? env.lookup_srt_symbol(static_cast<uint8_t>(out.rack), static_cast<uint32_t>(desc))
                              : "";
        return TrackedVal::make_descriptor(out.rack, desc, std::move(sym), 0, true);
    }
    return out;
}

/// Widen a pointer's offset to unknown when an unknown dynamic index is added.
TrackedVal add_unknown_offset(const TrackedVal& base) {
    if (!base.is_pointer()) return TrackedVal::make_top();
    TrackedVal out = base;
    out.offset_known = false;
    out.byte_offset = 0;
    return out;
}

/// Dereference a pointer `TrackedVal` in `PsbLoad(addr, elem_bytes)`.
TrackedVal deref_pointer(const TrackedVal& addr, uint32_t elem_bytes, const PointerAnalysisEnv& env) {
    switch (addr.kind) {
        case TrackedVal::Kind::SrtBase: {
            if (!addr.offset_known || addr.byte_offset < 0) {
                return TrackedVal::make_srt_rack(-1, 0, false);
            }
            // Mali Valhall SRT entries are 16 bytes each (rack_index * 0x10)
            int32_t rack = static_cast<int32_t>(addr.byte_offset / 16);
            return TrackedVal::make_srt_rack(rack, 0, true);
        }
        case TrackedVal::Kind::SrtRackPtr: {
            if (!addr.offset_known || addr.byte_offset < 0 || addr.rack < 0) {
                return TrackedVal::make_top();
            }
            // Each Valhall buffer/uniform descriptor in an SRT rack is 32 bytes (0x20):
            //   +0x00: 32-bit flags/type
            //   +0x04: 32-bit buffer size (bytes)
            //   +0x08: 64-bit buffer GPU VA
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

}  // namespace

// ============================================================================
// TrackedVal constructors & formatting
// ============================================================================

TrackedVal TrackedVal::make_bottom() { return TrackedVal{}; }
TrackedVal TrackedVal::make_top() { TrackedVal v; v.kind = Kind::Top; return v; }
TrackedVal TrackedVal::make_const(int64_t c) { TrackedVal v; v.kind = Kind::ConstInt; v.const_val = c; return v; }
TrackedVal TrackedVal::make_abi(std::string lbl) { TrackedVal v; v.kind = Kind::AbiPreload; v.label = std::move(lbl); return v; }
TrackedVal TrackedVal::make_fau_dst(int64_t off, bool ok) {
    TrackedVal v; v.kind = Kind::FauDstBase; v.byte_offset = off; v.offset_known = ok;
    v.label = "pilot_fau_dst_va (FAU_0)"; return v;
}
TrackedVal TrackedVal::make_srt_base(int64_t off, bool ok) {
    TrackedVal v; v.kind = Kind::SrtBase; v.byte_offset = off; v.offset_known = ok;
    v.label = "srt_base_va (SRT_0)"; return v;
}
TrackedVal TrackedVal::make_srt_rack(int32_t r, int64_t off, bool ok) {
    TrackedVal v; v.kind = Kind::SrtRackPtr; v.rack = r; v.byte_offset = off; v.offset_known = ok; return v;
}
TrackedVal TrackedVal::make_descriptor(int32_t r, int32_t d, std::string sym, int64_t foff, bool ok) {
    TrackedVal v; v.kind = Kind::DescriptorAddr; v.rack = r; v.desc_idx = d;
    v.symbol_name = std::move(sym); v.byte_offset = foff; v.offset_known = ok; return v;
}
TrackedVal TrackedVal::make_buffer_addr(int32_t r, int32_t d, std::string sym, int64_t off, bool drv, bool ok) {
    TrackedVal v; v.kind = Kind::BufferAddr; v.rack = r; v.desc_idx = d;
    v.symbol_name = std::move(sym); v.byte_offset = off; v.derived = drv; v.offset_known = ok; return v;
}
TrackedVal TrackedVal::make_buffer_size(int32_t r, int32_t d, std::string sym) {
    TrackedVal v; v.kind = Kind::BufferSize; v.rack = r; v.desc_idx = d; v.symbol_name = std::move(sym); return v;
}
TrackedVal TrackedVal::make_fau_uniform(uint32_t foff, std::string sym) {
    TrackedVal v; v.kind = Kind::FauUniformRef; v.fau_offset = foff; v.symbol_name = std::move(sym); return v;
}
TrackedVal TrackedVal::make_symbol_value(int32_t r, int32_t d, std::string sym, bool drv) {
    TrackedVal v; v.kind = Kind::SymbolValue; v.rack = r; v.desc_idx = d;
    v.symbol_name = std::move(sym); v.derived = drv; return v;
}
TrackedVal TrackedVal::make_data_section(int64_t off, bool ok) {
    TrackedVal v; v.kind = Kind::DataSectionPtr; v.byte_offset = off; v.offset_known = ok; return v;
}
TrackedVal TrackedVal::make_tls_ptr(int64_t off, bool ok) {
    TrackedVal v; v.kind = Kind::TlsPtr; v.byte_offset = off; v.offset_known = ok; return v;
}
TrackedVal TrackedVal::make_wls_ptr(int64_t off, bool ok) {
    TrackedVal v; v.kind = Kind::WlsPtr; v.byte_offset = off; v.offset_known = ok; return v;
}
TrackedVal TrackedVal::make_vertex_attr(uint32_t loc, std::string sym) {
    TrackedVal v; v.kind = Kind::VertexAttrValue; v.location = loc; v.symbol_name = std::move(sym); return v;
}
TrackedVal TrackedVal::make_varying(uint32_t loc, std::string sym) {
    TrackedVal v; v.kind = Kind::VaryingValue; v.location = loc; v.symbol_name = std::move(sym); return v;
}
TrackedVal TrackedVal::make_tex_sample(uint32_t img, uint32_t smp, std::string sym) {
    TrackedVal v; v.kind = Kind::TexSampleValue; v.location = img; v.sampler_idx = smp;
    v.symbol_name = std::move(sym); return v;
}
TrackedVal TrackedVal::make_blend_desc(uint32_t loc, std::string sym) {
    TrackedVal v; v.kind = Kind::BlendDescRef; v.location = loc; v.symbol_name = std::move(sym); return v;
}

bool TrackedVal::is_pointer() const {
    return kind_has_offset(kind);
}

std::string TrackedVal::format_short() const {
    std::ostringstream oss;
    switch (kind) {
        case Kind::Bottom: return "<bottom>";
        case Kind::Top: return "<top>";
        case Kind::ConstInt:
            oss << "0x" << std::hex << static_cast<uint64_t>(const_val);
            return oss.str();
        case Kind::AbiPreload:
            return label.empty() ? "ABI" : label;
        case Kind::FauDstBase:
            oss << "FAU_0";
            if (!offset_known) oss << "+?";
            else if (byte_offset != 0) oss << hex_off(byte_offset);
            return oss.str();
        case Kind::SrtBase:
            oss << "SRT_0";
            if (!offset_known) oss << "+?";
            else if (byte_offset != 0) oss << hex_off(byte_offset);
            return oss.str();
        case Kind::SrtRackPtr:
            oss << "SRT[" << rack << "]";
            if (!offset_known) oss << "+?";
            else if (byte_offset != 0) oss << hex_off(byte_offset);
            return oss.str();
        case Kind::DescriptorAddr:
            oss << "SRT[" << rack << "]->Desc[" << desc_idx << "]";
            if (!symbol_name.empty()) oss << " (" << symbol_name << ")";
            if (!offset_known) oss << "+?";
            else if (byte_offset != 0) oss << hex_off(byte_offset);
            return oss.str();
        case Kind::BufferAddr: {
            if (!symbol_name.empty()) {
                oss << "&" << symbol_name;
            } else {
                oss << "&Buf[" << desc_idx << "]";
            }
            if (!offset_known) oss << "+?";
            else if (byte_offset != 0) oss << hex_off(byte_offset);
            if (rack >= 0 && desc_idx >= 0) {
                oss << " (SRT[" << rack << "]->Buf[" << desc_idx << "].addr)";
            }
            return oss.str();
        }
        case Kind::BufferSize:
            if (!symbol_name.empty()) oss << "sizeof(" << symbol_name << ")";
            else oss << "sizeof(SRT[" << rack << "]->Buf[" << desc_idx << "])";
            return oss.str();
        case Kind::FauUniformRef:
            if (!symbol_name.empty()) return symbol_name;
            oss << "FAU+0x" << std::hex << fau_offset;
            return oss.str();
        case Kind::SymbolValue:
            if (derived) oss << "derived(";
            oss << (symbol_name.empty() ? "sym" : symbol_name);
            if (derived) oss << ")";
            return oss.str();
        case Kind::DataSectionPtr:
            oss << ".data";
            if (!offset_known) oss << "+?";
            else if (byte_offset != 0) oss << hex_off(byte_offset);
            return oss.str();
        case Kind::TlsPtr:
            oss << "TLS";
            if (!offset_known) oss << "+?";
            else if (byte_offset != 0) oss << hex_off(byte_offset);
            return oss.str();
        case Kind::WlsPtr:
            oss << "WLS";
            if (!offset_known) oss << "+?";
            else if (byte_offset != 0) oss << hex_off(byte_offset);
            return oss.str();
        case Kind::VertexAttrValue:
            oss << "attr[" << location << "]";
            if (!symbol_name.empty()) oss << " (" << symbol_name << ")";
            return oss.str();
        case Kind::VaryingValue:
            oss << "varying[" << location << "]";
            if (!symbol_name.empty()) oss << " (" << symbol_name << ")";
            return oss.str();
        case Kind::TexSampleValue:
            oss << "tex(img=" << location << ",smp=" << sampler_idx << ")";
            if (!symbol_name.empty()) oss << " (" << symbol_name << ")";
            return oss.str();
        case Kind::BlendDescRef:
            oss << "color_out[" << location << "]";
            if (!symbol_name.empty()) oss << " (" << symbol_name << ")";
            return oss.str();
    }
    return "<unknown>";
}

// ============================================================================
// TrackedValDomain complete lattice operations (join, meet, leq, equals)
// ============================================================================

bool TrackedValDomain::equals(const Value& a, const Value& b) {
    if (a.kind != b.kind) return false;
    if (a.kind == TrackedVal::Kind::Bottom || a.kind == TrackedVal::Kind::Top) return true;
    if (a.kind == TrackedVal::Kind::ConstInt) return a.const_val == b.const_val;
    return a.byte_offset == b.byte_offset &&
           a.offset_known == b.offset_known &&
           a.rack == b.rack &&
           a.desc_idx == b.desc_idx &&
           a.fau_offset == b.fau_offset &&
           a.location == b.location &&
           a.sampler_idx == b.sampler_idx &&
           a.derived == b.derived &&
           a.is_hi_half == b.is_hi_half &&
           a.symbol_name == b.symbol_name &&
           a.label == b.label;
}

bool TrackedValDomain::leq(const Value& a, const Value& b) {
    if (a.is_bottom() || b.is_top()) return true;
    if (b.is_bottom() || a.is_top()) return false;
    if (equals(a, b)) return true;
    // Same base pointer: concrete known offset <= symbolic unknown offset
    if (kind_has_offset(a.kind) && same_base_identity(a, b)) {
        if (a.offset_known && !b.offset_known && (!a.derived || b.derived)) return true;
        if (a.offset_known == b.offset_known && a.byte_offset == b.byte_offset && !a.derived && b.derived) return true;
    }
    if (a.kind == TrackedVal::Kind::SymbolValue && same_base_identity(a, b)) {
        if (!a.derived && b.derived) return true;
    }
    return false;
}

TrackedVal TrackedValDomain::join(const Value& a, const Value& b) {
    if (a.is_bottom()) return b;
    if (b.is_bottom()) return a;
    if (a.is_top() || b.is_top()) return top();
    if (equals(a, b)) return a;
    if (kind_has_offset(a.kind) && same_base_identity(a, b)) {
        TrackedVal out = a;
        if (!a.offset_known || !b.offset_known || a.byte_offset != b.byte_offset) {
            out.offset_known = false;
            out.byte_offset = 0;
        }
        out.derived = a.derived || b.derived;
        return out;
    }
    if (a.kind == TrackedVal::Kind::SymbolValue && same_base_identity(a, b)) {
        TrackedVal out = a;
        out.derived = a.derived || b.derived;
        return out;
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

bool TrackedValDomain::try_as_const_u64(const Value& v, uint64_t* out) {
    if (v.kind == TrackedVal::Kind::ConstInt) {
        if (out) *out = static_cast<uint64_t>(v.const_val);
        return true;
    }
    return false;
}

TrackedVal TrackedValDomain::make_const_u64(uint64_t v) {
    return TrackedVal::make_const(static_cast<int64_t>(v));
}

// ============================================================================
// SSA Pointer Transfer Visitor
// ============================================================================

namespace {

class SsaPointerTransferVisitor : public SsaDomainTransferVisitor<TrackedValDomain> {
public:
    SsaPointerTransferVisitor(
        const PointerAnalysisEnv& env,
        std::map<uint32_t, PilotFauWrite>* pilot_writes,
        std::vector<std::vector<std::string>>* ssa_notes)
        : env_(env), pilot_writes_(pilot_writes), ssa_notes_(ssa_notes) {}

    void on_const(compiler::SsaId dst, const compiler::SsaNode& node) override {
        using tables::SsaOp;
        switch (node.op) {
            case SsaOp::ConstU64:
                assign(dst, TrackedVal::make_const(static_cast<int64_t>(node.imm_u64)));
                break;
            case SsaOp::ConstS32:
                assign(dst, TrackedVal::make_const(static_cast<int64_t>(static_cast<int32_t>(node.imm_u32))));
                break;
            case SsaOp::ConstU32:
            case SsaOp::ConstBool:
            case SsaOp::ConstF32:
                assign(dst, TrackedVal::make_const(static_cast<int64_t>(node.imm_u32)));
                break;
            default:
                assign(dst, TrackedVal::make_top());
                break;
        }
    }

    void on_load_sideband(compiler::SsaId dst, uint32_t member_idx, bool /*is_64bit*/,
                          const compiler::SsaNode&) override {
        if (env_.role == PointerAnalysisEnv::StageRole::Pilot) {
            // Pass 1 pilot preamble: member 3 = pilot_fau_dst_va (FAU_0), member 2 = srt_base_va (SRT_0)
            if (member_idx == 3) {
                assign(dst, TrackedVal::make_fau_dst(0, true));
                return;
            }
            if (member_idx == 2) {
                assign(dst, TrackedVal::make_srt_base(0, true));
                return;
            }
        }
        assign(dst, TrackedVal::make_top());
    }

    void on_load_builtin(compiler::SsaId dst, uint32_t /*builtin_id*/,
                         const compiler::SsaNode& node) override {
        std::string lbl = node.aux_str.empty() ? "gl_BuiltIn" : node.aux_str;
        assign(dst, TrackedVal::make_abi(std::move(lbl)));
    }

    void on_load_fau(compiler::SsaId dst, uint32_t fau_word_idx,
                     const compiler::SsaNode&) override {
        uint32_t byte_off = fau_word_idx * 4u;
        // Check FAU memory (seeded from Pilot or prior writes)
        if (state_) {
            TrackedVal fv = state_->fau(byte_off);
            if (!fv.is_bottom()) {
                assign(dst, fv);
                return;
            }
            // High word of an 8-byte FAU entry?
            if ((byte_off & 4u) != 0) {
                TrackedVal lo = state_->fau(byte_off & ~7u);
                if (!lo.is_bottom() && lo.is_pointer()) {
                    lo.is_hi_half = true;
                    assign(dst, lo);
                    return;
                }
            }
        }
        assign(dst, TrackedVal::make_fau_uniform(byte_off));
    }

    void on_split64(compiler::SsaId dst, compiler::SsaId src64,
                    const compiler::SsaNode&) override {
        // Carry the 64-bit abstract value on the TupleU32x2 node itself
        assign(dst, eval(src64));
    }

    void on_tuple_extract(compiler::SsaId dst, compiler::SsaId tuple_val, uint32_t elem_idx,
                          const compiler::SsaNode& node) override {
        if (node.type == compiler::ValType::Token) return;
        TrackedVal tv = eval(tuple_val);
        if (tv.is_bottom() || tv.is_top()) {
            assign(dst, tv);
            return;
        }
        if (elem_idx == 0) {
            tv.is_hi_half = false;
            assign(dst, tv);
        } else if (elem_idx == 1) {
            if (tv.kind == TrackedVal::Kind::ConstInt) {
                assign(dst, TrackedVal::make_const(static_cast<int64_t>(static_cast<uint64_t>(tv.const_val) >> 32)));
            } else {
                tv.is_hi_half = true;
                assign(dst, tv);
            }
        } else {
            assign(dst, TrackedVal::make_top());
        }
    }

    void on_pack64(compiler::SsaId dst, compiler::SsaId lo, compiler::SsaId hi,
                   const compiler::SsaNode&) override {
        TrackedVal vlo = eval(lo);
        TrackedVal vhi = eval(hi);
        if (vlo.kind == TrackedVal::Kind::ConstInt && vhi.kind == TrackedVal::Kind::ConstInt) {
            uint64_t packed = (static_cast<uint64_t>(static_cast<uint32_t>(vhi.const_val)) << 32) |
                              static_cast<uint64_t>(static_cast<uint32_t>(vlo.const_val));
            assign(dst, TrackedVal::make_const(static_cast<int64_t>(packed)));
            return;
        }
        // Reconstruct a split 64-bit pointer or FAU resource
        if (vlo.is_pointer() || vlo.kind == TrackedVal::Kind::FauUniformRef ||
            vlo.kind == TrackedVal::Kind::SymbolValue) {
            vlo.is_hi_half = false;
            assign(dst, vlo);
            return;
        }
        if (vhi.is_pointer()) {
            vhi.is_hi_half = false;
            assign(dst, vhi);
            return;
        }
        assign(dst, TrackedVal::make_top());
    }

    void on_vec(compiler::SsaId dst, const compiler::SsaNode& node) override {
        if (node.op == tables::SsaOp::VecExtract && !node.args.empty()) {
            TrackedVal base = eval(node.args[0]);
            if (base.kind == TrackedVal::Kind::AbiPreload) {
                static const char kComp[] = "xyzw";
                std::string lbl = base.label;
                if (node.imm_u32 < 4) {
                    lbl.push_back('.');
                    lbl.push_back(kComp[node.imm_u32]);
                }
                assign(dst, TrackedVal::make_abi(std::move(lbl)));
                return;
            }
            if (!base.is_bottom() && !base.is_top()) {
                assign(dst, base);
                return;
            }
        }
        assign(dst, TrackedVal::make_top());
    }

    void on_convert(compiler::SsaId dst, compiler::SsaId src,
                    const compiler::SsaNode& node) override {
        TrackedVal v = eval(src);
        if (node.op == tables::SsaOp::UConvert32To64 || node.op == tables::SsaOp::SConvert32To64 ||
            node.op == tables::SsaOp::UConvert64To32) {
            if (v.kind == TrackedVal::Kind::ConstInt) {
                if (node.op == tables::SsaOp::UConvert32To64) {
                    assign(dst, TrackedVal::make_const(static_cast<int64_t>(static_cast<uint32_t>(v.const_val))));
                } else if (node.op == tables::SsaOp::SConvert32To64) {
                    assign(dst, TrackedVal::make_const(static_cast<int64_t>(static_cast<int32_t>(v.const_val))));
                } else {
                    assign(dst, TrackedVal::make_const(static_cast<int64_t>(static_cast<uint32_t>(v.const_val))));
                }
                return;
            }
            if (!v.is_bottom() && !v.is_top()) {
                assign(dst, v);
                return;
            }
        }
        assign(dst, TrackedVal::make_top());
    }

    void on_iadd(compiler::SsaId dst, compiler::SsaId lhs, compiler::SsaId rhs,
                 bool is_64bit, const compiler::SsaNode& node) override {
        TrackedVal a = eval(lhs);
        TrackedVal b = eval(rhs);
        if (a.is_pointer() && b.kind == TrackedVal::Kind::ConstInt) {
            TrackedVal out = add_pointer_offset(a, b.const_val, env_);
            assign(dst, out);
            record_note(dst, "ptr " + out.format_short());
            return;
        }
        if (b.is_pointer() && a.kind == TrackedVal::Kind::ConstInt) {
            TrackedVal out = add_pointer_offset(b, a.const_val, env_);
            assign(dst, out);
            record_note(dst, "ptr " + out.format_short());
            return;
        }
        if (a.is_pointer()) {
            TrackedVal out = add_unknown_offset(a);
            assign(dst, out);
            return;
        }
        if (b.is_pointer()) {
            TrackedVal out = add_unknown_offset(b);
            assign(dst, out);
            return;
        }
        if (a.kind == TrackedVal::Kind::SymbolValue || b.kind == TrackedVal::Kind::SymbolValue) {
            TrackedVal s = (a.kind == TrackedVal::Kind::SymbolValue) ? a : b;
            s.derived = true;
            assign(dst, s);
            return;
        }
        // Delegate constant folding & x+0 extensional equalities to base
        SsaDomainTransferVisitor<TrackedValDomain>::on_iadd(dst, lhs, rhs, is_64bit, node);
    }

    void on_isub(compiler::SsaId dst, compiler::SsaId lhs, compiler::SsaId rhs,
                 bool is_64bit, const compiler::SsaNode& node) override {
        TrackedVal a = eval(lhs);
        TrackedVal b = eval(rhs);
        if (a.is_pointer() && b.kind == TrackedVal::Kind::ConstInt) {
            TrackedVal out = add_pointer_offset(a, -b.const_val, env_);
            assign(dst, out);
            return;
        }
        if (a.kind == TrackedVal::Kind::SymbolValue) {
            a.derived = true;
            assign(dst, a);
            return;
        }
        SsaDomainTransferVisitor<TrackedValDomain>::on_isub(dst, lhs, rhs, is_64bit, node);
    }

    void on_imul(compiler::SsaId dst, compiler::SsaId lhs, compiler::SsaId rhs,
                 bool is_64bit, const compiler::SsaNode& node) override {
        TrackedVal a = eval(lhs);
        TrackedVal b = eval(rhs);
        if (a.kind == TrackedVal::Kind::SymbolValue || b.kind == TrackedVal::Kind::SymbolValue) {
            TrackedVal s = (a.kind == TrackedVal::Kind::SymbolValue) ? a : b;
            s.derived = true;
            assign(dst, s);
            return;
        }
        SsaDomainTransferVisitor<TrackedValDomain>::on_imul(dst, lhs, rhs, is_64bit, node);
    }

    void on_and(compiler::SsaId dst, compiler::SsaId lhs, compiler::SsaId rhs,
                bool is_64bit, const compiler::SsaNode& node) override {
        TrackedVal a = eval(lhs);
        TrackedVal b = eval(rhs);
        // Common Mali pointer masking (e.g. `ptr & 0xFFFFFFFFFFFFFFF0` or `ptr & ~3`) preserves pointer base
        if (a.is_pointer() && b.kind == TrackedVal::Kind::ConstInt && b.const_val != 0) {
            TrackedVal out = a;
            if (out.offset_known) out.byte_offset = out.byte_offset & b.const_val;
            assign(dst, out);
            return;
        }
        if (b.is_pointer() && a.kind == TrackedVal::Kind::ConstInt && a.const_val != 0) {
            TrackedVal out = b;
            if (out.offset_known) out.byte_offset = out.byte_offset & a.const_val;
            assign(dst, out);
            return;
        }
        SsaDomainTransferVisitor<TrackedValDomain>::on_and(dst, lhs, rhs, is_64bit, node);
    }

    void on_psb_load(compiler::SsaId dst, compiler::SsaId addr, uint32_t elem_bytes,
                     const compiler::SsaNode&) override {
        TrackedVal av = eval(addr);
        TrackedVal loaded = deref_pointer(av, elem_bytes, env_);
        assign(dst, loaded);
        if (!av.is_bottom() && !av.is_top()) {
            record_note(dst, "load " + av.format_short() + " -> " + loaded.format_short());
        }
    }

    void on_psb_store(compiler::SsaId addr, compiler::SsaId val, uint32_t elem_bytes,
                      const compiler::SsaNode& node) override {
        TrackedVal av = eval(addr);
        TrackedVal vv = eval(val);
        if (av.kind == TrackedVal::Kind::FauDstBase && av.offset_known && av.byte_offset >= 0) {
            uint32_t fau_off = static_cast<uint32_t>(av.byte_offset);
            if (state_) {
                state_->join_fau(fau_off, vv);
            }
            if (pilot_writes_ && !vv.is_bottom() && !vv.is_top()) {
                auto it = pilot_writes_->find(fau_off);
                if (it == pilot_writes_->end()) {
                    (*pilot_writes_)[fau_off] = PilotFauWrite{fau_off, elem_bytes, vv};
                } else {
                    it->second.value = TrackedValDomain::join(it->second.value, vv);
                    it->second.byte_size = std::max(it->second.byte_size, elem_bytes);
                }
            }
            record_note(node.id, "pilot store FAU+0x" + hex_off(fau_off).substr(3) + " <- " + vv.format_short());
        } else if (!av.is_bottom() && !av.is_top()) {
            record_note(node.id, "store " + av.format_short());
        }
    }

    void on_load_vert_attr(compiler::SsaId dst, const compiler::SsaNode& node) override {
        uint32_t loc = node.imm_u32;
        std::string sym = env_.lookup_input_symbol ? env_.lookup_input_symbol(loc) : "";
        TrackedVal v = TrackedVal::make_vertex_attr(loc, sym);
        assign(dst, v);
        record_note(dst, v.format_short());
    }

    void on_load_frag_varying(compiler::SsaId dst, const compiler::SsaNode& node) override {
        uint32_t loc = node.imm_u32;
        std::string sym = env_.lookup_input_symbol ? env_.lookup_input_symbol(loc) : "";
        TrackedVal v = TrackedVal::make_varying(loc, sym);
        assign(dst, v);
        record_note(dst, v.format_short());
    }

    void on_tex_sample(compiler::SsaId dst, const compiler::SsaNode& node) override {
        uint32_t img = node.imm_u32;
        uint32_t smp = node.imm_u32_2;
        std::string sym = env_.lookup_image_symbol ? env_.lookup_image_symbol(img) : "";
        TrackedVal v = TrackedVal::make_tex_sample(img, smp, sym);
        assign(dst, v);
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

// ============================================================================
// Public Entrypoints
// ============================================================================

PointerAnalysisResult run_ssa_pointer_analysis(
    const compiler::ValhallSsaModule& mod,
    const PointerAnalysisEnv& env,
    const SsaLivenessResult* liveness,
    uint32_t max_iterations) {
    PointerAnalysisResult out;
    out.ssa_notes.resize(mod.nodes.size());

    AbstractState<TrackedValDomain> entry(mod.nodes.size());
    for (const auto& kv : env.initial_fau) {
        entry.set_fau(kv.first, kv.second);
    }

    SsaPointerTransferVisitor visitor(env, &out.pilot_fau_writes, &out.ssa_notes);
    SsaDataflowResult<TrackedValDomain> df =
        run_ssa_forward_dataflow<TrackedValDomain>(mod, entry, visitor, liveness, max_iterations);

    out.converged = df.converged;
    out.iterations = df.iterations;
    out.final_state = std::move(df.final_state);
    if (liveness) {
        out.liveness = *liveness;
    }
    return out;
}

}  // namespace valhall::dataflow

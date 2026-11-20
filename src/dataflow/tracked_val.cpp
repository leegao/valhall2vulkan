#include "tracked_val.h"

#include <cstdio>
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

}  // namespace

// ============================================================================
// TrackedVal constructors & member helpers
// ============================================================================

TrackedVal TrackedVal::make_bottom() {
    TrackedVal v;
    v.kind = Kind::Bottom;
    return v;
}

TrackedVal TrackedVal::make_top() {
    TrackedVal v;
    v.kind = Kind::Top;
    return v;
}

TrackedVal TrackedVal::make_const(int64_t c) {
    TrackedVal v;
    v.kind = Kind::ConstInt;
    v.const_val = c;
    return v;
}

TrackedVal TrackedVal::make_abi(std::string lbl) {
    TrackedVal v;
    v.kind = Kind::AbiPreload;
    v.label = std::move(lbl);
    return v;
}

TrackedVal TrackedVal::make_fau_dst(int64_t off, bool ok) {
    TrackedVal v;
    v.kind = Kind::FauDstBase;
    v.byte_offset = off;
    v.offset_known = ok;
    v.label = "pilot_fau_dst_va (FAU_0)";
    return v;
}

TrackedVal TrackedVal::make_srt_base(int64_t off, bool ok) {
    TrackedVal v;
    v.kind = Kind::SrtBase;
    v.byte_offset = off;
    v.offset_known = ok;
    v.label = "srt_base_va (SRT_0)";
    return v;
}

TrackedVal TrackedVal::make_srt_rack(int32_t r, int64_t off, bool ok) {
    TrackedVal v;
    v.kind = Kind::SrtRackPtr;
    v.rack = r;
    v.byte_offset = off;
    v.offset_known = ok;
    return v;
}

TrackedVal TrackedVal::make_descriptor(int32_t r, int32_t d, std::string sym, int64_t foff, bool ok) {
    TrackedVal v;
    v.kind = Kind::DescriptorAddr;
    v.rack = r;
    v.desc_idx = d;
    v.symbol_name = std::move(sym);
    v.byte_offset = foff;
    v.offset_known = ok;
    return v;
}

TrackedVal TrackedVal::make_buffer_addr(int32_t r, int32_t d, std::string sym, int64_t off, bool drv, bool ok) {
    TrackedVal v;
    v.kind = Kind::BufferAddr;
    v.rack = r;
    v.desc_idx = d;
    v.symbol_name = std::move(sym);
    v.byte_offset = off;
    v.derived = drv;
    v.offset_known = ok;
    return v;
}

TrackedVal TrackedVal::make_buffer_size(int32_t r, int32_t d, std::string sym) {
    TrackedVal v;
    v.kind = Kind::BufferSize;
    v.rack = r;
    v.desc_idx = d;
    v.symbol_name = std::move(sym);
    return v;
}

TrackedVal TrackedVal::make_fau_uniform(uint32_t foff, std::string sym) {
    TrackedVal v;
    v.kind = Kind::FauUniformRef;
    v.fau_offset = foff;
    v.symbol_name = std::move(sym);
    return v;
}

TrackedVal TrackedVal::make_symbol_value(int32_t r, int32_t d, std::string sym, bool drv) {
    TrackedVal v;
    v.kind = Kind::SymbolValue;
    v.rack = r;
    v.desc_idx = d;
    v.symbol_name = std::move(sym);
    v.derived = drv;
    return v;
}

TrackedVal TrackedVal::make_data_section(int64_t off, bool ok) {
    TrackedVal v;
    v.kind = Kind::DataSectionPtr;
    v.byte_offset = off;
    v.offset_known = ok;
    return v;
}

TrackedVal TrackedVal::make_tls_ptr(int64_t off, bool ok) {
    TrackedVal v;
    v.kind = Kind::TlsPtr;
    v.byte_offset = off;
    v.offset_known = ok;
    return v;
}

TrackedVal TrackedVal::make_wls_ptr(int64_t off, bool ok) {
    TrackedVal v;
    v.kind = Kind::WlsPtr;
    v.byte_offset = off;
    v.offset_known = ok;
    return v;
}

TrackedVal TrackedVal::make_vertex_attr(uint32_t loc, std::string sym) {
    TrackedVal v;
    v.kind = Kind::VertexAttrValue;
    v.location = loc;
    v.symbol_name = std::move(sym);
    return v;
}

TrackedVal TrackedVal::make_varying(uint32_t loc, std::string sym) {
    TrackedVal v;
    v.kind = Kind::VaryingValue;
    v.location = loc;
    v.symbol_name = std::move(sym);
    return v;
}

TrackedVal TrackedVal::make_tex_sample(uint32_t img, uint32_t smp, std::string sym) {
    TrackedVal v;
    v.kind = Kind::TexSampleValue;
    v.location = img;
    v.sampler_idx = smp;
    v.symbol_name = std::move(sym);
    return v;
}

TrackedVal TrackedVal::make_blend_desc(uint32_t loc, std::string sym) {
    TrackedVal v;
    v.kind = Kind::BlendDescRef;
    v.location = loc;
    v.symbol_name = std::move(sym);
    return v;
}

bool TrackedVal::has_offset() const {
    switch (kind) {
        case Kind::FauDstBase:
        case Kind::SrtBase:
        case Kind::SrtRackPtr:
        case Kind::DescriptorAddr:
        case Kind::BufferAddr:
        case Kind::DataSectionPtr:
        case Kind::TlsPtr:
        case Kind::WlsPtr:
            return true;
        default:
            return false;
    }
}

bool TrackedVal::same_base_identity(const TrackedVal& other) const {
    if (kind != other.kind) return false;
    if (is_hi_half != other.is_hi_half) return false;
    switch (kind) {
        // Singleton per-dispatch / per-thread / per-workgroup base regions:
        // there is only ONE FAU_0, ONE SRT_0, ONE .data section, ONE TLS frame,
        // and ONE WLS block per shader invocation.
        case Kind::FauDstBase:
        case Kind::SrtBase:
        case Kind::DataSectionPtr:
        case Kind::TlsPtr:
        case Kind::WlsPtr:
            return true;
        case Kind::SrtRackPtr:
            return rack == other.rack;
        case Kind::DescriptorAddr:
        case Kind::BufferAddr:
        case Kind::BufferSize:
        case Kind::SymbolValue:
            return rack == other.rack && desc_idx == other.desc_idx && symbol_name == other.symbol_name;
        case Kind::FauUniformRef:
            return fau_offset == other.fau_offset && symbol_name == other.symbol_name;
        case Kind::AbiPreload:
            return label == other.label;
        case Kind::VertexAttrValue:
        case Kind::VaryingValue:
        case Kind::BlendDescRef:
            return location == other.location && symbol_name == other.symbol_name;
        case Kind::TexSampleValue:
            return location == other.location && sampler_idx == other.sampler_idx &&
                   symbol_name == other.symbol_name;
        default:
            return false;
    }
}

SymbolicMemoryLoc TrackedVal::to_memory_loc() const {
    switch (kind) {
        case Kind::FauDstBase:
            return SymbolicMemoryLoc::make_fau(offset_known && byte_offset >= 0
                                                   ? static_cast<uint32_t>(byte_offset)
                                                   : 0u);
        case Kind::SrtBase:
            return SymbolicMemoryLoc::make_srt(-1, byte_offset, offset_known);
        case Kind::SrtRackPtr:
            return SymbolicMemoryLoc::make_srt(rack, byte_offset, offset_known);
        case Kind::DescriptorAddr:
        case Kind::BufferAddr:
            return SymbolicMemoryLoc::make_buffer(rack, desc_idx, byte_offset, offset_known, symbol_name);
        case Kind::DataSectionPtr:
            return SymbolicMemoryLoc::make_data_section(byte_offset, offset_known);
        case Kind::TlsPtr:
            return SymbolicMemoryLoc::make_tls(byte_offset, offset_known);
        case Kind::WlsPtr:
            return SymbolicMemoryLoc::make_wls(byte_offset, offset_known);
        default:
            return SymbolicMemoryLoc{};
    }
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
            if (!symbol_name.empty()) oss << "&" << symbol_name;
            else oss << "&Buf[" << desc_idx << "]";
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
// TrackedValDomain complete lattice operations
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
    if (a.has_offset() && a.same_base_identity(b)) {
        if (a.offset_known && !b.offset_known && (!a.derived || b.derived)) return true;
        if (a.offset_known == b.offset_known && a.byte_offset == b.byte_offset && !a.derived && b.derived) return true;
    }
    if (a.kind == TrackedVal::Kind::SymbolValue && a.same_base_identity(b)) {
        if (!a.derived && b.derived) return true;
    }
    return false;
}

TrackedVal TrackedValDomain::join(const Value& a, const Value& b) {
    if (a.is_bottom()) return b;
    if (b.is_bottom()) return a;
    if (a.is_top() || b.is_top()) return top();
    if (equals(a, b)) return a;
    if (a.has_offset() && a.same_base_identity(b)) {
        TrackedVal out = a;
        if (!a.offset_known || !b.offset_known || a.byte_offset != b.byte_offset) {
            out.offset_known = false;
            out.byte_offset = 0;
        }
        out.derived = a.derived || b.derived;
        return out;
    }
    if (a.kind == TrackedVal::Kind::SymbolValue && a.same_base_identity(b)) {
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

std::string format_tracked_val_set(const TrackedValSet& s) {
    if (s.is_bottom()) return "<bottom>";
    if (s.is_top()) return "<top>";
    if (s.items.size() == 1) return s.items[0].format_short();
    std::ostringstream oss;
    oss << "{";
    for (size_t i = 0; i < s.items.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << s.items[i].format_short();
    }
    oss << "}";
    return oss.str();
}

TrackedVal collapse_tracked_val_set(const TrackedValSet& s) {
    if (s.is_bottom()) return TrackedVal::make_bottom();
    if (s.is_top()) return TrackedVal::make_top();
    TrackedVal acc = TrackedVal::make_bottom();
    for (const auto& item : s.items) {
        acc = TrackedValDomain::join(acc, item);
    }
    return acc;
}

}  // namespace valhall::dataflow

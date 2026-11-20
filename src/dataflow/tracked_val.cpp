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

bool is_int_width_convert(ConvertKind kind) {
    return kind == ConvertKind::U32ToU64 || kind == ConvertKind::S32ToS64 ||
           kind == ConvertKind::U64ToU32 || kind == ConvertKind::Bitcast;
}

const char* sideband_member_name(uint32_t member_idx) {
    switch (member_idx) {
        case 0: return "fau_words";
        case 1: return "shader_pc_base";
        case 2: return "srt_base_va";
        case 3: return "pilot_fau_dst_va";
        case 4: return "blend_descs";
        case 5: return "local_size_xyz";
        case 6: return "fb_info";
        default: return nullptr;
    }
}

}  // namespace

// PointerVal & PointerDomain

PointerVal PointerVal::make_bottom() {
    PointerVal v;
    v.kind = Kind::Bottom;
    return v;
}

PointerVal PointerVal::make_top() {
    PointerVal v;
    v.kind = Kind::Top;
    return v;
}

PointerVal PointerVal::make_fau_dst(int64_t off, bool ok) {
    PointerVal v;
    v.kind = Kind::FauDstBase;
    v.byte_offset = ok ? off : 0;
    v.offset_known = ok;
    v.label = "pilot_fau_dst_va (FAU_0)";
    return v;
}

PointerVal PointerVal::make_srt_base(int64_t off, bool ok) {
    PointerVal v;
    v.kind = Kind::SrtBase;
    v.byte_offset = ok ? off : 0;
    v.offset_known = ok;
    v.label = "srt_base_va (SRT_0)";
    return v;
}

PointerVal PointerVal::make_srt_rack(int32_t r, int64_t off, bool ok) {
    PointerVal v;
    v.kind = Kind::SrtRackPtr;
    v.rack = r;
    v.byte_offset = ok ? off : 0;
    v.offset_known = ok;
    return v;
}

PointerVal PointerVal::make_descriptor(int32_t r, int32_t d, std::string sym, int64_t foff, bool ok) {
    PointerVal v;
    v.kind = Kind::DescriptorAddr;
    v.rack = r;
    v.desc_idx = d;
    v.symbol_name = std::move(sym);
    v.byte_offset = ok ? foff : 0;
    v.offset_known = ok;
    return v;
}

PointerVal PointerVal::make_buffer_addr(int32_t r, int32_t d, std::string sym, int64_t off, bool drv, bool ok) {
    PointerVal v;
    v.kind = Kind::BufferAddr;
    v.rack = r;
    v.desc_idx = d;
    v.symbol_name = std::move(sym);
    v.byte_offset = ok ? off : 0;
    v.derived = drv;
    v.offset_known = ok;
    return v;
}

PointerVal PointerVal::make_data_section(int64_t off, bool ok) {
    PointerVal v;
    v.kind = Kind::DataSectionPtr;
    v.byte_offset = ok ? off : 0;
    v.offset_known = ok;
    return v;
}

PointerVal PointerVal::make_tls_ptr(int64_t off, bool ok) {
    PointerVal v;
    v.kind = Kind::TlsPtr;
    v.byte_offset = ok ? off : 0;
    v.offset_known = ok;
    return v;
}

PointerVal PointerVal::make_wls_ptr(int64_t off, bool ok) {
    PointerVal v;
    v.kind = Kind::WlsPtr;
    v.byte_offset = ok ? off : 0;
    v.offset_known = ok;
    return v;
}

bool PointerVal::same_base_identity(const PointerVal& other) const {
    if (kind != other.kind) return false;
    switch (kind) {
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
            return rack == other.rack && desc_idx == other.desc_idx && symbol_name == other.symbol_name;
        default:
            return false;
    }
}

SymbolicMemoryLoc PointerVal::to_memory_loc() const {
    switch (kind) {
        case Kind::FauDstBase:
            return SymbolicMemoryLoc::make_fau(byte_offset, offset_known);
        case Kind::SrtBase:
            return SymbolicMemoryLoc::make_srt(-1, byte_offset, offset_known);
        case Kind::SrtRackPtr:
            return SymbolicMemoryLoc::make_srt(rack, byte_offset, offset_known);
        case Kind::DescriptorAddr:
            return SymbolicMemoryLoc::make_descriptor(rack, desc_idx, byte_offset, offset_known, symbol_name);
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

PointerVal PointerVal::with_memory_loc(const SymbolicMemoryLoc& loc) const {
    if (!is_valid_pointer()) return PointerVal::make_top();
    PointerVal out = *this;
    out.byte_offset = loc.byte_offset;
    out.offset_known = loc.offset_known;
    return out;
}

bool PointerDomain::equals(const Value& a, const Value& b) {
    if (a.kind != b.kind) return false;
    if (a.is_bottom() || a.is_top()) return true;
    return a.byte_offset == b.byte_offset &&
           a.offset_known == b.offset_known &&
           a.rack == b.rack &&
           a.desc_idx == b.desc_idx &&
           a.derived == b.derived &&
           a.symbol_name == b.symbol_name &&
           a.label == b.label;
}

bool PointerDomain::leq(const Value& a, const Value& b) {
    if (a.is_bottom() || b.is_top()) return true;
    if (b.is_bottom() || a.is_top()) return false;
    if (equals(a, b)) return true;
    if (a.same_base_identity(b)) {
        if (a.offset_known && !b.offset_known && (!a.derived || b.derived)) return true;
        if (a.offset_known == b.offset_known && a.byte_offset == b.byte_offset && !a.derived && b.derived) return true;
    }
    return false;
}

PointerVal PointerDomain::join(const Value& a, const Value& b) {
    if (a.is_bottom()) return b;
    if (b.is_bottom()) return a;
    if (a.is_top() || b.is_top()) return top();
    if (equals(a, b)) return a;
    if (a.same_base_identity(b)) {
        PointerVal out = a;
        if (!a.offset_known || !b.offset_known || a.byte_offset != b.byte_offset) {
            out.offset_known = false;
            out.byte_offset = 0;
        }
        out.derived = a.derived || b.derived;
        return out;
    }
    return top();
}

PointerVal PointerDomain::meet(const Value& a, const Value& b) {
    if (a.is_top()) return b;
    if (b.is_top()) return a;
    if (a.is_bottom() || b.is_bottom()) return bottom();
    if (equals(a, b)) return a;
    if (leq(a, b)) return a;
    if (leq(b, a)) return b;
    return bottom();
}

std::string PointerDomain::format(const Value& v) {
    std::ostringstream oss;
    switch (v.kind) {
        case PointerVal::Kind::Bottom: return "<bottom>";
        case PointerVal::Kind::Top: return "<top>";
        case PointerVal::Kind::FauDstBase:
            oss << "FAU_0";
            if (!v.offset_known) oss << "+?";
            else if (v.byte_offset != 0) oss << hex_off(v.byte_offset);
            return oss.str();
        case PointerVal::Kind::SrtBase:
            oss << "SRT_0";
            if (!v.offset_known) oss << "+?";
            else if (v.byte_offset != 0) oss << hex_off(v.byte_offset);
            return oss.str();
        case PointerVal::Kind::SrtRackPtr:
            oss << "SRT[" << v.rack << "]";
            if (!v.offset_known) oss << "+?";
            else if (v.byte_offset != 0) oss << hex_off(v.byte_offset);
            return oss.str();
        case PointerVal::Kind::DescriptorAddr:
            oss << "SRT[" << v.rack << "]->Desc[" << v.desc_idx << "]";
            if (!v.symbol_name.empty()) oss << " (" << v.symbol_name << ")";
            if (!v.offset_known) oss << "+?";
            else if (v.byte_offset != 0) oss << hex_off(v.byte_offset);
            return oss.str();
        case PointerVal::Kind::BufferAddr:
            if (!v.symbol_name.empty()) oss << "&" << v.symbol_name;
            else oss << "&Buf[" << v.desc_idx << "]";
            if (!v.offset_known) oss << "+?";
            else if (v.byte_offset != 0) oss << hex_off(v.byte_offset);
            if (v.rack >= 0 && v.desc_idx >= 0) {
                oss << " (SRT[" << v.rack << "]->Buf[" << v.desc_idx << "].addr)";
            }
            return oss.str();
        case PointerVal::Kind::DataSectionPtr:
            oss << ".data";
            if (!v.offset_known) oss << "+?";
            else if (v.byte_offset != 0) oss << hex_off(v.byte_offset);
            return oss.str();
        case PointerVal::Kind::TlsPtr:
            oss << "TLS";
            if (!v.offset_known) oss << "+?";
            else if (v.byte_offset != 0) oss << hex_off(v.byte_offset);
            return oss.str();
        case PointerVal::Kind::WlsPtr:
            oss << "WLS";
            if (!v.offset_known) oss << "+?";
            else if (v.byte_offset != 0) oss << hex_off(v.byte_offset);
            return oss.str();
    }
    return "<unknown-ptr>";
}

PointerVal PointerDomain::apply_int_delta(const Value& ptr, int64_t delta,
                                          const SrtSymbolLookupFn& lookup_srt) {
    if (!ptr.is_valid_pointer()) return top();
    SymbolicMemoryLoc loc = ptr.to_memory_loc().with_added_offset(delta);
    PointerVal out = ptr.with_memory_loc(loc);
    if (out.kind == PointerVal::Kind::SrtRackPtr && loc.offset_known && loc.byte_offset >= 0 &&
        (loc.byte_offset % 32) == 0 && out.rack >= 0) {
        int32_t desc = static_cast<int32_t>(loc.byte_offset / 32);
        std::string sym = lookup_srt
                              ? lookup_srt(static_cast<uint8_t>(out.rack), static_cast<uint32_t>(desc))
                              : "";
        return PointerVal::make_descriptor(out.rack, desc, std::move(sym), 0, true);
    }
    return out;
}

PointerVal PointerDomain::widen_unknown_offset(const Value& ptr) {
    if (!ptr.is_valid_pointer()) return top();
    return ptr.with_memory_loc(ptr.to_memory_loc().with_unknown_offset());
}

PointerVal PointerDomain::apply_int_mask(const Value& ptr, int64_t mask) {
    if (!ptr.is_valid_pointer()) return top();
    return ptr.with_memory_loc(ptr.to_memory_loc().with_masked_offset(mask));
}

ConstIntVal PointerDomain::sub(const Value& a, const Value& b) {
    if (a.is_bottom() || b.is_bottom()) return ConstIntVal::make_bottom();
    if (a.is_valid_pointer() && b.is_valid_pointer() &&
        a.same_base_identity(b) && a.offset_known && b.offset_known) {
        return ConstIntVal::make(a.byte_offset - b.byte_offset);
    }
    return ConstIntVal::make_top();
}

PointerVal PointerDomain::convert(const Value& v, ConvertKind kind) {
    if (v.is_bottom()) return bottom();
    if (v.is_top()) return top();
    return is_int_width_convert(kind) ? v : top();
}

// SymbolValueDomain

bool SymbolValueDomain::equals(const Value& a, const Value& b) {
    if (a.state != b.state) return false;
    if (a.state != Value::State::Known) return true;
    return a.rack == b.rack && a.desc_idx == b.desc_idx &&
           a.symbol_name == b.symbol_name && a.derived == b.derived;
}

bool SymbolValueDomain::leq(const Value& a, const Value& b) {
    if (a.is_bottom() || b.is_top()) return true;
    if (b.is_bottom() || a.is_top()) return false;
    if (equals(a, b)) return true;
    return a.same_identity(b) && !a.derived && b.derived;
}

SymbolVal SymbolValueDomain::join(const Value& a, const Value& b) {
    if (a.is_bottom()) return b;
    if (b.is_bottom()) return a;
    if (a.is_top() || b.is_top()) return top();
    if (equals(a, b)) return a;
    if (a.same_identity(b)) {
        SymbolVal out = a;
        out.derived = a.derived || b.derived;
        return out;
    }
    return top();
}

SymbolVal SymbolValueDomain::meet(const Value& a, const Value& b) {
    if (a.is_top()) return b;
    if (b.is_top()) return a;
    if (a.is_bottom() || b.is_bottom()) return bottom();
    if (leq(a, b)) return a;
    if (leq(b, a)) return b;
    return bottom();
}

std::string SymbolValueDomain::format(const Value& v) {
    if (v.is_bottom()) return "<bottom>";
    if (v.is_top()) return "<top>";
    std::ostringstream oss;
    if (v.derived) oss << "derived(";
    oss << (v.symbol_name.empty() ? "sym" : v.symbol_name);
    if (v.derived) oss << ")";
    return oss.str();
}

SymbolVal SymbolValueDomain::convert(const Value& v, ConvertKind kind) {
    if (v.is_bottom()) return bottom();
    if (v.is_top()) return top();
    return is_int_width_convert(kind) ? v : top();
}

// ShaderAtomDomain

ShaderAtomVal ShaderAtomDomain::make_abi(std::string lbl) {
    ShaderAtomPayload p;
    p.kind = Kind::AbiPreload;
    p.label = std::move(lbl);
    return Value::make(std::move(p));
}

ShaderAtomVal ShaderAtomDomain::make_sideband(uint32_t member_idx) {
    ShaderAtomPayload p;
    p.kind = Kind::SidebandValue;
    p.location = member_idx;
    return Value::make(std::move(p));
}

ShaderAtomVal ShaderAtomDomain::make_buffer_size(int32_t r, int32_t d, std::string sym) {
    ShaderAtomPayload p;
    p.kind = Kind::BufferSize;
    p.rack = r;
    p.desc_idx = d;
    p.symbol_name = std::move(sym);
    return Value::make(std::move(p));
}

ShaderAtomVal ShaderAtomDomain::make_fau_uniform(uint32_t foff, std::string sym) {
    ShaderAtomPayload p;
    p.kind = Kind::FauUniformRef;
    p.fau_offset = foff;
    p.symbol_name = std::move(sym);
    return Value::make(std::move(p));
}

ShaderAtomVal ShaderAtomDomain::make_vertex_attr(uint32_t loc, std::string sym) {
    ShaderAtomPayload p;
    p.kind = Kind::VertexAttrValue;
    p.location = loc;
    p.symbol_name = std::move(sym);
    return Value::make(std::move(p));
}

ShaderAtomVal ShaderAtomDomain::make_varying(uint32_t loc, std::string sym) {
    ShaderAtomPayload p;
    p.kind = Kind::VaryingValue;
    p.location = loc;
    p.symbol_name = std::move(sym);
    return Value::make(std::move(p));
}

ShaderAtomVal ShaderAtomDomain::make_tex_sample(uint32_t img, uint32_t smp, std::string sym) {
    ShaderAtomPayload p;
    p.kind = Kind::TexSampleValue;
    p.location = img;
    p.sampler_idx = smp;
    p.symbol_name = std::move(sym);
    return Value::make(std::move(p));
}

ShaderAtomVal ShaderAtomDomain::make_blend_desc(uint32_t loc, std::string sym) {
    ShaderAtomPayload p;
    p.kind = Kind::BlendDescRef;
    p.location = loc;
    p.symbol_name = std::move(sym);
    return Value::make(std::move(p));
}

std::string ShaderAtomDomain::format(const Value& v) {
    if (v.is_bottom()) return "<bottom>";
    if (v.is_top()) return "<top>";
    const auto& p = v.value;
    std::ostringstream oss;
    switch (p.kind) {
        case Kind::AbiPreload:
            return p.label.empty() ? "ABI" : p.label;
        case Kind::SidebandValue: {
            if (const char* field = sideband_member_name(p.location)) {
                oss << "sideband." << field;
            } else {
                oss << "sideband[" << p.location << "]";
            }
            return oss.str();
        }
        case Kind::BufferSize:
            if (!p.symbol_name.empty()) oss << "sizeof(" << p.symbol_name << ")";
            else oss << "sizeof(SRT[" << p.rack << "]->Buf[" << p.desc_idx << "])";
            return oss.str();
        case Kind::FauUniformRef:
            if (!p.symbol_name.empty()) return p.symbol_name;
            oss << "FAU+0x" << std::hex << p.fau_offset;
            return oss.str();
        case Kind::VertexAttrValue:
            oss << "attr[" << p.location << "]";
            if (!p.symbol_name.empty()) oss << " (" << p.symbol_name << ")";
            return oss.str();
        case Kind::VaryingValue:
            oss << "varying[" << p.location << "]";
            if (!p.symbol_name.empty()) oss << " (" << p.symbol_name << ")";
            return oss.str();
        case Kind::TexSampleValue:
            oss << "tex(img=" << p.location << ",smp=" << p.sampler_idx << ")";
            if (!p.symbol_name.empty()) oss << " (" << p.symbol_name << ")";
            return oss.str();
        case Kind::BlendDescRef:
            oss << "color_out[" << p.location << "]";
            if (!p.symbol_name.empty()) oss << " (" << p.symbol_name << ")";
            return oss.str();
    }
    return "<unknown-atom>";
}

ShaderAtomVal ShaderAtomDomain::convert(const Value& v, ConvertKind kind) {
    if (v.is_bottom()) return bottom();
    if (v.is_top()) return top();
    return is_int_width_convert(kind) ? v : top();
}

// Formatting & Set Collapse Helpers

std::string format_tracked_val(const TrackedVal& v) {
    return TrackedValDomain::format(v);
}

std::string format_tracked_val_debug(const TrackedVal& v) {
    if (v.is_bottom()) return "[Bottom] <bottom>";
    if (v.is_top()) return "[Top] <top>";
    std::string tag = "[Unknown]";
    if (v.inner.is<ConstIntDomain>()) tag = "[ConstInt]";
    else if (v.inner.is<PointerDomain>()) tag = "[Pointer]";
    else if (v.inner.is<SymbolValueDomain>()) tag = "[Symbol]";
    else if (v.inner.is<ShaderAtomDomain>()) tag = "[Atom]";
    return tag + " " + TrackedValDomain::format(v);
}

std::string format_tracked_val_set(const TrackedValSet& s) {
    return TrackedValSetDomain::format(s);
}

TrackedVal collapse_tracked_val_set(const TrackedValSet& s) {
    if (s.is_bottom()) return TrackedValDomain::bottom();
    if (s.is_top()) return TrackedValDomain::top();
    TrackedVal acc = TrackedValDomain::bottom();
    for (const auto& item : s.items) {
        acc = TrackedValDomain::join(acc, item);
    }
    return acc;
}

}  // namespace valhall::dataflow

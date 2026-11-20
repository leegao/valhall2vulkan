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

PointerVal apply_pointer_delta(const PointerVal& base, int64_t delta, const SrtSymbolLookupFn& lookup_srt) {
    if (!base.is_valid_pointer()) return PointerVal::make_top();
    SymbolicMemoryLoc loc = base.to_memory_loc().with_added_offset(delta);
    PointerVal out = base.with_memory_loc(loc);
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

}  // namespace

// ConstIntDomain

ConstIntVal ConstIntDomain::join(const Value& a, const Value& b) {
    if (a.is_bottom()) return b;
    if (b.is_bottom()) return a;
    if (a.is_top() || b.is_top()) return top();
    return (a.value == b.value) ? a : top();
}

ConstIntVal ConstIntDomain::meet(const Value& a, const Value& b) {
    if (a.is_top()) return b;
    if (b.is_top()) return a;
    if (a.is_bottom() || b.is_bottom()) return bottom();
    return (a.value == b.value) ? a : bottom();
}

bool ConstIntDomain::leq(const Value& a, const Value& b) {
    if (a.is_bottom() || b.is_top()) return true;
    if (b.is_bottom() || a.is_top()) return false;
    return a.value == b.value;
}

bool ConstIntDomain::equals(const Value& a, const Value& b) {
    if (a.state != b.state) return false;
    return a.state != Value::State::Known || a.value == b.value;
}

std::string ConstIntDomain::format(const Value& v) {
    if (v.is_bottom()) return "<bottom>";
    if (v.is_top()) return "<top>";
    std::ostringstream oss;
    oss << "0x" << std::hex << static_cast<uint64_t>(v.value);
    return oss.str();
}

bool ConstIntDomain::try_as_const_u64(const Value& v, uint64_t* out) {
    if (v.is_known()) {
        if (out) *out = static_cast<uint64_t>(v.value);
        return true;
    }
    return false;
}

ConstIntVal ConstIntDomain::make_const_u64(uint64_t v) {
    return Value::make(static_cast<int64_t>(v));
}

ConstIntVal ConstIntDomain::add(const Value& a, const Value& b) {
    if (a.is_bottom() || b.is_bottom()) return bottom();
    if (a.is_known() && b.is_known()) return Value::make(a.value + b.value);
    return top();
}

ConstIntVal ConstIntDomain::sub(const Value& a, const Value& b) {
    if (a.is_bottom() || b.is_bottom()) return bottom();
    if (a.is_known() && b.is_known()) return Value::make(a.value - b.value);
    return top();
}

ConstIntVal ConstIntDomain::mul(const Value& a, const Value& b) {
    if (a.is_bottom() || b.is_bottom()) return bottom();
    if ((a.is_known() && a.value == 0) || (b.is_known() && b.value == 0)) return Value::make(0);
    if (a.is_known() && b.is_known()) return Value::make(a.value * b.value);
    return top();
}

ConstIntVal ConstIntDomain::div(const Value& a, const Value& b) {
    if (a.is_bottom() || b.is_bottom()) return bottom();
    if (a.is_known() && b.is_known() && b.value != 0) return Value::make(a.value / b.value);
    return top();
}

ConstIntVal ConstIntDomain::bit_and(const Value& a, const Value& b, bool is64) {
    if (a.is_bottom() || b.is_bottom()) return bottom();
    if ((a.is_known() && a.value == 0) || (b.is_known() && b.value == 0)) return Value::make(0);
    const uint64_t all_ones = is64 ? ~0ULL : 0xFFFFFFFFULL;
    if (b.is_known() && (static_cast<uint64_t>(b.value) & all_ones) == all_ones) return a;
    if (a.is_known() && (static_cast<uint64_t>(a.value) & all_ones) == all_ones) return b;
    if (a.is_known() && b.is_known()) return Value::make(a.value & b.value);
    return top();
}

ConstIntVal ConstIntDomain::bit_or(const Value& a, const Value& b, bool /*is64*/) {
    if (a.is_bottom() || b.is_bottom()) return bottom();
    if (b.is_known() && b.value == 0) return a;
    if (a.is_known() && a.value == 0) return b;
    if (a.is_known() && b.is_known()) return Value::make(a.value | b.value);
    return top();
}

ConstIntVal ConstIntDomain::bit_xor(const Value& a, const Value& b, bool /*is64*/) {
    if (a.is_bottom() || b.is_bottom()) return bottom();
    if (b.is_known() && b.value == 0) return a;
    if (a.is_known() && a.value == 0) return b;
    if (a.is_known() && b.is_known()) return Value::make(a.value ^ b.value);
    return top();
}

ConstIntVal ConstIntDomain::bit_not(const Value& a, bool is64) {
    if (a.is_bottom()) return bottom();
    if (a.is_known()) {
        uint64_t mask = is64 ? ~0ULL : 0xFFFFFFFFULL;
        return Value::make(static_cast<int64_t>((~static_cast<uint64_t>(a.value)) & mask));
    }
    return top();
}

ConstIntVal ConstIntDomain::shift(const Value& a, const Value& b, ShiftKind kind, bool is64) {
    if (a.is_bottom() || b.is_bottom()) return bottom();
    if (b.is_known() && b.value == 0) return a;
    if (a.is_known() && b.is_known()) {
        uint64_t ca = static_cast<uint64_t>(a.value);
        uint32_t sh = static_cast<uint32_t>(static_cast<uint64_t>(b.value) & (is64 ? 63u : 31u));
        uint64_t mask = is64 ? ~0ULL : 0xFFFFFFFFULL;
        uint64_t res = 0;
        switch (kind) {
            case ShiftKind::Shl:  res = (ca << sh) & mask; break;
            case ShiftKind::ShrU: res = (ca & mask) >> sh; break;
            case ShiftKind::ShrS:
                if (is64) res = static_cast<uint64_t>(static_cast<int64_t>(ca) >> sh);
                else res = static_cast<uint64_t>(static_cast<uint32_t>(static_cast<int32_t>(ca) >> sh));
                break;
            case ShiftKind::RotL: {
                uint32_t w = is64 ? 64u : 32u;
                res = (sh == 0) ? (ca & mask) : (((ca << sh) | ((ca & mask) >> (w - sh))) & mask);
                break;
            }
            case ShiftKind::RotR: {
                uint32_t w = is64 ? 64u : 32u;
                res = (sh == 0) ? (ca & mask) : ((((ca & mask) >> sh) | (ca << (w - sh))) & mask);
                break;
            }
        }
        return Value::make(static_cast<int64_t>(res));
    }
    return top();
}

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

PointerVal PointerDomain::widen_dynamic_offset(const Value& ptr) {
    if (!ptr.is_valid_pointer()) return top();
    return ptr.with_memory_loc(ptr.to_memory_loc().with_unknown_offset());
}

PointerVal PointerDomain::add(const Value& ptr, const ConstIntVal& delta, const SrtSymbolLookupFn& lookup_srt) {
    if (ptr.is_bottom() || delta.is_bottom()) return bottom();
    if (!ptr.is_valid_pointer()) return top();
    if (delta.is_known()) {
        if (delta.value == 0) return ptr;
        return apply_pointer_delta(ptr, delta.value, lookup_srt);
    }
    return widen_dynamic_offset(ptr);
}

PointerVal PointerDomain::add(const ConstIntVal& delta, const Value& ptr, const SrtSymbolLookupFn& lookup_srt) {
    return add(ptr, delta, lookup_srt);
}

PointerVal PointerDomain::sub(const Value& ptr, const ConstIntVal& delta, const SrtSymbolLookupFn& lookup_srt) {
    if (ptr.is_bottom() || delta.is_bottom()) return bottom();
    if (!ptr.is_valid_pointer()) return top();
    if (delta.is_known()) {
        if (delta.value == 0) return ptr;
        return apply_pointer_delta(ptr, -delta.value, lookup_srt);
    }
    return widen_dynamic_offset(ptr);
}

ConstIntVal PointerDomain::sub(const Value& a, const Value& b) {
    if (a.is_bottom() || b.is_bottom()) return ConstIntVal::make_bottom();
    if (a.is_valid_pointer() && b.is_valid_pointer() &&
        a.same_base_identity(b) && a.offset_known && b.offset_known) {
        return ConstIntVal::make(a.byte_offset - b.byte_offset);
    }
    return ConstIntVal::make_top();
}

PointerVal PointerDomain::bit_and(const Value& ptr, const ConstIntVal& mask, bool is64) {
    if (ptr.is_bottom() || mask.is_bottom()) return bottom();
    if (!ptr.is_valid_pointer() || !mask.is_known()) return top();
    const uint64_t all_ones = is64 ? ~0ULL : 0xFFFFFFFFULL;
    if ((static_cast<uint64_t>(mask.value) & all_ones) == all_ones) return ptr;
    if (mask.value == 0) return top();
    return ptr.with_memory_loc(ptr.to_memory_loc().with_masked_offset(mask.value));
}

PointerVal PointerDomain::bit_and(const ConstIntVal& mask, const Value& ptr, bool is64) {
    return bit_and(ptr, mask, is64);
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

// ShaderAtomVal & ShaderAtomDomain

ShaderAtomVal ShaderAtomVal::make_bottom() {
    ShaderAtomVal v;
    v.kind = Kind::Bottom;
    return v;
}

ShaderAtomVal ShaderAtomVal::make_top() {
    ShaderAtomVal v;
    v.kind = Kind::Top;
    return v;
}

ShaderAtomVal ShaderAtomVal::make_abi(std::string lbl) {
    ShaderAtomVal v;
    v.kind = Kind::AbiPreload;
    v.label = std::move(lbl);
    return v;
}

ShaderAtomVal ShaderAtomVal::make_sideband(uint32_t member_idx) {
    ShaderAtomVal v;
    v.kind = Kind::SidebandValue;
    v.location = member_idx;
    return v;
}

ShaderAtomVal ShaderAtomVal::make_buffer_size(int32_t r, int32_t d, std::string sym) {
    ShaderAtomVal v;
    v.kind = Kind::BufferSize;
    v.rack = r;
    v.desc_idx = d;
    v.symbol_name = std::move(sym);
    return v;
}

ShaderAtomVal ShaderAtomVal::make_fau_uniform(uint32_t foff, std::string sym) {
    ShaderAtomVal v;
    v.kind = Kind::FauUniformRef;
    v.fau_offset = foff;
    v.symbol_name = std::move(sym);
    return v;
}

ShaderAtomVal ShaderAtomVal::make_vertex_attr(uint32_t loc, std::string sym) {
    ShaderAtomVal v;
    v.kind = Kind::VertexAttrValue;
    v.location = loc;
    v.symbol_name = std::move(sym);
    return v;
}

ShaderAtomVal ShaderAtomVal::make_varying(uint32_t loc, std::string sym) {
    ShaderAtomVal v;
    v.kind = Kind::VaryingValue;
    v.location = loc;
    v.symbol_name = std::move(sym);
    return v;
}

ShaderAtomVal ShaderAtomVal::make_tex_sample(uint32_t img, uint32_t smp, std::string sym) {
    ShaderAtomVal v;
    v.kind = Kind::TexSampleValue;
    v.location = img;
    v.sampler_idx = smp;
    v.symbol_name = std::move(sym);
    return v;
}

ShaderAtomVal ShaderAtomVal::make_blend_desc(uint32_t loc, std::string sym) {
    ShaderAtomVal v;
    v.kind = Kind::BlendDescRef;
    v.location = loc;
    v.symbol_name = std::move(sym);
    return v;
}

bool ShaderAtomDomain::equals(const Value& a, const Value& b) {
    if (a.kind != b.kind) return false;
    if (a.is_bottom() || a.is_top()) return true;
    return a.rack == b.rack &&
           a.desc_idx == b.desc_idx &&
           a.fau_offset == b.fau_offset &&
           a.location == b.location &&
           a.sampler_idx == b.sampler_idx &&
           a.symbol_name == b.symbol_name &&
           a.label == b.label;
}

bool ShaderAtomDomain::leq(const Value& a, const Value& b) {
    if (a.is_bottom() || b.is_top()) return true;
    if (b.is_bottom() || a.is_top()) return false;
    return equals(a, b);
}

ShaderAtomVal ShaderAtomDomain::join(const Value& a, const Value& b) {
    if (a.is_bottom()) return b;
    if (b.is_bottom()) return a;
    if (a.is_top() || b.is_top()) return top();
    return equals(a, b) ? a : top();
}

ShaderAtomVal ShaderAtomDomain::meet(const Value& a, const Value& b) {
    if (a.is_top()) return b;
    if (b.is_top()) return a;
    if (a.is_bottom() || b.is_bottom()) return bottom();
    return equals(a, b) ? a : bottom();
}

std::string ShaderAtomDomain::format(const Value& v) {
    std::ostringstream oss;
    switch (v.kind) {
        case ShaderAtomVal::Kind::Bottom: return "<bottom>";
        case ShaderAtomVal::Kind::Top: return "<top>";
        case ShaderAtomVal::Kind::AbiPreload:
            return v.label.empty() ? "ABI" : v.label;
        case ShaderAtomVal::Kind::SidebandValue:
            oss << "sideband[" << v.location << "]";
            return oss.str();
        case ShaderAtomVal::Kind::BufferSize:
            if (!v.symbol_name.empty()) oss << "sizeof(" << v.symbol_name << ")";
            else oss << "sizeof(SRT[" << v.rack << "]->Buf[" << v.desc_idx << "])";
            return oss.str();
        case ShaderAtomVal::Kind::FauUniformRef:
            if (!v.symbol_name.empty()) return v.symbol_name;
            oss << "FAU+0x" << std::hex << v.fau_offset;
            return oss.str();
        case ShaderAtomVal::Kind::VertexAttrValue:
            oss << "attr[" << v.location << "]";
            if (!v.symbol_name.empty()) oss << " (" << v.symbol_name << ")";
            return oss.str();
        case ShaderAtomVal::Kind::VaryingValue:
            oss << "varying[" << v.location << "]";
            if (!v.symbol_name.empty()) oss << " (" << v.symbol_name << ")";
            return oss.str();
        case ShaderAtomVal::Kind::TexSampleValue:
            oss << "tex(img=" << v.location << ",smp=" << v.sampler_idx << ")";
            if (!v.symbol_name.empty()) oss << " (" << v.symbol_name << ")";
            return oss.str();
        case ShaderAtomVal::Kind::BlendDescRef:
            oss << "color_out[" << v.location << "]";
            if (!v.symbol_name.empty()) oss << " (" << v.symbol_name << ")";
            return oss.str();
    }
    return "<unknown-atom>";
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

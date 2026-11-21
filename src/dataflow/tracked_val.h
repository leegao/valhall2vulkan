#pragma once

#include "domains.h"
#include "memory_domain.h"

#include <cstdint>
#include <functional>
#include <string>

namespace valhall::dataflow {

using SrtSymbolLookupFn = std::function<std::string(uint8_t rack, uint32_t desc_idx)>;

// Integer Constant Sub-Domain (standard FlatDomain-backed IntegralDomain)

using ConstIntDomain = IntegralDomain<int64_t>;
using ConstIntVal = ConstIntDomain::Value;

// Pointer Sub-Domain (Base + Offset Symbolic GPU Address Lattice)

struct PointerVal {
    enum class Kind : uint8_t {
        Bottom = 0,
        Top,
        FauDstBase,      // Pilot shader FAU_0 destination base VA (+ optional byte offset)
        SrtBase,         // Shader Resource Table (SRT_0) base VA (+ optional byte offset)
        SrtRackPtr,      // Dereferenced SRT rack pointer (SRT[rack] + optional byte offset)
        DescriptorAddr,  // Address of descriptor `desc_idx` inside SRT[rack] (+ field offset)
        BufferAddr,      // Dereferenced 64-bit GPU buffer VA (`&sym` + optional byte offset)
        DataSectionPtr,  // PC-relative pointer into embedded EBIN `.data` / LUT section
        TlsPtr,          // Thread-local storage (TLS) scratch/stack pointer
        WlsPtr,          // Workgroup-local storage (WLS / shared memory) pointer
    };

    Kind kind = Kind::Bottom;
    int64_t byte_offset = 0;
    bool offset_known = true;
    int32_t rack = -1;
    int32_t desc_idx = -1;
    bool derived = false;
    std::string symbol_name;
    std::string label;

    static PointerVal make_bottom();
    static PointerVal make_top();
    static PointerVal make_fau_dst(int64_t off = 0, bool off_known = true);
    static PointerVal make_srt_base(int64_t off = 0, bool off_known = true);
    static PointerVal make_srt_rack(int32_t rack, int64_t off = 0, bool off_known = true);
    static PointerVal make_descriptor(int32_t rack, int32_t desc, std::string sym,
                                      int64_t field_off = 0, bool off_known = true);
    static PointerVal make_buffer_addr(int32_t rack, int32_t desc, std::string sym,
                                       int64_t off = 0, bool derived = false, bool off_known = true);
    static PointerVal make_data_section(int64_t off = 0, bool off_known = true);
    static PointerVal make_tls_ptr(int64_t off = 0, bool off_known = true);
    static PointerVal make_wls_ptr(int64_t off = 0, bool off_known = true);

    bool is_bottom() const { return kind == Kind::Bottom; }
    bool is_top() const { return kind == Kind::Top; }
    bool is_valid_pointer() const { return kind != Kind::Bottom && kind != Kind::Top; }
    bool same_base_identity(const PointerVal& other) const;

    SymbolicMemoryLoc to_memory_loc() const;
    PointerVal with_memory_loc(const SymbolicMemoryLoc& loc) const;
};

/// Symbolic GPU pointer lattice with height 3:
///   Bottom < (ptr + concrete_offset) < (ptr + Top_Z) < Top
///
/// Inherits commutative/associative integer-shift ops (`add`/`sub`/`iadd`/`isub`)
/// from `IntegralAdditiveActionMixin` and commutative/associative bitmasking
/// (`bit_and`) from `IntegralBitmaskActionMixin`.
struct PointerDomain
    : public IntegralAdditiveActionMixin<PointerDomain, PointerVal, ConstIntDomain>,
      public IntegralBitmaskActionMixin<PointerDomain, PointerVal, ConstIntDomain> {
    using Value = PointerVal;
    using AdditiveBase = IntegralAdditiveActionMixin<PointerDomain, PointerVal, ConstIntDomain>;
    using AdditiveBase::sub;
    using AdditiveBase::isub;

    static Value bottom() { return PointerVal::make_bottom(); }
    static Value top() { return PointerVal::make_top(); }

    static Value join(const Value& a, const Value& b);
    static Value meet(const Value& a, const Value& b);
    static bool leq(const Value& a, const Value& b);
    static bool equals(const Value& a, const Value& b);
    static std::string format(const Value& v);

    // Hooks required by IntegralAdditiveActionMixin & IntegralBitmaskActionMixin
    static Value apply_int_delta(const Value& ptr, int64_t delta,
                                 const SrtSymbolLookupFn& lookup_srt = {});

    /// Widen a concrete pointer offset `(ptr + delta)` up one lattice step to
    /// `(ptr + Top_Z)` (`offset_known = false`, formatted as `ptr+?`) while
    /// preserving base object identity in `Bottom < (ptr + delta) < (ptr + Top_Z) < Top`.
    static Value widen_unknown_offset(const Value& ptr);

    static Value apply_int_mask(const Value& ptr, int64_t mask);

    // Same-base pointer subtraction yields an exact ConstInt byte delta when both offsets are known.
    static ConstIntVal sub(const Value& a, const Value& b);
    static ConstIntVal isub(const Value& a, const Value& b, bool /*is64*/ = true) { return sub(a, b); }

    static Value convert(const Value& v, ConvertKind kind);
};

// Symbol Value Sub-Domain (Scalar/Struct Value Loaded from Buffer or .data)

struct SymbolVal {
    enum class State : uint8_t { Bottom = 0, Known, Top };

    State state = State::Bottom;
    int32_t rack = -1;
    int32_t desc_idx = -1;
    std::string symbol_name;
    int64_t byte_offset = 0;
    bool offset_known = false;
    bool derived = false;

    static SymbolVal make_bottom() { return SymbolVal{}; }
    static SymbolVal make_top() {
        SymbolVal v;
        v.state = State::Top;
        return v;
    }
    static SymbolVal make(int32_t r, int32_t d, std::string sym, bool drv = false,
                          int64_t off = 0, bool off_known = false) {
        SymbolVal v;
        v.state = State::Known;
        v.rack = r;
        v.desc_idx = d;
        v.symbol_name = std::move(sym);
        v.byte_offset = off;
        v.offset_known = off_known;
        v.derived = drv;
        return v;
    }

    bool is_bottom() const { return state == State::Bottom; }
    bool is_top() const { return state == State::Top; }
    bool same_identity(const SymbolVal& o) const {
        return state == o.state && rack == o.rack && desc_idx == o.desc_idx && symbol_name == o.symbol_name;
    }
    SymbolVal as_derived() const {
        SymbolVal out = *this;
        if (out.state == State::Known) out.derived = true;
        return out;
    }
};

/// Loaded buffer/data scalar value lattice with `derived` propagation.
/// Inherits `IntegralCombinableMixin` (excluding `PointerVal` so `SymbolVal + PointerVal`
/// flows to `PointerDomain` in `UnionDomain`).
struct SymbolValueDomain
    : public IntegralCombinableMixin<SymbolValueDomain, SymbolVal, ConstIntDomain, PointerVal> {
    using Value = SymbolVal;

    static Value bottom() { return Value::make_bottom(); }
    static Value top() { return Value::make_top(); }

    static Value join(const Value& a, const Value& b);
    static Value meet(const Value& a, const Value& b);
    static bool leq(const Value& a, const Value& b);
    static bool equals(const Value& a, const Value& b);
    static std::string format(const Value& v);

    static Value combine_with_scalar(const Value& s) { return s.as_derived(); }
    static Value convert(const Value& v, ConvertKind kind);
};

// Shader Atom Sub-Domain (Flat Lattice for ABI, Sideband, FAU & Shader I/O Atoms)

struct ShaderAtomPayload {
    enum class Kind : uint8_t {
        AbiPreload = 0,   // Live hardware ABI entry builtin (WorkgroupID, GlobalInvocationID, ...)
        SidebandValue,    // Member of synthetic `ValhallSidebandData` UBO (set=0, binding=0) carrying
                          // CSF/SPD runtime params (shader_pc_base, srt_base_va, pilot_fau_dst_va,
                          // blend_descs, local_size_xyz, fb_info) not covered by SPIR-V builtins
        BufferSize,       // Buffer descriptor byte-size field (`sizeof(sym)`)
        FauUniformRef,    // Value loaded from FAU uniform slot `fau_offset`
        VertexAttrValue,  // Loaded vertex attribute value (`location`)
        VaryingValue,     // Loaded fragment varying value (`location`)
        TexSampleValue,   // Result of texture sample (`location = img`, `sampler_idx`)
        BlendDescRef,     // Fragment blend / color output target (`location`)
    };

    Kind kind = Kind::AbiPreload;
    int32_t rack = -1;
    int32_t desc_idx = -1;
    uint32_t fau_offset = 0;
    uint32_t location = 0;
    uint32_t sampler_idx = 0;
    std::string symbol_name;
    std::string label;

    bool operator==(const ShaderAtomPayload& o) const {
        return kind == o.kind &&
               rack == o.rack &&
               desc_idx == o.desc_idx &&
               fau_offset == o.fau_offset &&
               location == o.location &&
               sampler_idx == o.sampler_idx &&
               symbol_name == o.symbol_name &&
               label == o.label;
    }
};

using ShaderAtomVal = FlatValue<ShaderAtomPayload>;

struct ShaderAtomDomain : public FlatDomain<ShaderAtomPayload> {
    using Base = FlatDomain<ShaderAtomPayload>;
    using Value = Base::Value;
    using Kind = ShaderAtomPayload::Kind;

    static Value make_abi(std::string lbl);
    static Value make_sideband(uint32_t member_idx);
    static Value make_buffer_size(int32_t rack, int32_t desc, std::string sym);
    static Value make_fau_uniform(uint32_t fau_off, std::string sym = "");
    static Value make_vertex_attr(uint32_t loc, std::string sym = "");
    static Value make_varying(uint32_t loc, std::string sym = "");
    static Value make_tex_sample(uint32_t img_idx, uint32_t smp_idx, std::string sym = "");
    static Value make_blend_desc(uint32_t loc, std::string sym = "");

    static std::string format(const Value& v);
    static Value convert(const Value& v, ConvertKind kind);
};

// Composed TrackedValDomain = LaneDomainWrapper<UnionDomain<ConstInt, Pointer, Symbol, ShaderAtom>>

using TrackedValUnionDomain = UnionDomain<
    ConstIntDomain,
    PointerDomain,
    SymbolValueDomain,
    ShaderAtomDomain>;

/// Complete lattice and algebraic domain for SSA-level Valhall pointer & resource
/// analysis. All lattice ops, arithmetic/logical/convert ops, and 64-bit lane ops
/// (`pack64`, `split64`, `tuple_extract`) are inherited automatically from
/// `LaneDomainWrapper<UnionDomain<...>>`.
struct TrackedValDomain : public LaneDomainWrapper<TrackedValUnionDomain> {
    using Base = LaneDomainWrapper<TrackedValUnionDomain>;
    using Value = Base::Value;

    static Value make_bottom() { return bottom(); }
    static Value make_top() { return top(); }
    static Value make_const(int64_t v) { return Value(ConstIntVal::make(v)); }
    static Value make_abi(std::string lbl) { return Value(ShaderAtomDomain::make_abi(std::move(lbl))); }
    static Value make_sideband(uint32_t member_idx) { return Value(ShaderAtomDomain::make_sideband(member_idx)); }
    static Value make_fau_dst(int64_t off = 0, bool off_known = true) {
        return Value(PointerVal::make_fau_dst(off, off_known));
    }
    static Value make_srt_base(int64_t off = 0, bool off_known = true) {
        return Value(PointerVal::make_srt_base(off, off_known));
    }
    static Value make_srt_rack(int32_t rack, int64_t off = 0, bool off_known = true) {
        return Value(PointerVal::make_srt_rack(rack, off, off_known));
    }
    static Value make_descriptor(int32_t rack, int32_t desc, std::string sym,
                                 int64_t field_off = 0, bool off_known = true) {
        return Value(PointerVal::make_descriptor(rack, desc, std::move(sym), field_off, off_known));
    }
    static Value make_buffer_addr(int32_t rack, int32_t desc, std::string sym,
                                  int64_t off = 0, bool derived = false, bool off_known = true) {
        return Value(PointerVal::make_buffer_addr(rack, desc, std::move(sym), off, derived, off_known));
    }
    static Value make_buffer_size(int32_t rack, int32_t desc, std::string sym) {
        return Value(ShaderAtomDomain::make_buffer_size(rack, desc, std::move(sym)));
    }
    static Value make_fau_uniform(uint32_t fau_off, std::string sym = "") {
        return Value(ShaderAtomDomain::make_fau_uniform(fau_off, std::move(sym)));
    }
    static Value make_symbol_value(int32_t rack, int32_t desc, std::string sym, bool derived = false,
                                   int64_t off = 0, bool off_known = false) {
        return Value(SymbolVal::make(rack, desc, std::move(sym), derived, off, off_known));
    }
    static Value make_data_section(int64_t off = 0, bool off_known = true) {
        return Value(PointerVal::make_data_section(off, off_known));
    }
    static Value make_tls_ptr(int64_t off = 0, bool off_known = true) {
        return Value(PointerVal::make_tls_ptr(off, off_known));
    }
    static Value make_wls_ptr(int64_t off = 0, bool off_known = true) {
        return Value(PointerVal::make_wls_ptr(off, off_known));
    }
    static Value make_vertex_attr(uint32_t loc, std::string sym = "") {
        return Value(ShaderAtomDomain::make_vertex_attr(loc, std::move(sym)));
    }
    static Value make_varying(uint32_t loc, std::string sym = "") {
        return Value(ShaderAtomDomain::make_varying(loc, std::move(sym)));
    }
    static Value make_tex_sample(uint32_t img_idx, uint32_t smp_idx, std::string sym = "") {
        return Value(ShaderAtomDomain::make_tex_sample(img_idx, smp_idx, std::move(sym)));
    }
    static Value make_blend_desc(uint32_t loc, std::string sym = "") {
        return Value(ShaderAtomDomain::make_blend_desc(loc, std::move(sym)));
    }
};

using TrackedVal = TrackedValDomain::Value;

/// Bounded finite-set domain over `TrackedVal` (tracks up to 4 distinct concrete
/// values before widening to Top). Automatically inherits all pointwise lifted
/// arithmetic, bitwise, shift, convert, and 64-bit lane operations.
using TrackedValSetDomain = FiniteSetDomain<TrackedValDomain, 4>;
using TrackedValSet = TrackedValSetDomain::Value;

/// Format a single `TrackedVal` concisely (e.g. "&_83_input_0+0x10 (SRT[4]->Buf[0].addr)").
std::string format_tracked_val(const TrackedVal& v);

/// Format a single `TrackedVal` with its active union-domain tag for debugging
/// (e.g. "[Pointer] &_83_input_0+0x10" or "[ConstInt] 0x10").
std::string format_tracked_val_debug(const TrackedVal& v);

/// Format a `TrackedValSet` concisely (`<bottom>`, `<top>`, single item, or `{a, b}`).
std::string format_tracked_val_set(const TrackedValSet& s);

/// Collapse a `TrackedValSet` down to a single `TrackedVal` via `TrackedValDomain::join`.
TrackedVal collapse_tracked_val_set(const TrackedValSet& s);

}  // namespace valhall::dataflow

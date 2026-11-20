#pragma once

#include "domains.h"
#include "memory_domain.h"

#include <cstdint>
#include <functional>
#include <string>
#include <type_traits>

namespace valhall::dataflow {

using SrtSymbolLookupFn = std::function<std::string(uint8_t rack, uint32_t desc_idx)>;

// Sub-Domain 1: ConstIntDomain — Flat 64-Bit Integer Constant Lattice

struct ConstIntVal {
    enum class State : uint8_t { Bottom = 0, Known, Top };

    State state = State::Bottom;
    int64_t value = 0;

    static ConstIntVal make_bottom() { return ConstIntVal{State::Bottom, 0}; }
    static ConstIntVal make_top() { return ConstIntVal{State::Top, 0}; }
    static ConstIntVal make(int64_t v) { return ConstIntVal{State::Known, v}; }

    bool is_bottom() const { return state == State::Bottom; }
    bool is_top() const { return state == State::Top; }
    bool is_known() const { return state == State::Known; }
};

struct ConstIntDomain {
    using Value = ConstIntVal;

    static Value bottom() { return Value::make_bottom(); }
    static Value top() { return Value::make_top(); }

    static Value join(const Value& a, const Value& b);
    static Value meet(const Value& a, const Value& b);
    static bool leq(const Value& a, const Value& b);
    static bool equals(const Value& a, const Value& b);
    static std::string format(const Value& v);

    static bool try_as_const_u64(const Value& v, uint64_t* out);
    static Value make_const_u64(uint64_t v);

    static Value add(const Value& a, const Value& b);
    static Value sub(const Value& a, const Value& b);
    static Value mul(const Value& a, const Value& b);
    static Value div(const Value& a, const Value& b);

    static Value iadd(const Value& a, const Value& b, bool /*is64*/ = true) { return add(a, b); }
    static Value isub(const Value& a, const Value& b, bool /*is64*/ = true) { return sub(a, b); }
    static Value imul(const Value& a, const Value& b, bool /*is64*/ = true) { return mul(a, b); }
    static Value idiv(const Value& a, const Value& b, bool /*is64*/ = true) { return div(a, b); }

    static Value bit_and(const Value& a, const Value& b, bool is64 = true);
    static Value bit_or(const Value& a, const Value& b, bool is64 = true);
    static Value bit_xor(const Value& a, const Value& b, bool is64 = true);
    static Value bit_not(const Value& a, bool is64 = true);
    static Value shift(const Value& a, const Value& b, ShiftKind kind, bool is64 = true);
};

// Sub-Domain 2: PointerDomain — Base + Offset Symbolic GPU Address Lattice

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

struct PointerDomain {
    using Value = PointerVal;

    static Value bottom() { return PointerVal::make_bottom(); }
    static Value top() { return PointerVal::make_top(); }

    static Value join(const Value& a, const Value& b);
    static Value meet(const Value& a, const Value& b);
    static bool leq(const Value& a, const Value& b);
    static bool equals(const Value& a, const Value& b);
    static std::string format(const Value& v);

    /// Widen a pointer's byte offset to unknown (`+?`) while keeping its base
    /// object identity intact (used when adding/subtracting a dynamic runtime
    /// scalar such as `gl_GlobalInvocationID.x * 16` or `Top`).
    static Value widen_dynamic_offset(const Value& ptr);

    // Exact pointer +/- constant-integer arithmetic (and commutative int + ptr).
    static Value add(const Value& ptr, const ConstIntVal& delta, const SrtSymbolLookupFn& lookup_srt = {});
    static Value add(const ConstIntVal& delta, const Value& ptr, const SrtSymbolLookupFn& lookup_srt = {});
    static Value sub(const Value& ptr, const ConstIntVal& delta, const SrtSymbolLookupFn& lookup_srt = {});

    // Pointer + dynamic non-pointer offset -> preserve base identity, widen offset to `+?`.
    // Adding two pointers (`PointerVal + PointerVal`) is intentionally excluded and yields Top.
    template <typename OtherVal,
              std::enable_if_t<!std::is_same_v<std::decay_t<OtherVal>, Value> &&
                               !std::is_same_v<std::decay_t<OtherVal>, ConstIntVal>, int> = 0>
    static Value add(const Value& ptr, const OtherVal&) {
        return widen_dynamic_offset(ptr);
    }

    template <typename OtherVal,
              std::enable_if_t<!std::is_same_v<std::decay_t<OtherVal>, Value> &&
                               !std::is_same_v<std::decay_t<OtherVal>, ConstIntVal>, int> = 0>
    static Value add(const OtherVal&, const Value& ptr) {
        return widen_dynamic_offset(ptr);
    }

    template <typename OtherVal,
              std::enable_if_t<!std::is_same_v<std::decay_t<OtherVal>, Value> &&
                               !std::is_same_v<std::decay_t<OtherVal>, ConstIntVal>, int> = 0>
    static Value sub(const Value& ptr, const OtherVal&) {
        return widen_dynamic_offset(ptr);
    }

    // Same-base pointer subtraction yields an exact ConstInt byte delta when both offsets are known.
    static ConstIntVal sub(const Value& a, const Value& b);

    // Normalized `iadd` / `isub` overloads forwarding to `add` / `sub`.
    static Value iadd(const Value& ptr, const ConstIntVal& delta, bool /*is64*/ = true) {
        return add(ptr, delta);
    }
    static Value iadd(const ConstIntVal& delta, const Value& ptr, bool /*is64*/ = true) {
        return add(delta, ptr);
    }
    static Value isub(const Value& ptr, const ConstIntVal& delta, bool /*is64*/ = true) {
        return sub(ptr, delta);
    }
    static ConstIntVal isub(const Value& a, const Value& b, bool /*is64*/ = true) {
        return sub(a, b);
    }

    template <typename OtherVal,
              std::enable_if_t<!std::is_same_v<std::decay_t<OtherVal>, Value> &&
                               !std::is_same_v<std::decay_t<OtherVal>, ConstIntVal>, int> = 0>
    static Value iadd(const Value& ptr, const OtherVal&, bool /*is64*/ = true) {
        return widen_dynamic_offset(ptr);
    }

    template <typename OtherVal,
              std::enable_if_t<!std::is_same_v<std::decay_t<OtherVal>, Value> &&
                               !std::is_same_v<std::decay_t<OtherVal>, ConstIntVal>, int> = 0>
    static Value iadd(const OtherVal&, const Value& ptr, bool /*is64*/ = true) {
        return widen_dynamic_offset(ptr);
    }

    template <typename OtherVal,
              std::enable_if_t<!std::is_same_v<std::decay_t<OtherVal>, Value> &&
                               !std::is_same_v<std::decay_t<OtherVal>, ConstIntVal>, int> = 0>
    static Value isub(const Value& ptr, const OtherVal&, bool /*is64*/ = true) {
        return widen_dynamic_offset(ptr);
    }

    // Pointer alignment masking (`ptr & const_mask` / `const_mask & ptr`).
    static Value bit_and(const Value& ptr, const ConstIntVal& mask, bool is64 = true);
    static Value bit_and(const ConstIntVal& mask, const Value& ptr, bool is64 = true);
};

// Sub-Domain 3: SymbolValueDomain — Scalar/Struct Value Loaded from Buffer/.data

struct SymbolVal {
    enum class State : uint8_t { Bottom = 0, Known, Top };

    State state = State::Bottom;
    int32_t rack = -1;
    int32_t desc_idx = -1;
    std::string symbol_name;
    bool derived = false;

    static SymbolVal make_bottom() { return SymbolVal{}; }
    static SymbolVal make_top() {
        SymbolVal v;
        v.state = State::Top;
        return v;
    }
    static SymbolVal make(int32_t r, int32_t d, std::string sym, bool drv = false) {
        SymbolVal v;
        v.state = State::Known;
        v.rack = r;
        v.desc_idx = d;
        v.symbol_name = std::move(sym);
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

struct SymbolValueDomain {
    using Value = SymbolVal;

    static Value bottom() { return Value::make_bottom(); }
    static Value top() { return Value::make_top(); }

    static Value join(const Value& a, const Value& b);
    static Value meet(const Value& a, const Value& b);
    static bool leq(const Value& a, const Value& b);
    static bool equals(const Value& a, const Value& b);
    static std::string format(const Value& v);

    // Arithmetic on a loaded symbol value marks it `derived(sym)`.
    // Excludes `PointerVal` so `SymbolVal + PointerVal` falls through to `PointerDomain`
    // (preserving the pointer base with dynamic offset).
    template <typename OtherVal,
              std::enable_if_t<!std::is_same_v<std::decay_t<OtherVal>, PointerVal>, int> = 0>
    static Value add(const Value& s, const OtherVal&) { return s.as_derived(); }
    template <typename OtherVal,
              std::enable_if_t<!std::is_same_v<std::decay_t<OtherVal>, PointerVal> &&
                               !std::is_same_v<std::decay_t<OtherVal>, Value>, int> = 0>
    static Value add(const OtherVal&, const Value& s) { return s.as_derived(); }

    template <typename OtherVal,
              std::enable_if_t<!std::is_same_v<std::decay_t<OtherVal>, PointerVal>, int> = 0>
    static Value sub(const Value& s, const OtherVal&) { return s.as_derived(); }

    template <typename OtherVal>
    static Value mul(const Value& s, const OtherVal&) { return s.as_derived(); }
    template <typename OtherVal,
              std::enable_if_t<!std::is_same_v<std::decay_t<OtherVal>, Value>, int> = 0>
    static Value mul(const OtherVal&, const Value& s) { return s.as_derived(); }

    template <typename OtherVal>
    static Value div(const Value& s, const OtherVal&) { return s.as_derived(); }

    template <typename OtherVal,
              std::enable_if_t<!std::is_same_v<std::decay_t<OtherVal>, PointerVal>, int> = 0>
    static Value iadd(const Value& s, const OtherVal&, bool /*is64*/ = true) { return s.as_derived(); }
    template <typename OtherVal,
              std::enable_if_t<!std::is_same_v<std::decay_t<OtherVal>, PointerVal> &&
                               !std::is_same_v<std::decay_t<OtherVal>, Value>, int> = 0>
    static Value iadd(const OtherVal&, const Value& s, bool /*is64*/ = true) { return s.as_derived(); }

    template <typename OtherVal,
              std::enable_if_t<!std::is_same_v<std::decay_t<OtherVal>, PointerVal>, int> = 0>
    static Value isub(const Value& s, const OtherVal&, bool /*is64*/ = true) { return s.as_derived(); }

    template <typename OtherVal>
    static Value imul(const Value& s, const OtherVal&, bool /*is64*/ = true) { return s.as_derived(); }
    template <typename OtherVal,
              std::enable_if_t<!std::is_same_v<std::decay_t<OtherVal>, Value>, int> = 0>
    static Value imul(const OtherVal&, const Value& s, bool /*is64*/ = true) { return s.as_derived(); }

    template <typename OtherVal>
    static Value idiv(const Value& s, const OtherVal&, bool /*is64*/ = true) { return s.as_derived(); }
};

// Sub-Domain 4: ShaderAtomDomain — Flat Lattice for ABI, Sideband, FAU & Shader I/O Atoms

struct ShaderAtomVal {
    enum class Kind : uint8_t {
        Bottom = 0,
        Top,
        AbiPreload,       // Live hardware ABI entry builtin (WorkgroupID, GlobalInvocationID, ...)
        SidebandValue,    // Symbolic sideband struct member (`sideband[member_idx]`)
        BufferSize,       // Buffer descriptor byte-size field (`sizeof(sym)`)
        FauUniformRef,    // Value loaded from FAU uniform slot `fau_offset`
        VertexAttrValue,  // Loaded vertex attribute value (`location`)
        VaryingValue,     // Loaded fragment varying value (`location`)
        TexSampleValue,   // Result of texture sample (`location = img`, `sampler_idx`)
        BlendDescRef,     // Fragment blend / color output target (`location`)
    };

    Kind kind = Kind::Bottom;
    int32_t rack = -1;
    int32_t desc_idx = -1;
    uint32_t fau_offset = 0;
    uint32_t location = 0;
    uint32_t sampler_idx = 0;
    std::string symbol_name;
    std::string label;

    static ShaderAtomVal make_bottom();
    static ShaderAtomVal make_top();
    static ShaderAtomVal make_abi(std::string lbl);
    static ShaderAtomVal make_sideband(uint32_t member_idx);
    static ShaderAtomVal make_buffer_size(int32_t rack, int32_t desc, std::string sym);
    static ShaderAtomVal make_fau_uniform(uint32_t fau_off, std::string sym = "");
    static ShaderAtomVal make_vertex_attr(uint32_t loc, std::string sym = "");
    static ShaderAtomVal make_varying(uint32_t loc, std::string sym = "");
    static ShaderAtomVal make_tex_sample(uint32_t img_idx, uint32_t smp_idx, std::string sym = "");
    static ShaderAtomVal make_blend_desc(uint32_t loc, std::string sym = "");

    bool is_bottom() const { return kind == Kind::Bottom; }
    bool is_top() const { return kind == Kind::Top; }
};

struct ShaderAtomDomain {
    using Value = ShaderAtomVal;

    static Value bottom() { return ShaderAtomVal::make_bottom(); }
    static Value top() { return ShaderAtomVal::make_top(); }

    static Value join(const Value& a, const Value& b);
    static Value meet(const Value& a, const Value& b);
    static bool leq(const Value& a, const Value& b);
    static bool equals(const Value& a, const Value& b);
    static std::string format(const Value& v);
};

// Composed TrackedValDomain = LaneDomainWrapper<UnionDomain<ConstInt, Pointer, Symbol, ShaderAtom>>

using TrackedValUnionDomain = UnionDomain<
    ConstIntDomain,
    PointerDomain,
    SymbolValueDomain,
    ShaderAtomDomain>;

/// Complete lattice and algebraic domain for SSA-level Valhall pointer & resource
/// analysis. All lattice ops, arithmetic/logical ops, and 64-bit lane ops
/// (`pack64`, `split64`, `tuple_extract`) are inherited automatically from
/// `LaneDomainWrapper<UnionDomain<...>>`.
struct TrackedValDomain : public LaneDomainWrapper<TrackedValUnionDomain> {
    using Base = LaneDomainWrapper<TrackedValUnionDomain>;
    using Value = Base::Value;

    static Value make_bottom() { return bottom(); }
    static Value make_top() { return top(); }
    static Value make_const(int64_t v) { return Value(ConstIntVal::make(v)); }
    static Value make_abi(std::string lbl) { return Value(ShaderAtomVal::make_abi(std::move(lbl))); }
    static Value make_sideband(uint32_t member_idx) { return Value(ShaderAtomVal::make_sideband(member_idx)); }
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
        return Value(ShaderAtomVal::make_buffer_size(rack, desc, std::move(sym)));
    }
    static Value make_fau_uniform(uint32_t fau_off, std::string sym = "") {
        return Value(ShaderAtomVal::make_fau_uniform(fau_off, std::move(sym)));
    }
    static Value make_symbol_value(int32_t rack, int32_t desc, std::string sym, bool derived = false) {
        return Value(SymbolVal::make(rack, desc, std::move(sym), derived));
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
        return Value(ShaderAtomVal::make_vertex_attr(loc, std::move(sym)));
    }
    static Value make_varying(uint32_t loc, std::string sym = "") {
        return Value(ShaderAtomVal::make_varying(loc, std::move(sym)));
    }
    static Value make_tex_sample(uint32_t img_idx, uint32_t smp_idx, std::string sym = "") {
        return Value(ShaderAtomVal::make_tex_sample(img_idx, smp_idx, std::move(sym)));
    }
    static Value make_blend_desc(uint32_t loc, std::string sym = "") {
        return Value(ShaderAtomVal::make_blend_desc(loc, std::move(sym)));
    }
};

using TrackedVal = TrackedValDomain::Value;

/// Bounded finite-set domain over `TrackedVal` (tracks up to 4 distinct concrete
/// values before widening to Top). Automatically inherits all pointwise lifted
/// arithmetic, bitwise, shift, and 64-bit lane operations from `FiniteSetDomain`.
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

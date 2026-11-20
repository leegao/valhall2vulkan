#pragma once

#include "domains.h"
#include "memory_domain.h"

#include <cstdint>
#include <functional>
#include <string>

namespace valhall::dataflow {

/// Broadened abstract pointer / resource / ABI value domain for SSA-level
/// Valhall dataflow analysis.
///
/// Forms a complete lattice / CPO (L, <=, join, meet, Bottom, Top) of finite
/// height h = 3, guaranteeing monotone fixpoint termination on all CFGs:
///   Bottom < Concrete(Kind, id, known_offset) < Symbolic(Kind, id, unknown_offset) < Top
struct TrackedVal {
    enum class Kind : uint8_t {
        Bottom = 0,          // Unreachable / uninitialized
        Top,                 // Unknown / over-approximated value
        ConstInt,            // Known 64-bit integer constant (or raw 32-bit bit pattern)
        AbiPreload,          // Live hardware ABI entry builtin (WorkgroupID, GlobalInvocationID, VertexIndex, ...)
        SidebandValue,       // Symbolic sideband struct member (`sideband[member_idx]`)
        FauDstBase,          // Pilot shader FAU_0 destination base VA (+ optional byte offset)
        SrtBase,             // Shader Resource Table (SRT_0) base VA (+ optional byte offset)
        SrtRackPtr,          // Dereferenced SRT rack pointer (SRT[rack] + optional descriptor byte offset)
        DescriptorAddr,      // Address of descriptor `desc_idx` inside SRT rack `rack` (+ optional field offset)
        BufferAddr,          // Dereferenced 64-bit GPU buffer VA (`&sym` + optional byte offset)
        BufferSize,          // Buffer descriptor byte-size field (`sizeof(sym)`)
        FauUniformRef,       // Value loaded from FAU uniform slot `fau_offset`
        SymbolValue,         // Scalar/struct value loaded from UBO/SSBO symbol (`sym.member` / derived)
        DataSectionPtr,      // PC-relative pointer into embedded EBIN `.data` / LUT section
        TlsPtr,              // Thread-local storage (TLS) scratch/stack pointer
        WlsPtr,              // Workgroup-local storage (WLS / shared memory) pointer
        VertexAttrValue,     // Loaded vertex attribute value (location = `location`)
        VaryingValue,        // Loaded fragment varying value (location = `location`)
        TexSampleValue,      // Result of texture sample (`image_idx`, `sampler_idx`)
        BlendDescRef,        // Fragment blend / color output target (`location`)
    };

    Kind kind = Kind::Bottom;
    int64_t const_val = 0;         // For ConstInt
    int64_t byte_offset = 0;       // Byte offset relative to base (when offset_known == true)
    bool offset_known = true;      // False if widened across differing offsets on same base
    int32_t rack = -1;             // SRT rack index (0..N)
    int32_t desc_idx = -1;         // Descriptor index within rack (0..N)
    uint32_t fau_offset = 0;       // Byte offset inside FAU_0
    uint32_t location = 0;         // Vertex attr / varying / color output / image / sideband member index
    uint32_t sampler_idx = 0;      // Sampler index for TexSampleValue
    bool derived = false;          // True if value was transformed by ALU/arithmetic
    bool is_hi_half = false;       // True if upper 32-bit half of a split 64-bit pointer
    std::string symbol_name;       // Resolved MBS2 symbol or member name (e.g. "_83_input_0")
    std::string label;             // Human-readable label (e.g. "gl_GlobalInvocationID.x")

    // Convenience constructors
    static TrackedVal make_bottom();
    static TrackedVal make_top();
    static TrackedVal make_const(int64_t v);
    static TrackedVal make_abi(std::string lbl);
    static TrackedVal make_sideband(uint32_t member_idx);
    static TrackedVal make_fau_dst(int64_t off = 0, bool off_known = true);
    static TrackedVal make_srt_base(int64_t off = 0, bool off_known = true);
    static TrackedVal make_srt_rack(int32_t rack, int64_t off = 0, bool off_known = true);
    static TrackedVal make_descriptor(int32_t rack, int32_t desc, std::string sym,
                                      int64_t field_off = 0, bool off_known = true);
    static TrackedVal make_buffer_addr(int32_t rack, int32_t desc, std::string sym,
                                       int64_t off = 0, bool derived = false, bool off_known = true);
    static TrackedVal make_buffer_size(int32_t rack, int32_t desc, std::string sym);
    static TrackedVal make_fau_uniform(uint32_t fau_off, std::string sym = "");
    static TrackedVal make_symbol_value(int32_t rack, int32_t desc, std::string sym, bool derived = false);
    static TrackedVal make_data_section(int64_t off = 0, bool off_known = true);
    static TrackedVal make_tls_ptr(int64_t off = 0, bool off_known = true);
    static TrackedVal make_wls_ptr(int64_t off = 0, bool off_known = true);
    static TrackedVal make_vertex_attr(uint32_t loc, std::string sym = "");
    static TrackedVal make_varying(uint32_t loc, std::string sym = "");
    static TrackedVal make_tex_sample(uint32_t img_idx, uint32_t smp_idx, std::string sym = "");
    static TrackedVal make_blend_desc(uint32_t loc, std::string sym = "");

    // Member predicates & helpers
    bool is_bottom() const { return kind == Kind::Bottom; }
    bool is_top() const { return kind == Kind::Top; }
    bool has_offset() const;
    bool is_pointer() const { return has_offset(); }
    bool same_base_identity(const TrackedVal& other) const;

    /// Convert a pointer-like `TrackedVal` into its corresponding `SymbolicMemoryLoc`.
    SymbolicMemoryLoc to_memory_loc() const;

    /// Apply a `SymbolicMemoryLoc` offset/widening transformation back onto this pointer `TrackedVal`.
    TrackedVal with_memory_loc(const SymbolicMemoryLoc& loc) const;

    /// Concise human-readable description (e.g. "&_83_input_0+0x10 (SRT[4]->Buf[0].addr)").
    std::string format_short() const;
};

using SrtSymbolLookupFn = std::function<std::string(uint8_t rack, uint32_t desc_idx)>;

/// Complete lattice traits and pointwise algebraic operations for `TrackedVal`.
struct TrackedValDomain {
    using Value = TrackedVal;

    static Value bottom() { return TrackedVal::make_bottom(); }
    static Value top() { return TrackedVal::make_top(); }

    static Value join(const Value& a, const Value& b);
    static Value meet(const Value& a, const Value& b);
    static bool leq(const Value& a, const Value& b);
    static bool equals(const Value& a, const Value& b);

    static bool try_as_const_u64(const Value& v, uint64_t* out);
    static Value make_const_u64(uint64_t v);

    // Pointwise abstract arithmetic, logical & lane operations (delegating pointer offsets to SymbolicMemoryLoc)
    static Value add(const Value& a, const Value& b, const SrtSymbolLookupFn& lookup_srt = {});
    static Value sub(const Value& a, const Value& b, const SrtSymbolLookupFn& lookup_srt = {});
    static Value mul(const Value& a, const Value& b);
    static Value div(const Value& a, const Value& b);
    static Value bit_and(const Value& a, const Value& b);

    // Optional domain-trait signatures automatically flowed by SsaDomainTransferVisitor & FiniteSetDomain
    static Value iadd(const Value& a, const Value& b, bool /*is64*/) { return add(a, b); }
    static Value isub(const Value& a, const Value& b, bool /*is64*/) { return sub(a, b); }
    static Value imul(const Value& a, const Value& b, bool /*is64*/) { return mul(a, b); }
    static Value bit_and(const Value& a, const Value& b, bool is64);
    static Value bit_or(const Value& a, const Value& b, bool is64);
    static Value bit_xor(const Value& a, const Value& b, bool is64);
    static Value bit_not(const Value& a, bool is64);
    static Value shift(const Value& a, const Value& b, ShiftKind kind, bool is64);
    static Value pack64(const Value& lo, const Value& hi);
    static Value split64(const Value& src64) { return src64; }
    static Value tuple_extract(const Value& tup, uint32_t elem_idx);
};

/// Bounded finite-set domain over `TrackedVal` (tracks up to 4 distinct concrete
/// values before widening to Top).
using TrackedValSetDomain = FiniteSetDomain<TrackedValDomain, 4>;
using TrackedValSet = TrackedValSetDomain::Value;

/// Lifted abstract arithmetic & bitwise operations over `TrackedValSet`.
struct TrackedValSetOps {
    static TrackedValSet add(const TrackedValSet& a, const TrackedValSet& b,
                             const SrtSymbolLookupFn& lookup_srt = {});
    static TrackedValSet sub(const TrackedValSet& a, const TrackedValSet& b,
                             const SrtSymbolLookupFn& lookup_srt = {});
    static TrackedValSet mul(const TrackedValSet& a, const TrackedValSet& b);
    static TrackedValSet div(const TrackedValSet& a, const TrackedValSet& b);
    static TrackedValSet bit_and(const TrackedValSet& a, const TrackedValSet& b);
};

/// Format a `TrackedValSet` concisely (`<bottom>`, `<top>`, single item, or `{a, b}`).
std::string format_tracked_val_set(const TrackedValSet& s);

/// Collapse a `TrackedValSet` down to a single `TrackedVal` via `TrackedValDomain::join`.
TrackedVal collapse_tracked_val_set(const TrackedValSet& s);

}  // namespace valhall::dataflow

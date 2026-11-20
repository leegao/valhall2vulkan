#pragma once

#include <cstdint>
#include <functional>
#include <sstream>
#include <string>
#include <tuple>

namespace valhall::dataflow {

/// Distinct GPU memory address spaces / regions modeled by symbolic memory.
///
/// Disjointness assumptions:
///   - Distinct `MemoryRegionKind`s are disjoint memory spaces.
///   - Within `SrtTable`, distinct `table_id` (rack indices) are disjoint.
///   - Within `Buffer`, distinct `(table_id, entry_id)` bindings are disjoint.
///   - `FauUniform` and `DataSection` are read-only (RO) in Main shaders.
enum class MemoryRegionKind : uint8_t {
    FauUniform = 0,   // FAU_0 uniform bank (RO in Main shaders; written in Pilot shaders)
    SrtTable,         // Shader Resource Table (SRT_0[table_id])
    Buffer,           // UBO / SSBO buffer descriptor (SRT[table_id]->Desc[entry_id])
    DataSection,      // Embedded read-only EBIN .data / LUT section
    Tls,              // Per-thread scratch / spill stack (thread_local_pointer)
    Wls,              // Workgroup-local shared memory (workgroup_local_pointer)
    Other,            // Unclassified symbolic memory location
};

/// Structured symbolic memory location with pointer-offset arithmetic,
/// widening, hashing, ordering, and region-aware alias queries.
struct SymbolicMemoryLoc {
    MemoryRegionKind region = MemoryRegionKind::Other;
    int32_t table_id = -1;      // e.g. SRT rack index (0..N)
    int32_t entry_id = -1;      // e.g. descriptor index within rack (0..N)
    int64_t byte_offset = 0;    // Byte offset within region (valid when offset_known)
    bool offset_known = true;   // False if dynamic / widened offset
    std::string key;            // Optional human-readable label (e.g. symbol name)

    static SymbolicMemoryLoc make_fau(int64_t byte_off = 0, bool off_known = true) {
        SymbolicMemoryLoc loc;
        loc.region = MemoryRegionKind::FauUniform;
        loc.byte_offset = off_known ? byte_off : 0;
        loc.offset_known = off_known;
        return loc;
    }

    static SymbolicMemoryLoc make_srt(int32_t rack, int64_t off = 0, bool off_known = true) {
        SymbolicMemoryLoc loc;
        loc.region = MemoryRegionKind::SrtTable;
        loc.table_id = rack;
        loc.byte_offset = off_known ? off : 0;
        loc.offset_known = off_known;
        return loc;
    }

    static SymbolicMemoryLoc make_buffer(int32_t rack, int32_t desc, int64_t off = 0,
                                         bool off_known = true, std::string label = "") {
        SymbolicMemoryLoc loc;
        loc.region = MemoryRegionKind::Buffer;
        loc.table_id = rack;
        loc.entry_id = desc;
        loc.byte_offset = off_known ? off : 0;
        loc.offset_known = off_known;
        loc.key = std::move(label);
        return loc;
    }

    static SymbolicMemoryLoc make_tls(int64_t off = 0, bool off_known = true) {
        SymbolicMemoryLoc loc;
        loc.region = MemoryRegionKind::Tls;
        loc.byte_offset = off_known ? off : 0;
        loc.offset_known = off_known;
        return loc;
    }

    static SymbolicMemoryLoc make_wls(int64_t off = 0, bool off_known = true) {
        SymbolicMemoryLoc loc;
        loc.region = MemoryRegionKind::Wls;
        loc.byte_offset = off_known ? off : 0;
        loc.offset_known = off_known;
        return loc;
    }

    static SymbolicMemoryLoc make_data_section(int64_t off = 0, bool off_known = true) {
        SymbolicMemoryLoc loc;
        loc.region = MemoryRegionKind::DataSection;
        loc.byte_offset = off_known ? off : 0;
        loc.offset_known = off_known;
        return loc;
    }

    /// Return a copy with a constant byte delta added to `byte_offset` (no-op if offset unknown).
    SymbolicMemoryLoc with_added_offset(int64_t delta) const {
        SymbolicMemoryLoc out = *this;
        if (out.offset_known) {
            out.byte_offset += delta;
        }
        return out;
    }

    /// Return a copy with `offset_known` widened to false (dynamic / unknown offset).
    SymbolicMemoryLoc with_unknown_offset() const {
        SymbolicMemoryLoc out = *this;
        out.offset_known = false;
        out.byte_offset = 0;
        return out;
    }

    /// Return a copy with `byte_offset` masked by `mask` (preserving base region identity).
    SymbolicMemoryLoc with_masked_offset(int64_t mask) const {
        SymbolicMemoryLoc out = *this;
        if (out.offset_known) {
            out.byte_offset = out.byte_offset & mask;
        }
        return out;
    }

    bool operator==(const SymbolicMemoryLoc& o) const {
        return region == o.region &&
               table_id == o.table_id &&
               entry_id == o.entry_id &&
               byte_offset == o.byte_offset &&
               offset_known == o.offset_known &&
               key == o.key;
    }

    bool operator!=(const SymbolicMemoryLoc& o) const { return !(*this == o); }

    bool operator<(const SymbolicMemoryLoc& o) const {
        return std::tie(region, table_id, entry_id, offset_known, byte_offset, key) <
               std::tie(o.region, o.table_id, o.entry_id, o.offset_known, o.byte_offset, o.key);
    }

    /// Region-aware disjointness / may-alias predicate.
    bool may_alias(const SymbolicMemoryLoc& o) const {
        if (region != o.region) return false;
        if (region == MemoryRegionKind::SrtTable && table_id >= 0 && o.table_id >= 0 &&
            table_id != o.table_id) {
            return false;
        }
        if (region == MemoryRegionKind::Buffer) {
            if (table_id >= 0 && o.table_id >= 0 && table_id != o.table_id) return false;
            if (entry_id >= 0 && o.entry_id >= 0 && entry_id != o.entry_id) return false;
        }
        if (offset_known && o.offset_known && byte_offset != o.byte_offset) {
            return false;
        }
        return true;
    }

    std::string format() const {
        std::ostringstream oss;
        switch (region) {
            case MemoryRegionKind::FauUniform:  oss << "FAU_0"; break;
            case MemoryRegionKind::SrtTable:
                if (table_id < 0) oss << "SRT_0";
                else oss << "SRT[" << table_id << "]";
                break;
            case MemoryRegionKind::Buffer:
                if (!key.empty()) oss << key;
                else oss << "Buf[" << table_id << "," << entry_id << "]";
                break;
            case MemoryRegionKind::DataSection: oss << ".data"; break;
            case MemoryRegionKind::Tls:         oss << "TLS"; break;
            case MemoryRegionKind::Wls:         oss << "WLS"; break;
            case MemoryRegionKind::Other:       oss << (key.empty() ? "Mem" : key); break;
        }
        if (!offset_known) {
            oss << "+?";
        } else if (byte_offset != 0) {
            oss << "+0x" << std::hex << byte_offset;
        }
        return oss.str();
    }

    struct Hash {
        size_t operator()(const SymbolicMemoryLoc& loc) const noexcept {
            size_t h = std::hash<uint8_t>{}(static_cast<uint8_t>(loc.region));
            auto mix = [&](size_t v) { h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); };
            mix(std::hash<int32_t>{}(loc.table_id));
            mix(std::hash<int32_t>{}(loc.entry_id));
            mix(std::hash<int64_t>{}(loc.byte_offset));
            mix(std::hash<bool>{}(loc.offset_known));
            mix(std::hash<std::string>{}(loc.key));
            return h;
        }
    };
};

}  // namespace valhall::dataflow

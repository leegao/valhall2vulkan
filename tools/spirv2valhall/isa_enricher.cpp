#include "isa_enricher.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <set>
#include <sstream>

namespace valhall::spirv2valhall {

namespace {

struct SymbolLookup {
    const Mbs2Symbol* find_by_rack_addr(uint8_t rack, uint32_t addr) const {
        auto it = by_rack_addr.find({rack, addr});
        return (it != by_rack_addr.end()) ? it->second : nullptr;
    }

    const Mbs2Symbol* find_input_by_location(uint16_t loc) const {
        auto it = sinp_by_loc.find(loc);
        return (it != sinp_by_loc.end()) ? it->second : nullptr;
    }

    const Mbs2Symbol* find_output_by_location(uint16_t loc) const {
        auto it = sout_by_loc.find(loc);
        return (it != sout_by_loc.end()) ? it->second : nullptr;
    }

    const Mbs2Symbol* find_img_by_addr(uint32_t addr) const {
        auto it = simg_by_addr.find(addr);
        return (it != simg_by_addr.end()) ? it->second : nullptr;
    }

    const Mbs2Symbol* find_smp_by_addr(uint32_t addr) const {
        auto it = ssmp_by_addr.find(addr);
        return (it != ssmp_by_addr.end()) ? it->second : nullptr;
    }

    std::map<std::pair<uint8_t, uint32_t>, const Mbs2Symbol*> by_rack_addr;
    std::map<uint16_t, const Mbs2Symbol*> sinp_by_loc;
    std::map<uint16_t, const Mbs2Symbol*> sout_by_loc;
    std::map<uint32_t, const Mbs2Symbol*> simg_by_addr;
    std::map<uint32_t, const Mbs2Symbol*> ssmp_by_addr;
    std::map<uint32_t, std::string> rloc_by_fau_off;
    std::map<uint32_t, Mbs2Fcst> fcst_by_fau_off;
    std::map<uint16_t, std::string> fotv_by_loc;
};

std::string resolve_member_at_offset(const Mbs2Symbol& sym, uint32_t byte_off) {
    if (!sym.type) return sym.name;
    const std::vector<Mbs2Tpse>* members = nullptr;
    if (sym.type->has_tpib) members = &sym.type->tpib.members;
    else if (sym.type->has_tpst) members = &sym.type->tpst.members;
    if (!members || members->empty()) return sym.name;

    const Mbs2Tpse* best = nullptr;
    for (const auto& m : *members) {
        if (m.offset <= byte_off) {
            if (!best || m.offset >= best->offset) best = &m;
        }
    }
    if (!best || best->name.empty()) return sym.name;

    uint32_t rem = byte_off - best->offset;
    std::string out = sym.name + "." + best->name;
    if (best->type && best->type->has_tpge && best->type->tpge.vector_size > 1) {
        uint32_t elem_sz = (1u << best->type->tpge.scalar_size);
        if (elem_sz > 0) {
            uint32_t comp = rem / elem_sz;
            static const char kComp[] = "xyzw";
            if (comp < 4) {
                out.push_back('.');
                out.push_back(kComp[comp]);
            }
        }
    } else if (rem > 0) {
        char buf[24];
        std::snprintf(buf, sizeof(buf), "+0x%x", rem);
        out += buf;
    }
    return out;
}

SymbolLookup build_symbol_lookup(const Mbs2Module& mod, const ExtractedEbin* ebin) {
    SymbolLookup lut;
    auto index_list = [&](const std::vector<Mbs2Symbol>& list) {
        for (const auto& sym : list) {
            lut.by_rack_addr[{sym.rack_index, sym.address}] = &sym;
            for (const auto& r : sym.rlocs) {
                // Map every 4-byte word covered by the relocation
                uint32_t words = std::max<uint32_t>(1u, (r.size + 3u) / 4u);
                for (uint32_t w = 0; w < words; ++w) {
                    uint32_t fau_off = r.reg_offset + w * 4u;
                    uint32_t uni_off = r.uni_offset + w * 4u;
                    lut.rloc_by_fau_off[fau_off] = resolve_member_at_offset(sym, uni_off);
                }
            }
        }
    };
    index_list(mod.cmmn.suni);
    index_list(mod.cmmn.s3bo);
    index_list(mod.cmmn.simg);
    index_list(mod.cmmn.ssmp);

    for (const auto& sym : mod.cmmn.sinp) {
        if (sym.location != 0xFFFFu) lut.sinp_by_loc[sym.location] = &sym;
        else lut.sinp_by_loc[static_cast<uint16_t>(sym.address)] = &sym;
    }
    for (const auto& sym : mod.cmmn.sout) {
        if (sym.location != 0xFFFFu) lut.sout_by_loc[sym.location] = &sym;
        else lut.sout_by_loc[static_cast<uint16_t>(sym.address)] = &sym;
    }
    for (const auto& sym : mod.cmmn.simg) {
        lut.simg_by_addr[sym.address] = &sym;
    }
    for (const auto& sym : mod.cmmn.ssmp) {
        lut.ssmp_by_addr[sym.address] = &sym;
    }
    for (const auto& f : mod.cmmn.fcsts) {
        lut.fcst_by_fau_off[f.reg_offset] = f;
    }
    if (ebin && ebin->meta.has_dbug) {
        for (const auto& f : ebin->meta.dbug.fcsts) {
            lut.fcst_by_fau_off[f.reg_offset] = f;
        }
    }
    if (mod.has_cfra) {
        for (const auto& ov : mod.cfra.output_vars) {
            lut.fotv_by_loc[ov.location] = ov.name;
        }
    }
    return lut;
}

std::string format_fcst_value(uint32_t v) {
    float f = 0.0f;
    std::memcpy(&f, &v, sizeof(float));
    char buf[96];
    // Show float interpretation when exponent is normal (not 0 or 255)
    uint32_t exp = (v >> 23) & 0xFFu;
    if (exp > 0 && exp < 255) {
        std::snprintf(buf, sizeof(buf), "0x%08x (%.6g)", v, static_cast<double>(f));
    } else {
        std::snprintf(buf, sizeof(buf), "0x%08x (%u)", v, v);
    }
    return buf;
}

struct TrackedVal {
    enum class Kind {
        None = 0,
        FauDstBase,       // Pilot r0:r1 = pilot_fau_dst_va
        SrtBase,          // Pilot r2:r3 = srt_base_va
        SrtRackPtr,       // Resource[rack].address (points to Buffer/Desc table)
        BufferAddr,       // Buffer[idx].address (VA of SSBO/UBO/PushConst)
        BufferSize,       // Buffer[idx].size
        SymbolValue,      // Scalar/derived value loaded from a buffer/symbol
        DataSectionPtr,   // PC-relative pointer into EBIN data_section
    };
    Kind kind = Kind::None;
    int32_t rack = -1;
    int32_t desc_idx = -1;
    std::string label;
    std::string symbol_name;
    bool derived = false;
};

int64_t get_mem_offset(const DecodedValhallInst& ins) {
    auto it = ins.fields.find("offset");
    if (it != ins.fields.end()) {
        // Sign-extend if imm_signed matches
        if (ins.has_imm) return ins.imm_signed;
        return static_cast<int64_t>(it->second);
    }
    if (ins.has_imm) return ins.imm_signed;
    return 0;
}

bool is_load_mnemonic(Mnemonic m) {
    return m == Mnemonic::LOAD || m == Mnemonic::LD_CHECKED ||
           m == Mnemonic::LD_MULTI;
}

bool is_store_mnemonic(Mnemonic m) {
    return m == Mnemonic::STORE || m == Mnemonic::ST_CHECKED ||
           m == Mnemonic::ST_MULTI;
}

void set_reg_pair(std::map<uint8_t, TrackedVal>& state, uint8_t base, bool is64, const TrackedVal& val) {
    state[base] = val;
    if (is64 && base + 1 < 64) {
        state[static_cast<uint8_t>(base + 1)] = val;
    }
}

void clear_reg_pair(std::map<uint8_t, TrackedVal>& state, uint8_t base, uint8_t count) {
    for (uint8_t i = 0; i < count; ++i) {
        if (base + i < 64) state.erase(static_cast<uint8_t>(base + i));
    }
}

std::map<uint8_t, std::string> init_abi_entry_regs(EbinRole role) {
    std::map<uint8_t, std::string> abi;
    switch (role) {
        case EbinRole::Pilot:
            abi[0] = "pilot_fau_dst_va.lo (FAU_0)";
            abi[1] = "pilot_fau_dst_va.hi (FAU_0)";
            abi[2] = "srt_base_va.lo (SRT_0)";
            abi[3] = "srt_base_va.hi (SRT_0)";
            break;
        case EbinRole::MainCompute:
            abi[55] = "WorkGroupID.x";
            abi[56] = "WorkGroupID.y";
            abi[57] = "WorkGroupID.z";
            abi[60] = "GlobalInvocationID.x";
            abi[61] = "GlobalInvocationID.y";
            abi[62] = "GlobalInvocationID.z";
            break;
        case EbinRole::VertexPosition:
        case EbinRole::VertexVarying:
            abi[59] = "VertexStreamIndex";
            abi[60] = "VertexIndex";
            abi[61] = "InstanceIndex";
            break;
        case EbinRole::Fragment:
            abi[59] = "FragCoord.z";
            abi[60] = "sample_coverage_mask";
            abi[61] = "varying_barycentric";
            break;
        default:
            break;
    }
    return abi;
}

std::string lookup_fau_annotation(
    uint32_t fau_off,
    bool is_64bit,
    const SymbolLookup& lut,
    const PilotAnalysisResult& pilot_info) {
    // 1. Check Pilot FAU map first (exact match or aligned 64-bit base)
    auto pit = pilot_info.fau_map.find(fau_off);
    if (pit != pilot_info.fau_map.end()) {
        return pit->second.expression + " (pilot FAU+0x" +
               ([&]() {
                   char b[16];
                   std::snprintf(b, sizeof(b), "%x", fau_off);
                   return std::string(b);
               })() + ")";
    }
    if (!is_64bit && (fau_off & 4u)) {
        auto pit_lo = pilot_info.fau_map.find(fau_off & ~7u);
        if (pit_lo != pilot_info.fau_map.end() && pit_lo->second.byte_size == 8) {
            return pit_lo->second.symbol_name + ".hi (pilot)";
        }
    }

    // 2. Check RLOC symbol relocation
    auto rit = lut.rloc_by_fau_off.find(fau_off);
    if (rit != lut.rloc_by_fau_off.end()) {
        char b[24];
        std::snprintf(b, sizeof(b), "RLOC+0x%x", fau_off);
        return rit->second + " (" + b + ")";
    }

    // 3. Check FCST compiler constant
    auto fit = lut.fcst_by_fau_off.find(fau_off);
    if (fit != lut.fcst_by_fau_off.end()) {
        return "FCST " + format_fcst_value(fit->second.value);
    }
    return "";
}

}  // namespace

PilotAnalysisResult analyze_pilot_shaders(
    const Mbs2Module& mod,
    const std::vector<ExtractedEbin>& ebins) {
    PilotAnalysisResult res;
    SymbolLookup lut = build_symbol_lookup(mod, nullptr);

    for (const auto& eb : ebins) {
        if (eb.role != EbinRole::Pilot) continue;
        res.found_pilot = true;
        res.pilot_ebin_index = eb.index;

        std::map<uint8_t, TrackedVal> state;
        TrackedVal fau_base;
        fau_base.kind = TrackedVal::Kind::FauDstBase;
        fau_base.label = "FAU_0";
        set_reg_pair(state, 0, true, fau_base);

        TrackedVal srt_base;
        srt_base.kind = TrackedVal::Kind::SrtBase;
        srt_base.label = "SRT_0";
        set_reg_pair(state, 2, true, srt_base);

        for (const auto& ins : eb.instructions) {
            if (!ins.valid) continue;
            const Mnemonic m = ins.canonical_mnemonic;

            if (is_load_mnemonic(m) && ins.sr_write.present && !ins.srcs.empty() &&
                ins.srcs[0].kind == SourceKind::Register) {
                uint8_t base_r = ins.srcs[0].reg_num;
                int64_t off = get_mem_offset(ins);
                auto it = state.find(base_r);
                if (it != state.end()) {
                    TrackedVal src_v = it->second;  // copy before overwriting dst regs
                    if (src_v.kind == TrackedVal::Kind::SrtBase && off >= 0) {
                        int32_t rack = static_cast<int32_t>(off / 16);
                        TrackedVal tv;
                        tv.kind = TrackedVal::Kind::SrtRackPtr;
                        tv.rack = rack;
                        tv.label = "SRT[" + std::to_string(rack) + "]";
                        set_reg_pair(state, ins.sr_write.base_reg, ins.sr_write.count >= 2, tv);
                        continue;
                    }
                    if (src_v.kind == TrackedVal::Kind::SrtRackPtr && off >= 0) {
                        int32_t desc_idx = static_cast<int32_t>(off / 32);
                        int32_t field_off = static_cast<int32_t>(off % 32);
                        const Mbs2Symbol* sym = lut.find_by_rack_addr(
                            static_cast<uint8_t>(src_v.rack), static_cast<uint32_t>(desc_idx));
                        std::string sname = sym ? sym->name
                                                : ("SRT[" + std::to_string(src_v.rack) + "].Buf[" + std::to_string(desc_idx) + "]");
                        TrackedVal tv;
                        tv.rack = src_v.rack;
                        tv.desc_idx = desc_idx;
                        tv.symbol_name = sname;
                        if (field_off == 4) {
                            tv.kind = TrackedVal::Kind::BufferSize;
                            tv.label = "sizeof(" + sname + ")";
                        } else {
                            tv.kind = TrackedVal::Kind::BufferAddr;
                            tv.label = "&" + sname + " (SRT[" + std::to_string(src_v.rack) +
                                       "]->Buf[" + std::to_string(desc_idx) + "].addr)";
                        }
                        set_reg_pair(state, ins.sr_write.base_reg, ins.sr_write.count >= 2, tv);
                        continue;
                    }
                    if (src_v.kind == TrackedVal::Kind::BufferAddr && off >= 0) {
                        const Mbs2Symbol* sym = lut.find_by_rack_addr(
                            static_cast<uint8_t>(src_v.rack), static_cast<uint32_t>(src_v.desc_idx));
                        std::string member = sym ? resolve_member_at_offset(*sym, static_cast<uint32_t>(off))
                                                 : (src_v.symbol_name + "+0x" + std::to_string(off));
                        TrackedVal tv;
                        tv.kind = TrackedVal::Kind::SymbolValue;
                        tv.rack = src_v.rack;
                        tv.desc_idx = src_v.desc_idx;
                        tv.symbol_name = member;
                        tv.label = member;
                        set_reg_pair(state, ins.sr_write.base_reg, ins.sr_write.count >= 2, tv);
                        continue;
                    }
                }
                clear_reg_pair(state, ins.sr_write.base_reg, ins.sr_write.count);
            } else if (is_store_mnemonic(m) && ins.sr_read.present && !ins.srcs.empty() &&
                       ins.srcs[0].kind == SourceKind::Register) {
                uint8_t dst_ptr_r = ins.srcs[0].reg_num;
                int64_t fau_off = get_mem_offset(ins);
                auto pit = state.find(dst_ptr_r);
                if (pit != state.end() && pit->second.kind == TrackedVal::Kind::FauDstBase && fau_off >= 0) {
                    auto vit = state.find(ins.sr_read.base_reg);
                    if (vit != state.end()) {
                        PilotFauEntry entry;
                        entry.fau_offset = static_cast<uint32_t>(fau_off);
                        entry.byte_size = static_cast<uint32_t>(ins.sr_read.count) * 4u;
                        entry.symbol_name = vit->second.symbol_name;
                        entry.srt_rack = vit->second.rack;
                        entry.desc_index = vit->second.desc_idx;
                        if (vit->second.derived) {
                            entry.expression = "derived(" + vit->second.symbol_name + ")";
                        } else {
                            entry.expression = vit->second.label;
                        }
                        res.fau_map[entry.fau_offset] = std::move(entry);
                    }
                }
            } else if (ins.has_dest) {
                // Preserve 56-bit pointer mask (`AND_IMM r_hi, r_hi, #0xffffff` or `AND r_hi, r_hi, r_mask`)
                if ((m == Mnemonic::AND_IMM || m == Mnemonic::AND) &&
                    !ins.srcs.empty() && ins.srcs[0].kind == SourceKind::Register &&
                    ins.srcs[0].reg_num == ins.dest_reg) {
                    auto it = state.find(ins.dest_reg);
                    if (it != state.end() && it->second.kind == TrackedVal::Kind::SrtRackPtr) {
                        continue;  // preserve SRT rack pointer across 56-bit mask
                    }
                }
                // Preserve CSEL bounds check on high word of BufferAddr
                if (m == Mnemonic::CSEL && ins.srcs.size() >= 3 &&
                    ins.srcs[2].kind == SourceKind::Register && ins.srcs[2].reg_num == ins.dest_reg) {
                    auto it = state.find(ins.dest_reg);
                    if (it != state.end() && it->second.kind == TrackedVal::Kind::BufferAddr) {
                        continue;
                    }
                }
                // Propagate derived scalar expression from SymbolValue
                bool propagated = false;
                for (const auto& s : ins.srcs) {
                    if (s.kind == SourceKind::Register) {
                        auto it = state.find(s.reg_num);
                        if (it != state.end() && it->second.kind == TrackedVal::Kind::SymbolValue) {
                            TrackedVal tv = it->second;
                            tv.derived = true;
                            set_reg_pair(state, ins.dest_reg, ins.dest_is_64bit, tv);
                            propagated = true;
                            break;
                        }
                    }
                }
                if (!propagated) {
                    clear_reg_pair(state, ins.dest_reg, ins.dest_is_64bit ? 2 : 1);
                }
            }
        }
        break;  // analyze first pilot EBIN
    }
    return res;
}

std::vector<std::string> enrich_ebin_instructions(
    const Mbs2Module& mod,
    const ExtractedEbin& ebin,
    const PilotAnalysisResult& pilot_info) {
    std::vector<std::string> notes(ebin.instructions.size());
    SymbolLookup lut = build_symbol_lookup(mod, &ebin);

    // Live ABI entry registers (removed as soon as written/clobbered!)
    std::map<uint8_t, std::string> live_abi = init_abi_entry_regs(ebin.role);
    // Symbolic pointer/value state tracked across instructions
    std::map<uint8_t, TrackedVal> ptr_state;

    if (ebin.role == EbinRole::Pilot) {
        TrackedVal fau_base;
        fau_base.kind = TrackedVal::Kind::FauDstBase;
        fau_base.label = "FAU_0";
        set_reg_pair(ptr_state, 0, true, fau_base);

        TrackedVal srt_base;
        srt_base.kind = TrackedVal::Kind::SrtBase;
        srt_base.label = "SRT_0";
        set_reg_pair(ptr_state, 2, true, srt_base);
    }

    for (size_t idx = 0; idx < ebin.instructions.size(); ++idx) {
        const auto& ins = ebin.instructions[idx];
        if (!ins.valid) continue;

        std::vector<std::string> tags;
        std::set<std::string> seen_tags;
        auto add_tag = [&](const std::string& t) {
            if (!t.empty() && seen_tags.insert(t).second) {
                tags.push_back(t);
            }
        };

        const Mnemonic m = ins.canonical_mnemonic;

        // 1. Tag live ABI entry registers read as sources (BEFORE clobber!)
        auto check_abi_read = [&](uint8_t reg, bool is64) {
            auto it = live_abi.find(reg);
            if (it != live_abi.end()) {
                if (is64 && live_abi.count(static_cast<uint8_t>(reg + 1))) {
                    // Compact pair tag for pilot r0:r1 / r2:r3
                    std::string base_name = it->second;
                    auto dot = base_name.find(".lo");
                    if (dot != std::string::npos) {
                        base_name.erase(dot, 3);
                        add_tag("r" + std::to_string(reg) + ":r" + std::to_string(reg + 1) + " = " + base_name);
                        return;
                    }
                }
                add_tag("r" + std::to_string(reg) + " = " + it->second);
            }
        };

        for (const auto& s : ins.srcs) {
            if (s.kind == SourceKind::Register) {
                check_abi_read(s.reg_num, s.is_64bit);
            } else if (s.kind == SourceKind::FauUniform) {
                uint32_t fau_off = static_cast<uint32_t>(s.fau_index64) * 8u + (s.word_select ? 4u : 0u);
                std::string fau_label = lookup_fau_annotation(fau_off, s.is_64bit, lut, pilot_info);
                if (!fau_label.empty()) {
                    std::string reg_str = s.is_64bit
                                              ? ("u" + std::to_string(s.fau_index64))
                                              : format_fau_reg(fau_off, 4);
                    add_tag(reg_str + " = " + fau_label);
                }
            } else if (s.kind == SourceKind::FauSpecial) {
                if (s.symbolic_name.rfind("blend_descriptor_", 0) == 0) {
                    uint16_t rt = static_cast<uint16_t>(std::atoi(s.symbolic_name.c_str() + 17));
                    auto oit = lut.fotv_by_loc.find(rt);
                    if (oit != lut.fotv_by_loc.end()) {
                        add_tag("RT" + std::to_string(rt) + " -> " + oit->second);
                    } else if (const Mbs2Symbol* osym = lut.find_output_by_location(rt)) {
                        add_tag("RT" + std::to_string(rt) + " -> " + osym->name);
                    }
                }
            }
        }
        if (ins.sr_read.present) {
            for (uint8_t k = 0; k < ins.sr_read.count; ++k) {
                check_abi_read(static_cast<uint8_t>(ins.sr_read.base_reg + k), false);
            }
        }

        // 2. Instruction-specific semantic enrichment & pointer chasing
        if (ins.mnemonic == Mnemonic::ADR && (ins.has_branch_offset || ins.has_imm)) {
            int64_t target = ins.has_branch_offset
                                 ? ins.branch_target_pc
                                 : (static_cast<int64_t>(ins.pc + 8) + ins.imm_signed);
            if (ebin.meta.data_section_offset != 0xFFFFFFFFu &&
                target >= static_cast<int64_t>(ebin.meta.data_section_offset)) {
                char buf[64];
                std::snprintf(buf, sizeof(buf), "data_section+0x%llx (LUT @ 0x%llx)",
                              static_cast<unsigned long long>(target - ebin.meta.data_section_offset),
                              static_cast<unsigned long long>(target));
                add_tag(buf);
                if (ins.has_dest) {
                    TrackedVal tv;
                    tv.kind = TrackedVal::Kind::DataSectionPtr;
                    tv.label = buf;
                    tv.symbol_name = "data_section";
                    set_reg_pair(ptr_state, ins.dest_reg, true, tv);
                }
            }
        } else if (m == Mnemonic::LD_ATTR_IMM || m == Mnemonic::LD_VAR_IMM) {
            uint16_t idx_val = 0;
            auto it_i = ins.fields.find("index");
            if (it_i != ins.fields.end()) idx_val = static_cast<uint16_t>(it_i->second);
            if (const Mbs2Symbol* sym = lut.find_input_by_location(idx_val)) {
                add_tag((m == Mnemonic::LD_ATTR_IMM ? "attr[" : "varying[") +
                        std::to_string(idx_val) + "] = " + sym->name);
            }
        } else if (m == Mnemonic::TEX_SINGLE || m == Mnemonic::VAR_TEX_SINGLE ||
                   m == Mnemonic::TEX_FETCH || m == Mnemonic::TEX_GATHER) {
            auto it_t = ins.fields.find("texture_index");
            if (it_t != ins.fields.end()) {
                if (const Mbs2Symbol* sym = lut.find_img_by_addr(it_t->second)) {
                    add_tag("tex[" + std::to_string(it_t->second) + "] = " + sym->name);
                }
            }
            auto it_s = ins.fields.find("sampler_index");
            if (it_s != ins.fields.end()) {
                if (const Mbs2Symbol* sym = lut.find_smp_by_addr(it_s->second)) {
                    add_tag("smp[" + std::to_string(it_s->second) + "] = " + sym->name);
                }
            }
        } else if (m == Mnemonic::SHADD && ins.has_dest && !ins.srcs.empty()) {
            // Track `SHADD.u64 [rD:rD+1], uK, idx, #shift` -> base buffer pointer!
            if (ins.srcs[0].kind == SourceKind::FauUniform) {
                uint32_t fau_off = static_cast<uint32_t>(ins.srcs[0].fau_index64) * 8u;
                auto pit = pilot_info.fau_map.find(fau_off);
                if (pit != pilot_info.fau_map.end() && !pit->second.symbol_name.empty()) {
                    TrackedVal tv;
                    tv.kind = TrackedVal::Kind::BufferAddr;
                    tv.rack = pit->second.srt_rack;
                    tv.desc_idx = pit->second.desc_index;
                    tv.symbol_name = pit->second.symbol_name;
                    tv.label = "&" + pit->second.symbol_name + "[...]";
                    set_reg_pair(ptr_state, ins.dest_reg, true, tv);
                }
            } else if (ins.srcs[0].kind == SourceKind::Register) {
                auto it = ptr_state.find(ins.srcs[0].reg_num);
                if (it != ptr_state.end()) {
                    TrackedVal tv = it->second;
                    add_tag("ptr -> " + tv.label);
                    set_reg_pair(ptr_state, ins.dest_reg, true, tv);
                }
            }
        } else if (is_load_mnemonic(m) && ins.sr_write.present && !ins.srcs.empty() &&
                   ins.srcs[0].kind == SourceKind::Register) {
            uint8_t base_r = ins.srcs[0].reg_num;
            int64_t off = get_mem_offset(ins);
            auto it = ptr_state.find(base_r);
            if (it != ptr_state.end()) {
                TrackedVal src_v = it->second;
                if (src_v.kind == TrackedVal::Kind::SrtBase && off >= 0) {
                    int32_t rack = static_cast<int32_t>(off / 16);
                    add_tag("chase SRT[" + std::to_string(rack) + "] (Resource[" + std::to_string(rack) + "].address)");
                    TrackedVal tv;
                    tv.kind = TrackedVal::Kind::SrtRackPtr;
                    tv.rack = rack;
                    tv.label = "SRT[" + std::to_string(rack) + "]";
                    set_reg_pair(ptr_state, ins.sr_write.base_reg, ins.sr_write.count >= 2, tv);
                } else if (src_v.kind == TrackedVal::Kind::SrtRackPtr && off >= 0) {
                    int32_t desc_idx = static_cast<int32_t>(off / 32);
                    int32_t field_off = static_cast<int32_t>(off % 32);
                    const Mbs2Symbol* sym = lut.find_by_rack_addr(
                        static_cast<uint8_t>(src_v.rack), static_cast<uint32_t>(desc_idx));
                    std::string sname = sym ? sym->name
                                            : ("Buf[" + std::to_string(desc_idx) + "]");
                    const char* field_name = (field_off == 4) ? "size" : (field_off == 8 ? "address" : "field");
                    add_tag("chase SRT[" + std::to_string(src_v.rack) + "]->Buf[" +
                            std::to_string(desc_idx) + "]." + field_name + " (" + sname + ")");
                    TrackedVal tv;
                    tv.rack = src_v.rack;
                    tv.desc_idx = desc_idx;
                    tv.symbol_name = sname;
                    tv.kind = (field_off == 4) ? TrackedVal::Kind::BufferSize : TrackedVal::Kind::BufferAddr;
                    tv.label = (field_off == 4) ? ("sizeof(" + sname + ")") : ("&" + sname);
                    set_reg_pair(ptr_state, ins.sr_write.base_reg, ins.sr_write.count >= 2, tv);
                } else if (src_v.kind == TrackedVal::Kind::BufferAddr) {
                    const Mbs2Symbol* sym = (src_v.rack >= 0)
                                                ? lut.find_by_rack_addr(static_cast<uint8_t>(src_v.rack),
                                                                        static_cast<uint32_t>(src_v.desc_idx))
                                                : nullptr;
                    std::string member = (sym && off >= 0)
                                             ? resolve_member_at_offset(*sym, static_cast<uint32_t>(off))
                                             : src_v.label;
                    add_tag("load " + member);
                    TrackedVal tv;
                    tv.kind = TrackedVal::Kind::SymbolValue;
                    tv.rack = src_v.rack;
                    tv.desc_idx = src_v.desc_idx;
                    tv.symbol_name = member;
                    tv.label = member;
                    set_reg_pair(ptr_state, ins.sr_write.base_reg, ins.sr_write.count >= 2, tv);
                } else if (src_v.kind == TrackedVal::Kind::DataSectionPtr) {
                    add_tag("load " + src_v.label);
                    clear_reg_pair(ptr_state, ins.sr_write.base_reg, ins.sr_write.count);
                } else {
                    clear_reg_pair(ptr_state, ins.sr_write.base_reg, ins.sr_write.count);
                }
            } else {
                clear_reg_pair(ptr_state, ins.sr_write.base_reg, ins.sr_write.count);
            }
        } else if (is_store_mnemonic(m) && ins.sr_read.present && !ins.srcs.empty() &&
                   ins.srcs[0].kind == SourceKind::Register) {
            uint8_t dst_ptr_r = ins.srcs[0].reg_num;
            int64_t off = get_mem_offset(ins);
            auto pit = ptr_state.find(dst_ptr_r);
            if (pit != ptr_state.end()) {
                if (pit->second.kind == TrackedVal::Kind::FauDstBase && off >= 0) {
                    std::string fau_reg = format_fau_reg(static_cast<uint32_t>(off),
                                                         static_cast<uint32_t>(ins.sr_read.count) * 4u);
                    auto vit = ptr_state.find(ins.sr_read.base_reg);
                    if (vit != ptr_state.end()) {
                        std::string val_desc = vit->second.derived
                                                   ? ("derived(" + vit->second.symbol_name + ")")
                                                   : vit->second.label;
                        char buf[32];
                        std::snprintf(buf, sizeof(buf), "FAU[+0x%llx] (%s) = ",
                                      static_cast<unsigned long long>(off), fau_reg.c_str());
                        add_tag(std::string(buf) + val_desc);
                    } else {
                        char buf[32];
                        std::snprintf(buf, sizeof(buf), "FAU[+0x%llx] (%s)",
                                      static_cast<unsigned long long>(off), fau_reg.c_str());
                        add_tag(buf);
                    }
                } else if (pit->second.kind == TrackedVal::Kind::BufferAddr) {
                    add_tag("store " + pit->second.label);
                }
            }
        } else if (ins.has_dest) {
            // Check 56-bit pointer mask in pilot shader
            if ((m == Mnemonic::AND_IMM || m == Mnemonic::AND) &&
                !ins.srcs.empty() && ins.srcs[0].kind == SourceKind::Register &&
                ins.srcs[0].reg_num == ins.dest_reg) {
                auto it = ptr_state.find(ins.dest_reg);
                if (it != ptr_state.end() && it->second.kind == TrackedVal::Kind::SrtRackPtr) {
                    add_tag("mask 56-bit VA (" + it->second.label + ")");
                }
            } else if (m == Mnemonic::CSEL && ins.srcs.size() >= 3 &&
                       ins.srcs[2].kind == SourceKind::Register && ins.srcs[2].reg_num == ins.dest_reg &&
                       ptr_state.count(ins.dest_reg) &&
                       ptr_state[ins.dest_reg].kind == TrackedVal::Kind::BufferAddr) {
                add_tag("bounds-check " + ptr_state[ins.dest_reg].label);
            } else {
                // Propagate derived symbol value or clear destination
                bool propagated = false;
                for (const auto& s : ins.srcs) {
                    if (s.kind == SourceKind::Register) {
                        auto it = ptr_state.find(s.reg_num);
                        if (it != ptr_state.end() && it->second.kind == TrackedVal::Kind::SymbolValue) {
                            TrackedVal tv = it->second;
                            tv.derived = true;
                            set_reg_pair(ptr_state, ins.dest_reg, ins.dest_is_64bit, tv);
                            propagated = true;
                            break;
                        }
                    }
                }
                bool keep_data_ptr = false;
                if (ins.mnemonic == Mnemonic::MOV && !ins.srcs.empty() &&
                    ins.srcs[0].kind == SourceKind::FauSpecial &&
                    ins.srcs[0].symbolic_name == "pc" &&
                    ptr_state.count(ins.dest_reg) &&
                    ptr_state[ins.dest_reg].kind == TrackedVal::Kind::DataSectionPtr) {
                    keep_data_ptr = true;
                }
                if (!propagated && !keep_data_ptr) {
                    clear_reg_pair(ptr_state, ins.dest_reg, ins.dest_is_64bit ? 2 : 1);
                }
            }
        }

        // 3. Invalidate any ABI entry register clobbered by this instruction's writes!
        if (ins.has_dest) {
            live_abi.erase(ins.dest_reg);
            if (ins.dest_is_64bit) live_abi.erase(static_cast<uint8_t>(ins.dest_reg + 1));
        }
        if (ins.sr_write.present) {
            for (uint8_t k = 0; k < ins.sr_write.count; ++k) {
                live_abi.erase(static_cast<uint8_t>(ins.sr_write.base_reg + k));
            }
        }

        if (!tags.empty()) {
            std::ostringstream toss;
            for (size_t i = 0; i < tags.size(); ++i) {
                if (i > 0) toss << " | ";
                toss << tags[i];
            }
            notes[idx] = toss.str();
        }
    }

    return notes;
}

std::string format_enriched_disassembly(
    const SpirvStageInfo& stage_info,
    uint8_t arch,
    const Mbs2Module& mod,
    const std::vector<ExtractedEbin>& ebins,
    const PilotAnalysisResult& pilot_info,
    bool include_modifiers,
    bool enrich) {
    std::ostringstream oss;
    oss << "; spirv2valhall stage=" << stage_info.malioc_flag
        << " entry=\"" << stage_info.entry_point << "\""
        << " arch=v" << static_cast<int>(arch)
        << " ebins=" << ebins.size() << "\n";

    for (const auto& eb : ebins) {
        oss << "\n; === EBIN[" << eb.index << "] [" << ebin_role_name(eb.role) << "]";
        if (!eb.meta.var_name.empty()) oss << " (" << eb.meta.var_name << ")";
        oss << " variant=0x" << std::hex << eb.variant_flags
            << " flags=0x" << eb.meta.flags
            << std::dec
            << " regs=" << eb.meta.nof_reg
            << " rmu=" << eb.meta.nof_rmu
            << " code_size=" << eb.code_size
            << " insts=" << eb.instructions.size()
            << " invalid=" << eb.invalid_count << " ===\n";

        std::vector<std::string> notes;
        if (enrich) {
            notes = enrich_ebin_instructions(mod, eb, pilot_info);
        }

        for (size_t i = 0; i < eb.instructions.size(); ++i) {
            const auto& dec = eb.instructions[i];
            char hdr[64];
            std::snprintf(hdr, sizeof(hdr), "  0x%04llx: %016llx  ",
                          static_cast<unsigned long long>(dec.pc),
                          static_cast<unsigned long long>(dec.raw_word));
            std::string body = dec.valid ? dec.disassembly : std::string("INVALID");
            if (include_modifiers && dec.valid && !dec.modifier_strs.empty()) {
                body += "  [";
                bool first = true;
                for (const auto& kv : dec.modifier_strs) {
                    if (!first) body += ",";
                    body += kv.first + "=" + kv.second;
                    first = false;
                }
                body += "]";
            }
            oss << hdr << body;
            if (enrich && i < notes.size() && !notes[i].empty()) {
                if (body.size() < 48) {
                    oss << std::string(48 - body.size(), ' ');
                } else {
                    oss << "  ";
                }
                oss << "; " << notes[i];
            }
            oss << "\n";
        }

        if (!eb.data_section.empty()) {
            oss << "  ; [Embedded Data Section: offset=0x" << std::hex << eb.code_size
                << ", size=" << std::dec << eb.data_section.size()
                << " bytes (total OBJC=" << eb.meta.objc_total_size << " bytes)]\n";
        }
    }
    return oss.str();
}

}  // namespace valhall::spirv2valhall

#include "isa_enricher.h"
#include "pointer_analysis.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <set>
#include <sstream>

namespace valhall::spirv2valhall {

namespace {

struct Mbs2SymbolIndex {
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
        if (m.offset <= byte_off && (!best || m.offset >= best->offset)) best = &m;
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

Mbs2SymbolIndex build_symbol_index(const Mbs2Module& mod, const ExtractedEbin* ebin) {
    Mbs2SymbolIndex idx;
    auto index_list = [&](const std::vector<Mbs2Symbol>& list) {
        for (const auto& sym : list) {
            idx.by_rack_addr[{sym.rack_index, sym.address}] = &sym;
            for (const auto& r : sym.rlocs) {
                uint32_t words = std::max<uint32_t>(1u, (r.size + 3u) / 4u);
                for (uint32_t w = 0; w < words; ++w) {
                    idx.rloc_by_fau_off[r.reg_offset + w * 4u] =
                        resolve_member_at_offset(sym, r.uni_offset + w * 4u);
                }
            }
        }
    };
    index_list(mod.cmmn.suni);
    index_list(mod.cmmn.s3bo);
    index_list(mod.cmmn.simg);
    index_list(mod.cmmn.ssmp);

    for (const auto& sym : mod.cmmn.sinp) {
        idx.sinp_by_loc[(sym.location != 0xFFFFu) ? sym.location : static_cast<uint16_t>(sym.address)] = &sym;
    }
    for (const auto& sym : mod.cmmn.sout) {
        idx.sout_by_loc[(sym.location != 0xFFFFu) ? sym.location : static_cast<uint16_t>(sym.address)] = &sym;
    }
    for (const auto& sym : mod.cmmn.simg) idx.simg_by_addr[sym.address] = &sym;
    for (const auto& sym : mod.cmmn.ssmp) idx.ssmp_by_addr[sym.address] = &sym;
    for (const auto& f : mod.cmmn.fcsts) idx.fcst_by_fau_off[f.reg_offset] = f;
    if (ebin && ebin->meta.has_dbug) {
        for (const auto& f : ebin->meta.dbug.fcsts) idx.fcst_by_fau_off[f.reg_offset] = f;
    }
    if (mod.has_cfra) {
        for (const auto& ov : mod.cfra.output_vars) idx.fotv_by_loc[ov.location] = ov.name;
    }
    return idx;
}

std::string format_fcst_value(uint32_t v) {
    float f = 0.0f;
    std::memcpy(&f, &v, sizeof(float));
    char buf[96];
    uint32_t exp = (v >> 23) & 0xFFu;
    if (exp > 0 && exp < 255) {
        std::snprintf(buf, sizeof(buf), "0x%08x (%.6g)", v, static_cast<double>(f));
    } else {
        std::snprintf(buf, sizeof(buf), "0x%08x (%u)", v, v);
    }
    return buf;
}

dataflow::PointerAnalysisEnv::StageRole to_df_role(EbinRole role) {
    switch (role) {
        case EbinRole::Pilot: return dataflow::PointerAnalysisEnv::StageRole::Pilot;
        case EbinRole::MainCompute: return dataflow::PointerAnalysisEnv::StageRole::MainCompute;
        case EbinRole::VertexPosition:
        case EbinRole::VertexVarying: return dataflow::PointerAnalysisEnv::StageRole::Vertex;
        case EbinRole::Fragment: return dataflow::PointerAnalysisEnv::StageRole::Fragment;
        default: return dataflow::PointerAnalysisEnv::StageRole::Unknown;
    }
}

dataflow::PointerAnalysisEnv build_df_env(
    const Mbs2SymbolIndex& sym_idx,
    const ExtractedEbin& ebin,
    const PilotAnalysisResult* pilot_info) {
    dataflow::PointerAnalysisEnv env;
    env.role = to_df_role(ebin.role);
    env.data_section_offset = ebin.meta.data_section_offset;

    if (pilot_info) {
        for (const auto& kv : pilot_info->fau_map) {
            const auto& pe = kv.second;
            if (pe.srt_rack >= 0 && pe.desc_index >= 0 && pe.byte_size == 8) {
                env.initial_fau[pe.fau_offset] = dataflow::TrackedVal::make_buffer_addr(
                    pe.srt_rack, pe.desc_index, pe.symbol_name, 0, false);
            } else {
                env.initial_fau[pe.fau_offset] = dataflow::TrackedVal::make_symbol_value(
                    pe.srt_rack, pe.desc_index, pe.symbol_name, true);
            }
        }
    }

    env.lookup_srt_symbol = [&sym_idx](uint8_t rack, uint32_t desc) -> std::string {
        auto it = sym_idx.by_rack_addr.find({rack, desc});
        return (it != sym_idx.by_rack_addr.end()) ? it->second->name : "";
    };
    env.lookup_symbol_member = [&sym_idx](uint8_t rack, uint32_t desc, uint32_t off) -> std::string {
        auto it = sym_idx.by_rack_addr.find({rack, desc});
        return (it != sym_idx.by_rack_addr.end()) ? resolve_member_at_offset(*it->second, off) : "";
    };
    env.lookup_input_symbol = [&sym_idx](uint32_t loc) -> std::string {
        auto it = sym_idx.sinp_by_loc.find(static_cast<uint16_t>(loc));
        return (it != sym_idx.sinp_by_loc.end()) ? it->second->name : "";
    };
    env.lookup_output_symbol = [&sym_idx](uint32_t loc) -> std::string {
        auto fit = sym_idx.fotv_by_loc.find(static_cast<uint16_t>(loc));
        if (fit != sym_idx.fotv_by_loc.end()) return fit->second;
        auto it = sym_idx.sout_by_loc.find(static_cast<uint16_t>(loc));
        return (it != sym_idx.sout_by_loc.end()) ? it->second->name : "";
    };
    env.lookup_image_symbol = [&sym_idx](uint32_t addr) -> std::string {
        auto it = sym_idx.simg_by_addr.find(addr);
        return (it != sym_idx.simg_by_addr.end()) ? it->second->name : "";
    };
    env.lookup_sampler_symbol = [&sym_idx](uint32_t addr) -> std::string {
        auto it = sym_idx.ssmp_by_addr.find(addr);
        return (it != sym_idx.ssmp_by_addr.end()) ? it->second->name : "";
    };
    return env;
}

std::string lookup_fau_annotation(
    uint32_t fau_off,
    bool is_64bit,
    const Mbs2SymbolIndex& lut,
    const PilotAnalysisResult& pilot_info) {
    auto pit = pilot_info.fau_map.find(fau_off);
    if (pit != pilot_info.fau_map.end()) {
        char b[24];
        std::snprintf(b, sizeof(b), "pilot FAU+0x%x", fau_off);
        return pit->second.expression + " (" + b + ")";
    }
    if (!is_64bit && (fau_off & 4u)) {
        auto pit_lo = pilot_info.fau_map.find(fau_off & ~7u);
        if (pit_lo != pilot_info.fau_map.end() && pit_lo->second.byte_size == 8) {
            return pit_lo->second.symbol_name + ".hi (pilot)";
        }
    }
    auto rit = lut.rloc_by_fau_off.find(fau_off);
    if (rit != lut.rloc_by_fau_off.end()) {
        char b[24];
        std::snprintf(b, sizeof(b), "RLOC+0x%x", fau_off);
        return rit->second + " (" + b + ")";
    }
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
    Mbs2SymbolIndex sym_idx = build_symbol_index(mod, nullptr);

    for (const auto& eb : ebins) {
        if (eb.role != EbinRole::Pilot) continue;
        res.found_pilot = true;
        res.pilot_ebin_index = eb.index;

        dataflow::PointerAnalysisEnv env = build_df_env(sym_idx, eb, nullptr);
        dataflow::PointerAnalysisResult df = dataflow::run_pointer_analysis(eb.instructions, env);

        for (const auto& kv : df.pilot_fau_writes) {
            const auto& pw = kv.second;
            PilotFauEntry entry;
            entry.fau_offset = pw.fau_offset;
            entry.byte_size = pw.byte_size;
            entry.symbol_name = pw.value.symbol_name;
            entry.expression = pw.value.format_short();
            entry.srt_rack = pw.value.rack;
            entry.desc_index = pw.value.desc_idx;
            res.fau_map[entry.fau_offset] = std::move(entry);
        }
        break;
    }
    return res;
}

std::vector<std::string> enrich_ebin_instructions(
    const Mbs2Module& mod,
    const ExtractedEbin& ebin,
    const PilotAnalysisResult& pilot_info) {
    std::vector<std::string> notes(ebin.instructions.size());
    Mbs2SymbolIndex sym_idx = build_symbol_index(mod, &ebin);

    // Run decoupled pointer & ABI dataflow analysis as its own initial pass!
    dataflow::PointerAnalysisEnv env = build_df_env(sym_idx, ebin, &pilot_info);
    dataflow::PointerAnalysisResult df = dataflow::run_pointer_analysis(ebin.instructions, env);

    for (size_t idx = 0; idx < ebin.instructions.size(); ++idx) {
        const auto& ins = ebin.instructions[idx];
        if (!ins.valid) continue;

        std::vector<std::string> tags;
        std::set<std::string> seen;
        auto add_tag = [&](const std::string& t) {
            if (!t.empty() && seen.insert(t).second) tags.push_back(t);
        };

        const auto& pre = df.pre_states[idx];

        // 1. Tag live ABI entry registers & Pilot base pointers directly from pre-state
        for (const auto& s : ins.srcs) {
            if (s.kind == SourceKind::Register) {
                const auto& tv = pre.get_reg(s.reg_num);
                if (tv.kind == dataflow::TrackedVal::Kind::AbiPreload) {
                    add_tag("r" + std::to_string(s.reg_num) + " = " + tv.label);
                } else if (tv.kind == dataflow::TrackedVal::Kind::FauDstBase ||
                           tv.kind == dataflow::TrackedVal::Kind::SrtBase) {
                    std::string rname = s.is_64bit
                                            ? ("r" + std::to_string(s.reg_num) + ":r" + std::to_string(s.reg_num + 1))
                                            : ("r" + std::to_string(s.reg_num));
                    add_tag(rname + " = " + tv.label);
                }
            } else if (s.kind == SourceKind::FauUniform) {
                uint32_t fau_off = static_cast<uint32_t>(s.fau_index64) * 8u + (s.word_select ? 4u : 0u);
                std::string fau_label = lookup_fau_annotation(fau_off, s.is_64bit, sym_idx, pilot_info);
                if (!fau_label.empty()) {
                    std::string reg_str = s.is_64bit
                                              ? ("u" + std::to_string(s.fau_index64))
                                              : format_fau_reg(fau_off, 4);
                    add_tag(reg_str + " = " + fau_label);
                }
            }
        }

        // 2. Append semantic dataflow events discovered by PointerAnalysis pass
        if (idx < df.inst_notes.size()) {
            for (const auto& ev : df.inst_notes[idx]) {
                add_tag(ev);
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

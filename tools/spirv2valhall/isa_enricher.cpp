#include "isa_enricher.h"
#include "liveness.h"
#include "pointer_analysis.h"
#include "valhall_cfg.h"
#include "valhall_pass1.h"
#include "valhall_to_spirv.h"

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

valhall::ShaderStage to_valhall_stage(dataflow::PointerAnalysisEnv::StageRole role) {
    switch (role) {
        case dataflow::PointerAnalysisEnv::StageRole::Pilot: return valhall::ShaderStage::Pilot;
        case dataflow::PointerAnalysisEnv::StageRole::MainCompute: return valhall::ShaderStage::Compute;
        case dataflow::PointerAnalysisEnv::StageRole::Vertex: return valhall::ShaderStage::Vertex;
        case dataflow::PointerAnalysisEnv::StageRole::Fragment: return valhall::ShaderStage::Fragment;
        default: return valhall::ShaderStage::Compute;
    }
}

void append_unique_note(std::vector<std::string>& notes, const std::string& s) {
    if (s.empty()) return;
    if (std::find(notes.begin(), notes.end(), s) == notes.end()) {
        notes.push_back(s);
    }
}

struct IsaProjectedAnalysis {
    dataflow::PointerAnalysisResult df;
    std::vector<std::vector<std::string>> inst_notes;
};

/// Explicitly lower decoded Valhall ISA -> CFG -> SSA (`run_pass1_lowering` with `tolerant_ssa = true`),
/// run backwards SSA liveness + forward SSA pointer analysis (`src/dataflow`), and project SSA findings
/// back onto each originating ISA instruction index (`orig_inst_idx`).
IsaProjectedAnalysis lower_and_run_ssa_pointer_analysis(
    const std::vector<DecodedValhallInst>& instructions,
    const dataflow::PointerAnalysisEnv& env) {
    IsaProjectedAnalysis res;
    res.inst_notes.resize(instructions.size());
    if (instructions.empty()) {
        res.df.converged = true;
        return res;
    }

    // 1. Explicitly lower ISA -> CFG -> SSA here in isa_enricher
    cfg::ControlFlowGraph primary_cfg = cfg::build_cfg(instructions);
    cfg::ControlFlowGraph empty_cfg;

    ValhallTranslateOptions topts;
    topts.stage = to_valhall_stage(env.role);
    topts.arch = (env.arch != 0) ? env.arch : 10;
    topts.tolerant_ssa = true;

    compiler::Pass1Result p1 = compiler::run_pass1_lowering(primary_cfg, empty_cfg, topts);
    if (!p1.ok) {
        return res;
    }

    // 2. Run pure SSA backwards liveness + forward SSA pointer analysis
    dataflow::SsaLivenessResult liveness = dataflow::run_ssa_liveness(p1.ssa);
    res.df = dataflow::run_ssa_pointer_analysis(p1.ssa, env, &liveness);

    // 3. Project SSA-level findings back onto originating Valhall ISA instruction indices
    for (size_t id = 1; id < p1.ssa.nodes.size(); ++id) {
        const auto& node = p1.ssa.nodes[id];
        if (node.orig_inst_idx < 0 || static_cast<size_t>(node.orig_inst_idx) >= instructions.size()) {
            continue;
        }
        const size_t idx = static_cast<size_t>(node.orig_inst_idx);

        for (compiler::SsaId arg : node.args) {
            dataflow::TrackedVal av = res.df.val(arg);
            if (const auto* atom = av.inner.get_if<dataflow::ShaderAtomDomain>();
                atom && atom->is_known() && atom->value.kind == dataflow::ShaderAtomDomain::Kind::AbiPreload) {
                append_unique_note(res.inst_notes[idx], "ABI: " + atom->value.label);
            } else if (const auto* ptr = av.inner.get_if<dataflow::PointerDomain>();
                       ptr && (ptr->kind == dataflow::PointerVal::Kind::FauDstBase ||
                               ptr->kind == dataflow::PointerVal::Kind::SrtBase)) {
                append_unique_note(res.inst_notes[idx], ptr->label);
            }
        }

        if (id < res.df.ssa_notes.size()) {
            for (const auto& note : res.df.ssa_notes[id]) {
                append_unique_note(res.inst_notes[idx], note);
            }
        }
    }

    return res;
}

dataflow::PointerAnalysisEnv build_df_env(
    const Mbs2SymbolIndex& sym_idx,
    const ExtractedEbin& ebin,
    const PilotAnalysisResult* pilot_info) {
    dataflow::PointerAnalysisEnv env;
    env.role = to_df_role(ebin.role);
    env.data_section_offset = ebin.meta.data_section_offset;

    for (const auto& kv : sym_idx.fcst_by_fau_off) {
        env.initial_fau[kv.first] = dataflow::TrackedValDomain::make_const_u64(kv.second.value);
    }

    if (pilot_info) {
        for (const auto& kv : pilot_info->fau_map) {
            const auto& pe = kv.second;
            if (pe.srt_rack >= 0 && pe.desc_index >= 0 && pe.byte_size == 8) {
                env.initial_fau[pe.fau_offset] = dataflow::TrackedValDomain::make_buffer_addr(
                    pe.srt_rack, pe.desc_index, pe.symbol_name, 0, false);
            } else {
                env.initial_fau[pe.fau_offset] = dataflow::TrackedValDomain::make_symbol_value(
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
        IsaProjectedAnalysis proj = lower_and_run_ssa_pointer_analysis(eb.instructions, env);

        for (const auto& kv : proj.df.pilot_fau_writes) {
            const auto& pw = kv.second;
            PilotFauEntry entry;
            entry.fau_offset = pw.fau_offset;
            entry.byte_size = pw.byte_size;
            entry.expression = dataflow::format_tracked_val(pw.value);
            if (const auto* ptr = pw.value.inner.get_if<dataflow::PointerDomain>()) {
                entry.symbol_name = ptr->symbol_name;
                entry.srt_rack = ptr->rack;
                entry.desc_index = ptr->desc_idx;
                if (ptr->kind == dataflow::PointerVal::Kind::BufferAddr) {
                    entry.value_kind = PilotValueKind::BufferAddr;
                } else if (ptr->kind == dataflow::PointerVal::Kind::DescriptorAddr) {
                    entry.value_kind = PilotValueKind::DescriptorAddr;
                }
                entry.has_pointer_offset = ptr->offset_known;
                entry.pointer_offset = ptr->byte_offset;
                entry.is_derived = ptr->derived;
            } else if (const auto* sym = pw.value.inner.get_if<dataflow::SymbolValueDomain>()) {
                entry.value_kind = PilotValueKind::SymbolValue;
                entry.symbol_name = sym->symbol_name;
                entry.srt_rack = sym->rack;
                entry.desc_index = sym->desc_idx;
                entry.is_loaded_value = true;
                entry.has_byte_offset = sym->offset_known;
                entry.byte_offset = sym->byte_offset;
                entry.is_derived = sym->derived;
            } else if (const auto* ci = pw.value.inner.get_if<dataflow::ConstIntDomain>();
                       ci && ci->is_known()) {
                entry.value_kind = PilotValueKind::ConstInt;
                entry.has_const_value = true;
                entry.const_value = ci->value;
            } else if (const auto* atom = pw.value.inner.get_if<dataflow::ShaderAtomDomain>();
                       atom && atom->is_known()) {
                entry.symbol_name = atom->value.symbol_name;
                entry.srt_rack = atom->value.rack;
                entry.desc_index = atom->value.desc_idx;
                if (atom->value.kind == dataflow::ShaderAtomDomain::Kind::BufferSize) {
                    entry.value_kind = PilotValueKind::BufferSize;
                } else if (atom->value.kind == dataflow::ShaderAtomDomain::Kind::AbiPreload) {
                    entry.value_kind = PilotValueKind::AbiPreload;
                }
            }
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

    // Explicitly lower ISA -> SSA here, run pure SSA liveness + pointer analysis, and project back
    dataflow::PointerAnalysisEnv env = build_df_env(sym_idx, ebin, &pilot_info);
    IsaProjectedAnalysis proj = lower_and_run_ssa_pointer_analysis(ebin.instructions, env);

    for (size_t idx = 0; idx < ebin.instructions.size(); ++idx) {
        const auto& ins = ebin.instructions[idx];
        if (!ins.valid) continue;

        std::vector<std::string> tags;
        std::set<std::string> seen;
        auto add_tag = [&](const std::string& t) {
            if (!t.empty() && seen.insert(t).second) tags.push_back(t);
        };

        // 1. Tag FAU uniform operands directly from Pilot + MBS2 RLOC/FCST metadata
        for (const auto& s : ins.srcs) {
            if (s.kind == SourceKind::FauUniform) {
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

        // 2. Append semantic dataflow annotations projected from SSA-level PointerAnalysis
        if (idx < proj.inst_notes.size()) {
            for (const auto& ev : proj.inst_notes[idx]) {
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

dataflow::FauSummaryMap build_fau_summary_for_ebin(
    const Mbs2Module& mbs2,
    const std::vector<ExtractedEbin>& ebins,
    uint8_t arch,
    const ExtractedEbin& primary_eb) {
    dataflow::FauSummaryMap summary;
    Mbs2SymbolIndex lut = build_symbol_index(mbs2, &primary_eb);

    for (const auto& eb : ebins) {
        if (eb.role != EbinRole::Pilot) continue;
        Mbs2SymbolIndex pilot_lut = build_symbol_index(mbs2, &eb);
        dataflow::PointerAnalysisEnv pilot_env = build_df_env(pilot_lut, eb, nullptr);
        cfg::ControlFlowGraph pilot_cfg = cfg::build_cfg(eb.instructions);
        ValhallTranslateOptions p_opts;
        p_opts.stage = valhall::ShaderStage::Pilot;
        p_opts.arch = arch;
        p_opts.tolerant_ssa = true;
        auto p1 = compiler::run_pass1_lowering(pilot_cfg, {}, p_opts);
        if (p1.ok) {
            dataflow::FauSummaryMap p_sum = dataflow::run_pilot_fau_summary(p1.ssa, pilot_env);
            for (const auto& kv : p_sum) {
                summary[kv.first] = kv.second;
            }
        }
        break;
    }

    std::map<uint32_t, uint32_t> fcst_words;
    for (const auto& kv : lut.fcst_by_fau_off) {
        fcst_words[kv.first] = kv.second.value;
    }
    dataflow::seed_fcst_into_fau_summary(fcst_words, summary, /*overwrite=*/false);
    return summary;
}

}  // namespace valhall::spirv2valhall

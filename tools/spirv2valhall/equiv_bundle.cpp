#include "equiv_bundle.h"
#include "valhall_cfg.h"
#include "valhall_pass1.h"
#include "valhall_ssa_json.h"

#include <cstdio>
#include <map>
#include <sstream>
#include <utility>

namespace valhall::spirv2valhall {

namespace {

void write_json_str(std::ostringstream& os, const std::string& s) {
    os << '"';
    for (char c : s) {
        switch (c) {
            case '"': os << "\\\""; break;
            case '\\': os << "\\\\"; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20u) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(c)));
                    os << buf;
                } else {
                    os << c;
                }
                break;
        }
    }
    os << '"';
}

const char* stage_name(ShaderStage st) {
    switch (st) {
        case ShaderStage::Compute: return "Compute";
        case ShaderStage::Vertex: return "Vertex";
        case ShaderStage::Fragment: return "Fragment";
        default: return "Unknown";
    }
}

valhall::ShaderStage to_compiler_stage(ShaderStage st) {
    switch (st) {
        case ShaderStage::Vertex: return valhall::ShaderStage::Vertex;
        case ShaderStage::Fragment: return valhall::ShaderStage::Fragment;
        case ShaderStage::Compute:
        default: return valhall::ShaderStage::Compute;
    }
}

const char* resource_kind_name(spirv2ssa::ResourceKind k) {
    using spirv2ssa::ResourceKind;
    switch (k) {
        case ResourceKind::UniformBuffer: return "UBO";
        case ResourceKind::StorageBuffer: return "SSBO";
        case ResourceKind::PushConstant: return "PushConstant";
        case ResourceKind::SampledImage: return "SampledImage";
        case ResourceKind::InputAttrOrVarying: return "InputAttrOrVarying";
        case ResourceKind::OutputColorOrVarying: return "OutputColorOrVarying";
        case ResourceKind::BuiltinInput: return "BuiltinInput";
        case ResourceKind::BuiltinOutput: return "BuiltinOutput";
    }
    return "Unknown";
}

/// Find the MBS2 symbol matching a given (rack, desc_index) or symbol name, returning its Vulkan binding.
bool lookup_mbs2_buffer_binding(
    const Mbs2Module& mod,
    int32_t rack,
    int32_t desc_index,
    const std::string& sym_name,
    std::string* out_kind,
    uint32_t* out_binding) {
    auto match_list = [&](const std::vector<Mbs2Symbol>& syms, const char* kind_str) -> bool {
        for (const auto& s : syms) {
            bool by_addr = (rack >= 0 && desc_index >= 0 &&
                            static_cast<int32_t>(s.rack_index) == rack &&
                            static_cast<int32_t>(s.address) == desc_index);
            bool by_name = (!sym_name.empty() && s.name == sym_name);
            if (by_addr || by_name) {
                if (out_kind) *out_kind = kind_str;
                if (out_binding) *out_binding = (s.binding != 0xFFFFu) ? s.binding : s.address;
                return true;
            }
        }
        return false;
    };
    if (match_list(mod.cmmn.s3bo, "ssbo")) return true;
    if (match_list(mod.cmmn.suni, "ubo")) return true;
    return false;
}

} // namespace

void populate_mbs2_translate_options(
    const ExtractResult& extract_res,
    const ExtractedEbin& primary_eb,
    ValhallTranslateOptions* vopts,
    bool remap_varyings) {
    if (!vopts) return;

    if (extract_res.mbs2.has_kern && extract_res.mbs2.kern.has_workgroup_size) {
        vopts->local_size_x = extract_res.mbs2.kern.wg_x;
        vopts->local_size_y = extract_res.mbs2.kern.wg_y;
        vopts->local_size_z = extract_res.mbs2.kern.wg_z;
    }

    if (remap_varyings) {
        for (const auto& sym : extract_res.mbs2.cmmn.sinp) {
            if (sym.rack_index == 11u && sym.location != 0xFFFFu) {
                vopts->frag_varying_addr_to_loc[sym.address] = sym.location;
            }
        }
    }
    for (const auto& sym : extract_res.mbs2.cmmn.sout) {
        if (sym.rack_index == 3u && !sym.name.empty()) {
            vopts->vert_sout_attr_names[sym.address] = sym.name;
        }
    }

    std::map<uint32_t, uint32_t> simg_addrs;
    for (const auto& sym : extract_res.mbs2.cmmn.simg) {
        valhall::CombinedSamplerBindingSpec spec;
        spec.tex_index = sym.address;
        spec.descriptor_set = (sym.rack_index >= 68u) ? (sym.rack_index - 68u)
                            : (sym.rack_index >= 4u ? (sym.rack_index - 4u) : 0u);
        spec.binding = (sym.binding != 0xFFFFu) ? sym.binding : sym.address;
        vopts->samplers.push_back(spec);
        simg_addrs[sym.address] = spec.binding;
    }

    std::map<uint32_t, uint32_t> fcst_all;
    for (const auto& f : extract_res.mbs2.cmmn.fcsts) fcst_all[f.reg_offset] = f.value;
    if (primary_eb.meta.has_dbug) {
        for (const auto& f : primary_eb.meta.dbug.fcsts) fcst_all[f.reg_offset] = f.value;
    }

    {
        std::map<std::pair<uint32_t, uint32_t>, std::string> pka_buf_by_rack_addr;
        auto add_pka = [&](const std::vector<Mbs2Symbol>& syms, bool is_ssbo) {
            for (const auto& sym : syms) {
                std::string bname = sym.name;
                if (!is_ssbo && sym.rack_index == 0u && sym.binding == 0xFFFFu) {
                    bname = "push_const";
                } else if (sym.binding != 0xFFFFu) {
                    uint32_t d_set = (sym.rack_index >= 4u) ? (sym.rack_index - 4u) : 0u;
                    bname = std::string(is_ssbo ? "ssbo:" : "ubo:") +
                            std::to_string(d_set) + ":" + std::to_string(sym.binding);
                }
                pka_buf_by_rack_addr[{sym.rack_index, sym.address}] = std::move(bname);
            }
        };
        add_pka(extract_res.mbs2.cmmn.suni, false);
        add_pka(extract_res.mbs2.cmmn.s3bo, true);
        for (const auto& kv : fcst_all) {
            uint32_t rack = (kv.second >> 24u) & 0xFFu;
            uint32_t addr = kv.second & 0xFFFFu;
            auto it = pka_buf_by_rack_addr.find({rack, addr});
            if (it != pka_buf_by_rack_addr.end()) {
                vopts->fau_pka_buf_names[kv.first] = it->second;
            }
        }
    }

    if (!simg_addrs.empty()) {
        for (const auto& kv : fcst_all) {
            if ((kv.first & 7u) != 0u) continue;
            auto hi_it = fcst_all.find(kv.first + 4u);
            if (hi_it == fcst_all.end()) continue;
            if (kv.second == 0xFFFFFFFFu) {
                uint32_t desc_addr = hi_it->second & 0xFFFFu;
                if (simg_addrs.count(desc_addr)) {
                    vopts->fau_tex_handle_indices[kv.first] = desc_addr;
                }
            }
        }
    }
}

bool build_equivalence_bundle_json(
    const std::string& spv_path,
    const ExtractResult& extract_res,
    std::string* out_json,
    std::string* err,
    const EquivBundleOptions& opt) {
    if (!out_json) {
        if (err) *err = "Null out_json pointer";
        return false;
    }

    // 1. Direct SPIR-V -> ValhallSSA lowering
    std::vector<uint32_t> spv_words;
    if (!read_spirv_words(spv_path, &spv_words, err)) return false;
    spirv2ssa::Spirv2SsaOptions dopts;
    dopts.tolerant = true;
    dopts.run_ssa_opt = opt.run_ssa_opt;
    dopts.ssa_opt_fp_fast_math = opt.fp_fast_math;
    dopts.ssa_opt_fp_tile_rewrites = opt.enable_fp_tile_rewrites;
    dopts.ssa_opt_fp_cpp_rewrites = opt.enable_fp_cpp_rewrites;
    auto direct_res = spirv2ssa::lower_spirv_to_valhall_ssa(spv_words, dopts);

    // 2. Reference Valhall ISA -> valhall_pass1 -> ValhallSSA lowering
    const ExtractedEbin* primary_eb = nullptr;
    const ExtractedEbin* secondary_eb = nullptr;
    if (extract_res.stage_info.stage == ShaderStage::Compute) {
        for (const auto& eb : extract_res.ebins) {
            if (eb.role != EbinRole::Pilot) {
                primary_eb = &eb;
                break;
            }
        }
    } else if (extract_res.stage_info.stage == ShaderStage::Vertex) {
        for (const auto& eb : extract_res.ebins) {
            if (eb.role == EbinRole::VertexVarying && !secondary_eb) secondary_eb = &eb;
            else if (!primary_eb) primary_eb = &eb;
        }
    } else {
        for (const auto& eb : extract_res.ebins) {
            if (eb.role != EbinRole::Pilot) {
                primary_eb = &eb;
                break;
            }
        }
    }

    if (!primary_eb) {
        if (err) *err = "No primary EBIN found in ExtractResult";
        return false;
    }

    ValhallTranslateOptions vopts;
    vopts.stage = to_compiler_stage(extract_res.stage_info.stage);
    vopts.arch = extract_res.arch;
    vopts.tolerant_ssa = true;
    vopts.run_ssa_opt = opt.run_ssa_opt;
    vopts.ssa_opt_fp_fast_math = opt.fp_fast_math;
    vopts.ssa_opt_fp_tile_rewrites = opt.enable_fp_tile_rewrites;
    vopts.ssa_opt_fp_cpp_rewrites = opt.enable_fp_cpp_rewrites;
    populate_mbs2_translate_options(extract_res, *primary_eb, &vopts, /*remap_varyings=*/false);

    cfg::ControlFlowGraph pri_cfg = cfg::build_cfg(primary_eb->instructions);
    cfg::ControlFlowGraph sec_cfg;
    if (secondary_eb) {
        sec_cfg = cfg::build_cfg(secondary_eb->instructions);
    }

    compiler::Pass1Result p1_res = compiler::run_pass1_lowering(pri_cfg, sec_cfg, vopts);

    // 3. Assemble JSON bundle
    std::ostringstream os;
    os << "{\"spv_path\":";
    write_json_str(os, spv_path);
    os << ",\"stage\":";
    write_json_str(os, stage_name(extract_res.stage_info.stage));
    os << ",\"arch\":" << static_cast<unsigned>(extract_res.arch)
       << ",\"direct_ok\":" << (direct_res.ok ? "true" : "false")
       << ",\"valhall_ok\":" << (p1_res.ok ? "true" : "false");

    if (!direct_res.error_message.empty()) {
        os << ",\"direct_error\":";
        write_json_str(os, direct_res.error_message);
    }
    if (!p1_res.error_message.empty()) {
        os << ",\"valhall_error\":";
        write_json_str(os, p1_res.error_message);
    }

    os << ",\"direct_unsupported\":[";
    for (size_t i = 0; i < direct_res.unsupported_ops.size(); ++i) {
        if (i > 0) os << ',';
        write_json_str(os, direct_res.unsupported_ops[i]);
    }
    os << ']';

    // Direct resource bindings
    os << ",\"direct_bindings\":[";
    for (size_t i = 0; i < direct_res.bindings.size(); ++i) {
        const auto& b = direct_res.bindings[i];
        if (i > 0) os << ',';
        os << "{\"kind\":\"" << resource_kind_name(b.kind) << '"'
           << ",\"sideband_slot\":" << b.sideband_slot
           << ",\"set\":" << b.descriptor_set
           << ",\"binding\":" << b.binding
           << ",\"location\":" << b.location
           << ",\"component\":" << b.component
           << ",\"builtin\":" << b.builtin
           << ",\"name\":";
        write_json_str(os, b.name);
        os << ",\"key\":";
        write_json_str(os, b.canonical_key);
        os << '}';
    }
    os << ']';

    // MBS2 execution context (local_size, FCSTs, RLOCs, Pilot FAU, sinp/sout, pc_disasm)
    os << ",\"mbs2_context\":{";
    if (extract_res.mbs2.has_kern && extract_res.mbs2.kern.has_workgroup_size) {
        os << "\"local_size\":[" << extract_res.mbs2.kern.wg_x
           << ',' << extract_res.mbs2.kern.wg_y
           << ',' << extract_res.mbs2.kern.wg_z << "],";
    }

    // FAU FCSTs (module-level + primary EBIN dbug)
    os << "\"fcsts\":[";
    bool first_fcst = true;
    auto emit_fcst = [&](const Mbs2Fcst& fc) {
        if (!first_fcst) os << ',';
        first_fcst = false;
        os << "{\"reg_offset\":" << fc.reg_offset
           << ",\"word_idx\":" << (fc.reg_offset / 4u)
           << ",\"size\":" << fc.size
           << ",\"value\":" << fc.value << '}';
    };
    for (const auto& fc : extract_res.mbs2.cmmn.fcsts) emit_fcst(fc);
    if (primary_eb->index < extract_res.mbs2.cmmn.ebins.size()) {
        const auto& em = extract_res.mbs2.cmmn.ebins[primary_eb->index];
        if (em.has_dbug) {
            for (const auto& fc : em.dbug.fcsts) emit_fcst(fc);
        }
    }
    os << ']';

    // FAU RLOCs from suni
    os << ",\"rlocs\":[";
    bool first_rloc = true;
    for (const auto& u : extract_res.mbs2.cmmn.suni) {
        uint32_t binding = (u.binding != 0xFFFFu) ? u.binding : u.address;
        for (const auto& r : u.rlocs) {
            if (!first_rloc) os << ',';
            first_rloc = false;
            os << "{\"symbol\":";
            write_json_str(os, u.name);
            os << ",\"binding\":" << binding
               << ",\"reg_offset\":" << r.reg_offset
               << ",\"word_idx\":" << (r.reg_offset / 4u)
               << ",\"uni_offset\":" << r.uni_offset
               << ",\"size\":" << r.size << '}';
        }
    }
    os << ']';

    // Pilot FAU entries
    os << ",\"pilot_fau\":[";
    bool first_pf = true;
    for (const auto& [off, entry] : extract_res.pilot_analysis.fau_map) {
        if (!first_pf) os << ',';
        first_pf = false;
        std::string kind = "unknown";
        uint32_t binding = 0;
        bool resolved = lookup_mbs2_buffer_binding(
            extract_res.mbs2, entry.srt_rack, entry.desc_index, entry.symbol_name, &kind, &binding);
        auto pilot_vk_name = [](PilotValueKind vk) -> const char* {
            switch (vk) {
                case PilotValueKind::BufferAddr: return "buffer_addr";
                case PilotValueKind::BufferSize: return "buffer_size";
                case PilotValueKind::DescriptorAddr: return "descriptor_addr";
                case PilotValueKind::SymbolValue: return "symbol_value";
                case PilotValueKind::ConstInt: return "const_int";
                case PilotValueKind::AbiPreload: return "abi_preload";
                default: return "unknown";
            }
        };
        os << "{\"fau_offset\":" << off
           << ",\"word_idx\":" << (off / 4u)
           << ",\"byte_size\":" << entry.byte_size
           << ",\"value_kind\":";
        write_json_str(os, pilot_vk_name(entry.value_kind));
        os << ",\"symbol\":";
        write_json_str(os, entry.symbol_name);
        os << ",\"expression\":";
        write_json_str(os, entry.expression);
        os << ",\"resolved\":" << (resolved ? "true" : "false")
           << ",\"kind\":";
        write_json_str(os, kind);
        os << ",\"binding\":" << binding
           << ",\"is_loaded_value\":" << (entry.is_loaded_value ? "true" : "false")
           << ",\"has_byte_offset\":" << (entry.has_byte_offset ? "true" : "false")
           << ",\"byte_offset\":" << entry.byte_offset
           << ",\"is_derived\":" << (entry.is_derived ? "true" : "false")
           << ",\"has_pointer_offset\":" << (entry.has_pointer_offset ? "true" : "false")
           << ",\"pointer_offset\":" << entry.pointer_offset
           << ",\"has_const_value\":" << (entry.has_const_value ? "true" : "false")
           << ",\"const_value\":" << entry.const_value
           << ",\"key\":";
        write_json_str(os, resolved ? (kind + ":0:" + std::to_string(binding)) : "");
        os << '}';
    }
    os << ']';

    // Shader inputs (sinp) mapping Valhall slot/addr -> SPIR-V location
    os << ",\"sinp\":[";
    for (size_t i = 0; i < extract_res.mbs2.cmmn.sinp.size(); ++i) {
        const auto& s = extract_res.mbs2.cmmn.sinp[i];
        if (i > 0) os << ',';
        os << "{\"name\":";
        write_json_str(os, s.name);
        os << ",\"addr\":" << s.address
           << ",\"location\":" << (s.location != 0xFFFFu ? static_cast<int32_t>(s.location) : -1)
           << ",\"rack\":" << static_cast<unsigned>(s.rack_index) << '}';
    }
    os << ']';

    // Shader outputs (sout) mapping Valhall slot/addr -> SPIR-V location
    os << ",\"sout\":[";
    for (size_t i = 0; i < extract_res.mbs2.cmmn.sout.size(); ++i) {
        const auto& s = extract_res.mbs2.cmmn.sout[i];
        if (i > 0) os << ',';
        os << "{\"name\":";
        write_json_str(os, s.name);
        os << ",\"addr\":" << s.address
           << ",\"location\":" << (s.location != 0xFFFFu ? static_cast<int32_t>(s.location) : -1)
           << ",\"rack\":" << static_cast<unsigned>(s.rack_index) << '}';
    }
    os << ']';

    // Pilot SRT descriptor table: (rack, address) -> canonical buffer key
    os << ",\"pilot_srt\":[";
    bool first_srt = true;
    auto emit_srt = [&](const std::vector<Mbs2Symbol>& syms, const char* kind_str) {
        for (const auto& s : syms) {
            if (!first_srt) os << ',';
            first_srt = false;
            uint32_t d_set = (s.rack_index >= 4u) ? (s.rack_index - 4u) : 0u;
            uint32_t bind = (s.binding != 0xFFFFu) ? s.binding : s.address;
            std::string key;
            if (std::string(kind_str) == "ubo" && s.rack_index == 0u && s.binding == 0xFFFFu) {
                key = "push_const";
            } else {
                key = std::string(kind_str) + ":" + std::to_string(d_set) + ":" + std::to_string(bind);
            }
            os << "{\"rack\":" << static_cast<unsigned>(s.rack_index)
               << ",\"addr\":" << s.address
               << ",\"kind\":\"" << kind_str << '"'
               << ",\"key\":";
            write_json_str(os, key);
            os << '}';
        }
    };
    emit_srt(extract_res.mbs2.cmmn.suni, "ubo");
    emit_srt(extract_res.mbs2.cmmn.s3bo, "ssbo");
    os << ']';

    // Embedded read-only data section (LUTs / constant tables addressed via PC-relative ADR)
    if (primary_eb != nullptr && !primary_eb->data_section.empty() &&
        primary_eb->meta.data_section_offset != 0xFFFFFFFFu) {
        static constexpr char kHex[] = "0123456789abcdef";
        os << ",\"ebin_rodata\":{\"offset\":" << primary_eb->meta.data_section_offset
           << ",\"size\":" << primary_eb->data_section.size()
           << ",\"hex\":\"";
        for (uint8_t byte : primary_eb->data_section) {
            os << kHex[(byte >> 4) & 0xFu] << kHex[byte & 0xFu];
        }
        os << "\"}";
    }

    // Disassembly map: orig_pc -> formatted Valhall ISA instruction string
    os << ",\"pc_disasm\":{";
    for (size_t i = 0; i < primary_eb->instructions.size(); ++i) {
        const auto& ins = primary_eb->instructions[i];
        if (i > 0) os << ',';
        os << '"' << ins.pc << "\":";
        write_json_str(os, ins.disassembly);
    }
    os << '}';
    os << '}'; // end mbs2_context

    // Serialized SSA modules
    os << ",\"direct_ssa\":" << compiler::serialize_ssa_module_json(direct_res.ssa);
    os << ",\"valhall_ssa\":" << compiler::serialize_ssa_module_json(p1_res.ssa);
    for (const auto& eb : extract_res.ebins) {
        if (eb.role == EbinRole::Pilot && !eb.instructions.empty()) {
            cfg::ControlFlowGraph pilot_cfg = cfg::build_cfg(eb.instructions);
            cfg::ControlFlowGraph empty_cfg;
            ValhallTranslateOptions pilot_opts;
            pilot_opts.stage = valhall::ShaderStage::Pilot;
            pilot_opts.arch = vopts.arch;
            pilot_opts.tolerant_ssa = true;
            compiler::Pass1Result pilot_p1 = compiler::run_pass1_lowering(pilot_cfg, empty_cfg, pilot_opts);
            if (pilot_p1.ok) {
                os << ",\"pilot_ssa\":" << compiler::serialize_ssa_module_json(pilot_p1.ssa);
            }
            break;
        }
    }
    os << '}';

    *out_json = os.str();
    return true;
}

} // namespace valhall::spirv2valhall

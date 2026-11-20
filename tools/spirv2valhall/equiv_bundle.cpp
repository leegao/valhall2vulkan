#include "equiv_bundle.h"
#include "valhall_cfg.h"
#include "valhall_pass1.h"
#include "valhall_ssa_json.h"

#include <cstdio>
#include <sstream>

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

bool build_equivalence_bundle_json(
    const std::string& spv_path,
    const ExtractResult& extract_res,
    std::string* out_json,
    std::string* err) {
    if (!out_json) {
        if (err) *err = "Null out_json pointer";
        return false;
    }

    // 1. Direct SPIR-V -> ValhallSSA lowering
    std::vector<uint32_t> spv_words;
    if (!read_spirv_words(spv_path, &spv_words, err)) return false;
    spirv2ssa::Spirv2SsaOptions dopts;
    dopts.tolerant = true;
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
    if (extract_res.mbs2.has_kern && extract_res.mbs2.kern.has_workgroup_size) {
        vopts.local_size_x = extract_res.mbs2.kern.wg_x;
        vopts.local_size_y = extract_res.mbs2.kern.wg_y;
        vopts.local_size_z = extract_res.mbs2.kern.wg_z;
    }

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

    // MBS2 execution context (FCSTs, RLOCs, Pilot FAU, sinp/sout, pc_disasm)
    os << ",\"mbs2_context\":{";

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
        os << "{\"fau_offset\":" << off
           << ",\"word_idx\":" << (off / 4u)
           << ",\"byte_size\":" << entry.byte_size
           << ",\"symbol\":";
        write_json_str(os, entry.symbol_name);
        os << ",\"expression\":";
        write_json_str(os, entry.expression);
        os << ",\"resolved\":" << (resolved ? "true" : "false")
           << ",\"kind\":";
        write_json_str(os, kind);
        os << ",\"binding\":" << binding
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
    os << '}';

    *out_json = os.str();
    return true;
}

} // namespace valhall::spirv2valhall

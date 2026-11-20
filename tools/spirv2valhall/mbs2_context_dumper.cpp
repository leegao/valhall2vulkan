#include "mbs2_context_dumper.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <map>
#include <sstream>

namespace valhall::spirv2valhall {

namespace {

const char* vela_lang_name(uint32_t lang) {
    switch (lang) {
        case 1: return "GLSL_ES/1";
        case 2: return "GLSL/2";
        case 3: return "OpenCL/3";
        case 7: return "SPIRV (0x7)";
        default: return "Unknown";
    }
}

const char* tpib_layout_name(uint32_t layout) {
    return (layout == 1) ? "std430" : "std140";
}

const char* tpib_kind_name(uint32_t kind) {
    switch (kind) {
        case 0: return "UBO";
        case 1: return "SSBO";
        case 2: return "InputBlock";
        case 3: return "OutputBlock";
        default: return "Block";
    }
}

std::string format_tpge(const Mbs2Tpge& g) {
    static const char* kBits[] = {"8", "16", "32", "64"};
    const char* bits = (g.scalar_size < 4) ? kBits[g.scalar_size] : "?";
    std::string base;
    if (g.vector_size <= 1) {
        switch (g.scalar_type) {
            case 1: base = (g.scalar_size == 2) ? "float" : (std::string("f") + bits); break;
            case 2: base = (g.scalar_size == 2) ? "int" : (std::string("i") + bits); break;
            case 3: base = (g.scalar_size == 2) ? "uint" : (std::string("u") + bits); break;
            case 4: base = "bool"; break;
            default: base = "scalar"; break;
        }
    } else {
        std::string pfx;
        switch (g.scalar_type) {
            case 1: pfx = (g.scalar_size == 3) ? "d" : ""; break;
            case 2: pfx = "i"; break;
            case 3: pfx = "u"; break;
            case 4: pfx = "b"; break;
            default: pfx = "s"; break;
        }
        base = pfx + "vec" + std::to_string(g.vector_size);
        if (g.scalar_size != 2 && !(g.scalar_type == 1 && g.scalar_size == 3)) {
            base += "(" + std::string(bits) + "b)";
        }
    }
    return base;
}

std::string format_fcst_val(uint32_t v) {
    float f = 0.0f;
    std::memcpy(&f, &v, sizeof(float));
    char buf[96];
    uint32_t exp = (v >> 23) & 0xFFu;
    if (exp > 0 && exp < 255) {
        std::snprintf(buf, sizeof(buf), "0x%08x (f32=%.6g)", v, static_cast<double>(f));
    } else {
        std::snprintf(buf, sizeof(buf), "0x%08x (u32=%u)", v, v);
    }
    return buf;
}

void dump_symbol_section(
    std::ostringstream& oss,
    const char* section_label,
    const std::vector<Mbs2Symbol>& syms) {
    if (syms.empty()) return;
    oss << ";   " << section_label << " (" << syms.size() << "):\n";
    for (size_t i = 0; i < syms.size(); ++i) {
        const auto& s = syms[i];
        oss << ";     [" << i << "] \"" << s.name << "\""
            << " rack=" << static_cast<int>(s.rack_index)
            << " (SRT+0x" << std::hex << (static_cast<uint32_t>(s.rack_index) * 16u) << ")"
            << " addr=" << std::dec << s.address
            << " (desc+0x" << std::hex << (s.address * 32u) << ")" << std::dec;
        if (s.binding != 0xFFFFu) oss << " binding=" << s.binding;
        if (s.location != 0xFFFFu) oss << " loc=" << s.location;
        oss << " flags=0x" << std::hex << s.flags << std::dec;
        if (s.type) {
            oss << " type=" << format_mbs2_type(s.type.get());
        }
        oss << "\n";

        // Dump interface block or struct members
        const std::vector<Mbs2Tpse>* members = nullptr;
        if (s.type && s.type->has_tpib) members = &s.type->tpib.members;
        else if (s.type && s.type->has_tpst) members = &s.type->tpst.members;
        if (members && !members->empty()) {
            for (const auto& m : *members) {
                oss << ";           +0x" << std::hex << std::setw(2) << std::setfill('0') << m.offset
                    << std::dec << ": " << m.name << " : " << format_mbs2_type(m.type.get()) << "\n";
            }
        }
        // Dump RLOCs
        for (const auto& r : s.rlocs) {
            oss << ";           RLOC: FAU+0x" << std::hex << r.reg_offset << std::dec
                << " (" << format_fau_reg(r.reg_offset, r.size) << ")"
                << " <- uni+0x" << std::hex << r.uni_offset << std::dec
                << " size=" << r.size << "B\n";
        }
        if (s.has_symd) {
            oss << ";           SYMD: size=" << s.symd.size << " align=" << s.symd.alignment
                << " addrspace=" << s.symd.addressspace << "\n";
        }
    }
}

}  // namespace

std::string format_mbs2_type(const Mbs2Type* t) {
    if (!t) return "void";
    if (t->has_tpge) return format_tpge(t->tpge);
    if (t->has_tpma) {
        return "mat" + std::to_string(t->tpma.nof_columns) + "x" +
               std::to_string(t->tpma.column_type.vector_size) +
               (t->tpma.is_row_major ? "(row_major)" : "");
    }
    if (t->has_tpar) {
        std::string elem = format_mbs2_type(t->tpar.element_type.get());
        if (t->tpar.array_size == 0) return elem + "[]";
        return elem + "[" + std::to_string(t->tpar.array_size) + "]";
    }
    if (t->has_tpib) {
        std::ostringstream oss;
        oss << tpib_kind_name(t->tpib.kind) << "(" << tpib_layout_name(t->tpib.layout)
            << ", stride=" << t->tpib.stride << "B, members=" << t->tpib.members.size() << ")";
        return oss.str();
    }
    if (t->has_tpst) {
        return "struct " + (t->tpst.type_name.empty() ? "<anon>" : t->tpst.type_name) +
               "(stride=" + std::to_string(t->tpst.stride) + "B)";
    }
    if (t->has_tppo) {
        return format_mbs2_type(t->tppo.pointed_type.get()) + "*";
    }
    if (t->has_tpsa) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "sampler(kind=0x%x)", t->tpsa_kind);
        return buf;
    }
    if (t->has_tpac) {
        return "atomic_counter(off=" + std::to_string(t->tpac_offset) + ")";
    }
    return "opaque";
}

std::string format_fau_reg(uint32_t byte_offset, uint32_t byte_size) {
    uint32_t slot = byte_offset / 8u;
    uint32_t lane = (byte_offset % 8u) / 4u;
    if (byte_size >= 8 && lane == 0) {
        uint32_t end_slot = (byte_offset + byte_size - 1u) / 8u;
        if (end_slot == slot) {
            return "u" + std::to_string(slot);
        }
        return "u" + std::to_string(slot) + "..u" + std::to_string(end_slot);
    }
    if (byte_size > 4) {
        uint32_t end_off = byte_offset + byte_size - 4u;
        return "u" + std::to_string(slot) + ".w" + std::to_string(lane) +
               "..u" + std::to_string(end_off / 8u) + ".w" + std::to_string((end_off % 8u) / 4u);
    }
    return "u" + std::to_string(slot) + ".w" + std::to_string(lane);
}

std::string format_csf_execution_context(
    const Mbs2Module& mod,
    const PilotAnalysisResult& pilot_info) {
    std::ostringstream oss;
    oss << "; ============================================================================\n";
    oss << "; CSF Execution Context & Descriptor Graph (from cmpbe_chunk_MBS2, minus EBINs)\n";
    oss << "; ============================================================================\n";

    // 1. Container & Hardware Context
    oss << "; [MBS2 Header] version=" << mod.version
        << " stage_chunk=" << (mod.stage_chunk.empty() ? "UNKNOWN" : mod.stage_chunk)
        << " VEHW={core=0x" << std::hex << mod.vehw.core
        << ", major=" << std::dec << mod.vehw.major
        << ", minor=" << mod.vehw.minor << "}"
        << " VELA=" << vela_lang_name(mod.cmmn.vela_lang)
        << " ast_globals=" << mod.ast_global_count << "\n";

    // 2. Stage-Specific Execution Context
    if (mod.has_kern) {
        oss << "; [Compute KERN] entry=\"" << mod.kern.name << "\""
            << " WG_SIZE=(" << mod.kern.wg_x << ", " << mod.kern.wg_y << ", " << mod.kern.wg_z << ")"
            << " WLS=" << mod.kern.workgroup_local_size << "B"
            << " kpars=" << mod.kern.kpars.size() << "\n";
        for (const auto& kp : mod.kern.kpars) {
            oss << ";   KPAR: declared=\"" << kp.declared_name
                << "\" uniform=\"" << kp.uniform_name << "\" flags=0x" << std::hex << kp.flags << std::dec << "\n";
        }
    }
    if (mod.has_cfra) {
        oss << "; [Fragment CFRA] outputs=" << mod.cfra.output_vars.size();
        if (mod.cfra.has_lcal) oss << " LCAL=" << static_cast<int>(mod.cfra.local_storage_size) << "B";
        if (mod.cfra.has_bldm) oss << " BLDM=0x" << std::hex << mod.cfra.bldm_flags << std::dec;
        if (mod.cfra.has_spas) {
            oss << " SPAS={read_mask=0x" << std::hex << mod.cfra.spas_read_mask
                << ", write_mask=0x" << mod.cfra.spas_write_mask << "}" << std::dec;
        }
        oss << "\n";
        for (const auto& ov : mod.cfra.output_vars) {
            oss << ";   OUTV: loc=" << ov.location << " (slots=" << ov.nb_locations_used
                << ") \"" << ov.name << "\" : " << format_mbs2_type(ov.type.get()) << "\n";
        }
        for (const auto& ac : mod.cfra.alpha_info) {
            oss << ";   ACHN: rt=" << static_cast<int>(ac.rt) << " kind=" << ac.kind
                << " sampler=\"" << ac.sampler_name << "\" uniform=\"" << ac.uniform_name << "\"\n";
        }
    }
    if (mod.has_cver) {
        oss << "; [Vertex CVER]";
        if (mod.cver.has_vmvw) oss << " VMVW.num_views=" << static_cast<int>(mod.cver.num_views);
        if (mod.cver.has_vrsm) oss << " VRSM.mask=0x" << std::hex << mod.cver.vrs_modes_mask << std::dec;
        if (mod.cmmn.written_pos_fifo_attrs) oss << " pos_fifo_attrs=" << mod.cmmn.written_pos_fifo_attrs;
        oss << "\n";
    }
    if (mod.has_mtes) {
        oss << "; [Tessellation MTES] n_vertices=" << mod.mtes.n_vertices
            << " prim_mode=" << mod.mtes.primitive_mode
            << " spacing=" << mod.mtes.vertex_spacing
            << " ordering=" << mod.mtes.ordering
            << " point_mode=" << (mod.mtes.point_mode ? "true" : "false") << "\n";
    }
    if (mod.has_mgeo) {
        oss << "; [Geometry MGEO] invocations=" << mod.mgeo.n_invocations
            << " max_vertices=" << mod.mgeo.max_vertices
            << " in_prim=" << mod.mgeo.input_primitive_mode
            << " out_prim=" << mod.mgeo.output_primitive_mode << "\n";
    }
    if (mod.has_momd) {
        oss << "; [MOMD] prealloc_uni=" << static_cast<int>(mod.momd.prealloc_uni_buffers)
            << " prealloc_fau=" << static_cast<int>(mod.momd.prealloc_fau)
            << " setup_fau_in_pilot=" << static_cast<int>(mod.momd.setup_fau_in_pilot)
            << " robustness=" << static_cast<int>(mod.momd.robustness)
            << " per_sample=" << static_cast<int>(mod.momd.per_sample_shading)
            << " plocs=" << mod.momd.plocs.size() << "\n";
    }

    // 3. SPD / EBIN Metadata Summary (minus raw bytecode bytes)
    oss << "; [SPD / EBIN Metadata (" << mod.cmmn.ebins.size() << " shader binaries, raw bytecode omitted)]\n";
    for (const auto& em : mod.cmmn.ebins) {
        const char* reg_alloc = (em.nof_reg <= 32) ? "32_Per_Thread" : "64_Per_Thread";
        oss << ";   EBIN[" << em.index << "] " << ebin_role_name(em.role);
        if (!em.var_name.empty()) oss << " (" << em.var_name << ")";
        oss << ": rt_num=0x" << std::hex << em.rt_num
            << " aux_idx=0x" << em.aux_idx
            << " variant=0x" << em.variant
            << " flags=0x" << em.flags << std::dec
            << " regs=" << em.nof_reg << " (" << reg_alloc << ")"
            << " rmu=" << em.nof_rmu << " (FAU_0[63:56])"
            << " tls=" << em.tls_size << "B"
            << " code=" << em.code_size << "B";
        if (em.data_section_offset != 0xFFFFFFFFu && em.objc_total_size > em.code_size) {
            oss << " data_section=+0x" << std::hex << em.data_section_offset << std::dec
                << " (" << (em.objc_total_size - em.code_size) << "B)";
        }
        oss << "\n";

        if (em.has_spdf && !em.spdf.rt_indices.empty()) {
            oss << ";     SPDf: rts=[";
            for (size_t i = 0; i < em.spdf.rt_indices.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << "RT" << static_cast<int>(em.spdf.rt_indices[i]);
            }
            oss << "]\n";
        }
        if (em.has_pdsc) {
            oss << ";     PDSC: varying0=" << em.pdsc.varying0 << " varying1=" << em.pdsc.varying1
                << " primary_wait=" << static_cast<int>(em.pdsc.primary_shader_wait) << "\n";
        }
        if (em.has_vlkn) {
            oss << ";     VLKN: used_set_mask=0x" << std::hex << em.vlkn.used_set_bitmask << std::dec
                << " stack_layers=" << em.vlkn.stack_layers
                << " dynamic_outputs=" << static_cast<int>(em.vlkn.dynamic_outputs) << "\n";
        }
        if (em.has_fnmd) {
            oss << ";     FNMD: reg_mode=" << em.fnmd.reg_mode
                << " max_reg_pressure=" << em.fnmd.max_register_pressure;
            if (!em.fnmd.pilot_name.empty()) oss << " pilot=\"" << em.fnmd.pilot_name << "\"";
            oss << "\n";
        }
    }

    // 4. Implicit Hardware Preload / ABI Entry Register Map
    oss << "; [Hardware ABI Entry Registers (live at PC 0x0000 before clobber)]\n";
    if (mod.stage_chunk == "CCOM") {
        oss << ";   Main Compute: r55..r57 = WorkGroupID.xyz, r60..r62 = GlobalInvocationID.xyz\n";
        if (pilot_info.found_pilot) {
            oss << ";   Pilot Shader: r0:r1 = pilot_fau_dst_va (FAU_0), r2:r3 = srt_base_va (SRT_0)\n";
        }
    } else if (mod.stage_chunk == "CVER") {
        oss << ";   Vertex: r59 = VertexStreamIndex, r60 = VertexIndex, r61 = InstanceIndex\n";
    } else if (mod.stage_chunk == "CFRA") {
        oss << ";   Fragment: r59 = FragCoord.z, r60 = sample_coverage_mask, r61 = varying_barycentric\n";
    }

    // 5. Descriptor Graph (SRT -> Resource[rack] -> Buffer/Texture/Sampler/Attribute)
    oss << "; [CSF Descriptor Graph & MBS2 Symbols]\n";
    dump_symbol_section(oss, "Uniforms / Push Constants (suni)", mod.cmmn.suni);
    dump_symbol_section(oss, "Storage Buffers / SSBOs (s3bo)", mod.cmmn.s3bo);
    dump_symbol_section(oss, "Images / Textures (simg)", mod.cmmn.simg);
    dump_symbol_section(oss, "Samplers (ssmp)", mod.cmmn.ssmp);
    dump_symbol_section(oss, "Shader Inputs (sinp)", mod.cmmn.sinp);
    dump_symbol_section(oss, "Shader Outputs (sout)", mod.cmmn.sout);

    if (!mod.cmmn.ubuf_sizes.empty()) {
        oss << ";   UBUF sizes:";
        for (size_t i = 0; i < mod.cmmn.ubuf_sizes.size(); ++i) {
            oss << " [" << i << "]=" << mod.cmmn.ubuf_sizes[i] << "B";
        }
        oss << "\n";
    }
    if (mod.cmmn.has_cxfb) {
        oss << ";   CXFB strides: [" << mod.cmmn.cxfb.strides[0] << ", "
            << mod.cmmn.cxfb.strides[1] << ", " << mod.cmmn.cxfb.strides[2]
            << ", " << mod.cmmn.cxfb.strides[3] << "]\n";
    }

    // Visual SRT Rack -> Descriptor Tree
    std::map<uint8_t, std::vector<const Mbs2Symbol*>> srt_racks;
    for (const auto& s : mod.cmmn.suni) srt_racks[s.rack_index].push_back(&s);
    for (const auto& s : mod.cmmn.s3bo) srt_racks[s.rack_index].push_back(&s);
    for (const auto& s : mod.cmmn.simg) srt_racks[s.rack_index].push_back(&s);
    for (const auto& s : mod.cmmn.ssmp) srt_racks[s.rack_index].push_back(&s);
    if (!srt_racks.empty()) {
        oss << ";   SRT_0 Pointer Hierarchy:\n";
        for (auto& kv : srt_racks) {
            uint8_t rack = kv.first;
            auto& vec = kv.second;
            std::sort(vec.begin(), vec.end(),
                      [](const Mbs2Symbol* a, const Mbs2Symbol* b) { return a->address < b->address; });
            oss << ";     SRT_0 + 0x" << std::hex << std::setw(2) << std::setfill('0')
                << (static_cast<uint32_t>(rack) * 16u) << std::dec
                << " (Resource[" << static_cast<int>(rack) << "])\n";
            for (const Mbs2Symbol* sym : vec) {
                oss << ";       +-- Desc[" << sym->address << "] (+0x" << std::hex
                    << std::setw(2) << std::setfill('0') << (sym->address * 32u) << std::dec
                    << "): \"" << sym->name << "\"";
                if (sym->binding != 0xFFFFu) oss << " (binding=" << sym->binding << ")";
                oss << "\n";
            }
        }
    }

    // 6. Unified FAU Slot Map (Pilot-populated + RLOC + FCST)
    struct FauRow {
        uint32_t offset = 0;
        uint32_t size = 4;
        std::string source;
        std::string detail;
    };
    std::map<uint32_t, FauRow> fau_rows;
    for (const auto& kv : pilot_info.fau_map) {
        FauRow row;
        row.offset = kv.second.fau_offset;
        row.size = kv.second.byte_size;
        row.source = "PILOT";
        row.detail = kv.second.expression;
        fau_rows[row.offset] = std::move(row);
    }
    auto add_rlocs = [&](const std::vector<Mbs2Symbol>& syms) {
        for (const auto& s : syms) {
            for (const auto& r : s.rlocs) {
                FauRow row;
                row.offset = r.reg_offset;
                row.size = r.size;
                row.source = "RLOC";
                row.detail = s.name + " (uni+0x" +
                             ([&]() {
                                 char b[16];
                                 std::snprintf(b, sizeof(b), "%x", r.uni_offset);
                                 return std::string(b);
                             })() + ", " + std::to_string(r.size) + "B)";
                fau_rows[row.offset] = std::move(row);
            }
        }
    };
    add_rlocs(mod.cmmn.suni);
    add_rlocs(mod.cmmn.s3bo);
    for (const auto& f : mod.cmmn.fcsts) {
        FauRow row;
        row.offset = f.reg_offset;
        row.size = f.size;
        row.source = "FCST";
        row.detail = format_fcst_val(f.value);
        fau_rows[row.offset] = std::move(row);
    }

    if (!fau_rows.empty()) {
        oss << "; [FAU_0 Unified Uniform Map (" << fau_rows.size() << " entries)]\n";
        for (const auto& kv : fau_rows) {
            const auto& r = kv.second;
            oss << ";   FAU+0x" << std::hex << std::setw(2) << std::setfill('0') << r.offset << std::dec
                << " (" << format_fau_reg(r.offset, r.size) << ", " << r.size << "B)"
                << " [" << r.source << "] " << r.detail << "\n";
        }
    }

    return oss.str();
}

}  // namespace valhall::spirv2valhall

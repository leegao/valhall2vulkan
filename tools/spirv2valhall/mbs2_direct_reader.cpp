#include "mbs2_direct_reader.h"

#include <cstring>

namespace valhall::spirv2valhall {

namespace {

std::string read_stri(const cmpbe_chunk_STRI& s) {
    if (!s.data || s.size == 0) return "";
    size_t len = s.size;
    while (len > 0 && s.data[len - 1] == 0) --len;
    return std::string(reinterpret_cast<const char*>(s.data), len);
}

std::string read_stri_ptr(const cmpbe_chunk_STRI* s) {
    return s ? read_stri(*s) : "";
}

Mbs2Tpge read_tpge(const cmpbe_chunk_TPGE& g) {
    Mbs2Tpge out;
    out.scalar_type = static_cast<uint32_t>(g.scalar_type);
    out.vector_size = g.vector_size;
    out.scalar_size = static_cast<uint32_t>(g.scalar_size);
    out.precision = static_cast<uint32_t>(g.precision);
    out.stride = g.stride;
    out.aux_qualifier = static_cast<uint32_t>(g.aux_qualifier);
    out.component = g.component;
    return out;
}

std::shared_ptr<Mbs2Type> read_type(const cmpbe_chunk_TYPE& t, int depth = 0);

Mbs2Tpse read_tpse(const cmpbe_chunk_TPSE& m, int depth) {
    Mbs2Tpse out;
    out.name = read_stri(m.name);
    out.offset = m.offset;
    out.xfb_offset = m.xfb_offset;
    out.flags = static_cast<uint32_t>(m.flags);
    out.location = m.location;
    out.semantics = static_cast<uint32_t>(m.semantics);
    out.type = read_type(m.type, depth + 1);
    return out;
}

std::shared_ptr<Mbs2Type> read_type(const cmpbe_chunk_TYPE& t, int depth) {
    if (depth > 16) return nullptr;
    auto out = std::make_shared<Mbs2Type>();
    if (t.tpac) {
        out->has_tpac = true;
        out->tpac_offset = t.tpac->offset;
    }
    if (t.tpge) {
        out->has_tpge = true;
        out->tpge = read_tpge(*t.tpge);
    }
    if (t.tpma) {
        out->has_tpma = true;
        out->tpma.nof_columns = t.tpma->nof_columns;
        out->tpma.is_row_major = (t.tpma->is_row_major != 0);
        out->tpma.major_stride = t.tpma->major_stride;
        out->tpma.column_type = read_tpge(t.tpma->column_type);
    }
    if (t.tpar) {
        out->has_tpar = true;
        out->tpar.array_size = t.tpar->array_size;
        out->tpar.element_type = read_type(t.tpar->element_type, depth + 1);
    }
    if (t.tpib) {
        out->has_tpib = true;
        out->tpib.layout = static_cast<uint32_t>(t.tpib->layout);
        out->tpib.kind = static_cast<uint32_t>(t.tpib->kind);
        out->tpib.stride = t.tpib->stride;
        out->tpib.instance_name = read_stri_ptr(t.tpib->instance_name);
        if (t.tpib->tpse) {
            for (uint32_t i = 0; i < t.tpib->nof_members; ++i) {
                out->tpib.members.push_back(read_tpse(t.tpib->tpse[i], depth + 1));
            }
        }
    }
    if (t.tpst) {
        out->has_tpst = true;
        out->tpst.stride = t.tpst->stride;
        out->tpst.type_name = read_stri(t.tpst->type_name);
        if (t.tpst->tpse) {
            for (uint32_t i = 0; i < t.tpst->nof_members; ++i) {
                out->tpst.members.push_back(read_tpse(t.tpst->tpse[i], depth + 1));
            }
        }
    }
    if (t.tppo) {
        out->has_tppo = true;
        out->tppo.scalar_size = static_cast<uint32_t>(t.tppo->scalar_size);
        out->tppo.address_space = static_cast<uint32_t>(t.tppo->address_space);
        out->tppo.pointed_type = read_type(t.tppo->pointed_type, depth + 1);
    }
    if (t.tpsa) {
        out->has_tpsa = true;
        out->tpsa_kind = t.tpsa->kind;
    }
    if (t.tpmd) {
        out->has_tpmd = true;
        out->tpmd_flags = t.tpmd->flags;
    }
    return out;
}

Mbs2Rloc read_rloc(const cmpbe_chunk_RLOC& r) {
    return Mbs2Rloc{r.sym_idx, r.reg_offset, r.uni_offset, r.size};
}

Mbs2Relo read_relo(const cmpbe_chunk_RELO& r) {
    return Mbs2Relo{r.offset, static_cast<uint32_t>(r.kind), r.section_idx};
}

Mbs2Fcst read_fcst(const cmpbe_chunk_FCST& f) {
    return Mbs2Fcst{f.size, f.reg_offset, f.value};
}

Mbs2Symbol read_symb(const cmpbe_chunk_SYMB& s) {
    Mbs2Symbol out;
    out.name = read_stri(s.name);
    out.flags = static_cast<uint32_t>(s.flags);
    out.semantics = static_cast<uint32_t>(s.semantics);
    out.rack_index = s.rack_index;
    out.address = s.address;
    out.binding = s.binding;
    out.location = s.location;
    out.type = read_type(s.type);
    if (s.rloc) {
        for (uint32_t i = 0; i < s.nof_rloc; ++i) out.rlocs.push_back(read_rloc(s.rloc[i]));
    }
    if (s.relo) {
        for (uint32_t i = 0; i < s.nof_relo; ++i) out.relos.push_back(read_relo(s.relo[i]));
    }
    if (s.sini && s.sini->values) {
        for (uint32_t i = 0; i < s.sini->count; ++i) out.sini_values.push_back(s.sini->values[i]);
    }
    if (s.symd) {
        out.has_symd = true;
        out.symd.size = s.symd->size;
        out.symd.alignment = s.symd->alignment;
        out.symd.addressspace = s.symd->addressspace;
        out.symd.sym_hidden = s.symd->sym_hidden;
        out.symd.sym_restrict = s.symd->sym_restrict;
        out.symd.may_read_yuv = (s.symd->may_read_yuv != 0);
        out.symd.force_invariance = s.symd->force_invariance;
    }
    return out;
}

std::vector<Mbs2Symbol> read_ssym(const cmpbe_chunk_SSYM& ss) {
    std::vector<Mbs2Symbol> out;
    if (!ss.symb || ss.count == 0) return out;
    out.reserve(ss.count);
    for (uint32_t i = 0; i < ss.count; ++i) out.push_back(read_symb(ss.symb[i]));
    return out;
}

EbinRole classify_role(const std::string& stage_chunk, const Mbs2EbinMeta& meta) {
    if (stage_chunk == "CCOM") {
        if ((meta.flags & 0x20000000u) != 0 ||
            (meta.rt_num == 0xFFFFu && meta.aux_idx == 0 && meta.variant == 0) ||
            (meta.has_fnmd && !meta.fnmd.pilot_name.empty())) {
            return EbinRole::Pilot;
        }
        return EbinRole::MainCompute;
    }
    if (stage_chunk == "CVER") {
        return ((meta.variant & 0x01u) != 0) ? EbinRole::VertexVarying : EbinRole::VertexPosition;
    }
    if (stage_chunk == "CFRA") return EbinRole::Fragment;
    if (stage_chunk == "CTSC") return EbinRole::TessControl;
    if (stage_chunk == "CTSE") return EbinRole::TessEval;
    if (stage_chunk == "CGEO") return EbinRole::Geometry;
    if (stage_chunk == "CBLE") return EbinRole::Blend;
    return EbinRole::Unknown;
}

ExtractedEbin read_ebin(
    const cmpbe_chunk_EBIN& e,
    uint32_t index,
    const std::string& stage_chunk,
    uint8_t arch,
    bool disassemble) {
    ExtractedEbin eb;
    eb.index = index;
    eb.meta.index = index;
    eb.meta.var_name = "EBIN[" + std::to_string(index) + "]";
    eb.meta.rt_num = e.rt_num;
    eb.meta.aux_idx = e.aux_idx;
    eb.meta.prepass_flags = static_cast<uint32_t>(e.prepass_flags);
    eb.meta.data_section_offset = e.data_section_offset;
    eb.meta.nof_reg = e.nof_reg;
    eb.meta.nof_rmu = e.nof_rmu;
    eb.meta.tls_size = e.tls_size;
    eb.meta.flags = static_cast<uint32_t>(e.flags);
    eb.meta.variant = static_cast<uint32_t>(e.variant);
    eb.meta.entrypoint_name = read_stri_ptr(e.entrypoint_name);

    if (e.dbug) {
        eb.meta.has_dbug = true;
        if (e.dbug->rloc) {
            for (uint32_t i = 0; i < e.dbug->nof_rloc; ++i) eb.meta.dbug.rlocs.push_back(read_rloc(e.dbug->rloc[i]));
        }
        if (e.dbug->fcst) {
            for (uint32_t i = 0; i < e.dbug->nof_fcst; ++i) eb.meta.dbug.fcsts.push_back(read_fcst(e.dbug->fcst[i]));
        }
    }
    if (e.grel && e.grel->relo) {
        for (uint32_t i = 0; i < e.grel->nof_global_relo; ++i) eb.meta.grel.push_back(read_relo(e.grel->relo[i]));
    }
    if (e.spdf) {
        eb.meta.has_spdf = true;
        for (uint32_t i = 0; i < e.spdf->rt_count; ++i) {
            if (e.spdf->rt_index) eb.meta.spdf.rt_indices.push_back(e.spdf->rt_index[i]);
            if (e.spdf->size) eb.meta.spdf.sizes.push_back(static_cast<uint32_t>(e.spdf->size[i]));
            if (e.spdf->type) eb.meta.spdf.types.push_back(static_cast<uint32_t>(e.spdf->type[i]));
        }
    }
    if (e.pdsc) {
        eb.meta.has_pdsc = true;
        eb.meta.pdsc.varying0 = e.pdsc->varying0;
        eb.meta.pdsc.varying1 = e.pdsc->varying1;
        eb.meta.pdsc.primary_shader_wait = e.pdsc->primary_shader_wait;
    }
    if (e.fnmd) {
        eb.meta.has_fnmd = true;
        if (e.fnmd->tvar) {
            for (uint32_t i = 0; i < e.fnmd->nof_tvar; ++i) eb.meta.fnmd.tvars.push_back(static_cast<uint32_t>(e.fnmd->tvar[i]));
        }
        eb.meta.fnmd.reg_mode = static_cast<uint32_t>(e.fnmd->reg_mode);
        eb.meta.fnmd.pilot_name = read_stri(e.fnmd->pilot_name);
        eb.meta.fnmd.late_codegen = read_stri(e.fnmd->late_codegen);
        eb.meta.fnmd.attrib_mask = e.fnmd->attrib_mask;
        eb.meta.fnmd.max_register_pressure = e.fnmd->max_register_pressure;
        eb.meta.fnmd.mrtt = e.fnmd->mrtt;
    }

    eb.meta.role = classify_role(stage_chunk, eb.meta);
    eb.role = eb.meta.role;
    eb.variant_flags = eb.meta.variant;

    if (e.bytecode.byte_code && e.bytecode.size > 0) {
        uint32_t total = e.bytecode.size;
        eb.meta.objc_total_size = total;
        uint32_t code_sz = total;
        if (eb.meta.data_section_offset != 0 &&
            eb.meta.data_section_offset != 0xFFFFFFFFu &&
            eb.meta.data_section_offset <= total) {
            code_sz = eb.meta.data_section_offset;
        }
        code_sz = (code_sz / 8u) * 8u;
        eb.code_size = code_sz;
        eb.meta.code_size = code_sz;
        eb.bytecode.assign(e.bytecode.byte_code, e.bytecode.byte_code + code_sz);
        if (code_sz < total) {
            eb.data_section.assign(e.bytecode.byte_code + code_sz, e.bytecode.byte_code + total);
        }
        if (disassemble) {
            for (size_t off = 0; off + 8 <= eb.bytecode.size(); off += 8) {
                uint64_t word = 0;
                std::memcpy(&word, &eb.bytecode[off], 8);
                auto dec = valhall::decode_valhall_instruction(word, arch, off);
                if (!dec.valid) eb.invalid_count++;
                eb.instructions.push_back(std::move(dec));
            }
        }
    }
    return eb;
}

void read_cmmn(
    const cmpbe_chunk_CMMN& c,
    const std::string& stage_chunk,
    uint8_t arch,
    bool disassemble,
    Mbs2Cmmn* out_cmmn,
    std::vector<ExtractedEbin>* out_ebins) {
    out_cmmn->vela_lang = static_cast<uint32_t>(c.vela.lang);
    out_cmmn->sinp = read_ssym(c.sinp);
    out_cmmn->sout = read_ssym(c.sout);
    out_cmmn->suni = read_ssym(c.suni);
    out_cmmn->simg = read_ssym(c.simg);
    out_cmmn->ssmp = read_ssym(c.ssmp);
    out_cmmn->s3bo = read_ssym(c.s3bo);
    if (c.ubuf.sizes) {
        for (uint32_t i = 0; i < c.ubuf.count; ++i) out_cmmn->ubuf_sizes.push_back(c.ubuf.sizes[i]);
    }
    out_cmmn->epilogue_jump_index = c.epilogue_jump_index;
    if (c.fcst) {
        for (uint32_t i = 0; i < c.nof_fcst; ++i) out_cmmn->fcsts.push_back(read_fcst(c.fcst[i]));
    }
    if (c.ebin) {
        for (uint32_t i = 0; i < c.nof_ebin; ++i) {
            ExtractedEbin eb = read_ebin(c.ebin[i], static_cast<uint32_t>(out_ebins->size()),
                                         stage_chunk, arch, disassemble);
            out_cmmn->ebins.push_back(eb.meta);
            out_ebins->push_back(std::move(eb));
        }
    }
    if (c.cxfb) {
        out_cmmn->has_cxfb = true;
        out_cmmn->cxfb.strides[0] = c.cxfb->stride_0;
        out_cmmn->cxfb.strides[1] = c.cxfb->stride_1;
        out_cmmn->cxfb.strides[2] = c.cxfb->stride_2;
        out_cmmn->cxfb.strides[3] = c.cxfb->stride_3;
    }
}

}  // namespace

bool read_mbs2_struct_direct(
    const cmpbe_chunk_MBS2* mbs2,
    uint8_t arch,
    bool disassemble,
    Mbs2Module* out_module,
    std::vector<ExtractedEbin>* out_ebins,
    std::string* err) {
    if (!mbs2 || !out_module || !out_ebins) {
        if (err) *err = "Null pointer in read_mbs2_struct_direct";
        return false;
    }
    *out_module = Mbs2Module{};
    out_ebins->clear();

    out_module->version = mbs2->version;
    out_module->vehw.core = static_cast<uint32_t>(mbs2->vehw.core);
    out_module->vehw.major = mbs2->vehw.major;
    out_module->vehw.minor = mbs2->vehw.minor;

    if (mbs2->ccom) {
        out_module->stage_chunk = "CCOM";
        read_cmmn(mbs2->ccom->cmmn, "CCOM", arch, disassemble, &out_module->cmmn, out_ebins);
        out_module->has_kern = true;
        out_module->kern.name = read_stri(mbs2->ccom->kern.name);
        out_module->kern.workgroup_local_size = mbs2->ccom->kern.workgroup_local_size;
        if (mbs2->ccom->kern.kpar) {
            for (uint32_t i = 0; i < mbs2->ccom->kern.nof_kpar; ++i) {
                Mbs2Kpar kp;
                kp.declared_name = read_stri(mbs2->ccom->kern.kpar[i].declared_name);
                kp.uniform_name = read_stri(mbs2->ccom->kern.kpar[i].uniform_name);
                kp.flags = mbs2->ccom->kern.kpar[i].flags;
                out_module->kern.kpars.push_back(std::move(kp));
            }
        }
        if (mbs2->ccom->kern.workgroup_size) {
            out_module->kern.has_workgroup_size = true;
            out_module->kern.wg_x = mbs2->ccom->kern.workgroup_size->x;
            out_module->kern.wg_y = mbs2->ccom->kern.workgroup_size->y;
            out_module->kern.wg_z = mbs2->ccom->kern.workgroup_size->z;
        }
    } else if (mbs2->cfra) {
        out_module->stage_chunk = "CFRA";
        out_module->has_cfra = true;
        read_cmmn(mbs2->cfra->cmmn, "CFRA", arch, disassemble, &out_module->cmmn, out_ebins);
        if (mbs2->cfra->output_vars && mbs2->cfra->output_vars->outputs) {
            for (uint8_t i = 0; i < mbs2->cfra->output_vars->nb_outputs; ++i) {
                const auto& o = mbs2->cfra->output_vars->outputs[i];
                Mbs2Outv ov;
                ov.name = read_stri(o.name);
                ov.type = read_type(o.type);
                ov.location = o.location;
                ov.nb_locations_used = o.nb_locations_used;
                out_module->cfra.output_vars.push_back(std::move(ov));
            }
        }
        if (mbs2->cfra->lcal) {
            out_module->cfra.has_lcal = true;
            out_module->cfra.local_storage_size = mbs2->cfra->lcal->local_storage_size;
        }
        if (mbs2->cfra->bldm) {
            out_module->cfra.has_bldm = true;
            out_module->cfra.bldm_flags = mbs2->cfra->bldm->flags;
        }
        if (mbs2->cfra->spas) {
            out_module->cfra.has_spas = true;
            out_module->cfra.spas_read_mask = mbs2->cfra->spas->resource_read_mask;
            out_module->cfra.spas_write_mask = mbs2->cfra->spas->resource_write_mask;
        }
    } else if (mbs2->cver) {
        out_module->stage_chunk = "CVER";
        out_module->has_cver = true;
        read_cmmn(mbs2->cver->cmmn, "CVER", arch, disassemble, &out_module->cmmn, out_ebins);
        if (mbs2->cver->vmvw) {
            out_module->cver.has_vmvw = true;
            out_module->cver.num_views = mbs2->cver->vmvw->num_views;
        }
        if (mbs2->cver->vrsm) {
            out_module->cver.has_vrsm = true;
            out_module->cver.vrs_modes_mask = mbs2->cver->vrsm->potential_vrs_modes_mask;
        }
    } else if (mbs2->ctsc) {
        out_module->stage_chunk = "CTSC";
        read_cmmn(mbs2->ctsc->cmmn, "CTSC", arch, disassemble, &out_module->cmmn, out_ebins);
        out_module->ctsc_workgroup_local_size = mbs2->ctsc->workgroup_local_size;
        out_module->has_mtes = true;
        out_module->mtes.n_vertices = mbs2->ctsc->mtes.n_vertices;
        out_module->mtes.primitive_mode = mbs2->ctsc->mtes.primitive_mode;
        out_module->mtes.vertex_spacing = mbs2->ctsc->mtes.vertex_spacing;
        out_module->mtes.ordering = mbs2->ctsc->mtes.ordering;
        out_module->mtes.point_mode = (mbs2->ctsc->mtes.point_mode != 0);
    } else if (mbs2->ctse) {
        out_module->stage_chunk = "CTSE";
        read_cmmn(mbs2->ctse->cmmn, "CTSE", arch, disassemble, &out_module->cmmn, out_ebins);
        out_module->has_mtes = true;
        out_module->mtes.n_vertices = mbs2->ctse->mtes.n_vertices;
        out_module->mtes.primitive_mode = mbs2->ctse->mtes.primitive_mode;
        out_module->mtes.vertex_spacing = mbs2->ctse->mtes.vertex_spacing;
        out_module->mtes.ordering = mbs2->ctse->mtes.ordering;
        out_module->mtes.point_mode = (mbs2->ctse->mtes.point_mode != 0);
    } else if (mbs2->cgeo) {
        out_module->stage_chunk = "CGEO";
        read_cmmn(mbs2->cgeo->cmmn, "CGEO", arch, disassemble, &out_module->cmmn, out_ebins);
        out_module->has_mgeo = true;
        out_module->mgeo.n_invocations = mbs2->cgeo->mgeo.n_invocations;
        out_module->mgeo.max_vertices = mbs2->cgeo->mgeo.max_vertices;
        out_module->mgeo.input_primitive_mode = mbs2->cgeo->mgeo.input_primitive_mode;
        out_module->mgeo.output_primitive_mode = mbs2->cgeo->mgeo.output_primitive_mode;
    } else if (mbs2->cble) {
        out_module->stage_chunk = "CBLE";
        out_module->cmmn.suni = read_ssym(mbs2->cble->suni);
        ExtractedEbin eb = read_ebin(mbs2->cble->ebin, 0, "CBLE", arch, disassemble);
        out_module->cmmn.ebins.push_back(eb.meta);
        out_ebins->push_back(std::move(eb));
    }

    if (mbs2->momd) {
        out_module->has_momd = true;
        out_module->momd.prealloc_uni_buffers = mbs2->momd->prealloc_uni_buffers;
        out_module->momd.prealloc_fau = mbs2->momd->prealloc_fau;
        out_module->momd.setup_fau_in_pilot = mbs2->momd->setup_fau_in_pilot;
        out_module->momd.robustness = mbs2->momd->robustness;
        out_module->momd.per_sample_shading = mbs2->momd->per_sample_shading;
        out_module->momd.max_prealloc_addr = mbs2->momd->max_prealloc_addr;
        if (mbs2->momd->ploc) {
            for (uint32_t i = 0; i < mbs2->momd->nof_ploc; ++i) {
                out_module->momd.plocs.push_back(Mbs2Ploc{
                    mbs2->momd->ploc[i].reg_offset,
                    mbs2->momd->ploc[i].size,
                    mbs2->momd->ploc[i].orig_offset,
                });
            }
        }
    }

    if (out_ebins->empty()) {
        if (err) *err = "No EBIN bytecode sections found in live cmpbe_chunk_MBS2";
        return false;
    }
    return true;
}

}  // namespace valhall::spirv2valhall

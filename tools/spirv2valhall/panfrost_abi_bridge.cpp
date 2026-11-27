#include "panfrost_abi_bridge.h"

#include "ssa_defau_devirt.h"

#include <cstring>
#include <unordered_map>

namespace valhall::spirv2valhall {

namespace {

constexpr uint32_t kF32OneBits = 0x3f800000u;   // 1.0f
constexpr uint32_t kF32ZeroBits = 0x00000000u;  // 0.0f

bool inst_has_end_flow(const valhall::DecodedValhallInst& ins) {
    auto it = ins.modifier_strs.find("flow");
    return (it != ins.modifier_strs.end() && it->second == "end");
}

}  // namespace

PanfrostIdvsSplit decode_and_split_panfrost_binary(
    const std::vector<uint8_t>& ebin_bytes,
    uint8_t arch,
    valhall::ShaderStage stage,
    bool idvs_enabled) {
    PanfrostIdvsSplit out;
    const size_t n_qwords = ebin_bytes.size() / 8u;
    if (n_qwords == 0) return out;

    // Identify an IDVS Position/Varying boundary: a `.end` instruction followed by
    // one or more 64-bit zero alignment padding words and a non-zero Varying shader.
    size_t split_qword = n_qwords;
    if (stage == valhall::ShaderStage::Vertex && idvs_enabled) {
        bool saw_end = false;
        bool saw_zero_after_end = false;
        for (size_t i = 0; i < n_qwords; ++i) {
            uint64_t word = 0;
            std::memcpy(&word, ebin_bytes.data() + i * 8u, 8u);
            if (word == 0u) {
                if (saw_end) saw_zero_after_end = true;
                continue;
            }
            if (saw_end && saw_zero_after_end) {
                split_qword = i;
                break;
            }
            auto di = valhall::decode_valhall_instruction(word, arch, i * 8u);
            if (inst_has_end_flow(di)) {
                saw_end = true;
            }
        }
    }

    if (split_qword < n_qwords) {
        out.is_idvs_split = true;
        out.split_pc = static_cast<uint32_t>(split_qword * 8u);
        for (size_t i = 0; i < split_qword; ++i) {
            uint64_t word = 0;
            std::memcpy(&word, ebin_bytes.data() + i * 8u, 8u);
            if (word == 0u) continue;
            out.primary_insts.push_back(valhall::decode_valhall_instruction(word, arch, i * 8u));
        }
        for (size_t i = split_qword; i < n_qwords; ++i) {
            uint64_t word = 0;
            std::memcpy(&word, ebin_bytes.data() + i * 8u, 8u);
            if (word == 0u) continue;
            uint32_t rel_pc = static_cast<uint32_t>((i - split_qword) * 8u);
            out.secondary_insts.push_back(valhall::decode_valhall_instruction(word, arch, rel_pc));
        }
    } else {
        for (size_t i = 0; i < n_qwords; ++i) {
            uint64_t word = 0;
            std::memcpy(&word, ebin_bytes.data() + i * 8u, 8u);
            if (word == 0u) continue;
            out.primary_insts.push_back(valhall::decode_valhall_instruction(word, arch, i * 8u));
        }
    }
    return out;
}

void populate_panfrost_translate_options(
    const PanfrostCompileResult& cres,
    const PanfrostKernelBinary& kb,
    valhall::ValhallTranslateOptions* vopts,
    PanfrostAbiBridgeReport* report) {
    vopts->stage = kb.stage;
    vopts->arch = kb.arch;
    vopts->local_size_x = kb.local_size[0];
    vopts->local_size_y = kb.local_size[1];
    vopts->local_size_z = kb.local_size[2];
    vopts->srt_pka_buf_names = cres.srt_pka_buf_names;

    for (const auto& [w, val] : cres.fau_const_words) {
        uint32_t byte_off = w * 4u;
        vopts->fau_fcst_words[byte_off] = val;
        auto sit = cres.srt_pka_buf_names.find(val);
        if (sit != cres.srt_pka_buf_names.end()) {
            vopts->fau_pka_buf_names[byte_off] = sit->second;
        }
    }

    // Map Panfrost synthetic texture/image descriptors into ValhallTranslateOptions::samplers
    // so TEX_SINGLE / LD_TEX resolve canonical "tex:S:B" keys matching Direct SPIR-V and Mali DDK.
    uint32_t mapped_samplers = 0;
    for (const auto& d : cres.descriptors) {
        if (d.kind == "tex_or_img" || d.kind == "combined_image_sampler") {
            decltype(vopts->samplers)::value_type sbi{};
            sbi.descriptor_set = d.set;
            sbi.binding = d.binding;
            sbi.tex_index = d.desc_idx;
            sbi.sampler_index = (d.kind == "combined_image_sampler") ? (d.desc_idx + 1u) : d.desc_idx;
            vopts->samplers.push_back(sbi);
            ++mapped_samplers;
        }
    }
    if (report) {
        report->descriptor_samplers_mapped = mapped_samplers;
    }
}

void apply_panfrost_vk_abi_bridge(
    valhall::compiler::ValhallSsaModule& ssa,
    const PanfrostCompileResult& cres,
    const PanfrostKernelBinary& kb,
    PanfrostAbiBridgeReport* report) {
    if (ssa.blocks.empty()) return;

    valhall::dataflow::FauSummaryMap summary;
    std::unordered_map<uint32_t, uint32_t> nwg_word_to_comp;
    uint32_t vp_words = 0;
    uint32_t cs_words = 0;

    // Decode panvk compacted sysval FAU slots (64-bit slots packed in ascending order
    // via BITSET_PREFIX_SUM in panfrost_vk_compile.c).
    const uint32_t sysval_count = static_cast<uint32_t>(cres.used_sysval_words.size());
    for (uint32_t i = 0; i < sysval_count; ++i) {
        uint32_t orig_slot = cres.used_sysval_words[i];
        uint32_t fau_byte = i * 8u;
        uint32_t fau_word = i * 2u;

        if (kb.stage == valhall::ShaderStage::Compute) {
            // struct panvk_compute_sysvals:
            //   slot 0: {base.x, base.y}, slot 1: {base.z, _pad}
            //   slot 5: {num_work_groups.x, num_work_groups.y}
            //   slot 6: {num_work_groups.z, local_group_size.x}
            //   slot 7: {local_group_size.y, local_group_size.z}
            if (orig_slot == 0u) {
                summary[fau_byte] = valhall::dataflow::SymExpr::make_const_u32(0u);
                summary[fau_byte + 4u] = valhall::dataflow::SymExpr::make_const_u32(0u);
                cs_words += 2u;
            } else if (orig_slot == 1u) {
                summary[fau_byte] = valhall::dataflow::SymExpr::make_const_u32(0u);
                cs_words += 1u;
            } else if (orig_slot == 5u) {
                nwg_word_to_comp[fau_word] = 0u;
                nwg_word_to_comp[fau_word + 1u] = 1u;
                cs_words += 2u;
            } else if (orig_slot == 6u) {
                nwg_word_to_comp[fau_word] = 2u;
                summary[fau_byte + 4u] = valhall::dataflow::SymExpr::make_const_u32(kb.local_size[0]);
                cs_words += 2u;
            } else if (orig_slot == 7u) {
                summary[fau_byte] = valhall::dataflow::SymExpr::make_const_u32(kb.local_size[1]);
                summary[fau_byte + 4u] = valhall::dataflow::SymExpr::make_const_u32(kb.local_size[2]);
                cs_words += 2u;
            }
        } else if (kb.stage == valhall::ShaderStage::Vertex ||
                   kb.stage == valhall::ShaderStage::Fragment) {
            // struct panvk_graphics_sysvals:
            //   slots 0..1 (bytes 0..15):   blend.constants[4]
            //   slots 2..4 (bytes 16..39):  common.{push_uniforms, printf_buffer_address, constant_data}
            //   slot 5 (bytes 40..47):      viewport.scale.{x, y}   -> identity (1.0f, 1.0f)
            //   slot 6 (bytes 48..55):      viewport.{scale.z, offset.x} -> (1.0f, 0.0f)
            //   slot 7 (bytes 56..63):      viewport.offset.{y, z}  -> identity (0.0f, 0.0f)
            //   slot 13 (bytes 104..111):   vs.{first_vertex, base_instance} -> (0, 0)
            //   slot 14 (bytes 112..119):   vs.noperspective_varyings -> 0
            if (orig_slot == 5u) {
                summary[fau_byte] = valhall::dataflow::SymExpr::make_const_u32(kF32OneBits);
                summary[fau_byte + 4u] = valhall::dataflow::SymExpr::make_const_u32(kF32OneBits);
                vp_words += 2u;
            } else if (orig_slot == 6u) {
                summary[fau_byte] = valhall::dataflow::SymExpr::make_const_u32(kF32OneBits);
                summary[fau_byte + 4u] = valhall::dataflow::SymExpr::make_const_u32(kF32ZeroBits);
                vp_words += 2u;
            } else if (orig_slot == 7u) {
                summary[fau_byte] = valhall::dataflow::SymExpr::make_const_u32(kF32ZeroBits);
                summary[fau_byte + 4u] = valhall::dataflow::SymExpr::make_const_u32(kF32ZeroBits);
                vp_words += 2u;
            } else if (orig_slot == 13u) {
                summary[fau_byte] = valhall::dataflow::SymExpr::make_const_u32(0u);
                summary[fau_byte + 4u] = valhall::dataflow::SymExpr::make_const_u32(0u);
            } else if (orig_slot == 14u) {
                summary[fau_byte] = valhall::dataflow::SymExpr::make_const_u32(0u);
            }
        }
    }

    // Rewrite num_workgroups sysval loads to VecExtract(LoadBuiltinInput(NumWorkgroups), comp).
    if (!nwg_word_to_comp.empty()) {
        valhall::compiler::SsaId nwg_id = valhall::compiler::kInvalidSsa;
        std::vector<valhall::compiler::SsaId> entry_prefix;
        size_t orig_count = ssa.nodes.size();
        for (size_t idx = 0; idx < orig_count; ++idx) {
            if (ssa.nodes[idx].op != valhall::tables::SsaOp::LoadFau) continue;
            auto it = nwg_word_to_comp.find(ssa.nodes[idx].imm_u32);
            if (it == nwg_word_to_comp.end()) continue;
            if (nwg_id == valhall::compiler::kInvalidSsa) {
                valhall::compiler::SsaNode bn{};
                bn.id = static_cast<valhall::compiler::SsaId>(ssa.nodes.size());
                bn.block_id = 0;
                bn.op = valhall::tables::SsaOp::LoadBuiltinInput;
                bn.type = valhall::compiler::ValType::V3U32;
                bn.imm_u32 = 24u;  // SpvBuiltInNumWorkgroups
                bn.aux_str = "gl_NumWorkGroups";
                ssa.nodes.push_back(std::move(bn));
                nwg_id = ssa.nodes.back().id;
                entry_prefix.push_back(nwg_id);
            }
            ssa.nodes[idx].op = valhall::tables::SsaOp::VecExtract;
            ssa.nodes[idx].type = valhall::compiler::ValType::U32;
            ssa.nodes[idx].args = {nwg_id};
            ssa.nodes[idx].imm_u32 = it->second;
            ssa.nodes[idx].aux_str.clear();
        }
        if (!entry_prefix.empty()) {
            auto& b0 = ssa.blocks[0].nodes;
            b0.insert(b0.begin(), entry_prefix.begin(), entry_prefix.end());
        }
    }

    // Decode panvk compacted user push-constant FAU slots into canonical "push_const" loads.
    for (uint32_t j = 0; j < cres.used_push_const_words.size(); ++j) {
        uint32_t pc_slot = cres.used_push_const_words[j];
        uint32_t fau_byte = (sysval_count + j) * 8u;
        summary[fau_byte] = valhall::dataflow::SymExpr::make_buffer_load(-1, -1, "push_const", pc_slot * 8u, 4u);
        summary[fau_byte + 4u] = valhall::dataflow::SymExpr::make_buffer_load(-1, -1, "push_const", pc_slot * 8u + 4u, 4u);
    }

    // Seed promoted FAU immediate constants.
    for (const auto& [w, val] : cres.fau_const_words) {
        summary[w * 4u] = valhall::dataflow::SymExpr::make_const_u32(val);
    }

    if (!summary.empty()) {
        valhall::dataflow::DeFauDevirtOptions dopt;
        dopt.record_fallbacks = false;
        valhall::dataflow::run_ssa_defau_devirt(ssa, summary, dopt);
    }

    if (report) {
        report->viewport_fau_words_devirt = vp_words;
        report->compute_sysval_words_devirt = cs_words;
        report->push_const_words_devirt = static_cast<uint32_t>(cres.used_push_const_words.size() * 2u);
        report->immediate_words_devirt = static_cast<uint32_t>(cres.fau_const_words.size());
    }
}

void apply_panfrost_cl_abi_bridge(
    valhall::compiler::ValhallSsaModule& ssa,
    const std::vector<PanfrostArgBinding>& bindings) {
    if (ssa.blocks.empty()) return;
    std::unordered_map<uint32_t, std::pair<std::string, bool>> word_to_buf;
    for (const auto& ab : bindings) {
        uint32_t w_lo = ab.fau_offset / 4u;
        word_to_buf[w_lo] = {ab.key, false};
        word_to_buf[w_lo + 1u] = {ab.key, true};
    }

    std::unordered_map<std::string, std::pair<valhall::compiler::SsaId, valhall::compiler::SsaId>> buf_lo_hi;
    valhall::compiler::SsaId nwg_id = valhall::compiler::kInvalidSsa;
    std::vector<valhall::compiler::SsaId> entry_prefix;

    auto emit_entry = [&](valhall::compiler::SsaNode n) -> valhall::compiler::SsaId {
        n.id = static_cast<valhall::compiler::SsaId>(ssa.nodes.size());
        n.block_id = 0;
        ssa.nodes.push_back(std::move(n));
        entry_prefix.push_back(ssa.nodes.back().id);
        return ssa.nodes.back().id;
    };

    auto get_buf_lo_hi = [&](const std::string& key) -> std::pair<valhall::compiler::SsaId, valhall::compiler::SsaId> {
        auto it = buf_lo_hi.find(key);
        if (it != buf_lo_hi.end()) return it->second;
        valhall::compiler::SsaNode sb{};
        sb.op = valhall::tables::SsaOp::LoadSidebandU64;
        sb.type = valhall::compiler::ValType::U64;
        sb.imm_u32 = 0x10u;
        sb.aux_str = key;
        auto base = emit_entry(std::move(sb));

        valhall::compiler::SsaNode sp{};
        sp.op = valhall::tables::SsaOp::Split64;
        sp.type = valhall::compiler::ValType::TupleU32x2;
        sp.args = {base};
        auto sp_id = emit_entry(std::move(sp));

        valhall::compiler::SsaNode lo{};
        lo.op = valhall::tables::SsaOp::TupleExtract;
        lo.type = valhall::compiler::ValType::U32;
        lo.imm_u32 = 0;
        lo.args = {sp_id};
        auto lo_id = emit_entry(std::move(lo));

        valhall::compiler::SsaNode hi{};
        hi.op = valhall::tables::SsaOp::TupleExtract;
        hi.type = valhall::compiler::ValType::U32;
        hi.imm_u32 = 1;
        hi.args = {sp_id};
        auto hi_id = emit_entry(std::move(hi));

        buf_lo_hi[key] = {lo_id, hi_id};
        return {lo_id, hi_id};
    };

    auto get_nwg = [&]() -> valhall::compiler::SsaId {
        if (nwg_id != valhall::compiler::kInvalidSsa) return nwg_id;
        valhall::compiler::SsaNode bn{};
        bn.op = valhall::tables::SsaOp::LoadBuiltinInput;
        bn.type = valhall::compiler::ValType::V3U32;
        bn.imm_u32 = 24u;  // SpvBuiltInNumWorkgroups
        bn.aux_str = "gl_NumWorkGroups";
        nwg_id = emit_entry(std::move(bn));
        return nwg_id;
    };

    size_t orig_count = ssa.nodes.size();
    for (size_t i = 0; i < orig_count; ++i) {
        if (ssa.nodes[i].op != valhall::tables::SsaOp::LoadFau) continue;
        uint32_t widx = ssa.nodes[i].imm_u32;
        auto bit = word_to_buf.find(widx);
        if (bit != word_to_buf.end()) {
            auto [lo_id, hi_id] = get_buf_lo_hi(bit->second.first);
            ssa.nodes[i].op = valhall::tables::SsaOp::Copy;
            ssa.nodes[i].args = {bit->second.second ? hi_id : lo_id};
            ssa.nodes[i].imm_u32 = 0;
            ssa.nodes[i].aux_str = bit->second.first;
        } else if (widx < 3u) {
            auto nw = get_nwg();
            ssa.nodes[i].op = valhall::tables::SsaOp::VecExtract;
            ssa.nodes[i].type = valhall::compiler::ValType::U32;
            ssa.nodes[i].args = {nw};
            ssa.nodes[i].imm_u32 = widx;
        }
    }

    if (!entry_prefix.empty()) {
        auto& b0_nodes = ssa.blocks[0].nodes;
        b0_nodes.insert(b0_nodes.begin(), entry_prefix.begin(), entry_prefix.end());
    }
}

}  // namespace valhall::spirv2valhall

#include "valhall_csf_pipeline.h"
#include <cstring>
#include <algorithm>
#include <iostream>

namespace valhall::pipeline {

// ============================================================================
// 1. Morton 16x16 U-Interleaved Tiling & Format Helpers
// ============================================================================
uint32_t mali_morton_16x16_pixel_index(uint32_t x, uint32_t y) {
    uint32_t tx = x & 15u;
    uint32_t ty = y & 15u;
    uint32_t b0 = (tx & 1u) ^ (ty & 1u);
    uint32_t b1 = (ty & 1u);
    uint32_t b2 = ((tx >> 1u) & 1u) ^ ((ty >> 1u) & 1u);
    uint32_t b3 = ((ty >> 1u) & 1u);
    uint32_t b4 = (tx >> 2u) & 1u;
    uint32_t b5 = (ty >> 2u) & 1u;
    uint32_t b6 = (tx >> 3u) & 1u;
    uint32_t b7 = (ty >> 3u) & 1u;
    return (b0 << 0) | (b1 << 1) | (b2 << 2) | (b3 << 3) |
           (b4 << 4) | (b5 << 5) | (b6 << 6) | (b7 << 7);
}

void mali_detile_morton_16x16_rgba8(const uint8_t* tiled_src, uint8_t* linear_dst, uint32_t width, uint32_t height) {
    uint32_t tiles_x = (width + 15u) / 16u;
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            uint32_t tile_idx = (y / 16u) * tiles_x + (x / 16u);
            uint32_t in_tile = mali_morton_16x16_pixel_index(x, y);
            uint32_t src_off = (tile_idx * 256u + in_tile) * 4u;
            uint32_t dst_off = (y * width + x) * 4u;
            std::memcpy(linear_dst + dst_off, tiled_src + src_off, 4u);
        }
    }
}

void mali_retile_morton_16x16_rgba8(const uint8_t* linear_src, uint8_t* tiled_dst, uint32_t width, uint32_t height) {
    uint32_t tiles_x = (width + 15u) / 16u;
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            uint32_t tile_idx = (y / 16u) * tiles_x + (x / 16u);
            uint32_t in_tile = mali_morton_16x16_pixel_index(x, y);
            uint32_t dst_off = (tile_idx * 256u + in_tile) * 4u;
            uint32_t src_off = (y * width + x) * 4u;
            std::memcpy(tiled_dst + dst_off, linear_src + src_off, 4u);
        }
    }
}

VkFormat translate_mali_pixel_format(csf::PixelFormat pf) {
    using F = csf::Format;
    switch (pf.format) {
        case F::RGBA8_UNORM: return pf.srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
        case F::RGBA8_SNORM: return VK_FORMAT_R8G8B8A8_SNORM;
        case F::RGBA8UI:     return VK_FORMAT_R8G8B8A8_UINT;
        case F::RGBA8I:      return VK_FORMAT_R8G8B8A8_SINT;
        case F::RGB32F:      return VK_FORMAT_R32G32B32_SFLOAT;
        case F::RGBA32F:     return VK_FORMAT_R32G32B32A32_SFLOAT;
        case F::RG32F:       return VK_FORMAT_R32G32_SFLOAT;
        case F::R32F:        return VK_FORMAT_R32_SFLOAT;
        case F::RGBA16F:     return VK_FORMAT_R16G16B16A16_SFLOAT;
        case F::RG16F:       return VK_FORMAT_R16G16_SFLOAT;
        case F::R16F:        return VK_FORMAT_R16_SFLOAT;
        case F::R32UI:       return VK_FORMAT_R32_UINT;
        case F::R32I:        return VK_FORMAT_R32_SINT;
        case F::RG32UI:      return VK_FORMAT_R32G32_UINT;
        case F::RG32I:       return VK_FORMAT_R32G32_SINT;
        case F::RGBA32UI:    return VK_FORMAT_R32G32B32A32_UINT;
        case F::RGBA32I:     return VK_FORMAT_R32G32B32A32_SINT;
        default:             return VK_FORMAT_R8G8B8A8_UNORM;
    }
}

VkCompareOp translate_mali_compare_func(csf::Func f) {
    switch (f) {
        case csf::Func::Never:     return VK_COMPARE_OP_NEVER;
        case csf::Func::Less:      return VK_COMPARE_OP_LESS;
        case csf::Func::Equal:     return VK_COMPARE_OP_EQUAL;
        case csf::Func::Lequal:    return VK_COMPARE_OP_LESS_OR_EQUAL;
        case csf::Func::Greater:   return VK_COMPARE_OP_GREATER;
        case csf::Func::Not_Equal: return VK_COMPARE_OP_NOT_EQUAL;
        case csf::Func::Gequal:    return VK_COMPARE_OP_GREATER_OR_EQUAL;
        case csf::Func::Always:    return VK_COMPARE_OP_ALWAYS;
    }
    return VK_COMPARE_OP_ALWAYS;
}

VkStencilOp translate_mali_stencil_op(csf::StencilOp op) {
    switch (op) {
        case csf::StencilOp::Keep:      return VK_STENCIL_OP_KEEP;
        case csf::StencilOp::Replace:   return VK_STENCIL_OP_REPLACE;
        case csf::StencilOp::Zero:      return VK_STENCIL_OP_ZERO;
        case csf::StencilOp::Incr_Sat:  return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
        case csf::StencilOp::Decr_Sat:  return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
        case csf::StencilOp::Invert:    return VK_STENCIL_OP_INVERT;
        case csf::StencilOp::Incr_Wrap: return VK_STENCIL_OP_INCREMENT_AND_WRAP;
        case csf::StencilOp::Decr_Wrap: return VK_STENCIL_OP_DECREMENT_AND_WRAP;
    }
    return VK_STENCIL_OP_KEEP;
}

VkPrimitiveTopology translate_mali_draw_mode(csf::DrawMode dm) {
    switch (dm) {
        case csf::DrawMode::Points:         return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        case csf::DrawMode::Lines:          return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        case csf::DrawMode::Line_strip:     return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
        case csf::DrawMode::Triangles:      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        case csf::DrawMode::Triangle_strip: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        case csf::DrawMode::Triangle_fan:   return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
        default:                            return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }
}

VkIndexType translate_mali_index_type(csf::IndexType it, bool* out_indexed) {
    switch (it) {
        case csf::IndexType::None:
            if (out_indexed) *out_indexed = false;
            return VK_INDEX_TYPE_UINT32;
        case csf::IndexType::UINT8:
            if (out_indexed) *out_indexed = true;
            return VK_INDEX_TYPE_UINT8_EXT;
        case csf::IndexType::UINT16:
            if (out_indexed) *out_indexed = true;
            return VK_INDEX_TYPE_UINT16;
        case csf::IndexType::UINT32:
            if (out_indexed) *out_indexed = true;
            return VK_INDEX_TYPE_UINT32;
    }
    if (out_indexed) *out_indexed = false;
    return VK_INDEX_TYPE_UINT32;
}

VulkanComputeSideband extract_compute_sideband(
    const CsfStagingRegisterFile& sr,
    const csf::CSRUNCOMPUTE& run_compute_cmd) {
    VulkanComputeSideband sb{};
    sb.cmd = run_compute_cmd;

    const csf::COMPUTESR srt_regs[4] = {
        csf::COMPUTESR::SRT_0, csf::COMPUTESR::SRT_1, csf::COMPUTESR::SRT_2, csf::COMPUTESR::SRT_3
    };
    const csf::COMPUTESR spd_regs[4] = {
        csf::COMPUTESR::SPD_0, csf::COMPUTESR::SPD_1, csf::COMPUTESR::SPD_2, csf::COMPUTESR::SPD_3
    };
    const csf::COMPUTESR tsd_regs[4] = {
        csf::COMPUTESR::TSD_0, csf::COMPUTESR::TSD_1, csf::COMPUTESR::TSD_2, csf::COMPUTESR::TSD_3
    };
    const csf::COMPUTESR fau_regs[4] = {
        csf::COMPUTESR::FAU_0, csf::COMPUTESR::FAU_1, csf::COMPUTESR::FAU_2, csf::COMPUTESR::FAU_3
    };

    uint64_t raw_srt = sr.get64(srt_regs[run_compute_cmd.srt_select & 3u]);
    uint64_t raw_spd = sr.get64(spd_regs[run_compute_cmd.spd_select & 3u]);
    uint64_t raw_tsd = sr.get64(tsd_regs[run_compute_cmd.tsd_select & 3u]);
    uint64_t raw_fau = sr.get64(fau_regs[run_compute_cmd.fau_select & 3u]);

    sb.srt_va = raw_srt & 0x00FFFFFFFFFFFFFFULL;
    sb.spd_va = raw_spd & 0x00FFFFFFFFFFFFFFULL;
    sb.tsd_va = raw_tsd & 0x00FFFFFFFFFFFFFFULL;
    sb.fau_va = raw_fau & 0x00FFFFFFFFFFFFFFULL;
    sb.fau_word_count = static_cast<uint32_t>((raw_fau >> 56) & 0xFFu) * 2u;
    if (sb.fau_word_count == 0) sb.fau_word_count = 64u;

    if (sb.spd_va != 0) {
        sb.spd = csf::ShaderProgram::unpack(reinterpret_cast<const uint32_t*>(sb.spd_va));
        sb.shader_program_va = sb.spd.binary;
    }

    // Unpack WG_SIZE (COMPUTE SR r33: [9:0]=wg_x-1, [19:10]=wg_y-1, [29:20]=wg_z-1)
    uint32_t wg_raw = sr.get32(csf::COMPUTESR::WG_SIZE);
    sb.workgroup_size[0] = ((wg_raw >> 0)  & 0x3FFu) + 1u;
    sb.workgroup_size[1] = ((wg_raw >> 10) & 0x3FFu) + 1u;
    sb.workgroup_size[2] = ((wg_raw >> 20) & 0x3FFu) + 1u;

    sb.wg_offset[0] = sr.get32(csf::COMPUTESR::JOB_OFFSET_X);
    sb.wg_offset[1] = sr.get32(csf::COMPUTESR::JOB_OFFSET_Y);
    sb.wg_offset[2] = sr.get32(csf::COMPUTESR::JOB_OFFSET_Z);

    sb.group_count[0] = std::max(1u, sr.get32(csf::COMPUTESR::JOB_SIZE_X));
    sb.group_count[1] = std::max(1u, sr.get32(csf::COMPUTESR::JOB_SIZE_Y));
    sb.group_count[2] = std::max(1u, sr.get32(csf::COMPUTESR::JOB_SIZE_Z));
    return sb;
}

static void unpack_srt_descriptors(
    uint64_t srt_va,
    std::vector<csf::Attribute>* out_attrs,
    std::vector<csf::Buffer>* out_bufs,
    std::vector<csf::Texture>* out_texs,
    std::vector<csf::GenericPlane>* out_planes,
    std::vector<csf::Sampler>* out_samplers) {
    if (srt_va == 0) return;

    // Walk up to 4 Resource entries in the SRT
    const uint32_t* srt_words = reinterpret_cast<const uint32_t*>(srt_va);
    for (uint32_t r_idx = 0; r_idx < 4; ++r_idx) {
        csf::Resource res = csf::Resource::unpack(srt_words + r_idx * 4u);
        if (res.address == 0 || res.size == 0) continue;
        uint32_t num_descs = static_cast<uint32_t>(res.size / 32u);
        const uint32_t* table_ptr = reinterpret_cast<const uint32_t*>(res.address);
        for (uint32_t i = 0; i < num_descs; ++i) {
            const uint32_t* d_words = table_ptr + i * 8u;
            csf::DescriptorType dtype = static_cast<csf::DescriptorType>(d_words[0] & 0xFu);
            if (dtype == csf::DescriptorType::Attribute && out_attrs) {
                out_attrs->push_back(csf::Attribute::unpack(d_words));
            } else if (dtype == csf::DescriptorType::Buffer && out_bufs) {
                out_bufs->push_back(csf::Buffer::unpack(d_words));
            } else if (dtype == csf::DescriptorType::Texture && out_texs) {
                csf::Texture t = csf::Texture::unpack(d_words);
                out_texs->push_back(t);
                if (out_planes) {
                    if (t.surfaces != 0) {
                        out_planes->push_back(csf::GenericPlane::unpack(reinterpret_cast<const uint32_t*>(t.surfaces)));
                    } else {
                        out_planes->push_back(csf::GenericPlane{});
                    }
                }
            } else if (dtype == csf::DescriptorType::Sampler && out_samplers) {
                out_samplers->push_back(csf::Sampler::unpack(d_words));
            }
        }
    }
}

VulkanGraphicsSideband extract_graphics_sideband(
    const CsfStagingRegisterFile& idvs_sr,
    const CsfStagingRegisterFile& frag_sr,
    const csf::CSRUNIDVS& run_idvs_cmd) {
    VulkanGraphicsSideband sb{};
    sb.cmd = run_idvs_cmd;

    sb.pos_spd_va  = idvs_sr.get64(csf::IDVSSR::SPD_0) & 0x00FFFFFFFFFFFFFFULL;
    sb.var_spd_va  = idvs_sr.get64(csf::IDVSSR::SPD_1) & 0x00FFFFFFFFFFFFFFULL;
    sb.frag_spd_va = idvs_sr.get64(csf::IDVSSR::SPD_2) & 0x00FFFFFFFFFFFFFFULL;

    sb.pos_srt_va  = idvs_sr.get64(csf::IDVSSR::SRT_0) & 0x00FFFFFFFFFFFFFFULL;
    sb.var_srt_va  = idvs_sr.get64(run_idvs_cmd.varying_srt_select ? csf::IDVSSR::SRT_1 : csf::IDVSSR::SRT_0) & 0x00FFFFFFFFFFFFFFULL;
    sb.frag_srt_va = idvs_sr.get64(run_idvs_cmd.fragment_srt_select ? csf::IDVSSR::SRT_2 : csf::IDVSSR::SRT_0) & 0x00FFFFFFFFFFFFFFULL;

    uint64_t pfau = idvs_sr.get64(csf::IDVSSR::FAU_0);
    sb.pos_fau_va    = pfau & 0x00FFFFFFFFFFFFFFULL;
    sb.pos_fau_words = static_cast<uint32_t>((pfau >> 56) & 0xFFu) * 2u;

    uint64_t vfau = idvs_sr.get64(run_idvs_cmd.varying_fau_select ? csf::IDVSSR::FAU_1 : csf::IDVSSR::FAU_0);
    sb.var_fau_va    = vfau & 0x00FFFFFFFFFFFFFFULL;
    sb.var_fau_words = static_cast<uint32_t>((vfau >> 56) & 0xFFu) * 2u;

    uint64_t ffau = idvs_sr.get64(csf::IDVSSR::FAU_2);
    sb.frag_fau_va    = ffau & 0x00FFFFFFFFFFFFFFULL;
    sb.frag_fau_words = static_cast<uint32_t>((ffau >> 56) & 0xFFu) * 2u;

    if (sb.pos_spd_va != 0)  sb.pos_spd  = csf::ShaderProgram::unpack(reinterpret_cast<const uint32_t*>(sb.pos_spd_va));
    if (sb.var_spd_va != 0)  sb.var_spd  = csf::ShaderProgram::unpack(reinterpret_cast<const uint32_t*>(sb.var_spd_va));
    if (sb.frag_spd_va != 0) sb.frag_spd = csf::ShaderProgram::unpack(reinterpret_cast<const uint32_t*>(sb.frag_spd_va));

    // Unpack FBD pointer from FRAGMENT SR r40:41
    uint64_t raw_fbd_ptr = frag_sr.get64(csf::FRAGMENTSR::FBD_POINTER);
    uint32_t fbd_ptr_words[2] = {
        static_cast<uint32_t>(raw_fbd_ptr & 0xFFFFFFFFULL),
        static_cast<uint32_t>(raw_fbd_ptr >> 32)
    };
    csf::FramebufferPointer fbp = csf::FramebufferPointer::unpack(fbd_ptr_words);
    sb.fbd_va = fbp.pointer;
    sb.rt_count = std::max(1u, fbp.render_target_count);
    sb.has_zs_crc_ext = fbp.zs_crc_extension_present;

    if (sb.fbd_va != 0) {
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(sb.fbd_va);
        sb.fbd = csf::FramebufferParameters::unpack(reinterpret_cast<const uint32_t*>(ptr));
        ptr += 128; // 72B FramebufferParameters + 56B FramebufferPadding = 128B FBD header!
        if (sb.has_zs_crc_ext) {
            sb.zs_ext = csf::ZSCRCExtension::unpack(reinterpret_cast<const uint32_t*>(ptr));
            ptr += 64;
        }
        sb.rtds.resize(sb.rt_count);
        for (uint32_t i = 0; i < sb.rt_count; ++i) {
            sb.rtds[i] = csf::RenderTarget::unpack(reinterpret_cast<const uint32_t*>(ptr));
            ptr += 64;
        }
    }

    uint32_t bbox_max = frag_sr.get32(csf::FRAGMENTSR::BBOX_MAX);
    uint32_t bbox_w = (bbox_max & 0xFFFFu) + 1u;
    uint32_t bbox_h = ((bbox_max >> 16) & 0xFFFFu) + 1u;
    uint32_t width  = (sb.fbd_va != 0 && sb.fbd.width > 0)  ? sb.fbd.width  : std::max(1u, bbox_w);
    uint32_t height = (sb.fbd_va != 0 && sb.fbd.height > 0) ? sb.fbd.height : std::max(1u, bbox_h);

    sb.viewport.x = 0.0f;
    sb.viewport.y = 0.0f;
    sb.viewport.width = static_cast<float>(width);
    sb.viewport.height = static_cast<float>(height);
    sb.viewport.minDepth = idvs_sr.reg_f32(static_cast<uint32_t>(csf::IDVSSR::LOW_DEPTH_CLAMP));
    sb.viewport.maxDepth = idvs_sr.reg_f32(static_cast<uint32_t>(csf::IDVSSR::HIGH_DEPTH_CLAMP));
    if (sb.viewport.maxDepth <= sb.viewport.minDepth) {
        sb.viewport.minDepth = 0.0f;
        sb.viewport.maxDepth = 1.0f;
    }
    sb.scissor.offset = {0, 0};
    sb.scissor.extent = {width, height};

    // Unpack ZSD (IDVS SR r52:53)
    sb.zsd_va = idvs_sr.get64(csf::IDVSSR::ZSD) & 0x00FFFFFFFFFFFFFFULL;
    sb.depth_stencil_state = {};
    sb.depth_stencil_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    if (sb.zsd_va != 0) {
        sb.zsd = csf::DepthStencil::unpack(reinterpret_cast<const uint32_t*>(sb.zsd_va));
        sb.depth_stencil_state.depthTestEnable =
            (sb.zsd.depth_function != csf::Func::Always || sb.zsd.depth_write_enable) ? VK_TRUE : VK_FALSE;
        sb.depth_stencil_state.depthWriteEnable = sb.zsd.depth_write_enable ? VK_TRUE : VK_FALSE;
        sb.depth_stencil_state.depthCompareOp = translate_mali_compare_func(sb.zsd.depth_function);
        sb.depth_stencil_state.stencilTestEnable = sb.zsd.stencil_test_enable ? VK_TRUE : VK_FALSE;
    }

    // Unpack Blend array (IDVS SR r50:51)
    sb.blend_va = idvs_sr.get64(csf::IDVSSR::BLEND_DESC) & 0x00FFFFFFFFFFFFFFULL;
    sb.blends.resize(sb.rt_count);
    sb.blend_attachments.resize(sb.rt_count);
    for (uint32_t i = 0; i < sb.rt_count; ++i) {
        VkPipelineColorBlendAttachmentState att{};
        att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                             VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        att.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        att.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        att.colorBlendOp = VK_BLEND_OP_ADD;
        att.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        att.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        att.alphaBlendOp = VK_BLEND_OP_ADD;
        if (sb.blend_va != 0) {
            const uint32_t* bptr = reinterpret_cast<const uint32_t*>(sb.blend_va + i * 16u);
            sb.blends[i] = csf::Blend::unpack(bptr);
            if (sb.blends[i].equation.color_mask != 0) {
                att.colorWriteMask = sb.blends[i].equation.color_mask & 0xFu;
            }
            att.blendEnable = (sb.blends[i].enable &&
                               sb.blends[i].internal.mode == csf::BlendMode::Fixed_Function) ? VK_TRUE : VK_FALSE;
        }
        sb.blend_attachments[i] = att;
    }

    // DCD0..2 and draw parameters
    uint32_t dcd0_raw = idvs_sr.get32(csf::IDVSSR::DCD0);
    uint32_t dcd1_raw = idvs_sr.get32(csf::IDVSSR::DCD1);
    uint32_t dcd2_raw = idvs_sr.get32(csf::IDVSSR::DCD2);
    sb.dcd0 = csf::DCDFlags0::unpack(&dcd0_raw);
    sb.dcd1 = csf::DCDFlags1::unpack(&dcd1_raw);
    sb.dcd2 = csf::DCDFlags2::unpack(&dcd2_raw);

    VkCullModeFlags cull = VK_CULL_MODE_NONE;
    if (sb.dcd0.cull_front_face) cull |= VK_CULL_MODE_FRONT_BIT;
    if (sb.dcd0.cull_back_face)  cull |= VK_CULL_MODE_BACK_BIT;
    sb.cull_mode = cull;
    sb.front_face = sb.dcd0.front_face_ccw ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;

    sb.vertex_count = idvs_sr.get32(csf::IDVSSR::INDEX_COUNT);
    sb.instance_count = std::max(1u, idvs_sr.get32(csf::IDVSSR::INSTANCE_COUNT));
    sb.first_vertex = idvs_sr.get32(csf::IDVSSR::INDEX_OFFSET);
    sb.vertex_offset = static_cast<int32_t>(idvs_sr.get32(csf::IDVSSR::VERTEX_OFFSET));
    sb.index_buffer_va = idvs_sr.get64(csf::IDVSSR::INDEX_BUFFER) & 0x00FFFFFFFFFFFFFFULL;
    sb.point_size = idvs_sr.reg_f32(static_cast<uint32_t>(csf::IDVSSR::PRIMITIVE_SIZE));
    if (sb.point_size <= 0.0f) sb.point_size = 1.0f;

    // Unpack Vertex SRT (Attributes + Buffers)
    unpack_srt_descriptors(sb.pos_srt_va, &sb.attributes, &sb.vertex_buffers, nullptr, nullptr, nullptr);
    if (sb.var_srt_va != 0 && sb.var_srt_va != sb.pos_srt_va) {
        unpack_srt_descriptors(sb.var_srt_va, &sb.attributes, &sb.vertex_buffers, nullptr, nullptr, nullptr);
    }

    // Unpack Fragment SRT (Textures + Planes + Samplers)
    unpack_srt_descriptors(sb.frag_srt_va, nullptr, nullptr, &sb.textures, &sb.planes, &sb.samplers);

    return sb;
}

// ============================================================================
// 4. ValhallVulkanEngine Implementation (Mesa Lavapipe)
// ============================================================================
ValhallVulkanEngine::ValhallVulkanEngine() = default;

ValhallVulkanEngine::~ValhallVulkanEngine() {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
        for (auto& b : tracked_buffers_) {
            if (b.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device_, b.buffer, nullptr);
            if (b.memory != VK_NULL_HANDLE) vkFreeMemory(device_, b.memory, nullptr);
        }
        tracked_buffers_.clear();
        if (cmd_pool_ != VK_NULL_HANDLE) vkDestroyCommandPool(device_, cmd_pool_, nullptr);
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
}

bool ValhallVulkanEngine::init(std::string* out_err) {
    if (device_ != VK_NULL_HANDLE) return true;
    setenv("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/lvp_icd.json", 0);

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "valhall2spirv";
    app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app.pEngineName = "ValhallVulkanEngine";
    app.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, nullptr, &instance_) != VK_SUCCESS) {
        if (out_err) *out_err = "vkCreateInstance failed";
        return false;
    }

    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (count == 0) {
        if (out_err) *out_err = "No Vulkan physical devices found";
        return false;
    }
    std::vector<VkPhysicalDevice> devs(count);
    vkEnumeratePhysicalDevices(instance_, &count, devs.data());
    phys_dev_ = devs[0];

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(phys_dev_, &props);
    device_name_ = props.deviceName;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = 0;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    VkPhysicalDeviceVulkan12Features f12{};
    f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    f12.bufferDeviceAddress = VK_TRUE;
    f12.shaderInt8 = VK_TRUE;
    f12.shaderFloat16 = VK_TRUE;
    f12.storageBuffer8BitAccess = VK_TRUE;
    f12.uniformAndStorageBuffer8BitAccess = VK_TRUE;
    f12.scalarBlockLayout = VK_TRUE;

    VkPhysicalDeviceVulkan11Features f11{};
    f11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    f11.pNext = &f12;
    f11.storageBuffer16BitAccess = VK_TRUE;
    f11.uniformAndStorageBuffer16BitAccess = VK_TRUE;

    VkPhysicalDeviceFeatures2 f2{};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &f11;
    f2.features.shaderInt64 = VK_TRUE;
    f2.features.shaderInt16 = VK_TRUE;
    f2.features.fragmentStoresAndAtomics = VK_TRUE;
    f2.features.vertexPipelineStoresAndAtomics = VK_TRUE;

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &f2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;

    if (vkCreateDevice(phys_dev_, &dci, nullptr, &device_) != VK_SUCCESS) {
        if (out_err) *out_err = "vkCreateDevice failed";
        return false;
    }
    vkGetDeviceQueue(device_, 0, 0, &queue_);

    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.queueFamilyIndex = 0;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(device_, &cpci, nullptr, &cmd_pool_) != VK_SUCCESS) {
        if (out_err) *out_err = "vkCreateCommandPool failed";
        return false;
    }
    return true;
}

uint32_t ValhallVulkanEngine::find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags props) const {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys_dev_, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) && ((mp.memoryTypes[i].propertyFlags & props) == props)) {
            return i;
        }
    }
    return 0;
}

GpuBuffer ValhallVulkanEngine::alloc_buffer(VkDeviceSize size_bytes, const void* init_data, VkDeviceSize init_size) {
    GpuBuffer gb{};
    gb.size = std::max<VkDeviceSize>(64, (size_bytes + 63) & ~VkDeviceSize(63));

    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = gb.size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(device_, &bci, nullptr, &gb.buffer);

    VkMemoryRequirements mr{};
    vkGetBufferMemoryRequirements(device_, gb.buffer, &mr);

    VkMemoryAllocateFlagsInfo mafi{};
    mafi.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    mafi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.pNext = &mafi;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = find_memory_type(mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    vkAllocateMemory(device_, &mai, nullptr, &gb.memory);
    vkBindBufferMemory(device_, gb.buffer, gb.memory, 0);
    vkMapMemory(device_, gb.memory, 0, gb.size, 0, &gb.mapped);
    std::memset(gb.mapped, 0, gb.size);
    if (init_data && init_size > 0) {
        std::memcpy(gb.mapped, init_data, std::min(gb.size, init_size));
    }

    VkBufferDeviceAddressInfo bdai{};
    bdai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    bdai.buffer = gb.buffer;
    gb.device_address = vkGetBufferDeviceAddress(device_, &bdai);
    tracked_buffers_.push_back(gb);
    return gb;
}

void ValhallVulkanEngine::free_buffer(GpuBuffer& buf) {
    if (buf.buffer == VK_NULL_HANDLE) return;
    vkDestroyBuffer(device_, buf.buffer, nullptr);
    vkFreeMemory(device_, buf.memory, nullptr);
    buf.buffer = VK_NULL_HANDLE;
    buf.memory = VK_NULL_HANDLE;
    buf.mapped = nullptr;
}

bool ValhallVulkanEngine::dispatch_spirv_compute(
    const std::vector<uint32_t>& spirv_words,
    const std::vector<VkBuffer>& ssbo_buffers,
    const void* push_constants,
    uint32_t push_constants_size,
    uint32_t group_x,
    uint32_t group_y,
    uint32_t group_z,
    std::string* out_err) {

    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spirv_words.size() * sizeof(uint32_t);
    smci.pCode = spirv_words.data();
    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device_, &smci, nullptr, &mod) != VK_SUCCESS) {
        if (out_err) *out_err = "vkCreateShaderModule failed";
        return false;
    }

    std::vector<VkDescriptorSetLayoutBinding> bindings(ssbo_buffers.size());
    for (size_t i = 0; i < ssbo_buffers.size(); ++i) {
        bindings[i].binding = static_cast<uint32_t>(i);
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = static_cast<uint32_t>(bindings.size());
    dslci.pBindings = bindings.data();
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    vkCreateDescriptorSetLayout(device_, &dslci, nullptr, &dsl);

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = push_constants_size;

    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &dsl;
    if (push_constants && push_constants_size > 0) {
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
    }
    VkPipelineLayout playout = VK_NULL_HANDLE;
    vkCreatePipelineLayout(device_, &plci, nullptr, &playout);

    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = mod;
    cpci.stage.pName = "main";
    cpci.layout = playout;

    VkPipeline pipe = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe) != VK_SUCCESS) {
        if (out_err) *out_err = "vkCreateComputePipelines failed";
        vkDestroyPipelineLayout(device_, playout, nullptr);
        vkDestroyDescriptorSetLayout(device_, dsl, nullptr);
        vkDestroyShaderModule(device_, mod, nullptr);
        return false;
    }

    VkDescriptorPoolSize pool_size{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, std::max<uint32_t>(1, ssbo_buffers.size()) };
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &pool_size;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    vkCreateDescriptorPool(device_, &dpci, nullptr, &dpool);

    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = dpool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &dsl;
    VkDescriptorSet dset = VK_NULL_HANDLE;
    vkAllocateDescriptorSets(device_, &dsai, &dset);

    std::vector<VkDescriptorBufferInfo> dbis(ssbo_buffers.size());
    std::vector<VkWriteDescriptorSet> writes(ssbo_buffers.size());
    for (size_t i = 0; i < ssbo_buffers.size(); ++i) {
        dbis[i].buffer = ssbo_buffers[i];
        dbis[i].offset = 0;
        dbis[i].range = VK_WHOLE_SIZE;

        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = dset;
        writes[i].dstBinding = static_cast<uint32_t>(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &dbis[i];
    }
    if (!writes.empty()) {
        vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = cmd_pool_;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(device_, &cbai, &cmd);

    VkCommandBufferBeginInfo cbbi{};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &cbbi);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, playout, 0, 1, &dset, 0, nullptr);
    if (push_constants && push_constants_size > 0) {
        vkCmdPushConstants(cmd, playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_constants_size, push_constants);
    }
    vkCmdDispatch(cmd, std::max(1u, group_x), std::max(1u, group_y), std::max(1u, group_z));
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue_);

    vkFreeCommandBuffers(device_, cmd_pool_, 1, &cmd);
    vkDestroyDescriptorPool(device_, dpool, nullptr);
    vkDestroyPipeline(device_, pipe, nullptr);
    vkDestroyPipelineLayout(device_, playout, nullptr);
    vkDestroyDescriptorSetLayout(device_, dsl, nullptr);
    vkDestroyShaderModule(device_, mod, nullptr);
    return true;
}

bool ValhallVulkanEngine::execute_valhall_pilot_shader(
    const uint8_t* pilot_code,
    size_t pilot_code_size,
    uint64_t fau_dst_va,
    uint64_t srt_va,
    uint8_t arch,
    std::string* out_err) {
    ValhallTranslateOptions opts{};
    opts.stage = ShaderStage::Pilot;
    opts.arch = arch;
    auto tr = translate_valhall_to_spirv(pilot_code, pilot_code_size, opts);
    if (!tr.ok) {
        if (out_err) *out_err = tr.error_message;
        return false;
    }

    ValhallSidebandData sb_data{};
    sb_data.srt_base_va = srt_va;
    sb_data.pilot_fau_dst_va = fau_dst_va;
    GpuBuffer sb_buf = alloc_buffer(sizeof(ValhallSidebandData), &sb_data, sizeof(ValhallSidebandData));

    return dispatch_spirv_compute(tr.spirv_words, { sb_buf.buffer }, nullptr, 0, 1, 1, 1, out_err);
}

bool ValhallVulkanEngine::execute_cs_run_compute(
    const CsfStagingRegisterFile& compute_sr,
    const csf::CSRUNCOMPUTE& run_compute_cmd,
    std::vector<uint32_t>* out_translated_spirv,
    std::string* out_disassembly,
    std::string* out_err) {

    VulkanComputeSideband csb = extract_compute_sideband(compute_sr, run_compute_cmd);
    if (csb.shader_program_va == 0) {
        if (out_err) *out_err = "SPD.binary VA is 0";
        return false;
    }

    const uint8_t* code_ptr = reinterpret_cast<const uint8_t*>(csb.shader_program_va);
    ValhallTranslateOptions opts{};
    opts.stage = ShaderStage::Compute;
    opts.arch = 11;
    opts.local_size_x = csb.workgroup_size[0];
    opts.local_size_y = csb.workgroup_size[1];
    opts.local_size_z = csb.workgroup_size[2];

    auto tr = translate_valhall_to_spirv(code_ptr, 512, opts);
    if (!tr.ok) {
        if (out_err) *out_err = tr.error_message;
        return false;
    }
    if (out_translated_spirv) *out_translated_spirv = tr.spirv_words;
    if (out_disassembly) *out_disassembly = tr.disassembly_listing;

    ValhallSidebandData sb_data{};
    if (csb.fau_va != 0) {
        const uint32_t* fau_src = reinterpret_cast<const uint32_t*>(csb.fau_va);
        uint32_t n = std::min<uint32_t>(64u, csb.fau_word_count);
        std::memcpy(sb_data.fau_words, fau_src, n * sizeof(uint32_t));
    }
    sb_data.shader_pc_base = csb.shader_program_va;
    sb_data.srt_base_va = csb.srt_va;
    sb_data.local_size_xyz[0] = csb.workgroup_size[0];
    sb_data.local_size_xyz[1] = csb.workgroup_size[1];
    sb_data.local_size_xyz[2] = csb.workgroup_size[2];
    sb_data.local_size_xyz[3] = 16u;

    GpuBuffer sb_buf = alloc_buffer(sizeof(ValhallSidebandData), &sb_data, sizeof(ValhallSidebandData));
    return dispatch_spirv_compute(
        tr.spirv_words,
        { sb_buf.buffer },
        nullptr,
        0,
        csb.group_count[0],
        csb.group_count[1],
        csb.group_count[2],
        out_err);
}

bool ValhallVulkanEngine::execute_cs_run_idvs_graphics(
    const CsfStagingRegisterFile& idvs_sr,
    const CsfStagingRegisterFile& frag_sr,
    const csf::CSRUNIDVS& run_idvs_cmd,
    const uint8_t* pos_ebin_code,
    size_t pos_ebin_size,
    const uint8_t* var_ebin_code,
    size_t var_ebin_size,
    const uint8_t* frag_ebin_code,
    size_t frag_ebin_size,
    std::vector<uint32_t>* out_vert_spirv,
    std::vector<uint32_t>* out_frag_spirv,
    std::string* out_err) {

    VulkanGraphicsSideband gsb = extract_graphics_sideband(idvs_sr, frag_sr, run_idvs_cmd);
    uint32_t fb_w = std::max(1u, static_cast<uint32_t>(gsb.viewport.width));
    uint32_t fb_h = std::max(1u, static_cast<uint32_t>(gsb.viewport.height));

    // 1. Translate Vertex Shader (fusing Position EBIN + Varying EBIN)
    ValhallTranslateOptions vopts{};
    vopts.stage = ShaderStage::Vertex;
    vopts.arch = 11;
    vopts.secondary_vertex_code = var_ebin_code;
    vopts.secondary_vertex_code_size = var_ebin_size;
    auto vtr = translate_valhall_to_spirv(pos_ebin_code, pos_ebin_size, vopts);
    if (!vtr.ok) {
        if (out_err) *out_err = "Vertex translation failed: " + vtr.error_message;
        return false;
    }
    if (out_vert_spirv) *out_vert_spirv = vtr.spirv_words;

    // 2. Translate Fragment Shader
    ValhallTranslateOptions fopts{};
    fopts.stage = ShaderStage::Fragment;
    fopts.arch = 11;
    fopts.sideband_binding = 0;
    if (!gsb.textures.empty()) {
        CombinedSamplerBindingSpec sspec{};
        sspec.descriptor_set = 0;
        sspec.binding = 1;
        sspec.tex_index = 0;
        fopts.samplers.push_back(sspec);
    }
    auto ftr = translate_valhall_to_spirv(frag_ebin_code, frag_ebin_size, fopts);
    if (!ftr.ok) {
        if (out_err) *out_err = "Fragment translation failed: " + ftr.error_message;
        return false;
    }
    if (out_frag_spirv) *out_frag_spirv = ftr.spirv_words;

    // 3. Create Shader Modules
    VkShaderModule vert_mod = VK_NULL_HANDLE, frag_mod = VK_NULL_HANDLE;
    {
        VkShaderModuleCreateInfo smci{};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = vtr.spirv_words.size() * sizeof(uint32_t);
        smci.pCode = vtr.spirv_words.data();
        if (vkCreateShaderModule(device_, &smci, nullptr, &vert_mod) != VK_SUCCESS) {
            if (out_err) *out_err = "vkCreateShaderModule(Vertex) failed";
            return false;
        }
        smci.codeSize = ftr.spirv_words.size() * sizeof(uint32_t);
        smci.pCode = ftr.spirv_words.data();
        if (vkCreateShaderModule(device_, &smci, nullptr, &frag_mod) != VK_SUCCESS) {
            vkDestroyShaderModule(device_, vert_mod, nullptr);
            if (out_err) *out_err = "vkCreateShaderModule(Fragment) failed";
            return false;
        }
    }

    // 4. Create Sideband Buffer
    ValhallSidebandData sb_data{};
    if (gsb.pos_fau_va != 0) {
        const uint32_t* fsrc = reinterpret_cast<const uint32_t*>(gsb.pos_fau_va);
        uint32_t n = std::min<uint32_t>(64u, gsb.pos_fau_words ? gsb.pos_fau_words : 64u);
        std::memcpy(sb_data.fau_words, fsrc, n * sizeof(uint32_t));
    }
    sb_data.shader_pc_base = gsb.pos_spd.binary;
    sb_data.srt_base_va = gsb.pos_srt_va;
    sb_data.fb_info[0] = fb_w;
    sb_data.fb_info[1] = fb_h;
    sb_data.fb_info[2] = 1;
    sb_data.fb_info[3] = gsb.rt_count;
    GpuBuffer sb_buf = alloc_buffer(sizeof(ValhallSidebandData), &sb_data, sizeof(ValhallSidebandData));

    // 5. Optional Texture + Sampler creation from Fragment SRT
    VkImage tex_img = VK_NULL_HANDLE;
    VkDeviceMemory tex_mem = VK_NULL_HANDLE;
    VkImageView tex_view = VK_NULL_HANDLE;
    VkSampler vk_sampler = VK_NULL_HANDLE;
    GpuBuffer tex_stage_buf{};
    bool has_texture = !gsb.textures.empty() && !gsb.planes.empty() && gsb.planes[0].pointer != 0;

    if (has_texture) {
        const auto& t = gsb.textures[0];
        const auto& pl = gsb.planes[0];
        uint32_t tw = std::max(1u, t.width);
        uint32_t th = std::max(1u, t.height);

        std::vector<uint8_t> linear_rgba(tw * th * 4u, 255u);
        const uint8_t* src_texels = reinterpret_cast<const uint8_t*>(pl.pointer);
        if (pl.clump_ordering == csf::ClumpOrdering::Tiled_U_Interleaved) {
            mali_detile_morton_16x16_rgba8(src_texels, linear_rgba.data(), tw, th);
        } else {
            uint32_t stride = (pl.row_stride > 0) ? pl.row_stride : (tw * 4u);
            for (uint32_t y = 0; y < th; ++y) {
                std::memcpy(linear_rgba.data() + y * tw * 4u, src_texels + y * stride, tw * 4u);
            }
        }
        tex_stage_buf = alloc_buffer(linear_rgba.size(), linear_rgba.data(), linear_rgba.size());

        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_R8G8B8A8_UNORM;
        ici.extent = { tw, th, 1 };
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        vkCreateImage(device_, &ici, nullptr, &tex_img);

        VkMemoryRequirements mr{};
        vkGetImageMemoryRequirements(device_, tex_img, &mr);
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = find_memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vkAllocateMemory(device_, &mai, nullptr, &tex_mem);
        vkBindImageMemory(device_, tex_img, tex_mem, 0);

        VkImageViewCreateInfo ivci{};
        ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivci.image = tex_img;
        ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivci.format = VK_FORMAT_R8G8B8A8_UNORM;
        ivci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCreateImageView(device_, &ivci, nullptr, &tex_view);

        VkSamplerCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter = VK_FILTER_NEAREST;
        sci.minFilter = VK_FILTER_NEAREST;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (!gsb.samplers.empty()) {
            const auto& ms = gsb.samplers[0];
            sci.magFilter = ms.magnify_nearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
            sci.minFilter = ms.minify_nearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
        }
        vkCreateSampler(device_, &sci, nullptr, &vk_sampler);
    }

    // 6. Create Color Attachment Image, View, RenderPass, and Framebuffer
    VkImage color_img = VK_NULL_HANDLE;
    VkDeviceMemory color_mem = VK_NULL_HANDLE;
    VkImageView color_view = VK_NULL_HANDLE;
    {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_R8G8B8A8_UNORM;
        ici.extent = { fb_w, fb_h, 1 };
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        vkCreateImage(device_, &ici, nullptr, &color_img);

        VkMemoryRequirements mr{};
        vkGetImageMemoryRequirements(device_, color_img, &mr);
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = find_memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vkAllocateMemory(device_, &mai, nullptr, &color_mem);
        vkBindImageMemory(device_, color_img, color_mem, 0);

        VkImageViewCreateInfo ivci{};
        ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivci.image = color_img;
        ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivci.format = VK_FORMAT_R8G8B8A8_UNORM;
        ivci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCreateImageView(device_, &ivci, nullptr, &color_view);
    }

    VkAttachmentDescription att_desc{};
    att_desc.format = VK_FORMAT_R8G8B8A8_UNORM;
    att_desc.samples = VK_SAMPLE_COUNT_1_BIT;
    att_desc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att_desc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att_desc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att_desc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att_desc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    att_desc.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    VkAttachmentReference color_ref{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_ref;

    VkRenderPassCreateInfo rpci{};
    rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpci.attachmentCount = 1;
    rpci.pAttachments = &att_desc;
    rpci.subpassCount = 1;
    rpci.pSubpasses = &subpass;
    VkRenderPass rpass = VK_NULL_HANDLE;
    vkCreateRenderPass(device_, &rpci, nullptr, &rpass);

    VkFramebufferCreateInfo fbci{};
    fbci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbci.renderPass = rpass;
    fbci.attachmentCount = 1;
    fbci.pAttachments = &color_view;
    fbci.width = fb_w;
    fbci.height = fb_h;
    fbci.layers = 1;
    VkFramebuffer fb = VK_NULL_HANDLE;
    vkCreateFramebuffer(device_, &fbci, nullptr, &fb);

    // 7. Descriptor Set Layout & Pipeline Layout
    std::vector<VkDescriptorSetLayoutBinding> dsl_bindings;
    dsl_bindings.push_back({ 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                             VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr });
    if (has_texture) {
        dsl_bindings.push_back({ 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                 VK_SHADER_STAGE_FRAGMENT_BIT, nullptr });
    }
    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = static_cast<uint32_t>(dsl_bindings.size());
    dslci.pBindings = dsl_bindings.data();
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    vkCreateDescriptorSetLayout(device_, &dslci, nullptr, &dsl);

    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &dsl;
    VkPipelineLayout playout = VK_NULL_HANDLE;
    vkCreatePipelineLayout(device_, &plci, nullptr, &playout);

    // 8. Vertex Input bindings from v11.xml Attribute + Buffer descriptors
    std::vector<VkVertexInputBindingDescription> vi_bindings;
    std::vector<VkVertexInputAttributeDescription> vi_attrs;
    std::vector<GpuBuffer> uploaded_vbufs;

    uint32_t num_attrs = static_cast<uint32_t>(gsb.attributes.size());
    for (uint32_t i = 0; i < num_attrs; ++i) {
        const auto& ma = gsb.attributes[i];
        uint32_t b_idx = ma.buffer_index;
        VkFormat fmt = translate_mali_pixel_format(ma.format);
        uint32_t elem_stride = 16u;
        if (fmt == VK_FORMAT_R32G32B32_SFLOAT) elem_stride = 12u;
        else if (fmt == VK_FORMAT_R32G32_SFLOAT) elem_stride = 8u;
        else if (fmt == VK_FORMAT_R32_SFLOAT) elem_stride = 4u;

        if (b_idx < gsb.vertex_buffers.size() && gsb.vertex_buffers[b_idx].address != 0) {
            const auto& mb = gsb.vertex_buffers[b_idx];
            uint32_t stride = (ma.stride > 0) ? ma.stride : ((mb.stride > 0) ? mb.stride : elem_stride);
            vi_bindings.push_back({ i, stride, VK_VERTEX_INPUT_RATE_VERTEX });
            vi_attrs.push_back({ i, i, fmt, 0u });
            uint32_t off = (ma.offset > 0) ? static_cast<uint32_t>(ma.offset) : 0u;
            uint32_t rem = (mb.size > off) ? (mb.size - off) : 64u;
            GpuBuffer vb = alloc_buffer(std::max(16u, rem),
                                        reinterpret_cast<const uint8_t*>(mb.address) + off,
                                        rem);
            uploaded_vbufs.push_back(vb);
        }
    }

    VkPipelineVertexInputStateCreateInfo visci{};
    visci.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    visci.vertexBindingDescriptionCount = static_cast<uint32_t>(vi_bindings.size());
    visci.pVertexBindingDescriptions = vi_bindings.data();
    visci.vertexAttributeDescriptionCount = static_cast<uint32_t>(vi_attrs.size());
    visci.pVertexAttributeDescriptions = vi_attrs.data();

    VkPipelineInputAssemblyStateCreateInfo iasci{};
    iasci.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    iasci.topology = gsb.topology;

    VkPipelineViewportStateCreateInfo vpsci{};
    vpsci.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vpsci.viewportCount = 1;
    vpsci.pViewports = &gsb.viewport;
    vpsci.scissorCount = 1;
    vpsci.pScissors = &gsb.scissor;

    VkPipelineRasterizationStateCreateInfo rsci{};
    rsci.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rsci.polygonMode = VK_POLYGON_MODE_FILL;
    rsci.cullMode = gsb.cull_mode;
    rsci.frontFace = gsb.front_face;
    rsci.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo mssci{};
    mssci.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    mssci.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendStateCreateInfo cbsci{};
    cbsci.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cbsci.attachmentCount = 1;
    VkPipelineColorBlendAttachmentState default_blend_att{};
    default_blend_att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                       VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    if (gsb.blend_attachments.empty()) {
        gsb.blend_attachments.push_back(default_blend_att);
    }
    cbsci.attachmentCount = static_cast<uint32_t>(gsb.blend_attachments.size());
    cbsci.pAttachments = gsb.blend_attachments.data();

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert_mod;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag_mod;
    stages[1].pName = "main";

    VkGraphicsPipelineCreateInfo gpci{};
    gpci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpci.stageCount = 2;
    gpci.pStages = stages;
    gpci.pVertexInputState = &visci;
    gpci.pInputAssemblyState = &iasci;
    gpci.pViewportState = &vpsci;
    gpci.pRasterizationState = &rsci;
    gpci.pMultisampleState = &mssci;
    gpci.pDepthStencilState = &gsb.depth_stencil_state;
    gpci.pColorBlendState = &cbsci;
    gpci.layout = playout;
    gpci.renderPass = rpass;
    gpci.subpass = 0;

    VkPipeline gpipe = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gpci, nullptr, &gpipe) != VK_SUCCESS) {
        if (out_err) *out_err = "vkCreateGraphicsPipelines failed";
        return false;
    }

    // 9. Descriptor Pool & Set
    VkDescriptorPoolSize pool_sizes[2] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 }
    };
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 2;
    dpci.poolSizeCount = 2;
    dpci.pPoolSizes = pool_sizes;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    vkCreateDescriptorPool(device_, &dpci, nullptr, &dpool);

    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = dpool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &dsl;
    VkDescriptorSet dset = VK_NULL_HANDLE;
    vkAllocateDescriptorSets(device_, &dsai, &dset);

    VkDescriptorBufferInfo dbi{ sb_buf.buffer, 0, VK_WHOLE_SIZE };
    std::vector<VkWriteDescriptorSet> writes;
    VkWriteDescriptorSet w0{};
    w0.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w0.dstSet = dset;
    w0.dstBinding = 0;
    w0.descriptorCount = 1;
    w0.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w0.pBufferInfo = &dbi;
    writes.push_back(w0);

    VkDescriptorImageInfo dii{};
    if (has_texture) {
        dii.sampler = vk_sampler;
        dii.imageView = tex_view;
        dii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet w1{};
        w1.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w1.dstSet = dset;
        w1.dstBinding = 1;
        w1.descriptorCount = 1;
        w1.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w1.pImageInfo = &dii;
        writes.push_back(w1);
    }
    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    // 10. Record & Submit Command Buffer (Upload texture -> RenderPass -> Readback)
    GpuBuffer readback_buf = alloc_buffer(fb_w * fb_h * 4u);

    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = cmd_pool_;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(device_, &cbai, &cmd);

    VkCommandBufferBeginInfo cbbi{};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &cbbi);

    if (has_texture) {
        uint32_t tw = std::max(1u, gsb.textures[0].width);
        uint32_t th = std::max(1u, gsb.textures[0].height);
        VkImageMemoryBarrier imb1{};
        imb1.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        imb1.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imb1.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        imb1.srcAccessMask = 0;
        imb1.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        imb1.image = tex_img;
        imb1.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &imb1);

        VkBufferImageCopy bic{};
        bic.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        bic.imageExtent = { tw, th, 1 };
        vkCmdCopyBufferToImage(cmd, tex_stage_buf.buffer, tex_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bic);

        VkImageMemoryBarrier imb2{};
        imb2.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        imb2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        imb2.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imb2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        imb2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        imb2.image = tex_img;
        imb2.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &imb2);
    }

    VkClearValue clear_val{};
    clear_val.color = {{ 0.0f, 0.0f, 0.0f, 1.0f }};
    VkRenderPassBeginInfo rpbi{};
    rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass = rpass;
    rpbi.framebuffer = fb;
    rpbi.renderArea.offset = { 0, 0 };
    rpbi.renderArea.extent = { fb_w, fb_h };
    rpbi.clearValueCount = 1;
    rpbi.pClearValues = &clear_val;

    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gpipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, playout, 0, 1, &dset, 0, nullptr);

    if (!uploaded_vbufs.empty()) {
        std::vector<VkBuffer> vbs;
        std::vector<VkDeviceSize> offs(uploaded_vbufs.size(), 0);
        for (const auto& vb : uploaded_vbufs) vbs.push_back(vb.buffer);
        vkCmdBindVertexBuffers(cmd, 0, static_cast<uint32_t>(vbs.size()), vbs.data(), offs.data());
    }

    vkCmdDraw(cmd, std::max(3u, gsb.vertex_count), gsb.instance_count, gsb.first_vertex, 0);
    vkCmdEndRenderPass(cmd);

    VkBufferImageCopy read_bic{};
    read_bic.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    read_bic.imageExtent = { fb_w, fb_h, 1 };
    vkCmdCopyImageToBuffer(cmd, color_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback_buf.buffer, 1, &read_bic);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue_);

    // Write back to RTD[0].rgb.writeback_buffer.base
    if (!gsb.rtds.empty() && gsb.rtds[0].rgb.writeback_buffer.base != 0) {
        uint8_t* dst = reinterpret_cast<uint8_t*>(gsb.rtds[0].rgb.writeback_buffer.base);
        if (gsb.rtds[0].rgb.writeback_block_format == csf::BlockFormat::Tiled_U_Interleaved) {
            mali_retile_morton_16x16_rgba8(readback_buf.bytes(), dst, fb_w, fb_h);
        } else {
            std::memcpy(dst, readback_buf.bytes(), fb_w * fb_h * 4u);
        }
    }

    vkFreeCommandBuffers(device_, cmd_pool_, 1, &cmd);
    vkDestroyDescriptorPool(device_, dpool, nullptr);
    vkDestroyPipeline(device_, gpipe, nullptr);
    vkDestroyPipelineLayout(device_, playout, nullptr);
    vkDestroyDescriptorSetLayout(device_, dsl, nullptr);
    vkDestroyFramebuffer(device_, fb, nullptr);
    vkDestroyRenderPass(device_, rpass, nullptr);
    vkDestroyImageView(device_, color_view, nullptr);
    vkDestroyImage(device_, color_img, nullptr);
    vkFreeMemory(device_, color_mem, nullptr);
    if (has_texture) {
        vkDestroySampler(device_, vk_sampler, nullptr);
        vkDestroyImageView(device_, tex_view, nullptr);
        vkDestroyImage(device_, tex_img, nullptr);
        vkFreeMemory(device_, tex_mem, nullptr);
    }
    vkDestroyShaderModule(device_, vert_mod, nullptr);
    vkDestroyShaderModule(device_, frag_mod, nullptr);
    return true;
}

} // namespace valhall::pipeline

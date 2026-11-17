#include "valhall_isa_generated.h"
#include "valhall_csf_descriptors_generated.h"
#include "valhall_to_spirv.h"
#include "valhall_csf_pipeline.h"
#include "cmpbe_chunks.h"
#include "spirv_reflect.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

namespace vert_ns {
#include "fixtures/dumped_vert.h"
}
namespace frag_ns {
#include "fixtures/dumped_frag.h"
}

using namespace valhall;
using namespace valhall::csf;
using namespace valhall::pipeline;

static void validate_with_spirv_reflect(const std::vector<uint32_t>& spirv_words, const char* label) {
    SpvReflectShaderModule mod{};
    SpvReflectResult res = spvReflectCreateShaderModule(
        spirv_words.size() * sizeof(uint32_t), spirv_words.data(), &mod);
    if (res != SPV_REFLECT_RESULT_SUCCESS) {
        std::cerr << "[FAIL] SPIRV-Reflect validation failed for " << label << " (code=" << res << ")\n";
        std::abort();
    }
    std::cout << "[PASS] SPIRV-Reflect validated " << label
              << " (entry=\"" << mod.entry_point_name << "\", stage=" << mod.shader_stage
              << ", words=" << spirv_words.size() << ")\n";
    spvReflectDestroyShaderModule(&mod);
}

int main() {
    std::cout << "=== [Test 4] End-to-End Graphics Pipeline Roundtrip (IDVS Position+Varying+Pilot + CFRA Fragment + v11.xml FBD/RTD/ZSD/Blend/Texture -> Lavapipe) ===\n";

    ValhallVulkanEngine vk;
    std::string err;
    if (!vk.init(&err)) {
        std::cerr << "Vulkan init failed: " << err << "\n";
        return 1;
    }

    const cmpbe_chunk_CVER* cver = &vert_ns::CVER_1;
    const cmpbe_chunk_CFRA* cfra = &frag_ns::CFRA_1;
    assert(cver && cver->cmmn.nof_ebin == 3);
    assert(cfra && cfra->cmmn.nof_ebin == 1);

    // EBIN[0] = Varying (0x81), EBIN[1] = Position (0x180), EBIN[2] = Pilot (0x0)
    const auto& var_ebin   = cver->cmmn.ebin[0];
    const auto& pos_ebin   = cver->cmmn.ebin[1];
    const auto& pilot_ebin = cver->cmmn.ebin[2];
    const auto& frag_ebin  = cfra->cmmn.ebin[0];

    // ------------------------------------------------------------------------
    // Step 1: Set up UBO (4x4 identity MVP matrix) in v11.xml Buffer & SRT,
    //         and run the Vertex Pilot Shader (EBIN[2]) on Lavapipe!
    // ------------------------------------------------------------------------
    float mvp[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    GpuBuffer ubo_buf = vk.alloc_buffer(sizeof(mvp), mvp, sizeof(mvp));

    Buffer ubo_desc{};
    ubo_desc.type = DescriptorType::Buffer;
    ubo_desc.size = sizeof(mvp);
    ubo_desc.address = ubo_buf.va();
    uint32_t ubo_desc_words[Buffer::kWordCount] = {0};
    ubo_desc.pack(ubo_desc_words);
    GpuBuffer ubo_desc_gpu = vk.alloc_buffer(sizeof(ubo_desc_words), ubo_desc_words, sizeof(ubo_desc_words));

    uint64_t vert_pilot_srt_qwords[16] = {0};
    vert_pilot_srt_qwords[8] = ubo_desc_gpu.va(); // SRT + 0x40
    GpuBuffer vert_pilot_srt_gpu = vk.alloc_buffer(sizeof(vert_pilot_srt_qwords), vert_pilot_srt_qwords, sizeof(vert_pilot_srt_qwords));

    GpuBuffer pos_fau_gpu = vk.alloc_buffer(256);

    bool pilot_ok = vk.execute_valhall_pilot_shader(
        pilot_ebin.bytecode.byte_code,
        pilot_ebin.bytecode.size,
        pos_fau_gpu.va(),
        vert_pilot_srt_gpu.va(),
        11,
        &err);
    assert(pilot_ok);

    // Verify Pilot EBIN[2] copied the 4x4 identity matrix into FAU + 0x30..0x6F (u6..u13)
    const float* fau_floats = reinterpret_cast<const float*>(pos_fau_gpu.bytes());
    assert(fau_floats[12] == 1.0f && fau_floats[17] == 1.0f && fau_floats[22] == 1.0f && fau_floats[27] == 1.0f);
    std::cout << "[PASS] Translated Vertex Pilot Shader (EBIN[2]) executed on Lavapipe and loaded 4x4 MVP matrix into FAU u6..u13\n";

    // Populate gl_mali_ViewportTransform in FAU u0..u5:
    //   u0.w0 (sx) = 1.0f, u0.w1 (sy) = 1.0f
    //   u1.w0 (sz) = 1.0f, u1.w1 (oz) = 0.0f
    //   u2.w0 (ox) = 0.0f, u2.w1 (oy) = 0.0f
    //   u5.w0 (guardband) = +INFINITY
    float* fau_f = reinterpret_cast<float*>(pos_fau_gpu.bytes());
    fau_f[0] = 1.0f; fau_f[1] = 1.0f;
    fau_f[2] = 1.0f; fau_f[3] = 0.0f;
    fau_f[4] = 0.0f; fau_f[5] = 0.0f;
    fau_f[10] = 1e30f;

    // ------------------------------------------------------------------------
    // Step 2: Set up Fullscreen Quad Vertex Buffers (pos vec3 + uv vec2)
    //         and v11.xml Attribute + Buffer descriptors in Vertex SRT
    // ------------------------------------------------------------------------
    const float quad_pos[6 * 3] = {
        -1.0f, -1.0f, 0.5f,
         1.0f, -1.0f, 0.5f,
         1.0f,  1.0f, 0.5f,
        -1.0f, -1.0f, 0.5f,
         1.0f,  1.0f, 0.5f,
        -1.0f,  1.0f, 0.5f
    };
    const float quad_uv[6 * 2] = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 1.0f
    };
    GpuBuffer vbuf_pos = vk.alloc_buffer(sizeof(quad_pos), quad_pos, sizeof(quad_pos));
    GpuBuffer vbuf_uv  = vk.alloc_buffer(sizeof(quad_uv), quad_uv, sizeof(quad_uv));

    uint32_t vert_desc_table[4 * 8] = {0};
    {
        Attribute a0{};
        a0.type = DescriptorType::Attribute;
        a0.format = PixelFormat{ RGBComponentOrder::RGBA, false, Format::RGB32F, 0 };
        a0.buffer_index = 0u;
        a0.stride = 12u;
        a0.pack(vert_desc_table + 0 * 8);

        Attribute a1{};
        a1.type = DescriptorType::Attribute;
        a1.format = PixelFormat{ RGBComponentOrder::RGBA, false, Format::RG32F, 0 };
        a1.buffer_index = 1u;
        a1.stride = 8u;
        a1.pack(vert_desc_table + 1 * 8);

        Buffer b0{};
        b0.type = DescriptorType::Buffer;
        b0.size = sizeof(quad_pos);
        b0.address = vbuf_pos.va();
        b0.stride = 12u;
        b0.pack(vert_desc_table + 2 * 8);

        Buffer b1{};
        b1.type = DescriptorType::Buffer;
        b1.size = sizeof(quad_uv);
        b1.address = vbuf_uv.va();
        b1.stride = 8u;
        b1.pack(vert_desc_table + 3 * 8);
    }
    GpuBuffer vert_desc_gpu = vk.alloc_buffer(sizeof(vert_desc_table), vert_desc_table, sizeof(vert_desc_table));

    uint32_t vert_srt_words[4 * Resource::kWordCount] = {0};
    {
        Resource r0{};
        r0.address = vert_desc_gpu.va();
        r0.contains_descriptors = true;
        r0.size = sizeof(vert_desc_table);
        r0.pack(vert_srt_words);
    }
    GpuBuffer vert_srt_gpu = vk.alloc_buffer(sizeof(vert_srt_words), vert_srt_words, sizeof(vert_srt_words));

    // ------------------------------------------------------------------------
    // Step 3: Set up 32x32 Morton 16x16 U-Interleaved Texture + GenericPlane + Sampler
    //         in Fragment SRT, plus FBD + RTD (with Tiled_U_Interleaved writeback!)
    // ------------------------------------------------------------------------
    constexpr uint32_t kWidth = 32, kHeight = 32;
    std::vector<uint8_t> expected_linear_rgba(kWidth * kHeight * 4u);
    for (uint32_t y = 0; y < kHeight; ++y) {
        for (uint32_t x = 0; x < kWidth; ++x) {
            uint32_t off = (y * kWidth + x) * 4u;
            expected_linear_rgba[off + 0] = static_cast<uint8_t>(x * 7u + 15u);
            expected_linear_rgba[off + 1] = static_cast<uint8_t>(y * 7u + 31u);
            expected_linear_rgba[off + 2] = static_cast<uint8_t>((x ^ y) * 5u + 64u);
            expected_linear_rgba[off + 3] = 255u;
        }
    }
    std::vector<uint8_t> morton_texels(kWidth * kHeight * 4u);
    mali_retile_morton_16x16_rgba8(expected_linear_rgba.data(), morton_texels.data(), kWidth, kHeight);
    GpuBuffer texel_gpu = vk.alloc_buffer(morton_texels.size(), morton_texels.data(), morton_texels.size());

    GenericPlane plane{};
    plane.type = DescriptorType::Plane;
    plane.plane_type = PlaneType::Generic;
    plane.clump_ordering = ClumpOrdering::Tiled_U_Interleaved;
    plane.clump_format = ClumpFormat::RAW32;
    plane.pointer = texel_gpu.va();
    plane.width = kWidth;
    plane.height = kHeight;
    uint32_t plane_words[GenericPlane::kWordCount] = {0};
    plane.pack(plane_words);
    GpuBuffer plane_gpu = vk.alloc_buffer(sizeof(plane_words), plane_words, sizeof(plane_words));

    uint32_t frag_desc_table[2 * 8] = {0};
    {
        Texture tex{};
        tex.type = DescriptorType::Texture;
        tex.dimension = TextureDimension::V_2D;
        tex.format = PixelFormat{ RGBComponentOrder::RGBA, false, Format::RGBA8_UNORM, 0 };
        tex.width = kWidth;
        tex.height = kHeight;
        tex.levels = 1u;
        tex.array_size = 1u;
        tex.depth = 1u;
        tex.surfaces = plane_gpu.va();
        tex.pack(frag_desc_table + 0 * 8);

        Sampler samp{};
        samp.type = DescriptorType::Sampler;
        samp.magnify_nearest = true;
        samp.minify_nearest = true;
        samp.pack(frag_desc_table + 1 * 8);
    }
    GpuBuffer frag_desc_gpu = vk.alloc_buffer(sizeof(frag_desc_table), frag_desc_table, sizeof(frag_desc_table));

    uint32_t frag_srt_words[4 * Resource::kWordCount] = {0};
    {
        Resource r0{};
        r0.address = frag_desc_gpu.va();
        r0.contains_descriptors = true;
        r0.size = sizeof(frag_desc_table);
        r0.pack(frag_srt_words);
    }
    GpuBuffer frag_srt_gpu = vk.alloc_buffer(sizeof(frag_srt_words), frag_srt_words, sizeof(frag_srt_words));

    // Allocate FBD + RTD (128B FBD + 64B RTD = 192B) and writeback buffer
    GpuBuffer rt_writeback_gpu = vk.alloc_buffer(kWidth * kHeight * 4u);
    uint8_t fbd_blob[192] = {0};
    {
        FramebufferParameters fbd{};
        fbd.width = kWidth;
        fbd.height = kHeight;
        fbd.bound_max_x = kWidth - 1u;
        fbd.bound_max_y = kHeight - 1u;
        fbd.sample_count = 1u;
        fbd.render_target_count = 1u;
        fbd.pack(reinterpret_cast<uint32_t*>(fbd_blob));

        RenderTarget rtd{};
        rtd.rgb.write_enable = true;
        rtd.rgb.internal_format = ColorBufferInternalFormat::R8G8B8A8;
        rtd.rgb.writeback_format = ColorFormat::RAW32;
        rtd.rgb.writeback_block_format = BlockFormat::Tiled_U_Interleaved;
        rtd.rgb.writeback_buffer.base = rt_writeback_gpu.va();
        rtd.rgb.writeback_buffer.row_stride = kWidth * 4u;
        rtd.pack(reinterpret_cast<uint32_t*>(fbd_blob + 128));
    }
    GpuBuffer fbd_gpu = vk.alloc_buffer(sizeof(fbd_blob), fbd_blob, sizeof(fbd_blob));

    // ZSD & Blend descriptors
    DepthStencil zsd{};
    zsd.type = DescriptorType::Depth_stencil;
    zsd.depth_function = Func::Always;
    zsd.depth_write_enable = false;
    uint32_t zsd_words[DepthStencil::kWordCount] = {0};
    zsd.pack(zsd_words);
    GpuBuffer zsd_gpu = vk.alloc_buffer(sizeof(zsd_words), zsd_words, sizeof(zsd_words));

    Blend blend{};
    blend.enable = false;
    blend.equation.color_mask = 0xFu;
    uint32_t blend_words[Blend::kWordCount] = {0};
    blend.pack(blend_words);
    GpuBuffer blend_gpu = vk.alloc_buffer(sizeof(blend_words), blend_words, sizeof(blend_words));

    // Populate IDVS SR & FRAGMENT SR
    CsfStagingRegisterFile idvs_sr{};
    idvs_sr.set64(IDVSSR::SRT_0, vert_srt_gpu.va());
    idvs_sr.set64(IDVSSR::SRT_2, frag_srt_gpu.va());
    idvs_sr.set64(IDVSSR::FAU_0, pos_fau_gpu.va() | (uint64_t(16) << 56));
    idvs_sr.set64(IDVSSR::BLEND_DESC, blend_gpu.va());
    idvs_sr.set64(IDVSSR::ZSD, zsd_gpu.va());
    idvs_sr.set32(IDVSSR::INDEX_COUNT, 6u);
    idvs_sr.set32(IDVSSR::INSTANCE_COUNT, 1u);
    idvs_sr.set_reg_f32(static_cast<uint32_t>(IDVSSR::LOW_DEPTH_CLAMP), 0.0f);
    idvs_sr.set_reg_f32(static_cast<uint32_t>(IDVSSR::HIGH_DEPTH_CLAMP), 1.0f);

    CsfStagingRegisterFile frag_sr{};
    FramebufferPointer fbp{};
    fbp.type = true;
    fbp.zs_crc_extension_present = false;
    fbp.render_target_count = 1u;
    fbp.pointer = fbd_gpu.va();
    uint32_t fbp_words[2] = {0};
    fbp.pack(fbp_words);
    frag_sr.set64(FRAGMENTSR::FBD_POINTER, uint64_t(fbp_words[0]) | (uint64_t(fbp_words[1]) << 32));
    frag_sr.set32(FRAGMENTSR::BBOX_MAX, ((kHeight - 1u) << 16) | (kWidth - 1u));

    CSRUNIDVS run_idvs{};
    run_idvs.fragment_srt_select = true;

    std::vector<uint32_t> vert_spirv, frag_spirv;
    bool gfx_ok = vk.execute_cs_run_idvs_graphics(
        idvs_sr,
        frag_sr,
        run_idvs,
        pos_ebin.bytecode.byte_code,
        pos_ebin.bytecode.size,
        var_ebin.bytecode.byte_code,
        var_ebin.bytecode.size,
        frag_ebin.bytecode.byte_code,
        frag_ebin.bytecode.size,
        &vert_spirv,
        &frag_spirv,
        &err);
    if (!gfx_ok) {
        std::cerr << "execute_cs_run_idvs_graphics failed: " << err << "\n";
        return 1;
    }

    validate_with_spirv_reflect(vert_spirv, "Translated Fused Vertex (Position+Varying) SPIR-V");
    validate_with_spirv_reflect(frag_spirv, "Translated Fragment (VAR_TEX_SINGLE+BLEND) SPIR-V");

    // ------------------------------------------------------------------------
    // Step 4: Detile the Morton 16x16 RTD writeback buffer and verify pixel accuracy!
    // ------------------------------------------------------------------------
    std::vector<uint8_t> rendered_linear(kWidth * kHeight * 4u, 0u);
    mali_detile_morton_16x16_rgba8(rt_writeback_gpu.bytes(), rendered_linear.data(), kWidth, kHeight);

    size_t matching_pixels = 0;
    for (size_t p = 0; p < kWidth * kHeight; ++p) {
        const uint8_t* exp = &expected_linear_rgba[p * 4];
        const uint8_t* act = &rendered_linear[p * 4];
        if (exp[0] == act[0] && exp[1] == act[1] && exp[2] == act[2] && exp[3] == act[3]) {
            ++matching_pixels;
        } else {
            std::cerr << "[FAIL] Pixel mismatch at " << p << ": expected=("
                      << int(exp[0]) << "," << int(exp[1]) << "," << int(exp[2]) << "," << int(exp[3])
                      << ") actual=("
                      << int(act[0]) << "," << int(act[1]) << "," << int(act[2]) << "," << int(act[3]) << ")\n";
            return 1;
        }
    }
    std::cout << "[PASS] 100% PIXEL-EXACT EQUIVALENCE across all " << matching_pixels << "/" << (kWidth * kHeight)
              << " rendered pixels (32x32 Morton 16x16 texture -> VAR_TEX_SINGLE -> BLEND -> Morton 16x16 RTD writeback)!\n";
    std::cout << "=== ALL GRAPHICS PIPELINE ROUNDTRIP TESTS PASSED ===\n";
    return 0;
}

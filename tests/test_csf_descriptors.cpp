#include "valhall_csf_descriptors_generated.h"
#include "valhall_csf_pipeline.h"
#include <cassert>
#include <cstring>
#include <iostream>

using namespace valhall;
using namespace valhall::csf;
using namespace valhall::pipeline;

int main() {
    std::cout << "=== [Test 2] CSF Descriptor Pack/Unpack & Sideband Translation Verification ===\n";

    // 1. Test ShaderProgram (SPD) pack/unpack roundtrip
    {
        ShaderProgram spd{};
        spd.type = DescriptorType::Shader;
        spd.stage = csf::ShaderStage::Compute;
        spd.register_allocation = ShaderRegisterAllocation::V_32_Per_Thread;
        spd.binary = 0x0000123456789000ULL;
        spd.preload.r48_r63 = 0x101u;

        uint32_t words[ShaderProgram::kWordCount] = {0};
        spd.pack(words);
        ShaderProgram dec = ShaderProgram::unpack(words);
        assert(dec.type == DescriptorType::Shader);
        assert(dec.stage == csf::ShaderStage::Compute);
        assert(dec.register_allocation == ShaderRegisterAllocation::V_32_Per_Thread);
        assert(dec.binary == 0x0000123456789000ULL);
        assert(dec.preload.r48_r63 == 0x101u);
        std::cout << "[PASS] ShaderProgram (SPD) 32-byte pack/unpack bit-exact roundtrip\n";
    }

    // 2. Test Buffer & Attribute descriptor pack/unpack roundtrips
    {
        Buffer buf{};
        buf.type = DescriptorType::Buffer;
        buf.buffer_type = BufferType::Simple;
        buf.size = 4096u;
        buf.address = 0x00007F0011223340ULL;
        buf.stride = 16u;

        uint32_t bwords[Buffer::kWordCount] = {0};
        buf.pack(bwords);
        Buffer bdec = Buffer::unpack(bwords);
        assert(bdec.type == DescriptorType::Buffer);
        assert(bdec.size == 4096u);
        assert(bdec.address == 0x00007F0011223340ULL);
        assert(bdec.stride == 16u);

        Attribute attr{};
        attr.type = DescriptorType::Attribute;
        attr.format = PixelFormat{ RGBComponentOrder::RGBA, false, Format::RGB32F, 0 };
        attr.buffer_index = 2u;
        attr.offset = 32;
        attr.stride = 12u;

        uint32_t awords[Attribute::kWordCount] = {0};
        attr.pack(awords);
        Attribute adec = Attribute::unpack(awords);
        assert(adec.type == DescriptorType::Attribute);
        assert(adec.format.format == Format::RGB32F);
        assert(adec.buffer_index == 2u);
        assert(adec.offset == 32);
        assert(adec.stride == 12u);
        assert(translate_mali_pixel_format(adec.format) == VK_FORMAT_R32G32B32_SFLOAT);
        std::cout << "[PASS] Buffer & Attribute 32-byte pack/unpack and VkFormat translation\n";
    }

    // 3. Test Texture, GenericPlane, and Sampler pack/unpack roundtrips
    {
        GenericPlane plane{};
        plane.type = DescriptorType::Plane;
        plane.plane_type = PlaneType::Generic;
        plane.clump_ordering = ClumpOrdering::Tiled_U_Interleaved;
        plane.clump_format = ClumpFormat::RAW32;
        plane.pointer = 0x000055AA66BB7700ULL;
        plane.width = 64u;
        plane.height = 64u;

        uint32_t pwords[GenericPlane::kWordCount] = {0};
        plane.pack(pwords);
        GenericPlane pdec = GenericPlane::unpack(pwords);
        assert(pdec.clump_ordering == ClumpOrdering::Tiled_U_Interleaved);
        assert(pdec.pointer == 0x000055AA66BB7700ULL);
        assert(pdec.width == 64u && pdec.height == 64u);

        Texture tex{};
        tex.type = DescriptorType::Texture;
        tex.dimension = TextureDimension::V_2D;
        tex.format = PixelFormat{ RGBComponentOrder::RGBA, false, Format::RGBA8_UNORM, 0 };
        tex.width = 64u;
        tex.height = 64u;
        tex.levels = 1u;
        tex.array_size = 1u;
        tex.depth = 1u;
        tex.surfaces = reinterpret_cast<uint64_t>(pwords);

        uint32_t twords[Texture::kWordCount] = {0};
        tex.pack(twords);
        Texture tdec = Texture::unpack(twords);
        assert(tdec.type == DescriptorType::Texture);
        assert(tdec.width == 64u && tdec.height == 64u);
        assert(tdec.surfaces == reinterpret_cast<uint64_t>(pwords));

        Sampler samp{};
        samp.type = DescriptorType::Sampler;
        samp.magnify_nearest = true;
        samp.minify_nearest = true;
        samp.compare_function = Func::Lequal;
        uint32_t swords[Sampler::kWordCount] = {0};
        samp.pack(swords);
        Sampler sdec = Sampler::unpack(swords);
        assert(sdec.magnify_nearest && sdec.minify_nearest);
        assert(sdec.compare_function == Func::Lequal);
        std::cout << "[PASS] Texture, GenericPlane, and Sampler 32-byte pack/unpack roundtrips\n";
    }

    // 4. Test DepthStencil (ZSD), Blend, FramebufferParameters (FBD), FramebufferPointer, and RenderTarget (RTD)
    {
        DepthStencil zsd{};
        zsd.type = DescriptorType::Depth_stencil;
        zsd.depth_write_enable = true;
        zsd.depth_function = Func::Less;
        zsd.stencil_test_enable = true;
        uint32_t zwords[DepthStencil::kWordCount] = {0};
        zsd.pack(zwords);
        DepthStencil zdec = DepthStencil::unpack(zwords);
        assert(zdec.depth_write_enable == true);
        assert(zdec.depth_function == Func::Less);
        assert(zdec.stencil_test_enable == true);

        FramebufferParameters fbd{};
        fbd.width = 128u;
        fbd.height = 64u;
        fbd.bound_max_x = 127u;
        fbd.bound_max_y = 63u;
        fbd.sample_count = 1u;
        fbd.render_target_count = 1u;
        uint32_t fwords[FramebufferParameters::kWordCount] = {0};
        fbd.pack(fwords);
        FramebufferParameters fdec = FramebufferParameters::unpack(fwords);
        assert(fdec.width == 128u && fdec.height == 64u);
        assert(fdec.render_target_count == 1u);

        RenderTarget rtd{};
        rtd.rgb.write_enable = true;
        rtd.rgb.writeback_block_format = BlockFormat::Tiled_U_Interleaved;
        rtd.rgb.writeback_buffer.base = 0x0000CAFEBABE0000ULL;
        uint32_t rwords[RenderTarget::kWordCount] = {0};
        rtd.pack(rwords);
        RenderTarget rdec = RenderTarget::unpack(rwords);
        assert(rdec.rgb.write_enable == true);
        assert(rdec.rgb.writeback_block_format == BlockFormat::Tiled_U_Interleaved);
        assert(rdec.rgb.writeback_buffer.base == 0x0000CAFEBABE0000ULL);
        std::cout << "[PASS] ZSD, FBD, and RTD descriptor pack/unpack roundtrips\n";
    }

    // 5. Test Morton 16x16 U-Interleaved detile/retile roundtrip
    {
        constexpr uint32_t W = 32, H = 32;
        std::vector<uint8_t> orig(W * H * 4u);
        for (size_t i = 0; i < orig.size(); ++i) orig[i] = static_cast<uint8_t>((i * 73u + 19u) & 0xFFu);
        std::vector<uint8_t> tiled(W * H * 4u, 0);
        std::vector<uint8_t> detiled(W * H * 4u, 0);
        mali_retile_morton_16x16_rgba8(orig.data(), tiled.data(), W, H);
        mali_detile_morton_16x16_rgba8(tiled.data(), detiled.data(), W, H);
        assert(orig == detiled);
        std::cout << "[PASS] Morton 16x16 U-Interleaved retile -> detile exact bijection\n";
    }

    // 6. Test TilerContext::private_state 256-bit (8-word) nested struct roundtrip
    {
        TilerContext tc{};
        tc.hierarchy_mask = 0x1FFu;
        tc.private_state.word0 = 0x12345678u;
        tc.private_state.word7 = 0xDEADBEEFu;
        uint32_t tc_words[TilerContext::kWordCount] = {0};
        tc.pack(tc_words);
        TilerContext tc_dec = TilerContext::unpack(tc_words);
        assert(tc_dec.hierarchy_mask == 0x1FFu);
        assert(tc_dec.private_state.word0 == 0x12345678u);
        assert(tc_dec.private_state.word7 == 0xDEADBEEFu);
        std::cout << "[PASS] TilerContext::private_state 256-bit multi-word nested struct roundtrip\n";
    }

    // 7. Test SRT base|count masking (& ~0x3FULL), count > 4 entries, BLEND_DESC (& ~0xFULL), and TILER_FLAGS merge
    {
        alignas(64) uint32_t srt_words[8 * Resource::kWordCount] = {0};
        uint32_t attr_desc[Attribute::kWordCount] = {0};
        Attribute attr{};
        attr.type = DescriptorType::Attribute;
        attr.format = PixelFormat{ RGBComponentOrder::RGBA, false, Format::RGB32F, 0 };
        attr.table = 2u;
        attr.buffer_index = 0u;
        attr.offset = 12;
        attr.stride = 24u;
        attr.pack(attr_desc);

        uint32_t buf_desc[Buffer::kWordCount] = {0};
        Buffer vbuf{};
        vbuf.type = DescriptorType::Buffer;
        vbuf.buffer_type = BufferType::Simple;
        vbuf.size = 192u;
        vbuf.address = 0x00007F0022334400ULL;
        vbuf.pack(buf_desc);

        // Place Attribute table in Resource[1] and Buffer table in Resource[4] (beyond legacy 4-entry cap!)
        Resource r1{};
        r1.address = reinterpret_cast<uint64_t>(attr_desc);
        r1.contains_descriptors = true;
        r1.size = sizeof(attr_desc);
        r1.pack(srt_words + 1 * Resource::kWordCount);

        Resource r4{};
        r4.address = reinterpret_cast<uint64_t>(buf_desc);
        r4.contains_descriptors = true;
        r4.size = sizeof(buf_desc);
        r4.pack(srt_words + 4 * Resource::kWordCount);

        alignas(16) uint32_t blend_words[Blend::kWordCount] = {0};
        Blend bld{};
        bld.enable = true;
        bld.internal.mode = BlendMode::Fixed_Function;
        bld.equation.color_mask = 0xFu;
        bld.pack(blend_words);

        alignas(64) uint32_t fbd_words[128 / 4 + 64 / 4] = {0};
        FramebufferParameters fbd{};
        fbd.width = 256u;
        fbd.height = 256u;
        fbd.bound_max_x = 255u;
        fbd.bound_max_y = 255u;
        fbd.sample_count = 1u;
        fbd.render_target_count = 1u;
        fbd.pack(fbd_words);
        RenderTarget rtd{};
        rtd.rgb.write_enable = true;
        rtd.pack(fbd_words + 128 / 4);

        uint32_t fbp_words[FramebufferPointer::kWordCount] = {0};
        FramebufferPointer fbp{};
        fbp.pointer = reinterpret_cast<uint64_t>(fbd_words);
        fbp.render_target_count = 1u;
        fbp.zs_crc_extension_present = false;
        fbp.pack(fbp_words);

        CsfStagingRegisterFile idvs_sr{}, frag_sr{};
        // Encode SRT_0 = 64B-aligned base | count(8) (mirrors real r54p3 libGLES_mali.so 0x...c1c8)
        uint64_t srt_base = reinterpret_cast<uint64_t>(srt_words);
        assert((srt_base & 0x3FULL) == 0);
        idvs_sr.set64(IDVSSR::SRT_0, srt_base | 8ULL);
        // Encode BLEND_DESC = 16B-aligned base | count(1) (mirrors 0x...c0c1)
        uint64_t blend_base = reinterpret_cast<uint64_t>(blend_words);
        assert((blend_base & 0xFULL) == 0);
        idvs_sr.set64(IDVSSR::BLEND_DESC, blend_base | 1ULL);
        // Set TILER_FLAGS in r56 (Triangles) while flags_override is 0
        PrimitiveFlags pf{};
        pf.draw_mode = DrawMode::Triangles;
        pf.index_type = IndexType::UINT16;
        uint32_t pf_words[2] = {0};
        pf.pack(pf_words);
        idvs_sr.set32(IDVSSR::TILER_FLAGS, pf_words[0]);
        frag_sr.set64(FRAGMENTSR::FBD_POINTER, uint64_t(fbp_words[0]) | (uint64_t(fbp_words[1]) << 32));

        CSRUNIDVS cmd{};
        cmd.flags_override = 0u;
        VulkanGraphicsSideband gsb = extract_graphics_sideband(idvs_sr, frag_sr, cmd);
        assert(gsb.pos_srt_va == srt_base);
        assert(gsb.pos_srt_count == 8u);
        assert(gsb.attributes.size() == 1u);
        assert(gsb.attributes[0].offset == 12);
        assert(gsb.vertex_buffers.size() == 1u);
        assert(gsb.vertex_buffers[0].address == 0x00007F0022334400ULL);
        assert(gsb.blend_va == blend_base);
        assert(gsb.blend_count == 1u);
        assert(gsb.blends.size() == 1u && gsb.blends[0].enable == true);
        assert(gsb.topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        assert(gsb.index_type == VK_INDEX_TYPE_UINT16);
        std::cout << "[PASS] SRT base|count (& ~0x3FULL), Resource[4], BLEND_DESC (& ~0xFULL), and TILER_FLAGS merge\n";
    }

    std::cout << "=== ALL CSF DESCRIPTOR & SIDEBAND TESTS PASSED ===" << std::endl;
    return 0;
}

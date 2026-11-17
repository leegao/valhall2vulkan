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

    std::cout << "=== ALL CSF DESCRIPTOR & SIDEBAND TESTS PASSED ===\n";
    return 0;
}

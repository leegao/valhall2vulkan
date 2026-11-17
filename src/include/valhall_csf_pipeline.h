#pragma once
#include "valhall_isa_generated.h"
#include "valhall_csf_descriptors_generated.h"
#include "valhall_to_spirv.h"
#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>
#include <functional>
#include <memory>

namespace valhall::pipeline {

// ============================================================================
// 1. Hardware 64-Word CSF Staging Register File (v11.xml - v14.xml)
// ============================================================================
struct CsfStagingRegisterFile {
    uint32_t regs[64] = {0};

    uint32_t reg32(uint32_t idx) const { return regs[idx & 63u]; }
    void set_reg32(uint32_t idx, uint32_t val) { regs[idx & 63u] = val; }

    float reg_f32(uint32_t idx) const {
        float f = 0.0f;
        uint32_t u = reg32(idx);
        std::memcpy(&f, &u, sizeof(float));
        return f;
    }
    void set_reg_f32(uint32_t idx, float f) {
        uint32_t u = 0;
        std::memcpy(&u, &f, sizeof(float));
        set_reg32(idx, u);
    }

    uint64_t reg64(uint32_t idx) const {
        uint32_t lo = regs[idx & 62u];
        uint32_t hi = regs[(idx & 62u) + 1u];
        return uint64_t(lo) | (uint64_t(hi) << 32);
    }
    void set_reg64(uint32_t idx, uint64_t val) {
        regs[idx & 62u] = static_cast<uint32_t>(val & 0xFFFFFFFFULL);
        regs[(idx & 62u) + 1u] = static_cast<uint32_t>(val >> 32);
    }

    // Typed accessors using v11.xml COMPUTESR, IDVSSR, FRAGMENTSR, FULLSCREENSR enums
    uint32_t get32(csf::COMPUTESR r) const { return reg32(static_cast<uint32_t>(r)); }
    uint64_t get64(csf::COMPUTESR r) const { return reg64(static_cast<uint32_t>(r)); }
    void set32(csf::COMPUTESR r, uint32_t v) { set_reg32(static_cast<uint32_t>(r), v); }
    void set64(csf::COMPUTESR r, uint64_t v) { set_reg64(static_cast<uint32_t>(r), v); }

    uint32_t get32(csf::IDVSSR r) const { return reg32(static_cast<uint32_t>(r)); }
    uint64_t get64(csf::IDVSSR r) const { return reg64(static_cast<uint32_t>(r)); }
    void set32(csf::IDVSSR r, uint32_t v) { set_reg32(static_cast<uint32_t>(r), v); }
    void set64(csf::IDVSSR r, uint64_t v) { set_reg64(static_cast<uint32_t>(r), v); }

    uint32_t get32(csf::FRAGMENTSR r) const { return reg32(static_cast<uint32_t>(r)); }
    uint64_t get64(csf::FRAGMENTSR r) const { return reg64(static_cast<uint32_t>(r)); }
    void set32(csf::FRAGMENTSR r, uint32_t v) { set_reg32(static_cast<uint32_t>(r), v); }
    void set64(csf::FRAGMENTSR r, uint64_t v) { set_reg64(static_cast<uint32_t>(r), v); }

    uint32_t get32(csf::FULLSCREENSR r) const { return reg32(static_cast<uint32_t>(r)); }
    uint64_t get64(csf::FULLSCREENSR r) const { return reg64(static_cast<uint32_t>(r)); }
    void set32(csf::FULLSCREENSR r, uint32_t v) { set_reg32(static_cast<uint32_t>(r), v); }
    void set64(csf::FULLSCREENSR r, uint64_t v) { set_reg64(static_cast<uint32_t>(r), v); }
};

// ============================================================================
// 2. Extracted Sideband Structures from v11.xml CSF Registers & Descriptors
// ============================================================================
struct VulkanComputeSideband {
    csf::CSRUNCOMPUTE cmd{};
    uint64_t srt_va = 0;
    uint64_t spd_va = 0;
    uint64_t tsd_va = 0;
    uint64_t fau_va = 0;
    uint32_t fau_word_count = 0;

    csf::ShaderProgram spd{};
    uint64_t shader_program_va = 0;

    uint32_t workgroup_size[3] = {1, 1, 1};
    uint32_t group_count[3] = {1, 1, 1};
    uint32_t wg_offset[3] = {0, 0, 0};
};

struct VulkanGraphicsSideband {
    csf::CSRUNIDVS cmd{};

    uint64_t pos_spd_va = 0;
    uint64_t var_spd_va = 0;
    uint64_t frag_spd_va = 0;

    uint64_t pos_srt_va = 0;
    uint64_t var_srt_va = 0;
    uint64_t frag_srt_va = 0;

    uint64_t pos_fau_va = 0;
    uint32_t pos_fau_words = 0;
    uint64_t var_fau_va = 0;
    uint32_t var_fau_words = 0;
    uint64_t frag_fau_va = 0;
    uint32_t frag_fau_words = 0;

    csf::ShaderProgram pos_spd{};
    csf::ShaderProgram var_spd{};
    csf::ShaderProgram frag_spd{};

    // Framebuffer & Render Target Descriptors (FBD + ZS CRC + RTDs)
    uint64_t fbd_va = 0;
    uint32_t rt_count = 1;
    bool has_zs_crc_ext = false;
    csf::FramebufferParameters fbd{};
    csf::ZSCRCExtension zs_ext{};
    std::vector<csf::RenderTarget> rtds;

    // Depth/Stencil (ZSD) & Blend Descriptors
    uint64_t zsd_va = 0;
    csf::DepthStencil zsd{};
    uint64_t blend_va = 0;
    std::vector<csf::Blend> blends;

    // Draw state from IDVS SR
    csf::DCDFlags0 dcd0{};
    csf::DCDFlags1 dcd1{};
    csf::DCDFlags2 dcd2{};
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkCullModeFlags cull_mode = VK_CULL_MODE_NONE;
    VkFrontFace front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    VkViewport viewport{};
    VkRect2D scissor{};
    VkPipelineDepthStencilStateCreateInfo depth_stencil_state{};
    std::vector<VkPipelineColorBlendAttachmentState> blend_attachments;

    uint32_t vertex_count = 0;
    uint32_t instance_count = 1;
    uint32_t first_vertex = 0;
    int32_t vertex_offset = 0;
    uint64_t index_buffer_va = 0;
    VkIndexType index_type = VK_INDEX_TYPE_UINT32;
    float point_size = 1.0f;

    // Vertex Attributes & Buffers from SRT
    std::vector<csf::Attribute> attributes;
    std::vector<csf::Buffer> vertex_buffers;

    // Textures & Samplers from SRT
    std::vector<csf::Texture> textures;
    std::vector<csf::GenericPlane> planes;
    std::vector<csf::Sampler> samplers;
};

// ============================================================================
// 3. Morton 16x16 U-Interleaved Tiling & Format Helpers
// ============================================================================
uint32_t mali_morton_16x16_pixel_index(uint32_t x, uint32_t y);
void mali_detile_morton_16x16_rgba8(const uint8_t* tiled_src, uint8_t* linear_dst, uint32_t width, uint32_t height);
void mali_retile_morton_16x16_rgba8(const uint8_t* linear_src, uint8_t* tiled_dst, uint32_t width, uint32_t height);

VkFormat translate_mali_pixel_format(csf::PixelFormat pf);
VkCompareOp translate_mali_compare_func(csf::Func f);
VkStencilOp translate_mali_stencil_op(csf::StencilOp op);
VkPrimitiveTopology translate_mali_draw_mode(csf::DrawMode dm);
VkIndexType translate_mali_index_type(csf::IndexType it, bool* out_indexed);

// Extracts VulkanComputeSideband from v11.xml COMPUTE SR + CS RUN_COMPUTE
VulkanComputeSideband extract_compute_sideband(
    const CsfStagingRegisterFile& sr,
    const csf::CSRUNCOMPUTE& run_compute_cmd);

// Extracts VulkanGraphicsSideband from v11.xml IDVS SR + FRAGMENT SR + CS RUN_IDVS
VulkanGraphicsSideband extract_graphics_sideband(
    const CsfStagingRegisterFile& idvs_sr,
    const CsfStagingRegisterFile& frag_sr,
    const csf::CSRUNIDVS& run_idvs_cmd);

// ============================================================================
// 4. Self-Contained Vulkan (Lavapipe) Execution Engine
// ============================================================================
struct GpuBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceAddress device_address = 0;
    void* mapped = nullptr;

    uint64_t va() const { return static_cast<uint64_t>(device_address); }
    uint8_t* bytes() const { return static_cast<uint8_t*>(mapped); }
    uint32_t* words() const { return static_cast<uint32_t*>(mapped); }
};

class ValhallVulkanEngine {
public:
    ValhallVulkanEngine();
    ~ValhallVulkanEngine();

    bool init(std::string* out_err = nullptr);
    bool is_initialized() const { return device_ != VK_NULL_HANDLE; }
    std::string device_name() const { return device_name_; }
    VkDevice device() const { return device_; }

    // Allocate a host-coherent GPU buffer with a real 64-bit Vulkan BufferDeviceAddress (VA)
    GpuBuffer alloc_buffer(VkDeviceSize size_bytes, const void* init_data = nullptr, VkDeviceSize init_size = 0);
    void free_buffer(GpuBuffer& buf);

    // Execute a raw SPIR-V compute shader with arbitrary StorageBuffer bindings
    bool dispatch_spirv_compute(
        const std::vector<uint32_t>& spirv_words,
        const std::vector<VkBuffer>& ssbo_buffers,
        const void* push_constants,
        uint32_t push_constants_size,
        uint32_t group_x,
        uint32_t group_y,
        uint32_t group_z,
        std::string* out_err = nullptr);

    // Execute a Mali Valhall Pilot Shader (EBIN stage=0, variant=0) on Vulkan!
    // The translated Pilot SPIR-V shader reads the v11.xml SRT & Buffer descriptors at srt_va
    // and writes computed FAU words directly into fau_dst_va on the GPU.
    bool execute_valhall_pilot_shader(
        const uint8_t* pilot_code,
        size_t pilot_code_size,
        uint64_t fau_dst_va,
        uint64_t srt_va,
        uint8_t arch = 11,
        std::string* out_err = nullptr);

    // Execute a Mali Valhall Compute Dispatch using v11.xml COMPUTE SR + CS RUN_COMPUTE!
    bool execute_cs_run_compute(
        const CsfStagingRegisterFile& compute_sr,
        const csf::CSRUNCOMPUTE& run_compute_cmd,
        std::vector<uint32_t>* out_translated_spirv = nullptr,
        std::string* out_disassembly = nullptr,
        std::string* out_err = nullptr);

    // Execute a Mali Valhall Graphics Draw (IDVS + Fragment) using v11.xml IDVS SR + FRAGMENT SR!
    bool execute_cs_run_idvs_graphics(
        const CsfStagingRegisterFile& idvs_sr,
        const CsfStagingRegisterFile& frag_sr,
        const csf::CSRUNIDVS& run_idvs_cmd,
        const uint8_t* pos_ebin_code,
        size_t pos_ebin_size,
        const uint8_t* var_ebin_code,
        size_t var_ebin_size,
        const uint8_t* frag_ebin_code,
        size_t frag_ebin_size,
        std::vector<uint32_t>* out_vert_spirv = nullptr,
        std::vector<uint32_t>* out_frag_spirv = nullptr,
        std::string* out_err = nullptr);

private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_dev_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = 0;
    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    std::string device_name_;
    std::vector<GpuBuffer> tracked_buffers_;

    uint32_t find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags props) const;
};

} // namespace valhall::pipeline

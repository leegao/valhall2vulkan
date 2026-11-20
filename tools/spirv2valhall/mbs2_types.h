#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace valhall::spirv2valhall {

enum class EbinRole {
    Unknown = 0,
    MainCompute,
    Pilot,
    VertexPosition,
    VertexVarying,
    Fragment,
    TessControl,
    TessEval,
    Geometry,
    Blend,
};

inline const char* ebin_role_name(EbinRole role) {
    switch (role) {
        case EbinRole::MainCompute: return "MAIN COMPUTE SHADER";
        case EbinRole::Pilot: return "PILOT SHADER";
        case EbinRole::VertexPosition: return "VERTEX POSITION SHADER";
        case EbinRole::VertexVarying: return "VERTEX VARYING SHADER";
        case EbinRole::Fragment: return "FRAGMENT SHADER";
        case EbinRole::TessControl: return "TESS CONTROL SHADER";
        case EbinRole::TessEval: return "TESS EVAL SHADER";
        case EbinRole::Geometry: return "GEOMETRY SHADER";
        case EbinRole::Blend: return "BLEND SHADER";
        default: return "SHADER";
    }
}

struct Mbs2Vehw {
    uint32_t core = 0;
    uint32_t major = 0;
    uint32_t minor = 0;
};

struct Mbs2Tpge {
    uint32_t scalar_type = 0;    // 1=bool, 2=int, 3=uint, 4=float
    uint8_t vector_size = 1;
    uint32_t scalar_size = 2;    // 0=8b, 1=16b, 2=32b, 3=64b
    uint32_t precision = 2;      // 0=low, 1=medium, 2=high
    uint32_t stride = 0;
    uint32_t aux_qualifier = 0;
    uint8_t component = 0;
};

struct Mbs2Tpma {
    uint8_t nof_columns = 0;
    bool is_row_major = false;
    uint16_t major_stride = 0;
    Mbs2Tpge column_type;
};

struct Mbs2Type;

struct Mbs2Tpse {
    std::string name;
    uint32_t offset = 0;
    uint16_t xfb_offset = 0xFFFFu;
    uint32_t flags = 0;
    uint16_t location = 0xFFFFu;
    uint32_t semantics = 0;
    std::shared_ptr<Mbs2Type> type;
};

struct Mbs2Tpib {
    uint32_t layout = 0;       // 0=std140, 1=std430
    uint32_t kind = 0;         // 0=UBO, 1=SSBO, 2=Input, 3=Output
    uint32_t stride = 0;
    std::string instance_name;
    std::vector<Mbs2Tpse> members;
};

struct Mbs2Tpst {
    uint32_t stride = 0;
    std::string type_name;
    std::vector<Mbs2Tpse> members;
};

struct Mbs2Tpar {
    uint32_t array_size = 0;  // 0 = runtime-sized array
    std::shared_ptr<Mbs2Type> element_type;
};

struct Mbs2Tppo {
    uint32_t scalar_size = 0;
    uint32_t address_space = 0;
    std::shared_ptr<Mbs2Type> pointed_type;
};

struct Mbs2Type {
    bool has_tpac = false;
    uint32_t tpac_offset = 0;

    bool has_tpge = false;
    Mbs2Tpge tpge;

    bool has_tpma = false;
    Mbs2Tpma tpma;

    bool has_tpar = false;
    Mbs2Tpar tpar;

    bool has_tpib = false;
    Mbs2Tpib tpib;

    bool has_tpst = false;
    Mbs2Tpst tpst;

    bool has_tppo = false;
    Mbs2Tppo tppo;

    bool has_tpsa = false;
    uint32_t tpsa_kind = 0;

    bool has_tpmd = false;
    uint32_t tpmd_flags = 0;
};

struct Mbs2Rloc {
    uint16_t sym_idx = 0;
    uint16_t reg_offset = 0;  // FAU byte offset (u<reg_offset/8>.w<(reg_offset%8)/4>)
    uint16_t uni_offset = 0;  // Offset within uniform/push-constant block
    uint16_t size = 0;        // Byte size
};

struct Mbs2Relo {
    uint32_t offset = 0;
    uint32_t kind = 0;
    uint8_t section_idx = 0;
};

struct Mbs2Fixp {
    uint32_t kind = 0;
    uint32_t offset = 0;
    uint32_t value = 0;
    bool needs_high = false;
};

struct Mbs2Symd {
    uint32_t size = 0;
    uint32_t alignment = 0;
    uint32_t addressspace = 0;
    uint8_t sym_hidden = 0;
    uint8_t sym_restrict = 0;
    bool may_read_yuv = false;
    uint8_t force_invariance = 0;
};

struct Mbs2Symbol {
    std::string name;
    uint32_t flags = 0;
    uint32_t semantics = 0;
    uint8_t rack_index = 0;
    uint32_t address = 0;
    uint16_t binding = 0xFFFFu;
    uint16_t location = 0xFFFFu;
    std::shared_ptr<Mbs2Type> type;
    std::vector<Mbs2Rloc> rlocs;
    std::vector<Mbs2Relo> relos;
    std::vector<Mbs2Fixp> fixps;
    std::vector<uint32_t> sini_values;
    bool has_symd = false;
    Mbs2Symd symd;
};

struct Mbs2Fcst {
    uint16_t size = 0;
    uint16_t reg_offset = 0;  // FAU byte offset
    uint32_t value = 0;
};

struct Mbs2Dbug {
    std::vector<Mbs2Rloc> rlocs;
    std::vector<Mbs2Fcst> fcsts;
};

struct Mbs2Spdf {
    std::vector<uint8_t> rt_indices;
    std::vector<uint32_t> sizes;
    std::vector<uint32_t> types;
};

struct Mbs2Pdsc {
    uint16_t varying0 = 0;
    uint16_t varying1 = 0;
    uint8_t primary_shader_wait = 0;
};

struct Mbs2Vlkn {
    uint16_t used_set_bitmask = 0;
    uint16_t stack_layers = 0;
    uint8_t dynamic_outputs = 0;
};

struct Mbs2Fnmd {
    std::vector<uint32_t> tvars;
    uint8_t is_other_function = 0;
    uint8_t needs_structured_cfg = 0;
    uint8_t is_vertex_piece = 0;
    uint8_t is_simd_state_caller = 0;
    uint8_t is_callable_shaders_caller = 0;
    uint8_t is_discarded_function = 0;
    uint32_t reg_mode = 0;
    std::string pilot_name;
    std::string late_codegen;
    uint32_t attrib_mask = 0;
    uint32_t max_register_pressure = 0;
    uint32_t mrtt = 0;
};

/// Metadata for a single EBIN chunk (excludes raw bytecode bytes).
struct Mbs2EbinMeta {
    uint32_t index = 0;
    std::string var_name;  // e.g. "EBIN_42"
    EbinRole role = EbinRole::Unknown;
    uint16_t rt_num = 0;
    uint16_t aux_idx = 0;
    uint32_t prepass_flags = 0;
    uint32_t data_section_offset = 0xFFFFFFFFu;
    uint16_t nof_reg = 0;
    uint16_t nof_rmu = 0;
    uint32_t tls_size = 0;
    uint32_t flags = 0;
    uint32_t variant = 0;
    std::string entrypoint_name;
    uint32_t objc_total_size = 0;
    uint32_t code_size = 0;
    bool has_dbug = false;
    Mbs2Dbug dbug;
    std::vector<Mbs2Relo> grel;
    bool has_spdf = false;
    Mbs2Spdf spdf;
    bool has_pdsc = false;
    Mbs2Pdsc pdsc;
    bool has_vlkn = false;
    Mbs2Vlkn vlkn;
    bool has_fnmd = false;
    Mbs2Fnmd fnmd;
};

struct Mbs2Cxfb {
    uint16_t strides[4] = {0, 0, 0, 0};
};

struct Mbs2Cmmn {
    uint32_t vela_lang = 0;
    std::vector<Mbs2Symbol> sinp;
    std::vector<Mbs2Symbol> sout;
    std::vector<Mbs2Symbol> suni;
    std::vector<Mbs2Symbol> simg;
    std::vector<Mbs2Symbol> ssmp;
    std::vector<Mbs2Symbol> s3bo;
    std::vector<uint32_t> ubuf_sizes;
    uint16_t epilogue_jump_index = 0;
    uint16_t written_pos_fifo_attrs = 0;
    std::vector<Mbs2Fcst> fcsts;
    std::vector<Mbs2EbinMeta> ebins;
    bool has_cxfb = false;
    Mbs2Cxfb cxfb;
};

struct Mbs2Kpar {
    std::string declared_name;
    std::string uniform_name;
    uint32_t flags = 0;
};

struct Mbs2Kern {
    std::string name;
    std::vector<Mbs2Kpar> kpars;
    uint32_t workgroup_local_size = 0;
    bool has_workgroup_size = false;
    uint32_t wg_x = 1;
    uint32_t wg_y = 1;
    uint32_t wg_z = 1;
};

struct Mbs2Outv {
    std::string name;
    std::shared_ptr<Mbs2Type> type;
    uint16_t location = 0;
    uint16_t nb_locations_used = 1;
};

struct Mbs2Achn {
    std::string sampler_name;
    std::string uniform_name;
    uint32_t kind = 0;
    uint8_t rt = 0;
    uint8_t sampler_component = 0;
    uint8_t uniform_component = 0;
};

struct Mbs2CfraExtras {
    std::vector<Mbs2Outv> output_vars;
    bool has_lcal = false;
    uint8_t local_storage_size = 0;
    bool has_bldm = false;
    uint32_t bldm_flags = 0;
    std::vector<Mbs2Achn> alpha_info;
    bool has_spas = false;
    uint16_t spas_read_mask = 0;
    uint16_t spas_write_mask = 0;
};

struct Mbs2CverExtras {
    bool has_vmvw = false;
    uint8_t num_views = 0;
    bool has_vrsm = false;
    uint16_t vrs_modes_mask = 0;
};

struct Mbs2Mtes {
    uint32_t n_vertices = 0;
    uint32_t primitive_mode = 0;
    uint32_t vertex_spacing = 0;
    uint32_t ordering = 0;
    bool point_mode = false;
};

struct Mbs2Mgeo {
    uint16_t n_invocations = 0;
    uint16_t max_vertices = 0;
    uint32_t input_primitive_mode = 0;
    uint32_t output_primitive_mode = 0;
};

struct Mbs2Ploc {
    uint16_t reg_offset = 0;
    uint16_t size = 0;
    uint16_t orig_offset = 0;
};

struct Mbs2Momd {
    uint8_t prealloc_uni_buffers = 0;
    uint8_t prealloc_fau = 0;
    uint8_t ble_preserve_registers = 0;
    uint8_t essl_on_spirv = 0;
    uint8_t has_callable_blend_shaders = 0;
    uint8_t atest_wait_on_6_7 = 0;
    uint8_t gfx_hal = 0;
    uint8_t robustness = 0;
    uint8_t per_sample_shading = 0;
    uint8_t setup_fau_in_pilot = 0;
    uint8_t has_simd_call = 0;
    uint8_t has_simd_with_static_stack_frame = 0;
    uint32_t max_prealloc_addr = 0;
    std::vector<Mbs2Ploc> plocs;
};

/// Complete structured representation of cmpbe_chunk_MBS2 (minus raw EBIN bytecode).
struct Mbs2Module {
    uint32_t version = 0;
    Mbs2Vehw vehw;
    std::string stage_chunk;  // "CCOM", "CFRA", "CVER", "CTSC", "CTSE", "CGEO", "CBLE"
    Mbs2Cmmn cmmn;

    bool has_kern = false;
    Mbs2Kern kern;

    bool has_cfra = false;
    Mbs2CfraExtras cfra;

    bool has_cver = false;
    Mbs2CverExtras cver;

    bool has_mtes = false;
    Mbs2Mtes mtes;
    uint32_t ctsc_workgroup_local_size = 0;

    bool has_mgeo = false;
    Mbs2Mgeo mgeo;

    bool has_momd = false;
    Mbs2Momd momd;

    size_t ast_global_count = 0;
};

/// Format an Mbs2Type into a concise GLSL-like type string (e.g. "ivec2", "uint[]", "UBO{...}").
std::string format_mbs2_type(const Mbs2Type* t);

/// Helper to format a FAU byte offset as "u<slot>.w<lane>" (e.g. 0x1c -> "u3.w1", 0x08 -> "u1.w0").
std::string format_fau_reg(uint32_t byte_offset, uint32_t byte_size = 4);

}  // namespace valhall::spirv2valhall

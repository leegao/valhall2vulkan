#include "panfrost_vk_compiler.h"

#include <dlfcn.h>
#include <elf.h>
#include <fcntl.h>
#include <link.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>

namespace valhall::spirv2valhall {

namespace {

// Mesa 26.3 ABI structures matching the prebuilt panfrost_compile DWARF layout.
struct SpirvToNirOptionsAbi {
    uint32_t environment;
    uint8_t pad0[20];
    const void* capabilities;
    uint32_t ubo_addr_format;
    uint32_t ssbo_addr_format;
    uint32_t phys_ssbo_addr_format;
    uint32_t push_const_addr_format;
    uint32_t shared_addr_format;
    uint32_t task_payload_addr_format;
    uint32_t global_addr_format;
    uint32_t temp_addr_format;
    uint32_t constant_addr_format;
    uint32_t min_ubo_alignment;
    uint32_t min_ssbo_alignment;
    uint8_t pad1[144 - 76];
};
static_assert(sizeof(SpirvToNirOptionsAbi) == 144, "spirv_to_nir_options ABI size mismatch");

struct PanCompileInputsAbi {
    uint64_t gpu_id;
    uint32_t gpu_variant;
    bool is_blend;
    bool is_blit;
    bool no_idvs;
    uint8_t pad0;
    uint32_t view_mask;
    uint8_t fragcolor_nr_cbufs;
    uint8_t pad1[3];
    uint32_t robust_modes;
    bool robust_descriptors;
    uint8_t pad2[3];
    const void* varying_layout;
    uint32_t fau_reserved;
    uint32_t fau_pushable_ubos;
    bool fau_promote_immediates;
    uint8_t pad3[7];
};
static_assert(sizeof(PanCompileInputsAbi) == 56, "pan_compile_inputs ABI size mismatch");

struct UtilDynarrayAbi {
    void* mem_ctx;
    void* data;
    uint32_t size;
    uint32_t capacity;
};
static_assert(sizeof(UtilDynarrayAbi) == 24, "util_dynarray ABI size mismatch");

struct ExecNodeAbi {
    ExecNodeAbi* next;
    ExecNodeAbi* prev;
};

struct ExecListAbi {
    ExecNodeAbi head_sentinel;
    ExecNodeAbi tail_sentinel;
};

struct NirCursorAbi {
    uint32_t option;
    uint32_t pad;
    void* ptr;
};

// Mesa 26.3 NIR enum constants verified against panfrost_compile DWARF.
constexpr uint32_t kNirVarShaderIn = 1u << 2;
constexpr uint32_t kNirVarShaderOut = 1u << 3;
constexpr uint32_t kNirVarMemUbo = 1u << 7;
constexpr uint32_t kNirVarMemPushConst = 1u << 8;
constexpr uint32_t kNirVarMemSsbo = 1u << 9;
constexpr uint32_t kNirVarMemShared = 1u << 24;
constexpr uint32_t kNirVarAll = 0x03ffffffu;

constexpr uint32_t kNirAddr64BitGlobal = 1;
constexpr uint32_t kNirAddr32BitIndexOffset = 5;
constexpr uint32_t kNirAddr32BitOffset = 9;

constexpr uint32_t kNirInstrTypeTex = 3;
constexpr uint32_t kNirInstrTypeIntrinsic = 4;

constexpr uint32_t kNirTexSrcTextureDeref = 13;
constexpr uint32_t kNirTexSrcSamplerDeref = 14;

constexpr uint32_t kNirIntrinLoadPushConstant = 524;
constexpr uint32_t kNirIntrinLoadViewportOffset = 729;
constexpr uint32_t kNirIntrinLoadViewportScale = 730;
constexpr uint32_t kNirIntrinLoadVulkanDescriptor = 742;
constexpr uint32_t kNirIntrinVulkanResourceIndex = 946;

constexpr size_t kPanShaderInfoFauOffset = 420;
constexpr size_t kPanFauWordsCount = 128;

/// Resolved Mesa 26.3 compiler function pointers loaded once from panfrost_compile.
struct MesaVkApi {
    bool loaded = false;
    std::string loaded_path;
    std::string load_error;

    void (*glsl_init)() = nullptr;
    const void* (*pan_get_opts)(unsigned, int, bool) = nullptr;
    void* (*spirv_to_nir)(const uint32_t*, size_t, void*, int, const char*, const void*, const void*) = nullptr;
    bool (*nir_lower_variable_initializers)(void*, uint32_t) = nullptr;
    bool (*nir_lower_returns)(void*) = nullptr;
    bool (*nir_inline_functions)(void*) = nullptr;
    void (*nir_remove_non_entrypoints)(void*) = nullptr;
    bool (*nir_lower_io_vars_to_temporaries)(void*, void*, uint32_t) = nullptr;
    bool (*nir_lower_global_vars_to_local)(void*) = nullptr;
    bool (*nir_split_var_copies)(void*) = nullptr;
    bool (*nir_opt_copy_prop_vars)(void*) = nullptr;
    bool (*nir_opt_combine_stores)(void*, uint32_t) = nullptr;
    bool (*nir_lower_var_copies)(void*) = nullptr;
    bool (*nir_lower_memcpy)(void*) = nullptr;
    bool (*nir_lower_vars_to_ssa)(void*) = nullptr;
    bool (*nir_lower_system_values)(void*) = nullptr;
    bool (*nir_lower_compute_system_values)(void*, const void*) = nullptr;
    bool (*nir_lower_vars_to_explicit_types)(void*, uint32_t, void*) = nullptr;
    void* glsl_cl_size_align = nullptr;
    void* type_size_vec4 = nullptr;
    bool (*nir_lower_explicit_io)(void*, uint32_t, uint32_t) = nullptr;
    void (*nir_assign_io_var_locations)(void*, uint32_t, unsigned*, int) = nullptr;
    bool (*nir_lower_io)(void*, uint32_t, void*, uint32_t) = nullptr;
    void (*pan_preprocess_nir)(void*, uint64_t) = nullptr;
    void (*pan_postprocess_nir)(void*, const void*, void*) = nullptr;
    void (*pan_varying_collect_formats)(void*, void*, uint64_t) = nullptr;
    void (*pan_build_varying_layout_compact)(void*, void*, uint64_t) = nullptr;
    bool (*nir_opt_constant_folding)(void*) = nullptr;
    bool (*nir_opt_dce)(void*) = nullptr;
    void (*pan_shader_compile)(void*, void*, void*, void*) = nullptr;
    void* (*nir_load_const_instr_create)(void*, unsigned, unsigned) = nullptr;
    void (*nir_instr_insert)(NirCursorAbi, void*) = nullptr;
    void (*nir_def_rewrite_uses)(void*, void*) = nullptr;
    void (*nir_instr_remove_v)(void*) = nullptr;
    void (*nir_tex_instr_remove_src)(void*, unsigned) = nullptr;
};

static bool load_elf_symtab(
    const std::string& elf_path,
    std::unordered_map<std::string, uint64_t>* out_syms,
    std::string* err) {
    int fd = ::open(elf_path.c_str(), O_RDONLY);
    if (fd < 0) {
        if (err) *err = "Failed to open " + elf_path;
        return false;
    }
    struct stat st {};
    if (::fstat(fd, &st) != 0 || st.st_size < static_cast<off_t>(sizeof(Elf64_Ehdr))) {
        ::close(fd);
        if (err) *err = "Invalid ELF file: " + elf_path;
        return false;
    }
    void* mapped = ::mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (mapped == MAP_FAILED) {
        if (err) *err = "mmap failed for " + elf_path;
        return false;
    }

    const auto* base = static_cast<const uint8_t*>(mapped);
    const auto* eh = reinterpret_cast<const Elf64_Ehdr*>(base);
    const auto* sh = reinterpret_cast<const Elf64_Shdr*>(base + eh->e_shoff);
    bool found_symtab = false;
    for (int i = 0; i < eh->e_shnum; ++i) {
        if (sh[i].sh_type != SHT_SYMTAB) continue;
        found_symtab = true;
        const auto* syms = reinterpret_cast<const Elf64_Sym*>(base + sh[i].sh_offset);
        size_t count = sh[i].sh_size / sizeof(Elf64_Sym);
        const char* strtab = reinterpret_cast<const char*>(base + sh[sh[i].sh_link].sh_offset);
        for (size_t k = 0; k < count; ++k) {
            if (syms[k].st_name != 0 && syms[k].st_value != 0) {
                (*out_syms)[strtab + syms[k].st_name] = syms[k].st_value;
            }
        }
    }
    ::munmap(mapped, static_cast<size_t>(st.st_size));
    if (!found_symtab) {
        if (err) *err = "No .symtab section in " + elf_path + " (cannot resolve Mesa Vulkan symbols)";
        return false;
    }
    return true;
}

static const MesaVkApi& get_mesa_vk_api(const std::string& panfrost_bin) {
    static MesaVkApi api;
    if (api.loaded && api.loaded_path == panfrost_bin) {
        return api;
    }
    api = MesaVkApi{};
    api.loaded_path = panfrost_bin;

    std::unordered_map<std::string, uint64_t> syms;
    if (!load_elf_symtab(panfrost_bin, &syms, &api.load_error)) {
        return api;
    }

    void* handle = ::dlopen(panfrost_bin.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        const char* dlerr = ::dlerror();
        api.load_error = std::string("dlopen(") + panfrost_bin + ") failed: " +
                         (dlerr ? dlerr : "unknown error");
        return api;
    }
    struct link_map* lm = nullptr;
    if (::dlinfo(handle, RTLD_DI_LINKMAP, &lm) != 0 || !lm) {
        api.load_error = "dlinfo(RTLD_DI_LINKMAP) failed for " + panfrost_bin;
        return api;
    }
    uintptr_t base = static_cast<uintptr_t>(lm->l_addr);

    auto resolve = [&](const char* name, auto& dst) -> bool {
        auto it = syms.find(name);
        if (it == syms.end()) {
            api.load_error = std::string("Missing Mesa symbol in panfrost_compile: ") + name;
            return false;
        }
        using FnType = std::remove_reference_t<decltype(dst)>;
        dst = reinterpret_cast<FnType>(base + it->second);
        return true;
    };

    bool ok =
        resolve("glsl_type_singleton_init_or_ref", api.glsl_init) &&
        resolve("pan_get_nir_shader_compiler_options", api.pan_get_opts) &&
        resolve("spirv_to_nir", api.spirv_to_nir) &&
        resolve("nir_lower_variable_initializers", api.nir_lower_variable_initializers) &&
        resolve("nir_lower_returns", api.nir_lower_returns) &&
        resolve("nir_inline_functions", api.nir_inline_functions) &&
        resolve("nir_remove_non_entrypoints", api.nir_remove_non_entrypoints) &&
        resolve("nir_lower_io_vars_to_temporaries", api.nir_lower_io_vars_to_temporaries) &&
        resolve("nir_lower_global_vars_to_local", api.nir_lower_global_vars_to_local) &&
        resolve("nir_split_var_copies", api.nir_split_var_copies) &&
        resolve("nir_opt_copy_prop_vars", api.nir_opt_copy_prop_vars) &&
        resolve("nir_opt_combine_stores", api.nir_opt_combine_stores) &&
        resolve("nir_lower_var_copies", api.nir_lower_var_copies) &&
        resolve("nir_lower_memcpy", api.nir_lower_memcpy) &&
        resolve("nir_lower_vars_to_ssa", api.nir_lower_vars_to_ssa) &&
        resolve("nir_lower_system_values", api.nir_lower_system_values) &&
        resolve("nir_lower_compute_system_values", api.nir_lower_compute_system_values) &&
        resolve("nir_lower_vars_to_explicit_types", api.nir_lower_vars_to_explicit_types) &&
        resolve("glsl_get_cl_type_size_align", api.glsl_cl_size_align) &&
        resolve("type_size_vec4", api.type_size_vec4) &&
        resolve("nir_lower_explicit_io", api.nir_lower_explicit_io) &&
        resolve("nir_assign_io_var_locations", api.nir_assign_io_var_locations) &&
        resolve("nir_lower_io", api.nir_lower_io) &&
        resolve("pan_preprocess_nir", api.pan_preprocess_nir) &&
        resolve("pan_postprocess_nir", api.pan_postprocess_nir) &&
        resolve("pan_varying_collect_formats", api.pan_varying_collect_formats) &&
        resolve("pan_build_varying_layout_compact", api.pan_build_varying_layout_compact) &&
        resolve("nir_opt_constant_folding", api.nir_opt_constant_folding) &&
        resolve("nir_opt_dce", api.nir_opt_dce) &&
        resolve("pan_shader_compile", api.pan_shader_compile) &&
        resolve("nir_load_const_instr_create", api.nir_load_const_instr_create) &&
        resolve("nir_instr_insert", api.nir_instr_insert) &&
        resolve("nir_def_rewrite_uses", api.nir_def_rewrite_uses) &&
        resolve("nir_instr_remove_v", api.nir_instr_remove_v) &&
        resolve("nir_tex_instr_remove_src", api.nir_tex_instr_remove_src);

    if (ok) {
        api.glsl_init();
        api.loaded = true;
    }
    return api;
}

struct SpvMeta {
    int mesa_stage = -1;
    valhall::ShaderStage stage = valhall::ShaderStage::Compute;
    std::string entry_point = "main";
    uint32_t local_size[3] = {1, 1, 1};
    std::map<uint32_t, std::string> srt_pka_buf_names;
};

static bool parse_spv_metadata(const std::vector<uint32_t>& words, SpvMeta* out, std::string* err) {
    if (words.size() < 5 || words[0] != 0x07230203u) {
        if (err) *err = "Invalid SPIR-V magic header";
        return false;
    }
    std::unordered_map<uint32_t, uint32_t> desc_set;
    std::unordered_map<uint32_t, uint32_t> binding;
    std::unordered_map<uint32_t, uint32_t> var_storage_class;
    std::unordered_map<uint32_t, bool> type_is_buffer_block;
    std::unordered_map<uint32_t, uint32_t> ptr_pointee;

    size_t i = 5;
    while (i < words.size()) {
        uint16_t wc = static_cast<uint16_t>(words[i] >> 16);
        uint16_t op = static_cast<uint16_t>(words[i] & 0xFFFFu);
        if (wc == 0 || i + wc > words.size()) break;
        if (op == 15 && wc >= 4 && out->mesa_stage < 0) {  // OpEntryPoint
            uint32_t em = words[i + 1];
            const char* nm = reinterpret_cast<const char*>(&words[i + 3]);
            out->entry_point = nm;
            if (em == 0) {
                out->mesa_stage = 0;
                out->stage = valhall::ShaderStage::Vertex;
            } else if (em == 4) {
                out->mesa_stage = 4;
                out->stage = valhall::ShaderStage::Fragment;
            } else if (em == 5) {
                out->mesa_stage = 5;
                out->stage = valhall::ShaderStage::Compute;
            }
        } else if (op == 16 && wc >= 6 && words[i + 2] == 17) {  // OpExecutionMode LocalSize
            out->local_size[0] = words[i + 3];
            out->local_size[1] = words[i + 4];
            out->local_size[2] = words[i + 5];
        } else if (op == 71 && wc >= 3) {  // OpDecorate
            uint32_t target = words[i + 1];
            uint32_t dec = words[i + 2];
            if (dec == 34 && wc >= 4) desc_set[target] = words[i + 3];  // DescriptorSet
            if (dec == 33 && wc >= 4) binding[target] = words[i + 3];   // Binding
            if (dec == 3) type_is_buffer_block[target] = true;          // BufferBlock
        } else if (op == 32 && wc >= 4) {  // OpTypePointer
            ptr_pointee[words[i + 1]] = words[i + 3];
        } else if (op == 59 && wc >= 4) {  // OpVariable
            uint32_t result_type = words[i + 1];
            uint32_t id = words[i + 2];
            uint32_t sc = words[i + 3];
            if (sc == 2 || sc == 12) {  // Uniform or StorageBuffer
                var_storage_class[id] = sc;
                if (sc == 2) {
                    auto pit = ptr_pointee.find(result_type);
                    if (pit != ptr_pointee.end() && type_is_buffer_block.count(pit->second)) {
                        var_storage_class[id] = 12;  // Legacy BufferBlock SSBO
                    }
                }
            }
        }
        i += wc;
    }

    if (out->mesa_stage < 0) {
        if (err) *err = "Unsupported or missing Vulkan SPIR-V entry point (only Vertex/Fragment/Compute supported)";
        return false;
    }

    // Build Valhall packed descriptor ID -> canonical buffer name mapping.
    // Matches Panfrost's Valhall LD_PKA/LEA_PKA encoding: ((4 + desc_set) << 24) | binding.
    for (const auto& [var_id, sc] : var_storage_class) {
        auto bit = binding.find(var_id);
        if (bit == binding.end()) continue;
        uint32_t ds = desc_set.count(var_id) ? desc_set[var_id] : 0u;
        uint32_t bd = bit->second;
        uint32_t packed = ((4u + ds) << 24) | (bd & 0xFFFFu);
        std::string prefix = (sc == 12) ? "ssbo_s" : "ubo_s";
        out->srt_pka_buf_names[packed] = prefix + std::to_string(ds) + "_b" + std::to_string(bd);
    }
    return true;
}

static void* get_entrypoint_impl(void* shader) {
    auto* fns = reinterpret_cast<ExecListAbi*>(static_cast<uint8_t*>(shader) + 472);
    ExecNodeAbi* fn = fns->head_sentinel.next;
    while (fn && fn->next) {
        void* impl = *reinterpret_cast<void**>(reinterpret_cast<uint8_t*>(fn) + 48);
        if (impl) return impl;
        fn = fn->next;
    }
    return nullptr;
}

/// Lower Vulkan resource-index/descriptor intrinsics, clear push-constant range/base,
/// and strip Vulkan texture/sampler deref sources prior to Panfrost post-processing.
static void walk_pre_cf_list(const MesaVkApi& api, void* shader, ExecListAbi* list) {
    ExecNodeAbi* node = list->head_sentinel.next;
    while (node && node->next) {
        auto* cf = reinterpret_cast<uint8_t*>(node);
        uint32_t cf_type = *reinterpret_cast<uint32_t*>(cf + 16);
        if (cf_type == 0) {  // nir_cf_node_block
            auto* ilist = reinterpret_cast<ExecListAbi*>(cf + 32);
            ExecNodeAbi* in = ilist->head_sentinel.next;
            while (in && in->next) {
                ExecNodeAbi* next_in = in->next;
                auto* ins = reinterpret_cast<uint8_t*>(in);
                uint8_t itype = *(ins + 24);
                if (itype == kNirInstrTypeIntrinsic) {
                    uint32_t intrin = *reinterpret_cast<uint32_t*>(ins + 56);
                    uint8_t* def_ptr = ins + 32;
                    uint8_t nc = *(ins + 60);
                    auto* cidx = reinterpret_cast<int32_t*>(ins + 64);
                    if (intrin == kNirIntrinVulkanResourceIndex) {
                        uint32_t dset = static_cast<uint32_t>(cidx[0]);
                        uint32_t bind = static_cast<uint32_t>(cidx[1]);
                        uint32_t packed = ((4u + dset) << 24) | (bind & 0xFFFFu);
                        unsigned comps = nc ? nc : 2u;
                        auto* lc = static_cast<uint8_t*>(api.nir_load_const_instr_create(shader, comps, 32));
                        auto* vals = reinterpret_cast<uint64_t*>(lc + 56);
                        vals[0] = packed;
                        if (comps >= 2) vals[1] = 0;
                        NirCursorAbi cur{2, 0, ins};
                        api.nir_instr_insert(cur, lc);
                        api.nir_def_rewrite_uses(def_ptr, lc + 32);
                        api.nir_instr_remove_v(ins);
                    } else if (intrin == kNirIntrinLoadVulkanDescriptor) {
                        void* src0_ssa = *reinterpret_cast<void**>(ins + 112 + 24);
                        api.nir_def_rewrite_uses(def_ptr, src0_ssa);
                        api.nir_instr_remove_v(ins);
                    } else if (intrin == kNirIntrinLoadPushConstant) {
                        // bi_emit_load_push_constant requires base == 0 and range == 0.
                        cidx[0] = 0;
                        cidx[2] = 0;
                    }
                } else if (itype == kNirInstrTypeTex) {
                    auto* srcs = *reinterpret_cast<uint8_t**>(ins + 72);
                    uint32_t nsrcs = *reinterpret_cast<uint32_t*>(ins + 80);
                    for (int si = static_cast<int>(nsrcs) - 1; si >= 0; --si) {
                        uint32_t stype = *reinterpret_cast<uint32_t*>(srcs + si * 40 + 32);
                        if (stype == kNirTexSrcTextureDeref || stype == kNirTexSrcSamplerDeref) {
                            api.nir_tex_instr_remove_src(ins, static_cast<unsigned>(si));
                            srcs = *reinterpret_cast<uint8_t**>(ins + 72);
                        }
                    }
                }
                in = next_in;
            }
        } else if (cf_type == 1) {  // nir_cf_node_if
            walk_pre_cf_list(api, shader, reinterpret_cast<ExecListAbi*>(cf + 72));
            walk_pre_cf_list(api, shader, reinterpret_cast<ExecListAbi*>(cf + 104));
        } else if (cf_type == 2) {  // nir_cf_node_loop
            walk_pre_cf_list(api, shader, reinterpret_cast<ExecListAbi*>(cf + 32));
            walk_pre_cf_list(api, shader, reinterpret_cast<ExecListAbi*>(cf + 64));
        }
        node = node->next;
    }
}

/// Lower post-process vertex viewport sysvals (`load_viewport_scale`, `load_viewport_offset`)
/// that panvk normally lowers via `panvk_lower_sysvals`.
static void walk_post_cf_list(const MesaVkApi& api, void* shader, ExecListAbi* list) {
    ExecNodeAbi* node = list->head_sentinel.next;
    while (node && node->next) {
        auto* cf = reinterpret_cast<uint8_t*>(node);
        uint32_t cf_type = *reinterpret_cast<uint32_t*>(cf + 16);
        if (cf_type == 0) {
            auto* ilist = reinterpret_cast<ExecListAbi*>(cf + 32);
            ExecNodeAbi* in = ilist->head_sentinel.next;
            while (in && in->next) {
                ExecNodeAbi* next_in = in->next;
                auto* ins = reinterpret_cast<uint8_t*>(in);
                if (*(ins + 24) == kNirInstrTypeIntrinsic) {
                    uint32_t intrin = *reinterpret_cast<uint32_t*>(ins + 56);
                    if (intrin == kNirIntrinLoadViewportScale || intrin == kNirIntrinLoadViewportOffset) {
                        auto* lc = static_cast<uint8_t*>(api.nir_load_const_instr_create(shader, 3, 32));
                        auto* vals = reinterpret_cast<uint64_t*>(lc + 56);
                        uint32_t bits = (intrin == kNirIntrinLoadViewportScale) ? 0x3f800000u : 0x00000000u;
                        vals[0] = bits;
                        vals[1] = bits;
                        vals[2] = bits;
                        NirCursorAbi cur{2, 0, ins};
                        api.nir_instr_insert(cur, lc);
                        api.nir_def_rewrite_uses(ins + 32, lc + 32);
                        api.nir_instr_remove_v(ins);
                    }
                }
                in = next_in;
            }
        } else if (cf_type == 1) {
            walk_post_cf_list(api, shader, reinterpret_cast<ExecListAbi*>(cf + 72));
            walk_post_cf_list(api, shader, reinterpret_cast<ExecListAbi*>(cf + 104));
        } else if (cf_type == 2) {
            walk_post_cf_list(api, shader, reinterpret_cast<ExecListAbi*>(cf + 32));
            walk_post_cf_list(api, shader, reinterpret_cast<ExecListAbi*>(cf + 64));
        }
        node = node->next;
    }
}

static void run_cf_walk(const MesaVkApi& api, void* shader, bool post) {
    auto* fns = reinterpret_cast<ExecListAbi*>(static_cast<uint8_t*>(shader) + 472);
    ExecNodeAbi* fn = fns->head_sentinel.next;
    while (fn && fn->next) {
        void* impl = *reinterpret_cast<void**>(reinterpret_cast<uint8_t*>(fn) + 48);
        if (impl) {
            auto* body = reinterpret_cast<ExecListAbi*>(static_cast<uint8_t*>(impl) + 48);
            if (post) {
                walk_post_cf_list(api, shader, body);
            } else {
                walk_pre_cf_list(api, shader, body);
            }
        }
        fn = fn->next;
    }
}

struct ChildResultHeader {
    uint32_t ok;
    uint32_t work_reg_count;
    uint32_t tls_size;
    uint32_t wls_size;
    uint32_t binary_size;
    uint32_t fau_count;
    uint32_t fau_words[kPanFauWordsCount];
    uint32_t fau_is_const[4];
};

static bool write_all(int fd, const void* buf, size_t len) {
    const auto* p = static_cast<const uint8_t*>(buf);
    while (len > 0) {
        ssize_t n = ::write(fd, p, len);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        p += n;
        len -= static_cast<size_t>(n);
    }
    return true;
}

static bool read_all(int fd, void* buf, size_t len) {
    auto* p = static_cast<uint8_t*>(buf);
    while (len > 0) {
        ssize_t n = ::read(fd, p, len);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;
        p += n;
        len -= static_cast<size_t>(n);
    }
    return true;
}

}  // namespace

PanfrostVkCompileOutput compile_vulkan_spv_via_panfrost(
    const std::string& spv_path,
    uint8_t arch,
    const std::string& panfrost_bin,
    bool verbose) {
    PanfrostVkCompileOutput out;
    if (arch < 6 || arch > 14) arch = 10;
    out.arch = arch;

    std::ifstream ifs(spv_path, std::ios::binary);
    if (!ifs) {
        out.error = "Failed to open SPIR-V file: " + spv_path;
        return out;
    }
    std::vector<uint8_t> raw_bytes((std::istreambuf_iterator<char>(ifs)), {});
    if (raw_bytes.size() < 20 || (raw_bytes.size() % 4) != 0) {
        out.error = "Invalid SPIR-V binary size: " + spv_path;
        return out;
    }
    std::vector<uint32_t> spv_words(raw_bytes.size() / 4);
    std::memcpy(spv_words.data(), raw_bytes.data(), raw_bytes.size());

    SpvMeta meta;
    if (!parse_spv_metadata(spv_words, &meta, &out.error)) {
        return out;
    }
    out.entry_point = meta.entry_point;
    out.stage = meta.stage;
    out.local_size[0] = meta.local_size[0];
    out.local_size[1] = meta.local_size[1];
    out.local_size[2] = meta.local_size[2];
    out.srt_pka_buf_names = std::move(meta.srt_pka_buf_names);

    const MesaVkApi& api = get_mesa_vk_api(panfrost_bin);
    if (!api.loaded) {
        out.error = api.load_error;
        return out;
    }

    int data_pipe[2] = {-1, -1};
    int err_pipe[2] = {-1, -1};
    if (::pipe(data_pipe) != 0 || ::pipe(err_pipe) != 0) {
        out.error = "Failed to create pipes for Panfrost Vulkan compiler worker";
        if (data_pipe[0] >= 0) { ::close(data_pipe[0]); ::close(data_pipe[1]); }
        if (err_pipe[0] >= 0) { ::close(err_pipe[0]); ::close(err_pipe[1]); }
        return out;
    }

    if (verbose) {
        std::cerr << "[panfrost-vk] compiling " << spv_path
                  << " stage=" << meta.mesa_stage
                  << " entry=" << meta.entry_point
                  << " arch=v" << static_cast<int>(arch) << "\n";
    }

    pid_t pid = ::fork();
    if (pid < 0) {
        ::close(data_pipe[0]); ::close(data_pipe[1]);
        ::close(err_pipe[0]); ::close(err_pipe[1]);
        out.error = "fork() failed for Panfrost Vulkan compiler worker";
        return out;
    }

    if (pid == 0) {
        ::close(data_pipe[0]);
        ::close(err_pipe[0]);
        ::dup2(err_pipe[1], STDERR_FILENO);
        ::close(err_pipe[1]);

        uint8_t all_caps[512];
        std::memset(all_caps, 1, sizeof(all_caps));

        SpirvToNirOptionsAbi sopts{};
        sopts.environment = 0;  // NIR_SPIRV_VULKAN
        sopts.capabilities = all_caps;
        sopts.ubo_addr_format = kNirAddr32BitIndexOffset;
        sopts.ssbo_addr_format = kNirAddr32BitIndexOffset;
        sopts.phys_ssbo_addr_format = kNirAddr64BitGlobal;
        sopts.push_const_addr_format = kNirAddr32BitOffset;
        sopts.shared_addr_format = kNirAddr32BitOffset;
        sopts.min_ubo_alignment = 16;
        sopts.min_ssbo_alignment = 16;

        const void* nopts = api.pan_get_opts(arch, meta.mesa_stage, false);
        void* nir = api.spirv_to_nir(
            spv_words.data(), spv_words.size(), nullptr, meta.mesa_stage,
            meta.entry_point.c_str(), &sopts, nopts);
        if (!nir) {
            std::fprintf(stderr, "spirv_to_nir returned null\n");
            ::_exit(2);
        }

        api.nir_lower_variable_initializers(nir, kNirVarAll);
        api.nir_lower_returns(nir);
        api.nir_inline_functions(nir);
        api.nir_remove_non_entrypoints(nir);

        if (meta.mesa_stage == 0 || meta.mesa_stage == 4) {
            void* ep = get_entrypoint_impl(nir);
            if (ep) {
                api.nir_lower_io_vars_to_temporaries(nir, ep, kNirVarShaderOut);
            }
        }

        api.nir_lower_global_vars_to_local(nir);
        api.nir_split_var_copies(nir);
        api.nir_opt_copy_prop_vars(nir);
        api.nir_opt_combine_stores(nir, kNirVarAll);
        api.nir_lower_var_copies(nir);
        api.nir_lower_memcpy(nir);
        api.nir_lower_vars_to_ssa(nir);
        api.nir_lower_system_values(nir);
        if (meta.mesa_stage == 5) {
            uint8_t cs_opts[64] = {};
            api.nir_lower_compute_system_values(nir, cs_opts);
        }

        uint64_t gpu_id = static_cast<uint64_t>(arch & 0xfu) << 28;
        api.pan_preprocess_nir(nir, gpu_id);

        if (meta.mesa_stage == 5) {
            api.nir_lower_vars_to_explicit_types(nir, kNirVarMemShared, api.glsl_cl_size_align);
            api.nir_lower_explicit_io(nir, kNirVarMemShared, kNirAddr32BitOffset);
        }
        api.nir_lower_explicit_io(nir, kNirVarMemUbo | kNirVarMemSsbo, kNirAddr32BitIndexOffset);
        api.nir_lower_explicit_io(nir, kNirVarMemPushConst, kNirAddr32BitOffset);
        if (meta.mesa_stage == 0 || meta.mesa_stage == 4) {
            unsigned num_in = 0, num_out = 0;
            api.nir_assign_io_var_locations(nir, kNirVarShaderIn, &num_in, meta.mesa_stage);
            api.nir_assign_io_var_locations(nir, kNirVarShaderOut, &num_out, meta.mesa_stage);
            api.nir_lower_io(nir, kNirVarShaderIn | kNirVarShaderOut, api.type_size_vec4,
                             (meta.mesa_stage == 4) ? 1u : 0u);
        }

        run_cf_walk(api, nir, false);
        api.nir_opt_constant_folding(nir);
        api.nir_opt_dce(nir);

        PanCompileInputsAbi inputs{};
        inputs.gpu_id = gpu_id;
        inputs.no_idvs = true;
        inputs.fau_reserved = 64;
        inputs.fau_promote_immediates = true;

        uint8_t vlayout_buf[4096] = {};
        if (meta.mesa_stage == 0) {
            api.pan_varying_collect_formats(vlayout_buf, nir, gpu_id);
            api.pan_build_varying_layout_compact(vlayout_buf, nir, gpu_id);
            inputs.varying_layout = vlayout_buf;
        }

        uint8_t info_buf[2048] = {};
        api.pan_postprocess_nir(nir, &inputs, info_buf);
        run_cf_walk(api, nir, true);

        UtilDynarrayAbi binary{};
        api.pan_shader_compile(nir, &inputs, &binary, info_buf);

        ChildResultHeader hdr{};
        hdr.ok = (binary.data != nullptr && binary.size > 0) ? 1u : 0u;
        std::memcpy(&hdr.work_reg_count, info_buf + 4, 4);
        std::memcpy(&hdr.tls_size, info_buf + 8, 4);
        std::memcpy(&hdr.wls_size, info_buf + 12, 4);
        hdr.binary_size = binary.size;
        std::memcpy(hdr.fau_words, info_buf + kPanShaderInfoFauOffset, sizeof(hdr.fau_words));
        std::memcpy(hdr.fau_is_const, info_buf + kPanShaderInfoFauOffset + 512, sizeof(hdr.fau_is_const));
        std::memcpy(&hdr.fau_count, info_buf + kPanShaderInfoFauOffset + 532, 4);

        write_all(data_pipe[1], &hdr, sizeof(hdr));
        if (hdr.ok && hdr.binary_size > 0) {
            write_all(data_pipe[1], binary.data, hdr.binary_size);
        }
        ::close(data_pipe[1]);
        ::_exit(0);
    }

    ::close(data_pipe[1]);
    ::close(err_pipe[1]);

    ChildResultHeader hdr{};
    bool got_hdr = read_all(data_pipe[0], &hdr, sizeof(hdr));
    std::vector<uint8_t> bin_bytes;
    if (got_hdr && hdr.ok && hdr.binary_size > 0 && hdr.binary_size <= (64u * 1024u * 1024u)) {
        bin_bytes.resize(hdr.binary_size);
        if (!read_all(data_pipe[0], bin_bytes.data(), hdr.binary_size)) {
            got_hdr = false;
        }
    }
    ::close(data_pipe[0]);

    std::string err_text;
    char buf[512];
    ssize_t n = 0;
    while ((n = ::read(err_pipe[0], buf, sizeof(buf))) > 0) {
        err_text.append(buf, static_cast<size_t>(n));
    }
    ::close(err_pipe[0]);

    int status = 0;
    ::waitpid(pid, &status, 0);

    if (!got_hdr || !hdr.ok || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        std::string reason;
        if (WIFSIGNALED(status)) {
            reason = "Panfrost Vulkan compiler terminated by signal " + std::to_string(WTERMSIG(status));
        } else if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
            reason = "Panfrost Vulkan compiler exited with code " + std::to_string(WEXITSTATUS(status));
        } else {
            reason = "Panfrost Vulkan compiler produced empty binary";
        }
        if (!err_text.empty()) {
            while (!err_text.empty() && (err_text.back() == '\n' || err_text.back() == '\r')) {
                err_text.pop_back();
            }
            reason += ": " + err_text;
        }
        out.error = std::move(reason);
        return out;
    }

    // Trim trailing all-zero 64-bit Valhall alignment padding words.
    while (bin_bytes.size() >= 8) {
        uint64_t tail = 0;
        std::memcpy(&tail, bin_bytes.data() + bin_bytes.size() - 8, 8);
        if (tail == 0) {
            bin_bytes.resize(bin_bytes.size() - 8);
        } else {
            break;
        }
    }

    out.ok = true;
    out.work_reg_count = hdr.work_reg_count;
    out.tls_size = hdr.tls_size;
    out.wls_size = hdr.wls_size;
    out.ebin_bytes = std::move(bin_bytes);

    uint32_t fcount = (hdr.fau_count <= kPanFauWordsCount) ? hdr.fau_count : kPanFauWordsCount;
    for (uint32_t w = 0; w < fcount; ++w) {
        if ((hdr.fau_is_const[w / 32u] >> (w % 32u)) & 1u) {
            out.fau_const_words[w] = hdr.fau_words[w];
        }
    }
    return out;
}

}  // namespace valhall::spirv2valhall

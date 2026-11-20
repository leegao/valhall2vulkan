#include "malioc_inproc.h"

#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <link.h>
#include <mutex>
#include <sys/mman.h>
#include <unistd.h>

namespace valhall::malioc_inproc {

namespace {

// Reverse-engineered ABI of libMali-Gxx_r55p0-00rel0.so (malioc v8.8.1 Build 73b41e):
struct MaliocFlexibleInputs {
    uint32_t count;
    uint32_t pad;
    const char** pairs;
};

struct MaliocHardwareCore {
    const char* name;
    const char* revision;
};

struct MaliocInputs {
    const char* source;
    MaliocFlexibleInputs flex;
    MaliocHardwareCore core;
    uint32_t flags;
};

struct MaliocOutputs {
    uint8_t opaque[512];
};

using pfn_malioc_compile_t = int (*)(const MaliocInputs*, MaliocOutputs*);
using pfn_malioc_release_outputs_t = void (*)(MaliocOutputs*);
using pfn_compile_multi_t = int (*)(
    void* a0, int shader_count, void* a2, void* a3, void* a4, void* a5,
    void* s0, void* s1, void* s2, void* s3, void*** out_compiled_program_ptr_ptr,
    void* s5, void* s6, void* s7);
using pfn_deserialize_to_c_t = uint64_t (*)(void* ctx, void* stream_obj_72b, char** out_c_str);

// Internal helpers called by cmpbe_v2_deserialize_MBS2_to_C (+0x1cada0):
using pfn_alloc_init_t = void (*)(void* alloc_cb_32b, void* malloc_fn, void* free_fn);
using pfn_arena_init_t = int (*)(void* arena_32b, uint64_t zero, void* alloc_cb_32b);
using pfn_arena_destroy_t = void (*)(void* arena_32b);
using pfn_reader_init_t = void (*)(void* reader_64b, void* alloc_fn, uint64_t zero1, uint64_t zero2,
                                   const uint8_t* data, uint32_t size);
using pfn_read_mbs2_t = int (*)(void* reader_64b, const cmpbe_chunk_MBS2** out_mbs2);

struct MaliLibRuntime {
    void* handle = nullptr;
    uintptr_t base = 0;
    pfn_malioc_compile_t malioc_compile = nullptr;
    pfn_malioc_release_outputs_t malioc_release_outputs = nullptr;
    pfn_compile_multi_t real_compile_multi = nullptr;
    pfn_deserialize_to_c_t real_deserialize_to_c = nullptr;
    pfn_alloc_init_t alloc_init = nullptr;
    pfn_arena_init_t arena_init = nullptr;
    pfn_arena_destroy_t arena_destroy = nullptr;
    pfn_reader_init_t reader_init = nullptr;
    pfn_read_mbs2_t read_mbs2 = nullptr;
    void* arena_alloc_fn = nullptr;
    bool ready = false;
    std::string loaded_path;
};

struct ActiveCapture {
    const std::function<void(const cmpbe_chunk_MBS2*)>* on_mbs2 = nullptr;
    bool emit_c_header = false;
    std::string* out_c_header = nullptr;
    size_t captured_count = 0;
};

std::mutex g_mali_mutex;
MaliLibRuntime g_rt;
thread_local ActiveCapture* g_capture = nullptr;

int hooked_compile_multi(
    void* a0, int shader_count, void* a2, void* a3, void* a4, void* a5,
    void* s0, void* s1, void* s2, void* s3, void*** out_compiled_program_ptr_ptr,
    void* s5, void* s6, void* s7) {
    int ret = g_rt.real_compile_multi(
        a0, shader_count, a2, a3, a4, a5,
        s0, s1, s2, s3, out_compiled_program_ptr_ptr, s5, s6, s7);

    if (ret == 0 && g_capture && out_compiled_program_ptr_ptr && *out_compiled_program_ptr_ptr &&
        shader_count > 0) {
        uint8_t* out_program = reinterpret_cast<uint8_t*>(*out_compiled_program_ptr_ptr);
        const size_t stride = 72;

        void* ctx_malloc = *reinterpret_cast<void**>(reinterpret_cast<uint8_t*>(a0) + 0x80);
        void* ctx_free = *reinterpret_cast<void**>(reinterpret_cast<uint8_t*>(a0) + 0x88);

        for (int i = 0; i < shader_count; ++i) {
            uint8_t stream_obj[72];
            std::memcpy(stream_obj, out_program + static_cast<size_t>(i) * stride, stride);

            // 1. Direct in-memory MBS2 struct deserialization via cmpbe_chunk_read_MBS2!
            if (g_capture->on_mbs2 && *g_capture->on_mbs2) {
                const uint8_t* mbs2_bytes = *reinterpret_cast<const uint8_t* const*>(stream_obj + 0x10);
                uint32_t mbs2_size = *reinterpret_cast<const uint32_t*>(stream_obj + 0x18);
                if (mbs2_bytes && mbs2_size > 0) {
                    alignas(16) uint8_t alloc_cb[32] = {};
                    alignas(16) uint8_t arena[32] = {};
                    alignas(16) uint8_t reader[64] = {};
                    g_rt.alloc_init(alloc_cb, ctx_malloc, ctx_free);
                    if (g_rt.arena_init(arena, 0, alloc_cb) == 0) {
                        g_rt.reader_init(reader, g_rt.arena_alloc_fn, 0, 0, mbs2_bytes, mbs2_size);
                        *reinterpret_cast<void**>(reader + 0x20) = arena;
                        const cmpbe_chunk_MBS2* live_mbs2 = nullptr;
                        if (g_rt.read_mbs2(reader, &live_mbs2) == 0 && live_mbs2 != nullptr) {
                            (*g_capture->on_mbs2)(live_mbs2);
                            g_capture->captured_count++;
                        }
                        g_rt.arena_destroy(arena);
                    }
                }
            }

            // 2. Optional decompiled MBS2 C header emission via cmpbe_v2_deserialize_MBS2_to_C
            if (g_capture->emit_c_header && g_capture->out_c_header && g_rt.real_deserialize_to_c) {
                char* c_code = nullptr;
                uint64_t st = g_rt.real_deserialize_to_c(a0, stream_obj, &c_code);
                if (st == 0 && c_code) {
                    if (i == 0) {
                        *g_capture->out_c_header += "#include \"cmpbe_chunks.h\"\n";
                    } else {
                        *g_capture->out_c_header += "\n";
                    }
                    *g_capture->out_c_header += c_code;
                }
            }
        }
    }
    return ret;
}

bool ensure_loaded_locked(const std::string& lib_path, std::string* err) {
    if (g_rt.ready) return true;
    int dl_flags = RTLD_NOW | RTLD_LOCAL;
#ifdef RTLD_DEEPBIND
    // Prevent ELF symbol interposition (e.g. host libLLVM.so colliding with libMali-Gxx's bundled LLVM)
    dl_flags |= RTLD_DEEPBIND;
#endif
    void* h = dlopen(lib_path.c_str(), dl_flags);
    if (!h) {
        if (err) *err = std::string("dlopen(") + lib_path + ") failed: " + (dlerror() ? dlerror() : "unknown");
        return false;
    }

    auto* lm = reinterpret_cast<struct link_map*>(h);
    uintptr_t base = static_cast<uintptr_t>(lm->l_addr);

    g_rt.handle = h;
    g_rt.base = base;
    g_rt.malioc_compile = reinterpret_cast<pfn_malioc_compile_t>(dlsym(h, "malioc_compile"));
    g_rt.malioc_release_outputs = reinterpret_cast<pfn_malioc_release_outputs_t>(dlsym(h, "malioc_release_outputs"));
    g_rt.real_compile_multi = reinterpret_cast<pfn_compile_multi_t>(base + 0x001c8400ULL);
    g_rt.real_deserialize_to_c = reinterpret_cast<pfn_deserialize_to_c_t>(base + 0x001cada0ULL);
    g_rt.alloc_init = reinterpret_cast<pfn_alloc_init_t>(base + 0x001802f0ULL);
    g_rt.arena_init = reinterpret_cast<pfn_arena_init_t>(base + 0x0017fe70ULL);
    g_rt.arena_destroy = reinterpret_cast<pfn_arena_destroy_t>(base + 0x0017ff20ULL);
    g_rt.reader_init = reinterpret_cast<pfn_reader_init_t>(base + 0x00333100ULL);
    g_rt.read_mbs2 = reinterpret_cast<pfn_read_mbs2_t>(base + 0x00333160ULL);
    g_rt.arena_alloc_fn = reinterpret_cast<void*>(base + 0x00180060ULL);

    if (!g_rt.malioc_compile || !g_rt.malioc_release_outputs) {
        if (err) *err = "Missing malioc_compile / malioc_release_outputs in " + lib_path;
        return false;
    }

    // Patch cmpbe_v2_compile_multiple_shaders@GOT (+0x01967ec0) in-process
    uintptr_t got_entry = base + 0x01967ec0ULL;
    uintptr_t page = got_entry & ~0xFFFULL;
    if (mprotect(reinterpret_cast<void*>(page), 4096, PROT_READ | PROT_WRITE) != 0) {
        if (err) *err = "mprotect failed while patching libMali-Gxx GOT entry";
        return false;
    }
    *reinterpret_cast<void**>(got_entry) = reinterpret_cast<void*>(&hooked_compile_multi);

    g_rt.loaded_path = lib_path;
    g_rt.ready = true;
    return true;
}

std::string normalize_shader_type(const std::string& stage_or_flag) {
    std::string s = stage_or_flag;
    while (!s.empty() && s[0] == '-') s.erase(s.begin());
    if (s == "comp") return "compute";
    if (s == "vert") return "vertex";
    if (s == "frag") return "fragment";
    if (s == "geom") return "geometry";
    if (s == "tesc") return "tessellation_control";
    if (s == "tese") return "tessellation_evaluation";
    return s;
}

}  // namespace

bool is_inproc_malioc_available(const std::string& mali_lib_path) {
    return access(mali_lib_path.c_str(), R_OK) == 0;
}

bool compile_spirv_in_process(
    const std::vector<uint32_t>& spirv_words,
    const InProcCompileOptions& opts,
    const std::function<void(const cmpbe_chunk_MBS2* mbs2)>& on_mbs2,
    InProcCompileResult* out,
    std::string* err) {
    if (!out) {
        if (err) *err = "Null InProcCompileResult pointer";
        return false;
    }
    *out = InProcCompileResult{};
    if (spirv_words.empty()) {
        if (err) *err = "Empty SPIR-V word buffer";
        return false;
    }

    std::lock_guard<std::mutex> lock(g_mali_mutex);
    if (!ensure_loaded_locked(opts.mali_lib_path, err)) {
        return false;
    }

    std::string shader_type = normalize_shader_type(opts.shader_type);
    std::string byte_len_str = std::to_string(spirv_words.size() * sizeof(uint32_t));
    std::string ep = opts.entry_point.empty() ? "main" : opts.entry_point;

    const char* flex_pairs[] = {
        "compiler_type", "openglessl",
        "shader_type", shader_type.c_str(),
        "spirv", "true",
        "spirv_binary_length", byte_len_str.c_str(),
        "spirv_entrypoint_name", ep.c_str(),
        "request-pilots", "true",
    };

    MaliocInputs in{};
    in.source = reinterpret_cast<const char*>(spirv_words.data());
    in.flex.count = static_cast<uint32_t>(sizeof(flex_pairs) / sizeof(flex_pairs[0]));
    in.flex.pairs = flex_pairs;
    in.core.name = opts.core.c_str();
    in.core.revision = opts.revision.c_str();
    in.flags = 0;

    MaliocOutputs raw_out{};

    ActiveCapture cap;
    cap.on_mbs2 = &on_mbs2;
    cap.emit_c_header = opts.emit_c_header;
    cap.out_c_header = &out->mbs2_c_header;

    if (opts.verbose) {
        std::cerr << "[malioc_inproc] compiling in-process: core=" << opts.core
                  << " stage=" << shader_type << " entry=\"" << ep
                  << "\" bytes=" << byte_len_str << "\n";
    }

    g_capture = &cap;
    int rc = g_rt.malioc_compile(&in, &raw_out);
    g_capture = nullptr;
    g_rt.malioc_release_outputs(&raw_out);

    out->shader_count = cap.captured_count;
    if (rc != 0 || cap.captured_count == 0) {
        out->ok = false;
        out->error_message = "malioc_compile in-process failed (rc=" + std::to_string(rc) +
                             ", captured_mbs2=" + std::to_string(cap.captured_count) + ")";
        if (err) *err = out->error_message;
        return false;
    }

    out->ok = true;
    return true;
}

}  // namespace valhall::malioc_inproc

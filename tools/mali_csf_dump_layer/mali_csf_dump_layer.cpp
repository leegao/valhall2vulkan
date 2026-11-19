#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "valhall_csf_descriptors_generated.h"

#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <elf.h>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <link.h>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ============================================================================
// 1. Minimal Mali kbase CSF UAPI Definitions (r38p1 + r44p1 + r49p1 + r54p3)
// ============================================================================
namespace {

constexpr uint8_t kKbaseIoctlType = 0x80;

struct kbase_ioctl_get_ddk_version {
    uint64_t version_buffer;
    uint32_t size;
    uint32_t padding;
};

struct kbase_ioctl_cs_queue_register {
    uint64_t buffer_gpu_addr;
    uint32_t buffer_size;
    uint8_t priority;
    uint8_t padding[3];
};

struct kbase_ioctl_cs_queue_register_ex {
    uint64_t buffer_gpu_addr;
    uint32_t buffer_size;
    uint8_t priority;
    uint8_t padding[3];
    uint64_t ex_offset_var_addr;
    uint64_t ex_buffer_base;
    uint32_t ex_buffer_size;
    uint8_t ex_event_size;
    uint8_t ex_event_state;
    uint8_t ex_padding[2];
};

struct kbase_ioctl_cs_queue_kick {
    uint64_t buffer_gpu_addr;
};

union kbase_ioctl_cs_queue_bind {
    struct {
        uint64_t buffer_gpu_addr;
        uint8_t group_handle;
        uint8_t csi_index;
        uint8_t padding[6];
    } in;
    struct {
        uint64_t mmap_handle;
    } out;
};

union kbase_ioctl_mem_alloc {
    struct {
        uint64_t va_pages;
        uint64_t commit_pages;
        uint64_t extension;
        uint64_t flags;
    } in;
    struct {
        uint64_t flags;
        uint64_t gpu_va;
    } out;
};

struct kbase_ioctl_mem_free {
    uint64_t gpu_addr;
};

#define MALI_IOCTL_GET_DDK_VERSION      _IOW(kKbaseIoctlType, 13, struct kbase_ioctl_get_ddk_version)
#define MALI_IOCTL_MEM_ALLOC            _IOWR(kKbaseIoctlType, 5, union kbase_ioctl_mem_alloc)
#define MALI_IOCTL_MEM_FREE             _IOW(kKbaseIoctlType, 7, struct kbase_ioctl_mem_free)
#define MALI_IOCTL_CS_QUEUE_REGISTER    _IOW(kKbaseIoctlType, 36, struct kbase_ioctl_cs_queue_register)
#define MALI_IOCTL_CS_QUEUE_KICK        _IOW(kKbaseIoctlType, 37, struct kbase_ioctl_cs_queue_kick)
#define MALI_IOCTL_CS_QUEUE_BIND        _IOWR(kKbaseIoctlType, 39, union kbase_ioctl_cs_queue_bind)
#define MALI_IOCTL_CS_QUEUE_REGISTER_EX _IOW(kKbaseIoctlType, 40, struct kbase_ioctl_cs_queue_register_ex)

constexpr size_t kCsUserIoInsertOffset = 0x1000;

// ============================================================================
// 2. Android System Property + Environment Configuration
// ============================================================================
using PropGetFn = int (*)(const char*, char*);

std::string get_config_str(const char* prop_name, const char* env_name, const char* def_val) {
    static PropGetFn sys_prop_get = reinterpret_cast<PropGetFn>(dlsym(RTLD_DEFAULT, "__system_property_get"));
    if (sys_prop_get && prop_name) {
        char buf[256] = {0};
        if (sys_prop_get(prop_name, buf) > 0 && buf[0] != '\0') {
            return std::string(buf);
        }
    }
    if (env_name) {
        if (const char* e = std::getenv(env_name)) {
            if (e[0] != '\0') return std::string(e);
        }
    }
    return def_val ? std::string(def_val) : std::string();
}

bool is_dump_enabled() {
    std::string v = get_config_str("debug.mali.csf_dump.enable", "MALI_CSF_DUMP_ENABLE", "1");
    return v == "1" || v == "true" || v == "TRUE" || v == "yes";
}

uint32_t get_target_frame() {
    std::string v = get_config_str("debug.mali.csf_dump.frame", "MALI_CSF_DUMP_FRAME", "1");
    char* endp = nullptr;
    unsigned long f = std::strtoul(v.c_str(), &endp, 10);
    return (endp && endp != v.c_str()) ? static_cast<uint32_t>(f) : 1u;
}

std::string get_dump_dir() {
#ifdef __ANDROID__
    const char* def_dir = "/data/local/tmp/mali_csf_dump";
#else
    const char* def_dir = "/tmp/mali_csf_dump";
#endif
    return get_config_str("debug.mali.csf_dump.dir", "MALI_CSF_DUMP_DIR", def_dir);
}

// ============================================================================
// 3. Global State: GPU/CPU VA Map, CSF Queues, Frame & Dump State
// ============================================================================
struct GpuMapping {
    uint64_t gpu_va = 0;
    uintptr_t cpu_va = 0;
    size_t size = 0;
};

struct CsfQueueState {
    uint64_t buffer_gpu_addr = 0;
    uint32_t buffer_size = 0;
    uint64_t mmap_handle = 0;
    uintptr_t user_io_cpu = 0;
    uint64_t last_extract = 0;
    uint8_t csi_index = 0;
};

struct CapturedIdvs {
    bool valid = false;
    bool is_idvs2 = false;
    uint64_t run_idvs_insn = 0;
    uint32_t idvs_sr[64] = {0};
    uint32_t frag_sr[64] = {0};
    bool has_frag_sr = false;
    uint64_t rt_color_va = 0;
    uint32_t fb_width = 0;
    uint32_t fb_height = 0;
    std::string out_dir;
};

struct LayerState {
    std::mutex mu;
    std::unordered_set<int> mali_fds;
    std::vector<GpuMapping> mappings;
    std::unordered_map<uint64_t, CsfQueueState> queues_by_gpu_va;
    std::unordered_map<uint64_t, uint64_t> queue_by_mmap_handle;
    std::string detected_abi = "unknown";
    uint32_t regs[96] = {0};
    std::atomic<uint32_t> current_frame{1};
    std::atomic<bool> dump_done{false};
    CapturedIdvs captured{};
    bool hooks_installed = false;

    // Original libc functions
    int (*orig_open)(const char*, int, ...) = nullptr;
    int (*orig_open64)(const char*, int, ...) = nullptr;
    int (*orig_open_2)(const char*, int) = nullptr;
    int (*orig_openat)(int, const char*, int, ...) = nullptr;
    int (*orig_openat64)(int, const char*, int, ...) = nullptr;
    int (*orig_close)(int) = nullptr;
    int (*orig_ioctl)(int, unsigned long, ...) = nullptr;
    void* (*orig_mmap)(void*, size_t, int, int, int, off_t) = nullptr;
    void* (*orig_mmap64)(void*, size_t, int, int, int, off64_t) = nullptr;
    int (*orig_munmap)(void*, size_t) = nullptr;
};

LayerState& state() {
    static LayerState s;
    return s;
}

const void* gpu_to_cpu_locked(const LayerState& s, uint64_t gpu_va, size_t min_bytes = 4) {
    uint64_t va48 = gpu_va & 0x0000FFFFFFFFFFFFULL;
    if (va48 == 0) return nullptr;
    for (const auto& m : s.mappings) {
        uint64_t m48 = m.gpu_va & 0x0000FFFFFFFFFFFFULL;
        if (va48 >= m48 && (va48 - m48) + min_bytes <= m.size) {
            return reinterpret_cast<const void*>(m.cpu_va + (va48 - m48));
        }
    }
    return nullptr;
}

void write_bin_file(const std::string& path, const void* data, size_t size) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (f && data && size > 0) {
        f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    }
}

void ensure_dir(const std::string& dir) {
    std::string cur;
    for (size_t i = 0; i < dir.size(); ++i) {
        cur.push_back(dir[i]);
        if (dir[i] == '/' || i + 1 == dir.size()) {
            if (!cur.empty() && cur != "/") {
                ::mkdir(cur.c_str(), 0755);
            }
        }
    }
}

// ============================================================================
// 4. vkcube_frame1 Format Dumper (v10/v11 IDVS + v12/v13/v14 IDVS2)
// ============================================================================
void dump_sr_u64(const std::string& path, const uint32_t* sr64) {
    uint64_t words[64] = {0};
    for (size_t i = 0; i < 64; ++i) {
        words[i] = static_cast<uint64_t>(sr64[i]);
    }
    write_bin_file(path, words, sizeof(words));
}

void dump_srt_and_buffers_locked(LayerState& s, const std::string& out_dir, uint64_t raw_pos_srt, uint64_t raw_frag_srt) {
    uint64_t pos_srt_va = raw_pos_srt & 0x0000FFFFFFFFFFFFULL & ~0x3FULL;
    uint64_t frag_srt_va = raw_frag_srt & 0x0000FFFFFFFFFFFFULL & ~0x3FULL;
    uint32_t pos_srt_count = static_cast<uint32_t>(raw_pos_srt & 0x3FULL);
    if (pos_srt_count == 0 && pos_srt_va != 0) pos_srt_count = 4u;

    if (const void* p = gpu_to_cpu_locked(s, pos_srt_va, 512)) {
        write_bin_file(out_dir + "/pos_srt.bin", p, 512);
        const uint32_t* srt_words = reinterpret_cast<const uint32_t*>(p);
        uint32_t buf_idx = 0;
        for (uint32_t r = 0; r < pos_srt_count; ++r) {
            valhall::csf::Resource res = valhall::csf::Resource::unpack(srt_words + r * 4u);
            uint64_t tbl_va = res.address & 0x0000FFFFFFFFFFFFULL;
            if (tbl_va >= 0x10000ULL && res.size > 0) {
                if (const void* tp = gpu_to_cpu_locked(s, tbl_va, static_cast<size_t>(res.size))) {
                    char name[64];
                    std::snprintf(name, sizeof(name), "/srt_buf%u.bin", buf_idx++);
                    write_bin_file(out_dir + name, tp, static_cast<size_t>(res.size));
                }
            }
        }
    }
    if (const void* p = gpu_to_cpu_locked(s, frag_srt_va, 512)) {
        write_bin_file(out_dir + "/frag_srt.bin", p, 512);
    }
}

void dump_spd_and_shader_locked(LayerState& s, const std::string& out_dir, uint64_t spd_va,
                                const char* spd_name, const char* bin_name) {
    if (const void* p = gpu_to_cpu_locked(s, spd_va, 32)) {
        write_bin_file(out_dir + "/" + spd_name, p, 32);
        valhall::csf::ShaderProgram spd = valhall::csf::ShaderProgram::unpack(reinterpret_cast<const uint32_t*>(p));
        uint64_t code_va = spd.binary & 0x0000FFFFFFFFFFFFULL;
        if (const void* cp = gpu_to_cpu_locked(s, code_va, 512)) {
            write_bin_file(out_dir + "/" + bin_name, cp, 512);
        }
    }
}

void dump_idvs_capture_locked(LayerState& s, uint64_t run_idvs_insn, const uint32_t* idvs_sr) {
    std::string out_dir = get_dump_dir();
    ensure_dir(out_dir);

    s.captured.valid = true;
    s.captured.is_idvs2 = false;
    s.captured.run_idvs_insn = run_idvs_insn;
    std::memcpy(s.captured.idvs_sr, idvs_sr, sizeof(s.captured.idvs_sr));
    s.captured.out_dir = out_dir;

    dump_sr_u64(out_dir + "/idvs_sr.bin", idvs_sr);
    write_bin_file(out_dir + "/csrunidvs_insn.bin", &run_idvs_insn, sizeof(run_idvs_insn));

    auto rd64 = [&](valhall::csf::IDVSSR r) -> uint64_t {
        uint32_t idx = static_cast<uint32_t>(r) & 62u;
        return static_cast<uint64_t>(idvs_sr[idx]) | (static_cast<uint64_t>(idvs_sr[idx + 1u]) << 32);
    };

    dump_srt_and_buffers_locked(s, out_dir, rd64(valhall::csf::IDVSSR::SRT_0), rd64(valhall::csf::IDVSSR::SRT_2));

    uint64_t raw_fau = rd64(valhall::csf::IDVSSR::FAU_0);
    uint64_t fau_va = raw_fau & 0x0000FFFFFFFFFFFFULL;
    size_t fau_bytes = std::max<size_t>(8u, static_cast<size_t>((raw_fau >> 56) & 0xFFu) * 8u);
    if (const void* p = gpu_to_cpu_locked(s, fau_va, fau_bytes)) {
        write_bin_file(out_dir + "/fau.bin", p, fau_bytes);
    }

    dump_spd_and_shader_locked(s, out_dir, rd64(valhall::csf::IDVSSR::SPD_0) & 0x0000FFFFFFFFFFFFULL, "pos_spd.bin", "pos_shader.bin");
    dump_spd_and_shader_locked(s, out_dir, rd64(valhall::csf::IDVSSR::SPD_1) & 0x0000FFFFFFFFFFFFULL, "var_spd.bin", "var_shader.bin");
    dump_spd_and_shader_locked(s, out_dir, rd64(valhall::csf::IDVSSR::SPD_2) & 0x0000FFFFFFFFFFFFULL, "frag_spd.bin", "frag_shader.bin");

    uint64_t tsd_va = rd64(valhall::csf::IDVSSR::TSD_0) & 0x0000FFFFFFFFFFFFULL;
    uint64_t tcd_va = rd64(valhall::csf::IDVSSR::TILER_CTX) & 0x0000FFFFFFFFFFFFULL;
    uint64_t zsd_va = rd64(valhall::csf::IDVSSR::ZSD) & 0x0000FFFFFFFFFFFFULL;
    uint64_t bd_va = rd64(valhall::csf::IDVSSR::BLEND_DESC) & 0x0000FFFFFFFFFFFFULL & ~0xFULL;

    if (const void* p = gpu_to_cpu_locked(s, tsd_va, 512)) write_bin_file(out_dir + "/tsd.bin", p, 512);
    if (const void* p = gpu_to_cpu_locked(s, tcd_va, 256)) write_bin_file(out_dir + "/tcd.bin", p, 256);
    if (const void* p = gpu_to_cpu_locked(s, zsd_va, 64))  write_bin_file(out_dir + "/zsd.bin", p, 64);
    if (const void* p = gpu_to_cpu_locked(s, bd_va, 64))   write_bin_file(out_dir + "/bd.bin", p, 64);
}

void dump_idvs2_capture_locked(LayerState& s, uint64_t run_idvs2_insn, const uint32_t* idvs_sr) {
    std::string out_dir = get_dump_dir();
    ensure_dir(out_dir);

    s.captured.valid = true;
    s.captured.is_idvs2 = true;
    s.captured.run_idvs_insn = run_idvs2_insn;
    std::memcpy(s.captured.idvs_sr, idvs_sr, sizeof(s.captured.idvs_sr));
    s.captured.out_dir = out_dir;

    dump_sr_u64(out_dir + "/idvs_sr.bin", idvs_sr);
    write_bin_file(out_dir + "/csrunidvs_insn.bin", &run_idvs2_insn, sizeof(run_idvs2_insn));

    auto rd64 = [&](valhall::csf::V13IDVSSR r) -> uint64_t {
        uint32_t idx = static_cast<uint32_t>(r) & 62u;
        return static_cast<uint64_t>(idvs_sr[idx]) | (static_cast<uint64_t>(idvs_sr[idx + 1u]) << 32);
    };

    dump_srt_and_buffers_locked(s, out_dir,
        rd64(valhall::csf::V13IDVSSR::VERTEX_SRT),
        rd64(valhall::csf::V13IDVSSR::FRAGMENT_SRT));

    uint64_t raw_fau = rd64(valhall::csf::V13IDVSSR::VERTEX_FAU);
    uint64_t fau_va = raw_fau & 0x0000FFFFFFFFFFFFULL;
    size_t fau_bytes = std::max<size_t>(8u, static_cast<size_t>((raw_fau >> 56) & 0xFFu) * 8u);
    if (const void* p = gpu_to_cpu_locked(s, fau_va, fau_bytes)) {
        write_bin_file(out_dir + "/fau.bin", p, fau_bytes);
    }
    uint64_t raw_frag_fau = rd64(valhall::csf::V13IDVSSR::FRAGMENT_FAU);
    uint64_t frag_fau_va = raw_frag_fau & 0x0000FFFFFFFFFFFFULL;
    size_t frag_fau_bytes = std::max<size_t>(8u, static_cast<size_t>((raw_frag_fau >> 56) & 0xFFu) * 8u);
    if (const void* p = gpu_to_cpu_locked(s, frag_fau_va, frag_fau_bytes)) {
        write_bin_file(out_dir + "/frag_fau.bin", p, frag_fau_bytes);
    }

    dump_spd_and_shader_locked(s, out_dir, rd64(valhall::csf::V13IDVSSR::VERTEX_SPD) & 0x0000FFFFFFFFFFFFULL, "pos_spd.bin", "pos_shader.bin");
    dump_spd_and_shader_locked(s, out_dir, rd64(valhall::csf::V13IDVSSR::FRAGMENT_SPD) & 0x0000FFFFFFFFFFFFULL, "frag_spd.bin", "frag_shader.bin");

    uint64_t tsd_va = rd64(valhall::csf::V13IDVSSR::VERTEX_TSD) & 0x0000FFFFFFFFFFFFULL;
    uint64_t tcd_va = rd64(valhall::csf::V13IDVSSR::TILER_CTX) & 0x0000FFFFFFFFFFFFULL;
    uint64_t zsd_va = rd64(valhall::csf::V13IDVSSR::ZSD) & 0x0000FFFFFFFFFFFFULL;
    uint64_t bd_va = rd64(valhall::csf::V13IDVSSR::BLEND_DESC) & 0x0000FFFFFFFFFFFFULL & ~0xFULL;

    if (const void* p = gpu_to_cpu_locked(s, tsd_va, 512)) write_bin_file(out_dir + "/tsd.bin", p, 512);
    if (const void* p = gpu_to_cpu_locked(s, tcd_va, 256)) write_bin_file(out_dir + "/tcd.bin", p, 256);
    if (const void* p = gpu_to_cpu_locked(s, zsd_va, 64))  write_bin_file(out_dir + "/zsd.bin", p, 64);
    if (const void* p = gpu_to_cpu_locked(s, bd_va, 64))   write_bin_file(out_dir + "/bd.bin", p, 64);
}

void dump_fragment_capture_locked(LayerState& s, const uint32_t* frag_sr, uint32_t fbd_reg_idx) {
    if (!s.captured.valid) return;
    s.captured.has_frag_sr = true;
    std::memcpy(s.captured.frag_sr, frag_sr, sizeof(s.captured.frag_sr));
    const std::string& out_dir = s.captured.out_dir;

    dump_sr_u64(out_dir + "/frag_sr.bin", frag_sr);

    uint32_t fbd_lo = frag_sr[fbd_reg_idx];
    uint32_t fbd_hi = frag_sr[fbd_reg_idx + 1u];
    uint32_t fbd_words[2] = { fbd_lo, fbd_hi };
    valhall::csf::FramebufferPointer fbp = valhall::csf::FramebufferPointer::unpack(fbd_words);
    uint64_t fbd_va = fbp.pointer & 0x0000FFFFFFFFFFFFULL;

    if (const void* p = gpu_to_cpu_locked(s, fbd_va, 512)) {
        write_bin_file(out_dir + "/fbd.bin", p, 512);
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(p);
        valhall::csf::FramebufferParameters fbd =
            valhall::csf::FramebufferParameters::unpack(reinterpret_cast<const uint32_t*>(ptr));
        s.captured.fb_width = fbd.width;
        s.captured.fb_height = fbd.height;
        ptr += 128 + (fbp.zs_crc_extension_present ? 64 : 0);
        valhall::csf::RenderTarget rtd =
            valhall::csf::RenderTarget::unpack(reinterpret_cast<const uint32_t*>(ptr));
        s.captured.rt_color_va = rtd.rgb.writeback_buffer.base & 0x0000FFFFFFFFFFFFULL;
    }

    auto rd64 = [&](uint32_t idx) -> uint64_t {
        return static_cast<uint64_t>(s.captured.idvs_sr[idx]) |
               (static_cast<uint64_t>(s.captured.idvs_sr[idx + 1u]) << 32);
    };
    uint32_t pos_srt_reg = s.captured.is_idvs2 ? static_cast<uint32_t>(valhall::csf::V13IDVSSR::VERTEX_SRT)
                                              : static_cast<uint32_t>(valhall::csf::IDVSSR::SRT_0);
    uint32_t frag_srt_reg = s.captured.is_idvs2 ? static_cast<uint32_t>(valhall::csf::V13IDVSSR::FRAGMENT_SRT)
                                               : static_cast<uint32_t>(valhall::csf::IDVSSR::SRT_2);
    uint32_t fau_reg = s.captured.is_idvs2 ? static_cast<uint32_t>(valhall::csf::V13IDVSSR::VERTEX_FAU)
                                           : static_cast<uint32_t>(valhall::csf::IDVSSR::FAU_0);

    std::ostringstream js;
    js << "{\n"
       << "  \"frame\": " << s.current_frame.load() << ",\n"
       << "  \"driver_abi\": \"" << s.detected_abi << "\",\n"
       << "  \"csf_draw_op\": \"" << (s.captured.is_idvs2 ? "RUN_IDVS2" : "RUN_IDVS") << "\",\n"
       << "  \"fb_width\": " << s.captured.fb_width << ",\n"
       << "  \"fb_height\": " << s.captured.fb_height << ",\n"
       << "  \"pos_srt_va\": \"0x" << std::hex << (rd64(pos_srt_reg) & 0x0000FFFFFFFFFFFFULL & ~0x3FULL) << "\",\n"
       << "  \"frag_srt_va\": \"0x" << (rd64(frag_srt_reg) & 0x0000FFFFFFFFFFFFULL & ~0x3FULL) << "\",\n"
       << "  \"fau_va\": \"0x" << (rd64(fau_reg) & 0x0000FFFFFFFFFFFFULL) << "\",\n"
       << "  \"fbd_va\": \"0x" << fbd_va << "\",\n"
       << "  \"rt_color_va\": \"0x" << s.captured.rt_color_va << "\"\n"
       << "}\n";
    std::string jstr = js.str();
    write_bin_file(out_dir + "/metadata.json", jstr.data(), jstr.size());
}

// ============================================================================
// 5. CSF Command Stream Interpreter (v10–v14: RUN_IDVS, RUN_IDVS2, RUN_FRAGMENT/2)
// ============================================================================
void walk_csf_stream_locked(LayerState& s, const uint64_t* words, size_t count, int depth) {
    if (!words || count == 0 || depth > 8) return;
    const uint32_t target_frame = get_target_frame();

    for (size_t i = 0; i < count; ++i) {
        uint64_t insn = words[i];
        uint32_t op_raw = static_cast<uint32_t>((insn >> 56) & 0xFFu);
        auto op = static_cast<valhall::csf::CSOpcode>(op_raw);

        if (op == valhall::csf::CSOpcode::MOVE48) {
            auto m = valhall::csf::CSMOVE48::unpack_u64(insn);
            uint32_t r = m.destination;
            if (r + 1 < 96) {
                s.regs[r] = static_cast<uint32_t>(m.immediate & 0xFFFFFFFFULL);
                s.regs[r + 1] = static_cast<uint32_t>((m.immediate >> 32) & 0xFFFFu);
            }
        } else if (op == valhall::csf::CSOpcode::MOVE32) {
            auto m = valhall::csf::CSMOVE32::unpack_u64(insn);
            if (m.destination < 96) {
                s.regs[m.destination] = m.immediate;
            }
        } else if (op == valhall::csf::CSOpcode::ADD_IMMEDIATE32) {
            auto a = valhall::csf::CSADDIMM32::unpack_u64(insn);
            if (a.source < 96 && a.destination < 96) {
                s.regs[a.destination] = static_cast<uint32_t>(static_cast<int32_t>(s.regs[a.source]) + a.immediate);
            }
        } else if (op == valhall::csf::CSOpcode::ADD_IMMEDIATE64) {
            auto a = valhall::csf::CSADDIMM64::unpack_u64(insn);
            if (a.source + 1 < 96 && a.destination + 1 < 96) {
                uint64_t src_v = static_cast<uint64_t>(s.regs[a.source]) |
                                 (static_cast<uint64_t>(s.regs[a.source + 1]) << 32);
                uint64_t dst_v = static_cast<uint64_t>(static_cast<int64_t>(src_v) + static_cast<int64_t>(a.immediate));
                s.regs[a.destination] = static_cast<uint32_t>(dst_v & 0xFFFFFFFFULL);
                s.regs[a.destination + 1] = static_cast<uint32_t>(dst_v >> 32);
            }
        } else if (op == valhall::csf::CSOpcode::CALL) {
            auto c = valhall::csf::CSCALL::unpack_u64(insn);
            uint32_t r_addr = c.address;
            uint32_t r_len = c.length;
            if (r_addr + 1 < 96 && r_len < 96) {
                uint64_t sub_va = static_cast<uint64_t>(s.regs[r_addr]) |
                                  (static_cast<uint64_t>(s.regs[r_addr + 1]) << 32);
                uint32_t sub_bytes = s.regs[r_len];
                if (sub_bytes >= 8 && sub_bytes <= (1u << 20)) {
                    if (const void* sp = gpu_to_cpu_locked(s, sub_va, sub_bytes)) {
                        walk_csf_stream_locked(s, reinterpret_cast<const uint64_t*>(sp), sub_bytes / 8u, depth + 1);
                    }
                }
            }
        } else if (op == valhall::csf::CSOpcode::RUN_IDVS) {
            if (is_dump_enabled() && !s.dump_done.load() &&
                s.current_frame.load() == target_frame && !s.captured.valid) {
                dump_idvs_capture_locked(s, insn, s.regs);
            }
        } else if (op_raw == static_cast<uint32_t>(valhall::csf::V13CSOpcode::RUN_IDVS2)) {
            if (is_dump_enabled() && !s.dump_done.load() &&
                s.current_frame.load() == target_frame && !s.captured.valid) {
                dump_idvs2_capture_locked(s, insn, s.regs);
            }
        } else if (op == valhall::csf::CSOpcode::RUN_FRAGMENT) {
            if (is_dump_enabled() && !s.dump_done.load() &&
                s.current_frame.load() == target_frame && s.captured.valid && !s.captured.has_frag_sr) {
                dump_fragment_capture_locked(s, s.regs, static_cast<uint32_t>(valhall::csf::FRAGMENTSR::FBD_POINTER));
            }
        } else if (op_raw == static_cast<uint32_t>(valhall::csf::V14CSOpcode::RUN_FRAGMENT2)) {
            if (is_dump_enabled() && !s.dump_done.load() &&
                s.current_frame.load() == target_frame && s.captured.valid && !s.captured.has_frag_sr) {
                dump_fragment_capture_locked(s, s.regs, static_cast<uint32_t>(valhall::csf::V14FRAGMENTSR::FBD_POINTER));
            }
        }
    }
}

void process_queue_kick_locked(LayerState& s, uint64_t buffer_gpu_addr) {
    auto it = s.queues_by_gpu_va.find(buffer_gpu_addr);
    if (it == s.queues_by_gpu_va.end()) return;
    CsfQueueState& q = it->second;
    if (q.buffer_size == 0) return;

    uint64_t insert_off = q.buffer_size;
    if (q.user_io_cpu != 0) {
        const volatile uint64_t* ins_ptr = reinterpret_cast<const volatile uint64_t*>(
            q.user_io_cpu + kCsUserIoInsertOffset + static_cast<size_t>(q.csi_index) * 16u);
        insert_off = *ins_ptr;
    }

    const uint8_t* ring = reinterpret_cast<const uint8_t*>(gpu_to_cpu_locked(s, q.buffer_gpu_addr, q.buffer_size));
    if (!ring) return;

    uint64_t cur = q.last_extract;
    while (cur < insert_off) {
        size_t off = static_cast<size_t>(cur % q.buffer_size);
        size_t avail = std::min<size_t>(q.buffer_size - off, static_cast<size_t>(insert_off - cur));
        if (avail < 8) break;
        walk_csf_stream_locked(s, reinterpret_cast<const uint64_t*>(ring + off), avail / 8u, 0);
        cur += (avail / 8u) * 8u;
    }
    q.last_extract = insert_off;
}

// ============================================================================
// 6. Hooked libc Entrypoints (open, close, ioctl, mmap, munmap)
// ============================================================================
bool is_mali_path(const char* path) {
    return path && std::strstr(path, "mali") != nullptr;
}

int hooked_open(const char* pathname, int flags, ...) {
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list ap;
        va_start(ap, flags);
        mode = static_cast<mode_t>(va_arg(ap, int));
        va_end(ap);
    }
    int fd = (flags & O_CREAT) ? state().orig_open(pathname, flags, mode)
                               : state().orig_open(pathname, flags);
    if (fd >= 0 && is_mali_path(pathname)) {
        std::lock_guard<std::mutex> lk(state().mu);
        state().mali_fds.insert(fd);
    }
    return fd;
}

int hooked_open64(const char* pathname, int flags, ...) {
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list ap;
        va_start(ap, flags);
        mode = static_cast<mode_t>(va_arg(ap, int));
        va_end(ap);
    }
    int fd = (flags & O_CREAT) ? state().orig_open64(pathname, flags, mode)
                               : state().orig_open64(pathname, flags);
    if (fd >= 0 && is_mali_path(pathname)) {
        std::lock_guard<std::mutex> lk(state().mu);
        state().mali_fds.insert(fd);
    }
    return fd;
}

int hooked_open_2(const char* pathname, int flags) {
    int fd = state().orig_open_2(pathname, flags);
    if (fd >= 0 && is_mali_path(pathname)) {
        std::lock_guard<std::mutex> lk(state().mu);
        state().mali_fds.insert(fd);
    }
    return fd;
}

int hooked_openat(int dirfd, const char* pathname, int flags, ...) {
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list ap;
        va_start(ap, flags);
        mode = static_cast<mode_t>(va_arg(ap, int));
        va_end(ap);
    }
    int fd = (flags & O_CREAT) ? state().orig_openat(dirfd, pathname, flags, mode)
                               : state().orig_openat(dirfd, pathname, flags);
    if (fd >= 0 && is_mali_path(pathname)) {
        std::lock_guard<std::mutex> lk(state().mu);
        state().mali_fds.insert(fd);
    }
    return fd;
}

int hooked_openat64(int dirfd, const char* pathname, int flags, ...) {
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list ap;
        va_start(ap, flags);
        mode = static_cast<mode_t>(va_arg(ap, int));
        va_end(ap);
    }
    int fd = (flags & O_CREAT) ? state().orig_openat64(dirfd, pathname, flags, mode)
                               : state().orig_openat64(dirfd, pathname, flags);
    if (fd >= 0 && is_mali_path(pathname)) {
        std::lock_guard<std::mutex> lk(state().mu);
        state().mali_fds.insert(fd);
    }
    return fd;
}

int hooked_close(int fd) {
    {
        std::lock_guard<std::mutex> lk(state().mu);
        state().mali_fds.erase(fd);
    }
    return state().orig_close(fd);
}

void* hooked_mmap_impl(void* (*real_mmap)(void*, size_t, int, int, int, off64_t),
                       void* addr, size_t length, int prot, int flags, int fd, off64_t offset) {
    void* ret = real_mmap(addr, length, prot, flags, fd, offset);
    if (ret != MAP_FAILED && fd >= 0) {
        std::lock_guard<std::mutex> lk(state().mu);
        if (state().mali_fds.count(fd)) {
            uint64_t off_u = static_cast<uint64_t>(offset);
            auto qit = state().queue_by_mmap_handle.find(off_u);
            if (qit != state().queue_by_mmap_handle.end()) {
                state().queues_by_gpu_va[qit->second].user_io_cpu = reinterpret_cast<uintptr_t>(ret);
            } else {
                state().mappings.push_back({ off_u, reinterpret_cast<uintptr_t>(ret), length });
            }
        }
    }
    return ret;
}

void* hooked_mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset) {
    return hooked_mmap_impl(
        reinterpret_cast<void* (*)(void*, size_t, int, int, int, off64_t)>(state().orig_mmap),
        addr, length, prot, flags, fd, static_cast<off64_t>(offset));
}

void* hooked_mmap64(void* addr, size_t length, int prot, int flags, int fd, off64_t offset) {
    return hooked_mmap_impl(state().orig_mmap64, addr, length, prot, flags, fd, offset);
}

int hooked_munmap(void* addr, size_t length) {
    {
        std::lock_guard<std::mutex> lk(state().mu);
        uintptr_t u = reinterpret_cast<uintptr_t>(addr);
        state().mappings.erase(
            std::remove_if(state().mappings.begin(), state().mappings.end(),
                           [u](const GpuMapping& m) { return m.cpu_va == u; }),
            state().mappings.end());
    }
    return state().orig_munmap(addr, length);
}

int hooked_ioctl(int fd, unsigned long request, ...) {
    va_list ap;
    va_start(ap, request);
    void* arg = va_arg(ap, void*);
    va_end(ap);

    int ret = state().orig_ioctl(fd, request, arg);
    if (ret < 0 || !arg) return ret;

    std::lock_guard<std::mutex> lk(state().mu);
    if (!state().mali_fds.count(fd)) return ret;

    if (request == MALI_IOCTL_GET_DDK_VERSION) {
        const auto* dv = reinterpret_cast<const kbase_ioctl_get_ddk_version*>(arg);
        if (dv->version_buffer && dv->size > 0) {
            const char* ver = reinterpret_cast<const char*>(static_cast<uintptr_t>(dv->version_buffer));
            state().detected_abi = std::string(ver, strnlen(ver, dv->size));
        }
    } else if (request == MALI_IOCTL_CS_QUEUE_REGISTER) {
        const auto* qr = reinterpret_cast<const kbase_ioctl_cs_queue_register*>(arg);
        CsfQueueState& q = state().queues_by_gpu_va[qr->buffer_gpu_addr];
        q.buffer_gpu_addr = qr->buffer_gpu_addr;
        q.buffer_size = qr->buffer_size;
    } else if (request == MALI_IOCTL_CS_QUEUE_REGISTER_EX) {
        const auto* qr = reinterpret_cast<const kbase_ioctl_cs_queue_register_ex*>(arg);
        CsfQueueState& q = state().queues_by_gpu_va[qr->buffer_gpu_addr];
        q.buffer_gpu_addr = qr->buffer_gpu_addr;
        q.buffer_size = qr->buffer_size;
    } else if (request == MALI_IOCTL_CS_QUEUE_BIND) {
        const auto* qb = reinterpret_cast<const kbase_ioctl_cs_queue_bind*>(arg);
        CsfQueueState& q = state().queues_by_gpu_va[qb->in.buffer_gpu_addr];
        q.csi_index = qb->in.csi_index;
        q.mmap_handle = qb->out.mmap_handle;
        state().queue_by_mmap_handle[qb->out.mmap_handle] = qb->in.buffer_gpu_addr;
    } else if (request == MALI_IOCTL_CS_QUEUE_KICK) {
        const auto* qk = reinterpret_cast<const kbase_ioctl_cs_queue_kick*>(arg);
        process_queue_kick_locked(state(), qk->buffer_gpu_addr);
    } else if (request == MALI_IOCTL_MEM_FREE) {
        const auto* mf = reinterpret_cast<const kbase_ioctl_mem_free*>(arg);
        uint64_t g48 = mf->gpu_addr & 0x0000FFFFFFFFFFFFULL;
        state().mappings.erase(
            std::remove_if(state().mappings.begin(), state().mappings.end(),
                           [g48](const GpuMapping& m) { return (m.gpu_va & 0x0000FFFFFFFFFFFFULL) == g48; }),
            state().mappings.end());
    }
    return ret;
}

// ============================================================================
// 7. ELF GOT/PLT Hooking for libGLES_mali.so (r38p1, r44p1, r49p1, r54p3)
// ============================================================================
void* resolve_hook_target(const char* sym) {
    if (std::strcmp(sym, "open") == 0) return reinterpret_cast<void*>(&hooked_open);
    if (std::strcmp(sym, "open64") == 0) return reinterpret_cast<void*>(&hooked_open64);
    if (std::strcmp(sym, "__open_2") == 0) return reinterpret_cast<void*>(&hooked_open_2);
    if (std::strcmp(sym, "openat") == 0) return reinterpret_cast<void*>(&hooked_openat);
    if (std::strcmp(sym, "openat64") == 0) return reinterpret_cast<void*>(&hooked_openat64);
    if (std::strcmp(sym, "close") == 0) return reinterpret_cast<void*>(&hooked_close);
    if (std::strcmp(sym, "ioctl") == 0) return reinterpret_cast<void*>(&hooked_ioctl);
    if (std::strcmp(sym, "mmap") == 0) return reinterpret_cast<void*>(&hooked_mmap);
    if (std::strcmp(sym, "mmap64") == 0) return reinterpret_cast<void*>(&hooked_mmap64);
    if (std::strcmp(sym, "munmap") == 0) return reinterpret_cast<void*>(&hooked_munmap);
    return nullptr;
}

std::string detect_ddk_version_from_phdr(const struct dl_phdr_info* info) {
    for (ElfW(Half) i = 0; i < info->dlpi_phnum; ++i) {
        const auto& ph = info->dlpi_phdr[i];
        if (ph.p_type != PT_LOAD || (ph.p_flags & PF_W) != 0 || ph.p_memsz < 16) continue;
        const char* base = reinterpret_cast<const char*>(info->dlpi_addr + ph.p_vaddr);
        size_t len = static_cast<size_t>(ph.p_filesz);
        for (size_t j = 0; j + 12 < len; ++j) {
            if (base[j] == 'r' &&
                base[j + 1] >= '0' && base[j + 1] <= '9' &&
                base[j + 2] >= '0' && base[j + 2] <= '9' &&
                base[j + 3] == 'p' &&
                base[j + 4] >= '0' && base[j + 4] <= '9' &&
                base[j + 5] == '-' &&
                base[j + 6] >= '0' && base[j + 6] <= '9' &&
                base[j + 7] >= '0' && base[j + 7] <= '9' &&
                base[j + 8] == 'e' && base[j + 9] == 'a' && base[j + 10] == 'c') {
                size_t k = j;
                while (k < len && k - j < 32 && base[k] >= 0x20 && base[k] < 0x7f && base[k] != '"') ++k;
                return std::string(base + j, k - j);
            }
        }
    }
    return {};
}

uintptr_t fix_dyn_ptr(ElfW(Addr) dlpi_addr, ElfW(Addr) val) {
    if (val == 0) return 0;
    return (val < dlpi_addr) ? (dlpi_addr + val) : val;
}

void patch_rela_table(ElfW(Addr) dlpi_addr, const char* strtab, const ElfW(Sym)* symtab,
                      const ElfW(Rela)* rela, size_t rela_sz, size_t page_sz) {
    if (!rela || rela_sz == 0) return;
    const size_t n_rel = rela_sz / sizeof(ElfW(Rela));
    for (size_t i = 0; i < n_rel; ++i) {
        const ElfW(Rela)& r = rela[i];
#if defined(__aarch64__)
        uint32_t sym_idx = ELF64_R_SYM(r.r_info);
        uint32_t r_type = ELF64_R_TYPE(r.r_info);
        if (r_type != R_AARCH64_JUMP_SLOT && r_type != R_AARCH64_GLOB_DAT) continue;
#elif defined(__x86_64__)
        uint32_t sym_idx = ELF64_R_SYM(r.r_info);
        uint32_t r_type = ELF64_R_TYPE(r.r_info);
        if (r_type != R_X86_64_JUMP_SLOT && r_type != R_X86_64_GLOB_DAT) continue;
#else
        uint32_t sym_idx = ELF64_R_SYM(r.r_info);
#endif
        const char* sym_name = strtab + symtab[sym_idx].st_name;
        void* replacement = resolve_hook_target(sym_name);
        if (!replacement) continue;

        void** got_slot = reinterpret_cast<void**>(dlpi_addr + r.r_offset);
        uintptr_t page_base = reinterpret_cast<uintptr_t>(got_slot) & ~(page_sz - 1u);
        if (mprotect(reinterpret_cast<void*>(page_base), page_sz, PROT_READ | PROT_WRITE) == 0) {
            *got_slot = replacement;
        }
    }
}

int patch_elf_got_cb(struct dl_phdr_info* info, size_t /*size*/, void* /*data*/) {
    if (!info->dlpi_name || std::strstr(info->dlpi_name, "libGLES_mali") == nullptr) {
        return 0;
    }
    if (state().detected_abi == "unknown") {
        std::string probed = detect_ddk_version_from_phdr(info);
        if (!probed.empty()) state().detected_abi = probed;
        else if (std::strstr(info->dlpi_name, "r49p1") != nullptr) state().detected_abi = "r49p1";
        else if (std::strstr(info->dlpi_name, "r44p1") != nullptr) state().detected_abi = "r44p1";
        else if (std::strstr(info->dlpi_name, "r38p1") != nullptr) state().detected_abi = "r38p1";
        else state().detected_abi = "csf-unknown";
    }

    const ElfW(Dyn)* dyn = nullptr;
    for (ElfW(Half) i = 0; i < info->dlpi_phnum; ++i) {
        if (info->dlpi_phdr[i].p_type == PT_DYNAMIC) {
            dyn = reinterpret_cast<const ElfW(Dyn)*>(info->dlpi_addr + info->dlpi_phdr[i].p_vaddr);
            break;
        }
    }
    if (!dyn) return 0;

    const char* strtab = nullptr;
    const ElfW(Sym)* symtab = nullptr;
    const ElfW(Rela)* jmprel = nullptr;
    const ElfW(Rela)* rela = nullptr;
    size_t jmprel_sz = 0;
    size_t rela_sz = 0;
    for (; dyn->d_tag != DT_NULL; ++dyn) {
        if (dyn->d_tag == DT_STRTAB) strtab = reinterpret_cast<const char*>(fix_dyn_ptr(info->dlpi_addr, dyn->d_un.d_ptr));
        else if (dyn->d_tag == DT_SYMTAB) symtab = reinterpret_cast<const ElfW(Sym)*>(fix_dyn_ptr(info->dlpi_addr, dyn->d_un.d_ptr));
        else if (dyn->d_tag == DT_JMPREL) jmprel = reinterpret_cast<const ElfW(Rela)*>(fix_dyn_ptr(info->dlpi_addr, dyn->d_un.d_ptr));
        else if (dyn->d_tag == DT_PLTRELSZ) jmprel_sz = dyn->d_un.d_val;
        else if (dyn->d_tag == DT_RELA) rela = reinterpret_cast<const ElfW(Rela)*>(fix_dyn_ptr(info->dlpi_addr, dyn->d_un.d_ptr));
        else if (dyn->d_tag == DT_RELASZ) rela_sz = dyn->d_un.d_val;
    }
    if (!strtab || !symtab) return 0;

    const size_t page_sz = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    patch_rela_table(info->dlpi_addr, strtab, symtab, jmprel, jmprel_sz, page_sz);
    patch_rela_table(info->dlpi_addr, strtab, symtab, rela, rela_sz, page_sz);
    return 0;
}

void install_mali_hooks() {
    std::lock_guard<std::mutex> lk(state().mu);
    LayerState& s = state();
    if (!s.orig_open) {
        s.orig_open = reinterpret_cast<int (*)(const char*, int, ...)>(dlsym(RTLD_NEXT, "open"));
        s.orig_open64 = reinterpret_cast<int (*)(const char*, int, ...)>(dlsym(RTLD_NEXT, "open64"));
        s.orig_open_2 = reinterpret_cast<int (*)(const char*, int)>(dlsym(RTLD_NEXT, "__open_2"));
        s.orig_openat = reinterpret_cast<int (*)(int, const char*, int, ...)>(dlsym(RTLD_NEXT, "openat"));
        s.orig_openat64 = reinterpret_cast<int (*)(int, const char*, int, ...)>(dlsym(RTLD_NEXT, "openat64"));
        s.orig_close = reinterpret_cast<int (*)(int)>(dlsym(RTLD_NEXT, "close"));
        s.orig_ioctl = reinterpret_cast<int (*)(int, unsigned long, ...)>(dlsym(RTLD_NEXT, "ioctl"));
        s.orig_mmap = reinterpret_cast<void* (*)(void*, size_t, int, int, int, off_t)>(dlsym(RTLD_NEXT, "mmap"));
        s.orig_mmap64 = reinterpret_cast<void* (*)(void*, size_t, int, int, int, off64_t)>(dlsym(RTLD_NEXT, "mmap64"));
        s.orig_munmap = reinterpret_cast<int (*)(void*, size_t)>(dlsym(RTLD_NEXT, "munmap"));
    }
    dl_iterate_phdr(patch_elf_got_cb, nullptr);
    s.hooks_installed = true;
}

__attribute__((constructor)) static void on_layer_load() {
    install_mali_hooks();
}

// ============================================================================
// 8. Vulkan Layer Dispatch Table (Instance & Device)
// ============================================================================
struct InstanceDispatch {
    PFN_vkGetInstanceProcAddr gipa = nullptr;
    PFN_vkDestroyInstance destroy_instance = nullptr;
    PFN_vkCreateDevice create_device = nullptr;
};

struct DeviceDispatch {
    PFN_vkGetDeviceProcAddr gdpa = nullptr;
    PFN_vkDestroyDevice destroy_device = nullptr;
    PFN_vkQueuePresentKHR queue_present = nullptr;
};

std::mutex g_vk_mu;
std::unordered_map<void*, InstanceDispatch> g_inst_disp;
std::unordered_map<void*, DeviceDispatch> g_dev_disp;

static void* dispatch_key(void* handle) {
    return *reinterpret_cast<void**>(handle);
}

VKAPI_ATTR VkResult VKAPI_CALL layer_CreateInstance(
    const VkInstanceCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkInstance* pInstance) {
    install_mali_hooks();

    const auto* layer_info = reinterpret_cast<const VkLayerInstanceCreateInfo*>(pCreateInfo->pNext);
    while (layer_info && (layer_info->sType != VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO ||
                          layer_info->function != VK_LAYER_LINK_INFO)) {
        layer_info = reinterpret_cast<const VkLayerInstanceCreateInfo*>(layer_info->pNext);
    }
    if (!layer_info || !layer_info->u.pLayerInfo) {
        std::fprintf(stderr, "layer_info=%p\n", (void*)layer_info);
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    PFN_vkGetInstanceProcAddr next_gipa = layer_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    const_cast<VkLayerInstanceCreateInfo*>(layer_info)->u.pLayerInfo = layer_info->u.pLayerInfo->pNext;
    auto next_create = reinterpret_cast<PFN_vkCreateInstance>(next_gipa(VK_NULL_HANDLE, "vkCreateInstance"));
    if (!next_create) return VK_ERROR_INITIALIZATION_FAILED;

    VkResult res = next_create(pCreateInfo, pAllocator, pInstance);
    install_mali_hooks();
    if (res == VK_SUCCESS && *pInstance) {
        InstanceDispatch d{};
        d.gipa = next_gipa;
        d.destroy_instance = reinterpret_cast<PFN_vkDestroyInstance>(next_gipa(*pInstance, "vkDestroyInstance"));
        d.create_device = reinterpret_cast<PFN_vkCreateDevice>(next_gipa(*pInstance, "vkCreateDevice"));
        std::lock_guard<std::mutex> lk(g_vk_mu);
        g_inst_disp[dispatch_key(*pInstance)] = d;
    }
    return res;
}

VKAPI_ATTR void VKAPI_CALL layer_DestroyInstance(VkInstance instance, const VkAllocationCallbacks* pAllocator) {
    InstanceDispatch d{};
    {
        std::lock_guard<std::mutex> lk(g_vk_mu);
        auto it = g_inst_disp.find(dispatch_key(instance));
        if (it != g_inst_disp.end()) {
            d = it->second;
            g_inst_disp.erase(it);
        }
    }
    if (d.destroy_instance) d.destroy_instance(instance, pAllocator);
}

VKAPI_ATTR VkResult VKAPI_CALL layer_CreateDevice(
    VkPhysicalDevice physicalDevice,
    const VkDeviceCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDevice* pDevice) {
    install_mali_hooks();

    const auto* layer_info = reinterpret_cast<const VkLayerDeviceCreateInfo*>(pCreateInfo->pNext);
    while (layer_info && (layer_info->sType != VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO ||
                          layer_info->function != VK_LAYER_LINK_INFO)) {
        layer_info = reinterpret_cast<const VkLayerDeviceCreateInfo*>(layer_info->pNext);
    }
    if (!layer_info) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr next_gipa = layer_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr next_gdpa = layer_info->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    const_cast<VkLayerDeviceCreateInfo*>(layer_info)->u.pLayerInfo = layer_info->u.pLayerInfo->pNext;
    auto next_create = reinterpret_cast<PFN_vkCreateDevice>(next_gipa(VK_NULL_HANDLE, "vkCreateDevice"));
    if (!next_create) return VK_ERROR_INITIALIZATION_FAILED;

    VkResult res = next_create(physicalDevice, pCreateInfo, pAllocator, pDevice);
    if (res == VK_SUCCESS && *pDevice) {
        DeviceDispatch d{};
        d.gdpa = next_gdpa;
        d.destroy_device = reinterpret_cast<PFN_vkDestroyDevice>(next_gdpa(*pDevice, "vkDestroyDevice"));
        d.queue_present = reinterpret_cast<PFN_vkQueuePresentKHR>(next_gdpa(*pDevice, "vkQueuePresentKHR"));
        std::lock_guard<std::mutex> lk(g_vk_mu);
        g_dev_disp[dispatch_key(*pDevice)] = d;
    }
    return res;
}

VKAPI_ATTR void VKAPI_CALL layer_DestroyDevice(VkDevice device, const VkAllocationCallbacks* pAllocator) {
    DeviceDispatch d{};
    {
        std::lock_guard<std::mutex> lk(g_vk_mu);
        auto it = g_dev_disp.find(dispatch_key(device));
        if (it != g_dev_disp.end()) {
            d = it->second;
            g_dev_disp.erase(it);
        }
    }
    if (d.destroy_device) d.destroy_device(device, pAllocator);
}

VKAPI_ATTR VkResult VKAPI_CALL layer_QueuePresentKHR(VkQueue queue, const VkPresentInfoKHR* pPresentInfo) {
    DeviceDispatch d{};
    {
        std::lock_guard<std::mutex> lk(g_vk_mu);
        auto it = g_dev_disp.find(dispatch_key(queue));
        if (it != g_dev_disp.end()) d = it->second;
    }
    VkResult res = d.queue_present ? d.queue_present(queue, pPresentInfo) : VK_SUCCESS;

    if (is_dump_enabled() && !state().dump_done.load()) {
        std::lock_guard<std::mutex> lk(state().mu);
        if (state().captured.valid && state().current_frame.load() == get_target_frame()) {
            if (state().captured.rt_color_va != 0 && state().captured.fb_height > 0) {
                size_t rt_bytes = static_cast<size_t>(state().captured.fb_height) * 16384u;
                if (const void* p = gpu_to_cpu_locked(state(), state().captured.rt_color_va, rt_bytes)) {
                    write_bin_file(state().captured.out_dir + "/rt_color_after.bin", p, rt_bytes);
                }
            }
            state().dump_done.store(true);
        }
        state().current_frame.fetch_add(1u);
    }
    return res;
}

} // namespace

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL layer_GetDeviceProcAddr(VkDevice device, const char* pName) {
    if (std::strcmp(pName, "vkGetDeviceProcAddr") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&layer_GetDeviceProcAddr);
    if (std::strcmp(pName, "vkDestroyDevice") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&layer_DestroyDevice);
    if (std::strcmp(pName, "vkQueuePresentKHR") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&layer_QueuePresentKHR);
    DeviceDispatch d{};
    {
        std::lock_guard<std::mutex> lk(g_vk_mu);
        auto it = g_dev_disp.find(dispatch_key(device));
        if (it != g_dev_disp.end()) d = it->second;
    }
    return d.gdpa ? d.gdpa(device, pName) : nullptr;
}

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL layer_GetInstanceProcAddr(VkInstance instance, const char* pName) {
    if (std::strcmp(pName, "vkGetInstanceProcAddr") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&layer_GetInstanceProcAddr);
    if (std::strcmp(pName, "vkCreateInstance") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&layer_CreateInstance);
    if (std::strcmp(pName, "vkDestroyInstance") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&layer_DestroyInstance);
    if (std::strcmp(pName, "vkCreateDevice") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&layer_CreateDevice);
    if (std::strcmp(pName, "vkGetDeviceProcAddr") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&layer_GetDeviceProcAddr);
    InstanceDispatch d{};
    {
        std::lock_guard<std::mutex> lk(g_vk_mu);
        auto it = g_inst_disp.find(dispatch_key(instance));
        if (it != g_inst_disp.end()) d = it->second;
    }
    return d.gipa ? d.gipa(instance, pName) : nullptr;
}

extern "C" {

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char* pName) {
    return layer_GetDeviceProcAddr(device, pName);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char* pName) {
    return layer_GetInstanceProcAddr(instance, pName);
}

VKAPI_ATTR VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* pVersionStruct) {
    install_mali_hooks();
    if (!pVersionStruct || pVersionStruct->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (pVersionStruct->loaderLayerInterfaceVersion >= 2) {
        pVersionStruct->loaderLayerInterfaceVersion = 2;
        pVersionStruct->pfnGetInstanceProcAddr = &layer_GetInstanceProcAddr;
        pVersionStruct->pfnGetDeviceProcAddr = &layer_GetDeviceProcAddr;
        pVersionStruct->pfnGetPhysicalDeviceProcAddr = nullptr;
    }
    return VK_SUCCESS;
}

// ============================================================================
// 9. Self-Test Entrypoint (In-Process CSF Stream Replay Verification)
// ============================================================================
bool mali_csf_dump_layer_selftest(const char* out_dir, uint32_t target_frame) {
    setenv("MALI_CSF_DUMP_ENABLE", "1", 1);
    setenv("MALI_CSF_DUMP_DIR", out_dir, 1);
    char fbuf[16];
    std::snprintf(fbuf, sizeof(fbuf), "%u", target_frame);
    setenv("MALI_CSF_DUMP_FRAME", fbuf, 1);

    std::lock_guard<std::mutex> lk(state().mu);
    state().dump_done.store(false);
    state().current_frame.store(target_frame);
    state().captured = CapturedIdvs{};
    return state().hooks_installed;
}

} // extern "C"

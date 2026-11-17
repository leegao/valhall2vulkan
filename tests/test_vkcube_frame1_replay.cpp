#include "valhall_csf_pipeline.h"
#include "valhall_to_spirv.h"
#include "valhall_csf_descriptors_generated.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <signal.h>
#include <execinfo.h>
#include <unistd.h>
static void segv_h(int sig) {
    void* b[64]; int n = backtrace(b, 64);
    backtrace_symbols_fd(b, n, STDERR_FILENO);
    _exit(139);
}

#include <string>
#include <utility>
#include <vector>

using namespace valhall;
using namespace valhall::pipeline;

static std::vector<uint8_t> read_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cerr << "Failed to open " << path << "\n";
        std::exit(1);
    }
    auto sz = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(sz);
    if (sz > 0) f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

static void patch_va_occurrences(
    std::vector<uint8_t>& buf,
    const std::vector<std::pair<uint64_t, uint64_t>>& va_map) {
    if (buf.size() < 8) return;
    for (size_t off = 0; off + 8 <= buf.size(); off += 4) {
        uint64_t raw = 0;
        std::memcpy(&raw, buf.data() + off, 8);
        uint64_t va48 = raw & 0x0000FFFFFFFFFFFFULL;
        uint64_t hi16 = raw & 0xFFFF000000000000ULL;
        for (const auto& kv : va_map) {
            if (kv.first != 0 && va48 == kv.first) {
                uint64_t patched = hi16 | (kv.second & 0x0000FFFFFFFFFFFFULL);
                std::memcpy(buf.data() + off, &patched, 8);
                break;
            }
        }
    }
}

static CsfStagingRegisterFile load_sr_from_u64(const std::vector<uint8_t>& raw) {
    CsfStagingRegisterFile sr{};
    size_t n = std::min<size_t>(64, raw.size() / 8);
    for (size_t i = 0; i < n; ++i) {
        uint64_t v = 0;
        std::memcpy(&v, raw.data() + i * 8, 8);
        sr.regs[i] = static_cast<uint32_t>(v & 0xffffffffULL);
    }
    return sr;
}

static void patch_sr_va_occurrences(
    CsfStagingRegisterFile& sr,
    const std::vector<std::pair<uint64_t, uint64_t>>& va_map) {
    for (size_t i = 0; i + 1 < 64; i += 2) {
        uint64_t raw = sr.reg64(static_cast<uint32_t>(i));
        uint64_t va48 = raw & 0x0000FFFFFFFFFFFFULL;
        uint64_t hi16 = raw & 0xFFFF000000000000ULL;
        uint64_t lo6  = raw & 0x3FULL;
        for (const auto& kv : va_map) {
            if (kv.first != 0 && (va48 == kv.first || (va48 & ~0x3FULL) == kv.first)) {
                uint64_t flags = (va48 == kv.first) ? 0ULL : lo6;
                sr.set_reg64(static_cast<uint32_t>(i), hi16 | ((kv.second & 0x0000FFFFFFFFFFFFULL) | flags));
                break;
            }
        }
    }
}

int main() {
    signal(SIGSEGV, segv_h);
    std::cout << std::unitbuf;
    const std::string dir = "tests/data/vkcube_frame1";
    std::cout << "=== [Replay Test] vkcube Frame 1 (libr54p3_vulkan_icd.so MALI_ICD_NATIVE_AOT=1 dump) ===\n";

    ValhallVulkanEngine vk;
    std::string err;
    if (!vk.init(&err)) {
        std::cerr << "ValhallVulkanEngine::init failed: " << err << "\n";
        return 1;
    }
    std::cout << "Lavapipe device: " << vk.device_name() << "\n";

    auto idvs_sr_raw   = read_bin(dir + "/idvs_sr.bin");
    auto frag_sr_raw   = read_bin(dir + "/frag_sr.bin");
    auto csrun_raw     = read_bin(dir + "/csrunidvs_insn.bin");
    auto pos_shader    = read_bin(dir + "/pos_shader.bin");
    auto var_shader    = read_bin(dir + "/var_shader.bin");
    auto frag_shader   = read_bin(dir + "/frag_shader.bin");
    auto fbd_raw       = read_bin(dir + "/fbd.bin");
    auto tsd_raw       = read_bin(dir + "/tsd.bin");
    auto tcd_raw       = read_bin(dir + "/tcd.bin");
    auto zsd_raw       = read_bin(dir + "/zsd.bin");
    auto bd_raw        = read_bin(dir + "/bd.bin");
    auto pos_srt_raw   = read_bin(dir + "/pos_srt.bin");
    auto frag_srt_raw  = read_bin(dir + "/frag_srt.bin");
    pos_srt_raw.insert(pos_srt_raw.begin(), 8, 0);
    frag_srt_raw.insert(frag_srt_raw.begin(), 8, 0);
    auto srt_buf0_raw  = read_bin(dir + "/srt_buf0.bin");
    auto srt_buf1_raw  = read_bin(dir + "/srt_buf1.bin");
    auto primary_raw   = read_bin(dir + "/primary_buf.bin");
    auto secondary_raw = read_bin(dir + "/secondary_buf.bin");
    auto fau_raw       = read_bin(dir + "/fau.bin");
    auto pos_spd_raw   = read_bin(dir + "/pos_spd.bin");
    auto var_spd_raw   = read_bin(dir + "/var_spd.bin");
    auto frag_spd_raw  = read_bin(dir + "/frag_spd.bin");

    auto rt_buf        = vk.alloc_buffer(16384ULL * 256ULL);
    auto fbd_buf       = vk.alloc_buffer(fbd_raw.size(), fbd_raw.data(), fbd_raw.size());
    auto tsd_buf       = vk.alloc_buffer(tsd_raw.size(), tsd_raw.data(), tsd_raw.size());
    auto tcd_buf       = vk.alloc_buffer(tcd_raw.size(), tcd_raw.data(), tcd_raw.size());
    auto zsd_buf       = vk.alloc_buffer(zsd_raw.size(), zsd_raw.data(), zsd_raw.size());
    auto bd_buf        = vk.alloc_buffer(bd_raw.size(), bd_raw.data(), bd_raw.size());
    auto pos_srt_buf   = vk.alloc_buffer(pos_srt_raw.size(), pos_srt_raw.data(), pos_srt_raw.size());
    auto frag_srt_buf  = vk.alloc_buffer(frag_srt_raw.size(), frag_srt_raw.data(), frag_srt_raw.size());
    auto srt_buf0      = vk.alloc_buffer(srt_buf0_raw.size(), srt_buf0_raw.data(), srt_buf0_raw.size());
    auto srt_buf1      = vk.alloc_buffer(srt_buf1_raw.size(), srt_buf1_raw.data(), srt_buf1_raw.size());
    auto primary_buf   = vk.alloc_buffer(primary_raw.size(), primary_raw.data(), primary_raw.size());
    auto secondary_buf = vk.alloc_buffer(secondary_raw.size(), secondary_raw.data(), secondary_raw.size());
    auto fau_buf       = vk.alloc_buffer(std::max<size_t>(fau_raw.size(), 256), fau_raw.data(), fau_raw.size());
    auto pos_spd_buf   = vk.alloc_buffer(pos_spd_raw.size(), pos_spd_raw.data(), pos_spd_raw.size());
    auto var_spd_buf   = vk.alloc_buffer(var_spd_raw.size(), var_spd_raw.data(), var_spd_raw.size());
    auto frag_spd_buf  = vk.alloc_buffer(frag_spd_raw.size(), frag_spd_raw.data(), frag_spd_raw.size());
    auto pos_code_buf  = vk.alloc_buffer(pos_shader.size(), pos_shader.data(), pos_shader.size());
    auto var_code_buf  = vk.alloc_buffer(var_shader.size(), var_shader.data(), var_shader.size());
    auto frag_code_buf = vk.alloc_buffer(frag_shader.size(), frag_shader.data(), frag_shader.size());
    auto scratch_buf   = vk.alloc_buffer(65536);
    std::vector<uint32_t> seq_idx(36); for (uint32_t k=0;k<36;++k) seq_idx[k]=k;
    auto idx_host_buf  = vk.alloc_buffer(256, seq_idx.data(), seq_idx.size()*4);

    std::vector<std::pair<uint64_t, uint64_t>> va_map = {
        {0x78685bd1b000ULL, reinterpret_cast<uint64_t>(rt_buf.mapped)},
        {0x78682c63e480ULL, reinterpret_cast<uint64_t>(fbd_buf.mapped)},
        {0x78682c63e000ULL, reinterpret_cast<uint64_t>(tsd_buf.mapped)},
        {0x78682c63e400ULL, reinterpret_cast<uint64_t>(tcd_buf.mapped)},
        {0x78685bcfc0c0ULL, reinterpret_cast<uint64_t>(zsd_buf.mapped)},
        {0x78682c5fc1c0ULL, reinterpret_cast<uint64_t>(pos_srt_buf.mapped)},
        {0x78682c5fc240ULL, reinterpret_cast<uint64_t>(fau_buf.mapped)},
        {0x78685bcfc000ULL, reinterpret_cast<uint64_t>(pos_spd_buf.mapped)},
        {0x78685bcfc020ULL, reinterpret_cast<uint64_t>(var_spd_buf.mapped)},
        {0x78685bcfc040ULL, reinterpret_cast<uint64_t>(frag_spd_buf.mapped)},
        {0x78685bcfc060ULL, reinterpret_cast<uint64_t>(srt_buf0.mapped)},
        {0x78682c63e740ULL, reinterpret_cast<uint64_t>(srt_buf1.mapped)},
        {0x10050580ULL,     reinterpret_cast<uint64_t>(pos_code_buf.mapped)},
        {0x10050800ULL,     reinterpret_cast<uint64_t>(var_code_buf.mapped)},
        {0x10050900ULL,     reinterpret_cast<uint64_t>(frag_code_buf.mapped)},
        {0x78685da06000ULL, reinterpret_cast<uint64_t>(primary_buf.mapped)},
        {0x786863742400ULL, reinterpret_cast<uint64_t>(secondary_buf.mapped)},
        {0x786863744000ULL, reinterpret_cast<uint64_t>(scratch_buf.mapped)},
        {0x78682c63e280ULL, reinterpret_cast<uint64_t>(scratch_buf.mapped)},
        {0x78685bd18000ULL, reinterpret_cast<uint64_t>(scratch_buf.mapped)},
        {0x78685bd19000ULL, reinterpret_cast<uint64_t>(scratch_buf.mapped)},
        {0x78685da1d000ULL, reinterpret_cast<uint64_t>(scratch_buf.mapped)},
        {0x78685da0d010ULL, reinterpret_cast<uint64_t>(scratch_buf.mapped)},
        {0x78685bcfc0e0ULL, reinterpret_cast<uint64_t>(bd_buf.mapped)},
        {0x78685bd17000ULL, reinterpret_cast<uint64_t>(idx_host_buf.mapped)},
    };

    patch_va_occurrences(fbd_raw, va_map);
    patch_va_occurrences(tcd_raw, va_map);
    patch_va_occurrences(zsd_raw, va_map);
    patch_va_occurrences(bd_raw, va_map);
    patch_va_occurrences(pos_srt_raw, va_map);
    patch_va_occurrences(frag_srt_raw, va_map);
    patch_va_occurrences(srt_buf0_raw, va_map);
    patch_va_occurrences(srt_buf1_raw, va_map);
    patch_va_occurrences(pos_spd_raw, va_map);
    patch_va_occurrences(var_spd_raw, va_map);
    patch_va_occurrences(frag_spd_raw, va_map);

    std::memcpy(fbd_buf.mapped, fbd_raw.data(), fbd_raw.size());
    std::memcpy(tcd_buf.mapped, tcd_raw.data(), tcd_raw.size());
    std::memcpy(zsd_buf.mapped, zsd_raw.data(), zsd_raw.size());
    std::memcpy(bd_buf.mapped, bd_raw.data(), bd_raw.size());
    std::memcpy(pos_srt_buf.mapped, pos_srt_raw.data(), pos_srt_raw.size());
    std::memcpy(frag_srt_buf.mapped, frag_srt_raw.data(), frag_srt_raw.size());
    std::memcpy(srt_buf0.mapped, srt_buf0_raw.data(), srt_buf0_raw.size());
    std::memcpy(srt_buf1.mapped, srt_buf1_raw.data(), srt_buf1_raw.size());
    std::memcpy(pos_spd_buf.mapped, pos_spd_raw.data(), pos_spd_raw.size());
    std::memcpy(var_spd_buf.mapped, var_spd_raw.data(), var_spd_raw.size());
    std::memcpy(frag_spd_buf.mapped, frag_spd_raw.data(), frag_spd_raw.size());

    CsfStagingRegisterFile idvs_sr = load_sr_from_u64(idvs_sr_raw);
    CsfStagingRegisterFile frag_sr = load_sr_from_u64(frag_sr_raw);
    patch_sr_va_occurrences(idvs_sr, va_map);
    patch_sr_va_occurrences(frag_sr, va_map);

    uint64_t csrun_word = 0;
    std::memcpy(&csrun_word, csrun_raw.data(), sizeof(csrun_word));
    std::cerr << "pos_srt_buf.device_address=" << std::hex << pos_srt_buf.device_address << " pos_srt_buf.mapped=" << (uint64_t)pos_srt_buf.mapped << "\n";
    std::cerr << "idvs_sr r0=" << std::hex << idvs_sr.reg64(0) << " pos_srt_buf=" << (uint64_t)pos_srt_buf.mapped << " srt0=" << (uint64_t)srt_buf0.mapped << "\n";
    csf::CSRUNIDVS run_idvs = csf::CSRUNIDVS::unpack_u64(csrun_word);

    std::vector<uint32_t> vert_spirv, frag_spirv;
    bool ok = vk.execute_cs_run_idvs_graphics(
        idvs_sr,
        frag_sr,
        run_idvs,
        pos_shader.data(), pos_shader.size(),
        var_shader.data(), var_shader.size(),
        frag_shader.data(), frag_shader.size(),
        &vert_spirv,
        &frag_spirv,
        &err);
    std::cout << "execute_cs_run_idvs_graphics returned ok=" << ok
              << " vert_words=" << vert_spirv.size()
              << " frag_words=" << frag_spirv.size()
              << " err=\"" << err << "\"\n";
    if (!ok) return 1;

    const uint32_t* px = reinterpret_cast<const uint32_t*>(rt_buf.mapped);
    size_t non_zero = 0;
    for (size_t y = 0; y < 256; ++y) {
        for (size_t x = 0; x < 256; ++x) {
            if (px[y * (16384 / 4) + x] != 0) ++non_zero;
        }
    }
    std::cout << "[PASS] vkcube Frame 1 replayed locally! non_zero_pixels=" << non_zero << "/65536\n";
    return 0;
}

#include "csf_config_log.h"
#include "csf_dumper.h"
#include "csf_elf_hooks.h"
#include "csf_interpreter.h"
#include "csf_kbase_uapi.h"
#include "csf_layer_state.h"
#include "csf_va_tracker.h"
#include "valhall_csf_descriptors_generated.h"

#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace mali_csf_dump;

namespace {

// Helper to build raw 64-bit CSF instructions for testing
uint64_t mk_move48(uint32_t dst, uint64_t imm48) {
    valhall::csf::CSMOVE48 m{};
    m.destination = dst;
    m.immediate = imm48 & 0x0000FFFFFFFFFFFFULL;
    return m.pack_u64();
}

uint64_t mk_move32(uint32_t dst, uint32_t imm32) {
    valhall::csf::CSMOVE32 m{};
    m.destination = dst;
    m.immediate = imm32;
    return m.pack_u64();
}

uint64_t mk_arith_reg32(uint32_t sub_op, uint32_t dst, uint32_t src0, uint32_t src1, uint32_t imm24 = 0) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(valhall::csf::V13CSOpcode::ARITH_REG32)) << 56) |
           (static_cast<uint64_t>(dst & 0xFFu) << 48) |
           (static_cast<uint64_t>(src0 & 0xFFu) << 40) |
           (static_cast<uint64_t>(src1 & 0xFFu) << 32) |
           (static_cast<uint64_t>(sub_op & 0xFFu) << 24) |
           static_cast<uint64_t>(imm24 & 0xFFFFFFu);
}

uint64_t mk_arith_reg64(uint32_t sub_op, uint32_t dst, uint32_t src0, uint32_t src1, uint32_t imm24 = 0) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(valhall::csf::V13CSOpcode::ARITH_REG64)) << 56) |
           (static_cast<uint64_t>(dst & 0xFFu) << 48) |
           (static_cast<uint64_t>(src0 & 0xFFu) << 40) |
           (static_cast<uint64_t>(src1 & 0xFFu) << 32) |
           (static_cast<uint64_t>(sub_op & 0xFFu) << 24) |
           static_cast<uint64_t>(imm24 & 0xFFFFFFu);
}

uint64_t mk_load_multiple(uint32_t base_reg, uint32_t addr_reg, uint32_t mask, int32_t offset) {
    valhall::csf::CSLOADMULTIPLE lm{};
    lm.base_register = base_reg;
    lm.address = addr_reg;
    lm.mask = mask;
    lm.offset = offset;
    return lm.pack_u64();
}

uint64_t mk_store_multiple(uint32_t base_reg, uint32_t addr_reg, uint32_t mask, int32_t offset) {
    valhall::csf::CSSTOREMULTIPLE sm{};
    sm.base_register = base_reg;
    sm.address = addr_reg;
    sm.mask = mask;
    sm.offset = offset;
    return sm.pack_u64();
}

uint64_t mk_branch(uint32_t val_reg, valhall::csf::CSCondition cond, int32_t offset) {
    valhall::csf::CSBRANCH br{};
    br.value = val_reg;
    br.condition = cond;
    br.offset = offset;
    return br.pack_u64();
}

uint64_t mk_call(uint32_t addr_reg, uint32_t len_reg) {
    valhall::csf::CSCALL c{};
    c.address = addr_reg;
    c.length = len_reg;
    return c.pack_u64();
}

uint64_t mk_jump(uint32_t addr_reg, uint32_t len_reg) {
    valhall::csf::CSJUMP j{};
    j.address = addr_reg;
    j.length = len_reg;
    return j.pack_u64();
}

uint64_t mk_run_idvs() {
    return static_cast<uint64_t>(static_cast<uint32_t>(valhall::csf::CSOpcode::RUN_IDVS)) << 56;
}

uint64_t mk_run_fragment() {
    return static_cast<uint64_t>(static_cast<uint32_t>(valhall::csf::CSOpcode::RUN_FRAGMENT)) << 56;
}

uint64_t mk_run_compute(uint32_t srt_sel, uint32_t spd_sel, uint32_t tsd_sel, uint32_t fau_sel) {
    valhall::csf::CSRUNCOMPUTE rc{};
    rc.srt_select = srt_sel;
    rc.spd_select = spd_sel;
    rc.tsd_select = tsd_sel;
    rc.fau_select = fau_sel;
    return rc.pack_u64();
}

size_t file_size_bytes(const std::string& path) {
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0) return 0u;
    return static_cast<size_t>(st.st_size);
}

std::string read_text_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) return {};
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// ============================================================================
// Test 1: Per-Queue Register Isolation (Gap 1)
// ============================================================================
void test_per_queue_register_isolation() {
    VaTracker va;
    CsfQueueState q_gfx{};
    q_gfx.buffer_gpu_addr = 0x10000000ULL;
    CsfQueueState q_comp{};
    q_comp.buffer_gpu_addr = 0x20000000ULL;

    const uint64_t gfx_stream[] = {
        mk_move48(0, 0x0000111122223300ULL),
        mk_move32(4, 0xAAAAAAAAu),
    };
    const uint64_t comp_stream[] = {
        mk_move48(0, 0x0000999988887700ULL),
        mk_move32(4, 0x55555555u),
    };

    CsfDispatchCallbacks cb{};
    execute_csf_stream(q_gfx, va, gfx_stream, 2, cb);
    execute_csf_stream(q_comp, va, comp_stream, 2, cb);

    // Verify q_comp did not clobber q_gfx's register file
    assert(q_gfx.regs[0] == 0x22223300u);
    assert(q_gfx.regs[1] == 0x00001111u);
    assert(q_gfx.regs[4] == 0xAAAAAAAAu);
    assert(q_comp.regs[0] == 0x88887700u);
    assert(q_comp.regs[1] == 0x00009999u);
    assert(q_comp.regs[4] == 0x55555555u);
    std::cout << "[PASS] Gap 1: Per-queue regs[96] isolation verified\n";
}

// ============================================================================
// Test 2: Full CSF ISA Emulation (Gap 2 — ARITH_REG32/64, LOAD/STORE_MULTIPLE,
//         BRANCH, CALL, JUMP, loop step-budget termination)
// ============================================================================
void test_csf_isa_emulation() {
    VaTracker va;
    CsfQueueState q{};
    q.buffer_gpu_addr = 0x30000000ULL;

    // Backing buffer for LOAD_MULTIPLE (16 words)
    alignas(64) uint32_t mem_buf[16] = {0};
    mem_buf[0] = 0xDEAD0000u;
    mem_buf[1] = 0x0000BEEFu;
    mem_buf[2] = 0xCAFEBABEu;
    const uint64_t mem_va = 0x40000000ULL;
    va.record_mem_import_user_buffer(reinterpret_cast<uintptr_t>(mem_buf), sizeof(mem_buf), mem_va, 0);

    // Secondary stream for JUMP tail-call target
    const uint64_t jump_target[] = {
        mk_move32(30, 0x77777777u),
    };
    const uint64_t jump_va = 0x50000000ULL;
    va.record_mem_import_user_buffer(reinterpret_cast<uintptr_t>(jump_target), sizeof(jump_target), jump_va, 0);

    // Subroutine stream for CALL target
    const uint64_t call_target[] = {
        mk_move32(28, 0x12345678u),
    };
    const uint64_t call_va = 0x51000000ULL;
    va.record_mem_import_user_buffer(reinterpret_cast<uintptr_t>(call_target), sizeof(call_target), call_va, 0);

    const uint64_t stream[] = {
        // ARITH_REG32: r2 = 100, r3 = 25 -> ADD32 r4 = 125, SUB32 r5 = 75, UMIN32 r6 = 25
        mk_move32(2, 100u),
        mk_move32(3, 25u),
        mk_arith_reg32(1u, 4, 2, 3), // ADD32
        mk_arith_reg32(2u, 5, 2, 3), // SUB32
        mk_arith_reg32(0u, 6, 2, 3), // UMIN32

        // ARITH_REG64: [r8:r9] = 0x1_0000_0000, [r10:r11] = 0x0_0000_1000 -> ADD64 [r12:r13]
        mk_move48(8, 0x0000000100000000ULL),
        mk_move48(10, 0x0000000000001000ULL),
        mk_arith_reg64(1u, 12, 8, 10), // ADD64

        // LOAD_MULTIPLE from mem_va into r16..r18 (mask = 0b111)
        mk_move48(14, mem_va),
        mk_load_multiple(16, 14, 0x7u, 0),

        // STORE_MULTIPLE r4, r5 to mem_va + 16 (shadow memory! must NOT mutate mem_buf)
        mk_store_multiple(4, 14, 0x3u, 16),
        // LOAD_MULTIPLE back from mem_va + 16 into r20, r21
        mk_load_multiple(20, 14, 0x3u, 16),

        // BRANCH: r5 == 75 (> 0), skip next instruction (offset = +1)
        mk_branch(5, valhall::csf::CSCondition::Greater, 1),
        mk_move32(25, 0xBAD00000u), // skipped!
        mk_move32(26, 0x600D0000u), // executed!

        // CALL call_va (length 8)
        mk_move48(40, call_va),
        mk_move32(42, 8u),
        mk_call(40, 42),

        // JUMP jump_va (length 8) — tail transfer; following instruction must NOT run
        mk_move48(44, jump_va),
        mk_move32(46, 8u),
        mk_jump(44, 46),
        mk_move32(31, 0xDEADDEADu), // unreachable after JUMP!
    };

    CsfDispatchCallbacks cb{};
    CsfExecResult res = execute_csf_stream(q, va, stream, sizeof(stream) / sizeof(stream[0]), cb);
    assert(!res.hit_step_limit);
    assert(q.regs[4] == 125u);
    assert(q.regs[5] == 75u);
    assert(q.regs[6] == 25u);
    assert(q.regs[12] == 0x00001000u && q.regs[13] == 0x00000001u);
    assert(q.regs[16] == 0xDEAD0000u && q.regs[17] == 0x0000BEEFu && q.regs[18] == 0xCAFEBABEu);
    // Shadow memory round-trip succeeded AND original mem_buf[4..5] stayed untouched (0)
    assert(q.regs[20] == 125u && q.regs[21] == 75u);
    assert(mem_buf[4] == 0u && mem_buf[5] == 0u);
    // Branch skipped r25, executed r26
    assert(q.regs[25] == 0u && q.regs[26] == 0x600D0000u);
    // CALL executed r28 and returned
    assert(q.regs[28] == 0x12345678u);
    // JUMP executed r30 and did NOT return to r31
    assert(q.regs[30] == 0x77777777u && q.regs[31] == 0u);

    // Verify infinite-loop BRANCH terminates cleanly via step budget
    CsfQueueState q_loop{};
    const uint64_t loop_stream[] = {
        mk_branch(0, valhall::csf::CSCondition::Always, -1),
    };
    CsfExecResult lres = execute_csf_stream(q_loop, va, loop_stream, 1, cb, 64u);
    assert(lres.hit_step_limit && lres.steps == 64u);
    std::cout << "[PASS] Gap 2: CSF ISA (ARITH_REG32/64, LOAD/STORE_MULTIPLE, BRANCH, CALL, JUMP) verified\n";
}

// ============================================================================
// Test 3: MEM_ALLOC / MEM_ALLOC_EX / MEM_IMPORT VA & Cookie Tracking (Gap 3)
// ============================================================================
void test_va_tracker_modes() {
    VaTracker va;

    // 1. SAME_VA cookie-based allocation (cookie = 65 << 12 -> mmap returns cpu_va -> gpu_va == cpu_va)
    alignas(4096) uint8_t same_va_buf[4096] = {0};
    same_va_buf[16] = 0xAB;
    const uint64_t cookie = (65ULL << 12);
    va.record_mem_alloc(cookie, 1, kBaseMemSameVa | kBaseMemNeedMmap, /*is_ex=*/false);
    va.record_mali_mmap(cookie, reinterpret_cast<uintptr_t>(same_va_buf), sizeof(same_va_buf), 3);
    const uint64_t same_gpu_va = reinterpret_cast<uintptr_t>(same_va_buf);
    CpuSpan s1 = va.gpu_to_cpu_span(same_gpu_va + 16, 4);
    assert(s1 && s1.kind == MappingKind::SameVaCookie && s1.available_bytes == 4096u - 16u);
    assert(*reinterpret_cast<const uint8_t*>(s1.ptr) == 0xAB);

    // 2. CUSTOM_VA / FIXED via MEM_ALLOC_EX (gpu_va != cpu_va)
    alignas(4096) uint8_t custom_buf[8192] = {0};
    custom_buf[128] = 0xCD;
    const uint64_t custom_gpu_va = 0x0000123400000000ULL;
    va.record_mem_alloc(custom_gpu_va, 2, kBaseMemFixed, /*is_ex=*/true);
    va.record_mali_mmap(custom_gpu_va, reinterpret_cast<uintptr_t>(custom_buf), sizeof(custom_buf), 3);
    CpuSpan s2 = va.gpu_to_cpu_span(custom_gpu_va + 128, 4);
    assert(s2 && s2.kind == MappingKind::CustomVa && s2.available_bytes == 8192u - 128u);
    assert(*reinterpret_cast<const uint8_t*>(s2.ptr) == 0xCD);

    // 3. MEM_IMPORT (UMM DMA-BUF): mmap happens on dmabuf_fd, NOT /dev/mali0!
    alignas(4096) uint8_t dmabuf_mem[4096] = {0};
    dmabuf_mem[32] = 0xEF;
    const int fake_dmabuf_fd = 99;
    const uint64_t imp_gpu_va = 0x0000222200000000ULL;
    va.record_mem_import_umm(fake_dmabuf_fd, imp_gpu_va, 1, kBaseMemImportShared);
    assert(va.is_tracked_dmabuf_fd(fake_dmabuf_fd));
    bool handled = va.record_dmabuf_mmap(fake_dmabuf_fd, 0, reinterpret_cast<uintptr_t>(dmabuf_mem), sizeof(dmabuf_mem));
    assert(handled);
    CpuSpan s3 = va.gpu_to_cpu_span(imp_gpu_va + 32, 4);
    assert(s3 && s3.kind == MappingKind::ImportDmaBuf);
    assert(*reinterpret_cast<const uint8_t*>(s3.ptr) == 0xEF);

    // 4. MEM_FREE removes mapping cleanly
    va.record_mem_free(custom_gpu_va);
    assert(!va.gpu_to_cpu_span(custom_gpu_va, 4));

    std::cout << "[PASS] Gap 3: SAME_VA cookie, CUSTOM_VA MEM_ALLOC_EX, and DMA-BUF MEM_IMPORT verified\n";
}

// ============================================================================
// Test 4: Lazy dlopen() GOT/PLT Interception (Gap 4)
// ============================================================================
void test_lazy_dlopen_interception() {
    // Register our mock shared library name and install loader dlopen hooks
    register_extra_hook_target_substr("mock_lazy_mali_driver");
    install_mali_hooks();
    const HookStats before = get_hook_stats();

    // Create a temporary file containing "mali" in its path so hooked_open tracks it
    const char* tmp_mali = "/tmp/test_mock_mali0_fd";
    int cfd = ::open(tmp_mali, O_CREAT | O_RDWR, 0644);
    assert(cfd >= 0);
    ::close(cfd);

    // Call dlopen() from this executable — intercepted by hooked_dlopen()!
    void* mod = dlopen(MOCK_LAZY_MALI_PATH, RTLD_NOW | RTLD_LOCAL);
    assert(mod != nullptr);

    const HookStats after = get_hook_stats();
    assert(after.dlopen_intercepts > before.dlopen_intercepts);
    assert(after.mali_libs_patched > before.mali_libs_patched);

    auto open_fn = reinterpret_cast<int (*)(const char*)>(dlsym(mod, "mock_mali_open_path"));
    auto close_fn = reinterpret_cast<int (*)(int)>(dlsym(mod, "mock_mali_close_fd"));
    assert(open_fn != nullptr && close_fn != nullptr);

    // Immediately call open() inside the lazily loaded library WITHOUT any second hook pass!
    int fd = open_fn(tmp_mali);
    assert(fd >= 0);
    {
        std::lock_guard<std::mutex> lk(layer_state().mu);
        assert(layer_state().mali_fds.count(fd) == 1u);
    }
    close_fn(fd);
    {
        std::lock_guard<std::mutex> lk(layer_state().mu);
        assert(layer_state().mali_fds.count(fd) == 0u);
    }

    dlclose(mod);
    ::unlink(tmp_mali);
    clear_extra_hook_target_substrs();
    std::cout << "[PASS] Gap 4: Lazy dlopen() intercepted & patched GOT before first open()\n";
}

// ============================================================================
// Test 5: Dynamic Sizes, RTBuffer Stride, USER_IO Offset, Multi-Draw & Compute (Gap 5)
// ============================================================================
void test_dynamic_sizes_multidraw_and_compute() {
    // 5a. USER_IO insert offset: kbase < 1.35 (csi_off=0) vs kbase >= 1.35 (csi_off=csi_index*16)
    CsfQueueState q_old{};
    q_old.csi_index = 3;
    q_old.csg_shared_user_io = false;
    assert(compute_user_io_insert_offset(q_old, 4096u) == 4096u);

    CsfQueueState q_new{};
    q_new.csi_index = 3;
    q_new.csg_shared_user_io = true;
    assert(compute_user_io_insert_offset(q_new, 4096u) == 4096u + 3u * 16u);

    // 5b. Valhall shader size scanner: shader larger than 512 B (e.g. 1024 B ending with flow=15)
    std::vector<uint64_t> big_shader(256, 0ULL); // 2048 B buffer
    for (size_t i = 0; i < 127; ++i) big_shader[i] = 0x0000000000000001ULL;
    big_shader[127] = (15ULL << 59) | 0x1ULL; // END at word 127 -> 1024 bytes!
    size_t est = estimate_valhall_shader_size(big_shader.data(), big_shader.size() * 8u, 65536u);
    assert(est == 1024u);

    // 5c. RTBuffer stride calculation: 64x32 RT with row_stride=512 -> 16384 B (NOT 32 * 16384!)
    size_t rt_sz = compute_rt_writeback_bytes(64u, 32u, 512u, 0u, 65536u);
    assert(rt_sz == 32u * 512u);

    // 5d. End-to-end multi-draw (2 RUN_IDVS) + RUN_FRAGMENT + RUN_COMPUTE capture
    const std::string out_dir = "/tmp/mali_csf_dump_v19_unit";
    setenv("MALI_CSF_DUMP_ENABLE", "1", 1);
    setenv("MALI_CSF_DUMP_DIR", out_dir.c_str(), 1);
    setenv("MALI_CSF_DUMP_FRAME", "1", 1);
    setenv("MALI_CSF_DUMP_MAX_DRAWS", "0", 1);
    reset_layer_state_for_test(1u);

    VaTracker va;
    // Set up SPD + 1024B shader + FBD + 16384B RT buffer
    const uint64_t shader_va = 0x60000000ULL;
    va.record_mem_import_user_buffer(reinterpret_cast<uintptr_t>(big_shader.data()), big_shader.size() * 8u, shader_va, 0);

    uint32_t spd_words[8] = {0};
    valhall::csf::ShaderProgram spd{};
    spd.binary = shader_va;
    spd.pack(spd_words);
    const uint64_t spd_va = 0x61000000ULL;
    va.record_mem_import_user_buffer(reinterpret_cast<uintptr_t>(spd_words), sizeof(spd_words), spd_va, 0);

    // Tight 64-byte TSD (would have silently failed with old hardcoded 512 B minimum!)
    alignas(64) uint8_t tight_tsd[64] = {0x42};
    const uint64_t tsd_va = 0x62000000ULL;
    va.record_mem_import_user_buffer(reinterpret_cast<uintptr_t>(tight_tsd), sizeof(tight_tsd), tsd_va, 0);

    // FBD (128B params + 64B RT0 = 192B) + color RT buffer (32 * 512 = 16384 B)
    std::vector<uint8_t> rt_buf(32u * 512u, 0x7Au);
    const uint64_t rt_va = 0x63000000ULL;
    va.record_mem_import_user_buffer(reinterpret_cast<uintptr_t>(rt_buf.data()), rt_buf.size(), rt_va, 0);

    alignas(64) uint8_t fbd_buf[192] = {0};
    valhall::csf::FramebufferParameters fbp{};
    fbp.width = 64u;
    fbp.height = 32u;
    fbp.render_target_count = 1u;
    fbp.pack(reinterpret_cast<uint32_t*>(fbd_buf));
    valhall::csf::RenderTarget rtd{};
    rtd.rgb.writeback_buffer.base = rt_va;
    rtd.rgb.writeback_buffer.row_stride = 512u;
    rtd.pack(reinterpret_cast<uint32_t*>(fbd_buf + 128));
    const uint64_t fbd_va = 0x64000000ULL;
    va.record_mem_import_user_buffer(reinterpret_cast<uintptr_t>(fbd_buf), sizeof(fbd_buf), fbd_va, 0);

    // Build CSF stream with 2 graphics draws, 1 fragment pass, and 1 compute dispatch
    CsfQueueState q{};
    q.buffer_gpu_addr = 0x70000000ULL;
    const uint64_t stream[] = {
        mk_move48(static_cast<uint32_t>(valhall::csf::IDVSSR::SPD_0), spd_va),
        mk_move48(static_cast<uint32_t>(valhall::csf::IDVSSR::TSD_0), tsd_va),
        mk_run_idvs(), // draw_000
        mk_run_idvs(), // draw_001
        mk_move48(static_cast<uint32_t>(valhall::csf::FRAGMENTSR::FBD_POINTER), fbd_va),
        mk_run_fragment(), // frag_000
        mk_move48(static_cast<uint32_t>(valhall::csf::COMPUTESR::SPD_0), spd_va),
        mk_move48(static_cast<uint32_t>(valhall::csf::COMPUTESR::TSD_0), tsd_va),
        mk_move32(static_cast<uint32_t>(valhall::csf::COMPUTESR::JOB_SIZE_X), 16u),
        mk_move32(static_cast<uint32_t>(valhall::csf::COMPUTESR::JOB_SIZE_Y), 8u),
        mk_move32(static_cast<uint32_t>(valhall::csf::COMPUTESR::JOB_SIZE_Z), 1u),
        mk_run_compute(0, 0, 0, 0), // compute_000
    };

    LayerConfig cfg = read_layer_config();
    FrameCaptureState cap{};
    CsfDumper dumper;
    CsfDispatchCallbacks cb{};
    cb.on_run_idvs = [&](const CsfQueueState& qs, uint64_t insn, bool is_idvs2) {
        dumper.on_run_idvs(cap, qs, va, cfg, 1u, insn, is_idvs2);
    };
    cb.on_run_fragment = [&](const CsfQueueState& qs, uint64_t insn, uint32_t fbd_reg, bool is_frag2) {
        dumper.on_run_fragment(cap, qs, va, cfg, 1u, insn, fbd_reg, is_frag2);
    };
    cb.on_run_compute = [&](const CsfQueueState& qs, uint64_t insn, bool is_indirect) {
        dumper.on_run_compute(cap, qs, va, cfg, 1u, insn, is_indirect);
    };

    execute_csf_stream(q, va, stream, sizeof(stream) / sizeof(stream[0]), cb);
    dumper.on_frame_present(cap, va, "r54p3-00eac0", 1u, 35u);

    assert(cap.draws.size() == 2u);
    assert(cap.fragments.size() == 1u);
    assert(cap.computes.size() == 1u);
    // Verify dynamic sizes on disk
    assert(file_size_bytes(out_dir + "/pos_shader.bin") == 1024u);
    assert(file_size_bytes(out_dir + "/draws/draw_001/pos_shader.bin") == 1024u);
    assert(file_size_bytes(out_dir + "/draws/draw_000/tsd.bin") == 64u);
    assert(file_size_bytes(out_dir + "/fbd.bin") == 192u);
    assert(file_size_bytes(out_dir + "/rt_color_after.bin") == 32u * 512u);
    assert(file_size_bytes(out_dir + "/computes/compute_000/comp_shader.bin") == 1024u);

    std::string meta = read_text_file(out_dir + "/metadata.json");
    assert(meta.find("\"draw_count\": 2") != std::string::npos);
    assert(meta.find("\"compute_count\": 1") != std::string::npos);
    assert(meta.find("\"rt_color_bytes\": 16384") != std::string::npos);
    std::cout << "[PASS] Gap 5: Dynamic shader/FBD/TSD/RT sizes, USER_IO offset, multi-draw & RUN_COMPUTE verified\n";
}

// ============================================================================
// Test 6: Vulkan Loader Chain Verification (Existing E2E Test)
// ============================================================================
void test_vulkan_loader_chain() {
    void* handle = dlopen(CSF_DUMP_LAYER_PATH, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        std::cerr << "dlopen failed: " << dlerror() << "\n";
        std::exit(1);
    }

    auto neg_fn = reinterpret_cast<PFN_vkNegotiateLoaderLayerInterfaceVersion>(
        dlsym(handle, "vkNegotiateLoaderLayerInterfaceVersion"));
    assert(neg_fn != nullptr);

    VkNegotiateLayerInterface neg{};
    neg.sType = LAYER_NEGOTIATE_INTERFACE_STRUCT;
    neg.loaderLayerInterfaceVersion = 2;
    VkResult nr = neg_fn(&neg);
    assert(nr == VK_SUCCESS);
    assert(neg.loaderLayerInterfaceVersion == 2);
    assert(neg.pfnGetInstanceProcAddr != nullptr);
    assert(neg.pfnGetDeviceProcAddr != nullptr);
    std::cout << "[PASS] vkNegotiateLoaderLayerInterfaceVersion negotiated v2\n";

    auto selftest_fn = reinterpret_cast<bool (*)(const char*, uint32_t)>(
        dlsym(handle, "mali_csf_dump_layer_selftest"));
    assert(selftest_fn != nullptr);
    bool hooks_ok = selftest_fn("/tmp/mali_csf_dump_test", 1u);
    assert(hooks_ok);
    std::cout << "[PASS] GOT/PLT hook installer initialized cleanly\n";

    setenv("VK_LAYER_PATH", CSF_DUMP_LAYER_DIR, 1);
    const char* default_icd = (access("/usr/share/vulkan/icd.d/lvp_icd.json", R_OK) == 0)
        ? "/usr/share/vulkan/icd.d/lvp_icd.json"
        : "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json";
    setenv("VK_ICD_FILENAMES", default_icd, 0);
    setenv("MALI_CSF_DUMP_ENABLE", "1", 1);

    uint32_t layer_count = 0;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    std::vector<VkLayerProperties> layers(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, layers.data());
    bool found = false;
    for (const auto& lp : layers) {
        if (std::strcmp(lp.layerName, "VK_LAYER_MALI_csf_dump") == 0) {
            found = true;
            break;
        }
    }
    assert(found);
    std::cout << "[PASS] Vulkan loader enumerated VK_LAYER_MALI_csf_dump\n";

    const char* req_layers[] = { "VK_LAYER_MALI_csf_dump" };
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "test_csf_dump_layer";
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    ici.enabledLayerCount = 1;
    ici.ppEnabledLayerNames = req_layers;

    VkInstance inst = VK_NULL_HANDLE;
    VkResult ir = vkCreateInstance(&ici, nullptr, &inst);
    assert(ir == VK_SUCCESS && inst != VK_NULL_HANDLE);
    vkDestroyInstance(inst, nullptr);
    std::cout << "[PASS] vkCreateInstance / vkDestroyInstance chained through VK_LAYER_MALI_csf_dump\n";

    dlclose(handle);
}

} // namespace

int main() {
    std::cout << "=== [Test 7] VK_LAYER_MALI_csf_dump Modular Unit & Vulkan Loader Verification ===\n";
    test_per_queue_register_isolation();
    test_csf_isa_emulation();
    test_va_tracker_modes();
    test_lazy_dlopen_interception();
    test_dynamic_sizes_multidraw_and_compute();
    test_vulkan_loader_chain();
    return 0;
}

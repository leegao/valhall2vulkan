#pragma once
// ============================================================================
// csf_interpreter.h
// Per-queue Mali CSF (v10–v14) command stream interpreter adapted from
// panthor2kbase's lifted_csffw.c, with full ARITH_REG32/64, ARITH_IMM32/64,
// LOGIC_OP32, LOAD_MULTIPLE, STORE_MULTIPLE (non-destructive shadow memory),
// BRANCH, CALL, JUMP, and RUN_IDVS / RUN_IDVS2 / RUN_FRAGMENT / RUN_FRAGMENT2 /
// RUN_COMPUTE / RUN_COMPUTE_INDIRECT dispatch hooks.
// ============================================================================

#include "csf_va_tracker.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>

namespace mali_csf_dump {

constexpr uint32_t kCsfMaxRegs = 96u;
constexpr uint32_t kCsfMaxCallDepth = 8u;
constexpr uint32_t kCsfMaxStepsPerKick = 65536u;

// Per-queue CSF execution state (isolates register files across concurrent queues)
struct CsfQueueState {
    uint64_t buffer_gpu_addr = 0;
    uint32_t buffer_size = 0;
    uint64_t mmap_handle = 0;
    uintptr_t user_io_cpu = 0;
    uint64_t last_extract = 0;
    uint8_t group_handle = 0;
    uint8_t csi_index = 0;
    bool csg_shared_user_io = false; // true on kbase >= 1.35 or shared mmap_handle

    // Per-queue 96x32-bit CSHWIF register file & iterator state
    uint32_t regs[kCsfMaxRegs] = {0};
    uint64_t iter_state64 = 0;

    // Non-destructive shadow memory overlay for STORE_MULTIPLE -> LOAD_MULTIPLE
    // (keyed by 48-bit aligned GPU VA, storing 32-bit words).
    std::unordered_map<uint64_t, uint32_t> shadow_mem;

    // Diagnostics & execution counters
    uint64_t insns_executed = 0;
    uint64_t calls_executed = 0;
    uint64_t jumps_executed = 0;
    uint64_t branches_taken = 0;
    uint64_t unknown_opcodes = 0;
    uint64_t fault_count = 0;
};

struct CsfDispatchCallbacks {
    std::function<void(const CsfQueueState& q, uint64_t insn, bool is_idvs2)> on_run_idvs;
    std::function<void(const CsfQueueState& q, uint64_t insn, uint32_t fbd_reg_idx, bool is_frag2)> on_run_fragment;
    std::function<void(const CsfQueueState& q, uint64_t insn, bool is_indirect)> on_run_compute;
};

struct CsfExecResult {
    uint32_t steps = 0;
    bool hit_step_limit = false;
    bool faulted = false;
};

// Compute the byte offset of CS_INSERT inside the queue's USER_IO mapping.
size_t compute_user_io_insert_offset(const CsfQueueState& q, size_t page_size);

// Execute a contiguous CSF command stream slice starting at `stream_va`
// (mapped in `va`) or directly from `(words, count)` if `stream_va == 0`.
CsfExecResult execute_csf_stream(CsfQueueState& q,
                                 const VaTracker& va,
                                 const uint64_t* words,
                                 size_t count,
                                 const CsfDispatchCallbacks& cb,
                                 uint32_t max_steps = kCsfMaxStepsPerKick);

// Advance a bound queue from `q.last_extract` up to `insert_off` (handling ring wrap).
CsfExecResult process_queue_ring(CsfQueueState& q,
                                 const VaTracker& va,
                                 uint64_t insert_off,
                                 const CsfDispatchCallbacks& cb,
                                 uint32_t max_steps = kCsfMaxStepsPerKick);

} // namespace mali_csf_dump

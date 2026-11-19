#pragma once
// ============================================================================
// csf_layer_state.h
// Central LayerState coordinating VaTracker, per-queue CsfQueueState maps,
// CsfDumper, and original libc function pointers.
// ============================================================================

#include "csf_dumper.h"
#include "csf_interpreter.h"
#include "csf_va_tracker.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <sys/types.h>
#include <unordered_map>
#include <unordered_set>

namespace mali_csf_dump {

struct LayerState {
    std::mutex mu;
    std::unordered_set<int> mali_fds;
    VaTracker va;
    std::unordered_map<uint64_t, CsfQueueState> queues_by_gpu_va;
    // Maps mmap_handle -> list of queue buffer_gpu_addrs bound to that handle
    // (1 queue per handle on kbase < 1.35; >1 queues per handle on kbase >= 1.35 CSG-shared USER_IO)
    std::unordered_map<uint64_t, std::vector<uint64_t>> queues_by_mmap_handle;

    std::string detected_abi = "unknown";
    uint16_t kbase_ver_major = 0;
    uint16_t kbase_ver_minor = 0;

    std::atomic<uint32_t> current_frame{1};
    std::atomic<bool> dump_done{false};
    FrameCaptureState capture{};
    CsfDumper dumper{};
    bool hooks_installed = false;

    // Original libc / libdl functions
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
    void* (*orig_dlopen)(const char*, int) = nullptr;
    void* (*orig_android_dlopen_ext)(const char*, int, const void*) = nullptr;

    bool is_kbase_1_35_or_newer() const {
        return (kbase_ver_major > 1u) || (kbase_ver_major == 1u && kbase_ver_minor >= 35u);
    }
};

LayerState& layer_state();

// Handle a CS_QUEUE_KICK ioctl while holding `s.mu`
void process_queue_kick_locked(LayerState& s, uint64_t buffer_gpu_addr);

// Handle a QueuePresentKHR frame boundary while holding `s.mu`
void process_frame_present_locked(LayerState& s);

// Reset capture & queue/VA state for selftests
void reset_layer_state_for_test(uint32_t target_frame);

} // namespace mali_csf_dump

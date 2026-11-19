#pragma once
// ============================================================================
// csf_va_tracker.h
// Tracks Mali GPU VA -> CPU VA mappings across SAME_VA (direct & cookie-based),
// CUSTOM_VA / FIXED, MEM_ALLOC, MEM_ALLOC_EX, and MEM_IMPORT (DMA-BUF UMM &
// USER_BUFFER).
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace mali_csf_dump {

enum class MappingKind : uint8_t {
    SameVaDirect = 0,
    SameVaCookie = 1,
    CustomVa     = 2,
    ImportDmaBuf = 3,
    ImportUserBuf = 4,
};

struct GpuMapping {
    uint64_t gpu_va = 0;
    uintptr_t cpu_va = 0;
    size_t size = 0;
    uint64_t flags = 0;
    int backing_fd = -1;
    MappingKind kind = MappingKind::SameVaDirect;
};

struct PendingAlloc {
    uint64_t handle_or_va = 0; // cookie (for SAME_VA) or real gpu_va (for CUSTOM_VA)
    size_t va_size = 0;
    uint64_t flags = 0;
    bool is_same_va = false;
    bool is_cookie = false;
};

struct PendingDmaBufImport {
    int dmabuf_fd = -1;
    uint64_t gpu_va = 0;
    size_t va_size = 0;
    uint64_t flags = 0;
};

struct CpuSpan {
    const void* ptr = nullptr;
    size_t available_bytes = 0;
    MappingKind kind = MappingKind::SameVaDirect;

    explicit operator bool() const { return ptr != nullptr && available_bytes > 0; }
};

struct VaTrackerStats {
    uint64_t mem_alloc_count = 0;
    uint64_t mem_alloc_ex_count = 0;
    uint64_t mem_import_count = 0;
    uint64_t mem_free_count = 0;
    uint64_t mmap_mali_count = 0;
    uint64_t mmap_dmabuf_count = 0;
    uint64_t lookup_hits = 0;
    uint64_t lookup_misses = 0;
};

class VaTracker {
public:
    void clear();

    // Record a completed KBASE_IOCTL_MEM_ALLOC or MEM_ALLOC_EX
    void record_mem_alloc(uint64_t out_gpu_va_or_cookie, uint64_t va_pages, uint64_t out_flags, bool is_ex);

    // Record a completed KBASE_IOCTL_MEM_IMPORT (UMM DMA-BUF or USER_BUFFER)
    void record_mem_import_umm(int dmabuf_fd, uint64_t out_gpu_va, uint64_t va_pages, uint64_t out_flags);
    void record_mem_import_user_buffer(uintptr_t cpu_ptr, size_t length, uint64_t out_gpu_va, uint64_t out_flags);

    // Record mmap() on /dev/mali0 fd (excluding queue user_io handles)
    void record_mali_mmap(uint64_t offset, uintptr_t cpu_va, size_t length, int mali_fd);

    // Record mmap() on a tracked DMA-BUF fd; returns true if fd was a tracked import
    bool record_dmabuf_mmap(int fd, uint64_t offset, uintptr_t cpu_va, size_t length);

    // Record munmap()
    void record_munmap(uintptr_t cpu_va, size_t length);

    // Record KBASE_IOCTL_MEM_FREE
    void record_mem_free(uint64_t gpu_va);

    // Record close() on a possibly tracked DMA-BUF fd
    void record_fd_close(int fd);

    bool is_tracked_dmabuf_fd(int fd) const;

    // Resolve a GPU VA to a CPU pointer + remaining bytes in that mapping.
    // Returns empty CpuSpan if unmapped or fewer than min_bytes remain.
    CpuSpan gpu_to_cpu_span(uint64_t gpu_va, size_t min_bytes = 4) const;

    // Convenience wrapper matching legacy signature
    const void* gpu_to_cpu(uint64_t gpu_va, size_t min_bytes = 4) const;

    size_t mapping_count() const { return mappings_.size(); }
    const VaTrackerStats& stats() const { return stats_; }

private:
    static uint64_t mask48(uint64_t va) { return va & 0x0000FFFFFFFFFFFFULL; }
    void upsert_mapping(const GpuMapping& m);

    std::vector<GpuMapping> mappings_;
    // Keyed by cookie (for SAME_VA) or masked gpu_va (for CUSTOM_VA)
    std::unordered_map<uint64_t, PendingAlloc> pending_allocs_;
    // Keyed by dmabuf_fd
    std::unordered_map<int, PendingDmaBufImport> dmabuf_imports_;
    mutable VaTrackerStats stats_{};
};

} // namespace mali_csf_dump

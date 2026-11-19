#include "csf_va_tracker.h"
#include "csf_config_log.h"
#include "csf_kbase_uapi.h"

#include <algorithm>
#include <cinttypes>

namespace mali_csf_dump {

void VaTracker::clear() {
    mappings_.clear();
    pending_allocs_.clear();
    dmabuf_imports_.clear();
    stats_ = VaTrackerStats{};
}

void VaTracker::upsert_mapping(const GpuMapping& m) {
    const uint64_t g48 = mask48(m.gpu_va);
    if (g48 == 0 || m.cpu_va == 0 || m.size == 0) return;

    // Remove any exact duplicate or overlapping stale entry at the same GPU or CPU range
    mappings_.erase(
        std::remove_if(mappings_.begin(), mappings_.end(),
                       [g48, &m](const GpuMapping& existing) {
                           const uint64_t eg48 = mask48(existing.gpu_va);
                           const bool same_gpu = (eg48 == g48 && existing.size == m.size);
                           const bool same_cpu = (existing.cpu_va == m.cpu_va && existing.size == m.size);
                           return same_gpu || same_cpu;
                       }),
        mappings_.end());
    mappings_.push_back(m);
}

void VaTracker::record_mem_alloc(uint64_t out_gpu_va_or_cookie,
                                 uint64_t va_pages,
                                 uint64_t out_flags,
                                 bool is_ex) {
    if (is_ex) stats_.mem_alloc_ex_count++;
    else       stats_.mem_alloc_count++;

    const uint64_t key = mask48(out_gpu_va_or_cookie);
    if (key == 0) return;

    PendingAlloc pa{};
    pa.handle_or_va = key;
    pa.va_size = static_cast<size_t>(va_pages * kKbasePageSize);
    pa.flags = out_flags;
    pa.is_same_va = (out_flags & kBaseMemSameVa) != 0;
    pa.is_cookie = is_kbase_cookie(key) || ((out_flags & kBaseMemNeedMmap) != 0 && pa.is_same_va);

    pending_allocs_[key] = pa;
    csf_log(LogLevel::Debug, "va",
            "MEM_ALLOC%s: handle=0x%" PRIx64 " pages=%" PRIu64 " flags=0x%" PRIx64 " same_va=%d cookie=%d",
            is_ex ? "_EX" : "", key, va_pages, out_flags,
            pa.is_same_va ? 1 : 0, pa.is_cookie ? 1 : 0);
}

void VaTracker::record_mem_import_umm(int dmabuf_fd,
                                      uint64_t out_gpu_va,
                                      uint64_t va_pages,
                                      uint64_t out_flags) {
    stats_.mem_import_count++;
    const uint64_t g48 = mask48(out_gpu_va);
    if (dmabuf_fd < 0 || g48 == 0) return;

    const size_t sz = static_cast<size_t>(va_pages * kKbasePageSize);
    dmabuf_imports_[dmabuf_fd] = PendingDmaBufImport{ dmabuf_fd, g48, sz, out_flags };

    // Also register in pending_allocs_ in case userspace mmaps via /dev/mali0 with offset=gpu_va/cookie
    PendingAlloc pa{};
    pa.handle_or_va = g48;
    pa.va_size = sz;
    pa.flags = out_flags;
    pa.is_same_va = (out_flags & kBaseMemSameVa) != 0;
    pa.is_cookie = is_kbase_cookie(g48);
    pending_allocs_[g48] = pa;

    csf_log(LogLevel::Debug, "va",
            "MEM_IMPORT(UMM): dmabuf_fd=%d gpu_va=0x%" PRIx64 " pages=%" PRIu64 " flags=0x%" PRIx64,
            dmabuf_fd, g48, va_pages, out_flags);
}

void VaTracker::record_mem_import_user_buffer(uintptr_t cpu_ptr,
                                              size_t length,
                                              uint64_t out_gpu_va,
                                              uint64_t out_flags) {
    stats_.mem_import_count++;
    const uint64_t g48 = mask48(out_gpu_va);
    if (cpu_ptr == 0 || length == 0 || g48 == 0) return;

    GpuMapping m{};
    m.gpu_va = g48;
    m.cpu_va = cpu_ptr;
    m.size = length;
    m.flags = out_flags;
    m.backing_fd = -1;
    m.kind = MappingKind::ImportUserBuf;
    upsert_mapping(m);

    csf_log(LogLevel::Debug, "va",
            "MEM_IMPORT(USER_BUFFER): gpu_va=0x%" PRIx64 " -> cpu_va=0x%" PRIxPTR " len=%zu",
            g48, cpu_ptr, length);
}

void VaTracker::record_mali_mmap(uint64_t offset, uintptr_t cpu_va, size_t length, int mali_fd) {
    stats_.mmap_mali_count++;
    if (cpu_va == 0 || length == 0) return;

    // Ignore special tracking handle (BASE_MEM_MAP_TRACKING_HANDLE = 3 << 12)
    if (offset == kBaseMemMapTrackingHandle || offset == kBaseMemInvalidHandle) {
        return;
    }

    const uint64_t off48 = mask48(offset);
    GpuMapping m{};
    m.cpu_va = cpu_va;
    m.size = length;
    m.backing_fd = mali_fd;

    auto pit = pending_allocs_.find(off48);
    if (pit != pending_allocs_.end()) {
        const PendingAlloc pa = pit->second;
        m.flags = pa.flags;
        if (pa.is_same_va && pa.is_cookie) {
            // For SAME_VA cookie-based mmap, kbase assigns gpu_va == cpu_va!
            m.gpu_va = mask48(static_cast<uint64_t>(cpu_va));
            m.kind = MappingKind::SameVaCookie;
        } else {
            // CUSTOM_VA / FIXED / non-cookie SAME_VA: gpu_va == handle_or_va (== offset)
            m.gpu_va = pa.handle_or_va;
            m.kind = pa.is_same_va ? MappingKind::SameVaDirect : MappingKind::CustomVa;
        }
        pending_allocs_.erase(pit);
    } else if (is_kbase_cookie(off48)) {
        // Fallback if MEM_ALLOC happened before hooks were installed: a cookie offset
        // in [64<<12, 128<<12) on /dev/mali0 is always a SAME_VA cookie -> gpu_va = cpu_va.
        m.gpu_va = mask48(static_cast<uint64_t>(cpu_va));
        m.kind = MappingKind::SameVaCookie;
        csf_log(LogLevel::Debug, "va",
                "mali mmap with untracked cookie 0x%" PRIx64 " -> inferred SAME_VA gpu_va=0x%" PRIx64,
                off48, m.gpu_va);
    } else {
        // Direct offset == gpu_va (SAME_VA direct or CUSTOM_VA mapped by real GPU VA)
        m.gpu_va = off48;
        m.kind = (off48 == mask48(static_cast<uint64_t>(cpu_va)))
                     ? MappingKind::SameVaDirect
                     : MappingKind::CustomVa;
    }

    upsert_mapping(m);
    csf_log(LogLevel::Debug, "va",
            "mali mmap: off=0x%" PRIx64 " -> gpu_va=0x%" PRIx64 " cpu_va=0x%" PRIxPTR " len=%zu kind=%u",
            off48, m.gpu_va, m.cpu_va, m.size, static_cast<unsigned>(m.kind));
}

bool VaTracker::record_dmabuf_mmap(int fd, uint64_t offset, uintptr_t cpu_va, size_t length) {
    auto it = dmabuf_imports_.find(fd);
    if (it == dmabuf_imports_.end()) return false;

    stats_.mmap_dmabuf_count++;
    const PendingDmaBufImport& imp = it->second;
    GpuMapping m{};
    m.gpu_va = mask48(imp.gpu_va + offset);
    m.cpu_va = cpu_va;
    m.size = length;
    m.flags = imp.flags;
    m.backing_fd = fd;
    m.kind = MappingKind::ImportDmaBuf;
    upsert_mapping(m);

    csf_log(LogLevel::Debug, "va",
            "dmabuf mmap: fd=%d off=0x%" PRIx64 " -> gpu_va=0x%" PRIx64 " cpu_va=0x%" PRIxPTR " len=%zu",
            fd, offset, m.gpu_va, m.cpu_va, m.size);
    return true;
}

void VaTracker::record_munmap(uintptr_t cpu_va, size_t length) {
    if (cpu_va == 0) return;
    const uintptr_t end = cpu_va + length;
    mappings_.erase(
        std::remove_if(mappings_.begin(), mappings_.end(),
                       [cpu_va, end, length](const GpuMapping& m) {
                           if (m.cpu_va == cpu_va) return true;
                           if (length > 0 && m.cpu_va >= cpu_va && (m.cpu_va + m.size) <= end) {
                               return true;
                           }
                           return false;
                       }),
        mappings_.end());
}

void VaTracker::record_mem_free(uint64_t gpu_va) {
    stats_.mem_free_count++;
    const uint64_t g48 = mask48(gpu_va);
    if (g48 == 0) return;

    pending_allocs_.erase(g48);
    mappings_.erase(
        std::remove_if(mappings_.begin(), mappings_.end(),
                       [g48](const GpuMapping& m) { return mask48(m.gpu_va) == g48; }),
        mappings_.end());
}

void VaTracker::record_fd_close(int fd) {
    dmabuf_imports_.erase(fd);
}

bool VaTracker::is_tracked_dmabuf_fd(int fd) const {
    return dmabuf_imports_.find(fd) != dmabuf_imports_.end();
}

CpuSpan VaTracker::gpu_to_cpu_span(uint64_t gpu_va, size_t min_bytes) const {
    const uint64_t va48 = mask48(gpu_va);
    if (va48 == 0 || min_bytes == 0) {
        stats_.lookup_misses++;
        return {};
    }
    for (const auto& m : mappings_) {
        const uint64_t m48 = mask48(m.gpu_va);
        if (va48 >= m48 && (va48 - m48) < m.size) {
            const size_t offset = static_cast<size_t>(va48 - m48);
            const size_t avail = m.size - offset;
            if (avail >= min_bytes) {
                stats_.lookup_hits++;
                return CpuSpan{
                    reinterpret_cast<const void*>(m.cpu_va + offset),
                    avail,
                    m.kind,
                };
            }
        }
    }
    stats_.lookup_misses++;
    return {};
}

const void* VaTracker::gpu_to_cpu(uint64_t gpu_va, size_t min_bytes) const {
    CpuSpan span = gpu_to_cpu_span(gpu_va, min_bytes);
    return span.ptr;
}

} // namespace mali_csf_dump

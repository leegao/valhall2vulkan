#pragma once
// ============================================================================
// csf_kbase_uapi.h
// Mali /dev/mali0 (kbase CSF) UAPI definitions across r38p1, r44p1, r49p1,
// r54p1, r54p3, and MediaTek vendor variants.
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <sys/ioctl.h>

namespace mali_csf_dump {

constexpr uint8_t kKbaseIoctlType = 0x80;
constexpr uint64_t kKbasePageShift = 12ULL;
constexpr uint64_t kKbasePageSize = 4096ULL;

// ---- BASE_MEM_* flags (from mali_kbase_uapi.h) ----
constexpr uint64_t kBaseMemProtCpuRd       = (1ULL << 0);
constexpr uint64_t kBaseMemProtCpuWr       = (1ULL << 1);
constexpr uint64_t kBaseMemProtGpuRd       = (1ULL << 2);
constexpr uint64_t kBaseMemProtGpuWr       = (1ULL << 3);
constexpr uint64_t kBaseMemProtGpuEx       = (1ULL << 4);
constexpr uint64_t kBaseMemFixed           = (1ULL << 8);
constexpr uint64_t kBaseMemGrowOnGpf       = (1ULL << 9);
constexpr uint64_t kBaseMemCoherentSystem  = (1ULL << 10);
constexpr uint64_t kBaseMemCoherentLocal   = (1ULL << 11);
constexpr uint64_t kBaseMemCachedCpu       = (1ULL << 12);
constexpr uint64_t kBaseMemSameVa          = (1ULL << 13);
constexpr uint64_t kBaseMemNeedMmap        = (1ULL << 14);
constexpr uint64_t kBaseMemImportShared    = (1ULL << 18);
constexpr uint64_t kBaseMemCsfEvent        = (1ULL << 19);
constexpr uint64_t kBaseMemFlagMapFixed    = (1ULL << 27);
constexpr uint64_t kBaseMemFixable         = (1ULL << 29);

// ---- Special kbase mmap handles / cookies (in bytes, page-aligned) ----
constexpr uint64_t kBaseMemInvalidHandle      = (0ULL << kKbasePageShift);
constexpr uint64_t kBaseMemMapTrackingHandle  = (3ULL << kKbasePageShift);
constexpr uint64_t kBaseMemCookieBase         = (64ULL << kKbasePageShift);
constexpr uint64_t kBaseMemFirstFreeAddress   = ((64ULL + 64ULL) << kKbasePageShift);

inline bool is_kbase_cookie(uint64_t offset_or_va) {
    return offset_or_va >= kBaseMemCookieBase && offset_or_va < kBaseMemFirstFreeAddress;
}

// ---- MEM_IMPORT types ----
constexpr uint32_t kBaseMemImportTypeInvalid    = 0u;
constexpr uint32_t kBaseMemImportTypeUmp        = 1u;
constexpr uint32_t kBaseMemImportTypeUmm        = 2u; // phandle -> int* dmabuf_fd
constexpr uint32_t kBaseMemImportTypeUserBuffer = 3u; // phandle -> base_mem_import_user_buffer*

struct base_mem_import_user_buffer {
    uint64_t ptr;
    uint64_t length;
};

// ---- Kbase ioctl structs ----
struct kbase_ioctl_version_check {
    uint16_t major;
    uint16_t minor;
};

struct kbase_ioctl_get_ddk_version {
    uint64_t version_buffer;
    uint32_t size;
    uint32_t padding;
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

union kbase_ioctl_mem_alloc_ex {
    struct {
        uint64_t va_pages;
        uint64_t commit_pages;
        uint64_t extension;
        uint64_t flags;
        uint64_t fixed_address;
        uint64_t extra[3];
    } in;
    struct {
        uint64_t flags;
        uint64_t gpu_va;
    } out;
};

// Legacy / MTK 32-byte MEM_ALLOC_EX (nr=50, 0xc0208032) shares the same
// first 32 bytes (in: va_pages, commit_pages, extension, flags; out: flags, gpu_va).
using kbase_ioctl_mem_alloc_ex_legacy = kbase_ioctl_mem_alloc;

struct kbase_ioctl_mem_free {
    uint64_t gpu_addr;
};

union kbase_ioctl_mem_import {
    struct {
        uint64_t flags;
        uint64_t phandle;
        uint32_t type;
        uint32_t padding;
    } in;
    struct {
        uint64_t flags;
        uint64_t gpu_va;
        uint64_t va_pages;
    } out;
};

// MediaTek 40-byte MEM_IMPORT variant (with 16-byte SLC tail) shares the exact
// same prefix for both `in` (24 bytes) and `out` (24 bytes).
union kbase_ioctl_mem_import_mtk {
    struct {
        uint64_t flags;
        uint64_t phandle;
        uint32_t type;
        uint32_t padding;
        uint64_t slc_extra[2];
    } in;
    struct {
        uint64_t flags;
        uint64_t gpu_va;
        uint64_t va_pages;
    } out;
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

struct kbase_ioctl_cs_queue_terminate {
    uint64_t buffer_gpu_addr;
};

// ---- Kbase ioctl command numbers (nr) ----
constexpr unsigned int kIoctlNrMemAlloc         = 5u;
constexpr unsigned int kIoctlNrMemFree          = 7u;
constexpr unsigned int kIoctlNrGetDdkVersion    = 13u;
constexpr unsigned int kIoctlNrMemImport        = 22u;
constexpr unsigned int kIoctlNrCsQueueRegister  = 36u;
constexpr unsigned int kIoctlNrCsQueueKick      = 37u;
constexpr unsigned int kIoctlNrCsQueueBind      = 39u;
constexpr unsigned int kIoctlNrCsQueueRegisterEx = 40u;
constexpr unsigned int kIoctlNrCsQueueTerminate = 41u;
constexpr unsigned int kIoctlNrMemAllocExLegacy = 50u;
constexpr unsigned int kIoctlNrVersionCheckCsf  = 52u;
constexpr unsigned int kIoctlNrMemAllocEx       = 59u;

// Concrete ioctl request codes for tests and direct comparisons
#define MALI_IOCTL_MEM_ALLOC             _IOWR(mali_csf_dump::kKbaseIoctlType, 5,  mali_csf_dump::kbase_ioctl_mem_alloc)
#define MALI_IOCTL_MEM_FREE              _IOW (mali_csf_dump::kKbaseIoctlType, 7,  mali_csf_dump::kbase_ioctl_mem_free)
#define MALI_IOCTL_GET_DDK_VERSION       _IOW (mali_csf_dump::kKbaseIoctlType, 13, mali_csf_dump::kbase_ioctl_get_ddk_version)
#define MALI_IOCTL_MEM_IMPORT            _IOWR(mali_csf_dump::kKbaseIoctlType, 22, mali_csf_dump::kbase_ioctl_mem_import)
#define MALI_IOCTL_MEM_IMPORT_MTK        _IOWR(mali_csf_dump::kKbaseIoctlType, 22, mali_csf_dump::kbase_ioctl_mem_import_mtk)
#define MALI_IOCTL_CS_QUEUE_REGISTER     _IOW (mali_csf_dump::kKbaseIoctlType, 36, mali_csf_dump::kbase_ioctl_cs_queue_register)
#define MALI_IOCTL_CS_QUEUE_KICK         _IOW (mali_csf_dump::kKbaseIoctlType, 37, mali_csf_dump::kbase_ioctl_cs_queue_kick)
#define MALI_IOCTL_CS_QUEUE_BIND         _IOWR(mali_csf_dump::kKbaseIoctlType, 39, mali_csf_dump::kbase_ioctl_cs_queue_bind)
#define MALI_IOCTL_CS_QUEUE_REGISTER_EX  _IOW (mali_csf_dump::kKbaseIoctlType, 40, mali_csf_dump::kbase_ioctl_cs_queue_register_ex)
#define MALI_IOCTL_CS_QUEUE_TERMINATE    _IOW (mali_csf_dump::kKbaseIoctlType, 41, mali_csf_dump::kbase_ioctl_cs_queue_terminate)
#define MALI_IOCTL_MEM_ALLOC_EX_LEGACY   _IOWR(mali_csf_dump::kKbaseIoctlType, 50, mali_csf_dump::kbase_ioctl_mem_alloc_ex_legacy)
#define MALI_IOCTL_VERSION_CHECK_CSF     _IOWR(mali_csf_dump::kKbaseIoctlType, 52, mali_csf_dump::kbase_ioctl_version_check)
#define MALI_IOCTL_MEM_ALLOC_EX          _IOWR(mali_csf_dump::kKbaseIoctlType, 59, mali_csf_dump::kbase_ioctl_mem_alloc_ex)

// ---- CSF USER_IO page layout ----
// Each bound queue (or CSG in kbase >= 1.35) maps 3 pages:
//   Page 0 (+0 * page_sz): Hardware doorbell page (WO)
//   Page 1 (+1 * page_sz): User Input page  (CS_INSERT at +csi_off)
//   Page 2 (+2 * page_sz): User Output page (CS_EXTRACT at +csi_off, CS_ACTIVE at +csi_off + 8)
// For per-queue USER_IO (kbase < 1.35), csi_off == 0 regardless of csi_index.
// For CSG-shared USER_IO (kbase >= 1.35), csi_off == csi_index * 16 (lsl #4).
constexpr size_t kCsUserIoCsiStride135 = 16u;

} // namespace mali_csf_dump

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "csf_elf_hooks.h"
#include "csf_config_log.h"
#include "csf_kbase_uapi.h"
#include "csf_layer_state.h"

#include <algorithm>
#include <cinttypes>
#include <cstdarg>
#include <cstring>
#include <dlfcn.h>
#include <elf.h>
#include <fcntl.h>
#include <link.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <unordered_set>
#include <vector>

namespace mali_csf_dump {

namespace {

struct HookGlobals {
    std::mutex mu;
    std::unordered_set<uintptr_t> patched_mali_bases;
    std::unordered_set<uintptr_t> patched_loader_bases;
    std::vector<std::string> extra_target_substrs;
    HookStats stats{};
};

HookGlobals& hook_globals() {
    static HookGlobals g;
    return g;
}

bool is_mali_path(const char* path) {
    return path && std::strstr(path, "mali") != nullptr;
}

bool is_mali_library_name(const char* name) {
    if (!name || name[0] == '\0') return false;
    if (std::strstr(name, "libGLES_mali") != nullptr) return true;
    for (const auto& sub : hook_globals().extra_target_substrs) {
        if (!sub.empty() && std::strstr(name, sub.c_str()) != nullptr) {
            return true;
        }
    }
    return false;
}

bool is_loader_or_main_name(const char* name) {
    if (!name) return false;
    // Empty dlpi_name is the main executable; also hook libvulkan loader so
    // lazy dlopen("libGLES_mali.so") inside vkCreateInstance is caught immediately.
    if (name[0] == '\0') return true;
    if (std::strstr(name, "libvulkan") != nullptr) return true;
    if (std::strstr(name, "vulkan") != nullptr && std::strstr(name, "VkLayer_mali_csf_dump") == nullptr) {
        return true;
    }
    return false;
}

// ============================================================================
// Hooked libc syscalls (installed into libGLES_mali.so GOT/PLT)
// ============================================================================
int hooked_open(const char* pathname, int flags, ...) {
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list ap;
        va_start(ap, flags);
        mode = static_cast<mode_t>(va_arg(ap, int));
        va_end(ap);
    }
    int fd = (flags & O_CREAT) ? layer_state().orig_open(pathname, flags, mode)
                               : layer_state().orig_open(pathname, flags);
    if (fd >= 0) dispatch_hooked_mali_open(fd, pathname);
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
    int fd = (flags & O_CREAT) ? layer_state().orig_open64(pathname, flags, mode)
                               : layer_state().orig_open64(pathname, flags);
    if (fd >= 0) dispatch_hooked_mali_open(fd, pathname);
    return fd;
}

int hooked_open_2(const char* pathname, int flags) {
    int fd = layer_state().orig_open_2(pathname, flags);
    if (fd >= 0) dispatch_hooked_mali_open(fd, pathname);
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
    int fd = (flags & O_CREAT) ? layer_state().orig_openat(dirfd, pathname, flags, mode)
                               : layer_state().orig_openat(dirfd, pathname, flags);
    if (fd >= 0) dispatch_hooked_mali_open(fd, pathname);
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
    int fd = (flags & O_CREAT) ? layer_state().orig_openat64(dirfd, pathname, flags, mode)
                               : layer_state().orig_openat64(dirfd, pathname, flags);
    if (fd >= 0) dispatch_hooked_mali_open(fd, pathname);
    return fd;
}

int hooked_close(int fd) {
    dispatch_hooked_close(fd);
    return layer_state().orig_close(fd);
}

void* hooked_mmap_impl(void* (*real_mmap)(void*, size_t, int, int, int, off64_t),
                       void* addr, size_t length, int prot, int flags, int fd, off64_t offset) {
    void* ret = real_mmap(addr, length, prot, flags, fd, offset);
    if (ret != MAP_FAILED && fd >= 0) {
        dispatch_hooked_mmap(fd, static_cast<uint64_t>(offset),
                             reinterpret_cast<uintptr_t>(ret), length);
    }
    return ret;
}

void* hooked_mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset) {
    return hooked_mmap_impl(
        reinterpret_cast<void* (*)(void*, size_t, int, int, int, off64_t)>(layer_state().orig_mmap),
        addr, length, prot, flags, fd, static_cast<off64_t>(offset));
}

void* hooked_mmap64(void* addr, size_t length, int prot, int flags, int fd, off64_t offset) {
    return hooked_mmap_impl(layer_state().orig_mmap64, addr, length, prot, flags, fd, offset);
}

int hooked_munmap(void* addr, size_t length) {
    dispatch_hooked_munmap(reinterpret_cast<uintptr_t>(addr), length);
    return layer_state().orig_munmap(addr, length);
}

int hooked_ioctl(int fd, unsigned long request, ...) {
    va_list ap;
    va_start(ap, request);
    void* arg = va_arg(ap, void*);
    va_end(ap);

    // Snapshot union input fields BEFORE orig_ioctl() overwrites them with kernel outputs!
    uint64_t saved_va_pages = 0;
    uint32_t saved_import_type = 0;
    int saved_dmabuf_fd = -1;
    uintptr_t saved_user_ptr = 0;
    size_t saved_user_len = 0;

    const unsigned int ioc_type = _IOC_TYPE(request);
    const unsigned int ioc_nr = _IOC_NR(request);
    if (arg && ioc_type == kKbaseIoctlType) {
        if (ioc_nr == kIoctlNrMemAlloc || ioc_nr == kIoctlNrMemAllocEx || ioc_nr == kIoctlNrMemAllocExLegacy) {
            const auto* ma = reinterpret_cast<const kbase_ioctl_mem_alloc*>(arg);
            saved_va_pages = ma->in.va_pages;
        } else if (ioc_nr == kIoctlNrMemImport) {
            const auto* mi = reinterpret_cast<const kbase_ioctl_mem_import*>(arg);
            saved_import_type = mi->in.type;
            if (mi->in.phandle != 0) {
                if (saved_import_type == kBaseMemImportTypeUmm) {
                    const int* fd_ptr = reinterpret_cast<const int*>(static_cast<uintptr_t>(mi->in.phandle));
                    saved_dmabuf_fd = *fd_ptr;
                } else if (saved_import_type == kBaseMemImportTypeUserBuffer) {
                    const auto* ub = reinterpret_cast<const base_mem_import_user_buffer*>(
                        static_cast<uintptr_t>(mi->in.phandle));
                    saved_user_ptr = static_cast<uintptr_t>(ub->ptr);
                    saved_user_len = static_cast<size_t>(ub->length);
                }
            }
        }
    }

    int ret = layer_state().orig_ioctl(fd, request, arg);
    if (ret < 0 || !arg) return ret;

    LayerState& s = layer_state();
    std::lock_guard<std::mutex> lk(s.mu);
    if (!s.mali_fds.count(fd) || ioc_type != kKbaseIoctlType) return ret;

    if (ioc_nr == kIoctlNrMemAlloc || ioc_nr == kIoctlNrMemAllocEx || ioc_nr == kIoctlNrMemAllocExLegacy) {
        const auto* ma = reinterpret_cast<const kbase_ioctl_mem_alloc*>(arg);
        s.va.record_mem_alloc(ma->out.gpu_va, saved_va_pages, ma->out.flags, ioc_nr != kIoctlNrMemAlloc);
    } else if (ioc_nr == kIoctlNrMemImport) {
        const auto* mi = reinterpret_cast<const kbase_ioctl_mem_import*>(arg);
        if (saved_import_type == kBaseMemImportTypeUmm) {
            s.va.record_mem_import_umm(saved_dmabuf_fd, mi->out.gpu_va, mi->out.va_pages, mi->out.flags);
        } else if (saved_import_type == kBaseMemImportTypeUserBuffer) {
            s.va.record_mem_import_user_buffer(saved_user_ptr, saved_user_len, mi->out.gpu_va, mi->out.flags);
        }
    } else {
        // Delegate remaining ioctls (VERSION_CHECK_CSF, GET_DDK_VERSION, CS_QUEUE_*, MEM_FREE)
        // to the shared post-ioctl handler (unlocking first is avoided by using an internal helper).
        if (ioc_nr == kIoctlNrVersionCheckCsf) {
            const auto* vc = reinterpret_cast<const kbase_ioctl_version_check*>(arg);
            s.kbase_ver_major = vc->major;
            s.kbase_ver_minor = vc->minor;
            csf_log(LogLevel::Info, "ioctl", "VERSION_CHECK_CSF: kbase %u.%u", vc->major, vc->minor);
        } else if (ioc_nr == kIoctlNrGetDdkVersion) {
            const auto* dv = reinterpret_cast<const kbase_ioctl_get_ddk_version*>(arg);
            if (dv->version_buffer && dv->size > 0) {
                const char* ver = reinterpret_cast<const char*>(static_cast<uintptr_t>(dv->version_buffer));
                s.detected_abi = std::string(ver, strnlen(ver, dv->size));
                csf_log(LogLevel::Info, "ioctl", "GET_DDK_VERSION: %s", s.detected_abi.c_str());
            }
        } else if (ioc_nr == kIoctlNrCsQueueRegister) {
            const auto* qr = reinterpret_cast<const kbase_ioctl_cs_queue_register*>(arg);
            CsfQueueState& q = s.queues_by_gpu_va[qr->buffer_gpu_addr];
            q.buffer_gpu_addr = qr->buffer_gpu_addr;
            q.buffer_size = qr->buffer_size;
        } else if (ioc_nr == kIoctlNrCsQueueRegisterEx) {
            const auto* qr = reinterpret_cast<const kbase_ioctl_cs_queue_register_ex*>(arg);
            CsfQueueState& q = s.queues_by_gpu_va[qr->buffer_gpu_addr];
            q.buffer_gpu_addr = qr->buffer_gpu_addr;
            q.buffer_size = qr->buffer_size;
        } else if (ioc_nr == kIoctlNrCsQueueBind) {
            const auto* qb = reinterpret_cast<const kbase_ioctl_cs_queue_bind*>(arg);
            CsfQueueState& q = s.queues_by_gpu_va[qb->in.buffer_gpu_addr];
            q.buffer_gpu_addr = qb->in.buffer_gpu_addr;
            q.group_handle = qb->in.group_handle;
            q.csi_index = qb->in.csi_index;
            q.mmap_handle = qb->out.mmap_handle;
            auto& qlist = s.queues_by_mmap_handle[qb->out.mmap_handle];
            if (std::find(qlist.begin(), qlist.end(), qb->in.buffer_gpu_addr) == qlist.end()) {
                qlist.push_back(qb->in.buffer_gpu_addr);
            }
            const bool shared = s.is_kbase_1_35_or_newer() || (qlist.size() > 1u);
            for (uint64_t qva : qlist) {
                s.queues_by_gpu_va[qva].csg_shared_user_io = shared;
                // If user_io was already mapped for an earlier queue on this shared handle, propagate it
                if (q.user_io_cpu == 0 && s.queues_by_gpu_va[qva].user_io_cpu != 0) {
                    q.user_io_cpu = s.queues_by_gpu_va[qva].user_io_cpu;
                }
            }
        } else if (ioc_nr == kIoctlNrCsQueueKick) {
            const auto* qk = reinterpret_cast<const kbase_ioctl_cs_queue_kick*>(arg);
            process_queue_kick_locked(s, qk->buffer_gpu_addr);
        } else if (ioc_nr == kIoctlNrCsQueueTerminate) {
            const auto* qt = reinterpret_cast<const kbase_ioctl_cs_queue_terminate*>(arg);
            auto qit = s.queues_by_gpu_va.find(qt->buffer_gpu_addr);
            if (qit != s.queues_by_gpu_va.end()) {
                auto mit = s.queues_by_mmap_handle.find(qit->second.mmap_handle);
                if (mit != s.queues_by_mmap_handle.end()) {
                    auto& v = mit->second;
                    v.erase(std::remove(v.begin(), v.end(), qt->buffer_gpu_addr), v.end());
                    if (v.empty()) s.queues_by_mmap_handle.erase(mit);
                }
                s.queues_by_gpu_va.erase(qit);
            }
        } else if (ioc_nr == kIoctlNrMemFree) {
            const auto* mf = reinterpret_cast<const kbase_ioctl_mem_free*>(arg);
            s.va.record_mem_free(mf->gpu_addr);
        }
    }
    return ret;
}

// ============================================================================
// Lazy dlopen / android_dlopen_ext interception (closes Gap 4)
// ============================================================================
void* hooked_dlopen(const char* filename, int flags) {
    void* handle = layer_state().orig_dlopen(filename, flags);
    if (handle && filename && is_mali_library_name(filename)) {
        {
            std::lock_guard<std::mutex> lk(hook_globals().mu);
            hook_globals().stats.dlopen_intercepts++;
        }
        csf_log(LogLevel::Info, "hooks", "Intercepted lazy dlopen(\"%s\"); patching GOT immediately", filename);
        install_mali_hooks();
    }
    return handle;
}

void* hooked_android_dlopen_ext(const char* filename, int flags, const void* extinfo) {
    void* handle = layer_state().orig_android_dlopen_ext(filename, flags, extinfo);
    if (handle && filename && is_mali_library_name(filename)) {
        {
            std::lock_guard<std::mutex> lk(hook_globals().mu);
            hook_globals().stats.dlopen_intercepts++;
        }
        csf_log(LogLevel::Info, "hooks", "Intercepted lazy android_dlopen_ext(\"%s\"); patching GOT immediately", filename);
        install_mali_hooks();
    }
    return handle;
}

// ============================================================================
// ELF GOT/PLT Relocation Patching (Android Bionic + glibc)
// ============================================================================
void* resolve_mali_hook_target(const char* sym) {
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

void* resolve_loader_hook_target(const char* sym) {
    if (std::strcmp(sym, "dlopen") == 0 && layer_state().orig_dlopen) {
        return reinterpret_cast<void*>(&hooked_dlopen);
    }
    if (std::strcmp(sym, "android_dlopen_ext") == 0 && layer_state().orig_android_dlopen_ext) {
        return reinterpret_cast<void*>(&hooked_android_dlopen_ext);
    }
    return nullptr;
}

std::string detect_ddk_version_from_phdr(const struct dl_phdr_info* info) {
    for (ElfW(Half) i = 0; i < info->dlpi_phnum; ++i) {
        const auto& ph = info->dlpi_phdr[i];
        if (ph.p_type != PT_LOAD || (ph.p_flags & PF_W) != 0 || ph.p_memsz < 16) continue;
        const char* base = reinterpret_cast<const char*>(info->dlpi_addr + ph.p_vaddr);
        const size_t len = static_cast<size_t>(ph.p_filesz);
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

uint32_t patch_rela_table(ElfW(Addr) dlpi_addr,
                          const char* strtab,
                          const ElfW(Sym)* symtab,
                          const ElfW(Rela)* rela,
                          size_t rela_sz,
                          size_t page_sz,
                          void* (*resolver)(const char*)) {
    if (!rela || rela_sz == 0) return 0;
    uint32_t patched = 0;
    const size_t n_rel = rela_sz / sizeof(ElfW(Rela));
    for (size_t i = 0; i < n_rel; ++i) {
        const ElfW(Rela)& r = rela[i];
#if defined(__aarch64__)
        const uint32_t sym_idx = ELF64_R_SYM(r.r_info);
        const uint32_t r_type = ELF64_R_TYPE(r.r_info);
        if (r_type != R_AARCH64_JUMP_SLOT && r_type != R_AARCH64_GLOB_DAT) continue;
#elif defined(__x86_64__)
        const uint32_t sym_idx = ELF64_R_SYM(r.r_info);
        const uint32_t r_type = ELF64_R_TYPE(r.r_info);
        if (r_type != R_X86_64_JUMP_SLOT && r_type != R_X86_64_GLOB_DAT) continue;
#else
        const uint32_t sym_idx = ELF64_R_SYM(r.r_info);
#endif
        const char* sym_name = strtab + symtab[sym_idx].st_name;
        void* replacement = resolver(sym_name);
        if (!replacement) continue;

        void** got_slot = reinterpret_cast<void**>(dlpi_addr + r.r_offset);
        if (*got_slot == replacement) continue;

        const uintptr_t page_base = reinterpret_cast<uintptr_t>(got_slot) & ~(page_sz - 1u);
        if (mprotect(reinterpret_cast<void*>(page_base), page_sz, PROT_READ | PROT_WRITE) == 0) {
            __atomic_store_n(got_slot, replacement, __ATOMIC_RELEASE);
            patched++;
        }
    }
    return patched;
}

void patch_phdr_dynamic(const struct dl_phdr_info* info,
                        void* (*resolver)(const char*),
                        uint32_t* out_slots) {
    const ElfW(Dyn)* dyn = nullptr;
    for (ElfW(Half) i = 0; i < info->dlpi_phnum; ++i) {
        if (info->dlpi_phdr[i].p_type == PT_DYNAMIC) {
            dyn = reinterpret_cast<const ElfW(Dyn)*>(info->dlpi_addr + info->dlpi_phdr[i].p_vaddr);
            break;
        }
    }
    if (!dyn) return;

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
    if (!strtab || !symtab) return;

    const size_t page_sz = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    uint32_t n = 0;
    n += patch_rela_table(info->dlpi_addr, strtab, symtab, jmprel, jmprel_sz, page_sz, resolver);
    n += patch_rela_table(info->dlpi_addr, strtab, symtab, rela, rela_sz, page_sz, resolver);
    if (out_slots) *out_slots += n;
}

int patch_elf_got_cb(struct dl_phdr_info* info, size_t /*size*/, void* /*data*/) {
    if (!info) return 0;
    const uintptr_t base = static_cast<uintptr_t>(info->dlpi_addr);

    if (info->dlpi_name && is_mali_library_name(info->dlpi_name)) {
        if (hook_globals().patched_mali_bases.count(base)) return 0;

        LayerState& s = layer_state();
        if (s.detected_abi == "unknown") {
            std::string probed = detect_ddk_version_from_phdr(info);
            if (!probed.empty()) s.detected_abi = probed;
            else if (std::strstr(info->dlpi_name, "r54p") != nullptr) s.detected_abi = "r54p";
            else if (std::strstr(info->dlpi_name, "r49p1") != nullptr) s.detected_abi = "r49p1";
            else if (std::strstr(info->dlpi_name, "r44p1") != nullptr) s.detected_abi = "r44p1";
            else if (std::strstr(info->dlpi_name, "r38p1") != nullptr) s.detected_abi = "r38p1";
            else s.detected_abi = "csf-unknown";
        }

        uint32_t slots = 0;
        patch_phdr_dynamic(info, &resolve_mali_hook_target, &slots);
        hook_globals().patched_mali_bases.insert(base);
        hook_globals().stats.mali_libs_patched++;
        hook_globals().stats.got_slots_patched += slots;
        csf_log(LogLevel::Info, "hooks", "Patched Mali target \"%s\" (%u GOT slots, abi=%s)",
                info->dlpi_name, slots, s.detected_abi.c_str());
        return 0;
    }

    if (is_loader_or_main_name(info->dlpi_name)) {
        if (hook_globals().patched_loader_bases.count(base)) return 0;
        uint32_t slots = 0;
        patch_phdr_dynamic(info, &resolve_loader_hook_target, &slots);
        hook_globals().patched_loader_bases.insert(base);
        if (slots > 0) {
            hook_globals().stats.loader_libs_patched++;
            hook_globals().stats.got_slots_patched += slots;
            csf_log(LogLevel::Debug, "hooks", "Patched dlopen GOT in \"%s\" (%u slots)",
                    (info->dlpi_name && info->dlpi_name[0]) ? info->dlpi_name : "<main>", slots);
        }
    }
    return 0;
}

} // namespace

void install_mali_hooks() {
    std::lock_guard<std::mutex> hlk(hook_globals().mu);
    LayerState& s = layer_state();
    {
        std::lock_guard<std::mutex> slk(s.mu);
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
            s.orig_dlopen = reinterpret_cast<void* (*)(const char*, int)>(dlsym(RTLD_NEXT, "dlopen"));
            s.orig_android_dlopen_ext = reinterpret_cast<void* (*)(const char*, int, const void*)>(
                dlsym(RTLD_NEXT, "android_dlopen_ext"));
        }
    }
    dl_iterate_phdr(patch_elf_got_cb, nullptr);
    {
        std::lock_guard<std::mutex> slk(s.mu);
        s.hooks_installed = true;
    }
}

void register_extra_hook_target_substr(const std::string& substr) {
    std::lock_guard<std::mutex> lk(hook_globals().mu);
    if (!substr.empty()) {
        hook_globals().extra_target_substrs.push_back(substr);
    }
}

void clear_extra_hook_target_substrs() {
    std::lock_guard<std::mutex> lk(hook_globals().mu);
    hook_globals().extra_target_substrs.clear();
}

HookStats get_hook_stats() {
    std::lock_guard<std::mutex> lk(hook_globals().mu);
    return hook_globals().stats;
}

// ============================================================================
// Direct syscall event dispatchers (used by hooks AND unit tests)
// ============================================================================
void dispatch_hooked_mali_open(int fd, const char* pathname) {
    if (fd >= 0 && is_mali_path(pathname)) {
        std::lock_guard<std::mutex> lk(layer_state().mu);
        layer_state().mali_fds.insert(fd);
    }
}

void dispatch_hooked_close(int fd) {
    std::lock_guard<std::mutex> lk(layer_state().mu);
    layer_state().mali_fds.erase(fd);
    layer_state().va.record_fd_close(fd);
}

void dispatch_hooked_mmap(int fd, uint64_t offset, uintptr_t cpu_va, size_t length) {
    std::lock_guard<std::mutex> lk(layer_state().mu);
    LayerState& s = layer_state();
    if (s.mali_fds.count(fd)) {
        auto qit = s.queues_by_mmap_handle.find(offset);
        if (qit != s.queues_by_mmap_handle.end()) {
            for (uint64_t qva : qit->second) {
                s.queues_by_gpu_va[qva].user_io_cpu = cpu_va;
            }
        } else {
            s.va.record_mali_mmap(offset, cpu_va, length, fd);
        }
    } else if (s.va.is_tracked_dmabuf_fd(fd)) {
        s.va.record_dmabuf_mmap(fd, offset, cpu_va, length);
    }
}

void dispatch_hooked_munmap(uintptr_t cpu_va, size_t length) {
    std::lock_guard<std::mutex> lk(layer_state().mu);
    layer_state().va.record_munmap(cpu_va, length);
}

int dispatch_hooked_ioctl(int fd, unsigned long request, void* arg) {
    if (!arg) return -1;
    const unsigned int ioc_type = _IOC_TYPE(request);
    const unsigned int ioc_nr = _IOC_NR(request);
    if (ioc_type != kKbaseIoctlType) return 0;

    LayerState& s = layer_state();
    std::lock_guard<std::mutex> lk(s.mu);
    if (!s.mali_fds.count(fd)) return 0;

    if (ioc_nr == kIoctlNrVersionCheckCsf) {
        const auto* vc = reinterpret_cast<const kbase_ioctl_version_check*>(arg);
        s.kbase_ver_major = vc->major;
        s.kbase_ver_minor = vc->minor;
    } else if (ioc_nr == kIoctlNrGetDdkVersion) {
        const auto* dv = reinterpret_cast<const kbase_ioctl_get_ddk_version*>(arg);
        if (dv->version_buffer && dv->size > 0) {
            const char* ver = reinterpret_cast<const char*>(static_cast<uintptr_t>(dv->version_buffer));
            s.detected_abi = std::string(ver, strnlen(ver, dv->size));
        }
    } else if (ioc_nr == kIoctlNrCsQueueRegister) {
        const auto* qr = reinterpret_cast<const kbase_ioctl_cs_queue_register*>(arg);
        CsfQueueState& q = s.queues_by_gpu_va[qr->buffer_gpu_addr];
        q.buffer_gpu_addr = qr->buffer_gpu_addr;
        q.buffer_size = qr->buffer_size;
    } else if (ioc_nr == kIoctlNrCsQueueRegisterEx) {
        const auto* qr = reinterpret_cast<const kbase_ioctl_cs_queue_register_ex*>(arg);
        CsfQueueState& q = s.queues_by_gpu_va[qr->buffer_gpu_addr];
        q.buffer_gpu_addr = qr->buffer_gpu_addr;
        q.buffer_size = qr->buffer_size;
    } else if (ioc_nr == kIoctlNrCsQueueBind) {
        const auto* qb = reinterpret_cast<const kbase_ioctl_cs_queue_bind*>(arg);
        CsfQueueState& q = s.queues_by_gpu_va[qb->in.buffer_gpu_addr];
        q.buffer_gpu_addr = qb->in.buffer_gpu_addr;
        q.group_handle = qb->in.group_handle;
        q.csi_index = qb->in.csi_index;
        q.mmap_handle = qb->out.mmap_handle;
        auto& qlist = s.queues_by_mmap_handle[qb->out.mmap_handle];
        if (std::find(qlist.begin(), qlist.end(), qb->in.buffer_gpu_addr) == qlist.end()) {
            qlist.push_back(qb->in.buffer_gpu_addr);
        }
        const bool shared = s.is_kbase_1_35_or_newer() || (qlist.size() > 1u);
        for (uint64_t qva : qlist) {
            s.queues_by_gpu_va[qva].csg_shared_user_io = shared;
            if (q.user_io_cpu == 0 && s.queues_by_gpu_va[qva].user_io_cpu != 0) {
                q.user_io_cpu = s.queues_by_gpu_va[qva].user_io_cpu;
            }
        }
    } else if (ioc_nr == kIoctlNrCsQueueKick) {
        const auto* qk = reinterpret_cast<const kbase_ioctl_cs_queue_kick*>(arg);
        process_queue_kick_locked(s, qk->buffer_gpu_addr);
    } else if (ioc_nr == kIoctlNrMemFree) {
        const auto* mf = reinterpret_cast<const kbase_ioctl_mem_free*>(arg);
        s.va.record_mem_free(mf->gpu_addr);
    }
    return 0;
}

} // namespace mali_csf_dump

#pragma once
// ============================================================================
// csf_elf_hooks.h
// ELF GOT/PLT patching for libGLES_mali.so and lazy dlopen / android_dlopen_ext
// interception on Android Bionic and GNU/Linux glibc.
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <string>

namespace mali_csf_dump {

struct HookStats {
    uint32_t mali_libs_patched = 0;
    uint32_t loader_libs_patched = 0;
    uint32_t got_slots_patched = 0;
    uint32_t dlopen_intercepts = 0;
};

// Install or refresh GOT/PLT hooks across loaded libraries:
//   - Hooks open/open64/__open_2/openat/openat64/close/ioctl/mmap/mmap64/munmap
//     inside libGLES_mali*.so (and any explicitly registered mock/target libs).
//   - Hooks dlopen (and android_dlopen_ext on Android) inside the Vulkan loader
//     and main executable so lazy-loaded libGLES_mali.so is patched immediately
//     upon dlopen() return, BEFORE its first open("/dev/mali0") / ioctl().
void install_mali_hooks();

// Register an additional shared-library substring to patch with Mali syscall
// hooks (used by local unit tests to test lazy dlopen interception without
// real libGLES_mali.so hardware binaries).
void register_extra_hook_target_substr(const std::string& substr);
void clear_extra_hook_target_substrs();

HookStats get_hook_stats();

// Exposed for direct unit testing of the hooked syscall handlers
int dispatch_hooked_ioctl(int fd, unsigned long request, void* arg);
void dispatch_hooked_mali_open(int fd, const char* pathname);
void dispatch_hooked_close(int fd);
void dispatch_hooked_mmap(int fd, uint64_t offset, uintptr_t cpu_va, size_t length);
void dispatch_hooked_munmap(uintptr_t cpu_va, size_t length);

} // namespace mali_csf_dump

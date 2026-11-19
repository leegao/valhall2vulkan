// ============================================================================
// mock_lazy_mali_driver.cpp
// Minimal mock shared library used ONLY by test_csf_dump_layer to verify
// lazy dlopen() GOT/PLT hooking (Gap 4) locally without real Mali hardware.
// ============================================================================

#include <fcntl.h>
#include <unistd.h>

extern "C" {

// Calls open() through this shared object's PLT/GOT so if our lazy dlopen()
// hook patched this library's GOT upon dlopen() return, open("/dev/null") or
// open("/tmp/mock_mali_dev") will be intercepted by hooked_open().
int mock_mali_open_path(const char* path) {
    int fd = open(path, O_RDONLY);
    return fd;
}

int mock_mali_close_fd(int fd) {
    return close(fd);
}

} // extern "C"

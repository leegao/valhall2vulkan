#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __GNUC__
#define COMPAT_WEAK __attribute__((weak))
#else
#define COMPAT_WEAK
#endif

/*
 * LLVM-21 prebuilt static archives compiled against glibc 2.38+ emit calls to
 * C23 __isoc23_* symbols. Provide weak forwarding shims so the binary links
 * and runs cleanly on Debian 12 (glibc 2.36) as well as newer glibc hosts.
 */
COMPAT_WEAK unsigned long __isoc23_strtoul(const char* nptr, char** endptr, int base) {
    return strtoul(nptr, endptr, base);
}

COMPAT_WEAK long __isoc23_strtol(const char* nptr, char** endptr, int base) {
    return strtol(nptr, endptr, base);
}

COMPAT_WEAK unsigned long long __isoc23_strtoull(const char* nptr, char** endptr, int base) {
    return strtoull(nptr, endptr, base);
}

COMPAT_WEAK long long __isoc23_strtoll(const char* nptr, char** endptr, int base) {
    return strtoll(nptr, endptr, base);
}

COMPAT_WEAK int __isoc23_sscanf(const char* str, const char* format, ...) {
    va_list args;
    va_start(args, format);
    int rc = vsscanf(str, format, args);
    va_end(args);
    return rc;
}

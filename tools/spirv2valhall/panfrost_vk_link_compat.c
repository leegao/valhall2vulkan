// Weak link-time compatibility shims for statically linking Mesa's
// libpanfrost_vk_compile.a on hosts without libSPIRV-Tools.a or with glibc < 2.38.

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(__GNUC__) || defined(__clang__)
#define VALHALL_WEAK __attribute__((weak))
#else
#define VALHALL_WEAK
#endif

// vtn_debug.c references SPIRV-Tools only from spirv_print_asm() (NIR_DEBUG=spirv).
// Weak stubs keep standalone static links working when libSPIRV-Tools.a is absent;
// if libSPIRV-Tools.a is linked, its strong definitions take precedence.
VALHALL_WEAK void* spvContextCreate(int env) {
    (void)env;
    return NULL;
}

VALHALL_WEAK int spvBinaryToText(
    const void* ctx, const uint32_t* binary, size_t word_count,
    uint32_t options, void** text, void** diagnostic) {
    (void)ctx;
    (void)binary;
    (void)word_count;
    (void)options;
    if (text) *text = NULL;
    if (diagnostic) *diagnostic = NULL;
    return -1;
}

VALHALL_WEAK void spvTextDestroy(void* text) { (void)text; }
VALHALL_WEAK int spvDiagnosticPrint(const void* diag) {
    (void)diag;
    return 0;
}
VALHALL_WEAK void spvDiagnosticDestroy(void* diag) { (void)diag; }

// Prebuilt Mesa objects compiled against glibc 2.38+ reference C23 strto*/sscanf
// entry points. Provide weak forwarding shims when linking on older glibc (e.g. 2.36).
#if defined(__GLIBC__) && (__GLIBC__ == 2 && __GLIBC_MINOR__ < 38)
VALHALL_WEAK long __isoc23_strtol(const char* nptr, char** endptr, int base) {
    return strtol(nptr, endptr, base);
}
VALHALL_WEAK long long __isoc23_strtoll(const char* nptr, char** endptr, int base) {
    return strtoll(nptr, endptr, base);
}
VALHALL_WEAK unsigned long __isoc23_strtoul(const char* nptr, char** endptr, int base) {
    return strtoul(nptr, endptr, base);
}
VALHALL_WEAK unsigned long long __isoc23_strtoull(const char* nptr, char** endptr, int base) {
    return strtoull(nptr, endptr, base);
}
VALHALL_WEAK int __isoc23_sscanf(const char* str, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int r = vsscanf(str, fmt, ap);
    va_end(ap);
    return r;
}
#endif

#define _GNU_SOURCE
#include <stdbool.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <link.h>

// Offsets inside libMali-Gxx_r55p0-00rel0.so (mali_offline_compiler v8.8.1 Build 73b41e):
//   cmpbe_v2_compile_multiple_shaders      @ +0x001c8400
//   cmpbe_v2_deserialize_MBS2_to_C         @ +0x001cada0
//   cmpbe_v2_compile_multiple_shaders@GOT  @ +0x01967ec0
typedef int (*pfn_compile_multi_t)(
    void* a0, int shader_count, void* a2, void* a3, void* a4, void* a5,
    void* s0, void* s1, void* s2, void* s3, void*** out_compiled_program_ptr_ptr,
    void* s5, void* s6, void* s7);

typedef uint64_t (*pfn_deserialize_t)(
    void* context_ptr, void* stream_obj_72b, char** out_c_str);

static pfn_compile_multi_t real_compile_multi = NULL;
static pfn_deserialize_t real_deserialize = NULL;

static int my_compile_multi(
    void* a0, int shader_count, void* a2, void* a3, void* a4, void* a5,
    void* s0, void* s1, void* s2, void* s3, void*** out_compiled_program_ptr_ptr,
    void* s5, void* s6, void* s7) {
    int ret = real_compile_multi(
        a0, shader_count, a2, a3, a4, a5,
        s0, s1, s2, s3, out_compiled_program_ptr_ptr, s5, s6, s7);

    if (ret == 0 && out_compiled_program_ptr_ptr && *out_compiled_program_ptr_ptr &&
        real_deserialize && shader_count > 0) {
        uint8_t* out_program = (uint8_t*)(*out_compiled_program_ptr_ptr);
        const size_t stride = 72;
        for (int i = 0; i < shader_count; ++i) {
            uint8_t fake_stream_obj[72];
            memcpy(fake_stream_obj, out_program + (size_t)i * stride, stride);
            char* c_code = NULL;
            uint64_t status = real_deserialize(a0, fake_stream_obj, &c_code);
            if (status == 0 && c_code) {
                const char* out_path = getenv("MOC_DUMP_PATH");
                if (!out_path) out_path = "dumped_mbs2.h";
                FILE* fp = fopen(out_path, "w");
                if (fp) {
                    fprintf(fp, "#include \"cmpbe_chunks.h\"\n%s", c_code);
                    fclose(fp);
                }
            }
        }
    }
    return ret;
}

void* dlopen(const char* filename, int flags) {
    static void* (*real_dlopen)(const char*, int) = NULL;
    if (!real_dlopen) {
        real_dlopen = (void* (*)(const char*, int))dlsym(RTLD_NEXT, "dlopen");
    }
    void* handle = real_dlopen(filename, flags);
    if (handle && filename && strstr(filename, "libMali-Gxx")) {
        struct link_map* lm = (struct link_map*)handle;
        uintptr_t base = (uintptr_t)lm->l_addr;
        real_compile_multi = (pfn_compile_multi_t)(base + 0x001c8400ULL);
        real_deserialize = (pfn_deserialize_t)(base + 0x001cada0ULL);
        uintptr_t got_entry = base + 0x01967ec0ULL;
        uintptr_t page = got_entry & ~0xFFFULL;
        if (mprotect((void*)page, 4096, PROT_READ | PROT_WRITE) == 0) {
            *(void**)got_entry = (void*)&my_compile_multi;
        }
    }
    return handle;
}

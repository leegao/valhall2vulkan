/*
 * Copyright 2024 Collabora Ltd
 * SPDX-License-Identifier: MIT
 */

#ifndef PANFROST_VK_COMPILE_H
#define PANFROST_VK_COMPILE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Options for panfrost_vk_compile().  Zero-initialize and override what you
 * need; in_path and out_path are required. */
struct panfrost_vk_compile_options {
   /* Required: input SPIR-V module path and output ISA binary path. */
   const char *in_path;
   const char *out_path;
   unsigned arch;

   /* Entry point name override, or NULL to use the name found in the module. */
   const char *entry_override;

   /* If non-NULL, write a metadata JSON file here. */
   const char *info_json_path;

   /* Shader stage override (a MESA_SHADER_* value), or -1 to detect. */
   int stage_override;

   /* Backend / lowering knobs (defaults match panvk + libMali on v9+). */
   bool robust;         /* SSBO/descriptor robustness bounds checks */
   bool no_idvs;        /* disable index-driven vertex shading */
   bool promote_imm;    /* FAU immediate promotion */
   bool clamp_version;  /* clamp SPIR-V >= 1.4 header to 1.3 */
   bool dump_nir;       /* print final lowered NIR to stderr */
   bool disasm;         /* disassemble emitted ISA to stderr */

   /* Value fed to the load_core_max_id_arm sysval. */
   uint32_t core_max_id;
};

/* Compile a SPIR-V module to Valhall/Bifrost ISA end-to-end.
 *
 * Returns 0 on success, non-zero on failure.  Diagnostics go to stderr.
 * The backend is selected by the process-global PAN_USE_KRAID env var (read
 * via pan_use_kraid()); set it before the first call if you need to override.
 */
int
panfrost_vk_compile(const struct panfrost_vk_compile_options *opts);

#ifdef __cplusplus
}
#endif

#endif /* PANFROST_VK_COMPILE_H */

// Reverse engineered struct and enum definitions for ARM Mali DDK MBS2 chunks
// Extracted from libGLES_mali.so (r54p3, .gnu_debugdata) and libGLES_mali.r44p1.so
// Compatible with cmpbe_v2_deserialize_MBS2_to_C / cmpbe_chunk_read_MBS2

#ifndef CMPBE_CHUNKS_H
#define CMPBE_CHUNKS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint8_t b8;

/* Forward declarations */
struct cmpbe_chunk_TYPE;
struct cmpbe_chunk_TPAC;
struct cmpbe_chunk_TPAR;
struct cmpbe_chunk_TPGE;
struct cmpbe_chunk_TPIB;
struct cmpbe_chunk_TPMA;
struct cmpbe_chunk_TPPO;
struct cmpbe_chunk_TPQU;
struct cmpbe_chunk_TPPR;
struct cmpbe_chunk_TPPW;
struct cmpbe_chunk_TPSA;
struct cmpbe_chunk_TPST;
struct cmpbe_chunk_TPSE;
struct cmpbe_chunk_TPMD;
struct cmpbe_chunk_RLOC;
struct cmpbe_chunk_RELO;
struct cmpbe_chunk_FIXP;
struct cmpbe_chunk_SINI;
struct cmpbe_chunk_SYMD;
struct cmpbe_chunk_SYMB;
struct cmpbe_chunk_DBUG;
struct cmpbe_chunk_FCST;
struct cmpbe_chunk_GREL;
struct cmpbe_chunk_SPDf;
struct cmpbe_chunk_PDSC;
struct cmpbe_chunk_VLKN;
struct cmpbe_chunk_FNMD;
struct cmpbe_chunk_SLOC;
struct cmpbe_chunk_EBIN;
struct cmpbe_chunk_CXFB;
struct cmpbe_chunk_CXFE;
struct cmpbe_chunk_KWGS;
struct cmpbe_chunk_KPAR;
struct cmpbe_chunk_KERN;
struct cmpbe_chunk_CBLE;
struct cmpbe_chunk_CCOM;
struct cmpbe_chunk_CVER;
struct cmpbe_chunk_CFRA;
struct cmpbe_chunk_CTSC;
struct cmpbe_chunk_CTSE;
struct cmpbe_chunk_CGEO;
struct cmpbe_chunk_SSRC;
struct cmpbe_chunk_MOMD;
struct cmpbe_chunk_PLOC;
struct cmpbe_chunk_FOTV;
struct cmpbe_chunk_OUTV;
struct cmpbe_chunk_LCAL;
struct cmpbe_chunk_BLDM;
struct cmpbe_chunk_AINF;
struct cmpbe_chunk_ACHN;
struct cmpbe_chunk_SPAS;
struct cmpbe_chunk_VMVW;
struct cmpbe_chunk_VRSM;
struct cmpbe_chunk_MTES;
struct cmpbe_chunk_MGEO;
struct cmpbe_chunk_GSVC;

/* Values from llvm::Mali::setVEHW (0x2417bb0) and YAML ScalarEnumerationTraits (0x22385f0) */
typedef enum {
    CMPBE_VEHW_CORE_DEFAULT = 0x00,
    CMPBE_VEHW_CORE_0x0B    = 0x0b, /* arch_major 12 */
    CMPBE_VEHW_CORE_0x0C    = 0x0c, /* arch_major 13 */
    CMPBE_VEHW_CORE_0x0D    = 0x0d, /* arch_major 14 */
    CMPBE_VEHW_CORE_0x0E    = 0x0e, /* arch_major 15 */
    CMPBE_VEHW_CORE_0x10    = 0x10, /* arch_major 17 */
    CMPBE_VEHW_CORE_0x11    = 0x11, /* arch_major 18 */
    CMPBE_VEHW_CORE_0x12    = 0x12, /* arch_major 19 */
    CMPBE_VEHW_CORE_0x15    = 0x15, /* arch_major 22 */
    CMPBE_VEHW_CORE_0x16    = 0x16, /* arch_major 23 */
    CMPBE_VEHW_CORE_0x17    = 0x17, /* arch_major 24 */
    CMPBE_VEHW_CORE_0x18    = 0x18, /* arch_major 25 */
    CMPBE_VEHW_CORE_0x19    = 0x19, /* arch_major 26 */
    CMPBE_VEHW_CORE_0x1B    = 0x1b, /* arch_major 28 */
    CMPBE_VEHW_CORE_VALHALL = 0x1c, /* arch_major 29 */
    CMPBE_VEHW_CORE_0x1E    = 0x1e,
    CMPBE_VEHW_CORE_0x1F    = 0x1f,
    CMPBE_VEHW_CORE_0x20    = 0x20,
    CMPBE_VEHW_CORE_0x21    = 0x21,
    CMPBE_VEHW_CORE_0x22    = 0x22,
    CMPBE_VEHW_CORE_0x23    = 0x23,
    CMPBE_VEHW_CORE_0x24    = 0x24,
    CMPBE_VEHW_CORE_0x25    = 0x25,
    CMPBE_VEHW_CORE_0x26    = 0x26
} cmpbe_chunk_VEHW_core;

/* Values from r54p3 YAML ScalarEnumerationTraits (0x2242fc0): 1, 2, 3, 4, 5, 7, 8 */
typedef enum {
    CMPBE_VELA_LANG_1     = 0x1,
    CMPBE_VELA_LANG_2     = 0x2,
    CMPBE_VELA_LANG_3     = 0x3,
    CMPBE_VELA_LANG_4     = 0x4,
    CMPBE_VELA_LANG_5     = 0x5,
    CMPBE_VELA_LANG_SPIRV = 0x7,
    CMPBE_VELA_LANG_8     = 0x8
} cmpbe_chunk_VELA_lang;

/* Values from r54p3 YAML traits (0x223c3b0) & SPIR2LIR::get_cmpbe_type_from_tpge (0x1ddf4f0) */
typedef enum {
    CMPBE_TYPE_SCALAR_BOOL  = 0x1,
    CMPBE_TYPE_SCALAR_INT   = 0x2,
    CMPBE_TYPE_SCALAR_UINT  = 0x3,
    CMPBE_TYPE_SCALAR_FLOAT = 0x4
} cmpbe_chunk_TPGE_scalar_type;

/* Values from r54p3 YAML traits (0x223c4e0) & llvm::Mali::getTPGEScalarSize (0x2dfc9b0) */
typedef enum {
    CMPBE_TYPE_SIZE_8BIT  = 0x0,
    CMPBE_TYPE_SIZE_16BIT = 0x1,
    CMPBE_TYPE_SIZE_32BIT = 0x2,
    CMPBE_TYPE_SIZE_64BIT = 0x3
} cmpbe_chunk_TPGE_scalar_size;

/* Values from r54p3 YAML traits (0x223c610): 0..3 */
typedef enum {
    CMPBE_PRECISION_LOW    = 0x0,
    CMPBE_PRECISION_MEDIUM = 0x1,
    CMPBE_PRECISION_HIGH   = 0x2,
    CMPBE_PRECISION_3      = 0x3
} cmpbe_chunk_TPGE_precision;

/* Values from r54p3 YAML bitset traits (0x223c740) & SPIR_MBS2Helper::apply_decorations (0x1df8100) */
typedef enum {
    CMPBE_AUX_QUAL_NONE          = 0x00,
    CMPBE_AUX_QUAL_SMOOTH        = 0x01,
    CMPBE_AUX_QUAL_FLAT          = 0x02,
    CMPBE_AUX_QUAL_NOPERSPECTIVE = 0x04,
    CMPBE_AUX_QUAL_INTERP_MASK   = 0x07,
    CMPBE_AUX_QUAL_PUSH_CONSTANT = 0x08,
    CMPBE_AUX_QUAL_CENTROID      = 0x10,
    CMPBE_AUX_QUAL_SAMPLE        = 0x20,
    CMPBE_AUX_QUAL_STORAGE_MASK  = 0x38
} cmpbe_chunk_TPGE_aux_qualifier;

/* Values from r54p3 YAML bitset traits (0x2239a90): bits 0x1..0x800 */
typedef enum {
    CMPBE_SYM_FLAG_NONE       = 0x000,
    CMPBE_SYM_FLAG_UNIFORM    = 0x024,
    CMPBE_SYM_FLAG_BUFFER_VAR = 0x824
} cmpbe_chunk_symbol_flags;

/* Values from r54p3 YAML enum traits (0x2239d90): 0..0x58 (85 entries) */
typedef enum {
    CMPBE_SEMANTICS_NONE = 0x0
} cmpbe_chunk_SYMB_semantics;

/* Values from r54p3 YAML bitset traits (0x223cdf0): 1, 2, 3, 4, 8, 0xc */
typedef enum {
    CMPBE_LAYOUT_STD140 = 0x0,
    CMPBE_LAYOUT_STD430 = 0x1
} cmpbe_chunk_TPIB_layout;

/* Values from r54p3 YAML enum traits (0x223d000) & SPIR_MBS2Helper::get_tpib_kind (0x1dfa6a0): 0..4 */
typedef enum {
    CMPBE_KIND_UNIFORM_BLOCK = 0x0,
    CMPBE_KIND_STORAGE_BLOCK = 0x1,
    CMPBE_KIND_INPUT_BLOCK   = 0x2,
    CMPBE_KIND_OUTPUT_BLOCK  = 0x3,
    CMPBE_KIND_4             = 0x4
} cmpbe_chunk_TPIB_kind;

/* Values from r54p3 YAML enum traits (0x223d9d0) & SPIR2LIR::get_cmpbe_addrspace_from_tppo (0x1de41e0): 0..3 */
typedef enum {
    CMPBE_TPPO_ADDRSPACE_PRIVATE   = 0x0,
    CMPBE_TPPO_ADDRSPACE_CONSTANT  = 0x1,
    CMPBE_TPPO_ADDRSPACE_WORKGROUP = 0x2,
    CMPBE_TPPO_ADDRSPACE_GLOBAL    = 0x3
} cmpbe_chunk_TPPO_address_space;

/* Values from r54p3 YAML bitset traits (0x223fb20): bits 0x1, 0x2, 0x4 */
typedef enum {
    CMPBE_PREPASS_FLAG_NONE = 0x0
} cmpbe_chunk_FSHA_prepass_flags;

/* Values from r54p3 YAML bitset traits (0x223fc70): bits 0x1..0x40000000 */
typedef enum {
    CMPBE_FSHA_FLAG_COMPUTE = 0x42000,
    CMPBE_FSHA_FLAG_AUX_ALT = 0x20042000
} cmpbe_chunk_FSHA_flags;

/* Values from r54p3 YAML bitset traits (0x2240410): 1, 4, 0x40, 0x80, 0x100, 0x400, 0x800, 0x1000, 0x2000, 0x4000 */
typedef enum {
    CMPBE_VARIANT_DEFAULT = 0x00,
    CMPBE_VARIANT_ASTC    = 0x80
} cmpbe_chunk_FSHA_variant;

typedef enum {
    CMPBE_MBS2_RESERVED_DEFAULT = 0x0
} cmpbe_chunk_MBS2_reserved_enum;

/* Values from r54p3 YAML enum traits (0x223e770): 2, 3, 4, 0xd */
typedef enum {
    CMPBE_RELO_KIND_2   = 0x2,
    CMPBE_RELO_KIND_3   = 0x3,
    CMPBE_RELO_KIND_4   = 0x4,
    CMPBE_RELO_KIND_0xD = 0xd
} cmpbe_chunk_RELO_kind;

/* Values from r54p3 YAML enum traits (0x2241e10): 0..8 */
typedef enum {
    CMPBE_REG_MODE_0 = 0x0,
    CMPBE_REG_MODE_1 = 0x1,
    CMPBE_REG_MODE_2 = 0x2,
    CMPBE_REG_MODE_3 = 0x3,
    CMPBE_REG_MODE_4 = 0x4,
    CMPBE_REG_MODE_5 = 0x5,
    CMPBE_REG_MODE_6 = 0x6,
    CMPBE_REG_MODE_7 = 0x7,
    CMPBE_REG_MODE_8 = 0x8
} cmpbe_chunk_Register_mode;

typedef uint32_t cmpbe_chunk_TPMD_flags;      /* bit 0 (0x1) = invariant */
typedef uint32_t cmpbe_chunk_KPAR_flags;
typedef uint32_t cmpbe_chunk_BLDM_flags;      /* bits 0x1..0x4000 */
typedef uint32_t cmpbe_chunk_SLOC_flags;
typedef uint32_t cmpbe_chunk_VLKN_depth_hint;
typedef uint32_t cmpbe_chunk_ACHN_kind;       /* 1..6 */
typedef uint32_t cmpbe_chunk_MTES_primitive_mode;  /* 0..3 */
typedef uint32_t cmpbe_chunk_MTES_vertex_spacing;  /* 0..3 */
typedef uint32_t cmpbe_chunk_MTES_ordering;
typedef uint32_t cmpbe_chunk_MGEO_input_primitive_mode;  /* 1, 2, 3, 4, 6 */
typedef uint32_t cmpbe_chunk_MGEO_output_primitive_mode;

/* size: 0x0c (12 bytes) */
typedef struct cmpbe_chunk_VEHW {
    cmpbe_chunk_VEHW_core core;      /* +0x00 */
    uint32_t major;                  /* +0x04 (was reserved1) */
    uint32_t minor;                  /* +0x08 (was reserved2) */
} cmpbe_chunk_VEHW;

/* size: 0x04 (4 bytes) */
typedef struct cmpbe_chunk_VELA {
    cmpbe_chunk_VELA_lang lang;      /* +0x00 */
} cmpbe_chunk_VELA;

/* size: 0x10 (16 bytes) */
typedef struct cmpbe_chunk_STRI {
    uint32_t size;                   /* +0x00 (was length) */
    const uint8_t* data;             /* +0x08 (was string_data) */
} cmpbe_chunk_STRI;

/* size: 0x10 (16 bytes) */
typedef struct cmpbe_chunk_BINS {
    uint32_t size;                   /* +0x00 */
    const uint8_t* data;             /* +0x08 */
} cmpbe_chunk_BINS;

/* size: 0x10 (16 bytes) */
typedef struct cmpbe_chunk_SSYM {
    uint32_t count;                  /* +0x00 */
    const struct cmpbe_chunk_SYMB* symb; /* +0x08 (was symbols) */
} cmpbe_chunk_SSYM;

/* size: 0x10 (16 bytes) */
typedef struct cmpbe_chunk_UBUF {
    uint32_t count;                  /* +0x00 */
    const uint32_t* sizes;           /* +0x08 (was const void* data) */
} cmpbe_chunk_UBUF;

/* size: 0x04 (4 bytes) */
typedef struct cmpbe_chunk_TPAC {
    uint32_t offset;                 /* +0x00 */
} cmpbe_chunk_TPAC;

/* size: 0x1c (28 bytes) */
typedef struct cmpbe_chunk_TPGE {
    cmpbe_chunk_TPGE_scalar_type scalar_type;     /* +0x00 */
    uint8_t vector_size;                          /* +0x04 (was component_count) */
    cmpbe_chunk_TPGE_scalar_size scalar_size;     /* +0x08 */
    cmpbe_chunk_TPGE_precision precision;         /* +0x0c */
    uint32_t stride;                              /* +0x10 (was alignment) */
    cmpbe_chunk_TPGE_aux_qualifier aux_qualifier; /* +0x14 */
    uint8_t component;                            /* +0x18 (was reserved) */
} cmpbe_chunk_TPGE;

/* size: 0x20 (32 bytes) */
typedef struct cmpbe_chunk_TPMA {
    uint8_t nof_columns;             /* +0x00 (was columns) */
    b8 is_row_major;                 /* +0x01 (was row_major) */
    uint16_t major_stride;           /* +0x02 (was stride) */
    cmpbe_chunk_TPGE column_type;    /* +0x04 (was base_type) */
} cmpbe_chunk_TPMA;

/* size: 0x04 (4 bytes) */
typedef struct cmpbe_chunk_TPQU {
    cmpbe_chunk_TPGE_scalar_size scalar_size; /* +0x00 */
} cmpbe_chunk_TPQU;

/* size: 0x04 (4 bytes) */
typedef struct cmpbe_chunk_TPPR {
    cmpbe_chunk_TPGE_scalar_size scalar_size; /* +0x00 */
} cmpbe_chunk_TPPR;

/* size: 0x04 (4 bytes) */
typedef struct cmpbe_chunk_TPPW {
    cmpbe_chunk_TPGE_scalar_size scalar_size; /* +0x00 */
} cmpbe_chunk_TPPW;

/* size: 0x04 (4 bytes) */
typedef struct cmpbe_chunk_TPSA {
    uint32_t kind;                   /* +0x00 (was dim_flags) */
} cmpbe_chunk_TPSA;

/* size: 0x08 (8 bytes) */
typedef struct cmpbe_chunk_TPMD {
    cmpbe_chunk_TPMD_flags flags;    /* +0x00 (bit 0 = invariant) */
    uint8_t unk4;                    /* +0x04 (r54p3 YAML key '354') */
    uint8_t unk5;                    /* +0x05 (r54p3 YAML key '355') */
    uint8_t unk6;                    /* +0x06 (r54p3 YAML key '356') */
} cmpbe_chunk_TPMD;

/* size: 0x60 (96 bytes) */
typedef struct cmpbe_chunk_TYPE {
    const struct cmpbe_chunk_TPAC* tpac; /* +0x00 (was unknown0) */
    const struct cmpbe_chunk_TPAR* tpar; /* +0x08 */
    const struct cmpbe_chunk_TPGE* tpge; /* +0x10 */
    const struct cmpbe_chunk_TPIB* tpib; /* +0x18 */
    const struct cmpbe_chunk_TPMA* tpma; /* +0x20 (was unknown4) */
    const struct cmpbe_chunk_TPPO* tppo; /* +0x28 (was unknown5) */
    const struct cmpbe_chunk_TPQU* tpqu; /* +0x30 (was unknown6) */
    const struct cmpbe_chunk_TPPR* tppr; /* +0x38 (was unknown7) */
    const struct cmpbe_chunk_TPPW* tppw; /* +0x40 (was unknown8) */
    const struct cmpbe_chunk_TPSA* tpsa; /* +0x48 (was unknown9) */
    const struct cmpbe_chunk_TPST* tpst; /* +0x50 */
    const struct cmpbe_chunk_TPMD* tpmd; /* +0x58 (was unknown11) */
} cmpbe_chunk_TYPE;

/* size: 0x68 (104 bytes) */
typedef struct cmpbe_chunk_TPAR {
    uint32_t array_size;             /* +0x00 */
    cmpbe_chunk_TYPE element_type;   /* +0x08 */
} cmpbe_chunk_TPAR;

/* size: 0x68 (104 bytes) */
typedef struct cmpbe_chunk_TPPO {
    cmpbe_chunk_TPGE_scalar_size scalar_size;     /* +0x00 */
    cmpbe_chunk_TPPO_address_space address_space; /* +0x04 */
    cmpbe_chunk_TYPE pointed_type;                /* +0x08 */
} cmpbe_chunk_TPPO;

/* size: 0x20 (32 bytes) */
typedef struct cmpbe_chunk_TPIB {
    cmpbe_chunk_TPIB_layout layout;          /* +0x00 */
    cmpbe_chunk_TPIB_kind kind;              /* +0x04 */
    uint32_t stride;                         /* +0x08 (was block_size) */
    uint32_t nof_members;                    /* +0x0c (was member_count) */
    const struct cmpbe_chunk_TPSE* tpse;     /* +0x10 (was members) */
    const struct cmpbe_chunk_STRI* instance_name; /* +0x18 (was reserved) */
} cmpbe_chunk_TPIB;

/* size: 0x20 (32 bytes) */
typedef struct cmpbe_chunk_TPST {
    uint32_t stride;                         /* +0x00 (was size) */
    uint32_t nof_members;                    /* +0x04 (was member_count) */
    cmpbe_chunk_STRI type_name;              /* +0x08 (was tag) */
    const struct cmpbe_chunk_TPSE* tpse;     /* +0x18 (was members) */
} cmpbe_chunk_TPST;

/* size: 0x88 (136 bytes) */
typedef struct cmpbe_chunk_TPSE {
    cmpbe_chunk_STRI name;                   /* +0x00 */
    uint32_t offset;                         /* +0x10 */
    uint16_t xfb_offset;                     /* +0x14 (was binding) */
    cmpbe_chunk_symbol_flags flags;          /* +0x18 */
    uint16_t location;                       /* +0x1c */
    cmpbe_chunk_SYMB_semantics semantics;    /* +0x20 */
    cmpbe_chunk_TYPE type;                   /* +0x28 */
} cmpbe_chunk_TPSE;

/* size: 0x08 (8 bytes) */
typedef struct cmpbe_chunk_RLOC {
    uint16_t sym_idx;                /* +0x00 (was segment) */
    uint16_t reg_offset;             /* +0x02 (was offset; used in test_moc_roundtrip_compute.cpp) */
    uint16_t uni_offset;             /* +0x04 (was type) */
    uint16_t size;                   /* +0x06 */
} cmpbe_chunk_RLOC;

/* size: 0x0c (12 bytes) */
typedef struct cmpbe_chunk_RELO {
    uint32_t offset;                 /* +0x00 */
    cmpbe_chunk_RELO_kind kind;      /* +0x04 */
    uint8_t section_idx;             /* +0x08 */
} cmpbe_chunk_RELO;

/* size: 0x10 (16 bytes) */
typedef struct cmpbe_chunk_FIXP {
    uint32_t kind;                   /* +0x00 */
    uint32_t offset;                 /* +0x04 */
    uint32_t value;                  /* +0x08 */
    b8 needs_high;                   /* +0x0c */
} cmpbe_chunk_FIXP;

/* size: 0x10 (16 bytes) */
typedef struct cmpbe_chunk_SINI {
    uint32_t count;                  /* +0x00 */
    const uint32_t* values;          /* +0x08 */
} cmpbe_chunk_SINI;

/* size: 0x10 (16 bytes in r54p3) */
typedef struct cmpbe_chunk_SYMD {
    uint32_t size;                   /* +0x00 */
    uint32_t alignment;              /* +0x04 */
    uint32_t addressspace;           /* +0x08 */
    uint8_t sym_hidden;              /* +0x0c */
    uint8_t sym_restrict;            /* +0x0d */
    b8 may_read_yuv;                 /* +0x0e */
    uint8_t force_invariance;        /* +0x0f */
} cmpbe_chunk_SYMD;

/* size: 0xc8 (200 bytes) */
typedef struct cmpbe_chunk_SYMB {
    cmpbe_chunk_STRI name;                   /* +0x00 */
    cmpbe_chunk_symbol_flags flags;          /* +0x10 */
    cmpbe_chunk_SYMB_semantics semantics;    /* +0x14 */
    uint8_t rack_index;                      /* +0x18 (was binding_set) */
    uint32_t address;                        /* +0x1c (was binding_id) */
    uint16_t binding;                        /* +0x20 (was location) */
    uint16_t location;                       /* +0x22 (was component) */
    cmpbe_chunk_TYPE type;                   /* +0x28 */
    uint32_t nof_rloc;                       /* +0x88 (was rloc_count) */
    const struct cmpbe_chunk_RLOC* rloc;     /* +0x90 (was relocations) */
    uint32_t nof_relo;                       /* +0x98 (was nested_count) */
    const struct cmpbe_chunk_RELO* relo;     /* +0xa0 (was nested_symbols) */
#ifdef CMPBE_LAYOUT_R54P3
    uint32_t nof_fixp;                       /* +0xa8 (r54p3 DDK) */
    const struct cmpbe_chunk_FIXP* fixp;     /* +0xb0 (r54p3 DDK) */
#endif
    const struct cmpbe_chunk_SINI* sini;     /* +0xb8 in r54p3 (was reserved1) */
    const struct cmpbe_chunk_SYMD* symd;     /* +0xc0 in r54p3 (was reserved2) */
} cmpbe_chunk_SYMB;

/* size: 0x08 (8 bytes) */
typedef struct cmpbe_chunk_FCST {
    uint16_t size;                   /* +0x00 */
    uint16_t reg_offset;             /* +0x02 (was constant_id; used in test_moc_roundtrip_compute.cpp) */
    uint32_t value;                  /* +0x04 */
} cmpbe_chunk_FCST;

/* size: 0x10 (16 bytes) */
typedef struct cmpbe_chunk_OBJC {
    uint32_t size;                   /* +0x00 */
    const uint8_t* byte_code;        /* +0x08 (alias: data) */
} cmpbe_chunk_OBJC;

/* size: 0x20 (32 bytes) */
typedef struct cmpbe_chunk_DBUG {
    uint32_t nof_rloc;                       /* +0x00 */
    const struct cmpbe_chunk_RLOC* rloc;     /* +0x08 */
    uint32_t nof_fcst;                       /* +0x10 */
    const struct cmpbe_chunk_FCST* fcst;     /* +0x18 */
} cmpbe_chunk_DBUG;

/* size: 0x10 (16 bytes) */
typedef struct cmpbe_chunk_GREL {
    uint32_t nof_global_relo;                /* +0x00 */
    const struct cmpbe_chunk_RELO* relo;     /* +0x08 */
} cmpbe_chunk_GREL;

/* size: 0x20 (32 bytes in r54p3; r44p1 also had return_addr) */
typedef struct cmpbe_chunk_SPDf {
    uint32_t rt_count;                               /* +0x00 (was count) */
    const uint8_t* rt_index;                         /* +0x08 (was locations) */
    const cmpbe_chunk_TPGE_scalar_size* size;        /* +0x10 (was sizes) */
    const cmpbe_chunk_TPGE_scalar_type* type;        /* +0x18 (was types) */
} cmpbe_chunk_SPDf;

/* size: 0x06 (6 bytes) */
typedef struct cmpbe_chunk_PDSC {
    uint16_t varying0;               /* +0x00 */
    uint16_t varying1;               /* +0x02 */
    uint8_t primary_shader_wait;     /* +0x04 */
} cmpbe_chunk_PDSC;

/* size: 0x10 (16 bytes) */
typedef struct cmpbe_chunk_VLKN {
    uint16_t used_set_bitmask;       /* +0x00 */
    uint16_t stack_layers;           /* +0x02 */
    uint8_t dynamic_outputs;         /* +0x04 */
    const cmpbe_chunk_VLKN_depth_hint* depth_hint; /* +0x08 (cmpbe_chunk_VDEP*) */
} cmpbe_chunk_VLKN;

/* size: 0x50 (80 bytes) */
typedef struct cmpbe_chunk_FNMD {
    uint32_t nof_tvar;                       /* +0x00 */
    const cmpbe_chunk_FSHA_variant* tvar;    /* +0x08 */
    uint8_t is_other_function;               /* +0x10 */
    uint8_t needs_structured_cfg;            /* +0x11 */
    uint8_t is_vertex_piece;                 /* +0x12 */
    uint8_t is_simd_state_caller;            /* +0x13 */
    uint8_t is_callable_shaders_caller;      /* +0x14 */
    uint8_t is_discarded_function;           /* +0x15 */
    cmpbe_chunk_Register_mode reg_mode;      /* +0x18 */
    uint8_t unk1c;                           /* +0x1c */
    cmpbe_chunk_STRI pilot_name;             /* +0x20 */
    cmpbe_chunk_STRI late_codegen;           /* +0x30 */
    uint32_t attrib_mask;                    /* +0x40 */
    uint32_t max_register_pressure;          /* +0x44 */
    uint32_t mrtt;                           /* +0x48 */
} cmpbe_chunk_FNMD;

/* size: 0x38 (56 bytes) */
typedef struct cmpbe_chunk_SLOC {
    uint32_t nof_bins;                       /* +0x00 */
    const struct cmpbe_chunk_BINS* bins;     /* +0x08 */
    uint32_t nof_u32_1;                      /* +0x10 */
    const uint32_t* u32_1;                   /* +0x18 */
    cmpbe_chunk_SLOC_flags flags;            /* +0x20 */
    const void* slin;                        /* +0x28 (cmpbe_chunk_SLIN*) */
    uint64_t reserved;                       /* +0x30 */
} cmpbe_chunk_SLOC;

/* size: 0x70 (112 bytes) */
typedef struct cmpbe_chunk_EBIN {
    uint16_t rt_num;                         /* +0x00 (was id) */
    uint16_t aux_idx;                        /* +0x02 (was pipeline_stage) */
    cmpbe_chunk_FSHA_prepass_flags prepass_flags; /* +0x04 */
    uint32_t data_section_offset;            /* +0x08 (was uncompressed_code_size) */
    uint16_t nof_reg;                        /* +0x0c (was register_count) */
    uint16_t nof_rmu;                        /* +0x0e (was local_thread_limit) */
    uint32_t tls_size;                       /* +0x10 (was scratch_allocation_bytes) */
    cmpbe_chunk_FSHA_flags flags;            /* +0x14 (was fsha_flags) */
    cmpbe_chunk_FSHA_variant variant;        /* +0x18 */
    const cmpbe_chunk_STRI* entrypoint_name; /* +0x20 (was unk0) */
    const cmpbe_chunk_DBUG* dbug;            /* +0x28 (was unk1) */
    const cmpbe_chunk_GREL* grel;            /* +0x30 (was unk2) */
    const cmpbe_chunk_SPDf* spdf;            /* +0x38 (was unk3) */
    const cmpbe_chunk_PDSC* pdsc;            /* +0x40 (was unk4) */
#ifdef CMPBE_LAYOUT_R54P3
    const cmpbe_chunk_VLKN* vlkn;            /* +0x48 (r54p3 DDK) */
#endif
    cmpbe_chunk_OBJC bytecode;               /* +0x50 (DDK: objc) */
    const cmpbe_chunk_FNMD* fnmd;            /* +0x60 (was unk5) */
    const cmpbe_chunk_SLOC* sloc;            /* +0x68 (was unk6) */
} cmpbe_chunk_EBIN;

/* ============================================================================
 * Common & Stage-Specific Chunks
 * ============================================================================ */

/* size: 0x04 (4 bytes) */
typedef struct cmpbe_chunk_CXFE {
    uint16_t buffer;                 /* +0x00 */
    uint16_t offset;                 /* +0x02 */
} cmpbe_chunk_CXFE;

/* size: 0x18 (24 bytes; in r54p3 CXFB is 0x8 bytes: stride_0..3) */
typedef struct cmpbe_chunk_CXFB {
    uint16_t stride_0;               /* +0x00 */
    uint16_t stride_1;               /* +0x02 */
    uint16_t stride_2;               /* +0x04 */
    uint16_t stride_3;               /* +0x06 */
} cmpbe_chunk_CXFB;

/* size: 0xa0 (160 bytes) */
typedef struct cmpbe_chunk_CMMN {
    cmpbe_chunk_VELA vela;                   /* +0x00 (was language) */
    cmpbe_chunk_SSYM sinp;                   /* +0x08 (was ssym_2; symb.sinp) */
    cmpbe_chunk_SSYM sout;                   /* +0x18 (was ssym_3; symb.sout) */
    cmpbe_chunk_SSYM suni;                   /* +0x28 (was ssym_24; symb.suni) */
    cmpbe_chunk_SSYM simg;                   /* +0x38 (was ssym_25; symb.simg) */
    cmpbe_chunk_SSYM ssmp;                   /* +0x48 (was ssym_26; symb.ssmp) */
    cmpbe_chunk_SSYM s3bo;                   /* +0x58 (was ssym_43; symb.s3bo) */
    cmpbe_chunk_UBUF ubuf;                   /* +0x68 (was ubuf_44) */
    uint16_t epilogue_jump_index;            /* +0x78 (was compilation_flags) */
#ifdef CMPBE_LAYOUT_R54P3
    uint16_t written_pos_fifo_attrs;         /* +0x7a (r54p3 DDK) */
#endif
    uint32_t nof_fcst;                       /* +0x7c (was constant_count) */
    const struct cmpbe_chunk_FCST* fcst;     /* +0x80 (was constants) */
    uint32_t nof_ebin;                       /* +0x88 (was binary_count) */
    const struct cmpbe_chunk_EBIN* ebin;     /* +0x90 (was binaries) */
    const struct cmpbe_chunk_CXFB* cxfb;     /* +0x98 (was common_tail) */
} cmpbe_chunk_CMMN;

/* size: 0x0c (12 bytes) */
typedef struct cmpbe_chunk_KWGS {
    uint32_t x;                      /* +0x00 (was workgroup_x) */
    uint32_t y;                      /* +0x04 (was workgroup_y) */
    uint32_t z;                      /* +0x08 (was workgroup_z) */
} cmpbe_chunk_KWGS;

/* size: 0x28 (40 bytes) */
typedef struct cmpbe_chunk_KPAR {
    cmpbe_chunk_STRI declared_name;          /* +0x00 */
    cmpbe_chunk_STRI uniform_name;           /* +0x10 */
    cmpbe_chunk_KPAR_flags flags;            /* +0x20 */
} cmpbe_chunk_KPAR;

/* size: 0x30 (48 bytes) */
typedef struct cmpbe_chunk_KERN {
    cmpbe_chunk_STRI name;                   /* +0x00 (was entry_point_name) */
    uint32_t nof_kpar;                       /* +0x10 (was kernel_flags) */
    const struct cmpbe_chunk_KPAR* kpar;     /* +0x18 (was reserved1) */
    uint32_t workgroup_local_size;           /* +0x20 (was argument_count) */
#ifndef CMPBE_LAYOUT_R54P3
    cmpbe_chunk_MBS2_reserved_enum reserved_enum; /* r55p0 malioc */
#endif
    const cmpbe_chunk_KWGS* workgroup_size;  /* +0x28 (was workgroup_sizes; custom_ext0/1 were here) */
} cmpbe_chunk_KERN;

/* size: 0x90 (144 bytes) */
typedef struct cmpbe_chunk_CBLE {
    cmpbe_chunk_SSYM suni;                   /* +0x00 */
    cmpbe_chunk_UBUF ubuf;                   /* +0x10 */
    cmpbe_chunk_EBIN ebin;                   /* +0x20 */
} cmpbe_chunk_CBLE;

/* size: 0xd0 (208 bytes) */
typedef struct cmpbe_chunk_CCOM {
    cmpbe_chunk_CMMN cmmn;                   /* +0x00 (was common) */
    cmpbe_chunk_KERN kern;                   /* +0xa0 (was kernel) */
#ifndef CMPBE_LAYOUT_R54P3
    const void* custom_ext0;                 /* r55p0 malioc */
    const void* custom_ext1;                 /* r55p0 malioc */
#endif
} cmpbe_chunk_CCOM;

/* size: 0x04 (4 bytes) */
typedef struct cmpbe_chunk_SPAS {
    uint16_t resource_read_mask;             /* +0x00 (was unk0) */
    uint16_t resource_write_mask;            /* +0x02 (was unk1) */
} cmpbe_chunk_SPAS;

/* size: 0x01 (1 byte) */
typedef struct cmpbe_chunk_VMVW {
    uint8_t num_views;                       /* +0x00 */
} cmpbe_chunk_VMVW;

/* size: 0x02 (2 bytes) */
typedef struct cmpbe_chunk_VRSM {
    uint16_t potential_vrs_modes_mask;       /* +0x00 */
} cmpbe_chunk_VRSM;

/* size: 0xb0 (176 bytes) */
typedef struct cmpbe_chunk_CVER {
    cmpbe_chunk_CMMN cmmn;                   /* +0x00 (was common) */
    const struct cmpbe_chunk_VMVW* vmvw;     /* +0xa0 (was unk0) */
    const struct cmpbe_chunk_VRSM* vrsm;     /* +0xa8 (was unk1) */
} cmpbe_chunk_CVER;

/* size: 0x78 (120 bytes) */
typedef struct cmpbe_chunk_OUTV {
    cmpbe_chunk_STRI name;                   /* +0x00 */
    cmpbe_chunk_TYPE type;                   /* +0x10 */
    uint16_t location;                       /* +0x70 */
    uint16_t nb_locations_used;              /* +0x72 */
} cmpbe_chunk_OUTV;

/* size: 0x10 (16 bytes) */
typedef struct cmpbe_chunk_FOTV {
    uint8_t nb_outputs;                      /* +0x00 */
    const struct cmpbe_chunk_OUTV* outputs;  /* +0x08 */
} cmpbe_chunk_FOTV;

/* size: 0x01 (1 byte) */
typedef struct cmpbe_chunk_LCAL {
    uint8_t local_storage_size;              /* +0x00 */
} cmpbe_chunk_LCAL;

/* size: 0x04 (4 bytes) */
typedef struct cmpbe_chunk_BLDM {
    cmpbe_chunk_BLDM_flags flags;            /* +0x00 */
} cmpbe_chunk_BLDM;

/* size: 0x28 (40 bytes) */
typedef struct cmpbe_chunk_ACHN {
    cmpbe_chunk_STRI sampler_name;           /* +0x00 */
    cmpbe_chunk_STRI uniform_name;           /* +0x10 */
    cmpbe_chunk_ACHN_kind kind;              /* +0x20 */
    uint8_t rt;                              /* +0x24 */
    uint8_t sampler_component;               /* +0x25 */
    uint8_t uniform_component;               /* +0x26 */
} cmpbe_chunk_ACHN;

/* size: 0x10 (16 bytes) */
typedef struct cmpbe_chunk_AINF {
    uint32_t count;                          /* +0x00 */
    const struct cmpbe_chunk_ACHN* achn;     /* +0x08 */
} cmpbe_chunk_AINF;

/* size: 0xc8 (200 bytes in r54p3) */
typedef struct cmpbe_chunk_CFRA {
    cmpbe_chunk_CMMN cmmn;                   /* +0x00 (was common) */
    const struct cmpbe_chunk_FOTV* output_vars; /* +0xa0 (was unk0) */
    const struct cmpbe_chunk_LCAL* lcal;     /* +0xa8 (was unk1) */
    const struct cmpbe_chunk_BLDM* bldm;     /* +0xb0 (was unk2) */
    const struct cmpbe_chunk_AINF* alpha_info; /* +0xb8 (was unk3) */
    const struct cmpbe_chunk_SPAS* spas;     /* +0xc0 */
} cmpbe_chunk_CFRA;

/* size: 0x14 (20 bytes) */
typedef struct cmpbe_chunk_MTES {
    uint32_t n_vertices;                         /* +0x00 */
    cmpbe_chunk_MTES_primitive_mode primitive_mode; /* +0x04 */
    cmpbe_chunk_MTES_vertex_spacing vertex_spacing; /* +0x08 */
    cmpbe_chunk_MTES_ordering ordering;          /* +0x0c */
    b8 point_mode;                               /* +0x10 */
} cmpbe_chunk_MTES;

/* size: 0xb8 (184 bytes) */
typedef struct cmpbe_chunk_CTSC {
    cmpbe_chunk_CMMN cmmn;                   /* +0x00 */
    uint32_t workgroup_local_size;           /* +0xa0 */
    cmpbe_chunk_MTES mtes;                   /* +0xa4 */
} cmpbe_chunk_CTSC;

/* size: 0xb8 (184 bytes) */
typedef struct cmpbe_chunk_CTSE {
    cmpbe_chunk_CMMN cmmn;                   /* +0x00 */
    cmpbe_chunk_MTES mtes;                   /* +0xa0 */
} cmpbe_chunk_CTSE;

/* size: 0x0c (12 bytes) */
typedef struct cmpbe_chunk_MGEO {
    uint16_t n_invocations;                          /* +0x00 */
    uint16_t max_vertices;                           /* +0x02 */
    cmpbe_chunk_MGEO_input_primitive_mode input_primitive_mode;   /* +0x04 */
    cmpbe_chunk_MGEO_output_primitive_mode output_primitive_mode; /* +0x08 */
} cmpbe_chunk_MGEO;

/* size: 0x18 (24 bytes) */
typedef struct cmpbe_chunk_GSVC {
    uint16_t nof_layers;                     /* +0x00 */
    const uint32_t* index_counts;            /* +0x08 */
    uint32_t nof_vertices;                   /* +0x10 */
    uint32_t primitives_generated;           /* +0x14 */
} cmpbe_chunk_GSVC;

/* size: 0xc0 (192 bytes) */
typedef struct cmpbe_chunk_CGEO {
    cmpbe_chunk_CMMN cmmn;                   /* +0x00 */
    cmpbe_chunk_MGEO mgeo;                   /* +0xa0 */
    const struct cmpbe_chunk_GSVC* gsvc;     /* +0xb0 */
    const struct cmpbe_chunk_VRSM* vrsm;     /* +0xb8 */
} cmpbe_chunk_CGEO;

/* size: 0x10 (16 bytes) */
typedef struct cmpbe_chunk_SSRC {
    uint32_t extra;                          /* +0x00 */
    uint32_t nb_bins;                        /* +0x04 */
    const struct cmpbe_chunk_BINS* shader_sources; /* +0x08 */
} cmpbe_chunk_SSRC;

/* size: 0x06 (6 bytes) */
typedef struct cmpbe_chunk_PLOC {
    uint16_t reg_offset;                     /* +0x00 */
    uint16_t size;                           /* +0x02 */
    uint16_t orig_offset;                    /* +0x04 */
} cmpbe_chunk_PLOC;

/* size: 0x30 (48 bytes) */
typedef struct cmpbe_chunk_MOMD {
    uint8_t prealloc_uni_buffers;                /* +0x00 */
    uint8_t prealloc_fau;                        /* +0x01 */
    uint8_t ble_preserve_registers;              /* +0x02 */
    uint8_t essl_on_spirv;                       /* +0x03 */
    uint8_t has_callable_blend_shaders;          /* +0x04 */
    uint8_t atest_wait_on_6_7;                   /* +0x05 */
    uint8_t gfx_hal;                             /* +0x06 */
    uint8_t robustness;                          /* +0x07 */
    uint8_t per_sample_shading;                  /* +0x08 */
    uint8_t setup_fau_in_pilot;                  /* +0x09 */
    uint8_t has_simd_call;                       /* +0x0a */
    uint8_t has_simd_with_static_stack_frame;    /* +0x0b */
    uint8_t unk_0c[9];                           /* +0x0c..+0x14 (r54p3 YAML keys) */
    uint32_t max_prealloc_addr;                  /* +0x18 */
    uint16_t unk_1c;                             /* +0x1c..+0x1d (2x uint8_t in r54p3) */
    uint32_t nof_ploc;                           /* +0x20 */
    const struct cmpbe_chunk_PLOC* ploc;         /* +0x28 */
} cmpbe_chunk_MOMD;

// Top-Level MBS2 Container

/* size: 0x60 (96 bytes; note: original header's extension_tree8 does not exist) */
typedef struct cmpbe_chunk_MBS2 {
    uint32_t version;                        /* +0x00 */
    cmpbe_chunk_VEHW vehw;                   /* +0x04 (was hardware) */
    const struct cmpbe_chunk_CBLE* cble;     /* +0x10 (was interface_descriptor) */
    const struct cmpbe_chunk_CCOM* ccom;     /* +0x18 (was compilation_command) */
    const struct cmpbe_chunk_CFRA* cfra;     /* +0x20 (was extension_tree0) */
    const struct cmpbe_chunk_CVER* cver;     /* +0x28 (was extension_tree1) */
    const struct cmpbe_chunk_CTSC* ctsc;     /* +0x30 (was extension_tree2) */
    const struct cmpbe_chunk_CTSE* ctse;     /* +0x38 (was extension_tree3) */
    const struct cmpbe_chunk_CGEO* cgeo;     /* +0x40 (was extension_tree4) */
    const struct cmpbe_chunk_SSRC* ssrc;     /* +0x48 (was extension_tree5) */
    const struct cmpbe_chunk_BINS* variants; /* +0x50 (was extension_tree6) */
    const struct cmpbe_chunk_MOMD* momd;     /* +0x58 (was extension_tree7) */
#ifndef CMPBE_LAYOUT_R54P3
    const void* extension_tree8;             /* r55p0 malioc */
#endif
} cmpbe_chunk_MBS2;

#ifdef __cplusplus
}
#endif

#endif /* CMPBE_CHUNKS_H */

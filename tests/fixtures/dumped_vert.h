
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
static const cmpbe_chunk_VEHW VEHW_0 =
{
	(cmpbe_chunk_VEHW_core) 0x1c,
	(uint32_t) 0x0,
	(uint32_t) 0x0,
};
static const cmpbe_chunk_VELA VELA_1 =
{
	(cmpbe_chunk_VELA_lang) 0x7,
};
static const uint8_t b_2[] = "_25_inPos";
static const cmpbe_chunk_STRI STRI_2 = { 9, (uint8_t*)b_2 };
static const cmpbe_chunk_TPGE TPGE_3 =
{
	(cmpbe_chunk_TPGE_scalar_type) 0x1,
	(uint8_t) 0x3,
	(cmpbe_chunk_TPGE_scalar_size) 0x2,
	(cmpbe_chunk_TPGE_precision) 0x1,
	(uint32_t) 0x1,
	(cmpbe_chunk_TPGE_aux_qualifier) 0x9,
	(uint8_t) 0x0,
};
static const cmpbe_chunk_TYPE TYPE_4 =
{
	NULL,
	NULL,
	(cmpbe_chunk_TPGE*)&TPGE_3,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
};
static const cmpbe_chunk_SYMB SYMB_2 =
{
	STRI_2,
	(cmpbe_chunk_symbol_flags) 0x26,
	(cmpbe_chunk_SYMB_semantics) 0x0,
	(uint8_t) 0x1,
	(uint32_t) 0x0,
	(uint16_t) 0xffff,
	(uint16_t) 0x0,
	TYPE_4,
	(uint32_t) 0x0,
	NULL,
	(uint32_t) 0x0,
	NULL,
	NULL,
	NULL,
};
static const uint8_t b_5[] = "_39_inUV";
static const cmpbe_chunk_STRI STRI_5 = { 8, (uint8_t*)b_5 };
static const cmpbe_chunk_TPGE TPGE_6 =
{
	(cmpbe_chunk_TPGE_scalar_type) 0x1,
	(uint8_t) 0x2,
	(cmpbe_chunk_TPGE_scalar_size) 0x2,
	(cmpbe_chunk_TPGE_precision) 0x1,
	(uint32_t) 0x1,
	(cmpbe_chunk_TPGE_aux_qualifier) 0x9,
	(uint8_t) 0x0,
};
static const cmpbe_chunk_TYPE TYPE_7 =
{
	NULL,
	NULL,
	(cmpbe_chunk_TPGE*)&TPGE_6,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
};
static const cmpbe_chunk_SYMB SYMB_5 =
{
	STRI_5,
	(cmpbe_chunk_symbol_flags) 0x26,
	(cmpbe_chunk_SYMB_semantics) 0x0,
	(uint8_t) 0x1,
	(uint32_t) 0x1,
	(uint16_t) 0xffff,
	(uint16_t) 0x1,
	TYPE_7,
	(uint32_t) 0x0,
	NULL,
	(uint32_t) 0x0,
	NULL,
	NULL,
	NULL,
};
static cmpbe_chunk_SYMB a_2[2] = { SYMB_2, SYMB_5, };
static const cmpbe_chunk_SSYM SSYM_8 =
{
	(uint32_t) 0x2,
	a_2,
};
static const uint8_t b_9[] = "gl_Position";
static const cmpbe_chunk_STRI STRI_9 = { 11, (uint8_t*)b_9 };
static const cmpbe_chunk_TPGE TPGE_10 =
{
	(cmpbe_chunk_TPGE_scalar_type) 0x1,
	(uint8_t) 0x4,
	(cmpbe_chunk_TPGE_scalar_size) 0x2,
	(cmpbe_chunk_TPGE_precision) 0x1,
	(uint32_t) 0x1,
	(cmpbe_chunk_TPGE_aux_qualifier) 0x9,
	(uint8_t) 0x0,
};
static const cmpbe_chunk_TYPE TYPE_11 =
{
	NULL,
	NULL,
	(cmpbe_chunk_TPGE*)&TPGE_10,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
};
static const cmpbe_chunk_SYMB SYMB_9 =
{
	STRI_9,
	(cmpbe_chunk_symbol_flags) 0x124,
	(cmpbe_chunk_SYMB_semantics) 0x40,
	(uint8_t) 0x3,
	(uint32_t) 0x0,
	(uint16_t) 0xffff,
	(uint16_t) 0xffff,
	TYPE_11,
	(uint32_t) 0x0,
	NULL,
	(uint32_t) 0x0,
	NULL,
	NULL,
	NULL,
};
static const uint8_t b_12[] = "_37_outUV";
static const cmpbe_chunk_STRI STRI_12 = { 9, (uint8_t*)b_12 };
static const cmpbe_chunk_TPGE TPGE_13 =
{
	(cmpbe_chunk_TPGE_scalar_type) 0x1,
	(uint8_t) 0x2,
	(cmpbe_chunk_TPGE_scalar_size) 0x2,
	(cmpbe_chunk_TPGE_precision) 0x1,
	(uint32_t) 0x1,
	(cmpbe_chunk_TPGE_aux_qualifier) 0x9,
	(uint8_t) 0x0,
};
static const cmpbe_chunk_TYPE TYPE_14 =
{
	NULL,
	NULL,
	(cmpbe_chunk_TPGE*)&TPGE_13,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
};
static const cmpbe_chunk_SYMB SYMB_12 =
{
	STRI_12,
	(cmpbe_chunk_symbol_flags) 0x126,
	(cmpbe_chunk_SYMB_semantics) 0x0,
	(uint8_t) 0x3,
	(uint32_t) 0x0,
	(uint16_t) 0xffff,
	(uint16_t) 0x0,
	TYPE_14,
	(uint32_t) 0x0,
	NULL,
	(uint32_t) 0x0,
	NULL,
	NULL,
	NULL,
};
static cmpbe_chunk_SYMB a_9[2] = { SYMB_9, SYMB_12, };
static const cmpbe_chunk_SSYM SSYM_15 =
{
	(uint32_t) 0x2,
	a_9,
};
static const uint8_t b_16[] = "_19_ubo";
static const cmpbe_chunk_STRI STRI_16 = { 7, (uint8_t*)b_16 };
static const uint8_t b_17[] = "mvp";
static const cmpbe_chunk_STRI STRI_17 = { 3, (uint8_t*)b_17 };
static const cmpbe_chunk_TPGE TPGE_18 =
{
	(cmpbe_chunk_TPGE_scalar_type) 0x1,
	(uint8_t) 0x4,
	(cmpbe_chunk_TPGE_scalar_size) 0x2,
	(cmpbe_chunk_TPGE_precision) 0x1,
	(uint32_t) 0x10,
	(cmpbe_chunk_TPGE_aux_qualifier) 0x8,
	(uint8_t) 0x0,
};
static const cmpbe_chunk_TPMA TPMA_19 =
{
	(uint8_t) 0x4,
	(b8) 0x0,
	(uint16_t) 0x10,
	TPGE_18,
};
static const cmpbe_chunk_TYPE TYPE_20 =
{
	NULL,
	NULL,
	NULL,
	NULL,
	(cmpbe_chunk_TPMA*)&TPMA_19,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
};
static const cmpbe_chunk_TPSE TPSE_17 =
{
	STRI_17,
	(uint32_t) 0x0,
	(uint16_t) 0xffff,
	(cmpbe_chunk_symbol_flags) 0x0,
	(uint16_t) 0xffff,
	(cmpbe_chunk_SYMB_semantics) 0x0,
	TYPE_20,
};
static cmpbe_chunk_TPSE a_17[1] = { TPSE_17, };
static const cmpbe_chunk_TPIB TPIB_21 =
{
	(cmpbe_chunk_TPIB_layout) 0x0,
	(cmpbe_chunk_TPIB_kind) 0x0,
	(uint32_t) 0x40,
	(uint32_t) 0x1,
	a_17,
	NULL,
};
static const cmpbe_chunk_TYPE TYPE_22 =
{
	NULL,
	NULL,
	NULL,
	(cmpbe_chunk_TPIB*)&TPIB_21,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
};
static const cmpbe_chunk_SYMB SYMB_16 =
{
	STRI_16,
	(cmpbe_chunk_symbol_flags) 0x34,
	(cmpbe_chunk_SYMB_semantics) 0x0,
	(uint8_t) 0x4,
	(uint32_t) 0x0,
	(uint16_t) 0x0,
	(uint16_t) 0xffff,
	TYPE_22,
	(uint32_t) 0x0,
	NULL,
	(uint32_t) 0x0,
	NULL,
	NULL,
	NULL,
};
static const uint8_t b_23[] = "gl_mali_ViewportTransform";
static const cmpbe_chunk_STRI STRI_23 = { 25, (uint8_t*)b_23 };
static const cmpbe_chunk_TPGE TPGE_24 =
{
	(cmpbe_chunk_TPGE_scalar_type) 0x1,
	(uint8_t) 0x4,
	(cmpbe_chunk_TPGE_scalar_size) 0x2,
	(cmpbe_chunk_TPGE_precision) 0x1,
	(uint32_t) 0x10,
	(cmpbe_chunk_TPGE_aux_qualifier) 0x8,
	(uint8_t) 0x0,
};
static const cmpbe_chunk_TYPE TYPE_25 =
{
	NULL,
	NULL,
	(cmpbe_chunk_TPGE*)&TPGE_24,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
};
static const cmpbe_chunk_TPAR TPAR_26 =
{
	(uint32_t) 0x2,
	TYPE_25,
};
static const cmpbe_chunk_TYPE TYPE_27 =
{
	NULL,
	(cmpbe_chunk_TPAR*)&TPAR_26,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
};
static const cmpbe_chunk_TPAR TPAR_28 =
{
	(uint32_t) 0x1,
	TYPE_27,
};
static const cmpbe_chunk_TYPE TYPE_29 =
{
	NULL,
	(cmpbe_chunk_TPAR*)&TPAR_28,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
};
static const cmpbe_chunk_RLOC RLOC_30 =
{
	(uint16_t) 0x1,
	(uint16_t) 0x0,
	(uint16_t) 0x0,
	(uint16_t) 0xc,
};
static const cmpbe_chunk_RLOC RLOC_31 =
{
	(uint16_t) 0x1,
	(uint16_t) 0xc,
	(uint16_t) 0x18,
	(uint16_t) 0x4,
};
static const cmpbe_chunk_RLOC RLOC_32 =
{
	(uint16_t) 0x1,
	(uint16_t) 0x10,
	(uint16_t) 0x10,
	(uint16_t) 0x8,
};
static const cmpbe_chunk_RLOC RLOC_33 =
{
	(uint16_t) 0x1,
	(uint16_t) 0x18,
	(uint16_t) 0x4,
	(uint16_t) 0x4,
};
static const cmpbe_chunk_RLOC RLOC_34 =
{
	(uint16_t) 0x1,
	(uint16_t) 0x1c,
	(uint16_t) 0xc,
	(uint16_t) 0x4,
};
static const cmpbe_chunk_RLOC RLOC_35 =
{
	(uint16_t) 0x1,
	(uint16_t) 0x20,
	(uint16_t) 0x0,
	(uint16_t) 0x4,
};
static const cmpbe_chunk_RLOC RLOC_36 =
{
	(uint16_t) 0x1,
	(uint16_t) 0x24,
	(uint16_t) 0x1c,
	(uint16_t) 0x4,
};
static cmpbe_chunk_RLOC a_30[7] = { RLOC_30, RLOC_31, RLOC_32, RLOC_33, RLOC_34, RLOC_35, RLOC_36, };
static const cmpbe_chunk_SYMB SYMB_23 =
{
	STRI_23,
	(cmpbe_chunk_symbol_flags) 0x4,
	(cmpbe_chunk_SYMB_semantics) 0x17,
	(uint8_t) 0x0,
	(uint32_t) 0x4000c,
	(uint16_t) 0xffff,
	(uint16_t) 0xffff,
	TYPE_29,
	(uint32_t) 0x7,
	a_30,
	(uint32_t) 0x0,
	NULL,
	NULL,
	NULL,
};
static cmpbe_chunk_SYMB a_16[2] = { SYMB_16, SYMB_23, };
static const cmpbe_chunk_SSYM SSYM_37 =
{
	(uint32_t) 0x2,
	a_16,
};
static const cmpbe_chunk_SSYM SSYM_38 =
{
	(uint32_t) 0x0,
	NULL,
};
static const cmpbe_chunk_SSYM SSYM_39 =
{
	(uint32_t) 0x0,
	NULL,
};
static const cmpbe_chunk_SSYM SSYM_40 =
{
	(uint32_t) 0x0,
	NULL,
};
static const cmpbe_chunk_UBUF UBUF_41 =
{
	(uint32_t) 0x0,
	NULL,
};
static const cmpbe_chunk_FCST FCST_42 =
{
	(uint16_t) 0x4,
	(uint16_t) 0x28,
	(uint32_t) 0x44800000,
};
static cmpbe_chunk_FCST a_42[1] = { FCST_42, };
static const uint8_t b_43[] = "|}\x11\x12\x04\x80""f\x00{\x0d\x00\x00\x04\x82^\x08""B\x00\x00*\x04@ax";
static const cmpbe_chunk_OBJC OBJC_43 = { 24, (uint8_t*)b_43 };
static const cmpbe_chunk_EBIN EBIN_44 =
{
	(uint16_t) 0x2,
	(uint16_t) 0x3800,
	(cmpbe_chunk_FSHA_prepass_flags) 0x0,
	(uint32_t) 0xffffffff,
	(uint16_t) 0x20,
	(uint16_t) 0xe,
	(uint32_t) 0x0,
	(cmpbe_chunk_FSHA_flags) 0x2000,
	(cmpbe_chunk_FSHA_variant) 0x81,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	OBJC_43,
	NULL,
	NULL,
};
static const uint8_t b_45[] = "|}\x01\x22\x06\x80""f\x08\x93\x01\xc0\x00\x04\xc3\xb2\x00\x8f\x00""C\x00\x00\xc3\xb2\x00\x97\x02""C\x00\x00\xc3\xb2\x00""C\x9b\x00\x00\x00\xc7\xa4\x00\x92\x01\xc0\x00\x04\xfe\xb2\x00\x8e\x00~\x00\x00\xfe\xb2\x00\x96\x02~\x00\x00\xfe\xb2\x00~\x9a\x00\x00\x00\xc6\xa4\x00\x91\x01\xc0\x00\x04\xff\xb2\x00\x8d\x00\x7f\x00\x00\xff\xb2\x00\x95\x02\x7f\x00\x00\xff\xb2\x00\x7f\x99\x00\x00\x00\xc5\xa4\x00\x90""A\xc0\x00\x04\xc1\xb2\x00\x8c@A\x00\x00\xc0\xb2\x00\x94""B@\x00\x00\xc0\xb2\x00@\x98\x00\x00\x00\xc4\xa4\x00{\x0d\x00\x00\x04\x80^\x00\x06\x07\xc0\x80\xa1\xca\xf4\x00\x04\xd5\xc0\x00\x84\xc8\xb2\x00\x05\xd5H\x00\x80\xc8\xb2\x00\x06\xd5H\x00\x80\xc8\xb2\x00\x07\xda\x03@\x80\xc9\xa4\x00HI\x03@\x00\xc8\xa4\x00H\x00\x00\x00\x00\xc8\x9c\x00\x08\xc0\x00\x00P\xc9\xa4\x00G\xc0IH\x04\xcbT\x01""F\x0b\xc0\x00\x04\xc9\xb2\x00\x09\xc0\x00\x00\x12\xc8\xa4\x00J\xc0IH\x03\xc9Q\x01I\x82\x83\x00\x00\xca\xb2\x00""D\x80\xc0\x00\x04\xc8\xb2\x00""E\x81\xc0\x00\x04\xc9\xb2\x00H\x0b\x84\x00\x00\xc8\xb2\x00I\x0b\x85\x00\x00\xc9\xb2\x00\x86\x87\x03@\xa0\xc2\xa4\x00""B\x8a\xc0\x80\x02\xc2\xf4\x00\x88\x89\x03@\xa0\xc3\xa4\x00""C\x8a""B\x80\x02\xc3\xf4\x00\x0b\xd0""C\x01\x00\xc3\xf4\x00""C\xc0KC\x00KP\x09@\x00\x00""9\x08Hax";
static const cmpbe_chunk_OBJC OBJC_45 = { 336, (uint8_t*)b_45 };
static const cmpbe_chunk_EBIN EBIN_46 =
{
	(uint16_t) 0x2,
	(uint16_t) 0x3800,
	(cmpbe_chunk_FSHA_prepass_flags) 0x0,
	(uint32_t) 0xffffffff,
	(uint16_t) 0x20,
	(uint16_t) 0xe,
	(uint32_t) 0x0,
	(cmpbe_chunk_FSHA_flags) 0x2000,
	(cmpbe_chunk_FSHA_variant) 0x180,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	OBJC_45,
	NULL,
	NULL,
};
static const uint8_t b_47[] = "B@\x00(\xf4\x82`\x00\xc0\x0f\x00\x00\x00\xc4\x10\x01\xc0\x1f\x00\x00\x00\xfd\x10\x01\xc0\x2f\x00\x00\x00\xfe\x10\x09""C\xff\xff\xff\x00\xc3\x18\x01\x02\x04\x00\x18\x82\x88`\x00""B\x08\x00(\xf4\x82`\x08\x08""D\x03\xc5\x01\xc5P\x01\x02\x00\x00\x00\x00\xc4\x91\x00""D\x00\x00""8\xf8\x84`\x00\x08}\x03\xc5\x01\xfdP\x01\x02\x00\x00\x00\x00\xfc\x91\x00\x08~\x03\xc5\x01\xffP\x01\x02\x00\x00\x00\x00\xfe\x91\x08\x00""0\x00x\x08""Da\x00|\x10\x00""8\xf8\x84`\x08\x00@\x00x\x08""Da\x00~ \x00""8\xf8\x84`\x00\xc0?\x00\x00\x00\xff\x10\x01H\x7f""C\xc5\x01\xc3P\x09\x00P\x00x\x08""Da\x00""B0\x00""8\xf8\x84`\x08@`\x00""8\x08""Dax";
static const cmpbe_chunk_OBJC OBJC_47 = { 184, (uint8_t*)b_47 };
static const cmpbe_chunk_EBIN EBIN_48 =
{
	(uint16_t) 0xffff,
	(uint16_t) 0x0,
	(cmpbe_chunk_FSHA_prepass_flags) 0x0,
	(uint32_t) 0xffffffff,
	(uint16_t) 0x20,
	(uint16_t) 0xe,
	(uint32_t) 0x0,
	(cmpbe_chunk_FSHA_flags) 0x20042000,
	(cmpbe_chunk_FSHA_variant) 0x0,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	OBJC_47,
	NULL,
	NULL,
};
static cmpbe_chunk_EBIN a_44[3] = { EBIN_44, EBIN_46, EBIN_48, };
static const cmpbe_chunk_CMMN CMMN_1 =
{
	VELA_1,
	SSYM_8,
	SSYM_15,
	SSYM_37,
	SSYM_38,
	SSYM_39,
	SSYM_40,
	UBUF_41,
	(uint16_t) 0x0,
	(uint32_t) 0x1,
	a_42,
	(uint32_t) 0x3,
	a_44,
	NULL,
};
static const cmpbe_chunk_CVER CVER_1 =
{
	CMMN_1,
	NULL,
	NULL,
};
cmpbe_chunk_MBS2 MBS2 =
{
	/*version = */48,
	VEHW_0,
	NULL,
	NULL,
	NULL,
	(cmpbe_chunk_CVER*)&CVER_1,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
};

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

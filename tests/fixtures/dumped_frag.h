
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
static const uint8_t b_2[] = "_17_inUV";
static const cmpbe_chunk_STRI STRI_2 = { 8, (uint8_t*)b_2 };
static const cmpbe_chunk_TPGE TPGE_3 =
{
	(cmpbe_chunk_TPGE_scalar_type) 0x1,
	(uint8_t) 0x2,
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
	(uint8_t) 0xb,
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
static cmpbe_chunk_SYMB a_2[1] = { SYMB_2, };
static const cmpbe_chunk_SSYM SSYM_5 =
{
	(uint32_t) 0x1,
	a_2,
};
static const cmpbe_chunk_SSYM SSYM_6 =
{
	(uint32_t) 0x0,
	NULL,
};
static const cmpbe_chunk_SSYM SSYM_7 =
{
	(uint32_t) 0x0,
	NULL,
};
static const uint8_t b_8[] = "_13_texSampler";
static const cmpbe_chunk_STRI STRI_8 = { 14, (uint8_t*)b_8 };
static const cmpbe_chunk_TPSA TPSA_9 =
{
	(uint32_t) 0x85,
};
static const cmpbe_chunk_TYPE TYPE_10 =
{
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	(cmpbe_chunk_TPSA*)&TPSA_9,
	NULL,
	NULL,
};
static const cmpbe_chunk_SYMB SYMB_8 =
{
	STRI_8,
	(cmpbe_chunk_symbol_flags) 0x24,
	(cmpbe_chunk_SYMB_semantics) 0x0,
	(uint8_t) 0x44,
	(uint32_t) 0x1,
	(uint16_t) 0x1,
	(uint16_t) 0xffff,
	TYPE_10,
	(uint32_t) 0x0,
	NULL,
	(uint32_t) 0x0,
	NULL,
	NULL,
	NULL,
};
static cmpbe_chunk_SYMB a_8[1] = { SYMB_8, };
static const cmpbe_chunk_SSYM SSYM_11 =
{
	(uint32_t) 0x1,
	a_8,
};
static const cmpbe_chunk_SSYM SSYM_12 =
{
	(uint32_t) 0x0,
	NULL,
};
static const cmpbe_chunk_SSYM SSYM_13 =
{
	(uint32_t) 0x0,
	NULL,
};
static const cmpbe_chunk_UBUF UBUF_14 =
{
	(uint32_t) 0x0,
	NULL,
};
static const cmpbe_chunk_FCST FCST_15 =
{
	(uint16_t) 0x4,
	(uint16_t) 0x0,
	(uint32_t) 0xffffffff,
};
static const cmpbe_chunk_FCST FCST_16 =
{
	(uint16_t) 0x4,
	(uint16_t) 0x4,
	(uint32_t) 0x4000001,
};
static const cmpbe_chunk_FCST FCST_17 =
{
	(uint16_t) 0x4,
	(uint16_t) 0x8,
	(uint32_t) 0xb000000,
};
static cmpbe_chunk_FCST a_15[3] = { FCST_15, FCST_16, FCST_17, };
static uint8_t d_18[1] = { (uint8_t)0x0 };
static cmpbe_chunk_TPGE_scalar_size d_19[1] = { (cmpbe_chunk_TPGE_scalar_size)0x2 };
static cmpbe_chunk_TPGE_scalar_type d_20[1] = { (cmpbe_chunk_TPGE_scalar_type)0x1 };
static const cmpbe_chunk_SPDf SPDf_21 =
{
	(uint32_t) 0x1,
	d_18,
	d_19,
	d_20,
};
static const uint8_t b_22[] = "\x80!\x82\x1d\xb4\x80""8I\xf0\x00|2\x08@\x7fx";
static const cmpbe_chunk_OBJC OBJC_22 = { 16, (uint8_t*)b_22 };
static const cmpbe_chunk_EBIN EBIN_23 =
{
	(uint16_t) 0xffff,
	(uint16_t) 0x3000,
	(cmpbe_chunk_FSHA_prepass_flags) 0x0,
	(uint32_t) 0xffffffff,
	(uint16_t) 0x20,
	(uint16_t) 0x2,
	(uint32_t) 0x0,
	(cmpbe_chunk_FSHA_flags) 0x8a11000,
	(cmpbe_chunk_FSHA_variant) 0x0,
	NULL,
	NULL,
	NULL,
	(cmpbe_chunk_SPDf*)&SPDf_21,
	NULL,
	OBJC_22,
	NULL,
	NULL,
};
static cmpbe_chunk_EBIN a_23[1] = { EBIN_23, };
static const cmpbe_chunk_CMMN CMMN_1 =
{
	VELA_1,
	SSYM_5,
	SSYM_6,
	SSYM_7,
	SSYM_11,
	SSYM_12,
	SSYM_13,
	UBUF_14,
	(uint16_t) 0x0,
	(uint32_t) 0x3,
	a_15,
	(uint32_t) 0x1,
	a_23,
	NULL,
};
static const cmpbe_chunk_SPAS SPAS_24 =
{
	(uint16_t) 0x0,
	(uint16_t) 0x1,
};
static const cmpbe_chunk_CFRA CFRA_1 =
{
	CMMN_1,
	NULL,
	NULL,
	NULL,
	NULL,
	(cmpbe_chunk_SPAS*)&SPAS_24,
};
cmpbe_chunk_MBS2 MBS2 =
{
	/*version = */48,
	VEHW_0,
	NULL,
	NULL,
	(cmpbe_chunk_CFRA*)&CFRA_1,
	NULL,
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

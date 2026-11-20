#include "mbs2_ast_parser.h"
#include "json_lite.h"
#include "moc_runner.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

namespace valhall::spirv2valhall {

namespace {

bool dir_exists(const std::string& path) {
    struct stat st{};
    return !path.empty() && stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

std::string resolve_cmpbe_include_dir(const std::string& override_dir) {
    if (!override_dir.empty()) return override_dir;
#ifdef SPIRV2VALHALL_DEFAULT_CMPBE_INCLUDE_DIR
    if (dir_exists(SPIRV2VALHALL_DEFAULT_CMPBE_INCLUDE_DIR)) {
        return SPIRV2VALHALL_DEFAULT_CMPBE_INCLUDE_DIR;
    }
#endif
    return "tools/spirv2valhall";
}

std::string resolve_clang_resource_dir(const std::string& override_dir) {
    if (!override_dir.empty()) return override_dir;
    const char* env_res = std::getenv("CLANG_RESOURCE_DIR");
    if (env_res && env_res[0] != '\0' && dir_exists(env_res)) return env_res;
#ifdef SPIRV2VALHALL_DEFAULT_CLANG_RESOURCE_DIR
    if (dir_exists(SPIRV2VALHALL_DEFAULT_CLANG_RESOURCE_DIR)) {
        return SPIRV2VALHALL_DEFAULT_CLANG_RESOURCE_DIR;
    }
#endif
    if (dir_exists("third_party/prebuilts/llvm-21/lib/clang/21")) {
        return "third_party/prebuilts/llvm-21/lib/clang/21";
    }
    return "";
}

std::vector<uint8_t> decode_hex(const std::string& hex) {
    std::vector<uint8_t> out;
    out.reserve(hex.size() / 2);
    auto nib = [](char c) -> uint8_t {
        if (c >= '0' && c <= '9') return static_cast<uint8_t>(c - '0');
        if (c >= 'a' && c <= 'f') return static_cast<uint8_t>(c - 'a' + 10);
        if (c >= 'A' && c <= 'F') return static_cast<uint8_t>(c - 'A' + 10);
        return 0;
    };
    for (size_t i = 0; i + 1 < hex.size(); i += 2) {
        out.push_back(static_cast<uint8_t>((nib(hex[i]) << 4) | nib(hex[i + 1])));
    }
    return out;
}

class Mbs2AstWalker {
public:
    explicit Mbs2AstWalker(const JsonValue& root) {
        const JsonValue* globals = root.get("globals");
        if (globals && globals->is_array()) {
            for (const auto& g : globals->arr_val) {
                std::string name = g.get_string("name");
                if (!name.empty()) {
                    globals_[name] = &g;
                }
            }
        }
    }

    size_t global_count() const { return globals_.size(); }

    const JsonValue* find_global_init(const std::string& name) const {
        auto it = globals_.find(name);
        if (it == globals_.end()) return nullptr;
        return it->second->get("init");
    }

    // Unwrap casts, unary '&', and resolve top-level variable references.
    const JsonValue* resolve_expr(const JsonValue* node, std::string* out_ref_name = nullptr, int depth = 0) const {
        if (!node || node->is_null() || depth > 32) return nullptr;
        std::string kind = node->get_string("kind");
        if (kind == "cast") {
            return resolve_expr(node->get("expr"), out_ref_name, depth + 1);
        }
        if (kind == "unary") {
            std::string op = node->get_string("op");
            if (op == "&" || op == "+") {
                return resolve_expr(node->get("operand"), out_ref_name, depth + 1);
            }
            return node;
        }
        if (kind == "ident") {
            std::string name = node->get_string("name");
            if (out_ref_name && out_ref_name->empty()) *out_ref_name = name;
            const JsonValue* init = find_global_init(name);
            if (!init) return node;
            return resolve_expr(init, out_ref_name, depth + 1);
        }
        if (kind == "literal" && node->get_string("lit_kind") == "null") {
            return nullptr;
        }
        return node;
    }

    const JsonValue* get_field(const JsonValue* struct_node, const std::string& field_name) const {
        const JsonValue* s = resolve_expr(struct_node);
        if (!s || s->get_string("kind") != "init_list") return nullptr;
        const JsonValue* fields = s->get("fields");
        if (!fields || !fields->is_object()) return nullptr;
        return fields->get(field_name);
    }

    uint64_t eval_u64(const JsonValue* node, uint64_t fallback = 0) const {
        const JsonValue* r = resolve_expr(node);
        if (!r) return fallback;
        std::string kind = r->get_string("kind");
        if (kind == "unary" && r->get_string("op") == "-") {
            return static_cast<uint64_t>(-static_cast<int64_t>(eval_u64(r->get("operand"), 0)));
        }
        if (kind == "literal") {
            std::string lk = r->get_string("lit_kind");
            if (lk == "int") {
                const JsonValue* iv = r->get("int_value");
                if (iv && iv->is_int()) return static_cast<uint64_t>(iv->int_val);
                std::string sv = r->get_string("value");
                if (!sv.empty()) {
                    if (sv[0] == '-') return static_cast<uint64_t>(std::strtoll(sv.c_str(), nullptr, 10));
                    return std::strtoull(sv.c_str(), nullptr, 10);
                }
            } else if (lk == "bool") {
                return r->get_bool("value") ? 1u : 0u;
            }
        }
        return fallback;
    }

    std::vector<uint8_t> eval_bytes(const JsonValue* node) const {
        const JsonValue* r = resolve_expr(node);
        if (!r) return {};
        if (r->get_string("kind") == "literal" && r->get_string("lit_kind") == "string") {
            const JsonValue* hex = r->get("bytes_hex");
            if (hex && hex->is_string()) {
                return decode_hex(hex->str_val);
            }
            std::string v = r->get_string("value");
            return std::vector<uint8_t>(v.begin(), v.end());
        }
        if (r->get_string("kind") == "init_list") {
            const JsonValue* items = r->get("items");
            if (items && items->is_array()) {
                std::vector<uint8_t> out;
                out.reserve(items->arr_val.size());
                for (const auto& item : items->arr_val) {
                    out.push_back(static_cast<uint8_t>(eval_u64(&item, 0) & 0xFFu));
                }
                return out;
            }
        }
        return {};
    }

    std::string parse_stri(const JsonValue* node) const {
        const JsonValue* s = resolve_expr(node);
        if (!s) return "";
        uint32_t sz = static_cast<uint32_t>(eval_u64(get_field(s, "size"), 0));
        auto bytes = eval_bytes(get_field(s, "data"));
        if (sz > 0 && sz < bytes.size()) bytes.resize(sz);
        // Strip trailing NULs
        while (!bytes.empty() && bytes.back() == 0) bytes.pop_back();
        return std::string(bytes.begin(), bytes.end());
    }

    const std::vector<JsonValue>* resolve_array_items(const JsonValue* ptr_node) const {
        const JsonValue* r = resolve_expr(ptr_node);
        if (!r || r->get_string("kind") != "init_list") return nullptr;
        const JsonValue* items = r->get("items");
        if (!items || !items->is_array()) return nullptr;
        return &items->arr_val;
    }

    Mbs2Tpge parse_tpge(const JsonValue* node) const {
        Mbs2Tpge t;
        t.scalar_type = static_cast<uint32_t>(eval_u64(get_field(node, "scalar_type")));
        t.vector_size = static_cast<uint8_t>(eval_u64(get_field(node, "vector_size"), 1));
        t.scalar_size = static_cast<uint32_t>(eval_u64(get_field(node, "scalar_size"), 2));
        t.precision = static_cast<uint32_t>(eval_u64(get_field(node, "precision"), 2));
        t.stride = static_cast<uint32_t>(eval_u64(get_field(node, "stride")));
        t.aux_qualifier = static_cast<uint32_t>(eval_u64(get_field(node, "aux_qualifier")));
        t.component = static_cast<uint8_t>(eval_u64(get_field(node, "component")));
        return t;
    }

    Mbs2Tpse parse_tpse(const JsonValue* node) const {
        Mbs2Tpse m;
        m.name = parse_stri(get_field(node, "name"));
        m.offset = static_cast<uint32_t>(eval_u64(get_field(node, "offset")));
        m.xfb_offset = static_cast<uint16_t>(eval_u64(get_field(node, "xfb_offset"), 0xFFFFu));
        m.flags = static_cast<uint32_t>(eval_u64(get_field(node, "flags")));
        m.location = static_cast<uint16_t>(eval_u64(get_field(node, "location"), 0xFFFFu));
        m.semantics = static_cast<uint32_t>(eval_u64(get_field(node, "semantics")));
        m.type = parse_type(get_field(node, "type"));
        return m;
    }

    std::shared_ptr<Mbs2Type> parse_type(const JsonValue* node, int depth = 0) const {
        const JsonValue* s = resolve_expr(node);
        if (!s || depth > 16) return nullptr;
        auto t = std::make_shared<Mbs2Type>();

        if (const JsonValue* tpac = resolve_expr(get_field(s, "tpac"))) {
            t->has_tpac = true;
            t->tpac_offset = static_cast<uint32_t>(eval_u64(get_field(tpac, "offset")));
        }
        if (const JsonValue* tpge = resolve_expr(get_field(s, "tpge"))) {
            t->has_tpge = true;
            t->tpge = parse_tpge(tpge);
        }
        if (const JsonValue* tpma = resolve_expr(get_field(s, "tpma"))) {
            t->has_tpma = true;
            t->tpma.nof_columns = static_cast<uint8_t>(eval_u64(get_field(tpma, "nof_columns")));
            t->tpma.is_row_major = eval_u64(get_field(tpma, "is_row_major")) != 0;
            t->tpma.major_stride = static_cast<uint16_t>(eval_u64(get_field(tpma, "major_stride")));
            t->tpma.column_type = parse_tpge(get_field(tpma, "column_type"));
        }
        if (const JsonValue* tpar = resolve_expr(get_field(s, "tpar"))) {
            t->has_tpar = true;
            t->tpar.array_size = static_cast<uint32_t>(eval_u64(get_field(tpar, "array_size")));
            t->tpar.element_type = parse_type(get_field(tpar, "element_type"), depth + 1);
        }
        if (const JsonValue* tpib = resolve_expr(get_field(s, "tpib"))) {
            t->has_tpib = true;
            t->tpib.layout = static_cast<uint32_t>(eval_u64(get_field(tpib, "layout")));
            t->tpib.kind = static_cast<uint32_t>(eval_u64(get_field(tpib, "kind")));
            t->tpib.stride = static_cast<uint32_t>(eval_u64(get_field(tpib, "stride")));
            t->tpib.instance_name = parse_stri(get_field(tpib, "instance_name"));
            uint32_t n = static_cast<uint32_t>(eval_u64(get_field(tpib, "nof_members")));
            if (const auto* arr = resolve_array_items(get_field(tpib, "tpse"))) {
                for (size_t i = 0; i < arr->size() && (n == 0 || i < n); ++i) {
                    t->tpib.members.push_back(parse_tpse(&(*arr)[i]));
                }
            }
        }
        if (const JsonValue* tpst = resolve_expr(get_field(s, "tpst"))) {
            t->has_tpst = true;
            t->tpst.stride = static_cast<uint32_t>(eval_u64(get_field(tpst, "stride")));
            t->tpst.type_name = parse_stri(get_field(tpst, "type_name"));
            uint32_t n = static_cast<uint32_t>(eval_u64(get_field(tpst, "nof_members")));
            if (const auto* arr = resolve_array_items(get_field(tpst, "tpse"))) {
                for (size_t i = 0; i < arr->size() && (n == 0 || i < n); ++i) {
                    t->tpst.members.push_back(parse_tpse(&(*arr)[i]));
                }
            }
        }
        if (const JsonValue* tppo = resolve_expr(get_field(s, "tppo"))) {
            t->has_tppo = true;
            t->tppo.scalar_size = static_cast<uint32_t>(eval_u64(get_field(tppo, "scalar_size")));
            t->tppo.address_space = static_cast<uint32_t>(eval_u64(get_field(tppo, "address_space")));
            t->tppo.pointed_type = parse_type(get_field(tppo, "pointed_type"), depth + 1);
        }
        if (const JsonValue* tpsa = resolve_expr(get_field(s, "tpsa"))) {
            t->has_tpsa = true;
            t->tpsa_kind = static_cast<uint32_t>(eval_u64(get_field(tpsa, "kind")));
        }
        if (const JsonValue* tpmd = resolve_expr(get_field(s, "tpmd"))) {
            t->has_tpmd = true;
            t->tpmd_flags = static_cast<uint32_t>(eval_u64(get_field(tpmd, "flags")));
        }
        return t;
    }

    Mbs2Rloc parse_rloc(const JsonValue* node) const {
        Mbs2Rloc r;
        r.sym_idx = static_cast<uint16_t>(eval_u64(get_field(node, "sym_idx")));
        r.reg_offset = static_cast<uint16_t>(eval_u64(get_field(node, "reg_offset")));
        r.uni_offset = static_cast<uint16_t>(eval_u64(get_field(node, "uni_offset")));
        r.size = static_cast<uint16_t>(eval_u64(get_field(node, "size")));
        return r;
    }

    Mbs2Relo parse_relo(const JsonValue* node) const {
        Mbs2Relo r;
        r.offset = static_cast<uint32_t>(eval_u64(get_field(node, "offset")));
        r.kind = static_cast<uint32_t>(eval_u64(get_field(node, "kind")));
        r.section_idx = static_cast<uint8_t>(eval_u64(get_field(node, "section_idx")));
        return r;
    }

    Mbs2Fcst parse_fcst(const JsonValue* node) const {
        Mbs2Fcst f;
        f.size = static_cast<uint16_t>(eval_u64(get_field(node, "size")));
        f.reg_offset = static_cast<uint16_t>(eval_u64(get_field(node, "reg_offset")));
        f.value = static_cast<uint32_t>(eval_u64(get_field(node, "value")));
        return f;
    }

    Mbs2Symbol parse_symb(const JsonValue* node) const {
        Mbs2Symbol sym;
        sym.name = parse_stri(get_field(node, "name"));
        sym.flags = static_cast<uint32_t>(eval_u64(get_field(node, "flags")));
        sym.semantics = static_cast<uint32_t>(eval_u64(get_field(node, "semantics")));
        sym.rack_index = static_cast<uint8_t>(eval_u64(get_field(node, "rack_index")));
        sym.address = static_cast<uint32_t>(eval_u64(get_field(node, "address")));
        sym.binding = static_cast<uint16_t>(eval_u64(get_field(node, "binding"), 0xFFFFu));
        sym.location = static_cast<uint16_t>(eval_u64(get_field(node, "location"), 0xFFFFu));
        sym.type = parse_type(get_field(node, "type"));

        uint32_t n_rloc = static_cast<uint32_t>(eval_u64(get_field(node, "nof_rloc")));
        if (const auto* arr = resolve_array_items(get_field(node, "rloc"))) {
            for (size_t i = 0; i < arr->size() && (n_rloc == 0 || i < n_rloc); ++i) {
                sym.rlocs.push_back(parse_rloc(&(*arr)[i]));
            }
        }
        uint32_t n_relo = static_cast<uint32_t>(eval_u64(get_field(node, "nof_relo")));
        if (const auto* arr = resolve_array_items(get_field(node, "relo"))) {
            for (size_t i = 0; i < arr->size() && (n_relo == 0 || i < n_relo); ++i) {
                sym.relos.push_back(parse_relo(&(*arr)[i]));
            }
        }
        uint32_t n_fixp = static_cast<uint32_t>(eval_u64(get_field(node, "nof_fixp")));
        if (const auto* arr = resolve_array_items(get_field(node, "fixp"))) {
            for (size_t i = 0; i < arr->size() && (n_fixp == 0 || i < n_fixp); ++i) {
                Mbs2Fixp fp;
                fp.kind = static_cast<uint32_t>(eval_u64(get_field(&(*arr)[i], "kind")));
                fp.offset = static_cast<uint32_t>(eval_u64(get_field(&(*arr)[i], "offset")));
                fp.value = static_cast<uint32_t>(eval_u64(get_field(&(*arr)[i], "value")));
                fp.needs_high = eval_u64(get_field(&(*arr)[i], "needs_high")) != 0;
                sym.fixps.push_back(fp);
            }
        }
        if (const JsonValue* sini = resolve_expr(get_field(node, "sini"))) {
            uint32_t cnt = static_cast<uint32_t>(eval_u64(get_field(sini, "count")));
            if (const auto* arr = resolve_array_items(get_field(sini, "values"))) {
                for (size_t i = 0; i < arr->size() && (cnt == 0 || i < cnt); ++i) {
                    sym.sini_values.push_back(static_cast<uint32_t>(eval_u64(&(*arr)[i])));
                }
            }
        }
        if (const JsonValue* symd = resolve_expr(get_field(node, "symd"))) {
            sym.has_symd = true;
            sym.symd.size = static_cast<uint32_t>(eval_u64(get_field(symd, "size")));
            sym.symd.alignment = static_cast<uint32_t>(eval_u64(get_field(symd, "alignment")));
            sym.symd.addressspace = static_cast<uint32_t>(eval_u64(get_field(symd, "addressspace")));
            sym.symd.sym_hidden = static_cast<uint8_t>(eval_u64(get_field(symd, "sym_hidden")));
            sym.symd.sym_restrict = static_cast<uint8_t>(eval_u64(get_field(symd, "sym_restrict")));
            sym.symd.may_read_yuv = eval_u64(get_field(symd, "may_read_yuv")) != 0;
            sym.symd.force_invariance = static_cast<uint8_t>(eval_u64(get_field(symd, "force_invariance")));
        }
        return sym;
    }

    std::vector<Mbs2Symbol> parse_ssym(const JsonValue* node) const {
        std::vector<Mbs2Symbol> out;
        uint32_t cnt = static_cast<uint32_t>(eval_u64(get_field(node, "count")));
        if (cnt == 0) return out;
        if (const auto* arr = resolve_array_items(get_field(node, "symb"))) {
            for (size_t i = 0; i < arr->size() && i < cnt; ++i) {
                out.push_back(parse_symb(&(*arr)[i]));
            }
        }
        return out;
    }

    static EbinRole classify_ebin_role(const std::string& stage_chunk, const Mbs2EbinMeta& meta) {
        if (stage_chunk == "CCOM") {
            if ((meta.flags & 0x20000000u) != 0 ||
                (meta.rt_num == 0xFFFFu && meta.aux_idx == 0 && meta.variant == 0) ||
                (meta.has_fnmd && !meta.fnmd.pilot_name.empty())) {
                return EbinRole::Pilot;
            }
            return EbinRole::MainCompute;
        }
        if (stage_chunk == "CVER") {
            if ((meta.variant & 0x01u) != 0) return EbinRole::VertexVarying;
            return EbinRole::VertexPosition;
        }
        if (stage_chunk == "CFRA") return EbinRole::Fragment;
        if (stage_chunk == "CTSC") return EbinRole::TessControl;
        if (stage_chunk == "CTSE") return EbinRole::TessEval;
        if (stage_chunk == "CGEO") return EbinRole::Geometry;
        if (stage_chunk == "CBLE") return EbinRole::Blend;
        return EbinRole::Unknown;
    }

    ExtractedEbin parse_ebin(
        const JsonValue* node,
        uint32_t index,
        const std::string& stage_chunk,
        uint8_t arch,
        bool disassemble) const {
        std::string ref_name;
        const JsonValue* s = resolve_expr(node, &ref_name);

        ExtractedEbin eb;
        eb.index = index;
        eb.meta.index = index;
        eb.meta.var_name = ref_name;
        if (!s) return eb;

        eb.meta.rt_num = static_cast<uint16_t>(eval_u64(get_field(s, "rt_num")));
        eb.meta.aux_idx = static_cast<uint16_t>(eval_u64(get_field(s, "aux_idx")));
        eb.meta.prepass_flags = static_cast<uint32_t>(eval_u64(get_field(s, "prepass_flags")));
        eb.meta.data_section_offset = static_cast<uint32_t>(eval_u64(get_field(s, "data_section_offset"), 0xFFFFFFFFu));
        eb.meta.nof_reg = static_cast<uint16_t>(eval_u64(get_field(s, "nof_reg")));
        eb.meta.nof_rmu = static_cast<uint16_t>(eval_u64(get_field(s, "nof_rmu")));
        eb.meta.tls_size = static_cast<uint32_t>(eval_u64(get_field(s, "tls_size")));
        eb.meta.flags = static_cast<uint32_t>(eval_u64(get_field(s, "flags")));
        eb.meta.variant = static_cast<uint32_t>(eval_u64(get_field(s, "variant")));
        eb.meta.entrypoint_name = parse_stri(get_field(s, "entrypoint_name"));

        if (const JsonValue* dbug = resolve_expr(get_field(s, "dbug"))) {
            eb.meta.has_dbug = true;
            uint32_t nr = static_cast<uint32_t>(eval_u64(get_field(dbug, "nof_rloc")));
            if (const auto* arr = resolve_array_items(get_field(dbug, "rloc"))) {
                for (size_t i = 0; i < arr->size() && (nr == 0 || i < nr); ++i) {
                    eb.meta.dbug.rlocs.push_back(parse_rloc(&(*arr)[i]));
                }
            }
            uint32_t nf = static_cast<uint32_t>(eval_u64(get_field(dbug, "nof_fcst")));
            if (const auto* arr = resolve_array_items(get_field(dbug, "fcst"))) {
                for (size_t i = 0; i < arr->size() && (nf == 0 || i < nf); ++i) {
                    eb.meta.dbug.fcsts.push_back(parse_fcst(&(*arr)[i]));
                }
            }
        }
        if (const JsonValue* grel = resolve_expr(get_field(s, "grel"))) {
            uint32_t ng = static_cast<uint32_t>(eval_u64(get_field(grel, "nof_global_relo")));
            if (const auto* arr = resolve_array_items(get_field(grel, "relo"))) {
                for (size_t i = 0; i < arr->size() && (ng == 0 || i < ng); ++i) {
                    eb.meta.grel.push_back(parse_relo(&(*arr)[i]));
                }
            }
        }
        if (const JsonValue* spdf = resolve_expr(get_field(s, "spdf"))) {
            eb.meta.has_spdf = true;
            uint32_t cnt = static_cast<uint32_t>(eval_u64(get_field(spdf, "rt_count")));
            if (const auto* arr = resolve_array_items(get_field(spdf, "rt_index"))) {
                for (size_t i = 0; i < arr->size() && (cnt == 0 || i < cnt); ++i) {
                    eb.meta.spdf.rt_indices.push_back(static_cast<uint8_t>(eval_u64(&(*arr)[i])));
                }
            }
            if (const auto* arr = resolve_array_items(get_field(spdf, "size"))) {
                for (size_t i = 0; i < arr->size() && (cnt == 0 || i < cnt); ++i) {
                    eb.meta.spdf.sizes.push_back(static_cast<uint32_t>(eval_u64(&(*arr)[i])));
                }
            }
            if (const auto* arr = resolve_array_items(get_field(spdf, "type"))) {
                for (size_t i = 0; i < arr->size() && (cnt == 0 || i < cnt); ++i) {
                    eb.meta.spdf.types.push_back(static_cast<uint32_t>(eval_u64(&(*arr)[i])));
                }
            }
        }
        if (const JsonValue* pdsc = resolve_expr(get_field(s, "pdsc"))) {
            eb.meta.has_pdsc = true;
            eb.meta.pdsc.varying0 = static_cast<uint16_t>(eval_u64(get_field(pdsc, "varying0")));
            eb.meta.pdsc.varying1 = static_cast<uint16_t>(eval_u64(get_field(pdsc, "varying1")));
            eb.meta.pdsc.primary_shader_wait = static_cast<uint8_t>(eval_u64(get_field(pdsc, "primary_shader_wait")));
        }
        if (const JsonValue* vlkn = resolve_expr(get_field(s, "vlkn"))) {
            eb.meta.has_vlkn = true;
            eb.meta.vlkn.used_set_bitmask = static_cast<uint16_t>(eval_u64(get_field(vlkn, "used_set_bitmask")));
            eb.meta.vlkn.stack_layers = static_cast<uint16_t>(eval_u64(get_field(vlkn, "stack_layers")));
            eb.meta.vlkn.dynamic_outputs = static_cast<uint8_t>(eval_u64(get_field(vlkn, "dynamic_outputs")));
        }
        if (const JsonValue* fnmd = resolve_expr(get_field(s, "fnmd"))) {
            eb.meta.has_fnmd = true;
            uint32_t nt = static_cast<uint32_t>(eval_u64(get_field(fnmd, "nof_tvar")));
            if (const auto* arr = resolve_array_items(get_field(fnmd, "tvar"))) {
                for (size_t i = 0; i < arr->size() && (nt == 0 || i < nt); ++i) {
                    eb.meta.fnmd.tvars.push_back(static_cast<uint32_t>(eval_u64(&(*arr)[i])));
                }
            }
            eb.meta.fnmd.is_other_function = static_cast<uint8_t>(eval_u64(get_field(fnmd, "is_other_function")));
            eb.meta.fnmd.needs_structured_cfg = static_cast<uint8_t>(eval_u64(get_field(fnmd, "needs_structured_cfg")));
            eb.meta.fnmd.is_vertex_piece = static_cast<uint8_t>(eval_u64(get_field(fnmd, "is_vertex_piece")));
            eb.meta.fnmd.reg_mode = static_cast<uint32_t>(eval_u64(get_field(fnmd, "reg_mode")));
            eb.meta.fnmd.pilot_name = parse_stri(get_field(fnmd, "pilot_name"));
            eb.meta.fnmd.late_codegen = parse_stri(get_field(fnmd, "late_codegen"));
            eb.meta.fnmd.attrib_mask = static_cast<uint32_t>(eval_u64(get_field(fnmd, "attrib_mask")));
            eb.meta.fnmd.max_register_pressure = static_cast<uint32_t>(eval_u64(get_field(fnmd, "max_register_pressure")));
            eb.meta.fnmd.mrtt = static_cast<uint32_t>(eval_u64(get_field(fnmd, "mrtt")));
        }

        eb.meta.role = classify_ebin_role(stage_chunk, eb.meta);
        eb.role = eb.meta.role;
        eb.variant_flags = eb.meta.variant;

        // Decode OBJC bytecode blob
        const JsonValue* objc = resolve_expr(get_field(s, "bytecode"));
        if (objc) {
            uint32_t objc_sz = static_cast<uint32_t>(eval_u64(get_field(objc, "size")));
            std::vector<uint8_t> raw = eval_bytes(get_field(objc, "byte_code"));
            if (objc_sz > 0 && objc_sz < raw.size()) {
                raw.resize(objc_sz);
            }
            eb.meta.objc_total_size = static_cast<uint32_t>(raw.size());

            uint32_t code_sz = static_cast<uint32_t>(raw.size());
            if (eb.meta.data_section_offset != 0 &&
                eb.meta.data_section_offset != 0xFFFFFFFFu &&
                eb.meta.data_section_offset <= raw.size()) {
                code_sz = eb.meta.data_section_offset;
            }
            code_sz = (code_sz / 8u) * 8u;
            eb.code_size = code_sz;
            eb.meta.code_size = code_sz;
            eb.bytecode.assign(raw.begin(), raw.begin() + code_sz);
            if (code_sz < raw.size()) {
                eb.data_section.assign(raw.begin() + code_sz, raw.end());
            }

            if (disassemble) {
                for (size_t off = 0; off + 8 <= eb.bytecode.size(); off += 8) {
                    uint64_t word = 0;
                    std::memcpy(&word, &eb.bytecode[off], 8);
                    auto dec = valhall::decode_valhall_instruction(word, arch, off);
                    if (!dec.valid) eb.invalid_count++;
                    eb.instructions.push_back(std::move(dec));
                }
            }
        }
        return eb;
    }

    void parse_cmmn(
        const JsonValue* cmmn_node,
        const std::string& stage_chunk,
        uint8_t arch,
        bool disassemble,
        Mbs2Cmmn* out_cmmn,
        std::vector<ExtractedEbin>* out_ebins) const {
        const JsonValue* s = resolve_expr(cmmn_node);
        if (!s) return;

        if (const JsonValue* vela = resolve_expr(get_field(s, "vela"))) {
            out_cmmn->vela_lang = static_cast<uint32_t>(eval_u64(get_field(vela, "lang")));
        }
        out_cmmn->sinp = parse_ssym(get_field(s, "sinp"));
        out_cmmn->sout = parse_ssym(get_field(s, "sout"));
        out_cmmn->suni = parse_ssym(get_field(s, "suni"));
        out_cmmn->simg = parse_ssym(get_field(s, "simg"));
        out_cmmn->ssmp = parse_ssym(get_field(s, "ssmp"));
        out_cmmn->s3bo = parse_ssym(get_field(s, "s3bo"));

        if (const JsonValue* ubuf = resolve_expr(get_field(s, "ubuf"))) {
            uint32_t cnt = static_cast<uint32_t>(eval_u64(get_field(ubuf, "count")));
            if (const auto* arr = resolve_array_items(get_field(ubuf, "sizes"))) {
                for (size_t i = 0; i < arr->size() && (cnt == 0 || i < cnt); ++i) {
                    out_cmmn->ubuf_sizes.push_back(static_cast<uint32_t>(eval_u64(&(*arr)[i])));
                }
            }
        }

        out_cmmn->epilogue_jump_index = static_cast<uint16_t>(eval_u64(get_field(s, "epilogue_jump_index")));
        out_cmmn->written_pos_fifo_attrs = static_cast<uint16_t>(eval_u64(get_field(s, "written_pos_fifo_attrs")));

        uint32_t n_fcst = static_cast<uint32_t>(eval_u64(get_field(s, "nof_fcst")));
        if (const auto* arr = resolve_array_items(get_field(s, "fcst"))) {
            for (size_t i = 0; i < arr->size() && (n_fcst == 0 || i < n_fcst); ++i) {
                out_cmmn->fcsts.push_back(parse_fcst(&(*arr)[i]));
            }
        }

        uint32_t n_ebin = static_cast<uint32_t>(eval_u64(get_field(s, "nof_ebin")));
        if (const auto* arr = resolve_array_items(get_field(s, "ebin"))) {
            for (size_t i = 0; i < arr->size() && (n_ebin == 0 || i < n_ebin); ++i) {
                ExtractedEbin eb = parse_ebin(&(*arr)[i], static_cast<uint32_t>(out_ebins->size()),
                                              stage_chunk, arch, disassemble);
                out_cmmn->ebins.push_back(eb.meta);
                out_ebins->push_back(std::move(eb));
            }
        }

        if (const JsonValue* cxfb = resolve_expr(get_field(s, "cxfb"))) {
            out_cmmn->has_cxfb = true;
            out_cmmn->cxfb.strides[0] = static_cast<uint16_t>(eval_u64(get_field(cxfb, "stride_0")));
            out_cmmn->cxfb.strides[1] = static_cast<uint16_t>(eval_u64(get_field(cxfb, "stride_1")));
            out_cmmn->cxfb.strides[2] = static_cast<uint16_t>(eval_u64(get_field(cxfb, "stride_2")));
            out_cmmn->cxfb.strides[3] = static_cast<uint16_t>(eval_u64(get_field(cxfb, "stride_3")));
        }
    }

    bool parse_root(uint8_t arch, bool disassemble, Mbs2Module* out_mod,
                    std::vector<ExtractedEbin>* out_ebins, std::string* err) const {
        const JsonValue* mbs2 = find_global_init("MBS2");
        if (!mbs2) {
            if (err) *err = "No top-level 'MBS2' variable found in clang_ast_dumper output";
            return false;
        }
        *out_mod = Mbs2Module{};
        out_mod->ast_global_count = globals_.size();
        out_mod->version = static_cast<uint32_t>(eval_u64(get_field(mbs2, "version")));
        if (const JsonValue* vehw = resolve_expr(get_field(mbs2, "vehw"))) {
            out_mod->vehw.core = static_cast<uint32_t>(eval_u64(get_field(vehw, "core")));
            out_mod->vehw.major = static_cast<uint32_t>(eval_u64(get_field(vehw, "major")));
            out_mod->vehw.minor = static_cast<uint32_t>(eval_u64(get_field(vehw, "minor")));
        }

        if (const JsonValue* ccom = resolve_expr(get_field(mbs2, "ccom"))) {
            out_mod->stage_chunk = "CCOM";
            parse_cmmn(get_field(ccom, "cmmn"), "CCOM", arch, disassemble, &out_mod->cmmn, out_ebins);
            if (const JsonValue* kern = resolve_expr(get_field(ccom, "kern"))) {
                out_mod->has_kern = true;
                out_mod->kern.name = parse_stri(get_field(kern, "name"));
                out_mod->kern.workgroup_local_size = static_cast<uint32_t>(eval_u64(get_field(kern, "workgroup_local_size")));
                uint32_t nk = static_cast<uint32_t>(eval_u64(get_field(kern, "nof_kpar")));
                if (const auto* arr = resolve_array_items(get_field(kern, "kpar"))) {
                    for (size_t i = 0; i < arr->size() && (nk == 0 || i < nk); ++i) {
                        Mbs2Kpar kp;
                        kp.declared_name = parse_stri(get_field(&(*arr)[i], "declared_name"));
                        kp.uniform_name = parse_stri(get_field(&(*arr)[i], "uniform_name"));
                        kp.flags = static_cast<uint32_t>(eval_u64(get_field(&(*arr)[i], "flags")));
                        out_mod->kern.kpars.push_back(std::move(kp));
                    }
                }
                if (const JsonValue* kwgs = resolve_expr(get_field(kern, "workgroup_size"))) {
                    out_mod->kern.has_workgroup_size = true;
                    out_mod->kern.wg_x = static_cast<uint32_t>(eval_u64(get_field(kwgs, "x"), 1));
                    out_mod->kern.wg_y = static_cast<uint32_t>(eval_u64(get_field(kwgs, "y"), 1));
                    out_mod->kern.wg_z = static_cast<uint32_t>(eval_u64(get_field(kwgs, "z"), 1));
                }
            }
        } else if (const JsonValue* cfra = resolve_expr(get_field(mbs2, "cfra"))) {
            out_mod->stage_chunk = "CFRA";
            out_mod->has_cfra = true;
            parse_cmmn(get_field(cfra, "cmmn"), "CFRA", arch, disassemble, &out_mod->cmmn, out_ebins);
            if (const JsonValue* fotv = resolve_expr(get_field(cfra, "output_vars"))) {
                uint32_t nb = static_cast<uint32_t>(eval_u64(get_field(fotv, "nb_outputs")));
                if (const auto* arr = resolve_array_items(get_field(fotv, "outputs"))) {
                    for (size_t i = 0; i < arr->size() && (nb == 0 || i < nb); ++i) {
                        Mbs2Outv ov;
                        ov.name = parse_stri(get_field(&(*arr)[i], "name"));
                        ov.type = parse_type(get_field(&(*arr)[i], "type"));
                        ov.location = static_cast<uint16_t>(eval_u64(get_field(&(*arr)[i], "location")));
                        ov.nb_locations_used = static_cast<uint16_t>(eval_u64(get_field(&(*arr)[i], "nb_locations_used"), 1));
                        out_mod->cfra.output_vars.push_back(std::move(ov));
                    }
                }
            }
            if (const JsonValue* lcal = resolve_expr(get_field(cfra, "lcal"))) {
                out_mod->cfra.has_lcal = true;
                out_mod->cfra.local_storage_size = static_cast<uint8_t>(eval_u64(get_field(lcal, "local_storage_size")));
            }
            if (const JsonValue* bldm = resolve_expr(get_field(cfra, "bldm"))) {
                out_mod->cfra.has_bldm = true;
                out_mod->cfra.bldm_flags = static_cast<uint32_t>(eval_u64(get_field(bldm, "flags")));
            }
            if (const JsonValue* ainf = resolve_expr(get_field(cfra, "alpha_info"))) {
                uint32_t cnt = static_cast<uint32_t>(eval_u64(get_field(ainf, "count")));
                if (const auto* arr = resolve_array_items(get_field(ainf, "achn"))) {
                    for (size_t i = 0; i < arr->size() && (cnt == 0 || i < cnt); ++i) {
                        Mbs2Achn ac;
                        ac.sampler_name = parse_stri(get_field(&(*arr)[i], "sampler_name"));
                        ac.uniform_name = parse_stri(get_field(&(*arr)[i], "uniform_name"));
                        ac.kind = static_cast<uint32_t>(eval_u64(get_field(&(*arr)[i], "kind")));
                        ac.rt = static_cast<uint8_t>(eval_u64(get_field(&(*arr)[i], "rt")));
                        ac.sampler_component = static_cast<uint8_t>(eval_u64(get_field(&(*arr)[i], "sampler_component")));
                        ac.uniform_component = static_cast<uint8_t>(eval_u64(get_field(&(*arr)[i], "uniform_component")));
                        out_mod->cfra.alpha_info.push_back(std::move(ac));
                    }
                }
            }
            if (const JsonValue* spas = resolve_expr(get_field(cfra, "spas"))) {
                out_mod->cfra.has_spas = true;
                out_mod->cfra.spas_read_mask = static_cast<uint16_t>(eval_u64(get_field(spas, "resource_read_mask")));
                out_mod->cfra.spas_write_mask = static_cast<uint16_t>(eval_u64(get_field(spas, "resource_write_mask")));
            }
        } else if (const JsonValue* cver = resolve_expr(get_field(mbs2, "cver"))) {
            out_mod->stage_chunk = "CVER";
            out_mod->has_cver = true;
            parse_cmmn(get_field(cver, "cmmn"), "CVER", arch, disassemble, &out_mod->cmmn, out_ebins);
            if (const JsonValue* vmvw = resolve_expr(get_field(cver, "vmvw"))) {
                out_mod->cver.has_vmvw = true;
                out_mod->cver.num_views = static_cast<uint8_t>(eval_u64(get_field(vmvw, "num_views")));
            }
            if (const JsonValue* vrsm = resolve_expr(get_field(cver, "vrsm"))) {
                out_mod->cver.has_vrsm = true;
                out_mod->cver.vrs_modes_mask = static_cast<uint16_t>(eval_u64(get_field(vrsm, "potential_vrs_modes_mask")));
            }
        } else if (const JsonValue* ctsc = resolve_expr(get_field(mbs2, "ctsc"))) {
            out_mod->stage_chunk = "CTSC";
            parse_cmmn(get_field(ctsc, "cmmn"), "CTSC", arch, disassemble, &out_mod->cmmn, out_ebins);
            out_mod->ctsc_workgroup_local_size = static_cast<uint32_t>(eval_u64(get_field(ctsc, "workgroup_local_size")));
            if (const JsonValue* mtes = resolve_expr(get_field(ctsc, "mtes"))) {
                out_mod->has_mtes = true;
                out_mod->mtes.n_vertices = static_cast<uint32_t>(eval_u64(get_field(mtes, "n_vertices")));
                out_mod->mtes.primitive_mode = static_cast<uint32_t>(eval_u64(get_field(mtes, "primitive_mode")));
                out_mod->mtes.vertex_spacing = static_cast<uint32_t>(eval_u64(get_field(mtes, "vertex_spacing")));
                out_mod->mtes.ordering = static_cast<uint32_t>(eval_u64(get_field(mtes, "ordering")));
                out_mod->mtes.point_mode = eval_u64(get_field(mtes, "point_mode")) != 0;
            }
        } else if (const JsonValue* ctse = resolve_expr(get_field(mbs2, "ctse"))) {
            out_mod->stage_chunk = "CTSE";
            parse_cmmn(get_field(ctse, "cmmn"), "CTSE", arch, disassemble, &out_mod->cmmn, out_ebins);
            if (const JsonValue* mtes = resolve_expr(get_field(ctse, "mtes"))) {
                out_mod->has_mtes = true;
                out_mod->mtes.n_vertices = static_cast<uint32_t>(eval_u64(get_field(mtes, "n_vertices")));
                out_mod->mtes.primitive_mode = static_cast<uint32_t>(eval_u64(get_field(mtes, "primitive_mode")));
                out_mod->mtes.vertex_spacing = static_cast<uint32_t>(eval_u64(get_field(mtes, "vertex_spacing")));
                out_mod->mtes.ordering = static_cast<uint32_t>(eval_u64(get_field(mtes, "ordering")));
                out_mod->mtes.point_mode = eval_u64(get_field(mtes, "point_mode")) != 0;
            }
        } else if (const JsonValue* cgeo = resolve_expr(get_field(mbs2, "cgeo"))) {
            out_mod->stage_chunk = "CGEO";
            parse_cmmn(get_field(cgeo, "cmmn"), "CGEO", arch, disassemble, &out_mod->cmmn, out_ebins);
            if (const JsonValue* mgeo = resolve_expr(get_field(cgeo, "mgeo"))) {
                out_mod->has_mgeo = true;
                out_mod->mgeo.n_invocations = static_cast<uint16_t>(eval_u64(get_field(mgeo, "n_invocations")));
                out_mod->mgeo.max_vertices = static_cast<uint16_t>(eval_u64(get_field(mgeo, "max_vertices")));
                out_mod->mgeo.input_primitive_mode = static_cast<uint32_t>(eval_u64(get_field(mgeo, "input_primitive_mode")));
                out_mod->mgeo.output_primitive_mode = static_cast<uint32_t>(eval_u64(get_field(mgeo, "output_primitive_mode")));
            }
        } else if (const JsonValue* cble = resolve_expr(get_field(mbs2, "cble"))) {
            out_mod->stage_chunk = "CBLE";
            out_mod->cmmn.suni = parse_ssym(get_field(cble, "suni"));
            ExtractedEbin eb = parse_ebin(get_field(cble, "ebin"), 0, "CBLE", arch, disassemble);
            out_mod->cmmn.ebins.push_back(eb.meta);
            out_ebins->push_back(std::move(eb));
        }

        if (const JsonValue* momd = resolve_expr(get_field(mbs2, "momd"))) {
            out_mod->has_momd = true;
            out_mod->momd.prealloc_uni_buffers = static_cast<uint8_t>(eval_u64(get_field(momd, "prealloc_uni_buffers")));
            out_mod->momd.prealloc_fau = static_cast<uint8_t>(eval_u64(get_field(momd, "prealloc_fau")));
            out_mod->momd.ble_preserve_registers = static_cast<uint8_t>(eval_u64(get_field(momd, "ble_preserve_registers")));
            out_mod->momd.essl_on_spirv = static_cast<uint8_t>(eval_u64(get_field(momd, "essl_on_spirv")));
            out_mod->momd.has_callable_blend_shaders = static_cast<uint8_t>(eval_u64(get_field(momd, "has_callable_blend_shaders")));
            out_mod->momd.atest_wait_on_6_7 = static_cast<uint8_t>(eval_u64(get_field(momd, "atest_wait_on_6_7")));
            out_mod->momd.gfx_hal = static_cast<uint8_t>(eval_u64(get_field(momd, "gfx_hal")));
            out_mod->momd.robustness = static_cast<uint8_t>(eval_u64(get_field(momd, "robustness")));
            out_mod->momd.per_sample_shading = static_cast<uint8_t>(eval_u64(get_field(momd, "per_sample_shading")));
            out_mod->momd.setup_fau_in_pilot = static_cast<uint8_t>(eval_u64(get_field(momd, "setup_fau_in_pilot")));
            out_mod->momd.has_simd_call = static_cast<uint8_t>(eval_u64(get_field(momd, "has_simd_call")));
            out_mod->momd.has_simd_with_static_stack_frame = static_cast<uint8_t>(eval_u64(get_field(momd, "has_simd_with_static_stack_frame")));
            out_mod->momd.max_prealloc_addr = static_cast<uint32_t>(eval_u64(get_field(momd, "max_prealloc_addr")));
            uint32_t np = static_cast<uint32_t>(eval_u64(get_field(momd, "nof_ploc")));
            if (const auto* arr = resolve_array_items(get_field(momd, "ploc"))) {
                for (size_t i = 0; i < arr->size() && (np == 0 || i < np); ++i) {
                    Mbs2Ploc pl;
                    pl.reg_offset = static_cast<uint16_t>(eval_u64(get_field(&(*arr)[i], "reg_offset")));
                    pl.size = static_cast<uint16_t>(eval_u64(get_field(&(*arr)[i], "size")));
                    pl.orig_offset = static_cast<uint16_t>(eval_u64(get_field(&(*arr)[i], "orig_offset")));
                    out_mod->momd.plocs.push_back(pl);
                }
            }
        }

        if (out_ebins->empty()) {
            if (err) *err = "No EBIN bytecode sections found in parsed MBS2 AST";
            return false;
        }
        return true;
    }

private:
    std::map<std::string, const JsonValue*> globals_;
};

}  // namespace

std::string resolve_clang_ast_dumper_path(const std::string& override_path) {
    if (!override_path.empty()) return override_path;
    const char* env_bin = std::getenv("SPIRV2VALHALL_CLANG_AST_DUMPER");
    if (env_bin && env_bin[0] != '\0') return env_bin;
#ifdef SPIRV2VALHALL_DEFAULT_CLANG_AST_DUMPER_PATH
    return SPIRV2VALHALL_DEFAULT_CLANG_AST_DUMPER_PATH;
#else
    return "build/clang_ast_dumper";
#endif
}

bool parse_mbs2_from_ast_json(
    const std::string& ast_json_text,
    uint8_t arch,
    bool disassemble,
    Mbs2Module* out_module,
    std::vector<ExtractedEbin>* out_ebins,
    std::string* err) {
    if (!out_module || !out_ebins) {
        if (err) *err = "Null output pointer in parse_mbs2_from_ast_json";
        return false;
    }
    out_ebins->clear();

    JsonValue root;
    if (!parse_json(ast_json_text, &root, err)) {
        return false;
    }
    Mbs2AstWalker walker(root);
    return walker.parse_root(arch, disassemble, out_module, out_ebins, err);
}

bool parse_mbs2_header_with_clang_ast(
    const std::string& header_text,
    uint8_t arch,
    const Mbs2AstParseOptions& opts,
    Mbs2Module* out_module,
    std::vector<ExtractedEbin>* out_ebins,
    std::string* err) {
    if (!out_module || !out_ebins) {
        if (err) *err = "Null output pointer in parse_mbs2_header_with_clang_ast";
        return false;
    }

    std::string dumper_bin = resolve_clang_ast_dumper_path(opts.clang_ast_dumper_path);
    std::string inc_dir = resolve_cmpbe_include_dir(opts.cmpbe_include_dir);
    std::string res_dir = resolve_clang_resource_dir(opts.clang_resource_dir);

    char src_tmpl[] = "/tmp/spirv2valhall_ast_in_XXXXXX";
    int src_fd = mkstemp(src_tmpl);
    if (src_fd < 0) {
        if (err) *err = "Failed to create temporary input file for clang_ast_dumper";
        return false;
    }
    close(src_fd);
    std::string src_path = std::string(src_tmpl) + ".cpp";
    std::remove(src_tmpl);

    char json_tmpl[] = "/tmp/spirv2valhall_ast_out_XXXXXX";
    int json_fd = mkstemp(json_tmpl);
    if (json_fd < 0) {
        if (err) *err = "Failed to create temporary JSON output file for clang_ast_dumper";
        return false;
    }
    close(json_fd);
    std::string json_path = std::string(json_tmpl) + ".json";
    std::remove(json_tmpl);

    {
        std::ofstream ofs(src_path, std::ios::binary);
        if (!ofs.good()) {
            if (err) *err = "Failed to write temporary MBS2 header: " + src_path;
            return false;
        }
        ofs << header_text;
    }

    std::string cmd = shell_quote(dumper_bin) + " " + shell_quote(src_path) +
                      " -o " + shell_quote(json_path) +
                      " --globals --strict";
    if (opts.verbose) cmd += " --verbose";
    cmd += " -- -x c++ -std=c++17"
           " -include " + shell_quote(inc_dir + "/cmpbe_chunks.h") +
           " -I" + shell_quote(inc_dir) +
           " -Wno-writable-strings -Wno-c++11-narrowing -Wno-missing-field-initializers";
    if (!res_dir.empty()) {
        cmd += " -resource-dir " + shell_quote(res_dir);
    }
    cmd += " 2>&1";

    if (opts.verbose) {
        std::cerr << "[spirv2valhall/ast] cmd: " << cmd << "\n";
    }

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        std::remove(src_path.c_str());
        std::remove(json_path.c_str());
        if (err) *err = "Failed to launch clang_ast_dumper subprocess";
        return false;
    }
    std::string dumper_out;
    char buf[512];
    while (fgets(buf, sizeof(buf), pipe) != nullptr) {
        dumper_out += buf;
    }
    int rc = pclose(pipe);

    if (opts.verbose && !dumper_out.empty()) {
        std::cerr << dumper_out;
    }

    std::remove(src_path.c_str());

    if (rc != 0) {
        std::remove(json_path.c_str());
        if (err) {
            *err = "clang_ast_dumper failed (exit=" + std::to_string(rc) + "):\n" + dumper_out;
        }
        return false;
    }

    std::ifstream jfs(json_path);
    if (!jfs.good()) {
        std::remove(json_path.c_str());
        if (err) *err = "Failed to open clang_ast_dumper JSON output: " + json_path;
        return false;
    }
    std::ostringstream joss;
    joss << jfs.rdbuf();
    jfs.close();
    std::remove(json_path.c_str());

    return parse_mbs2_from_ast_json(joss.str(), arch, opts.disassemble, out_module, out_ebins, err);
}

}  // namespace valhall::spirv2valhall

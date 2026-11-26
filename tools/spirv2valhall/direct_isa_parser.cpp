#include "direct_isa_parser.h"

#include "valhall_cfg.h"
#include "valhall_pass1.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

namespace valhall::spirv2valhall {

namespace {

std::string trim(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

std::string strip_comment(const std::string& s) {
    size_t p = s.find('#');
    size_t q = s.find(';');
    size_t cut = std::min(p, q);
    return cut == std::string::npos ? s : s.substr(0, cut);
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        if (c == '\\') out += "\\\\";
        else if (c == '"') out += "\\\"";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20) out += ' ';
        else out += static_cast<char>(c);
    }
    return out;
}

bool parse_fau_kv(const std::string& tok, std::pair<uint32_t, uint32_t>* out) {
    std::string t = trim(tok);
    if (t.empty()) return false;
    size_t eq = t.find('=');
    if (eq == std::string::npos) return false;
    std::string lhs = trim(t.substr(0, eq));
    std::string rhs = trim(t.substr(eq + 1));
    if (!lhs.empty() && (lhs[0] == 'u' || lhs[0] == 'U')) lhs = lhs.substr(1);
    if (lhs.empty() || rhs.empty()) return false;
    uint32_t idx = static_cast<uint32_t>(std::strtoul(lhs.c_str(), nullptr, 0));
    uint32_t val = static_cast<uint32_t>(std::strtoul(rhs.c_str(), nullptr, 0));
    *out = {idx, val};
    return true;
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool parse_valhall_text(std::istream& is,
                        std::vector<DirectIsaCase>* out_cases,
                        std::string* out_err) {
    DirectIsaCase cur;
    cur.case_id = "default";
    bool has_any = false;

    std::string raw_line;
    int line_no = 0;
    while (std::getline(is, raw_line)) {
        ++line_no;
        std::string line = trim(strip_comment(raw_line));
        if (line.empty()) continue;

        if (line[0] == '@') {
            std::istringstream ls(line);
            std::string dir;
            ls >> dir;
            if (dir == "@case") {
                if (has_any && !cur.words.empty()) {
                    out_cases->push_back(cur);
                    cur = DirectIsaCase{};
                }
                std::string name;
                ls >> name;
                cur.case_id = name.empty() ? ("case_" + std::to_string(out_cases->size())) : name;
                has_any = true;
            } else if (dir == "@arch") {
                int a = 11;
                ls >> a;
                cur.arch = static_cast<uint8_t>(a);
                has_any = true;
            } else if (dir == "@stage") {
                std::string st;
                ls >> st;
                if (!st.empty()) cur.stage = st;
                has_any = true;
            } else if (dir == "@fau") {
                std::string kv;
                while (ls >> kv) {
                    std::pair<uint32_t, uint32_t> p;
                    if (parse_fau_kv(kv, &p)) cur.fau_words.push_back(p);
                }
                has_any = true;
            } else {
                if (out_err) *out_err = "Unknown directive '" + dir + "' at line " + std::to_string(line_no);
                return false;
            }
            continue;
        }

        // Expect a 64-bit hex instruction word (with or without 0x prefix).
        std::istringstream ls(line);
        std::string tok;
        ls >> tok;
        if (tok.rfind("0x", 0) == 0 || tok.rfind("0X", 0) == 0) {
            tok = tok.substr(2);
        }
        if (tok.empty() || !std::all_of(tok.begin(), tok.end(), [](unsigned char c) { return std::isxdigit(c); })) {
            if (out_err) *out_err = "Expected 64-bit hex word at line " + std::to_string(line_no) + ": " + line;
            return false;
        }
        uint64_t w = std::strtoull(tok.c_str(), nullptr, 16);
        cur.words.push_back(w);
        has_any = true;
    }

    if (has_any && !cur.words.empty()) {
        out_cases->push_back(cur);
    }
    return true;
}

bool parse_tsv_batch(std::istream& is,
                     std::vector<DirectIsaCase>* out_cases,
                     std::string* /*out_err*/) {
    std::string line;
    while (std::getline(is, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string case_id, arch_s, stage_s, words_s, fau_s;
        if (!std::getline(ss, case_id, '\t') ||
            !std::getline(ss, arch_s, '\t') ||
            !std::getline(ss, stage_s, '\t') ||
            !std::getline(ss, words_s, '\t')) {
            continue;
        }
        std::getline(ss, fau_s, '\t');  // optional 5th column: u0=0x3f800000,u1=0x40000000

        DirectIsaCase c;
        c.case_id = trim(case_id);
        c.arch = static_cast<uint8_t>(std::atoi(arch_s.c_str()));
        c.stage = trim(stage_s);

        std::stringstream ws(words_s);
        std::string tok;
        while (std::getline(ws, tok, ',')) {
            tok = trim(tok);
            if (tok.empty()) continue;
            c.words.push_back(std::strtoull(tok.c_str(), nullptr, 16));
        }

        if (!fau_s.empty()) {
            std::stringstream fs(fau_s);
            std::string ftok;
            while (std::getline(fs, ftok, ',')) {
                std::pair<uint32_t, uint32_t> p;
                if (parse_fau_kv(ftok, &p)) c.fau_words.push_back(p);
            }
        }
        if (!c.words.empty()) {
            out_cases->push_back(std::move(c));
        }
    }
    return true;
}

}  // namespace

bool parse_direct_isa_file(const std::string& path,
                           std::vector<DirectIsaCase>* out_cases,
                           std::string* out_err) {
    std::ifstream ifs(path);
    if (!ifs.good()) {
        if (out_err) *out_err = "Cannot open direct-isa file: " + path;
        return false;
    }
    if (ends_with(path, ".valhall")) {
        return parse_valhall_text(ifs, out_cases, out_err);
    }
    return parse_tsv_batch(ifs, out_cases, out_err);
}

int run_direct_isa_file(const std::string& path) {
    std::vector<DirectIsaCase> cases;
    std::string err;
    if (!parse_direct_isa_file(path, &cases, &err)) {
        std::cerr << "[spirv2valhall] ERROR: " << err << "\n";
        return 1;
    }

    for (const auto& c : cases) {
        std::vector<uint8_t> bytes;
        bytes.reserve(c.words.size() * 8);
        for (uint64_t w : c.words) {
            for (int b = 0; b < 8; ++b) {
                bytes.push_back(static_cast<uint8_t>((w >> (b * 8)) & 0xFFu));
            }
        }

        valhall::ShaderStage stage = valhall::ShaderStage::Compute;
        if (c.stage == "fragment") stage = valhall::ShaderStage::Fragment;
        else if (c.stage == "vertex") stage = valhall::ShaderStage::Vertex;

        auto insts = valhall::decode_valhall_shader(bytes.data(), bytes.size(), c.arch, false);
        bool decoded_ok = !insts.empty() && insts[0].valid && insts[0].mnemonic != valhall::Mnemonic::INVALID;
        std::string mn = decoded_ok ? valhall::mnemonic_to_string(insts[0].mnemonic) : "";
        std::string cmn = decoded_ok ? valhall::mnemonic_to_string(insts[0].canonical_mnemonic) : "";
        std::string var = decoded_ok ? valhall::variant_to_string(insts[0].variant) : "";
        std::string disasm = decoded_ok ? insts[0].disassembly : "";

        bool pass1_ok = false;
        size_t ssa_nodes = 0;
        std::vector<std::string> ssa_ops;
        std::string err_msg;

        if (decoded_ok) {
            valhall::ValhallTranslateOptions vopts;
            vopts.stage = stage;
            vopts.arch = c.arch;
            vopts.tolerant_ssa = true;
            vopts.run_ssa_opt = false;
            auto cfg = valhall::cfg::build_cfg(insts);
            valhall::cfg::ControlFlowGraph empty_sec;
            auto p1 = valhall::compiler::run_pass1_lowering(cfg, empty_sec, vopts);
            pass1_ok = p1.ok;
            err_msg = p1.error_message;
            if (p1.ok) {
                ssa_nodes = p1.ssa.nodes.size();
                for (const auto& n : p1.ssa.nodes) {
                    if (ssa_ops.size() < 24) {
                        ssa_ops.push_back(valhall::tables::ssa_op_to_string(n.op));
                    }
                }
            }
        } else {
            err_msg = "decode failed";
        }

        std::cout << "{\"case_id\":\"" << json_escape(c.case_id)
                  << "\",\"decoded_ok\":" << (decoded_ok ? "true" : "false")
                  << ",\"mnemonic\":\"" << json_escape(mn)
                  << "\",\"canonical_mnemonic\":\"" << json_escape(cmn)
                  << "\",\"variant\":\"" << json_escape(var)
                  << "\",\"disasm\":\"" << json_escape(disasm)
                  << "\",\"fau_count\":" << c.fau_words.size()
                  << ",\"pass1_ok\":" << (pass1_ok ? "true" : "false")
                  << ",\"ssa_nodes\":" << ssa_nodes
                  << ",\"ssa_ops\":[";
        for (size_t k = 0; k < ssa_ops.size(); ++k) {
            if (k) std::cout << ",";
            std::cout << "\"" << json_escape(ssa_ops[k]) << "\"";
        }
        std::cout << "],\"error\":\"" << json_escape(err_msg) << "\"}\n";
    }
    return 0;
}

}  // namespace valhall::spirv2valhall

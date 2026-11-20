#include "valhall_ssa_json.h"

#include <cstdio>
#include <sstream>

namespace valhall::compiler {

namespace {

void write_json_escaped(std::ostringstream& os, const std::string& s) {
    os << '"';
    for (char c : s) {
        switch (c) {
            case '"': os << "\\\""; break;
            case '\\': os << "\\\\"; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20u) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(c)));
                    os << buf;
                } else {
                    os << c;
                }
                break;
        }
    }
    os << '"';
}

} // namespace

const char* val_type_name(ValType vt) {
    switch (vt) {
        case ValType::Void: return "Void";
        case ValType::Bool: return "Bool";
        case ValType::U32: return "U32";
        case ValType::S32: return "S32";
        case ValType::F32: return "F32";
        case ValType::U64: return "U64";
        case ValType::S64: return "S64";
        case ValType::V2F32: return "V2F32";
        case ValType::V2U32: return "V2U32";
        case ValType::V2S32: return "V2S32";
        case ValType::V4F32: return "V4F32";
        case ValType::V4U32: return "V4U32";
        case ValType::V4S32: return "V4S32";
        case ValType::V3U32: return "V3U32";
        case ValType::V2Bool: return "V2Bool";
        case ValType::TupleU32x2: return "TupleU32x2";
        case ValType::TupleValToken: return "TupleValToken";
        case ValType::Token: return "Token";
    }
    return "Void";
}

std::string serialize_ssa_module_json(const ValhallSsaModule& mod) {
    std::ostringstream os;
    os << "{\"blocks\":[";
    for (size_t b = 0; b < mod.blocks.size(); ++b) {
        const auto& blk = mod.blocks[b];
        if (b > 0) os << ',';
        os << "{\"id\":" << blk.id
           << ",\"preds\":[";
        for (size_t i = 0; i < blk.preds.size(); ++i) {
            if (i > 0) os << ',';
            os << blk.preds[i];
        }
        os << "],\"phis\":[";
        for (size_t i = 0; i < blk.phis.size(); ++i) {
            if (i > 0) os << ',';
            os << blk.phis[i];
        }
        os << "],\"nodes\":[";
        for (size_t i = 0; i < blk.nodes.size(); ++i) {
            if (i > 0) os << ',';
            os << blk.nodes[i];
        }
        os << "],\"branch_target\":" << blk.branch_target_block
           << ",\"fallthrough\":" << blk.fallthrough_block
           << ",\"is_cond\":" << (blk.is_conditional_branch ? "true" : "false")
           << ",\"is_uncond\":" << (blk.is_unconditional_branch ? "true" : "false")
           << ",\"is_end\":" << (blk.is_terminated_by_end ? "true" : "false")
           << ",\"is_discard\":" << (blk.is_terminated_by_discard ? "true" : "false")
           << ",\"is_loop_header\":" << (blk.is_loop_header ? "true" : "false")
           << ",\"loop_backedge\":" << blk.loop_backedge_block
           << ",\"loop_merge\":" << blk.loop_merge_block
           << ",\"sel_merge\":" << blk.selection_merge_block
           << ",\"branch_cond\":" << blk.branch_cond
           << '}';
    }
    os << "],\"nodes\":[";
    for (size_t i = 1; i < mod.nodes.size(); ++i) {
        const auto& n = mod.nodes[i];
        if (i > 1) os << ',';
        os << "{\"id\":" << n.id
           << ",\"op\":\"" << tables::ssa_op_to_string(n.op) << '"'
           << ",\"type\":\"" << val_type_name(n.type) << '"'
           << ",\"blk\":" << n.block_id
           << ",\"args\":[";
        for (size_t a = 0; a < n.args.size(); ++a) {
            if (a > 0) os << ',';
            os << n.args[a];
        }
        os << ']';
        if (!n.phi_incomings.empty()) {
            os << ",\"phi\":[";
            for (size_t p = 0; p < n.phi_incomings.size(); ++p) {
                if (p > 0) os << ',';
                os << '[' << n.phi_incomings[p].pred_block << ',' << n.phi_incomings[p].value << ']';
            }
            os << ']';
        }
        if (n.imm_u32 != 0) os << ",\"imm_u32\":" << n.imm_u32;
        if (n.imm_u32_2 != 0) os << ",\"imm_u32_2\":" << n.imm_u32_2;
        if (n.imm_u64 != 0) os << ",\"imm_u64\":" << n.imm_u64;
        if (n.aux_index != 0) os << ",\"aux_index\":" << n.aux_index;
        if (n.aux_bool) os << ",\"aux_bool\":true";
        if (n.aux_bool2) os << ",\"aux_bool2\":true";
        if (!n.aux_str.empty()) {
            os << ",\"aux_str\":";
            write_json_escaped(os, n.aux_str);
        }
        if (n.orig_inst_idx >= 0) {
            os << ",\"orig_pc\":" << n.orig_pc;
        }
        os << '}';
    }
    os << "]}";
    return os.str();
}

} // namespace valhall::compiler

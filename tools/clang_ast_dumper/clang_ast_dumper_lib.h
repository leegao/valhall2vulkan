#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace valhall::ast_dumper {

/// Pure C++17 in-memory representation of a Clang AST initializer expression
/// (no LLVM/Clang headers required by callers).
struct AstExprNode {
    enum class Kind : uint8_t {
        Null = 0,
        IntLiteral,
        BoolLiteral,
        StringLiteral,
        Ident,
        UnaryOp,
        Cast,
        InitList,
        Other,
    };

    Kind kind = Kind::Null;
    int64_t int_val = 0;
    uint64_t uint_val = 0;
    bool is_signed = false;
    bool bool_val = false;
    std::string name;         // For Ident
    std::string op;           // For UnaryOp (e.g. "&", "-")
    std::string str_val;      // Printable string value
    std::vector<uint8_t> bytes;  // Exact raw bytes for StringLiteral (lossless binary!)
    std::string record_type;  // Struct name for record InitList
    std::shared_ptr<AstExprNode> sub;  // For Cast / UnaryOp
    std::vector<std::shared_ptr<AstExprNode>> items;              // Positional elements
    std::map<std::string, std::shared_ptr<AstExprNode>> fields;   // Named struct fields
};

struct AstGlobalVar {
    std::string name;
    std::string type;
    std::shared_ptr<AstExprNode> init;
};

struct AstGlobalsDumpResult {
    bool ok = false;
    size_t unhandled_count = 0;
    std::vector<AstGlobalVar> globals;
    std::string error_message;
};

/// Parse C/C++ source code in-process via Clang-21 (`runToolOnCodeWithArgs`) and return
/// all main-file top-level VarDecls directly as C++ structs (zero subprocess, zero JSON).
AstGlobalsDumpResult dump_globals_in_process(
    const std::string& source_code,
    const std::string& virtual_filename,
    const std::vector<std::string>& clang_args,
    bool strict = true,
    bool verbose = false);

/// Entry point for the standalone `clang_ast_dumper` CLI executable.
int run_clang_ast_dumper_cli(int argc, const char** argv);

}  // namespace valhall::ast_dumper

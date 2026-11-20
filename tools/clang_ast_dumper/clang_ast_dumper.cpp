#include "clang_ast_dumper_lib.h"

int main(int argc, const char** argv) {
    return valhall::ast_dumper::run_clang_ast_dumper_cli(argc, argv);
}

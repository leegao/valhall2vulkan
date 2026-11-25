cmake_minimum_required(VERSION 3.16)
project(valhall2spirv LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
    set(VALHALL_PYTHON "${Python3_EXECUTABLE}")
else()
    set(VALHALL_PYTHON "python3")
endif()

find_path(VULKAN_INCLUDE_DIR vulkan/vulkan.h)
if(NOT VULKAN_INCLUDE_DIR AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/predownloaded/vulkan_headers.zip")
    set(_vk_deps_dir "${CMAKE_CURRENT_BINARY_DIR}/_deps/vulkan_headers")
    if(NOT EXISTS "${_vk_deps_dir}/include/vulkan/vulkan.h")
        file(MAKE_DIRECTORY "${_vk_deps_dir}")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E tar xf "${CMAKE_CURRENT_SOURCE_DIR}/predownloaded/vulkan_headers.zip"
            WORKING_DIRECTORY "${_vk_deps_dir}"
            RESULT_VARIABLE _vk_extract_rc
        )
        if(NOT _vk_extract_rc EQUAL 0)
            message(FATAL_ERROR "Failed to extract predownloaded/vulkan_headers.zip")
        endif()
    endif()
    set(VULKAN_INCLUDE_DIR "${_vk_deps_dir}/include" CACHE PATH "Vulkan headers include dir" FORCE)
endif()

find_library(VULKAN_LIBRARY
    NAMES vulkan libvulkan.so.1 vulkan-1
    HINTS /usr/lib/x86_64-linux-gnu
    REQUIRED
)

add_custom_target(codegen
    COMMAND "${VALHALL_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_valhall_spec.py"
    COMMAND "${VALHALL_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_spirv_builder.py"
    COMMAND "${VALHALL_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_spirv_parser.py"
    COMMAND "${VALHALL_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_valhall_tables.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Regenerating Valhall ISA, CSF descriptor, and SpirvBuilder C++17 sources"
)

find_program(PDFLATEX_EXECUTABLE
    NAMES pdflatex-quiet pdflatex
    DOC "PDFLaTeX compiler"
)

if(PDFLATEX_EXECUTABLE)
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/tools/reference/generated/valhall-isa-reference.tex"
        COMMAND "${VALHALL_PYTHON}" -m tools.reference.gen_valhall_spec_tex
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/reference/template.tex"
            "${CMAKE_CURRENT_SOURCE_DIR}/specs/kraid/isa-v9-v14.xml"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/reference/gen_valhall_spec_tex.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/reference/isa_semantics.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/reference/v3_enrichments.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/reference/fmt_helpers.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/reference/csf_abi_map.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/reference/operands_doc.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/reference/mined_examples.py"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Generating tools/reference/generated/valhall-isa-reference.tex..."
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${CMAKE_BINARY_DIR}/reference/valhall-isa-reference.pdf"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/reference"
        COMMAND "${PDFLATEX_EXECUTABLE}" -interaction=nonstopmode -output-directory "${CMAKE_BINARY_DIR}/reference" "${CMAKE_CURRENT_SOURCE_DIR}/tools/reference/generated/valhall-isa-reference.tex" 2>&1 > /dev/null
        COMMAND "${PDFLATEX_EXECUTABLE}" -interaction=nonstopmode -output-directory "${CMAKE_BINARY_DIR}/reference" "${CMAKE_CURRENT_SOURCE_DIR}/tools/reference/generated/valhall-isa-reference.tex" 2>&1 > /dev/null
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${CMAKE_BINARY_DIR}/reference/valhall-isa-reference.pdf" "${CMAKE_SOURCE_DIR}/valhall-isa-reference.pdf"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/reference/generated/valhall-isa-reference.tex"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/reference/template.tex"
            "${CMAKE_CURRENT_SOURCE_DIR}/specs/kraid/isa-v9-v14.xml"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/reference/gen_valhall_spec_tex.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/reference/isa_semantics.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/reference/v3_enrichments.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/reference/fmt_helpers.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/reference/csf_abi_map.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/reference/operands_doc.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/reference/mined_examples.py"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Generating Valhall valhall-isa-reference.pdf (${PDFLATEX_EXECUTABLE}))"
        VERBATIM
    )
    add_custom_target(valhall_isa_reference ALL
        DEPENDS "${CMAKE_BINARY_DIR}/reference/valhall-isa-reference.pdf"
    )
else()
    message(STATUS "pdflatex not found; skipping Valhall ISA reference PDF generation")
endif()

add_test(NAME test_valhall_isa_reference
    COMMAND "${VALHALL_PYTHON}" -m unittest tools/reference/test_valhall_isa_reference.py
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_library(spirv_reflect STATIC
    third_party/spirv-reflect/spirv_reflect.c
)
target_include_directories(spirv_reflect PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/spirv-reflect"
)
target_compile_options(spirv_reflect PRIVATE -Wall -Wextra -Wno-unused-parameter)

set_source_files_properties(src/generated/valhall_isa_generated.cpp PROPERTIES
    COMPILE_OPTIONS "-O1;-Wall;-Wextra;-Wno-unused-parameter"
)
set_source_files_properties(src/generated/valhall_spirv_builder_generated.cpp PROPERTIES
    COMPILE_OPTIONS "-ffunction-sections;-fdata-sections;-Wall;-Wextra;-Wno-unused-parameter"
)

set(VALHALL_OPERANDS_GENERATED_CPP "${CMAKE_CURRENT_SOURCE_DIR}/src/generated/valhall_operands_generated.cpp")
add_custom_command(
    OUTPUT "${VALHALL_OPERANDS_GENERATED_CPP}"
    COMMAND "${VALHALL_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_valhall_tables.py"
        --lowering "${CMAKE_CURRENT_SOURCE_DIR}/specs/lowering/valhall_lowering.json"
        --out-dir "${CMAKE_CURRENT_SOURCE_DIR}/src/generated"
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/specs/lowering/valhall_lowering.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_valhall_tables.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/valhall_lowering_dsl.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Regenerating Valhall lowering tables and operands from specs/lowering/valhall_lowering.json"
    VERBATIM
)

add_library(valhall2spirv STATIC
    src/generated/valhall_isa_generated.cpp
    src/generated/valhall_csf_descriptors_generated.cpp
    src/generated/valhall_spirv_builder_generated.cpp
    src/generated/valhall_spirv_parser_generated.cpp
    src/generated/valhall_tables_generated.cpp
    src/ssa/valhall_ssa.cpp
    src/ssa/valhall_ssa_json.cpp
    src/spirv2ssa/spirv_module.cpp
    src/spirv2ssa/spirv2ssa_mem.cpp
    src/spirv2ssa/spirv2ssa_ops.cpp
    src/spirv2ssa/spirv2ssa.cpp
    src/generated/valhall_operands_generated.cpp
    src/valhall_pass1.cpp
    src/valhall_pass1_handlers.cpp
    src/valhall_tiler.cpp
    src/valhall_spirv_builder.cpp
    src/valhall_cfg.cpp
    src/dataflow/worklist.cpp
    src/dataflow/liveness.cpp
    src/dataflow/tracked_val.cpp
    src/dataflow/transfer.cpp
    src/dataflow/pointer_analysis.cpp
    src/dataflow/ssa_opt_utils.cpp
    src/dataflow/ssa_dce.cpp
    src/dataflow/ssa_cse.cpp
    src/dataflow/ssa_sccp.cpp
    src/dataflow/ssa_spill_mem2reg.cpp
    src/dataflow/ssa_sroa.cpp
    src/dataflow/ssa_buffer_symbolize.cpp
    src/dataflow/ssa_defau_devirt.cpp
    src/dataflow/ssa_fact_prover.cpp
    src/dataflow/ssa_fp_opt.cpp
    src/dataflow/ssa_opt_pipeline.cpp
    src/valhall_to_spirv.cpp
    src/valhall_csf_pipeline.cpp
)
target_include_directories(valhall2spirv PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ssa"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/spirv2ssa"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/dataflow"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/generated"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/spirv-reflect/include"
    "${VULKAN_INCLUDE_DIR}"
)
target_compile_options(valhall2spirv PUBLIC -Wno-attributes)
target_compile_options(valhall2spirv PRIVATE -Wall -Wextra -Wno-unused-parameter)
if(UNIX AND NOT APPLE)
    target_link_options(valhall2spirv INTERFACE -Wl,--gc-sections)
endif()
target_link_libraries(valhall2spirv PUBLIC
    spirv_reflect
    "${VULKAN_LIBRARY}"
)

add_library(moc_hook SHARED
    tools/moc_hook/moc_hook.c
)
set_target_properties(moc_hook PROPERTIES OUTPUT_NAME moc_hook)
target_compile_options(moc_hook PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(moc_hook PRIVATE ${CMAKE_DL_LIBS})

if(NOT ANDROID)
    add_library(malioc_inproc STATIC
        tools/malioc_inproc/malioc_inproc.cpp
    )
    target_include_directories(malioc_inproc PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/malioc_inproc"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/spirv2valhall"
    )
    target_compile_options(malioc_inproc PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)
    target_link_libraries(malioc_inproc PUBLIC ${CMAKE_DL_LIBS})
endif()

add_library(spirv2valhall_lib STATIC
    tools/spirv2valhall/spirv_inspect.cpp
    tools/spirv2valhall/mbs2_direct_reader.cpp
    tools/spirv2valhall/mbs2_context_dumper.cpp
    tools/spirv2valhall/isa_enricher.cpp
    tools/spirv2valhall/spirv_roundtrip.cpp
    tools/spirv2valhall/equiv_bundle.cpp
    tools/spirv2valhall/spirv2valhall.cpp
)
target_include_directories(spirv2valhall_lib PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/spirv2valhall"
)
target_compile_options(spirv2valhall_lib PRIVATE -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(spirv2valhall_lib PUBLIC valhall2spirv)
if(TARGET malioc_inproc)
    target_link_libraries(spirv2valhall_lib PUBLIC malioc_inproc)
endif()

add_executable(spirv2valhall
    tools/spirv2valhall/main.cpp
)
target_compile_options(spirv2valhall PRIVATE -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(spirv2valhall PRIVATE spirv2valhall_lib)

foreach(test_name IN ITEMS test_isa_decoder test_csf_descriptors test_spirv_builder test_two_pass_compiler test_dataflow test_ssa_opt test_spirv2ssa)
    add_executable(${test_name} "tests/${test_name}.cpp")
    target_include_directories(${test_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    target_compile_options(${test_name} PRIVATE -Wall -Wextra -Wno-unused-parameter)
    target_link_libraries(${test_name} PRIVATE valhall2spirv spirv2valhall_lib)
    add_test(NAME ${test_name}
        COMMAND ${test_name}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
endforeach()

add_executable(test_spirv2valhall tests/test_spirv2valhall.cpp)
target_include_directories(test_spirv2valhall PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
target_compile_options(test_spirv2valhall PRIVATE -Wall -Wextra -Wno-unused-parameter)
target_compile_definitions(test_spirv2valhall PRIVATE
    TEST_BUILD_DIR="${CMAKE_CURRENT_BINARY_DIR}"
    SPIRV2VALHALL_BIN_PATH="$<TARGET_FILE:spirv2valhall>"
)
target_link_libraries(test_spirv2valhall PRIVATE spirv2valhall_lib)
add_dependencies(test_spirv2valhall spirv2valhall moc_hook)
add_test(NAME test_spirv2valhall
    COMMAND test_spirv2valhall
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_executable(test_moc_roundtrip_compute tests/test_moc_roundtrip_compute.cpp)
target_include_directories(test_moc_roundtrip_compute PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
target_compile_options(test_moc_roundtrip_compute PRIVATE -Wall -Wextra -Wno-unused-parameter)
target_compile_definitions(test_moc_roundtrip_compute PRIVATE
    TEST_BUILD_DIR="${CMAKE_CURRENT_BINARY_DIR}"
    MOC_HOOK_PATH="$<TARGET_FILE:moc_hook>"
)
target_link_libraries(test_moc_roundtrip_compute PRIVATE valhall2spirv spirv2valhall_lib)
add_dependencies(test_moc_roundtrip_compute moc_hook spirv2valhall)
add_test(NAME test_moc_roundtrip_compute
    COMMAND test_moc_roundtrip_compute
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_executable(test_moc_roundtrip_graphics tests/test_moc_roundtrip_graphics.cpp)
target_include_directories(test_moc_roundtrip_graphics PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
target_compile_options(test_moc_roundtrip_graphics PRIVATE -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(test_moc_roundtrip_graphics PRIVATE valhall2spirv spirv2valhall_lib)
add_test(NAME test_moc_roundtrip_graphics
    COMMAND test_moc_roundtrip_graphics
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_executable(test_vkcube_frame1_replay tests/test_vkcube_frame1_replay.cpp)
target_include_directories(test_vkcube_frame1_replay PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
target_compile_options(test_vkcube_frame1_replay PRIVATE -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(test_vkcube_frame1_replay PRIVATE valhall2spirv)
add_test(NAME test_vkcube_frame1_replay
    COMMAND test_vkcube_frame1_replay
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_library(mali_csf_dump_core STATIC
    tools/mali_csf_dump_layer/csf_config_log.cpp
    tools/mali_csf_dump_layer/csf_va_tracker.cpp
    tools/mali_csf_dump_layer/csf_interpreter.cpp
    tools/mali_csf_dump_layer/csf_dumper.cpp
    tools/mali_csf_dump_layer/csf_elf_hooks.cpp
    tools/mali_csf_dump_layer/csf_layer_state.cpp
)
set_target_properties(mali_csf_dump_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(mali_csf_dump_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/mali_csf_dump_layer"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/generated"
    "${VULKAN_INCLUDE_DIR}"
)
target_compile_options(mali_csf_dump_core PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(mali_csf_dump_core PUBLIC ${CMAKE_DL_LIBS})

add_library(VkLayer_mali_csf_dump SHARED
    tools/mali_csf_dump_layer/mali_csf_dump_layer.cpp
)
target_include_directories(VkLayer_mali_csf_dump PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/mali_csf_dump_layer"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/generated"
    "${VULKAN_INCLUDE_DIR}"
)
target_compile_options(VkLayer_mali_csf_dump PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(VkLayer_mali_csf_dump PRIVATE mali_csf_dump_core ${CMAKE_DL_LIBS})
add_custom_command(TARGET VkLayer_mali_csf_dump POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/mali_csf_dump_layer/VkLayer_mali_csf_dump.json"
        "$<TARGET_FILE_DIR:VkLayer_mali_csf_dump>/VkLayer_mali_csf_dump.json"
)

if(NOT ANDROID)
    add_library(mock_lazy_mali_driver SHARED
        tools/mali_csf_dump_layer/mock_lazy_mali_driver.cpp
    )
    target_compile_options(mock_lazy_mali_driver PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)

    add_executable(test_csf_dump_layer tests/test_csf_dump_layer.cpp)
    target_include_directories(test_csf_dump_layer PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/mali_csf_dump_layer"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/generated"
        "${VULKAN_INCLUDE_DIR}"
    )
    target_compile_options(test_csf_dump_layer PRIVATE -Wall -Wextra -Wno-unused-parameter)
    target_compile_definitions(test_csf_dump_layer PRIVATE
        CSF_DUMP_LAYER_PATH="$<TARGET_FILE:VkLayer_mali_csf_dump>"
        CSF_DUMP_LAYER_DIR="$<TARGET_FILE_DIR:VkLayer_mali_csf_dump>"
        MOCK_LAZY_MALI_PATH="$<TARGET_FILE:mock_lazy_mali_driver>"
    )
    target_link_libraries(test_csf_dump_layer PRIVATE mali_csf_dump_core "${VULKAN_LIBRARY}" ${CMAKE_DL_LIBS})
    add_dependencies(test_csf_dump_layer VkLayer_mali_csf_dump mock_lazy_mali_driver)
    add_test(NAME test_csf_dump_layer
        COMMAND test_csf_dump_layer
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
endif()

add_test(NAME test_panfrost_z3_model
    COMMAND "${VALHALL_PYTHON}" -m unittest tools/verify/test_panfrost_z3_model.py
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_test(NAME test_lowering_extractor
    COMMAND "${VALHALL_PYTHON}" -m unittest tools/verify/test_lowering_extractor.py
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_test(NAME test_all_paths_miter
    COMMAND "${VALHALL_PYTHON}" -m unittest tools/verify/test_all_paths_miter.py
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_test(NAME test_shader_equiv
    COMMAND "${VALHALL_PYTHON}" -m unittest tools/verify/test_shader_equiv.py
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_test(NAME test_canonicalize
    COMMAND "${VALHALL_PYTHON}" -m unittest tools/verify/test_canonicalize.py
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_test(NAME test_cec
    COMMAND "${VALHALL_PYTHON}" -m unittest tools/verify/test_cec.py
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_test(NAME test_coverage_synthesizer
    COMMAND "${VALHALL_PYTHON}" -m unittest tools/coverage/test_coverage_synthesizer.py
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_test(NAME test_corpus_synth
    COMMAND "${VALHALL_PYTHON}" -m unittest tools/corpus_synth/test_corpus_synth.py
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

add_test(NAME test_v31_domains_and_cfg
    COMMAND "${VALHALL_PYTHON}" -m unittest tools/symbolic_ssa/test_v31_domains_and_cfg.py
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

add_test(NAME test_v35_backfill_and_soundness
    COMMAND "${VALHALL_PYTHON}" -m unittest tools/symbolic_ssa/test_v35_backfill_and_soundness.py
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

add_test(NAME test_v36_concolic_and_lowering
    COMMAND "${VALHALL_PYTHON}" -m unittest tools/symbolic_ssa/test_v36_concolic_and_lowering.py
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

add_test(NAME test_v37_sweep_and_fba18fbe
    COMMAND "${VALHALL_PYTHON}" -m unittest tools/symbolic_ssa/test_v37_sweep_and_fba18fbe.py
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

add_test(NAME test_v38_backfill_pushconst_subword_fau
    COMMAND "${VALHALL_PYTHON}" -m unittest tools/symbolic_ssa/test_v38_backfill_pushconst_subword_fau.py
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

add_test(
    NAME test_v39_rodata_and_abi_parity
    COMMAND ${Python3_EXECUTABLE} -m unittest tools.symbolic_ssa.test_v39_rodata_and_abi_parity
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

add_test(
    NAME test_v40_clamp_idvs_fexp
    COMMAND ${Python3_EXECUTABLE} -m unittest tools.symbolic_ssa.test_v40_clamp_idvs_fexp
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

add_test(
    NAME test_v41_ld_pka_srt_resolution
    COMMAND ${Python3_EXECUTABLE} -m unittest tools.symbolic_ssa.test_v41_ld_pka_srt_resolution
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
    add_test(
        NAME test_v42_sweep_lowering_and_record
        COMMAND ${Python3_EXECUTABLE} -m unittest tools.symbolic_ssa.test_v42_sweep_lowering_and_record
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

add_test(NAME test_v43_dominance_asymmetry_and_iabs
    COMMAND ${Python3_EXECUTABLE} -m unittest tools/symbolic_ssa/test_v43_dominance_asymmetry_and_iabs.py -v
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(test_v43_dominance_asymmetry_and_iabs PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}:$ENV{PYTHONPATH}"
)

add_test(NAME test_v44_base_vertex_and_triage
    COMMAND ${Python3_EXECUTABLE} -m unittest tools/symbolic_ssa/test_v44_base_vertex_and_triage.py -v
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(test_v44_base_vertex_and_triage PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}:$ENV{PYTHONPATH}"
)

add_test(
  NAME test_v45_frag_w_tex_wide_and_fp_range
  COMMAND ${Python3_EXECUTABLE} -m unittest tools/symbolic_ssa/test_v45_frag_w_tex_wide_and_fp_range.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(test_v45_frag_w_tex_wide_and_fp_range PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}:/working_dir/c_43dda7b22cb4bd91/pydeps")


set(_VALHALL_USE_SYSTEM_CLANG FALSE)
set(_VALHALL_USE_BUNDLED_LLVM21 FALSE)
set(VALHALL_LLVM21_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/prebuilts/llvm-21")

if(NOT ANDROID)
    find_program(VALHALL_LLVM_CONFIG NAMES llvm-config llvm-config-21 llvm-config-20 llvm-config-19 llvm-config-18)
    if(VALHALL_LLVM_CONFIG)
        execute_process(COMMAND "${VALHALL_LLVM_CONFIG}" --includedir OUTPUT_VARIABLE _SYS_LLVM_INC OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _rc1)
        execute_process(COMMAND "${VALHALL_LLVM_CONFIG}" --libdir OUTPUT_VARIABLE _SYS_LLVM_LIB OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _rc2)
        execute_process(COMMAND "${VALHALL_LLVM_CONFIG}" --has-rtti OUTPUT_VARIABLE _SYS_LLVM_RTTI OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND "${VALHALL_LLVM_CONFIG}" --version OUTPUT_VARIABLE _SYS_LLVM_VER OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(_rc1 EQUAL 0 AND _rc2 EQUAL 0 AND EXISTS "${_SYS_LLVM_INC}/clang/AST/ASTContext.h" AND
           (EXISTS "${_SYS_LLVM_LIB}/libclang-cpp.so" OR EXISTS "${_SYS_LLVM_LIB}/libclangTooling.a"))
            set(_VALHALL_USE_SYSTEM_CLANG TRUE)
            message(STATUS "Clang AST dumper: using system LLVM/Clang ${_SYS_LLVM_VER} (${VALHALL_LLVM_CONFIG})")
        endif()
    endif()

    if(NOT _VALHALL_USE_SYSTEM_CLANG AND
       EXISTS "${VALHALL_LLVM21_DIR}/lib/libclangTooling.a" AND
       EXISTS "${VALHALL_LLVM21_DIR}/include/clang/AST/ASTContext.h")
        set(_VALHALL_USE_BUNDLED_LLVM21 TRUE)
        message(WARNING "Clang AST dumper: using bundled third_party/prebuilts/llvm-21")
    endif()
endif()

if(_VALHALL_USE_SYSTEM_CLANG OR _VALHALL_USE_BUNDLED_LLVM21)
    add_library(clang_ast_dumper_lib STATIC tools/clang_ast_dumper/clang_ast_dumper_lib.cpp)
    if(_VALHALL_USE_BUNDLED_LLVM21)
        add_library(clang_glibc_compat STATIC tools/clang_ast_dumper/glibc_compat.c)
        target_compile_options(clang_glibc_compat PRIVATE -O2 -Wall -Wextra)
    endif()
    target_include_directories(clang_ast_dumper_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/tools/clang_ast_dumper")

    if(_VALHALL_USE_SYSTEM_CLANG)
        target_include_directories(clang_ast_dumper_lib PRIVATE "${_SYS_LLVM_INC}")
        set(_DUMPER_CXX_FLAGS "-O2;-Wall;-Wextra;-Wno-unused-parameter")
        if(_SYS_LLVM_RTTI STREQUAL "NO")
            list(APPEND _DUMPER_CXX_FLAGS "-fno-rtti")
        endif()
        set_source_files_properties(tools/clang_ast_dumper/clang_ast_dumper_lib.cpp PROPERTIES
            COMPILE_OPTIONS "${_DUMPER_CXX_FLAGS}"
        )

        if(EXISTS "${_SYS_LLVM_LIB}/libclang-cpp.so")
            find_library(_SYS_LLVM_SHARED NAMES LLVM LLVM-${_SYS_LLVM_VER} HINTS "${_SYS_LLVM_LIB}")
            target_link_libraries(clang_ast_dumper_lib PUBLIC
                "${_SYS_LLVM_LIB}/libclang-cpp.so"
                $<$<BOOL:${_SYS_LLVM_SHARED}>:${_SYS_LLVM_SHARED}>
                pthread
                dl
            )
        else()
            execute_process(COMMAND "${VALHALL_LLVM_CONFIG}" --libs core support option targetparser frontendopenmp frontendhlsl frontendoffloading bitreader bitwriter analysis transformutils scalaropts profiledata debuginfodwarf object objectyaml textapi mc mcparser bitstreamreader irreader asmparser binaryformat remarks demangle
                OUTPUT_VARIABLE _SYS_LLVM_LIBS_RAW OUTPUT_STRIP_TRAILING_WHITESPACE)
            execute_process(COMMAND "${VALHALL_LLVM_CONFIG}" --system-libs
                OUTPUT_VARIABLE _SYS_LLVM_SYSLIBS_RAW OUTPUT_STRIP_TRAILING_WHITESPACE)
            separate_arguments(_SYS_LLVM_LIBS UNIX_COMMAND "${_SYS_LLVM_LIBS_RAW}")
            separate_arguments(_SYS_LLVM_SYSLIBS UNIX_COMMAND "${_SYS_LLVM_SYSLIBS_RAW}")
            set(_SYS_CLANG_LIBS
                clangTooling clangFrontend clangSerialization clangDriver clangParse clangSema
                clangAnalysis clangASTMatchers clangEdit clangAST clangLex clangBasic
                clangAPINotes clangSupport
            )
            set(_SYS_CLANG_ARCHIVES "")
            foreach(_clib IN LISTS _SYS_CLANG_LIBS)
                if(EXISTS "${_SYS_LLVM_LIB}/lib${_clib}.a")
                    list(APPEND _SYS_CLANG_ARCHIVES "${_SYS_LLVM_LIB}/lib${_clib}.a")
                endif()
            endforeach()
            target_link_libraries(clang_ast_dumper_lib PUBLIC
                -Wl,--start-group
                ${_SYS_CLANG_ARCHIVES}
                ${_SYS_LLVM_LIBS}
                -Wl,--end-group
                ${_SYS_LLVM_SYSLIBS}
            )
        endif()

        find_program(_SYS_CLANG_BIN NAMES clang clang-21 clang-20 clang-19 clang-18)
        set(_DUMPER_RESOURCE_ARGS "")
        if(_SYS_CLANG_BIN)
            execute_process(COMMAND "${_SYS_CLANG_BIN}" -print-resource-dir OUTPUT_VARIABLE _SYS_RES_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(EXISTS "${_SYS_RES_DIR}/include")
                set(_DUMPER_RESOURCE_ARGS -resource-dir "${_SYS_RES_DIR}")
                set(_TEST_RES_DIR "${_SYS_RES_DIR}")
            endif()
        endif()
    else()
        find_library(VALHALL_ZSTD_LIBRARY NAMES zstd libzstd.so.1 HINTS /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu REQUIRED)
        find_library(VALHALL_ZLIB_LIBRARY NAMES z libz.so.1 HINTS /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu REQUIRED)

        target_include_directories(clang_ast_dumper_lib PRIVATE "${VALHALL_LLVM21_DIR}/include")
        set_source_files_properties(tools/clang_ast_dumper/clang_ast_dumper_lib.cpp PROPERTIES
            COMPILE_OPTIONS "-O2;-fno-rtti;-fno-exceptions;-Wall;-Wextra;-Wno-unused-parameter"
        )
        set_source_files_properties(tools/clang_ast_dumper/glibc_compat.c PROPERTIES
            COMPILE_OPTIONS "-O2;-Wall;-Wextra"
        )

        set(_VALHALL_CLANG_LLVM_LIBS
            clangTooling clangFrontend clangSerialization clangDriver clangParse clangSema
            clangAnalysis clangASTMatchers clangEdit clangAST clangLex clangBasic
            clangAPINotes clangSupport
            LLVMFrontendOpenMP LLVMFrontendHLSL LLVMFrontendOffloading LLVMFrontendAtomic
            LLVMFrontendDirective LLVMWindowsDriver LLVMTransformUtils LLVMScalarOpts
            LLVMAnalysis LLVMProfileData LLVMDebugInfoDWARF LLVMDebugInfoDWARFLowLevel
            LLVMObject LLVMObjectYAML LLVMTextAPI LLVMOption LLVMTargetParser
            LLVMMCParser LLVMMC LLVMBitstreamReader LLVMBitReader LLVMIRReader
            LLVMAsmParser LLVMBinaryFormat LLVMCore LLVMRemarks LLVMSupport LLVMDemangle
        )
        set(_VALHALL_CLANG_LLVM_ARCHIVES "")
        foreach(_lib IN LISTS _VALHALL_CLANG_LLVM_LIBS)
            list(APPEND _VALHALL_CLANG_LLVM_ARCHIVES "${VALHALL_LLVM21_DIR}/lib/lib${_lib}.a")
        endforeach()

        target_link_libraries(clang_ast_dumper_lib PUBLIC
            -Wl,--start-group
            ${_VALHALL_CLANG_LLVM_ARCHIVES}
            clang_glibc_compat
            -Wl,--end-group
            "${VALHALL_ZLIB_LIBRARY}"
            "${VALHALL_ZSTD_LIBRARY}"
            pthread
            dl
        )
        set(_DUMPER_RESOURCE_ARGS -resource-dir "${VALHALL_LLVM21_DIR}/lib/clang/21")
        set(_TEST_RES_DIR "${VALHALL_LLVM21_DIR}/lib/clang/21")
    endif()

    add_executable(clang_ast_dumper tools/clang_ast_dumper/clang_ast_dumper.cpp)
    target_link_libraries(clang_ast_dumper PRIVATE clang_ast_dumper_lib)

    set(VALHALL_HOST_HANDLERS_AST_JSON "${CMAKE_CURRENT_SOURCE_DIR}/tools/verify/generated/valhall_host_handlers.json")
    add_custom_command(
        OUTPUT "${VALHALL_HOST_HANDLERS_AST_JSON}"
        COMMAND $<TARGET_FILE:clang_ast_dumper>
            "${CMAKE_CURRENT_SOURCE_DIR}/src/valhall_pass1_handlers.cpp"
            -o "${VALHALL_HOST_HANDLERS_AST_JSON}"
            --root "${CMAKE_CURRENT_SOURCE_DIR}"
            --strict
            --
            -std=c++17
            -Xclang -fno-spell-checking
            -Wno-unknown-attributes
            ${_DUMPER_RESOURCE_ARGS}
            "-I${CMAKE_CURRENT_SOURCE_DIR}/src/include"
            "-I${CMAKE_CURRENT_SOURCE_DIR}/src/ssa"
            "-I${CMAKE_CURRENT_SOURCE_DIR}/src/generated"
            "-I${VULKAN_INCLUDE_DIR}"
        DEPENDS
            clang_ast_dumper
            "${CMAKE_CURRENT_SOURCE_DIR}/src/valhall_pass1_handlers.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/include/valhall_pass1_handlers.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/ssa/valhall_ssa.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/include/valhall_operands.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/generated/valhall_tables_generated.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/generated/valhall_isa_generated.h"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Dumping Clang AST JSON for src/valhall_pass1_handlers.cpp -> tools/verify/generated/valhall_host_handlers.json"
        VERBATIM
    )
    add_custom_target(valhall_host_handlers_ast ALL
        DEPENDS "${VALHALL_HOST_HANDLERS_AST_JSON}"
    )

    add_test(NAME test_clang_ast_dumper
        COMMAND "${VALHALL_PYTHON}" -m unittest tools/verify/test_clang_ast_dumper.py
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    set_tests_properties(test_clang_ast_dumper PROPERTIES
        ENVIRONMENT "CLANG_AST_DUMPER_BIN=$<TARGET_FILE:clang_ast_dumper>;VALHALL_HOST_HANDLERS_AST_JSON=${VALHALL_HOST_HANDLERS_AST_JSON};CLANG_RESOURCE_DIR=${_TEST_RES_DIR};VULKAN_INCLUDE_DIR=${VULKAN_INCLUDE_DIR}"
    )
else()
    message(WARNING "libclang-cpp.sp is not available on this machine, tools/verify/valhall_host_handlers.json will not be regenerated.")
endif()

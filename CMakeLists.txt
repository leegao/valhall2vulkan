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
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Regenerating Valhall ISA and CSF descriptor C++17 sources"
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

add_library(valhall2spirv STATIC
    src/generated/valhall_isa_generated.cpp
    src/generated/valhall_csf_descriptors_generated.cpp
    src/valhall_spirv_builder.cpp
    src/valhall_cfg.cpp
    src/valhall_to_spirv.cpp
    src/valhall_csf_pipeline.cpp
)
target_include_directories(valhall2spirv PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/generated"
    "${VULKAN_INCLUDE_DIR}"
)
target_compile_options(valhall2spirv PRIVATE -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(valhall2spirv PUBLIC
    spirv_reflect
    "${VULKAN_LIBRARY}"
)

foreach(test_name IN ITEMS test_isa_decoder test_csf_descriptors)
    add_executable(${test_name} "tests/${test_name}.cpp")
    target_include_directories(${test_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    target_compile_options(${test_name} PRIVATE -Wall -Wextra -Wno-unused-parameter)
    target_link_libraries(${test_name} PRIVATE valhall2spirv)
    add_test(NAME ${test_name}
        COMMAND ${test_name}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
endforeach()

add_library(moc_hook SHARED
    tools/moc_hook/moc_hook.c
)
set_target_properties(moc_hook PROPERTIES OUTPUT_NAME moc_hook)
target_compile_options(moc_hook PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(moc_hook PRIVATE ${CMAKE_DL_LIBS})

add_executable(test_moc_roundtrip_compute tests/test_moc_roundtrip_compute.cpp)
target_include_directories(test_moc_roundtrip_compute PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
target_compile_options(test_moc_roundtrip_compute PRIVATE -Wall -Wextra -Wno-unused-parameter)
target_compile_definitions(test_moc_roundtrip_compute PRIVATE
    TEST_BUILD_DIR="${CMAKE_CURRENT_BINARY_DIR}"
    MOC_HOOK_PATH="$<TARGET_FILE:moc_hook>"
)
target_link_libraries(test_moc_roundtrip_compute PRIVATE valhall2spirv)
add_dependencies(test_moc_roundtrip_compute moc_hook)
add_test(NAME test_moc_roundtrip_compute
    COMMAND test_moc_roundtrip_compute
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_executable(test_moc_roundtrip_graphics tests/test_moc_roundtrip_graphics.cpp)
target_include_directories(test_moc_roundtrip_graphics PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
target_compile_options(test_moc_roundtrip_graphics PRIVATE -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(test_moc_roundtrip_graphics PRIVATE valhall2spirv)
add_test(NAME test_moc_roundtrip_graphics
    COMMAND test_moc_roundtrip_graphics
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

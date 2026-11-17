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


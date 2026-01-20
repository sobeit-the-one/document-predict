cmake_minimum_required(VERSION 3.23)
project(document-predict-cpp CXX)

# Enable compile_commands.json generation for better IntelliSense support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set policy version minimum to handle submodule dependencies
if(POLICY CMP0001)
    cmake_policy(SET CMP0001 NEW)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable common library build for llama.cpp and disable CURL
set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build common utils library" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "Use CURL for downloads" FORCE)

# GPU backend options (enable the one matching your hardware)
# For NVIDIA GPUs - requires CUDA toolkit
option(ENABLE_CUDA "Enable CUDA backend for NVIDIA GPUs" ON)
# For cross-platform GPU support (works with NVIDIA, AMD, Intel)
option(ENABLE_VULKAN "Enable Vulkan backend" OFF)

if(ENABLE_CUDA)
    set(GGML_CUDA ON CACHE BOOL "Enable CUDA" FORCE)
endif()
if(ENABLE_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "Enable Vulkan" FORCE)
endif()

# Enable native CPU optimizations
set(GGML_NATIVE ON CACHE BOOL "Enable native CPU optimizations" FORCE)

# Set output directories to unify all outputs in build/bin/
# Use CACHE with FORCE to override llama.cpp's settings
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE PATH "Output directory for executables" FORCE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE PATH "Output directory for libraries" FORCE)
# Override per-configuration directories to remove Release/Debug subdirectories
foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/bin CACHE PATH "Output directory for executables (${OUTPUTCONFIG})" FORCE)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/bin CACHE PATH "Output directory for libraries (${OUTPUTCONFIG})" FORCE)
endforeach()

# Configure minja options before adding subdirectory
set(MINJA_TEST_ENABLED OFF CACHE BOOL "minja: Build with test" FORCE)
set(MINJA_EXAMPLE_ENABLED OFF CACHE BOOL "minja: Build with example" FORCE)

# Add subdirectories
add_subdirectory(llama.cpp)
add_subdirectory(argparse)
add_subdirectory(minja)
add_subdirectory(libpredict)

# Fix MSVC C++20 char8_t issue for llama.cpp
# In C++20, u8"..." literals produce const char8_t[] instead of const char[],
# which breaks llama-chat.cpp. The /Zc:char8_t- flag restores C++17 behavior.
# We use target_compile_options because set_source_files_properties has
# directory scope and won't work from parent CMakeLists.txt.
if(MSVC)
    target_compile_options(llama PRIVATE /Zc:char8_t-)
endif()

# Create executable (CLI)
set(TARGET document-predict)
add_executable(${TARGET} 
    src/main.cpp
    src/prompt_file.cpp
)

# Add include directories for executable
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/libpredict/include
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/minja/include
    ${CMAKE_BINARY_DIR}/_deps/variant-lite-src/include
    ${CMAKE_BINARY_DIR}/_deps/expected-lite-src/include
    ${CMAKE_BINARY_DIR}/_deps/optional-lite-src/include
    ${CMAKE_BINARY_DIR}/_deps/string-view-lite-src/include
    ${CMAKE_BINARY_DIR}/_deps/fmt-src/include
)

# Link libraries for executable
target_link_libraries(${TARGET} PRIVATE 
    libpredict
    argparse::argparse
    minja
)

target_compile_features(${TARGET} PRIVATE cxx_std_20)

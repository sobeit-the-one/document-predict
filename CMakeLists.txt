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

# Create library (DLL)
set(LIB_TARGET document-predict-lib)
add_library(${LIB_TARGET} SHARED
    src/document_predict.cpp
)

# Set DLL export definition for Windows
if(WIN32)
    target_compile_definitions(${LIB_TARGET} PRIVATE DOCUMENT_PREDICT_EXPORTS)
endif()

# Add include directories for library
target_include_directories(${LIB_TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include
)

# Link libraries for library
target_link_libraries(${LIB_TARGET} PRIVATE 
    common
    llama 
    minja
    ${CMAKE_THREAD_LIBS_INIT}
)

target_compile_features(${LIB_TARGET} PRIVATE cxx_std_20)

# Create executable (CLI)
set(TARGET document-predict)
add_executable(${TARGET} 
    src/main.cpp
    src/prompt_file.cpp
)

# Add include directories for executable
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
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
    ${LIB_TARGET}
    argparse::argparse
    minja
)

target_compile_features(${TARGET} PRIVATE cxx_std_20)

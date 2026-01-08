#pragma once

#ifdef _WIN32
    #ifdef DOCUMENT_PREDICT_EXPORTS
        #define DOCUMENT_PREDICT_API __declspec(dllexport)
    #else
        #define DOCUMENT_PREDICT_API __declspec(dllimport)
    #endif
#else
    #define DOCUMENT_PREDICT_API
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Result structure (C-compatible)
struct DocumentPredictResult {
    bool success;
    char* output;  // Allocated string, caller must free with document_predict_free_result
    char* error_message;  // Allocated string if success is false, caller must free
};

/**
 * Generate document prediction (C API - Rust FFI compatible)
 * 
 * @param model_path Path to GGUF model file
 * @param prompt_content The prompt/content to generate from (already formatted, no template processing)
 * @param max_tokens Maximum tokens to generate
 * @param ctx_size Context window size
 * @param n_threads Number of CPU threads (-1 for auto-detect)
 * @param n_gpu_layers Number of layers to offload to GPU (0 for CPU-only, -1 for auto)
 * @param temperature Sampling temperature
 * @param top_k Top-k sampling
 * @param top_p Top-p sampling
 * @param min_p Min-p sampling
 * @param seed RNG seed (-1 for random)
 * 
 * @return Result structure with output or error message
 *         Caller must free result using document_predict_free_result()
 */
DOCUMENT_PREDICT_API struct DocumentPredictResult* document_predict_generate(
    const char* model_path,
    const char* prompt_content,
    int32_t max_tokens,
    int32_t ctx_size,
    int32_t n_threads,
    int32_t n_gpu_layers,
    float temperature,
    int32_t top_k,
    float top_p,
    float min_p,
    int32_t seed
);

/**
 * Free a result structure allocated by document_predict_generate
 * @param result Result to free (can be NULL)
 */
DOCUMENT_PREDICT_API void document_predict_free_result(struct DocumentPredictResult* result);

#ifdef __cplusplus
}
#endif

// C++ API (optional, for easier use from C++)
#ifdef __cplusplus

#include <string>
#include <vector>
#include <optional>

namespace document_predict {

struct Config {
    std::string model_path;
    std::string prompt_content;  // The prompt/content to generate from (already formatted)
    
    int32_t max_tokens = 128;
    int32_t ctx_size = 4096;
    int32_t n_threads = -1;  // -1 for auto
    int32_t n_gpu_layers = 0;
    
    float temperature = 0.8f;
    int32_t top_k = 40;
    float top_p = 0.9f;
    float min_p = 0.05f;
    int32_t seed = -1;  // -1 for random
};

struct Result {
    bool success;
    std::string output;
    std::string error_message;
};

/**
 * Generate document prediction (C++ API)
 * @param config Configuration parameters
 * @return Result with output or error message
 */
DOCUMENT_PREDICT_API Result generate(const Config& config);

} // namespace document_predict

#endif // __cplusplus

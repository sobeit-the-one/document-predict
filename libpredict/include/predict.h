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
 * Progress callback for C API
 * @param current Current number of tokens generated
 * @param total Maximum tokens to generate
 * @param user_data User-provided context pointer
 * @return true to continue generation, false to abort
 */
typedef bool (*DocumentPredictProgressCallback)(int32_t current, int32_t total, void* user_data);

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
 * @param repeat_penalty Repetition penalty (1.0 = disabled, default 1.1)
 * @param penalty_last_n How many tokens to consider for repetition penalty (-1 for context size, default 64)
 * @param soft_max_tokens If true, progressively bias toward EOS as generation approaches max_tokens
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
    int32_t seed,
    float repeat_penalty,
    int32_t penalty_last_n,
    bool soft_max_tokens
);

/**
 * Generate document prediction with progress callback (C API)
 * Same as document_predict_generate but with optional progress reporting.
 * 
 * @param progress_callback Optional callback for progress updates (can be NULL)
 * @param user_data User context passed to progress_callback (can be NULL)
 */
DOCUMENT_PREDICT_API struct DocumentPredictResult* document_predict_generate_with_progress(
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
    int32_t seed,
    float repeat_penalty,
    int32_t penalty_last_n,
    bool soft_max_tokens,
    DocumentPredictProgressCallback progress_callback,
    void* user_data
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
#include <functional>

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
    
    float repeat_penalty = 1.1f;  // 1.0 = disabled
    int32_t penalty_last_n = 64;  // -1 for context size
    
    bool soft_max_tokens = false;  // Progressive EOS bias as generation approaches max_tokens
    bool verbose = false;  // Show llama.cpp logging (GPU info, warnings, etc.)
    
    // Optional progress callback: (current, total) -> bool (return false to abort)
    std::function<bool(int32_t, int32_t)> progress_callback = nullptr;
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

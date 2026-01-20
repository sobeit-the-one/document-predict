#include "predict.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"

#include <memory>
#include <cstring>
#include <sstream>
#include <functional>

// Null log callback to suppress llama.cpp debug output
static void null_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
}

static void read_sampling_defaults_from_model(
    const llama_model* model,
    common_params_sampling& sparams) {
    
    auto get_int32 = [&](const char* key, int32_t& dst) {
        char buf[64] = {0};
        if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) > 0) {
            char* end = nullptr;
            int32_t v = strtol(buf, &end, 10);
            if (end && end != buf) {
                dst = v;
            }
        }
    };
    
    auto get_float = [&](const char* key, float& dst) {
        char buf[128] = {0};
        if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) > 0) {
            char* end = nullptr;
            float v = strtof(buf, &end);
            if (end && end != buf) {
                dst = v;
            }
        }
    };
    
    // Read sampling sequence
    char buf[512] = {0};
    if (llama_model_meta_val_str(model, 
            llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE), 
            buf, sizeof(buf)) > 0) {
        const std::vector<std::string> sampler_names = string_split<std::string>(std::string(buf), ';');
        if (!sampler_names.empty()) {
            sparams.samplers = common_sampler_types_from_names(sampler_names, true);
        }
    }
    
    // Read individual sampling parameters
    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TOP_K), sparams.top_k);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TOP_P), sparams.top_p);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIN_P), sparams.min_p);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TEMP), sparams.temp);
    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N), sparams.penalty_last_n);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT), sparams.penalty_repeat);
    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT), sparams.mirostat);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU), sparams.mirostat_tau);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA), sparams.mirostat_eta);
}

static std::string generate_completion(
    const std::string& model_path,
    const std::string& prompt,
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
    bool verbose,
    const std::function<bool(int32_t, int32_t)>& progress_callback = nullptr) {
    
    // Initialize backend
    llama_backend_init();
    if (!verbose) {
        llama_log_set(null_log_callback, nullptr);
    }
    ggml_backend_load_all();
    
    // Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    
    llama_model* model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == nullptr) {
        llama_backend_free();
        throw std::runtime_error("Failed to load model");
    }
    
    // Initialize sampling parameters
    common_params_sampling sparams;
    sparams.seed = (seed == -1) ? LLAMA_DEFAULT_SEED : static_cast<uint32_t>(seed);
    read_sampling_defaults_from_model(model, sparams);
    
    // Override with provided values
    sparams.temp = temperature;
    sparams.top_k = top_k;
    sparams.top_p = top_p;
    sparams.min_p = min_p;
    sparams.penalty_repeat = repeat_penalty;
    sparams.penalty_last_n = penalty_last_n;
    
    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = ctx_size;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    
    // Batch sizes for efficient prompt processing
    // n_batch: logical batch size (max tokens per llama_decode call)
    // n_ubatch: physical batch size (actual GPU computation batch)
    ctx_params.n_batch = 2048;   // Process up to 2048 tokens at once
    ctx_params.n_ubatch = 512;   // GPU processes 512 tokens per physical batch

    // TODO: Flash attn should eventually be optional
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    
    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        llama_model_free(model);
        llama_backend_free();
        throw std::runtime_error("Failed to create context");
    }
    
    // Create sampler
    common_sampler* sampler = common_sampler_init(model, sparams);
    if (sampler == nullptr) {
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        throw std::runtime_error("Failed to create sampler");
    }
    
    // Tokenize input
    std::vector<llama_token> input_tokens = common_tokenize(ctx, prompt, true, true);
    if (input_tokens.empty()) {
        common_sampler_free(sampler);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        throw std::runtime_error("Failed to tokenize input");
    }
    
    const llama_vocab* vocab = llama_model_get_vocab(model);
    
    // Get EOS token for soft max tokens biasing
    llama_token eos_token = llama_vocab_eos(vocab);
    
    // Soft max tokens parameters
    const float eos_bias_start = 0.5f;   // Start biasing at 50% of max_tokens
    const float eos_bias_max = 10.0f;    // Maximum logit bias at 100%
    
    // Check if prompt fits in context
    const int32_t n_ctx = llama_n_ctx(ctx);
    if (static_cast<int32_t>(input_tokens.size()) > n_ctx - 4) {
        common_sampler_free(sampler);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        throw std::runtime_error("Prompt too long: " + std::to_string(input_tokens.size()) + 
            " tokens exceeds context size " + std::to_string(n_ctx));
    }
    
    // Process prompt in batches (required when prompt > n_batch)
    const int32_t n_batch = ctx_params.n_batch;
    
    if (llama_model_has_encoder(model)) {
        // Encoder-decoder model: encode full prompt
        llama_batch batch = llama_batch_get_one(input_tokens.data(), static_cast<int32_t>(input_tokens.size()));
        if (llama_encode(ctx, batch)) {
            common_sampler_free(sampler);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            throw std::runtime_error("Failed to encode prompt");
        }
        
        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_vocab_bos(vocab);
        }
        llama_batch start_batch = llama_batch_get_one(&decoder_start_token_id, 1);
        if (llama_decode(ctx, start_batch)) {
            common_sampler_free(sampler);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            throw std::runtime_error("Failed to decode start token");
        }
    } else {
        // Decoder-only model: process prompt in chunks
        for (size_t i = 0; i < input_tokens.size(); i += n_batch) {
            const size_t chunk_size = std::min(static_cast<size_t>(n_batch), input_tokens.size() - i);
            llama_batch batch = llama_batch_get_one(input_tokens.data() + i, static_cast<int32_t>(chunk_size));
            
            if (llama_decode(ctx, batch)) {
                common_sampler_free(sampler);
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                throw std::runtime_error("Failed to decode prompt chunk at position " + std::to_string(i));
            }
        }
    }
    
    // Reset perf counters to measure generation only (not prompt processing)
    llama_perf_context_reset(ctx);
    
    // Generate completion
    std::ostringstream output;
    int n_remain = max_tokens;
    int n_generated = 0;
    
    while (n_remain > 0) {
        // Apply progressive EOS bias if soft_max_tokens is enabled
        if (soft_max_tokens && max_tokens > 0 && eos_token != LLAMA_TOKEN_NULL) {
            float progress = static_cast<float>(n_generated) / static_cast<float>(max_tokens);
            if (progress > eos_bias_start) {
                // Quadratic ramp: bias increases faster as we approach max_tokens
                float t = (progress - eos_bias_start) / (1.0f - eos_bias_start);
                float bias = eos_bias_max * t * t;
                
                // Get logits and apply bias to EOS token
                float* logits = llama_get_logits_ith(ctx, -1);
                if (logits) {
                    logits[eos_token] += bias;
                }
            }
        }
        
        llama_token new_token = common_sampler_sample(sampler, ctx, -1);
        
        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }
        
        char buf[128];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        if (n > 0) {
            output.write(buf, n);
        }
        
        common_sampler_accept(sampler, new_token, true);
        
        n_remain--;
        n_generated++;
        
        // Invoke progress callback if provided
        if (progress_callback && !progress_callback(n_generated, max_tokens)) {
            break;  // Callback returned false, abort generation
        }
        
        // Decode the new token to produce logits for next iteration
        llama_batch next_batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(ctx, next_batch)) {
            break;
        }
    }
    
    std::string result = output.str();
    
    // Print performance timings if verbose
    if (verbose) {
        llama_perf_context_print(ctx);
    }
    
    // Cleanup
    common_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    
    return result;
}

// C API implementation

extern "C" {

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
    bool soft_max_tokens) {
    
    auto* result = new DocumentPredictResult{};
    result->success = false;
    result->output = nullptr;
    result->error_message = nullptr;
    
    try {
        // Validate required parameters
        if (!model_path || !prompt_content) {
            result->error_message = _strdup("Invalid configuration: missing required parameters");
            return result;
        }
        
        // Generate completion directly from prompt content
        std::string output = generate_completion(
            model_path,
            prompt_content,
            max_tokens,
            ctx_size,
            n_threads,
            n_gpu_layers,
            temperature,
            top_k,
            top_p,
            min_p,
            seed,
            repeat_penalty,
            penalty_last_n,
            soft_max_tokens,
            false  // verbose - not exposed in simple C API
        );
        
        result->success = true;
        result->output = _strdup(output.c_str());
        return result;
        
    } catch (const std::exception& e) {
        result->error_message = _strdup(e.what());
        return result;
    } catch (...) {
        result->error_message = _strdup("Unknown error occurred");
        return result;
    }
}

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
    void* user_data) {
    
    auto* result = new DocumentPredictResult{};
    result->success = false;
    result->output = nullptr;
    result->error_message = nullptr;
    
    try {
        // Validate required parameters
        if (!model_path || !prompt_content) {
            result->error_message = _strdup("Invalid configuration: missing required parameters");
            return result;
        }
        
        // Wrap C callback in std::function if provided
        std::function<bool(int32_t, int32_t)> callback_wrapper = nullptr;
        if (progress_callback) {
            callback_wrapper = [progress_callback, user_data](int32_t current, int32_t total) {
                return progress_callback(current, total, user_data);
            };
        }
        
        // Generate completion directly from prompt content
        std::string output = generate_completion(
            model_path,
            prompt_content,
            max_tokens,
            ctx_size,
            n_threads,
            n_gpu_layers,
            temperature,
            top_k,
            top_p,
            min_p,
            seed,
            repeat_penalty,
            penalty_last_n,
            soft_max_tokens,
            false,  // verbose - not exposed in C API with progress
            callback_wrapper
        );
        
        result->success = true;
        result->output = _strdup(output.c_str());
        return result;
        
    } catch (const std::exception& e) {
        result->error_message = _strdup(e.what());
        return result;
    } catch (...) {
        result->error_message = _strdup("Unknown error occurred");
        return result;
    }
}

DOCUMENT_PREDICT_API void document_predict_free_result(struct DocumentPredictResult* result) {
    if (result) {
        if (result->output) {
            free(result->output);
        }
        if (result->error_message) {
            free(result->error_message);
        }
        delete result;
    }
}

} // extern "C"

// C++ API implementation

#ifdef __cplusplus

namespace document_predict {

Result generate(const Config& config) {
    Result result{};
    
    try {
        // Validate required parameters
        if (config.model_path.empty() || config.prompt_content.empty()) {
            result.success = false;
            result.error_message = "Invalid configuration: missing required parameters";
            return result;
        }
        
        // Generate completion directly from prompt content
        result.output = generate_completion(
            config.model_path,
            config.prompt_content,
            config.max_tokens,
            config.ctx_size,
            config.n_threads,
            config.n_gpu_layers,
            config.temperature,
            config.top_k,
            config.top_p,
            config.min_p,
            config.seed,
            config.repeat_penalty,
            config.penalty_last_n,
            config.soft_max_tokens,
            config.verbose,
            config.progress_callback
        );
        
        result.success = true;
        return result;
        
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
        return result;
    } catch (...) {
        result.success = false;
        result.error_message = "Unknown error occurred";
        return result;
    }
}

} // namespace document_predict

#endif // __cplusplus

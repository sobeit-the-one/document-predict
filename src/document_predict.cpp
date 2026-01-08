#include "document_predict.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"

#include <memory>
#include <cstring>

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
    int32_t seed) {
    
    // Initialize backend
    llama_backend_init();
    llama_log_set(null_log_callback, nullptr);
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
    
    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = ctx_size;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    
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
    
    // Prepare batch for prompt
    llama_batch batch = llama_batch_get_one(input_tokens.data(), static_cast<int32_t>(input_tokens.size()));
    
    // Encode/decode prompt
    if (llama_model_has_encoder(model)) {
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
        batch = llama_batch_get_one(&decoder_start_token_id, 1);
    } else {
        if (llama_decode(ctx, batch)) {
            common_sampler_free(sampler);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            throw std::runtime_error("Failed to decode prompt");
        }
    }
    
    // Generate completion
    std::ostringstream output;
    int n_remain = max_tokens;
    
    while (n_remain > 0) {
        if (llama_decode(ctx, batch)) {
            break;
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
        batch = llama_batch_get_one(&new_token, 1);
        
        n_remain--;
    }
    
    std::string result = output.str();
    
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
    int32_t seed) {
    
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
            seed
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
            config.seed
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

#include "argparse/argparse.hpp"
#include "predict.h"
#include "prompt_file.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

// Simple tqdm-style progress bar for manual updates
// (tqdm.cpp is iterator-based, so we create a simple wrapper for callback use)
class ProgressBar {
public:
    ProgressBar(size_t total, const std::string& desc = "", const std::string& unit = "tok")
        : total_(total), current_(0), desc_(desc), unit_(unit), 
          start_time_(std::chrono::steady_clock::now()) {}
    
    void update(size_t current) {
        current_ = current;
        render();
    }
    
    void finish() {
        current_ = total_;
        render();
        fprintf(stderr, "\n");
    }
    
    void clear() {
        fprintf(stderr, "\r%*s\r", 80, "");
        fflush(stderr);
    }

private:
    void render() {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double>(now - start_time_).count();
        
        float progress = total_ > 0 ? static_cast<float>(current_) / total_ : 0.0f;
        int percent = static_cast<int>(progress * 100);
        
        // Calculate speed and ETA
        double speed = elapsed > 0 ? current_ / elapsed : 0;
        double eta = speed > 0 ? (total_ - current_) / speed : 0;
        
        // Build the progress bar (20 chars wide)
        const int bar_width = 20;
        int filled = static_cast<int>(progress * bar_width);
        
        std::string bar;
        for (int i = 0; i < bar_width; ++i) {
            if (i < filled) bar += '#';
            else bar += ' ';
        }
        
        // Format: desc: 100%|####################| 128/128 [00:05<00:00, 25.6tok/s]
        fprintf(stderr, "\r%s%3d%%|%s| %zu/%zu [%02d:%02d<%02d:%02d, %.1f%s/s]",
            desc_.empty() ? "" : (desc_ + ": ").c_str(),
            percent,
            bar.c_str(),
            current_, total_,
            static_cast<int>(elapsed) / 60, static_cast<int>(elapsed) % 60,
            static_cast<int>(eta) / 60, static_cast<int>(eta) % 60,
            speed, unit_.c_str());
        fflush(stderr);
    }
    
    size_t total_;
    size_t current_;
    std::string desc_;
    std::string unit_;
    std::chrono::steady_clock::time_point start_time_;
};

int main(int argc, char** argv) {
    argparse::ArgumentParser program("document-predict");
    
    // Required arguments
    program.add_argument("-m", "--model")
        .required()
        .help("Path to GGUF model file");
    
    program.add_argument("-f", "--file")
        .required()
        .help("Path to input file (plain text or Jinja2 template if --jinja is used)");
    
    program.add_argument("-o", "--output")
        .help("Path to output file (if not specified, output to console)");
    
    program.add_argument("--jinja")
        .flag()
        .help("Enable Jinja2 template processing (treat input file as template)");
    
    program.add_argument("--var")
        .append()
        .help("Template variable in format key=value (can be used multiple times, only used with --jinja)");

    // Generation parameters
    program.add_argument("-n", "--max-tokens")
        .default_value(128)
        .scan<'i', int>()
        .help("Maximum tokens to generate (default: 128)");
    
    program.add_argument("-c", "--ctx-size")
        .default_value(4096)
        .scan<'i', int>()
        .help("Context window size (default: 4096)");
    
    program.add_argument("-t", "--threads")
        .default_value(-1)
        .scan<'i', int>()
        .help("Number of CPU threads (default: auto-detect)");
    
    program.add_argument("-ngl", "--n-gpu-layers")
        .default_value(0)
        .scan<'i', int>()
        .help("Number of layers to offload to GPU (default: 0 for CPU-only, -1 for auto)");
    
    // Sampling parameters (will be overridden by GGUF defaults if available)
    program.add_argument("--temperature")
        .default_value(0.8f)
        .scan<'g', float>()
        .help("Sampling temperature (default: from GGUF or 0.8)");
    
    program.add_argument("--top-k")
        .default_value(40)
        .scan<'i', int>()
        .help("Top-k sampling (default: from GGUF or 40)");
    
    program.add_argument("--top-p")
        .default_value(0.9f)
        .scan<'g', float>()
        .help("Top-p sampling (default: from GGUF or 0.9)");
    
    program.add_argument("--min-p")
        .default_value(0.05f)
        .scan<'g', float>()
        .help("Min-p sampling (default: from GGUF or 0.05)");
    
    program.add_argument("-s", "--seed")
        .default_value(-1)
        .scan<'i', int>()
        .help("RNG seed (default: -1 for random)");
    
    program.add_argument("--repeat-penalty")
        .default_value(1.1f)
        .scan<'g', float>()
        .help("Repetition penalty (1.0 = disabled, default: 1.1)");
    
    program.add_argument("--repeat-last-n")
        .default_value(64)
        .scan<'i', int>()
        .help("Number of tokens to consider for repetition penalty (-1 = context size, default: 64)");
    
    program.add_argument("--soft-max-tokens")
        .flag()
        .help("Progressively bias toward EOS as generation approaches max-tokens (helps avoid mid-sentence cutoffs)");
    
    program.add_argument("--progress")
        .flag()
        .help("Show progress bar during token generation");
    
    program.add_argument("-v", "--verbose")
        .flag()
        .help("Show llama.cpp logging (GPU detection, backend info, warnings)");
    
    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }
    
    // Get arguments
    std::string model_path = program.get<std::string>("--model");
    std::string file_path = program.get<std::string>("--file");
    bool use_jinja = program.get<bool>("--jinja");
    
    // Read input file
    std::ifstream file_stream(file_path);
    if (!file_stream.is_open()) {
        std::cerr << "Error: Failed to read input file: " << file_path << std::endl;
        return 1;
    }
    std::ostringstream buffer;
    buffer << file_stream.rdbuf();
    std::string file_content = buffer.str();
    file_stream.close();
    
    // Process with Jinja2 if enabled
    std::string final_prompt;
    if (use_jinja) {
        // Treat input file as Jinja2 template and render it
        std::vector<PromptVariable> variables;
        variables.push_back({"file_path", file_path});
        
        // Parse template variables from command line (--var key=value)
        if (program.is_used("--var")) {
            auto var_args = program.get<std::vector<std::string>>("--var");
            for (const auto& var_arg : var_args) {
                size_t eq_pos = var_arg.find('=');
                if (eq_pos == std::string::npos || eq_pos == 0) {
                    std::cerr << "Error: Invalid template variable format: " << var_arg << std::endl;
                    std::cerr << "Expected format: key=value" << std::endl;
                    return 1;
                }
                std::string key = var_arg.substr(0, eq_pos);
                std::string value = var_arg.substr(eq_pos + 1);
                variables.push_back({key, value});
            }
        }
        
        PromptFile prompt_file(file_path);
        auto prompt_result = prompt_file.load_prompt(variables);
        if (!prompt_result) {
            std::cerr << "Error: Failed to load or render template" << std::endl;
            return 1;
        }
        final_prompt = prompt_result.value();
    } else {
        // Use file content directly (no template processing)
        final_prompt = file_content;
    }
    
    int max_tokens = program.get<int>("--max-tokens");
    int ctx_size = program.get<int>("--ctx-size");
    int n_threads = program.get<int>("--threads");
    int n_gpu_layers = program.get<int>("--n-gpu-layers");
    int seed = program.get<int>("--seed");
    
    float temperature = program.get<float>("--temperature");
    int top_k = program.get<int>("--top-k");
    float top_p = program.get<float>("--top-p");
    float min_p = program.get<float>("--min-p");
    float repeat_penalty = program.get<float>("--repeat-penalty");
    int repeat_last_n = program.get<int>("--repeat-last-n");
    bool soft_max_tokens = program.get<bool>("--soft-max-tokens");
    bool show_progress = program.get<bool>("--progress");
    bool verbose = program.get<bool>("--verbose");
    
    // Use C++ API - DLL only handles LLM inference
    document_predict::Config config;
    config.model_path = model_path;
    config.prompt_content = final_prompt;
    config.max_tokens = max_tokens;
    config.ctx_size = ctx_size;
    config.n_threads = n_threads;
    config.n_gpu_layers = n_gpu_layers;
    config.seed = seed;
    config.temperature = temperature;
    config.top_k = top_k;
    config.top_p = top_p;
    config.min_p = min_p;
    config.repeat_penalty = repeat_penalty;
    config.penalty_last_n = repeat_last_n;
    config.soft_max_tokens = soft_max_tokens;
    config.verbose = verbose;
    
    // Set up progress bar if enabled
    std::unique_ptr<ProgressBar> progress_bar;
    if (show_progress) {
        progress_bar = std::make_unique<ProgressBar>(max_tokens, "Generating", "tok");
        config.progress_callback = [&progress_bar](int32_t current, int32_t total) {
            progress_bar->update(current);
            return true;  // Continue generation
        };
    }
    
    // Generate prediction using library
    auto result = document_predict::generate(config);
    
    // Finish and clear progress bar if it was shown
    if (progress_bar) {
        progress_bar->clear();
    }
    
    if (!result.success) {
        std::cerr << "Error: " << result.error_message << std::endl;
        return 1;
    }
    
    // Output input file content + prediction to file or console
    bool has_output = program.is_used("--output");
    if (has_output) {
        std::string output_path = program.get<std::string>("--output");
        std::ofstream output_file(output_path);
        if (!output_file.is_open()) {
            std::cerr << "Error: Failed to open output file: " << output_path << std::endl;
            return 1;
        }
        // Output input content first, then prediction
        output_file << final_prompt << result.output;
        output_file.close();
    } else {
        // Output to console
        std::cout << final_prompt << result.output << std::endl;
    }
    
    return 0;
}

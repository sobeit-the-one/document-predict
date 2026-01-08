#include "argparse/argparse.hpp"
#include "document_predict.h"
#include "prompt_file.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

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
        // Note: file_content would be the template source itself, which is usually not useful
        // Users can add custom variables via template if needed
        
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
    
    // Generate prediction using library
    auto result = document_predict::generate(config);
    
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

# document-predict-cpp

A command-line tool and library for **document prediction** using Large Language Models (LLMs). This project demonstrates that LLMs are fundamentally **document predictors**, not "generative AI" in the traditional sense.

## Philosophy: LLMs as Document Predictors

### The Core Insight

Large Language Models are not "generating" new content from scratch. They are **predicting the most likely continuation** of a document based on patterns learned from training data. This is fundamentally different from true generation:

- **True Generation**: Creating something entirely new, independent of context
- **Document Prediction**: Predicting what comes next in a sequence, based on statistical patterns

### Why This Matters

1. **LLMs predict tokens, not ideas**: Every token is chosen based on probability distributions over the vocabulary, conditioned on the preceding context.

2. **Continuation, not creation**: When you give an LLM a prompt, it's not "creating" a response—it's predicting what text would most likely follow that prompt in a document.

3. **Statistical pattern matching**: The model has learned statistical patterns from training data. It predicts continuations that match these patterns.

4. **Context-dependent**: The "generation" is entirely dependent on the input context. Change the prompt, and you get a different prediction.

### Demonstration

This tool is designed to make this concept explicit:

- **Input**: A document (or partial document)
- **Process**: The model predicts the most likely continuation
- **Output**: The original document + the predicted continuation

By showing the input alongside the prediction, we make it clear that the model is **completing** a document, not generating something from nothing.

## Features

- **Pure C++ implementation** using `llama.cpp` for efficient inference
- **GPU acceleration support** via layer offloading
- **Optional Jinja2 templating** for flexible prompt formatting
- **DLL/Shared library** for integration into other applications
- **Rust FFI compatible** C API
- **Automatic parameter detection** from GGUF model metadata

## Building

### Prerequisites

- CMake 3.23 or higher
- C++20 compatible compiler
- Git (for submodules)
- **For GPU acceleration**: CUDA Toolkit (NVIDIA) or Vulkan SDK

### Build Steps

```bash
# Clone with submodules
git clone --recursive <repository-url>
cd document-predict-cpp

# Configure
cmake -B build -S .

# Build (IMPORTANT: Use Release mode for performance!)
cmake --build build --config Release

# Outputs will be in build/bin/
```

> **Important**: Always build with `--config Release`. Debug builds are 5-10x slower due to disabled optimizations and runtime checks.

### GPU Support

GPU acceleration is enabled by default for NVIDIA GPUs (CUDA). To use a different backend:

```bash
# For Vulkan (cross-platform: NVIDIA, AMD, Intel)
cmake -B build -S . -DENABLE_CUDA=OFF -DENABLE_VULKAN=ON

# For CPU only
cmake -B build -S . -DENABLE_CUDA=OFF
```

### Submodules

This project uses the following submodules:

- `llama.cpp` - Core LLM inference engine
- `argparse` - Command-line argument parsing
- `minja` - Jinja2-like template engine (for CLI only)

## Usage

### Basic Usage

```bash
# Simple document prediction (plain text input, GPU accelerated)
document-predict.exe -m model.gguf -f input.txt -ngl -1

# With output file and progress display
document-predict.exe -m model.gguf -f input.txt -o output.txt -ngl -1 --progress

# Longer generation with soft max tokens (smoother endings)
document-predict.exe -m model.gguf -f input.txt -o output.txt -ngl -1 -n 512 --soft-max-tokens

# Using Jinja2 template
document-predict.exe -m model.gguf -f template.j2 --jinja -o result.txt -ngl -1
```

### Command-Line Arguments

**Required:**

- `-m, --model`: Path to GGUF model file
- `-f, --file`: Path to input file (plain text or Jinja2 template)

**Optional:**

- `-o, --output`: Path to output file (if not specified, outputs to console)
- `--jinja`: Enable Jinja2 template processing
- `--var key=value`: Template variable (can be used multiple times, only used with `--jinja`)
- `--progress`: Show generation progress
- `--verbose`: Enable verbose output (model loading info, performance stats)

**Generation Parameters:**

- `-n, --max-tokens`: Maximum tokens to generate (default: 128)
- `-c, --ctx-size`: Context window size (default: 4096)
- `-t, --threads`: Number of CPU threads (default: auto-detect)
- `-ngl, --n-gpu-layers`: GPU layers to offload (default: 0, use -1 for all)
- `--temperature`: Sampling temperature (default: from GGUF or 0.8)
- `--top-k`: Top-k sampling (default: from GGUF or 40)
- `--top-p`: Top-p sampling (default: from GGUF or 0.9)
- `--min-p`: Min-p sampling (default: from GGUF or 0.05)
- `-s, --seed`: RNG seed (default: -1 for random)
- `--repeat-penalty`: Repetition penalty (default: from GGUF or 1.0)
- `--penalty-last-n`: Tokens to consider for penalty (default: from GGUF or 64)
- `--soft-max-tokens`: Gradually bias toward EOS as max tokens approaches (smoother endings)

### Examples

#### Example 1: Simple Document Completion

**input.txt:**

```text
The history of artificial intelligence began in the 1950s when researchers first
```

**Command:**

```bash
document-predict.exe -m llama-3.1-8b.gguf -f input.txt -o output.txt -n 256 -ngl -1
```

**output.txt:**

```text
The history of artificial intelligence began in the 1950s when researchers first
explored the possibility of creating machines that could think and learn. The field
has since evolved through multiple phases, from early symbolic systems to modern
neural networks and deep learning architectures...
```

Notice how the model **predicted** what would most likely follow the partial sentence, completing the document.

#### Example 2: Using Templates

**template.j2:**

```text
You are a helpful assistant.

User: {{ user_query }}

Assistant:
```

**Command:**

```bash
document-predict.exe -m model.gguf -f template.j2 --jinja --var user_query="What is AI?" -o response.txt
```

The template is rendered with the provided variables (`file_path` is always available), then the model predicts what an assistant would say next.

#### Example 3: Multiple Template Variables

**template.j2:**

```text
{{ greeting }}, {{ name }}!

Today is {{ date }}. Here's your personalized message:

```

**Command:**

```bash
document-predict.exe -m model.gguf -f template.j2 --jinja \
  --var greeting="Hello" \
  --var name="Alice" \
  --var date="January 15, 2024" \
  -o output.txt
```

Multiple variables can be passed using multiple `--var` arguments.

## Library API

### C API (Rust FFI Compatible)

```c
// Basic generation
struct DocumentPredictResult* document_predict_generate(
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

// Generation with progress callback
typedef bool (*DocumentPredictProgressCallback)(int32_t current, int32_t total, void* user_data);

struct DocumentPredictResult* document_predict_generate_with_progress(
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

void document_predict_free_result(struct DocumentPredictResult* result);
```

### C++ API

```cpp
namespace document_predict {
    struct Config {
        std::string model_path;
        std::string prompt_content;
        int32_t max_tokens = 128;
        int32_t ctx_size = 4096;
        int32_t n_threads = 0;  // 0 = auto-detect
        int32_t n_gpu_layers = 0;
        float temperature = 0.8f;
        int32_t top_k = 40;
        float top_p = 0.9f;
        float min_p = 0.05f;
        int32_t seed = -1;
        float repeat_penalty = 1.0f;
        int32_t penalty_last_n = 64;
        bool soft_max_tokens = false;
        bool verbose = false;
        std::function<bool(int32_t, int32_t)> progress_callback = nullptr;
    };
    
    struct Result {
        bool success;
        std::string output;
        std::string error_message;
    };
    
    Result generate(const Config& config);
}
```

## Architecture

### Components

1. **CLI (`document-predict.exe`)**: Command-line interface that handles:
   - File I/O
   - Jinja2 template rendering (if enabled)
   - Progress display
   - Calls the library for inference

2. **Library (`libpredict.dll`)**: Core inference engine that:
   - Loads GGUF models via llama.cpp
   - Manages context, batching, and sampling
   - Handles GPU offloading and Flash Attention
   - Performs token prediction with configurable parameters
   - Supports progress callbacks for long-running generations

### Design Philosophy

The library is **pure inference**—it takes a prompt string and returns a prediction. Template processing, file handling, and formatting are handled by the CLI or calling application. This separation makes the library:

- **Language-agnostic**: C API works from any language
- **Simple**: Single responsibility (inference)
- **Reusable**: Can be integrated into any application

## Technical Details

### Model Support

- Any GGUF format model compatible with `llama.cpp`
- Automatic detection of default sampling parameters from GGUF metadata
- Support for encoder-decoder models (via `llama.cpp`)

### Sampling

The tool uses `llama.cpp`'s common sampling utilities, which support:

- Top-k sampling
- Top-p (nucleus) sampling
- Min-p sampling
- Temperature-based sampling
- Mirostat sampling (if configured in model)

Default parameters are read from GGUF metadata when available, with command-line arguments as overrides.

### GPU Acceleration

GPU acceleration is supported via layer offloading:

- Set `-ngl` / `--n-gpu-layers` to offload layers to GPU
- Use `-1` to offload all layers to GPU (recommended)
- Supports CUDA (NVIDIA), Vulkan (cross-platform), Metal (macOS), and other backends via `llama.cpp`

**Performance features:**

- Flash Attention enabled by default for faster inference
- Configurable batch sizes for efficient prompt processing
- Native CPU optimizations when GPU not available

## Why "Document Prediction" Not "Generation"

### Statistical Nature

LLMs work by:

1. Tokenizing input into a sequence
2. Computing probability distributions over the vocabulary for each position
3. Sampling from these distributions to predict the next token
4. Repeating until a stopping condition

This is **prediction**, not generation. The model is predicting: "Given this context, what token is most likely to come next?"

### Training Process

During training, LLMs learn to predict the next token in sequences from their training data. They're not learning to "create" content—they're learning to **predict** what text would appear next in documents similar to their training corpus.

### Practical Implications

Understanding LLMs as predictors helps explain:

- **Hallucinations**: The model predicts text that matches training patterns, even if factually incorrect
- **Context sensitivity**: Different prompts lead to different predictions
- **Reproducibility**: Same prompt + seed = same prediction
- **Limitations**: The model can only predict patterns it has seen during training

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

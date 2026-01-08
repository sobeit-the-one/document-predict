#include "prompt_file.hpp"
#include <minja/minja.hpp>
#include <fstream>
#include <sstream>
#include <stdexcept>

using json = nlohmann::ordered_json;

PromptFile::PromptFile(const std::filesystem::path& path)
    : path_(path) {
}

std::optional<std::string> PromptFile::load_prompt(const std::vector<PromptVariable>& variables) {
    // Read the template file
    std::ifstream file(path_);
    if (!file.is_open()) {
        return std::nullopt;
    }
    
    std::ostringstream buffer;
    buffer << file.rdbuf();
    std::string template_content = buffer.str();
    file.close();
    
    // Parse the template
    minja::Options options;
    auto template_node = minja::Parser::parse(template_content, options);
    
    // Build context from variables
    json context_json = json::object();
    for (const auto& var : variables) {
        context_json[var.name] = var.value;
    }
    
    auto context = minja::Context::make(minja::Value(context_json));
    
    // Render the template
    try {
        std::string result = template_node->render(context);
        return result;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}
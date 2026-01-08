#pragma once

#include <filesystem>
#include <optional>

struct PromptVariable {
    std::string name;
    std::string value;
};

class PromptFile {
public:
    PromptFile(const std::filesystem::path& path);

    std::optional<std::string> load_prompt(const std::vector<PromptVariable>& variables);

private:
    std::filesystem::path path_;
};
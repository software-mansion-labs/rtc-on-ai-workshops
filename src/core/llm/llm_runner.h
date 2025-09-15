#pragma once

#include "../tokenization/tokenizer_adapter.h"
#include "chat_template.h"
#include "text_prefiller.h"
#include "token_generator.h"
#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/error.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace rtc_runner {

class LlmRunner {
public:
  explicit LlmRunner(const std::string &model_path,
                     const std::string &tokenizer_path);

  bool is_loaded() const;
  executorch::runtime::Error load();
  /**
   * Generate a complete text response from the given prompt.
   * Orchestrates the full pipeline: tokenization, prefill, and autoregressive
   * generation.
   * @param prompt The input text prompt to generate a response for
   * @return Error code indicating success or failure of the generation process
   */
  executorch::runtime::Error generate(const std::string &prompt);

  // Chat-specific methods
  executorch::runtime::Error chat(const std::string &user_message,
                                  const std::string &system_prompt = "");
  executorch::runtime::Error
  chat_with_history(const std::vector<ChatMessage> &messages);
  static constexpr int32_t MAX_CONTEXT_LEN = 2048;
  static constexpr int32_t VOCAB_SIZE = 128256;

private:
  float temperature_;
  std::string tokenizer_path_;

  // Core components
  std::unique_ptr<executorch::extension::Module> module_;
  std::unique_ptr<TokenizerAdapter> tokenizer_;
  std::unique_ptr<TextPrefiller> text_prefiller_;
  std::unique_ptr<TokenGenerator> text_token_generator_;
};

} // namespace rtc_runner
#pragma once

#include "../tokenization/tokenizer_adapter.h"
#include "chat_template.h"
#include "simple_token_generator.h"
#include "text_prefiller.h"
#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/error.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace rtc_runner {

class LlamaRunner {
public:
  explicit LlamaRunner(const std::string &model_path,
                       const std::string &tokenizer_path);

  bool is_loaded() const;
  executorch::runtime::Error load();
  executorch::runtime::Error generate(const std::string &prompt);
  void stop();

  // Chat-specific methods
  executorch::runtime::Error chat(const std::string &user_message,
                                  const std::string &system_prompt = "");
  executorch::runtime::Error
  chat_with_history(const std::vector<ChatMessage> &messages);
  static constexpr int32_t MAX_CONTEXT_LEN = 2048;
  static constexpr int32_t VOCAB_SIZE = 128256;

private:
  float temperature_;
  bool should_stop_ = false;
  std::string tokenizer_path_;

  // Core components
  std::unique_ptr<executorch::extension::Module> module_;
  std::unique_ptr<TokenizerAdapter> tokenizer_;
  std::unique_ptr<TextPrefiller> text_prefiller_;
  std::unique_ptr<SimpleTokenGenerator> text_token_generator_;
};

} // namespace rtc_runner
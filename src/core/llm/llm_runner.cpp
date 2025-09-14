#include "llama_runner.h"
#include <iostream>
#include <unordered_set>

namespace rtc_runner {

LlmRunner::LlmRunner(const std::string &model_path,
                     const std::string &tokenizer_path)
    : tokenizer_path_(tokenizer_path) {
  module_ = std::make_unique<executorch::extension::Module>(
      model_path,
      executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
}

bool LlmRunner::is_loaded() const {
  return module_->is_loaded() && tokenizer_ && text_prefiller_ &&
         text_token_generator_;
}

executorch::runtime::Error LlmRunner::load() {
  if (is_loaded()) {
    return executorch::runtime::Error::Ok;
  }

  auto load_result = module_->load_method("forward");
  if (load_result != executorch::runtime::Error::Ok) {
    std::cout << "❌ Failed to load forward method" << std::endl;
    return load_result;
  }

  // Load tokenizer
  try {
    tokenizer_ = std::make_unique<TokenizerAdapter>(tokenizer_path_);
  } catch (const std::exception &e) {
    std::cout << "❌ Failed to load tokenizer: " << e.what() << std::endl;
    return executorch::runtime::Error::InvalidArgument;
  }

  auto eos_ids = std::make_unique<std::unordered_set<uint64_t>>(
      std::unordered_set<uint64_t>{tokenizer_->eos_tok()});

  // Initialize components
  text_prefiller_ = std::make_unique<TextPrefiller>(module_.get());

  text_token_generator_ = std::make_unique<SimpleTokenGenerator>(
      tokenizer_.get(), module_.get(), std::move(eos_ids));

  return executorch::runtime::Error::Ok;
}

executorch::runtime::Error LlmRunner::generate(const std::string &prompt) {
  /*
  TODO: Implement the complete text generation pipeline

  This function orchestrates the complete LLM inference process including
  prompt processing, prefill, and autoregressive generation.

  Steps to implement:
  1. Validate the prompt is not empty
  2. Ensure the model is loaded (call load() if needed)
  3. Encode the prompt using tokenizer_->encode()
  4. Validate the prompt length against MAX_CONTEXT_LEN
  5. Prefill phase:
     - Initialize position to 0
     - Call text_prefiller_->prefill() with prompt_tokens and position
     - Handle errors and extract the first generated token
     - Decode and print the first token
  6. Generation phase:
     - Add the first token to prompt_tokens
     - Create a token callback lambda for console output
     - Call text_token_generator_->generate() with updated tokens
     - Handle errors and print results
  7. Print completion message if max context length reached

  Key details:
  - Use tokenizer_->encode() to convert string to tokens
  - Check num_prompt_tokens >= MAX_CONTEXT_LEN for length validation
  - Use prefill_res.ok() to check prefill success
  - Use prefill_res.get() to extract the token
  - Use tokenizer_->decode() to convert token to string
  - Print with std::flush for immediate output
  - Pass num_prompt_tokens as start_pos to generate()
  - Lambda callback: [](const std::string &piece) { std::cout << piece <<
  std::flush; }

  Error handling:
  - Return InvalidArgument for empty prompt or too long prompt
  - Return the actual error from load(), prefill(), or generate()
  */

  // TODO: Your implementation here
  return executorch::runtime::Error::NotImplemented;
}

void LlmRunner::stop() {
  if (is_loaded()) {
    text_token_generator_->stop();
  }
  should_stop_ = true;
}

// Chat-specific methods
executorch::runtime::Error LlmRunner::chat(const std::string &user_message,
                                           const std::string &system_prompt) {
  std::string formatted_prompt =
      ChatTemplate::format_user_prompt(user_message, system_prompt);
  return generate(formatted_prompt);
}

executorch::runtime::Error
LlmRunner::chat_with_history(const std::vector<ChatMessage> &messages) {
  std::string formatted_prompt = ChatTemplate::format_conversation(messages);
  return generate(formatted_prompt);
}

} // namespace rtc_runner
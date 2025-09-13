#pragma once

#include <executorch/extension/module/module.h>

namespace rtc_runner {

class TextPrefiller {
public:
  TextPrefiller(executorch::extension::Module *module);
  /**
   * Prefill an LLM Module with the given text input.
   * @param prompt_tokens The text prompt tokens to the LLM Module. Encoded by
   * tokenizer.
   * @param start_pos The starting position in KV cache of the input in the LLM
   * Module.
   * @return The next token of the LLM Module after prefill.
   */
  ::executorch::runtime::Result<uint64_t>
  prefill(std::vector<uint64_t> &prompt_tokens, int64_t &start_pos);

private:
  executorch::extension::Module *module_;
};

} // namespace rtc_runner

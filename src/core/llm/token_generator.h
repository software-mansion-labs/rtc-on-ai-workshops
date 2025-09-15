#pragma once

#include "../tokenization/tokenizer_adapter.h"
#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/result.h>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace rtc_runner {

class TokenGenerator {
public:
  TokenGenerator(TokenizerAdapter *tokenizer,
                 executorch::extension::Module *module,
                 std::unique_ptr<std::unordered_set<uint64_t>> &&eos_ids);

  /**
   * Generate tokens autoregressively from the given input tokens.
   * @param tokens The input token sequence including prompt and any previously
   * generated tokens
   * @param start_pos The starting position in KV cache for generation (after
   * prefill)
   * @param token_callback Callback function called for each generated token's
   * decoded text
   * @return The number of tokens generated, or error if generation fails
   */
  ::executorch::runtime::Result<int64_t>
  generate(std::vector<uint64_t> tokens, int64_t start_pos,
           std::function<void(const std::string &)> token_callback);

private:
  TokenizerAdapter *tokenizer_;
  executorch::extension::Module *module_;
  std::unique_ptr<std::unordered_set<uint64_t>> eos_ids_;
};

} // namespace rtc_runner
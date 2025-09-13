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

class SimpleTokenGenerator {
public:
  SimpleTokenGenerator(TokenizerAdapter *tokenizer,
                       executorch::extension::Module *module,
                       std::unique_ptr<std::unordered_set<uint64_t>> &&eos_ids);

  ::executorch::runtime::Result<int64_t>
  generate(std::vector<uint64_t> tokens, int64_t start_pos,
           std::function<void(const std::string &)> token_callback);

  void stop();

private:
  TokenizerAdapter *tokenizer_;
  executorch::extension::Module *module_;
  std::unique_ptr<std::unordered_set<uint64_t>> eos_ids_;
  bool should_stop_ = false;
};

} // namespace rtc_runner
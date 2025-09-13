#include "token_generator.h"
#include "../../utils/argmax_utils.h"
#include "./llm_runner.h"
#include <executorch/extension/tensor/tensor.h>

namespace rtc_runner {

TokenGenerator::TokenGenerator(
    TokenizerAdapter *tokenizer, executorch::extension::Module *module,
    std::unique_ptr<std::unordered_set<uint64_t>> &&eos_ids)
    : tokenizer_(tokenizer), module_(module), eos_ids_(std::move(eos_ids)) {}

::executorch::runtime::Result<int64_t> TokenGenerator::generate(
    std::vector<uint64_t> tokens, int64_t start_pos,
    std::function<void(const std::string &)> token_callback) {

  /*
  TODO: Implement the token generation function

  This function should generate tokens one by one in a loop until EOS or max
  length.

  Steps to implement:
  1. Validate input (check if tokens is empty)
  2. Initialize position and current token from the last token in input
  3. Create tensor data and shape for single token processing
  4. Create managed tensors for model input
  5. Loop until max context length:
     a. Call module_->forward() with current token and position
     b. Extract logits from the result
     c. Get next token using logits_to_token()
     d. Increment position
     e. Update token_data with new token
     f. Decode token to string and call token_callback
     g. Check if token is EOS and break if so
  6. Return the number of tokens generated

  Key details:
  - Use tokens.back() to get the starting token
  - Token tensor shape should be {1, 1} for single token
  - Position tensor shape should be {1}
  - Use ScalarType::Long for both tensors
  - Check eos_ids_->find(cur_token) != eos_ids_->end() for EOS detection
  - Return pos - start_pos as the number of generated tokens

  Constants:
  - LlmRunner::MAX_CONTEXT_LEN - 1 as the maximum position
  */

  // TODO: Your implementation here
  int64_t pos = start_pos;
  uint64_t cur_token = tokens.back();

  std::vector<uint64_t> token_data = {cur_token};
  std::vector<executorch::aten::SizesType> token_shape = {1, 1};

  auto tokens_managed = executorch::extension::from_blob(
      token_data.data(), token_shape, executorch::aten::ScalarType::Long);
  auto start_pos_managed = executorch::extension::from_blob(
      &pos, {1}, executorch::aten::ScalarType::Long);

  while (pos < LlmRunner::MAX_CONTEXT_LEN - 1) {
    auto logits_res = module_->forward({tokens_managed, start_pos_managed});
    executorch::aten::Tensor logits_tensor = logits_res.get()[0].toTensor();

    // Greedy sampling
    cur_token = logits_to_token(logits_tensor);

    pos++;
    token_data[0] = cur_token;

    // Decode and callback
    std::string piece = tokenizer_->decode(cur_token);
    token_callback(piece);

    if (eos_ids_->find(cur_token) != eos_ids_->end()) {
      break;
    }
  }

  return pos - start_pos;
}
} // namespace rtc_runner
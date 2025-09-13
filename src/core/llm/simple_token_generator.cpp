#include "simple_token_generator.h"
#include "../../utils/argmax_utils.h"
#include "./llama_runner.h"
#include <executorch/extension/tensor/tensor.h>

namespace rtc_runner {

SimpleTokenGenerator::SimpleTokenGenerator(
    TokenizerAdapter *tokenizer, executorch::extension::Module *module,
    std::unique_ptr<std::unordered_set<uint64_t>> &&eos_ids)
    : tokenizer_(tokenizer), module_(module), eos_ids_(std::move(eos_ids)) {}

::executorch::runtime::Result<int64_t> SimpleTokenGenerator::generate(
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
  5. Set should_stop_ to false
  6. Loop until max context length or should_stop_ is true:
     a. Call module_->forward() with current token and position
     b. Extract logits from the result
     c. Get next token using logits_to_token()
     d. Increment position
     e. Update token_data with new token
     f. Decode token to string and call token_callback
     g. Check if token is EOS and break if so
  7. Return the number of tokens generated

  Key details:
  - Use tokens.back() to get the starting token
  - Token tensor shape should be {1, 1} for single token
  - Position tensor shape should be {1}
  - Use ScalarType::Long for both tensors
  - Check eos_ids_->find(cur_token) != eos_ids_->end() for EOS detection
  - Return pos - start_pos as the number of generated tokens

  Constants:
  - LlamaRunner::MAX_CONTEXT_LEN - 1 as the maximum position
  */

  // TODO: Your implementation here
  return ::executorch::runtime::Error::NotImplemented;
}

void SimpleTokenGenerator::stop() { should_stop_ = true; }

} // namespace rtc_runner
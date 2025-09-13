#include "text_prefiller.h"
#include "../../utils/argmax_utils.h"
#include <executorch/extension/tensor/tensor.h>
#include <iostream>

namespace rtc_runner {

TextPrefiller::TextPrefiller(executorch::extension::Module *module)
    : module_(module) {}

::executorch::runtime::Result<uint64_t>
TextPrefiller::prefill(std::vector<uint64_t> &prompt_tokens,
                       int64_t &start_pos) {
  /*
  TODO: Implement the parallel prefill function

  This function should:
  1. Validate input (check if prompt_tokens is empty)
  2. Load the "forward" method if not already loaded
  3. Create tensors from prompt_tokens and start_pos
  4. Execute the model forward pass
  5. Extract the next token from the output logits
  6. Update start_pos by the number of processed tokens
  7. Return the next token

  Required steps:
  - Use executorch::extension::from_blob() to create tensors
  - Call module_->forward() with the input tensors
  - Use logits_to_token() to get the next token from logits
  - Handle all error cases appropriately

  Hints:
  - Check prompt_tokens.empty() first
  - Use module_->is_method_loaded("forward") to check if method is loaded
  - tensor shape for tokens: {1, num_prompt_tokens}
  - tensor shape for start_pos: {1}
  - ScalarType should be Long for both tensors
  */

  // TODO: Your implementation here
  if (!module_->is_method_loaded("forward")) {
    auto load_result = module_->load_method("forward");
    if (load_result != ::executorch::runtime::Error::Ok) {
      return load_result;
    }
  }

  int32_t num_prompt_tokens = prompt_tokens.size();
  uint64_t cur_token;

  auto tokens = executorch::extension::from_blob(
      prompt_tokens.data(), {1, num_prompt_tokens},
      executorch::aten::ScalarType::Long);

  auto start_pos_tensor = executorch::extension::from_blob(
      &start_pos, {1}, executorch::aten::ScalarType::Long);

  auto outputs_res = module_->forward({tokens, start_pos_tensor});
  if (outputs_res.error() != ::executorch::runtime::Error::Ok) {
    return outputs_res.error();
  }

  auto logits = outputs_res.get()[0].toTensor();
  start_pos += num_prompt_tokens;
  cur_token = logits_to_token(logits);

  return cur_token;
}

} // namespace rtc_runner
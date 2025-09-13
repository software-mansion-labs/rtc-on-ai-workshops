#pragma once

#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/compiler.h>

namespace rtc_runner {

template <typename T>
inline int32_t argmax(T* probabilities, int32_t size) {
  int max_i = 0;
  T max_p = probabilities[0];
  for (int i = 1; i < size; i++) {
    if (probabilities[i] > max_p) {
      max_i = i;
      max_p = probabilities[i];
    }
  }
  return max_i;
}

inline int32_t logits_to_token(const executorch::aten::Tensor &logits_tensor) {
  int32_t result = 0;
  ET_SWITCH_THREE_TYPES(Float, Half, BFloat16, logits_tensor.scalar_type(),
                        unused, "logits_to_token", CTYPE, [&]() {
                          auto *logits =
                              logits_tensor.mutable_data_ptr<CTYPE>();
                          if (logits_tensor.dim() == 3) {
                            auto num_tokens = logits_tensor.size(1);
                            auto vocab_size = logits_tensor.size(2);
                            auto *logits_last = logits;
                            logits_last += (num_tokens - 1) * vocab_size;
                            result = argmax(logits_last, vocab_size);
                          } else {
                            result = argmax(logits, logits_tensor.numel());
                          }
                        });
  return result;
}

} // namespace rtc_runner
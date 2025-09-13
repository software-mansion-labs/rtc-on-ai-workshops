#pragma once

#include <fstream>
#include <memory>
#include <string>
#include <tokenizers_cpp.h>
#include <vector>

namespace rtc_runner {

constexpr uint64_t EOS_TOKEN = 128009;

class TokenizerAdapter {
public:
  explicit TokenizerAdapter(const std::string &tokenizer_path);

  std::vector<uint64_t> encode(const std::string &text);
  std::string decode(uint64_t token); // Single token decode for speech-to-text
  uint64_t eos_tok() const { return eos_token_; }
  void set_eos_token(uint64_t eos_token) { eos_token_ = eos_token; }
  std::shared_ptr<tokenizers::Tokenizer> get_tokenizer() const;

private:
  std::shared_ptr<tokenizers::Tokenizer> tokenizer_;
  uint64_t eos_token_ = EOS_TOKEN; // Default to LLaMA EOS token, can be changed
  std::string loadBytesFromFile(const std::string &path);
};

} // namespace rtc_runner
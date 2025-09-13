#include "tokenizer_adapter.h"
#include <stdexcept>

namespace rtc_runner {

TokenizerAdapter::TokenizerAdapter(const std::string &tokenizer_path) {
  tokenizer_ =
      tokenizers::Tokenizer::FromBlobJSON(loadBytesFromFile(tokenizer_path));
}

std::vector<uint64_t> TokenizerAdapter::encode(const std::string &text) {
  auto tokens = tokenizer_->Encode(text);
  std::vector<uint64_t> result;
  for (int32_t token : tokens) {
    result.push_back(static_cast<uint64_t>(token));
  }
  return result;
}

std::string TokenizerAdapter::decode(uint64_t token) {
  return tokenizer_->Decode({static_cast<int32_t>(token)});
}

std::shared_ptr<tokenizers::Tokenizer> TokenizerAdapter::get_tokenizer() const {
  return tokenizer_;
}

std::string TokenizerAdapter::loadBytesFromFile(const std::string &path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    throw std::runtime_error("Failed to open tokenizer file: " + path);
  }
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}

} // namespace rtc_runner
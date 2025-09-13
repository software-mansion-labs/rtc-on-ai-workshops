#include "core/audio/audio_recorder.h"
#include "core/llm/llama_runner.h"
#include "core/stt/speech_to_text.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <thread>
#include <vector>

using executorch::runtime::Error;
using namespace rtc_runner;

int main() {
  std::cout << "LOCAL AI" << std::endl;
  std::cout << std::string(50, '=') << std::endl;

  std::unique_ptr<TokenizerAdapter> tokenizer_;
  tokenizer_ = std::make_unique<TokenizerAdapter>("models/llm/tokenizer.json");

  const std::vector<uint64_t> prompt_tokens =
      tokenizer_->encode("Hello, how are you?");

  for (uint64_t t : prompt_tokens) {
    std::cout << t << " ";
    std::cout << tokenizer_->decode(t) << std::endl;
  }

  return 0;
}
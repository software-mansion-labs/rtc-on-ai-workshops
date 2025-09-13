#include "core/audio/audio_recorder.h"
#include "core/llm/llm_runner.h"
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

  std::unique_ptr<executorch::extension::Module> module =
      std::make_unique<executorch::extension::Module>(
          "models/llm/llama3_2_bf16.pte");

  std::unique_ptr<TokenizerAdapter> tokenizer =
      std::make_unique<TokenizerAdapter>("models/llm/tokenizer.json");

  auto eos_ids = std::make_unique<std::unordered_set<uint64_t>>(
      std::unordered_set<uint64_t>{tokenizer->eos_tok()});

  std::unique_ptr<TokenGenerator> text_token_generator =
      std::make_unique<TokenGenerator>(tokenizer.get(), module.get(),
                                       std::move(eos_ids));

  text_token_generator->generate(
      {15496, 11, 1268, 527, 499}, 5,
      [](const std::string &piece) { std::cout << piece << std::flush; });

  return 0;
}
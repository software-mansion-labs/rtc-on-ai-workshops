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

  std::unique_ptr<TextPrefiller> text_prefiller =
      std::make_unique<TextPrefiller>(module.get());

  std::vector<uint64_t> prompt_tokens = {15496, 11, 1268, 527, 499};
  int64_t start_pos = 0;

  auto prefill_res = text_prefiller->prefill(prompt_tokens, start_pos);

  if (prefill_res.ok()) {
    uint64_t next_token = prefill_res.get();
    std::cout << "Next token after prefill: " << next_token;
  } else {
    std::cout << "Prefill error: " << static_cast<int>(prefill_res.error());
  }

  return 0;
}
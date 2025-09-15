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

  LlmRunner runner("models/llm/llama3_2_bf16.pte", "models/llm/tokenizer.json");

  runner.load();

  runner.generate("Hello, how are you?");

  return 0;
}
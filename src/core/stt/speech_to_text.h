#pragma once

#include "../tokenization/tokenizer_adapter.h"
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace rtc_runner {

class SpeechToText {
public:
  explicit SpeechToText(const std::string &encoderPath,
                        const std::string &decoderPath,
                        const std::string &tokenizerPath);

  ~SpeechToText() = default;

  // Load the models
  bool initialize();

  // Process audio and return transcribed text
  std::string transcribe(std::span<const float> waveform);

  // Check if models are loaded
  bool isLoaded() const;

private:
  static constexpr int FFT_WINDOW_SIZE =
      512; // Back to original for model compatibility
  static constexpr int STFT_HOP_LENGTH = 160; // 10ms hop length
  static constexpr int INNER_DIM = 256;       // Model expects 256 features
  static constexpr int MAX_TOKENS = 256;

  std::string encoderPath_;
  std::string decoderPath_;
  std::string tokenizerPath_;
  std::unique_ptr<executorch::extension::Module> encoder_;
  std::unique_ptr<executorch::extension::Module> decoder_;
  std::unique_ptr<TokenizerAdapter> tokenizer_;

  executorch::runtime::EValue encoderOutput_;

  executorch::extension::TensorPtr
  prepareAudioInput(std::span<const float> waveform);

  executorch::extension::TensorPtr
  prepareTokenInput(const std::vector<int64_t> &tokens);

  std::string decode();

  int32_t extractNextToken(const executorch::aten::Tensor &logitsTensor);

  // Special token IDs for Whisper
  static constexpr int64_t START_OF_TRANSCRIPT = 50257;
  static constexpr int64_t END_OF_TRANSCRIPT = 50256;
  static constexpr int64_t NO_TIMESTAMPS = 50363;

  bool initialized_ = false;
};

} // namespace rtc_runner
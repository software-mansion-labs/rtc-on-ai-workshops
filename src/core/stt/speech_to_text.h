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

  /**
   * Encode raw audio waveform into feature representation for speech
   * recognition. Converts audio through STFT preprocessing and runs encoder
   * forward pass.
   * @param waveform Raw audio samples (16kHz mono, normalized -1.0 to 1.0)
   * @return Encoded audio features as EValue, or empty EValue on error
   */
  executorch::runtime::EValue encode(std::span<const float> waveform);

  /**
   * Decode encoded audio features into text using autoregressive generation.
   * Performs Whisper-style decoding with special tokens and EOS detection.
   * @param encoderOutput Encoded audio features from the encode() method
   * @return Decoded text string, or empty string on error
   */
  std::string decode(const executorch::runtime::EValue &encoderOutput);

  // Process audio and return transcribed text (uses encode + decode)
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

  executorch::extension::TensorPtr
  prepareAudioInput(std::span<const float> waveform);

  executorch::extension::TensorPtr
  prepareTokenInput(const std::vector<int64_t> &tokens);

  int32_t extractNextToken(const executorch::aten::Tensor &logitsTensor);

  // Special token IDs for Whisper
  static constexpr int64_t START_OF_TRANSCRIPT = 50257;
  static constexpr int64_t END_OF_TRANSCRIPT = 50256;
  static constexpr int64_t NO_TIMESTAMPS = 50363;

  bool initialized_ = false;
};

} // namespace rtc_runner
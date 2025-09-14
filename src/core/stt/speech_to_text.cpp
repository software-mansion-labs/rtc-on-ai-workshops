#include "speech_to_text.h"
#include "../../utils/argmax_utils.h"
#include "../audio/dsp_utils.h"
#include "../tokenization/tokenizer_adapter.h"
#include <algorithm>
#include <iostream>

// Use the custom span from dsp_utils.h directly

namespace rtc_runner {

SpeechToText::SpeechToText(const std::string &encoderPath,
                           const std::string &decoderPath,
                           const std::string &tokenizerPath)
    : encoderPath_(encoderPath), decoderPath_(decoderPath),
      tokenizerPath_(tokenizerPath) {}

bool SpeechToText::initialize() {
  try {
    // Load encoder
    encoder_ = std::make_unique<executorch::extension::Module>(
        encoderPath_,
        executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);

    auto encoder_result = encoder_->load_method("forward");
    if (encoder_result != executorch::runtime::Error::Ok) {
      std::cerr << "Failed to load encoder method, error code: "
                << static_cast<int>(encoder_result) << std::endl;
      return false;
    }

    // Load decoder
    decoder_ = std::make_unique<executorch::extension::Module>(
        decoderPath_,
        executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);

    auto decoder_result = decoder_->load_method("forward");
    if (decoder_result != executorch::runtime::Error::Ok) {
      std::cerr << "Failed to load decoder method, error code: "
                << static_cast<int>(decoder_result) << std::endl;
      return false;
    }

    // Load tokenizer and set EOS token for Whisper
    tokenizer_ = std::make_unique<TokenizerAdapter>(tokenizerPath_);
    tokenizer_->set_eos_token(END_OF_TRANSCRIPT); // 50256 for Whisper

    initialized_ = true;
    std::cout << "✅ Speech-to-text models and tokenizer loaded successfully!"
              << std::endl;
    return true;

  } catch (const std::exception &e) {
    std::cerr << "Failed to initialize speech-to-text: " << e.what()
              << std::endl;
    return false;
  }
}

bool SpeechToText::isLoaded() const {
  return initialized_ && encoder_ && decoder_ &&
         encoder_->is_method_loaded("forward") &&
         decoder_->is_method_loaded("forward");
}

executorch::extension::TensorPtr
SpeechToText::prepareAudioInput(std::span<const float> waveform) {
  // Convert audio to STFT features (similar to Whisper preprocessing)

  // Convert std::span to our custom span
  std::span<float> audioSpan(const_cast<float *>(waveform.data()),
                             waveform.size());
  auto preprocessedData =
      dsp::stftFromWaveform(audioSpan, FFT_WINDOW_SIZE, STFT_HOP_LENGTH);

  const auto numFrames = preprocessedData.size() / INNER_DIM;
  std::vector<int32_t> inputShape = {static_cast<int32_t>(numFrames),
                                     INNER_DIM};

  return executorch::extension::make_tensor_ptr(std::move(inputShape),
                                                std::move(preprocessedData));
}

executorch::extension::TensorPtr
SpeechToText::prepareTokenInput(const std::vector<int64_t> &tokens) {
  std::vector<int32_t> tokens32;
  tokens32.reserve(tokens.size());
  for (auto token : tokens) {
    tokens32.push_back(static_cast<int32_t>(token));
  }

  std::vector<int32_t> tensorSizes = {1, static_cast<int32_t>(tokens32.size())};
  return executorch::extension::make_tensor_ptr(std::move(tensorSizes),
                                                std::move(tokens32));
}

std::string SpeechToText::transcribe(std::span<const float> waveform) {
  /*
  TODO: Implement the complete speech-to-text transcription pipeline

  This function orchestrates the complete STT inference process:
  1. Audio preprocessing (encoder input)
  2. Encoder forward pass (audio features → encoded representation)
  3. Decoder autoregressive generation (encoded features → text tokens → text)

  Steps to implement:
  1. Check if models are loaded using isLoaded()
  2. Prepare audio input tensor using prepareAudioInput()
  3. Run encoder forward pass to get audio feature representation
  4. Store encoder output for decoder use
  5. Call decode() method to convert features to text
  6. Handle all errors with appropriate error messages

  Key details:
  - Use encoder_->forward() with audio tensor
  - Check encoderResult.ok() for success
  - Store result as encoderOutput_ = encoderResult.get().at(0)
  - Return empty string on any failure
  - Use try-catch for exception handling

  Error handling:
  - Return empty string if models not loaded
  - Print error codes for encoder failures
  - Catch and print any exceptions
  */

  // TODO: Your implementation here
  return "";
}

std::string
SpeechToText::decode(const executorch::runtime::EValue &encoderOutput) {
  /*
  TODO: Implement autoregressive decoding for speech-to-text

  This function performs autoregressive decoding to convert encoded audio
  features into text tokens, then converts tokens to readable text.

  Whisper-style decoding process:
  1. Initialize with special tokens [START_OF_TRANSCRIPT, language_token,
  NO_TIMESTAMPS]
  2. Loop until MAX_TOKENS or EOS token:
     a. Convert current tokens to tensor input
     b. Run decoder forward pass with token tensor and encoder output
     c. Extract next token from logits
     d. Check for EOS (end of transcript)
     e. Add token to sequence and decode to text
  3. Return accumulated text result

  Steps to implement:
  1. Check if encoderOutput_ is available (use encoderOutput_.isNone())
  2. Initialize token sequence: {START_OF_TRANSCRIPT, 50258, NO_TIMESTAMPS}
     - START_OF_TRANSCRIPT = start token
     - 50258 = <|en|> language token for English
     - NO_TIMESTAMPS = no timestamp mode
  3. Initialize empty result string
  4. Autoregressive loop (step < MAX_TOKENS):
     a. Create token tensor using prepareTokenInput(tokens)
     b. Call decoder_->execute("forward", {tokenTensor, encoderOutput_})
     c. Check decoderResult.ok() for errors
     d. Extract logits: decoderResult.get().at(0).toTensor()
     e. Get next token using extractNextToken(logitsTensor)
     f. Check if nextToken equals tokenizer_->eos_tok() (end condition)
     g. Add nextToken to tokens vector
     h. Decode token to text:
  tokenizer_->decode(static_cast<uint64_t>(nextToken)) i. Append text to result
  if not empty
  5. Return final result string

  Key details:
  - Use decoder_->execute() not decoder_->forward()
  - Cast nextToken to uint64_t for tokenizer_->decode()
  - Break loop on EOS token or error
  - Print step number and error code on decoder failures

  Constants used:
  - START_OF_TRANSCRIPT, NO_TIMESTAMPS (defined in header)
  - MAX_TOKENS for maximum generation length
  */

  // TODO: Your implementation here
  return "";
}

int32_t
SpeechToText::extractNextToken(const executorch::aten::Tensor &logitsTensor) {
  // Similar to WhisperStrategy::extractOutputToken, but for next token
  // prediction
  const auto innerDim = logitsTensor.size(1);
  const auto vocabSize = logitsTensor.size(2);

  // Get the logits for the last token position (for next token prediction)
  const auto *dataPtr =
      static_cast<const float *>(logitsTensor.const_data_ptr());
  const auto *lastTokenLogits = dataPtr + (innerDim - 1) * vocabSize;

  // Find the token with highest probability using argmax
  return logits_to_token(logitsTensor);
}

} // namespace rtc_runner
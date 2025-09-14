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

executorch::runtime::EValue
SpeechToText::encode(std::span<const float> waveform) {
  /*
  TODO: Implement audio encoding for speech-to-text

  This method converts raw audio waveform into encoded features that can be
  used by the decoder. This is the first step of the STT pipeline.

  Steps to implement:
  1. Check if models are loaded using isLoaded()
  2. Prepare audio input tensor using prepareAudioInput()
  3. Run encoder forward pass to get audio feature representation
  4. Return the encoded features as EValue
  5. Handle all errors with appropriate error messages

  Key details:
  - Use encoder_->forward() with audio tensor
  - Check encoderResult.ok() for success
  - Return encoderResult.get().at(0) on success
  - Return empty EValue() on any failure
  - Use try-catch for exception handling

  Error handling:
  - Return empty EValue if models not loaded
  - Print error codes for encoder failures
  - Catch and print any exceptions
  */

  if (!isLoaded()) {
    std::cerr << "Models not loaded. Call initialize() first." << std::endl;
    return executorch::runtime::EValue();
  }

  try {
    const auto audioTensor = prepareAudioInput(waveform);

    const auto encoderResult = encoder_->forward(
        std::vector<executorch::runtime::EValue>{audioTensor});
    if (!encoderResult.ok()) {
      std::cerr << "Encoder forward pass failed, error code: "
                << static_cast<int>(encoderResult.error()) << std::endl;
      return executorch::runtime::EValue();
    }

    return encoderResult.get().at(0);

  } catch (const std::exception &e) {
    std::cerr << "Audio encoding failed: " << e.what() << std::endl;
    return executorch::runtime::EValue();
  }
}

std::string SpeechToText::transcribe(std::span<const float> waveform) {
  // Simple orchestration: encode then decode
  auto encoderOutput = encode(waveform);
  if (encoderOutput.isNone()) {
    return "";
  }
  return decode(encoderOutput);
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
  1. Check if encoderOutput is available (use encoderOutput.isNone())
  2. Initialize token sequence: {START_OF_TRANSCRIPT, 50258, NO_TIMESTAMPS}
     - START_OF_TRANSCRIPT = start token
     - 50258 = <|en|> language token for English
     - NO_TIMESTAMPS = no timestamp mode
  3. Initialize empty result string
  4. Autoregressive loop (step < MAX_TOKENS):
     a. Create token tensor using prepareTokenInput(tokens)
     b. Call decoder_->execute("forward", {tokenTensor, encoderOutput})
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

  if (encoderOutput.isNone()) {
    std::cerr << "No encoder output available" << std::endl;
    return "";
  }

  // Start with initial tokens for Whisper
  std::vector<int64_t> tokens = {START_OF_TRANSCRIPT, 50258,
                                 NO_TIMESTAMPS}; // 50258 = <|en|>
  std::string result;

  // Autoregressive decoding
  for (int step = 0; step < MAX_TOKENS; ++step) {
    const auto tokenTensor = prepareTokenInput(tokens);

    const auto decoderResult =
        decoder_->execute("forward", {tokenTensor, encoderOutput});
    if (!decoderResult.ok()) {
      std::cerr << "Decoder forward pass failed at step " << step
                << ", error code: " << static_cast<int>(decoderResult.error())
                << std::endl;
      break;
    }

    // Get the logits tensor and extract next token
    const auto logitsTensor = decoderResult.get().at(0).toTensor();

    // Get the last token's logits (for next token prediction)
    const auto vocabSize = logitsTensor.size(2);
    const auto seqLen = logitsTensor.size(1);

    // Extract the next token using proper logits processing (similar to
    // WhisperStrategy)
    int32_t nextToken = extractNextToken(logitsTensor);

    // Check for end of transcript using tokenizer's EOS token
    if (static_cast<uint64_t>(nextToken) == tokenizer_->eos_tok()) {
      break;
    }

    tokens.push_back(nextToken);

    // Convert token to text using TokenizerAdapter
    std::string tokenText =
        tokenizer_->decode(static_cast<uint64_t>(nextToken));
    if (!tokenText.empty()) {
      result += tokenText;
    }
  }

  return result;
}

int32_t
SpeechToText::extractNextToken(const executorch::aten::Tensor &logitsTensor) {
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
#include "dsp_utils.h"
#include <algorithm>
#include <cstring>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace rtc_runner {
namespace dsp {

FFT::FFT(size_t size) : size_(size) { computeTwiddles(); }

FFT::~FFT() = default;

void FFT::computeTwiddles() {
  twiddles_.resize(size_);
  for (size_t i = 0; i < size_; ++i) {
    float angle = -2.0f * M_PI * i / size_;
    twiddles_[i] = std::complex<float>(std::cos(angle), std::sin(angle));
  }
}

void FFT::doFFT(const float *input, std::vector<std::complex<float>> &output) {
  output.resize(size_);

  // Copy input to output as complex numbers
  for (size_t i = 0; i < size_; ++i) {
    output[i] = std::complex<float>(input[i], 0.0f);
  }

  // Simple DFT implementation (not optimized, but functional)
  std::vector<std::complex<float>> temp(size_);
  for (size_t k = 0; k < size_; ++k) {
    temp[k] = std::complex<float>(0, 0);
    for (size_t n = 0; n < size_; ++n) {
      size_t twiddle_idx = (k * n) % size_;
      temp[k] += output[n] * twiddles_[twiddle_idx];
    }
  }
  output = std::move(temp);
}

std::vector<float> hannWindow(size_t size) {
  std::vector<float> window(size);
  for (size_t i = 0; i < size; i++) {
    window[i] = 0.5f * (1 - std::cos(2 * M_PI * i / size));
  }
  return window;
}

std::vector<float> stftFromWaveform(std::span<float> waveform,
                                    std::size_t fftWindowSize,
                                    std::size_t hopSize) {
  // Initialize FFT
  FFT fft(fftWindowSize);

  const auto numFrames = 1 + (waveform.size() - fftWindowSize) / hopSize;
  const auto numBins = fftWindowSize / 2; // 256 bins for the model
  const auto hann = hannWindow(fftWindowSize);
  auto inBuffer = std::vector<float>(fftWindowSize);
  auto outBuffer = std::vector<std::complex<float>>(fftWindowSize);

  // Output magnitudes in dB
  std::vector<float> magnitudes;
  magnitudes.reserve(numFrames * numBins);
  const auto magnitudeScale = 1.0f / static_cast<float>(fftWindowSize);
  constexpr auto epsilon = std::numeric_limits<float>::epsilon();
  constexpr auto dbConversionFactor = 20.0f;

  for (size_t t = 0; t < numFrames; ++t) {
    const size_t offset = t * hopSize;
    // Clear the input buffer first
    std::fill(inBuffer.begin(), inBuffer.end(), 0.0f);

    // Fill frame with windowed signal
    const size_t samplesToRead =
        std::min(fftWindowSize, waveform.size() - offset);
    for (size_t i = 0; i < samplesToRead; i++) {
      inBuffer[i] = waveform[offset + i] * hann[i];
    }

    fft.doFFT(inBuffer.data(), outBuffer);

    // Calculate magnitudes in dB (all 256 bins as expected by model)
    for (size_t i = 0; i < numBins; i++) {
      const auto magnitude = std::abs(outBuffer[i]) * magnitudeScale;
      const auto magnitude_db =
          dbConversionFactor * log10f(magnitude + epsilon);
      magnitudes.push_back(magnitude_db);
    }
  }

  return magnitudes;
}

} // namespace dsp
} // namespace rtc_runner
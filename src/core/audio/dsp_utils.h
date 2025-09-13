#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <span>
#include <vector>

namespace rtc_runner {
namespace dsp {

class FFT {
public:
  explicit FFT(size_t size);
  ~FFT();
  void doFFT(const float *input, std::vector<std::complex<float>> &output);

private:
  size_t size_;
  std::vector<std::complex<float>> twiddles_;
  void computeTwiddles();
};

std::vector<float> hannWindow(size_t size);
std::vector<float> stftFromWaveform(std::span<float> waveform,
                                    std::size_t fftWindowSize,
                                    std::size_t hopSize);

} // namespace dsp
} // namespace rtc_runner
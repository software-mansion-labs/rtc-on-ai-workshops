#pragma once

#include <atomic>
#include <memory>
#include <portaudio.h>
#include <span>
#include <string>
#include <vector>

namespace rtc_runner {

class AudioRecorder {
public:
  AudioRecorder();
  ~AudioRecorder();

  bool initialize();
  bool startRecording();
  bool stopRecording();
  bool saveToWav(const std::string &filename);
  bool loadFromWav(const std::string &filename);
  bool isRecording() const { return recording_.load(); }
  void cleanup();

  // Get the raw audio data for speech-to-text processing
  std::span<const float> getAudioData() const { return audioBuffer_; }
  size_t getAudioDataSize() const { return audioBuffer_.size(); }

private:
  static constexpr int SAMPLE_RATE = 16000;
  static constexpr int FRAMES_PER_BUFFER = 512;
  static constexpr int CHANNELS = 1;

  static int recordCallback(const void *inputBuffer, void *outputBuffer,
                            unsigned long framesPerBuffer,
                            const PaStreamCallbackTimeInfo *timeInfo,
                            PaStreamCallbackFlags statusFlags, void *userData);

  PaStream *stream_;
  std::vector<float> audioBuffer_;
  std::atomic<bool> recording_;
  std::atomic<bool> initialized_;

  void writeWavHeader(FILE *file, int sampleRate, int channels, int dataSize);
};

} // namespace rtc_runner
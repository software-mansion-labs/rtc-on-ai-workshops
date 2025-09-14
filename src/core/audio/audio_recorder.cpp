#include "audio_recorder.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <thread>

namespace rtc_runner {

AudioRecorder::AudioRecorder()
    : stream_(nullptr), recording_(false), initialized_(false) {}

AudioRecorder::~AudioRecorder() { cleanup(); }

bool AudioRecorder::initialize() {
  PaError err = Pa_Initialize();
  if (err != paNoError) {
    std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
    return false;
  }

  initialized_.store(true);
  return true;
}

void AudioRecorder::cleanup() {
  if (recording_.load()) {
    stopRecording();
  }

  if (initialized_.load()) {
    Pa_Terminate();
    initialized_.store(false);
  }
}

int AudioRecorder::recordCallback(const void *inputBuffer, void *outputBuffer,
                                  unsigned long framesPerBuffer,
                                  const PaStreamCallbackTimeInfo *timeInfo,
                                  PaStreamCallbackFlags statusFlags,
                                  void *userData) {
  AudioRecorder *recorder = static_cast<AudioRecorder *>(userData);
  const float *input = static_cast<const float *>(inputBuffer);

  (void)outputBuffer; // Unused
  (void)timeInfo;     // Unused

  if (statusFlags != 0) {
  }

  if (!input) {
    std::cerr << "No input buffer in callback" << std::endl;
    return paContinue;
  }

  if (!recorder->recording_.load()) {
    return paContinue;
  }

  try {
    // Append audio data to buffer
    size_t currentSize = recorder->audioBuffer_.size();
    size_t newSamples = framesPerBuffer * CHANNELS;
    recorder->audioBuffer_.resize(currentSize + newSamples);

    std::memcpy(&recorder->audioBuffer_[currentSize], input,
                newSamples * sizeof(float));

    // Debug: Print every 8000 frames (roughly every 0.5 seconds at 16kHz) to
    // show it's working
    static int frameCount = 0;
    frameCount += framesPerBuffer;
    if (frameCount >= 8000) {
      frameCount = 0;
    }
  } catch (const std::exception &e) {
    std::cerr << "Error in audio callback: " << e.what() << std::endl;
  }

  return paContinue;
}

bool AudioRecorder::startRecording() {
  if (!initialized_.load()) {
    std::cerr << "AudioRecorder not initialized" << std::endl;
    return false;
  }

  if (recording_.load()) {
    std::cerr << "Already recording" << std::endl;
    return false;
  }

  // Clear previous audio data
  audioBuffer_.clear();

  PaStreamParameters inputParameters;
  inputParameters.device = Pa_GetDefaultInputDevice();
  if (inputParameters.device == paNoDevice) {
    std::cerr << "❌ No default input device found" << std::endl;

    return false;
  }

  const PaDeviceInfo *deviceInfo = Pa_GetDeviceInfo(inputParameters.device);

  inputParameters.channelCount = CHANNELS;
  inputParameters.sampleFormat = paFloat32;
  inputParameters.suggestedLatency = deviceInfo->defaultLowInputLatency;
  inputParameters.hostApiSpecificStreamInfo = nullptr;

  PaError err = Pa_OpenStream(&stream_, &inputParameters,
                              nullptr, // no output
                              SAMPLE_RATE, FRAMES_PER_BUFFER, paClipOff,
                              recordCallback, this);

  if (err != paNoError) {
    std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
    return false;
  }

  err = Pa_StartStream(stream_);
  if (err != paNoError) {
    std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
    Pa_CloseStream(stream_);
    stream_ = nullptr;
    return false;
  }

  recording_.store(true);

  // Give a moment for the stream to start and check if we're getting data
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  return true;
}

bool AudioRecorder::stopRecording() {
  if (!recording_.load()) {
    return false;
  }

  recording_.store(false);

  if (stream_) {
    PaError err = Pa_StopStream(stream_);
    if (err != paNoError) {
      std::cerr << "PortAudio error stopping stream: " << Pa_GetErrorText(err)
                << std::endl;
    }

    err = Pa_CloseStream(stream_);
    if (err != paNoError) {
      std::cerr << "PortAudio error closing stream: " << Pa_GetErrorText(err)
                << std::endl;
    }

    stream_ = nullptr;
  }

  return true;
}

void AudioRecorder::writeWavHeader(FILE *file, int sampleRate, int channels,
                                   int dataSize) {
  // WAV file header
  fwrite("RIFF", 1, 4, file);
  int chunkSize = 36 + dataSize;
  fwrite(&chunkSize, 4, 1, file);
  fwrite("WAVE", 1, 4, file);

  // fmt subchunk
  fwrite("fmt ", 1, 4, file);
  int subchunk1Size = 16;
  fwrite(&subchunk1Size, 4, 1, file);
  short audioFormat = 1; // PCM
  fwrite(&audioFormat, 2, 1, file);
  short numChannels = channels;
  fwrite(&numChannels, 2, 1, file);
  fwrite(&sampleRate, 4, 1, file);
  int byteRate = sampleRate * channels * 2; // 16-bit samples
  fwrite(&byteRate, 4, 1, file);
  short blockAlign = channels * 2;
  fwrite(&blockAlign, 2, 1, file);
  short bitsPerSample = 16;
  fwrite(&bitsPerSample, 2, 1, file);

  // data subchunk
  fwrite("data", 1, 4, file);
  fwrite(&dataSize, 4, 1, file);
}

bool AudioRecorder::saveToWav(const std::string &filename) {
  if (audioBuffer_.empty()) {
    std::cerr << "No audio data to save" << std::endl;
    return false;
  }

  FILE *file = fopen(filename.c_str(), "wb");
  if (!file) {
    std::cerr << "Failed to open file for writing: " << filename << std::endl;
    return false;
  }

  int dataSize = audioBuffer_.size() * sizeof(short);
  writeWavHeader(file, SAMPLE_RATE, CHANNELS, dataSize);

  // Convert float samples to 16-bit PCM and write
  for (float sample : audioBuffer_) {
    short pcmSample = static_cast<short>(sample * 32767.0f);
    fwrite(&pcmSample, sizeof(short), 1, file);
  }

  fclose(file);
  std::cout << "📁 Audio saved to: " << filename << std::endl;
  return true;
}

bool AudioRecorder::loadFromWav(const std::string &filename) {
  FILE *file = fopen(filename.c_str(), "rb");
  if (!file) {
    std::cerr << "❌ Cannot open file: " << filename << std::endl;
    return false;
  }

  // Read WAV header
  struct WavHeader {
    char riff[4];
    uint32_t fileSize;
    char wave[4];
    char fmt[4];
    uint32_t fmtSize;
    uint16_t audioFormat;
    uint16_t numChannels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
    char data[4];
    uint32_t dataSize;
  } header;

  if (fread(&header, sizeof(header), 1, file) != 1) {
    fclose(file);
    return false;
  }

  if (std::strncmp(header.riff, "RIFF", 4) != 0 ||
      std::strncmp(header.wave, "WAVE", 4) != 0) {
    fclose(file);
    return false;
  }

  // Read audio data
  std::vector<int16_t> rawData(header.dataSize / sizeof(int16_t));
  if (fread(rawData.data(), sizeof(int16_t), rawData.size(), file) !=
      rawData.size()) {
    fclose(file);
    return false;
  }

  fclose(file);

  // Convert to float [-1.0, 1.0] and store in audioBuffer_
  audioBuffer_.clear();
  audioBuffer_.reserve(rawData.size());

  for (int16_t sample : rawData) {
    audioBuffer_.push_back(static_cast<float>(sample) / 32768.0f);
  }

  return true;
}

} // namespace rtc_runner
#include "core/audio/audio_recorder.h"
#include "core/llm/llm_runner.h"
#include "core/stt/speech_to_text.h"
#include <iostream>
#include <span>
#include <vector>

using namespace rtc_runner;

int main() {
  std::cout << "🎙️  Speech-to-Text Test" << std::endl;
  std::cout << std::string(30, '=') << std::endl;

  AudioRecorder audioRecorder;
  SpeechToText speechToText("models/stt/encoder.pte", "models/stt/decoder.pte",
                            "models/stt/tokenizer.json");
  speechToText.initialize();

  if (!audioRecorder.loadFromWav(
          "/Users/norbertklockiewicz/Desktop/work/swm_ai/"
          "rtc-on-ai-workshops/recordings/recording_1.wav")) {
    std::cerr << "Could not load audio file" << std::endl;
    return 1;
  }

  auto audioData = audioRecorder.getAudioData();
  try {
    std::string transcription = speechToText.transcribe(audioData);

    if (!transcription.empty()) {
      std::cout << "Result: \"" << transcription << "\"" << std::endl;
    } else {
      std::cout << "Transcription failed - no text generated" << std::endl;
    }

  } catch (const std::exception &e) {
    std::cerr << "Transcription error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
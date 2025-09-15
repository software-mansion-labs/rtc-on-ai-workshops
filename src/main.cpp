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
  std::cout << "Voice-to-LLM Chat Application" << std::endl;
  std::cout << std::string(50, '=') << std::endl;

  LlmRunner runner("models/llm/llama3_2_bf16.pte", "models/llm/tokenizer.json");

  AudioRecorder audioRecorder;
  SpeechToText speechToText("models/stt/encoder.pte", "models/stt/decoder.pte",
                            "models/stt/tokenizer.json");

  // Initialize audio recorder
  if (!audioRecorder.initialize()) {
    std::cerr << "Failed to initialize audio recorder" << std::endl;
    return 1;
  }

  // Initialize speech-to-text (required for voice interaction)
  bool sttAvailable = speechToText.initialize();
  if (!sttAvailable) {
    std::cerr << " Speech-to-text models are required for voice interaction!"
              << std::endl;
    std::cerr << "   Make sure the models are available in models/stt/"
              << std::endl;
    return 1;
  }

  // Default system prompt for the assistant
  const std::string system_prompt =
      "You are a helpful, harmless, and honest AI assistant. "
      "Always provide accurate and helpful responses.";

  std::cout << "\n Voice chat is ready!" << std::endl;
  std::cout << "\n Commands:" << std::endl;
  std::cout << "'record' - Start voice recording" << std::endl;
  std::cout << "'stop'   - Stop recording and send to AI" << std::endl;
  std::cout << "'quit'   - Exit application" << std::endl;
  std::cout << "'text'   - Switch to text mode (optional)" << std::endl;
  std::cout << "\n Primary mode: Voice interaction" << std::endl;
  std::cout << "===========================================\n" << std::endl;

  std::string user_input;
  int recordingCounter = 1;
  bool isRecording = false;
  bool textMode = false;

  while (true) {
    if (isRecording) {
      std::cout << "RECORDING... (type 'stop' to finish): ";
    } else if (textMode) {
      std::cout << "You (text): ";
    } else {
      std::cout << "Ready for voice (type 'record'): ";
    }
    std::getline(std::cin, user_input);

    if (user_input == "quit" || user_input == "exit") {
      std::cout << "Goodbye!" << std::endl;
      break;
    }

    if (user_input == "text") {
      textMode = !textMode;
      if (textMode) {
        std::cout << "Switched to text mode. Type normally to chat."
                  << std::endl;
      } else {
        std::cout << "Switched back to voice mode. Use 'record' command."
                  << std::endl;
      }
      continue;
    }

    if (user_input == "record") {
      if (!isRecording) {
        std::cout << "Recording started... Speak now!" << std::endl;
        std::cout << "   (Type 'stop' when finished speaking)" << std::endl;
        audioRecorder.startRecording();
        isRecording = true;
      } else {
        std::cout << "Already recording! Type 'stop' to finish." << std::endl;
      }
      continue;
    }

    if (user_input == "stop") {
      if (isRecording) {
        std::cout << "Processing your voice..." << std::endl;
        audioRecorder.stopRecording();
        std::string filename =
            "recording_" + std::to_string(recordingCounter++) + ".wav";
        audioRecorder.saveToWav(filename);
        isRecording = false;

        // Transcribe the audio and send to LLM
        try {
          // Get the raw audio data from the recorder
          auto audioData = audioRecorder.getAudioData();

          if (audioData.empty()) {
            std::cout << "No audio data captured" << std::endl;
          } else {
            // Transcribe the audio using the speech-to-text model
            std::string transcription = speechToText.transcribe(audioData);

            if (!transcription.empty()) {
              std::cout << "You said: \"" << transcription << "\"" << std::endl;
              std::cout << "AI is thinking..." << std::endl;

              // Send the transcribed text to the LLM
              auto result = runner.chat(transcription, system_prompt);
              if (result != Error::Ok) {
                std::cout << "Chat failed with error: "
                          << static_cast<int>(result) << std::endl;
              }
              std::cout << std::endl;
            } else {
              std::cout << "Could not understand the audio. Please try again."
                        << std::endl;
              std::cout
                  << "Make sure to speak clearly and check your microphone."
                  << std::endl;
            }
          }

        } catch (const std::exception &e) {
          std::cerr << "Voice processing failed: " << e.what() << std::endl;
          std::cout << "Please try recording again." << std::endl;
        }
      }
      continue;
    }

    // Handle text input in text mode
    if (textMode && !user_input.empty()) {
      std::cout << "AI is responding..." << std::endl;
      auto result = runner.chat(user_input, system_prompt);
      if (result != Error::Ok) {
        std::cout << "Chat failed with error: " << static_cast<int>(result)
                  << std::endl;
      }
      std::cout << std::endl;
    } else if (!textMode && !user_input.empty() && user_input != "record" &&
               user_input != "stop") {
      // In voice mode, ignore non-command text input
      std::cout << "Use 'record' to start voice input, or 'text' to switch "
                   "to text mode."
                << std::endl;
    }
  }

  return 0;
}
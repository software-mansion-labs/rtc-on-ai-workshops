# RTC Voice-to-LLM Chat Application

A real-time voice chat application that records audio, converts speech to text using Whisper, and generates responses using LLaMA models.

## 🏗️ Project Structure

```
rtc_on/
├── src/                          # Source code
│   ├── main.cpp                  # Main application entry
│   ├── core/                     # Core functionality
│   │   ├── audio/               # Audio processing
│   │   │   ├── audio_recorder.{h,cpp}    # PortAudio recording
│   │   │   └── dsp_utils.{h,cpp}         # STFT/DSP processing  
│   │   ├── llm/                 # LLM functionality
│   │   │   ├── llama_runner.{h,cpp}      # LLaMA model runner
│   │   │   ├── chat_template.{h,cpp}     # Chat formatting
│   │   │   ├── text_prefiller.{h,cpp}    # Text generation helpers
│   │   │   └── simple_token_generator.{h,cpp}
│   │   ├── stt/                 # Speech-to-text
│   │   │   └── speech_to_text.{h,cpp}    # Whisper model integration
│   │   └── tokenization/        # Tokenizer functionality  
│   │       └── tokenizer_adapter.{h,cpp} # HuggingFace tokenizer wrapper
│   └── utils/                   # Utility functions
│       ├── argmax_utils.h       # Token selection utilities
│       └── keyboard_input.{h,cpp} # Input handling
├── models/                      # Model files (gitignored)
│   ├── llm/                     # LLM models
│   │   ├── llama3_2_bf16.pte   # LLaMA model
│   │   └── tokenizer.json       # LLM tokenizer
│   └── stt/                     # Speech-to-text models  
│       ├── encoder.pte          # Whisper encoder
│       ├── decoder.pte          # Whisper decoder
│       └── tokenizer.json       # Whisper tokenizer
├── recordings/                  # Audio recordings (gitignored)
├── scripts/                     # Build scripts
│   └── build.sh                 # Build automation
├── third_party/                 # External dependencies
│   └── executorch/              # ExecutorTorch framework
├── build/                       # Build output (gitignored)
├── CMakeLists.txt              # Build configuration
└── README.md                   # This file
```

## 🚀 Quick Start

1. **Build the project:**
   ```bash
   ./scripts/build.sh --clean --run
   ```

2. **Run the application:**
   ```bash
   ./build/rtc_on_workshops
   ```

3. **Voice interaction:**
   - Type `record` to start recording
   - Speak your message
   - Type `stop` to process and send to AI
   - Type `quit` to exit

## 🎯 Features

- **Voice-first interaction** with fallback to text mode
- **Real-time audio recording** using PortAudio
- **Speech-to-text** using Whisper (encoder-decoder architecture)  
- **LLM responses** using LLaMA models via ExecutorTorch
- **Modular architecture** with clear separation of concerns

## 🛠️ Dependencies

- **ExecutorTorch**: ML model inference framework
- **PortAudio**: Cross-platform audio I/O library  
- **HuggingFace Tokenizers**: Text tokenization
- **C++20**: Modern C++ standard

## 📝 Usage

### Voice Mode (Default)
- `record` - Start voice recording
- `stop` - Stop recording and send to AI
- `text` - Switch to text input mode
- `quit` - Exit application

### Text Mode  
- Type messages normally
- `text` - Switch back to voice mode
- `quit` - Exit application
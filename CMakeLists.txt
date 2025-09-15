cmake_minimum_required(VERSION 3.19)
project(rtc_on_workshops)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set options for executorch build.
option(EXECUTORCH_ENABLE_LOGGING "" ON)
set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON)
set(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR ON)
set(EXECUTORCH_BUILD_EXTENSION_MODULE ON)
set(EXECUTORCH_BUILD_EXTENSION_TENSOR ON)
set(EXECUTORCH_BUILD_KERNELS_OPTIMIZED ON)
set(EXECUTORCH_BUILD_XNNPACK ON)
set(EXECUTORCH_BUILD_KERNELS_CUSTOM ON)
set(EXECUTORCH_BUILD_KERNELS_PORTABLE ON)
set(EXECUTORCH_BUILD_KERNELS_QUANTIZED ON)

# Include the executorch subdirectory.
add_subdirectory(third-party/executorch)

# Fetch tokenizers_cpp
include(FetchContent)
FetchContent_Declare(
  tokenizers_cpp
  GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp
  GIT_TAG main
)
FetchContent_MakeAvailable(tokenizers_cpp)

# Find required packages for audio recording
if(APPLE)
    # On macOS, try to find PortAudio via Homebrew
    find_library(PORTAUDIO_LIBRARIES portaudio PATHS /opt/homebrew/opt/portaudio/lib /opt/homebrew/lib /usr/local/lib)
    find_path(PORTAUDIO_INCLUDE_DIRS portaudio.h PATHS /opt/homebrew/opt/portaudio/include /opt/homebrew/include /usr/local/include)
    
    if(NOT PORTAUDIO_LIBRARIES)
        message(FATAL_ERROR "PortAudio not found. Install with: brew install portaudio")
    endif()
    
    message(STATUS "Found PortAudio: ${PORTAUDIO_LIBRARIES}")
    message(STATUS "PortAudio include dir: ${PORTAUDIO_INCLUDE_DIRS}")
else()
    # On Linux, use pkg-config
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)
endif()

add_executable(rtc_on_workshops 
  src/main.cpp
  src/core/llm/text_prefiller.cpp
  src/core/llm/chat_template.cpp
  src/core/tokenization/tokenizer_adapter.cpp
  src/core/llm/token_generator.cpp
  src/core/llm/llm_runner.cpp
  src/core/audio/audio_recorder.cpp
  src/core/stt/speech_to_text.cpp
  src/core/audio/dsp_utils.cpp
)
target_link_libraries(
  rtc_on_workshops
  PRIVATE executorch
          extension_module # Provides the Module class
          extension_tensor # Provides the TensorPtr class
          optimized_native_cpu_ops_lib
          xnnpack_backend # Provides the XNNPACK CPU acceleration backend
          tokenizers_cpp # Provides tokenizer functionality
          optimized_kernels
          portable_kernels
          quantized_ops_lib
          $<LINK_LIBRARY:WHOLE_ARCHIVE,custom_ops>
          ${PORTAUDIO_LIBRARIES}
)

# Add include directories for PortAudio
target_include_directories(rtc_on_workshops PRIVATE ${PORTAUDIO_INCLUDE_DIRS})
if(NOT APPLE)
    target_compile_options(rtc_on_workshops PRIVATE ${PORTAUDIO_CFLAGS_OTHER})
endif()
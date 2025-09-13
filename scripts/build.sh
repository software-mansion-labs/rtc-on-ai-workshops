#!/bin/bash

# Fast ExecutorTorch Build Script
echo "🚀 Building ExecutorTorch project..."

# Parse command line arguments
CLEAN_BUILD=false
RUN_AFTER=false

for arg in "$@"; do
    case $arg in
        --clean|-c)
            CLEAN_BUILD=true
            shift
            ;;
        --run|-r)
            RUN_AFTER=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --clean, -c    Clean rebuild (slow, ~10-15 min)"
            echo "  --run, -r      Run executable after build"
            echo "  --help, -h     Show this help"
            echo ""
            echo "Examples:"
            echo "  $0              # Fast incremental build (~10-30 sec)"
            echo "  $0 --clean     # Clean rebuild from scratch" 
            echo "  $0 --run       # Fast build + run chatbot"
            echo "  $0 --clean --run  # Clean rebuild + run"
            exit 0
            ;;
    esac
done

# Handle clean build
if [ "$CLEAN_BUILD" = true ]; then
    echo "🧹 Clean rebuild requested..."
    if [ -d "build" ]; then
        echo "   Removing previous build directory..."
        rm -rf build
    fi
    echo "   Creating fresh build directory..."
    mkdir build
    cd build
    echo "⚙️  Configuring with CMake (this may take a few minutes)..."
    cmake .. -DCMAKE_BUILD_TYPE=Release
else
    echo "⚡ Fast incremental build..."
    # Create build directory if it doesn't exist
    if [ ! -d "build" ]; then
        echo "   No build directory found, creating one..."
        mkdir build
        cd build
        echo "⚙️  Initial CMake configuration..."
        cmake .. -DCMAKE_BUILD_TYPE=Release
    else
        echo "   Using existing build directory..."
        cd build
    fi
fi

# Build the project
echo "🔨 Building project (targeting rtc_on_workshops only)..."
start_time=$(date +%s)
make -j$(nproc 2>/dev/null || echo 4) rtc_on_workshops

# Check if build was successful
if [ $? -eq 0 ]; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "✅ Build successful in ${duration}s!"
    
    if [ "$RUN_AFTER" = true ]; then
        echo "🤖 Starting chatbot..."
        echo "   (Run from project root for tokenizer to work)"
        cd ..
        ./build/rtc_on_workshops
    else
        echo "💡 To run: cd .. && ./build/rtc_on_workshops"
    fi
else
    echo "❌ Build failed!"
    exit 1
fi
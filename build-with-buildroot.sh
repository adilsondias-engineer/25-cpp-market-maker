#!/bin/bash
# Build market_maker with Buildroot cross-compiler
# This script sets up the environment and builds the project

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILDROOT_HOST="/work/tos/buildroot/output/host"

if [ ! -d "$BUILDROOT_HOST" ]; then
    echo "Error: Buildroot host directory not found at $BUILDROOT_HOST"
    echo "Please build Buildroot first: cd /work/tos/buildroot && ./run.sh"
    exit 1
fi

# Set Buildroot toolchain paths
export PATH="$BUILDROOT_HOST/bin:$PATH"

# Set cross-compilation variables
export CC=x86_64-buildroot-linux-gnu-gcc
export CXX=x86_64-buildroot-linux-gnu-g++
export CMAKE_SYSROOT="$BUILDROOT_HOST/x86_64-buildroot-linux-gnu/sysroot"

echo "=========================================="
echo "Building market_maker with Buildroot toolchain"
echo "=========================================="
echo "CC: $CC"
echo "CXX: $CXX"
echo "SYSROOT: $CMAKE_SYSROOT"
echo ""

# Verify toolchain
if ! command -v "$CC" >/dev/null 2>&1; then
    echo "Error: $CC not found in PATH"
    echo "PATH: $PATH"
    exit 1
fi

# Create build directory
cd "$SCRIPT_DIR"
rm -rf build
mkdir build
cd build

echo "Configuring CMake..."
# Explicitly exclude vcpkg to prevent linking host-built libraries
cmake .. \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=x86_64 \
    -DCMAKE_C_COMPILER="$CC" \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_SYSROOT="$CMAKE_SYSROOT" \
    -DCMAKE_FIND_ROOT_PATH="$CMAKE_SYSROOT" \
    -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
    -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
    -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
    -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY \
    -DCMAKE_PREFIX_PATH="$CMAKE_SYSROOT/usr" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

echo ""
echo "Building..."
make -j$(nproc)

echo ""
echo "=========================================="
echo "Build complete!"
echo "=========================================="
echo "Binary: $SCRIPT_DIR/build/market_maker"
echo ""
echo "Copying to Buildroot overlay:"
mkdir -p /work/tos/trading-linux/buildroot-external/board/trading/overlay/opt/trading/bin
mkdir -p /work/tos/trading-linux/buildroot-external/board/trading/overlay/opt/trading/config
cp market_maker /work/tos/trading-linux/buildroot-external/board/trading/overlay/opt/trading/bin
cp ../config.json /work/tos/trading-linux/buildroot-external/board/trading/overlay/opt/trading/config/p25_config.json
echo "Done"
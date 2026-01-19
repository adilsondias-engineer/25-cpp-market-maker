cmake_minimum_required(VERSION 3.20)
project(marketMaker CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Project 25: Market Maker + XGBoost Inference
# Disruptor Consumer from P24
message(STATUS "==============================================")
message(STATUS "  Project 25: Market Maker + XGBoost Inference")
message(STATUS "  Disruptor Consumer from P24")
message(STATUS "==============================================")

# Add vcpkg installed directory to CMAKE_PREFIX_PATH if it exists
# BUT: Skip vcpkg when cross-compiling (e.g., with Buildroot toolchain)
# vcpkg libraries are built with host compiler and won't work with cross-compiler
if(CMAKE_CROSSCOMPILING)
    message(STATUS "Cross-compiling detected - skipping vcpkg (use Buildroot packages instead)")
    # Explicitly remove vcpkg paths to prevent CMake from finding host-built libraries
    list(REMOVE_ITEM CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-linux/share")
    # Also remove from CMAKE_FIND_ROOT_PATH if set
    list(REMOVE_ITEM CMAKE_FIND_ROOT_PATH "${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-linux")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-linux/share")
    list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-linux/share")
    message(STATUS "Added vcpkg_installed to CMAKE_PREFIX_PATH")
endif()

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "CMAKE_BUILD_TYPE not set, defaulting to Release")
endif()

# Generate compile_commands.json for IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/../common)

# Source files (Disruptor-only, no TCP client)
set(SOURCES
    src/main.cpp
    src/bbo_parser.cpp
    src/market_maker_fsm.cpp
    src/position_tracker.cpp
    src/order_producer.cpp
)

# Assembly optimization files (x86-64 FMA)
set(ASM_SOURCES
    src/asm/trading_math_asm.S
)

# Enable assembly language support
enable_language(ASM)

# Executable (includes assembly sources)
add_executable(market_maker ${SOURCES} ${ASM_SOURCES})

# spdlog - use system library when cross-compiling, FetchContent otherwise
if(CMAKE_CROSSCOMPILING)
    message(STATUS "Cross-compiling: using system spdlog with external fmt")
    find_package(spdlog REQUIRED)
    find_package(fmt REQUIRED)
    # Tell spdlog to use external fmt (Buildroot configuration)
    add_compile_definitions(SPDLOG_FMT_EXTERNAL)
    # Link spdlog with external fmt
    target_link_libraries(market_maker PRIVATE spdlog::spdlog fmt::fmt)
else()
    # Native build: use FetchContent with bundled fmt to avoid system fmt version conflicts
    message(STATUS "Native build: using FetchContent spdlog with bundled fmt")
    include(FetchContent)
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.14.1
        GIT_SHALLOW TRUE
    )
    set(SPDLOG_FMT_EXTERNAL OFF CACHE BOOL "" FORCE)  # Use bundled fmt
    set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)  # Static/header-only
    FetchContent_MakeAvailable(spdlog)
    # Link spdlog (header-only with bundled fmt)
    target_link_libraries(market_maker PRIVATE spdlog::spdlog_header_only)
endif()

# nlohmann_json - try find_package first, fallback to FetchContent
find_package(nlohmann_json CONFIG QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found, fetching from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()
target_link_libraries(market_maker PRIVATE nlohmann_json::nlohmann_json)

# Link pthread library (required for RT scheduling and CPU pinning)
find_package(Threads REQUIRED)
target_link_libraries(market_maker PRIVATE Threads::Threads)

# XGBoost for GPU inference (moved from P24 for lower latency pipeline)
# XGBoost is installed in Buildroot at /opt/xgboost
set(XGBOOST_ROOT "/opt/xgboost" CACHE PATH "XGBoost installation directory")
if(CMAKE_CROSSCOMPILING)
    # For cross-compilation, use Buildroot's XGBoost in sysroot
    set(XGBOOST_INCLUDE_DIR "${CMAKE_SYSROOT}/opt/xgboost/include")
    set(XGBOOST_LIBRARY "${CMAKE_SYSROOT}/opt/xgboost/lib/libxgboost.so")
else()
    # For native build, look in standard locations
    find_path(XGBOOST_INCLUDE_DIR xgboost/c_api.h
        HINTS ${XGBOOST_ROOT}/include /opt/xgboost/include /usr/include /usr/local/include
    )
    find_library(XGBOOST_LIBRARY xgboost
        HINTS ${XGBOOST_ROOT}/lib /opt/xgboost/lib /usr/lib /usr/local/lib
    )
endif()

if(XGBOOST_INCLUDE_DIR AND XGBOOST_LIBRARY)
    message(STATUS "Found XGBoost: ${XGBOOST_LIBRARY}")
    message(STATUS "XGBoost include: ${XGBOOST_INCLUDE_DIR}")
    target_include_directories(market_maker PRIVATE ${XGBOOST_INCLUDE_DIR})
    target_link_libraries(market_maker PRIVATE ${XGBOOST_LIBRARY})
    target_compile_definitions(market_maker PRIVATE HAVE_XGBOOST=1)
else()
    message(WARNING "XGBoost not found - building without ML inference support")
    message(WARNING "  XGBOOST_INCLUDE_DIR: ${XGBOOST_INCLUDE_DIR}")
    message(WARNING "  XGBOOST_LIBRARY: ${XGBOOST_LIBRARY}")
endif()

# Ensure libstdc++ is linked (required for C++ ABI functions like __cxa_call_terminate)
# This is especially important when cross-compiling
# The undefined reference to __cxa_call_terminate indicates libstdc++ is not being linked
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # Explicitly link libstdc++ to ensure C++ ABI functions are available
    # This fixes the __cxa_call_terminate undefined reference error
    target_link_libraries(market_maker PRIVATE stdc++fs)  # C++17 filesystem
    
    # For cross-compilation, ensure we're using Buildroot's libstdc++
    if(CMAKE_CROSSCOMPILING)
        message(STATUS "Cross-compiling: using Buildroot's libstdc++ from ${CMAKE_SYSROOT}")
        # CMAKE_SYSROOT should point to Buildroot's sysroot
        # The linker should automatically find libstdc++ there
    endif()
endif()

# Installation
install(TARGETS market_maker DESTINATION bin)

# Linux-specific optimizations
if(UNIX AND NOT APPLE)
    target_compile_options(market_maker PRIVATE -march=native -O3)
endif()

# Standalone benchmark (no external dependencies)
add_executable(trading_math_benchmark
    benchmark/trading_math_benchmark.cpp
    ${ASM_SOURCES}
)
target_include_directories(trading_math_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(trading_math_benchmark PRIVATE Threads::Threads)
target_compile_options(trading_math_benchmark PRIVATE -march=native -O3)

# Print build information
message(STATUS "")
message(STATUS "Build configuration:")
message(STATUS "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "  CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
message(STATUS "")

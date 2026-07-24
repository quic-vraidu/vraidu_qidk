# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
cmake_minimum_required(VERSION 3.18.1)
project("vlmassistant")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

# All Genie SDK headers (GenieNode.h, GeniePipeline.h, GenieCommon.h, etc.)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/genie/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(vlmassistant SHARED native-lib.cpp)

find_library(log-lib log)

target_link_libraries(
        vlmassistant
        ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/arm64-v8a/libGenie.so
        ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/arm64-v8a/libQnnHtp.so
        android
        ${log-lib})

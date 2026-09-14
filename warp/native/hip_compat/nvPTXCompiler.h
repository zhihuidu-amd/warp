// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// HIP compatibility shim: satisfies `#include <nvPTXCompiler.h>` on ROCm.
//
// The PTX compiler turns PTX into a cubin for a specific NVIDIA architecture.
// There is no ROCm equivalent and none is needed: hiprtc emits a code object
// directly, so the PTX path is never taken on HIP. Warp still needs the
// declarations to compile, so they are provided here and every entry point
// reports failure. If one is ever reached on HIP, it fails loudly rather than
// silently producing an empty module.

#pragma once

#include <cstddef>

typedef enum {
    NVPTXCOMPILE_SUCCESS = 0,
    NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE = 1,
    NVPTXCOMPILE_ERROR_INVALID_INPUT = 2,
    NVPTXCOMPILE_ERROR_COMPILATION_FAILURE = 3,
    NVPTXCOMPILE_ERROR_INTERNAL = 4,
    NVPTXCOMPILE_ERROR_OUT_OF_MEMORY = 5,
    NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE = 6,
    NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION = 7,
} nvPTXCompileResult;

typedef struct nvPTXCompiler* nvPTXCompilerHandle;

static inline nvPTXCompileResult nvPTXCompilerCreate(nvPTXCompilerHandle* compiler, size_t size, const char* ptx)
{
    (void)compiler;
    (void)size;
    (void)ptx;
    return NVPTXCOMPILE_ERROR_INTERNAL;
}

static inline nvPTXCompileResult
nvPTXCompilerCompile(nvPTXCompilerHandle compiler, int num_options, const char* const* options)
{
    (void)compiler;
    (void)num_options;
    (void)options;
    return NVPTXCOMPILE_ERROR_INTERNAL;
}

static inline nvPTXCompileResult nvPTXCompilerGetCompiledProgramSize(nvPTXCompilerHandle compiler, size_t* size)
{
    (void)compiler;
    if (size)
        *size = 0;
    return NVPTXCOMPILE_ERROR_INTERNAL;
}

static inline nvPTXCompileResult nvPTXCompilerGetCompiledProgram(nvPTXCompilerHandle compiler, void* binary)
{
    (void)compiler;
    (void)binary;
    return NVPTXCOMPILE_ERROR_INTERNAL;
}

static inline nvPTXCompileResult nvPTXCompilerDestroy(nvPTXCompilerHandle* compiler)
{
    (void)compiler;
    return NVPTXCOMPILE_ERROR_INTERNAL;
}

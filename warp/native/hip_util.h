#pragma once

#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__)
#error "hip_util.h should only be included for HIP builds."
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "hip_compat/hip_device_compat.h"

#ifndef HIP_VERSION
#if defined(HIP_VERSION_MAJOR) && defined(HIP_VERSION_MINOR) && defined(HIP_VERSION_PATCH)
#define HIP_VERSION (HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 + HIP_VERSION_PATCH)
#else
#define HIP_VERSION 0
#endif  // defined(HIP_VERSION_MAJOR) && defined(HIP_VERSION_MINOR) && defined(HIP_VERSION_PATCH)
#endif  // HIP_VERSION
// CUDA_VERSION serves two DIFFERENT purposes in Warp, and they need different
// values on ROCm.
//
// 1. Compile-time feature gates: ~25 sites of the form
//        #if CUDA_VERSION >= 12080
//    selecting which CUDA driver APIs exist, plus a hard
//        #if CUDA_VERSION < 12000
//        #error Building Warp requires CUDA Toolkit version 12.0 or higher
//    in cuda_util.cpp. HIP provides the modern spellings, so these gates must
//    all be taken. CUDA_VERSION therefore has to compare HIGH.
//
// 2. wp_cuda_toolkit_version() in warp.cu returns CUDA_VERSION to Python, which
//    decodes it as (v // 1000, (v % 1000) // 10) and derives
//    min_driver_version from it, then gates device enumeration on
//        driver_version >= min_driver_version
//    That needs a value in CUDA's own major*1000 + minor*10 scheme.
//
// Aliasing CUDA_VERSION to HIP_VERSION (70125424 on ROCm 7.1) satisfies (1) --
// every gate passes -- but gives (2) a "toolkit" of (70125, 42), so the minimum
// driver became (70125, 0) against a driver reporting about (6, 4). The gate was
// never satisfied, no GPU was registered, and Warp silently reported a CPU-only
// device list after building, linking and importing without a single warning.
//
// Setting CUDA_VERSION to a CUDA-scheme value instead (7020) breaks (1): it is
// below the 12000 minimum, so cuda_util.cpp fails with the #error above and the
// feature gates deselect the APIs HIP actually provides.
//
// So: keep CUDA_VERSION high for the feature gates, and report the version
// separately through WP_HIP_TOOLKIT_VERSION, which wp_cuda_toolkit_version()
// returns instead (see warp.cu). The two concerns are genuinely distinct and
// cannot share one macro.
#ifndef CUDA_VERSION
#define CUDA_VERSION HIP_VERSION
#endif  // CUDA_VERSION

// The ROCm version in CUDA's major*1000 + minor*10 encoding, for the Python
// runtime's version comparison only -- never for a feature gate.
#if defined(HIP_VERSION_MAJOR) && defined(HIP_VERSION_MINOR)
#define WP_HIP_TOOLKIT_VERSION ((HIP_VERSION_MAJOR) * 1000 + (HIP_VERSION_MINOR) * 10)
#else
#define WP_HIP_TOOLKIT_VERSION 0
#endif

// warp.cu returns CUDA_VERSION from wp_cuda_toolkit_version(); on HIP it
// returns WP_HIP_TOOLKIT_VERSION instead, under a one-line guard there.
#ifndef NVRTC_SUCCESS
#define NVRTC_SUCCESS HIPRTC_SUCCESS
#endif  // NVRTC_SUCCESS
#ifndef nvrtcGetErrorString
#define nvrtcGetErrorString hiprtcGetErrorString
#endif  // nvrtcGetErrorString
#ifndef nvrtcCreateProgram
#define nvrtcCreateProgram hiprtcCreateProgram
#endif  // nvrtcCreateProgram
// GPU architecture is an INTEGER in Warp and a STRING on ROCm, and the two do
// not round-trip. Warp stores arch as 10*major + minor and formats
//     --gpu-architecture=sm_%d      (or compute_%d for PTX)
// On gfx942, hipDeviceAttributeComputeCapabilityMajor/Minor report 9 and 4, so
// arch becomes 94 and the trailing "2" is simply gone -- no integer encoding
// recovers "gfx942" from it, and hiprtc rejects "sm_94" ("CUDA kernel build
// failed with error code 6").
//
// Rather than change Warp's integer plumbing, rewrite the option where it
// reaches hiprtc and take the architecture from the device itself, which is
// authoritative and needs no reconstruction. The snprintf format strings are
// literals a macro cannot touch, but nvrtcCompileProgram is already an alias
// here, so the substitution happens on the option array instead.
static inline const char* wp_hip_arch_string()
{
    // Cached: hipGetDeviceProperties is not cheap and this runs per JIT.
    // Sized to hipDeviceProp_t::gcnArchName (256 bytes) -- anything smaller is
    // a -Werror=format-truncation failure, since Warp builds with -Werror.
    static char cached[256] = {0};
    if (cached[0])
        return cached;

    int ordinal = 0;
    if (hipGetDevice(&ordinal) != hipSuccess)
        ordinal = 0;

    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, ordinal) != hipSuccess)
        return "gfx942";  // last resort; hiprtc still reports its own error

    // gcnArchName carries target features, e.g. "gfx942:sramecc+:xnack-".
    // hiprtc accepts the full string and the features affect codegen, so keep
    // them rather than truncating at the colon.
    snprintf(cached, sizeof(cached), "%s", props.gcnArchName);
    return cached;
}

static inline hiprtcResult wp_hiprtcCompileProgram(
    hiprtcProgram prog, int numOptions, const char** options)
{
    // Translate the options Warp builds for NVRTC into the spellings hiprtc
    // accepts. Two need changing:
    //
    //   --gpu-architecture=sm_NN  ->  --offload-arch=gfxNNN[:features]
    //   --include-path=DIR        ->  -IDIR
    //
    // hiprtc does not understand --include-path: it matches the prefix
    // "--include" and then treats the remainder as a filename, failing with
    //     fatal error: '-path=/path/to/warp/native' file not found
    // which surfaces only as HIPRTC_ERROR_COMPILATION at kernel JIT time.
    //
    // Build the translated list as strings first. Every option is copied, so
    // there is no mix of owned and borrowed pointers to keep straight, and the
    // c_str() pointers are only taken once the vector has stopped growing --
    // taking them earlier would dangle on reallocation.
    std::vector<std::string> translated;
    bool have_arch = false;

    translated.reserve(size_t(numOptions > 0 ? numOptions : 0) + 2);

    // Warp's device headers gate several runtime-compilation behaviours on
    // __CUDACC_RTC__, which NVRTC defines and hiprtc does not (it defines
    // __HIPCC_RTC__ instead). Measured on gfx942/ROCm 7.2:
    //
    //   __CUDACC_RTC__  NOT defined      __HIPCC_RTC__   DEFINED
    //   __CUDA__        NOT defined      __clang__       DEFINED
    //
    // so tile.h's `#if WP_ENABLE_CUDA || defined(__CUDACC_RTC__) ||
    // (defined(__clang__) && defined(__CUDA__))` is false under hiprtc and its
    // outer #else hand-defines `struct float4`, colliding with HIP's built-in:
    //   tile.h:33:20: error: definition of type 'float4' conflicts with type alias
    //
    // Defining it is correct rather than expedient -- every behaviour it gates
    // holds for hiprtc as well: float4 is built in (verified by compiling
    // `__device__ float4 x;` with no includes), the target is 64-bit, and
    // volume.h defines the same macro itself to stop PNanoVDB pulling <stdint.h>.
    translated.push_back("-D__CUDACC_RTC__");

    // __CUDACC__ is the macro that makes Warp's builtins DEVICE functions:
    //
    //     #if !defined(__CUDACC__)
    //     #define CUDA_CALLABLE            // host-only
    //     #else
    //     #define CUDA_CALLABLE __host__ __device__
    //
    // hiprtc defines neither __CUDACC__ nor __CUDACC_RTC__, so without this
    // every wp::tid/address/load/add compiled as a host function and the
    // generated kernel failed with
    //     error: reference to __host__ function 'init' in __global__ function
    //     error: no matching function for call to 'tid'
    // and a dozen more -- errors that read like the generated code is wrong
    // when the real cause is that the whole builtin library lost its
    // __device__ annotations.
    //
    // The AOT path already passes -D__CUDACC__ per source in build_dll.py;
    // this is the JIT equivalent. rocThrust also keys on __CUDACC__, but no
    // Thrust code reaches hiprtc -- deterministic.cu is AOT-only and excluded
    // from HIP builds entirely.
    translated.push_back("-D__CUDACC__");

    // With __CUDACC__ defined, crt.h would pull in cuda_crt.h -- NVIDIA's
    // barebones-Clang shim -- which redefines size_t and declares NVVM PTX
    // intrinsics that do not exist on AMD. crt.h now excludes it for HIP, but
    // that header also supplied two things hiprtc does not provide under
    // WP_NO_CRT, so replace them here:
    //
    //   __brkpt()  a CUDA device builtin with no HIP equivalent. The AOT path
    //              already maps it this way in build_dll.py; this is the JIT
    //              equivalent, kept identical on purpose.
    //   assert()   Warp's device code calls it throughout vec.h/mat.h. hiprtc
    //              has no assert.h, and device-side abort is not what these
    //              call sites want in a release build, so compile it out --
    //              matching cuda_crt.h's own NDEBUG behaviour.
    translated.push_back("-D__brkpt()=__builtin_trap()");
    translated.push_back("-Dassert(e)=((void)0)");
    // NOT -Dmemset=__builtin_memset: hiprtc's own hiprtc_runtime.h declares
    // memset/memcpy for device code, and the macro rewrites that declaration
    // into
    //   error: __device__ function '__builtin_memset' cannot overload
    //          __host__ __device__ function '__builtin_memset'
    //
    // The one remaining call site is radix_sort_pairs_cpu_core in
    // tile_radix_sort.h -- a CPU-only helper with no CUDA_CALLABLE that is
    // nevertheless visible to the JIT because builtin.h includes the header
    // unconditionally. NVIDIA never trips over it because cuda_crt.h declares
    // a __device__ memset, so the call resolves even though the function is
    // never called from device code. crt.h supplies the declaration for HIP
    // instead, next to the exclusion that removed it.
    for (int i = 0; i < numOptions; ++i) {
        const char* o = options ? options[i] : nullptr;
        if (!o)
            continue;
        if (strncmp(o, "--gpu-architecture=", 19) == 0 || strncmp(o, "-arch=", 6) == 0) {
            if (!have_arch) {
                translated.push_back(std::string("--offload-arch=") + wp_hip_arch_string());
                have_arch = true;
            }
            continue;  // drop the sm_/compute_ form entirely
        }
        if (strncmp(o, "--include-path=", 15) == 0) {
            translated.push_back(std::string("-I") + (o + 15));
            continue;
        }
        // NVRTC-only options. hiprtc rejects each with "unknown argument" and
        // fails the whole compile, so they are dropped rather than translated.
        // None changes program semantics on this path:
        //
        //   --Ofast-compile=N                 compile-speed/codegen tradeoff
        //   -pch, --pch-dir=                  precompiled headers; hiprtc has
        //                                     no equivalent, only a build-time
        //                                     cost
        //   --fmad=true|false                 fused multiply-add contraction.
        //                                     NOT dropped silently below --
        //                                     see the -ffp-contract mapping
        //   --device-as-default-execution-space
        //   --extra-device-vectorization      optimizer hints
        //   --restrict                        aliasing hint
        //   --diag-suppress=...               suppresses nvcc diagnostic ids
        //                                     that do not exist in clang
        static const char* const nvrtc_only[] = {
            "--Ofast-compile",
            "-pch",
            "--pch-dir=",
            "--device-as-default-execution-space",
            "--extra-device-vectorization",
            "--restrict",
            "--diag-suppress=",
        };
        bool dropped = false;
        for (const char* p : nvrtc_only) {
            if (strncmp(o, p, strlen(p)) == 0) {
                dropped = true;
                break;
            }
        }
        if (dropped)
            continue;

        // --fmad controls FMA contraction, which affects numerical results, so
        // map it rather than drop it. clang spells it -ffp-contract.
        if (strcmp(o, "--fmad=true") == 0) {
            translated.push_back("-ffp-contract=fast");
            continue;
        }
        if (strcmp(o, "--fmad=false") == 0) {
            translated.push_back("-ffp-contract=off");
            continue;
        }

        translated.emplace_back(o);
    }
    if (!have_arch)
        translated.push_back(std::string("--offload-arch=") + wp_hip_arch_string());

    std::vector<const char*> rewritten;
    rewritten.reserve(translated.size());
    for (const std::string& s : translated)
        rewritten.push_back(s.c_str());

    if (getenv("WP_HIP_DEBUG_JIT")) {
        fprintf(stderr, "[wp_hip] hiprtc options:");
        for (const char* o : rewritten)
            fprintf(stderr, " %s", o);
        fprintf(stderr, "\n");
    }

    return hiprtcCompileProgram(prog, int(rewritten.size()), rewritten.data());
}

#ifndef nvrtcCompileProgram
#define nvrtcCompileProgram wp_hiprtcCompileProgram
#endif  // nvrtcCompileProgram
#ifndef nvrtcDestroyProgram
#define nvrtcDestroyProgram hiprtcDestroyProgram
#endif  // nvrtcDestroyProgram
#ifndef nvrtcGetProgramLogSize
#define nvrtcGetProgramLogSize hiprtcGetProgramLogSize
#endif  // nvrtcGetProgramLogSize
#ifndef nvrtcGetProgramLog
#define nvrtcGetProgramLog hiprtcGetProgramLog
#endif  // nvrtcGetProgramLog
#ifndef nvrtcGetPTXSize
#define nvrtcGetPTXSize hiprtcGetCodeSize
#endif  // nvrtcGetPTXSize
#ifndef nvrtcGetPTX
#define nvrtcGetPTX hiprtcGetCode
#endif  // nvrtcGetPTX
#ifndef nvrtcGetCUBINSize
#define nvrtcGetCUBINSize hiprtcGetBitcodeSize
#endif  // nvrtcGetCUBINSize
#ifndef nvrtcGetCUBIN
#define nvrtcGetCUBIN hiprtcGetBitcode
#endif  // nvrtcGetCUBIN
#if defined(nvrtcGetNumSupportedArchs)
#undef nvrtcGetNumSupportedArchs
#endif  // defined(nvrtcGetNumSupportedArchs)
static inline hiprtcResult nvrtcGetNumSupportedArchs(int* count)
{
    if (count) {
        *count = 0;
    }
    return HIPRTC_SUCCESS;
}

#if defined(nvrtcGetSupportedArchs)
#undef nvrtcGetSupportedArchs
#endif  // defined(nvrtcGetSupportedArchs)
static inline hiprtcResult nvrtcGetSupportedArchs(int* archs)
{
    (void)archs;
    return HIPRTC_SUCCESS;
}
#ifndef nvrtcVersion
#define nvrtcVersion hiprtcVersion
#endif  // nvrtcVersion
#ifndef CUDAAPI
#define CUDAAPI
#endif  // CUDAAPI
#if defined(CUDART_CB)
#undef CUDART_CB
#endif  // defined(CUDART_CB)
#define CUDART_CB
#ifndef CU_GET_PROC_ADDRESS_DEFAULT
#define CU_GET_PROC_ADDRESS_DEFAULT 0
#endif  // CU_GET_PROC_ADDRESS_DEFAULT
#ifndef CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE
#define CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE
#endif  // CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE
#ifndef CU_POINTER_ATTRIBUTE_IS_MANAGED
#define CU_POINTER_ATTRIBUTE_IS_MANAGED HIP_POINTER_ATTRIBUTE_IS_MANAGED
#endif  // CU_POINTER_ATTRIBUTE_IS_MANAGED
#ifndef CU_POINTER_ATTRIBUTE_MEMORY_TYPE
#define CU_POINTER_ATTRIBUTE_MEMORY_TYPE HIP_POINTER_ATTRIBUTE_MEMORY_TYPE
#endif  // CU_POINTER_ATTRIBUTE_MEMORY_TYPE
#ifndef CU_IPC_HANDLE_SIZE
#define CU_IPC_HANDLE_SIZE sizeof(CUipcMemHandle)
#endif  // CU_IPC_HANDLE_SIZE
#ifndef CUDA_SUCCESS
#define CUDA_SUCCESS hipSuccess
#endif  // CUDA_SUCCESS
#ifndef cudaErrorInvalidValue
#define cudaErrorInvalidValue hipErrorInvalidValue
#endif  // cudaErrorInvalidValue
// The driver-API spelling. HIP unifies the driver and runtime error enums, so
// both names resolve to the same hipError_t value.
#ifndef CUDA_ERROR_INVALID_VALUE
#define CUDA_ERROR_INVALID_VALUE hipErrorInvalidValue
#endif  // CUDA_ERROR_INVALID_VALUE
// Stream creation flag. The runtime spelling; HIP names it hipStreamNonBlocking.
#ifndef cudaStreamNonBlocking
#define cudaStreamNonBlocking hipStreamNonBlocking
#endif  // cudaStreamNonBlocking
#ifndef cudaSuccess
#define cudaSuccess hipSuccess
#endif  // cudaSuccess
#ifndef cudaGetErrorString
#define cudaGetErrorString hipGetErrorString
#endif  // cudaGetErrorString
#ifndef cudaGetLastError
#define cudaGetLastError hipGetLastError
#endif  // cudaGetLastError
#ifndef cudaDeviceSynchronize
#define cudaDeviceSynchronize hipDeviceSynchronize
#endif  // cudaDeviceSynchronize
#ifndef cudaGetDevice
#define cudaGetDevice hipGetDevice
#endif  // cudaGetDevice
#ifndef cudaGetDeviceCount
#define cudaGetDeviceCount hipGetDeviceCount
#endif  // cudaGetDeviceCount
#ifndef cudaGetDeviceProperties
#define cudaGetDeviceProperties hipGetDeviceProperties
#endif  // cudaGetDeviceProperties
#ifndef cudaDeviceCanAccessPeer
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#endif  // cudaDeviceCanAccessPeer
#ifndef cudaPointerGetAttributes
#define cudaPointerGetAttributes hipPointerGetAttributes
#endif  // cudaPointerGetAttributes
#ifndef cudaMemcpy
#define cudaMemcpy hipMemcpy
#endif  // cudaMemcpy
#ifndef cudaMemcpyAsync
#define cudaMemcpyAsync hipMemcpyAsync
#endif  // cudaMemcpyAsync
#ifndef cudaMemcpyDeviceToDevice
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#endif  // cudaMemcpyDeviceToDevice
#ifndef cudaMemcpyDeviceToHost
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#endif  // cudaMemcpyDeviceToHost
#ifndef cudaMemcpyHostToDevice
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#endif  // cudaMemcpyHostToDevice
#ifndef cudaMemcpyHostToHost
#define cudaMemcpyHostToHost hipMemcpyHostToHost
#endif  // cudaMemcpyHostToHost
#ifndef cudaMemcpyDefault
#define cudaMemcpyDefault hipMemcpyDefault
#endif  // cudaMemcpyDefault
#ifndef cudaMemset
#define cudaMemset hipMemset
#endif  // cudaMemset
#ifndef cudaMemsetAsync
#define cudaMemsetAsync hipMemsetAsync
#endif  // cudaMemsetAsync
#ifndef cudaMalloc
#define cudaMalloc hipMalloc
#endif  // cudaMalloc
#ifndef cudaMallocHost
#define cudaMallocHost hipHostMalloc
#endif  // cudaMallocHost
#ifndef cudaFree
#define cudaFree hipFree
#endif  // cudaFree
#ifndef cudaFreeHost
#define cudaFreeHost hipHostFree
#endif  // cudaFreeHost
#ifndef cudaMallocManaged
#define cudaMallocManaged hipMallocManaged
#endif  // cudaMallocManaged
#ifndef cudaMallocAsync
#define cudaMallocAsync hipMallocAsync
#endif  // cudaMallocAsync
#ifndef cudaFreeAsync
#define cudaFreeAsync hipFreeAsync
#endif  // cudaFreeAsync
#ifndef cudaDeviceGetDefaultMemPool
#define cudaDeviceGetDefaultMemPool hipDeviceGetDefaultMemPool
#endif  // cudaDeviceGetDefaultMemPool
#ifndef cudaMemPoolSetAttribute
#define cudaMemPoolSetAttribute hipMemPoolSetAttribute
#endif  // cudaMemPoolSetAttribute
#ifndef cudaMemPoolGetAttribute
#define cudaMemPoolGetAttribute hipMemPoolGetAttribute
#endif  // cudaMemPoolGetAttribute
#ifndef cudaMemPoolGetAccess
#define cudaMemPoolGetAccess hipMemPoolGetAccess
#endif  // cudaMemPoolGetAccess
#ifndef cudaMemPoolSetAccess
#define cudaMemPoolSetAccess hipMemPoolSetAccess
#endif  // cudaMemPoolSetAccess
#ifndef cudaStreamSynchronize
#define cudaStreamSynchronize hipStreamSynchronize
#endif  // cudaStreamSynchronize
#ifndef cudaStreamIsCapturing
#define cudaStreamIsCapturing hipStreamIsCapturing
#endif  // cudaStreamIsCapturing
#ifndef cudaStreamBeginCapture
#define cudaStreamBeginCapture hipStreamBeginCapture
#endif  // cudaStreamBeginCapture
#ifndef cudaStreamBeginCaptureToGraph
#define cudaStreamBeginCaptureToGraph hipStreamBeginCaptureToGraph
#endif  // cudaStreamBeginCaptureToGraph
#ifndef cudaStreamEndCapture
#define cudaStreamEndCapture hipStreamEndCapture
#endif  // cudaStreamEndCapture
#ifndef cudaEventCreate
#define cudaEventCreate hipEventCreate
#endif  // cudaEventCreate
#ifndef cudaEventRecord
#define cudaEventRecord hipEventRecord
#endif  // cudaEventRecord
#ifndef cudaEventSynchronize
#define cudaEventSynchronize hipEventSynchronize
#endif  // cudaEventSynchronize
#ifndef cudaEventElapsedTime
#define cudaEventElapsedTime hipEventElapsedTime
#endif  // cudaEventElapsedTime
#ifndef cudaEventDestroy
#define cudaEventDestroy hipEventDestroy
#endif  // cudaEventDestroy
#ifndef cudaMemAdvise
#define cudaMemAdvise hipMemAdvise
#endif  // cudaMemAdvise
#ifndef cudaMemPrefetchAsync
#define cudaMemPrefetchAsync hipMemPrefetchAsync
#endif  // cudaMemPrefetchAsync
#ifndef cudaFuncSetAttribute
#define cudaFuncSetAttribute hipFuncSetAttribute
#endif  // cudaFuncSetAttribute
#ifndef cudaFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#endif  // cudaFuncAttributeMaxDynamicSharedMemorySize
// HIP has TWO function-attribute enums and they are not interchangeable:
//   hipFunction_attribute  (HIP_FUNC_ATTRIBUTE_*)  -- driver API, get and set
//   hipFuncAttribute       (hipFuncAttribute*)     -- runtime hipFuncSetAttribute
// Warp declares both cuFuncSetAttribute_f and cuFuncGetAttribute_f as taking
// CUfunction_attribute, which aliases the driver enum, so this must use the
// HIP_FUNC_ATTRIBUTE_ spelling. Naming the runtime enumerator here made the
// setter call fail to resolve. wp_hipFuncSetAttribute below converts to the
// runtime enum at the boundary.
#ifndef CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
#define CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
#endif  // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
// Queried through cuFuncGetAttribute_f, which takes a CUfunction_attribute
// (aliased to hipFunction_attribute) -- these are the HIP_FUNC_ATTRIBUTE_*
// enumerators, not the hipFuncAttribute* set used by the setter above.
#ifndef CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
#define CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
#endif  // CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
#ifndef CU_FUNC_ATTRIBUTE_NUM_REGS
#define CU_FUNC_ATTRIBUTE_NUM_REGS HIP_FUNC_ATTRIBUTE_NUM_REGS
#endif  // CU_FUNC_ATTRIBUTE_NUM_REGS
#ifndef CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
#define CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
#endif  // CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
// JIT options passed to cuModuleLoadDataEx_f, typed CUjit_option (hipJitOption).
#ifndef CU_JIT_ERROR_LOG_BUFFER
#define CU_JIT_ERROR_LOG_BUFFER hipJitOptionErrorLogBuffer
#endif  // CU_JIT_ERROR_LOG_BUFFER
#ifndef CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
#define CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES hipJitOptionErrorLogBufferSizeBytes
#endif  // CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
#ifndef cudaCpuDeviceId
#define cudaCpuDeviceId hipCpuDeviceId
#endif  // cudaCpuDeviceId
#ifndef cudaInvalidDeviceId
#define cudaInvalidDeviceId hipInvalidDeviceId
#endif  // cudaInvalidDeviceId
#ifndef cudaGraphDestroy
#define cudaGraphDestroy hipGraphDestroy
#endif  // cudaGraphDestroy
#ifndef cudaGraphExecDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy
#endif  // cudaGraphExecDestroy
#ifndef cudaGraphAddMemFreeNode
#define cudaGraphAddMemFreeNode hipGraphAddMemFreeNode
#endif  // cudaGraphAddMemFreeNode
#ifndef cudaGraphAddMemcpyNode1D
#define cudaGraphAddMemcpyNode1D hipGraphAddMemcpyNode1D
#endif  // cudaGraphAddMemcpyNode1D

#ifndef CU_STREAM_ADD_CAPTURE_DEPENDENCIES
#define CU_STREAM_ADD_CAPTURE_DEPENDENCIES 0
#endif  // CU_STREAM_ADD_CAPTURE_DEPENDENCIES
#ifndef CU_STREAM_CAPTURE_STATUS_NONE
#define CU_STREAM_CAPTURE_STATUS_NONE hipStreamCaptureStatusNone
#endif  // CU_STREAM_CAPTURE_STATUS_NONE
#ifndef CU_STREAM_CAPTURE_STATUS_ACTIVE
#define CU_STREAM_CAPTURE_STATUS_ACTIVE hipStreamCaptureStatusActive
#endif  // CU_STREAM_CAPTURE_STATUS_ACTIVE
#ifndef cudaGraphExecMemcpyNodeSetParams1D
#define cudaGraphExecMemcpyNodeSetParams1D hipGraphExecMemcpyNodeSetParams1D
#endif  // cudaGraphExecMemcpyNodeSetParams1D
#ifndef cudaGraphInstantiateWithFlags
#define cudaGraphInstantiateWithFlags hipGraphInstantiateWithFlags
#endif  // cudaGraphInstantiateWithFlags
#ifndef cudaGraphUpload
#define cudaGraphUpload hipGraphUpload
#endif  // cudaGraphUpload
#ifndef cudaGraphLaunch
#define cudaGraphLaunch hipGraphLaunch
#endif  // cudaGraphLaunch
#ifndef cudaGraphGetNodes
#define cudaGraphGetNodes hipGraphGetNodes
#endif  // cudaGraphGetNodes
#ifndef cudaGraphChildGraphNodeGetGraph
#define cudaGraphChildGraphNodeGetGraph hipGraphChildGraphNodeGetGraph
#endif  // cudaGraphChildGraphNodeGetGraph
#ifndef cudaGraphAddChildGraphNode
#define cudaGraphAddChildGraphNode hipGraphAddChildGraphNode
#endif  // cudaGraphAddChildGraphNode
#ifndef cudaGraphDebugDotPrint
#define cudaGraphDebugDotPrint hipGraphDebugDotPrint
#endif  // cudaGraphDebugDotPrint
#ifndef cudaUserObjectCreate
#define cudaUserObjectCreate hipUserObjectCreate
#endif  // cudaUserObjectCreate
#ifndef cudaGraphRetainUserObject
#define cudaGraphRetainUserObject hipGraphRetainUserObject
#endif  // cudaGraphRetainUserObject
#ifndef cudaStreamSetCaptureDependencies
#define cudaStreamSetCaptureDependencies hipStreamSetCaptureDependencies
#endif  // cudaStreamSetCaptureDependencies

#ifndef cudaGraphInstantiateFlagAutoFreeOnLaunch
#define cudaGraphInstantiateFlagAutoFreeOnLaunch hipGraphInstantiateFlagAutoFreeOnLaunch
#endif  // cudaGraphInstantiateFlagAutoFreeOnLaunch
#ifndef cudaGraphUserObjectMove
#define cudaGraphUserObjectMove hipGraphUserObjectMove
#endif  // cudaGraphUserObjectMove
#ifndef cudaUserObjectNoDestructorSync
#define cudaUserObjectNoDestructorSync hipUserObjectNoDestructorSync
#endif  // cudaUserObjectNoDestructorSync
#ifndef cudaMemPoolAttrReleaseThreshold
#define cudaMemPoolAttrReleaseThreshold hipMemPoolAttrReleaseThreshold
#endif  // cudaMemPoolAttrReleaseThreshold
#ifndef cudaMemPoolAttrUsedMemCurrent
#define cudaMemPoolAttrUsedMemCurrent hipMemPoolAttrUsedMemCurrent
#endif  // cudaMemPoolAttrUsedMemCurrent
#ifndef cudaMemPoolAttrUsedMemHigh
#define cudaMemPoolAttrUsedMemHigh hipMemPoolAttrUsedMemHigh
#endif  // cudaMemPoolAttrUsedMemHigh
#ifndef cudaMemAccessFlagsProtNone
#define cudaMemAccessFlagsProtNone hipMemAccessFlagsProtNone
#endif  // cudaMemAccessFlagsProtNone
#ifndef cudaMemAccessFlagsProtReadWrite
#define cudaMemAccessFlagsProtReadWrite hipMemAccessFlagsProtReadWrite
#endif  // cudaMemAccessFlagsProtReadWrite
#ifndef cudaMemLocationTypeDevice
#define cudaMemLocationTypeDevice hipMemLocationTypeDevice
#endif  // cudaMemLocationTypeDevice
#ifndef cudaStreamCaptureStatusNone
#define cudaStreamCaptureStatusNone hipStreamCaptureStatusNone
#endif  // cudaStreamCaptureStatusNone
#ifndef cudaStreamCaptureStatusActive
#define cudaStreamCaptureStatusActive hipStreamCaptureStatusActive
#endif  // cudaStreamCaptureStatusActive
#ifndef cudaStreamCaptureModeThreadLocal
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#endif  // cudaStreamCaptureModeThreadLocal

#ifndef CUDA_ERROR_NOT_INITIALIZED
#define CUDA_ERROR_NOT_INITIALIZED hipErrorNotInitialized
#endif  // CUDA_ERROR_NOT_INITIALIZED
#ifndef CUDA_ERROR_NOT_READY
#define CUDA_ERROR_NOT_READY hipErrorNotReady
#endif  // CUDA_ERROR_NOT_READY
#ifndef CUDA_ERROR_NOT_SUPPORTED
#define CUDA_ERROR_NOT_SUPPORTED hipErrorNotSupported
#endif  // CUDA_ERROR_NOT_SUPPORTED
#ifndef CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED
#define CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED hipErrorPeerAccessAlreadyEnabled
#endif  // CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED
#ifndef CUDA_ERROR_PEER_ACCESS_NOT_ENABLED
#define CUDA_ERROR_PEER_ACCESS_NOT_ENABLED hipErrorPeerAccessNotEnabled
#endif  // CUDA_ERROR_PEER_ACCESS_NOT_ENABLED
// ROCm has no equivalent of cudaErrorCallRequiresNewerDriver. Map it to the
// generic "not supported" status so comparisons against it stay well-formed
// and simply never match.
#ifndef cudaErrorCallRequiresNewerDriver
#define cudaErrorCallRequiresNewerDriver hipErrorNotSupported
#endif  // cudaErrorCallRequiresNewerDriver
#ifndef CU_STREAM_DEFAULT
#define CU_STREAM_DEFAULT hipStreamDefault
#endif  // CU_STREAM_DEFAULT
#ifndef CU_EVENT_DEFAULT
#define CU_EVENT_DEFAULT hipEventDefault
#endif  // CU_EVENT_DEFAULT
#ifndef CU_EVENT_DISABLE_TIMING
#define CU_EVENT_DISABLE_TIMING hipEventDisableTiming
#endif  // CU_EVENT_DISABLE_TIMING
#ifndef CU_EVENT_RECORD_DEFAULT
#define CU_EVENT_RECORD_DEFAULT 0
#endif  // CU_EVENT_RECORD_DEFAULT
#ifndef CU_EVENT_WAIT_DEFAULT
#define CU_EVENT_WAIT_DEFAULT 0
#endif  // CU_EVENT_WAIT_DEFAULT
#ifndef CU_EVENT_RECORD_EXTERNAL
#define CU_EVENT_RECORD_EXTERNAL 0
#endif  // CU_EVENT_RECORD_EXTERNAL
#ifndef CU_EVENT_WAIT_EXTERNAL
#define CU_EVENT_WAIT_EXTERNAL 0
#endif  // CU_EVENT_WAIT_EXTERNAL
#ifndef CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
#define CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS 0
#endif  // CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
#ifndef WP_HAS_MEMCPY_BATCH
#define WP_HAS_MEMCPY_BATCH (HIP_VERSION >= 70100000)
#endif  // WP_HAS_MEMCPY_BATCH

#ifndef CU_MEMCPY_SRC_ACCESS_ORDER_STREAM
#if WP_HAS_MEMCPY_BATCH
#define CU_MEMCPY_SRC_ACCESS_ORDER_STREAM hipMemcpySrcAccessOrderStream
#else
#define CU_MEMCPY_SRC_ACCESS_ORDER_STREAM 0
#endif  // WP_HAS_MEMCPY_BATCH
#endif  // CU_MEMCPY_SRC_ACCESS_ORDER_STREAM

using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
using cudaEvent_t = hipEvent_t;
using cudaDeviceProp = hipDeviceProp_t;
using cudaPointerAttributes = hipPointerAttribute_t;
using cudaMemcpyKind = hipMemcpyKind;
using cudaMemoryAdvise = hipMemoryAdvise;
using cudaStreamCaptureStatus = hipStreamCaptureStatus;
using cudaStreamCaptureMode = hipStreamCaptureMode;
using cudaGraph_t = hipGraph_t;
using cudaGraphNode_t = hipGraphNode_t;
using cudaGraphExec_t = hipGraphExec_t;
using cudaMemPool_t = hipMemPool_t;
using cudaMemAccessFlags = hipMemAccessFlags;
using cudaMemLocation = hipMemLocation;
using cudaMemAccessDesc = hipMemAccessDesc;
using cudaUserObject_t = hipUserObject_t;
using cudaResourceDesc = hipResourceDesc;
using cudaArray_t = hipArray_t;

// ---------------------------------------------------------------------------
// Opaque handles: integer in CUDA, pointer in HIP.
//
// CUDA types CUdeviceptr, CUtexObject and cudaSurfaceObject_t as unsigned
// integers, so Warp moves them through uint64_t with static_cast. HIP types
// them as pointers, where that cast is ill-formed. These wrappers hold the HIP
// pointer but convert to and from integers, so the existing casts stay valid
// and no call site changes.
// ---------------------------------------------------------------------------
template <typename T>
struct wp_hip_handle {
    T value{};

    wp_hip_handle() = default;
    // NOLINTNEXTLINE(google-explicit-constructor)
    constexpr wp_hip_handle(T v) : value(v) {}
    // NOLINTNEXTLINE(google-explicit-constructor)
    explicit constexpr wp_hip_handle(unsigned long long v) : value(reinterpret_cast<T>(v)) {}

    constexpr operator T() const { return value; }
    constexpr operator unsigned long long() const
    {
        return reinterpret_cast<unsigned long long>(value);
    }
    constexpr operator unsigned long() const
    {
        return reinterpret_cast<unsigned long>(value);
    }
    constexpr bool operator==(const wp_hip_handle& o) const { return value == o.value; }
    constexpr bool operator!=(const wp_hip_handle& o) const { return value != o.value; }
};

using cudaSurfaceObject_t = wp_hip_handle<hipSurfaceObject_t>;
#ifndef cudaResourceTypeArray
#define cudaResourceTypeArray hipResourceTypeArray
#endif  // cudaResourceTypeArray
#ifndef cudaCreateSurfaceObject
#define cudaCreateSurfaceObject wp_hipCreateSurfaceObject
#endif  // cudaCreateSurfaceObject
#ifndef cudaDestroySurfaceObject
#define cudaDestroySurfaceObject hipDestroySurfaceObject
#endif  // cudaDestroySurfaceObject
using nvrtcProgram = hiprtcProgram;
using nvrtcResult = hiprtcResult;
using CUresult = hipError_t;
using CUdevice = hipDevice_t;
// HIP has a real context type (hipCtx_t). Aliasing CUcontext to a stand-in
// struct would compile but then fail to convert wherever the driver entry
// points expect the genuine type.
using CUcontext = hipCtx_t;
using CUstream = hipStream_t;
using CUevent = hipEvent_t;
using CUmodule = hipModule_t;
using CUfunction = hipFunction_t;
// Deliberately an integer, exactly as CUDA declares it -- NOT a wp_hip_handle.
//
// The wrapper approach fails here. Warp writes reinterpret_cast<CUdeviceptr>(p)
// in warp.cu, and reinterpret_cast to a class type is ill-formed no matter what
// converting constructors that class provides; a user-defined conversion is
// never considered. Those call sites are upstream files this backend must not
// modify, so the type itself has to accept the cast.
//
// An integer satisfies every use: reinterpret_cast from void* and static_cast
// from uint64_t both work, and the assignment back out to uint64_t is implicit.
// The only places needing care are the driver entry points that take a
// CUdeviceptr* out-parameter, where HIP expects void**; each is bridged by a
// wp_hip* wrapper below that reinterprets the pointer at the boundary. That is
// sound because hipDeviceptr_t is void* and this type is the same width.
using CUdeviceptr = unsigned long long;
static_assert(
    sizeof(CUdeviceptr) == sizeof(hipDeviceptr_t),
    "CUdeviceptr must be pointer-width: the wrappers below reinterpret "
    "CUdeviceptr* as void** for HIP's out-parameters.");
using CUuuid = hipUUID;
using CUdevice_attribute = hipDeviceAttribute_t;
using CUipcEventHandle = hipIpcEventHandle_t;
using CUipcMemHandle = hipIpcMemHandle_t;
using cuuint64_t = uint64_t;
using CUgraphicsResource = hipGraphicsResource_t;
using CUarray = hipArray_t;
using CUtexObject = wp_hip_handle<hipTextureObject_t>;
using CUgraph = hipGraph_t;
using CUgraphExec = hipGraphExec_t;
using CUgraphNode = hipGraphNode_t;
using CUgraphNodeType = hipGraphNodeType;
// hipGraphNodeParams has no `conditional` member because ROCm has no
// conditional node type. Wrap it so the conditional-node code in warp.cu still
// compiles; that code is unreachable on HIP (see the conditional graph node
// note below), and cuGraphAddNode_f is given the HIP-native part.
using cudaGraphConditionalHandle = unsigned long long;

struct wp_hip_conditional_node_params {
    cudaGraphConditionalHandle handle;
    int type;
    unsigned int size;
    hipCtx_t ctx;
    hipGraph_t phGraph_out[2];
};

struct wp_hip_graph_node_params : hipGraphNodeParams {
    wp_hip_conditional_node_params conditional;
};

using CUgraphNodeParams = wp_hip_graph_node_params;
using CUgraphEdgeData = void;
using CUstreamCaptureStatus = hipStreamCaptureStatus;
using CUjit_option = hipJitOption;
using CUpointer_attribute = hipPointer_attribute;
// hipFunction_attribute is the query enum used by hipFuncGetAttribute;
// hipFuncAttribute is the distinct enum used when setting attributes.
using CUfunction_attribute = hipFunction_attribute;
// HIP has no equivalent of the CUDA `CUoccupancyB2DSize` callback (the only HIP
// equivalents take a fixed `size_t` shared-memory size). Provide a stub typedef
// so call sites that reference the type compile under HIP; HIP code paths must
// ignore the callback (see cuOccupancyMaxPotentialBlockSize_f).
typedef size_t (CUDAAPI* CUoccupancyB2DSize)(int blockSize);
#if HIP_VERSION >= 70100000
using CUmemcpyAttributes = hipMemcpyAttributes;
#else
struct CUmemcpyAttributes {
    int dummy;
};
#endif  // HIP_VERSION >= 70100000
using CUDA_ARRAY_DESCRIPTOR = HIP_ARRAY_DESCRIPTOR;
using CUDA_ARRAY3D_DESCRIPTOR = HIP_ARRAY3D_DESCRIPTOR;
using CUmipmappedArray = hipmipmappedArray;
// Thread-block clusters are a Hopper+ CUDA feature with no AMD equivalent, and
// the CUDA and HIP launch configs differ in shape: CUDA spells the grid as flat
// gridDimX/Y/Z fields and carries a clusterDim attribute, while HIP uses a dim3
// and has no cluster member at all.
//
// warp.cu probes the cluster limit with cuOccupancyMaxActiveClusters, which
// hip_util.h already types as a stub because ROCm has no such entry point. These
// CUDA-shaped definitions let that probe compile; it reports no cluster support
// on AMD, which is correct.
struct wp_hip_cluster_dim {
    unsigned int x, y, z;
};

struct wp_hip_launch_attribute_value {
    wp_hip_cluster_dim clusterDim;
};

struct wp_hip_launch_attribute {
    int id;
    wp_hip_launch_attribute_value value;
};

struct wp_hip_launch_config {
    unsigned int gridDimX, gridDimY, gridDimZ;
    unsigned int blockDimX, blockDimY, blockDimZ;
    unsigned int sharedMemBytes;
    hipStream_t hStream;
    wp_hip_launch_attribute* attrs;
    unsigned int numAttrs;
};

using CUlaunchAttribute = wp_hip_launch_attribute;
using CUlaunchConfig = wp_hip_launch_config;

#ifndef CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
#define CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION 0
#endif  // CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
#ifndef cudaStreamGetId
#define cudaStreamGetId hipStreamGetId
#endif  // cudaStreamGetId

// CUDA_MEMCPY3D is NOT a plain alias to HIP_MEMCPY3D: its two device-pointer
// fields have to hold a CUdeviceptr, which is an integer here (see the
// CUdeviceptr definition above). texture.cpp writes
//     copy_params.dstDevice = static_cast<CUdeviceptr>(handle);
// and assigning an integer to HIP's void* field is invalid ("invalid conversion
// from 'long long unsigned int' to 'void*'"). Mirroring the layout with integer
// fields keeps that assignment legal; wp_hipDrvMemcpy3D* converts to the real
// HIP struct at the call boundary.
//
// Defined once, outside the HIP_VERSION split below, so the two branches cannot
// drift apart. Field names and semantics follow HIP_MEMCPY3D exactly; if ROCm
// adds a field Warp starts using, this has to be extended.
struct wp_hip_memcpy3d {
    size_t srcXInBytes{}, srcY{}, srcZ{};
    size_t srcLOD{};
    hipMemoryType srcMemoryType{};
    const void* srcHost{};
    CUdeviceptr srcDevice{};
    hipArray_t srcArray{};
    void* reserved0{};
    size_t srcPitch{}, srcHeight{};

    size_t dstXInBytes{}, dstY{}, dstZ{};
    size_t dstLOD{};
    hipMemoryType dstMemoryType{};
    void* dstHost{};
    CUdeviceptr dstDevice{};
    hipArray_t dstArray{};
    void* reserved1{};
    size_t dstPitch{}, dstHeight{};

    size_t WidthInBytes{}, Height{}, Depth{};

    HIP_MEMCPY3D to_hip() const
    {
        HIP_MEMCPY3D p{};
        p.srcXInBytes = srcXInBytes; p.srcY = srcY; p.srcZ = srcZ;
        p.srcLOD = srcLOD;
        p.srcMemoryType = srcMemoryType;
        p.srcHost = srcHost;
        p.srcDevice = reinterpret_cast<hipDeviceptr_t>(srcDevice);
        p.srcArray = srcArray;
        p.srcPitch = srcPitch; p.srcHeight = srcHeight;

        p.dstXInBytes = dstXInBytes; p.dstY = dstY; p.dstZ = dstZ;
        p.dstLOD = dstLOD;
        p.dstMemoryType = dstMemoryType;
        p.dstHost = dstHost;
        p.dstDevice = reinterpret_cast<hipDeviceptr_t>(dstDevice);
        p.dstArray = dstArray;
        p.dstPitch = dstPitch; p.dstHeight = dstHeight;

        p.WidthInBytes = WidthInBytes; p.Height = Height; p.Depth = Depth;
        return p;
    }
};
using CUDA_MEMCPY3D = wp_hip_memcpy3d;

#if HIP_VERSION >= 70000000
using CUDA_MEMCPY2D = hip_Memcpy2D;
using CUDA_RESOURCE_DESC = HIP_RESOURCE_DESC;
using CUDA_TEXTURE_DESC = HIP_TEXTURE_DESC;
using CUDA_RESOURCE_VIEW_DESC = HIP_RESOURCE_VIEW_DESC;

using CUarray_format = hipArray_Format;
using CUaddress_mode = HIPaddress_mode;

#ifndef CU_AD_FORMAT_UNSIGNED_INT8
#define CU_AD_FORMAT_UNSIGNED_INT8 HIP_AD_FORMAT_UNSIGNED_INT8
#endif  // CU_AD_FORMAT_UNSIGNED_INT8
#ifndef CU_AD_FORMAT_UNSIGNED_INT16
#define CU_AD_FORMAT_UNSIGNED_INT16 HIP_AD_FORMAT_UNSIGNED_INT16
#endif  // CU_AD_FORMAT_UNSIGNED_INT16
#ifndef CU_AD_FORMAT_UNSIGNED_INT32
#define CU_AD_FORMAT_UNSIGNED_INT32 HIP_AD_FORMAT_UNSIGNED_INT32
#endif  // CU_AD_FORMAT_UNSIGNED_INT32
#ifndef CU_AD_FORMAT_SIGNED_INT8
#define CU_AD_FORMAT_SIGNED_INT8 HIP_AD_FORMAT_SIGNED_INT8
#endif  // CU_AD_FORMAT_SIGNED_INT8
#ifndef CU_AD_FORMAT_SIGNED_INT16
#define CU_AD_FORMAT_SIGNED_INT16 HIP_AD_FORMAT_SIGNED_INT16
#endif  // CU_AD_FORMAT_SIGNED_INT16
#ifndef CU_AD_FORMAT_SIGNED_INT32
#define CU_AD_FORMAT_SIGNED_INT32 HIP_AD_FORMAT_SIGNED_INT32
#endif  // CU_AD_FORMAT_SIGNED_INT32
#ifndef CU_AD_FORMAT_HALF
#define CU_AD_FORMAT_HALF HIP_AD_FORMAT_HALF
#endif  // CU_AD_FORMAT_HALF
#ifndef CU_AD_FORMAT_FLOAT
#define CU_AD_FORMAT_FLOAT HIP_AD_FORMAT_FLOAT
#endif  // CU_AD_FORMAT_FLOAT

#ifndef CU_TR_ADDRESS_MODE_WRAP
#define CU_TR_ADDRESS_MODE_WRAP HIP_TR_ADDRESS_MODE_WRAP
#endif  // CU_TR_ADDRESS_MODE_WRAP
#ifndef CU_TR_ADDRESS_MODE_CLAMP
#define CU_TR_ADDRESS_MODE_CLAMP HIP_TR_ADDRESS_MODE_CLAMP
#endif  // CU_TR_ADDRESS_MODE_CLAMP
#ifndef CU_TR_ADDRESS_MODE_MIRROR
#define CU_TR_ADDRESS_MODE_MIRROR HIP_TR_ADDRESS_MODE_MIRROR
#endif  // CU_TR_ADDRESS_MODE_MIRROR
#ifndef CU_TR_ADDRESS_MODE_BORDER
#define CU_TR_ADDRESS_MODE_BORDER HIP_TR_ADDRESS_MODE_BORDER
#endif  // CU_TR_ADDRESS_MODE_BORDER

#ifndef CU_TR_FILTER_MODE_POINT
#define CU_TR_FILTER_MODE_POINT HIP_TR_FILTER_MODE_POINT
#endif  // CU_TR_FILTER_MODE_POINT
#ifndef CU_TR_FILTER_MODE_LINEAR
#define CU_TR_FILTER_MODE_LINEAR HIP_TR_FILTER_MODE_LINEAR
#endif  // CU_TR_FILTER_MODE_LINEAR

#ifndef CU_TRSF_NORMALIZED_COORDINATES
#define CU_TRSF_NORMALIZED_COORDINATES HIP_TRSF_NORMALIZED_COORDINATES
#endif  // CU_TRSF_NORMALIZED_COORDINATES

#ifndef CU_RESOURCE_TYPE_ARRAY
#define CU_RESOURCE_TYPE_ARRAY HIP_RESOURCE_TYPE_ARRAY
#endif  // CU_RESOURCE_TYPE_ARRAY

#ifndef CU_MEMORYTYPE_HOST
#define CU_MEMORYTYPE_HOST hipMemoryTypeHost
#endif  // CU_MEMORYTYPE_HOST
#ifndef CU_MEMORYTYPE_DEVICE
#define CU_MEMORYTYPE_DEVICE hipMemoryTypeDevice
#endif  // CU_MEMORYTYPE_DEVICE
#ifndef CU_MEMORYTYPE_ARRAY
#define CU_MEMORYTYPE_ARRAY hipMemoryTypeArray
#endif  // CU_MEMORYTYPE_ARRAY
using CUmemorytype = hipMemoryType;
// Distinct from CUmemorytype above despite the near-identical spelling: this is
// the memory-pool handle returned by CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE, not
// the host/device enum. clang suggests CUmemorytype for the missing name, which
// would compile and then compare a pool handle as if it were a memory kind.
using CUmemoryPool = hipMemPool_t;
#ifndef CUDA_ARRAY3D_SURFACE_LDST
#define CUDA_ARRAY3D_SURFACE_LDST hipArraySurfaceLoadStore
#endif  // CUDA_ARRAY3D_SURFACE_LDST
#else
using CUDA_MEMCPY2D = HIP_MEMCPY2D;
using CUDA_RESOURCE_DESC = hipResourceDesc;
using CUDA_TEXTURE_DESC = hipTextureDesc;
using CUDA_RESOURCE_VIEW_DESC = hipResourceViewDesc;
#endif  // HIP_VERSION >= 70000000

#ifndef CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID
#define CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID hipDeviceAttributePciDomainId
#endif  // CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID
#ifndef CU_DEVICE_ATTRIBUTE_PCI_BUS_ID
#define CU_DEVICE_ATTRIBUTE_PCI_BUS_ID hipDeviceAttributePciBusId
#endif  // CU_DEVICE_ATTRIBUTE_PCI_BUS_ID
#ifndef CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID
#define CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID hipDeviceAttributePciDeviceId
#endif  // CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID
#ifndef CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING
#define CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING hipDeviceAttributeUnifiedAddressing
#endif  // CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING
#ifndef CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED
#define CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED hipDeviceAttributeMemoryPoolsSupported
#endif  // CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED
#ifndef CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT hipDeviceAttributeMultiprocessorCount
#endif  // CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
#ifndef CU_DEVICE_ATTRIBUTE_INTEGRATED
#define CU_DEVICE_ATTRIBUTE_INTEGRATED hipDeviceAttributeIntegrated
#endif  // CU_DEVICE_ATTRIBUTE_INTEGRATED
#ifndef CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
#define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN hipDeviceAttributeSharedMemPerBlockOptin
#endif  // CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
#ifndef CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR hipDeviceAttributeComputeCapabilityMajor
#endif  // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
#ifndef CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR hipDeviceAttributeComputeCapabilityMinor
#endif  // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
#ifndef CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED
#define CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED ((CUdevice_attribute)-1)
#endif  // CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED


// The driver entry points below take these handles by pointer. A
// wp_hip_handle<T> is a standard-layout struct whose only member is a T, so a
// wp_hip_handle<T>* is layout-compatible with a T* and the reinterpret_cast is
// well-defined. Wrapping here keeps the casts out of the call sites.
static inline hipError_t wp_hipCreateSurfaceObject(
    wp_hip_handle<hipSurfaceObject_t>* pSurfObject, const hipResourceDesc* pResDesc)
{
    return hipCreateSurfaceObject(
        reinterpret_cast<hipSurfaceObject_t*>(pSurfObject), pResDesc);
}

static inline hipError_t wp_hipMemcpyBatchAsync(
    CUdeviceptr* dsts, CUdeviceptr* srcs, size_t* sizes,
    size_t count, hipMemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs, size_t* failIdx,
    hipStream_t stream)
{
    return hipMemcpyBatchAsync(
        reinterpret_cast<void**>(dsts), reinterpret_cast<void**>(srcs), sizes, count, attrs,
        attrsIdxs, numAttrs, failIdx, stream);
}

static inline hipError_t wp_hipGraphicsResourceGetMappedPointer(
    CUdeviceptr* pDevPtr, size_t* pSize, hipGraphicsResource_t resource)
{
    return hipGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(pDevPtr), pSize, resource);
}

static inline hipError_t wp_hipModuleGetGlobal(
    CUdeviceptr* dptr, size_t* bytes, hipModule_t hmod, const char* name)
{
    return hipModuleGetGlobal(reinterpret_cast<hipDeviceptr_t*>(dptr), bytes, hmod, name);
}

static inline hipError_t wp_hipIpcOpenMemHandle(
    CUdeviceptr* pdptr, hipIpcMemHandle_t handle, unsigned int flags)
{
    return hipIpcOpenMemHandle(reinterpret_cast<void**>(pdptr), handle, flags);
}

// These three take the device pointer BY VALUE. CUdeviceptr is an integer (see
// its definition above), so they cannot alias the HIP entry points directly the
// way the out-parameter forms do -- the integer will not convert to void*.
static inline hipError_t wp_hipPointerGetAttribute(
    void* data, hipPointer_attribute attribute, CUdeviceptr ptr)
{
    return hipPointerGetAttribute(data, attribute, reinterpret_cast<hipDeviceptr_t>(ptr));
}

static inline hipError_t wp_hipIpcGetMemHandle(hipIpcMemHandle_t* handle, CUdeviceptr devPtr)
{
    return hipIpcGetMemHandle(handle, reinterpret_cast<void*>(devPtr));
}

static inline hipError_t wp_hipIpcCloseMemHandle(CUdeviceptr devPtr)
{
    return hipIpcCloseMemHandle(reinterpret_cast<void*>(devPtr));
}

static inline hipError_t wp_hipTexObjectCreate(
    wp_hip_handle<hipTextureObject_t>* pTexObject, const HIP_RESOURCE_DESC* pResDesc,
    const HIP_TEXTURE_DESC* pTexDesc, const HIP_RESOURCE_VIEW_DESC* pResViewDesc)
{
    return hipTexObjectCreate(
        reinterpret_cast<hipTextureObject_t*>(pTexObject), pResDesc, pTexDesc, pResViewDesc);
}

// ---------------------------------------------------------------------------
// Versioned driver-API function pointer types.
//
// Warp declares its driver entry points using the explicitly versioned names
// from CUDA's cudaTypedefs.h (PFN_cuFoo_vNNNN). ROCm has no equivalent header,
// so each name is synthesised from the HIP function it already aliases to.
// decltype keeps the signature in step with ROCm automatically: if HIP changes
// one, this stops compiling rather than silently mismatching.
// ---------------------------------------------------------------------------
// Several HIP entry points that Warp resolves dynamically are marked
// deprecated (the context API, the profiler controls). Warp builds with
// -Werror, so suppress the diagnostic around these declarations: taking a
// function's address to type a driver entry point is not a use of it.
#if defined(__GNUC__) || defined(__clang__)
#define WP_HIP_PFN(_fn, _pfn)                                        _Pragma("GCC diagnostic push")                                   _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"")     using _pfn = decltype(&_fn);                                     _Pragma("GCC diagnostic pop")
#else
#define WP_HIP_PFN(_fn, _pfn) using _pfn = decltype(&_fn)
#endif

WP_HIP_PFN(hipArray3DCreate, PFN_cuArray3DCreate_v3020);
WP_HIP_PFN(hipArray3DGetDescriptor, PFN_cuArray3DGetDescriptor_v3020);
WP_HIP_PFN(hipArrayCreate, PFN_cuArrayCreate_v3020);
WP_HIP_PFN(hipArrayDestroy, PFN_cuArrayDestroy_v2000);
WP_HIP_PFN(hipCtxCreate, PFN_cuCtxCreate_v3020);
WP_HIP_PFN(hipCtxDestroy, PFN_cuCtxDestroy_v4000);
WP_HIP_PFN(hipCtxDisablePeerAccess, PFN_cuCtxDisablePeerAccess_v4000);
WP_HIP_PFN(hipCtxEnablePeerAccess, PFN_cuCtxEnablePeerAccess_v4000);
WP_HIP_PFN(hipCtxGetCurrent, PFN_cuCtxGetCurrent_v4000);
WP_HIP_PFN(hipCtxGetDevice, PFN_cuCtxGetDevice_v2000);
WP_HIP_PFN(hipCtxPopCurrent, PFN_cuCtxPopCurrent_v4000);
WP_HIP_PFN(hipCtxPushCurrent, PFN_cuCtxPushCurrent_v4000);
WP_HIP_PFN(hipCtxSetCurrent, PFN_cuCtxSetCurrent_v4000);
WP_HIP_PFN(hipCtxSynchronize, PFN_cuCtxSynchronize_v2000);
WP_HIP_PFN(hipDeviceCanAccessPeer, PFN_cuDeviceCanAccessPeer_v4000);
WP_HIP_PFN(hipDeviceGetAttribute, PFN_cuDeviceGetAttribute_v2000);
// ROCm spells this hipGetDeviceCount, not hipDeviceGetCount.
WP_HIP_PFN(hipGetDeviceCount, PFN_cuDeviceGetCount_v2000);
WP_HIP_PFN(hipDeviceGetName, PFN_cuDeviceGetName_v2000);
WP_HIP_PFN(hipDeviceGetUuid, PFN_cuDeviceGetUuid_v11040);
WP_HIP_PFN(hipDeviceGet, PFN_cuDeviceGet_v2000);
WP_HIP_PFN(hipDevicePrimaryCtxRelease, PFN_cuDevicePrimaryCtxRelease_v11000);
WP_HIP_PFN(hipDevicePrimaryCtxRetain, PFN_cuDevicePrimaryCtxRetain_v7000);
// Re-encode into CUDA's major*1000 + minor*10 scheme, matching CUDA_VERSION
// above. ROCm documents that "there is no mapping/correlation between HIP
// driver version and CUDA driver version", and hipDriverGetVersion returns
// HIP's wide encoding. Warp compares this against min_driver_version, which it
// derives from CUDA_VERSION, so both sides must use the same scheme or device
// enumeration is skipped and no GPU is ever registered.
static inline hipError_t wp_hipDriverGetVersion(int* driverVersion)
{
    if (!driverVersion)
        return hipErrorInvalidValue;
    int raw = 0;
    hipError_t status = hipDriverGetVersion(&raw);
    if (status != hipSuccess)
        return status;
    // HIP: major*10000000 + minor*100000 + patch  ->  CUDA: major*1000 + minor*10
    const int major = raw / 10000000;
    const int minor = (raw / 100000) % 100;
    *driverVersion = major * 1000 + minor * 10;
    return status;
}
WP_HIP_PFN(wp_hipDriverGetVersion, PFN_cuDriverGetVersion_v2020);
// The driver API always takes flags; the plain HIP spelling does not.
WP_HIP_PFN(hipEventCreateWithFlags, PFN_cuEventCreate_v2000);
WP_HIP_PFN(hipEventDestroy, PFN_cuEventDestroy_v4000);
WP_HIP_PFN(hipEventQuery, PFN_cuEventQuery_v2000);
WP_HIP_PFN(hipEventRecordWithFlags, PFN_cuEventRecordWithFlags_v11010);
WP_HIP_PFN(hipEventRecord, PFN_cuEventRecord_v2000);
WP_HIP_PFN(hipEventSynchronize, PFN_cuEventSynchronize_v2000);
WP_HIP_PFN(hipFuncGetAttribute, PFN_cuFuncGetAttribute_v2020);
// Setting uses hipFuncAttribute; querying uses hipFunction_attribute, which
// CUfunction_attribute aliases. Convert at the boundary.
static inline hipError_t wp_hipFuncSetAttribute(
    hipFunction_t f, hipFunction_attribute attrib, int value)
{
    return hipFuncSetAttribute(
        reinterpret_cast<const void*>(f), static_cast<hipFuncAttribute>(attrib), value);
}
WP_HIP_PFN(wp_hipFuncSetAttribute, PFN_cuFuncSetAttribute_v9000);
// The driver API returns the message through an out-parameter, while the HIP
// runtime spellings return it directly. ROCm provides the driver-style forms
// under hipDrv*.
// hipDrv* are marked nodiscard, but the driver API's callers legitimately
// ignore the status and just check whether the out-parameter was filled.
// Wrap them so the discarded result is explicit and local.
static inline hipError_t wp_hipDrvGetErrorName(hipError_t e, const char** s)
{
    hipError_t status = hipDrvGetErrorName(e, s);
    if (status != hipSuccess && s)
        *s = nullptr;
    return status;
}
static inline hipError_t wp_hipDrvGetErrorString(hipError_t e, const char** s)
{
    hipError_t status = hipDrvGetErrorString(e, s);
    if (status != hipSuccess && s)
        *s = nullptr;
    return status;
}
// ROCm marks the whole hipError_t enum [[nodiscard]] on C++17 with no opt-out
// macro. Warp both returns these statuses and calls the accessors purely for
// effect, and a conditional expression requires the same enum on both arms, so
// the type has to stay hipError_t. Warp builds with -Werror, so relax just
// this diagnostic for translation units that use the shim; the codebase still
// checks statuses where they matter, via check_cu/check_cuda.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wunused-result"
#endif
WP_HIP_PFN(wp_hipDrvGetErrorName, PFN_cuGetErrorName_v6000);
WP_HIP_PFN(wp_hipDrvGetErrorString, PFN_cuGetErrorString_v6000);
WP_HIP_PFN(hipGetProcAddress, PFN_cuGetProcAddress_v12000);

// ---------------------------------------------------------------------------
// Driver loading. cuda_util.cpp's init_cuda_driver() does, literally:
//
//     hCudaDriver = dlopen("libcuda.so", RTLD_NOW);          // then "libcuda.so.1"
//     pfn_cuGetProcAddress = dlsym(hCudaDriver, "cuGetProcAddress");
//
// Neither exists on ROCm: there is no libcuda.so, and the entry-point loader is
// hipGetProcAddress in libamdhip64.so. The dlopen therefore returned NULL, every
// driver entry point stayed null, and wp_cuda_driver_is_initialized() reported
// false -- so Warp registered no GPU. The library built, linked and imported
// perfectly and simply had no driver behind it.
//
// The library and symbol names are string literals, which a macro cannot
// rewrite. But dlopen and dlsym are *identifiers*, so redirect the calls
// instead and translate the arguments here. Keeps cuda_util.cpp untouched.
// dlopen/dlsym are redirected by the macros at the end of this block, so these
// two helpers must be defined BEFORE them -- otherwise the calls inside would be
// rewritten to call themselves.
#include <dlfcn.h>

#include <cstdio>
#include <cstring>

static inline void* wp_hip_dlopen(const char* filename, int flags)
{
    if (filename && strncmp(filename, "libcuda.so", 10) == 0)
        filename = "libamdhip64.so";
    return dlopen(filename, flags);
}

// Stands in for cuGetProcAddress. Cannot be hipGetProcAddress directly: Warp
// asks for CUDA spellings ("cuCtxCreate") with CUDA version numbers (3020),
// and HIP knows neither. Translate the name to its hip* equivalent and drop
// the version, which has no meaning across the two APIs.
static inline hipError_t wp_hip_get_proc_address(
    const char* symbol, void** pfn, int version, uint64_t flags,
    hipDriverProcAddressQueryResult* symbolStatus)
{
    (void)version;  // CUDA-versioned; not meaningful to HIP
    if (!symbol || !pfn)
        return hipErrorInvalidValue;

    // Most names follow "cuFoo" -> "hipFoo", but not all: HIP moved a few verbs
    // around. Verified against all 80 entry points cuda_util.cpp resolves --
    // 77 follow the rule, and these do not. cuDeviceGetCount is the one that
    // matters: it is how Warp counts GPUs, so a miss here means no device is
    // ever registered.
    static const struct {
        const char* cuda;
        const char* hip;
    } exceptions[] = {
        {"cuDeviceGetCount", "hipGetDeviceCount"},
    };
    for (const auto& e : exceptions) {
        if (strcmp(symbol, e.cuda) == 0)
            return hipGetProcAddress(e.hip, pfn, 0, flags, symbolStatus);
    }

    // "cuFoo" -> "hipFoo". Warp only ever requests cu* driver entry points here.
    char translated[160];
    if (strncmp(symbol, "cu", 2) == 0 && symbol[2] != '\0') {
        int n = snprintf(translated, sizeof(translated), "hip%s", symbol + 2);
        if (n <= 0 || static_cast<size_t>(n) >= sizeof(translated))
            return hipErrorInvalidValue;
        symbol = translated;
    }

    hipError_t status = hipGetProcAddress(symbol, pfn, 0, flags, symbolStatus);
    // Warp tolerates a null entry point and reports it per call site, so a
    // missing symbol is not fatal here -- but it must not look like success.
    if (status == hipSuccess && (!pfn || !*pfn))
        return hipErrorNotFound;
    return status;
}

static inline void* wp_hip_dlsym(void* handle, const char* symbol)
{
    // Warp resolves exactly one symbol this way; everything else goes through
    // the loader it returns.
    if (symbol && strcmp(symbol, "cuGetProcAddress") == 0)
        return reinterpret_cast<void*>(&wp_hip_get_proc_address);
    return dlsym(handle, symbol);
}

#define dlopen wp_hip_dlopen
#define dlsym wp_hip_dlsym
// The driver API passes per-edge data and puts the dependency count after
// it; HIP has no edge-data parameter. Drop it: Warp only uses default
// edges here, which is what HIP assumes.
static inline hipError_t wp_hipGraphAddNode(
    hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies,
    const CUgraphEdgeData* edgeData, size_t numDependencies, hipGraphNodeParams* nodeParams)
{
    (void)edgeData;
    return hipGraphAddNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams);
}
WP_HIP_PFN(wp_hipGraphAddNode, PFN_cuGraphAddNode_v12030);
WP_HIP_PFN(hipGraphNodeGetDependentNodes, PFN_cuGraphNodeGetDependentNodes_v10000);
WP_HIP_PFN(hipGraphNodeGetType, PFN_cuGraphNodeGetType_v10000);
// The GL interop entry points live in hip_gl_interop.h, which requires an
// OpenGL header to be included first. Warp resolves these dynamically and
// handles a null entry point, so declare the types without adding a GL
// build dependency.
using PFN_cuGraphicsGLRegisterBuffer_v3000 =
    hipError_t (*)(hipGraphicsResource**, unsigned int, unsigned int);
using PFN_cuGraphicsGLRegisterImage_v3000 =
    hipError_t (*)(hipGraphicsResource**, unsigned int, unsigned int, unsigned int);
WP_HIP_PFN(hipGraphicsMapResources, PFN_cuGraphicsMapResources_v3000);
WP_HIP_PFN(wp_hipGraphicsResourceGetMappedPointer, PFN_cuGraphicsResourceGetMappedPointer_v3020);
WP_HIP_PFN(hipGraphicsSubResourceGetMappedArray, PFN_cuGraphicsSubResourceGetMappedArray_v3000);
WP_HIP_PFN(hipGraphicsUnmapResources, PFN_cuGraphicsUnmapResources_v3000);
WP_HIP_PFN(hipGraphicsUnregisterResource, PFN_cuGraphicsUnregisterResource_v3000);
WP_HIP_PFN(hipInit, PFN_cuInit_v2000);
WP_HIP_PFN(wp_hipIpcCloseMemHandle, PFN_cuIpcCloseMemHandle_v4010);
WP_HIP_PFN(hipIpcGetEventHandle, PFN_cuIpcGetEventHandle_v4010);
WP_HIP_PFN(wp_hipIpcGetMemHandle, PFN_cuIpcGetMemHandle_v4010);
WP_HIP_PFN(hipIpcOpenEventHandle, PFN_cuIpcOpenEventHandle_v4010);
WP_HIP_PFN(wp_hipIpcOpenMemHandle, PFN_cuIpcOpenMemHandle_v11000);
// The driver API launches a hipFunction_t with flat dimensions; the runtime
// spelling hipLaunchKernel takes a host symbol and dim3.
WP_HIP_PFN(hipModuleLaunchKernel, PFN_cuLaunchKernel_v4000);
WP_HIP_PFN(hipMemGetInfo, PFN_cuMemGetInfo_v3020);
// The driver-API copies take a descriptor struct, while the HIP runtime
// spellings (hipMemcpy2D/hipMemcpy3D) take loose arguments or a different
// struct. ROCm provides the descriptor forms under hipMemcpyParam2D* and
// hipDrvMemcpy3D*.
WP_HIP_PFN(hipMemcpyParam2DAsync, PFN_cuMemcpy2DAsync_v3020);
WP_HIP_PFN(hipMemcpyParam2D, PFN_cuMemcpy2D_v3020);
// CUDA_MEMCPY3D is wp_hip_memcpy3d, not HIP_MEMCPY3D (see its definition), so
// these cannot alias the HIP entry points directly -- the descriptor has to be
// converted first. Doing it here means texture.cpp needs no change.
static inline hipError_t wp_hipDrvMemcpy3D(const wp_hip_memcpy3d* pCopy)
{
    if (!pCopy)
        return hipErrorInvalidValue;
    HIP_MEMCPY3D p = pCopy->to_hip();
    return hipDrvMemcpy3D(&p);
}
static inline hipError_t wp_hipDrvMemcpy3DAsync(const wp_hip_memcpy3d* pCopy, hipStream_t stream)
{
    if (!pCopy)
        return hipErrorInvalidValue;
    HIP_MEMCPY3D p = pCopy->to_hip();
    return hipDrvMemcpy3DAsync(&p, stream);
}
WP_HIP_PFN(wp_hipDrvMemcpy3DAsync, PFN_cuMemcpy3DAsync_v3020);
WP_HIP_PFN(wp_hipDrvMemcpy3D, PFN_cuMemcpy3D_v3020);
WP_HIP_PFN(wp_hipMemcpyBatchAsync, PFN_cuMemcpyBatchAsync_v12080);
// hipCtxGetDevice reports the device of the CURRENT context, so read a specific
// context's device by making it current briefly.
static inline hipError_t wp_hip_ctx_device(hipCtx_t ctx, int* device)
{
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    if (!ctx)
        return hipGetDevice(device);
    if (hipCtxPushCurrent(ctx) != hipSuccess)
        return hipErrorInvalidValue;
    hipError_t status = hipCtxGetDevice(device);
    hipCtx_t popped = nullptr;
    (void)hipCtxPopCurrent(&popped);
    return status;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
}

// The driver API identifies the peers by context; HIP takes device ordinals.
// hipCtx_t carries its device, so translate rather than guard the call site.
// Takes CUdeviceptr (an integer) rather than hipDeviceptr_t: Warp calls this
// with (CUdeviceptr) casts, which no longer convert to void* implicitly.
static inline hipError_t wp_hipMemcpyPeerAsync(
    CUdeviceptr dst, hipCtx_t dstCtx, CUdeviceptr src, hipCtx_t srcCtx, size_t count,
    hipStream_t stream)
{
    int dst_dev = 0;
    int src_dev = 0;
    if (wp_hip_ctx_device(dstCtx, &dst_dev) != hipSuccess)
        return hipErrorInvalidValue;
    if (wp_hip_ctx_device(srcCtx, &src_dev) != hipSuccess)
        return hipErrorInvalidValue;
    return hipMemcpyPeerAsync(
        reinterpret_cast<void*>(dst), dst_dev, reinterpret_cast<void*>(src), src_dev, count,
        stream);
}
WP_HIP_PFN(wp_hipMemcpyPeerAsync, PFN_cuMemcpyPeerAsync_v4000);
// HIP takes a non-const descriptor here where the driver API takes const.
// Adapt in one place rather than casting at the call site.
static inline hipError_t wp_hipMipmappedArrayCreate(
    hipMipmappedArray_t* pHandle, const HIP_ARRAY3D_DESCRIPTOR* pDesc, unsigned int numMipmapLevels)
{
    return hipMipmappedArrayCreate(pHandle, const_cast<HIP_ARRAY3D_DESCRIPTOR*>(pDesc), numMipmapLevels);
}
WP_HIP_PFN(wp_hipMipmappedArrayCreate, PFN_cuMipmappedArrayCreate_v5000);
WP_HIP_PFN(hipMipmappedArrayDestroy, PFN_cuMipmappedArrayDestroy_v5000);
WP_HIP_PFN(hipMipmappedArrayGetLevel, PFN_cuMipmappedArrayGetLevel_v5000);
WP_HIP_PFN(hipModuleGetFunction, PFN_cuModuleGetFunction_v2000);
WP_HIP_PFN(wp_hipModuleGetGlobal, PFN_cuModuleGetGlobal_v3020);
// hipModuleLoadDataEx is fine once CUjit_option aliases hipJitOption; the
// wrapper exists only to keep the driver-API parameter spelling.
static inline hipError_t wp_hipModuleLoadDataEx(
    hipModule_t* module, const void* image, unsigned int numOptions, CUjit_option* options,
    void** optionValues)
{
    return hipModuleLoadDataEx(
        module, image, numOptions, options, optionValues);
}
WP_HIP_PFN(wp_hipModuleLoadDataEx, PFN_cuModuleLoadDataEx_v2010);
WP_HIP_PFN(hipModuleUnload, PFN_cuModuleUnload_v2000);
// OccupancyMaxActiveClusters has no ROCm equivalent; the entry point resolves to
// null at runtime and callers already handle a missing driver entry.
using PFN_cuOccupancyMaxActiveClusters_v11070 = hipError_t (*)(int*, hipFunction_t, const CUlaunchConfig*);
// Overloaded in the HIP headers, so decltype(&f) is ambiguous; state the
// C signature that the driver entry point actually has.
// The driver API passes a per-block shared-memory callback; HIP takes a flat
// size. Warp always passes a null callback, so zero is the faithful value.
static inline hipError_t wp_hipOccupancyMaxPotentialBlockSize(
    int* gridSize, int* blockSize, hipFunction_t f, CUoccupancyB2DSize b2dSize,
    size_t dynSharedMemPerBlk, int blockSizeLimit)
{
    (void)b2dSize;
    return hipModuleOccupancyMaxPotentialBlockSize(
        gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit);
}
WP_HIP_PFN(wp_hipOccupancyMaxPotentialBlockSize, PFN_cuOccupancyMaxPotentialBlockSize_v6050);
WP_HIP_PFN(wp_hipPointerGetAttribute, PFN_cuPointerGetAttribute_v4000);
WP_HIP_PFN(hipProfilerStart, PFN_cuProfilerStart_v4000);
WP_HIP_PFN(hipProfilerStop, PFN_cuProfilerStop_v4000);
WP_HIP_PFN(hipStreamCreateWithPriority, PFN_cuStreamCreateWithPriority_v5050);
WP_HIP_PFN(hipStreamCreateWithFlags, PFN_cuStreamCreate_v2000);
WP_HIP_PFN(hipStreamDestroy, PFN_cuStreamDestroy_v4000);
// v2 is the form that reports the graph and its dependencies.
// Warp passes a uint64_t* for the capture id while HIP declares
// 'unsigned long long*'. They have the same width but are distinct types on
// LP64, so bridge them here rather than casting at the call site.
static inline hipError_t wp_hipStreamGetCaptureInfo(
    hipStream_t stream, hipStreamCaptureStatus* captureStatus_out, cuuint64_t* id_out,
    hipGraph_t* graph_out, const hipGraphNode_t** dependencies_out, size_t* numDependencies_out)
{
    unsigned long long id = 0;
    hipError_t status = hipStreamGetCaptureInfo_v2(
        stream, captureStatus_out, &id, graph_out, dependencies_out, numDependencies_out);
    if (id_out)
        *id_out = static_cast<cuuint64_t>(id);
    return status;
}
WP_HIP_PFN(wp_hipStreamGetCaptureInfo, PFN_cuStreamGetCaptureInfo_v11030);
// StreamGetCtx has no ROCm equivalent; the entry point resolves to
// null at runtime and callers already handle a missing driver entry.
using PFN_cuStreamGetCtx_v9020 = hipError_t (*)(hipStream_t, hipCtx_t*);
WP_HIP_PFN(hipStreamGetPriority, PFN_cuStreamGetPriority_v5050);
WP_HIP_PFN(hipStreamQuery, PFN_cuStreamQuery_v2000);
WP_HIP_PFN(hipStreamSynchronize, PFN_cuStreamSynchronize_v2000);
WP_HIP_PFN(hipStreamUpdateCaptureDependencies, PFN_cuStreamUpdateCaptureDependencies_v11030);
WP_HIP_PFN(hipStreamWaitEvent, PFN_cuStreamWaitEvent_v3020);
WP_HIP_PFN(wp_hipTexObjectCreate, PFN_cuTexObjectCreate_v5000);
WP_HIP_PFN(hipTexObjectDestroy, PFN_cuTexObjectDestroy_v5000);

// Graph memory-node inspection: the enumerators and accessors Warp uses to walk
// a captured graph looking for allocation nodes.
#ifndef CU_GRAPH_NODE_TYPE_MEM_ALLOC
#define CU_GRAPH_NODE_TYPE_MEM_ALLOC hipGraphNodeTypeMemAlloc
#endif  // CU_GRAPH_NODE_TYPE_MEM_ALLOC
#ifndef CU_GRAPH_NODE_TYPE_MEM_FREE
#define CU_GRAPH_NODE_TYPE_MEM_FREE hipGraphNodeTypeMemFree
#endif  // CU_GRAPH_NODE_TYPE_MEM_FREE
#ifndef cudaMemAllocNodeParams
#define cudaMemAllocNodeParams hipMemAllocNodeParams
#endif  // cudaMemAllocNodeParams
#ifndef cudaGraphMemAllocNodeGetParams
#define cudaGraphMemAllocNodeGetParams hipGraphMemAllocNodeGetParams
#endif  // cudaGraphMemAllocNodeGetParams
#ifndef cudaGraphMemFreeNodeGetParams
#define cudaGraphMemFreeNodeGetParams hipGraphMemFreeNodeGetParams
#endif  // cudaGraphMemFreeNodeGetParams

#ifndef CU_RESOURCE_TYPE_MIPMAPPED_ARRAY
#define CU_RESOURCE_TYPE_MIPMAPPED_ARRAY HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY
#endif  // CU_RESOURCE_TYPE_MIPMAPPED_ARRAY

#ifndef cudaThreadExchangeStreamCaptureMode
#define cudaThreadExchangeStreamCaptureMode hipThreadExchangeStreamCaptureMode
#endif  // cudaThreadExchangeStreamCaptureMode
#ifndef cudaStreamCaptureMode
#define cudaStreamCaptureMode hipStreamCaptureMode
#endif  // cudaStreamCaptureMode
#ifndef cudaStreamCaptureModeRelaxed
#define cudaStreamCaptureModeRelaxed hipStreamCaptureModeRelaxed
#endif  // cudaStreamCaptureModeRelaxed
#ifndef cudaStreamCaptureModeThreadLocal
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#endif  // cudaStreamCaptureModeThreadLocal
#ifndef cudaStreamCaptureModeGlobal
#define cudaStreamCaptureModeGlobal hipStreamCaptureModeGlobal
#endif  // cudaStreamCaptureModeGlobal

// Device attributes and managed-memory queries used by wp_cuda_device_*.
#ifndef CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS
#define CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS hipDeviceAttributeConcurrentManagedAccess
#endif  // CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS
#ifndef CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST
#define CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST     hipDeviceAttributeDirectManagedMemAccessFromHost
#endif  // CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST
#ifndef CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED
#define CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED hipDeviceAttributeHostNativeAtomicSupported
#endif  // CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED
#ifndef CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY
#define CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY hipDeviceAttributeManagedMemory
#endif  // CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY
#ifndef CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS
#define CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS hipDeviceAttributePageableMemoryAccess
#endif  // CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS
#ifndef cudaMemAttachGlobal
#define cudaMemAttachGlobal hipMemAttachGlobal
#endif  // cudaMemAttachGlobal
#ifndef cudaStreamGetFlags
#define cudaStreamGetFlags hipStreamGetFlags
#endif  // cudaStreamGetFlags

// Graph memory pool accounting.
#ifndef cudaDeviceGetGraphMemAttribute
#define cudaDeviceGetGraphMemAttribute hipDeviceGetGraphMemAttribute
#endif  // cudaDeviceGetGraphMemAttribute
#ifndef cudaDeviceGraphMemTrim
#define cudaDeviceGraphMemTrim hipDeviceGraphMemTrim
#endif  // cudaDeviceGraphMemTrim
#ifndef cudaGraphMemAttrUsedMemCurrent
#define cudaGraphMemAttrUsedMemCurrent hipGraphMemAttrUsedMemCurrent
#endif  // cudaGraphMemAttrUsedMemCurrent
#ifndef cudaGraphMemAttrUsedMemHigh
#define cudaGraphMemAttrUsedMemHigh hipGraphMemAttrUsedMemHigh
#endif  // cudaGraphMemAttrUsedMemHigh
#ifndef cudaGraphMemAttrReservedMemCurrent
#define cudaGraphMemAttrReservedMemCurrent hipGraphMemAttrReservedMemCurrent
#endif  // cudaGraphMemAttrReservedMemCurrent
#ifndef cudaGraphMemAttrReservedMemHigh
#define cudaGraphMemAttrReservedMemHigh hipGraphMemAttrReservedMemHigh
#endif  // cudaGraphMemAttrReservedMemHigh

// Empty graph nodes and the capture-dependency setter both exist on ROCm.
#ifndef cudaGraphAddEmptyNode
#define cudaGraphAddEmptyNode hipGraphAddEmptyNode
#endif  // cudaGraphAddEmptyNode
#ifndef CU_STREAM_SET_CAPTURE_DEPENDENCIES
#define CU_STREAM_SET_CAPTURE_DEPENDENCIES hipStreamSetCaptureDependencies
#endif  // CU_STREAM_SET_CAPTURE_DEPENDENCIES

// Conditional graph nodes have NO ROCm equivalent.
//
// CUDA 12.4+ can embed an if/while node whose body is re-evaluated during
// replay. HIP has no such node type, which is why a data-dependent loop has to
// be captured as a static graph on AMD.
//
// Warp already gates the feature at runtime:
// warp._src.context.is_conditional_graph_supported() requires a CUDA Toolkit
// and driver of 12.4+, and on HIP the toolkit version is derived from
// HIP_VERSION and never satisfies that, so these paths are unreachable. The
// declarations exist only so the file compiles; the handle constructor reports
// failure, so a path reached in error fails loudly rather than silently
// building a graph with no condition.
static inline hipError_t cudaGraphConditionalHandleCreate(cudaGraphConditionalHandle* handle, hipGraph_t graph,
                                                          unsigned int default_value = 0,
                                                          unsigned int flags = 0)
{
    (void)graph;
    (void)default_value;
    (void)flags;
    if (handle) {
        *handle = 0;
    }
    return hipErrorNotSupported;
}

#ifndef CU_GRAPH_NODE_TYPE_CONDITIONAL
#define CU_GRAPH_NODE_TYPE_CONDITIONAL ((hipGraphNodeType)-1)
#endif  // CU_GRAPH_NODE_TYPE_CONDITIONAL
#ifndef CU_GRAPH_COND_TYPE_IF
#define CU_GRAPH_COND_TYPE_IF 0
#endif  // CU_GRAPH_COND_TYPE_IF
#ifndef CU_GRAPH_COND_TYPE_WHILE
#define CU_GRAPH_COND_TYPE_WHILE 1
#endif  // CU_GRAPH_COND_TYPE_WHILE

// Graph node types. Every one Warp inspects has a direct HIP equivalent.
#ifndef CU_GRAPH_NODE_TYPE_KERNEL
#define CU_GRAPH_NODE_TYPE_KERNEL hipGraphNodeTypeKernel
#endif  // CU_GRAPH_NODE_TYPE_KERNEL
#ifndef CU_GRAPH_NODE_TYPE_MEMCPY
#define CU_GRAPH_NODE_TYPE_MEMCPY hipGraphNodeTypeMemcpy
#endif  // CU_GRAPH_NODE_TYPE_MEMCPY
#ifndef CU_GRAPH_NODE_TYPE_MEMSET
#define CU_GRAPH_NODE_TYPE_MEMSET hipGraphNodeTypeMemset
#endif  // CU_GRAPH_NODE_TYPE_MEMSET
#ifndef CU_GRAPH_NODE_TYPE_HOST
#define CU_GRAPH_NODE_TYPE_HOST hipGraphNodeTypeHost
#endif  // CU_GRAPH_NODE_TYPE_HOST
#ifndef CU_GRAPH_NODE_TYPE_GRAPH
#define CU_GRAPH_NODE_TYPE_GRAPH hipGraphNodeTypeGraph
#endif  // CU_GRAPH_NODE_TYPE_GRAPH
#ifndef CU_GRAPH_NODE_TYPE_EMPTY
#define CU_GRAPH_NODE_TYPE_EMPTY hipGraphNodeTypeEmpty
#endif  // CU_GRAPH_NODE_TYPE_EMPTY
#ifndef CU_GRAPH_NODE_TYPE_WAIT_EVENT
#define CU_GRAPH_NODE_TYPE_WAIT_EVENT hipGraphNodeTypeWaitEvent
#endif  // CU_GRAPH_NODE_TYPE_WAIT_EVENT
#ifndef CU_GRAPH_NODE_TYPE_EVENT_RECORD
#define CU_GRAPH_NODE_TYPE_EVENT_RECORD hipGraphNodeTypeEventRecord
#endif  // CU_GRAPH_NODE_TYPE_EVENT_RECORD
#ifndef CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL
#define CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL hipGraphNodeTypeExtSemaphoreSignal
#endif  // CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL
#ifndef CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT
#define CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT hipGraphNodeTypeExtSemaphoreWait
#endif  // CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT
#ifndef CU_GRAPH_NODE_TYPE_BATCH_MEM_OP
#define CU_GRAPH_NODE_TYPE_BATCH_MEM_OP hipGraphNodeTypeBatchMemOp
#endif  // CU_GRAPH_NODE_TYPE_BATCH_MEM_OP

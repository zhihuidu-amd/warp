#pragma once

#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__)
#error "hip_util.h should only be included for HIP builds."
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>

#ifndef HIP_VERSION
#if defined(HIP_VERSION_MAJOR) && defined(HIP_VERSION_MINOR) && defined(HIP_VERSION_PATCH)
#define HIP_VERSION (HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 + HIP_VERSION_PATCH)
#else
#define HIP_VERSION 0
#endif  // defined(HIP_VERSION_MAJOR) && defined(HIP_VERSION_MINOR) && defined(HIP_VERSION_PATCH)
#endif  // HIP_VERSION
#ifndef CUDA_VERSION
#define CUDA_VERSION HIP_VERSION
#endif  // CUDA_VERSION
#ifndef NVRTC_SUCCESS
#define NVRTC_SUCCESS HIPRTC_SUCCESS
#endif  // NVRTC_SUCCESS
#ifndef nvrtcGetErrorString
#define nvrtcGetErrorString hiprtcGetErrorString
#endif  // nvrtcGetErrorString
#ifndef nvrtcCreateProgram
#define nvrtcCreateProgram hiprtcCreateProgram
#endif  // nvrtcCreateProgram
#ifndef nvrtcCompileProgram
#define nvrtcCompileProgram hiprtcCompileProgram
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
#ifndef CU_IPC_HANDLE_SIZE
#define CU_IPC_HANDLE_SIZE sizeof(CUipcMemHandle)
#endif  // CU_IPC_HANDLE_SIZE
#ifndef CUDA_SUCCESS
#define CUDA_SUCCESS hipSuccess
#endif  // CUDA_SUCCESS
#ifndef cudaErrorInvalidValue
#define cudaErrorInvalidValue hipErrorInvalidValue
#endif  // cudaErrorInvalidValue
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
#ifndef CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
#define CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES hipFuncAttributeMaxDynamicSharedMemorySize
#endif  // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
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
using cudaSurfaceObject_t = hipSurfaceObject_t;
#ifndef cudaResourceTypeArray
#define cudaResourceTypeArray hipResourceTypeArray
#endif  // cudaResourceTypeArray
#ifndef cudaCreateSurfaceObject
#define cudaCreateSurfaceObject hipCreateSurfaceObject
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
using CUdeviceptr = hipDeviceptr_t;
using CUuuid = hipUUID;
using CUdevice_attribute = hipDeviceAttribute_t;
using CUipcEventHandle = hipIpcEventHandle_t;
using CUipcMemHandle = hipIpcMemHandle_t;
using cuuint64_t = uint64_t;
using CUgraphicsResource = hipGraphicsResource_t;
using CUarray = hipArray_t;
using CUtexObject = hipTextureObject_t;
using CUgraph = hipGraph_t;
using CUgraphNode = hipGraphNode_t;
using CUgraphNodeType = hipGraphNodeType;
using CUgraphNodeParams = void;
using CUgraphEdgeData = void;
using CUstreamCaptureStatus = hipStreamCaptureStatus;
using CUjit_option = hipJitOption;
using CUpointer_attribute = hipPointer_attribute;
using CUfunction_attribute = hipFuncAttribute;
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
using CUlaunchConfig = hipLaunchConfig_t;
#ifndef cudaStreamGetId
#define cudaStreamGetId hipStreamGetId
#endif  // cudaStreamGetId
#if HIP_VERSION >= 70000000
using CUDA_MEMCPY2D = hip_Memcpy2D;
using CUDA_MEMCPY3D = HIP_MEMCPY3D;
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
#ifndef CUDA_ARRAY3D_SURFACE_LDST
#define CUDA_ARRAY3D_SURFACE_LDST hipArraySurfaceLoadStore
#endif  // CUDA_ARRAY3D_SURFACE_LDST
#else
using CUDA_MEMCPY2D = HIP_MEMCPY2D;
using CUDA_MEMCPY3D = HIP_MEMCPY3D;
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
WP_HIP_PFN(hipDriverGetVersion, PFN_cuDriverGetVersion_v2020);
// The driver API always takes flags; the plain HIP spelling does not.
WP_HIP_PFN(hipEventCreateWithFlags, PFN_cuEventCreate_v2000);
WP_HIP_PFN(hipEventDestroy, PFN_cuEventDestroy_v4000);
WP_HIP_PFN(hipEventQuery, PFN_cuEventQuery_v2000);
WP_HIP_PFN(hipEventRecordWithFlags, PFN_cuEventRecordWithFlags_v11010);
WP_HIP_PFN(hipEventRecord, PFN_cuEventRecord_v2000);
WP_HIP_PFN(hipEventSynchronize, PFN_cuEventSynchronize_v2000);
WP_HIP_PFN(hipFuncGetAttribute, PFN_cuFuncGetAttribute_v2020);
WP_HIP_PFN(hipFuncSetAttribute, PFN_cuFuncSetAttribute_v9000);
// The driver API returns the message through an out-parameter, while the HIP
// runtime spellings return it directly. ROCm provides the driver-style forms
// under hipDrv*.
// hipDrv* are marked nodiscard, but the driver API's callers legitimately
// ignore the status and just check whether the out-parameter was filled.
// Wrap them so the discarded result is explicit and local.
static inline hipError_t wp_hipDrvGetErrorName(hipError_t e, const char** s)
{
    return hipDrvGetErrorName(e, s);
}
static inline hipError_t wp_hipDrvGetErrorString(hipError_t e, const char** s)
{
    return hipDrvGetErrorString(e, s);
}
WP_HIP_PFN(wp_hipDrvGetErrorName, PFN_cuGetErrorName_v6000);
WP_HIP_PFN(wp_hipDrvGetErrorString, PFN_cuGetErrorString_v6000);
WP_HIP_PFN(hipGetProcAddress, PFN_cuGetProcAddress_v12000);
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
WP_HIP_PFN(hipGraphicsResourceGetMappedPointer, PFN_cuGraphicsResourceGetMappedPointer_v3020);
WP_HIP_PFN(hipGraphicsSubResourceGetMappedArray, PFN_cuGraphicsSubResourceGetMappedArray_v3000);
WP_HIP_PFN(hipGraphicsUnmapResources, PFN_cuGraphicsUnmapResources_v3000);
WP_HIP_PFN(hipGraphicsUnregisterResource, PFN_cuGraphicsUnregisterResource_v3000);
WP_HIP_PFN(hipInit, PFN_cuInit_v2000);
WP_HIP_PFN(hipIpcCloseMemHandle, PFN_cuIpcCloseMemHandle_v4010);
WP_HIP_PFN(hipIpcGetEventHandle, PFN_cuIpcGetEventHandle_v4010);
WP_HIP_PFN(hipIpcGetMemHandle, PFN_cuIpcGetMemHandle_v4010);
WP_HIP_PFN(hipIpcOpenEventHandle, PFN_cuIpcOpenEventHandle_v4010);
WP_HIP_PFN(hipIpcOpenMemHandle, PFN_cuIpcOpenMemHandle_v11000);
WP_HIP_PFN(hipLaunchKernel, PFN_cuLaunchKernel_v4000);
WP_HIP_PFN(hipMemGetInfo, PFN_cuMemGetInfo_v3020);
// The driver-API copies take a descriptor struct, while the HIP runtime
// spellings (hipMemcpy2D/hipMemcpy3D) take loose arguments or a different
// struct. ROCm provides the descriptor forms under hipMemcpyParam2D* and
// hipDrvMemcpy3D*.
WP_HIP_PFN(hipMemcpyParam2DAsync, PFN_cuMemcpy2DAsync_v3020);
WP_HIP_PFN(hipMemcpyParam2D, PFN_cuMemcpy2D_v3020);
WP_HIP_PFN(hipDrvMemcpy3DAsync, PFN_cuMemcpy3DAsync_v3020);
WP_HIP_PFN(hipDrvMemcpy3D, PFN_cuMemcpy3D_v3020);
WP_HIP_PFN(hipMemcpyBatchAsync, PFN_cuMemcpyBatchAsync_v12080);
WP_HIP_PFN(hipMemcpyPeerAsync, PFN_cuMemcpyPeerAsync_v4000);
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
WP_HIP_PFN(hipModuleGetGlobal, PFN_cuModuleGetGlobal_v3020);
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
using PFN_cuOccupancyMaxActiveClusters_v11070 = hipError_t (*)(int*, hipFunction_t, const hipLaunchConfig_t*);
// Overloaded in the HIP headers, so decltype(&f) is ambiguous; state the
// C signature that the driver entry point actually has.
using PFN_cuOccupancyMaxPotentialBlockSize_v6050 =
    hipError_t (*)(int*, int*, hipFunction_t, size_t, int);
WP_HIP_PFN(hipPointerGetAttribute, PFN_cuPointerGetAttribute_v4000);
WP_HIP_PFN(hipProfilerStart, PFN_cuProfilerStart_v4000);
WP_HIP_PFN(hipProfilerStop, PFN_cuProfilerStop_v4000);
WP_HIP_PFN(hipStreamCreateWithPriority, PFN_cuStreamCreateWithPriority_v5050);
WP_HIP_PFN(hipStreamCreateWithFlags, PFN_cuStreamCreate_v2000);
WP_HIP_PFN(hipStreamDestroy, PFN_cuStreamDestroy_v4000);
// v2 is the form that reports the graph and its dependencies.
WP_HIP_PFN(hipStreamGetCaptureInfo_v2, PFN_cuStreamGetCaptureInfo_v11030);
// StreamGetCtx has no ROCm equivalent; the entry point resolves to
// null at runtime and callers already handle a missing driver entry.
using PFN_cuStreamGetCtx_v9020 = hipError_t (*)(hipStream_t, hipCtx_t*);
WP_HIP_PFN(hipStreamGetPriority, PFN_cuStreamGetPriority_v5050);
WP_HIP_PFN(hipStreamQuery, PFN_cuStreamQuery_v2000);
WP_HIP_PFN(hipStreamSynchronize, PFN_cuStreamSynchronize_v2000);
WP_HIP_PFN(hipStreamUpdateCaptureDependencies, PFN_cuStreamUpdateCaptureDependencies_v11030);
WP_HIP_PFN(hipStreamWaitEvent, PFN_cuStreamWaitEvent_v3020);
WP_HIP_PFN(hipTexObjectCreate, PFN_cuTexObjectCreate_v5000);
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

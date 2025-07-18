#ifndef GPU_ATTRIBUTES_HPP
#define GPU_ATTRIBUTES_HPP
#pragma ONCE

#define SMAX_CUDA_MODE

#if SMAX_CUDA_MODE
#define devattr_t cudaDeviceAttr

#elif SMAX_HIP_MODE
#define devattr_t hipDeviceAttribute_t

#endif

enum gpuAttributes : devattr_t {
    AccessPolicyMaxWindowSize = hipDeviceAttributeAccessPolicyMaxWindowSize,
    AsyncEngineCount = hipDeviceAttributeAsyncEngineCount,
    CanMapHostMemory = hipDeviceAttributeCanMapHostMemory,
    CanUseHostPointerForRegisteredMem =
        hipDeviceAttributeCanUseHostPointerForRegisteredMem,
    ClockRate = hipDeviceAttributeClockRate,
    ComputeMode = hipDeviceAttributeComputeMode,
    ComputePreemptionSupported = hipDeviceAttributeComputePreemptionSupported,
    ConcurrentKernels = hipDeviceAttributeConcurrentKernels,
    ConcurrentManagedAccess = hipDeviceAttributeConcurrentManagedAccess,
    CooperativeLaunch = hipDeviceAttributeCooperativeLaunch,
    CooperativeMultiDeviceLaunch =
        hipDeviceAttributeCooperativeMultiDeviceLaunch,
    DeviceOverlap = hipDeviceAttributeDeviceOverlap,
    DirectManagedMemAccessFromHost =
        hipDeviceAttributeDirectManagedMemAccessFromHost,
    GlobalL1CacheSupported = hipDeviceAttributeGlobalL1CacheSupported,
    HostNativeAtomicSupported = hipDeviceAttributeHostNativeAtomicSupported,
    Integrated = hipDeviceAttributeIntegrated,
    IsMultiGpuBoard = hipDeviceAttributeIsMultiGpuBoard,
    KernelExecTimeout = hipDeviceAttributeKernelExecTimeout,
    L2CacheSize = hipDeviceAttributeL2CacheSize,
    LocalL1CacheSupported = hipDeviceAttributeLocalL1CacheSupported,
    Luid = hipDeviceAttributeLuid,
    LuidDeviceNodeMask = hipDeviceAttributeLuidDeviceNodeMask,
    ComputeCapabilityMajor = hipDeviceAttributeComputeCapabilityMajor,
    ManagedMemory = hipDeviceAttributeManagedMemory,
    MaxBlocksPerMultiProcessor = hipDeviceAttributeMaxBlocksPerMultiProcessor,
    MaxBlockDimX = hipDeviceAttributeMaxBlockDimX,
    MaxBlockDimY = hipDeviceAttributeMaxBlockDimY,
    MaxBlockDimZ = hipDeviceAttributeMaxBlockDimZ,
    MaxGridDimX = hipDeviceAttributeMaxGridDimX,
    MaxGridDimY = hipDeviceAttributeMaxGridDimY,
    MaxGridDimZ = hipDeviceAttributeMaxGridDimZ,
    MaxSurface1D = hipDeviceAttributeMaxSurface1D,
    MaxSurface1DLayered = hipDeviceAttributeMaxSurface1DLayered,
    MaxSurface2D = hipDeviceAttributeMaxSurface2D,
    MaxSurface2DLayered = hipDeviceAttributeMaxSurface2DLayered,
    MaxSurface3D = hipDeviceAttributeMaxSurface3D,
    MaxSurfaceCubemap = hipDeviceAttributeMaxSurfaceCubemap,
    MaxSurfaceCubemapLayered = hipDeviceAttributeMaxSurfaceCubemapLayered,
    MaxTexture1DWidth = hipDeviceAttributeMaxTexture1DWidth,
    MaxTexture1DLayered = hipDeviceAttributeMaxTexture1DLayered,
    MaxTexture1DLinear = hipDeviceAttributeMaxTexture1DLinear,
    MaxTexture1DMipmap = hipDeviceAttributeMaxTexture1DMipmap,
    MaxTexture2DWidth = hipDeviceAttributeMaxTexture2DWidth,
    MaxTexture2DHeight = hipDeviceAttributeMaxTexture2DHeight,
    MaxTexture2DGather = hipDeviceAttributeMaxTexture2DGather,
    MaxTexture2DLayered = hipDeviceAttributeMaxTexture2DLayered,
    MaxTexture2DLinear = hipDeviceAttributeMaxTexture2DLinear,
    MaxTexture2DMipmap = hipDeviceAttributeMaxTexture2DMipmap,
    MaxTexture3DWidth = hipDeviceAttributeMaxTexture3DWidth,
    MaxTexture3DHeight = hipDeviceAttributeMaxTexture3DHeight,
    MaxTexture3DDepth = hipDeviceAttributeMaxTexture3DDepth,
    MaxTexture3DAlt = hipDeviceAttributeMaxTexture3DAlt,
    MaxTextureCubemap = hipDeviceAttributeMaxTextureCubemap,
    MaxTextureCubemapLayered = hipDeviceAttributeMaxTextureCubemapLayered,
    MaxThreadsDim = hipDeviceAttributeMaxThreadsDim,
    MaxThreadsPerBlock = hipDeviceAttributeMaxThreadsPerBlock,
    MaxThreadsPerMultiProcessor = hipDeviceAttributeMaxThreadsPerMultiProcessor,
    MaxPitch = hipDeviceAttributeMaxPitch,
    MemoryBusWidth = hipDeviceAttributeMemoryBusWidth,
    MemoryClockRate = hipDeviceAttributeMemoryClockRate,
    ComputeCapabilityMinor = hipDeviceAttributeComputeCapabilityMinor,
    MultiGpuBoardGroupID = hipDeviceAttributeMultiGpuBoardGroupID,
    MultiprocessorCount = hipDeviceAttributeMultiprocessorCount,
    Name = hipDeviceAttributeName,
    PageableMemoryAccess = hipDeviceAttributePageableMemoryAccess,
    PageableMemoryAccessUsesHostPageTables =
        hipDeviceAttributePageableMemoryAccessUsesHostPageTables,
    PciBusId = hipDeviceAttributePciBusId,
    PciDeviceId = hipDeviceAttributePciDeviceId,
    PciDomainID = hipDeviceAttributePciDomainID,
    PersistingL2CacheMaxSize = hipDeviceAttributePersistingL2CacheMaxSize,
    MaxRegistersPerBlock = hipDeviceAttributeMaxRegistersPerBlock,
    MaxRegistersPerMultiprocessor =
        hipDeviceAttributeMaxRegistersPerMultiprocessor,
    ReservedSharedMemPerBlock = hipDeviceAttributeReservedSharedMemPerBlock,
    MaxSharedMemoryPerBlock = hipDeviceAttributeMaxSharedMemoryPerBlock,
    SharedMemPerBlockOptin = hipDeviceAttributeSharedMemPerBlockOptin,
    SharedMemPerMultiprocessor = hipDeviceAttributeSharedMemPerMultiprocessor,
    SingleToDoublePrecisionPerfRatio =
        hipDeviceAttributeSingleToDoublePrecisionPerfRatio,
    StreamPrioritiesSupported = hipDeviceAttributeStreamPrioritiesSupported,
    SurfaceAlignment = hipDeviceAttributeSurfaceAlignment,
    TccDriver = hipDeviceAttributeTccDriver,
    TextureAlignment = hipDeviceAttributeTextureAlignment,
    TexturePitchAlignment = hipDeviceAttributeTexturePitchAlignment,
    TotalConstantMemory = hipDeviceAttributeTotalConstantMemory,
    TotalGlobalMem = hipDeviceAttributeTotalGlobalMem,
    UnifiedAddressing = hipDeviceAttributeUnifiedAddressing,
    Uuid = hipDeviceAttributeUuid,
    WarpSize = hipDeviceAttributeWarpSize,
    MemoryPoolsSupported = hipDeviceAttributeMemoryPoolsSupported,
    VirtualMemoryManagementSupported =
        hipDeviceAttributeVirtualMemoryManagementSupported,
    Arch = hipDeviceAttributeArch,
    MaxSharedMemoryPerMultiprocessor =
        hipDeviceAttributeMaxSharedMemoryPerMultiprocessor,
    GcnArch = hipDeviceAttributeGcnArch,
    GcnArchName = hipDeviceAttributeGcnArchName,
    HdpMemFlushCntl = hipDeviceAttributeHdpMemFlushCntl,
    HdpRegFlushCntl = hipDeviceAttributeHdpRegFlushCntl,
    CooperativeMultiDeviceUnmatchedFunc =
        hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc,
    CooperativeMultiDeviceUnmatchedGridDim =
        hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim,
    CooperativeMultiDeviceUnmatchedBlockDim =
        hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim,
    CooperativeMultiDeviceUnmatchedSharedMem =
        hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem,
    IsLargeBar = hipDeviceAttributeIsLargeBar,
    AsicRevision = hipDeviceAttributeAsicRevision,
    CanUseStreamWaitValue = hipDeviceAttributeCanUseStreamWaitValue,
    ImageSupport = hipDeviceAttributeImageSupport,
    PhysicalMultiProcessorCount = hipDeviceAttributePhysicalMultiProcessorCount,
    FineGrainSupport = hipDeviceAttributeFineGrainSupport,
    WallClockRate = hipDeviceAttributeWallClockRate,
    VendorSpecificBegin = hipDeviceAttributeVendorSpecificBegin
}

#endif
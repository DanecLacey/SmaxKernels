#ifndef CUDA_HELPER_CUH
#define CUDA_HELPER_CUH
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <memory>

#include "cuda_error_handler.cuh"
#include "stream_helper.cuh"

namespace SH = stream_helper;


namespace cuda_helpers {
  // Type alias for a unique_ptr managing CUDA memory
  struct device_deleter {
    void operator()(void *ptr) { CHECK_CUDA_ERR(cudaFree(ptr)); }
  };

  struct host_deleter {
    void operator()(void *ptr) { CHECK_CUDA_ERR(cudaFreeHost(ptr)); }
  };

  template <typename VT>
  using device_unique_ptr = std::unique_ptr<VT, device_deleter>;

  template <typename VT>
  using host_unique_ptr = std::unique_ptr<VT, host_deleter>;

  template <typename VT>
  inline size_t sizeInBytes(const VT *ptr, size_t N) {
    return N * sizeof(VT);
  }

  template <typename VT>
  inline size_t sizeInBytes(size_t N) {
    return N * sizeof(VT);
  }

  template <typename VT>
  inline float sizeInGBytes(const VT *ptr, size_t N) {
    return static_cast<float>(sizeInBytes(ptr, N)) / 1e9;
  }

  template <typename VT>
  inline float sizeInGBytes(size_t N) {
    return static_cast<float>(sizeInBytes<VT>(N)) / 1e9;
  }

  template <typename VT, typename DT>
  inline float sizeInGBytes(const std::unique_ptr<VT, DT> &ptr, size_t N) {
    return sizeInGBytes(ptr.get(), N);
  }

  template <typename VT, typename DT>
  inline size_t sizeInBytes(const std::unique_ptr<VT, DT> &ptr, size_t N) {
    return sizeInBytes(ptr.get(), N);
  }

  template <typename VT>
  inline host_unique_ptr<VT> allocHost(size_t N) {
    VT *ptr;
    CHECK_CUDA_ERR(cudaMallocHost(&ptr, N * sizeof(VT)));
    return host_unique_ptr<VT>(ptr);
  }

  template <typename VT>
  inline device_unique_ptr<VT> allocDevice(size_t N) {
    VT *ptr;
    CHECK_CUDA_ERR(cudaMalloc(&ptr, N * sizeof(VT)));
    return device_unique_ptr<VT>(ptr);
  }

  template <typename VT>
  inline device_unique_ptr<VT> allocManaged(size_t N) {
    VT *ptr;
    CHECK_CUDA_ERR(cudaMallocManaged(&ptr, N * sizeof(VT)));
    return device_unique_ptr<VT>(ptr);
  }

  template <typename VT>
  inline void memsetDevice(VT *dev, int value, size_t N) {
    CHECK_CUDA_ERR(cudaMemset(dev, value, N * sizeof(VT)));
  }

  template <typename VT>
  inline void memsetDevice(device_unique_ptr<VT> &dev, int value, size_t N) {
    memsetDevice(dev.get(), value, N);
  }

  template <typename VT>
  inline void initHost(VT *ptr, size_t N, const VT &value = VT()) {
    std::fill(ptr, ptr + N, value);
  }

  template <typename VT>
  inline void initHost(host_unique_ptr<VT> &ptr, size_t N,
                       const VT &value = VT()) {
    initHost(ptr.get(), N, value);
  }

  template <typename VT>
  inline void memcpyH2D(const VT *host, VT *dev, size_t N) {
    CHECK_CUDA_ERR(
      cudaMemcpy(dev, host, N * sizeof(VT), cudaMemcpyHostToDevice));
  }

  template <typename VT>
  inline void memcpyH2D(const host_unique_ptr<VT> &host,
                        device_unique_ptr<VT> &dev, size_t N) {
    memcpyH2D(host.get(), dev.get(), N);
  }

  template <typename VT>
  inline void memcpyD2H(const VT *dev, VT *host, size_t N) {
    CHECK_CUDA_ERR(
      cudaMemcpy(host, dev, N * sizeof(VT), cudaMemcpyDeviceToHost));
  }

  template <typename VT>
  inline void memcpyD2H(const device_unique_ptr<VT> &dev,
                        host_unique_ptr<VT> &host, size_t N) {
    memcpyD2H(dev.get(), host.get(), N);
  }

  template <typename VT>
  inline void asyncMemcpyH2D(const VT *host, VT *dev, size_t N,
                             const SH::cudaStream &stream) {
    CHECK_CUDA_ERR(cudaMemcpyAsync(dev, host, N * sizeof(VT),
                                   cudaMemcpyHostToDevice, stream));
  }

  template <typename VT>
  inline void asyncMemcpyH2D(const host_unique_ptr<VT> &host,
                             device_unique_ptr<VT> &dev, size_t N,
                             const SH::cudaStream &stream) {
    asyncMemcpyH2D(host.get(), dev.get(), N, stream);
  }

  template <typename VT>
  inline void asyncMemcpyD2H(const VT *dev, VT *host, size_t N,
                             const SH::cudaStream &stream) {
    CHECK_CUDA_ERR(cudaMemcpyAsync(host, dev, N * sizeof(VT),
                                   cudaMemcpyDeviceToHost, stream));
  }

  template <typename VT>
  inline void asyncMemcpyD2H(const device_unique_ptr<VT> &dev,
                             host_unique_ptr<VT> &host, size_t N,
                             const SH::cudaStream &stream = {}) {
    asyncMemcpyD2H(dev.get(), host.get(), N, stream);
  }

  unsigned int getWarpSize(int devID = 0) {
    CHECK_CUDA_ERR(cudaSetDevice(devID));
    thread_local static int ws = 0;
    if (ws == 0) {
      CHECK_CUDA_ERR(cudaDeviceGetAttribute(&ws, cudaDevAttrWarpSize, devID));
    }
    return ws;
  }

  unsigned int getSMCount(int devID = 0) {
    CHECK_CUDA_ERR(cudaSetDevice(devID));
    thread_local static int SMcount = -1;
    if (SMcount == -1) {
      CHECK_CUDA_ERR(cudaDeviceGetAttribute(
        &SMcount, cudaDevAttrMultiProcessorCount, devID));
    }
    return SMcount;
  }
}  // namespace cuda_helpers

#endif
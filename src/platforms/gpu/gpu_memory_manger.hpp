#ifndef GPU_MEMORY_MANAGER_HPP
#define GPU_MEMORY_MANAGER_HPP
#pragma once

#include <memory>

#include "gpu_backend.hpp"
#include "gpu_error_handler.hpp"
#include "gpu_stream_event_manager.hpp"
#include <iostream>

namespace SMAX {
struct device_deleter {
    void operator()(void *ptr) { GPU_SAFE_BACKEND_CALL(Free, (ptr)); }
};

struct host_deleter {
    void operator()(void *ptr) { GPU_SAFE_BACKEND_CALL(FreeHost, (ptr)); }
};

template <typename VT>
using device_unique_ptr = std::unique_ptr<VT, device_deleter>;

template <typename VT>
using host_unique_ptr = std::unique_ptr<VT, host_deleter>;

template <typename VT> inline size_t sizeInBytes(const VT *ptr, size_t N) {
    return N * sizeof(VT);
}

template <typename VT> inline size_t sizeInBytes(size_t N) {
    return N * sizeof(VT);
}

template <typename VT, typename DT>
inline size_t sizeInBytes(const std::unique_ptr<VT, DT> &ptr, size_t N) {
    return sizeInBytes(ptr.get(), N);
}

template <typename VT, typename DT>
inline void memSet(const std::unique_ptr<VT, DT> &ptr, size_t N, int val) {
    GPU_SAFE_BACKEND_CALL(Memset, (ptr.get(), val, N * sizeof(VT)));
}

template <typename VT> inline host_unique_ptr<VT> allocHost(size_t N) {
    VT *ptr;
    GPU_SAFE_BACKEND_CALL(MallocHost, (&ptr, N * sizeof(VT)));
    return host_unique_ptr<VT>(ptr);
}

template <typename VT> inline device_unique_ptr<VT> allocGPU(size_t N) {
    VT *ptr;
    GPU_SAFE_BACKEND_CALL(Malloc, (&ptr, N * sizeof(VT)));
    return device_unique_ptr<VT>(ptr);
}

template <typename VT> inline device_unique_ptr<VT> allocManaged(size_t N) {
    VT *ptr;
    GPU_SAFE_BACKEND_CALL(MallocManaged, (&ptr, N * sizeof(VT)));
    return device_unique_ptr<VT>(ptr);
}

template <typename VT>
inline void memcpyH2D(const void *host_src, void *dev_dest, size_t N) {
    GPU_SAFE_BACKEND_CALL(Memcpy, (dev_dest, host_src, N * sizeof(VT),
                                   GPU_BACKEND(MemcpyHostToDevice)));
}

template <typename VT>
inline void memcpyH2D(const host_unique_ptr<VT> &host_src,
                      device_unique_ptr<VT> &dev_dest, size_t N) {
    memcpyH2D<VT>(host_src.get(), dev_dest.get(), N);
}

template <typename VT>
inline void memcpyD2H(const void *dev_src, void *host_dest, size_t N) {
    GPU_SAFE_BACKEND_CALL(Memcpy, (host_dest, dev_src, N * sizeof(VT),
                                   GPU_BACKEND(MemcpyDeviceToHost)));
}

template <typename VT>
inline void memcpyD2H(const device_unique_ptr<VT> &dev_src,
                      host_unique_ptr<VT> &host_dest, size_t N) {
    memcpyD2H<VT>(dev_src.get(), host_dest.get(), N);
}

template <typename VT>
inline void asyncMemcpyH2D(const void *host_src, void *dev_dest, size_t N,
                           const gpu_stream &stream) {
    GPU_SAFE_BACKEND_CALL(MemcpyAsync,
                          (dev_dest, host_src, N * sizeof(VT),
                           GPU_BACKEND(MemcpyHostToDevice), stream));
}

template <typename VT>
inline void asyncMemcpyH2D(const host_unique_ptr<VT> &host_src,
                           device_unique_ptr<VT> &dev_dest, size_t N,
                           const gpu_stream &stream) {
    asyncMemcpyH2D<VT>(host_src.get(), dev_dest.get(), N, stream);
}

template <typename VT>
inline void asyncMemcpyD2H(const void *dev_src, void *host_dest, size_t N,
                           const gpu_stream &stream) {
    GPU_SAFE_BACKEND_CALL(MemcpyAsync,
                          (host_dest, dev_src, N * sizeof(VT),
                           GPU_BACKEND(MemcpyDeviceToHost), stream));
}

template <typename VT>
inline void asyncMemcpyD2H(const device_unique_ptr<VT> &dev_src,
                           host_unique_ptr<VT> &host_dest, size_t N,
                           const gpu_stream &stream) {
    asyncMemcpyD2H<VT>(dev_src.get(), host_dest.get(), N, stream);
}

template <typename elem_type>
void allocate_copy_to_device(const void *host_data, void *&device_data_out, size_t n_elems) {
#if SMAX_CUDA_MODE

    IF_SMAX_DEBUG_3(std::cout << "Allocating: " << n_elems
                              << " spaces of size: " << sizeof(elem_type)
                              << " on device" << std::endl);

    elem_type *d_data = allocGPU<elem_type>(n_elems).release();
    memcpyH2D<elem_type>(host_data, d_data, n_elems);
    // Must cast to void for interop with core data structures
    device_data_out = static_cast<void *>(d_data);
#else
    // TODO: Throw error
#endif
};

template <typename elem_type>
void transfer_DtoH(const void *d_data, void *&h_data_out, size_t n_elems) {

#if SMAX_CUDA_MODE
    elem_type *h_data = static_cast<elem_type *>(h_data_out);

    IF_SMAX_DEBUG_3(std::cout << "Copying back: " << n_elems
                              << " spaces of size: " << sizeof(elem_type)
                              << " on host" << std::endl);
    memcpyD2H<elem_type>(d_data, h_data, n_elems);
    // GPU_SAFE_BACKEND_CALL(Free, (const_cast<void *>(d_data))); // highly discouraged to remove const but 
    h_data_out = static_cast<void *>(h_data);
#else
    // TODO: Throw error
#endif
};

} // namespace SMAX

#endif
#pragma once

#include "../../macros.hpp"
#include "../../memory_utils.hpp"
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

namespace SMAX {

template <typename elem_type>
void transfer_HtoD(const void *h_data, void *&d_data_out, size_t n_elems) {
#if SMAX_CUDA_MODE
    elem_type *d_data = nullptr;

    IF_SMAX_DEBUG(std::cout << "Allocating: " << n_elems << " spaces of size: "
                            << sizeof(elem_type) << " on device" << std::endl);

    cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&d_data),
                                 n_elems * sizeof(elem_type));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err)
                  << std::endl;
        d_data_out = nullptr;
        return;
    }

    // printf("host_ptr = [");
    // for (int i = 0; i < n_elems; ++i) {
    //     std::cout << h_data[i] << ", ";
    // }
    // printf("]\n");

    err = cudaMemcpy(d_data, h_data, n_elems * sizeof(elem_type),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy (Host to Device) failed: "
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_data);
        d_data_out = nullptr;
        return;
    }

    // Must cast to void for interop with core data structures
    d_data_out = static_cast<void *>(d_data);
#else
    // TODO: Throw error
#endif
};

template <typename elem_type>
void transfer_DtoH(const void *d_data, void *&h_data_out, size_t n_elems) {

#if SMAX_CUDA_MODE
    elem_type *h_data = static_cast<elem_type *>(h_data_out);

    IF_SMAX_DEBUG(std::cout << "Copying back: " << n_elems
                            << " spaces of size: " << sizeof(elem_type)
                            << " on host" << std::endl);

    cudaError_t err = cudaMemcpy(h_data, d_data, n_elems * sizeof(elem_type),
                                 cudaMemcpyDeviceToHost);

    // printf("y = [");
    // for (int i = 0; i < n_elems; ++i) {
    //     std::cout << h_data[i] << ", ";
    // }
    // printf("]\n");

    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy (Device to Host) failed: "
                  << cudaGetErrorString(err) << std::endl;
    }

    // Must cast to void for interop with core data structures
    h_data_out = static_cast<void *>(h_data);
#else
    // TODO: Throw error
#endif
};

} // namespace SMAX
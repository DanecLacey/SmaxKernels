#ifndef STREAM_HELPER_CUH
#define STREAM_HELPER_CUH

#pragma once

#include <cuda_runtime.h>

#include "cuda_error_handler.cuh"

namespace stream_helper {

  class cudaStream {
   public:
    cudaStream(unsigned int flags = cudaStreamDefault, int priority = 0) {
      CHECK_CUDA_ERR(cudaStreamCreateWithPriority(&stream_, flags, priority));
    }

    ~cudaStream() { CHECK_CUDA_ERR(cudaStreamDestroy(stream_)); }

    cudaStream(const cudaStream &) = delete;
    cudaStream &operator=(const cudaStream &) = delete;

    cudaStream(cudaStream &&other) noexcept : stream_(other.stream_) {
      other.stream_ = nullptr;
    }

    cudaStream &operator=(cudaStream &&other) noexcept {
      if (this != &other) {
        CHECK_CUDA_ERR(cudaStreamDestroy(stream_));
      }
      stream_ = other.stream_;
      other.stream_ = nullptr;
      return *this;
    }

    // Disallow construction from an `int`, e.g., `0`.
    cudaStream(int) = delete;
    // Disallow construction from `nullptr`.
    cudaStream(std::nullptr_t) = delete;

    cudaStream_t get() const noexcept { return stream_; }

    int getPriority() const {
      int prio;
      CHECK_CUDA_ERR(cudaStreamGetPriority(stream_, &prio));
      return prio;
    }

    void synchronize() { CHECK_CUDA_ERR(cudaStreamSynchronize(stream_)); }

    operator cudaStream_t() const noexcept { return stream_; }

    static std::pair<int, int> getPriorityRange() {
      if (least == 0 && highest == 0) {
        CHECK_CUDA_ERR(cudaDeviceGetStreamPriorityRange(&least, &highest));
      }
      return {least, highest};
    }

   private:
    cudaStream_t stream_;
    inline static thread_local int least = 0;
    inline static thread_local int highest = 0;
  };
}  // namespace stream_helper

#endif

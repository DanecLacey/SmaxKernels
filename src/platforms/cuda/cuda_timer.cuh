#ifndef CUDA_TIMER_CUH
#define CUDA_TIMER_CUH
#pragma once

#include "cuda_error_handler.cuh"
#include "event_helper.cuh"
#include "stream_helper.cuh"

#include <string>
#include <stdio.h>

namespace cuda_timer_helper {
  namespace EH = event_helper;
  namespace SH = stream_helper;

  class cudaTimer {
   public:
    cudaTimer()
        : start_(cudaEventDefault), stop_(cudaEventDefault), stream_(0) {}

    explicit cudaTimer(const SH::cudaStream &stream)
        : start_(cudaEventDefault),
          stop_(cudaEventDefault),
          stream_(stream.get()) {}

    ~cudaTimer() = default;

    void setStream(const SH::cudaStream &stream) { stream_ = stream.get(); }

    inline void start() { CHECK_CUDA_ERR(cudaEventRecord(start_, stream_)); }

    inline void stop() { CHECK_CUDA_ERR(cudaEventRecord(stop_, stream_)); }

    float elapsedMilliseconds() {
      float ms = 0.0f;
      CHECK_CUDA_ERR(cudaEventSynchronize(stop_));
      CHECK_CUDA_ERR(cudaEventElapsedTime(&ms, start_, stop_));
      return ms;
    }

    float elapsedSeconds() { return elapsedMilliseconds() * 1e-3f; }

   private:
    EH::cudaEvent start_;
    EH::cudaEvent stop_;
    cudaStream_t stream_;
  };
  
  class cudaScopedTimer {
    public:
     explicit cudaScopedTimer(const std::string &label,
                              const SH::cudaStream &stream = {})
         : label_(label), timer_(stream) {
       timer_.start();
     }
     ~cudaScopedTimer() {
       timer_.stop();
       float ms = timer_.elapsedMilliseconds();
       fprintf(stderr, "[%s] Elapsed: %.3f ms\n", label_.c_str(), ms);
     }
 
    private:
     std::string label_;
     cuda_timer_helper::cudaTimer timer_;
   };

}  // namespace cuda_timer_helper


#endif

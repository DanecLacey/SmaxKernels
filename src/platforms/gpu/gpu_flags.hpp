#ifndef GPU_FLAGS_HPP
#define GPU_FLAGS_HPP
#pragma once

#include "gpu_backend.hpp"

namespace SMAX {

enum class stream_create_flags : unsigned int {
    standard_stream = GPU_BACKEND(StreamDefault),
    non_blocking_stream = GPU_BACKEND(StreamNonBlocking),
    default_stream
};

/// @brief higher priority indicated by higher value i.e. P5 is scheduled before
enum stream_priority : unsigned int { P0 = 0, P1, P2, P3, P4, P5 };

enum class event_create_flags : unsigned int {
    default_flag = GPU_BACKEND(EventDefault),
    blocking_sync = GPU_BACKEND(EventBlockingSync),
    disable_timing = GPU_BACKEND(EventDisableTiming),
    inter_process =
        GPU_BACKEND(EventInterprocess) | GPU_BACKEND(EventDisableTiming)
};

enum class event_record_flags : unsigned int {
    default_flag = GPU_BACKEND(EventRecordDefault),
    external_flag = GPU_BACKEND(EventRecordExternal)
};

enum class event_wait_flags : unsigned int {
#if SMAX_CUDA_MODE
    default_flag = GPU_BACKEND(EventWaitDefault),
    external_flag = GPU_BACKEND(EventWaitExternal)
#else
    default_flag = 0,
    external_flag = 0
#endif
};

} // namespace SMAX

#endif
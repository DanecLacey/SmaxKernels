/**
 * @file gpu_stream_manager.hpp
 * @brief GPU stream management interface for CUDA/HIP interoperability
 * @author SMAX
 * @date July 6, 2025
 */

#ifndef DEVICE_STREAM_MANAGER_HPP
#define DEVICE_STREAM_MANAGER_HPP
#pragma once

#include "gpu_backend.hpp"
#include "gpu_error_handler.hpp"
#include "gpu_flags.hpp"

namespace SMAX {
class gpu_event;
class gpu_stream;
/**
 * @class gpuStream
 * @brief A wrapper class for GPU streams providing CUDA/HIP interoperability
 *
 * This class provides a unified interface for managing GPU streams across
 * different GPU backends (CUDA/HIP). It handles stream creation, destruction,
 * synchronization, and event-based coordination.
 */
class gpu_stream {
  private:
    using dev_raw_stream =
        GPU_BACKEND(Stream_t); ///< Backend-specific stream type
    dev_raw_stream stream_;    ///< The underlying GPU stream handle

  public:
    /**
     * @brief Constructs a new GPU stream
     * @param flag Stream creation flags (default: standardStream)
     * @param priority Stream priority level (default: P0 - least priority)
     * @throws gpu_exception on stream creation failure
     */
    gpu_stream(stream_create_flags flag = stream_create_flags::default_stream,
               stream_priority priority = stream_priority::P0) {
        if (stream_create_flags::default_stream == flag) {
            stream_ = nullptr;
            return;
        }

        GPU_SAFE_BACKEND_CALL(StreamCreateWithPriority,
                              (&stream_, static_cast<unsigned int>(flag),
                               -static_cast<int>(priority)));
    }

    /**
     * @brief Destructor - automatically destroys the GPU stream
     * @note This will synchronize the stream before destruction
     */
    ~gpu_stream() {
        if (stream_ != nullptr) {
            GPU_SAFE_BACKEND_CALL(StreamDestroy, (stream_));
        }
    }

    /**
     * @brief Copy constructor - deleted to prevent copying
     * GPU streams are non-copyable resources
     */
    gpu_stream(const gpu_stream &) = delete;

    /**
     * @brief Integer constructor - deleted to prevent accidental construction
     */
    gpu_stream(int) = delete;

    /**
     * @brief Nullptr constructor - deleted to prevent null initialization
     */
    gpu_stream(std::nullptr_t) = delete;

    /**
     * @brief Copy assignment operator - deleted to prevent copying
     * GPU streams are non-copyable resources
     */
    gpu_stream &operator=(const gpu_stream &) = delete;

    /**
     * @brief Synchronizes the stream, blocking until all operations complete
     */
    void synchronize() {
#if DEBUG
        CHECK_DEVICE_LASTERR("Asynchronous Error in Old call");
#endif
        GPU_SAFE_BACKEND_CALL(StreamSynchronize, (stream_));
    }

    /**
     * @brief Queries the stream status without blocking
     * @return true if all operations in the stream have completed, false
     * otherwise
     * @throws gpu_exception on query failure
     */
    inline bool query() {
        auto err = GPU_BACKEND(StreamQuery)(stream_);
        if (err != GPU_BACKEND(Success) || err != GPU_BACKEND(ErrorNotReady)) {
            SMAX::checkDeviceError(err, "streamQuery", __FILE__, __LINE__ - 2);
        }
        return err == GPU_BACKEND(Success);
    }

    /**
     * @brief Implicit conversion operator to backend stream type
     * @return The underlying stream handle
     */
    operator dev_raw_stream() const { return this->stream_; }

    /**
     * @brief Gets the underlying stream handle
     * @return The raw backend stream handle
     */
    inline dev_raw_stream get() const { return *this; }

    /**
     * @brief Releases ownership of the stream handle
     * @return The raw backend stream handle
     * @note After calling this, the gpuStream object no longer manages the
     * stream
     */
    inline dev_raw_stream release() {
        dev_raw_stream tmp = stream_;
        stream_ = nullptr;
        return tmp;
    }
};

/**
 * @brief Makes the stream wait for an event to complete
 * @param event The event to wait for
 * @param flag Event wait flags (default: Default)
 * @throws gpu_exception on failure to set up the wait
 */
// inline void
// wait_on_event(const gpu_event &event,
//               event_wait_flags flag = event_wait_flags::default_flag) {
//     GPU_SAFE_BACKEND_CALL(StreamWaitEvent,(
//         stream_, event, static_cast<unsigned int>(flag)));
// }

class gpu_event {
  private:
    using dev_raw_event = GPU_BACKEND(Event_t);
    dev_raw_event event_;

  public:
    gpu_event(event_create_flags flag = event_create_flags::default_flag) {
        GPU_SAFE_BACKEND_CALL(EventCreateWithFlags,
                              (&event_, static_cast<unsigned int>(flag)));
    }
    ~gpu_event() {
        if (event_ != nullptr)
            GPU_SAFE_BACKEND_CALL(EventDestroy, (event_));
    }

    gpu_event(const gpu_event &) = delete;
    gpu_event(int) = delete;
    gpu_event(std::nullptr_t) = delete;
    gpu_event &operator=(const gpu_event &) = delete;

    void synchronize() const {
#if DEBUG
        CHECK_DEVICE_LASTERR("Asynchronous Error in Old call");
#endif
        GPU_SAFE_BACKEND_CALL(EventSynchronize, (event_));
    }
    bool query() {
        auto err = GPU_BACKEND(EventQuery)(event_);
        if (err != GPU_BACKEND(Success) || err != GPU_BACKEND(ErrorNotReady)) {
            SMAX::checkDeviceError(err, "eventQuery", __FILE__, __LINE__ - 2);
        }
        return err == GPU_BACKEND(Success);
    }
    operator dev_raw_event() const noexcept { return event_; }
    dev_raw_event get() { return *this; }

    dev_raw_event release() {
        dev_raw_event tmp = event_;
        event_ = nullptr;
        return tmp;
    }

    void record_in_stream(
        const gpu_stream &stream =
            gpu_stream(stream_create_flags::default_stream),
        event_record_flags flag = event_record_flags::default_flag) const {
        GPU_SAFE_BACKEND_CALL(
            EventRecordWithFlags,
            (event_, stream, static_cast<unsigned int>(flag)));
    }
    float elapsed_time_millSec(const gpu_event &eventend) const {
        eventend.synchronize();
        float dur{0.0};
        GPU_SAFE_BACKEND_CALL(EventElapsedTime,
                              (&dur, this->event_, eventend.event_));
        return dur;
    }
};

} // namespace SMAX

#endif
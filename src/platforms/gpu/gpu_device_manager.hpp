#ifndef GPU_DEVICE_HPP
#define GPU_DEVICE_HPP
#pragma once

#include "gpu_backend.hpp"
#include "gpu_error_handler.hpp"

namespace SMAX
{
    typedef int device_t;
    class device
    {
    public:
        void set(int devId)
        {
            CHECK_DEVICE_ERR(GPU_BACKEND(SetDevice)(devId));
            devId_ = devId;
        }

        int get(int devId)
        {
            CHECK_DEVICE_ERR(GPU_BACKEND(GetDevice)(&devId_));
            return devId_;
        }
        void Synchronize()
        {
#if DEBUG
            CHECK_DEVICE_LASTERR("Asynchronous Error in Old call");
#endif
            CHECK_DEVICE_ERR(GPU_BACKEND(DeviceSynchronize)());
        }
        void reset() { CHECK_DEVICE_ERR(GPU_BACKEND(DeviceReset)()); }

        static int getDeviceCount()
        {
            int count = 0;
            CHECK_DEVICE_ERR(GPU_BACKEND(GetDeviceCount)(&count));
            return count;
        }
        int getMultiProcessorCount()
        {
            int smcount = 0;
            CHECK_DEVICE_ERR(GPU_BACKEND(DeviceGetAttribute)(
                &smcount, GPU_BACKEND(DevAttrMultiProcessorCount), devId_));
            return smcount;
        }

        int getAttrvale()
        {
            int attr = 0;
            //            CHECK_DEVICE_ERR(GPU_BACKEND(DeviceGetAttribute)(
            //                &attr, GPU_BACKEND(....), devId_));
            return attr;
        }

    private:
        device_t devId_;
    };

} // namespace SMAX

#endif
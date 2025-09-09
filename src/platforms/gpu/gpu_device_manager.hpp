#pragma once
#ifndef GPU_DEVICE_MANAGER_HPP
#define GPU_DEVICE_MANAGER_HPP

#include "gpu_backend.hpp"
#include "gpu_error_handler.hpp"

namespace SMAX {
    namespace device{
        inline void synchronize(){
            GPU_SAFE_BACKEND_CALL(DeviceSynchronize, ());
        }        
    }
}

#endif
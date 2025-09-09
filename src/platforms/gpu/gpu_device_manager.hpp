#pragma once
#ifndef GPU_MANAGER_HPP
#define GPU_MANAGER_HPP

#include "gpu_backend.hpp"
#include "gpu_error_handler.hpp"

namespace SMAX {
    namespace device{
        void synchronize(){
            GPU_SAFE_BACKEND_CALL(DeviceSynchronize, ());
        }
    }
}

#endif
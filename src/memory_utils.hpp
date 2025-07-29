#pragma once

#include <cstddef>
#include <errno.h>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <variant>
#include <vector>

namespace SMAX {

// To ensure users can pass any int-like metadata
using Variant = std::variant<long long, unsigned long long, void *, void **>;
using ULL = unsigned long long int; // But they will all be cast to this type

// one‐level overloads
template <typename T> inline T &as_dref(void *ptr) {
    return *static_cast<T *>(ptr);
}

template <typename T> inline const T &as_dref(const void *ptr) {
    return *static_cast<const T *>(ptr);
}

template <typename T> inline T *&as_ptr_ref(void **slot) {
    // reinterpret the pointed-at void* as a T*
    return *reinterpret_cast<T **>(slot);
}

// Cast void* to T (pointer or otherwise) without dereferencing
template <typename T> inline T as(void *ptr) { return static_cast<T>(ptr); }

template <typename T> inline T as(const void *ptr) {
    return static_cast<T>(ptr);
}

inline ULL get_ull(const Variant &v) {
    return std::visit(
        [](auto &&x) -> ULL {
            using T = std::decay_t<decltype(x)>;

            if constexpr (!std::is_pointer_v<T>) {
                // We don't allow negative
                if constexpr (std::is_signed_v<T>) {
                    if (x < 0) {
                    // 2025.06.20 DL TODO: Temporary workaround
#if !(SMAX_CUDA_MODE || SMAX_HIP_MODE)
                        throw std::runtime_error("metadata value < 0");
#endif
                    }
                }

                return static_cast<ULL>(x);
            } else {
            // 2025.06.20 DL TODO: Temporary workaround
#if !(SMAX_CUDA_MODE || SMAX_HIP_MODE)
                throw std::runtime_error("This should never be called");
#endif
                return reinterpret_cast<ULL>(x);
            }
        },
        v);
}

} // namespace SMAX

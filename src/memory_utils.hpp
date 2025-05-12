#ifndef SMAX_MEMORY_UTILS_HPP
#define SMAX_MEMORY_UTILS_HPP

#include <cstddef>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

namespace SMAX {

// one‐level overloads
template <typename T> inline T &as(void *ptr) { return *static_cast<T *>(ptr); }

template <typename T> inline const T &as(const void *ptr) {
    return *static_cast<const T *>(ptr);
}

template <typename T> inline T *&as_ptr_ref(void **slot) {
    // reinterpret the pointed-at void* as a T*
    return *reinterpret_cast<T **>(slot);
}

} // namespace SMAX

#endif // SMAX_MEMORY_UTILS_HPP

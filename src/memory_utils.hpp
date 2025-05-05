#ifndef SMAX_MEMORY_UTILS_HPP
#define SMAX_MEMORY_UTILS_HPP

#include <cstddef>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

namespace SMAX {

// Generic accessor for casting a void* to a reference of type T
template <typename T> inline T &as(void *ptr) { return *static_cast<T *>(ptr); }

template <typename T> inline const T &as(const void *ptr) {
    return *static_cast<const T *>(ptr);
}

} // namespace SMAX

#endif // SMAX_MEMORY_UTILS_HPP

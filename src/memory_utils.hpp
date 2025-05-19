#pragma once

#include <cstddef>
#include <errno.h>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

namespace SMAX {

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

} // namespace SMAX

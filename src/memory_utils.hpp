#pragma once

#include <cstddef>
#include <errno.h>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

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

// Capture rvalues as void*, so they have a place in memory and can be
// dereferenced later
template <typename T>
void *to_void_ptr(T &&arg, std::vector<std::unique_ptr<char[]>> &storage) {
    using Decayed = std::decay_t<T>;

    if constexpr (std::is_pointer_v<Decayed>) {
        return reinterpret_cast<void *>(arg);
    } else {
        auto buf = std::make_unique<char[]>(sizeof(Decayed));
        *reinterpret_cast<Decayed *>(buf.get()) = std::forward<T>(arg);
        void *ptr = buf.get();
        storage.push_back(std::move(buf)); // ensure memory is kept alive
        return ptr;
    }
}

} // namespace SMAX

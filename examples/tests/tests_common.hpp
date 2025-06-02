#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

#include <limits>

template <typename T> constexpr T get_abs_epsilon() {
    // Lower‐bound on “absolute zero difference” for small values.
    // For float, maybe 1e-8; for double, 1e-14, etc.
    if constexpr (std::is_same_v<T, float>)
        return 1e-7f;
    else if constexpr (std::is_same_v<T, double>)
        return 1e-14;
    else
        return T(1e-7);
}

template <typename T> constexpr T get_rel_epsilon() {
    // A multiple of machine epsilon—e.g. 4× or 8×, depending on
    // how many floating‐point ops you do in your kernel.
    return T(std::numeric_limits<T>::epsilon() * 4);
}

template <typename T> bool almost_equal(T x, T y) {
    T diff = std::abs(x - y);
    T abs_eps = get_abs_epsilon<T>();
    if (diff <= abs_eps)
        return true;

    T max_xy = std::max(std::abs(x), std::abs(y));
    T rel_eps = get_rel_epsilon<T>();
    return (diff <= rel_eps * max_xy);
}

template <typename T>
void compare_arrays(const T *a, const T *b, int N, std::string desc) {
    for (int i = 0; i < N; ++i) {
        if (!almost_equal(a[i], b[i])) {
            std::ostringstream oss;
            oss << desc << " : Mismatch at index " << i << ": first=" << a[i]
                << ", second=" << b[i];
            throw std::runtime_error(oss.str());
        }
    }
}

template <typename T>
void compare_values(const T a, const T b, std::string desc) {
    if (a != b) {
        // Build an error message with index and differing values
        std::ostringstream oss;
        oss << desc << " : Value mismatch : first=" << a << ", second=" << b;
        throw std::runtime_error(oss.str());
    }
}

template <typename T> void print_array(const T *a, int N, std::string desc) {
    std::cout << desc << ": [";
    for (int i = 0; i < N; ++i) {
        std::cout << a[i] << ", ";
    }
    std::cout << "]" << std::endl;
}
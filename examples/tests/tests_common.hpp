#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

template <typename T> constexpr T get_epsilon() {
    return static_cast<T>(1e-7); // For float or general fallback
}

template <> constexpr double get_epsilon<double>() { return 1e-14; }

template <typename T>
void compare_arrays(const T *a, const T *b, int N, std::string desc) {
    for (int i = 0; i < N; ++i) {
        T diff = std::abs(a[i] - b[i]);
        if (diff > get_epsilon<T>()) {
            // Build an error message with index and differing values
            std::ostringstream oss;
            oss << desc << " : Mismatch at index " << i << ": first=" << a[i]
                << ", second=" << b[i];
            throw std::runtime_error(oss.str());
        }
    }
}

template <typename T>
void compare_value(const T a, const T b, std::string desc) {
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
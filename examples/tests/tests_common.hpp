#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

template <typename T>
void compare_arrays(const T *a, const T *b, int N, std::string desc) {
    for (int i = 0; i < N; ++i) {
        if (a[i] != b[i]) {
            // Build an error message with index and differing values
            std::ostringstream oss;
            oss << desc << " : Array mismatch at index " << i
                << ": first=" << a[i] << ", second=" << b[i];
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
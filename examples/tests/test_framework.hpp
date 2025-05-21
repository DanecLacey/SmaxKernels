#pragma once
#include <functional>
#include <map>
#include <string>
#include <vector>

inline std::map<std::string, std::function<void()>> &get_test_registry() {
    static std::map<std::string, std::function<void()>> registry;
    return registry;
}

#define REGISTER_TEST(name)                                                    \
    void name();                                                               \
    struct name##_registrar {                                                  \
        name##_registrar() { get_test_registry()[#name] = name; }              \
    };                                                                         \
    static name##_registrar name##_instance;                                   \
    void name()

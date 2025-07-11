#pragma once

#include <sys/time.h>

namespace SMAX {

class Stopwatch {

    long double wtime{};

  public:
    timeval *begin;
    timeval *end;
    Stopwatch(timeval *_begin, timeval *_end) : begin(_begin), end(_end) {};

    void start(void) { gettimeofday(begin, 0); }

    void stop(void) {
        gettimeofday(end, 0);
        long seconds = end->tv_sec - begin->tv_sec;
        long microseconds = end->tv_usec - begin->tv_usec;
        wtime += seconds + microseconds * 1e-6;
    }

    long double check(void) {
        gettimeofday(end, 0);
        long seconds = end->tv_sec - begin->tv_sec;
        long microseconds = end->tv_usec - begin->tv_usec;
        return seconds + microseconds * 1e-6;
    }

    long double get_wtime() { return wtime; }
};

class Timers {
  public:
    std::unordered_map<std::string, Stopwatch *> all;

    ~Timers() {
        for (auto &[name, sw] : all) {
            delete sw;
        }
    }

    void add(const std::string &name, Stopwatch *sw) { all[name] = sw; }

    Stopwatch *get(const std::string &name) {
        auto it = all.find(name);
        return it != all.end() ? it->second : nullptr;
    }
};

#define CREATE_SMAX_STOPWATCH(timer_name)                                      \
    timeval *timer_name##_time_start = new timeval;                            \
    timeval *timer_name##_time_end = new timeval;                              \
    Stopwatch *timer_name##_time =                                             \
        new Stopwatch(timer_name##_time_start, timer_name##_time_end);         \
    this->timers->add(#timer_name, timer_name##_time);

} // namespace SMAX

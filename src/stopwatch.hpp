#ifndef STOPWATCH_HPP
#define STOPWATCH_HPP

#include <sys/time.h>

namespace SMAX {

class Stopwatch {

    long double wtime{};

  public:
    timeval *begin;
    timeval *end;
    Stopwatch(timeval *_begin, timeval *_end) : begin(_begin), end(_end) {};
    Stopwatch() : begin(), end() {};

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

struct Timers {
    Stopwatch *total_time;
    Stopwatch *register_A_time;
    Stopwatch *register_B_time;
    Stopwatch *register_C_time;
    Stopwatch *initialize_time;
    Stopwatch *apply_time;
    Stopwatch *finalize_time;

    ~Timers() {
        delete total_time;
        delete register_A_time;
        delete register_B_time;
        delete register_C_time;
        delete initialize_time;
        delete apply_time;
        delete finalize_time;
    }
};

#define CREATE_STOPWATCH(timer_name)                                           \
    timeval *timer_name##_time_start = new timeval;                            \
    timeval *timer_name##_time_end = new timeval;                              \
    Stopwatch *timer_name##_time =                                             \
        new Stopwatch(timer_name##_time_start, timer_name##_time_end);         \
    timers->timer_name##_time = timer_name##_time;

void init_timers(Timers *timers) {
    CREATE_STOPWATCH(total)
    CREATE_STOPWATCH(register_A)
    CREATE_STOPWATCH(register_B)
    CREATE_STOPWATCH(register_C)
    CREATE_STOPWATCH(initialize)
    CREATE_STOPWATCH(apply)
    CREATE_STOPWATCH(finalize)
}

} // namespace SMAX

#endif // STOPWATCH_HPP

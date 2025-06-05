#pragma once

#ifdef USE_TIMERS
#undef USE_TIMERS
#endif

#ifdef USE_FAST_MMIO
#undef USE_FAST_MMIO
#endif

#ifdef USE_LIKWID
#undef USE_LIKWID
#endif

#ifdef USE_OPENMP
#undef USE_OPENMP
#endif

#ifdef LINE_COUNT_WARNING
#undef LINE_COUNT_WARNING
#endif

#ifdef LINE_COUNT_ERROR
#undef LINE_COUNT_ERROR
#endif

#ifdef BUF_SIZE
#undef BUF_SIZE
#endif

#ifdef GET_THREAD_ID
#undef GET_THREAD_ID
#endif

#ifdef GET_THREAD_COUNT
#undef GET_THREAD_COUNT
#endif

#ifdef RESTRICT
#undef RESTRICT
#endif

#ifdef CHECK_ERROR
#undef CHECK_ERROR
#endif

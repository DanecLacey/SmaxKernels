#ifndef TIMING_H
#define TIMING_H

#if !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 199309L
#endif

#include <time.h>

double getTimeStamp();

double getTimeResolution();

double getTimeStamp_();

#endif
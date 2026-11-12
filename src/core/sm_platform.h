#ifndef SM_PLATFORM_H
#define SM_PLATFORM_H

#include "sm_utils.h"

#if defined(_WIN32)
    #include <process.h>
    #define sm_getpid_m() ((u64)_getpid())
#elif defined(__unix__) || defined(__APPLE__)
    #include <unistd.h>
    #define sm_getpid_m() ((u64)getpid())
#else
    #define sm_getpid_m() ((u64)0)
#endif

#endif
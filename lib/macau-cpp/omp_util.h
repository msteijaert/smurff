#ifndef OMP_UTIL_H
#define OMP_UTIL_H

#if defined(_OPENMP)
#include <iostream>
#include <omp.h>

inline int nthreads() 
{
    return omp_get_num_threads();
}


inline int thread_limit() 
{
    int nt = -1;
#pragma omp parallel
    {
#pragma omp single
        nt = omp_get_num_threads();
    }
return nt;
}

inline int thread_num()
{
    return omp_get_thread_num(); 
}

inline void threads_init() {
    std::cout << "Using OpenMP with up to " << thread_limit() << " threads.\n";
}

#else
inline int thread_num() { return 0; }
inline int nthreads(void) { return 1; }
inline int thread_limit(void) { return 1; }
inline void threads_init() { }
#endif
#endif

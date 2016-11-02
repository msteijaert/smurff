#ifndef OMP_UTIL_H
#define OMP_UTIL_H

#if defined(_OPENMP)
#include <omp.h>

inline int thread_num() { return omp_get_thread_num(); }
inline int nthreads() 
{
#pragma omp parallel 
   {
#pragma omp single
      {
         return omp_get_num_threads();
      }
   }
}
#else
inline int thread_num() { return 0; }
inline int nthreads(void) { return 1; }
#endif

#endif

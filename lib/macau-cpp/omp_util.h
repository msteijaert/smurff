#ifndef OMP_UTIL_H
#define OMP_UTIL_H

#if defined(_OPENMP)
#include <omp.h>

inline int thread_num() { return omp_get_thread_num(); }
inline int nthreads() 
{
    int nt = -1;
#pragma omp parallel 
   {
#pragma omp single
      {
         nt = omp_get_num_threads();
      }
   }

   return nt;
}
#else
inline int thread_num() { return 0; }
inline int nthreads(void) { return 1; }
#endif

#pragma omp declare reduction (VectorPlus : VectorNd : omp_out += omp_in) initializer(omp_priv = VectorNd::Zero())
#pragma omp declare reduction (MatrixPlus : MatrixNNd : omp_out += omp_in) initializer(omp_priv = MatrixNNd::Zero())

#endif

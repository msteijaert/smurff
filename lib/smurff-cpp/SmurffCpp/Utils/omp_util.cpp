#include "omp_util.h"

#include <iostream>

// _OPENMP will be enabled if -fopenmp flag is passed to the compiler (use cmake release build)
#if defined(_OPENMP)

#include <omp.h>

#ifdef MKL_THREAD_LIBRARY
#include <mkl.h>
#endif

int nthreads() 
{
   return omp_get_num_threads(); 
}

int thread_limit() 
{
   static int nt = -1;
   if (nt < 0) 
   #pragma omp parallel
   {
      #pragma omp single
      nt = omp_get_num_threads();
   }
   return nt;
}

int thread_num() 
{
   return omp_get_thread_num(); 
}

void threads_init(bool verbose) 
{
    if (verbose)
    {
        std::cout << "Using OpenMP with up to " << thread_limit() << " threads.\n";
        #ifdef OPENBLAS
        std::cout << "Using BLAS with up to " << openblas_get_num_threads() << " threads.\n";
        #endif
    }
  
   #if defined(MKL_THREAD_LIBRARY_GNU)
       mkl_set_threading_layer( MKL_THREADING_GNU );
   #elif defined(MKL_THREAD_LIBRARY_INTEL)
       mkl_set_threading_layer( MKL_THREADING_INTEL );
   #elif defined(MKL_THREAD_LIBRARY_SEQUENTIAL)
       THROWERROR("Shouldn't have MKL_THREAD_LIBRARY == sequential when OpenMP is enabled");
   #endif
}

#else

int thread_num() 
{
   return 0;
}

int nthreads() 
{
   return 1; 
}

int thread_limit() 
{
   return 1; 
}

void threads_init(bool verbose) 
{ 
    if (verbose)
    {
        std::cout << "No threading library used.\n";
    }

}

#endif

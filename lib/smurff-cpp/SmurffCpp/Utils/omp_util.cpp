#include "omp_util.h"

#include <iostream>

// _OPENMP will be enabled if -fopenmp flag is passed to the compiler (use cmake release build)
#if defined(_OPENMP)

#include <omp.h>

#ifdef MKL_THREAD_LIBRARY
#include <mkl.h>
#endif

#include <Utils/Error.h>

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

static int prev_threading_layer = -1;

void threads_init(int verbose) 
{
    if (verbose)
    {
        std::cout << "Using OpenMP with up to " << thread_limit() << " threads.\n";
        #ifdef OPENBLAS
        std::cout << "Using BLAS with up to " << openblas_get_num_threads() << " threads.\n";
        #endif
    }
}

void threads_enable(int verbose) 
{
   #if defined(MKL_THREAD_LIBRARY_GNU)
       prev_threading_layer = mkl_set_threading_layer(  MKL_THREADING_GNU );
       if (verbose > 2) {
           std::cout << "  Using GNU threads now. Was using " << prev_threading_layer << " before \n";
       }
   #elif defined(MKL_THREAD_LIBRARY_INTEL) || defined(MKL_THREAD_LIBRARY_LLVM)
       prev_threading_layer = mkl_set_threading_layer( MKL_THREADING_INTEL );
       if (verbose > 2) {
           std::cout << "  Using Intel threads now. Was using " << prev_threading_layer << " before \n";
       }
   #elif defined(MKL_THREAD_LIBRARY_SEQUENTIAL)
       THROWERROR("Shouldn't have MKL_THREAD_LIBRARY == sequential when OpenMP is enabled");
   #else
       THROWERROR("Unkown threading library define");
   #endif
}

void threads_disable(int verbose) 
{
   #ifdef MKL_THREAD_LIBRARY
       THROWERROR_ASSERT (prev_threading_layer >= 0);

       if (verbose > 2)
       {
           std::cout << "  Back to using prev threading layer (" << prev_threading_layer << ")\n";
       }

       mkl_set_threading_layer(prev_threading_layer);
       prev_threading_layer = -1;
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

void threads_init(int verbose) 
{ 
    if (verbose)
    {
        std::cout << "No threading library used.\n";
    }

}

void threads_enable(int verbose) { }
void threads_disable(int verbose) { }

#endif

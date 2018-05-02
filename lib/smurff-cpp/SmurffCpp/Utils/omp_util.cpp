#include "omp_util.h"

#include <iostream>


static int m_prev_threading_layer = -1;
static int m_verbose = 0;
static int m_num_threads = 1;

// _OPENMP will be enabled if -fopenmp flag is passed to the compiler (use cmake release build)
#if defined(_OPENMP)

#include <omp.h>

#ifdef MKL_THREAD_LIBRARY
#include <mkl.h>
#endif

#include <Utils/Error.h>

int nthreads()
{
   int omp_nthreads = omp_get_num_threads(); 
   THROWERROR_ASSERT(omp_nnthreads = m_nthreads);
   return m_nthreads;
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


void threads_init(int verbose, int num_threads) 
{
    m_verbose = verbose;
    m_num_theads = num_threads;

    if (num_threads > 0)
    {
        omp_set_num_threads(num_threads);
    }

    if (verbose)
    {
        std::cout << "Using OpenMP with up to " << ThreadConfig::thread_limit() << " threads.\n";
    }
}

void threads_enable() 
{
   #if defined(MKL_THREAD_LIBRARY_GNU)
       ThreadConfig::m_prev_threading_layer = mkl_set_threading_layer(  MKL_THREADING_GNU );
       if (m_verbose > 2) {
           std::cout << "  Using GNU threads now. Was using " << m_prev_threading_layer << " before \n";
       }
   #elif defined(MKL_THREAD_LIBRARY_INTEL) || defined(MKL_THREAD_LIBRARY_LLVM)
       m_prev_threading_layer = mkl_set_threading_layer( MKL_THREADING_INTEL );
       if (ThreadConfig::m_verbose > 2) {
           std::cout << "  Using Intel threads now. Was using " << ThreadConfig::m_prev_threading_layer << " before \n";
       }
   #elif defined(MKL_THREAD_LIBRARY_SEQUENTIAL)
       THROWERROR("Shouldn't have MKL_THREAD_LIBRARY == sequential when OpenMP is enabled");
   #endif
}

void threads_disable() 
{
   #ifdef MKL_THREAD_LIBRARY
       THROWERROR_ASSERT (ThreadConfig::m_prev_threading_layer >= 0);

       if (ThreadConfig::m_verbose > 2)
       {
           std::cout << "  Back to using m_prev threading layer (" << ThreadConfig::m_prev_threading_layer << ")\n";
       }

       mkl_set_threading_layer(ThreadConfig::m_prev_threading_layer);
       ThreadConfig::m_prev_threading_layer = -1;
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

void threads_init(int verbose, int) 
{ 
    if (verbose)
    {
        std::cout << "No threading library used.\n";
    }

}

void threads_enable() { }
void threads_disable() { }

#endif

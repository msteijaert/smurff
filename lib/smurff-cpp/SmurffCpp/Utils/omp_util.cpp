#include "omp_util.h"

#include <iostream>

#include <SmurffCpp/Utils/Error.h>

namespace smurff
{
namespace threads
{


    // _OPENMP will be enabled if -fopenmp flag is passed to the compiler (use cmake release build)
    #if defined(_OPENMP)

    #include <omp.h>

    #ifdef MKL_THREAD_LIBRARY
    #include <mkl.h>
    #endif

    static bool m_is_init = false;
    static int  m_prev_threading_layer = -1;
    static int  m_verbose = 0;
    static int  m_num_threads = 1;

    int get_num_threads()
    {
        return omp_get_num_threads(); 
    }

    int get_max_threads()
    {
        int ret = omp_get_max_threads();
        THROWERROR_ASSERT(m_num_threads <= ret);
        return ret;
    }

    int get_thread_num()
    {
        int ret = omp_get_thread_num(); 
        THROWERROR_ASSERT(ret <= get_max_threads());
        return ret;
    }


    void init(int verbose, int num_threads) 
    {
        m_verbose = verbose;
        m_num_threads = num_threads;
        bool m_is_init = true;

        if (num_threads > 0)
        {
            omp_set_num_threads(num_threads);
        }

        if (verbose)
        {
            std::cout << "Using OpenMP with up to " << get_max_threads() << " threads.\n";
        }
    }

    void enable() 
    {
    #if defined(MKL_THREAD_LIBRARY_GNU)
        m_prev_threading_layer = mkl_set_threading_layer(  MKL_THREADING_GNU );
        if (m_verbose > 2) {
            std::cout << "  Using GNU threads now. Was using " << m_prev_threading_layer << " before \n";
        }
    #elif defined(MKL_THREAD_LIBRARY_INTEL) || defined(MKL_THREAD_LIBRARY_LLVM)
        m_prev_threading_layer = mkl_set_threading_layer( MKL_THREADING_INTEL );
        if (m_verbose > 2) {
            std::cout << "  Using Intel threads now. Was using " << m_prev_threading_layer << " before \n";
        }
    #elif defined(MKL_THREAD_LIBRARY_SEQUENTIAL)
        THROWERROR("Shouldn't have MKL_THREAD_LIBRARY == sequential when OpenMP is enabled");
    #endif
    }

    void disable() 
    {
    #ifdef MKL_THREAD_LIBRARY
        THROWERROR_ASSERT (m_prev_threading_layer >= 0);

        if (m_verbose > 2)
        {
            std::cout << "  Back to using m_prev threading layer (" << m_prev_threading_layer << ")\n";
        }

        mkl_set_threading_layer(m_prev_threading_layer);
        m_prev_threading_layer = -1;
    #endif
    }

    #else

    void init(int verbose, int) 
    { 
        if (verbose)
        {
            std::cout << "No threading library used.\n";
        }

    }

    int num()      { return 0; }
    int get_num_threads() { return 1; }
    int get_max_threads()    { return 1; }
    void enable()  { }
    void disable() { }

    #endif // _OPENMP
}

}
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

    static int  m_verbose = 0;

    int get_num_threads()
    {
        return omp_get_num_threads(); 
    }

    int get_max_threads()
    {
        return omp_get_max_threads();
    }

    int get_thread_num()
    {
        return omp_get_thread_num(); 
    }


    void init(int verbose, int num_threads) 
    {
        m_verbose = verbose;

        if (num_threads > 0)
        {
            omp_set_num_threads(num_threads);
        } 

        if (verbose)
        {
            std::cout << "Using OpenMP with up to " << get_max_threads() << " threads.\n";
        }
    }

    #else

    void init(int verbose, int) 
    { 
        if (verbose)
        {
            std::cout << "No threading library used.\n";
        }

    }

    int  get_num_threads() { return 1; }
    int  get_max_threads() { return 1; }
    int  get_thread_num() { return 0; } 

    #endif // _OPENMP
}

}

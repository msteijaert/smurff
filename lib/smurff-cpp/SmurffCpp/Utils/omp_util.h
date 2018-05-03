#pragma once

namespace smurff
{
    namespace threads 
    {
        void init(int verbose, int num_threads);
        void enable();
        void disable();

        int  get_num_threads();
        int  get_max_threads();
        int  get_thread_num();

    }
}



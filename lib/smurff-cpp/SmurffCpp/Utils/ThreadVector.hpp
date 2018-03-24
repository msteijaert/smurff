#pragma once

#include <algorithm>
#include <numeric>
#include <vector>

#include "omp_util.h"

namespace smurff
{
   template<typename T>
   class thread_vector
   {
       public:
           thread_vector(const T &t = T()) : _m(thread_limit(), t), _i(t) {}
           template<typename F>
           T combine(F f) const {
               return std::accumulate(_m.begin(), _m.end(), _i, f);
           }
           T combine() const {
               return std::accumulate(_m.begin(), _m.end(), _i, std::plus<T>());
           }
   
           T &local() {
               return _m.at(thread_num());
           }
           void reset() {
               for(auto &t: _m) t = _i;
           }
           template<typename F>
           T combine_and_reset(F f) const {
               T ret = combine(f);
               reset();
               return ret;
           }
           T combine_and_reset() {
               T ret = combine();
               reset();
               return ret;
           }
           void init(const T &t) {
               _i = t;
               reset();
           }
           void init(const std::vector<T> &v) {
               _m = v;
           }
   
   
       private:
           std::vector<T> _m;
           T _i;
   };
}

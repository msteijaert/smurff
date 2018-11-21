#pragma once

#include <cmath>

#include <SmurffCpp/Utils/PVec.hpp>

namespace smurff {

struct ResultItem
{
   ResultItem(const PVec<> &c, float v, float p1s, float pa, float var, int n = 0)
   : coords(c), val(v), pred_1sample(p1s), pred_avg(pa), var(var), nsamples(n), keep_samples(0)
   {
   }


   ResultItem(const PVec<> &c, float v = NAN, int n = 0)
   : coords(c), val(v), pred_1sample(NAN), pred_avg(NAN), var(NAN),
     nsamples(0), keep_samples(n)
   {
      pred_all.resize(n);
   }

   smurff::PVec<> coords;

   float val;
   float pred_1sample;
   float pred_avg;
   float var;

   int nsamples;
   int keep_samples;

   std::vector<float> pred_all;

   void update(float pred) {
      if (nsamples < keep_samples)
         pred_all[nsamples] = pred;

      nsamples++;
      if (nsamples > 1)
      {
        // update pred_1sample, pred_avg and var
        float delta = pred - pred_avg;
        pred_avg = (pred_avg + delta / nsamples);
        var += delta * (pred - pred_avg);
      }
      else
      {
        pred_avg = pred;
        var = 0;
      }
      pred_1sample = pred;
   }
};

inline std::ostream &operator<<(std::ostream& os, const ResultItem& r)
{
   os << r.coords << ": "
      << r.val << ","
      << r.pred_1sample << ","
      << r.pred_avg << ","
      << r.var << "[ ";
      for (auto i: r.pred_all) os << i << ", ";
      os << "]";
      return os;
}

} //end namespace
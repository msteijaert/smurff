#pragma once

#include <SmurffCpp/Utils/PVec.hpp>

namespace smurff {

struct ResultItem
{
   smurff::PVec<> coords;

   double val;
   double pred_1sample;
   double pred_avg;
   double var;

   std::vector<double> pred_all;

   void update(double pred) {
      pred_all.push_back(pred);
      auto N = pred_all.size();
      double delta = pred - pred_avg;

      // update pred_1sample, pred_avg and var
      pred_avg = (pred_avg + delta / N);
      var += delta * (pred - pred_avg);
      pred_1sample = pred;
   }
};

}
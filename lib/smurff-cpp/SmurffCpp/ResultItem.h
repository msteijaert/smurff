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

   void update(double pred, int N) {
      double delta = pred - pred_avg;

      // update pred_1sample, pred_avg and var
      pred_avg = (pred_avg + delta / N);
      var += delta * (pred - pred_avg);
      pred_1sample = pred;
   }
};

}
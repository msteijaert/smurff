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
};

}
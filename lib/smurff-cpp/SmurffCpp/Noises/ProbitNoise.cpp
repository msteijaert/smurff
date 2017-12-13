#include "ProbitNoise.h"

#include <assert.h>
#include <cmath>

#include <SmurffCpp/Utils/TruncNorm.h>
#include <SmurffCpp/Model.h>

using namespace smurff;

ProbitNoise::ProbitNoise()
   : INoiseModel()
{

}

double ProbitNoise::sample(const SubModel& model, const PVec<> &pos, double val11)
{
    assert(val11 >= -1.0 && val11 <= 1.0);
    double pred11 = model.predict(pos);
    double pred01 = 0.5 * (pred11 + 1.0);   
    return rand_truncnorm(pred01 * val11, 1.0, 0.0);
}

std::ostream& ProbitNoise::info(std::ostream& os, std::string indent)
{
   os << "Probit Noise\n";
   return os;
}

std::string ProbitNoise::getStatus()
{
   return std::string();
}

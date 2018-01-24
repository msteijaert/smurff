#include "ProbitNoise.h"

#include <SmurffCpp/Utils/Error.h>

#include <SmurffCpp/Utils/TruncNorm.h>
#include <SmurffCpp/Model.h>

using namespace smurff;

ProbitNoise::ProbitNoise(double t)
   : INoiseModel(), threshold(t)
{

}

/* original code from jaak:
 *
 * double y = 2 * it.value() - 1; // y == sign (-1. or +1.)
 * z = y * rand_truncnorm(y * col.dot(u), 1.0, 0.0);
 * rr.noalias() += col * z
 */


double ProbitNoise::sample(const SubModel& model, const PVec<> &pos, double val)
{
    double sign = (val < threshold) ? -1. : 1.;
    double pred = model.predict(pos);
    return sign * rand_truncnorm(pred * sign, 1.0, 0.0);
}

std::ostream& ProbitNoise::info(std::ostream& os, std::string indent)
{
   os << "Probit Noise with threshold " << threshold << std::endl;
   return os;
}

std::string ProbitNoise::getStatus()
{
   return std::string("Probit ") + std::to_string(threshold);
}

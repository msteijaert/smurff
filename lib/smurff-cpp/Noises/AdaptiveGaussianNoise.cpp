#include "AdaptiveGaussianNoise.h"

#include "mvnormal.h"
#include <DataMatrices/Data.h>

using namespace smurff;

AdaptiveGaussianNoise::AdaptiveGaussianNoise(double sinit, double smax)
: INoiseModel(), sn_max(smax), sn_init(sinit)
{

}

//  AdaptiveGaussianNoise  ////
void AdaptiveGaussianNoise::init(const Data* data)
{
   var_total = data->var_total();

   // Var(noise) = Var(total) / (SN + 1)
   alpha     = (sn_init + 1.0) / var_total;
   alpha_max = (sn_max + 1.0) / var_total;
}

void AdaptiveGaussianNoise::update(const Data* data, const SubModel& model)
{
   double sumsq = data->sumsq(model);

   // (a0, b0) correspond to a prior of 1 sample of noise with full variance
   double a0 = 0.5;
   double b0 = 0.5 * var_total;
   double aN = a0 + data->nnz() / 2.0;
   double bN = b0 + sumsq / 2.0;
   alpha = rgamma(aN, 1.0 / bN);

   if (alpha > alpha_max)
   {
      alpha = alpha_max;
   }
}

double AdaptiveGaussianNoise::getAlpha()
{
   return alpha;
}

std::ostream &AdaptiveGaussianNoise::info(std::ostream &os, std::string indent)
{
   os << "Adaptive gaussian noise with max precision of " << alpha_max << "\n";
   return os;
}

std::string AdaptiveGaussianNoise::getStatus()
{
   return std::string("Prec: ") + to_string_with_precision(alpha, 2);
}

void AdaptiveGaussianNoise::setSNInit(double a)
{
   sn_init = a;
}

void AdaptiveGaussianNoise::setSNMax(double a)
{
   sn_max  = a;
}








#include "UnusedNoise.h"

#include <assert.h>
#include <cmath>

using namespace smurff;

UnusedNoise::UnusedNoise()
: INoiseModel()
{
}

void UnusedNoise::getMuLambda(const SubModel& model, int mode, int d, Eigen::VectorXd& rr, Eigen::MatrixXd& MM)
{
   assert(false);
}

std::ostream& UnusedNoise::info(std::ostream& os, std::string indent)
{
   os << "Noisemodel is not used here.\n";
   return os;
}

std::string UnusedNoise::getStatus()
{
   return std::string("Unused");
}

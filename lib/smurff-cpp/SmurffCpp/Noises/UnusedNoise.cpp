#include "UnusedNoise.h"

#include <SmurffCpp/Utils/Error.h>

using namespace smurff;

UnusedNoise::UnusedNoise()
: INoiseModel()
{
}

void UnusedNoise::getMuLambda(const SubModel& model, int mode, int d, Eigen::VectorXd& rr, Eigen::MatrixXd& MM)
{
   THROWERROR_NOTIMPL();
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

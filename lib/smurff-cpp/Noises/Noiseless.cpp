#include "Noiseless.h"

using namespace smurff;

Noiseless::Noiseless()
   : INoiseModel()
{

}

double Noiseless::getAlpha()
{
   return 1.;
}

std::ostream& Noiseless::info(std::ostream& os, std::string indent)
{
   os << "No noise\n";
   return os;
}

std::string Noiseless::getStatus()
{
   return std::string("No noise");
}
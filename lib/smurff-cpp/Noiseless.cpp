#include "Noiseless.h"

using namespace Eigen;
using namespace smurff;

Noiseless::Noiseless(Data* p) 
   : INoiseModel(p) 
{

}

void Noiseless::init() 
{

}

void Noiseless::update(const SubModel& sm)
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
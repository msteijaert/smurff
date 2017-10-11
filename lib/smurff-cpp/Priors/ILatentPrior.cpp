#include "ILatentPrior.h"

using namespace smurff;
using namespace Eigen;

//most of the methods are new
//there are also some new fields as well

ILatentPrior::ILatentPrior(BaseSession &m, int p, std::string name)
   : session(m), mode(p), name(name) 
{

} 

void ILatentPrior::init() 
{
   rrs.init(VectorNd::Zero(num_latent()));
   MMs.init(MatrixNNd::Zero(num_latent(), num_latent()));
}

Model &ILatentPrior::model() const 
{ 
   return session.model; 
}

Data &ILatentPrior::data() const 
{ 
   return session.data(); 
}

double ILatentPrior::predict(const PVec<> &pos) const {
    return model().predict(pos, data());
}

INoiseModel &ILatentPrior::noise() 
{ 
   return data().noise(); 
}

MatrixXd &ILatentPrior::U() 
{ 
   return model().U(mode); 
}

MatrixXd &ILatentPrior::V() 
{ 
   return model().V(mode); 
}

int ILatentPrior::num_latent() const 
{ 
   return model().nlatent(); 
}

int ILatentPrior::num_cols() const 
{ 
   return model().U(mode).cols(); 
}

std::ostream &ILatentPrior::info(std::ostream &os, std::string indent) 
{
   os << indent << mode << ": " << name << "\n";
   return os;
}

bool ILatentPrior::run_slave() 
{
   return false; 
}

//this method is ok except for:
//dont see any update_pnm equvalent in old code

void ILatentPrior::sample_latents() 
{
   data().update_pnm(model(), mode);

   #pragma omp parallel for schedule(guided)
   for(int n = 0; n < U().cols(); n++) 
   {
      #pragma omp task
      sample_latent(n); 
   }
}

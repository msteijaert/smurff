#include "ILatentPrior.h"

using namespace smurff;
using namespace Eigen;

ILatentPrior::ILatentPrior(std::shared_ptr<BaseSession> session, int mode, std::string name)
   : m_session(session), m_mode(mode), m_name(name) 
{

} 

void ILatentPrior::init() 
{
   rrs.init(VectorXd::Zero(num_latent()));
   MMs.init(MatrixXd::Zero(num_latent(), num_latent()));
}

const Model& ILatentPrior::model() const 
{ 
   return m_session->model();
}

Model& ILatentPrior::model() 
{ 
   return m_session->model();
}

std::shared_ptr<Data> ILatentPrior::data() const 
{ 
   return m_session->data(); 
}

double ILatentPrior::predict(const PVec<> &pos) const 
{
    return model().predict(pos, data());
}

std::shared_ptr<INoiseModel> ILatentPrior::noise() 
{ 
   return data()->noise(); 
}

MatrixXd &ILatentPrior::U() 
{ 
   return model().U(m_mode); 
}

MatrixXd &ILatentPrior::V() 
{ 
   return model().V(m_mode); 
}

int ILatentPrior::num_latent() const 
{ 
   return model().nlatent(); 
}

int ILatentPrior::num_cols() const 
{ 
   return model().U(m_mode).cols(); 
}

std::ostream &ILatentPrior::info(std::ostream &os, std::string indent) 
{
   os << indent << m_mode << ": " << m_name << "\n";
   return os;
}

bool ILatentPrior::run_slave() 
{
   return false; 
}

//update_pnm is smth new and there is no equvalent in old code

void ILatentPrior::sample_latents() 
{
   data()->update_pnm(model(), m_mode);

   #pragma omp parallel for schedule(guided)
   for(int n = 0; n < U().cols(); n++) 
   {
      #pragma omp task
      sample_latent(n);
   }

   update_prior();
}

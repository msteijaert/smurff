#include "ILatentPrior.h"

using namespace smurff;
using namespace Eigen;

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

//macau
/*
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <math.h>
#include <iomanip>

#include "mvnormal.h"
#include "macau.h"
#include "chol.h"
#include "linop.h"

#include "truncnorm.h"
extern "C" {
  #include <sparse.h>
}

using namespace std; 
using namespace Eigen;

void ILatentPrior::sample_latents(FixedGaussianNoise & noise, Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat,
                    double mean_value, const Eigen::MatrixXd &samples, const int num_latent) {
  this->sample_latents(U, mat, mean_value, samples, noise.alpha, num_latent);
}

void ILatentPrior::sample_latents(AdaptiveGaussianNoise & noise, Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat,
                    double mean_value, const Eigen::MatrixXd &samples, const int num_latent) {
  this->sample_latents(U, mat, mean_value, samples, noise.alpha, num_latent);
}

void ILatentPrior::sample_latents(FixedGaussianNoise & noiseModel, MatrixData & matrixData,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent) {
  if (mode == 0) {
    this->sample_latents(noiseModel, *samples[0], matrixData.Yt, matrixData.mean_value, *samples[1], num_latent);
  } else {
    this->sample_latents(noiseModel, *samples[1], matrixData.Y,  matrixData.mean_value, *samples[0], num_latent);
  }
}

void ILatentPrior::sample_latents(AdaptiveGaussianNoise & noiseModel, MatrixData & matrixData,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent) {
  if (mode == 0) {
    this->sample_latents(noiseModel, *samples[0], matrixData.Yt, matrixData.mean_value, *samples[1], num_latent);
  } else {
    this->sample_latents(noiseModel, *samples[1], matrixData.Y,  matrixData.mean_value, *samples[0], num_latent);
  }
}

void ILatentPrior::sample_latents(ProbitNoise & noiseModel, MatrixData & matrixData,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent) {
  if (mode == 0) {
    this->sample_latents(noiseModel, *samples[0], matrixData.Yt, matrixData.mean_value, *samples[1], num_latent);
  } else {
    this->sample_latents(noiseModel, *samples[1], matrixData.Y,  matrixData.mean_value, *samples[0], num_latent);
  }
}

void ILatentPrior::sample_latents(FixedGaussianNoise& noiseModel, TensorData & data,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent) {
  sample_latents(noiseModel.alpha, data, samples, mode, num_latent);
}

void ILatentPrior::sample_latents(AdaptiveGaussianNoise& noiseModel, TensorData & data,
                            std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent) {
  sample_latents(noiseModel.alpha, data, samples, mode, num_latent);
}
*/
#include "NormalPrior.h"

#include <iomanip>

#include <SmurffCpp/Utils/chol.h>
#include <SmurffCpp/Utils/linop.h>
#include <SmurffCpp/IO/MatrixIO.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <SmurffCpp/Utils/Distribution.h>

using namespace Eigen;
using namespace smurff;

//  base class NormalPrior

NormalPrior::NormalPrior(std::shared_ptr<BaseSession> session, int mode, std::string name)
   : ILatentPrior(session, mode, name)
{

}

void NormalPrior::init()
{
   //does not look that there was such init previously
   ILatentPrior::init();

   //this is some new initialization
   initUU();

   const int K = num_latent();
   mu.resize(K);
   mu.setZero();

   Lambda.resize(K, K);
   Lambda.setIdentity();
   Lambda *= 10;

   // parameters of Inv-Whishart distribution
   WI.resize(K, K);
   WI.setIdentity();
   mu0.resize(K);
   mu0.setZero();
   b0 = 2;
   df = K;
}

const Eigen::VectorXd NormalPrior::getMu(int) const
{
    return mu;
}

void NormalPrior::update_prior()
{
   const int N = num_cols();
   
   const auto cov = UUcol.combine_and_reset();
   const auto sum = Ucol.combine_and_reset();

   std::tie(mu, Lambda) = CondNormalWishart(N, cov, sum, mu0, b0, WI, df);
}

void NormalPrior::sample_latent(int n)
{
   const auto &mu_u = getMu(n);

   VectorXd &rr = rrs.local();
   MatrixXd &MM = MMs.local();

   rr.setZero();
   MM.setZero();

   // add pnm
   data()->get_pnm(model(), m_mode, n, rr, MM);

   // add hyperparams
   rr.noalias() += Lambda * mu_u;
   MM.noalias() += Lambda;

   //Solve system of linear equations for x: MM * x = rr

   Eigen::LLT<MatrixXd> chol = MM.llt(); // compute the Cholesky decomposition X = L * U
   if(chol.info() != Eigen::Success)
   {
      throw std::runtime_error("Cholesky Decomposition failed!");
   }

   chol.matrixL().solveInPlace(rr); // solve for y: y = L^-1 * b
   rr.noalias() += nrandn(num_latent());
   chol.matrixU().solveInPlace(rr); // solve for x: x = U^-1 * y
   
   U()->col(n).noalias() = rr; // rr is equal to x
   Ucol.local().noalias() += rr;
   UUcol.local().noalias() += rr * rr.transpose();
}

void NormalPrior::save(std::string prefix, std::string suffix)
{
}

void NormalPrior::restore(std::string prefix, std::string suffix)
{
   initUU();
}

std::ostream &NormalPrior::status(std::ostream &os, std::string indent) const
{
   os << indent << m_name << ": mu = " <<  mu.norm() << std::endl;
   return os;
}

void NormalPrior::initUU()
{
    const int K = num_latent();
    Ucol.init(VectorXd::Zero(K));
    UUcol.init(MatrixXd::Zero(K, K));
    UUcol.local() = *U() * U()->transpose();
    Ucol.local() = U()->rowwise().sum();
}

//macau Probit sample latents
/*
void BPMFPrior::sample_latents(ProbitNoise & noise, Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat,
   double mean_value, const Eigen::MatrixXd &samples, const int num_latent)
{
   const int N = U.cols();

   #pragma omp parallel for schedule(dynamic, 2)
   for(int n = 0; n < N; n++) 
   {
      sample_latent_blas_probit(U, n, mat, mean_value, samples, mu, Lambda, num_latent);
   }
}
*/

//macau Tensor methods
/*
void BPMFPrior::sample_latents(ProbitNoise& noiseModel, TensorData & data,
                               std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, 
                               int mode, 
                               const int num_latent
                              ) 
{
  // TODO
  throw std::runtime_error("Unimplemented: sample_latents");
}
*/
/*
void BPMFPrior::sample_latents(double noisePrecision,
                               TensorData & data, // array of sparse views per dimention
                               std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, //vector of sample matrices
                               const int mode, //dimention index
                               const int num_latent //number of latent dimentions
                              ) 
{
  auto& sparseMode = (*data.Y)[mode]; // select sparse view by dimention index
  auto& U = samples[mode]; // select U matrix by dimention index
  const int N = U->cols();
  VectorView<Eigen::MatrixXd> view(samples, mode); // select all other samples except from dimention index

  #pragma omp parallel for schedule(dynamic, 2)
  for (int n = 0; n < N; n++) //iterate through each column of U
  {
    sample_latent_tensor(U, n, sparseMode, view, data.mean_value, noisePrecision, mu, Lambda);
  }
}
*/

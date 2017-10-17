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

NormalPrior::NormalPrior(BaseSession &m, int p, std::string name)
   : ILatentPrior(m, p, name)
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

void NormalPrior::sample_latents()
{
   //this will go in cycle nad call sample_latent(n)
   ILatentPrior::sample_latents();

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
   data().get_pnm(model(),mode,n,rr,MM);

   // add hyperparams
   rr.noalias() += Lambda * mu_u;
   MM.noalias() += Lambda;

   Eigen::LLT<MatrixXd> chol = MM.llt();
   if(chol.info() != Eigen::Success)
   {
      throw std::runtime_error("Cholesky Decomposition failed!");
   }

   chol.matrixL().solveInPlace(rr);
   rr.noalias() += nrandn(num_latent());
   chol.matrixU().solveInPlace(rr);

   U().col(n).noalias() = rr;
   Ucol.local().noalias() += rr;
   UUcol.local().noalias() += rr * rr.transpose();
}

void NormalPrior::save(std::string prefix, std::string suffix)
{
   smurff::matrix_io::eigen::write_matrix(prefix + "-U" + std::to_string(mode) + "-latentmean" + suffix, mu);
}

void NormalPrior::restore(std::string prefix, std::string suffix)
{
   smurff::matrix_io::eigen::read_matrix(prefix + "-U" + std::to_string(mode) + "-latentmean" + suffix, mu);
   initUU();
}

std::ostream &NormalPrior::status(std::ostream &os, std::string indent) const
{
   os << indent << name << ": mu = " <<  mu.norm() << std::endl;
   return os;
}

void NormalPrior::initUU()
{
    const int K = num_latent();
    Ucol.init(VectorXd::Zero(K));
    UUcol.init(MatrixXd::Zero(K, K));
    UUcol.local() = U() * U().transpose();
    Ucol.local() = U().rowwise().sum();
}

//macau Probit sample latents
/*
void BPMFPrior::sample_latents(ProbitNoise & noise, Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat,
   double mean_value, const Eigen::MatrixXd &samples, const int num_latent)
{
   const int N = U.cols();

   #pragma omp parallel for schedule(dynamic, 2)
   for(int n = 0; n < N; n++) {
   sample_latent_blas_probit(U, n, mat, mean_value, samples, mu, Lambda, num_latent);
   }
}
*/

//macau Tensor methods
/*
void BPMFPrior::sample_latents(ProbitNoise& noiseModel, TensorData & data,
                               std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent) {
  // TODO
  throw std::runtime_error("Unimplemented: sample_latents");
}
*/
/*
void BPMFPrior::sample_latents(double noisePrecision,
                               TensorData & data,
                               std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples,
                               const int mode,
                               const int num_latent) {
  auto& sparseMode = (*data.Y)[mode];
  auto& U = samples[mode];
  const int N = U->cols();
  VectorView<Eigen::MatrixXd> view(samples, mode);

#pragma omp parallel for schedule(dynamic, 2)
  for (int n = 0; n < N; n++) {
    sample_latent_tensor(U, n, sparseMode, view, data.mean_value, noisePrecision, mu, Lambda);
  }
}
*/

#include "NormalPrior.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <iomanip>

#include "mvnormal.h"
#include "session.h"
#include "chol.h"
#include "linop.h"
#include "Data.h"

using namespace Eigen;

using namespace smurff;


//  base class NormalPrior

NormalPrior::NormalPrior(BaseSession &m, int p, std::string name)
   : ILatentPrior(m, p, name)
{

}

//method is nearly identical


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

//new method

const Eigen::VectorXd NormalPrior::getMu(int) const
{
    return mu;
}

//this method is ok except that
//CondNormalWishart now uses more arguments (cov and sum)

void NormalPrior::sample_latents()
{
   //this will go in cycle nad call sample_latent(n)
   ILatentPrior::sample_latents();

   const int N = num_cols();

   //this are new variables required for CondNormalWishart ?
   const auto cov = UUcol.combine_and_reset();
   const auto sum = Ucol.combine_and_reset();

   //this corresponds to update_prior - tie(mu, Lambda) = CondNormalWishart(U, mu0, b0, WI, df);
   std::tie(mu, Lambda) = CondNormalWishart(N, cov / N, sum / N, mu0, b0, WI, df);
}

//this method should correspond to sample_latent_blas from GlobalPrior
//previously it was called as:
/*
template<class FType>
void MacauPrior<FType>::sample_latents(Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat, double mean_value,
                    const Eigen::MatrixXd &samples, double alpha, const int num_latent)
{
#pragma omp parallel for schedule(dynamic, 2)
  for(int n = 0; n < U.cols(); n++) {
    // TODO: try moving mu + Uhat.col(n) inside sample_latent for speed
    sample_latent_blas(U, n, mat, mean_value, samples, alpha, mu + Uhat.col(n), Lambda, num_latent);
  }
}
*/

void NormalPrior::sample_latent(int n)
{
   const auto &mu_u = getMu(n);

   VectorNd &rr = rrs.local();
   MatrixNNd &MM = MMs.local();

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

//macau
/*
void sample_latent_blas(MatrixXd &s, int mm, const SparseMatrix<double> &mat, double mean_rating,
    const MatrixXd &samples, double alpha, const VectorXd &mu_u, const MatrixXd &Lambda_u,
    const int num_latent)
{
  MatrixXd MM = Lambda_u;
  VectorXd rr = VectorXd::Zero(num_latent);
  for (SparseMatrix<double>::InnerIterator it(mat, mm); it; ++it) {
    auto col = samples.col(it.row());
    MM.triangularView<Eigen::Lower>() += alpha * col * col.transpose();
    rr.noalias() += col * ((it.value() - mean_rating) * alpha);
  }

  Eigen::LLT<MatrixXd> chol = MM.llt();
  if(chol.info() != Eigen::Success) {
    throw std::runtime_error("Cholesky Decomposition failed!");
  }

  rr.noalias() += Lambda_u * mu_u;
  chol.matrixL().solveInPlace(rr);
  for (int i = 0; i < num_latent; i++) {
    rr[i] += randn0();
  }
  chol.matrixU().solveInPlace(rr);
  s.col(mm).noalias() = rr;
}
*/

//macau - not used anywhere. not sure if it is needed
//maybe current implementation of sample latents in priors is build upon this function?
/*
void sample_latent(MatrixXd &s, int mm, const SparseMatrix<double> &mat, double mean_rating,
    const MatrixXd &samples, double alpha, const VectorXd &mu_u, const MatrixXd &Lambda_u,
    const int num_latent)
{
  // TODO: add cholesky update version
  MatrixXd MM = MatrixXd::Zero(num_latent, num_latent);
  VectorXd rr = VectorXd::Zero(num_latent);
  for (SparseMatrix<double>::InnerIterator it(mat, mm); it; ++it) {
    auto col = samples.col(it.row());
    MM.noalias() += col * col.transpose();
    rr.noalias() += col * ((it.value() - mean_rating) * alpha);
  }

  Eigen::LLT<MatrixXd> chol = (Lambda_u + alpha * MM).llt();
  if(chol.info() != Eigen::Success) {
    throw std::runtime_error("Cholesky Decomposition failed!");
  }

  rr.noalias() += Lambda_u * mu_u;
  chol.matrixL().solveInPlace(rr);
  for (int i = 0; i < num_latent; i++) {
    rr[i] += randn0();
  }
  chol.matrixU().solveInPlace(rr);
  s.col(mm).noalias() = rr;
}
*/

//method is identical

void NormalPrior::save(std::string prefix, std::string suffix)
{
   smurff::matrix_io::eigen::write_matrix(prefix + "-U" + std::to_string(mode) + "-latentmean" + suffix, mu);
}

//new method

void NormalPrior::restore(std::string prefix, std::string suffix)
{
   smurff::matrix_io::eigen::read_matrix(prefix + "-U" + std::to_string(mode) + "-latentmean" + suffix, mu);
   initUU();
}

//new method

std::ostream &NormalPrior::status(std::ostream &os, std::string indent) const
{
   os << indent << name << ": mu = " <<  mu.norm() << std::endl;
   return os;
}

//new method

void NormalPrior::initUU()
{
    const int K = num_latent();
    Ucol.init(VectorNd::Zero(K));
    UUcol.init(MatrixNNd::Zero(K, K));
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
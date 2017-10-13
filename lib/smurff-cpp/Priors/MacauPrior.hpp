#pragma once

#include <memory>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <IO/MatrixIO.h>
#include <Utils/linop.h>

#include "mvnormal.h"
#include "model.h"

#include "NormalPrior.h"
#include "SparseDoubleFeat.h"
#include "SparseFeat.h"

namespace smurff {

//previously MacauPrior was not inheriting from NormalPrior
//that is why some of fields (related to NormalPrior) are now removed from this class

//sample_beta method is now virtual. because we override it in MPIMacauPrior
//we also have this method in MacauOnePrior but it is not virtual
//maybe make it virtual?

//why remove update_prior method ?

/// Prior with side information
template<class FType>
class MacauPrior : public NormalPrior
{
public:
   Eigen::MatrixXd Uhat;
   std::unique_ptr<FType> F;  /* side information */
   Eigen::MatrixXd FtF;       /* F'F */
   Eigen::MatrixXd beta;      /* link matrix */
   bool use_FtF;
   double lambda_beta;
   double lambda_beta_mu0; /* Hyper-prior for lambda_beta */
   double lambda_beta_nu0; /* Hyper-prior for lambda_beta */

   double tol = 1e-6;

public:
   //new method
   MacauPrior(BaseSession& m, int p)
      : NormalPrior(m, p, "MacauPrior")
   {

   }

   virtual ~MacauPrior() {}

   //method is identical
   //this was previously part of init method

   void init() override
   {
      //this was previously part of init method but now has moved to NormalPrior init
      NormalPrior::init();

      assert((F->rows() == num_cols()) && "Number of rows in train must be equal to number of rows in features");

      if (use_FtF)
      {
         FtF.resize(F->cols(), F->cols());
         At_mul_A(FtF, *F);
      }

      Uhat.resize(this->num_latent(), F->rows());
      Uhat.setZero();

      beta.resize(this->num_latent(), F->cols());
      beta.setZero();
   }

   //this method looks ok
   //however not sure about CondNormalWishart
   //it is already called in NormalPrior
   //why call again ?

   void sample_latents() override
   {
      //sample_latents implementation has moved to NormalPrior

      NormalPrior::sample_latents();

      //this was previously part of update_prior

      // residual (Uhat is later overwritten):
      Uhat.noalias() = U() - Uhat;
      Eigen::MatrixXd BBt = A_mul_At_combo(beta);

      // sampling Gaussian
      std::tie(this->mu, this->Lambda) = CondNormalWishart(Uhat, this->mu0, this->b0, this->WI + lambda_beta * BBt, this->df + beta.cols());
      sample_beta();
      compute_uhat(Uhat, *F, beta);
      lambda_beta = sample_lambda_beta(beta, this->Lambda, lambda_beta_nu0, lambda_beta_mu0);
   }

   //method is identical
   //this was previously part of init method

   void addSideInfo(std::unique_ptr<FType> &Fmat, bool comp_FtF = false)
   {
      // side information
      F = std::move(Fmat);
      use_FtF = comp_FtF;

      // initial value (should be determined automatically)
      lambda_beta = 5.0;
      // Hyper-prior for lambda_beta (mean 1.0, var of 1e+3):
      lambda_beta_mu0 = 1.0;
      lambda_beta_nu0 = 1e-3;
   }

   //new method

   double getLinkLambda()
   {
      return lambda_beta;
   }

   //new method

   const Eigen::VectorXd getMu(int n) const override
   {
      return this->mu + Uhat.col(n);
   }

   //this method was previously part of sample_beta
   //this method is identical except that is was not parallel
   //MatrixXd Ft_y = A_mul_B(tmp, *F) + sqrt(lambda_beta) * MvNormal_prec_omp(Lambda, num_feat);

   void compute_Ft_y_omp(Eigen::MatrixXd& Ft_y)
   {
      const int num_feat = beta.cols();

      // Ft_y = (U .- mu + Normal(0, Lambda^-1)) * F + sqrt(lambda_beta) * Normal(0, Lambda^-1)
      // Ft_y is [ D x F ] matrix
      Eigen::MatrixXd tmp = (U() + MvNormal_prec_omp(Lambda, num_cols())).colwise() - mu;
      Ft_y = A_mul_B(tmp, *F);
      Eigen::MatrixXd tmp2 = MvNormal_prec_omp(Lambda, num_feat);

      #pragma omp parallel for schedule(static)
      for (int f = 0; f < num_feat; f++)
      {
         for (int d = 0; d < num_latent(); d++)
         {
            Ft_y(d, f) += sqrt(lambda_beta) * tmp2(d, f);
         }
      }
   }

   //this method is identical

   // Update beta and Uhat
   virtual void sample_beta()
   {
      if (use_FtF)
         sample_beta_direct();
      else
         sample_beta_cg();
   }

   //new method

   void setLambdaBeta(double lb)
   {
      lambda_beta = lb;
   }

   //new method

   void setTol(double t)
   {
      tol = t;
   }

   //method is identical
   //part has now moved to NormalPrior

   void save(std::string prefix, std::string suffix) override
   {
      NormalPrior::save(prefix, suffix);
      prefix += "-F" + std::to_string(mode);
      smurff::matrix_io::eigen::write_matrix(prefix + "-link" + suffix, this->beta);
   }

   //new method

   void restore(std::string prefix, std::string suffix) override
   {
      NormalPrior::restore(prefix, suffix);
      prefix += "-F" + std::to_string(mode);
      smurff::matrix_io::eigen::read_matrix(prefix + "-link" + suffix, this->beta);
   }

private:

   //new method

   std::ostream &printSideInfo(std::ostream &os, const SparseDoubleFeat &F)
   {
      os << "SparseDouble [" << F.rows() << ", " << F.cols() << "]\n";
      return os;
   }

   //new method

   std::ostream &printSideInfo(std::ostream &os, const Eigen::MatrixXd &F)
   {
      os << "DenseDouble [" << F.rows() << ", " << F.cols() << "]\n";
      return os;
   }

   //new method

   std::ostream &printSideInfo(std::ostream &os, const SparseFeat &F)
   {
      os << "SparseBinary [" << F.rows() << ", " << F.cols() << "]\n";
      return os;
   }

public:

   //new method

   std::ostream &info(std::ostream &os, std::string indent) override
   {
      NormalPrior::info(os, indent);
      os << indent << " SideInfo: "; printSideInfo(os, *F);
      os << indent << " Method: " << (use_FtF ? "Cholesky Decomposition" : "CG Solver") << "\n";
      os << indent << " Tol: " << tol << "\n";
      os << indent << " LambdaBeta: " << lambda_beta << "\n";
      return os;
   }

   //beta norm was previouly part of getLinkNorm

   std::ostream &status(std::ostream &os, std::string indent) const override
   {
      os << indent << "  " << name << ": Beta = " << beta.norm() << "\n";
      return os;
   }

private:

   //this method is identical except that:
   //compute_Ft_y_omp was previously part of this method

   //diagonal was calculated as:
   /*
   for (int i = 0; i < K.cols(); i++)
   {
      K(i,i) += lambda_beta;
   }
   */

   // direct method
   void sample_beta_direct()
   {
      Eigen::MatrixXd Ft_y;
      this->compute_Ft_y_omp(Ft_y);

      Eigen::MatrixXd K(FtF.rows(), FtF.cols());
      K.triangularView<Eigen::Lower>() = FtF;
      K.diagonal().array() += lambda_beta;
      chol_decomp(K);
      chol_solve_t(K, Ft_y);
      beta = Ft_y;
   }

   //this method is identical except that:
   //compute_Ft_y_omp was previously part of this method

   // BlockCG solver
   void sample_beta_cg()
   {
      Eigen::MatrixXd Ft_y;
      this->compute_Ft_y_omp(Ft_y);

      solve_blockcg(beta, *F, lambda_beta, Ft_y, tol, 32, 8);
   }

public:

   //method is identical

   static std::pair<double,double> posterior_lambda_beta(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu)
   {
      const int D = beta.rows();
      Eigen::MatrixXd BB(D, D);
      A_mul_At_combo(BB, beta);
      double nux = nu + beta.rows() * beta.cols();
      double mux = mu * nux / (nu + mu * (BB.selfadjointView<Eigen::Lower>() * Lambda_u).trace() );
      double b   = nux / 2;
      double c   = 2 * mux / nux;
      return std::make_pair(b, c);
   }

   //method is identical

   static double sample_lambda_beta(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu)
   {
      auto gamma_post = posterior_lambda_beta(beta, Lambda_u, nu, mu);
      return rgamma(gamma_post.first, gamma_post.second);
   }
};

// specialization for dense matrices --> always direct method */
template<>
void MacauPrior<Eigen::Matrix<double, -1, -1, 0, -1, -1>>::sample_beta_cg();

}

//==========================

//macau Probit sample latents
/*
template<class FType>
void MacauPrior<FType>::sample_latents(ProbitNoise & noise, Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat,
                                       double mean_value, const Eigen::MatrixXd &samples, const int num_latent) {
    const int N = U.cols();
#pragma omp parallel for schedule(dynamic, 2)
  for(int n = 0; n < N; n++) {
    // TODO: try moving mu + Uhat.col(n) inside sample_latent for speed
    sample_latent_blas_probit(U, n, mat, mean_value, samples, mu + Uhat.col(n), Lambda, num_latent);
  }
}
*/

//macau Tensor method
/*
template<class FType>
void MacauPrior<FType>::sample_latents(ProbitNoise& noiseModel, TensorData & data,
                               std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent) {
  // TODO:
}
*/

//macau Tensor method
/*
template<class FType>
void MacauPrior<FType>::sample_latents(double noisePrecision,
                                       TensorData & data,
                                       std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples,
                                       int mode,
                                       const int num_latent) {
  auto& sparseMode = (*data.Y)[mode];
  auto& U = samples[mode];
  const int N = U->cols();
  VectorView<Eigen::MatrixXd> view(samples, mode);

#pragma omp parallel for schedule(dynamic, 2)
  for (int n = 0; n < N; n++) {
    Eigen::VectorXd mu2 = mu + Uhat.col(n);
    sample_latent_tensor(U, n, sparseMode, view, data.mean_value, noisePrecision, mu2, Lambda);
  }
}
*/
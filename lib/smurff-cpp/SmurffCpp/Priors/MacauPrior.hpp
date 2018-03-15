#pragma once

#include <memory>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <SmurffCpp/IO/MatrixIO.h>
#include <SmurffCpp/IO/GenericIO.h>

#include <SmurffCpp/Utils/linop.h>
#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/Utils/Error.h>

#include <SmurffCpp/Priors/NormalPrior.h>
#include <SmurffCpp/SparseDoubleFeat.h>
#include <SmurffCpp/SparseFeat.h>

namespace smurff {

//sample_beta method is now virtual. because we override it in MPIMacauPrior
//we also have this method in MacauOnePrior but it is not virtual
//maybe make it virtual?

/// Prior with side information
template<class FType>
class MacauPrior : public NormalPrior
{
public:
   typedef FType SideInfo;

   Eigen::MatrixXd Uhat;
   Eigen::MatrixXd FtF;       // F'F
   Eigen::MatrixXd beta;      // link matrix
   Eigen::MatrixXd HyperU, HyperU2;
   Eigen::MatrixXd Ft_y;
   
   bool enable_beta_precision_sampling;
   double beta_precision_mu0; // Hyper-prior for beta_precision
   double beta_precision_nu0; // Hyper-prior for beta_precision

   std::vector<std::shared_ptr<FType> > side_info_values;
   std::vector<double> beta_precision_values;
   std::vector<bool> direct_values;
   std::vector<double> tol_values;

   //these must be removed
   std::shared_ptr<FType> Features;  // side information
   bool use_FtF;
   double beta_precision;
   double tol = 1e-6;

private:
   MacauPrior()
      : NormalPrior(){}

public:
   MacauPrior(std::shared_ptr<BaseSession> session, uint32_t mode)
      : NormalPrior(session, mode, "MacauPrior")
   {
      beta_precision = MacauPriorConfig::BETA_PRECISION_DEFAULT_VALUE;
      tol = MacauPriorConfig::TOL_DEFAULT_VALUE;

      enable_beta_precision_sampling = Config::ENABLE_BETA_PRECISION_SAMPLING_DEFAULT_VALUE;
   }

   virtual ~MacauPrior() {}

   void init() override
   {
      NormalPrior::init();

      THROWERROR_ASSERT_MSG(Features->rows() == num_cols(), "Number of rows in train must be equal to number of rows in features");

      if (use_FtF)
      {
         FtF.resize(Features->cols(), Features->cols());
         smurff::linop::At_mul_A(FtF, *Features);
      }

      Uhat.resize(this->num_latent(), Features->rows());
      Uhat.setZero();

      beta.resize(this->num_latent(), Features->cols());
      beta.setZero();
   }

   void update_prior() override
   {
      // residual (Uhat is later overwritten):
      Uhat.noalias() = U() - Uhat;
      Eigen::MatrixXd BBt = smurff::linop::A_mul_At_combo(beta);

      // sampling Gaussian
      std::tie(this->mu, this->Lambda) = CondNormalWishart(Uhat, this->mu0, this->b0, this->WI + beta_precision * BBt, this->df + beta.cols());
      sample_beta();
      smurff::linop::compute_uhat(Uhat, *Features, beta);

      if(enable_beta_precision_sampling)
         beta_precision = sample_beta_precision(beta, this->Lambda, beta_precision_nu0, beta_precision_mu0);
   }

   const Eigen::VectorXd getMu(int n) const override
   {
      return this->mu + Uhat.col(n);
   }

   void compute_Ft_y_omp(Eigen::MatrixXd& Ft_y)
   {
      const int num_feat = beta.cols();

      // Ft_y = (U .- mu + Normal(0, Lambda^-1)) * F + std::sqrt(beta_precision) * Normal(0, Lambda^-1)
      // Ft_y is [ D x F ] matrix
      HyperU = (U() + MvNormal_prec_omp(Lambda, num_cols())).colwise() - mu;

      Ft_y = smurff::linop::A_mul_B(HyperU, *Features);
      HyperU2 = MvNormal_prec_omp(Lambda, num_feat);

      #pragma omp parallel for schedule(static)
      for (int f = 0; f < num_feat; f++)
      {
         for (int d = 0; d < num_latent(); d++)
         {
            Ft_y(d, f) += std::sqrt(beta_precision) * HyperU2(d, f);
         }
      }
   }

   // Update beta and Uhat
   virtual void sample_beta()
   {
      if (use_FtF)
         sample_beta_direct();
      else
         sample_beta_cg();
   }

public:

   void addSideInfo(std::shared_ptr<FType>& side_info, bool direct = false)
   {
      //FIXME: remove old code

      // side information
      Features = side_info;
      use_FtF = direct;

      //FIXME: this code should push multiple side info items that are passed?

      // side information
      side_info_values.push_back(side_info);
      direct_values.push_back(direct);

      // Hyper-prior for beta_precision (mean 1.0, var of 1e+3):
      beta_precision_mu0 = 1.0;
      beta_precision_nu0 = 1e-3;
   }

   void setBetaPrecisionValues(const std::vector<std::shared_ptr<MacauPriorConfigItem> >& config_items)
   {
      beta_precision_values.clear();

      for (auto& item : config_items)
         beta_precision_values.push_back(item->getBetaPrecision());

      //FIXME: remove old code
      beta_precision = config_items.front()->getBetaPrecision();
   }

   void setTolValues(const std::vector<std::shared_ptr<MacauPriorConfigItem> >& config_items)
   {
      tol_values.clear();

      for (auto& item : config_items)
         tol_values.push_back(item->getTol());

      //FIXME: remove old code
      tol = config_items.front()->getTol();
   }

   void setEnableBetaPrecisionSampling(bool value)
   {
      enable_beta_precision_sampling = value;
   }

public:

   void save(std::shared_ptr<const StepFile> sf) const override
   {
      NormalPrior::save(sf);

      std::string path = sf->getPriorFileName(m_mode);
      smurff::matrix_io::eigen::write_matrix(path, this->beta);
   }

   void restore(std::shared_ptr<const StepFile> sf) override
   {
      NormalPrior::restore(sf);

      std::string path = sf->getPriorFileName(m_mode);

      THROWERROR_FILE_NOT_EXIST(path);

      smurff::matrix_io::eigen::read_matrix(path, this->beta);
   }

private:

   std::ostream &printSideInfo(std::ostream &os, const SparseDoubleFeat &F)
   {
      os << "SparseDouble [" << F.rows() << ", " << F.cols() << "]" << std::endl;
      return os;
   }

   std::ostream &printSideInfo(std::ostream &os, const Eigen::MatrixXd &F)
   {
      os << "DenseDouble [" << F.rows() << ", " << F.cols() << "]" << std::endl;
      return os;
   }

   std::ostream &printSideInfo(std::ostream &os, const SparseFeat &F)
   {
      os << "SparseBinary [" << F.rows() << ", " << F.cols() << "]" << std::endl;
      return os;
   }

public:

   std::ostream &info(std::ostream &os, std::string indent) override
   {
      NormalPrior::info(os, indent);
      os << indent << " SideInfo: "; printSideInfo(os, *Features);
      os << indent << " Method: " << (use_FtF ? "Cholesky Decomposition" : "CG Solver") << std::endl;
      os << indent << " Tol: " << tol << std::endl;
      os << indent << " BetaPrecision: " << beta_precision << std::endl;
      return os;
   }

   std::ostream &status(std::ostream &os, std::string indent) const override
   {
      os << indent << m_name << ": " << std::endl;
      indent += "  ";
      os << indent << "FtF          = " << FtF.norm() << std::endl;
      os << indent << "HyperU       = " << HyperU.norm() << std::endl;
      os << indent << "HyperU2      = " << HyperU2.norm() << std::endl;
      os << indent << "Beta         = " << beta.norm() << std::endl;
      os << indent << "beta_precision  = " << beta_precision << std::endl;
      os << indent << "Ft_y         = " << Ft_y.norm() << std::endl;
      return os;
   }

private:

   // direct method
   void sample_beta_direct()
   {
      this->compute_Ft_y_omp(Ft_y);

      Eigen::MatrixXd K(FtF.rows(), FtF.cols());
      K.triangularView<Eigen::Lower>() = FtF;
      K.diagonal().array() += beta_precision;
      chol_decomp(K);
      chol_solve_t(K, Ft_y);
      beta = Ft_y;
   }

   // BlockCG solver
   void sample_beta_cg();

public:

   static std::pair<double,double> posterior_beta_precision(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu)
   {
      const int D = beta.rows();
      Eigen::MatrixXd BB(D, D);
      smurff::linop::A_mul_At_combo(BB, beta);
      double nux = nu + beta.rows() * beta.cols();
      double mux = mu * nux / (nu + mu * (BB.selfadjointView<Eigen::Lower>() * Lambda_u).trace() );
      double b   = nux / 2;
      double c   = 2 * mux / nux;
      return std::make_pair(b, c);
   }

   static double sample_beta_precision(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu)
   {
      auto gamma_post = posterior_beta_precision(beta, Lambda_u, nu, mu);
      return rgamma(gamma_post.first, gamma_post.second);
   }
};

template<class FType>
void MacauPrior<FType>::sample_beta_cg()
{
   Eigen::MatrixXd Ft_y;
   this->compute_Ft_y_omp(Ft_y);

   smurff::linop::solve_blockcg(beta, *Features, beta_precision, Ft_y, tol, 32, 8);
}

// specialization for dense matrices --> always direct method
template<>
void MacauPrior<Eigen::Matrix<double, -1, -1, 0, -1, -1>>::sample_beta_cg();

}

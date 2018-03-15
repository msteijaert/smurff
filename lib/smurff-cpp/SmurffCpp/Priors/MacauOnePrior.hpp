#pragma once

#include <memory>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <SmurffCpp/DataMatrices/ScarceMatrixData.h>
#include <SmurffCpp/IO/MatrixIO.h>
#include <SmurffCpp/IO/GenericIO.h>

#include <SmurffCpp/Utils/linop.h>

#include <SmurffCpp/Priors/NormalOnePrior.h>

namespace smurff {

//Why remove init method and put everything in constructor if we have
//init method in other priors and the other method addSideInfo which we use in pair

template<class FType>
class MacauOnePrior : public NormalOnePrior
{
public:
   typedef FType SideInfo;

   Eigen::MatrixXd Uhat;

   Eigen::VectorXd F_colsq;   // sum-of-squares for every feature (column)

   Eigen::MatrixXd beta;      // link matrix
   
   Eigen::VectorXd beta_precision;

   bool enable_beta_precision_sampling;
   double beta_precision_a0; // Hyper-prior for beta_precision
   double beta_precision_b0; // Hyper-prior for beta_precision

   std::vector<std::shared_ptr<FType> > side_info_values;
   std::vector<double> beta_precision_values;

   //these must be removed
   std::shared_ptr<FType> Features;  // side information
   double bp0;

public:
   MacauOnePrior(std::shared_ptr<BaseSession> session, uint32_t mode)
      : NormalOnePrior(session, mode, "MacauOnePrior")
   {
      bp0 = MacauPriorConfig::BETA_PRECISION_DEFAULT_VALUE;

      enable_beta_precision_sampling = Config::ENABLE_BETA_PRECISION_SAMPLING_DEFAULT_VALUE;
   }

   void init() override
   {
      NormalOnePrior::init();

      // init SideInfo related
      Uhat = Eigen::MatrixXd::Constant(num_latent(), Features->rows(), 0.0);
      beta = Eigen::MatrixXd::Constant(num_latent(), Features->cols(), 0.0);

      // initial value (should be determined automatically)
      // Hyper-prior for beta_precision (mean 1.0):
      beta_precision = Eigen::VectorXd::Constant(num_latent(), bp0);
      beta_precision_a0 = 0.1;
      beta_precision_b0 = 0.1;
   }

   void update_prior() override
   {
      sample_mu_lambda(U());
      sample_beta(U());
      smurff::linop::compute_uhat(Uhat, *Features, beta);

      if(enable_beta_precision_sampling)
         sample_beta_precision();
   }
    
   const Eigen::VectorXd getMu(int n) const override
   {
      return this->mu + Uhat.col(n);
   }

public:
   void addSideInfo(std::shared_ptr<FType>& side_info, bool direct = false)
   {
      //FIXME: remove old code
      Features = side_info;

      //FIXME: this code should push multiple side info items that are passed?

      // side information
      side_info_values.push_back(side_info);

      F_colsq = smurff::linop::col_square_sum(*Features);
   }

   void setBetaPrecisionValues(const std::vector<std::shared_ptr<MacauPriorConfigItem> >& config_items)
   {
      beta_precision_values.clear();

      for (auto& item : config_items)
         beta_precision_values.push_back(item->getBetaPrecision());

      //FIXME: remove old code
      bp0 = config_items.front()->getBetaPrecision();
   }

   void setTolValues(const std::vector<std::shared_ptr<MacauPriorConfigItem> >& config_items)
   {
   }

   void setEnableBetaPrecisionSampling(bool value)
   {
      enable_beta_precision_sampling = value;
   }

public:

   //used in update_prior

   void sample_beta(const Eigen::MatrixXd &U)
   {
      // updating beta and beta_var
      const int nfeat = beta.cols();
      const int N = U.cols();
      const int blocksize = 4;

      Eigen::MatrixXd Z;

      #pragma omp parallel for private(Z) schedule(static, 1)
      for (int dstart = 0; dstart < num_latent(); dstart += blocksize)
      {
         const int dcount = std::min(blocksize, num_latent() - dstart);
         Z.resize(dcount, U.cols());

         for (int i = 0; i < N; i++)
         {
            for (int d = 0; d < dcount; d++)
            {
               int dx = d + dstart;
               Z(d, i) = U(dx, i) - mu(dx) - Uhat(dx, i);
            }
         }

         for (int f = 0; f < nfeat; f++)
         {
            Eigen::VectorXd zx(dcount), delta_beta(dcount), randvals(dcount);
            // zx = Z[dstart : dstart + dcount, :] * F[:, f]
            smurff::linop::At_mul_Bt(zx, *Features, f, Z);
            // TODO: check if sampling randvals for whole [nfeat x dcount] matrix works faster
            bmrandn_single( randvals );

            for (int d = 0; d < dcount; d++)
            {
               int dx = d + dstart;
               double A_df     = beta_precision(dx) + Lambda(dx,dx) * F_colsq(f);
               double B_df     = Lambda(dx,dx) * (zx(d) + beta(dx,f) * F_colsq(f));
               double A_inv    = 1.0 / A_df;
               double beta_new = B_df * A_inv + std::sqrt(A_inv) * randvals(d);
               delta_beta(d)   = beta(dx,f) - beta_new;

               beta(dx, f)     = beta_new;
            }
            // Z[dstart : dstart + dcount, :] += F[:, f] * delta_beta'
            smurff::linop::add_Acol_mul_bt(Z, *Features, f, delta_beta);
         }
      }
   }

   //used in update_prior

   void sample_mu_lambda(const Eigen::MatrixXd &U)
   {
      Eigen::MatrixXd WI(num_latent(), num_latent());
      WI.setIdentity();
      int N = U.cols();

      Eigen::MatrixXd Udelta(num_latent(), N);
      #pragma omp parallel for schedule(static)
      for (int i = 0; i < N; i++)
      {
         for (int d = 0; d < num_latent(); d++)
         {
            Udelta(d, i) = U(d, i) - Uhat(d, i);
         }
      }
      std::tie(mu, Lambda) = CondNormalWishart(Udelta, Eigen::VectorXd::Constant(num_latent(), 0.0), 2.0, WI, num_latent());
   }

   //used in update_prior

   void sample_beta_precision()
   {
      double beta_precision_a = beta_precision_a0 + beta.cols() / 2.0;
      Eigen::VectorXd beta_precision_b = Eigen::VectorXd::Constant(beta.rows(), beta_precision_b0);
      const int D = beta.rows();
      const int F = beta.cols();
      #pragma omp parallel
      {
         Eigen::VectorXd tmp(D);
         tmp.setZero();
         #pragma omp for schedule(static)
         for (int f = 0; f < F; f++)
         {
            for (int d = 0; d < D; d++)
            {
               tmp(d) += std::pow(beta(d, f), 2);
            }
         }
         #pragma omp critical
         {
            beta_precision_b += tmp / 2;
         }
      }
      for (int d = 0; d < D; d++)
      {
         beta_precision(d) = rgamma(beta_precision_a, 1.0 / beta_precision_b(d));
      }
   }

public:

   void save(std::shared_ptr<const StepFile> sf) const override
   {
      NormalOnePrior::save(sf);

      std::string path = sf->getPriorFileName(m_mode);
      smurff::matrix_io::eigen::write_matrix(path, beta);
   }

   void restore(std::shared_ptr<const StepFile> sf) override
   {
      NormalOnePrior::restore(sf);

      std::string path = sf->getPriorFileName(m_mode);

      THROWERROR_FILE_NOT_EXIST(path);

      smurff::matrix_io::eigen::read_matrix(path, beta);
   }

   std::ostream &status(std::ostream &os, std::string indent) const override
   {
      os << indent << "  " << m_name << ": Beta = " << beta.norm() << std::endl;
      return os;
   }
};

}

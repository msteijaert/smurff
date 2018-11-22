#include <viennacl/matrix.hpp>
#include <viennacl/linalg/lu.hpp>

#include "MacauPrior.h"

#include <SmurffCpp/IO/MatrixIO.h>
#include <SmurffCpp/IO/GenericIO.h>

#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/Utils/Error.h>
#include <SmurffCpp/Utils/counters.h>

#include <SmurffCpp/Utils/linop.h>

#include <ios>

using namespace smurff;

MacauPrior::MacauPrior()
    : NormalPrior()
{
}

MacauPrior::MacauPrior(std::shared_ptr<Session> session, uint32_t mode)
    : NormalPrior(session, mode, "MacauPrior")
{
    beta_precision = SideInfoConfig::BETA_PRECISION_DEFAULT_VALUE;
    tol = SideInfoConfig::TOL_DEFAULT_VALUE;

    enable_beta_precision_sampling = Config::ENABLE_BETA_PRECISION_SAMPLING_DEFAULT_VALUE;
}

MacauPrior::~MacauPrior()
{
}

void MacauPrior::init()
{
   NormalPrior::init();

   THROWERROR_ASSERT_MSG(Features->rows() == num_item(), "Number of rows in train must be equal to number of rows in features");

   if (use_FtF)
   {
      std::uint64_t dim = num_feat();
      FtF_plus_precision.resize(dim, dim);
      Features->At_mul_A(FtF_plus_precision);
      FtF_plus_precision.diagonal().array() += beta_precision;
   }

   Uhat.resize(num_latent(), Features->rows());
   Uhat.setZero();

   m_beta = std::make_shared<Eigen::MatrixXd>(num_latent(), num_feat());
   beta().setZero();

   BBt = beta() * beta().transpose();

   m_session->model().setLinkMatrix(m_mode, m_beta);
}

void MacauPrior::update_prior()
{
    /*
>> compute_uhat:                     0.5012     (12%) in        110
>> main:                             4.1396     (100%) in       1
>> rest of update_prior:             0.1684     (4%) in 110
>> sample hyper mu/Lambda:           0.3804     (9%) in 110
>> sample_beta:                      1.4927     (36%) in        110
>> sample_latents:                   3.8824     (94%) in        220
>> step:                             3.9824     (96%) in        111
>> update_prior:                     2.5436     (61%) in        110
*/
    COUNTER("update_prior");
    {
        COUNTER("rest of update_prior");


    }

    // sampling Gaussian
    {
        COUNTER("sample hyper mu/Lambda");
        //uses: U, Uhat
        // writes: Udelta
        // complexity: num_latent x num_items
        Udelta = U() - Uhat;
        // uses: Udelta
        // complexity: num_latent x num_items
        std::tie(mu, Lambda) = CondNormalWishart(Udelta, mu0, b0, WI + beta_precision * BBt, df + num_feat());
    }

    // uses: U, F
    // writes: Ft_y
    // uses: U, F
    // writes: Ft_y
    // complexity: num_latent x num_feat x num_item
    compute_Ft_y(Ft_y);

    sample_beta();

    {
        COUNTER("compute_uhat");
        // Uhat = beta * F
        // uses: beta, F
        // output: Uhat
        // complexity: num_feat x num_latent x num_item
        Features->compute_uhat(Uhat, beta());
    }

    if (enable_beta_precision_sampling)
    {
        // uses: beta
        // writes: FtF
        COUNTER("sample_beta_precision");
        double old_beta = beta_precision;
        beta_precision = sample_beta_precision(BBt, Lambda, beta_precision_nu0, beta_precision_mu0, beta().cols());
        FtF_plus_precision.diagonal().array() += beta_precision - old_beta;
   }
}

void MacauPrior::sample_beta()
{
    bool use_ViennaCL = true;

    COUNTER("sample_beta");
    if (use_FtF)
    {
        // uses: FtF, Ft_y, 
        // writes: m_beta
        // complexity: num_feat^3

        if (use_ViennaCL)
        {
            typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenRowMajor;

            using namespace viennacl;
            using namespace viennacl::linalg;

            //allocate
            matrix<double> vcl_FtF(FtF_plus_precision.rows(), FtF_plus_precision.cols());
            matrix<double> vcl_Ft_y_t(Ft_y.cols(), Ft_y.rows());

            //map
            Eigen::Map<EigenRowMajor> map_beta(beta().data(), beta().cols(), beta().rows());
            Eigen::Map<EigenRowMajor> map_Ft_y(Ft_y.data(), Ft_y.cols(), Ft_y.rows());

            //copy
            copy(FtF_plus_precision, vcl_FtF);
            copy(map_Ft_y, vcl_Ft_y_t);

            // calculate
            lu_factorize(vcl_FtF);
            lu_substitute(vcl_FtF, vcl_Ft_y_t);

            //copy back
            copy(vcl_Ft_y_t, map_beta);

        }
        else
        {
            beta() = FtF_plus_precision.llt().solve(Ft_y.transpose()).transpose();
        }

    } 
    else
    {
        // uses: Features, beta_precision, Ft_y, 
        // writes: beta
        // complexity: num_feat x num_feat x num_iter
        blockcg_iter = Features->solve_blockcg(beta(), beta_precision, Ft_y, tol, 32, 8, throw_on_cholesky_error);
    }
    // complexity: num_feat x num_feat x num_latent
    BBt = beta() * beta().transpose();
}

const Eigen::VectorXd MacauPrior::getMu(int n) const
{
   return mu + Uhat.col(n);
}

void MacauPrior::compute_Ft_y(Eigen::MatrixXd& Ft_y)
{
    COUNTER("compute Ft_y");
   // Ft_y = (U .- mu + Normal(0, Lambda^-1)) * F + std::sqrt(beta_precision) * Normal(0, Lambda^-1)
   // Ft_y is [ num_latent x num_feat ] matrix

   //HyperU: num_latent x num_item
   HyperU = (U() + MvNormal_prec(Lambda, num_item())).colwise() - mu;
   Ft_y = Features->A_mul_B(HyperU); // num_latent x num_feat

   //--  add beta_precision 
   HyperU2 = MvNormal_prec(Lambda, num_feat()); // num_latent x num_feat
   Ft_y += std::sqrt(beta_precision) * HyperU2;
}

void MacauPrior::addSideInfo(const std::shared_ptr<ISideInfo>& side, double bp, double to, bool di, bool sa, bool th)
{
    Features = side;
    beta_precision = bp;
    tol = to;
    use_FtF = di;
    enable_beta_precision_sampling = sa;
    throw_on_cholesky_error = th;

    // Hyper-prior for beta_precision (mean 1.0, var of 1e+3):
    beta_precision_mu0 = 1.0;
    beta_precision_nu0 = 1e-3;
}

bool MacauPrior::save(std::shared_ptr<const StepFile> sf) const
{
    NormalPrior::save(sf);

   std::string path = sf->makeLinkMatrixFileName(m_mode);
   smurff::matrix_io::eigen::write_matrix(path, beta());

    return true;
}

void MacauPrior::restore(std::shared_ptr<const StepFile> sf)
{
    NormalPrior::restore(sf);

    std::string path = sf->getLinkMatrixFileName(m_mode);

   smurff::matrix_io::eigen::read_matrix(path, beta());
}

std::ostream& MacauPrior::info(std::ostream &os, std::string indent)
{
   NormalPrior::info(os, indent);
   os << indent << " SideInfo: ";
   Features->print(os);
   os << indent << " Method: ";
   if (use_FtF)
   {
      os << "Cholesky Decomposition";
      double needs_gb = (double)num_feat() / 1024. * (double)num_feat() / 1024. / 1024.;
      if (needs_gb > 1.0) os << " (needing " << needs_gb << " GB of memory)";
      os << std::endl;
   } else {
      os << "CG Solver with tolerance: " << std::scientific << tol << std::fixed << std::endl;
   }
   os << indent << " BetaPrecision: ";
   if (enable_beta_precision_sampling)
   {
       os << "sampled around ";
   }
   else
   {
       os << "fixed at ";
   }
   os << beta_precision << std::endl;
   return os;
}

std::ostream &MacauPrior::status(std::ostream &os, std::string indent) const
{
   os << indent << m_name << ": " << std::endl;
   indent += "  ";
   os << indent << "blockcg iter = " << blockcg_iter << std::endl;
   os << indent << "FtF_plus_precision= " << FtF_plus_precision.norm() << std::endl;
   os << indent << "HyperU       = " << HyperU.norm() << std::endl;
   os << indent << "HyperU2      = " << HyperU2.norm() << std::endl;
   os << indent << "Beta         = " << beta().norm() << std::endl;
   os << indent << "beta_precision  = " << beta_precision << std::endl;
   os << indent << "Ft_y         = " << Ft_y.norm() << std::endl;
   return os;
}

std::pair<double, double> MacauPrior::posterior_beta_precision(const Eigen::MatrixXd & BBt, Eigen::MatrixXd & Lambda_u, double nu, double mu, int N)
{
   double nux = nu + N * BBt.cols();
   double mux = mu * nux / (nu + mu * (BBt.selfadjointView<Eigen::Lower>() * Lambda_u).trace());
   double b = nux / 2;
   double c = 2 * mux / nux;
   return std::make_pair(b, c);
}

double MacauPrior::sample_beta_precision(const Eigen::MatrixXd & BBt, Eigen::MatrixXd & Lambda_u, double nu, double mu, int N)
{
   auto gamma_post = posterior_beta_precision(BBt, Lambda_u, nu, mu, N);
   return rgamma(gamma_post.first, gamma_post.second);
}

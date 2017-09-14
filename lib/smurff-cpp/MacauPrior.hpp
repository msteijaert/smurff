#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

#include "mvnormal.h"
#include "linop.h"
#include "model.h"

#include "NormalPrior.h"

namespace smurff {

//previously MacauPrior was not inheriting from NormalPrior
//that is why some of fields (related to NormalPrior) are now removed from this class

//sample_beta method is now virtual. because we override it in MPIMacauPrior
//we also have this method in MacauOnePrior but it is not virtual
//maybe make it virtual?

//why remove update_prior method ?

std::pair<double,double> posterior_lambda_beta(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu);
double sample_lambda_beta(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu);

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
    MacauPrior(BaseSession &m, int p);
    virtual ~MacauPrior() {}
    void init() override;

    void sample_latents() override;
            
    void addSideInfo(std::unique_ptr<FType> &Fmat, bool comp_FtF = false);

    double getLinkLambda();
    const Eigen::VectorXd getMu(int n) const override;

    void compute_Ft_y_omp(Eigen::MatrixXd &);
    virtual void sample_beta();
    void setLambdaBeta(double lb);

    void setTol(double t);
    void save(std::string prefix, std::string suffix) override;
    void restore(std::string prefix, std::string suffix) override;
    std::ostream &info(std::ostream &os, std::string indent) override;
    std::ostream &status(std::ostream &os, std::string indent) const override;

  private:
    void sample_beta_direct();
    void sample_beta_cg();
};

}
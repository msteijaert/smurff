#ifndef MACAUPRIOR_H
#define MACAUPRIOR_H

#include <Eigen/Dense>
#include <memory>

namespace Macau {

std::pair<double,double> posterior_lambda_beta(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu);
double sample_lambda_beta(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu);

}

#endif


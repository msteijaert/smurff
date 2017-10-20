#pragma once

#define EIGEN_RUNTIME_NO_MALLOC
//#define EIGEN_DONT_PARALLELIZE 1

#include <map>

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace smurff
{
   double randn0();
   double randn(double = .0);
   
   void bmrandn(double* x, long n);
   void bmrandn(Eigen::MatrixXd & X);
   
   double bmrandn_single();
   void bmrandn_single(double* x, long n);
   void bmrandn_single(Eigen::VectorXd & x);
   void bmrandn_single(Eigen::MatrixXd & X);
   
   void init_bmrng();
   void init_bmrng(int seed);
   
   double rand_unif();
   double rand_unif(double low, double high);
   
   double rgamma(double shape, double scale);
   
   // return a random matrix of size n, m
   
   auto nrandn(int n) -> decltype(Eigen::VectorXd::NullaryExpr(n, std::cref(randn)) ); 
   auto nrandn(int n, int m) -> decltype(Eigen::ArrayXXd::NullaryExpr(n, m, std::ptr_fun(randn)) );
   
   // Wishart distribution
   
   std::pair<Eigen::VectorXd, Eigen::MatrixXd> NormalWishart(const Eigen::VectorXd & mu, double kappa, const Eigen::MatrixXd & T, double nu);
   std::pair<Eigen::VectorXd, Eigen::MatrixXd> CondNormalWishart(const Eigen::MatrixXd &U, const Eigen::VectorXd &mu, const double kappa, const Eigen::MatrixXd &T, const int nu);
   std::pair<Eigen::VectorXd, Eigen::MatrixXd> CondNormalWishart(const int N, const Eigen::MatrixXd &NS, const Eigen::VectorXd &NU, const Eigen::VectorXd &mu, const double kappa, const Eigen::MatrixXd &T, const int nu);
   
   // Multivariate normal gaussian

   Eigen::MatrixXd MvNormal_prec(const Eigen::MatrixXd & Lambda, int nn = 1);
   Eigen::MatrixXd MvNormal_prec(const Eigen::MatrixXd & Lambda, const Eigen::VectorXd & mean, int nn = 1);
   Eigen::MatrixXd MvNormal_prec_omp(const Eigen::MatrixXd & Lambda, int nn = 1);
   Eigen::MatrixXd MvNormal(const Eigen::MatrixXd covar, const Eigen::VectorXd mean, int nn = 1);
}

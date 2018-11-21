#pragma once

#include <map>

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace smurff
{
   float randn0();
   float randn(float = .0);
   
   void bmrandn(float* x, long n);
   void bmrandn(Eigen::MatrixXf & X);
   
   float bmrandn_single_thread();
   void bmrandn_single_thread(float* x, long n);
   void bmrandn_single_thread(Eigen::VectorXf & x);
   void bmrandn_single_thread(Eigen::MatrixXf & X);
   
   void init_bmrng();
   void init_bmrng(int seed);
   
   float rand_unif();
   float rand_unif(float low, float high);
   
   float rgamma(float shape, float scale);
   
   // return a random matrix of size n, m
   
   auto nrandn(int n) -> decltype(Eigen::VectorXf::NullaryExpr(n, std::cref(randn)) ); 
   auto nrandn(int n, int m) -> decltype(Eigen::ArrayXXf::NullaryExpr(n, m, std::ptr_fun(randn)) );
   
   // Wishart distribution
   
   std::pair<Eigen::VectorXf, Eigen::MatrixXf> NormalWishart(const Eigen::VectorXf & mu, float kappa, const Eigen::MatrixXf & T, float nu);
   std::pair<Eigen::VectorXf, Eigen::MatrixXf> CondNormalWishart(const Eigen::MatrixXf &U, const Eigen::VectorXf &mu, const float kappa, const Eigen::MatrixXf &T, const int nu);
   std::pair<Eigen::VectorXf, Eigen::MatrixXf> CondNormalWishart(const int N, const Eigen::MatrixXf &NS, const Eigen::VectorXf &NU, const Eigen::VectorXf &mu, const float kappa, const Eigen::MatrixXf &T, const int nu);
   
   // Multivariate normal gaussian

   Eigen::MatrixXf MvNormal_prec(const Eigen::MatrixXf & Lambda, int nn = 1);
   Eigen::MatrixXf MvNormal_prec(const Eigen::MatrixXf & Lambda, const Eigen::VectorXf & mean, int nn = 1);
   Eigen::MatrixXf MvNormal(const Eigen::MatrixXf covar, const Eigen::VectorXf mean, int nn = 1);
}

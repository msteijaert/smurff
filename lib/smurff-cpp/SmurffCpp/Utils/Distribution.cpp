#include "Distribution.h"

// From:
// http://stackoverflow.com/questions/6142576/sample-from-multivariate-normal-gaussian-distribution-in-c
 
#include <iostream>
#include <random>
#include <chrono>

#include <Eigen/Dense>

#include "utils.h"
#include "omp_util.h"

using namespace std;
using namespace Eigen;
using namespace std::chrono;

static thread_vector<std::mt19937> bmrngs;

double smurff::randn0()
{
   return smurff::bmrandn_single();
}

double smurff::randn(double) 
{
   return smurff::bmrandn_single();
}

void smurff::bmrandn(double* x, long n) 
{
   #pragma omp parallel 
   {
      std::uniform_real_distribution<double> unif(-1.0, 1.0);
      auto& bmrng = bmrngs.local();
      
      #pragma omp for schedule(static)
      for (long i = 0; i < n; i += 2) 
      {
         double x1, x2, w;
         do 
         {
           x1 = unif(bmrng);
           x2 = unif(bmrng);
           w = x1 * x1 + x2 * x2;
         } while ( w >= 1.0 );
   
         w = std::sqrt( (-2.0 * std::log( w ) ) / w );
         x[i] = x1 * w;

         if (i + 1 < n) 
         {
           x[i+1] = x2 * w;
         }
      }
   }
}
   
void smurff::bmrandn(MatrixXd & X) 
{
   long n = X.rows() * (long)X.cols();
   smurff::bmrandn(X.data(), n);
}

double smurff::bmrandn_single() 
{
   //TODO: add bmrng as input
   std::uniform_real_distribution<double> unif(-1.0, 1.0);
   auto& bmrng = bmrngs.local();
  
   double x1, x2, w;
   do 
   {
      x1 = unif(bmrng);
      x2 = unif(bmrng);
      w = x1 * x1 + x2 * x2;
   } while ( w >= 1.0 );

   w = std::sqrt( (-2.0 * std::log( w ) ) / w );
   return x1 * w;
}

// to be called within OpenMP parallel loop (also from serial code is fine)
void smurff::bmrandn_single(double* x, long n) 
{
   std::uniform_real_distribution<double> unif(-1.0, 1.0);
   auto& bmrng = bmrngs.local();

   for (long i = 0; i < n; i += 2) 
   {
      double x1, x2, w;

      do 
      {
         x1 = unif(bmrng);
         x2 = unif(bmrng);
         w = x1 * x1 + x2 * x2;
      } while ( w >= 1.0 );
 
      w = std::sqrt( (-2.0 * std::log( w ) ) / w );
      x[i] = x1 * w;

      if (i + 1 < n) 
      {
         x[i+1] = x2 * w;
      }
   }
}
  
void smurff::bmrandn_single(Eigen::VectorXd & x) 
{
   smurff::bmrandn_single(x.data(), x.size());
}
 
void smurff::bmrandn_single(MatrixXd & X) 
{
   long n = X.rows() * (long)X.cols();
   smurff::bmrandn_single(X.data(), n);
}

void smurff::init_bmrng() 
{
   auto ms = (duration_cast< milliseconds >(system_clock::now().time_since_epoch())).count();
   smurff::init_bmrng(ms);
}

void smurff::init_bmrng(int seed) 
{
    std::vector<std::mt19937> v;
    for (int i = 0; i < thread_limit(); i++)
    {
        v.push_back(std::mt19937(seed + i * 1999));
    }
    bmrngs.init(v);
}
   
double smurff::rand_unif() 
{
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    auto& bmrng = bmrngs.local();
    return unif(bmrng);
}
 
double smurff::rand_unif(double low, double high) 
{
   std::uniform_real_distribution<double> unif(low, high);
   auto& bmrng = bmrngs.local();
   return unif(bmrng);
}

// returns random number according to Gamma distribution
// with the given shape (k) and scale (theta). See wiki.
double smurff::rgamma(double shape, double scale) 
{
   std::gamma_distribution<double> gamma(shape, scale);
   return gamma(bmrngs.local());
}

auto smurff::nrandn(int n) -> decltype(VectorXd::NullaryExpr(n, std::cref(randn))) 
{
   return VectorXd::NullaryExpr(n, std::cref(randn));
}

auto smurff::nrandn(int n, int m) -> decltype(ArrayXXd::NullaryExpr(n, m, ptr_fun(randn)))
{
   return ArrayXXd::NullaryExpr(n, m, ptr_fun(randn)); 
}


MatrixXd WishartUnit(int m, int df)
{
   MatrixXd c(m,m);
   c.setZero();
   auto& rng = bmrngs.local();

   for ( int i = 0; i < m; i++ ) 
   {
      std::gamma_distribution<> gam(0.5*(df - i));
      c(i,i) = std::sqrt(2.0 * gam(rng));
      VectorXd r = smurff::nrandn(m-i-1);
      c.block(i,i+1,1,m-i-1) = r.transpose();
   }

   MatrixXd ret = c.transpose() * c;

   #ifdef TEST_MVNORMAL
   cout << "WISHART UNIT {\n" << endl;
   cout << "  m:\n" << m << endl;
   cout << "  df:\n" << df << endl;
   cout << "  ret;\n" << ret << endl;
   cout << "  c:\n" << c << endl;
   cout << "}\n" << ret << endl;
   #endif

   return ret;
}

MatrixXd Wishart(const MatrixXd &sigma, const int df)
{
   //  Get R, the upper triangular Cholesky factor of SIGMA.
   auto chol = sigma.llt();
   MatrixXd r = chol.matrixL();

   //  Get AU, a sample from the unit Wishart distribution.
   MatrixXd au = WishartUnit(sigma.cols(), df);

   //  Construct the matrix A = R' * AU * R.
   MatrixXd a = r * au * chol.matrixU();

   #ifdef TEST_MVNORMAL
   cout << "WISHART {\n" << endl;
   cout << "  sigma:\n" << sigma << endl;
   cout << "  r:\n" << r << endl;
   cout << "  au:\n" << au << endl;
   cout << "  df:\n" << df << endl;
   cout << "  a:\n" << a << endl;
   cout << "}\n" << endl;
   #endif

  return a;
}

// from julia package Distributions: conjugates/normalwishart.jl
std::pair<VectorXd, MatrixXd> smurff::NormalWishart(const VectorXd & mu, double kappa, const MatrixXd & T, double nu)
{
   MatrixXd Lam = Wishart(T, nu);
   MatrixXd mu_o = smurff::MvNormal_prec(Lam * kappa, mu);

   #ifdef TEST_MVNORMAL
   cout << "NORMAL WISHART {\n" << endl;
   cout << "  mu:\n" << mu << endl;
   cout << "  kappa:\n" << kappa << endl;
   cout << "  T:\n" << T << endl;
   cout << "  nu:\n" << nu << endl;
   cout << "  mu_o\n" << mu_o << endl;
   cout << "  Lam\n" << Lam << endl;
   cout << "}\n" << endl;
   #endif

   return std::make_pair(mu_o , Lam);
}

std::pair<VectorXd, MatrixXd> smurff::CondNormalWishart(const int N, const MatrixXd &NS, const VectorXd &NU, const VectorXd &mu, const double kappa, const MatrixXd &T, const int nu)
{
   int nu_c = nu + N;

   double kappa_c = kappa + N;
   auto mu_c = (kappa * mu + NU) / (kappa + N);
   auto X    = (T + NS + kappa * mu * mu.adjoint() - kappa_c * mu_c * mu_c.adjoint());
   Eigen::MatrixXd T_c = X.inverse();
    
   return NormalWishart(mu_c, kappa_c, T_c, nu_c);
}

std::pair<VectorXd, MatrixXd> smurff::CondNormalWishart(const MatrixXd &U, const VectorXd &mu, const double kappa, const MatrixXd &T, const int nu)
{
   auto N = U.cols();
   auto NS = U * U.adjoint();
   auto NU = U.rowwise().sum();
   return CondNormalWishart(N, NS, NU, mu, kappa, T, nu);
}

// Normal(0, Lambda^-1) for nn columns
MatrixXd smurff::MvNormal_prec(const MatrixXd & Lambda, int nn)
{
   int size = Lambda.rows(); // Dimensionality (rows)

   LLT<MatrixXd> chol(Lambda);

   MatrixXd r = MatrixXd::NullaryExpr(size, nn, std::cref(randn));
   chol.matrixU().solveInPlace(r);
   return r;
}

MatrixXd smurff::MvNormal_prec(const MatrixXd & Lambda, const VectorXd & mean, int nn)
{
   MatrixXd r = MvNormal_prec(Lambda, nn);
   return r.colwise() + mean;
}

MatrixXd smurff::MvNormal_prec_omp(const MatrixXd & Lambda, int nn)
{
   int size = Lambda.rows(); // Dimensionality (rows)

   LLT<MatrixXd> chol(Lambda);

   MatrixXd r(size, nn);
   smurff::bmrandn(r);
   // TODO: check if solveInPlace is parallelized:
   chol.matrixU().solveInPlace(r);
   return r;
}

// Draw nn samples from a size-dimensional normal distribution
// with a specified mean and covariance
MatrixXd smurff::MvNormal(const MatrixXd covar, const VectorXd mean, int nn) 
{
   int size = mean.rows(); // Dimensionality (rows)
   MatrixXd normTransform(size,size);

   LLT<MatrixXd> cholSolver(covar);
   normTransform = cholSolver.matrixL();

   auto normSamples = MatrixXd::NullaryExpr(size, nn, std::cref(randn));
   MatrixXd samples = (normTransform * normSamples).colwise() + mean;

   return samples;
}

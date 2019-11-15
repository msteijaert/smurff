
// From:
// http://stackoverflow.com/questions/6142576/sample-from-multivariate-normal-gaussian-distribution-in-c
 
#include <iostream>
#include <chrono>
#include <functional>


#include "ThreadVector.hpp"

#include "omp_util.h"

#ifdef USE_BOOST_RANDOM
#include <boost/random.hpp>
#define MERSENNE_TWISTER boost::random::mt19937
#define UNIFORM_REAL_DISTRIBUTION boost::random::uniform_real_distribution<double>
#define GAMMA_DISTRIBUTION boost::random::gamma_distribution<double>
#else
#include <random>
#define MERSENNE_TWISTER std::mt19937
#define UNIFORM_REAL_DISTRIBUTION std::uniform_real_distribution<double>
#define GAMMA_DISTRIBUTION std::gamma_distribution<double>
#endif

#include <Eigen/Dense>

#include "Distribution.h"

using namespace Eigen;

static smurff::thread_vector<MERSENNE_TWISTER> *bmrngs;

double smurff::randn0()
{
   return smurff::bmrandn_single_thread();
}

double smurff::randn(double) 
{
   return smurff::bmrandn_single_thread();
}

void smurff::bmrandn(double* x, long n) 
{
   #pragma omp parallel 
   {
      UNIFORM_REAL_DISTRIBUTION unif(-1.0, 1.0);
      auto& bmrng = bmrngs->local();
      
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
   
void smurff::bmrandn(Eigen::MatrixXd & X) 
{
   long n = X.rows() * (long)X.cols();
   smurff::bmrandn(X.data(), n);
}

double smurff::bmrandn_single_thread() 
{
   //TODO: add bmrng as input
   UNIFORM_REAL_DISTRIBUTION unif(-1.0, 1.0);
   auto& bmrng = bmrngs->local();
  
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
void smurff::bmrandn_single_thread(double* x, long n) 
{
   UNIFORM_REAL_DISTRIBUTION unif(-1.0, 1.0);
   auto& bmrng = bmrngs->local();

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
  
void smurff::bmrandn_single_thread(Eigen::VectorXd & x) 
{
   smurff::bmrandn_single_thread(x.data(), x.size());
}
 
void smurff::bmrandn_single_thread(Eigen::MatrixXd & X) 
{
   long n = X.rows() * (long)X.cols();
   smurff::bmrandn_single_thread(X.data(), n);
}


void smurff::init_bmrng() 
{
   using namespace std::chrono;
   auto ms = (duration_cast< milliseconds >(system_clock::now().time_since_epoch())).count();
   smurff::init_bmrng(ms);
}

void smurff::init_bmrng(int seed) 
{
    std::vector<MERSENNE_TWISTER> v;
    for (int i = 0; i < threads::get_max_threads(); i++)
    {
        v.push_back(MERSENNE_TWISTER(seed + i * 1999));
    }
    bmrngs = new smurff::thread_vector<MERSENNE_TWISTER>();
    bmrngs->init(v);
}
   
double smurff::rand_unif() 
{
   UNIFORM_REAL_DISTRIBUTION unif(0.0, 1.0);
   auto& bmrng = bmrngs->local();
   return unif(bmrng);
}
 
double smurff::rand_unif(double low, double high) 
{
   UNIFORM_REAL_DISTRIBUTION unif(low, high);
   auto& bmrng = bmrngs->local();
   return unif(bmrng);
}

// returns random number according to Gamma distribution
// with the given shape (k) and scale (theta). See wiki.
double smurff::rgamma(double shape, double scale) 
{
   GAMMA_DISTRIBUTION gamma(shape, scale);
   return gamma(bmrngs->local());
}

auto smurff::nrandn(int n) -> decltype(Eigen::VectorXd::NullaryExpr(n, std::cref(randn))) 
{
   return Eigen::VectorXd::NullaryExpr(n, std::cref(randn));
}

auto smurff::nrandn(int n, int m) -> decltype(Eigen::ArrayXXd::NullaryExpr(n, m, std::cref(randn)))
{
   return Eigen::ArrayXXd::NullaryExpr(n, m, std::cref(randn)); 
}


Eigen::MatrixXd WishartUnit(int m, int df)
{
   Eigen::MatrixXd c(m,m);
   c.setZero();
   auto& rng = bmrngs->local();

   for ( int i = 0; i < m; i++ ) 
   {
      GAMMA_DISTRIBUTION gam(0.5*(df - i));
      c(i,i) = std::sqrt(2.0 * gam(rng));
      Eigen::VectorXd r = smurff::nrandn(m-i-1);
      c.block(i,i+1,1,m-i-1) = r.transpose();
   }

   Eigen::MatrixXd ret = c.transpose() * c;

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

MatrixXd Wishart(const Eigen::MatrixXd &sigma, const int df)
{
   //  Get R, the upper triangular Cholesky factor of SIGMA.
   auto chol = sigma.llt();
   Eigen::MatrixXd r = chol.matrixL();

   //  Get AU, a sample from the unit Wishart distribution.
   Eigen::MatrixXd au = WishartUnit(sigma.cols(), df);

   //  Construct the matrix A = R' * AU * R.
   Eigen::MatrixXd a = r * au * chol.matrixU();

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
std::pair<Eigen::VectorXd, Eigen::MatrixXd> smurff::NormalWishart(const Eigen::VectorXd & mu, double kappa, const Eigen::MatrixXd & T, double nu)
{
   Eigen::MatrixXd Lam = Wishart(T, nu);
   Eigen::MatrixXd mu_o = smurff::MvNormal_prec(Lam * kappa, mu);

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

std::pair<Eigen::VectorXd, Eigen::MatrixXd> smurff::CondNormalWishart(const int N, const Eigen::MatrixXd &NS, const Eigen::VectorXd &NU, const Eigen::VectorXd &mu, const double kappa, const Eigen::MatrixXd &T, const int nu)
{
   int nu_c = nu + N;

   double kappa_c = kappa + N;
   auto mu_c = (kappa * mu + NU) / (kappa + N);
   auto X    = (T + NS + kappa * mu * mu.adjoint() - kappa_c * mu_c * mu_c.adjoint());
   Eigen::MatrixXd T_c = X.inverse();
    
   return NormalWishart(mu_c, kappa_c, T_c, nu_c);
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> smurff::CondNormalWishart(const Eigen::MatrixXd &U, const Eigen::VectorXd &mu, const double kappa, const Eigen::MatrixXd &T, const int nu)
{
   auto N = U.cols();
   auto NS = U * U.adjoint();
   auto NU = U.rowwise().sum();
   return CondNormalWishart(N, NS, NU, mu, kappa, T, nu);
}

// Normal(0, Lambda^-1) for nn columns
MatrixXd smurff::MvNormal_prec(const Eigen::MatrixXd & Lambda, int ncols)
{
   int nrows = Lambda.rows(); // Dimensionality (rows)
   LLT<Eigen::MatrixXd> chol(Lambda);

   Eigen::MatrixXd r(nrows, ncols);
   smurff::bmrandn(r);

   return chol.matrixU().solve(r);
}

Eigen::MatrixXd smurff::MvNormal_prec(const Eigen::MatrixXd & Lambda, const Eigen::VectorXd & mean, int nn)
{
   Eigen::MatrixXd r = MvNormal_prec(Lambda, nn);
   return r.colwise() + mean;
}

// Draw nn samples from a size-dimensional normal distribution
// with a specified mean and covariance
Eigen::MatrixXd smurff::MvNormal(const Eigen::MatrixXd covar, const Eigen::VectorXd mean, int nn) 
{
   int size = mean.rows(); // Dimensionality (rows)
   Eigen::MatrixXd normTransform(size,size);

   LLT<Eigen::MatrixXd> cholSolver(covar);
   normTransform = cholSolver.matrixL();

   auto normSamples = Eigen::MatrixXd::NullaryExpr(size, nn, std::cref(randn));
   Eigen::MatrixXd samples = (normTransform * normSamples).colwise() + mean;

   return samples;
}

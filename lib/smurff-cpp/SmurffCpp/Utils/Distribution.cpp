
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
   
template<typename M>
void smurff::bmrandn(M & X) 
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
  
template<typename V>
void smurff::bmrandn_single_thread(V & x) 
{
   smurff::bmrandn_single_thread(x.data(), x.size());
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

template <typename M>
M WishartUnit(int m, int df)
{
   M c(m,m);
   c.setZero();
   auto& rng = bmrngs->local();

   for ( int i = 0; i < m; i++ ) 
   {
      GAMMA_DISTRIBUTION gam(0.5*(df - i));
      c(i,i) = std::sqrt(2.0 * gam(rng));
      auto r = smurff::nrandn<M>(m-i-1, 1);
      c.block(i,i+1,1,m-i-1) = r.transpose();
   }

   M ret = c.transpose() * c;

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


template <typename M>
M Wishart(const M &sigma, const int df)
{
   //  Get R, the upper triangular Cholesky factor of SIGMA.
   auto chol = sigma.llt();
   M r = chol.matrixL();

   //  Get AU, a sample from the unit Wishart distribution.
   M au = WishartUnit<M>(sigma.cols(), df);

   //  Construct the matrix A = R' * AU * R.
   M a = r * au * chol.matrixU();

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
template<typename V, typename M>
std::pair<V, M> smurff::NormalWishart(const V & mu, double kappa, const M & T, double nu)
{
   M Lam = Wishart(T, nu);
   M Lam_kap  = Lam * kappa;
   M mu_o = smurff::MvNormal_prec(Lam_kap, mu);

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

   return std::make_pair(mu_o, Lam);
}

template<typename V, typename M>
std::pair<V, M> smurff::CondNormalWishart(const int N, const M &NS, const V &NU, const V &mu, const double kappa, const M &T, const int nu)
{
   int nu_c = nu + N;

   double kappa_c = kappa + N;
   V mu_c = (kappa * mu + NU) / (kappa + N);
   auto X    = (T + NS + kappa * mu * mu.adjoint() - kappa_c * mu_c * mu_c.adjoint());
   M T_c = X.inverse();
    
   return NormalWishart(mu_c, kappa_c, T_c, nu_c);
}

template<typename V, typename M>
std::pair<V, M> smurff::CondNormalWishart(const M &U, const V &mu, const double kappa, const M &T, const int nu)
{
   auto N = U.cols();
   M NS = U * U.adjoint();
   V NU = U.rowwise().sum();
   return CondNormalWishart(N, NS, NU, mu, kappa, T, nu);
}

// Normal(0, Lambda^-1) for nn columns
template <typename M>
M smurff::MvNormal_prec(const M & Lambda, int ncols)
{
   int nrows = Lambda.rows(); // Dimensionality (rows)
   LLT<M> chol(Lambda);

   M r(nrows, ncols);
   smurff::bmrandn(r);

   return chol.matrixU().solve(r);
}

template<typename V, typename M>
M smurff::MvNormal_prec(const M & Lambda, const V & mean, int nn)
{
   M r = MvNormal_prec(Lambda, nn);
   return r.colwise() + mean;
}

// Draw nn samples from a size-dimensional normal distribution
// with a specified mean and covariance
template<typename V, typename M>
M smurff::MvNormal(const M & covar, const V & mean, int nn) 
{
   int size = mean.rows(); // Dimensionality (rows)
   M normTransform(size,size);

   LLT<M> cholSolver(covar);
   normTransform = cholSolver.matrixL();

   auto normSamples = M::NullaryExpr(size, nn, std::cref(randn));
   M samples = (normTransform * normSamples).colwise() + mean;

   return samples;
}

template void smurff::bmrandn(Eigen::MatrixXd & X);
template void smurff::bmrandn_single_thread(Eigen::MatrixXd & X);
template void smurff::bmrandn_single_thread(Eigen::VectorXd & X);

// Wishart distribution
template std::pair<Eigen::VectorXd, Eigen::MatrixXd> smurff::NormalWishart(const Eigen::VectorXd & mu, double kappa, const Eigen::MatrixXd & T, double nu);
template std::pair<Eigen::VectorXd, Eigen::MatrixXd> smurff::CondNormalWishart(const Eigen::MatrixXd &U, const Eigen::VectorXd &mu, const double kappa, const Eigen::MatrixXd &T, const int nu);
template std::pair<Eigen::VectorXd, Eigen::MatrixXd> smurff::CondNormalWishart(const int N, const Eigen::MatrixXd &NS, const Eigen::VectorXd &NU, const Eigen::VectorXd &mu, const double kappa, const Eigen::MatrixXd &T, const int nu);

// Multivariate normal gaussian
template Eigen::MatrixXd smurff::MvNormal_prec(const Eigen::MatrixXd & Lambda, int nn = 1);
template Eigen::MatrixXd smurff::MvNormal_prec(const Eigen::MatrixXd & Lambda, const Eigen::VectorXd & mean, int nn = 1);
template Eigen::MatrixXd smurff::MvNormal(const Eigen::MatrixXd & covar, const Eigen::VectorXd & mean, int nn = 1);
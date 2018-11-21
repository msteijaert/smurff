
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
#define UNIFORM_REAL_DISTRIBUTION boost::random::uniform_real_distribution<float>
#define GAMMA_DISTRIBUTION boost::random::gamma_distribution<float>
#else
#include <random>
#define MERSENNE_TWISTER std::mt19937
#define UNIFORM_REAL_DISTRIBUTION std::uniform_real_distribution<float>
#define GAMMA_DISTRIBUTION std::gamma_distribution<float>
#endif

#include <Eigen/Dense>

#include "Distribution.h"

static smurff::thread_vector<MERSENNE_TWISTER> *bmrngs;

float smurff::randn0()
{
   return smurff::bmrandn_single_thread();
}

float smurff::randn(float) 
{
   return smurff::bmrandn_single_thread();
}

void smurff::bmrandn(float* x, long n) 
{
   #pragma omp parallel 
   {
      UNIFORM_REAL_DISTRIBUTION unif(-1.0, 1.0);
      auto& bmrng = bmrngs->local();
      
      #pragma omp for schedule(static)
      for (long i = 0; i < n; i += 2) 
      {
         float x1, x2, w;
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
   
void smurff::bmrandn(Eigen::MatrixXf & X) 
{
   long n = X.rows() * (long)X.cols();
   smurff::bmrandn(X.data(), n);
}

float smurff::bmrandn_single_thread() 
{
   //TODO: add bmrng as input
   UNIFORM_REAL_DISTRIBUTION unif(-1.0, 1.0);
   auto& bmrng = bmrngs->local();
  
   float x1, x2, w;
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
void smurff::bmrandn_single_thread(float* x, long n) 
{
   UNIFORM_REAL_DISTRIBUTION unif(-1.0, 1.0);
   auto& bmrng = bmrngs->local();

   for (long i = 0; i < n; i += 2) 
   {
      float x1, x2, w;

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
  
void smurff::bmrandn_single_thread(Eigen::VectorXf & x) 
{
   smurff::bmrandn_single_thread(x.data(), x.size());
}
 
void smurff::bmrandn_single_thread(Eigen::MatrixXf & X) 
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
   
float smurff::rand_unif() 
{
   UNIFORM_REAL_DISTRIBUTION unif(0.0, 1.0);
   auto& bmrng = bmrngs->local();
   return unif(bmrng);
}
 
float smurff::rand_unif(float low, float high) 
{
   UNIFORM_REAL_DISTRIBUTION unif(low, high);
   auto& bmrng = bmrngs->local();
   return unif(bmrng);
}

// returns random number according to Gamma distribution
// with the given shape (k) and scale (theta). See wiki.
float smurff::rgamma(float shape, float scale) 
{
   GAMMA_DISTRIBUTION gamma(shape, scale);
   return gamma(bmrngs->local());
}

auto smurff::nrandn(int n) -> decltype(Eigen::VectorXf::NullaryExpr(n, std::cref(randn))) 
{
   return Eigen::VectorXf::NullaryExpr(n, std::cref(randn));
}

auto smurff::nrandn(int n, int m) -> decltype(Eigen::ArrayXXf::NullaryExpr(n, m, std::ptr_fun(randn)))
{
   return Eigen::ArrayXXf::NullaryExpr(n, m, std::ptr_fun(randn)); 
}


Eigen::MatrixXf WishartUnit(int m, int df)
{
   Eigen::MatrixXf c(m,m);
   c.setZero();
   auto& rng = bmrngs->local();

   for ( int i = 0; i < m; i++ ) 
   {
      GAMMA_DISTRIBUTION gam(0.5*(df - i));
      c(i,i) = std::sqrt(2.0 * gam(rng));
      Eigen::VectorXf r = smurff::nrandn(m-i-1);
      c.block(i,i+1,1,m-i-1) = r.transpose();
   }

   Eigen::MatrixXf ret = c.transpose() * c;

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

Eigen::MatrixXf Wishart(const Eigen::MatrixXf &sigma, const int df)
{
   //  Get R, the upper triangular Cholesky factor of SIGMA.
   auto chol = sigma.llt();
   Eigen::MatrixXf r = chol.matrixL();

   //  Get AU, a sample from the unit Wishart distribution.
   Eigen::MatrixXf au = WishartUnit(sigma.cols(), df);

   //  Construct the matrix A = R' * AU * R.
   Eigen::MatrixXf a = r * au * chol.matrixU();

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
std::pair<Eigen::VectorXf, Eigen::MatrixXf> smurff::NormalWishart(const Eigen::VectorXf & mu, float kappa, const Eigen::MatrixXf & T, float nu)
{
   Eigen::MatrixXf Lam = Wishart(T, nu);
   Eigen::MatrixXf mu_o = smurff::MvNormal_prec(Lam * kappa, mu);

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

std::pair<Eigen::VectorXf, Eigen::MatrixXf> smurff::CondNormalWishart(const int N, const Eigen::MatrixXf &NS, const Eigen::VectorXf &NU, const Eigen::VectorXf &mu, const float kappa, const Eigen::MatrixXf &T, const int nu)
{
   int nu_c = nu + N;

   float kappa_c = kappa + N;
   auto mu_c = (kappa * mu + NU) / (kappa + N);
   auto X    = (T + NS + kappa * mu * mu.adjoint() - kappa_c * mu_c * mu_c.adjoint());
   Eigen::MatrixXf T_c = X.inverse();
    
   return NormalWishart(mu_c, kappa_c, T_c, nu_c);
}

std::pair<Eigen::VectorXf, Eigen::MatrixXf> smurff::CondNormalWishart(const Eigen::MatrixXf &U, const Eigen::VectorXf &mu, const float kappa, const Eigen::MatrixXf &T, const int nu)
{
   auto N = U.cols();
   auto NS = U * U.adjoint();
   auto NU = U.rowwise().sum();
   return CondNormalWishart(N, NS, NU, mu, kappa, T, nu);
}

// Normal(0, Lambda^-1) for nn columns
Eigen::MatrixXf smurff::MvNormal_prec(const Eigen::MatrixXf & Lambda, int ncols)
{
   int nrows = Lambda.rows(); // Dimensionality (rows)
   Eigen::LLT<Eigen::MatrixXf> chol(Lambda);

   Eigen::MatrixXf r(nrows, ncols);
   smurff::bmrandn(r);

   return chol.matrixU().solve(r);
}

Eigen::MatrixXf smurff::MvNormal_prec(const Eigen::MatrixXf & Lambda, const Eigen::VectorXf & mean, int nn)
{
   Eigen::MatrixXf r = MvNormal_prec(Lambda, nn);
   return r.colwise() + mean;
}

// Draw nn samples from a size-dimensional normal distribution
// with a specified mean and covariance
Eigen::MatrixXf smurff::MvNormal(const Eigen::MatrixXf covar, const Eigen::VectorXf mean, int nn) 
{
   int size = mean.rows(); // Dimensionality (rows)
   Eigen::MatrixXf normTransform(size,size);

   Eigen::LLT<Eigen::MatrixXf> cholSolver(covar);
   normTransform = cholSolver.matrixL();

   auto normSamples = Eigen::MatrixXf::NullaryExpr(size, nn, std::cref(randn));
   Eigen::MatrixXf samples = (normTransform * normSamples).colwise() + mean;

   return samples;
}

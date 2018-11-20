#pragma once

#include <map>
#include <functional>

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace smurff
{
   double randn0();
   double randn(double = .0);
   
   void bmrandn(double* x, long n);

   template<typename M>
   void bmrandn(M & X);
   
   double bmrandn_single_thread();
   void bmrandn_single_thread(double* x, long n);

   template<typename M>
   void bmrandn_single_thread(M & X);
   
   void init_bmrng();
   void init_bmrng(int seed);
   
   double rand_unif();
   double rand_unif(double low, double high);
   
   double rgamma(double shape, double scale);
   
   // return a random matrix of size n, m
   
   template<typename V>
   auto nrandn(int n) -> decltype(V::NullaryExpr(n, std::cref(randn)) )
   {
      return V::NullaryExpr(n, std::cref(randn));
   }

   template <typename M>
   auto nrandn(int n, int m) -> decltype(M::NullaryExpr(n, m, std::ptr_fun(randn)))
   {
      return M::NullaryExpr(n, m, std::ptr_fun(randn));
   }

   // Wishart distribution
   
   template<typename V, typename M>
   std::pair<V, M> NormalWishart(const V & mu, double kappa, const M & T, double nu);
   template<typename V, typename M>
   std::pair<V, M> CondNormalWishart(const M &U, const V &mu, const double kappa, const M &T, const int nu);
   template<typename V, typename M>
   std::pair<V, M> CondNormalWishart(const int N, const M &NS, const V &NU, const V &mu, const double kappa, const M &T, const int nu);
   
   // Multivariate normal gaussian

   template<typename M>
   M MvNormal_prec(const M & Lambda, int nn = 1);
   template<typename V, typename M>
   M MvNormal_prec(const M & Lambda, const V & mean, int nn = 1);
   template<typename V, typename M>
   M MvNormal(const M & covar, const V & mean, int nn = 1);
   
}

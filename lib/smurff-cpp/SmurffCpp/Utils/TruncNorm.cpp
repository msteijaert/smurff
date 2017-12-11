//need to add a define on windows to have math constants like M_SQRT1_2
#ifdef _WINDOWS
#define _USE_MATH_DEFINES
#endif

#include <cmath>

#include <SmurffCpp/Utils/InvNormCdf.h>
#include <SmurffCpp/Utils/Distribution.h>

double norm_cdf(double x) {
	return 0.5 * erfc(-x * M_SQRT1_2);
}

double rand_truncnorm_icdf(double low_cut) {
	double u = smurff::rand_unif(norm_cdf(low_cut), 1.0);
	return inv_norm_cdf(u);
}

double rand_truncnorm_rej(double low_cut) {
  double u, v, xbar;
  while (true) {
    u = smurff::rand_unif();
    xbar = sqrt(low_cut*low_cut - 2 * log(1 - u));
    v = smurff::rand_unif();
    if (v <= xbar / low_cut) {
      return xbar;
    }
  }
}

double rand_truncnorm(double low_cut) {
  if (low_cut > 3.0) {
    return rand_truncnorm_rej(low_cut);
  }
  return rand_truncnorm_icdf(low_cut);
}

double rand_truncnorm(double mean, double std, double low_cut) {
	double abar, xbar;
	abar = (low_cut - mean) / std;
	xbar = rand_truncnorm(abar);
	return std * xbar + mean;
}


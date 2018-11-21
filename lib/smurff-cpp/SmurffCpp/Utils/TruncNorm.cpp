//need to add a define on windows to have math constants like M_SQRT1_2
#ifdef _WINDOWS
#define _USE_MATH_DEFINES
#endif

#include <cmath>

#include <SmurffCpp/Utils/InvNormCdf.h>
#include <SmurffCpp/Utils/Distribution.h>

float norm_cdf(float x) {
	return 0.5 * erfc(-x * M_SQRT1_2);
}

float rand_truncnorm_icdf(float low_cut) {
	float u = smurff::rand_unif(norm_cdf(low_cut), 1.0);
	return inv_norm_cdf(u);
}

float rand_truncnorm_rej(float low_cut) {
  float u, v, xbar;
  while (true) {
    u = smurff::rand_unif();
    xbar = std::sqrt(low_cut*low_cut - 2 * std::log(1 - u));
    v = smurff::rand_unif();
    if (v <= xbar / low_cut) {
      return xbar;
    }
  }
}

float rand_truncnorm(float low_cut) {
  if (low_cut > 3.0) {
    return rand_truncnorm_rej(low_cut);
  }
  return rand_truncnorm_icdf(low_cut);
}

float rand_truncnorm(float mean, float std, float low_cut) {
	float abar, xbar;
	abar = (low_cut - mean) / std;
	xbar = rand_truncnorm(abar);
	return std * xbar + mean;
}


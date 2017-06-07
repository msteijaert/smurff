#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <cmath>

#include "mvnormal.h"
#include "noisemodels.h"
#include "model.h"
#include "data.h"

using namespace Eigen;

namespace Macau {

std::ostream &FixedGaussianNoise::info(std::ostream &os, std::string indent)
{ 
    os << "Fixed gaussian noise with precision: " << alpha << "\n";
    return os;
}

std::ostream &AdaptiveGaussianNoise::info(std::ostream &os, std::string indent)
{ 
    os << "Adaptive gaussian noise with max precision of " << alpha_max << "\n";
    return os;
}

//  AdaptiveGaussianNoise  ////
void AdaptiveGaussianNoise::init() {
    double var_total = data.var_total();
 
    // Var(noise) = Var(total) / (SN + 1)
    alpha     = (sn_init + 1.0) / var_total;
    alpha_max = (sn_max + 1.0) / var_total;
}

void AdaptiveGaussianNoise::update(const SubModel &m)
{
    double sumsq = data.sumsq(m);

    // (a0, b0) correspond to a prior of 1 sample of noise with full variance
    double a0 = 0.5;
    double b0 = 0.5 * var_total;
    double aN = a0 + data.nnz() / 2.0;
    double bN = b0 + sumsq / 2.0;
    alpha = rgamma(aN, 1.0 / bN);
    if (alpha > alpha_max) {
        alpha = alpha_max;
    }
}

std::ostream &ProbitNoise::info(std::ostream &os, std::string indent)
{ 
    os << "Probit Noise\n";
    return os;
}

} // end namespace Macau


#include <Eigen/Sparse>

#include <iostream>
#include <cassert>
#include <fstream>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <memory>
#include <cmath>

#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Sparse>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <signal.h>

#include "utils.h"
#include "data.h"
#include "model.h"
#include "mvnormal.h"

using namespace std; 
using namespace Eigen;

namespace Macau {

void Model::init(int nl, const std::vector<int> &d, std::string init_model) {
    num_latent = nl;
    dims = d;
    for(unsigned i = 0; i < d.size(); ++i) {
        samples.push_back(Eigen::MatrixXd(num_latent, d[i]));
        auto &M = samples.back();
        if (init_model == "random") bmrandn(M);
        else if (init_model == "zero") M.setZero();
        else assert(false);
    }
}

void Model::save(std::string prefix, std::string suffix) {
    int i = 0;
    for(auto &U : samples) {
        write_dense(prefix + "-U" + std::to_string(i++) + "-latents" + suffix, U);
    }
}

void Model::restore(std::string prefix, std::string suffix) {
    int i = 0;
    for(auto &U : samples) {
        read_dense(prefix + "-U" + std::to_string(i++) + "-latents" + suffix, U);
    }
}

std::ostream &Model::info(std::ostream &os, std::string indent)
{
    os << indent << "Num-latents: " << num_latent << "\n";
    return os;
}

SubModel Model::full()
{
    return SubModel(*this);
}

double Model::predict(const std::vector<int> &pos, const Data &data) const
{
    return dot(pos) + data.offset_to_mean(pos);
}


} // end namespace

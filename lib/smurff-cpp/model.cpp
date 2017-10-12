
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
#include <DataMatrices/Data.h>
#include "model.h"
#include "mvnormal.h"

#include "MatrixIO.h"

using namespace std;
using namespace Eigen;

namespace smurff {

void Model::init(int nl, const PVec<> &d, std::string init_model) {
    num_latent = nl;
    m_dims = std::unique_ptr<PVec<> >(new PVec<>(d));
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
    for(auto &U : samples)
    {
      smurff::matrix_io::eigen::write_matrix(prefix + "-U" + std::to_string(i++) + "-latents" + suffix, U);
    }
}

void Model::restore(std::string prefix, std::string suffix) {
    int i = 0;
    for(auto &U : samples)
    {
      smurff::matrix_io::eigen::read_matrix(prefix + "-U" + std::to_string(i++) + "-latents" + suffix, U);
    }
}

std::ostream &Model::info(std::ostream &os, std::string indent) const
{
    os << indent << "Num-latents: " << num_latent << "\n";
    return os;
}

std::ostream &Model::status(std::ostream &os, std::string indent) const
{
    Eigen::ArrayXd P = Eigen::ArrayXd::Ones(num_latent);
    for(int d = 0; d < nmodes(); ++d) P *= U(d).rowwise().norm().array();
    os << indent << "  Latent-wise norm: " << P.transpose() << "\n";
    return os;
}

SubModel Model::full()
{
    return SubModel(*this);
}

double Model::predict(const PVec<> &pos, const Data &data) const
{
    return dot(pos) + data.offset_to_mean(pos);
}


} // end namespace

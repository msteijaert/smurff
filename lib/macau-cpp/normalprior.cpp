#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <iomanip>

#include "mvnormal.h"
#include "session.h"
#include "chol.h"
#include "linop.h"
#include "noisemodels.h"
#include "latentprior.h"

extern "C" {
  #include <sparse.h>
}

using namespace std; 
using namespace Eigen;

namespace Macau {

ILatentPrior::ILatentPrior(BaseSession &m, int p, std::string name)
    : sessions(1, &m), pos(p), name(name), rrs(VectorNd::Zero(m.model->num_latent)),
                  MMs(MatrixNNd::Zero(m.model->num_latent, m.model->num_latent)) {} 

std::ostream &ILatentPrior::info(std::ostream &os, std::string indent) 
{
    os << indent << pos << ": " << name << "\n";
    return os;
}

BaseSession &ILatentPrior::sys(int s)
{
    return *sessions.at(s); 
}

Model &ILatentPrior::model(int s)
{
    return *sys(s).model; 
}

MatrixXd &ILatentPrior::U(int s)
{
    return model(s).U(pos);
}

MatrixXd &ILatentPrior::V(int s)
{
    return model(s).V(pos);
}

INoiseModel &ILatentPrior::noise(int s)
{
    return *sys(s).noise;
}

int ILatentPrior::num_cols()
{
    int ret = 0;
    for(int i = 0; i<num_sys(); ++i) ret += U(i).cols();
    return ret;
}

void ILatentPrior::add(BaseSession &b)
{
    sessions.push_back(&b);
}

void ILatentPrior::sample_latents() 
{
    for(unsigned s = 0; s < sessions.size(); s++) {
        auto &model = *sessions.at(s)->model;
        model.update_pnm(pos);
#pragma omp parallel for schedule(guided)
        for(int n = 0; n < model.U(pos).cols(); n++) {
#pragma omp task
            sample_latent(s, n); 
        }
    }
}

void ILatentPrior::pnm(int s, int n, VectorNd &rr, MatrixNNd &MM)
{
    auto &model = *sessions.at(s)->model;
    model.get_pnm(pos, n, rr, MM); 
}

/**
 *  base class NormalPrior 
 */

NormalPrior::NormalPrior(BaseSession &m, int p, std::string name)
    : ILatentPrior(m, p, name),
      Ucol(VectorNd::Zero(num_latent())),
      UUcol(MatrixNNd::Zero(num_latent(), num_latent()))
{
    const int K = num_latent();
    mu.resize(K);
    mu.setZero();

    Lambda.resize(K, K);
    Lambda.setIdentity();
    Lambda *= 10;

    // parameters of Inv-Whishart distribution
    WI.resize(K, K);
    WI.setIdentity();
    mu0.resize(K);
    mu0.setZero();
    b0 = 2;
    df = K;
}

void NormalPrior::sample_latents() {

    const int N = num_cols();
    const auto cov = UUcol.combine();
    const auto sum = Ucol.combine();
    tie(mu, Lambda) = CondNormalWishart(N, cov / N, sum / N, mu0, b0, WI, df);

    UUcol.reset();
    Ucol.reset();

    ILatentPrior::sample_latents();
}

void NormalPrior::sample_latent(int s, int n)
{
    auto &U = this->U(s);
    const auto &mu_u = getMu(n);
    const double alpha = noise(s).getAlpha();


    VectorNd &rr = rrs.local();
    MatrixNNd &MM = MMs.local();

    rr.setZero();
    MM.setZero();

    // add pnm
    pnm(s,n,rr,MM);

    // add noise
    rr.array() *= alpha;
    MM.array() *= alpha;

    // add hyperparams
    rr.noalias() += Lambda * mu_u;
    MM.noalias() += Lambda;

    Eigen::LLT<MatrixXd> chol = MM.llt();
    if(chol.info() != Eigen::Success) {
        throw std::runtime_error("Cholesky Decomposition failed!");
    }

    chol.matrixL().solveInPlace(rr);
    rr.noalias() += nrandn(num_latent());
    chol.matrixU().solveInPlace(rr);

    U.col(n).noalias() = rr;
    Ucol.local().noalias() += rr;
    UUcol.local().noalias() += rr * rr.transpose();

}

void NormalPrior::save(std::string prefix, std::string suffix) {
  write_dense(prefix + "-U" + std::to_string(pos) + "-latentmean" + suffix, mu);
}


/*
 * Master Prior
 */

template<class Prior>
MasterPrior<Prior>::MasterPrior(BaseSession &m, int p) 
    : Prior(m, p)
{
    this->name = "Master" + this->name;
}

template<class Prior>
void MasterPrior<Prior>::init() 
{
    Prior::init();
 
    // create+init slave priors
    for(auto &s : slaves) {
        for(auto &p : this->sys().priors) {
            s.template addPrior<SlavePrior>();
            if (p->pos != this->pos) p->add(s);
        }

        s.init();

        assert(this->U().cols() == s.priors.at(this->pos)->U().cols());
   }
}

template<class Prior>
std::ostream &MasterPrior<Prior>::info(std::ostream &os, std::string indent) 
{
    Prior::info(os, indent);
    os << indent << "with slaves {\n";
    for(auto &s : slaves) s.info(os, indent + "  ");
    os << indent << "}\n";
    return os;

}

template<class Prior>
void MasterPrior<Prior>::save(std::string prefix, std::string suffix)
{
    Prior::save(prefix, suffix);
    int i = 0;
    for(auto &s : slaves) s.save(prefix + "-S" + to_string(i++), suffix);
}
 
template<typename P1, typename P2>
std::pair<P1, P2> &operator+=(std::pair<P1, P2> &a, const std::pair<P1, P2> &b) {
    a.first += b.first;
    a.second += b.second;
    return a;
}

template<class Prior>
void MasterPrior<Prior>::pnm(int s, int n, VectorNd &rr, MatrixNNd &MM) 
{
    // first the master
    Prior::pnm(s, n, rr, MM);

    // no slaves for the slaves
    if (s > 0) return;

    // then the slaves
    assert(slaves.size() > 0 && "No slaves");
    for(auto &slave : slaves) {
        auto &slave_prior = slave.priors.at(this->pos);
        slave_prior->pnm(s, n, rr, MM);
    }
}

template<class Prior>
void MasterPrior<Prior>::sample_latent(int s, int d) {
    Prior::sample_latent(s,d);

    // no slaves on slaves
    if (s>0) return;

    // if s == 0 
    for(auto &slave : this->slaves) {
        auto &slave_prior = slave.priors.at(this->pos);
        slave_prior->U(s).col(d) = this->U(s).col(d);
    }
}


template<class Prior>
void MasterPrior<Prior>::sample_latents() {
    assert(slaves.size() > 0 && "No slaves");
    for(auto &s : this->slaves) s.model->update_pnm(this->pos);
    Prior::sample_latents(); // includes slaves!
    for(auto &s : this->slaves) s.noise->update();
}

template<class Prior>
template<class Model>
Model& MasterPrior<Prior>::addSlave()
{
    slaves.push_back(BaseSession());
    auto &slave_macau = slaves.back();
    slave_macau.name = "Slave " + std::to_string(slaves.size());
    Model *n = new Model(this->num_latent());
    slave_macau.model.reset(n);
    slave_macau.noise.reset(this->noise().copyTo(*n)); 
    return *n;
}

template<class Prior>
double MasterPrior<Prior>::getLinkNorm() {
    assert(slaves.size() > 0 && "No slaves");
    double ret = .0;
    for(auto &s : this->slaves) {
        ret += s.model->V(this->pos).norm();
    }
    return ret;
}

template class MasterPrior<NormalPrior>;
template DenseDenseMF &MasterPrior<NormalPrior>::addSlave();
template SparseDenseMF &MasterPrior<NormalPrior>::addSlave();
template SparseMF &MasterPrior<NormalPrior>::addSlave();

template class MasterPrior<SpikeAndSlabPrior>;
template DenseDenseMF &MasterPrior<SpikeAndSlabPrior>::addSlave();
template SparseDenseMF &MasterPrior<SpikeAndSlabPrior>::addSlave();
template SparseMF &MasterPrior<SpikeAndSlabPrior>::addSlave();

} // end namespace Macau

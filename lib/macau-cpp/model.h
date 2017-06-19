#ifndef MODEL_H
#define MODEL_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

#include "noisemodels.h"
#include "matrix_io.h"
#include "utils.h"

namespace smurff {

struct SubModel;

struct Model {
    Model() : num_latent(-1) {}
    void init(int nl, const PVec &indices, std::string init_model);

    //-- access for all
    const Eigen::MatrixXd &U(int f) const {
        return samples.at(f); 
    }
    Eigen::MatrixXd::ConstColXpr col(int f, int i) const {
        return U(f).col(i); 
    }
    Eigen::MatrixXd &U(int f) {
        return samples.at(f); 
    }

    double dot(const PVec &indices) const  {
        Eigen::ArrayXd P = Eigen::ArrayXd::Ones(num_latent);
        for(int d = 0; d < nmodes(); ++d) P *= col(d, indices.at(d)).array();
        return P.sum();
    }
    double predict(const PVec &pos, const Data &data) const;

    //-- for when nmodes == 2
    Eigen::MatrixXd &V(int f) {
        assert(nmodes() == 2);
        return samples.at((f+1)%2);
    }
    const Eigen::MatrixXd &V(int f) const {
        assert(nmodes() == 2);
        return samples.at((f+1)%2);
    }

    // basic stuff
    int nmodes() const { return samples.size(); }
    int nlatent() const { return num_latent; }
    int nsamples() const { return std::accumulate(samples.begin(), samples.end(), 0,
            [](const int &a, const Eigen::MatrixXd &b) { return a + b.cols(); }); }
    PVec dims;
    SubModel full();

    //-- output to file
    void save(std::string, std::string);
    void restore(std::string, std::string);
    std::ostream &info(std::ostream &os, std::string indent);

  private:
    std::vector<Eigen::MatrixXd> samples;
    int num_latent;
};

struct SubModel {
    SubModel(const Model &m, const PVec o, const PVec d) 
        : model(m), off(o), dims(d) {}

    SubModel(const SubModel &m, const PVec o, const PVec d) 
        : model(m.model), off(o + m.off), dims(d) {}

    SubModel(const Model &m) : model(m), off(m.nmodes()), dims(m.dims) {}

    Eigen::MatrixXd::ConstBlockXpr U(int f) const {
        return model.U(f).block(0, off.at(f), model.nlatent(), dims.at(f));
    }

    Eigen::MatrixXd::ConstBlockXpr V(int f) const {
        assert(nmodes() == 2);
        return U((f+1)%2);
    }

    double dot(const PVec &pos) const  {
        return model.dot(off + pos);
    }

    int nlatent() const { return model.nlatent(); }
    int nmodes() const { return model.nmodes(); }

private:
    const Model &model;
    PVec off;
    PVec dims;
};





}; // end namespace smurff

#endif

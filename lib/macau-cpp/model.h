#ifndef MODEL_H
#define MODEL_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

#include "noisemodels.h"
#include "matrix_io.h"
#include "utils.h"

namespace Macau {

struct SubModel;

struct Model {
    Model() : num_latent(-1) {}
    void init(int nl, const std::vector<int> &indices, std::string init_model);

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

    double dot(const std::vector<int> &indices) const  {
        Eigen::ArrayXd P = Eigen::ArrayXd::Ones(num_latent);
        for(int d = 0; d < nmodes(); ++d) P *= col(d, indices.at(d)).array();
        return P.sum();
    }

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
    std::vector<int> dims() const {
        std::vector<int> ret;
        for(auto s : samples) ret.push_back(s.cols());
        return ret;
    }

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
    SubModel(const Model &m, const std::vector<int> o, const std::vector<int> d) 
        : model(m), off(o), dims(d) {}

    SubModel(const SubModel &m, const std::vector<int> o, const std::vector<int> d) 
        : model(m.model), dims(d)
    {
        for(int i=0; i<nmodes(); ++i) {
            off.push_back(o[i] + m.off[i]);
        }
    }

    SubModel(const Model &m) : model(m), off(std::vector<int>(m.nmodes(), 0)), dims(m.dims()) {}

    Eigen::MatrixXd::ConstBlockXpr U(int f) const {
        return model.U(f).block(0, off.at(f), model.nlatent(), dims.at(f));
    }

    Eigen::MatrixXd::ConstBlockXpr V(int f) const {
        assert(nmodes() == 2);
        return U((f+1)%2);
    }

    double dot(const std::vector<int> &indices) const  {
        auto oi = indices;
        std::transform(oi.begin(), oi.end(), off.begin(), oi.begin(), std::plus<int>());
        return model.dot(indices);
    }

    int nlatent() const { return model.nlatent(); }
    int nmodes() const { return model.nmodes(); }

private:
    const Model &model;
    std::vector<int> off;
    std::vector<int> dims;
};





}; // end namespace Macau

#endif

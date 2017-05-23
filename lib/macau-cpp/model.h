#ifndef MODEL_H
#define MODEL_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

#include "noisemodels.h"
#include "matrix_io.h"
#include "utils.h"

namespace Macau {

struct Model {
    Model() : num_latent(-1), mean_rating(NAN) {}
    void init(int nl, double mean_rating, const std::vector<int> &indices, std::string init_model);

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

    double predict(const std::vector<int> &indices) const  {
        Eigen::ArrayXd P = Eigen::ArrayXd::Ones(num_latent);
        for(int d = 0; d < nmodes(); ++d) P *= col(d, indices.at(d)).array();
        return P.sum() + mean_rating;
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

    //-- output to file
    void save(std::string, std::string);
    void restore(std::string, std::string);
    std::ostream &info(std::ostream &os, std::string indent);

  private:
    std::vector<Eigen::MatrixXd> samples;
    int num_latent;
    double mean_rating;
};

}; // end namespace Macau

#endif

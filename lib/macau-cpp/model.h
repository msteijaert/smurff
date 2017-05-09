#ifndef MODEL_H
#define MODEL_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

#include "matrix_io.h"
#include "utils.h"

namespace Macau {

struct Model {
    void init(int nl, const std::vector<int> &indices);

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

    double predict(const std::vector<int> &indices, double mean_rating) const  {
        Eigen::ArrayXd P = Eigen::ArrayXd::Zero(num_latent) ;
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
};

struct Data {
    // helper functions for noise
    virtual double sumsq(const Model &) const = 0;
    virtual double var_total() const = 0;

    // helper functions for priors
    virtual void get_pnm(const Model &,int,int,VectorNd &, MatrixNNd &) = 0;
    virtual void update_pnm(const Model &,int) = 0;
 
    //-- print info
    std::ostream &info(std::ostream &os, std::string indent);

    // virtual functions data-related
    double mean_rating = .0;
    virtual void             init()       = 0;
    virtual int              nnz()  const = 0;
    virtual int              size() const = 0;
    virtual std::vector<int> dims() const = 0;

    std::string name;
};

struct MatrixData: public Data {
    virtual int nrow()      const = 0;
    virtual int ncol()      const = 0;
    int size()              const override { return nrow() * ncol(); }
    std::vector<int> dims() const override { return {nrow(), ncol()}; }
};

struct MatricesData: public Data {
  private:
      std::map<std::pair<int,int>, std::unique_ptr<MatrixData>> matrices;

};
template<typename YType>
struct MatrixDataTempl : public MatrixData {
    MatrixDataTempl(YType Y) : Y(Y) {}
    void init_base();
    void init() override;

    int nrow()                      const override { return Y.rows(); }
    int ncol()                      const override { return Y.cols(); }
    int nnz()                       const override { return Y.nonZeros(); }

    double var_total() const override;
    double sumsq(const Model &) const override;

    YType Y;
    std::vector<YType> Yc; // centered versions
};

struct ScarceMatrixData : public MatrixDataTempl<SparseMatrixD> {
    //-- c'tor
    ScarceMatrixData(SparseMatrixD Y)
        : MatrixDataTempl<SparseMatrixD>(Y) 
    {
        name = "ScarceMatrixData [with NAs]";
    }

    void get_pnm(const Model &,int,int,VectorNd &, MatrixNNd &) override;
    void update_pnm(const Model &,int) override {}
};

struct ScarceBinaryMatrixData : public MatrixDataTempl<SparseMatrixD> {
    //-- c'tor
    ScarceBinaryMatrixData(SparseMatrixD &Y) : MatrixDataTempl<SparseMatrixD>(Y) 
    {
        name = "ScarceMatrixData [containing 0,1,NA] (Probit Noise Sampler)";
    }

    void get_pnm(const Model &,int,int,VectorNd &, MatrixNNd &) override;
    void update_pnm(const Model &,int) override {};
};

template<class YType>
struct FullMatrixData : public MatrixDataTempl<YType> {
    //-- c'tor
    FullMatrixData(YType Y) : MatrixDataTempl<YType>(Y)
    {
        this->name = "MatrixData [fully known]";
    }

    void get_pnm(const Model &,int,int,VectorNd &, MatrixNNd &) override;
    void update_pnm(const Model &,int) override;

  private:
    Eigen::MatrixXd VV[2];
};

struct DenseMatrixData : public FullMatrixData<Eigen::MatrixXd> {
    //-- c'tor
    DenseMatrixData(Eigen::MatrixXd Y) : FullMatrixData<Eigen::MatrixXd>(Y)
    {
        this->name = "DenseMatrixData [fully known]";
    }
};

struct SparseMatrixData : public FullMatrixData<Eigen::SparseMatrix<double>> {
    //-- c'tor
    SparseMatrixData(Eigen::SparseMatrix<double> Y) : FullMatrixData<Eigen::SparseMatrix<double>>(Y)
    {
        this->name = "SparseMatrixData [fully known]";
    }
};

}; // end namespace Macau

#endif

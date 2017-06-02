#ifndef DATA_H
#define DATA_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

#include "noisemodels.h"
#include "matrix_io.h"
#include "utils.h"

namespace Macau {

struct Data {
    // helper functions for noise
    virtual double sumsq(const Model &) const = 0;
    virtual double var_total() const = 0;

    // update noise and precision/mean
    virtual void update(const Model &model) { noise->update(model); }
    virtual void get_pnm(const Model &,int,int,VectorNd &, MatrixNNd &) = 0;
    virtual void update_pnm(const Model &,int) = 0;

    //-- print info
    virtual std::ostream &info(std::ostream &os, std::string indent);

    // set noise models
    FixedGaussianNoise &setPrecision(double p);
    AdaptiveGaussianNoise &setAdaptivePrecision(double sn_init, double sn_max);
    ProbitNoise &setProbit();

    // virtual functions data-related
    double mean_rating = .0;
    virtual void             init()       = 0;
    virtual int              nnz()  const = 0;
    virtual int              size() const = 0;
    virtual std::vector<int> dims() const = 0;

    std::string                  name;
    std::unique_ptr<INoiseModel> noise;
};

struct MatrixData: public Data {
    virtual int nrow()      const = 0;
    virtual int ncol()      const = 0;
    int size()              const override { return nrow() * ncol(); }
    std::vector<int> dims() const override { return {ncol(), nrow()}; }
    std::ostream &info(std::ostream &os, std::string indent) override;
};

struct MatricesData: public MatrixData {
    // add data
    MatrixData &add(int, int, const std::unique_ptr<MatrixData>);

    // helper functions for noise
    // but 
    double sumsq(const Model &) const override { assert(false); return NAN; }
    double var_total() const override { assert(false); return NAN; }

    // update noise and precision/mean
    void get_pnm(const Model &,int,int,VectorNd &, MatrixNNd &) override;
    void update_pnm(const Model &model, int mode) override;
  
    //-- print info
    std::ostream &info(std::ostream &os, std::string indent) override;

    // virtual functions data-related
    void init()       override;
    int  nnz()  const override;
    int  nrow() const override;           
    int  ncol() const override;           

  private:
    Eigen::Matrix<std::unique_ptr<MatrixData>, Eigen::Dynamic, Eigen::Dynamic> matrices;
    std::vector<int> rowdims, coldims;
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
        name = "ScarceBinaryMatrixData [containing 0,1,NA]";
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

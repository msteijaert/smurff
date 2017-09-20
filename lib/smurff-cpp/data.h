#ifndef DATA_H
#define DATA_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <numeric>
#include <memory>

#include "matrix_io.h"
#include "utils.h"

#include "model.h"
#include "INoiseModel.h"

// AGE: I dont like the idea of adding this include. this all happens because we have implementation of MatricesData in header.
#include "UnusedNoise.h"

#include "Data.h"
#include "MatrixData.h"
#include "MatricesData.h"
#include "MatrixDataTempl.hpp"
#include "ScarceMatrixData.h"

namespace smurff {

struct ScarceBinaryMatrixData : public ScarceMatrixData {
    //-- c'tor
    ScarceBinaryMatrixData(SparseMatrixD &Y) : ScarceMatrixData(Y)
    {
        name = "ScarceBinaryMatrixData [containing 0,1,NA]";
    }

    void get_pnm(const SubModel &,int,int,VectorNd &, MatrixNNd &) override;
    void update_pnm(const SubModel &,int) override {};

    int    nna()   const override { return size() - nnz(); }

};

template<class YType>
struct FullMatrixData : public MatrixDataTempl<YType> {
    //-- c'tor
    FullMatrixData(YType Y) : MatrixDataTempl<YType>(Y)
    {
        this->name = "MatrixData [fully known]";
    }


    void get_pnm(const SubModel &,int,int,VectorNd &, MatrixNNd &) override;
    void update_pnm(const SubModel &,int) override;

    int    nna()   const override { return 0; }

  private:
    Eigen::MatrixXd VV[2];

    double compute_mode_mean(int,int) override;
};

struct DenseMatrixData : public FullMatrixData<Eigen::MatrixXd> {
    //-- c'tor
    DenseMatrixData(Eigen::MatrixXd Y) : FullMatrixData<Eigen::MatrixXd>(Y)
    {
        this->name = "DenseMatrixData [fully known]";
    }
    void center(double) override;
    double train_rmse(const SubModel &) const override;
};

struct SparseMatrixData : public FullMatrixData<Eigen::SparseMatrix<double>> {
    //-- c'tor
    SparseMatrixData(Eigen::SparseMatrix<double> Y) : FullMatrixData<Eigen::SparseMatrix<double>>(Y)
    {
        this->name = "SparseMatrixData [fully known]";
    }
    void center(double) override;
    double train_rmse(const SubModel &) const override;
};

}; // end namespace smurff

#endif

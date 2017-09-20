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
#include "ScarceBinaryMatrixData.h"
#include "FullMatrixData.hpp"
#include "DenseMatrixData.h"

namespace smurff {

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

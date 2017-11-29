#pragma once

#include <vector>
#include <memory>

#include <Eigen/Dense>
#include <Eigen/Sparse>

//convert array of coordinates to [nnz x nmodes] matrix
Eigen::MatrixXi toMatrix(int* columns, int nrows, int ncols);

//convert two coordinate arrays to [N x 2] matrix
Eigen::MatrixXi toMatrix(int* col1, int* col2, int nrows);

//convert array of values to vector
Eigen::VectorXd toVector(double* vals, int size);

//convert array of values to vector
Eigen::VectorXi toVector(int* vals, int size);
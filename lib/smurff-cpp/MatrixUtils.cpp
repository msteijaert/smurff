#include "MatrixUtils.h"

template<>
Eigen::SparseMatrix<double> sparse_to_eigen<const smurff::MatrixConfig>(const smurff::MatrixConfig& matrixConfig)
{
   Eigen::SparseMatrix<double> out(matrixConfig.getNRow(), matrixConfig.getNCol());
   std::shared_ptr<std::vector<std::uint32_t> > rowsPtr = matrixConfig.getRowsPtr();
   std::shared_ptr<std::vector<std::uint32_t> > colsPtr = matrixConfig.getColsPtr();
   std::shared_ptr<std::vector<double> > valuesPtr = matrixConfig.getValuesPtr();

   std::vector<Eigen::Triplet<double> > eigenTriplets;
   for (std::uint64_t i = 0; i < matrixConfig.getNNZ(); i++)
   {
      std::uint32_t row = rowsPtr->operator[](i);
      std::uint32_t col = colsPtr->operator[](i);
      double val = matrixConfig.isBinary() ? 1.0 : valuesPtr->operator[](i);
      eigenTriplets.push_back(Eigen::Triplet<double>(row, col, val));
   }

   out.setFromTriplets(eigenTriplets.begin(), eigenTriplets.end());
   return out;
}

template<>
Eigen::SparseMatrix<double> sparse_to_eigen<smurff::MatrixConfig>(smurff::MatrixConfig& matrixConfig)
{
   return sparse_to_eigen<const smurff::MatrixConfig>(matrixConfig);
}

Eigen::MatrixXd sparse_to_dense(const SparseBinaryMatrix& in)
{
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(in.nrow, in.ncol);
    for(int i=0; i<in.nnz; ++i) out(in.rows[i], in.cols[i]) = 1.;
    return out;
}

Eigen::MatrixXd sparse_to_dense(const SparseDoubleMatrix& in)
{
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(in.nrow, in.ncol);
    for(int i=0; i<in.nnz; ++i) out(in.rows[i], in.cols[i]) = in.vals[i];
    return out;
}
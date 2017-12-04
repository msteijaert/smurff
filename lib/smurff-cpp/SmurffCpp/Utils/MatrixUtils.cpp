#include "MatrixUtils.h"

#include <numeric>
#include <set>
#include <unsupported/Eigen/SparseExtra>

Eigen::MatrixXd smurff::matrix_utils::dense_to_eigen(const smurff::MatrixConfig& matrixConfig)
{
   if(!matrixConfig.isDense())
      throw std::runtime_error("matrix config should be dense");

   std::vector<double> Yvalues = matrixConfig.getValues(); //eigen map can not take const values pointer. have to make copy
   return Eigen::Map<Eigen::MatrixXd>(Yvalues.data(), matrixConfig.getNRow(), matrixConfig.getNCol());
}

Eigen::MatrixXd smurff::matrix_utils::dense_to_eigen(smurff::MatrixConfig& matrixConfig)
{
   const smurff::MatrixConfig& mc = matrixConfig;
   return smurff::matrix_utils::dense_to_eigen(mc);
}

template<>
Eigen::SparseMatrix<double> smurff::matrix_utils::sparse_to_eigen<const smurff::MatrixConfig>(const smurff::MatrixConfig& matrixConfig)
{
   if(matrixConfig.isDense())
      throw std::runtime_error("matrix config should be sparse");

   Eigen::SparseMatrix<double> out(matrixConfig.getNRow(), matrixConfig.getNCol());
   std::shared_ptr<std::vector<std::uint32_t> > rowsPtr = matrixConfig.getRowsPtr();
   std::shared_ptr<std::vector<std::uint32_t> > colsPtr = matrixConfig.getColsPtr();
   std::shared_ptr<std::vector<double> > valuesPtr = matrixConfig.getValuesPtr();

   std::vector<Eigen::Triplet<double> > eigenTriplets;
   for (std::uint64_t i = 0; i < matrixConfig.getNNZ(); i++)
   {
      std::uint32_t row = rowsPtr->operator[](i);
      std::uint32_t col = colsPtr->operator[](i);
      assert(row >= 0 && row < matrixConfig.getNRow());
      assert(col >= 0 && col < matrixConfig.getNCol());
      double val = valuesPtr->operator[](i);
      eigenTriplets.push_back(Eigen::Triplet<double>(row, col, val));
   }

   assert(eigenTriplets.size() == matrixConfig.getNNZ());
   out.setFromTriplets(eigenTriplets.begin(), eigenTriplets.end());
   assert(out.nonZeros() == (int)matrixConfig.getNNZ());
   return out;
}

template<>
Eigen::SparseMatrix<double> smurff::matrix_utils::sparse_to_eigen<smurff::MatrixConfig>(smurff::MatrixConfig& matrixConfig)
{
   return smurff::matrix_utils::sparse_to_eigen<const smurff::MatrixConfig>(matrixConfig);
}

Eigen::MatrixXd smurff::matrix_utils::sparse_to_dense(const SparseBinaryMatrix& in)
{
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(in.nrow, in.ncol);
    for(int i=0; i<in.nnz; ++i) out(in.rows[i], in.cols[i]) = 1.;
    return out;
}

Eigen::MatrixXd smurff::matrix_utils::sparse_to_dense(const SparseDoubleMatrix& in)
{
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(in.nrow, in.ncol);
    for(int i=0; i<in.nnz; ++i) out(in.rows[i], in.cols[i]) = in.vals[i];
    return out;
}

std::ostream& smurff::matrix_utils::operator << (std::ostream& os, const MatrixConfig& mc)
{
   const std::vector<std::uint32_t>& rows = mc.getRows();
   const std::vector<std::uint32_t>& cols = mc.getCols();
   const std::vector<double>& values = mc.getValues();
   const std::vector<std::uint32_t>& columns = mc.getColumns();

   if(rows.size() != cols.size() || rows.size() != values.size())
      throw "Invalid sizes";

   os << "rows: " << std::endl;
   for(std::uint64_t i = 0; i < rows.size(); i++)
      os << rows[i] << ", ";
   os << std::endl;

   os << "cols: " << std::endl;
   for(std::uint64_t i = 0; i < cols.size(); i++)
      os << cols[i] << ", ";
   os << std::endl;

   os << "columns: " << std::endl;
   for(std::uint64_t i = 0; i < columns.size(); i++)
      os << columns[i] << ", ";
   os << std::endl;

   os << "values: " << std::endl;
   for(std::uint64_t i = 0; i < values.size(); i++)
      os << values[i] << ", ";
   os << std::endl;

   os << "NRow: " << mc.getNRow() << " NCol: " << mc.getNCol() << std::endl;

   Eigen::SparseMatrix<double> X(mc.getNRow(), mc.getNCol());

   std::vector<Eigen::Triplet<double> > triplets;
   for(std::uint64_t i = 0; i < mc.getNNZ(); i++)
      triplets.push_back(Eigen::Triplet<double>(rows[i], cols[i], values[i]));

   os << "NTriplets: " << triplets.size() << std::endl;

   X.setFromTriplets(triplets.begin(), triplets.end());

   os << X << std::endl;

   return os;
}

bool smurff::matrix_utils::is_explicit_binary(const Eigen::SparseMatrix<double>& M)
{
   auto *values = M.valuePtr();
   for(int i = 0; i < M.nonZeros(); ++i) 
   {
      if (values[i] != 1.0 && values[i] != 0.0)
         return false;
   }

   std::cout << "Detected binary matrix\n";

   return true;
}

bool smurff::matrix_utils::equals(const Eigen::MatrixXd& m1, const Eigen::MatrixXd& m2, double precision)
{
   if (m1.rows() != m2.rows() || m1.cols() != m2.cols())
      return false;

   for (Eigen::Index i = 0; i < m1.rows(); i++)
   {
      for (Eigen::Index j = 0; j < m1.cols(); j++)
      {
         Eigen::MatrixXd::Scalar m1_v = m1(i, j);
         Eigen::MatrixXd::Scalar m2_v = m2(i, j);

         if (std::abs(m1_v - m2_v) > precision)
            return false;
      }
   }

   return true;
}

#include "MatrixUtils.h"

#include <numeric>
#include <set>
#include <vector>
#include <iterator>

#include <SmurffCpp/Utils/Error.h>

Eigen::MatrixXf smurff::matrix_utils::dense_to_eigen(const smurff::MatrixConfig& matrixConfig)
{
   if(!matrixConfig.isDense())
   {
      THROWERROR("matrix config should be dense");
   }

   std::vector<float> float_values(matrixConfig.getValues().begin(), matrixConfig.getValues().end());

   return Eigen::Map<const Eigen::MatrixXf>(float_values.data(), matrixConfig.getNRow(), matrixConfig.getNCol());
}

std::shared_ptr<smurff::MatrixConfig> smurff::matrix_utils::eigen_to_dense(const Eigen::MatrixXf &eigenMatrix, NoiseConfig n)
{
   std::vector<double> values(eigenMatrix.data(),  eigenMatrix.data() + eigenMatrix.size());
   return std::make_shared<smurff::MatrixConfig>(eigenMatrix.rows(), eigenMatrix.cols(), values, n);
}

struct sparse_vec_iterator
{
  sparse_vec_iterator(const smurff::MatrixConfig& matrixConfig, int pos)
     : config(matrixConfig), pos(pos) {}

  const smurff::MatrixConfig& config;
  int pos;

  bool operator!=(const sparse_vec_iterator &other) const {
     THROWERROR_ASSERT(&config == &other.config);
     return pos != other.pos;
  }

  sparse_vec_iterator &operator++() { pos++; return *this; }

  typedef Eigen::Triplet<float> T;
  T v;

  T* operator->() {
     // also convert from 1-base to 0-base
     uint32_t row = config.getRows()[pos];
     uint32_t col = config.getCols()[pos];
     float val = config.getValues()[pos];
     v = T(row, col, val);
     return &v;
  }
};

Eigen::SparseMatrix<float> smurff::matrix_utils::sparse_to_eigen(const smurff::MatrixConfig& matrixConfig)
{
   if(matrixConfig.isDense())
   {
      THROWERROR("matrix config should be sparse");
   }

   Eigen::SparseMatrix<float> out(matrixConfig.getNRow(), matrixConfig.getNCol());

   sparse_vec_iterator begin(matrixConfig, 0);
   sparse_vec_iterator end(matrixConfig, matrixConfig.getNNZ());

   out.setFromTriplets(begin, end);

   THROWERROR_ASSERT_MSG(out.nonZeros() == (int)matrixConfig.getNNZ(), "probable presence of duplicate records in " + matrixConfig.getFilename());

   return out;
}

std::shared_ptr<smurff::MatrixConfig> smurff::matrix_utils::eigen_to_sparse(const Eigen::SparseMatrix<float> &X, NoiseConfig n, bool isScarce)
{
   std::uint64_t nrow = X.rows();
   std::uint64_t ncol = X.cols();

   std::vector<uint32_t> rows;
   std::vector<uint32_t> cols;
   std::vector<double> values;

   for (int k = 0; k < X.outerSize(); ++k)
   {
      for (Eigen::SparseMatrix<float>::InnerIterator it(X,k); it; ++it)
      {
         rows.push_back(it.row());
         cols.push_back(it.col());
         values.push_back(it.value());
      }
   }

   return std::make_shared<smurff::MatrixConfig>(nrow, ncol, rows, cols, values, n, isScarce);
}

std::ostream& smurff::matrix_utils::operator << (std::ostream& os, const MatrixConfig& mc)
{
   const std::vector<std::uint32_t>& rows = mc.getRows();
   const std::vector<std::uint32_t>& cols = mc.getCols();
   const std::vector<double>& values = mc.getValues();
   const std::vector<std::uint32_t>& columns = mc.getColumns();

   if(rows.size() != cols.size() || rows.size() != values.size())
   {
      THROWERROR("Invalid sizes");
   }

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

   Eigen::SparseMatrix<float> X(mc.getNRow(), mc.getNCol());

   std::vector<Eigen::Triplet<float> > triplets;
   for(std::uint64_t i = 0; i < mc.getNNZ(); i++)
      triplets.push_back(Eigen::Triplet<float>(rows[i], cols[i], values[i]));

   os << "NTriplets: " << triplets.size() << std::endl;

   X.setFromTriplets(triplets.begin(), triplets.end());

   os << X << std::endl;

   return os;
}

bool smurff::matrix_utils::equals(const Eigen::MatrixXf& m1, const Eigen::MatrixXf& m2, float precision)
{
   if (m1.rows() != m2.rows() || m1.cols() != m2.cols())
      return false;

   for (Eigen::Index i = 0; i < m1.rows(); i++)
   {
      for (Eigen::Index j = 0; j < m1.cols(); j++)
      {
         Eigen::MatrixXf::Scalar m1_v = m1(i, j);
         Eigen::MatrixXf::Scalar m2_v = m2(i, j);

         if (std::abs(m1_v - m2_v) > precision)
            return false;
      }
   }

   return true;
}

bool smurff::matrix_utils::equals_vector(const Eigen::VectorXf& v1, const Eigen::VectorXf& v2, float precision)
{
   if (v1.size() != v2.size())
      return false;

   for (auto i = 0; i < v1.size(); i++)
   {
      if (std::abs(v1(i) - v2(i)) > precision)
         return false;
   }

   return true;
}

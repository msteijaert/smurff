#include "MatrixUtils.h"

#include <numeric>
#include <set>
#include <vector>
#include <unsupported/Eigen/SparseExtra>

#include <SmurffCpp/Utils/Error.h>

Eigen::MatrixXd smurff::matrix_utils::dense_to_eigen(const smurff::MatrixConfig& matrixConfig)
{
   if(!matrixConfig.isDense())
   {
      THROWERROR("matrix config should be dense");
   }

   return Eigen::Map<const Eigen::MatrixXd>(matrixConfig.getValues().data(), matrixConfig.getNRow(), matrixConfig.getNCol());
}

Eigen::MatrixXd smurff::matrix_utils::dense_to_eigen(smurff::MatrixConfig& matrixConfig)
{
   const smurff::MatrixConfig& mc = matrixConfig;
   return smurff::matrix_utils::dense_to_eigen(mc);
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

  typedef Eigen::Triplet<double> T;
  T v;

  T* operator->() {
     // also convert from 1-base to 0-base
     uint32_t row = config.getRows()[pos];
     uint32_t col = config.getCols()[pos];
     double val = config.getValues()[pos];
     v = T(row, col, val);
     return &v;
  }
};

Eigen::SparseMatrix<double> smurff::matrix_utils::sparse_to_eigen(const smurff::MatrixConfig& matrixConfig)
{
   if(matrixConfig.isDense())
   {
      THROWERROR("matrix config should be sparse");
   }

   Eigen::SparseMatrix<double> out(matrixConfig.getNRow(), matrixConfig.getNCol());

   sparse_vec_iterator begin(matrixConfig, 0);
   sparse_vec_iterator end(matrixConfig, matrixConfig.getNNZ());

   out.setFromTriplets(begin, end);

   THROWERROR_ASSERT_MSG(out.nonZeros() == (int)matrixConfig.getNNZ(), "probable presence of duplicate records in " + matrixConfig.getFilename());

   return out;
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

smurff::matrix_utils::sparse_eigen_struct smurff::matrix_utils::csr_to_eigen(const CSR& csr) {
    std::vector<Eigen::Triplet<double>>* triplet_list = new std::vector<Eigen::Triplet<double>>();
    Eigen::SparseMatrix<double, Eigen::RowMajor>* row_major = new Eigen::SparseMatrix<double, Eigen::RowMajor>(csr.nrow, csr.ncol);
    Eigen::SparseMatrix<double, Eigen::ColMajor>* col_major = new Eigen::SparseMatrix<double, Eigen::ColMajor>(csr.nrow, csr.ncol);

    for (size_t row = 0; row < csr.nrow; row++) {
        for (size_t i = csr.row_ptr[row]; i < csr.row_ptr[row + 1]; i++) {
            triplet_list->push_back(Eigen::Triplet<double>(row, csr.cols[i], csr.vals[i]));
        }
    }
    row_major->setFromTriplets(triplet_list->begin(), triplet_list->end());
    col_major->setFromTriplets(triplet_list->begin(), triplet_list->end());

    delete triplet_list;
    return {row_major, col_major};
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

   Eigen::SparseMatrix<double> X(mc.getNRow(), mc.getNCol());

   std::vector<Eigen::Triplet<double> > triplets;
   for(std::uint64_t i = 0; i < mc.getNNZ(); i++)
      triplets.push_back(Eigen::Triplet<double>(rows[i], cols[i], values[i]));

   os << "NTriplets: " << triplets.size() << std::endl;

   X.setFromTriplets(triplets.begin(), triplets.end());

   os << X << std::endl;

   return os;
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

bool smurff::matrix_utils::equals_vector(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2, double precision)
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

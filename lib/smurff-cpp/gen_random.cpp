#include <limits>
#include <iostream>

#include "mvnormal.h"
#include "config.h"
#include "gen_random.h"

using namespace Eigen;
using namespace smurff;

MatrixXd ones_Ydense(int N, int D, int K)
{
    MatrixXd X_tmp = MatrixXd::Ones(N,K);
    MatrixXd W_tmp = MatrixXd::Ones(D,K);
    return X_tmp * W_tmp.transpose();
}

MatrixXd random_Ydense(int N, int D, int K)
{
    MatrixXd X = nrandn(K,N);
    MatrixXd W = nrandn(K,D);
    return W * X.transpose() + nrandn(N,D).matrix();
}

SparseMatrixD sparsify(MatrixXd X, MatrixXd W, double s)
{
    const int N = X.cols();
    const int D = W.cols();

    SparseMatrixD Y(N,D);
    std::default_random_engine gen;
    std::uniform_real_distribution<double> udist(0.0,1.0);
    std::normal_distribution<> ndist;

    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    std::vector<int> row_counts(N, 0);
    std::vector<int> col_counts(D, 0);

    int empty_rows = 0;
    int empty_cols = 0;
    do {
        for(int i=0;i<N;++i) {
            if (row_counts[i] > 0) continue;
            for(int j=0;j<D;++j) {
                auto p=udist(gen);
                if(p < s) {
                    double v_ij = W.col(j).transpose() * X.col(i); //+ ndist(gen);
                    tripletList.push_back(T(i,j,v_ij));
                    row_counts[i]++;
                    col_counts[j]++;
                }
            }
        }

        for(int j=0;j<D;++j) {
            if (col_counts[j] > 0) continue;
            for(int i=0;i<N;++i) {
                auto p=udist(gen);
                if(p < s) {
                    double v_ij = W.col(j).transpose() * X.col(i); //+ ndist(gen);
                    tripletList.push_back(T(i,j,v_ij));
                    row_counts[i]++;
                    col_counts[j]++;
                }
            }
        }

        empty_rows = 0; for(int i=0; i<N; ++i) if (row_counts[i] == 0) empty_rows++;
        empty_cols = 0; for(int i=0; i<D; ++i) if (col_counts[i] == 0) empty_cols++;
    } while (empty_rows > 0 || empty_cols > 0);

    Y.setFromTriplets(tripletList.begin(), tripletList.end());   //create the matrix

    std::cout << Y.nonZeros() << " entries out of " << N*D << " (" << 100. * Y.nonZeros() / N / D << "%)" << std::endl;
    std::cout << empty_rows << " empty rows out of " << N << std::endl;
    std::cout << empty_cols << " empty cols out of " << D << std::endl;

    return Y;
}

SparseMatrixD ones_Ysparse(int N, int D, int K, double s)
{
    MatrixXd X = MatrixXd::Ones(K,N);
    MatrixXd W = MatrixXd::Ones(K,D);
    return sparsify(X, W, s);
}

SparseMatrixD random_Ysparse(int N, int D, int K, double s)
{
    MatrixXd X = nrandn(K,N);
    MatrixXd W = nrandn(K,D);
    return sparsify(X, W, s);
}

SparseMatrixD extract(SparseMatrixD &Y, double s, std::default_random_engine::result_type seed)
{
   unsigned Yout_nnz = Y.nonZeros() * s;
   unsigned Ynew_nnz = Y.nonZeros() - Yout_nnz;

   SparseMatrixD predictions(Y.rows(), Y.cols());
   std::default_random_engine gen(seed);
   std::uniform_real_distribution<double> udist(0.0,1.0);

   Y.prune([&udist, &gen, &predictions, s](
               const SparseMatrixD::Index& row,
               const SparseMatrixD::Index& col,
               const SparseMatrixD::Scalar& value)->bool
      {
         bool prune = udist(gen) < s;
         if (prune) predictions.insert(row,col) = value;
         return !prune;
      }
   );

   assert(Y.nonZeros() == Ynew_nnz);
   assert(predictions.nonZeros() == Yout_nnz);

   return predictions;
}

SparseMatrixD extract(const Eigen::MatrixXd &Yin, double s, std::default_random_engine::result_type seed)
{
   unsigned Yout_nnz = Yin.nonZeros() * s;
   SparseMatrixD Yout(Yin.rows(), Yin.cols());

   std::default_random_engine gen(seed);
   std::uniform_real_distribution<double> udist(0.0,1.0);

   for (int k = 0; k < Yin.cols(); ++k)
   {
      for (int l = 0; l < Yin.rows(); ++l)
      {
         auto p = udist(gen);
         if(p < s)
            Yout.coeffRef(l,k) = Yin(l,k);
      }
   }

   assert(Yout.nonZeros() == Yout_nnz);

   return Yout;
}

//
// ----------------------------------------------------------------------------
// !!! WARNING !!!
// ----------------------------------------------------------------------------
// Function below is indeterministic by its own nature.
// It cannot guarantee expected extraction ratio 's'.
//
// !!! IT IS RECOMMENDED NOT TO USE THIS FUNCTION AT ALL !!!
//
// GitHub issue #30:
//    https://github.com/ExaScience/smurff/issues/30
//
// ----------------------------------------------------------------------------
// Problem explained
// ----------------------------------------------------------------------------
// s = 0.3, remove = true, seed = 1234
// Yin = 1 2 3
//       4 5 6
//       7 8 9
//
// In this case it is expected that this function should produce 2 matrices.
// 'Ynew' with 7 nnz values and 'Yout' with 2 nnz values.
//
// But it does not work this way.
// It produces 'Ynew' with 8 nnz values and 'Yout' with 1 nnz value.
// Because of that assertion fails:
//    assert(Ynew_values.size() < Ynew_nnz)
//
// So additional check was added to prevent overflow:
//    Ynew_values.size() >= Ynew_nnz
//
// There are also tests in tests.cpp named 'gen_random/extract'.
// They all have [!hide] tag so that they don't run with the rest of the tests.
// These tests fail with assertion.
//
MatrixConfig extract(MatrixConfig &Yin, double s, bool remove, std::default_random_engine::result_type seed)
{
   size_t Yout_nnz = static_cast<size_t>(Yin.getNNZ() * s);
   size_t Ynew_nnz = Yin.getNNZ() - Yout_nnz;

   std::default_random_engine gen(seed);
   std::uniform_real_distribution<double> udist(0.0, 1.0);

   std::shared_ptr<std::vector<size_t> > Yin_rowsPtr = Yin.getRowsPtr();
   std::shared_ptr<std::vector<size_t> > Yin_colsPtr = Yin.getColsPtr();
   std::shared_ptr<std::vector<double> > Yin_valuesPtr = Yin.getValuesPtr();

   std::vector<size_t> Yout_rows;
   std::vector<size_t> Yout_cols;
   std::vector<double> Yout_values;

   std::vector<size_t> Ynew_rows;
   std::vector<size_t> Ynew_cols;
   std::vector<double> Ynew_values;

   for (size_t i = 0; i < Yin.getNNZ(); i++)
   {
      if ((udist(gen) < s && Yout_rows.size() < Yout_nnz) || Ynew_values.size() >= Ynew_nnz)
      {
         Yout_rows.push_back(Yin_rowsPtr->operator[](i));
         Yout_cols.push_back(Yin_colsPtr->operator[](i));
         Yout_values.push_back(Yin_valuesPtr->operator[](i));
      }
      else if (remove) // Applies only to sparse matrix
      {
         assert(Ynew_values.size() < Ynew_nnz);
         Ynew_rows.push_back(Yin_rowsPtr->operator[](i));
         Ynew_cols.push_back(Yin_colsPtr->operator[](i));
         Ynew_values.push_back(Yin_valuesPtr->operator[](i));
      }
   }

   assert(Yout_values.size() == Yout_nnz);
   // Applies only to sparse matrix
   if (remove)
   {
      assert(Ynew_values.size() == Ynew_nnz);
      MatrixConfig Ynew(Yin.getNRow(), Yin.getNCol(), Ynew_rows, Ynew_cols, Ynew_values, Yin.getNoiseConfig());
      std::swap(Yin, Ynew);
   }

   MatrixConfig Yout(Yin.getNRow(), Yin.getNCol(), Yout_rows, Yout_cols, Yout_values, Yin.getNoiseConfig());
   return Yout;
}

MatrixConfig extract(MatrixConfig &in, double s, std::default_random_engine::result_type seed)
{
   return extract(in, s, !in.isDense(), seed);
}
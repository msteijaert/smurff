#include <iostream>
#include "mvnormal.h"

#include "gen_random.h"

using namespace Eigen;

typedef Eigen::SparseMatrix<double> SparseMatrixD;

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

SparseMatrixD extract(SparseMatrixD &Y, double s)
{
    SparseMatrixD predictions(Y.rows(), Y.cols());
    std::default_random_engine gen;
    std::uniform_real_distribution<double> udist(0.0,1.0);

    Y.prune([&udist, &gen, &predictions, s](
                const SparseMatrixD::Index& row,
                const SparseMatrixD::Index& col,
                const SparseMatrixD::Scalar& value)->bool
        {
            bool prune = udist(gen) < s;
            if (prune) predictions.insert(row,col) = value;
            return prune;
        }
    );

    return predictions;
}

SparseMatrixD extract(const Eigen::MatrixXd &Yin, double s)
{
    SparseMatrixD Yout(Yin.rows(), Yin.cols());

    std::default_random_engine gen;
    std::uniform_real_distribution<double> udist(0.0,1.0);

    for (int k = 0; k < Yin.cols(); ++k) {
        for (int l = 0; l < Yin.rows(); ++l) {
            auto p=udist(gen);
            if(p < s) Yout.coeffRef(l,k) = Yin(l,k);
        }
    }
    return Yout;
}


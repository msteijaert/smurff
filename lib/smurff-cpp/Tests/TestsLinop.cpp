#include "catch.hpp"

#include <SmurffCpp/Utils/linop.h>
#include <SmurffCpp/Utils/Distribution.h>

using namespace smurff;

static NoiseConfig fixed_ncfg(NoiseTypes::fixed);

TEST_CASE( "MatrixXd/compute_uhat", "compute_uhat for MatrixXd" ) {
   Eigen::MatrixXd beta(2, 4), feat(6, 4), uhat(2, 6), uhat_true(2, 6);
   beta << 0.56,  0.55,  0.3 , -1.78,
           1.63, -0.71,  0.8 , -0.28;
   feat <<  -0.83,  -0.26,  -0.52,  -0.27,
             0.91,  -0.48,   0.50,  -0.20,
            -0.59,   1.94,  -1.09,   0.86,
            -0.08,   0.62,  -1.10,   0.96,
             1.44,   0.89,  -0.45,   0.2,
            -1.33,  -1.42,   0.03,  -2.32;
   uhat_true <<  -0.2832,  0.7516,  -1.1212,  -1.7426,  0.8049,   2.6128,
                 -1.5087,  2.2801,  -3.4519,  -1.7194,  1.2993,  -0.4861;
   smurff::linop::compute_uhat(uhat, feat, beta);
   for (int i = 0; i < uhat.rows(); i++) {
     for (int j = 0; j < uhat.cols(); j++) {
       REQUIRE( uhat(i,j) == Approx(uhat_true(i,j)) );
     }
   }
}

TEST_CASE( "linop/solve_blockcg_dense/fail", "BlockCG solver for dense (3rhs separately) [!hide]" ) 
{
   int rows[9] = { 0, 3, 3, 2, 5, 4, 1, 2, 4 };
   int cols[9] = { 1, 0, 2, 1, 3, 0, 1, 3, 2 };
   Eigen::MatrixXd B(3, 4), X(3, 4), X_true(3, 4), sf(6, 4);
 
    sf = Eigen::MatrixXd::Zero(6, 4);
    for (int i = 0; i < 9; i++) {
       sf(rows[i], cols[i]) = 1.0;
    }
 
   B << 0.56,  0.55,  0.3 , -1.78,
        0.34,  0.05, -1.48,  1.11,
        0.09,  0.51, -0.63,  1.59;
 
   // this system is unsolvable
   REQUIRE_THROWS(smurff::linop::solve_blockcg(X, sf, 0.5, B, 1e-6, true));
}

TEST_CASE( "linop/solve_blockcg_dense/ok", "BlockCG solver for dense (3rhs separately)" ) 
{
   double reg = 0.5;

   Eigen::MatrixXd KK(6, 6);
   KK <<  1.7488399 , -1.92816395, -1.39618642, -0.2769755 , -0.52815529, 0.24624319,
        -1.92816395,  3.34435465,  2.07258617,  0.4417173 ,  0.84673143, -0.35075244,
        -1.39618642,  2.07258617,  2.1623261 ,  0.25923918,  0.64428255, -0.2329581,
        -0.2769755 ,  0.4417173 ,  0.25923918,  0.6147927 ,  0.15112057, -0.00692033,
        -0.52815529,  0.84673143,  0.64428255,  0.15112057,  0.80141217, -0.19682322,
         0.24624319, -0.35075244, -0.2329581 , -0.00692033, -0.19682322, 0.56240547;

   Eigen::MatrixXd K = KK.llt().matrixU();

   REQUIRE(((K.transpose() * K) - KK).norm() < 1e-3);

   Eigen::MatrixXd X_true(3, 6);
   X_true << 0.35555556,  0.40709677, -0.16444444, -0.87483871, -0.16444444, -0.87483871,
             1.69333333, -0.12709677, -1.94666667,  0.49483871, -1.94666667,  0.49483871,
             0.66      , -0.04064516, -0.78      ,  0.65225806, -0.78      ,  0.65225806;

   Eigen::MatrixXd B = ((K.transpose() * K + Eigen::MatrixXd::Identity(6,6) * reg) * X_true.transpose()).transpose();
   Eigen::MatrixXd X(3, 6);

   //-- Solves the system (K' * K + reg * I) * X = B for X for m right-hand sides
   smurff::linop::solve_blockcg(X, K, 0.5, B, 1e-6, true);

   for (int i = 0; i < X.rows(); i++) {
     for (int j = 0; j < X.cols(); j++) {
       REQUIRE( X(i,j) == Approx(X_true(i,j)) );
     }
   }
}

TEST_CASE( "linop/A_mul_At_omp", "A_mul_At with OpenMP" ) 
{
   init_bmrng(12345);
   Eigen::MatrixXd A(2, 42);
   Eigen::MatrixXd AAt(2, 2);
   bmrandn(A);
   smurff::linop::A_mul_At_omp(AAt, A);
   Eigen::MatrixXd AAt_true(2, 2);
   AAt_true.triangularView<Eigen::Lower>() = A * A.transpose();
 
   REQUIRE( AAt(0,0) == Approx(AAt_true(0,0)) );
   REQUIRE( AAt(1,1) == Approx(AAt_true(1,1)) );
   REQUIRE( AAt(1,0) == Approx(AAt_true(1,0)) );
}

TEST_CASE( "linop/A_mul_At_combo", "A_mul_At with OpenMP (returning matrix)" ) 
{
   init_bmrng(12345);
   Eigen::MatrixXd A(2, 42);
   Eigen::MatrixXd AAt_true(2, 2);
   bmrandn(A);
   Eigen::MatrixXd AAt = smurff::linop::A_mul_At_combo(A);
   AAt_true = A * A.transpose();
 
   REQUIRE( AAt.rows() == 2);
   REQUIRE( AAt.cols() == 2);
 
   REQUIRE( AAt(0,0) == Approx(AAt_true(0,0)) );
   REQUIRE( AAt(1,1) == Approx(AAt_true(1,1)) );
   REQUIRE( AAt(0,1) == Approx(AAt_true(0,1)) );
   REQUIRE( AAt(1,0) == Approx(AAt_true(1,0)) );
}

TEST_CASE( "linop/A_mul_B_omp", "Fast parallel A_mul_B for small A") 
{
   Eigen::MatrixXd A(2, 2);
   Eigen::MatrixXd B(2, 5);
   Eigen::MatrixXd C(2, 5);
   Eigen::MatrixXd Ctr(2, 5);
   A << 3.0, -2.00,
        1.0,  0.91;
   B << 0.52, 0.19, 0.25, -0.73, -2.81,
       -0.15, 0.31,-0.40,  0.91, -0.08;
   smurff::linop::A_mul_B_omp(0, C, 1.0, A, B);
   Ctr = A * B;
   REQUIRE( (C - Ctr).norm() == Approx(0.0) );
}

TEST_CASE( "linop/A_mul_B_omp/speed", "Speed of A_mul_B_omp") 
{
   Eigen::MatrixXd B(32, 1000);
   Eigen::MatrixXd X(32, 1000);
   Eigen::MatrixXd Xtr(32, 1000);
   Eigen::MatrixXd A(32, 32);
   for (int col = 0; col < B.cols(); col++) {
     for (int row = 0; row < B.rows(); row++) {
       B(row, col) = sin(row * col);
     }
   }
   for (int col = 0; col < A.cols(); col++) {
     for (int row = 0; row < A.rows(); row++) {
       A(row, col) = sin(row*(row+0.2)*col);
     }
   }
   Xtr = A * B;
   smurff::linop::A_mul_B_omp(0, X, 1.0, A, B);
   REQUIRE( (X - Xtr).norm() == Approx(0.0) );
}

TEST_CASE( "linop/A_mul_B_add", "Fast parallel A_mul_B with adding") 
{
   Eigen::MatrixXd A(2, 2);
   Eigen::MatrixXd B(2, 5);
   Eigen::MatrixXd C(2, 5);
   Eigen::MatrixXd Ctr(2, 5);
   A << 3.0, -2.00,
        1.0,  0.91;
   B << 0.52, 0.19, 0.25, -0.73, -2.81,
       -0.15, 0.31,-0.40,  0.91, -0.08;
   C << 0.21, 0.70, 0.53, -0.18, -2.14,
       -0.35,-0.82,-0.27,  0.15, -0.10;
   Ctr = C;
   smurff::linop::A_mul_B_omp(1.0, C, 1.0, A, B);
   Ctr += A * B;
   REQUIRE( (C - Ctr).norm() == Approx(0.0) );
}

TEST_CASE( "linop/At_mul_A_blas", "A'A with BLAS is correct") 
{
   Eigen::MatrixXd A(3, 2);
   Eigen::MatrixXd AtA(2, 2);
   Eigen::MatrixXd AtAtr(2, 2);
   A <<  1.7, -3.1,
         0.7,  2.9,
        -1.3,  1.5;
   AtAtr = A.transpose() * A;
   smurff::linop::At_mul_A_blas(A, AtA.data());
   smurff::linop::makeSymmetric(AtA);
   REQUIRE( (AtA - AtAtr).norm() == Approx(0.0) );
}
 
TEST_CASE( "linop/A_mul_At_blas", "AA' with BLAS is correct") 
{
   Eigen::MatrixXd A(3, 2);
   Eigen::MatrixXd AA(3, 3);
   Eigen::MatrixXd AAtr(3, 3);
   A <<  1.7, -3.1,
         0.7,  2.9,
        -1.3,  1.5;
   AAtr = A * A.transpose();
   smurff::linop::A_mul_At_blas(A, AA.data());
   smurff::linop::makeSymmetric(AA);
   REQUIRE( (AA - AAtr).norm() == Approx(0.0) );
}

TEST_CASE( "linop/A_mul_B_blas", "A_mul_B_blas is correct") 
{
   Eigen::MatrixXd A(3, 2);
   Eigen::MatrixXd B(2, 5);
   Eigen::MatrixXd C(3, 5);
   Eigen::MatrixXd Ctr(3, 5);
   A << 3.0, -2.00,
        1.0,  0.91,
        1.9, -1.82;
   B << 0.52, 0.19, 0.25, -0.73, -2.81,
       -0.15, 0.31,-0.40,  0.91, -0.08;
   C << 0.21, 0.70, 0.53, -0.18, -2.14,
       -0.35,-0.82,-0.27,  0.15, -0.10,
       +2.34,-0.81,-0.47,  0.31, -0.14;
   smurff::linop::A_mul_B_blas(C, A, B);
   Ctr = A * B;
   REQUIRE( (C - Ctr).norm() == Approx(0.0) );
}
 
TEST_CASE( "linop/At_mul_B_blas", "At_mul_B_blas is correct") 
{
   Eigen::MatrixXd A(2, 3);
   Eigen::MatrixXd B(2, 5);
   Eigen::MatrixXd C(3, 5);
   Eigen::MatrixXd Ctr(3, 5);
   A << 3.0, -2.00,  1.0,
        0.91, 1.90, -1.82;
   B << 0.52, 0.19, 0.25, -0.73, -2.81,
       -0.15, 0.31,-0.40,  0.91, -0.08;
   C << 0.21, 0.70, 0.53, -0.18, -2.14,
       -0.35,-0.82,-0.27,  0.15, -0.10,
       +2.34,-0.81,-0.47,  0.31, -0.14;
   Ctr = C;
   smurff::linop::At_mul_B_blas(C, A, B);
   Ctr = A.transpose() * B;
   REQUIRE( (C - Ctr).norm() == Approx(0.0) );
}

TEST_CASE( "linop/A_mul_Bt_blas", "A_mul_Bt_blas is correct") 
{
   Eigen::MatrixXd A(3, 2);
   Eigen::MatrixXd B(5, 2);
   Eigen::MatrixXd C(3, 5);
   Eigen::MatrixXd Ctr(3, 5);
   A << 3.0, -2.00,
        1.0,  0.91,
        1.9, -1.82;
   B << 0.52,  0.19,
        0.25, -0.73,
       -2.81, -0.15,
        0.31, -0.40,
        0.91, -0.08;
   C << 0.21, 0.70, 0.53, -0.18, -2.14,
       -0.35,-0.82,-0.27,  0.15, -0.10,
       +2.34,-0.81,-0.47,  0.31, -0.14;
   smurff::linop::A_mul_Bt_blas(C, A, B);
   Ctr = A * B.transpose();
   REQUIRE( (C - Ctr).norm() == Approx(0.0) );
}

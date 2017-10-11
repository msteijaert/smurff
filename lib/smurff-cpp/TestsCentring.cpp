#include "catch.hpp"

#include <vector>

#include <Eigen/Core>

#include <Noises/Noiseless.h>

#include <DataMatrices/ScarceBinaryMatrixData.h>
#include <DataMatrices/DenseMatrixData.h>
#include <DataMatrices/SparseMatrixData.h>

using namespace smurff;

//===

void test_dimentions_scarse(Data* dims)
{
   REQUIRE(dims->nmode() == 2); // number of dimensions
   REQUIRE(dims->nnz() == 8); // number of non zero elements
   REQUIRE(dims->nna() == 4); // number of NA elements
   REQUIRE(dims->dim(0) == 4); // size of dimension
   REQUIRE(dims->dim(1) == 3); // size of dimension
   REQUIRE(dims->size() == 12); // number of all elements (dimension dot product)
}

TEST_CASE("ScarceBinaryMatrixData data dimentions")
{
   std::vector<Eigen::Triplet<double> > initialMatrixTriplets = {
      { 0, 0, 1 },
      { 0, 1, 2 },
      { 0, 2, 3 },
      { 2, 0, 7 },
      { 2, 1, 8 },
      { 2, 2, 9 },
      { 3, 0, 10 },
      { 3, 2, 12 },
   };

   Eigen::SparseMatrix<double> initialMatrix(4, 3);
   initialMatrix.setFromTriplets(initialMatrixTriplets.begin(), initialMatrixTriplets.end());

   ScarceBinaryMatrixData sbm(initialMatrix);

   test_dimentions_scarse(&sbm);
}

TEST_CASE("ScarceMatrixData data dimentions")
{
   std::vector<Eigen::Triplet<double> > initialMatrixTriplets = {
      { 0, 0, 1 },
      { 0, 1, 2 },
      { 0, 2, 3 },
      { 2, 0, 7 },
      { 2, 1, 8 },
      { 2, 2, 9 },
      { 3, 0, 10 },
      { 3, 2, 12 },
   };

   Eigen::SparseMatrix<double> initialMatrix(4, 3);
   initialMatrix.setFromTriplets(initialMatrixTriplets.begin(), initialMatrixTriplets.end());

   ScarceMatrixData scm(initialMatrix);

   test_dimentions_scarse(&scm);
}

TEST_CASE("SparseMatrixData data dimentions")
{
   std::vector<Eigen::Triplet<double> > initialMatrixTriplets = {
      { 0, 0, 1 },
      { 0, 1, 2 },
      { 0, 2, 3 },
      { 2, 0, 7 },
      { 2, 1, 8 },
      { 2, 2, 9 },
      { 3, 0, 10 },
      { 3, 2, 12 },
   };

   Eigen::SparseMatrix<double> initialMatrix(4, 3);
   initialMatrix.setFromTriplets(initialMatrixTriplets.begin(), initialMatrixTriplets.end());

   SparseMatrixData smd(initialMatrix);

   Data* dims = &smd;

  REQUIRE(dims->nmode() == 2); // number of dimensions
  REQUIRE(dims->nnz() == 8); // number of non zero elements
  REQUIRE(dims->nna() == 0); // number of NA elements
  REQUIRE(dims->dim(0) == 4); // size of dimension
  REQUIRE(dims->dim(1) == 3); // size of dimension
  REQUIRE(dims->size() == 12); // number of all elements (dimension dot product)
}

TEST_CASE("DenseMatrixData data dimentions")
{
   Eigen::MatrixXd initialMatrix(4, 3);
   initialMatrix << 1, 2, 3, 0, 0, 0, 7, 8, 9, 10, 0, 12;

   DenseMatrixData dmd(initialMatrix);

   Data* dims = &dmd;

  REQUIRE(dims->nmode() == 2); // number of dimensions
  REQUIRE(dims->nnz() == 12); // number of non zero elements
  REQUIRE(dims->nna() == 0); // number of NA elements
  REQUIRE(dims->dim(0) == 4); // size of dimension
  REQUIRE(dims->dim(1) == 3); // size of dimension
  REQUIRE(dims->size() == 12); // number of all elements (dimension dot product)
}

void test_arithmetic_scarse(Data* arith)
{
   REQUIRE(arith->sum() == 52.0);
}

TEST_CASE("ScarceBinaryMatrixData arithmetic")
{
   std::vector<Eigen::Triplet<double> > initialMatrixTriplets = {
      { 0, 0, 1 },
      { 0, 1, 2 },
      { 0, 2, 3 },
      { 2, 0, 7 },
      { 2, 1, 8 },
      { 2, 2, 9 },
      { 3, 0, 10 },
      { 3, 2, 12 },
   };

   Eigen::SparseMatrix<double> initialMatrix(4, 3);
   initialMatrix.setFromTriplets(initialMatrixTriplets.begin(), initialMatrixTriplets.end());

   ScarceBinaryMatrixData sbm(initialMatrix);

   test_arithmetic_scarse(&sbm);
}

TEST_CASE("ScarceMatrixData arithmetic")
{
   std::vector<Eigen::Triplet<double> > initialMatrixTriplets = {
      { 0, 0, 1 },
      { 0, 1, 2 },
      { 0, 2, 3 },
      { 2, 0, 7 },
      { 2, 1, 8 },
      { 2, 2, 9 },
      { 3, 0, 10 },
      { 3, 2, 12 },
   };

   Eigen::SparseMatrix<double> initialMatrix(4, 3);
   initialMatrix.setFromTriplets(initialMatrixTriplets.begin(), initialMatrixTriplets.end());

   ScarceMatrixData scm(initialMatrix);

   test_arithmetic_scarse(&scm);
}

TEST_CASE("SparseMatrixData arithmetic")
{
   std::vector<Eigen::Triplet<double> > initialMatrixTriplets = {
      { 0, 0, 1 },
      { 0, 1, 2 },
      { 0, 2, 3 },
      { 2, 0, 7 },
      { 2, 1, 8 },
      { 2, 2, 9 },
      { 3, 0, 10 },
      { 3, 2, 12 },
   };

   Eigen::SparseMatrix<double> initialMatrix(4, 3);
   initialMatrix.setFromTriplets(initialMatrixTriplets.begin(), initialMatrixTriplets.end());

   SparseMatrixData smd(initialMatrix);

   Data* arith = &smd;
   REQUIRE(arith->sum() == 52.0);
}

TEST_CASE("DenseMatrixData arithmetic")
{
   Eigen::MatrixXd initialMatrix(4, 3);
   initialMatrix << 1, 2, 3, 0, 0, 0, 7, 8, 9, 10, 0, 12;

   DenseMatrixData dmd(initialMatrix);

   Data* arith = &dmd;
   REQUIRE(arith->sum() == 52.0);
}

//===

TEST_CASE("DenseMatrixData IMeanCentering CENTER_NONE")
{
   Eigen::MatrixXd initialMatrix(4, 3);
   initialMatrix << 1, 2, 3, 0, 0, 0, 7, 8, 9, 10, 0, 12;

   DenseMatrixData dmd(initialMatrix);

   IMeanCentering* mnce = &dmd;

   mnce->setCenterMode(IMeanCentering::CenterModeTypes::CENTER_NONE);

   dmd.setNoiseModel(new Noiseless());
   dmd.init();

   REQUIRE(mnce->getGlobalMean() == Approx(4.33).epsilon(0.01));
   REQUIRE(mnce->getCwiseMean() == Approx(4.33).epsilon(0.01));

   REQUIRE(initialMatrix.isApprox(dmd.getYc().at(0).transpose()));
   REQUIRE(initialMatrix.isApprox(dmd.getYc().at(1)));
}

TEST_CASE("DenseMatrixData IMeanCentering CENTER_GLOBAL")
{
   Eigen::MatrixXd initialMatrix(4, 3);
   initialMatrix << 1, 2, 3, 0, 0, 0, 7, 8, 9, 10, 0, 12;

   DenseMatrixData dmd(initialMatrix);

   IMeanCentering* mnce = &dmd;

   mnce->setCenterMode(IMeanCentering::CenterModeTypes::CENTER_GLOBAL);

   dmd.setNoiseModel(new Noiseless());
   dmd.init();

   REQUIRE(mnce->getGlobalMean() == Approx(4.33).epsilon(0.01));
   REQUIRE(mnce->getCwiseMean() == Approx(4.33).epsilon(0.01));

   Eigen::MatrixXd expectedMatrix(4, 3);
   expectedMatrix << -3.33, -2.33, -1.33, -4.33, -4.33, -4.33, 2.67,  3.67,  4.67, 5.67, -4.33,  7.67;

   REQUIRE(expectedMatrix.isApprox(dmd.getYc().at(0).transpose(), 0.01));
   REQUIRE(expectedMatrix.isApprox(dmd.getYc().at(1), 0.01));
}

TEST_CASE("DenseMatrixData IMeanCentering CENTER_VIEW")
{
   Eigen::MatrixXd initialMatrix(4, 3);
   initialMatrix << 1, 2, 3, 0, 0, 0, 7, 8, 9, 10, 0, 12;

   DenseMatrixData dmd(initialMatrix);

   IMeanCentering* mnce = &dmd;

   mnce->setCenterMode(IMeanCentering::CenterModeTypes::CENTER_VIEW);

   dmd.setNoiseModel(new Noiseless());
   dmd.init();

   REQUIRE(mnce->getGlobalMean() == Approx(4.33).epsilon(0.01));
   REQUIRE(mnce->getCwiseMean() == Approx(4.33).epsilon(0.01));

   Eigen::MatrixXd expectedMatrix(4, 3);
   expectedMatrix << -3.33, -2.33, -1.33, -4.33, -4.33, -4.33, 2.67,  3.67,  4.67, 5.67, -4.33,  7.67;

   REQUIRE(expectedMatrix.isApprox(dmd.getYc().at(0).transpose(), 0.01));
   REQUIRE(expectedMatrix.isApprox(dmd.getYc().at(1), 0.01));
}

TEST_CASE("DenseMatrixData IMeanCentering CENTER_COLS")
{
   Eigen::MatrixXd initialMatrix(4, 3);
   initialMatrix << 1, 2, 3, 0, 0, 0, 7, 8, 9, 10, 0, 12;

   DenseMatrixData dmd(initialMatrix);

   IMeanCentering* mnce = &dmd;

   //substract each row mean from each row
   mnce->setCenterMode(IMeanCentering::CenterModeTypes::CENTER_COLS);

   dmd.setNoiseModel(new Noiseless());
   dmd.init();

   REQUIRE(mnce->getGlobalMean() == Approx(4.33).epsilon(0.01));
   REQUIRE(mnce->getCwiseMean() == Approx(4.33).epsilon(0.01));

   Eigen::VectorXd rowMeanExpected(4);
   rowMeanExpected << 2.0, 0.0, 8.0, 7.33;

   Eigen::VectorXd colMeanExpected(3);
   colMeanExpected << 4.5, 2.5, 6.0;

   REQUIRE(rowMeanExpected.isApprox(mnce->getModeMean(0), 0.01));
   REQUIRE(colMeanExpected.isApprox(mnce->getModeMean(1), 0.01));

   Eigen::MatrixXd expectedMatrix(4, 3);
   expectedMatrix << -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 2.67, -7.33,  4.67;

   REQUIRE(expectedMatrix.isApprox(dmd.getYc().at(0).transpose(), 0.01));
   REQUIRE(expectedMatrix.isApprox(dmd.getYc().at(1), 0.01));
}

TEST_CASE("DenseMatrixData IMeanCentering CENTER_ROWS")
{
   Eigen::MatrixXd initialMatrix(4, 3);
   initialMatrix << 1, 2, 3, 0, 0, 0, 7, 8, 9, 10, 0, 12;

   DenseMatrixData dmd(initialMatrix);

   IMeanCentering* mnce = &dmd;

   //substract each column mean from each columns
   mnce->setCenterMode(IMeanCentering::CenterModeTypes::CENTER_ROWS);

   dmd.setNoiseModel(new Noiseless());
   dmd.init();

   REQUIRE(mnce->getGlobalMean() == Approx(4.33).epsilon(0.01));
   REQUIRE(mnce->getCwiseMean() == Approx(4.33).epsilon(0.01));

   Eigen::VectorXd rowMeanExpected(4);
   rowMeanExpected << 2.0, 0.0, 8.0, 7.33;

   Eigen::VectorXd colMeanExpected(3);
   colMeanExpected << 4.5, 2.5, 6.0;

   REQUIRE(rowMeanExpected.isApprox(mnce->getModeMean(0), 0.01));
   REQUIRE(colMeanExpected.isApprox(mnce->getModeMean(1), 0.01));

   Eigen::MatrixXd expectedMatrix(4, 3);
   expectedMatrix << -3.5, -0.5, -3.0, -4.5, -2.5, -6, 2.5, 5.5, 3, 5.5, -2.5, 6;

   REQUIRE(expectedMatrix.isApprox(dmd.getYc().at(0).transpose(), 0.01));
   REQUIRE(expectedMatrix.isApprox(dmd.getYc().at(1), 0.01));
}

//===

TEST_CASE("SparseMatrixData IMeanCentering CENTER_NONE")
{
   std::vector<Eigen::Triplet<double> > initialMatrixTriplets = {
      { 0, 0, 1 },
      { 0, 1, 2 },
      { 0, 2, 3 },
      { 2, 0, 7 },
      { 2, 1, 8 },
      { 2, 2, 9 },
      { 3, 0, 10 },
      { 3, 2, 12 },
   };

   Eigen::SparseMatrix<double> initialMatrix(4, 3);
   initialMatrix.setFromTriplets(initialMatrixTriplets.begin(), initialMatrixTriplets.end());

   SparseMatrixData smd(initialMatrix);

   IMeanCentering* mnce = &smd;

   mnce->setCenterMode(IMeanCentering::CenterModeTypes::CENTER_NONE);

   smd.setNoiseModel(new Noiseless());
   smd.init();

   REQUIRE(mnce->getGlobalMean() == Approx(4.33).epsilon(0.01));
   REQUIRE(mnce->getCwiseMean() == Approx(4.33).epsilon(0.01));

   REQUIRE(initialMatrix.isApprox(smd.getYc().at(0).transpose()));
   REQUIRE(initialMatrix.isApprox(smd.getYc().at(1)));
}

//===

TEST_CASE("ScarceMatrixData IMeanCentering CENTER_NONE")
{
   std::vector<Eigen::Triplet<double> > initialMatrixTriplets = {
      { 0, 0, 1 },
      { 0, 1, 2 },
      { 0, 2, 3 },
      { 2, 0, 7 },
      { 2, 1, 8 },
      { 2, 2, 9 },
      { 3, 0, 10 },
      { 3, 2, 12 },
   };

   Eigen::SparseMatrix<double> initialMatrix(4, 3);
   initialMatrix.setFromTriplets(initialMatrixTriplets.begin(), initialMatrixTriplets.end());

   ScarceMatrixData scm(initialMatrix);

   IMeanCentering* mnce = &scm;

   mnce->setCenterMode(IMeanCentering::CenterModeTypes::CENTER_NONE);

   scm.setNoiseModel(new Noiseless());
   scm.init();

   REQUIRE(mnce->getGlobalMean() == Approx(6.5).epsilon(0.01));
   REQUIRE(mnce->getCwiseMean() == Approx(6.5).epsilon(0.01));

   REQUIRE(initialMatrix.isApprox(scm.getYc().at(0).transpose()));
   REQUIRE(initialMatrix.isApprox(scm.getYc().at(1)));
}

TEST_CASE("ScarceMatrixData IMeanCentering CENTER_GLOBAL")
{
   std::vector<Eigen::Triplet<double> > initialMatrixTriplets = {
      { 0, 0, 1 },
      { 0, 1, 2 },
      { 0, 2, 3 },
      { 2, 0, 7 },
      { 2, 1, 8 },
      { 2, 2, 9 },
      { 3, 0, 10 },
      { 3, 2, 12 },
   };

   Eigen::SparseMatrix<double> initialMatrix(4, 3);
   initialMatrix.setFromTriplets(initialMatrixTriplets.begin(), initialMatrixTriplets.end());

   ScarceMatrixData scm(initialMatrix);

   IMeanCentering* mnce = &scm;

   mnce->setCenterMode(IMeanCentering::CenterModeTypes::CENTER_GLOBAL);

   scm.setNoiseModel(new Noiseless());
   scm.init();

   REQUIRE(mnce->getGlobalMean() == Approx(6.5).epsilon(0.01));
   REQUIRE(mnce->getCwiseMean() == Approx(6.5).epsilon(0.01));

   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets = {
      { 0, 0, -5.5 },
      { 0, 1, -4.5 },
      { 0, 2, -3.5 },
      { 2, 0, 0.5 },
      { 2, 1, 1.5 },
      { 2, 2, 2.5 },
      { 3, 0, 3.5 },
      { 3, 2, 5.5 },
   };

   Eigen::SparseMatrix<double> expectedMatrix(4, 3);
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   REQUIRE(expectedMatrix.isApprox(scm.getYc().at(0).transpose(), 0.01));
   REQUIRE(expectedMatrix.isApprox(scm.getYc().at(1), 0.01));
}

TEST_CASE("ScarceMatrixData IMeanCentering CENTER_VIEW")
{
   std::vector<Eigen::Triplet<double> > initialMatrixTriplets = {
      { 0, 0, 1 },
      { 0, 1, 2 },
      { 0, 2, 3 },
      { 2, 0, 7 },
      { 2, 1, 8 },
      { 2, 2, 9 },
      { 3, 0, 10 },
      { 3, 2, 12 },
   };

   Eigen::SparseMatrix<double> initialMatrix(4, 3);
   initialMatrix.setFromTriplets(initialMatrixTriplets.begin(), initialMatrixTriplets.end());

   ScarceMatrixData scm(initialMatrix);

   IMeanCentering* mnce = &scm;

   mnce->setCenterMode(IMeanCentering::CenterModeTypes::CENTER_VIEW);

   scm.setNoiseModel(new Noiseless());
   scm.init();

   REQUIRE(mnce->getGlobalMean() == Approx(6.5).epsilon(0.01));
   REQUIRE(mnce->getCwiseMean() == Approx(6.5).epsilon(0.01));

   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets = {
      { 0, 0, -5.5 },
      { 0, 1, -4.5 },
      { 0, 2, -3.5 },
      { 2, 0, 0.5 },
      { 2, 1, 1.5 },
      { 2, 2, 2.5 },
      { 3, 0, 3.5 },
      { 3, 2, 5.5 },
   };

   Eigen::SparseMatrix<double> expectedMatrix(4, 3);
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   REQUIRE(expectedMatrix.isApprox(scm.getYc().at(0).transpose(), 0.01));
   REQUIRE(expectedMatrix.isApprox(scm.getYc().at(1), 0.01));
}

TEST_CASE("ScarceMatrixData IMeanCentering CENTER_COLS")
{
   std::vector<Eigen::Triplet<double> > initialMatrixTriplets = {
      { 0, 0, 1 },
      { 0, 1, 2 },
      { 0, 2, 3 },
      { 2, 0, 7 },
      { 2, 1, 8 },
      { 2, 2, 9 },
      { 3, 0, 10 },
      { 3, 2, 12 },
   };

   Eigen::SparseMatrix<double> initialMatrix(4, 3);
   initialMatrix.setFromTriplets(initialMatrixTriplets.begin(), initialMatrixTriplets.end());

   ScarceMatrixData scm(initialMatrix);

   IMeanCentering* mnce = &scm;

   mnce->setCenterMode(IMeanCentering::CenterModeTypes::CENTER_COLS);

   scm.setNoiseModel(new Noiseless());
   scm.init();

   REQUIRE(mnce->getGlobalMean() == Approx(6.5).epsilon(0.01));
   REQUIRE(mnce->getCwiseMean() == Approx(6.5).epsilon(0.01));

   Eigen::VectorXd rowMeanExpected(4);
   rowMeanExpected << 2.0, 6.5, 8.0, 11.0; //6.5 is expected for zero rows

   Eigen::VectorXd colMeanExpected(3);
   colMeanExpected << 6.0, 5.0, 8.0;

   REQUIRE(rowMeanExpected.isApprox(mnce->getModeMean(0), 0.01));
   REQUIRE(colMeanExpected.isApprox(mnce->getModeMean(1), 0.01));

   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets = {
      { 0, 0, -1.0 },
      { 0, 1, -0.0 },
      { 0, 2, 1.0 },
      { 2, 0, -1.0 },
      { 2, 1, 0.0 },
      { 2, 2, 1.0 },
      { 3, 0, -1.0 },
      { 3, 2, 1.0 },
   };

   Eigen::SparseMatrix<double> expectedMatrix(4, 3);
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   REQUIRE(expectedMatrix.isApprox(scm.getYc().at(0).transpose(), 0.01));
   REQUIRE(expectedMatrix.isApprox(scm.getYc().at(1), 0.01));
}

TEST_CASE("ScarceMatrixData IMeanCentering CENTER_ROWS")
{
   std::vector<Eigen::Triplet<double> > initialMatrixTriplets = {
      { 0, 0, 1 },
      { 0, 1, 2 },
      { 0, 2, 3 },
      { 2, 0, 7 },
      { 2, 1, 8 },
      { 2, 2, 9 },
      { 3, 0, 10 },
      { 3, 2, 12 },
   };

   Eigen::SparseMatrix<double> initialMatrix(4, 3);
   initialMatrix.setFromTriplets(initialMatrixTriplets.begin(), initialMatrixTriplets.end());

   ScarceMatrixData scm(initialMatrix);

   IMeanCentering* mnce = &scm;

   mnce->setCenterMode(IMeanCentering::CenterModeTypes::CENTER_ROWS);

   scm.setNoiseModel(new Noiseless());
   scm.init();

   REQUIRE(mnce->getGlobalMean() == Approx(6.5).epsilon(0.01));
   REQUIRE(mnce->getCwiseMean() == Approx(6.5).epsilon(0.01));

   Eigen::VectorXd rowMeanExpected(4);
   rowMeanExpected << 2.0, 6.5, 8.0, 11.0; //6.5 is expected for zero rows

   Eigen::VectorXd colMeanExpected(3);
   colMeanExpected << 6.0, 5.0, 8.0;

   REQUIRE(rowMeanExpected.isApprox(mnce->getModeMean(0), 0.01));
   REQUIRE(colMeanExpected.isApprox(mnce->getModeMean(1), 0.01));

   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets = {
      { 0, 0, -5.0 },
      { 0, 1, -3.0 },
      { 0, 2, -5.0 },
      { 2, 0, 1.0 },
      { 2, 1, 3.0 },
      { 2, 2, 1.0 },
      { 3, 0, 4.0 },
      { 3, 2, 4.0 },
   };

   Eigen::SparseMatrix<double> expectedMatrix(4, 3);
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   REQUIRE(expectedMatrix.isApprox(scm.getYc().at(0).transpose(), 0.01));
   REQUIRE(expectedMatrix.isApprox(scm.getYc().at(1), 0.01));
}

//===

TEST_CASE("ScarceBinaryMatrixData IMeanCentering CENTER_NONE")
{
   std::vector<Eigen::Triplet<double> > initialMatrixTriplets = {
      { 0, 0, 1 },
      { 0, 1, 2 },
      { 0, 2, 3 },
      { 2, 0, 7 },
      { 2, 1, 8 },
      { 2, 2, 9 },
      { 3, 0, 10 },
      { 3, 2, 12 },
   };

   Eigen::SparseMatrix<double> initialMatrix(4, 3);
   initialMatrix.setFromTriplets(initialMatrixTriplets.begin(), initialMatrixTriplets.end());

   ScarceBinaryMatrixData sbm(initialMatrix);

   IMeanCentering* mnce = &sbm;

   mnce->setCenterMode(IMeanCentering::CenterModeTypes::CENTER_NONE);

   sbm.setNoiseModel(new Noiseless());
   sbm.init();

   REQUIRE(mnce->getGlobalMean() == Approx(6.5).epsilon(0.01));
   REQUIRE(mnce->getCwiseMean() == Approx(6.5).epsilon(0.01));

   REQUIRE(initialMatrix.isApprox(sbm.getYc().at(0).transpose()));
   REQUIRE(initialMatrix.isApprox(sbm.getYc().at(1)));
}

TEST_CASE("ScarceBinaryMatrixData IMeanCentering CENTER_GLOBAL")
{
   std::vector<Eigen::Triplet<double> > initialMatrixTriplets = {
      { 0, 0, 1 },
      { 0, 1, 2 },
      { 0, 2, 3 },
      { 2, 0, 7 },
      { 2, 1, 8 },
      { 2, 2, 9 },
      { 3, 0, 10 },
      { 3, 2, 12 },
   };

   Eigen::SparseMatrix<double> initialMatrix(4, 3);
   initialMatrix.setFromTriplets(initialMatrixTriplets.begin(), initialMatrixTriplets.end());

   ScarceBinaryMatrixData sbm(initialMatrix);

   IMeanCentering* mnce = &sbm;

   mnce->setCenterMode(IMeanCentering::CenterModeTypes::CENTER_GLOBAL);

   sbm.setNoiseModel(new Noiseless());
   sbm.init();

   REQUIRE(mnce->getGlobalMean() == Approx(6.5).epsilon(0.01));
   REQUIRE(mnce->getCwiseMean() == Approx(6.5).epsilon(0.01));

   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets = {
      { 0, 0, -5.5 },
      { 0, 1, -4.5 },
      { 0, 2, -3.5 },
      { 2, 0, 0.5 },
      { 2, 1, 1.5 },
      { 2, 2, 2.5 },
      { 3, 0, 3.5 },
      { 3, 2, 5.5 },
   };

   Eigen::SparseMatrix<double> expectedMatrix(4, 3);
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   REQUIRE(expectedMatrix.isApprox(sbm.getYc().at(0).transpose(), 0.01));
   REQUIRE(expectedMatrix.isApprox(sbm.getYc().at(1), 0.01));
}

TEST_CASE("ScarceBinaryMatrixData IMeanCentering CENTER_VIEW")
{
   std::vector<Eigen::Triplet<double> > initialMatrixTriplets = {
      { 0, 0, 1 },
      { 0, 1, 2 },
      { 0, 2, 3 },
      { 2, 0, 7 },
      { 2, 1, 8 },
      { 2, 2, 9 },
      { 3, 0, 10 },
      { 3, 2, 12 },
   };

   Eigen::SparseMatrix<double> initialMatrix(4, 3);
   initialMatrix.setFromTriplets(initialMatrixTriplets.begin(), initialMatrixTriplets.end());

   ScarceBinaryMatrixData sbm(initialMatrix);

   IMeanCentering* mnce = &sbm;

   mnce->setCenterMode(IMeanCentering::CenterModeTypes::CENTER_VIEW);

   sbm.setNoiseModel(new Noiseless());
   sbm.init();

   REQUIRE(mnce->getGlobalMean() == Approx(6.5).epsilon(0.01));
   REQUIRE(mnce->getCwiseMean() == Approx(6.5).epsilon(0.01));

   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets = {
      { 0, 0, -5.5 },
      { 0, 1, -4.5 },
      { 0, 2, -3.5 },
      { 2, 0, 0.5 },
      { 2, 1, 1.5 },
      { 2, 2, 2.5 },
      { 3, 0, 3.5 },
      { 3, 2, 5.5 },
   };

   Eigen::SparseMatrix<double> expectedMatrix(4, 3);
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   REQUIRE(expectedMatrix.isApprox(sbm.getYc().at(0).transpose(), 0.01));
   REQUIRE(expectedMatrix.isApprox(sbm.getYc().at(1), 0.01));
}

TEST_CASE("ScarceBinaryMatrixData IMeanCentering CENTER_COLS")
{
   std::vector<Eigen::Triplet<double> > initialMatrixTriplets = {
      { 0, 0, 1 },
      { 0, 1, 2 },
      { 0, 2, 3 },
      { 2, 0, 7 },
      { 2, 1, 8 },
      { 2, 2, 9 },
      { 3, 0, 10 },
      { 3, 2, 12 },
   };

   Eigen::SparseMatrix<double> initialMatrix(4, 3);
   initialMatrix.setFromTriplets(initialMatrixTriplets.begin(), initialMatrixTriplets.end());

   ScarceBinaryMatrixData sbm(initialMatrix);

   IMeanCentering* mnce = &sbm;

   mnce->setCenterMode(IMeanCentering::CenterModeTypes::CENTER_COLS);

   sbm.setNoiseModel(new Noiseless());
   sbm.init();

   REQUIRE(mnce->getGlobalMean() == Approx(6.5).epsilon(0.01));
   REQUIRE(mnce->getCwiseMean() == Approx(6.5).epsilon(0.01));

   Eigen::VectorXd rowMeanExpected(4);
   rowMeanExpected << 2.0, 6.5, 8.0, 11.0; //6.5 is expected for zero rows

   Eigen::VectorXd colMeanExpected(3);
   colMeanExpected << 6.0, 5.0, 8.0;

   REQUIRE(rowMeanExpected.isApprox(mnce->getModeMean(0), 0.01));
   REQUIRE(colMeanExpected.isApprox(mnce->getModeMean(1), 0.01));

   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets = {
      { 0, 0, -1.0 },
      { 0, 1, -0.0 },
      { 0, 2, 1.0 },
      { 2, 0, -1.0 },
      { 2, 1, 0.0 },
      { 2, 2, 1.0 },
      { 3, 0, -1.0 },
      { 3, 2, 1.0 },
   };

   Eigen::SparseMatrix<double> expectedMatrix(4, 3);
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   REQUIRE(expectedMatrix.isApprox(sbm.getYc().at(0).transpose(), 0.01));
   REQUIRE(expectedMatrix.isApprox(sbm.getYc().at(1), 0.01));
}

TEST_CASE("ScarceBinaryMatrixData IMeanCentering CENTER_ROWS")
{
   std::vector<Eigen::Triplet<double> > initialMatrixTriplets = {
      { 0, 0, 1 },
      { 0, 1, 2 },
      { 0, 2, 3 },
      { 2, 0, 7 },
      { 2, 1, 8 },
      { 2, 2, 9 },
      { 3, 0, 10 },
      { 3, 2, 12 },
   };

   Eigen::SparseMatrix<double> initialMatrix(4, 3);
   initialMatrix.setFromTriplets(initialMatrixTriplets.begin(), initialMatrixTriplets.end());

   ScarceBinaryMatrixData sbm(initialMatrix);

   IMeanCentering* mnce = &sbm;

   mnce->setCenterMode(IMeanCentering::CenterModeTypes::CENTER_ROWS);

   sbm.setNoiseModel(new Noiseless());
   sbm.init();

   REQUIRE(mnce->getGlobalMean() == Approx(6.5).epsilon(0.01));
   REQUIRE(mnce->getCwiseMean() == Approx(6.5).epsilon(0.01));

   Eigen::VectorXd rowMeanExpected(4);
   rowMeanExpected << 2.0, 6.5, 8.0, 11.0; //6.5 is expected for zero rows

   Eigen::VectorXd colMeanExpected(3);
   colMeanExpected << 6.0, 5.0, 8.0;

   REQUIRE(rowMeanExpected.isApprox(mnce->getModeMean(0), 0.01));
   REQUIRE(colMeanExpected.isApprox(mnce->getModeMean(1), 0.01));

   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets = {
      { 0, 0, -5.0 },
      { 0, 1, -3.0 },
      { 0, 2, -5.0 },
      { 2, 0, 1.0 },
      { 2, 1, 3.0 },
      { 2, 2, 1.0 },
      { 3, 0, 4.0 },
      { 3, 2, 4.0 },
   };

   Eigen::SparseMatrix<double> expectedMatrix(4, 3);
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   REQUIRE(expectedMatrix.isApprox(sbm.getYc().at(0).transpose(), 0.01));
   REQUIRE(expectedMatrix.isApprox(sbm.getYc().at(1), 0.01));
}

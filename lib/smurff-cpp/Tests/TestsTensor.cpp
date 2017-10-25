#include "catch.hpp"

#include <iostream>
#include <string>
#include <sstream>

#include <SmurffCpp/sparsetensor.h>
#include <SmurffCpp/Configs/TensorConfig.h>


TEST_CASE("test tensor constructor 1")
{
   std::vector<int> tensorConfigDims = { 2, 3, 4, 2 };
   std::vector<int> tensorConfigColumns =
      {
         // 1D  
         0, 1, 0, 1, 0, 1,    0, 1, 0, 1, 0, 1,    0, 1, 0, 1, 0, 1,    0, 1, 0, 1, 0, 1,    0, 1, 0, 1, 0, 1,    0, 1, 0, 1, 0, 1,    0, 1, 0, 1, 0, 1,    0, 1, 0, 1, 0, 1,
         // 2D  
         0, 0, 1, 1, 2, 2,    0, 0, 1, 1, 2, 2,    0, 0, 1, 1, 2, 2,    0, 0, 1, 1, 2, 2,    0, 0, 1, 1, 2, 2,    0, 0, 1, 1, 2, 2,    0, 0, 1, 1, 2, 2,    0, 0, 1, 1, 2, 2,
         // 3D  
         0, 0, 0, 0, 0, 0,    1, 1, 1, 1, 1, 1,    2, 2, 2, 2, 2, 2,    3, 3, 3, 3, 3, 3,    0, 0, 0, 0, 0, 0,    1, 1, 1, 1, 1, 1,    2, 2, 2, 2, 2, 2,    3, 3, 3, 3, 3, 3,
         // 4D  
         0, 0, 0, 0, 0, 0,    0, 0, 0, 0, 0, 0,    0, 0, 0, 0, 0, 0,    0, 0, 0, 0, 0, 0,    1, 1, 1, 1, 1, 1,    1, 1, 1, 1, 1, 1,    1, 1, 1, 1, 1, 1,    1, 1, 1, 1, 1, 1
      };
   std::vector<double> tensorConfigValues =
      {
                  0, 1, 2, 3, 4, 5,    6, 7, 8, 9, 10, 11,    
                  12, 13, 14, 15, 16, 17,    18, 19, 20, 21, 22, 23,    
                  24, 25, 26, 27, 28, 29,   30, 31, 32, 33, 34, 35,   
                  36, 37, 38, 39, 40, 41,    42, 43, 44, 45, 46, 47
      };
   //TensorConfig tensorConfig(tensorConfigDims, tensorConfigColumns, tensorConfigValues, NoiseConfig());

   TensorData td(4);
   //td.setTrain(rows, cols, values, nnz, nrows, ncols);
   td.setTrain(tensorConfigColumns.data(), tensorConfigDims.size(), tensorConfigValues.data(), tensorConfigValues.size(), tensorConfigDims.data());

   /*
   std::cout << "Tensor Data test" << std::endl;

   for(std::unique_ptr<SparseMode>& val : *td.Y)
   {
      std::cout << "num_modes: " << val->num_modes << std::endl;
      std::cout << "nnz: " << val->nnz << std::endl;
      std::cout << "mode: " << val->mode << std::endl;

      std::cout << "row_ptr:" << std::endl << val->row_ptr << std::endl;
      std::cout << "indices:" << std::endl << val->indices.transpose() << std::endl;
      std::cout << "values:" << std::endl << val->values.transpose() << std::endl;

      std::cout << std::endl;
   }
   */
}

TEST_CASE("test tensor constructor 2")
{
   std::vector<int> tensorConfigDims = { 2, 3, 4 };
   std::vector<int> tensorConfigColumns =
      {
         // 1D 
         0,  1,  0,  1,  0,  1,     0,  1,  0,  1,  0,  1,     0,  1,  0,  1,  0,  1,     0,  1,  0,  1,  0,  1,
         // 2D 
         0,  0,  1,  1,  2,  2,     0,  0,  1,  1,  2,  2,     0,  0,  1,  1,  2,  2,     0,  0,  1,  1,  2,  2,
         // 3D 
         0,  0,  0,  0,  0,  0,     1,  1,  1,  1,  1,  1,     2,  2,  2,  2,  2,  2,     3,  3,  3,  3,  3,  3,
      };
   std::vector<double> tensorConfigValues =
      {
                  0,  3,  1,  4,  2,  5,     6,  9,  7, 10,  8, 11,    12, 15, 13, 16, 14, 17,    18, 21, 19, 22, 20, 23
      };

   TensorData td(3);
   //td.setTrain(rows, cols, values, nnz, nrows, ncols);
   td.setTrain(tensorConfigColumns.data(), tensorConfigDims.size(), tensorConfigValues.data(), tensorConfigValues.size(), tensorConfigDims.data());

   /*
   std::cout << "Tensor Data test" << std::endl;

   for(std::unique_ptr<SparseMode>& val : *td.Y)
   {
      std::cout << "num_modes: " << val->num_modes << std::endl;
      std::cout << "nnz: " << val->nnz << std::endl;
      std::cout << "mode: " << val->mode << std::endl;

      std::cout << "row_ptr:" << std::endl << val->row_ptr << std::endl;
      std::cout << "indices:" << std::endl << val->indices.transpose() << std::endl;
      std::cout << "values:" << std::endl << val->values.transpose() << std::endl;

      std::cout << std::endl;
   }
   */
}

//smurff

/* 
TEST_CASE("sparsetensor/sparsemode", "SparseMode constructor") {
  Eigen::MatrixXi C(5, 3);
  C << 0, 1, 0,
       0, 0, 0,
       1, 3, 1,
       2, 3, 0,
       1, 0, 1;
  Eigen::VectorXd v(5);
  v << 0.1, 0.2, 0.3, 0.4, 0.5;

  // mode 0
  SparseMode sm0(C, v, 0, 4);

  REQUIRE( sm0.num_modes == 3);
  REQUIRE( sm0.row_ptr.size() == 5 );
  REQUIRE( sm0.nnz == 5 );
  REQUIRE( sm0.row_ptr(0) == 0 );
  REQUIRE( sm0.row_ptr(1) == 2 );
  REQUIRE( sm0.row_ptr(2) == 4 );
  REQUIRE( sm0.row_ptr(3) == 5 );
  REQUIRE( sm0.row_ptr(4) == 5 );
  REQUIRE( sm0.modeSize() == 4 );

  Eigen::MatrixXi I0(5, 2);
  I0 << 1, 0,
        0, 0,
        3, 1,
        0, 1,
        3, 0;
  Eigen::VectorXd v0(5);
  v0 << 0.1, 0.2, 0.3, 0.5, 0.4;
  REQUIRE( (sm0.indices - I0).norm() == 0 );
  REQUIRE( (sm0.values  - v0).norm() == 0 );

  // mode 1
  SparseMode sm1(C, v, 1, 4);
  Eigen::VectorXi ptr1(5);
  ptr1 << 0, 2, 3, 3, 5;
  I0   << 0, 0,
          1, 1,
          0, 0,
          1, 1,
          2, 0;
  v0 << 0.2, 0.5, 0.1, 0.3, 0.4;
  REQUIRE( sm1.num_modes == 3);
  REQUIRE( (sm1.row_ptr - ptr1).norm() == 0 );
  REQUIRE( (sm1.indices - I0).norm()   == 0 );
  REQUIRE( (sm1.values  - v0).norm()   == 0 );
  REQUIRE( sm1.modeSize() == 4 );
}

TEST_CASE("bpmfutils/eval_rmse_tensor", "Testing eval_rmse_tensor") {
  Eigen::MatrixXi C(5, 3);
  C << 0, 1, 0,
       0, 0, 0,
       1, 3, 1,
       1, 0, 1,
       2, 3, 0;
  Eigen::VectorXd v(5);
  v << 0.1, 0.2, 0.3, 0.4, 0.5;

  // mode 0
  SparseMode sm0(C, v, 0, 4);
  int nlatent = 5;
  double gmean = 0.9;

  std::vector< std::unique_ptr<Eigen::MatrixXd> > samples;
  Eigen::VectorXi dims(3);
  dims << 4, 5, 2;

  for (int d = 0; d < 3; d++) {
    Eigen::MatrixXd* x = new Eigen::MatrixXd(nlatent, dims(d));
    bmrandn(*x);
    samples.push_back( std::move(std::unique_ptr<Eigen::MatrixXd>(x)) );
  }

  Eigen::VectorXd pred(5);
  Eigen::VectorXd pred_var(5);
  pred.setZero();
  pred_var.setZero();

  eval_rmse_tensor(sm0, 0, pred, pred_var, samples, gmean);

  for (int i = 0; i < C.rows(); i++) {
    auto v0 = gmean + samples[0]->col(C(i, 0)).
                  cwiseProduct( samples[1]->col(C(i, 1)) ).
                  cwiseProduct( samples[2]->col(C(i, 2)) ).sum();
    REQUIRE(v0 == Approx(pred(i)));
  }
}

TEST_CASE("sparsetensor/sparsetensor", "TensorData constructor") {
  Eigen::MatrixXi C(5, 3);
  C << 0, 1, 0,
       0, 0, 0,
       1, 3, 1,
       2, 3, 0,
       1, 0, 1;
  Eigen::VectorXd v(5);
  v << 0.1, 0.2, 0.3, 0.4, 0.5;
  Eigen::VectorXi dims(3);
  dims << 4, 4, 2;

  TensorData st(3);
  st.setTrain(C, v, dims);
  REQUIRE( st.Y->size() == 3 );
  REQUIRE( (*st.Y)[0]->nonZeros() == 5 );
  REQUIRE( st.mean_value == Approx(v.mean()) );
  REQUIRE( st.N == 3 );
  REQUIRE( st.dims(0) == dims(0) );
  REQUIRE( st.dims(1) == dims(1) );
  REQUIRE( st.dims(2) == dims(2) );

  // test data
  Eigen::MatrixXi Cte(6, 3);
  Cte << 1, 1, 0,
         0, 0, 0,
         1, 3, 0,
         0, 3, 0,
         2, 3, 1,
         2, 0, 0;
  Eigen::VectorXd vte(6);
  vte << -0.1, -0.2, -0.3, -0.4, -0.5, -0.6;
  st.setTest(Cte, vte, dims);

  // fetch test data:
  Eigen::MatrixXd testData = st.getTestData();

  REQUIRE( st.getTestNonzeros() == Cte.rows() );
  REQUIRE( testData.rows() == Cte.rows() );
  REQUIRE( testData.cols() == 4 );

  Eigen::MatrixXd testDataTr(6, 4);
  testDataTr << 0, 0, 0, -0.2,
                0, 3, 0, -0.4,
                1, 1, 0, -0.1,
                1, 3, 0, -0.3,
                2, 3, 1, -0.5,
                2, 0, 0, -0.6;
  REQUIRE( (testDataTr - testData).norm() == 0);
}

TEST_CASE("sparsetensor/vectorview", "VectorView test") {
	std::vector<std::unique_ptr<int> > vec2;
	vec2.push_back( std::unique_ptr<int>(new int(0)) );
	vec2.push_back( std::unique_ptr<int>(new int(2)) );
	vec2.push_back( std::unique_ptr<int>(new int(4)) );
	vec2.push_back( std::unique_ptr<int>(new int(6)) );
	vec2.push_back( std::unique_ptr<int>(new int(8)) );
	VectorView<int> vv2(vec2, 1);
	REQUIRE( *vv2.get(0) == 0 );
	REQUIRE( *vv2.get(1) == 4 );
	REQUIRE( *vv2.get(2) == 6 );
	REQUIRE( *vv2.get(3) == 8 );
	REQUIRE( vv2.size() == 4 );
}

TEST_CASE("latentprior/sample_tensor", "Test whether sampling tensor is correct") {
  Eigen::MatrixXi C(5, 3);
  C << 0, 1, 0,
       0, 0, 0,
       1, 3, 1,
       2, 3, 0,
       1, 0, 1;
  Eigen::VectorXd v(5);
  v << 0.15, 0.23, 0.31, 0.47, 0.59;

	Eigen::VectorXd mu(3);
	Eigen::MatrixXd Lambda(3, 3);
	mu << 0.03, -0.08, 0.12;
	Lambda << 1.2, 0.11, 0.17,
				    0.11, 1.4, 0.08,
						0.17, 0.08, 1.7;

	double mvalue = 0.2;
	double alpha  = 7.5;
  int nlatent = 3;

  std::vector< std::unique_ptr<Eigen::MatrixXd> > samples;
	std::vector< std::unique_ptr<SparseMode> > sparseModes;

  Eigen::VectorXi dims(3);
  dims << 4, 5, 2;
  TensorData st(3);
  st.setTrain(C, v, dims);

  for (int d = 0; d < 3; d++) {
    Eigen::MatrixXd* x = new Eigen::MatrixXd(nlatent, dims(d));
    bmrandn(*x);
    samples.push_back( std::move(std::unique_ptr<Eigen::MatrixXd>(x)) );

		SparseMode* sm  = new SparseMode(C, v, d, dims(d));
		sparseModes.push_back( std::move(std::unique_ptr<SparseMode>(sm)) );
  }

	VectorView<Eigen::MatrixXd> vv0(samples, 0);
  sample_latent_tensor(samples[0], 0, sparseModes[0], vv0, mvalue, alpha, mu, Lambda);
}

TEST_CASE("macauoneprior/sample_tensor_uni", "Testing sampling tensor univariate") {
  int rows[9] = { 0, 3, 3, 2, 5, 4, 1, 2, 4 };
  int cols[9] = { 1, 0, 2, 1, 3, 0, 1, 3, 2 };
  SparseFeat* sf = new SparseFeat(6, 4, 9, rows, cols);
  auto sfptr = std::unique_ptr<SparseFeat>(sf);

  Eigen::MatrixXi C(5, 3);
  C << 0, 1, 0,
       0, 0, 0,
       1, 3, 1,
       2, 3, 0,
       1, 0, 1;
  Eigen::VectorXd v(5);
  v << 0.15, 0.23, 0.31, 0.47, 0.59;

	Eigen::VectorXd mu(3);
	Eigen::MatrixXd Lambda(3, 3);
	mu << 0.03, -0.08, 0.12;
	Lambda << 1.2, 0.11, 0.17,
				    0.11, 1.4, 0.08,
						0.17, 0.08, 1.7;

	double mvalue = 0.2;
	double alpha  = 7.5;
  int nlatent = 3;

  MacauOnePrior<SparseFeat> prior(nlatent, sfptr);

  std::vector< std::unique_ptr<Eigen::MatrixXd> > samples;

  Eigen::VectorXi dims(3);
  dims << 6, 5, 2;
  TensorData st(3);
  st.setTrain(C, v, dims);

  for (int d = 0; d < 3; d++) {
    Eigen::MatrixXd* x = new Eigen::MatrixXd(nlatent, dims(d));
    bmrandn(*x);
    samples.push_back( std::move(std::unique_ptr<Eigen::MatrixXd>(x)) );
  }

  prior.sample_latents(alpha, st, samples, 0, nlatent);
}
*/
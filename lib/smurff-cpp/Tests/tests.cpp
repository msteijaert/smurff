#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>
#include <limits>

#include <SmurffCpp/Model.h>
#include <SmurffCpp/result.h>

#include <SmurffCpp/Utils/TruncNorm.h>
#include <SmurffCpp/Utils/InvNormCdf.h>
#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/Utils/counters.h>
#include <SmurffCpp/Utils/MatrixUtils.h>
#include <SmurffCpp/Utils/linop.h>

#include <SmurffCpp/Configs/MatrixConfig.h>

#include <SmurffCpp/Priors/ILatentPrior.h>
#include <SmurffCpp/Priors/MacauPrior.h>
#include <SmurffCpp/Priors/MacauOnePrior.h>

#include <SmurffCpp/Noises/NoiseFactory.h>

#include <SmurffCpp/DataMatrices/Data.h>
#include <SmurffCpp/DataMatrices/ScarceMatrixData.h>
#include <SmurffCpp/DataMatrices/FullMatrixData.hpp>
#include <SmurffCpp/DataMatrices/SparseMatrixData.h>
#include <SmurffCpp/DataMatrices/DenseMatrixData.h>

#include <SmurffCpp/SideInfo/DenseSideInfo.h>

// https://github.com/catchorg/Catch2/blob/master/docs/assertions.md#floating-point-comparisons
// By default Catch.hpp sets epsilon to std::numeric_limits<float>::epsilon()*100
#define APPROX_EPSILON std::numeric_limits<float>::epsilon()*100

using namespace smurff;

static NoiseConfig fixed_ncfg(NoiseTypes::fixed);

TEST_CASE( "mvnormal/rgamma", "generaring random gamma variable" ) {
  init_bmrng(1234);
  float g = rgamma(100.0, 0.01);
  REQUIRE( g > 0 );
}

TEST_CASE( "latentprior/sample_beta_precision", "sampling beta precision from gamma distribution" ) {
  init_bmrng(1234);
  Eigen::MatrixXf beta(2, 3), Lambda_u(2, 2);
  beta << 3.0, -2.00,  0.5,
          1.0,  0.91, -0.2;
  Lambda_u << 0.5, 0.1,
              0.1, 0.3;
  
  Eigen::MatrixXf BBt = beta * beta.transpose();                  
  auto post = MacauPrior::posterior_beta_precision(BBt, Lambda_u, 0.01, 0.05, beta.cols());
  REQUIRE( post.first  == Approx(3.005) );
  REQUIRE( post.second == Approx(0.2631083888) );

  float beta_precision = MacauPrior::sample_beta_precision(BBt, Lambda_u, 0.01, 0.05, beta.cols());
  REQUIRE( beta_precision > 0 );
}

TEST_CASE( "utils/eval_rmse", "Test if prediction variance is correctly calculated")
{
  std::vector<std::uint32_t> rows = {0};
  std::vector<std::uint32_t> cols = {0};
  std::vector<double>        vals = {4.5};

  std::shared_ptr<Model> model(new Model());
  
  std::shared_ptr<MatrixConfig> S(new MatrixConfig(1, 1, rows, cols, vals, fixed_ncfg, false));
  std::shared_ptr<Data> data(new ScarceMatrixData(smurff::matrix_utils::sparse_to_eigen(*S)));
  std::shared_ptr<Result> p(new Result(S));

  data->setNoiseModel(NoiseFactory::create_noise_model(fixed_ncfg));

  data->init();
  model->init(2, PVec<>({1, 1}), ModelInitTypes::zero); //latent dimention has size 2

  auto &t = p->m_predictions.at(0);

  // first iteration
  model->U(0) << 1.0, 0.0;
  model->U(1) << 1.0, 0.0;

  p->update(model, false);

  REQUIRE(t.pred_avg == Approx(1.0 * 1.0 + 0.0 * 0.0));
  REQUIRE(t.var == Approx(0.0));
  REQUIRE(p->rmse_1sample == Approx(std::sqrt(std::pow(4.5 - (1.0 * 1.0 + 0.0 * 0.0), 2) / 1 )));
  REQUIRE(p->rmse_avg ==     Approx(std::sqrt(std::pow(4.5 - (1.0 * 1.0 + 0.0 * 0.0) / 1, 2) / 1 )));

  //// second iteration
  model->U(0) << 2.0, 0.0;
  model->U(1) << 1.0, 0.0;

  p->update(model, false);

  REQUIRE(t.pred_avg == Approx(((1.0 * 1.0 + 0.0 * 0.0) + (2.0 * 1.0 + 0.0 * 0.0)) / 2));
  REQUIRE(t.var == Approx(0.5));
  REQUIRE(p->rmse_1sample == Approx(std::sqrt(std::pow(4.5 - (2.0 * 1.0 + 0.0 * 0.0), 2) / 1 )));
  REQUIRE(p->rmse_avg == Approx(std::sqrt(std::pow(4.5 - ((1.0 * 1.0 + 0.0 * 0.0) + (2.0 * 1.0 + 0.0 * 0.0)) / 2, 2) / 1)));

  //// third iteration

  model->U(0) << 2.0, 0.0;
  model->U(1) << 3.0, 0.0;

  p->update(model, false);

  REQUIRE(t.pred_avg == Approx(((1.0 * 1.0 + 0.0 * 0.0) + (2.0 * 1.0 + 0.0 * 0.0)+ (2.0 * 3.0 + 0.0 * 0.0)) / 3));
  REQUIRE(t.var == Approx(14.0)); // accumulated variance
  REQUIRE(p->rmse_1sample == Approx(std::sqrt(std::pow(4.5 - (2.0 * 3.0 + 0.0 * 0.0), 2) / 1 )));
  REQUIRE(p->rmse_avg == Approx(std::sqrt(std::pow(4.5 - ((1.0 * 1.0 + 0.0 * 0.0) + (2.0 * 1.0 + 0.0 * 0.0) + (2.0 * 3.0 + 0.0 * 0.0)) / 3, 2) / 1)));
}

TEST_CASE("utils/auc","AUC ROC") {
  struct TestItem {
      float pred, val;
  };
  std::vector<TestItem> items = {
   { 20.0, 1.0 },
   { 19.0, 0.0 },
   { 18.0, 1.0 },
   { 17.0, 0.0 },
   { 16.0, 1.0 },
   { 15.0, 0.0 },
   { 14.0, 0.0 },
   { 13.0, 1.0 },
   { 12.0, 0.0 },
   { 11.0, 1.0 },
   { 10.0, 0.0 },
   { 9.0,  0.0 },
   { 8.0,  0.0 },
   { 7.0,  0.0 },
   { 6.0,  0.0 },
   { 5.0,  0.0 },
   { 4.0,  0.0 },
   { 3.0,  0.0 },
   { 2.0,  0.0 },
   { 1.0,  0.0 }
  };

  REQUIRE ( calc_auc(items, 0.5) == Approx(0.84) );
}

TEST_CASE( "ScarceMatrixData/var_total", "Test if variance of Scarce Matrix is correctly calculated") {
  std::vector<std::uint32_t> rows = {0, 1};
  std::vector<std::uint32_t> cols = {0, 0};
  std::vector<double>        vals = {1., 2.};

  const MatrixConfig S(2, 2, rows, cols, vals, fixed_ncfg, false);
  std::shared_ptr<Data> data(new ScarceMatrixData(matrix_utils::sparse_to_eigen(S)));

  data->setNoiseModel(NoiseFactory::create_noise_model(fixed_ncfg));

  data->init();
  REQUIRE(data->var_total() == Approx(0.25));
}

TEST_CASE( "DenseMatrixData/var_total", "Test if variance of Dense Matrix is correctly calculated") {
  Eigen::MatrixXf Y(2, 2);
  Y << 1., 2., 3., 4.;

  std::shared_ptr<Data> data(new DenseMatrixData(Y));

  data->setNoiseModel(NoiseFactory::create_noise_model(fixed_ncfg));

  data->init();
  REQUIRE(data->var_total() == Approx(1.25));
}

using namespace std;

MacauPrior* make_dense_prior(int nlatent, const std::vector<double> & ptr, int nrows, int ncols, bool comp_FtF) 
{
   auto ret = new MacauPrior(0, 0);
   std::shared_ptr<MatrixConfig> Fmat_ptr = std::make_shared<MatrixConfig>(nrows, ncols, ptr, fixed_ncfg);
   std::shared_ptr<DenseSideInfo> side_info = std::make_shared<DenseSideInfo>(Fmat_ptr);
   ret->addSideInfo(side_info, 10.0, 1e-6, comp_FtF, true, false);
   ret->FtF_plus_precision.resize(ncols, ncols);
   ret->Features->At_mul_A(ret->FtF_plus_precision);
   return ret;
}

TEST_CASE("macauprior/make_dense_prior", "Making MacauPrior with MatrixConfig") {
    std::vector<double> x = {0.1, 0.4, -0.7, 0.3, 0.11, 0.23};

    // ColMajor case
    auto prior = make_dense_prior(3, x, 3, 2, true);

    Eigen::MatrixXf Ftrue(3, 2);
    Ftrue <<  0.1, 0.3, 0.4, 0.11, -0.7, 0.23;
    auto features_downcast1 = std::dynamic_pointer_cast<DenseSideInfo>(prior->Features); //for the purpose of the test
    REQUIRE( (*(features_downcast1->get_features()) - Ftrue).norm() == Approx(0) );
    Eigen::MatrixXf tmp = Eigen::MatrixXf::Zero(2, 2);
    tmp.triangularView<Eigen::Lower>()  = prior->FtF_plus_precision;
    tmp.triangularView<Eigen::Lower>() -= Ftrue.transpose() * Ftrue;
    REQUIRE( tmp.norm() == Approx(0) );
}

TEST_CASE("inv_norm_cdf/inv_norm_cdf", "Inverse normal CDF") {
	REQUIRE( inv_norm_cdf(0.0)  == -std::numeric_limits<float>::infinity());
	REQUIRE( inv_norm_cdf(0.5)  == Approx(0) );
	REQUIRE( inv_norm_cdf(0.9)  == Approx(1.2815515655446004) );
	REQUIRE( inv_norm_cdf(0.01) == Approx(-2.3263478740408408) );
}

TEST_CASE("truncnorm/norm_cdf", "Normal CDF") {
	REQUIRE( norm_cdf(0.0)  == Approx(0.5));
	REQUIRE( norm_cdf(-1.0) == Approx(0.15865525393145707) );
	REQUIRE( norm_cdf(-3.0) == Approx(0.0013498980316300933) );
	REQUIRE( norm_cdf(4.0)  == Approx(0.99996832875816688) );
}

TEST_CASE( "truncnorm/rand_truncnorm", "generaring random truncnorm variable" ) {
  init_bmrng(1234);
  for (int i = 0; i < 10; i++) {
    REQUIRE( rand_truncnorm(2.0) >= 2.0 );
    REQUIRE( rand_truncnorm(3.0) >= 3.0 );
    REQUIRE( rand_truncnorm(5.0) >= 5.0 );
    REQUIRE( rand_truncnorm(50.0) >= 50.0 );
    REQUIRE( rand_truncnorm(30, 2.0, 50.0) >= 50.0 );
  }
}

TEST_CASE("Benchmark from old 'data.cpp' file", "[!hide]")
{
   const int N = 32 * 1024;
   const int K = 96;
   const int R = 20;

   {
       init_bmrng(1234);
       Eigen::MatrixXf U(K,N);
       bmrandn(U);

       Eigen::MatrixXf M(K,K) ;
       float start = tick();
       for(int i=0; i<R; ++i) {
           M.setZero();
           for(int j=0; j<N;++j) {
               const auto &col = U.col(j);
               M.noalias() += col * col.transpose();
           }
       }
       float stop = tick();
       std::cout << "norm U: " << U.norm() << std::endl;
       std::cout << "norm M: " << M.norm() << std::endl;
       std::cout << "MatrixXd: " << stop - start << std::endl;
   }

   {
       init_bmrng(1234);
       Eigen::Matrix<float, K, Eigen::Dynamic> U(K,N);
       U = nrandn(K,N);

       Eigen::Matrix<float,K,K> M;
       float start = tick();
       for(int i=0; i<R; ++i) {
           M.setZero();
           for(int j=0; j<N;++j) {
               const auto &col = U.col(j);
               M.noalias() += col * col.transpose();
           }
       }
       float stop = tick();
       std::cout << "norm U: " << U.norm() << std::endl;
       std::cout << "norm M: " << M.norm() << std::endl;
       std::cout << "MatrixXd: " << stop - start << std::endl;
   }
}

TEST_CASE("randn", "Test random number generation")
{
#if defined(USE_BOOST_RANDOM) 
   #if defined(TEST_RANDOM_OK)
      INFO("Testing with correct BOOST random - all testcases should pass\n");
   #else
      WARN("Wrong BOOST version (should be 1.5x) - expect many failures\n");
   #endif
#else
   WARN("Testing with std random - expect many failures\n");
#endif

#if defined(USE_BOOST_RANDOM) 
   init_bmrng(1234);

   float rnd = 0.0;
   rnd = smurff::randn();
   REQUIRE(rnd == Approx(-1.38981).epsilon(APPROX_EPSILON));

   rnd = smurff::randn();
   REQUIRE(rnd == Approx(0.444601).epsilon(APPROX_EPSILON));

   rnd = smurff::randn();
   REQUIRE(rnd == Approx(-1.13281).epsilon(APPROX_EPSILON));

   rnd = smurff::randn();
   REQUIRE(rnd == Approx(0.708248).epsilon(APPROX_EPSILON));

   rnd = smurff::randn();
   REQUIRE(rnd == Approx(0.369621).epsilon(APPROX_EPSILON));

   rnd = smurff::randn();
   REQUIRE(rnd == Approx(-0.465294).epsilon(APPROX_EPSILON));

   rnd = smurff::randn();
   REQUIRE(rnd == Approx(-0.637987).epsilon(APPROX_EPSILON));

   rnd = smurff::randn();
   REQUIRE(rnd == Approx(0.510229).epsilon(APPROX_EPSILON));

   rnd = smurff::randn();
   REQUIRE(rnd == Approx(0.28734).epsilon(APPROX_EPSILON));

   rnd = smurff::randn();
   REQUIRE(rnd == Approx(1.22677).epsilon(APPROX_EPSILON));
   #else
   std::cout << "Running std" << std::endl;
   init_bmrng(1234);
   
   float rnd = 0.0;
   rnd = smurff::randn();
   REQUIRE(rnd == Approx(-0.00989496).epsilon(APPROX_EPSILON));

   rnd = smurff::randn();
   REQUIRE(rnd == Approx(0.557211).epsilon(APPROX_EPSILON));

   rnd = smurff::randn();
   REQUIRE(rnd == Approx(-0.511044).epsilon(APPROX_EPSILON));

   rnd = smurff::randn();
   REQUIRE(rnd == Approx(-0.321061).epsilon(APPROX_EPSILON));

   rnd = smurff::randn();
   REQUIRE(rnd == Approx(-0.59018).epsilon(APPROX_EPSILON));

   rnd = smurff::randn();
   REQUIRE(rnd == Approx(-0.0465393).epsilon(APPROX_EPSILON));

   rnd = smurff::randn();
   REQUIRE(rnd == Approx(-0.824126).epsilon(APPROX_EPSILON));

   rnd = smurff::randn();
   REQUIRE(rnd == Approx(0.315523).epsilon(APPROX_EPSILON));

   rnd = smurff::randn();
   REQUIRE(rnd == Approx(-2.20691).epsilon(APPROX_EPSILON));

   rnd = smurff::randn();
   REQUIRE(rnd == Approx(0.190217).epsilon(APPROX_EPSILON));
   
   #endif
}

TEST_CASE("rgamma", "Test random number generation")
{
   #ifdef USE_BOOST_RANDOM
   std::cout << "Running boost" << std::endl;
   init_bmrng(1234);

   float rnd = 0.0;
   rnd = smurff::rgamma(1, 2);
   REQUIRE(rnd == Approx(0.425197).epsilon(APPROX_EPSILON));

   rnd = smurff::rgamma(1, 2);
   REQUIRE(rnd == Approx(1.37697).epsilon(APPROX_EPSILON));

   rnd = smurff::rgamma(1, 2);
   REQUIRE(rnd == Approx(1.9463).epsilon(APPROX_EPSILON));

   rnd = smurff::rgamma(1, 2);
   REQUIRE(rnd == Approx(3.40572).epsilon(APPROX_EPSILON));

   rnd = smurff::rgamma(1, 2);
   REQUIRE(rnd == Approx(1.15154).epsilon(APPROX_EPSILON));

   rnd = smurff::rgamma(1, 2);
   REQUIRE(rnd == Approx(1.89408).epsilon(APPROX_EPSILON));

   rnd = smurff::rgamma(1, 2);
   REQUIRE(rnd == Approx(3.07757).epsilon(APPROX_EPSILON));

   rnd = smurff::rgamma(1, 2);
   REQUIRE(rnd == Approx(2.95121).epsilon(APPROX_EPSILON));

   rnd = smurff::rgamma(1, 2);
   REQUIRE(rnd == Approx(3.02804).epsilon(APPROX_EPSILON));

   rnd = smurff::rgamma(1, 2);
   REQUIRE(rnd == Approx(3.94182).epsilon(APPROX_EPSILON));
   
   #else
   std::cout << "Running std" << std::endl;
   init_bmrng(1234);
   
   float rnd = 0.0;
   rnd = smurff::rgamma(1, 2);
   REQUIRE(rnd == Approx(4.96088).epsilon(APPROX_EPSILON));

   rnd = smurff::rgamma(1, 2);
   REQUIRE(rnd == Approx(2.35473).epsilon(APPROX_EPSILON));

   rnd = smurff::rgamma(1, 2);
   REQUIRE(rnd == Approx(2.40984).epsilon(APPROX_EPSILON));

   rnd = smurff::rgamma(1, 2);
   REQUIRE(rnd == Approx(1.08649).epsilon(APPROX_EPSILON));

   rnd = smurff::rgamma(1, 2);
   REQUIRE(rnd == Approx(1.19907).epsilon(APPROX_EPSILON));

   rnd = smurff::rgamma(1, 2);
   REQUIRE(rnd == Approx(0.27702).epsilon(APPROX_EPSILON));

   rnd = smurff::rgamma(1, 2);
   REQUIRE(rnd == Approx(0.956223).epsilon(APPROX_EPSILON));

   rnd = smurff::rgamma(1, 2);
   REQUIRE(rnd == Approx(5.37583).epsilon(APPROX_EPSILON));

   rnd = smurff::rgamma(1, 2);
   REQUIRE(rnd == Approx(0.030197).epsilon(APPROX_EPSILON));

   rnd = smurff::rgamma(1, 2);
   REQUIRE(rnd == Approx(0.0270837).epsilon(APPROX_EPSILON));
   
   #endif
}
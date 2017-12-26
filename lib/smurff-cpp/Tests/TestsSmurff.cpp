#include "catch.hpp"

#include <Eigen/Core>

#include <SmurffCpp/Configs/Config.h>
#include <SmurffCpp/Sessions/SessionFactory.h>

using namespace smurff;

std::shared_ptr<MatrixConfig> getTrainDenseMatrixConfig()
{
   std::vector<double> trainMatrixConfigVals = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };
   std::shared_ptr<MatrixConfig> trainMatrixConfig =
      std::make_shared<MatrixConfig>(3, 4, std::move(trainMatrixConfigVals), NoiseConfig());
   return trainMatrixConfig;
}

std::shared_ptr<TensorConfig> getTrainDenseTensor2dConfig()
{
   std::vector<double> trainTensorConfigVals = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };
   std::shared_ptr<TensorConfig> trainTensorConfig =
      std::make_shared<TensorConfig>(std::initializer_list<uint64_t>({ 3, 4 }), std::move(trainTensorConfigVals), NoiseConfig());
   return trainTensorConfig;
}

std::shared_ptr<MatrixConfig> getTrainSparseMatrixConfig()
{
   std::vector<std::uint32_t> trainMatrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2 };
   std::vector<std::uint32_t> trainMatrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   std::vector<double> trainMatrixConfigVals = { 1, 2, 3, 4, 9, 10, 11, 12 };
   std::shared_ptr<MatrixConfig> trainMatrixConfig =
      std::make_shared<MatrixConfig>(3, 4, std::move(trainMatrixConfigRows), std::move(trainMatrixConfigCols), std::move(trainMatrixConfigVals), NoiseConfig(), false);
   return trainMatrixConfig;
}

std::shared_ptr<TensorConfig> getTrainSparseTensor2dConfig()
{
   std::vector<std::uint32_t> trainTensorConfigCols =
      {
         0, 0, 0, 0, 2, 2, 2, 2,
         0, 1, 2, 3, 0, 1, 2, 3
      };
   std::vector<double> trainTensorConfigVals = { 1, 2, 3, 4, 9, 10, 11, 12 };
   std::shared_ptr<TensorConfig> trainTensorConfig =
      std::make_shared<TensorConfig>(std::initializer_list<uint64_t>({ 3, 4 }), std::move(trainTensorConfigCols), std::move(trainTensorConfigVals), NoiseConfig(), true);
   return trainTensorConfig;
}

std::shared_ptr<MatrixConfig> getTestSparseMatrixConfig()
{
   std::vector<std::uint32_t> testMatrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2};
   std::vector<std::uint32_t> testMatrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   std::vector<double> testMatrixConfigVals = { 1, 2, 3, 4, 9, 10, 11, 12 };
   std::shared_ptr<MatrixConfig> testMatrixConfig =
      std::make_shared<MatrixConfig>(3, 4, std::move(testMatrixConfigRows), std::move(testMatrixConfigCols), std::move(testMatrixConfigVals), NoiseConfig(), false);
   return testMatrixConfig;
}

std::shared_ptr<TensorConfig> getTestSparseTensor2dConfig()
{
   std::vector<std::uint32_t> testTensorConfigCols =
      {
         0, 0, 0, 0, 2, 2, 2, 2,
         0, 1, 2, 3, 0, 1, 2, 3
      };
   std::vector<double> testTensorConfigVals = { 1, 2, 3, 4, 9, 10, 11, 12 };
   std::shared_ptr<TensorConfig> testTensorConfig =
      std::make_shared<TensorConfig>(std::initializer_list<uint64_t>({ 3, 4 }), std::move(testTensorConfigCols), std::move(testTensorConfigVals), NoiseConfig(), true);
   return testTensorConfig;
}

std::shared_ptr<MatrixConfig> getRowFeaturesDenseMatrixConfig()
{
   std::vector<double> rowFeaturesDenseMatrixConfigVals = { 1, 2, 3 };
   std::shared_ptr<MatrixConfig> rowFeaturesDenseMatrixConfig =
      std::make_shared<MatrixConfig>(3, 1, std::move(rowFeaturesDenseMatrixConfigVals), NoiseConfig());
   return rowFeaturesDenseMatrixConfig;
}

std::shared_ptr<MatrixConfig> getColFeaturesDenseMatrixConfig()
{
   std::vector<double> colFeaturesDenseMatrixConfigVals = { 1, 2, 3, 4 };
   std::shared_ptr<MatrixConfig> colFeaturesDenseMatrixConfig =
      std::make_shared<MatrixConfig>(1, 4, std::move(colFeaturesDenseMatrixConfigVals), NoiseConfig());
   return colFeaturesDenseMatrixConfig;
}

void REQUIRE_RESULT_ITEMS(const std::vector<ResultItem>& actualResultItems, const std::vector<ResultItem>& expectedResultItems)
{
   REQUIRE(actualResultItems.size() == expectedResultItems.size());
   for (std::vector<ResultItem>::size_type i = 0; i < actualResultItems.size(); i++)
   {
      const ResultItem& actualResultItem = actualResultItems[i];
      const ResultItem& expectedResultItem = expectedResultItems[i];
      REQUIRE(actualResultItem.coords == expectedResultItem.coords);
      REQUIRE(actualResultItem.val == expectedResultItem.val);
      REQUIRE(actualResultItem.pred_1sample == Approx(expectedResultItem.pred_1sample));
      REQUIRE(actualResultItem.pred_avg == Approx(expectedResultItem.pred_avg));
      REQUIRE(actualResultItem.var == Approx(expectedResultItem.var));
      REQUIRE(actualResultItem.stds == Approx(expectedResultItem.stds));
   }
}

//
//      train: dense matrix
//       test: sparse matrix
//     priors: normal normal
//   features: none none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior normal normal --features none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234")
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();

   Config config;
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getFeatures().push_back(std::vector<std::shared_ptr<MatrixConfig> >());
   config.getFeatures().push_back(std::vector<std::shared_ptr<MatrixConfig> >());
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);

   std::shared_ptr<ISession> session = SessionFactory::create_py_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master ce08b46ac61a783a7958720ec9e1760780eeb170
   double expectedRmseAvg = 0.4148777232391693;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  2.2997524192635832,  1.7370592984929338, 20.7793461157125172, 0.6512052580222043 },
         { { 0, 1 },  2,  3.3981721039544817,  2.1159960504024529, 26.1403820382513352, 0.7303952248297554 },
         { { 0, 2 },  3,  3.0884498605823421,  2.7526565775059657, 28.3120906318335912, 0.7601300993886533 },
         { { 0, 3 },  4,  4.1232674062768409,  3.4710086454388214, 42.3947323487428065, 0.9301605341886411 },
         { { 2, 0 },  9,  9.1825164425321546,  8.4690736380474014, 59.3684003734182042, 1.1007270359270638 },
         { { 2, 1 }, 10,  8.2950709803313494,  9.5655479877679443, 56.3404527095019247, 1.0722896797679131 },
         { { 2, 2 }, 11, 11.5399538613641557, 10.9574323035202568, 55.9898197868022578, 1.0689477926466417 },
         { { 2, 3 }, 12, 10.8882167080340153, 12.0825997901525266, 58.3087165680419020, 1.0908592060898872 }
      };

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//
//      train: sparse matrix
//       test: sparse matrix
//     priors: normal normal
//   features: none none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE("--train <train_sparse_matrix> --test <test_sparse_matrix> --prior normal normal --features none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234")
{
   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();

   Config config;
   config.setTrain(trainSparseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getFeatures().push_back(std::vector<std::shared_ptr<MatrixConfig> >());
   config.getFeatures().push_back(std::vector<std::shared_ptr<MatrixConfig> >());
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);

   std::shared_ptr<ISession> session = SessionFactory::create_py_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master ce08b46ac61a783a7958720ec9e1760780eeb170
   double expectedRmseAvg = 0.4651976649366950;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  1.8406390905521013,  1.9192742413731576, 27.2861183357222679, 0.7462302311430991 },
         { { 0, 1 },  2,  3.0465108545083197,  2.2366571231945818, 19.5411133550200837, 0.6315047361145470 },
         { { 0, 2 },  3,  2.2841282851594640,  2.7546921870135730, 20.3127668143279116, 0.6438526706606847 },
         { { 0, 3 },  4,  3.4365954536537018,  3.2682022743359518, 35.1709389601064686, 0.8472155950476647 },
         { { 2, 0 },  9,  9.7228110254532929,  8.7068818224013871, 82.8820820565497627, 1.3005656701522481 },
         { { 2, 1 }, 10,  8.6048462197675892,  9.6328831298778255, 51.2317266860101270, 1.0225191648920988 },
         { { 2, 2 }, 11, 11.7980563069531819, 11.0483951554508089, 60.7393874710037238, 1.1133639728960754 },
         { { 2, 3 }, 12, 11.0020815144410644, 12.1070524500051224, 57.7075853863754986, 1.0852215553571283 }
      };

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//
//      train: dense matrix
//       test: sparse matrix
//     priors: normal normal
//   features: dense_features dense_features
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior normal normal --features <dense_row_features> <dense_col_features> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234")
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> rowFeaturesDenseMatrixConfig = getRowFeaturesDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> colFeaturesDenseMatrixConfig = getColFeaturesDenseMatrixConfig();

   Config config;
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getFeatures().push_back({ rowFeaturesDenseMatrixConfig });
   config.getFeatures().push_back({ colFeaturesDenseMatrixConfig });
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);

   std::shared_ptr<ISession> session = SessionFactory::create_py_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master ce08b46ac61a783a7958720ec9e1760780eeb170
   double expectedRmseAvg = 0.4923359705805635;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  1.7896283495233598,  1.8265540720487814, 25.6711703934737514, 0.7238103595566349 },
         { { 0, 1 },  2,  2.6549376380335210,  2.3323840195614722, 26.0913060795546237, 0.7297092806567729 },
         { { 0, 2 },  3,  2.5387870361790914,  2.8075245347069995, 21.2224050836179536, 0.6581111667711905 },
         { { 0, 3 },  4,  3.3528800970371373,  3.2787994341024289, 24.4926713728696939, 0.7070010156859848 },
         { { 2, 0 },  9,  7.6022367820348444,  8.3017756544085533, 46.4979311605163446, 0.9741341645906803 },
         { { 2, 1 }, 10,  9.8804262990750100,  9.7705386422539515, 48.0887718030733495, 0.9906581177102272 },
         { { 2, 2 }, 11, 11.2885241899954085, 11.1050562687125005, 32.6329648023197620, 0.8160752866719649 },
         { { 2, 3 }, 12, 11.2319412121555118, 12.1926220674740815, 35.9673203004125952, 0.8567537247694809 }
      };

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//
//      train: sparse matrix
//       test: sparse matrix
//     priors: normal normal
//   features: dense_features dense_features
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE("--train <train_sparse_matrix> --test <test_sparse_matrix> --prior normal normal --features <dense_row_features> <dense_col_features> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234")
{
   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> rowFeaturesDenseMatrixConfig = getRowFeaturesDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> colFeaturesDenseMatrixConfig = getColFeaturesDenseMatrixConfig();

   Config config;
   config.setTrain(trainSparseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getFeatures().push_back({ rowFeaturesDenseMatrixConfig });
   config.getFeatures().push_back({ colFeaturesDenseMatrixConfig });
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);

   std::shared_ptr<ISession> session = SessionFactory::create_py_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master ce08b46ac61a783a7958720ec9e1760780eeb170
   double expectedRmseAvg = 0.4563100827505874;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  1.6697048043624072,  1.7860638196059275, 19.9868356423350306, 0.6386662703991912 },
         { { 0, 1 },  2,  2.6367636593649024,  2.3310901876843597, 19.9825092740564187, 0.6385971435227666 },
         { { 0, 2 },  3,  2.3610396931275712,  2.8085686864824990, 24.1839877813573274, 0.7025316868640935 },
         { { 0, 3 },  4,  3.2114239722071991,  3.3387965007673466, 30.2702648899819629, 0.7859774220413812 },
         { { 2, 0 },  9,  7.3490691710242917,  8.3719770489661975, 82.7833479941962906, 1.2997907837472191 },
         { { 2, 1 }, 10,  9.9165105505840092,  9.7634869539392319, 56.2937919647504827, 1.0718455566170930 },
         { { 2, 2 }, 11, 10.9750710585449802, 10.9002682602838359, 36.4271629701464619, 0.8622131344317728 },
         { { 2, 3 }, 12, 11.2829665833916017, 11.9359655639288071, 47.3940838789614460, 0.9834765892543950 }
      };

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//
//      train: dense matrix
//       test: sparse matrix
//     priors: spikeandslab normal
//   features: none none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior spikeandslab normal --features none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234")
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();

   Config config;
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.getPriorTypes().push_back(PriorTypes::spikeandslab);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getFeatures().push_back(std::vector<std::shared_ptr<MatrixConfig> >());
   config.getFeatures().push_back(std::vector<std::shared_ptr<MatrixConfig> >());
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);

   std::shared_ptr<ISession> session = SessionFactory::create_py_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master ce08b46ac61a783a7958720ec9e1760780eeb170
   double expectedRmseAvg = 0.6206681329233389;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  2.2163520632118265,  1.9816048194880975,  8.3860512644964107, 0.4136954234180861 },
         { { 0, 1 },  2,  2.3906588067293937,  2.2802560118438056, 11.3278673278835811, 0.4808128179189643 },
         { { 0, 2 },  3,  2.8775907307172859,  2.5633705554310948, 16.6061313351497866, 0.5821517323626657 },
         { { 0, 3 },  4,  3.5095516143926275,  2.9356547633773604, 20.8406592945208118, 0.6521652991682402 },
         { { 2, 0 },  9,  8.9273597722179225,  8.3556267846542660, 33.6499437364078204, 0.8286938793311894 },
         { { 2, 1 }, 10,  9.6294589720400019,  9.6085963375869454, 40.2492467273388002, 0.9063184862492425 },
         { { 2, 2 }, 11, 11.5907973993468563, 10.7575895951788176, 44.7432284092971955, 0.9555768469326893 },
         { { 2, 3 }, 12, 14.1363055179274415, 12.2984752409140778, 41.4906604164368460, 0.9201892043292059 }
      };

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//
//      train: dense matrix
//       test: sparse matrix
//     priors: spikeandslab normal
//   features: dense_features none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior spikeandslab normal --features dense_row_features none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234")
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> rowFeaturesDenseMatrixConfig = getRowFeaturesDenseMatrixConfig();

   Config config;
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.getPriorTypes().push_back(PriorTypes::spikeandslab);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getFeatures().push_back({ rowFeaturesDenseMatrixConfig });
   config.getFeatures().push_back(std::vector<std::shared_ptr<MatrixConfig> >());
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);

   std::shared_ptr<ISession> session = SessionFactory::create_py_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master ce08b46ac61a783a7958720ec9e1760780eeb170
   double expectedRmseAvg = 0.5988320428930997;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  2.1737472697789544,  2.1326695136378913,  9.0707339729104497, 0.4302522746660560 },
         { { 0, 1 },  2,  2.8748962739977384,  2.5234536310788340,  9.5445975972433263, 0.4413476023115910 },
         { { 0, 2 },  3,  3.1564722791558459,  2.8226989925935135, 12.0817925194030078, 0.4965553285118696 },
         { { 0, 3 },  4,  3.1490842622856450,  3.1196855189022488, 15.6021167629049504, 0.5642787840967740 },
         { { 2, 0 },  9,  7.9104883498311311,  8.3287292399052095, 40.0700707279277282, 0.9042989248406264 },
         { { 2, 1 }, 10, 10.4620411943034313,  9.8860621122426480, 31.5998605105781785, 0.8030536174258722 },
         { { 2, 2 }, 11, 11.4867250383556900, 11.0548611199080451, 36.6189180603447610, 0.8644795303385632 },
         { { 2, 3 }, 12, 11.4598392903238953, 12.1971530948012337, 32.5296713249058911, 0.8147826970213754 }
      };

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//
//      train: sparse 2D-tensor (matrix)
//       test: sparse 2D-tensor (matrix)
//     priors: normal normal
//   features: none none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE("--train <train_sparse_tensor> --test <test_sparse_tensor> --prior normal normal --features none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234")
{
   std::shared_ptr<TensorConfig> trainSparseTensorConfig = getTrainSparseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();

   Config config;
   config.setTrain(trainSparseTensorConfig);
   config.setTest(testSparseTensorConfig);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getFeatures().push_back(std::vector<std::shared_ptr<MatrixConfig> >());
   config.getFeatures().push_back(std::vector<std::shared_ptr<MatrixConfig> >());
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);

   std::shared_ptr<ISession> session = SessionFactory::create_py_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master ce08b46ac61a783a7958720ec9e1760780eeb170
   double expectedRmseAvg = 0.4323854663303525;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  1.8541410944574841,  1.7767970217065547, 19.6988847520554238, 0.6340489383039738 },
         { { 0, 1 },  2,  3.2665045876208256,  2.2199148669997726, 20.9991128860445428, 0.6546398431236756 },
         { { 0, 2 },  3,  2.9603957155794047,  2.7068860487677879, 27.5289354824938073, 0.7495432007875279 },
         { { 0, 3 },  4,  3.8432554038064999,  3.2382031823632884, 24.6239205123522140, 0.7088927916462434 },
         { { 2, 0 },  9,  9.7079896960556766,  8.8272086786817141, 71.8317171716461758, 1.2107656303621475 },
         { { 2, 1 }, 10,  8.4116500121248148,  9.6397015671928123, 53.7651171828327819, 1.0474957228769801 },
         { { 2, 2 }, 11, 11.6675678529136988, 11.1319935784364024, 57.0847109463449840, 1.0793489245586045 },
         { { 2, 3 }, 12, 10.8696565965651395, 11.9768249192672869, 54.3060790060203473, 1.0527522627159138 }
      };

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}
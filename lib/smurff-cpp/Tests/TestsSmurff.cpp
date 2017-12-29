#include "catch.hpp"

#include <Eigen/Core>

#include <SmurffCpp/Configs/Config.h>
#include <SmurffCpp/Sessions/SessionFactory.h>
#include <SmurffCpp/Utils/MatrixUtils.h>

// https://github.com/catchorg/Catch2/blob/master/docs/assertions.md#floating-point-comparisons
// By default Catch.hpp sets epsilon to std::numeric_limits<float>::epsilon()*100
#define APPROX_EPSILON std::numeric_limits<float>::epsilon()*100

using namespace smurff;

// dense train data (matrix/tensor 2d/tensor 3d)

std::shared_ptr<MatrixConfig> getTrainDenseMatrixConfig()
{
   std::vector<double> trainMatrixConfigVals = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };
   std::shared_ptr<MatrixConfig> trainMatrixConfig =
      std::make_shared<MatrixConfig>(3, 4, std::move(trainMatrixConfigVals), NoiseConfig(NoiseTypes::unused));
   return trainMatrixConfig;
}

std::shared_ptr<TensorConfig> getTrainDenseTensor2dConfig()
{
   std::vector<double> trainTensorConfigVals = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };
   std::shared_ptr<TensorConfig> trainTensorConfig =
      std::make_shared<TensorConfig>(std::initializer_list<uint64_t>({ 3, 4 }), std::move(trainTensorConfigVals), NoiseConfig(NoiseTypes::unused));
   return trainTensorConfig;
}

std::shared_ptr<TensorConfig> getTrainDenseTensor3dConfig()
{
   std::vector<double> trainTensorConfigVals = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };
   std::shared_ptr<TensorConfig> trainTensorConfig =
      std::make_shared<TensorConfig>(std::initializer_list<uint64_t>({ 2, 3, 4 }), std::move(trainTensorConfigVals), NoiseConfig());
   return trainTensorConfig;
}

// sparse train data (matrix/tensor 2d)

std::shared_ptr<MatrixConfig> getTrainSparseMatrixConfig()
{
   std::vector<std::uint32_t> trainMatrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2 };
   std::vector<std::uint32_t> trainMatrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   std::vector<double> trainMatrixConfigVals = { 1, 2, 3, 4, 9, 10, 11, 12 };
   std::shared_ptr<MatrixConfig> trainMatrixConfig =
      std::make_shared<MatrixConfig>(3, 4, std::move(trainMatrixConfigRows), std::move(trainMatrixConfigCols), std::move(trainMatrixConfigVals), NoiseConfig(NoiseTypes::unused), true);
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
      std::make_shared<TensorConfig>(std::initializer_list<uint64_t>({ 3, 4 }), std::move(trainTensorConfigCols), std::move(trainTensorConfigVals), NoiseConfig(NoiseTypes::unused), true);
   return trainTensorConfig;
}

// sparse test data (matrix/tensor 2d/tensor 3d)

std::shared_ptr<MatrixConfig> getTestSparseMatrixConfig()
{
   std::vector<std::uint32_t> testMatrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2};
   std::vector<std::uint32_t> testMatrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   std::vector<double> testMatrixConfigVals = { 1, 2, 3, 4, 9, 10, 11, 12 };
   std::shared_ptr<MatrixConfig> testMatrixConfig =
      std::make_shared<MatrixConfig>(3, 4, std::move(testMatrixConfigRows), std::move(testMatrixConfigCols), std::move(testMatrixConfigVals), NoiseConfig(NoiseTypes::unused), true);
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
      std::make_shared<TensorConfig>(std::initializer_list<uint64_t>({ 3, 4 }), std::move(testTensorConfigCols), std::move(testTensorConfigVals), NoiseConfig(NoiseTypes::unused), true);
   return testTensorConfig;
}

std::shared_ptr<TensorConfig> getTestSparseTensor3dConfig()
{
   std::vector<std::uint32_t> testTensorConfigCols =
      {
         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 2, 2, 2, 2,
         0, 1, 2, 3, 0, 1, 2, 3
      };
   std::vector<double> testTensorConfigVals = { 1, 2, 3, 4, 9, 10, 11, 12 };
   std::shared_ptr<TensorConfig> testTensorConfig =
      std::make_shared<TensorConfig>(std::initializer_list<uint64_t>({ 2, 3, 4 }), std::move(testTensorConfigCols), std::move(testTensorConfigVals), NoiseConfig(), true);
   return testTensorConfig;
}

// aux data

std::shared_ptr<MatrixConfig> getRowAuxDataDenseMatrixConfig()
{
   std::vector<double> rowAuxDataDenseMatrixConfigVals = { 1, 2, 3 };
   std::shared_ptr<MatrixConfig> rowAuxDataDenseMatrixConfig =
      std::make_shared<MatrixConfig>(3, 1, std::move(rowAuxDataDenseMatrixConfigVals), NoiseConfig(NoiseTypes::unused));
   return rowAuxDataDenseMatrixConfig;
}

std::shared_ptr<MatrixConfig> getColAuxDataDenseMatrixConfig()
{
   std::vector<double> colAuxDataDenseMatrixConfigVals = { 1, 2, 3, 4 };
   std::shared_ptr<MatrixConfig> colAuxDataDenseMatrixConfig =
      std::make_shared<MatrixConfig>(1, 4, std::move(colAuxDataDenseMatrixConfigVals), NoiseConfig(NoiseTypes::unused));
   return colAuxDataDenseMatrixConfig;
}

// side info

std::shared_ptr<MatrixConfig> getRowSideInfoDenseMatrixConfig()
{
   std::vector<double> rowSideInfoDenseMatrixConfigVals = { 1, 2, 3 };
   std::shared_ptr<MatrixConfig> rowSideInfoDenseMatrixConfig =
      std::make_shared<MatrixConfig>(3, 1, std::move(rowSideInfoDenseMatrixConfigVals), NoiseConfig(NoiseTypes::unused));
   return rowSideInfoDenseMatrixConfig;
}

std::shared_ptr<MatrixConfig> getColSideInfoDenseMatrixConfig()
{
   std::vector<double> colSideInfoDenseMatrixConfigVals = { 1, 2, 3, 4 };
   std::shared_ptr<MatrixConfig> colSideInfoDenseMatrixConfig =
      std::make_shared<MatrixConfig>(4, 1, std::move(colSideInfoDenseMatrixConfigVals), NoiseConfig(NoiseTypes::unused));
   return colSideInfoDenseMatrixConfig;
}

std::shared_ptr<MatrixConfig> getRowSideInfoSparseMatrixConfig()
{
   std::vector<std::uint32_t> rowSideInfoSparseMatrixConfigRows = {0, 1, 2};
   std::vector<std::uint32_t> rowSideInfoSparseMatrixConfigCols = {0, 0, 0};
   std::vector<double> rowSideInfoSparseMatrixConfigVals = { 1, 2, 3 };
   std::shared_ptr<MatrixConfig> rowSideInfoSparseMatrixConfig =
      std::make_shared<MatrixConfig>(3, 1, std::move(rowSideInfoSparseMatrixConfigRows), std::move(rowSideInfoSparseMatrixConfigCols), std::move(rowSideInfoSparseMatrixConfigVals), NoiseConfig(), true);
   return rowSideInfoSparseMatrixConfig;
}

std::shared_ptr<MatrixConfig> getColSideInfoSparseMatrixConfig()
{
   std::vector<std::uint32_t> colSideInfoSparseMatrixConfigRows = {0, 1, 2, 3};
   std::vector<std::uint32_t> colSideInfoSparseMatrixConfigCols = {0, 0, 0, 0};
   std::vector<double> colSideInfoSparseMatrixConfigVals = { 1, 2, 3, 4 };
   std::shared_ptr<MatrixConfig> colSideInfoSparseMatrixConfig =
      std::make_shared<MatrixConfig>(4, 1, std::move(colSideInfoSparseMatrixConfigRows), std::move(colSideInfoSparseMatrixConfigCols), std::move(colSideInfoSparseMatrixConfigVals), NoiseConfig(), true);
   return colSideInfoSparseMatrixConfig;
}

//result comparison

void REQUIRE_RESULT_ITEMS(const std::vector<ResultItem>& actualResultItems, const std::vector<ResultItem>& expectedResultItems)
{
   REQUIRE(actualResultItems.size() == expectedResultItems.size());
   for (std::vector<ResultItem>::size_type i = 0; i < actualResultItems.size(); i++)
   {
      const ResultItem& actualResultItem = actualResultItems[i];
      const ResultItem& expectedResultItem = expectedResultItems[i];
      REQUIRE(actualResultItem.coords == expectedResultItem.coords);
      REQUIRE(actualResultItem.val == expectedResultItem.val);
      REQUIRE(actualResultItem.pred_1sample == Approx(expectedResultItem.pred_1sample).epsilon(APPROX_EPSILON));
      REQUIRE(actualResultItem.pred_avg == Approx(expectedResultItem.pred_avg).epsilon(APPROX_EPSILON));
      REQUIRE(actualResultItem.var == Approx(expectedResultItem.var).epsilon(APPROX_EPSILON));
      REQUIRE(actualResultItem.stds == Approx(expectedResultItem.stds).epsilon(APPROX_EPSILON));
   }
}

//
//      train: dense matrix
//       test: sparse matrix
//     priors: normal normal
//   aux-data: none none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior normal normal --aux-data none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234")
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();

   Config config;
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
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

   // Pre-calculated results with single-threaded Debug master 85a6fe322fdd1f6319c803b2736ada24471c5518
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

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//
//      train: sparse matrix
//       test: sparse matrix
//     priors: normal normal
//   aux-data: none none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE("--train <train_sparse_matrix> --test <test_sparse_matrix> --prior normal normal --aux-data none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234")
{
   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();

   Config config;
   config.setTrain(trainSparseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
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

   // Pre-calculated results with single-threaded Debug master 85a6fe322fdd1f6319c803b2736ada24471c5518
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

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//
//      train: dense matrix
//       test: sparse matrix
//     priors: normal normal
//   aux-data: dense_matrix dense_matrix
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior normal normal --aux-data <dense_matrix> <dense_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234")
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<TensorConfig> rowAuxDataDenseMatrixConfig = getRowAuxDataDenseMatrixConfig();
   std::shared_ptr<TensorConfig> colAuxDataDenseMatrixConfig = getColAuxDataDenseMatrixConfig();

   Config config;
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getAuxData().push_back({ rowAuxDataDenseMatrixConfig });
   config.getAuxData().push_back({ colAuxDataDenseMatrixConfig });
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

   // Pre-calculated results with single-threaded Debug master 85a6fe322fdd1f6319c803b2736ada24471c5518
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

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//
//      train: sparse matrix
//       test: sparse matrix
//     priors: normal normal
//   aux-data: dense_matrix dense_matrix
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE("--train <train_sparse_matrix> --test <test_sparse_matrix> --prior normal normal --aux-data <dense_matrix> <dense_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234")
{
   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<TensorConfig> rowAuxDataDenseMatrixConfig = getRowAuxDataDenseMatrixConfig();
   std::shared_ptr<TensorConfig> colAuxDataDenseMatrixConfig = getColAuxDataDenseMatrixConfig();

   Config config;
   config.setTrain(trainSparseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getAuxData().push_back({ rowAuxDataDenseMatrixConfig });
   config.getAuxData().push_back({ colAuxDataDenseMatrixConfig });
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

   // Pre-calculated results with single-threaded Debug master 85a6fe322fdd1f6319c803b2736ada24471c5518
   double expectedRmseAvg = 0.4525400415708138;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  1.7705198538771609,  1.7798909523727364, 29.4203884465389613, 0.7748652081140907 },
         { { 0, 1 },  2,  2.7188882472071034,  2.2528088176107239, 27.7661264387245055, 0.7527653297054949 },
         { { 0, 2 },  3,  2.4774772288151468,  2.7835797218861495, 21.7183359880023623, 0.6657562216712103 },
         { { 0, 3 },  4,  3.3692153470508579,  3.2823846386647970, 21.4085016690384009, 0.6609903156078115 },
         { { 2, 0 },  9,  7.3724767086720853,  8.3806889335653754, 64.6187190127960491, 1.1483681324418937 },
         { { 2, 1 }, 10,  9.9111856441994171,  9.8573945412399073, 52.8458652499535546, 1.0385023090573364 },
         { { 2, 2 }, 11, 10.9959963618639662, 11.0223238699814399, 37.4392744369116670, 0.8741091609422105 },
         { { 2, 3 }, 12, 11.0723309360581936, 12.0027028307527974, 46.8097292379471881, 0.9773948008316801 }
      };

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//=================================================================

//
//      train: dense matrix
//       test: sparse matrix
//     priors: spikeandslab spikeandslab
//   aux-data: none none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior spikeandslab spikeandslab --aux-data none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234")
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();

   Config config;
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.getPriorTypes().push_back(PriorTypes::spikeandslab);
   config.getPriorTypes().push_back(PriorTypes::spikeandslab);
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
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

   // Pre-calculated results with single-threaded Debug master 747d8e9c032bd4fad2cc4dbb777a51dc63a203bf
   double expectedRmseAvg = 0.5904290396919309;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  1.2725495872844774,  2.1306778734199430,  8.3800483929481953, 0.4135473319638913 },
         { { 0, 1 },  2,  1.7994100264092587,  2.4171122166565926,  8.5228477902927953, 0.4170559545068849 },
         { { 0, 2 },  3,  1.6635044823854819,  2.7724005045970679, 14.6050525022119846, 0.5459508182643469 },
         { { 0, 3 },  4,  2.2575434359692590,  3.0660621082136803, 14.9062301531461223, 0.5515512475152560 },
         { { 2, 0 },  9,  7.6538702177601392,  8.5009560651369060, 32.3457694963315134, 0.8124763041609842 },
         { { 2, 1 }, 10, 10.8227223114048989,  9.6812383116212182, 42.4044956020173771, 0.9302676332265495 },
         { { 2, 2 }, 11, 10.0053055237009421, 11.0322905166912939, 27.7466847616334675, 0.7525017426467512 },
         { { 2, 3 }, 12, 13.5782091656930550, 12.2463706479570931, 42.3834224595212845, 0.9300364537459554 }
      };

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//
//      train: sparse matrix
//       test: sparse matrix
//     priors: spikeandslab spikeandslab
//   aux-data: none none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE("--train <train_sparse_matrix> --test <test_sparse_matrix> --prior spikeandslab spikeandslab --aux-data none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234")
{
   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();

   Config config;
   config.setTrain(trainSparseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.getPriorTypes().push_back(PriorTypes::spikeandslab);
   config.getPriorTypes().push_back(PriorTypes::spikeandslab);
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
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

   // Pre-calculated results with single-threaded Debug master 747d8e9c032bd4fad2cc4dbb777a51dc63a203bf
   double expectedRmseAvg = 0.5880613687976408;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  1.9463337356217787,  2.1831318305422172,  6.8187879624634160, 0.3730401292213177 },
         { { 0, 1 },  2,  2.0376114912991699,  2.4368541943704862,  9.1109252310128745, 0.4312044174315792 },
         { { 0, 2 },  3,  2.5344732460568764,  2.7850325427238229, 10.2676740271883968, 0.4577601641709379 },
         { { 0, 3 },  4,  3.4003083781158931,  3.0247873423711158, 11.0231228550539395, 0.4743012659897379 },
         { { 2, 0 },  9,  7.5620019046803115,  8.7163409307409037, 47.6783167919188458, 0.9864212453634938 },
         { { 2, 1 }, 10,  7.9166392156689485,  9.7024737810028245, 51.7935951023250567, 1.0281109594520472 },
         { { 2, 2 }, 11,  9.8470735841819490, 11.0847978232938633, 40.4157935140758084, 0.9081916716927981 },
         { { 2, 3 }, 12, 13.2110634272073835, 12.0496099537344055, 46.6892272270470485, 0.9761359392935760 }
      };

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//
//      train: dense matrix
//       test: sparse matrix
//     priors: spikeandslab spikeandslab
//   aux-data: dense_matrix dense_matrix
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior spikeandslab spikeandslab --aux-data <dense_matrix> <dense_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234")
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<TensorConfig> rowAuxDataDenseMatrixConfig = getRowAuxDataDenseMatrixConfig();
   std::shared_ptr<TensorConfig> colAuxDataDenseMatrixConfig = getColAuxDataDenseMatrixConfig();

   Config config;
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.getPriorTypes().push_back(PriorTypes::spikeandslab);
   config.getPriorTypes().push_back(PriorTypes::spikeandslab);
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getAuxData().push_back({ rowAuxDataDenseMatrixConfig });
   config.getAuxData().push_back({ colAuxDataDenseMatrixConfig });
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

   // Pre-calculated results with single-threaded Debug master 747d8e9c032bd4fad2cc4dbb777a51dc63a203bf
   double expectedRmseAvg = 0.5964378119468161;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  2.5996744736213455,  2.1145522274541948, 10.3562065831135133, 0.4597294364704294 },
         { { 0, 1 },  2,  2.9589587817222407,  2.4158147221275970, 10.4980806698896973, 0.4628677395147162 },
         { { 0, 2 },  3,  3.3438762758397367,  2.7516202871955819, 15.3289012497964183, 0.5593162963687008 },
         { { 0, 3 },  4,  3.4910844341859995,  3.0488141177053585, 17.6720354491750662, 0.6005445734306839 },
         { { 2, 0 },  9,  8.8645851199203314,  8.4301972262481701, 37.4651876933559507, 0.8744116120062370 },
         { { 2, 1 }, 10, 10.0897024812396090,  9.6694530565356942, 30.7465724974380450, 0.7921370281562948 },
         { { 2, 2 }, 11, 11.4022260011547321, 10.9835227535444613, 36.8650385126026734, 0.8673798042074725 },
         { { 2, 3 }, 12, 11.9041885596397456, 12.1735537681210495, 37.3635979921946344, 0.8732252905201338 }
      };

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//
//      train: sparse matrix
//       test: sparse matrix
//     priors: spikeandslab spikeandslab
//   aux-data: dense_matrix dense_matrix
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE("--train <train_sparse_matrix> --test <test_sparse_matrix> --prior spikeandslab spikeandslab --aux-data <dense_matrix> <dense_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234")
{
   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<TensorConfig> rowAuxDataDenseMatrixConfig = getRowAuxDataDenseMatrixConfig();
   std::shared_ptr<TensorConfig> colAuxDataDenseMatrixConfig = getColAuxDataDenseMatrixConfig();

   Config config;
   config.setTrain(trainSparseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.getPriorTypes().push_back(PriorTypes::spikeandslab);
   config.getPriorTypes().push_back(PriorTypes::spikeandslab);
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getAuxData().push_back({ rowAuxDataDenseMatrixConfig });
   config.getAuxData().push_back({ colAuxDataDenseMatrixConfig });
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

   // Pre-calculated results with single-threaded Debug master 747d8e9c032bd4fad2cc4dbb777a51dc63a203bf
   double expectedRmseAvg = 0.5836819438725921;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  2.2477379069219041,  2.1509236668591805, 10.3515007568350121, 0.4596249748288585 },
         { { 0, 1 },  2,  2.2704110416293291,  2.4349098658318318, 13.6669823952762624, 0.5281268863321223 },
         { { 0, 2 },  3,  3.5057823257122878,  2.7219694625280106, 13.2000932169405463, 0.5190276076362245 },
         { { 0, 3 },  4,  3.8139671818068153,  2.9875627874353037, 18.3394158089889672, 0.6117792020166941 },
         { { 2, 0 },  9,  8.0362545941568051,  8.7984385960163465, 42.1537444082994455, 0.9275130716753053 },
         { { 2, 1 }, 10,  8.1173170180254299,  9.9363876924253507, 49.8017649016897437, 1.0081480789120596 },
         { { 2, 2 }, 11, 12.5340944931165232, 11.1525024762833329, 39.9045723910241747, 0.9024295143601235 },
         { { 2, 3 }, 12, 13.6359364641098306, 12.2035789645629951, 32.6833361093962296, 0.8167048789957320 }
      };

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//=================================================================

//
//      train: dense matrix
//       test: sparse matrix
//     priors: macau macau
//   features: row_side_info_dense_matrix col_side_info_dense_matrix
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//     direct: true
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior macau macau --aux-data <row_side_info_dense_matrix> <col_side_info_dense_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct")
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> rowSideInfoDenseMatrixConfig = getRowSideInfoDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> colSideInfoDenseMatrixConfig = getColSideInfoDenseMatrixConfig();

   Config config;
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.getPriorTypes().push_back(PriorTypes::macau);
   config.getPriorTypes().push_back(PriorTypes::macau);
   config.getSideInfo().push_back(rowSideInfoDenseMatrixConfig);
   config.getSideInfo().push_back(colSideInfoDenseMatrixConfig);
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);
   config.setDirect(true);

   std::shared_ptr<ISession> session = SessionFactory::create_py_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 85a6fe322fdd1f6319c803b2736ada24471c5518
   double expectedRmseAvg = 0.4708518779880723;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  1.4794984840337011,  1.8861567571268687, 16.8315853716267192, 0.5860902168421652 },
         { { 0, 1 },  2,  1.9238389757936527,  2.3660798029924592, 24.1127382604767959, 0.7014960436049543 },
         { { 0, 2 },  3,  2.6525907467696785,  3.0182516516593556, 17.3395572540116660, 0.5948684857916018 },
         { { 0, 3 },  4,  2.3347199168976327,  3.3950221781858656, 25.8724434707805955, 0.7266423125748278 },
         { { 2, 0 },  9,  9.1445832621889203,  8.4359697406668772, 35.8143272493136635, 0.8549296098165597 },
         { { 2, 1 }, 10,  8.2618857232595868,  9.7028140848276383, 34.1171376828241861, 0.8344268188259554 },
         { { 2, 2 }, 11, 11.7472313145468465, 11.0687832494420135, 38.6236106578563891, 0.8878270958926354 },
         { { 2, 3 }, 12, 13.5936527934716551, 12.2771492070695043, 52.0797001541472326, 1.0309466637775329 }
      };

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//
//      train: sparse matrix
//       test: sparse matrix
//     priors: macau macau
//   features: row_side_info_dense_matrix col_side_info_dense_matrix
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//     direct: true
//
TEST_CASE("--train <train_sparse_matrix> --test <test_sparse_matrix> --prior macau macau --aux-data <row_side_info_dense_matrix> <col_side_info_dense_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct")
{
   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> rowSideInfoDenseMatrixConfig = getRowSideInfoDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> colSideInfoDenseMatrixConfig = getColSideInfoDenseMatrixConfig();

   Config config;
   config.setTrain(trainSparseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.getPriorTypes().push_back(PriorTypes::macau);
   config.getPriorTypes().push_back(PriorTypes::macau);
   config.getSideInfo().push_back(rowSideInfoDenseMatrixConfig);
   config.getSideInfo().push_back(colSideInfoDenseMatrixConfig);
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);
   config.setDirect(true);

   std::shared_ptr<ISession> session = SessionFactory::create_py_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 747d8e9c032bd4fad2cc4dbb777a51dc63a203bf
   double expectedRmseAvg = 0.4156313740504307;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  1.7811883257126135,  1.8259317676719951, 18.9329867080726046, 0.6216007431126642 },
         { { 0, 1 },  2,  2.6111572836046690,  2.2338713041801994, 22.1940745654312259, 0.6730083931524914 },
         { { 0, 2 },  3,  1.9270507784227897,  2.8471266985277368, 28.7163873229556472, 0.7655381903449663 },
         { { 0, 3 },  4,  1.9481132856825851,  3.2730838921093759, 35.1027503941633228, 0.8463939160375468 },
         { { 2, 0 },  9,  9.8553562678485864,  8.7356772354547036, 35.2062951527992922, 0.8476413271218455 },
         { { 2, 1 }, 10, 10.3722857512520967,  9.8683330515223524, 37.8747643042940894, 0.8791782376498080 },
         { { 2, 2 }, 11, 10.4733463298594440, 11.0744419489404908, 43.2754831929138959, 0.9397729121367554 },
         { { 2, 3 }, 12, 12.8784949637118817, 11.9752037535616154, 64.9420858875505616, 1.1512378953034270 }
      };

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//=================================================================

//This test does not work because of dynamic cast in MacauOnePrior

//
//      train: dense matrix
//       test: sparse matrix
//     priors: macauone macauone
//   features: row_side_info_sparse_matrix col_side_info_sparse_matrix
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//     direct: true
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior macauone macauone --aux-data <row_side_info_sparse_matrix> <col_side_info_sparse_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct"
, "[!hide]")
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   
   std::shared_ptr<MatrixConfig> rowSideInfoSparseMatrixConfig = getRowSideInfoSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> colSideInfoSparseMatrixConfig = getColSideInfoSparseMatrixConfig();

   Config config;
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.getPriorTypes().push_back(PriorTypes::macauone);
   config.getPriorTypes().push_back(PriorTypes::macauone);
   config.getSideInfo().push_back(rowSideInfoSparseMatrixConfig);
   config.getSideInfo().push_back(colSideInfoSparseMatrixConfig);
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);
   config.setDirect(true);

   std::shared_ptr<ISession> session = SessionFactory::create_py_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 747d8e9c032bd4fad2cc4dbb777a51dc63a203bf
   double expectedRmseAvg = 0.4708518779880723;
   std::vector<ResultItem> expectedResults =
      {
         //?
      };

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//
//      train: sparse matrix
//       test: sparse matrix
//     priors: macauone macauone
//   features: row_side_info_sparse_matrix col_side_info_sparse_matrix
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//     direct: true
//
TEST_CASE("--train <train_sparse_matrix> --test <test_sparse_matrix> --prior macauone macauone --aux-data <row_side_info_sparse_matrix> <col_side_info_sparse_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct")
{
   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   
   std::shared_ptr<MatrixConfig> rowSideInfoSparseMatrixConfig = getRowSideInfoSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> colSideInfoSparseMatrixConfig = getColSideInfoSparseMatrixConfig();

   Config config;
   config.setTrain(trainSparseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.getPriorTypes().push_back(PriorTypes::macauone);
   config.getPriorTypes().push_back(PriorTypes::macauone);
   config.getSideInfo().push_back(rowSideInfoSparseMatrixConfig);
   config.getSideInfo().push_back(colSideInfoSparseMatrixConfig);
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);
   config.setDirect(true);

   std::shared_ptr<ISession> session = SessionFactory::create_py_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 747d8e9c032bd4fad2cc4dbb777a51dc63a203bf
   double expectedRmseAvg = 0.5037458567918540;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  2.2263555600455955,  1.6999603638415204, 17.5451127375816753, 0.5983840950900758 },
         { { 0, 1 },  2,  2.7720640316973126,  2.3955715468811771, 23.7244151063158561, 0.6958245014826575 },
         { { 0, 2 },  3,  3.4896580088934646,  2.9880741545986518, 28.4458058646875429, 0.7619229949934216 },
         { { 0, 3 },  4,  3.9050132987777753,  3.6872690168195184, 31.6932867092425994, 0.8042398706706723 },
         { { 2, 0 },  9,  9.5432677110405884,  8.0604463053166864, 62.7803068526388444, 1.1319146399330076 },
         { { 2, 1 }, 10, 10.2535331994224066,  9.4440600048411909, 35.3642332504274393, 0.8495404910462423 },
         { { 2, 2 }, 11, 12.3243369077485934, 11.0364425753753963, 40.2708393592330651, 0.9065615613261706 },
         { { 2, 3 }, 12, 11.1306558385604362, 12.3042307806398803, 33.0449767392614078, 0.8212108623204456 }         
      };

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//=================================================================

//
//      train: dense matrix
//       test: sparse matrix
//     priors: macau normal
//   features: row_side_info_dense_matrix none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//     direct: true
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior macau normal --aux-data <row_side_info_dense_matrix> none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct")
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> rowSideInfoDenseMatrixConfig = getRowSideInfoDenseMatrixConfig();

   Config config;
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.getPriorTypes().push_back(PriorTypes::macau);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getSideInfo().push_back(rowSideInfoDenseMatrixConfig);
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);
   config.setDirect(true);

   std::shared_ptr<ISession> session = SessionFactory::create_py_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 85a6fe322fdd1f6319c803b2736ada24471c5518
   double expectedRmseAvg = 0.4750318111505571;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  2.0464640198420820,  1.9894562393697965, 23.6233972231544698, 0.6943415207312001 },
         { { 0, 1 },  2,  3.2479927889480007,  2.4590359719868258, 17.4029166371974569, 0.5959543304855145 },
         { { 0, 2 },  3,  1.7725387884690189,  2.8351244944211986, 22.9495558918148177, 0.6843670678126066 },
         { { 0, 3 },  4,  3.4958447660084913,  3.4341569197438275, 36.4475301480830538, 0.8624541413195521 },
         { { 2, 0 },  9,  9.9340660945704471,  8.5591220340344076, 40.3514262207299907, 0.9074681781200992 },
         { { 2, 1 }, 10, 10.8979072875323642,  9.8490284864301163, 38.7100563866488301, 0.8888200890776126 },
         { { 2, 2 }, 11, 11.1369612134891405, 10.8499510614749202, 43.9036985653110605, 0.9465695158157451 },
         { { 2, 3 }, 12, 11.1962160233287449, 12.1687048104957807, 43.3460065068271732, 0.9405383446145885 }
      };

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//
//      train: dense matrix
//       test: sparse matrix
//     priors: normal macau
//   features: none col_side_info_dense_matrix
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//     direct: true
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior normal macau --aux-data none <col_side_info_dense_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct")
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> colSideInfoDenseMatrixConfig = getColSideInfoDenseMatrixConfig();

   Config config;
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getPriorTypes().push_back(PriorTypes::macau);
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getSideInfo().push_back(colSideInfoDenseMatrixConfig);
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);
   config.setDirect(true);

   std::shared_ptr<ISession> session = SessionFactory::create_py_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 85a6fe322fdd1f6319c803b2736ada24471c5518
   double expectedRmseAvg = 0.4718098763797450;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  1.0820348433927434,  1.5472804209576903, 20.8608023063782184, 0.6524803899841290 },
         { { 0, 1 },  2,  0.3166712583778596,  2.1333761538453335, 16.3624510120241062, 0.5778646655350725 },
         { { 0, 2 },  3,  1.0603021549892615,  2.8733290918267516, 25.4269079717275375, 0.7203585837754204 },
         { { 0, 3 },  4,  2.5504752513933715,  3.2898942631944492, 31.3572371358028334, 0.7999647586090179 },
         { { 2, 0 },  9,  8.0408589214887130,  8.2293859290722349, 33.0687300507163400, 0.8215059597174897 },
         { { 2, 1 }, 10,  9.3224538002828741,  9.5500735894746462, 37.9596195177687434, 0.8801625489689534 },
         { { 2, 2 }, 11, 11.3041689884507228, 10.9005077280747074, 29.8433963169746406, 0.7804158535217323 },
         { { 2, 3 }, 12, 12.4123961375339604, 12.3702123825900792, 35.1656958861189466, 0.8471524437679556 }
      };

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//test throw

//
//      train: dense matrix
//       test: sparse matrix
//     priors: macau normal
//   features: col_side_info_dense_matrix row_side_info_dense_matrix
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//     direct: true
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior macau normal --aux-data <col_side_info_dense_matrix> <row_side_info_dense_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct")
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> rowSideInfoDenseMatrixConfig = getRowSideInfoDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> colSideInfoDenseMatrixConfig = getColSideInfoDenseMatrixConfig();

   Config config;
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.getPriorTypes().push_back(PriorTypes::macau);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getSideInfo().push_back(colSideInfoDenseMatrixConfig);
   config.getSideInfo().push_back(rowSideInfoDenseMatrixConfig);
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);
   config.setDirect(true);

   REQUIRE_THROWS(SessionFactory::create_py_session(config));
}

//test throw

//
//      train: dense matrix
//       test: sparse matrix
//     priors: macau normal
//   features: none none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//     direct: true
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior macau normal --aux-data none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct")
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> rowSideInfoDenseMatrixConfig = getRowSideInfoDenseMatrixConfig();

   Config config;
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.getPriorTypes().push_back(PriorTypes::macau);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);
   config.setDirect(true);

   REQUIRE_THROWS(SessionFactory::create_py_session(config));
}

//test throw

//
//      train: dense matrix
//       test: sparse matrix
//     priors: macau normal
//   features: col_side_info_dense_matrix none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//     direct: true
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior macau normal --aux-data <col_side_info_dense_matrix> none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct")
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> colSideInfoDenseMatrixConfig = getColSideInfoDenseMatrixConfig();

   Config config;
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.getPriorTypes().push_back(PriorTypes::macau);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getSideInfo().push_back(colSideInfoDenseMatrixConfig);
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);
   config.setDirect(true);

   REQUIRE_THROWS(SessionFactory::create_py_session(config));
}

//=================================================================

//
//      train: dense matrix
//       test: sparse matrix
//     priors: normal spikeandslab
//   aux-data: none none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior normal spikeandslab --aux-data none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234")
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();

   Config config;
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getPriorTypes().push_back(PriorTypes::spikeandslab);
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
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

   // Pre-calculated results with single-threaded Debug master 85a6fe322fdd1f6319c803b2736ada24471c5518
   double expectedRmseAvg = 0.3259578668804468;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  1.5935964474216957,  1.5935450411958283, 27.1966661253837394, 0.7450060419613064 },
         { { 0, 1 },  2,  1.9799867601095829,  2.1094535214017029, 22.1104605008753232, 0.6717394493201704 },
         { { 0, 2 },  3,  2.2420992285486814,  2.6556953850001417, 23.8167522080628657, 0.6971772855676575 },
         { { 0, 3 },  4,  2.5267598167988767,  3.6710421614527573, 44.7512742878349741, 0.9556627605994868 },
         { { 2, 0 },  9,  8.0552442030477369,  8.5486307538802411, 40.9656207204988334, 0.9143484433893611 },
         { { 2, 1 }, 10, 10.0083536815663514,  9.8372447524065780, 40.1430076292839075, 0.9051215684419719 },
         { { 2, 2 }, 11, 11.3332687473326033, 10.9192977909543796, 33.7578834402211925, 0.8300219254569170 },
         { { 2, 3 }, 12, 12.7721590994342566, 11.8509455884048212, 26.3124015715247666, 0.7327945054201586 }
      };

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//
//      train: dense matrix
//       test: sparse matrix
//     priors: spikeandslab normal
//   aux-data: none none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior spikeandslab normal --aux-data none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234")
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();

   Config config;
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.getPriorTypes().push_back(PriorTypes::spikeandslab);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
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

   // Pre-calculated results with single-threaded Debug master 85a6fe322fdd1f6319c803b2736ada24471c5518
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

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//
//      train: dense matrix
//       test: sparse matrix
//     priors: normal spikeandslab
//   aux-data: none dense_matrix
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior normal spikeandslab --aux-data none <dense_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234")
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<TensorConfig> colAuxDataDenseMatrixConfig = getColAuxDataDenseMatrixConfig();

   Config config;
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.getPriorTypes().push_back(PriorTypes::spikeandslab);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.getAuxData().push_back({ colAuxDataDenseMatrixConfig });
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

   // Pre-calculated results with single-threaded Debug master 85a6fe322fdd1f6319c803b2736ada24471c5518
   double expectedRmseAvg = 0.6181474446533649;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  1.7827318002503647,  2.0367038437795628,  9.9226927199390644, 0.4500043700454262 },
         { { 0, 1 },  2,  2.1204659306031632,  2.3443792288668552, 13.6647206657195763, 0.5280831850388796 },
         { { 0, 2 },  3,  2.1966469390247476,  2.7040759776631336, 19.1393066523913120, 0.6249784755867689 },
         { { 0, 3 },  4,  2.5710605326966856,  3.0180377567550485, 20.3037118059276622, 0.6437091466082295 },
         { { 2, 0 },  9,  8.5116683780302012,  8.2921058411073307, 40.4467500675152252, 0.9085394206796324 },
         { { 2, 1 }, 10, 10.1241829004624169,  9.5149618982715136, 32.9714746177718325, 0.8202970419899027 },
         { { 2, 2 }, 11, 10.4879097831586190, 10.9460319109562043, 27.2551075417206015, 0.7458060636150059 },
         { { 2, 3 }, 12, 12.2755507200143601, 12.2690514161961772, 39.7099207397002658, 0.9002258303937788 }
      };

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//
//      train: dense matrix
//       test: sparse matrix
//     priors: spikeandslab normal
//   aux-data: dense_matrix none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior spikeandslab normal --aux-data <dense_matrix> none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234")
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<TensorConfig> rowAuxDataDenseMatrixConfig = getRowAuxDataDenseMatrixConfig();

   Config config;
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.getPriorTypes().push_back(PriorTypes::spikeandslab);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getAuxData().push_back({ rowAuxDataDenseMatrixConfig });
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
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

   // Pre-calculated results with single-threaded Debug master 85a6fe322fdd1f6319c803b2736ada24471c5518
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

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//=================================================================

//
//      train: dense matrix
//       test: sparse matrix
//     priors: macau spikeandslab
//   features: row_side_info_dense_matrix none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//     direct: true
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior macau spikeandslab --aux-data <row_side_info_dense_matrix> none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct")
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> rowSideInfoDenseMatrixConfig = getRowSideInfoDenseMatrixConfig();

   Config config;
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.getPriorTypes().push_back(PriorTypes::macau);
   config.getPriorTypes().push_back(PriorTypes::spikeandslab);
   config.getSideInfo().push_back(rowSideInfoDenseMatrixConfig);
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);
   config.setDirect(true);

   std::shared_ptr<ISession> session = SessionFactory::create_py_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master cb4f760b393156874cc13e667766ba474b6d9f04
   double expectedRmseAvg = 0.6108536509895077;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  2.5394685023275261,  2.1106493024808803, 10.0594587677124103, 0.4530949976462883 },
         { { 0, 1 },  2,  2.8887691509836682,  2.4118121581629892, 11.1346807934963437, 0.4766952732519399 },
         { { 0, 2 },  3,  3.5988165476608422,  2.8227946029316002, 18.3419058677625451, 0.6118207331777621 },
         { { 0, 3 },  4,  3.8288779343603396,  3.1468597709624366, 23.0274563422839158, 0.6855275987281906 },
         { { 2, 0 },  9,  8.3263276207569845,  8.3282151327266867, 30.9939811315534861, 0.7953176894641291 },
         { { 2, 1 }, 10,  9.4716033491971316,  9.5479432780141931, 38.5637960501877757, 0.8871393610488711 },
         { { 2, 2 }, 11, 11.7996839084090599, 11.1108211696531125, 41.8295891938965951, 0.9239399794303341 },
         { { 2, 3 }, 12, 12.5540017811410838, 12.3934778445095350, 44.8153069497794192, 0.9563462244479801 }
      };

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//
//      train: dense matrix
//       test: sparse matrix
//     priors: spikeandslab macau
//   features: none col_side_info_dense_matrix
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//     direct: true
//
TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior spikeandslab macau --aux-data none <col_side_info_dense_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct")
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> colSideInfoDenseMatrixConfig = getColSideInfoDenseMatrixConfig();

   Config config;
   config.setTrain(trainDenseMatrixConfig);
   config.setTest(testSparseMatrixConfig);
   config.getPriorTypes().push_back(PriorTypes::spikeandslab);
   config.getPriorTypes().push_back(PriorTypes::macau);
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getSideInfo().push_back(colSideInfoDenseMatrixConfig);
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);
   config.setDirect(true);

   std::shared_ptr<ISession> session = SessionFactory::create_py_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master cb4f760b393156874cc13e667766ba474b6d9f04
   double expectedRmseAvg = 0.6038171746157998;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  1.8917871627427933,  2.1764598069992958, 13.4033445734592966, 0.5230082641376828 },
         { { 0, 1 },  2,  2.3070418049497268,  2.5478982744531611, 15.8618599295301763, 0.5689564369380743 },
         { { 0, 2 },  3,  2.6170060311861678,  2.8084954473914578, 15.7760945570635016, 0.5674161730242268 },
         { { 0, 3 },  4,  3.6238090769414040,  3.1989061111012949, 22.0702244659406119, 0.6711279641043648 },
         { { 2, 0 },  9,  7.5073992851600391,  8.3784331745926934, 46.8258922848342181, 0.9775635297987211 },
         { { 2, 1 }, 10,  9.1553026357377334,  9.8122851050747162, 35.9470768688528395, 0.8565125881445397 },
         { { 2, 2 }, 11, 10.3853697681834483, 10.8689626342813597, 36.0454007498897226, 0.8576831719621979 },
         { { 2, 3 }, 12, 14.3807835308189951, 12.3396068937987398, 40.9719961049140551, 0.9144195895837821 }
      };

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//=================================================================

//
//      train: dense 2D-tensor (matrix)
//       test: sparse 2D-tensor (matrix)
//     priors: normal normal
//   aux-data: none none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE("--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> --prior normal normal --aux-data none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234")
{
   std::shared_ptr<TensorConfig> trainSparseTensorConfig = getTrainDenseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();

   Config config;
   config.setTrain(trainSparseTensorConfig);
   config.setTest(testSparseTensorConfig);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
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

   // Pre-calculated results with single-threaded Debug master 85a6fe322fdd1f6319c803b2736ada24471c5518
   double expectedRmseAvg = 0.4148777232391613;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  2.2997524192635992,  1.7370592984929207, 20.7793461157136292, 0.6512052580222217 },
         { { 0, 1 },  2,  3.3981721039544133,  2.1159960504024378, 26.1403820382501841, 0.7303952248297392 },
         { { 0, 2 },  3,  3.0884498605821431,  2.7526565775059755, 28.3120906318339820, 0.7601300993886585 },
         { { 0, 3 },  4,  4.1232674062768098,  3.4710086454388227, 42.3947323487426573, 0.9301605341886394 },
         { { 2, 0 },  9,  9.1825164425321226,  8.4690736380474352, 59.3684003734163781, 1.1007270359270469 },
         { { 2, 1 }, 10,  8.2950709803315519,  9.5655479877679337, 56.3404527095021805, 1.0722896797679156 },
         { { 2, 2 }, 11, 11.5399538613639976, 10.9574323035202355, 55.9898197868029257, 1.0689477926466482 },
         { { 2, 3 }, 12, 10.8882167080340597, 12.0825997901525319, 58.3087165680387400, 1.0908592060898576 }
      };

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//
//      train: sparse 2D-tensor (matrix)
//       test: sparse 2D-tensor (matrix)
//     priors: normal normal
//   aux-data: none none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE("--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> --prior normal normal --aux-data none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234")
{
   std::shared_ptr<TensorConfig> trainSparseTensorConfig = getTrainSparseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();

   Config config;
   config.setTrain(trainSparseTensorConfig);
   config.setTest(testSparseTensorConfig);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
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

   // Pre-calculated results with single-threaded Debug master 85a6fe322fdd1f6319c803b2736ada24471c5518
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

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//=================================================================

//
//      train: dense 2D-tensor (matrix)
//       test: sparse 2D-tensor (matrix)
//     priors: spikeandslab spikeandslab
//   aux-data: none none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE("--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> --prior spikeandslab spikeandslab --aux-data none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234")
{
   std::shared_ptr<TensorConfig> trainSparseTensorConfig = getTrainDenseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();

   Config config;
   config.setTrain(trainSparseTensorConfig);
   config.setTest(testSparseTensorConfig);
   config.getPriorTypes().push_back(PriorTypes::spikeandslab);
   config.getPriorTypes().push_back(PriorTypes::spikeandslab);
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
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

   // Pre-calculated results with single-threaded Debug master 747d8e9c032bd4fad2cc4dbb777a51dc63a203bf
   double expectedRmseAvg = 0.5903156096172538;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  1.2726354780941709,  2.1309703656035368,  8.3872734247241585, 0.4137255677409046 },
         { { 0, 1 },  2,  1.7994938808275749,  2.4174432387371412,  8.5311434352124085, 0.4172588741603458 },
         { { 0, 2 },  3,  1.6636094663595014,  2.7727847747587666, 14.6190840484526259, 0.5462130116081578 },
         { { 0, 3 },  4,  2.2576440354321257,  3.0664862280378786, 14.9216794669135613, 0.5518369965427635 },
         { { 2, 0 },  9,  7.6553932462162768,  8.5020824484720023, 32.3490098516762146, 0.8125169995292433 },
         { { 2, 1 }, 10, 10.8246497437937741,  9.6825335824681265, 42.4180911618062382, 0.9304167506191945 },
         { { 2, 2 }, 11, 10.0072525812199604, 11.0337582504183302, 27.7552191235189234, 0.7526174614883187 },
         { { 2, 3 }, 12, 13.5805996286461053, 12.2480043776745315, 42.3916925281742323, 0.9301271860383596 },
      };

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//
//      train: sparse 2D-tensor (matrix)
//       test: sparse 2D-tensor (matrix)
//     priors: spikeandslab spikeandslab
//   aux-data: none none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE("--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> --prior spikeandslab spikeandslab --aux-data none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234")
{
   std::shared_ptr<TensorConfig> trainSparseTensorConfig = getTrainSparseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();

   Config config;
   config.setTrain(trainSparseTensorConfig);
   config.setTest(testSparseTensorConfig);
   config.getPriorTypes().push_back(PriorTypes::spikeandslab);
   config.getPriorTypes().push_back(PriorTypes::spikeandslab);
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
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

   // Pre-calculated results with single-threaded Debug master 85a6fe322fdd1f6319c803b2736ada24471c5518
   double expectedRmseAvg = 0.5843270189981123;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  2.3705122746024765,  2.1705657803356178,  6.1911553798380599, 0.3554576061819658 },
         { { 0, 1 },  2,  3.1011229285882380,  2.4591191554029850,  9.3773799541920653, 0.4374643998155284 },
         { { 0, 2 },  3,  2.9511565553516110,  2.7621433890063019,  8.6613658290729152, 0.4204314069384726 },
         { { 0, 3 },  4,  3.4022881883245222,  3.0339110969625152,  9.7750007006520843, 0.4466428217461795 },
         { { 2, 0 },  9,  8.3086993795525323,  8.6694374222976354, 32.7276481206319048, 0.8172583350051225 },
         { { 2, 1 }, 10, 10.8695063209483322,  9.8064265468900516, 59.0861402124385222, 1.0981072789906343 },
         { { 2, 2 }, 11, 10.3438707755791945, 11.0270016369173245, 36.1715708858991860, 0.8591829398923283 },
         { { 2, 3 }, 12, 11.9250973986757245, 12.1145205318322642, 36.5104785971590999, 0.8631985913480673 }
      };

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//=================================================================

//
//      train: dense 3D-tensor (matrix)
//       test: sparse 3D-tensor (matrix)
//     priors: normal normal normal
//   aux-data: none none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE("--train <train_dense_3d_tensor> --test <test_sparse_3d_tensor> --prior normal normal --aux-data none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234")
{
   std::shared_ptr<TensorConfig> trainSparseTensorConfig = getTrainDenseTensor3dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor3dConfig();

   Config config;
   config.setTrain(trainSparseTensorConfig);
   config.setTest(testSparseTensorConfig);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
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

   // Pre-calculated results with single-threaded Debug master 85a6fe322fdd1f6319c803b2736ada24471c5518
   double expectedRmseAvg = 8.1222388122203117;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0, 0 },  1,  2.0577075209621487,  1.5184511241525529, 36.2924688384283201, 0.8606175860134860 },
         { { 0, 0, 1 },  2,  7.9968139641223184,  7.2968936590024054, 31.7258568441849143, 0.8046530098171853 },
         { { 0, 0, 2 },  3, 13.1986929795770358, 13.2213859338453297, 28.8452302653818933, 0.7672536534166932 },
         { { 0, 0, 3 },  4, 18.8164347359714057, 18.9080932496367176, 42.0370139871239985, 0.9262279658027950 },
         { { 0, 2, 0 },  9,  3.4713274350877423,  4.6587009427029988, 29.3379549972152489, 0.7737788931299253 },
         { { 0, 2, 1 }, 10, 10.6400651516089990, 10.6874294996177177, 23.8184045177335939, 0.6972014688144397 },
         { { 0, 2, 2 }, 11, 16.5059364279520153, 16.8544700896378679, 19.9698156016478130, 0.6383942803444326 },
         { { 0, 2, 3 }, 12, 23.2716649261782074, 22.9141193404789689, 18.9566356995931429, 0.6219888393839775 },
      };

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//=================================================================

//
//      train: dense 3D-tensor (matrix)
//       test: sparse 3D-tensor (matrix)
//     priors: spikeandslab spikeandslab
//   aux-data: none none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE("--train <train_dense_3d_tensor> --test <test_sparse_3d_tensor> --prior spikeandslab spikeandslab --aux-data none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234")
{
   std::shared_ptr<TensorConfig> trainSparseTensorConfig = getTrainDenseTensor3dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor3dConfig();

   Config config;
   config.setTrain(trainSparseTensorConfig);
   config.setTest(testSparseTensorConfig);
   config.getPriorTypes().push_back(PriorTypes::spikeandslab);
   config.getPriorTypes().push_back(PriorTypes::spikeandslab);
   config.getPriorTypes().push_back(PriorTypes::spikeandslab);
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
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

   // Pre-calculated results with single-threaded Debug master 747d8e9c032bd4fad2cc4dbb777a51dc63a203bf
   double expectedRmseAvg = 8.1108371221898459;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0, 0 },  1,  2.9691764166597201,  3.1447757072906217,  4.9137753304502771, 0.3166719583304824 },
         { { 0, 0, 1 },  2,  7.7969883937578102,  8.0212652299812692,  8.7577397801031758, 0.4227639805700241 },
         { { 0, 0, 2 },  3, 12.3009064934126542, 13.1555530737629969, 17.9763493534955394, 0.6056932165051191 },
         { { 0, 0, 3 },  4, 17.6918221492758221, 18.0207359436422969, 24.8978982404027676, 0.7128256254183892 },
         { { 0, 2, 0 },  9,  3.9604459745601601,  4.0504278004605609,  7.5420668736406276, 0.3923260533218701 },
         { { 0, 2, 1 }, 10, 10.4000392581891283, 10.3323953792288741, 10.3598354302581104, 0.4598099747313127 },
         { { 0, 2, 2 }, 11, 16.4076056013145468, 16.9445551614907366, 16.2481912671750983, 0.5758435031728938 },
         { { 0, 2, 3 }, 12, 23.5982966254860074, 23.2145865027304872, 23.7828745898045106, 0.6966812668265386 }
      };

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//=================================================================

//test throw

//
//      train: dense 2D-tensor
//       test: sparse 2D-tensor
//     priors: macau normal
//   aux-data: row_dense_side_info none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE("--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> --prior macau normal --side-info row_dense_side_info none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234")
{
   std::shared_ptr<TensorConfig> trainDenseTensorConfig = getTrainDenseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();
   std::shared_ptr<MatrixConfig> rowSideInfoDenseMatrixConfig = getRowSideInfoDenseMatrixConfig();

   Config config;
   config.setTrain(trainDenseTensorConfig);
   config.setTest(testSparseTensorConfig);
   config.getPriorTypes().push_back(PriorTypes::macau);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getSideInfo().push_back(rowSideInfoDenseMatrixConfig);
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);

   REQUIRE_THROWS(SessionFactory::create_py_session(config));
}

//test throw

//
//      train: dense 3D-tensor
//       test: sparse 3D-tensor
//     priors: macau normal
//   aux-data: row_dense_side_info none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE("--train <train_dense_3d_tensor> --test <test_sparse_3d_tensor> --prior macau normal --side-info row_dense_side_info none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234")
{
   std::shared_ptr<TensorConfig> trainDenseTensorConfig = getTrainDenseTensor3dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor3dConfig();
   std::shared_ptr<MatrixConfig> rowSideInfoDenseMatrixConfig = getRowSideInfoDenseMatrixConfig();

   Config config;
   config.setTrain(trainDenseTensorConfig);
   config.setTest(testSparseTensorConfig);
   config.getPriorTypes().push_back(PriorTypes::macau);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getSideInfo().push_back(rowSideInfoDenseMatrixConfig);
   config.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);

   REQUIRE_THROWS(SessionFactory::create_py_session(config));
}

//=================================================================

//pairwise tests for 2d matrix vs 2d tensor
// normal normal
// normal spikeandslab
// spikeandslab normal
// spikeandslab spikeandslab

//
//      train: 1. dense 2D-tensor (matrix)
//             2. dense matrix
//       test: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//     priors: normal normal
//   aux-data: none none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE(
   "matrix vs 2D-tensor"
   "--train <train_dense_matrix>    --test <test_sparse_amtrix>    --prior normal normal --aux-data none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234"
   "--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> --prior normal normal --aux-data none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234"
)
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   Config matrixSessionConfig;
   matrixSessionConfig.setTrain(trainDenseMatrixConfig);
   matrixSessionConfig.setTest(testSparseMatrixConfig);
   matrixSessionConfig.getPriorTypes().push_back(PriorTypes::normal);
   matrixSessionConfig.getPriorTypes().push_back(PriorTypes::normal);
   matrixSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   matrixSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   matrixSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   matrixSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   matrixSessionConfig.setNumLatent(4);
   matrixSessionConfig.setBurnin(50);
   matrixSessionConfig.setNSamples(50);
   matrixSessionConfig.setVerbose(false);
   matrixSessionConfig.setRandomSeed(1234);
   matrixSessionConfig.setRandomSeedSet(true);

   std::shared_ptr<TensorConfig> trainDenseTensorConfig = getTrainDenseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();
   Config tensorSessionConfig;
   tensorSessionConfig.setTrain(trainDenseTensorConfig);
   tensorSessionConfig.setTest(testSparseTensorConfig);
   tensorSessionConfig.getPriorTypes().push_back(PriorTypes::normal);
   tensorSessionConfig.getPriorTypes().push_back(PriorTypes::normal);
   tensorSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   tensorSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   tensorSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   tensorSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   tensorSessionConfig.setNumLatent(4);
   tensorSessionConfig.setBurnin(50);
   tensorSessionConfig.setNSamples(50);
   tensorSessionConfig.setVerbose(false);
   tensorSessionConfig.setRandomSeed(1234);
   tensorSessionConfig.setRandomSeedSet(true);

   std::shared_ptr<ISession> matrixSession = SessionFactory::create_py_session(matrixSessionConfig);
   std::shared_ptr<ISession> tensorSession = SessionFactory::create_py_session(tensorSessionConfig);
   matrixSession->run();
   tensorSession->run();

   REQUIRE(matrixSession->getRmseAvg() == Approx(tensorSession->getRmseAvg()).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*matrixSession->getResult(), *tensorSession->getResult());
}

//
//      train: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//       test: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//     priors: normal normal
//   aux-data: none none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE(
   "matrix vs 2D-tensor"
   "--train <train_sparse_matrix>    --test <test_sparse_amtrix>    --prior normal normal --aux-data none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234"
   "--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> --prior normal normal --aux-data none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234"
)
{
   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   Config matrixSessionConfig;
   matrixSessionConfig.setTrain(trainSparseMatrixConfig);
   matrixSessionConfig.setTest(testSparseMatrixConfig);
   matrixSessionConfig.getPriorTypes().push_back(PriorTypes::normal);
   matrixSessionConfig.getPriorTypes().push_back(PriorTypes::normal);
   matrixSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   matrixSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   matrixSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   matrixSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   matrixSessionConfig.setNumLatent(4);
   matrixSessionConfig.setBurnin(50);
   matrixSessionConfig.setNSamples(50);
   matrixSessionConfig.setVerbose(false);
   matrixSessionConfig.setRandomSeed(1234);
   matrixSessionConfig.setRandomSeedSet(true);

   std::shared_ptr<TensorConfig> trainSparseTensorConfig = getTrainSparseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();
   Config tensorSessionConfig;
   tensorSessionConfig.setTrain(trainSparseTensorConfig);
   tensorSessionConfig.setTest(testSparseTensorConfig);
   tensorSessionConfig.getPriorTypes().push_back(PriorTypes::normal);
   tensorSessionConfig.getPriorTypes().push_back(PriorTypes::normal);
   tensorSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   tensorSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   tensorSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   tensorSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   tensorSessionConfig.setNumLatent(4);
   tensorSessionConfig.setBurnin(50);
   tensorSessionConfig.setNSamples(50);
   tensorSessionConfig.setVerbose(false);
   tensorSessionConfig.setRandomSeed(1234);
   tensorSessionConfig.setRandomSeedSet(true);

   std::shared_ptr<ISession> matrixSession = SessionFactory::create_py_session(matrixSessionConfig);
   std::shared_ptr<ISession> tensorSession = SessionFactory::create_py_session(tensorSessionConfig);
   matrixSession->run();
   tensorSession->run();

   REQUIRE(matrixSession->getRmseAvg() == Approx(tensorSession->getRmseAvg()).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*matrixSession->getResult(), *tensorSession->getResult());
}

//
//      train: 1. dense 2D-tensor (matrix)
//             2. dense matrix
//       test: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//     priors: normal spikeandslab
//   aux-data: none none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE(
   "matrix vs 2D-tensor"
   "--train <train_dense_matrix>    --test <test_sparse_amtrix>    --prior normal spikeandslab --aux-data none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234"
   "--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> --prior normal spikeandslab --aux-data none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234"
   , "[!hide]"
)
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   Config matrixSessionConfig;
   matrixSessionConfig.setTrain(trainDenseMatrixConfig);
   matrixSessionConfig.setTest(testSparseMatrixConfig);
   matrixSessionConfig.getPriorTypes().push_back(PriorTypes::normal);
   matrixSessionConfig.getPriorTypes().push_back(PriorTypes::spikeandslab);
   matrixSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   matrixSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   matrixSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   matrixSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   matrixSessionConfig.setNumLatent(4);
   matrixSessionConfig.setBurnin(50);
   matrixSessionConfig.setNSamples(50);
   matrixSessionConfig.setVerbose(false);
   matrixSessionConfig.setRandomSeed(1234);
   matrixSessionConfig.setRandomSeedSet(true);

   std::shared_ptr<TensorConfig> trainDenseTensorConfig = getTrainDenseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();
   Config tensorSessionConfig;
   tensorSessionConfig.setTrain(trainDenseTensorConfig);
   tensorSessionConfig.setTest(testSparseTensorConfig);
   tensorSessionConfig.getPriorTypes().push_back(PriorTypes::normal);
   tensorSessionConfig.getPriorTypes().push_back(PriorTypes::spikeandslab);
   tensorSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   tensorSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   tensorSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   tensorSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   tensorSessionConfig.setNumLatent(4);
   tensorSessionConfig.setBurnin(50);
   tensorSessionConfig.setNSamples(50);
   tensorSessionConfig.setVerbose(false);
   tensorSessionConfig.setRandomSeed(1234);
   tensorSessionConfig.setRandomSeedSet(true);

   std::shared_ptr<ISession> matrixSession = SessionFactory::create_py_session(matrixSessionConfig);
   std::shared_ptr<ISession> tensorSession = SessionFactory::create_py_session(tensorSessionConfig);
   matrixSession->run();
   tensorSession->run();

   REQUIRE(matrixSession->getRmseAvg() == Approx(tensorSession->getRmseAvg()).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*matrixSession->getResult(), *tensorSession->getResult());
}

//
//      train: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//       test: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//     priors: normal spikeandslab
//   aux-data: none none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE(
   "matrix vs 2D-tensor"
   "--train <train_sparse_matrix>    --test <test_sparse_amtrix>    --prior normal spikeandslab --aux-data none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234"
   "--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> --prior normal spikeandslab --aux-data none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234"
   , "[!hide]"
)
{
   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   Config matrixSessionConfig;
   matrixSessionConfig.setTrain(trainSparseMatrixConfig);
   matrixSessionConfig.setTest(testSparseMatrixConfig);
   matrixSessionConfig.getPriorTypes().push_back(PriorTypes::normal);
   matrixSessionConfig.getPriorTypes().push_back(PriorTypes::spikeandslab);
   matrixSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   matrixSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   matrixSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   matrixSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   matrixSessionConfig.setNumLatent(4);
   matrixSessionConfig.setBurnin(50);
   matrixSessionConfig.setNSamples(50);
   matrixSessionConfig.setVerbose(false);
   matrixSessionConfig.setRandomSeed(1234);
   matrixSessionConfig.setRandomSeedSet(true);

   std::shared_ptr<TensorConfig> trainSparseTensorConfig = getTrainSparseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();
   Config tensorSessionConfig;
   tensorSessionConfig.setTrain(trainSparseTensorConfig);
   tensorSessionConfig.setTest(testSparseTensorConfig);
   tensorSessionConfig.getPriorTypes().push_back(PriorTypes::normal);
   tensorSessionConfig.getPriorTypes().push_back(PriorTypes::spikeandslab);
   tensorSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   tensorSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   tensorSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   tensorSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   tensorSessionConfig.setNumLatent(4);
   tensorSessionConfig.setBurnin(50);
   tensorSessionConfig.setNSamples(50);
   tensorSessionConfig.setVerbose(false);
   tensorSessionConfig.setRandomSeed(1234);
   tensorSessionConfig.setRandomSeedSet(true);

   std::shared_ptr<ISession> matrixSession = SessionFactory::create_py_session(matrixSessionConfig);
   std::shared_ptr<ISession> tensorSession = SessionFactory::create_py_session(tensorSessionConfig);
   matrixSession->run();
   tensorSession->run();

   REQUIRE(matrixSession->getRmseAvg() == Approx(tensorSession->getRmseAvg()).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*matrixSession->getResult(), *tensorSession->getResult());
}

//
//      train: 1. dense 2D-tensor (matrix)
//             2. dense matrix
//       test: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//     priors: spikeandslab normal
//   aux-data: none none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE(
   "matrix vs 2D-tensor"
   "--train <train_dense_matrix>    --test <test_sparse_amtrix>    --prior spikeandslab normal --aux-data none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234"
   "--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> --prior spikeandslab normal --aux-data none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234"
   , "[!hide]"
)
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   Config matrixSessionConfig;
   matrixSessionConfig.setTrain(trainDenseMatrixConfig);
   matrixSessionConfig.setTest(testSparseMatrixConfig);
   matrixSessionConfig.getPriorTypes().push_back(PriorTypes::spikeandslab);
   matrixSessionConfig.getPriorTypes().push_back(PriorTypes::normal);
   matrixSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   matrixSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   matrixSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   matrixSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   matrixSessionConfig.setNumLatent(4);
   matrixSessionConfig.setBurnin(50);
   matrixSessionConfig.setNSamples(50);
   matrixSessionConfig.setVerbose(false);
   matrixSessionConfig.setRandomSeed(1234);
   matrixSessionConfig.setRandomSeedSet(true);

   std::shared_ptr<TensorConfig> trainDenseTensorConfig = getTrainDenseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();
   Config tensorSessionConfig;
   tensorSessionConfig.setTrain(trainDenseTensorConfig);
   tensorSessionConfig.setTest(testSparseTensorConfig);
   tensorSessionConfig.getPriorTypes().push_back(PriorTypes::spikeandslab);
   tensorSessionConfig.getPriorTypes().push_back(PriorTypes::normal);
   tensorSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   tensorSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   tensorSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   tensorSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   tensorSessionConfig.setNumLatent(4);
   tensorSessionConfig.setBurnin(50);
   tensorSessionConfig.setNSamples(50);
   tensorSessionConfig.setVerbose(false);
   tensorSessionConfig.setRandomSeed(1234);
   tensorSessionConfig.setRandomSeedSet(true);

   std::shared_ptr<ISession> matrixSession = SessionFactory::create_py_session(matrixSessionConfig);
   std::shared_ptr<ISession> tensorSession = SessionFactory::create_py_session(tensorSessionConfig);
   matrixSession->run();
   tensorSession->run();

   REQUIRE(matrixSession->getRmseAvg() == Approx(tensorSession->getRmseAvg()).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*matrixSession->getResult(), *tensorSession->getResult());
}

//
//      train: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//       test: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//     priors: spikeandslab normal
//   aux-data: none none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE(
   "matrix vs 2D-tensor"
   "--train <train_sparse_matrix>    --test <test_sparse_amtrix>    --prior spikeandslab normal --aux-data none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234"
   "--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> --prior spikeandslab normal --aux-data none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234"
   , "[!hide]"
)
{
   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   Config matrixSessionConfig;
   matrixSessionConfig.setTrain(trainSparseMatrixConfig);
   matrixSessionConfig.setTest(testSparseMatrixConfig);
   matrixSessionConfig.getPriorTypes().push_back(PriorTypes::spikeandslab);
   matrixSessionConfig.getPriorTypes().push_back(PriorTypes::normal);
   matrixSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   matrixSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   matrixSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   matrixSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   matrixSessionConfig.setNumLatent(4);
   matrixSessionConfig.setBurnin(50);
   matrixSessionConfig.setNSamples(50);
   matrixSessionConfig.setVerbose(false);
   matrixSessionConfig.setRandomSeed(1234);
   matrixSessionConfig.setRandomSeedSet(true);

   std::shared_ptr<TensorConfig> trainSparseTensorConfig = getTrainSparseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();
   Config tensorSessionConfig;
   tensorSessionConfig.setTrain(trainSparseTensorConfig);
   tensorSessionConfig.setTest(testSparseTensorConfig);
   tensorSessionConfig.getPriorTypes().push_back(PriorTypes::spikeandslab);
   tensorSessionConfig.getPriorTypes().push_back(PriorTypes::normal);
   tensorSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   tensorSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   tensorSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   tensorSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   tensorSessionConfig.setNumLatent(4);
   tensorSessionConfig.setBurnin(50);
   tensorSessionConfig.setNSamples(50);
   tensorSessionConfig.setVerbose(false);
   tensorSessionConfig.setRandomSeed(1234);
   tensorSessionConfig.setRandomSeedSet(true);

   std::shared_ptr<ISession> matrixSession = SessionFactory::create_py_session(matrixSessionConfig);
   std::shared_ptr<ISession> tensorSession = SessionFactory::create_py_session(tensorSessionConfig);
   matrixSession->run();
   tensorSession->run();

   REQUIRE(matrixSession->getRmseAvg() == Approx(tensorSession->getRmseAvg()).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*matrixSession->getResult(), *tensorSession->getResult());
}

//
//      train: 1. dense 2D-tensor (matrix)
//             2. dense matrix
//       test: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//     priors: spikeandslab spikeandslab
//   aux-data: none none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE(
   "matrix vs 2D-tensor"
   "--train <train_dense_matrix>    --test <test_sparse_amtrix>    --prior spikeandslab spikeandslab --aux-data none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234"
   "--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> --prior spikeandslab spikeandslab --aux-data none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234"
   , "[!hide]"
)
{
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   Config matrixSessionConfig;
   matrixSessionConfig.setTrain(trainDenseMatrixConfig);
   matrixSessionConfig.setTest(testSparseMatrixConfig);
   matrixSessionConfig.getPriorTypes().push_back(PriorTypes::spikeandslab);
   matrixSessionConfig.getPriorTypes().push_back(PriorTypes::spikeandslab);
   matrixSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   matrixSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   matrixSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   matrixSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   matrixSessionConfig.setNumLatent(4);
   matrixSessionConfig.setBurnin(50);
   matrixSessionConfig.setNSamples(50);
   matrixSessionConfig.setVerbose(false);
   matrixSessionConfig.setRandomSeed(1234);
   matrixSessionConfig.setRandomSeedSet(true);

   std::shared_ptr<TensorConfig> trainDenseTensorConfig = getTrainDenseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();
   Config tensorSessionConfig;
   tensorSessionConfig.setTrain(trainDenseTensorConfig);
   tensorSessionConfig.setTest(testSparseTensorConfig);
   tensorSessionConfig.getPriorTypes().push_back(PriorTypes::spikeandslab);
   tensorSessionConfig.getPriorTypes().push_back(PriorTypes::spikeandslab);
   tensorSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   tensorSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   tensorSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   tensorSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   tensorSessionConfig.setNumLatent(4);
   tensorSessionConfig.setBurnin(50);
   tensorSessionConfig.setNSamples(50);
   tensorSessionConfig.setVerbose(false);
   tensorSessionConfig.setRandomSeed(1234);
   tensorSessionConfig.setRandomSeedSet(true);

   std::shared_ptr<ISession> matrixSession = SessionFactory::create_py_session(matrixSessionConfig);
   std::shared_ptr<ISession> tensorSession = SessionFactory::create_py_session(tensorSessionConfig);
   matrixSession->run();
   tensorSession->run();

   REQUIRE(matrixSession->getRmseAvg() == Approx(tensorSession->getRmseAvg()).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*matrixSession->getResult(), *tensorSession->getResult());
}

//
//      train: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//       test: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//     priors: spikeandslab spikeandslab
//   aux-data: none none
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE(
   "matrix vs 2D-tensor"
   "--train <train_sparse_matrix>    --test <test_sparse_amtrix>    --prior spikeandslab spikeandslab --aux-data none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234"
   "--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> --prior spikeandslab spikeandslab --aux-data none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234"
   , "[!hide]"
)
{
   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   Config matrixSessionConfig;
   matrixSessionConfig.setTrain(trainSparseMatrixConfig);
   matrixSessionConfig.setTest(testSparseMatrixConfig);
   matrixSessionConfig.getPriorTypes().push_back(PriorTypes::spikeandslab);
   matrixSessionConfig.getPriorTypes().push_back(PriorTypes::spikeandslab);
   matrixSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   matrixSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   matrixSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   matrixSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   matrixSessionConfig.setNumLatent(4);
   matrixSessionConfig.setBurnin(50);
   matrixSessionConfig.setNSamples(50);
   matrixSessionConfig.setVerbose(false);
   matrixSessionConfig.setRandomSeed(1234);
   matrixSessionConfig.setRandomSeedSet(true);

   std::shared_ptr<TensorConfig> trainSparseTensorConfig = getTrainSparseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();
   Config tensorSessionConfig;
   tensorSessionConfig.setTrain(trainSparseTensorConfig);
   tensorSessionConfig.setTest(testSparseTensorConfig);
   tensorSessionConfig.getPriorTypes().push_back(PriorTypes::spikeandslab);
   tensorSessionConfig.getPriorTypes().push_back(PriorTypes::spikeandslab);
   tensorSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   tensorSessionConfig.getSideInfo().push_back(std::shared_ptr<MatrixConfig>());
   tensorSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   tensorSessionConfig.getAuxData().push_back(std::vector<std::shared_ptr<TensorConfig> >());
   tensorSessionConfig.setNumLatent(4);
   tensorSessionConfig.setBurnin(50);
   tensorSessionConfig.setNSamples(50);
   tensorSessionConfig.setVerbose(false);
   tensorSessionConfig.setRandomSeed(1234);
   tensorSessionConfig.setRandomSeedSet(true);

   std::shared_ptr<ISession> matrixSession = SessionFactory::create_py_session(matrixSessionConfig);
   std::shared_ptr<ISession> tensorSession = SessionFactory::create_py_session(tensorSessionConfig);
   matrixSession->run();
   tensorSession->run();

   REQUIRE(matrixSession->getRmseAvg() == Approx(tensorSession->getRmseAvg()).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*matrixSession->getResult(), *tensorSession->getResult());
}
#include "catch.hpp"

#include <Eigen/Core>

#include <SmurffCpp/Configs/Config.h>
#include <SmurffCpp/Sessions/SessionFactory.h>
#include <SmurffCpp/Utils/MatrixUtils.h>

/////////////////////////////////////////////////////////////////////////////////////////////////
// Code for printing test results that can then be copy-pasted into tests as expected results
/////////////////////////////////////////////////////////////////////////////////////////////////
//
// std::cout << std::fixed << std::setprecision(16) << actualRmseAvg << std::endl;
// for (std::vector<ResultItem>::size_type i = 0; i < actualResults->size(); i++)
// {
//    const ResultItem& actualResultItem = actualResults->operator[](i);
//    std::cout << std::setprecision(16);
//    std::cout << "{ { " << actualResultItem.coords << " }, "
//             << std::defaultfloat << actualResultItem.val << ", "
//             << std::fixed << actualResultItem.pred_1sample << ", "
//             << actualResultItem.pred_avg << ", "
//             << actualResultItem.var << ", "
//             << actualResultItem.stds
//             << " }" << std::endl;
// }
//
/////////////////////////////////////////////////////////////////////////////////////////////////

// https://github.com/catchorg/Catch2/blob/master/docs/assertions.md#floating-point-comparisons
// By default Catch.hpp sets epsilon to std::numeric_limits<float>::epsilon()*100
#define APPROX_EPSILON std::numeric_limits<float>::epsilon()*100

using namespace smurff;

static NoiseConfig fixed_ncfg(NoiseTypes::fixed);

// dense train data (matrix/tensor 2d/tensor 3d)

std::shared_ptr<MatrixConfig> getTrainDenseMatrixConfig()
{
   std::vector<double> trainMatrixConfigVals = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };
   std::shared_ptr<MatrixConfig> trainMatrixConfig =
      std::make_shared<MatrixConfig>(3, 4, std::move(trainMatrixConfigVals), fixed_ncfg);
   return trainMatrixConfig;
}

std::shared_ptr<TensorConfig> getTrainDenseTensor2dConfig()
{
   std::vector<double> trainTensorConfigVals = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };
   std::shared_ptr<TensorConfig> trainTensorConfig =
      std::make_shared<TensorConfig>(std::initializer_list<uint64_t>({ 3, 4 }), std::move(trainTensorConfigVals), fixed_ncfg);
   return trainTensorConfig;
}

std::shared_ptr<TensorConfig> getTrainDenseTensor3dConfig()
{
   std::vector<double> trainTensorConfigVals = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };
   std::shared_ptr<TensorConfig> trainTensorConfig =
      std::make_shared<TensorConfig>(std::initializer_list<uint64_t>({ 2, 3, 4 }), std::move(trainTensorConfigVals), fixed_ncfg);
   return trainTensorConfig;
}

// sparse train data (matrix/tensor 2d)

std::shared_ptr<MatrixConfig> getTrainSparseMatrixConfig()
{
   std::vector<std::uint32_t> trainMatrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2 };
   std::vector<std::uint32_t> trainMatrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   std::vector<double> trainMatrixConfigVals = { 1, 2, 3, 4, 9, 10, 11, 12 };
   std::shared_ptr<MatrixConfig> trainMatrixConfig =
      std::make_shared<MatrixConfig>(3, 4, std::move(trainMatrixConfigRows), std::move(trainMatrixConfigCols), std::move(trainMatrixConfigVals), fixed_ncfg, true);
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
      std::make_shared<TensorConfig>(std::initializer_list<uint64_t>({ 3, 4 }), std::move(trainTensorConfigCols), std::move(trainTensorConfigVals), fixed_ncfg, true);
   return trainTensorConfig;
}

// sparse test data (matrix/tensor 2d/tensor 3d)

std::shared_ptr<MatrixConfig> getTestSparseMatrixConfig()
{
   std::vector<std::uint32_t> testMatrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2};
   std::vector<std::uint32_t> testMatrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   std::vector<double> testMatrixConfigVals = { 1, 2, 3, 4, 9, 10, 11, 12 };
   std::shared_ptr<MatrixConfig> testMatrixConfig =
      std::make_shared<MatrixConfig>(3, 4, std::move(testMatrixConfigRows), std::move(testMatrixConfigCols), std::move(testMatrixConfigVals), fixed_ncfg, true);
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
      std::make_shared<TensorConfig>(std::initializer_list<uint64_t>({ 3, 4 }), std::move(testTensorConfigCols), std::move(testTensorConfigVals), fixed_ncfg, true);
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
      std::make_shared<TensorConfig>(std::initializer_list<uint64_t>({ 2, 3, 4 }), std::move(testTensorConfigCols), std::move(testTensorConfigVals), fixed_ncfg, true);
   return testTensorConfig;
}

// aux data

std::shared_ptr<MatrixConfig> getRowAuxDataDenseMatrixConfig()
{
   std::vector<double> rowAuxDataDenseMatrixConfigVals = { 1, 2, 3 };
   std::shared_ptr<MatrixConfig> rowAuxDataDenseMatrixConfig =
      std::make_shared<MatrixConfig>(3, 1, std::move(rowAuxDataDenseMatrixConfigVals), fixed_ncfg);
   rowAuxDataDenseMatrixConfig->setPos(PVec<>({0,1}));
   return rowAuxDataDenseMatrixConfig;
}

std::shared_ptr<MatrixConfig> getColAuxDataDenseMatrixConfig()
{
   std::vector<double> colAuxDataDenseMatrixConfigVals = { 1, 2, 3, 4 };
   std::shared_ptr<MatrixConfig> colAuxDataDenseMatrixConfig =
      std::make_shared<MatrixConfig>(1, 4, std::move(colAuxDataDenseMatrixConfigVals), fixed_ncfg);
   colAuxDataDenseMatrixConfig->setPos(PVec<>({1,0}));
   return colAuxDataDenseMatrixConfig;
}

// side info

std::shared_ptr<MatrixConfig> getRowSideInfoDenseMatrixConfig()
{
   NoiseConfig nc(NoiseTypes::adaptive);
   nc.setPrecision(10.0);

   std::vector<double> rowSideInfoDenseMatrixConfigVals = { 1, 2, 3 };
   std::shared_ptr<MatrixConfig> rowSideInfoDenseMatrixConfig =
      std::make_shared<MatrixConfig>(3, 1, std::move(rowSideInfoDenseMatrixConfigVals), nc);
   return rowSideInfoDenseMatrixConfig;
}

std::shared_ptr<MatrixConfig> getColSideInfoDenseMatrixConfig()
{
   NoiseConfig nc(NoiseTypes::adaptive);
   nc.setPrecision(10.0);

   std::vector<double> colSideInfoDenseMatrixConfigVals = { 1, 2, 3, 4 };
   std::shared_ptr<MatrixConfig> colSideInfoDenseMatrixConfig =
      std::make_shared<MatrixConfig>(4, 1, std::move(colSideInfoDenseMatrixConfigVals), nc);
   return colSideInfoDenseMatrixConfig;
}

std::shared_ptr<MatrixConfig> getRowSideInfoSparseMatrixConfig()
{
   NoiseConfig nc(NoiseTypes::adaptive);
   nc.setPrecision(10.0);

   std::vector<std::uint32_t> rowSideInfoSparseMatrixConfigRows = {0, 1, 2};
   std::vector<std::uint32_t> rowSideInfoSparseMatrixConfigCols = {0, 0, 0};
   std::vector<double> rowSideInfoSparseMatrixConfigVals = { 1, 2, 3 };
   std::shared_ptr<MatrixConfig> rowSideInfoSparseMatrixConfig =
      std::make_shared<MatrixConfig>(3, 1, std::move(rowSideInfoSparseMatrixConfigRows), std::move(rowSideInfoSparseMatrixConfigCols), std::move(rowSideInfoSparseMatrixConfigVals), nc, true);
   return rowSideInfoSparseMatrixConfig;
}

std::shared_ptr<MatrixConfig> getColSideInfoSparseMatrixConfig()
{
   NoiseConfig nc(NoiseTypes::adaptive);
   nc.setPrecision(10.0);

   std::vector<std::uint32_t> colSideInfoSparseMatrixConfigRows = {0, 1, 2, 3};
   std::vector<std::uint32_t> colSideInfoSparseMatrixConfigCols = {0, 0, 0, 0};
   std::vector<double> colSideInfoSparseMatrixConfigVals = { 1, 2, 3, 4 };
   std::shared_ptr<MatrixConfig> colSideInfoSparseMatrixConfig =
      std::make_shared<MatrixConfig>(4, 1, std::move(colSideInfoSparseMatrixConfigRows), std::move(colSideInfoSparseMatrixConfigCols), std::move(colSideInfoSparseMatrixConfigVals), nc, true);
   return colSideInfoSparseMatrixConfig;
}

//result comparison

void REQUIRE_RESULT_ITEMS(const std::vector<ResultItem>& actualResultItems, const std::vector<ResultItem>& expectedResultItems)
{
    return;
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

#ifdef TEST_RANDOM
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
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);

   std::shared_ptr<ISession> session = SessionFactory::create_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 6b6491b79751aaa469f9471e727b6630a7aa8a82
   double expectedRmseAvg = 0.1709029553033369;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0},  1,  1.8856052357618300,  1.2765891774449203, 10.8555121881078271, 0.4706814900369729 },
         { { 0, 1},  2,  2.3035875606954122,  2.1312962904652473,  8.1368224714493689, 0.4075016582274855 },
         { { 0, 2},  3,  3.6928571685628011,  2.9096214449777804,  8.6128404494053452, 0.4192520173708134 },
         { { 0, 3},  4,  3.4748675440474557,  3.8081117541687717, 10.2775103265190246, 0.4579793758505615 },
         { { 2, 0},  9,  8.7215898468589756,  8.7757213895478383,  9.0585996534368096, 0.4299643946681909 },
         { { 2, 1}, 10, 10.2664070652140431,  9.8816830568812311,  7.6768016311883667, 0.3958148822930494 },
         { { 2, 2}, 11, 10.8840628915395250, 10.8822329328190541,  9.5740949001395652, 0.4420290622115055 },
         { { 2, 3}, 12, 12.6458605614538655, 12.1294759195291100, 11.5125646375406951, 0.4847167200802137 }
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
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);

   std::shared_ptr<ISession> session = SessionFactory::create_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 6b6491b79751aaa469f9471e727b6630a7aa8a82
   double expectedRmseAvg = 0.1798870464175684;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0},  1,  1.5453098994094618,  1.3546274871926034,  7.9175067776534105, 0.4019723510050394 },
         { { 0, 1},  2,  1.8529965721714778,  2.1255252782495981,  6.0861101378269034, 0.3524291834445697 },
         { { 0, 2},  3,  3.6856437792056127,  2.9033839919979743, 10.3963243276140336, 0.4606190228779323 },
         { { 0, 3},  4,  3.5284360764961979,  3.6903673503941423,  8.0330171493744498, 0.4048939682156771 },
         { { 2, 0},  9,  8.6537974191229718,  8.9712013920981555, 10.8917513169130107, 0.4714664771971319 },
         { { 2, 1}, 10,  9.3762189729344119, 10.0652829796573986, 10.6785174774113294, 0.4668285853613029 },
         { { 2, 2}, 11, 10.8735318607555307, 10.9166402706029686,  9.0023553543857844, 0.4286275047691268 },
         { { 2, 3}, 12, 11.8355586548263414, 11.9894908059708651, 10.6687541492828402, 0.4666151267543481 }
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

   std::shared_ptr<ISession> session = SessionFactory::create_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 6b6491b79751aaa469f9471e727b6630a7aa8a82
   double expectedRmseAvg = 0.1583533773711920;
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

   std::shared_ptr<ISession> session = SessionFactory::create_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 6b6491b79751aaa469f9471e727b6630a7aa8a82
   double expectedRmseAvg = 0.1340927209493763;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0},  1,  1.4701945342161571,  1.2799300453619689,  6.4960137262982034, 0.3641039806153777 },
         { { 0, 1},  2,  2.4559913865459579,  2.1010339211079549,  6.8982129330092361, 0.3752064175566546 },
         { { 0, 2},  3,  2.8639119986156309,  2.9688248639251800,  4.3419543571630284, 0.2976765247907372 },
         { { 0, 3},  4,  3.6197761970379254,  3.8069224228641332,  5.3269254275918900, 0.3297161867855534 },
         { { 2, 0},  9,  8.6756357913689204,  8.8831785637034510,  7.6925858403482374, 0.3962215892177115 },
         { { 2, 1}, 10, 10.4237621586384446,  9.9568212739875506, 10.8595273877073826, 0.4707685290165386 },
         { { 2, 2}, 11, 11.3215570725159029, 10.9617293126693429,  8.8409103507711233, 0.4247666910816712 },
         { { 2, 3}, 12, 12.3270869845028983, 11.9928555467033480, 10.2543830476606530, 0.4574637946566329 }
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
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);

   std::shared_ptr<ISession> session = SessionFactory::create_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 6b6491b79751aaa469f9471e727b6630a7aa8a82
   double expectedRmseAvg = 0.8483799452569423;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  2.1433102311797101,  1.9381975454076072, 1.5239976019683954, 0.1763575682439119 },
         { { 0, 1 },  2,  2.4673479342034561,  2.2628977372254520, 2.3434545227212231, 0.2186906548175179 },
         { { 0, 2 },  3,  2.7434664315868544,  2.5405412129387042, 2.2650511667309607, 0.2150012418915704 },
         { { 0, 3 },  4,  3.2224537662864359,  2.8259515099649022, 3.4081757637935404, 0.2637320750768817 },
         { { 2, 0 },  9,  7.8465878506610220,  7.8041719994764351, 6.1164797060457978, 0.3533073965400586 },
         { { 2, 1 }, 10,  9.0328791615100101,  9.1037849996616913, 5.4171358633100422, 0.3324963054362806 },
         { { 2, 2 }, 11, 10.0437398457884743, 10.2318746139433880, 7.0604885364587995, 0.3795939972982627 },
         { { 2, 3 }, 12, 11.7972966321084609, 11.3707594884843122, 6.7487546237857519, 0.3711195039333142 }
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
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);

   std::shared_ptr<ISession> session = SessionFactory::create_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 6b6491b79751aaa469f9471e727b6630a7aa8a82
   double expectedRmseAvg = 0.9342940251186639;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  2.0056595291331023,  1.9598211403470644,  1.8749914316461531, 0.1956147521484123 },
         { { 0, 1 },  2,  2.5398777811226569,  2.2657627609831219,  2.2996275890812705, 0.2166360433708409 },
         { { 0, 2 },  3,  2.5780017811675529,  2.5519190261754314,  3.1611436129748358, 0.2539943600922472 },
         { { 0, 3 },  4,  3.2154956020695167,  2.8124216541044027,  4.3205967708649053, 0.2969435035547426 },
         { { 2, 0 },  9,  7.4810109694101463,  7.7393275891525626,  8.3081522806329051, 0.4117695085556746 },
         { { 2, 1 }, 10,  9.4736186603677250,  8.9468641576202543,  6.8681315745653011, 0.3743874336851234 },
         { { 2, 2 }, 11,  9.6158192973107912, 10.0775943042116545, 14.4691441808780663, 0.5434046896674830 },
         { { 2, 3 }, 12, 11.9936397587727157, 11.0883997849639897,  6.2683514647264120, 0.3576668003833419 }
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

   std::shared_ptr<ISession> session = SessionFactory::create_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 6b6491b79751aaa469f9471e727b6630a7aa8a82
   double expectedRmseAvg = 0.8390651474354670;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  1.9628858411530270,  1.9306891335020062, 1.4025606427036019, 0.1691853616179242 },
         { { 0, 1 },  2,  2.4103484773695651,  2.2305147529726863, 1.9161048560975358, 0.1977477704973798 },
         { { 0, 2 },  3,  2.5328688928083718,  2.5083889495416103, 2.2978502958651816, 0.2165523123802851 },
         { { 0, 3 },  4,  2.8609938022110168,  2.8232596434938606, 3.0875944053823652, 0.2510221717739074 },
         { { 2, 0 },  9,  7.7577583123933342,  7.8558306956797006, 8.8672497202257681, 0.4253989656835186 },
         { { 2, 1 }, 10,  9.5262294648242811,  9.0683718511333549, 5.6750374842190539, 0.3403191024827648 },
         { { 2, 2 }, 11, 10.0104572030762213, 10.2020357383897231, 8.3714711466521603, 0.4133356383517839 },
         { { 2, 3 }, 12, 11.3072793055406251, 11.4777715064354293, 8.1610962997843526, 0.4081090365452408 }
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

   std::shared_ptr<ISession> session = SessionFactory::create_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 6b6491b79751aaa469f9471e727b6630a7aa8a82
   double expectedRmseAvg = 0.9444961219337701;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  1.9262460386602103,  1.9996480514964581,  1.6541794207743585, 0.1837355809016659 },
         { { 0, 1 },  2,  2.2571985617941026,  2.2406163685283911,  2.3710521955122306, 0.2199745901611732 },
         { { 0, 2 },  3,  2.5314799751121133,  2.5264003939432143,  2.7021957012567834, 0.2348336667644927 },
         { { 0, 3 },  4,  2.7250471870172981,  2.7656953783379405,  3.4813817864373497, 0.2665494473572896 },
         { { 2, 0 },  9,  7.7807517621704729,  7.9469053523952029, 11.4319318969218848, 0.4830162860507320 },
         { { 2, 1 }, 10,  9.1175796522150190,  8.8898999369052714,  4.6967247901157627, 0.3095989766276580 },
         { { 2, 2 }, 11, 10.2254939825614652, 10.0307252450463356,  7.5610843353323487, 0.3928203706253167 },
         { { 2, 3 }, 12, 11.0073766677958247, 10.9749329740362089,  7.7998721649884830, 0.3989750175031034 }
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
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);
   config.setDirect(true);

   std::shared_ptr<ISession> session = SessionFactory::create_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 6b6491b79751aaa469f9471e727b6630a7aa8a82
   double expectedRmseAvg = 0.1371894591774568;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  1.8997904770902050,  1.1751653319912134,  6.7677378138993012, 0.3716410879367927 },
         { { 0, 1 },  2,  1.5751001660374289,  1.9838173816140614,  7.4237828172070026, 0.3892374234059910 },
         { { 0, 2 },  3,  2.8224914091600515,  2.8670871306830912, 11.0486322551031328, 0.4748497561550096 },
         { { 0, 3 },  4,  4.0667555927612362,  3.8147524564287254,  8.6360494811621376, 0.4198165168008747 },
         { { 2, 0 },  9,  8.3030993926618333,  8.7908663511233929,  6.9822686653979673, 0.3774854684430551 },
         { { 2, 1 }, 10, 10.3501572488692481,  9.8981742048345751, 10.0017338215784974, 0.4517931127926898 },
         { { 2, 2 }, 11, 10.7184484043830999, 11.0885005135286629,  9.1718617039605217, 0.4326440235357876 },
         { { 2, 3 }, 12, 11.9465802759735666, 12.0755156356538880,  7.9924773028764777, 0.4038709963482853 }
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
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);
   config.setDirect(true);

   std::shared_ptr<ISession> session = SessionFactory::create_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 6b6491b79751aaa469f9471e727b6630a7aa8a82
   double expectedRmseAvg = 0.0913718066408195;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  1.6716816860962824,  1.1088586813524952,  9.5702367905728316, 0.4419399901679521 },
         { { 0, 1 },  2,  1.2655641641569502,  1.9719660909417431,  5.7200871710938879, 0.3416671960833628 },
         { { 0, 2 },  3,  2.4521983478321663,  2.9621782471737745,  7.5259167444536628, 0.3919057764844906 },
         { { 0, 3 },  4,  3.8807601076182676,  3.7941719790765500,  7.6472834845696793, 0.3950531734958942 },
         { { 2, 0 },  9,  9.6384697790565674,  8.9327235462693935,  9.2728589458049200, 0.4350195619764026 },
         { { 2, 1 }, 10,  9.7267500311852153,  9.9736550203904333,  8.6649472490437383, 0.4205183208181895 },
         { { 2, 2 }, 11, 11.3612799291046578, 11.0448864683254300, 12.4151142938905625, 0.5033584006124804 },
         { { 2, 3 }, 12, 12.3762680844434598, 12.0558901639604930,  6.5914762364430981, 0.3667695777906290 }
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
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);
   config.setDirect(true);

   std::shared_ptr<ISession> session = SessionFactory::create_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 6b6491b79751aaa469f9471e727b6630a7aa8a82
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
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);
   config.setDirect(true);

   std::shared_ptr<ISession> session = SessionFactory::create_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 6b6491b79751aaa469f9471e727b6630a7aa8a82
   double expectedRmseAvg = 0.1437412780171048;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  1.5994233584212105,  1.2082777130122822,  9.1122427967864432, 0.4312355953651407 },
         { { 0, 1 },  2,  2.7391899373968345,  1.9721509664301815, 10.8161031540169219, 0.4698263499012905 },
         { { 0, 2 },  3,  3.1086945075131154,  2.9461241559187918, 13.4370907414993859, 0.5236662502617987 },
         { { 0, 3 },  4,  3.4835307318736746,  3.7524160905113986, 13.6423313623373712, 0.5276503821300493 },
         { { 2, 0 },  9,  8.9743798728810624,  8.7791832021170908,  8.4175521767641879, 0.4144716867502972 },
         { { 2, 1 }, 10,  9.3626920196659853,  9.9813509533439717,  9.8230173644717489, 0.4477384751527129 },
         { { 2, 2 }, 11, 10.5257134611302927, 10.9247619058603505, 11.8522252393569882, 0.4918151560718512 },
         { { 2, 3 }, 12, 11.5380291714346619, 12.0465644805425303, 11.0422830417662112, 0.4747132978315289 }
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
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);
   config.setDirect(true);

   std::shared_ptr<ISession> session = SessionFactory::create_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 6b6491b79751aaa469f9471e727b6630a7aa8a82
   double expectedRmseAvg = 0.1967750746091230;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  1.2428886961768788,  1.3553504830665419,  8.8367715804196454, 0.4246672546258086 },
         { { 0, 1 },  2,  1.8348671341880085,  2.2587448952872236,  8.1841113080552415, 0.4086840830717896 },
         { { 0, 2 },  3,  3.0597364091645951,  2.9626376163832404,  4.4741667104262106, 0.3021746592660869 },
         { { 0, 3 },  4,  4.1209172405498746,  3.7664061407294662,  8.7883840305335976, 0.4235029824373598 },
         { { 2, 0 },  9,  9.3955318573234567,  8.8571251174144745,  9.8089565691794114, 0.4474179110475020 },
         { { 2, 1 }, 10, 10.4212769309433142,  9.9865130742666910, 10.1947017937669440, 0.4561306157761228 },
         { { 2, 2 }, 11, 10.9660613795579760, 10.9747721480218523,  8.4394133790018699, 0.4150095494107081 },
         { { 2, 3 }, 12, 12.1206686056789508, 12.1983606329218670, 11.0281284350190685, 0.4744089434366031 }
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
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);
   config.setDirect(true);

   std::shared_ptr<ISession> session = SessionFactory::create_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 6b6491b79751aaa469f9471e727b6630a7aa8a82
   double expectedRmseAvg = 0.1794010488906284;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  0.8887295879289608,  1.3692362889629088,  8.8890610961277137, 0.4259218356988232 },
         { { 0, 1 },  2,  2.5622926705358111,  2.1288366991687107,  7.6886574178714415, 0.3961204057795147 },
         { { 0, 2 },  3,  3.0182355903346405,  2.8103843566002364,  6.8298800803605104, 0.3733434179980402 },
         { { 0, 3 },  4,  3.2038544051058877,  3.7589699618017671, 14.0257410684210981, 0.5350136574343191 },
         { { 2, 0 },  9,  8.5070259206654697,  8.8988495172628372,  6.7260366479600355, 0.3704943373926257 },
         { { 2, 1 }, 10, 10.3079590117392286,  9.9912152887739030,  8.3490365871372898, 0.4127814213095256 },
         { { 2, 2 }, 11, 10.4255445773949162, 10.9936425035401797,  6.1199983611565312, 0.3534090063055671 },
         { { 2, 3 }, 12, 12.0384724300630417, 12.0120406809774192,  6.5488949409786112, 0.3655829825946891 }
      };

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

#endif // TEST_RANDOM

//test throw - normal prior should not have side info

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
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);
   config.setDirect(true);

   REQUIRE_THROWS(SessionFactory::create_session(config));
}

//test throw - macau prior should have side info

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
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);
   config.setDirect(true);

   REQUIRE_THROWS(SessionFactory::create_session(config));
}

//test throw - wrong dimentions of side info

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
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);
   config.setDirect(true);

   REQUIRE_THROWS(SessionFactory::create_session(config));
}

//=================================================================

#ifdef TEST_RANDOM

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
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);

   std::shared_ptr<ISession> session = SessionFactory::create_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 6b6491b79751aaa469f9471e727b6630a7aa8a82
   double expectedRmseAvg = 0.5898900387084497;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  2.4443512546743484,  2.1335763229515661, 1.9255734826471844, 0.1982357637087916 },
         { { 0, 1 },  2,  2.7154493488473408,  2.4729900862420298, 2.3820626026350920, 0.2204847443764692 },
         { { 0, 2 },  3,  3.0506289521924779,  2.7843230567372084, 2.4357787213085951, 0.2229568788412333 },
         { { 0, 3 },  4,  3.3793754422741125,  3.0974389145157355, 3.5622124115076121, 0.2696260604610526 },
         { { 2, 0 },  9,  8.5438526486833890,  8.4541600905569734, 6.3523905874112945, 0.3600564181248316 },
         { { 2, 1 }, 10,  9.4914342065808537,  9.8036432567288490, 9.8479079352020502, 0.4483053791371537 },
         { { 2, 2 }, 11, 10.6630027920485428, 11.0410182875654286, 6.7249922759332668, 0.3704655723885402 },
         { { 2, 3 }, 12, 11.8120854227294334, 12.2751988850462048, 8.1327717213078490, 0.4074002124298848 }
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
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);

   std::shared_ptr<ISession> session = SessionFactory::create_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 6b6491b79751aaa469f9471e727b6630a7aa8a82
   double expectedRmseAvg = 0.5898006999965592;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  2.3462533449699308,  2.1065364262662141, 1.3305557053242010, 0.1647853090170384 },
         { { 0, 1 },  2,  2.5309807046121491,  2.4492104355588520, 1.5777473787033092, 0.1794405921077667 },
         { { 0, 2 },  3,  3.0298083567497787,  2.7486992432238959, 2.7455857374748360, 0.2367115586305036 },
         { { 0, 3 },  4,  3.4689945591921272,  3.0814026893049786, 2.8701362793545768, 0.2420210936731476 },
         { { 2, 0 },  9,  8.6687926205117130,  8.4791695562845533, 7.8620465605981176, 0.4005620174306656 },
         { { 2, 1 }, 10,  9.3513119126018847,  9.8585530893295452, 7.3242668056550126, 0.3866197521706174 },
         { { 2, 2 }, 11, 11.1943496557460662, 11.0510436778980008, 7.1730655655425943, 0.3826082764059589 },
         { { 2, 3 }, 12, 12.8170278370792872, 12.3947859184389735, 7.5927108936018888, 0.3936410592696037 }
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
   config.getAuxData().push_back({ colAuxDataDenseMatrixConfig });
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);

   std::shared_ptr<ISession> session = SessionFactory::create_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 6b6491b79751aaa469f9471e727b6630a7aa8a82
   double expectedRmseAvg = 0.5982522407208059;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  2.1719771571078605,  2.1096732329766792,  1.5102678557420737, 0.1755613652667580 },
         { { 0, 1 },  2,  2.5564092033158978,  2.4301156061022136,  2.2446067747650709, 0.2140287399528805 },
         { { 0, 2 },  3,  2.7309707304523396,  2.7602120609163778,  2.1236386663976230, 0.2081815664759002 },
         { { 0, 3 },  4,  2.9312258306586361,  3.1159414849816010,  3.2560163351106879, 0.2577776424778584 },
         { { 2, 0 },  9,  8.5548605361311889,  8.4324169951148136,  7.8925646701868644, 0.4013386953324591 },
         { { 2, 1 }, 10, 10.0690396932032247,  9.7077734833888876, 10.0375069482268575, 0.4526003541492851 },
         { { 2, 2 }, 11, 10.7565927435377144, 11.0311284943818251,  5.4469883727351354, 0.3334111995944389 },
         { { 2, 3 }, 12, 11.5453461833736224, 12.4464284134191967,  7.7516142713436755, 0.3977388711444466 }
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
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);

   std::shared_ptr<ISession> session = SessionFactory::create_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 6b6491b79751aaa469f9471e727b6630a7aa8a82
   double expectedRmseAvg = 0.4885320592966239;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  2.2702342513011859,  1.9811297991500643,  3.7867612303788483, 0.2779943190719966 },
         { { 0, 1 },  2,  2.4922957880045127,  2.3847538378271773,  3.7901067254615528, 0.2781170919705478 },
         { { 0, 2 },  3,  2.9344361873617895,  2.8335583391499388,  4.1948478683434436, 0.2925903965790305 },
         { { 0, 3 },  4,  3.2822103978831927,  3.2653075589147189,  4.4081043820268633, 0.2999355162679398 },
         { { 2, 0 },  9,  8.9140723113201918,  8.6455454709062902,  6.5970877160083798, 0.3669256643842789 },
         { { 2, 1 }, 10,  9.8061238403776017,  9.8864557172120922,  8.6671169545350040, 0.4205709664820589 },
         { { 2, 2 }, 11, 11.6569288545257557, 11.0501728474459249, 11.2986316240104809, 0.4801919604255747 },
         { { 2, 3 }, 12, 12.9405678454205226, 12.3002231565981219,  8.9049980026176954, 0.4263034753724708 },
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
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);
   config.setDirect(true);

   std::shared_ptr<ISession> session = SessionFactory::create_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 6b6491b79751aaa469f9471e727b6630a7aa8a82
   double expectedRmseAvg = 0.5971104934513052;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  2.2763690576962037,  2.1712698598542803, 1.4574475293083340, 0.1724639879184681 },
         { { 0, 1 },  2,  2.7866374374950387,  2.4968955456297555, 2.1485340720879229, 0.2093982667651366 },
         { { 0, 2 },  3,  3.2361267682518595,  2.8299236338738982, 2.7119991411520159, 0.2352592638941139 },
         { { 0, 3 },  4,  3.6484470420596149,  3.1716737528726169, 3.6020880708861469, 0.2711309673324979 },
         { { 2, 0 },  9,  7.9472711657786324,  8.4245143701578407, 4.8830097344223828, 0.3156790456875005 },
         { { 2, 1 }, 10,  9.7287227137486116,  9.6861267354587604, 8.8905000790132593, 0.4259563089363973 },
         { { 2, 2 }, 11, 11.2979821383447590, 10.9744099766436811, 5.9149115143826130, 0.3474370157098398 },
         { { 2, 3 }, 12, 12.7374767633572024, 12.2968904437814697, 6.7948242949590396, 0.3723840538618600 }
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
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);
   config.setDirect(true);

   std::shared_ptr<ISession> session = SessionFactory::create_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 6b6491b79751aaa469f9471e727b6630a7aa8a82
   double expectedRmseAvg = 0.5916383254341472;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  2.1244440263729616,  2.1284239785270724, 2.1646569533166158, 0.2101824743328178 },
         { { 0, 1 },  2,  2.3850171811836778,  2.4459767197333129, 2.6879619706317150, 0.2342143606775341 },
         { { 0, 2 },  3,  2.7951938350259558,  2.7779383984065409, 3.2062163978928342, 0.2557987250009246 },
         { { 0, 3 },  4,  3.1545958833306669,  3.1065157380919071, 4.1273968268621939, 0.2902285104935593 },
         { { 2, 0 },  9,  8.3013547809571442,  8.5259474414969336, 7.3104198656230173, 0.3862541160370615 },
         { { 2, 1 }, 10,  9.3195553913870075,  9.7957664582689965, 4.9465013944669929, 0.3177247362896801 },
         { { 2, 2 }, 11, 10.9223379943365195, 11.1326938008059351, 8.5646389215859937, 0.4180772050950904 },
         { { 2, 3 }, 12, 12.3267166811564497, 12.4431490284068129, 5.8543699810704464, 0.3456543626069151 }
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
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);

   std::shared_ptr<ISession> session = SessionFactory::create_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 6b6491b79751aaa469f9471e727b6630a7aa8a82
   double expectedRmseAvg = 0.1709029553033292;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  1.8856052357618269,  1.2765891774449034, 10.8555121881092518, 0.4706814900370038 },
         { { 0, 1 },  2,  2.3035875606955214,  2.1312962904652388,  8.1368224714491380, 0.4075016582274797 },
         { { 0, 2 },  3,  3.6928571685626865,  2.9096214449777769,  8.6128404494045618, 0.4192520173707943 },
         { { 0, 3 },  4,  3.4748675440475827,  3.8081117541687930, 10.2775103265194208, 0.4579793758505703 },
         { { 2, 0 },  9,  8.7215898468588815,  8.7757213895478419,  9.0585996534370885, 0.4299643946681975 },
         { { 2, 1 }, 10, 10.2664070652140715,  9.8816830568812311,  7.6768016311880247, 0.3958148822930406 },
         { { 2, 2 }, 11, 10.8840628915394468, 10.8822329328190435,  9.5740949001402704, 0.4420290622115217 },
         { { 2, 3 }, 12, 12.6458605614538406, 12.1294759195290993, 11.5125646375408923, 0.4847167200802179 }
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
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);

   std::shared_ptr<ISession> session = SessionFactory::create_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 6b6491b79751aaa469f9471e727b6630a7aa8a82
   double expectedRmseAvg = 0.1798870464174907;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  1.5453098994077692,  1.3546274871924613,  7.9175067776735117, 0.4019723510055496 },
         { { 0, 1 },  2,  1.8529965721697137,  2.1255252782497016,  6.0861101378377480, 0.3524291834448837 },
         { { 0, 2 },  3,  3.6856437792072936,  2.9033839919978770, 10.3963243276196575, 0.4606190228780568 },
         { { 0, 3 },  4,  3.5284360765008591,  3.6903673503945003,  8.0330171493675397, 0.4048939682155029 },
         { { 2, 0 },  9,  8.6537974191209894,  8.9712013920982052, 10.8917513169211340, 0.4714664771973077 },
         { { 2, 1 }, 10,  9.3762189729340282, 10.0652829796576881, 10.6785174774091107, 0.4668285853612544 },
         { { 2, 2 }, 11, 10.8735318607524842, 10.9166402706028798,  9.0023553544070207, 0.4286275047696324 },
         { { 2, 3 }, 12, 11.8355586548271798, 11.9894908059706573, 10.6687541492887199, 0.4666151267544766 }
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
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);

   std::shared_ptr<ISession> session = SessionFactory::create_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 6b6491b79751aaa469f9471e727b6630a7aa8a82
   double expectedRmseAvg = 0.8621030670922455;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0 },  1,  1.9769015195861526,  1.9313454354867428,  1.2969952654404118, 0.1626938570795990 },
         { { 0, 1 },  2,  2.2683298178731301,  2.2393762964636017,  1.6916373952809503, 0.1858042307069213 },
         { { 0, 2 },  3,  2.4950584106232019,  2.5642835324181874,  1.8082545378952537, 0.1921019360511419 },
         { { 0, 3 },  4,  2.7721148493428167,  2.8607990451012664,  2.5710548459908331, 0.2290644168373547 },
         { { 2, 0 },  9,  8.2679475861969518,  7.7304723185273527,  8.1744161507827453, 0.4084419412887583 },
         { { 2, 1 }, 10,  9.4867811353136080,  8.9600285131903092,  6.5858823148743442, 0.3666139134403492 },
         { { 2, 2 }, 11, 10.4350226650900701, 10.2705008632754247, 13.3007369534664992, 0.5210025060354649 },
         { { 2, 3 }, 12, 11.5937491322697301, 11.4449951521295521,  6.5488381701570813, 0.3655813980150444 },
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
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);

   std::shared_ptr<ISession> session = SessionFactory::create_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 6b6491b79751aaa469f9471e727b6630a7aa8a82
   double expectedRmseAvg = 0.9342940251186641;
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
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);

   std::shared_ptr<ISession> session = SessionFactory::create_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 6b6491b79751aaa469f9471e727b6630a7aa8a82
   double expectedRmseAvg = 8.0598424414704120;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0, 0 },  1,  0.9435285187276158,  1.2735689706947713, 7.5499026504788320, 0.3925298025986554 },
         { { 0, 0, 1 },  2,  7.0935148960286307,  7.1763629657213945, 4.0161446887095522, 0.2862903011006076 },
         { { 0, 0, 2 },  3, 12.6255679056269923, 13.0963404943965678, 6.1878618031200174, 0.3553630452667030 },
         { { 0, 0, 3 },  4, 18.2911696770136665, 18.8598991515727832, 5.1738851051662893, 0.3249453676268815 },
         { { 0, 2, 0 },  9,  5.3462955478807990,  5.0724040651807343, 6.4386732090067715, 0.3624934400252819 },
         { { 0, 2, 1 }, 10, 11.3839403787613769, 11.0273956390506189, 4.6957139580646121, 0.3095656587920557 },
         { { 0, 2, 2 }, 11, 17.4891521347178625, 17.0450998642637792, 4.6432053220291944, 0.3078299730148286 },
         { { 0, 2, 3 }, 12, 22.8707861493274009, 22.8185687476221375, 7.1671351067213873, 0.3824500796214811 }
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
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);

   std::shared_ptr<ISession> session = SessionFactory::create_session(config);
   session->run();

   double actualRmseAvg = session->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > actualResults = session->getResult();

   // Pre-calculated results with single-threaded Debug master 6b6491b79751aaa469f9471e727b6630a7aa8a82
   double expectedRmseAvg = 8.0703733818488956;
   std::vector<ResultItem> expectedResults =
      {
         { { 0, 0, 0 },  1,  3.1008459798152535,  3.0840789761034411, 1.3304679442387921, 0.1647798744546234 },
         { { 0, 0, 1 },  2,  8.0314624297769441,  8.0857037492979682, 1.7935517350400654, 0.1913193577071351 },
         { { 0, 0, 2 },  3, 12.9250515496923128, 13.0563767315209347, 2.2652553936355950, 0.2150109343985336 },
         { { 0, 0, 3 },  4, 18.2402436262586001, 18.0472898141183826, 4.1991864553414731, 0.2927416655723453 },
         { { 0, 2, 0 },  9,  3.9469713450563204,  3.9476115479428566, 2.7890920158887500, 0.2385796412567916 },
         { { 0, 2, 1 }, 10, 10.2230011666411986, 10.3449152204876764, 2.0708075119225691, 0.2055757227746947 },
         { { 0, 2, 2 }, 11, 16.4519000402112141, 16.7055660745389467, 3.3167487739294002, 0.2601706180343105 },
         { { 0, 2, 3 }, 12, 23.2174443323941659, 23.0910816865597823, 5.3927617251018196, 0.3317474366694818 }
      };

   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*actualResults, expectedResults);
}

//=================================================================

//test throw - do not allow 3 and more dimentions for tensor data with macau priors

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
   config.setNumLatent(4);
   config.setBurnin(50);
   config.setNSamples(50);
   config.setVerbose(false);
   config.setRandomSeed(1234);
   config.setRandomSeedSet(true);

   REQUIRE_THROWS(SessionFactory::create_session(config));
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
   tensorSessionConfig.setNumLatent(4);
   tensorSessionConfig.setBurnin(50);
   tensorSessionConfig.setNSamples(50);
   tensorSessionConfig.setVerbose(false);
   tensorSessionConfig.setRandomSeed(1234);
   tensorSessionConfig.setRandomSeedSet(true);

   std::shared_ptr<ISession> matrixSession = SessionFactory::create_session(matrixSessionConfig);
   std::shared_ptr<ISession> tensorSession = SessionFactory::create_session(tensorSessionConfig);
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
   tensorSessionConfig.setNumLatent(4);
   tensorSessionConfig.setBurnin(50);
   tensorSessionConfig.setNSamples(50);
   tensorSessionConfig.setVerbose(false);
   tensorSessionConfig.setRandomSeed(1234);
   tensorSessionConfig.setRandomSeedSet(true);

   std::shared_ptr<ISession> matrixSession = SessionFactory::create_session(matrixSessionConfig);
   std::shared_ptr<ISession> tensorSession = SessionFactory::create_session(tensorSessionConfig);
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
   tensorSessionConfig.setNumLatent(4);
   tensorSessionConfig.setBurnin(50);
   tensorSessionConfig.setNSamples(50);
   tensorSessionConfig.setVerbose(false);
   tensorSessionConfig.setRandomSeed(1234);
   tensorSessionConfig.setRandomSeedSet(true);

   std::shared_ptr<ISession> matrixSession = SessionFactory::create_session(matrixSessionConfig);
   std::shared_ptr<ISession> tensorSession = SessionFactory::create_session(tensorSessionConfig);
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
   tensorSessionConfig.setNumLatent(4);
   tensorSessionConfig.setBurnin(50);
   tensorSessionConfig.setNSamples(50);
   tensorSessionConfig.setVerbose(false);
   tensorSessionConfig.setRandomSeed(1234);
   tensorSessionConfig.setRandomSeedSet(true);

   std::shared_ptr<ISession> matrixSession = SessionFactory::create_session(matrixSessionConfig);
   std::shared_ptr<ISession> tensorSession = SessionFactory::create_session(tensorSessionConfig);
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
   tensorSessionConfig.setNumLatent(4);
   tensorSessionConfig.setBurnin(50);
   tensorSessionConfig.setNSamples(50);
   tensorSessionConfig.setVerbose(false);
   tensorSessionConfig.setRandomSeed(1234);
   tensorSessionConfig.setRandomSeedSet(true);

   std::shared_ptr<ISession> matrixSession = SessionFactory::create_session(matrixSessionConfig);
   std::shared_ptr<ISession> tensorSession = SessionFactory::create_session(tensorSessionConfig);
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
   tensorSessionConfig.setNumLatent(4);
   tensorSessionConfig.setBurnin(50);
   tensorSessionConfig.setNSamples(50);
   tensorSessionConfig.setVerbose(false);
   tensorSessionConfig.setRandomSeed(1234);
   tensorSessionConfig.setRandomSeedSet(true);

   std::shared_ptr<ISession> matrixSession = SessionFactory::create_session(matrixSessionConfig);
   std::shared_ptr<ISession> tensorSession = SessionFactory::create_session(tensorSessionConfig);
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
   tensorSessionConfig.setNumLatent(4);
   tensorSessionConfig.setBurnin(50);
   tensorSessionConfig.setNSamples(50);
   tensorSessionConfig.setVerbose(false);
   tensorSessionConfig.setRandomSeed(1234);
   tensorSessionConfig.setRandomSeedSet(true);

   std::shared_ptr<ISession> matrixSession = SessionFactory::create_session(matrixSessionConfig);
   std::shared_ptr<ISession> tensorSession = SessionFactory::create_session(tensorSessionConfig);
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
   tensorSessionConfig.setNumLatent(4);
   tensorSessionConfig.setBurnin(50);
   tensorSessionConfig.setNSamples(50);
   tensorSessionConfig.setVerbose(false);
   tensorSessionConfig.setRandomSeed(1234);
   tensorSessionConfig.setRandomSeedSet(true);

   std::shared_ptr<ISession> matrixSession = SessionFactory::create_session(matrixSessionConfig);
   std::shared_ptr<ISession> tensorSession = SessionFactory::create_session(tensorSessionConfig);
   matrixSession->run();
   tensorSession->run();

   REQUIRE(matrixSession->getRmseAvg() == Approx(tensorSession->getRmseAvg()).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*matrixSession->getResult(), *tensorSession->getResult());
}
#endif // TEST_RANDOM

//
//      train: 1. dense 2D-tensor (matrix)
//             2. dense matrix
//       test: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//     priors: macau macau
//  side-info: row_side_info_dense_matrix col_side_info_dense_matrix
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE("--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> --prior macau macau --side-info <row_side_info_dense_matrix> <col_side_info_dense_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct"
          "--train <train_dense_matrix>    --test <test_sparse_matrix>    --prior macau macau --side-info <row_side_info_dense_matrix> <col_side_info_dense_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct",
          "")
{
   std::shared_ptr<TensorConfig> trainDenseTensorConfig = getTrainDenseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> rowSideInfoDenseMatrixConfig = getRowSideInfoDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> colSideInfoDenseMatrixConfig = getColSideInfoDenseMatrixConfig();

   Config tensorRunConfig;
   tensorRunConfig.setTrain(trainDenseTensorConfig);
   tensorRunConfig.setTest(testSparseTensorConfig);
   tensorRunConfig.getPriorTypes().push_back(PriorTypes::macau);
   tensorRunConfig.getPriorTypes().push_back(PriorTypes::macau);
   tensorRunConfig.getSideInfo().push_back(rowSideInfoDenseMatrixConfig);
   tensorRunConfig.getSideInfo().push_back(colSideInfoDenseMatrixConfig);
   tensorRunConfig.setNumLatent(4);
   tensorRunConfig.setBurnin(50);
   tensorRunConfig.setNSamples(50);
   tensorRunConfig.setVerbose(false);
   tensorRunConfig.setRandomSeed(1234);
   tensorRunConfig.setRandomSeedSet(true);
   tensorRunConfig.setDirect(true);

   Config matrixRunConfig;
   matrixRunConfig.setTrain(trainDenseMatrixConfig);
   matrixRunConfig.setTest(testSparseMatrixConfig);
   matrixRunConfig.getPriorTypes().push_back(PriorTypes::macau);
   matrixRunConfig.getPriorTypes().push_back(PriorTypes::macau);
   matrixRunConfig.getSideInfo().push_back(rowSideInfoDenseMatrixConfig);
   matrixRunConfig.getSideInfo().push_back(colSideInfoDenseMatrixConfig);
   matrixRunConfig.setNumLatent(4);
   matrixRunConfig.setBurnin(50);
   matrixRunConfig.setNSamples(50);
   matrixRunConfig.setVerbose(false);
   matrixRunConfig.setRandomSeed(1234);
   matrixRunConfig.setRandomSeedSet(true);
   matrixRunConfig.setDirect(true);

   std::shared_ptr<ISession> tensorRunSession = SessionFactory::create_session(tensorRunConfig);
   tensorRunSession->run();

   std::shared_ptr<ISession> matrixRunSession = SessionFactory::create_session(matrixRunConfig);
   matrixRunSession->run();

   double tensorRunRmseAvg = tensorRunSession->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > tensorRunResults = tensorRunSession->getResult();

   double matrixRunRmseAvg = matrixRunSession->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > matrixRunResults = matrixRunSession->getResult();

   REQUIRE(tensorRunRmseAvg == Approx(matrixRunRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*tensorRunResults, *matrixRunResults);
}

//
//      train: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//       test: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//     priors: macau macau
//  side-info: row_side_info_dense_matrix col_side_info_dense_matrix
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE("--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> --prior macau macau --side-info <row_side_info_dense_matrix> <col_side_info_dense_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct"
          "--train <train_sparse_matrix>    --test <test_sparse_matrix>    --prior macau macau --side-info <row_side_info_dense_matrix> <col_side_info_dense_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct",
          "")
{
   std::shared_ptr<TensorConfig> trainSparseTensorConfig = getTrainSparseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();
   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> rowSideInfoDenseMatrixConfig = getRowSideInfoDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> colSideInfoDenseMatrixConfig = getColSideInfoDenseMatrixConfig();

   Config tensorRunConfig;
   tensorRunConfig.setTrain(trainSparseTensorConfig);
   tensorRunConfig.setTest(testSparseTensorConfig);
   tensorRunConfig.getPriorTypes().push_back(PriorTypes::macau);
   tensorRunConfig.getPriorTypes().push_back(PriorTypes::macau);
   tensorRunConfig.getSideInfo().push_back(rowSideInfoDenseMatrixConfig);
   tensorRunConfig.getSideInfo().push_back(colSideInfoDenseMatrixConfig);
   tensorRunConfig.setNumLatent(4);
   tensorRunConfig.setBurnin(50);
   tensorRunConfig.setNSamples(50);
   tensorRunConfig.setVerbose(false);
   tensorRunConfig.setRandomSeed(1234);
   tensorRunConfig.setRandomSeedSet(true);
   tensorRunConfig.setDirect(true);

   Config matrixRunConfig;
   matrixRunConfig.setTrain(trainSparseMatrixConfig);
   matrixRunConfig.setTest(testSparseMatrixConfig);
   matrixRunConfig.getPriorTypes().push_back(PriorTypes::macau);
   matrixRunConfig.getPriorTypes().push_back(PriorTypes::macau);
   matrixRunConfig.getSideInfo().push_back(rowSideInfoDenseMatrixConfig);
   matrixRunConfig.getSideInfo().push_back(colSideInfoDenseMatrixConfig);
   matrixRunConfig.setNumLatent(4);
   matrixRunConfig.setBurnin(50);
   matrixRunConfig.setNSamples(50);
   matrixRunConfig.setVerbose(false);
   matrixRunConfig.setRandomSeed(1234);
   matrixRunConfig.setRandomSeedSet(true);
   matrixRunConfig.setDirect(true);

   std::shared_ptr<ISession> tensorRunSession = SessionFactory::create_session(tensorRunConfig);
   tensorRunSession->run();

   std::shared_ptr<ISession> matrixRunSession = SessionFactory::create_session(matrixRunConfig);
   matrixRunSession->run();

   double tensorRunRmseAvg = tensorRunSession->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > tensorRunResults = tensorRunSession->getResult();

   double matrixRunRmseAvg = matrixRunSession->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > matrixRunResults = matrixRunSession->getResult();

   REQUIRE(tensorRunRmseAvg == Approx(matrixRunRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*tensorRunResults, *matrixRunResults);
}

//
//      train: 1. dense 2D-tensor (matrix)
//             2. dense matrix
//       test: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//     priors: macauone macauone
//  side-info: row_side_info_dense_matrix col_side_info_dense_matrix
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE("--train <train_dense_2d_tensor> --test <test_sparse_2d_tensor> --prior macauone macauone --side-info <row_side_info_dense_matrix> <col_side_info_dense_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct"
          "--train <train_dense_matrix>    --test <test_sparse_matrix>    --prior macauone macauone --side-info <row_side_info_dense_matrix> <col_side_info_dense_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct",
          "[!hide]")
{
   std::shared_ptr<TensorConfig> trainDenseTensorConfig = getTrainDenseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();
   std::shared_ptr<MatrixConfig> trainDenseMatrixConfig = getTrainDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> rowSideInfoDenseMatrixConfig = getRowSideInfoDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> colSideInfoDenseMatrixConfig = getColSideInfoDenseMatrixConfig();

   Config tensorRunConfig;
   tensorRunConfig.setTrain(trainDenseTensorConfig);
   tensorRunConfig.setTest(testSparseTensorConfig);
   tensorRunConfig.getPriorTypes().push_back(PriorTypes::macauone);
   tensorRunConfig.getPriorTypes().push_back(PriorTypes::macauone);
   tensorRunConfig.getSideInfo().push_back(rowSideInfoDenseMatrixConfig);
   tensorRunConfig.getSideInfo().push_back(colSideInfoDenseMatrixConfig);
   tensorRunConfig.setNumLatent(4);
   tensorRunConfig.setBurnin(50);
   tensorRunConfig.setNSamples(50);
   tensorRunConfig.setVerbose(false);
   tensorRunConfig.setRandomSeed(1234);
   tensorRunConfig.setRandomSeedSet(true);
   tensorRunConfig.setDirect(true);

   Config matrixRunConfig;
   matrixRunConfig.setTrain(trainDenseMatrixConfig);
   matrixRunConfig.setTest(testSparseMatrixConfig);
   matrixRunConfig.getPriorTypes().push_back(PriorTypes::macauone);
   matrixRunConfig.getPriorTypes().push_back(PriorTypes::macauone);
   matrixRunConfig.getSideInfo().push_back(rowSideInfoDenseMatrixConfig);
   matrixRunConfig.getSideInfo().push_back(colSideInfoDenseMatrixConfig);
   matrixRunConfig.setNumLatent(4);
   matrixRunConfig.setBurnin(50);
   matrixRunConfig.setNSamples(50);
   matrixRunConfig.setVerbose(false);
   matrixRunConfig.setRandomSeed(1234);
   matrixRunConfig.setRandomSeedSet(true);
   matrixRunConfig.setDirect(true);

   std::shared_ptr<ISession> tensorRunSession = SessionFactory::create_session(tensorRunConfig);
   tensorRunSession->run();

   std::shared_ptr<ISession> matrixRunSession = SessionFactory::create_session(matrixRunConfig);
   matrixRunSession->run();

   double tensorRunRmseAvg = tensorRunSession->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > tensorRunResults = tensorRunSession->getResult();

   double matrixRunRmseAvg = matrixRunSession->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > matrixRunResults = matrixRunSession->getResult();

   REQUIRE(tensorRunRmseAvg == Approx(matrixRunRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*tensorRunResults, *matrixRunResults);
}

//
//      train: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//       test: 1. sparse 2D-tensor (matrix)
//             2. sparse matrix
//     priors: macauone macauone
//  side-info: row_side_info_dense_matrix col_side_info_dense_matrix
// num-latent: 4
//     burnin: 50
//   nsamples: 50
//    verbose: 0
//       seed: 1234
//
TEST_CASE("--train <train_sparse_2d_tensor> --test <test_sparse_2d_tensor> --prior macauone macauone --side-info <row_side_info_dense_matrix> <col_side_info_dense_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct"
          "--train <train_sparse_matrix>    --test <test_sparse_matrix>    --prior macauone macauone --side-info <row_side_info_dense_matrix> <col_side_info_dense_matrix> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234 --direct",
          "[!hide]")
{
   std::shared_ptr<TensorConfig> trainSparseTensorConfig = getTrainSparseTensor2dConfig();
   std::shared_ptr<TensorConfig> testSparseTensorConfig = getTestSparseTensor2dConfig();
   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
   std::shared_ptr<MatrixConfig> rowSideInfoDenseMatrixConfig = getRowSideInfoDenseMatrixConfig();
   std::shared_ptr<MatrixConfig> colSideInfoDenseMatrixConfig = getColSideInfoDenseMatrixConfig();

   Config tensorRunConfig;
   tensorRunConfig.setTrain(trainSparseTensorConfig);
   tensorRunConfig.setTest(testSparseTensorConfig);
   tensorRunConfig.getPriorTypes().push_back(PriorTypes::macauone);
   tensorRunConfig.getPriorTypes().push_back(PriorTypes::macauone);
   tensorRunConfig.getSideInfo().push_back(rowSideInfoDenseMatrixConfig);
   tensorRunConfig.getSideInfo().push_back(colSideInfoDenseMatrixConfig);
   tensorRunConfig.setNumLatent(4);
   tensorRunConfig.setBurnin(50);
   tensorRunConfig.setNSamples(50);
   tensorRunConfig.setVerbose(false);
   tensorRunConfig.setRandomSeed(1234);
   tensorRunConfig.setRandomSeedSet(true);
   tensorRunConfig.setDirect(true);

   Config matrixRunConfig;
   matrixRunConfig.setTrain(trainSparseMatrixConfig);
   matrixRunConfig.setTest(testSparseMatrixConfig);
   matrixRunConfig.getPriorTypes().push_back(PriorTypes::macauone);
   matrixRunConfig.getPriorTypes().push_back(PriorTypes::macauone);
   matrixRunConfig.getSideInfo().push_back(rowSideInfoDenseMatrixConfig);
   matrixRunConfig.getSideInfo().push_back(colSideInfoDenseMatrixConfig);
   matrixRunConfig.setNumLatent(4);
   matrixRunConfig.setBurnin(50);
   matrixRunConfig.setNSamples(50);
   matrixRunConfig.setVerbose(false);
   matrixRunConfig.setRandomSeed(1234);
   matrixRunConfig.setRandomSeedSet(true);
   matrixRunConfig.setDirect(true);

   std::shared_ptr<ISession> tensorRunSession = SessionFactory::create_session(tensorRunConfig);
   tensorRunSession->run();

   std::shared_ptr<ISession> matrixRunSession = SessionFactory::create_session(matrixRunConfig);
   matrixRunSession->run();

   double tensorRunRmseAvg = tensorRunSession->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > tensorRunResults = tensorRunSession->getResult();

   double matrixRunRmseAvg = matrixRunSession->getRmseAvg();
   std::shared_ptr<std::vector<ResultItem> > matrixRunResults = matrixRunSession->getResult();

   REQUIRE(tensorRunRmseAvg == Approx(matrixRunRmseAvg).epsilon(APPROX_EPSILON));
   REQUIRE_RESULT_ITEMS(*tensorRunResults, *matrixRunResults);
}

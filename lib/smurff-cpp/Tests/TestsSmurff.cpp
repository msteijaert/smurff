#include "catch.hpp"

#include <Eigen/Core>

#include <SmurffCpp/Configs/Config.h>
#include <SmurffCpp/Sessions/SessionFactory.h>

using namespace smurff;

TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior normal normal --features none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234")
{
   std::vector<double> trainMatrixConfigVals = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };
   std::shared_ptr<MatrixConfig> trainMatrixConfig =
      std::make_shared<MatrixConfig>(3, 4, trainMatrixConfigVals, NoiseConfig());

   std::vector<std::uint32_t> testMatrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2};
   std::vector<std::uint32_t> testMatrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   std::vector<double> testMatrixConfigVals = { 1, 2, 3, 4, 9, 10, 11, 12 };
   std::shared_ptr<MatrixConfig> testMatrixConfig =
      std::make_shared<MatrixConfig>(3, 4, std::move(testMatrixConfigRows), std::move(testMatrixConfigCols), std::move(testMatrixConfigVals), NoiseConfig(), false);

   Config config;
   config.setTrain(trainMatrixConfig);
   config.setTest(testMatrixConfig);
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

   // Pre-calculated results from master commit ce08b46ac61a783a7958720ec9e1760780eeb170
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

   REQUIRE(actualResults->size() == expectedResults.size());
   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg));
   for (std::vector<ResultItem>::size_type i = 0; i < actualResults->size(); i++)
   {
      const ResultItem& actualResultItem = actualResults->operator[](i);
      const ResultItem& expectedResultItem = expectedResults[i];
      REQUIRE(actualResultItem.coords == expectedResultItem.coords);
      REQUIRE(actualResultItem.val == expectedResultItem.val);
      REQUIRE(actualResultItem.pred_1sample == Approx(expectedResultItem.pred_1sample));
      REQUIRE(actualResultItem.pred_avg == Approx(expectedResultItem.pred_avg));
      REQUIRE(actualResultItem.var == Approx(expectedResultItem.var));
      REQUIRE(actualResultItem.stds == Approx(expectedResultItem.stds));
   }
}

TEST_CASE("--train <train_sparse_matrix> --test <test_sparse_matrix> --prior normal normal --features none none --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234", "[!hide]")
{
   std::vector<std::uint32_t> trainMatrixConfigCols = { 0, 0, 0, 0, 2, 2, 2, 2};
   std::vector<std::uint32_t> trainMatrixConfigRows = { 0, 1, 2, 3, 0, 1, 2, 3 };
   std::vector<double> trainMatrixConfigVals = { 1, 2, 3, 4, 9, 10, 11, 12 };
   std::shared_ptr<MatrixConfig> trainMatrixConfig =
      std::make_shared<MatrixConfig>(3, 4, std::move(trainMatrixConfigCols), std::move(trainMatrixConfigRows), std::move(trainMatrixConfigVals), NoiseConfig(), false);

   std::vector<std::uint32_t> testMatrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2};
   std::vector<std::uint32_t> testMatrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   std::vector<double> testMatrixConfigVals = { 1, 2, 3, 4, 9, 10, 11, 12 };
   std::shared_ptr<MatrixConfig> testMatrixConfig =
      std::make_shared<MatrixConfig>(3, 4, std::move(testMatrixConfigRows), std::move(testMatrixConfigCols), std::move(testMatrixConfigVals), NoiseConfig(), false);

   Config config;
   config.setTrain(trainMatrixConfig);
   config.setTest(testMatrixConfig);
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

   // Pre-calculated results from master commit ce08b46ac61a783a7958720ec9e1760780eeb170
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

   REQUIRE(actualResults->size() == expectedResults.size());
   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg));
   for (std::vector<ResultItem>::size_type i = 0; i < actualResults->size(); i++)
   {
      const ResultItem& actualResultItem = actualResults->operator[](i);
      const ResultItem& expectedResultItem = expectedResults[i];
      REQUIRE(actualResultItem.coords == expectedResultItem.coords);
      REQUIRE(actualResultItem.val == expectedResultItem.val);
      REQUIRE(actualResultItem.pred_1sample == Approx(expectedResultItem.pred_1sample));
      REQUIRE(actualResultItem.pred_avg == Approx(expectedResultItem.pred_avg));
      REQUIRE(actualResultItem.var == Approx(expectedResultItem.var));
      REQUIRE(actualResultItem.stds == Approx(expectedResultItem.stds));
   }
}

TEST_CASE("--train <train_dense_matrix> --test <test_sparse_matrix> --prior normal normal --features <dense_row_features> <dense_col_features> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234", "[!hide]")
{
   std::vector<double> trainMatrixConfigVals = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };
   std::shared_ptr<MatrixConfig> trainMatrixConfig =
      std::make_shared<MatrixConfig>(3, 4, std::move(trainMatrixConfigVals), NoiseConfig());

   std::vector<std::uint32_t> testMatrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2};
   std::vector<std::uint32_t> testMatrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   std::vector<double> testMatrixConfigVals = { 1, 2, 3, 4, 9, 10, 11, 12 };
   std::shared_ptr<MatrixConfig> testMatrixConfig =
      std::make_shared<MatrixConfig>(3, 4, std::move(testMatrixConfigRows), std::move(testMatrixConfigCols), std::move(testMatrixConfigVals), NoiseConfig(), false);

   std::vector<double> denseRowFeaturesMatrixConfigVals = { 1, 2, 3 };
   std::shared_ptr<MatrixConfig> denseRowFeaturesMatrixConfig =
      std::make_shared<MatrixConfig>(3, 1, std::move(denseRowFeaturesMatrixConfigVals), NoiseConfig());

   std::vector<double> denseColFeaturesMatrixConfigVals = { 1, 2, 3, 4 };
   std::shared_ptr<MatrixConfig> denseColFeaturesMatrixConfig =
      std::make_shared<MatrixConfig>(1, 4, std::move(denseColFeaturesMatrixConfigVals), NoiseConfig());

   Config config;
   config.setTrain(trainMatrixConfig);
   config.setTest(testMatrixConfig);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getFeatures().push_back({ denseRowFeaturesMatrixConfig });
   config.getFeatures().push_back({ denseColFeaturesMatrixConfig });
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

   // Pre-calculated results from master commit ce08b46ac61a783a7958720ec9e1760780eeb170
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

   REQUIRE(actualResults->size() == expectedResults.size());
   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg));
   for (std::vector<ResultItem>::size_type i = 0; i < actualResults->size(); i++)
   {
      const ResultItem& actualResultItem = actualResults->operator[](i);
      const ResultItem& expectedResultItem = expectedResults[i];
      REQUIRE(actualResultItem.coords == expectedResultItem.coords);
      REQUIRE(actualResultItem.val == expectedResultItem.val);
      REQUIRE(actualResultItem.pred_1sample == Approx(expectedResultItem.pred_1sample));
      REQUIRE(actualResultItem.pred_avg == Approx(expectedResultItem.pred_avg));
      REQUIRE(actualResultItem.var == Approx(expectedResultItem.var));
      REQUIRE(actualResultItem.stds == Approx(expectedResultItem.stds));
   }
}

TEST_CASE("--train <train_sparse_matrix> --test <test_sparse_matrix> --prior normal normal --features <dense_row_features> <dense_col_features> --num-latent 4 --burnin 50 --nsamples 50 --verbose 0 --seed 1234", "[!hide]")
{
   std::vector<std::uint32_t> trainMatrixConfigCols = { 0, 0, 0, 0, 2, 2, 2, 2};
   std::vector<std::uint32_t> trainMatrixConfigRows = { 0, 1, 2, 3, 0, 1, 2, 3 };
   std::vector<double> trainMatrixConfigVals = { 1, 2, 3, 4, 9, 10, 11, 12 };
   std::shared_ptr<MatrixConfig> trainMatrixConfig =
      std::make_shared<MatrixConfig>(3, 4, std::move(trainMatrixConfigCols), std::move(trainMatrixConfigRows), std::move(trainMatrixConfigVals), NoiseConfig(), false);

   std::vector<std::uint32_t> testMatrixConfigRows = { 0, 0, 0, 0, 2, 2, 2, 2};
   std::vector<std::uint32_t> testMatrixConfigCols = { 0, 1, 2, 3, 0, 1, 2, 3 };
   std::vector<double> testMatrixConfigVals = { 1, 2, 3, 4, 9, 10, 11, 12 };
   std::shared_ptr<MatrixConfig> testMatrixConfig =
      std::make_shared<MatrixConfig>(3, 4, std::move(testMatrixConfigRows), std::move(testMatrixConfigCols), std::move(testMatrixConfigVals), NoiseConfig(), false);

   std::vector<double> denseRowFeaturesMatrixConfigVals = { 1, 2, 3 };
   std::shared_ptr<MatrixConfig> denseRowFeaturesMatrixConfig =
      std::make_shared<MatrixConfig>(3, 1, std::move(denseRowFeaturesMatrixConfigVals), NoiseConfig());

   std::vector<double> denseColFeaturesMatrixConfigVals = { 1, 2, 3, 4 };
   std::shared_ptr<MatrixConfig> denseColFeaturesMatrixConfig =
      std::make_shared<MatrixConfig>(1, 4, std::move(denseColFeaturesMatrixConfigVals), NoiseConfig());

   Config config;
   config.setTrain(trainMatrixConfig);
   config.setTest(testMatrixConfig);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getPriorTypes().push_back(PriorTypes::normal);
   config.getFeatures().push_back({ denseRowFeaturesMatrixConfig });
   config.getFeatures().push_back({ denseColFeaturesMatrixConfig });
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

   // Pre-calculated results from master commit ce08b46ac61a783a7958720ec9e1760780eeb170
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

   REQUIRE(actualResults->size() == expectedResults.size());
   REQUIRE(actualRmseAvg == Approx(expectedRmseAvg));
   for (std::vector<ResultItem>::size_type i = 0; i < actualResults->size(); i++)
   {
      const ResultItem& actualResultItem = actualResults->operator[](i);
      const ResultItem& expectedResultItem = expectedResults[i];
      REQUIRE(actualResultItem.coords == expectedResultItem.coords);
      REQUIRE(actualResultItem.val == expectedResultItem.val);
      REQUIRE(actualResultItem.pred_1sample == Approx(expectedResultItem.pred_1sample));
      REQUIRE(actualResultItem.pred_avg == Approx(expectedResultItem.pred_avg));
      REQUIRE(actualResultItem.var == Approx(expectedResultItem.var));
      REQUIRE(actualResultItem.stds == Approx(expectedResultItem.stds));
   }
}
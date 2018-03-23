#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <chrono>
#include <memory>
#include <cmath>

#include <SmurffCpp/DataMatrices/Data.h>
#include <SmurffCpp/ConstVMatrixIterator.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <SmurffCpp/Model.h>
#include <SmurffCpp/result.h>

#include <SmurffCpp/Utils/Error.h>
#include <SmurffCpp/Utils/StepFile.h>
#include <SmurffCpp/Utils/StringUtils.h>
#include <SmurffCpp/Utils/IniUtils.h>

#include <SmurffCpp/IO/GenericIO.h>

#define RMSE_AVG_TAG "rmse_avg"
#define RMSE_1SAMPLE_TAG "rmse_1sample"
#define AUC_AVG_TAG "auc_avg"
#define AUC_1SAMPLE_TAG "auc_1sample"
#define SAMPLE_ITER_TAG "sample_iter"
#define BURNIN_ITER_TAG "burnin_iter"

using namespace std;
using namespace Eigen;

using namespace smurff;

//Y - test sparse matrix
void Result::set(std::shared_ptr<TensorConfig> Y)
{
   if (!Y)
   {
      THROWERROR("test data is not initialized");
   }

   if(Y->isDense())
   {
      THROWERROR("test data should be sparse");
   }

   std::shared_ptr<std::vector<std::uint32_t> > columnsPtr = Y->getColumnsPtr();
   std::shared_ptr<std::vector<double> > valuesPtr = Y->getValuesPtr();

   std::vector<int> coords(Y->getNModes());
   m_predictions = std::make_shared<std::vector<ResultItem>>();
   for(std::uint64_t i = 0; i < Y->getNNZ(); i++)
   {
      for(std::uint64_t m = 0; m < Y->getNModes(); m++)
         coords[m] = static_cast<int>(columnsPtr->operator[](Y->getNNZ() * m + i));

      m_predictions->push_back({smurff::PVec<>(coords), valuesPtr->operator[](i)});
   }

   m_dims = Y->getDims();

   init();
}

void Result::init()
{
   total_pos = 0;
   if (classify)
   {
      if (m_predictions && !m_predictions->empty())
      {
         for(std::vector<ResultItem>::const_iterator it = m_predictions->begin(); it != m_predictions->end(); it++)
         {
            int is_positive = it->val > threshold;
            total_pos += is_positive;
         }
      }
   }
}

//--- output model to files
void Result::save(std::shared_ptr<const StepFile> sf) const
{
   savePred(sf);
   savePredState(sf);
}

void Result::savePred(std::shared_ptr<const StepFile> sf) const
{
   if (isEmpty())
      return;

   std::string fname_pred = sf->getPredFileName();
   std::ofstream predfile;
   predfile.open(fname_pred);

   for (std::size_t d = 0; d < m_dims.size(); d++)
      predfile << "coord" << d << ",";

   predfile << "y,pred_1samp,pred_avg,var,std" << std::endl;

   for (std::vector<ResultItem>::const_iterator it = m_predictions->begin(); it != m_predictions->end(); it++)
   {
      it->coords.save(predfile)
         << "," << to_string(it->val)
         << "," << to_string(it->pred_1sample)
         << "," << to_string(it->pred_avg)
         << "," << to_string(it->var)
         << "," << to_string(it->stds)
         << std::endl;
   }

   predfile.close();
}

void Result::savePredState(std::shared_ptr<const StepFile> sf) const
{
   if (isEmpty())
      return;

   std::string predStateName = sf->getPredStateFileName();
   std::ofstream predStatefile;
   predStatefile.open(predStateName);

   predStatefile << RMSE_AVG_TAG << " = " << rmse_avg << std::endl;
   predStatefile << RMSE_1SAMPLE_TAG << " = " << rmse_1sample << std::endl;
   predStatefile << AUC_AVG_TAG << " = " << auc_avg << std::endl;
   predStatefile << AUC_1SAMPLE_TAG << " = " << auc_1sample << std::endl;
   predStatefile << SAMPLE_ITER_TAG << " = " << sample_iter << std::endl;
   predStatefile << BURNIN_ITER_TAG << " = " << burnin_iter << std::endl;
   
   predStatefile.close();
}

void Result::restore(std::shared_ptr<const StepFile> sf)
{
   restorePred(sf);
   restoreState(sf);
}

void Result::restorePred(std::shared_ptr<const StepFile> sf)
{
   std::string fname_pred = sf->getPredFileName();

   THROWERROR_FILE_NOT_EXIST(fname_pred);

   //since predictions were set in set method - clear them
   std::size_t oldSize = m_predictions->size();
   m_predictions->clear();

   //open file with predictions
   std::ifstream predFile;
   predFile.open(fname_pred);

   //parse header
   std::string header;
   getline(predFile, header);

   std::vector<std::string> headerTokens;
   smurff::split(header, headerTokens, ',');

   //parse all lines
   std::vector<std::string> tokens;
   std::vector<int> coords;
   std::string line;

   while (getline(predFile, line))
   {
      //split line
      smurff::split(line, tokens, ',');

      //construct coordinates
      coords.clear();

      std::size_t nCoords = m_dims.size();

      for (std::size_t c = 0; c < nCoords; c++)
         coords.push_back(stoi(tokens[c].c_str()));

      //parse other values
      double val = stod(tokens.at(nCoords).c_str());
      double pred_1sample = stod(tokens.at(nCoords + 1).c_str());
      double pred_avg = stod(tokens.at(nCoords + 2).c_str());
      double var = stod(tokens.at(nCoords + 3).c_str());
      double stds = stod(tokens.at(nCoords + 4).c_str());

      //construct result item
      m_predictions->push_back({ smurff::PVec<>(coords), val, pred_1sample, pred_avg, var, stds });
   }

   //just a sanity check, not sure if it is needed
   THROWERROR_ASSERT_MSG(oldSize == m_predictions->size(), "Incorrect predictions size after restore");

   predFile.close();
}

void Result::restoreState(std::shared_ptr<const StepFile> sf)
{
   std::string predStateName = sf->getPredStateFileName();

   INIFile iniReader;
   iniReader.open(predStateName);

   auto value = iniReader.get(std::string(), RMSE_AVG_TAG);
   rmse_avg = stod(value.c_str());
   
   value = iniReader.get(std::string(), RMSE_1SAMPLE_TAG);
   rmse_1sample = stod(value.c_str());
   
   value = iniReader.get(std::string(), AUC_AVG_TAG);
   auc_avg = stod(value.c_str());
   
   value = iniReader.get(std::string(), AUC_1SAMPLE_TAG);
   auc_1sample = stod(value.c_str());
   
   value = iniReader.get(std::string(), SAMPLE_ITER_TAG);
   sample_iter = stoi(value.c_str());
   
   value = iniReader.get(std::string(), BURNIN_ITER_TAG);
   burnin_iter = stoi(value.c_str());
}

//--- update RMSE and AUC

//model - holds samples (U matrices)
void Result::update(std::shared_ptr<const Model> model, bool burnin)
{
   if (!m_predictions || m_predictions->empty())
      return;

   const size_t NNZ = m_predictions->size();

   if (burnin)
   {
      double se_1sample = 0.0;

      #pragma omp parallel for schedule(guided) reduction(+:se_1sample)
      for(size_t k = 0; k < m_predictions->size(); ++k)
      {
         auto &t = m_predictions->operator[](k);
         t.pred_1sample = model->predict(t.coords); //dot product of i'th columns in each U matrix
         se_1sample += std::pow(t.val - t.pred_1sample, 2);
      }

      burnin_iter++;
      rmse_1sample = std::sqrt(se_1sample / NNZ);

      if (classify)
      {
         auc_1sample = calc_auc(*m_predictions, threshold,
               [](const ResultItem &a, const ResultItem &b) { return a.pred_1sample < b.pred_1sample;});
      }
   }
   else
   {
      double se_1sample = 0.0;
      double se_avg = 0.0;

      #pragma omp parallel for schedule(guided) reduction(+:se_1sample, se_avg)
      for(size_t k = 0; k < m_predictions->size(); ++k)
      {
         auto &t = m_predictions->operator[](k);
         const double pred = model->predict(t.coords); //dot product of i'th columns in each U matrix
         se_1sample += std::pow(t.val - pred, 2);

         double delta = pred - t.pred_avg;
         double pred_avg = (t.pred_avg + delta / (sample_iter + 1));
         t.var += delta * (pred - pred_avg);

         const double inorm = 1.0 / sample_iter;
         t.stds = std::sqrt(t.var * inorm);
         t.pred_avg = pred_avg;
         t.pred_1sample = pred;
         se_avg += std::pow(t.val - pred_avg, 2);
      }

      sample_iter++;
      rmse_1sample = std::sqrt(se_1sample / NNZ);
      rmse_avg = std::sqrt(se_avg / NNZ);

      if (classify)
      {
         auc_1sample = calc_auc(*m_predictions, threshold,
               [](const ResultItem &a, const ResultItem &b) { return a.pred_1sample < b.pred_1sample;});

         auc_avg = calc_auc(*m_predictions, threshold,
               [](const ResultItem &a, const ResultItem &b) { return a.pred_avg < b.pred_avg;});
      }
   }
}

std::ostream &Result::info(std::ostream &os, std::string indent)
{
   if (m_predictions && !m_predictions->empty())
   {
      std::uint64_t dtotal = 1;
      for(size_t d = 0; d < m_dims.size(); d++)
         dtotal *= m_dims[d];

      double test_fill_rate = 100. * m_predictions->size() / dtotal;

      os << indent << "Test data: " << m_predictions->size();

      os << " [";
      for(size_t d = 0; d < m_dims.size(); d++)
      {
         if(d == m_dims.size() - 1)
            os << m_dims[d];
         else
            os << m_dims[d] << " x ";
      }
      os << "]";

      os << " (" << test_fill_rate << "%)" << std::endl;

      if (classify)
      {
         double pos = 100. * (double)total_pos / (double)m_predictions->size();
         os << indent << "Binary classification threshold: " << threshold << std::endl;
         os << indent << "  " << pos << "% positives in test data" << std::endl;
      }
   }
   else
   {
      os << indent << "Test data: -" << std::endl;

      if (classify)
      {
         os << indent << "Binary classification threshold: " << threshold << std::endl;
         os << indent << "  " << "-" << "% positives in test data" << std::endl;
      }
   }

   return os;
}

bool Result::isEmpty() const
{
   return !m_predictions || m_predictions->empty();
}
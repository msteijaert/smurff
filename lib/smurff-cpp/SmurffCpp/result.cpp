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

#include <SmurffCpp/IO/GenericIO.h>
#include <SmurffCpp/IO/MatrixIO.h>

#define GLOBAL_TAG "global"
#define RMSE_AVG_TAG "rmse_avg"
#define RMSE_1SAMPLE_TAG "rmse_1sample"
#define AUC_AVG_TAG "auc_avg"
#define AUC_1SAMPLE_TAG "auc_1sample"
#define SAMPLE_ITER_TAG "sample_iter"
#define BURNIN_ITER_TAG "burnin_iter"

using namespace std;

using namespace smurff;

Result::Result() {}

//Y - test sparse matrix
Result::Result(std::shared_ptr<TensorConfig> Y, int nsamples)
    : m_dims(Y->getDims())
{
   if (!Y)
   {
      THROWERROR("test data is not initialized");
   }

   if(Y->isDense())
   {
      THROWERROR("test data should be sparse");
   }

   for(std::uint64_t i = 0; i < Y->getNNZ(); i++)
   {
      const auto p = Y->get(i);
      m_predictions.push_back(ResultItem(p.first, p.second, nsamples));
   }
}

Result::Result(PVec<> lo, PVec<> hi, double value, int nsamples)
    : m_dims(hi - lo)
{

   for(auto it = PVecIterator(lo, hi); !it.done(); ++it)
   {
      m_predictions.push_back(ResultItem(*it, value, nsamples));
   }
}

void Result::init()
{
   total_pos = 0;
   if (classify)
   {
         for (const auto &p : m_predictions)
         {
               int is_positive = p.val > threshold;
               total_pos += is_positive;
         }
   }
}

//--- output model to files
void Result::save(std::shared_ptr<const StepFile> sf) const
{
   savePred(sf);
   savePredState(sf);
}


template<typename Accessor>
std::shared_ptr<const MatrixConfig> Result::toMatrixConfig(const Accessor &acc) const
{
   std::vector<std::uint32_t> rows;
   std::vector<std::uint32_t> cols;
   std::vector<double> values;

   for (const auto &p : m_predictions)
   {
      rows.push_back(p.coords.at(0));
      cols.push_back(p.coords.at(1));
      values.push_back(acc(p));
   }

   return std::make_shared<MatrixConfig>(m_dims.at(0), m_dims.at(1), rows, cols, values, NoiseConfig(), false);
}

void Result::savePred(std::shared_ptr<const StepFile> sf) const
{
   if (isEmpty())
      return;

   std::string fname_pred = sf->makePredFileName();
   std::ofstream predFile;

   if (sf->isBinary())
   {
      predFile.open(fname_pred, std::ios::out | std::ios::binary);
      THROWERROR_ASSERT_MSG(predFile.is_open(), "Error opening file: " + fname_pred);
      predFile.write((const char *)(&m_predictions[0]), m_predictions.size() * sizeof(m_predictions[0]));

      if (m_dims.size() == 2)
      {
         std::string pred_avg_path = sf->makePredAvgFileName();
         auto pred_avg = toMatrixConfig([](const ResultItem &p) { return p.pred_avg; });
         smurff::matrix_io::write_matrix(pred_avg_path, pred_avg);

         std::string pred_var_path = sf->makePredVarFileName();
         auto pred_var = toMatrixConfig([](const ResultItem &p) { return p.var; });
         smurff::matrix_io::write_matrix(pred_var_path, pred_var);
      }
   }
   else
   {

      predFile.open(fname_pred, std::ios::out);
      THROWERROR_ASSERT_MSG(predFile.is_open(), "Error opening file: " + fname_pred);

      for (std::size_t d = 0; d < m_dims.size(); d++)
         predFile << "coord" << d << ",";

      predFile << "y,pred_1samp,pred_avg,var" << std::endl;

      for (std::vector<ResultItem>::const_iterator it = m_predictions.begin(); it != m_predictions.end(); it++)
      {
         it->coords.save(predFile)
             << "," << to_string(it->val)
             << "," << to_string(it->pred_1sample)
             << "," << to_string(it->pred_avg)
             << "," << to_string(it->var)
             << std::endl;
      }
   }

   predFile.close();

}

void Result::savePredState(std::shared_ptr<const StepFile> sf) const
{
   if (isEmpty())
      return;

   std::string predStateName = sf->makePredStateFileName();

   INIFile predStatefile;
   predStatefile.create(predStateName);

   predStatefile.startSection(GLOBAL_TAG);
   predStatefile.appendItem(GLOBAL_TAG, RMSE_AVG_TAG, std::to_string(rmse_avg));
   predStatefile.appendItem(GLOBAL_TAG, RMSE_1SAMPLE_TAG, std::to_string(rmse_1sample));
   predStatefile.appendItem(GLOBAL_TAG, AUC_AVG_TAG, std::to_string(auc_avg));
   predStatefile.appendItem(GLOBAL_TAG, AUC_1SAMPLE_TAG, std::to_string(auc_1sample));
   predStatefile.appendItem(GLOBAL_TAG, SAMPLE_ITER_TAG, std::to_string(sample_iter));
   predStatefile.appendItem(GLOBAL_TAG, BURNIN_ITER_TAG, std::to_string(burnin_iter));
   
   predStatefile.flush();
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

   //open file with predictions
   std::ifstream predFile;

   std::string fname_ext = fname_pred.substr(fname_pred.find_last_of("."));

   if (fname_ext == ".bin") {
      predFile.open(fname_pred, std::ios::in | std::ios::binary);
      THROWERROR_ASSERT_MSG(predFile.is_open(), "Error opening file: " + fname_pred);
      predFile.read((char *)(&(m_predictions)[0]), m_predictions.size() * sizeof((m_predictions)[0]));  
   }
   else if (fname_ext == ".csv")
   {
      //since predictions were set in set method - clear them
      std::size_t oldSize = m_predictions.size();
      m_predictions.clear();

      predFile.open(fname_pred);
      THROWERROR_ASSERT_MSG(predFile.is_open(), "Error opening file: " + fname_pred);

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

         //construct result item
         m_predictions.push_back(ResultItem(smurff::PVec<>(coords), val, pred_1sample, pred_avg, var, sample_iter));
      }

      //just a sanity check, not sure if it is needed
      THROWERROR_ASSERT_MSG(oldSize == m_predictions.size(), "Incorrect predictions size after restore");
   } 
   else 
   {
      THROWERROR("Unknown extension: " + fname_pred);
   }


   predFile.close();
}

void Result::restoreState(std::shared_ptr<const StepFile> sf)
{
   std::string predStateName = sf->getPredStateFileName();

   INIFile iniReader;
   iniReader.open(predStateName);

   auto value = iniReader.get(GLOBAL_TAG, RMSE_AVG_TAG);
   rmse_avg = stod(value.c_str());
   
   value = iniReader.get(GLOBAL_TAG, RMSE_1SAMPLE_TAG);
   rmse_1sample = stod(value.c_str());
   
   value = iniReader.get(GLOBAL_TAG, AUC_AVG_TAG);
   auc_avg = stod(value.c_str());
   
   value = iniReader.get(GLOBAL_TAG, AUC_1SAMPLE_TAG);
   auc_1sample = stod(value.c_str());
   
   value = iniReader.get(GLOBAL_TAG, SAMPLE_ITER_TAG);
   sample_iter = stoi(value.c_str());
   
   value = iniReader.get(GLOBAL_TAG, BURNIN_ITER_TAG);
   burnin_iter = stoi(value.c_str());
}

//--- update RMSE and AUC

//model - holds samples (U matrices)
void Result::update(std::shared_ptr<const Model> model, bool burnin)
{
   if (m_predictions.empty())
      return;

   const size_t NNZ = m_predictions.size();

   if (burnin)
   {
      double se_1sample = 0.0;

      #pragma omp parallel for schedule(guided) reduction(+:se_1sample)
      for(size_t k = 0; k < m_predictions.size(); ++k)
      {
         auto &t = m_predictions.operator[](k);
         t.pred_1sample = model->predict(t.coords); //dot product of i'th columns in each U matrix
         se_1sample += std::pow(t.val - t.pred_1sample, 2);
      }

      burnin_iter++;
      rmse_1sample = std::sqrt(se_1sample / NNZ);

      if (classify)
      {
         auc_1sample = calc_auc(m_predictions, threshold,
               [](const ResultItem &a, const ResultItem &b) { return a.pred_1sample < b.pred_1sample;});
      }
   }
   else
   {
      double se_1sample = 0.0;
      double se_avg = 0.0;

      #pragma omp parallel for schedule(guided) reduction(+:se_1sample, se_avg)
      for(size_t k = 0; k < m_predictions.size(); ++k)
      {
         auto &t = m_predictions.operator[](k);
         const double pred = model->predict(t.coords); //dot product of i'th columns in each U matrix
         t.update(pred);

         se_1sample += std::pow(t.val - pred, 2);
         se_avg += std::pow(t.val - t.pred_avg, 2);
      }

      sample_iter++;
      rmse_1sample = std::sqrt(se_1sample / NNZ);
      rmse_avg = std::sqrt(se_avg / NNZ);

      if (classify)
      {
         auc_1sample = calc_auc(m_predictions, threshold,
               [](const ResultItem &a, const ResultItem &b) { return a.pred_1sample < b.pred_1sample;});

         auc_avg = calc_auc(m_predictions, threshold,
               [](const ResultItem &a, const ResultItem &b) { return a.pred_avg < b.pred_avg;});
      }
   }
}

std::ostream &Result::info(std::ostream &os, std::string indent)
{
   if (!m_predictions.empty())
   {
      std::uint64_t dtotal = 1;
      for(size_t d = 0; d < m_dims.size(); d++)
         dtotal *= m_dims[d];

      double test_fill_rate = 100. * m_predictions.size() / dtotal;

      os << indent << "Test data: " << m_predictions.size();

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
         double pos = 100. * (double)total_pos / (double)m_predictions.size();
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
   return m_predictions.empty();
}

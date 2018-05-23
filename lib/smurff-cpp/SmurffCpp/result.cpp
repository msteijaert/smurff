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
#include <SmurffCpp/IO/TensorIO.h>

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

   m_predictions = std::make_shared<std::vector<ResultItem>>();
   for(std::uint64_t i = 0; i < Y->getNNZ(); i++)
   {
      const auto p = Y->get(i);
      ResultItem r = {p.first, p.second};
      m_predictions->push_back(r);
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

std::shared_ptr<TensorConfig> Result::toSparseTensor() const
{
      // tensor of 1 dimension higher than Ytest
      // in the extra dimension we store val, pred_1sample, pred_avg and var
      std::vector<std::uint64_t> dims = m_dims;
      dims.push_back(ResultItem::size);

      const std::uint64_t num_modes  =  getNModes() + 1;
      const std::uint64_t num_values =  getNNZ() * ResultItem::size;
      const std::uint64_t num_coords =  num_values * num_modes;

      std::vector<std::uint32_t> columns(num_coords);
      std::vector<double>        values(num_values);

      for (std::uint64_t i = 0; i<m_predictions->size(); ++i)
      {
            auto push = [&columns, &values, num_values, i](const PVec<> &coords, double value, int off) {
                  std::uint64_t pos = (i * ResultItem::size) + off;
                  values[pos] = value;

                  for (unsigned i=0; i<coords.size(); ++i)
                  {
                        columns[pos] = coords[i];
                        pos += num_values;
                  }
                  columns[pos] = off;
            };

            const auto &item = (*m_predictions)[i];
            push(item.coords, item.val, 0);
            push(item.coords, item.pred_1sample, 1);
            push(item.coords, item.pred_avg, 2);
            push(item.coords, item.var, 3);
      }

      return std::make_shared<TensorConfig>(dims, columns, values, NoiseConfig(), true);
}

void Result::savePred(std::shared_ptr<const StepFile> sf) const
{
   if (isEmpty())
      return;

   std::string fname_pred = sf->getPredFileName();
   const auto &st = toSparseTensor();
   smurff::tensor_io::write_tensor(fname_pred, st);
}

void Result::savePredState(std::shared_ptr<const StepFile> sf) const
{
   if (isEmpty())
      return;

   std::string predStateName = sf->getPredStateFileName();

   INIFile predStatefile;
   predStatefile.create(predStateName);

   predStatefile.appendItem(std::string(), RMSE_AVG_TAG, std::to_string(rmse_avg));
   predStatefile.appendItem(std::string(), RMSE_1SAMPLE_TAG, std::to_string(rmse_1sample));
   predStatefile.appendItem(std::string(), AUC_AVG_TAG, std::to_string(auc_avg));
   predStatefile.appendItem(std::string(), AUC_1SAMPLE_TAG, std::to_string(auc_1sample));
   predStatefile.appendItem(std::string(), SAMPLE_ITER_TAG, std::to_string(sample_iter));
   predStatefile.appendItem(std::string(), BURNIN_ITER_TAG, std::to_string(burnin_iter));
   
   predStatefile.flush();
}

void Result::restore(std::shared_ptr<const StepFile> sf)
{
   restorePred(sf);
   restoreState(sf);
}

void Result::fromSparseTensor(const std::shared_ptr<TensorConfig> &tc)
{
      const auto &columns = tc->getColumns();
      const auto &values = tc->getValues();
      
      assert(tc->getNNZ() % ResultItem::size == 0);
      assert(ResultItem::size == 4);

      for(std::uint64_t i = 0; i < tc->getNNZ(); i+=ResultItem::size)
      {
            std::vector<int> coords;
            std::uint64_t pos = i;
            
            double val          = values[pos];
            double pred_1sample = values[pos+1];
            double pred_avg     = values[pos+2];
            double var          = values[pos+3];

            for(int j=0; j<tc->getNModes() - 1; ++j) 
            {
                  coords.push_back(columns[pos]);
                  pos += tc->getNNZ();
            }

            assert(columns[pos] == 0);

            m_predictions->push_back({smurff::PVec<>(coords), val, pred_1sample, pred_avg, var});
      }
}

void Result::restorePred(std::shared_ptr<const StepFile> sf)
{
   //since predictions were set in set method - clear them
   std::size_t oldSize = m_predictions->size();
   m_predictions->clear();

   std::string fname_pred = sf->getPredFileName();
   const auto &st = smurff::tensor_io::read_tensor(fname_pred, true);
   fromSparseTensor(st);

   //just a sanity check, not sure if it is needed
   THROWERROR_ASSERT_MSG(oldSize == m_predictions->size(), "Incorrect predictions size after restore");
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

std::uint64_t Result::getNNZ() const
{
   return m_predictions->size();
}

std::uint64_t Result::getNModes() const
{
   return m_dims.size();
}
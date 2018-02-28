#include <SmurffCpp/Utils/StepFile.h>

#include <iostream>
#include <fstream>

#include <SmurffCpp/Model.h>
#include <SmurffCpp/result.h>
#include <SmurffCpp/Priors/ILatentPrior.h>

#include <SmurffCpp/IO/INIReader.h>
#include <SmurffCpp/Utils/Error.h>
#include <SmurffCpp/IO/GenericIO.h>

#define STEP_SAMPLE_PREFIX "-sample-"
#define STEP_BURNIN_PREFIX "-burnin-"
#define STEP_INI_SUFFIX "-step.ini"

#define MODEL_PREFIX "model_"
#define PRIOR_PREFIX "prior_"

#define NUM_MODELS_TAG "num_models"
#define NUM_PRIORS_TAG "num_priors"
#define PRED_TAG "pred"
#define PRED_STATE_TAG "pred_state"

using namespace smurff;

StepFile::StepFile(std::int32_t isample, std::string prefix, std::string extension, bool create, bool burnin)
   : m_isample(isample), m_prefix(prefix), m_extension(extension), m_burnin(burnin)
{
   if (create)
   {
      std::ofstream stepFile;
      stepFile.open(getStepFileName(), std::ios::trunc);
      stepFile.close();
   }
}

StepFile::StepFile(const std::string& path, std::string prefix, std::string extension)
   : m_prefix(prefix), m_extension(extension)
{
   m_isample = tryGetIsampleFromPathInternal(path, STEP_BURNIN_PREFIX, STEP_INI_SUFFIX);

   if (m_isample < 0)
   {
      m_isample = tryGetIsampleFromPathInternal(path, STEP_SAMPLE_PREFIX, STEP_INI_SUFFIX);

      if (m_isample < 0)
      {
         THROWERROR("Invalid step file name");
      }
      else
      {
         m_burnin = false;
      }
   }
   else
   {
      m_burnin = true;
   }
}

std::string StepFile::getStepFileName() const
{
   std::string prefix = getStepPrefix();
   return prefix + STEP_INI_SUFFIX;
}

std::int32_t StepFile::tryGetIsampleFromPathInternal(const std::string& path, const std::string& prefix, const std::string& suffix) const
{
   std::size_t idx0 = path.find(prefix);
   if (idx0 == std::string::npos)
      return -1;

   std::size_t idx1 = path.find(suffix);
   THROWERROR_ASSERT_MSG(idx1 != std::string::npos, "Invalid step file name");

   std::size_t start = idx0 + prefix.length();
   std::string indexStr = path.substr(start, idx1 - start);

   std::int32_t index;
   std::stringstream ss;
   ss << indexStr;
   ss >> index;

   return index;
}

std::string StepFile::getStepPrefix() const
{
   std::string prefix = m_burnin ? STEP_BURNIN_PREFIX : STEP_SAMPLE_PREFIX;
   return m_prefix + prefix + std::to_string(m_isample);
}

std::string StepFile::getModelFileName(std::uint64_t index) const
{
   std::string prefix = getStepPrefix();
   return prefix + "-U" + std::to_string(index) + "-latents" + m_extension;
}

std::string StepFile::getPriorFileName(std::uint32_t mode) const
{
   std::string prefix = getStepPrefix();
   return prefix + "-F" + std::to_string(mode) + "-link" + m_extension;
}

std::string StepFile::getPredFileName() const
{
   std::string prefix = getStepPrefix();
   return prefix + "-predictions.csv";
}

std::string StepFile::getPredStateFileName() const
{
   std::string prefix = getStepPrefix();
   return prefix + "-predictions-state.ini";
}

void StepFile::saveModel(std::shared_ptr<const Model> model) const
{
   model->save(shared_from_this());

   //save models

   std::ofstream stepFile;
   stepFile.open(getStepFileName(), std::ios::app);

   stepFile << "#models" << std::endl;
   stepFile << NUM_MODELS_TAG << " = " << model->nmodes() << std::endl;

   for (std::uint64_t mIndex = 0; mIndex < model->nmodes(); mIndex++)
   {
      std::string path = getModelFileName(mIndex);
      stepFile << MODEL_PREFIX << mIndex << " = " << path << std::endl;
   }

   stepFile.close();
}

void StepFile::savePred(std::shared_ptr<const Result> m_pred) const
{
   m_pred->save(shared_from_this());

   //save predictions

   std::ofstream stepFile;
   stepFile.open(getStepFileName(), std::ios::app);

   stepFile << "#predictions" << std::endl;
   stepFile << PRED_TAG << " = " << getPredFileName() << std::endl;
   stepFile << PRED_STATE_TAG << " = " << getPredStateFileName() << std::endl;

   stepFile.close();
}

void StepFile::savePriors(const std::vector<std::shared_ptr<ILatentPrior> >& priors) const
{
   for (auto &p : priors)
   {
      p->save(shared_from_this());
   }

   //save priors

   std::ofstream stepFile;
   stepFile.open(getStepFileName(), std::ios::app);

   stepFile << "#priors" << std::endl;
   stepFile << NUM_PRIORS_TAG << " = " << priors.size() << std::endl;

   for (std::uint64_t pIndex = 0; pIndex < priors.size(); pIndex++)
   {
      std::string priorPath = getPriorFileName(priors.at(pIndex)->getMode());
      stepFile << PRIOR_PREFIX << pIndex << " = " << priorPath << std::endl;
   }

   stepFile.close();
}

void StepFile::restoreModel(std::shared_ptr<Model> model) const
{
   model->restore(shared_from_this());
}

void StepFile::restorePred(std::shared_ptr<Result> m_pred) const
{
   m_pred->restore(shared_from_this());
}

void StepFile::restorePriors(std::vector<std::shared_ptr<ILatentPrior> >& priors) const
{
   for (auto &p : priors)
   {
      p->restore(shared_from_this());
   }
}

void StepFile::removeModel() const
{
   std::uint64_t index = 0;
   while (true)
   {
      std::string path = getModelFileName(index++);
      if (!generic_io::file_exists(path))
         break;

      std::remove(path.c_str());
   }   
}

void StepFile::removePred() const
{
   std::remove(getPredFileName().c_str());
   std::remove(getPredStateFileName().c_str());
}

void StepFile::removePriors() const
{
   std::uint32_t mode = 0;
   while (true)
   {
      std::string path = getPriorFileName(mode++);
      if (!generic_io::file_exists(path))
         break;

      std::remove(path.c_str());
   }
}

void StepFile::save(std::shared_ptr<const Model> model, std::shared_ptr<const Result> pred, const std::vector<std::shared_ptr<ILatentPrior> >& priors) const
{
   saveModel(model);
   savePred(pred);
   savePriors(priors);
}

void StepFile::restore(std::shared_ptr<Model> model, std::shared_ptr<Result> pred, std::vector<std::shared_ptr<ILatentPrior> >& priors) const
{
   restoreModel(model);
   restorePred(pred);
   restorePriors(priors);
}

void StepFile::remove(bool model, bool pred, bool priors) const
{
   //remove all model files
   if(model)
      removeModel();

   //remove all pred files
   if(pred)
      removePred();

   //remove all prior files
   if(priors)
      removePriors();

   //remove step file itself
   std::remove(getStepFileName().c_str());
}

std::int32_t StepFile::getIsample() const
{
   return m_isample;
}

std::int32_t StepFile::getNSamples() const
{
   std::string name = getStepFileName();
   INIReader reader(name);
   THROWERROR_ASSERT_MSG(reader.ParseError() >= 0, "Can't load '" + name + "'\n");

   std::int32_t num_models = reader.GetInteger("", NUM_MODELS_TAG, 0);
   return num_models;
}

bool StepFile::getBurnin() const
{
   return m_burnin;
}
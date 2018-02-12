#include <SmurffCpp/Utils/StepFile.h>

#include <SmurffCpp/Model.h>
#include <SmurffCpp/result.h>
#include <SmurffCpp/Priors/ILatentPrior.h>

#include <iostream>
#include <fstream>

using namespace smurff;

StepFile::StepFile(int isample, std::string prefix, std::string extension)
   : m_isample(isample), m_prefix(prefix), m_extension(extension)
{
}

std::string StepFile::getStepFileName() const
{
   std::string prefix = getSamplePrefix();
   return prefix + "-step.ini";
}

std::string StepFile::getSamplePrefix() const
{
   return m_prefix + "-sample-" + std::to_string(m_isample);
}

std::string StepFile::getModelFileName(std::uint64_t index) const
{
   std::string prefix = getSamplePrefix();
   return prefix + "-U" + std::to_string(index) + "-latents" + m_extension;
}

std::string StepFile::getPriorFileName(uint32_t mode) const
{
   std::string prefix = getSamplePrefix();
   return prefix + "-F" + std::to_string(mode) + "-link" + m_extension;
}

std::string StepFile::getPredFileName() const
{
   std::string prefix = getSamplePrefix();
   return prefix + "-predictions.csv";
}

void StepFile::saveModel(std::shared_ptr<Model> model) const
{
   model->save(shared_from_this());
}

void StepFile::savePred(std::shared_ptr<Result> m_pred) const
{
   return m_pred->save(shared_from_this());
}

void StepFile::savePriors(std::vector<std::shared_ptr<ILatentPrior> >& priors) const
{
   for (auto &p : priors)
   {
      p->save(shared_from_this());
   }
}

void StepFile::save(std::shared_ptr<Model> model, std::shared_ptr<Result> pred, std::vector<std::shared_ptr<ILatentPrior> >& priors) const
{
   saveModel(model);
   savePred(pred);
   savePriors(priors);

   std::string stepFilePath = getStepFileName();

   std::ofstream stepFile;
   stepFile.open(stepFilePath);

   //save models

   stepFile << "#models" << std::endl;
   stepFile << "num_models = " << model->nmodes() << std::endl;

   for (std::uint64_t mIndex = 0; mIndex < model->nmodes(); mIndex++)
   {
      std::string path = getModelFileName(mIndex);
      stepFile << "model_" << mIndex << " = " << path << std::endl;
   }

   //save predictions

   stepFile << "#predictions" << std::endl;
   stepFile << "pred = " << getPredFileName() << std::endl;
   
   stepFile << "#priors" << std::endl;
   stepFile << "num_priors = " << priors.size() << std::endl;

   //save priors

   std::uint64_t pIndex = 0;
   for (std::uint64_t pIndex = 0; pIndex < priors.size(); pIndex++)
   {
      std::string priorPath = getPriorFileName(priors.at(pIndex)->getMode());
      stepFile << "model_" << pIndex << " = " << priorPath << std::endl;
   }

   stepFile.close();
}
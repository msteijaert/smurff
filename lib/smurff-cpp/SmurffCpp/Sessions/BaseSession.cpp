#include "BaseSession.h"

#include <fstream>

#include <SmurffCpp/Priors/ILatentPrior.h>
#include <SmurffCpp/Model.h>
#include <SmurffCpp/result.h>

#include <SmurffCpp/Utils/Error.h>

using namespace smurff;

BaseSession::BaseSession()
   : m_model(std::make_shared<Model>())
   , m_pred(std::make_shared<Result>())
{
}

void BaseSession::addPrior(std::shared_ptr<ILatentPrior> prior)
{
   prior->setMode(m_priors.size());
   m_priors.push_back(prior);
}

void BaseSession::step()
{
   for(auto &p : m_priors)
      p->sample_latents();
   data()->update(m_model);
}

std::ostream &BaseSession::info(std::ostream &os, std::string indent)
{
   os << indent << name << " {\n";
   os << indent << "  Data: {\n";
   data()->info(os, indent + "    ");
   os << indent << "  }\n";
   os << indent << "  Model: {\n";
   m_model->info(os, indent + "    ");
   os << indent << "  }\n";
   os << indent << "  Priors: {\n";
   for( auto &p : m_priors)
      p->info(os, indent + "    ");
   os << indent << "  }\n";
   os << indent << "  Result: {\n";
   m_pred->info(os, indent + "    ");
   os << indent << "  }\n";
   return os;
}

std::string BaseSession::save(std::string prefix, std::string suffix)
{
   std::vector<std::string> modelPaths;
   m_model->save(prefix, suffix, modelPaths);

   std::string predPath = m_pred->save(prefix);

   std::vector<std::string> priorPaths;
   for(auto &p : m_priors)
   {
      std::string priorPath = p->save(prefix, suffix);
      priorPaths.push_back(priorPath);
   }

   std::string stepFilePath = getRootFileName(prefix + "-step.ini");

   std::ofstream stepFile;
   stepFile.open(stepFilePath);

   stepFile << "#models" << std::endl;
   stepFile << "num_models = " << modelPaths.size() << std::endl;

   std::int32_t mIndex = 0;
   for (auto m : modelPaths)
   {
      stepFile << "model_" << mIndex++ << " = " << m << std::endl;
   }

   stepFile << "#priors" << std::endl;
   stepFile << "num_priors = " << priorPaths.size() << std::endl;

   std::int32_t pIndex = 0;
   for (auto p : priorPaths)
   {
      stepFile << "model_" << pIndex++ << " = " << p << std::endl;
   }

   stepFile << "#predictions" << std::endl;
   stepFile << "prediction = " << predPath << std::endl;

   stepFile.close();
   return stepFilePath;
}

void BaseSession::restore(std::string prefix, std::string suffix)
{
   m_model->restore(prefix, suffix);
   for(auto &p : m_priors)
      p->restore(prefix, suffix);
}

std::shared_ptr<std::vector<ResultItem> > BaseSession::getResult()
{
   return m_pred->m_predictions;
}

double BaseSession::getRmseAvg()
{
   return m_pred->rmse_avg;
}

MatrixConfig BaseSession::getSample(int mode)
{
   THROWERROR_NOTIMPL_MSG("getSample is unimplemented");
}

std::string BaseSession::getRootFileName(std::string prefix) const
{
   return prefix + "-step.ini";
}
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

bool BaseSession::step()
{
   for(auto &p : m_priors)
      p->sample_latents();
   data()->update(m_model);
   return true;
}

std::ostream &BaseSession::info(std::ostream &os, std::string indent)
{
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

void BaseSession::save(std::shared_ptr<StepFile> stepFile) const
{
   stepFile->save(m_model, m_pred, m_priors);
}

void BaseSession::restore(std::shared_ptr<StepFile> stepFile)
{
   stepFile->restore(m_model, m_pred, m_priors);
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
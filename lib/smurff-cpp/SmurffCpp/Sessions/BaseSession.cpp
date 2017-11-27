#include "BaseSession.h"

#include <SmurffCpp/Priors/ILatentPrior.h>
#include <SmurffCpp/Model.h>
#include <SmurffCpp/result.h>

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

void BaseSession::save(std::string prefix, std::string suffix)
{
   m_model->save(prefix, suffix);
   m_pred->save(prefix);
   for(auto &p : m_priors)
      p->save(prefix, suffix);
}

void BaseSession::restore(std::string prefix, std::string suffix)
{
   m_model->restore(prefix, suffix);
   for(auto &p : m_priors)
      p->restore(prefix, suffix);
}

MatrixConfig BaseSession::getResult()
{
   std::vector<std::uint32_t> resultRows;
   std::vector<std::uint32_t> resultCols;
   std::vector<double> resultVals;
   resultRows.reserve(m_pred->predictions.size());
   resultCols.reserve(m_pred->predictions.size());
   resultVals.reserve(m_pred->predictions.size());

   for (const Result::Item& i : m_pred->predictions)
   {
      resultRows.push_back(i.row);
      resultCols.push_back(i.col);
      resultVals.push_back(i.val);
   }

   return MatrixConfig( m_pred->m_nrows
                      , m_pred->m_ncols
                      , std::move(resultRows)
                      , std::move(resultCols)
                      , std::move(resultVals)
                      , NoiseConfig()
                      );
}

MatrixConfig BaseSession::getSample(int mode)
{
   throw std::runtime_error("getSample is unimplemented");
}
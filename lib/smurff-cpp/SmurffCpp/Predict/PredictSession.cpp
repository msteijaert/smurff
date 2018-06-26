#include <memory>

#include <Eigen/Sparse>
#include <Eigen/Core>

#include <SmurffCpp/Utils/counters.h>
#include <SmurffCpp/Utils/RootFile.h>
#include <SmurffCpp/result.h>
#include <SmurffCpp/ResultItem.h>

#include <SmurffCpp/Model.h>

#include <SmurffCpp/Predict/PredictSession.h>

namespace smurff {

PredictSession::PredictSession(std::shared_ptr<RootFile> rf)
    : m_rootfile(rf), m_has_config(false), m_num_latent(-1), m_dims(PVec<>(0))
{
   restore();

}

PredictSession::PredictSession(std::shared_ptr<RootFile> rf, const Config &config)
    : m_rootfile(rf), m_config(config), m_has_config(true), m_num_latent(-1), m_dims(PVec<>(0))
{
   restore();

}
PredictSession::PredictSession(const Config &config)
    : m_config(config), m_has_config(true), m_num_latent(-1), m_dims(PVec<>(0))
{
   THROWERROR_ASSERT(config.getRootName().size())
   m_rootfile = std::make_shared<RootFile>(config.getRootName());
   restore();
}

void PredictSession::run()
{
   THROWERROR_ASSERT(m_has_config);
   init();
   while (step()) ;
}

void PredictSession::init()
{
   THROWERROR_ASSERT(m_config.getTest());
   m_result = std::make_shared<Result>(m_config.getTest());
   m_pos = m_stepdata.begin();
}

bool PredictSession::step() 
{
   double start = tick();
   m_result->update(m_pos->second.m_model, false);
   double stop = tick();
   m_secs_per_iter = stop - start;

   std::cout << getStatus()->asString() << std::endl;

   m_pos++;
   return m_pos != m_stepdata.end();
}

std::shared_ptr<StatusItem> PredictSession::getStatus() const
{
   std::shared_ptr<StatusItem> ret = std::make_shared<StatusItem>();
   ret->phase = "Predict";
   ret->iter = m_result->sample_iter;
   ret->phase_iter = m_stepdata.size();

   ret->train_rmse = NAN;

   ret->rmse_avg = m_result->rmse_avg;
   ret->rmse_1sample = m_result->rmse_1sample;

   ret->auc_avg = m_result->auc_avg;
   ret->auc_1sample = m_result->auc_1sample;

   ret->elapsed_iter = m_secs_per_iter;

   auto model = m_pos->second.m_model;
   for (int i = 0; i < model->nmodes(); ++i)
   {
      ret->model_norms.push_back(model->U(i).norm());
    }

   return ret;
}

std::shared_ptr<Result> PredictSession::getResult() const
{
   return m_result;
}

std::ostream& PredictSession::info(std::ostream &os, std::string indent) const
{
   os << indent << "PredictSession {\n";
   os << indent << "  root-file  : " << m_rootfile->getRootFileName() << "\n";
   os << indent << "  num-samples: " << getNumSteps() << "\n";
   os << indent << "  num-latent : " << getNumLatent() << "\n";
   os << indent << "  model size : " << getModelDims() << "\n";
   os << indent << "}\n";
   return os;
}

void PredictSession::restore()
{
   auto stepfiles = m_rootfile->openSampleStepFiles(); 
   for (const auto &sf : stepfiles)
   {
      int sample_number = sf->getIsample();
      auto model = sf->restoreModel();
      StepData step{model};
      if (m_num_latent <= 0) {
         m_num_latent = model->nlatent();
         m_dims = model->getDims();
      }
      else
      {
         THROWERROR_ASSERT(m_num_latent == model->nlatent());
         THROWERROR_ASSERT(m_dims == model->getDims());
      }

      m_stepdata.insert(std::make_pair(sample_number, step));
   }

}

// predict one element
ResultItem PredictSession::predict(PVec<> pos, const StepFile &sf) {
   ResultItem ret{pos};
   predict(ret, sf);
   return ret;
}

// predict one element
void PredictSession::predict(ResultItem &res, const StepFile &sf) {
   auto model = sf.restoreModel();
   auto pred = model->predict(res.coords);
   res.update(pred);
}

// predict one element
void PredictSession::predict(ResultItem &res) {
   auto stepfiles = m_rootfile->openSampleStepFiles();

   for(const auto &sf : stepfiles)
      predict(res, *sf);
}

ResultItem PredictSession::predict(PVec<> pos) {
   ResultItem ret{pos};
   predict(ret);
   return ret;
}

// predict all elements in Ytest
std::shared_ptr<Result> PredictSession::predict(std::shared_ptr<TensorConfig> Y)
{
   auto res = std::make_shared<Result>(Y);

   for (const auto &s : m_stepdata)
   {
      res->update(s.second.m_model, false);
   }

   return res;
}

// predict element or elements based on sideinfo
template<class Feat>
std::shared_ptr<Result> predict(std::vector<std::shared_ptr<Feat>>) {


}

} // end namespace smurff

#include <memory>

#include <Eigen/Sparse>
#include <Eigen/Core>

#include <SmurffCpp/Utils/RootFile.h>
#include <SmurffCpp/result.h>
#include <SmurffCpp/ResultItem.h>

#include <SmurffCpp/Model.h>

#include <SmurffCpp/Predict/PredictSession.h>

namespace smurff {

PredictSession::PredictSession(std::shared_ptr<RootFile> rf)
    : m_rootfile(rf), m_num_latent(-1), m_dims(PVec<>(0))
{
   restore();
}

std::ostream& PredictSession::info(std::ostream &os, std::string indent) const
{
   os << indent << "PredictSession {\n";
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

   info(std::cout, "");
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

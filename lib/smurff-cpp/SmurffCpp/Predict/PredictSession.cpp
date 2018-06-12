#include <memory>

#include <Eigen/Sparse>
#include <Eigen/Core>

#include <SmurffCpp/Utils/RootFile.h>
#include <SmurffCpp/result.h>
#include <SmurffCpp/ResultItem.h>

#include <SmurffCpp/Model.h>

#include <SmurffCpp/Predict/PredictSession.h>

namespace smurff {

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
   auto stepfiles = m_root_file->openSampleStepFiles();

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
   auto res = std::make_shared<Result>();
   res->set(Y);

   auto stepfiles = m_root_file->openSampleStepFiles();

   for (const auto &sf : stepfiles)
   {
      auto model = sf->restoreModel();
      res->update(model, false);
   }

   return res;
}

// predict element or elements based on sideinfo
template<class Feat>
std::shared_ptr<Result> predict(std::vector<std::shared_ptr<Feat>>) {


}

} // end namespace smurff

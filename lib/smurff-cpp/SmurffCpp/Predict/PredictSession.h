#pragma once

#include <memory>

#include <Eigen/Sparse>
#include <Eigen/Core>

#include <SmurffCpp/Utils/PVec.hpp>

namespace smurff {

class RootFile;
class Result;
struct ResultItem;

class PredictSession 
{
private:
    std::shared_ptr<RootFile> m_root_file;

public:
    PredictSession(std::shared_ptr<RootFile> rf)
        : m_root_file(rf) {}

    // predict one element - based on position only
    ResultItem predict(PVec<> Ytest);
    ResultItem predict(PVec<> Ytest, const StepFile &sf);

    // predict one element - based on ResultItem
    void predict(ResultItem &);
    void predict(ResultItem &, const StepFile &sf);

    // predict all elements in Ytest
    std::shared_ptr<Result> predict(std::shared_ptr<TensorConfig> Y);
    void predict(Result &, const StepFile &);

    // predict element or elements based on sideinfo
    template<class Feat>
    std::shared_ptr<Result> predict(std::vector<std::shared_ptr<Feat>>);
    template<class Feat>
    std::shared_ptr<Result> predict(std::vector<std::shared_ptr<Feat>>, int sample);
    
};

}

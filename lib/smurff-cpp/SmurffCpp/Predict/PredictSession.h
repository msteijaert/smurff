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
    std::shared_ptr<RootFile> m_rootfile;

    struct StepData {
        std::shared_ptr<Model> m_model;
        std::vector<std::shared_ptr<Eigen::MatrixXd>> m_link_matrices;
    };

    std::map<int, StepData> m_stepdata;

    int m_num_latent;
    PVec<> m_dims;

public:
    int getNumSteps() const
    {
        return m_stepdata.size();
    }

    PVec<> getModelDims() const
    {
       return m_dims;
    }

    int getNumLatent() const
    {
        return m_num_latent;
    }


public:
    PredictSession(std::shared_ptr<RootFile> rf);

    std::ostream& info(std::ostream &os, std::string indent) const;

    // predict one element - based on position only
    ResultItem predict(PVec<> Ytest);
    
    void restore();
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

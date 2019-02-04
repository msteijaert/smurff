#pragma once

#include <memory>

#include <Eigen/Sparse>
#include <Eigen/Core>

#include <SmurffCpp/Utils/PVec.hpp>
#include <SmurffCpp/IO/MatrixIO.h>
#include <SmurffCpp/Sessions/ISession.h>
#include <SmurffCpp/Model.h>


namespace smurff {

class RootFile;
class Result;
struct ResultItem;

class PredictSession : public ISession
{
private:
    std::shared_ptr<RootFile> m_model_rootfile;
    std::shared_ptr<RootFile> m_pred_rootfile;
    Config m_config;
    bool m_has_config;

    std::shared_ptr<Result> m_result;
    std::vector<std::shared_ptr<StepFile>>::reverse_iterator m_pos;

    double m_secs_per_iter;
    double m_secs_total;
    int m_iter;

    std::vector<std::shared_ptr<StepFile>> m_stepfiles;

    int m_num_latent;
    PVec<> m_dims;
    bool m_is_init;

private:
    std::shared_ptr<Model> restoreModel(const std::shared_ptr<StepFile> &, int skip_mode = -1);
    std::shared_ptr<Model> restoreModel(int i);

public:
    int    getNumSteps()  const { return m_stepfiles.size(); } 
    int    getNumLatent() const { return m_num_latent; } 
    PVec<> getModelDims() const { return m_dims; } 

public:
    // ISession interface 
    void run() override;
    bool step() override;
    void init() override;

    std::shared_ptr<StatusItem> getStatus() const override;
    std::shared_ptr<Result>     getResult() const override;

    std::shared_ptr<RootFile> getRootFile() const override {
        return m_pred_rootfile;
    }

private:
    void save();

    std::shared_ptr<RootFile> getModelRoot() const {
        return m_model_rootfile;
    }

  public:
    PredictSession(const Config &config);
    PredictSession(std::shared_ptr<RootFile> rf, const Config &config);
    PredictSession(std::shared_ptr<RootFile> rf);

    std::ostream& info(std::ostream &os, std::string indent) const override;

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
    template <class Feat>
    std::shared_ptr<Eigen::MatrixXd> predict(int mode, const Feat &f, int save_freq = 0);
};

// predict element or elements based on sideinfo
template <class Feat>
std::shared_ptr<Eigen::MatrixXd> PredictSession::predict(int mode, const Feat &f, int save_freq)
{
    std::shared_ptr<Eigen::MatrixXd> average(nullptr);

    for (int step = 0; step < getNumSteps(); step++)
    {
        if (m_config.getVerbose())
        {
            std::cout << "Out-of-matrix prediction step " << step << "/" << getNumSteps() << "." << std::endl;
        }
 
        const auto &sf = m_stepfiles.at(step);
        auto predictions = restoreModel(sf, mode)->predict(mode, f);
        if (!average)
            average = std::make_shared<Eigen::MatrixXd>(predictions);
        else
            *average += predictions;

        if (save_freq > 0 && (step % save_freq) == 0)
        {
            auto filename = m_config.getSavePrefix() + "/predictions-sample-" + std::to_string(step) + m_config.getSaveExtension();
            if (m_config.getVerbose())
            {
                std::cout << "-- Saving sample " << step << " to " << filename << "." << std::endl;
            }
            matrix_io::eigen::write_matrix(filename, predictions);
        }
    }

    (*average) /= (double)getNumSteps();

    if (save_freq != 0)
    {
        auto filename = m_config.getSavePrefix() + "/predictions-average" + m_config.getSaveExtension();
        if (m_config.getVerbose())
        {
            std::cout << "-- Saving average predictions to " << filename << "." << std::endl;
        }
        matrix_io::eigen::write_matrix(filename, *average);
    }

    return average;
}

} // end namespace smurff
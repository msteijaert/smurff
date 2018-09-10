#include <memory>

#include <Eigen/Sparse>
#include <Eigen/Core>

#include <SmurffCpp/Utils/counters.h>
#include <SmurffCpp/Utils/RootFile.h>
#include <SmurffCpp/Utils/MatrixUtils.h>
#include <SmurffCpp/IO/MatrixIO.h>
#include <SmurffCpp/result.h>
#include <SmurffCpp/ResultItem.h>

#include <SmurffCpp/Model.h>

#include <SmurffCpp/Predict/PredictSession.h>

namespace smurff
{

PredictSession::PredictSession(std::shared_ptr<RootFile> rf)
    : m_model_rootfile(rf), m_pred_rootfile(0),
      m_has_config(false), m_num_latent(-1),
      m_dims(PVec<>(0)), m_is_init(false)
{
    m_stepfiles = m_model_rootfile->openSampleStepFiles();
}

PredictSession::PredictSession(std::shared_ptr<RootFile> rf, const Config &config)
    : m_model_rootfile(rf), m_pred_rootfile(0),
      m_config(config), m_has_config(true), m_num_latent(-1),
      m_dims(PVec<>(0)), m_is_init(false)
{
    m_stepfiles = m_model_rootfile->openSampleStepFiles();
}
PredictSession::PredictSession(const Config &config)
    : m_pred_rootfile(0), m_config(config), m_has_config(true),
      m_num_latent(-1), m_dims(PVec<>(0)), m_is_init(false)
{
    THROWERROR_ASSERT(config.getRootName().size())
    m_model_rootfile = std::make_shared<RootFile>(config.getRootName());
    m_stepfiles = m_model_rootfile->openSampleStepFiles();
}

void PredictSession::run()
{
    THROWERROR_ASSERT(m_has_config);


    if (m_config.getTest())
    {
        init();
        while (step())
            ;

        return;
    }
    else
    {
        std::shared_ptr<MatrixConfig> side_info;
        int mode;

        if (m_config.getRowFeatures())
        {
            side_info = m_config.getRowFeatures();
            mode = 0;
        }
        else if (m_config.getColFeatures())
        {
            side_info = m_config.getColFeatures();
            mode = 1;
        }
        else
        {
            THROWERROR("Need either test, row features or col features")
        }

        if (side_info->isDense())
        {
            auto dense_matrix = matrix_utils::dense_to_eigen(*side_info);
            predict(mode, dense_matrix, m_config.getSaveFreq());
        }
        else
        {
            auto sparse_matrix = matrix_utils::sparse_to_eigen(*side_info);
            predict(mode, sparse_matrix, m_config.getSaveFreq());
        }
    }
}

void PredictSession::init()
{
    THROWERROR_ASSERT(m_has_config);
    THROWERROR_ASSERT(m_config.getTest());
    m_result = std::make_shared<Result>(m_config.getTest(), m_config.getNSamples());

    m_pos = m_stepfiles.rbegin();
    m_iter = 0;
    m_is_init = true;

    THROWERROR_ASSERT_MSG(m_config.getSavePrefix() != getModelRoot()->getPrefix(),
                          "Cannot have same prefix for model and predictions - both have " + m_config.getSavePrefix());

    if (m_config.getSaveFreq())
    {
        // create root file
        m_pred_rootfile = std::make_shared<RootFile>(m_config.getSavePrefix(), m_config.getSaveExtension(), true);
        m_pred_rootfile->createCsvStatusFile();
        m_pred_rootfile->flushLast();
    }

    if (m_config.getVerbose())
        info(std::cout, "");
}

bool PredictSession::step()
{
    THROWERROR_ASSERT(m_has_config);
    THROWERROR_ASSERT(m_is_init);
    THROWERROR_ASSERT(m_pos != m_stepfiles.rend());

    double start = tick();
    auto model = restoreModel(*m_pos);
    m_result->update(model, false);
    double stop = tick();
    m_iter++;
    m_secs_per_iter = stop - start;
    m_secs_total += m_secs_per_iter;

    if (m_config.getVerbose())
        std::cout << getStatus()->asString() << std::endl;

    if (m_config.getSaveFreq() > 0 && (m_iter % m_config.getSaveFreq()) == 0)
        save();

    auto next_pos = m_pos;
    next_pos++;
    bool last_iter = next_pos == m_stepfiles.rend();

    //save last iter
    if (last_iter && m_config.getSaveFreq() == -1)
        save();

    m_pos++;
    return !last_iter;
}

void PredictSession::save()
{
    //save this iteration
    std::shared_ptr<StepFile> stepFile = getRootFile()->createSampleStepFile(m_iter);

    if (m_config.getVerbose())
    {
        std::cout << "-- Saving predictions into '" << stepFile->getStepFileName() << "'." << std::endl;
    }

    stepFile->savePred(m_result);

    m_pred_rootfile->addCsvStatusLine(*getStatus());
    m_pred_rootfile->flushLast();
}

std::shared_ptr<StatusItem> PredictSession::getStatus() const
{
    std::shared_ptr<StatusItem> ret = std::make_shared<StatusItem>();
    ret->phase = "Predict";
    ret->iter = (*m_pos)->getIsample();
    ret->phase_iter = m_stepfiles.size();

    ret->train_rmse = NAN;

    ret->rmse_avg = m_result->rmse_avg;
    ret->rmse_1sample = m_result->rmse_1sample;

    ret->auc_avg = m_result->auc_avg;
    ret->auc_1sample = m_result->auc_1sample;

    ret->elapsed_iter = m_secs_per_iter;
    ret->elapsed_total = m_secs_total;

    return ret;
}

std::shared_ptr<Result> PredictSession::getResult() const
{
    return m_result;
}

std::ostream &PredictSession::info(std::ostream &os, std::string indent) const
{
    os << indent << "PredictSession {\n";
    os << indent << "  Model {\n";
    os << indent << "    model root-file: " << getModelRoot()->getFullPath() << "\n";
    os << indent << "    num-samples: " << getNumSteps() << "\n";
    os << indent << "    num-latent: " << getNumLatent() << "\n";
    os << indent << "    dimensions: " << getModelDims() << "\n";
    os << indent << "  }\n";
    os << indent << "  Predictions {\n";
    m_result->info(os, indent + "    ");
    if (m_config.getSaveFreq() > 0)
    {
        os << indent << "    Save predictions: every " << m_config.getSaveFreq() << " iteration\n";
        os << indent << "    Save extension: " << m_config.getSaveExtension() << "\n";
        os << indent << "    Output root-file: " << getRootFile()->getFullPath() << "\n";
    }
    else if (m_config.getSaveFreq() < 0)
    {
        os << indent << "    Save predictions after last iteration\n";
        os << indent << "    Save extension: " << m_config.getSaveExtension() << "\n";
        os << indent << "    Output root-file: " << getRootFile()->getFullPath() << "\n";
    }
    else
    {
        os << indent << "    Don't save predictions\n";
    }
    os << indent << "  }" << std::endl;
    os << indent << "}\n";
    return os;
}

std::shared_ptr<Model> PredictSession::restoreModel(const std::shared_ptr<StepFile> &sf)
{
    auto model = sf->restoreModel();

    if (m_num_latent <= 0)
    {
        m_num_latent = model->nlatent();
        m_dims = model->getDims();
    }
    else
    {
        THROWERROR_ASSERT(m_num_latent == model->nlatent());
        THROWERROR_ASSERT(m_dims == model->getDims());
    }

    THROWERROR_ASSERT(m_num_latent > 0);

    return model;
}

std::shared_ptr<Model> PredictSession::restoreModel(int i)
{
    return restoreModel(m_stepfiles.at(i));
}

// predict one element
ResultItem PredictSession::predict(PVec<> pos, const StepFile &sf)
{
    ResultItem ret{pos};
    predict(ret, sf);
    return ret;
}

// predict one element
void PredictSession::predict(ResultItem &res, const StepFile &sf)
{
    auto model = sf.restoreModel();
    auto pred = model->predict(res.coords);
    res.update(pred);
}

// predict one element
void PredictSession::predict(ResultItem &res)
{
    auto stepfiles = getModelRoot()->openSampleStepFiles();

    for (const auto &sf : stepfiles)
        predict(res, *sf);
}

ResultItem PredictSession::predict(PVec<> pos)
{
    ResultItem ret{pos};
    predict(ret);
    return ret;
}

// predict all elements in Ytest
std::shared_ptr<Result> PredictSession::predict(std::shared_ptr<TensorConfig> Y)
{
    auto res = std::make_shared<Result>(Y);

    for (const auto s : m_stepfiles)
    {
        auto model = restoreModel(s);
        res->update(model, false);
    }

    return res;
}

} // end namespace smurff

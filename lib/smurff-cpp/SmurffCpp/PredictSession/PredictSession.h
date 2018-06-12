#pragma once

#include <memory>

#include <Eigen/Sparse>
#include <Eigen/Core>

#include <SmurffCpp/Utils/PVec.hpp>

namespace smurff {

class RootFile;
class Result;
class ResultItem;

class PredictSession 
{
private:
    std::shared_ptr<RootFile> m_root_file;

public:
    // predict one element
    std::shared_ptr<ResultItem> predict(PVec<> Ytest, int sample = -1);

    // predict all elements in Ytest
    std::shared_ptr<Result> predict(Eigen::SparseMatrix<double> Ytest, int sample = -1);

    // predict element or elements based on sideinfo
    template<class Feat>
    std::shared_ptr<Result> predict(std::vector<std::shared_ptr<Feat>>, int sample = -1);
    
};

}

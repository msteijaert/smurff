#pragma once

#include <memory>
#include <vector>

#include <SmurffCpp/Configs/MatrixConfig.h>
#include <SmurffCpp/Sessions/Session.h>
#include <SmurffCpp/DataMatrices/MatrixData.h>

namespace smurff {

class MatrixDataFactory
{
public:
   static std::shared_ptr<MatrixData> create_matrix(std::shared_ptr<Session> session);
};

}
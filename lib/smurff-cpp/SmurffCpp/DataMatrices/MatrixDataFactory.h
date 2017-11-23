#pragma once

#include <memory>
#include <vector>

#include <SmurffCpp/Configs/MatrixConfig.h>
#include <SmurffCpp/Sessions/Session.h>
#include <SmurffCpp/DataMatrices/Data.h>

namespace smurff {

class MatrixDataFactory
{
public:
   static std::shared_ptr<Data> create_matrix_data(std::shared_ptr<Session> session);
};

}
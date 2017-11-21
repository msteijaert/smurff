#pragma once

#include <memory>
#include <vector>

#include <SmurffCpp/Configs/TensorConfig.h>
#include <SmurffCpp/Sessions/Session.h>

namespace smurff {

class TensorDataFactory
{
public:
   static std::shared_ptr<Data> create_tensor(std::shared_ptr<Session> session);
};

}
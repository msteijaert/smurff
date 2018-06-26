#pragma once

#include <SmurffCpp/Sessions/Session.h>
#include <SmurffCpp/Predict/PredictSession.h>

namespace smurff {
   std::shared_ptr<ISession> create_cmd_session(int argc, char** argv);
}

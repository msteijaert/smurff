#pragma once

#include <SmurffCpp/Sessions/Session.h>
#include <SmurffCpp/Predict/PredictSession.h>

namespace smurff {
    Config parse_options(int argc, char *argv[]);
    std::shared_ptr<ISession> create_cmd_session(int argc, char **argv);
}

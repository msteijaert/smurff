#include <SmurffCpp/Priors/ILatentPrior.h>
#include "CmdSession.h"

using namespace smurff;

int main(int argc, char** argv) {
    CmdSession session;
    session.setFromArgs(argc, argv);
    session.run();
    return 0;
}

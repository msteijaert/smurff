#include <SmurffCpp/Priors/ILatentPrior.h>
#include <SmurffCpp/Sessions/CmdSession.h>

using namespace smurff;

int main(int argc, char** argv) {
    CmdSession session;
    session.setFromArgs(argc, argv);
    session.run();
    return 0;
}

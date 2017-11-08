#include <SmurffCpp/Priors/ILatentPrior.h>
#include <SmurffCpp/Sessions/SessionFactory.h>

using namespace smurff;

int main(int argc, char** argv) 
{
    std::shared_ptr<smurff::Session> session = SessionFactory::create_cmd_session(argc, argv);
    session->run();
    return 0;
}

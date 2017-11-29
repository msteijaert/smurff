#include <SmurffCpp/Priors/ILatentPrior.h>
#include <SmurffCpp/Sessions/SessionFactory.h>
#include <SmurffCpp/Utils/counters.h>

using namespace smurff;

int main(int argc, char** argv) 
{
    std::shared_ptr<smurff::ISession> session = SessionFactory::create_cmd_session(argc, argv);
    { COUNTER("main"); session->run(); }
#ifdef PROFILING
    perf_data.print();
#endif
    return 0;
}

#include "PythonSession.h"

#include <signal.h>

#include <SmurffCpp/Priors/ILatentPrior.h>

using namespace smurff;

bool PythonSession::keepRunning = true;
bool PythonSession::keepRunningVerbose = false;

bool PythonSession::step()
{
    
    keepRunning = true;
    keepRunningVerbose = m_config.getVerbose();

#ifdef _WINDOWS
    signal(SIGINT, intHandler);
#else
    // save old handler, add our handler
    struct sigaction newHandler;
    struct sigaction oldHandler;
    memset(&newHandler, 0, sizeof(newHandler));
    sigemptyset( &newHandler.sa_mask );
    newHandler.sa_handler = intHandler;
    sigaction(SIGINT, &newHandler, &oldHandler );
#endif

    // run step
    bool not_done = Session::step();

#ifdef _WINDOWS
    // can't do this on windows
#else
    // restore old handler
    sigaction(SIGINT, &oldHandler, NULL );
#endif

    return not_done;
}

void PythonSession::intHandler(int)
{
   keepRunning = false;

   if (keepRunningVerbose)
   {
       std::cout << "[Received Ctrl-C. Stopping after finishing the current iteration.]\n";
   }
}

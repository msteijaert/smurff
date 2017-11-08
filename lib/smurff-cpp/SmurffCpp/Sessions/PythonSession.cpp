#include "PythonSession.h"

#include <signal.h>

#include <SmurffCpp/Priors/ILatentPrior.h>

using namespace smurff;

bool PythonSession::keepRunning = true;

void PythonSession::step() 
{
   if (!keepRunning) 
   return;
   signal(SIGINT, intHandler);
   Session::step();
}

void PythonSession::intHandler(int) 
{
   keepRunning = false;
   printf("[Received Ctrl-C. Stopping after finishing the current iteration.]\n");
} 
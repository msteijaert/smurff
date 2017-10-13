#pragma once

#include "Session.h"

namespace smurff {

class CmdSession :  public Session 
{
public:
   void setFromArgs(int argc, char** argv);
};

}
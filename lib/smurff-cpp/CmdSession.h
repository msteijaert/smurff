#pragma once

#include "Session2.h"

namespace smurff {

class CmdSession :  public Session 
{
public:
   void setFromArgs(int argc, char** argv);
};

}
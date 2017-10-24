#pragma once

#include <Smurff/CmdSession.h>

namespace smurff {

class MPISession : public CmdSession 
{
public:
   int world_rank;
   int world_size;

public:
   MPISession();
   void run();
};

}

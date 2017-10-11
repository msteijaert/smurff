#pragma once

namespace smurff {

class MPISession : public CmdSession 
{
public:
   int world_rank;
   int world_size;

public:
   MPISession();

   void run();

   std::ostream &info(std::ostream &os, std::string indent) override;
};

}
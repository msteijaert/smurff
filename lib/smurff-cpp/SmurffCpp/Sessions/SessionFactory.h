#pragma once

#include <memory>

#include <SmurffCpp/Sessions/Session.h>

namespace smurff {

   class SessionFactory
   {
   public:
      static std::shared_ptr<Session> create_cmd_session(int argc, char** argv);

      #ifdef MPI_FOUND
      static std::shared_ptr<Session> create_mpi_session(int argc, char** argv);
      #endif

      static std::shared_ptr<Session> create_py_session(Config& cfg);

   private:
      static void initialize_session(std::shared_ptr<Session> session);
   };

}
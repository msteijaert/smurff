#pragma once

#include <stdexcept>
#include <string>
#include <sstream>

#define CONCAT_VAR(n1, n2) n1 ## n2

#define THROWERROR_BASE(msg, ssvar, except_type) { \
   std::stringstream ssvar; \
   ssvar << "line: " << __LINE__ << " file: " << __FILE__ << " function: " << __FUNCTION__ << std::endl << (msg); \
   throw except_type(ssvar.str()); }

#define THROWERROR(msg) THROWERROR_BASE(msg, CONCAT_VAR(ss, __LINE__), std::runtime_error)

#define THROWERROR_SPEC(except_type, msg) THROWERROR_BASE(msg, CONCAT_VAR(ss, __LINE__), except_type)

namespace smurff
{
   void not_implemented(std::string message);

   void die_unless_file_exists(std::string fname);
}
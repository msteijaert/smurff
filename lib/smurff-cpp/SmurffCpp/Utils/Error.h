#pragma once

#include <stdexcept>
#include <string>
#include <sstream>

#define CONCAT_VAR(n1, n2) n1 ## n2

#define THROWERROR_BASE(msg, ssvar) { \
   std::stringstream ssvar; \
   ssvar << "line: " << __LINE__ << " file: " << __FILE__ << " function: " << __FUNCTION__ << std::endl << (msg); \
   throw std::runtime_error(ssvar.str()); }

#define THROWERROR(msg) THROWERROR_BASE(msg, CONCAT_VAR(ss, __LINE__))
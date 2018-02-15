#pragma once

#include <vector>
#include <string>

namespace smurff
{

   std::string& ltrim(std::string& s);

   std::string& rtrim(std::string& s);

   std::string& trim(std::string& s);

   void split(const std::string& str, std::vector<std::string >& tokens, char delim);

   bool startsWith(const std::string& str, const std::string& prefix);
}
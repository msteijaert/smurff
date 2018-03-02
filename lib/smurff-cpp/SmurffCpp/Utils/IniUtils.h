#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace smurff
{
   void loadIni(const std::string& path, std::vector<std::pair<std::string, std::string> >& values);

   void loadIni(const std::string& path, std::unordered_map<std::string, std::string>& values);

   std::vector<std::pair<std::string, std::string> >::const_iterator iniFind(const std::vector<std::pair<std::string, std::string> >& values, std::string tag);

   void iniRemove(std::vector<std::pair<std::string, std::string> >& values, std::string tag);
}
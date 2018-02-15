#pragma once

#include <string>
#include <unordered_map>

namespace smurff
{

   void loadIni(const std::string& path, std::unordered_map<std::string, std::string>& values);

}
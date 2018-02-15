#include <SmurffCpp/Utils/IniUtils.h>

#include <fstream>
#include <vector>

#include <SmurffCpp/Utils/StringUtils.h>
#include <SmurffCpp/Utils/Error.h>
#include <SmurffCpp/IO/GenericIO.h>

void smurff::loadIni(const std::string& path, std::unordered_map<std::string, std::string>& values)
{
   THROWERROR_FILE_NOT_EXIST(path);

   std::ifstream file;
   file.open(path);

   std::vector<std::string > tokens;
   std::string line;
   while (getline(file, line))
   {
      smurff::split(line, tokens, '=');

      THROWERROR_ASSERT_MSG(tokens.size() == 2, "Invalid key value pair in ini file");

      values.insert(std::make_pair(smurff::trim(tokens.at(0)), smurff::trim(tokens.at(1))));
   }

   file.close();
}
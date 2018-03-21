#include <SmurffCpp/Utils/IniUtils.h>

#include <fstream>
#include <vector>
#include <functional>

#include <SmurffCpp/Utils/StringUtils.h>
#include <SmurffCpp/Utils/Error.h>
#include <SmurffCpp/IO/GenericIO.h>

void loadIniBase(const std::string& path, std::function<void(std::pair<std::string, std::string>)> inserter)
{
   THROWERROR_FILE_NOT_EXIST(path);

   std::ifstream file;
   file.open(path);

   std::vector<std::string > tokens;
   std::string line;
   while (getline(file, line))
   {
      smurff::split(line, tokens, '=');

      if (tokens.size() == 1)
      {
         //skip comment
         THROWERROR_ASSERT_MSG(smurff::startsWith(smurff::trim(tokens.at(0)), "#"), "Invalid key value pair in ini file");
      }
      else if (tokens.size() == 2)
      {
         inserter(std::make_pair(smurff::trim(tokens.at(0)), smurff::trim(tokens.at(1))));
      }
      else if(tokens.size() > 2)
      {
         THROWERROR("Invalid key value pair in ini file");
      }
   }

   file.close();
}

void vector_inserter(std::vector<std::pair<std::string, std::string> >& values, std::pair<std::string, std::string> value)
{
   values.push_back(value);
}

void smurff::loadIni(const std::string& path, std::vector<std::pair<std::string, std::string> >& values)
{
   using namespace std::placeholders;

   auto func = std::bind(vector_inserter, std::ref(values), _1);
   loadIniBase(path, func);
}

void map_inserter(std::unordered_map<std::string, std::string>& values, std::pair<std::string, std::string> value)
{
   values.insert(value);
}

void smurff::loadIni(const std::string& path, std::unordered_map<std::string, std::string>& values)
{
   using namespace std::placeholders;

   auto func = std::bind(map_inserter, std::ref(values), _1);
   loadIniBase(path, func);
}

std::vector<std::pair<std::string, std::string> >::const_iterator smurff::iniFind(const std::vector<std::pair<std::string, std::string> >& values, std::string tag)
{
   for (std::vector<std::pair<std::string, std::string> >::const_iterator it = values.begin(); it != values.end(); ++it)
   {
      if (it->first == tag)
         return it;
   }
   return values.end();
}

void smurff::iniRemove(std::vector<std::pair<std::string, std::string> >& values, std::string tag)
{
   for (std::vector<std::pair<std::string, std::string> >::iterator it = values.begin(); it != values.end(); ++it)
   {
      if (it->first == tag)
      {
         values.erase(it);
         break;
      }
   }
}

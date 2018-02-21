#include <SmurffCpp/Utils/StringUtils.h>

#include <algorithm> 
#include <functional> 
#include <cctype>
#include <locale>
#include <sstream>

std::string& smurff::ltrim(std::string& s)
{
   s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
   return s;
}

std::string& smurff::rtrim(std::string& s)
{
   s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
   return s;
}

std::string& smurff::trim(std::string& s)
{
   return ltrim(rtrim(s));
}

void smurff::split(const std::string& str, std::vector<std::string >& tokens, char delim)
{
   tokens.clear();

   std::stringstream ss(str);
   std::string token;

   while (getline(ss, token, delim))
      tokens.push_back(token);
}

bool smurff::startsWith(const std::string& str, const std::string& prefix)
{
   //protection from out of range exception
   if (str.length() < prefix.length())
      return false;

   return std::equal(prefix.begin(), prefix.end(), str.begin());
}
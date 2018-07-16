#include <SmurffCpp/Utils/StringUtils.h>

#include <algorithm> 
#include <functional> 
#include <cctype>
#include <locale>
#include <sstream>

template<>
std::string smurff::_util::convert<std::string>(const std::string& value)
{
   return value;
}

template<>
int smurff::_util::convert<int>(const std::string& value)
{
   return std::stoi(value);
}

template<>
double smurff::_util::convert<double>(const std::string& value)
{
   return std::stod(value);
}

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

bool smurff::startsWith(const std::string& str, const std::string& prefix)
{
   //protection from out of range exception
   if (str.length() < prefix.length())
      return false;

   return std::equal(prefix.begin(), prefix.end(), str.begin());
}

std::string smurff::stripPrefix(const std::string& str,  const std::string& prefix)
{
   //protection from out of range exception
   if (str.length() < prefix.length())
      return str;

   if (str.substr(0, prefix.size()) != prefix)
      return str;
   
   return fileName(str);
}


std::string smurff::fileName(const std::string& str)
{
   auto pos = str.find_last_of("/");
   if (std::string::npos == pos)
      return str;
     
   return str.substr(pos+1);
}

std::string smurff::dirName(const std::string& str)
{
   auto pos = str.find_last_of("/");
   if (std::string::npos == pos) 
      return std::string();
     
   return str.substr(0, pos+1);
}

std::string smurff::addDirName(const std::string& str,  const std::string& dirname)
{
   if (str.size() >0 && str[0] == '/')
      return str;
     
   return dirname + str;
}




#pragma once

#include <vector>
#include <string>
#include <sstream>

namespace smurff
{
   std::string& ltrim(std::string& s, const std::string& delimiters = " \f\n\r\t\v" );
   std::string& rtrim(std::string& s, const std::string& delimiters = " \f\n\r\t\v" );
   std::string&  trim(std::string& s, const std::string& delimiters = " \f\n\r\t\v" );

   namespace _util
   {
      template<typename T>
      T convert(const std::string& value);

      template<>
      std::string convert<std::string>(const std::string& value);

      template<>
      int convert<int>(const std::string& value);

      template<>
      double convert<double>(const std::string& value);
   }

   template<typename T>
   void split(const std::string& str, std::vector<T>& tokens, char delim)
   {
      tokens.clear();

      std::stringstream ss(str);
      std::string token;

      while (std::getline(ss, token, delim))
         tokens.push_back(_util::convert<T>(token));
   }

   bool startsWith(const std::string& str, const std::string& prefix);

   std::string stripPrefix(const std::string& str, const std::string& prefix);
   std::string fileName(const std::string& str);
   std::string dirName(const std::string& str);

   std::string addDirName(const std::string& str,  const std::string& prefix);
}

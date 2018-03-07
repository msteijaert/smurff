// Read an INI file into easy-to-access name/value pairs.
// this code is based on https://github.com/Blandinium/inih/blob/master/cpp/INIReader.cpp 61bf1b3  on Dec 18, 2014

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>

#include "ini.h"
#include "INIReader.h"

INIReader::INIReader(const std::string& filename)
{
   m_error = ini_parse(filename.c_str(), ValueHandler, this);
}

INIReader::~INIReader()
{
}

int INIReader::getParseError() const
{
   return m_error;
}

std::string INIReader::get(const std::string& section, const std::string& name, const std::string& default_value) const
{
   std::string key = MakeKey(section, name);
   auto valuesIt = m_values.find(key);
   if (valuesIt == m_values.end())
      return default_value;
   else
      return valuesIt->second;
}

int INIReader::getInteger(const std::string& section, const std::string& name, int default_value) const
{
   std::string valstr = get(section, name, "");
   const char* value = valstr.c_str();
   char* end;
   // This parses "1234" (decimal) and also "0x4D2" (hex)
   long n = strtol(value, &end, 0);
   return end > value ? n : default_value;
}

double INIReader::getReal(const std::string& section, const std::string& name, double default_value) const
{
   std::string valstr = get(section, name, "");
   const char* value = valstr.c_str();
   char* end;
   double n = strtod(value, &end);
   return end > value ? n : default_value;
}

bool INIReader::getBoolean(const std::string& section, const std::string& name, bool default_value) const
{
   std::string valstr = get(section, name, "");
   // Convert to lower case to make string comparisons case-insensitive
   std::transform(valstr.begin(), valstr.end(), valstr.begin(), ::tolower);
   if (valstr == "true" || valstr == "yes" || valstr == "on" || valstr == "1")
      return true;
   else if (valstr == "false" || valstr == "no" || valstr == "off" || valstr == "0")
      return false;
   else
      return default_value;
}

const std::set<std::string>& INIReader::getSections() const
{
   return m_sections;
}

std::map<std::string, std::vector<std::string> >::const_iterator INIReader::getFields(const std::string& section) const
{
   auto fieldSetIt = m_fields.find(section);
   if (fieldSetIt == m_fields.end())
      return m_fields.end();
   return fieldSetIt;
}

std::string INIReader::MakeKey(const std::string& section, const std::string& name)
{
   return section + "=" + name;    
}

int INIReader::ValueHandler(void* user, const char* section, const char* name, const char* value)
{
   INIReader* reader = (INIReader*)user;

   // Insert the section in the sections set
   reader->m_sections.insert(section);

   // Add the value to the lookup map
   std::string key = MakeKey(section, name);

   auto valuesIt = reader->m_values.find(key);
   if (valuesIt == reader->m_values.end())
   {
      reader->m_values.emplace(key, value);

      //if value was not in the lookup map - we need to add it in list of m_fields as well
      //we store fields in original order as vector so the only efficient way to make sure that values are unique
      //is to use m_values as a lookup first

      //create entry for new section if required
      auto fieldSetIt = reader->m_fields.find(section);
      if (fieldSetIt == reader->m_fields.end())
      {
         auto item = reader->m_fields.emplace(section, std::vector<std::string>());
         fieldSetIt = item.first;
      }

      //add field name
      fieldSetIt->second.push_back(name);
   }
   else
   {
      std::cout << "Warning: duplicate key: '" << name << "' in section: '" << section << "'" << std::endl;
   }

   return 1;
}

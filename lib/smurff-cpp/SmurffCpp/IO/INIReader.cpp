// Read an INI file into easy-to-access name/value pairs.
// this code is based on https://github.com/Blandinium/inih/blob/master/cpp/INIReader.cpp 61bf1b3  on Dec 18, 2014

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <fstream>

#include "ini.h"
#include "INIReader.h"

#include <SmurffCpp/Utils/Error.h>

INIFile::INIFile()
   : m_error(0)
{
}

INIFile::~INIFile()
{
}

void INIFile::open(const std::string& filename)
{
   m_filePath = filename;
   m_error = ini_parse(filename.c_str(), ValueHandler, this);
}

void INIFile::create(const std::string& filename)
{
   m_filePath = filename;

   std::ofstream stepFile;
   stepFile.open(filename, std::ios::trunc);
   THROWERROR_ASSERT_MSG(stepFile, "Error opening file: " + filename);
   stepFile.close();
}

int INIFile::getParseError() const
{
   return m_error;
}

std::string INIFile::get(const std::string& section, const std::string& name, const std::string& default_value) const
{
   std::string key = MakeKey(section, name);
   auto valuesIt = m_values.find(key);
   if (valuesIt == m_values.end())
      return default_value;
   else
      return valuesIt->second;
}

int INIFile::getInteger(const std::string& section, const std::string& name, int default_value) const
{
   std::string valstr = get(section, name, "");
   const char* value = valstr.c_str();
   char* end;
   // This parses "1234" (decimal) and also "0x4D2" (hex)
   long n = strtol(value, &end, 0);
   return end > value ? n : default_value;
}

double INIFile::getReal(const std::string& section, const std::string& name, double default_value) const
{
   std::string valstr = get(section, name, "");
   const char* value = valstr.c_str();
   char* end;
   double n = strtod(value, &end);
   return end > value ? n : default_value;
}

bool INIFile::getBoolean(const std::string& section, const std::string& name, bool default_value) const
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

std::string INIFile::get(const std::string& section, const std::string& name) const
{
   std::string key = MakeKey(section, name);
   auto valuesIt = m_values.find(key);
   if (valuesIt == m_values.end())
   {
      THROWERROR("section: " + section + " name:" + name + " not found");
   }
   else
   {
      return valuesIt->second;
   }
}

std::pair<bool, std::string> INIFile::tryGet(const std::string& section, const std::string& name) const
{
   std::string key = MakeKey(section, name);
   auto valuesIt = m_values.find(key);
   if (valuesIt == m_values.end())
      return std::make_pair(false, std::string());
   else
      return std::make_pair(true, valuesIt->second);
}

const std::set<std::string>& INIFile::getSections() const
{
   return m_sections;
}

std::map<std::string, std::vector<std::string> >::const_iterator INIFile::getFields(const std::string& section) const
{
   auto fieldSetIt = m_fields.find(section);
   if (fieldSetIt == m_fields.end())
      THROWERROR("section: " + section + " not found");
   return fieldSetIt;
}

std::string INIFile::MakeKey(const std::string& section, const std::string& name)
{
   return section + "=" + name;
}

int INIFile::ValueHandler(void* user, const char* section, const char* name, const char* value)
{
   INIFile* reader = (INIFile*)user;

   reader->insertItem(section, name, value);

   return 1;
}

int INIFile::insertItem(const std::string& section, const std::string& name, const std::string& value)
{
   // Insert the section in the sections set
   m_sections.insert(section);

   // Add the value to the lookup map
   std::string key = MakeKey(section, name);

   auto valuesIt = m_values.find(key);
   if (valuesIt == m_values.end())
   {
      m_values.emplace(key, value);

      //if value was not in the lookup map - we need to add it in list of m_fields as well
      //we store fields in original order as vector so the only efficient way to make sure that values are unique
      //is to use m_values as a lookup first

      //create entry for new section if required
      auto fieldSetIt = m_fields.find(section);
      if (fieldSetIt == m_fields.end())
      {
         auto item = m_fields.emplace(section, std::vector<std::string>());
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

bool INIFile::empty() const
{
   return m_values.empty() && m_sections.empty() && m_fields.empty();
}

void INIFile::appendItem(const std::string& tag, const std::string& value)
{
   m_writeBuffer.push_back(std::make_pair(tag, value));

   insertItem(std::string(), tag, value);
}

void INIFile::flush()
{
   std::ofstream stepFile;
   stepFile.open(m_filePath, std::ios::app);
   THROWERROR_ASSERT_MSG(stepFile, "Error opening file: " + m_filePath);

   for (auto& item : m_writeBuffer)
   {
      stepFile << item.first << " = " << item.second << std::endl;
   }

   stepFile.close();

   m_writeBuffer.clear();
}

void INIFile::removeItem(const std::string& section, const std::string& tag)
{
   //create a key
   std::string key = MakeKey(section, tag);

   //try erase value from values map
   auto valuesIt = m_values.find(key);
   if (valuesIt != m_values.end())
      m_values.erase(valuesIt);
   
   //try find section in field groups
   auto fieldsGroupIt = m_fields.find(section);
   if (fieldsGroupIt != m_fields.end())
   {
      //try find field in field group
      for (auto fieldIt = fieldsGroupIt->second.begin(); fieldIt != fieldsGroupIt->second.end(); ++fieldIt)
      {
         //erase field from section
         if (*fieldIt == tag)
         { 
            fieldsGroupIt->second.erase(fieldIt);
            break;
         }
      }

      //check if section is now empty
      if (fieldsGroupIt->second.empty())
      {
         //try erase section from sections map
         auto sectionsIt = m_sections.find(fieldsGroupIt->first);
         if (sectionsIt != m_sections.end())
            m_sections.erase(sectionsIt);

         //erase section from field groups
         m_fields.erase(fieldsGroupIt);
      }
   }
}

void INIFile::appendComment(std::string comment)
{
   std::ofstream stepFile;
   stepFile.open(m_filePath, std::ios::app);
   THROWERROR_ASSERT_MSG(stepFile, "Error opening file: " + m_filePath);
   stepFile << "#" << comment << std::endl;
   stepFile.close();
}
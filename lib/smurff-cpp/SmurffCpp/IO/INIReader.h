// this code is based on https://github.com/Blandinium/inih/blob/master/cpp/INIReader.h 61bf1b3  on Dec 18, 2014

// Read an INI file into easy-to-access name/value pairs.

// inih and INIReader are released under the New BSD license (see LICENSE.txt).
// Go to the project home page for more info: http://code.google.com/p/inih/

//This code is heavily changed to support our needs

#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

class INIReader
{
private:
   int m_error;

   std::map<std::string, std::string> m_values;

   std::set<std::string> m_sections;

   std::map<std::string, std::vector<std::string> > m_fields;

public:
    // Construct INIReader and parse given filename. See ini.h for more info about the parsing.
    INIReader(const std::string& filename);
    
    ~INIReader();

    // Return the result of ini_parse(), i.e., 0 on success, 
    // line number of first error on parse error, 
    // or -1 on file open error.
    int getParseError() const;

    // Get a string value from INI file, 
    // returning default_value if not found.
    std::string get(const std::string& section, const std::string& name, const std::string& default_value) const;

    // Get an integer (long) value from INI file, 
    // returning default_value if not found or not a valid integer (decimal "1234", "-1234", or hex "0x4d2").
    int getInteger(const std::string& section, const std::string& name, int default_value) const;

    // Get a real (floating point double) value from INI file, 
    // returning default_value if not found or not a valid floating point value according to strtod().
    double getReal(const std::string& section, const std::string& name, double default_value) const;

    // Get a boolean value from INI file, 
    // returning default_value if not found or if not a valid true/false value. 
    // Valid true values are "true", "yes", "on", "1",
    // and valid false values are "false", "no", "off", "0" (not case sensitive).
    bool getBoolean(const std::string& section, const std::string& name, bool default_value) const;

    // Returns all the section names from the INI file in alphabetical order
    const std::set<std::string>& getSections() const;

    // Returns all the field names from a section in the INI file in original order
    // Returns end iterator if section name is not found
    std::map<std::string, std::vector<std::string> >::const_iterator getFields(const std::string& section) const;

private:
    static std::string MakeKey(const std::string& section, const std::string& name);

    //callback for handling values parsed by ini_parse
    static int ValueHandler(void* user, const char* section, const char* name, const char* value);
};
#include "Error.h"

#include <fstream>

bool file_exists(const char *fileName)
{
   std::ifstream infile(fileName);
   return infile.good();
}

bool file_exists(const std::string fileName)
{
   return file_exists(fileName.c_str());
}

void smurff::not_implemented(std::string message) 
{
   THROWERROR(std::string("[Not implemented]: '") + message +  "'\n");
}

void smurff::die_unless_file_exists(std::string fname) 
{
   if (fname.size() && ! file_exists(fname)) 
   {
      THROWERROR(std::string("[ERROR]\nFile '") + fname +  "' not found.\n");
   }
}
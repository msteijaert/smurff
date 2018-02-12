#pragma once

#include <string>

#include <SmurffCpp/Configs/Config.h>

#include <SmurffCpp/Utils/StepFile.h>

namespace smurff {

class RootFile
{
private:
   std::string m_path;
   std::string m_prefix;
   std::string m_extension;

public:
   RootFile(std::string path);

   RootFile(std::string prefix, std::string extension);

private:
   std::string getRootFileName() const;
   std::string getOptionsFileName() const;

private:
   void appendToRootFile(std::string tag, std::string item, bool truncate) const;

public:
   void saveConfig(Config config);

public:
   std::shared_ptr<StepFile> createStepFile(int isample) const;
};

}
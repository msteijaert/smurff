#pragma once

#include <string>
#include <unordered_map>

#include <SmurffCpp/Configs/Config.h>

#include <SmurffCpp/Utils/StepFile.h>

namespace smurff {

class RootFile
{
private:
   std::string m_path;
   std::string m_prefix;
   std::string m_extension;

   mutable std::unordered_map<std::string, std::string> m_iniStorage;

public:
   RootFile(std::string path);

   RootFile(std::string prefix, std::string extension);

private:
   std::string getRootFileName() const;
   std::string getOptionsFileName() const;

private:
   void appendToRootFile(std::string tag, std::string item) const;

public:
   void saveConfig(Config& config);

   void restoreConfig(Config& config);

public:
   std::shared_ptr<StepFile> createStepFile(std::int32_t isample) const;

public:
   std::shared_ptr<StepFile> getLastStepFile() const;

   std::shared_ptr<StepFile> getStepFile(std::int32_t isample) const;
};

}
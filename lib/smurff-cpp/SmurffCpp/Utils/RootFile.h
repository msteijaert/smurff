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
   //this constructor should be used to open existing root file when previous session is continued
   //it will read list of references to step files and load them into m_iniStorage
   //you can then call getLastStepFile or getStepFile to access existing step file
   RootFile(std::string path);

   //this constructor should be used to create a new root file on the first run of session
   //items are then appended to it when createStepFile is called
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
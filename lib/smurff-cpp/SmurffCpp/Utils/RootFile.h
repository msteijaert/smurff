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

   //AGE: should preserve order of elements as in file
   mutable std::vector<std::pair<std::string, std::string> > m_iniStorage;

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
   void appendToRootFile(std::string tag, std::string value) const;

public:
   void saveConfig(Config& config);

   void restoreConfig(Config& config);

private:
   void restoreState(std::string& save_prefix, std::string& save_extension);

   std::string restoreGetOptionsFileName() const;

public:
   std::shared_ptr<StepFile> createSampleStepFile(std::int32_t isample) const;

   std::shared_ptr<StepFile> createBurninStepFile(std::int32_t isample) const;

public:
   void removeSampleStepFile(std::int32_t isample) const;

   void removeBurninStepFile(std::int32_t isample) const;

private:
   std::shared_ptr<StepFile> createStepFileInternal(std::int32_t isample, bool burnin) const;

private:
   void removeStepFileInternal(std::int32_t isample, bool burnin) const;

public:
   std::shared_ptr<StepFile> openLastStepFile() const;

/*
public:
   std::shared_ptr<StepFile> openSampleStepFile(std::int32_t isample) const;

   std::shared_ptr<StepFile> openSampleStepFile(std::string path) const;
*/

public:
   void flush() const;

   void flushLast() const;

/*
public:
   std::vector<std::pair<std::string, std::string> >::const_iterator stepFilesBegin() const;
   std::vector<std::pair<std::string, std::string> >::const_iterator stepFilesEnd() const;
*/
};

}
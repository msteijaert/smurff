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

   mutable std::string m_cur_section;

   //preserves order of elements in the file
   mutable std::shared_ptr<INIFile> m_iniReader;

public:
   //this constructor should be used to open existing root file when previous session is continued
   //it will read list of references to step files and load them into m_iniStorage
   //you can then call getLastStepFile or getStepFile to access existing step file
   RootFile(std::string path);

   //this constructor should be used to create a new root file on the first run of session
   //items are then appended to it when createStepFile is called
   RootFile(std::string prefix, std::string extension);

public:
   std::string getRootFileName() const;
   std::string getOptionsFileName() const;

private:
   void appendToRootFile(std::string section, std::string tag, std::string value) const;

   void appendCommentToRootFile(std::string comment) const;

public:
   void saveConfig(Config& config);

   void restoreConfig(Config& config);

private:
   void restoreState(std::string& save_prefix, std::string& save_extension);

   std::string restoreGetOptionsFileName() const;

public:
   std::shared_ptr<StepFile> createSampleStepFile(std::int32_t isample) const;

   std::shared_ptr<StepFile> createCheckpointStepFile(std::int32_t isample) const;

public:
   void removeSampleStepFile(std::int32_t isample) const;

   void removeCheckpointStepFile(std::int32_t isample) const;

private:
   std::shared_ptr<StepFile> createStepFileInternal(std::int32_t isample, bool burnin) const;

private:
   void removeStepFileInternal(std::int32_t isample, bool burnin) const;

public:
   std::shared_ptr<StepFile> openLastStepFile() const;

   std::vector<std::shared_ptr<StepFile>> openSampleStepFiles() const;

/*
public:
   std::shared_ptr<StepFile> openSampleStepFile(std::int32_t isample) const;

   std::shared_ptr<StepFile> openSampleStepFile(std::string path) const;
*/

public:
   void flushLast() const;

/*
public:
   std::vector<std::pair<std::string, std::string> >::const_iterator stepFilesBegin() const;
   std::vector<std::pair<std::string, std::string> >::const_iterator stepFilesEnd() const;
*/
};

}

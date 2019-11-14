#pragma once

#include <string>
#include <unordered_map>

#include <SmurffCpp/Configs/Config.h>

#include <SmurffCpp/Utils/StepFile.h>

namespace smurff {

struct StatusItem;

class RootFile
{
private:
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
   RootFile(std::string prefix, std::string extension, bool create);

public:
   std::string getPrefix() const;
   std::string getFullPath() const;
   std::string getOptionsFileName() const;
   std::string getCsvStatusFileName() const;

private:
   std::string getFullPathFromIni(const std::string &section, const std::string &field) const;

private:
   void appendToRootFile(std::string section, std::string tag, std::string value) const;

   void appendCommentToRootFile(std::string comment) const;

public:
   void saveConfig(Config& config);

   void restoreConfig(Config& config);

private:
   std::string restoreGetOptionsFileName() const;

public:
   std::shared_ptr<StepFile> createSampleStepFile(std::int32_t isample, bool final) const;

   std::shared_ptr<StepFile> createCheckpointStepFile(std::int32_t isample) const;

public:
   void removeSampleStepFile(std::int32_t isample) const;

   void removeCheckpointStepFile(std::int32_t isample) const;

private:
   std::shared_ptr<StepFile> createStepFileInternal(std::int32_t isample, bool burnin, bool final) const;

private:
   void removeStepFileInternal(std::int32_t isample, bool burnin) const;

public:
   std::shared_ptr<StepFile> openLastCheckpoint() const;

   std::shared_ptr<StepFile> openSampleStepFile(int isample) const;

   std::vector<std::shared_ptr<StepFile>> openSampleStepFiles() const;

public:
   void flushLast() const;

public:
  void createCsvStatusFile() const;
  void addCsvStatusLine(const StatusItem &status_item) const;

  /*
public:
   std::vector<std::pair<std::string, std::string> >::const_iterator stepFilesBegin() const;
   std::vector<std::pair<std::string, std::string> >::const_iterator stepFilesEnd() const;
*/
};

}

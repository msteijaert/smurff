#include "RootFile.h"

#include <iostream>
#include <fstream>

#include <SmurffCpp/Utils/Error.h>
#include <SmurffCpp/Utils/StringUtils.h>
#include <SmurffCpp/Utils/IniUtils.h>
#include <SmurffCpp/IO/GenericIO.h>

#define OPTIONS_TAG "options"
#define CHECKPOINT_STEP_PREFIX "checkpoint_step_"
#define SAMPLE_STEP_PREFIX "sample_step_"

using namespace smurff;

RootFile::RootFile(std::string path)
   : m_path(path)
{
   //load all entries in ini file to be able to go through step variables
   loadIni(m_path, m_iniStorage);

   //lightweight restore of prefix and extension
   restoreState(m_prefix, m_extension);
}

RootFile::RootFile(std::string prefix, std::string extension)
   : m_prefix(prefix), m_extension(extension)
{
   m_path = getRootFileName();
}

std::string RootFile::getRootFileName() const
{
   return m_prefix + "-root.ini";
}

std::string RootFile::getOptionsFileName() const
{
   return m_prefix + "-options.ini";
}

void RootFile::appendToRootFile(std::string tag, std::string value) const
{
   m_iniStorage.push_back(std::make_pair(tag, value));

   //this function should be used here when it is required to write to root file immediately
   //currently we call flushLast method separately
   //this guarantees integrity of a step - step file entry is written after everything else is calculated/written
   //flushLast();
}

void RootFile::saveConfig(Config& config)
{
   std::string configPath = getOptionsFileName();
   config.save(configPath);
   appendToRootFile(OPTIONS_TAG, configPath);
}

std::string RootFile::restoreGetOptionsFileName() const
{
   THROWERROR_ASSERT_MSG(!m_iniStorage.empty(), "Root ini file is not loaded");

   auto optionsIt = iniFind(m_iniStorage, OPTIONS_TAG);
   THROWERROR_ASSERT_MSG(optionsIt != m_iniStorage.end(), "Options tag is not found in root ini file");

   return optionsIt->second;
}

void RootFile::restoreConfig(Config& config)
{
   //get options filename
   std::string optionsFileName = restoreGetOptionsFileName();

   //restore config
   bool success = config.restore(optionsFileName);
   THROWERROR_ASSERT_MSG(success, "Could not load ini file '" + optionsFileName + "'");
}

void RootFile::restoreState(std::string& save_prefix, std::string& save_extension)
{
   //get options filename
   std::string optionsFileName = restoreGetOptionsFileName();

   //lightweight restore
   bool success = Config::restoreSaveInfo(optionsFileName, save_prefix, save_extension);
   THROWERROR_ASSERT_MSG(success, "Could not load ini file '" + optionsFileName + "'");
}

std::shared_ptr<StepFile> RootFile::createSampleStepFile(std::int32_t isample) const
{
   return createStepFileInternal(isample, false);
}

std::shared_ptr<StepFile> RootFile::createCheckpointStepFile(std::int32_t isample) const
{
   return createStepFileInternal(isample, true);
}

std::shared_ptr<StepFile> RootFile::createStepFileInternal(std::int32_t isample, bool checkpoint) const
{
   std::shared_ptr<StepFile> stepFile = std::make_shared<StepFile>(isample, m_prefix, m_extension, true, checkpoint);

   std::string stepFileName = stepFile->getStepFileName();
   std::string tagPrefix = checkpoint ? CHECKPOINT_STEP_PREFIX : SAMPLE_STEP_PREFIX;
   std::string stepTag = tagPrefix + std::to_string(isample);
   appendToRootFile(stepTag, stepFileName);

   return stepFile;
}

void RootFile::removeSampleStepFile(std::int32_t isample) const
{
   removeStepFileInternal(isample, false);
}

void RootFile::removeCheckpointStepFile(std::int32_t isample) const
{
   removeStepFileInternal(isample, true);
}

void RootFile::removeStepFileInternal(std::int32_t isample, bool checkpoint) const
{
   std::shared_ptr<StepFile> stepFile = std::make_shared<StepFile>(isample, m_prefix, m_extension, false, checkpoint);
   stepFile->remove(true, true, true);

   std::string stepFileName = stepFile->getStepFileName();
   std::string tagPrefix = checkpoint ? CHECKPOINT_STEP_PREFIX : SAMPLE_STEP_PREFIX;
   std::string stepTag = "# removed " + tagPrefix + std::to_string(isample);
   appendToRootFile(stepTag, stepFileName);
}

std::shared_ptr<StepFile> RootFile::openLastStepFile() const
{
   std::string lastCheckpointItem;
   std::string lastStepItem;

   for (auto& item : m_iniStorage)
   {
      if (startsWith(item.first, CHECKPOINT_STEP_PREFIX))
         lastCheckpointItem = item.second;

      if (startsWith(item.first, SAMPLE_STEP_PREFIX))
         lastStepItem = item.second;
   }

   //try open sample file
   //if no sample file then try open checkpoint file
   //if no checkpoint file then return empty file
   if (lastStepItem.empty())
   {
      if (lastCheckpointItem.empty())
      {
         return std::shared_ptr<StepFile>();
      }
      else
      {
         return std::make_shared<StepFile>(lastCheckpointItem, m_prefix, m_extension);
      }  
   }
   else
   {
      return std::make_shared<StepFile>(lastStepItem, m_prefix, m_extension);
   }   
}

/*
std::shared_ptr<StepFile> RootFile::openSampleStepFile(std::int32_t isample) const
{
   std::shared_ptr<StepFile> stepFile = std::make_shared<StepFile>(isample, m_prefix, m_extension, false, false);
   return stepFile;
}

std::shared_ptr<StepFile> RootFile::openSampleStepFile(std::string path) const
{
   std::shared_ptr<StepFile> stepFile = std::make_shared<StepFile>(path, m_prefix, m_extension);
   return stepFile;
}
*/

void RootFile::flush() const
{
   std::string configPath = getRootFileName();

   std::ofstream rootFile;
   rootFile.open(configPath, std::ios::trunc);

   for (auto item : m_iniStorage)
   {
      rootFile << item.first << " = " << item.second << std::endl;
   }

   rootFile.close();
}

void RootFile::flushLast() const
{
   auto& last = m_iniStorage.back();

   std::string configPath = getRootFileName();

   std::ofstream rootFile;
   rootFile.open(configPath, std::ios::app);
   rootFile << last.first << " = " << last.second << std::endl;
   rootFile.close();
}

//AGE: to properly implement this we need smth like boost::adaptors::filtered
/*
std::vector<std::pair<std::string, std::string> >::const_iterator RootFile::stepFilesBegin() const
{
   auto it = m_iniStorage.begin();
   std::advance(it, 1);
   return it;
}

std::vector<std::pair<std::string, std::string> >::const_iterator RootFile::stepFilesEnd() const
{
   return m_iniStorage.end();
}
*/

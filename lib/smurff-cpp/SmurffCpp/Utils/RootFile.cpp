#include "RootFile.h"

#include <iostream>
#include <fstream>

#include <SmurffCpp/IO/INIReader.h>
#include <SmurffCpp/Utils/Error.h>
#include <SmurffCpp/Utils/StringUtils.h>
#include <SmurffCpp/Utils/IniUtils.h>

#define OPTIONS_TAG "options"
#define BURNIN_STEP_PREFIX "burnin_step_"
#define SAMPLE_STEP_PREFIX "sample_step_"

using namespace smurff;

RootFile::RootFile(std::string path)
   : m_path(path)
{
   //load all entries in ini file to be able to go through step variables
   loadIni(m_path, m_iniStorage);

   //AGE: I know that this is an extra call to restoreConfig
   //however it is a small price to pay for making constructor being solid (assigning all fields here, instead of restoreConfig)
   Config config;
   restoreConfig(config);

   m_prefix = config.getSavePrefix();
   THROWERROR_ASSERT_MSG(!m_prefix.empty(), "Save prefix is empty");

   m_extension = config.getSaveExtension();
   THROWERROR_ASSERT_MSG(!m_extension.empty(), "Save extension is empty");
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

void RootFile::restoreConfig(Config& config)
{
   THROWERROR_ASSERT_MSG(!m_iniStorage.empty(), "Root ini file is not loaded");

   auto optionsIt = iniFind(m_iniStorage, OPTIONS_TAG);
   THROWERROR_ASSERT_MSG(optionsIt != m_iniStorage.end(), "Options tag is not found in root ini file");
   
   std::string optionsFileName = optionsIt->second;

   bool success = config.restore(optionsFileName);
   THROWERROR_ASSERT_MSG(success, "Could not load ini file '" + optionsFileName + "'");
}

std::shared_ptr<StepFile> RootFile::createSampleStepFile(std::int32_t isample) const
{
   return createStepFileInternal(isample, false);
}

std::shared_ptr<StepFile> RootFile::createBurninStepFile(std::int32_t isample) const
{
   return createStepFileInternal(isample, true);
}

std::shared_ptr<StepFile> RootFile::createStepFileInternal(std::int32_t isample, bool burnin) const
{
   std::shared_ptr<StepFile> stepFile = std::make_shared<StepFile>(isample, m_prefix, m_extension, true, burnin);

   std::string stepFileName = stepFile->getStepFileName();
   std::string tagPrefix = burnin ? BURNIN_STEP_PREFIX : SAMPLE_STEP_PREFIX;
   std::string stepTag = tagPrefix + std::to_string(isample);
   appendToRootFile(stepTag, stepFileName);

   return stepFile;
}

void RootFile::removeSampleStepFile(std::int32_t isample) const
{
   removeStepFileInternal(isample, false);
}

void RootFile::removeBurninStepFile(std::int32_t isample) const
{
   removeStepFileInternal(isample, true);
}

void RootFile::removeStepFileInternal(std::int32_t isample, bool burnin) const
{
   std::shared_ptr<StepFile> stepFile = std::make_shared<StepFile>(isample, m_prefix, m_extension, false, burnin);
   stepFile->remove(true, true, true);
}

std::shared_ptr<StepFile> RootFile::openLastStepFile() const
{
   std::string lastBurninItem;
   std::string lastStepItem;

   for (auto& item : m_iniStorage)
   {
      if (startsWith(item.first, BURNIN_STEP_PREFIX))
         lastBurninItem = item.second;

      if (startsWith(item.first, SAMPLE_STEP_PREFIX))
         lastStepItem = item.second;
   }

   //try open sample file
   //if no sample file then try open burnin file
   //if no burnin file then return empty file
   if (lastStepItem.empty())
   {
      if (lastBurninItem.empty())
      {
         return std::shared_ptr<StepFile>();
      }
      else
      {
         return std::make_shared<StepFile>(lastBurninItem, m_prefix, m_extension);
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
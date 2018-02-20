#include "RootFile.h"

#include <iostream>
#include <fstream>

#include <SmurffCpp/IO/INIReader.h>
#include <SmurffCpp/Utils/Error.h>
#include <SmurffCpp/Utils/StringUtils.h>
#include <SmurffCpp/Utils/IniUtils.h>

#define OPTIONS_TAG "options"
#define STEP_PREFIX "step_"

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
   std::string configPath = getRootFileName();

   std::ofstream rootFile;
   rootFile.open(configPath, std::ios::app);
   rootFile << tag << " = " << value << std::endl;
   rootFile.close();

   m_iniStorage.push_back(std::make_pair(tag, value));
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

std::shared_ptr<StepFile> RootFile::createStepFile(std::int32_t isample) const
{
   std::shared_ptr<StepFile> stepFile = std::make_shared<StepFile>(isample, m_prefix, m_extension, true);
   std::string stepFileName = stepFile->getStepFileName();
   std::string stepTag = STEP_PREFIX + std::to_string(isample);
   appendToRootFile(stepTag, stepFileName);

   return stepFile;
}

std::shared_ptr<StepFile> RootFile::openLastStepFile() const
{
   std::string lastItem;

   for (auto& item : m_iniStorage)
   {
      if (startsWith(item.first, STEP_PREFIX))
         lastItem = item.second;
   }

   if (lastItem.empty())
      return std::shared_ptr<StepFile>();
   else
      return std::make_shared<StepFile>(lastItem, m_prefix, m_extension);
}

std::shared_ptr<StepFile> RootFile::openStepFile(std::int32_t isample) const
{
   std::shared_ptr<StepFile> stepFile = std::make_shared<StepFile>(isample, m_prefix, m_extension, false);
   return stepFile;
}

std::shared_ptr<StepFile> RootFile::openStepFile(std::string path) const
{
   std::shared_ptr<StepFile> stepFile = std::make_shared<StepFile>(path, m_prefix, m_extension);
   return stepFile;
}

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
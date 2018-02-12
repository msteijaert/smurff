#include "RootFile.h"

#include <iostream>
#include <fstream>

#include <SmurffCpp/IO/INIReader.h>
#include <SmurffCpp/Utils/Error.h>

using namespace smurff;

RootFile::RootFile(std::string path)
   : m_path(path)
{
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

void RootFile::appendToRootFile(std::string tag, std::string value, bool truncate) const
{
   std::string configPath = getRootFileName();

   std::ofstream rootFile;

   if (truncate)
      rootFile.open(configPath, std::ios::out | std::ios::trunc);
   else
      rootFile.open(configPath, std::ios::out | std::ios::app);

   rootFile << tag << " = " << value << std::endl;
   rootFile.close();
}

void RootFile::saveConfig(Config& config)
{
   std::string configPath = getOptionsFileName();
   config.save(configPath);
   appendToRootFile("options", configPath, true);
}

void RootFile::restoreConfig(Config& config)
{
   INIReader reader(m_path);
   THROWERROR_ASSERT_MSG(reader.ParseError() >= 0, "Can't load '" + m_path + "'\n");

   std::string optionsFileName = reader.Get("", "options", "");
   THROWERROR_ASSERT_MSG(!optionsFileName.empty(), "Can't load '" + m_path + "'\n");

   bool success = config.restore(optionsFileName);
   THROWERROR_ASSERT_MSG(success, "Could not load ini file '" + optionsFileName + "'");
}

std::shared_ptr<StepFile> RootFile::createStepFile(int isample) const
{
   std::shared_ptr<StepFile> stepFile = std::make_shared<StepFile>(isample, m_prefix, m_extension);
   std::string stepFileName = stepFile->getStepFileName();
   appendToRootFile("step_" + std::to_string(isample), stepFileName, false);
   return stepFile;
}

std::shared_ptr<StepFile> RootFile::getLastStepFile() const
{
   INIReader reader(m_path);
   THROWERROR_ASSERT_MSG(reader.ParseError() >= 0, "Can't load '" + m_path + "'\n");

   int isample = 1;
   while (true)
   {
      std::string stepFile = reader.Get("", "step_" + std::to_string(isample), "<invalid>");
      if (stepFile == "<invalid>")
      {
         isample--;
         break;
      }
      isample++;
   }

   if (isample < 1)
      return std::shared_ptr<StepFile>();

   std::shared_ptr<StepFile> stepFile = std::make_shared<StepFile>(isample, m_prefix, m_extension);
   return stepFile;
}
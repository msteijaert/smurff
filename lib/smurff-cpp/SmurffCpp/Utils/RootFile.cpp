#include "RootFile.h"

#include <iostream>
#include <fstream>

using namespace smurff;

RootFile::RootFile(std::string path)
   : m_path(path)
{
}

RootFile::RootFile(std::string prefix, std::string extension)
   : m_prefix(prefix), m_extension(extension)
{
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

void RootFile::saveConfig(Config config)
{
   std::string configPath = getOptionsFileName();
   config.save(configPath);
   appendToRootFile("options", configPath, true);
}

std::shared_ptr<StepFile> RootFile::createStepFile(int isample) const
{
   std::shared_ptr<StepFile> stepFile = std::make_shared<StepFile>(isample, m_prefix, m_extension);
   std::string stepFileName = stepFile->getStepFileName();
   appendToRootFile("step_" + std::to_string(isample), stepFileName, false);
   return stepFile;
}
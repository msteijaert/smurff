#include "RootFile.h"

#include <iostream>
#include <fstream>

#include <SmurffCpp/Utils/Error.h>
#include <SmurffCpp/Utils/StringUtils.h>
#include <SmurffCpp/IO/GenericIO.h>
#include <SmurffCpp/StatusItem.h>

#define OPTIONS_TAG "options"
#define STEPS_TAG "steps"
#define STATUS_TAG "status"
#define CHECKPOINT_STEP_PREFIX "checkpoint_step_"
#define SAMPLE_STEP_PREFIX "sample_step_"

using namespace smurff;

RootFile::RootFile(std::string path)
   : m_path(path)
{
   //load all entries in ini file to be able to go through step variables
   m_iniReader = std::make_shared<INIFile>();
   m_iniReader->open(m_path);

   //lightweight restore of prefix and extension
   restoreState(m_prefix, m_extension);
}

RootFile::RootFile(std::string prefix, std::string extension)
   : m_prefix(prefix), m_extension(extension)
{
   m_path = getRootFileName();

   //create root file
   m_iniReader = std::make_shared<INIFile>();
   m_iniReader->create(m_path);

}

std::string RootFile::getPrefix() const
{
   return m_prefix;
}

std::string RootFile::getRootFileName() const
{
   return m_prefix + "-root.ini";
}

std::string RootFile::getOptionsFileName() const
{
   return m_prefix + "-options.ini";
}

std::string RootFile::getCsvStatusFileName() const
{
   return m_prefix + "-status.csv";
}



void RootFile::appendToRootFile(std::string section, std::string tag, std::string value) const
{
    if (m_cur_section != section) {
      m_iniReader->startSection(section);
      m_cur_section = section;
   }
   
   m_iniReader->appendItem(section, tag, value);

   //this function should be used here when it is required to write to root file immediately
   //currently we call flushLast method separately
   //this guarantees integrity of a step - step file entry is written after everything else is calculated/written
   //flushLast();
}

void RootFile::appendCommentToRootFile(std::string comment) const
{
   m_iniReader->appendComment(comment);
}

void RootFile::createCsvStatusFile() const
{
    //write header to status file
    const std::string statusPath = getCsvStatusFileName();
    std::ofstream csv_out(getCsvStatusFileName(), std::ofstream::out);
    csv_out << StatusItem::getCsvHeader() << std::endl;
    appendToRootFile(STATUS_TAG, STATUS_TAG, statusPath);
}

void RootFile::addCsvStatusLine(const StatusItem &status_item) const
{
    const std::string statusPath = getCsvStatusFileName();
    std::ofstream csv_out(statusPath, std::ofstream::out | std::ofstream::app);
    THROWERROR_ASSERT_MSG(csv_out, "Could not open status csv file: " + statusPath);;
    csv_out << status_item.asCsvString() << std::endl;
}

void RootFile::saveConfig(Config& config)
{
   std::string configPath = getOptionsFileName();
   config.save(configPath);
   appendToRootFile(OPTIONS_TAG, OPTIONS_TAG, configPath);
}

std::string RootFile::restoreGetOptionsFileName() const
{
   THROWERROR_ASSERT_MSG(m_iniReader, "Root ini file is not loaded");

   return m_iniReader->get(OPTIONS_TAG, OPTIONS_TAG);
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
   appendToRootFile(STEPS_TAG, stepTag, stepFileName);

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
   appendCommentToRootFile(stepTag + " " + stepFileName);
}

std::shared_ptr<StepFile> RootFile::openLastStepFile() const
{
   std::string lastCheckpointItem;
   std::string lastStepItem;

   for (auto& section : m_iniReader->getSections())
   {
      m_cur_section = section;

      auto fieldsIt = m_iniReader->getFields(section);
      for (auto& field : fieldsIt->second)
      {
         if (startsWith(field, CHECKPOINT_STEP_PREFIX))
            lastCheckpointItem = m_iniReader->get(section, field);

         if (startsWith(field, SAMPLE_STEP_PREFIX))
            lastStepItem = m_iniReader->get(section, field);
      }
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


std::vector<std::shared_ptr<StepFile>> RootFile::openSampleStepFiles() const
{
   std::vector<std::shared_ptr<StepFile>> samples;

   for (auto& section : m_iniReader->getSections())
   {
      m_cur_section = section;
      auto fieldsIt = m_iniReader->getFields(section);
      for (auto& field : fieldsIt->second)
      {
         if (!startsWith(field, SAMPLE_STEP_PREFIX))
            continue;

         std::string stepItem = m_iniReader->get(section, field);

         if (stepItem.empty())
             continue;

         samples.push_back(std::make_shared<StepFile>(stepItem, m_prefix, m_extension));
      }
   }


   return samples;
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

void RootFile::flushLast() const
{
   m_iniReader->flush();
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

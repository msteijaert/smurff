#pragma once

#include <string>
#include <memory>
#include <vector>
#include <cstdint>

namespace smurff {

   class Model;
   class Result;
   class ILatentPrior;

   class StepFile : public std::enable_shared_from_this<StepFile>
   {
   private:
      std::int32_t m_isample;
      std::string m_prefix;
      std::string m_extension;

   public:
      //this constructor should be used to create a step file on a first run of session
      StepFile(std::int32_t isample, std::string prefix, std::string extension);

      //this constructor should be used to  open existing step file when previous session is continued
      StepFile(const std::string& path, std::string prefix, std::string extension);

   private:
      std::string getSamplePrefix() const;

      std::int32_t StepFile::getIsampleFromPath(const std::string& path) const;

   public:
      std::string getStepFileName() const;

   public:
      std::string getModelFileName(std::uint64_t index) const;
      std::string getPriorFileName(std::uint32_t mode) const;
      std::string getPredFileName() const;

   public:
      void saveModel(std::shared_ptr<const Model> model) const;
      void savePred(std::shared_ptr<const Result> m_pred) const;
      void savePriors(const std::vector<std::shared_ptr<ILatentPrior> >& priors) const;

   public:
      void restoreModel(std::shared_ptr<Model> model) const;
      void restorePred(std::shared_ptr<Result> m_pred) const;
      void restorePriors(std::vector<std::shared_ptr<ILatentPrior> >& priors) const;

   public:
      void save(std::shared_ptr<const Model> model, std::shared_ptr<const Result> pred, const std::vector<std::shared_ptr<ILatentPrior> >& priors) const;

      void restore(std::shared_ptr<Model> model, std::shared_ptr<Result> m_pred, std::vector<std::shared_ptr<ILatentPrior> >& priors) const;

   public:
      std::int32_t getIsample() const;

      std::int32_t getNSamples() const;
   };
}
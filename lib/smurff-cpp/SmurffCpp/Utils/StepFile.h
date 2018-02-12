#pragma once

#include <string>
#include <memory>
#include <vector>

namespace smurff {

   class Model;
   struct Result;
   class ILatentPrior;

   class StepFile : public std::enable_shared_from_this<StepFile>
   {
   private:
      int m_isample;
      std::string m_prefix;
      std::string m_extension;

   public:
      StepFile(int isample, std::string prefix, std::string extension);

   public:
      std::string getSamplePrefix() const;

   public:
      std::string getStepFileName() const;

   public:
      std::string getModelFileName(std::uint64_t index) const;
      std::string getPriorFileName(std::uint32_t mode) const;
      std::string getPredFileName() const;

   private:
      void saveModel(std::shared_ptr<Model> model) const;
      void savePred(std::shared_ptr<Result> m_pred) const;
      void savePriors(std::vector<std::shared_ptr<ILatentPrior> >& priors) const;

   public:
      void save(std::shared_ptr<Model> model, std::shared_ptr<Result> pred, std::vector<std::shared_ptr<ILatentPrior> >& priors) const;
   };
}
#pragma once

#include <string>
#include <memory>
#include <vector>
#include <set>
#include <cstdint>

#include <SmurffCpp/IO/INIFile.h>

namespace smurff {

   class Model;
   class Result;
   class ILatentPrior;
   class MatrixConfig;

   class StepFile : public std::enable_shared_from_this<StepFile>
   {
   private:
      std::int32_t m_isample;
      std::string m_prefix;
      std::string m_extension;
      bool m_checkpoint;
      bool m_final;

      mutable std::string m_cur_section;

      //preserves order of elements in the file
      mutable std::shared_ptr<INIFile> m_iniReader;

   public:
      //this constructor should be used to create a step file on a first run of session
      StepFile(std::int32_t isample, std::string prefix, std::string extension, bool create, bool checkpoint, bool final);

      //this constructor should be used to  open existing step file when previous session is continued
      StepFile(const std::string& path, std::string prefix, std::string extension);

   private:
      std::string getStepPrefix() const;

   public:
      bool isBinary() const;
      std::string getStepFileName() const;

   public:
      bool hasModel(std::uint64_t index) const;
      bool hasMu(std::uint64_t index) const;
      bool hasLinkMatrix(std::uint32_t mode) const;
      bool hasPred() const;

      std::string getModelFileName(std::uint64_t index) const;
      std::string getMuFileName(std::uint64_t index) const;
      std::string getLinkMatrixFileName(std::uint32_t mode) const;
      std::string getPredFileName() const;
      std::string getPredStateFileName() const;
      std::string getPredAvgFileName() const;
      std::string getPredVarFileName() const;

      std::string makeModelFileName(std::uint64_t index) const;
      std::string makeLinkMatrixFileName(std::uint32_t mode) const;
      std::string makeMuFileName(std::uint32_t mode) const;
      std::string makePredFileName() const;
      std::string makePredStateFileName() const;
      std::string makePredAvgFileName() const;
      std::string makePredVarFileName() const;

      std::string makePostMuFileName(std::uint64_t index) const;
      std::string makePostLambdaFileName(std::uint64_t index) const;

   public:
      void saveModel(std::shared_ptr<const Model> model, bool saveAggr) const;
      void savePred(std::shared_ptr<const Result> m_pred) const;
      void savePriors(const std::vector<std::shared_ptr<ILatentPrior> >& priors) const;

      void save(std::shared_ptr<const Model> model, std::shared_ptr<const Result> pred, const std::vector<std::shared_ptr<ILatentPrior> >& priors) const;

   public:
      void restoreModel(std::shared_ptr<Model> model, int skip_mode = -1) const;
      void restorePred(std::shared_ptr<Result> m_pred) const;
      void restorePriors(std::vector<std::shared_ptr<ILatentPrior> >& priors) const;
      
      //-- used in PredictSession
      std::shared_ptr<Model> restoreModel(int skip_mode = -1) const;

      void restore(std::shared_ptr<Model> model, std::shared_ptr<Result> pred, std::vector<std::shared_ptr<ILatentPrior> >& priors) const;

   public:
      void removeModel() const;
      void removePred() const;
      void removePriors() const;

      void remove(bool model, bool pred, bool priors) const;

   public:
      std::int32_t getIsample() const;

      bool isCheckpoint() const;

   public:
      std::int32_t getNModes() const;

   public:
      std::string getIniValueBase(const std::string& section, const std::string& tag) const;

      bool hasIniValueBase(const std::string &section, const std::string& tag) const;
      std::pair<bool, std::string> tryGetIniValueBase(const std::string& section, const std::string& tag) const;

      std::pair<bool, std::string> tryGetIniValueFullPath(const std::string& section, const std::string& tag) const;

      void appendToStepFile(std::string section, std::string tag, std::string value) const;

      void appendCommentToStepFile(std::string comment) const;

      void removeFromStepFile(std::string section, std::string tag) const;

      void flushLast() const;
   };
}

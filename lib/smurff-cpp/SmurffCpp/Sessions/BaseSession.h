#pragma once

#include <string>
#include <vector>
#include <memory>

#include <SmurffCpp/Sessions/ISession.h>
#include <SmurffCpp/Utils/Error.h>

namespace smurff {

class ILatentPrior;
class Data;
class Model;
class SessionFactory;
struct Result;

class BaseSession : public ISession
{
   friend class SessionFactory;

protected:
   std::shared_ptr<Model> m_model;
   std::shared_ptr<Result> m_pred;

protected:
   std::vector<std::shared_ptr<ILatentPrior> > m_priors;
   std::string name;

protected:
   bool is_init = false;

   //train data
   std::shared_ptr<Data> data_ptr;

protected:
   BaseSession();

public:
   virtual ~BaseSession() {}

public:
   std::shared_ptr<Data> data() const
   {
      THROWERROR_ASSERT(data_ptr != 0);

      return data_ptr;
   }

   std::shared_ptr<const Model> model() const
   {
      return m_model;
   }

   std::shared_ptr<Model> model()
   {
      return m_model;
   }

public:
   void addPrior(std::shared_ptr<ILatentPrior> prior);

public:
   void step() override;

public:
   virtual std::ostream &info(std::ostream &, std::string indent);

   std::string save(std::string prefix, std::string suffix);

   void restore(std::string prefix, std::string suffix);

public:
   std::shared_ptr<std::vector<ResultItem> > getResult() override;
   MatrixConfig getSample(int dim) override;
   double getRmseAvg() override;

private:
   std::string getRootFileName(std::string prefix) const;
};

}

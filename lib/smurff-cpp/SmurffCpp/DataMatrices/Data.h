#pragma once

#include <memory>
#include <vector>
#include <string>
#include <iostream>

#include <SmurffCpp/Noises/INoiseModel.h>
#include <SmurffCpp/Utils/PVec.hpp>

#include <Eigen/Core>

#include <SmurffCpp/model.h>
#include <SmurffCpp/Configs/Config.h>

namespace smurff
{

   //AGE: This is a fake interface that allows to isolate all code dependencies on matrix centring funcionality

   class IDataCentringBaseFake
   {
   public:
      virtual ~IDataCentringBaseFake(){}

   public:
      static CenterModeTypes stringToCenterMode(std::string c)
      {
         return CenterModeTypes::CENTER_INVALID;
      }

      void setCenterMode(std::string c)
      {

      }

      void setCenterMode(CenterModeTypes type)
      {

      }

      double offset_to_mean(const PVec<>& pos) const
      {
         return 0.0;
      }

      double getModeMeanItem(int m, int c) const
      {
         return 0.0;
      }

      double getGlobalMean() const
      {
         return 0.0;
      }

      std::string getCenterModeName() const
      {
         return std::string();
      }

      double getCwiseMean() const
      {
         return 0.0;
      }

      bool getMeanComputed() const
      {
         return false;
      }
   };

   //AGE: This is a fake interface that allows to isolate all code dependencies on matrix centring funcionality

   template<typename YType>
   class IDataCentringFake : public IDataCentringBaseFake
   {
   private:
      std::shared_ptr<std::vector<YType> > Ycentered;

   public:
      const std::vector<YType>& getYc() const
      {
         return *Ycentered.get();
      }

      std::shared_ptr<std::vector<YType> > getYcPtr() const
      {
         return Ycentered;
      }
   };

   class Data
   {
      //AGE: Only MatricesData should call init methods, center methods etc
      friend class MatricesData;

   public:
      std::string name;

   //AGE: this is a temp fake field to hold fake interface to isolate matrix centring dependencies

   private:
      std::shared_ptr<IDataCentringBaseFake> m_center;

   public:
      template<typename T>
      std::shared_ptr<IDataCentringFake<T> > getCenter() const
      {
         return std::dynamic_pointer_cast<IDataCentringFake<T> >(m_center);
      }

      std::shared_ptr<IDataCentringBaseFake> getCenter() const
      {
         return m_center;
      }

   protected:
      Data();

   public:
      virtual ~Data(){}

   protected:
      virtual void init_pre() = 0;
      virtual void init_post();

   public:
      virtual void init();
      virtual void update(const SubModel& model);

   //#### arithmetic functions ####
   public:
      virtual double sum() const = 0;

   //#### prediction functions ####
   public:
      virtual double predict(const PVec<>& pos, const SubModel& model) const;

   //#### dimention functions ####
   public:
      virtual int nmode() const = 0; // number of dimensions
      virtual int nnz() const = 0; // number of non zero elements
      virtual int nna() const = 0; // number of NA elements
      virtual PVec<> dim() const = 0; // dimension vector

   public:
      int size() const; // number of all elements (dimension dot product)
      int dim(int m) const; // size of dimension

   //#### view functions ####

   public:
      virtual int nview(int mode) const;
      virtual int view(int mode, int pos) const;
      virtual int view_size(int m, int v) const;

   //#### noise, precision, mean functions ####

   private:
      std::unique_ptr<INoiseModel> noise_ptr; // noise model for this data

   public:
      virtual double train_rmse(const SubModel& model) const = 0;
      virtual void get_pnm(const SubModel& model, int mode, int d, Eigen::VectorXd& rr, Eigen::MatrixXd& MM) = 0;
      virtual void update_pnm(const SubModel& model, int mode) = 0;

   public:
      virtual double sumsq(const SubModel& model) const = 0;
      virtual double var_total() const = 0;

   public:
      INoiseModel& noise() const;
      void setNoiseModel(INoiseModel* nm);

   //#### info functions ####
   public:
      virtual std::ostream& info(std::ostream& os, std::string indent);
      virtual std::ostream& status(std::ostream& os, std::string indent) const;
   };
}

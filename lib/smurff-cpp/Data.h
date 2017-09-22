#pragma once

#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <Eigen/Core>

#include "model.h"
#include "INoiseModel.h"
#include "PVec.h"

namespace smurff
{
   class IDataArithmetic
   {
   protected:
      IDataArithmetic() {}

   public:
      virtual ~IDataArithmetic() {}

   public:
      virtual double sum() const = 0;
   };

   class IDataDimensions
   {
   protected:
      IDataDimensions() {}

   public:
      virtual ~IDataDimensions() {}

   public:
      virtual int nmode() const = 0; // number of dimensions
      virtual int nnz() const = 0; // number of non zero elements
      virtual int nna() const = 0; // number of NA elements
      virtual PVec dim() const = 0; // dimension vector

      int size() const; // number of all elements (dimension dot product)
      int dim(int m) const; // size of dimension
   };

   class IDataBase : public IDataArithmetic, public IDataDimensions
   {
   protected:
      IDataBase()
         : IDataArithmetic()
         , IDataDimensions()
      {
      }

   public:
      virtual ~IDataBase() {}
   };

   class IMeanCentering
   {
   public:
      enum class CenterModeTypes : int
      {
         CENTER_INVALID = -10,
         CENTER_NONE = -3,
         CENTER_GLOBAL = -2,
         CENTER_VIEW = -1,
         CENTER_COLS = 0,
         CENTER_ROWS = 1
      };

   private:
      const IDataBase* m_dataBase;

      double m_cwise_mean = NAN;
      double m_global_mean = NAN;
      double m_var = NAN;
      CenterModeTypes m_center_mode;

   private:
      std::vector<Eigen::VectorXd> m_mode_mean;
      bool m_mean_computed = false;
      bool m_centered = false;

   public:
      void init_pre_mean_centering()
      {
         m_cwise_mean = m_dataBase->sum() / (m_dataBase->size() - m_dataBase->nna());
      }

   protected:
      IMeanCentering(const IDataBase* dataBase)
         : m_dataBase(dataBase)
         , m_center_mode(CenterModeTypes::CENTER_INVALID)
      {
      }

   public:
      virtual ~IMeanCentering() {}

   protected:
      virtual double compute_mode_mean(int m, int c) = 0;

   public:
      virtual void center(double upper_mean)
      {
         assert(!m_centered);
         m_global_mean = upper_mean;
         m_centered = true;
      }

      virtual double offset_to_mean(const PVec& pos) const = 0;

   public:
      virtual void setCenterMode(std::string c);

   public:
      double mean(int m, int c) const;
      void compute_mode_mean();

   public:
      double getCwiseMean() const
      {
         return m_cwise_mean;
      }

      double getGlobalMean() const
      {
         return m_global_mean;
      }

      void setGlobalMean(double value)
      {
         m_global_mean = value;
      }

      double getVar() const
      {
         return m_var;
      }

      CenterModeTypes getCenterMode() const
      {
         return m_center_mode;
      }

      bool getMeanComputed() const
      {
         return m_mean_computed;
      }

      const Eigen::VectorXd& getModeMean(size_t i)
      {
         return m_mode_mean.at(i);
      }

   public:
      std::string getCenterModeName() const
      {
         switch (m_center_mode)
         {
            case CenterModeTypes::CENTER_INVALID:
               throw std::runtime_error("Invalid center mode");
            case CenterModeTypes::CENTER_NONE:
               return "none";
            case CenterModeTypes::CENTER_GLOBAL:
               return "global";
            case CenterModeTypes::CENTER_VIEW:
               return "view";
            case CenterModeTypes::CENTER_COLS:
               return "cols";
            case CenterModeTypes::CENTER_ROWS:
               return "rows";
            default:
               throw std::runtime_error("Invalid center mode");
         }
      }
   };

   class Data : public IDataBase, public IMeanCentering
   {
   public:
      // noise model for this dataset
      std::unique_ptr<INoiseModel> noise_ptr;

      // name
      std::string name;

   public:
      Data();
      virtual ~Data();

   public:
      // init
      virtual void init_pre() = 0;
      virtual void init_post();
      virtual void init();

      // helper functions for noise
      virtual double sumsq(const SubModel& model) const = 0;
      virtual double var_total() const = 0;
      INoiseModel& noise() const;

      // update noise and precision/mean
      virtual double train_rmse(const SubModel& model) const = 0;
      virtual void update(const SubModel& model);
      virtual void get_pnm(const SubModel& model, int mode, int d, Eigen::VectorXd& rr, Eigen::MatrixXd& MM) = 0;
      virtual void update_pnm(const SubModel& model, int mode) = 0;

      //-- print info
      virtual std::ostream& info(std::ostream& os, std::string indent);
      virtual std::ostream& status(std::ostream& os, std::string indent) const;

      virtual int nview(int mode) const;
      virtual int view(int mode, int pos) const;
      virtual int view_size(int m,int) const;

      // helper for predictions
      double predict(const PVec& pos, const SubModel& model) const;
   };
}

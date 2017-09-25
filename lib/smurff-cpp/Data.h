#pragma once

#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <Eigen/Core>

#include "model.h"
#include "INoiseModel.h"
#include "PVec.h"

#define CENTER_MODE_STR_NONE "none"
#define CENTER_MODE_STR_GLOBAL "global"
#define CENTER_MODE_STR_VIEW "view"
#define CENTER_MODE_STR_ROWS "rows"
#define CENTER_MODE_STR_COLS "cols"

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

   // data and dimentions related
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

   class Data;

   // mean and centering
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

   protected:
      double m_cwise_mean = NAN;

   private:
      double m_global_mean = NAN;
      double m_var = NAN;
      CenterModeTypes m_center_mode;

   protected:
      std::vector<Eigen::VectorXd> m_mode_mean;

   protected:
      bool m_mean_computed = false;

   private:
      bool m_centered = false;

   protected:
      IMeanCentering()
         : m_center_mode(CenterModeTypes::CENTER_INVALID)
      {
      }

   public:
      virtual ~IMeanCentering() {}

   protected:
      virtual void compute_mode_mean() = 0;
      virtual double compute_mode_mean_mn(int mode, int pos) = 0;
      virtual void init_pre_mean_centering() = 0;

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
      static std::string centerModeToString(CenterModeTypes cm)
      {
         switch (cm)
         {
            case CenterModeTypes::CENTER_INVALID:
               return std::string();
            case CenterModeTypes::CENTER_NONE:
               return CENTER_MODE_STR_NONE;
            case CenterModeTypes::CENTER_GLOBAL:
               return CENTER_MODE_STR_GLOBAL;
            case CenterModeTypes::CENTER_VIEW:
               return CENTER_MODE_STR_VIEW;
            case CenterModeTypes::CENTER_ROWS:
               return CENTER_MODE_STR_ROWS;
            case CenterModeTypes::CENTER_COLS:
               return CENTER_MODE_STR_COLS;
            default:
               return std::string();
         }
      }

      static CenterModeTypes stringToCenterMode(std::string c)
      {
         if (c == CENTER_MODE_STR_NONE)
            return CenterModeTypes::CENTER_NONE;
         else if (c == CENTER_MODE_STR_GLOBAL)
            return CenterModeTypes::CENTER_GLOBAL;
         else if (c == CENTER_MODE_STR_VIEW)
            return CenterModeTypes::CENTER_VIEW;
         else if (c == CENTER_MODE_STR_ROWS)
            return CenterModeTypes::CENTER_ROWS;
         else if (c == CENTER_MODE_STR_COLS)
            return CenterModeTypes::CENTER_COLS;
         else
            return CenterModeTypes::CENTER_INVALID;
      }

   public:
      std::string getCenterModeName() const
      {
         std::string name = centerModeToString(m_center_mode);
         if(name.empty())
            throw std::runtime_error("Invalid center mode");
         return name;
      }
   };

   class IView
   {
   private:
      const IDataDimensions* m_data_dim;

   protected:
      IView(const IDataDimensions* data_dim)
         :  m_data_dim(data_dim)
      {}

      virtual ~IView(){}

   public:
      virtual int nview(int mode) const;
      virtual int view(int mode, int pos) const;
      virtual int view_size(int m, int v) const;
   };

   // update noise and precision/mean
   class INoisePrecisionMean
   {
   private:
      // noise model for this dataset
      std::unique_ptr<INoiseModel> noise_ptr;

   protected:
      INoisePrecisionMean();

   public:
      virtual ~ INoisePrecisionMean(){}

   public:
      virtual double train_rmse(const SubModel& model) const = 0;
      virtual void get_pnm(const SubModel& model, int mode, int d, Eigen::VectorXd& rr, Eigen::MatrixXd& MM) = 0;
      virtual void update_pnm(const SubModel& model, int mode) = 0;

   public:
      // helper functions for noise
      virtual double sumsq(const SubModel& model) const = 0;
      virtual double var_total() const = 0;

   public:
      virtual void update(const SubModel& model) = 0;

   public:
      INoiseModel& noise() const;

      void setNoiseModel(INoiseModel* nm)
      {
         noise_ptr.reset(nm);
      }
   };

   class Data : public IDataDimensions
              , public IDataArithmetic
              , public INoisePrecisionMean
              , public IMeanCentering
              , public IView
   {
   public:
      // name
      std::string name;

   public:
      Data()
      : IDataDimensions()
      , IDataArithmetic()
      , INoisePrecisionMean()
      , IMeanCentering()
      , IView(this)
   {
   }

   public:
      virtual ~Data(){}

   public:
      // init
      virtual void init_pre() = 0;
      virtual void init_post();
      virtual void init();

      //-- print info
      virtual std::ostream& info(std::ostream& os, std::string indent);
      virtual std::ostream& status(std::ostream& os, std::string indent) const;

      void update(const SubModel& model) override
      {
         noise().update(model, this);
      }

      void compute_mode_mean() override
      {
         assert(!m_mean_computed);
         m_mode_mean.resize(nmode());
         for (int m = 0; m < nmode(); ++m)
         {
             auto &M = m_mode_mean.at(m);
             M.resize(dim(m));
             for (int n = 0; n < dim(m); n++)
               M(n) = compute_mode_mean_mn(m, n);
         }
         m_mean_computed = true;
      }

      void init_pre_mean_centering()
      {
         m_cwise_mean = sum() / (size() - nna());
      }

      // helper for predictions
      double predict(const PVec& pos, const SubModel& model) const;
   };
}

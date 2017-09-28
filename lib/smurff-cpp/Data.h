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

      //AGE: algorithms for counting depend on matrix data implementation
   public:
      virtual int nmode() const = 0; // number of dimensions
      virtual int nnz() const = 0; // number of non zero elements
      virtual int nna() const = 0; // number of NA elements
      virtual PVec dim() const = 0; // dimension vector

   public:
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

   private:
      double m_cwise_mean = NAN; // mean of non NA elements in matrix
      bool m_cwise_mean_initialized = false;
      double m_global_mean = NAN;
      double m_var = NAN;
      CenterModeTypes m_center_mode;

   private:
      std::vector<Eigen::VectorXd> m_mode_mean;
      bool m_mean_computed = false;
      bool m_centered = false;

   protected:
      IMeanCentering();

   public:
      virtual ~IMeanCentering() {}

      //AGE: methods are pure virtual because they depend on multiple interfaces from Data class
      //introducing Data class dependency into this class is not a good idea
   protected:
      void compute_mode_mean_internal(const Data* data); //depends on cwise mean. Also depends on initial data from Ycentered (which is Y)
      virtual void compute_mode_mean() = 0;
      virtual double compute_mode_mean_mn(int mode, int pos) = 0;

      void init_cwise_mean_internal(const Data* data); //does not depend on any init
      virtual void init_cwise_mean() = 0;  //does not depend on any init

      //AGE: implementation depends on matrix data type
   protected:
      virtual void center(double upper_mean); //depends on cwise mean. Also depends on mod mean being calculated.

   public:
      virtual void setCenterMode(std::string c);
      virtual void setCenterMode(CenterModeTypes type);

      //AGE: implementation depends on matrix data type
   public:
      virtual double offset_to_mean(const PVec& pos) const = 0;

      //AGE: getters
   public:
      double getCwiseMean() const;
      double getGlobalMean() const;
      double getVar() const;
      CenterModeTypes getCenterMode() const;
      bool getMeanComputed() const;
      double getModeMeanItem(int m, int c) const;
      const Eigen::VectorXd& getModeMean(size_t i) const;
      std::string getCenterModeName() const;

   public:
      void setCentered(bool value)
      {
         m_centered = value;
      }

   public:
      static std::string centerModeToString(CenterModeTypes cm);

      static CenterModeTypes stringToCenterMode(std::string c);
   };

   class IView
   {
   private:
      const IDataDimensions* m_data_dim;

   protected:
      IView(const IDataDimensions* data_dim);

   public:
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

      //AGE: implementation depends on matrix data type
   public:
      virtual double train_rmse(const SubModel& model) const = 0;
      virtual void get_pnm(const SubModel& model, int mode, int d, Eigen::VectorXd& rr, Eigen::MatrixXd& MM) = 0;
      virtual void update_pnm(const SubModel& model, int mode) = 0;

   public:
      // helper functions for noise
      virtual double sumsq(const SubModel& model) const = 0;
      virtual double var_total() const = 0;

   protected:
      void update_internal(const Data* data, const SubModel& model);

   public:
      virtual void update(const SubModel& model) = 0;

   protected:
      void init_noise_internal(const Data* data);

   public:
      INoiseModel& noise() const;

      void setNoiseModel(INoiseModel* nm);
   };

   class IDataPredict
   {
   protected:
      IDataPredict(){}

   public:
      virtual ~IDataPredict(){};

   protected:
      double predict_internal(const Data* data, const PVec& pos, const SubModel& model) const;

   public:
      // helper for predictions
      virtual double predict(const PVec& pos, const SubModel& model) const = 0;
   };

   class Data : public IDataDimensions
              , public IDataArithmetic
              , public INoisePrecisionMean
              , public IMeanCentering
              , public IDataPredict
              , public IView
   {
      //AGE: Only MatricesData should call init methods, center methods etc
      friend class MatricesData;

   public:
      std::string name;

   protected:
      Data();

   public:
      virtual ~Data(){}

   protected:
      virtual void init_pre() = 0;
      virtual void init_post();

   public:
      virtual void init();

      void update(const SubModel& model) override;

      void compute_mode_mean() override;

      void init_cwise_mean() override;

      double predict(const PVec& pos, const SubModel& model) const override;

   public:
      // print info
      virtual std::ostream& info(std::ostream& os, std::string indent);
      virtual std::ostream& status(std::ostream& os, std::string indent) const;
   };
}

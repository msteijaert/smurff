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
   
   class Data : public IMeanCentering
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
      virtual void update(const SubModel& model);

   // #### mean centring functions  ####
   public:
      void compute_mode_mean() override;

      void init_cwise_mean() override;

   //#### arithmetic functions ####
   public:
      virtual double sum() const = 0;

   //#### prediction functions ####
   public:
      virtual double predict(const PVec& pos, const SubModel& model) const;

   //#### dimention functions ####
   public:
      virtual int nmode() const = 0; // number of dimensions
      virtual int nnz() const = 0; // number of non zero elements
      virtual int nna() const = 0; // number of NA elements
      virtual PVec dim() const = 0; // dimension vector

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

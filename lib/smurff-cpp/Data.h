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
   class Data
   {
   protected:
      std::vector<Eigen::VectorXd> mode_mean;
      bool mean_computed = false;
      bool centered = false;

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
      virtual void center(double upper_mean) = 0;
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

      // virtual functions data-related
      virtual int    nmode() const = 0;
      virtual int    nnz()   const = 0;
              int    size()  const;
      virtual int    nna()   const = 0;
      virtual PVec   dim()   const = 0;
      virtual double sum()   const = 0;
              int dim(int m) const;
      // for matrices (nmode() == 2)
      virtual int nrow()     const;
      virtual int ncol()     const;

      virtual int nview(int mode) const;
      virtual int view(int mode, int pos) const;
      virtual int view_size(int m,int) const;

      // mean & centering
      double cwise_mean = NAN, global_mean = NAN;
      double var = NAN;
      double mean(int m, int c) const;
      virtual double compute_mode_mean(int m, int c) = 0;
                void compute_mode_mean();
      virtual double offset_to_mean(const PVec& pos) const = 0;

      virtual void setCenterMode(std::string c);
      enum { CENTER_INVALID = -10, CENTER_NONE = -3, CENTER_GLOBAL = -2, CENTER_VIEW = -1, CENTER_COLS = 0, CENTER_ROWS = 1}
      center_mode;

      // helper for predictions
      double predict(const PVec& pos, const SubModel& model) const;
   };
}

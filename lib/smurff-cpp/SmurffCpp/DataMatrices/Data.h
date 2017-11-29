#pragma once

#include <memory>
#include <vector>
#include <string>
#include <iostream>

#include <SmurffCpp/Noises/INoiseModel.h>
#include <SmurffCpp/Utils/PVec.hpp>

#include <Eigen/Core>

#include <SmurffCpp/Model.h>
#include <SmurffCpp/Configs/Config.h>

namespace smurff
{
   class Data
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

   //#### arithmetic functions ####
   public:
      virtual double sum() const = 0;

   //#### prediction functions ####
   public:
      virtual double predict(const PVec<>& pos, const SubModel& model) const;

   //#### dimention functions ####
   public:
      virtual std::uint64_t nmode() const = 0; // number of dimensions
      virtual std::uint64_t nnz() const = 0; // number of non zero elements
      virtual std::uint64_t nna() const = 0; // number of NA elements
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
      std::shared_ptr<INoiseModel> noise_ptr; // noise model for this data

   public:
      virtual double train_rmse(const SubModel& model) const = 0;
      virtual void get_pnm(const SubModel& model, uint32_t mode, int d, Eigen::VectorXd& rr, Eigen::MatrixXd& MM) = 0;
      virtual void update_pnm(const SubModel& model, uint32_t mode) = 0;

   public:
      virtual double sumsq(const SubModel& model) const = 0;
      virtual double var_total() const = 0;

   public:
      std::shared_ptr<INoiseModel> noise() const;
      void setNoiseModel(std::shared_ptr<INoiseModel> nm);

   //#### info functions ####
   public:
      virtual std::ostream& info(std::ostream& os, std::string indent);
      virtual std::ostream& status(std::ostream& os, std::string indent) const;
   };
}

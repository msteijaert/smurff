#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

#include "mvnormal.h"
#include "linop.h"
#include "model.h"
#include "session.h"

namespace smurff {

class ILatentPrior 
{
public:
   BaseSession &session;
   int mode;
   std::string name = "xxxx";

   thread_vector<VectorNd> rrs;
   thread_vector<MatrixNNd> MMs;

public:
   // c-tor
   ILatentPrior(BaseSession &s, int m, std::string name = "xxxx");
   virtual ~ILatentPrior() {}
   virtual void init();

   // utility
   Model &model() const;
   Data  &data() const;
   INoiseModel &noise();
   Eigen::MatrixXd &U();
   Eigen::MatrixXd &V();
   int num_latent() const;
   int num_cols() const;

   virtual void save(std::string prefix, std::string suffix) = 0;
   virtual void restore(std::string prefix, std::string suffix) = 0;
   virtual std::ostream &info(std::ostream &os, std::string indent);
   virtual std::ostream &status(std::ostream &os, std::string indent) const = 0;

   // work
   virtual bool run_slave(); // returns true if some work happened...

   virtual void sample_latents();
   virtual void sample_latent(int n) = 0;

   //TODO: missing implementation
   void add(BaseSession &b);
};

//macau
/*
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include "mvnormal.h"
#include "linop.h"
#include "sparsetensor.h"

 // forward declarations
class FixedGaussianNoise;
class AdaptiveGaussianNoise;
class ProbitNoise;
class MatrixData;

// interface
class ILatentPrior {
  public:
    virtual void sample_latents(Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat, double mean_value,
                        const Eigen::MatrixXd &samples, double alpha, const int num_latent) = 0;
    virtual void sample_latents(FixedGaussianNoise& noise, Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat,
                        double mean_value, const Eigen::MatrixXd &samples, const int num_latent);
    virtual void sample_latents(AdaptiveGaussianNoise& noise, Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat,
                        double mean_value, const Eigen::MatrixXd &samples, const int num_latent);
    virtual void sample_latents(ProbitNoise& noise, Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat,
                        double mean_value, const Eigen::MatrixXd &samples, const int num_latent) = 0;
    // general functions (called from outside)
    void sample_latents(FixedGaussianNoise& noiseModel, MatrixData & matrixData,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent);
    void sample_latents(AdaptiveGaussianNoise& noiseModel, MatrixData & matrixData,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent);
    void sample_latents(ProbitNoise& noiseModel, MatrixData & matrixData,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent);
    // for tensor
    void sample_latents(FixedGaussianNoise& noiseModel, TensorData & data,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent);
    void sample_latents(AdaptiveGaussianNoise& noiseModel, TensorData & data,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent);
    virtual void sample_latents(ProbitNoise& noiseModel, TensorData & data,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent) = 0;
    virtual void sample_latents(double noisePrecision, TensorData & data,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent) = 0;

    void virtual update_prior(const Eigen::MatrixXd &U) {};
    virtual double getLinkNorm() { return NAN; };
    virtual double getLinkLambda() { return NAN; };
    virtual void saveModel(std::string prefix) {};
    virtual ~ILatentPrior() {};
};
*/
}

#ifndef GFA_H
#define GFA_H

#define EIGEN_RUNTIME_NO_MALLOC 1

#include <Eigen/Dense>
using namespace Eigen;
#include <random>

double randn(double);
auto nrandn(int n) -> decltype( Eigen::ArrayXd::NullaryExpr(n, std::ptr_fun(randn)) );

/*
 *
 Group factor analysis.

 @param Y Either 1. data sources paired in one mode: a list of M data
   matrices, where Y[[m]] is a matrix with N rows (samples) and D_m columns
   (features), or 2. data sources paired in two modes: a list with two
   elements similar to 1, where the first data view is paired in both modes
   (Y[[1]][[1]]==t(Y[[2]][[1]])). NOTE: All of these should be centered, so
   that the mean of each feature is zero. NOTE: The algorithm is roughly
   invariant to the scale of the data, but extreme values should be avoided.
   Data with roughly unit variance or similar scale is recommended.
 @param K A scalar. The number of components (i.e. latent variables).
   Recommended to be set high enough so that the sampler can determine the
   model complexity by shutting down excessive components.
 @param opts List of options; see function \code{\link{getDefaultOpts}}.
 @param projection Fixed projections. Only intended for sequential prediction
   use via function \code{\link{sequentialGfaPrediction}}.
 @param filename A string. If provided, will save the sampling chain to this
   file every 100 iterations
 @return The trained model, which is a list that contains the following
   elements (or, in the case of pairing in two modes, the elements are lists
   of length 2):
   \itemize{
   \item W: The projections (final posterior sample); D times K matrix.
   \item X: The latent variables (final sample); N times K matrix.
   \item Z: The spike-and-slab parameters (final sample); D times K matrix.
   \item r: The probability of slab in Z (final sample).
   \item rz: The probability of slab in the spike-and-slab prior of X
          (final sample).
   \item tau: The noise precisions (final sample); D-element vector.
   \item alpha: The precisions of the projection weights W (final sample);
          D times K matrix.
   \item beta: The precisions of the latent variables X (final sample);
          N times K matrix.
   \item groups: A list denoting which features belong to each data source.
   \item D: Data dimensionalities; M-element vector.
   \item K: The number of component inferred. May be less than inital K.
   \item posterior: the posterior samples of, by default, X, W and tau.
   }
   And the following elements:
   \itemize{
   \item cost: The likelihood of all the posterior samples.
   \item aic: The Akaike information criteria of all the posterior samples.
   \item opts: The options used for the GFA model.
   \item conv: An estimate of the convergence of the models reconstruction
   based on Geweke diagnostic. Values around 0.05 imply a converged model.
   \item time: The computational time (in seconds) used to sample the model.
   }
 @examples
 X <- matrix(rnorm(20*3),20,3)
 W <- matrix(rnorm(30*3),30,3)
 Y <- tcrossprod(X,W) + matrix(rnorm(20*30),20,30)
 res <- gfa(list(Y[,1:10],Y[,11:30]),K=5,opts=getDefaultOpts())
*/

class Model
{
	//GFA returns the trained model, which is a dict that contains the following elements:

	MatrixXd Z; //The mean of the latent variables; N times K matrix
	MatrixXd W; // The mean projections; D_i times K matrices
	VectorXd tau; //The mean precisions (inverse variance, so 1/tau gives the variances denoted by sigma in the paper); M-element vector

	VectorXd cost; //Vector collecting the variational lower bounds for each iteration

};

#endif

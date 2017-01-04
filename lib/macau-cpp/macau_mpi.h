#pragma once

#include <memory>
#include <Eigen/Dense>
#include "macau.h"
#include "linop.h"

void run_macau_mpi(Macau* macau, int world_rank);

template<class Prior>
void update_prior_mpi(Prior &prior, const Eigen::MatrixXd &U, int world_rank);

template<class Prior>
void sample_beta_mpi(Prior &prior, const Eigen::MatrixXd &U, int world_rank);

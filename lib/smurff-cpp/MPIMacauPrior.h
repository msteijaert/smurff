#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

#include "mvnormal.h"
#include "linop.h"
#include "model.h"
#include "session.h"
#include "data.h"

#include "MacauPrior.h"

namespace smurff 
{

// Prior with side information
template<class FType>
class MPIMacauPrior : public MacauPrior<FType> 
{
public:
   int world_rank;
   int world_size;

private:
   int* rhs_for_rank = NULL;
   double* rec     = NULL;
   int* sendcounts = NULL;
   int* displs     = NULL;   

public:
   //TODO: missing implementation
   MPIMacauPrior(BaseSession &m, int p);

   //TODO: missing declaration
   MPIMacauPrior(ScarceMatrixData &m, int p, INoiseModel &n);

   virtual ~MPIMacauPrior() {}

   void addSideInfo(std::unique_ptr<FType> &Fmat, bool comp_FtF = false);

   //TODO: missing implementation
   std::ostream &info(std::ostream &os, std::string indent) override;

   virtual void sample_beta();
   virtual bool run_slave();

   int rhs() const;
};

}
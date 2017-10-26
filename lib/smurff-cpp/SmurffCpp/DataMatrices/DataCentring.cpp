#include "DataCentring.h"

IMeanCenteringOld::IMeanCenteringOld()
: m_center_mode(CenterModeTypes::CENTER_INVALID)
{
}

void IMeanCenteringOld::compute_mode_mean_internal(const Data* data)
{
   assert(!m_mean_computed);
   m_mode_mean.resize(data->nmode());
   for (int m = 0; m < data->nmode(); ++m)
   {
       auto &M = m_mode_mean.at(m);
       M.resize(data->dim(m));
       for (int n = 0; n < data->dim(m); n++)
         M(n) = compute_mode_mean_mn(m, n);
   }
   m_mean_computed = true;
}

void IMeanCenteringOld::init_cwise_mean_internal(const Data* data)
{
   assert(!m_cwise_mean_initialized);
   m_cwise_mean = data->sum() / (data->size() - data->nna());
   m_cwise_mean_initialized = true;
}

void IMeanCentering::center(double upper_mean)
{
   assert(!m_centered);
   m_global_mean = upper_mean;
}

void IMeanCenteringOld::setCenterMode(std::string c)
{
   //-- centering model
   m_center_mode = stringToCenterMode(c);
   if(m_center_mode == CenterModeTypes::CENTER_INVALID)
      throw std::runtime_error("Invalid center mode");
}

void IMeanCenteringOld::setCenterMode(CenterModeTypes type)
{
   m_center_mode = type;
}

double IMeanCenteringOld::getCwiseMean() const
{
   assert(m_cwise_mean_initialized);
   return m_cwise_mean;
}

double IMeanCenteringOld::getGlobalMean() const
{
   assert(m_centered);
   return m_global_mean;
}

double IMeanCenteringOld::getVar() const
{
   return m_var;
}

CenterModeTypes IMeanCenteringOld::getCenterMode() const
{
   return m_center_mode;
}

bool IMeanCenteringOld::getMeanComputed() const
{
   return m_mean_computed;
}

const Eigen::VectorXd& IMeanCenteringOld::getModeMean(size_t i) const
{
   assert(m_mean_computed);
   return m_mode_mean.at(i);
}

double IMeanCenteringOld::getModeMeanItem(int m, int c) const
{
   assert(m_mean_computed);
   return m_mode_mean.at(m)(c);
}

std::string IMeanCenteringOld::getCenterModeName() const
{
   return centerModeToString(m_center_mode);
}

//===

// #### mean centring functions  ####

void DataCentring::compute_mode_mean()
{
   compute_mode_mean_internal(this);
}

void DataCentring::init_cwise_mean()
{
   init_cwise_mean_internal(this);
}

//===

void DenseMatrixDataCentring::center(double global_mean)
{
    IMeanCenteringOld::center(global_mean);

    if (getCenterMode() == CenterModeTypes::CENTER_GLOBAL)
    {
      getYcPtr()->at(0).array() -= global_mean;
      getYcPtr()->at(1).array() -= global_mean;
    }
    else if (getCenterMode() == CenterModeTypes::CENTER_VIEW)
    {
      getYcPtr()->at(0).array() -= getCwiseMean();
      getYcPtr()->at(1).array() -= getCwiseMean();
    }
    else if (getCenterMode() == CenterModeTypes::CENTER_COLS)
    {
      getYcPtr()->at(0).rowwise() -= getModeMean(0).transpose();
      getYcPtr()->at(1) = getYcPtr()->at(0).transpose();
    }
    else if (getCenterMode() == CenterModeTypes::CENTER_ROWS)
    {
      getYcPtr()->at(1).rowwise() -= getModeMean(1).transpose();
      getYcPtr()->at(0) = getYcPtr()->at(1).transpose();
    }
    else if (getCenterMode() == CenterModeTypes::CENTER_NONE)
    {
      //do nothing
    }
    else
    {
       throw std::logic_error("Invalid center mode");
    }

    setCentered(true);
}

//===

void MatricesDataCentring::setCenterMode(std::string mode)
{
   DataCentring::setCenterMode(mode);
   for(auto &p : blocks) 
      p.data().setCenterMode(mode);
}

void MatricesDataCentring::setCenterMode(CenterModeTypes type)
{
   DataCentring::setCenterMode(type);
   for(auto &p : blocks) 
      p.data().setCenterMode(type);
}

void MatricesDataCentring::center(double global_mean)
{
    IMeanCenteringOld::center(global_mean);

    // center sub-matrices
    assert(global_mean == getCwiseMean());

    for(auto &p : blocks)
      p.data().center(getCwiseMean());

   setCentered(true);
}

double MatricesDataCentring::compute_mode_mean_mn(int mode, int pos)
{
   double sum = .0;
   int N = 0;
   int count = 0;

   apply(mode, pos, [&](const Block &b) {
       double local_mean = b.data().getModeMeanItem(mode, pos - b.start(mode));
       sum += local_mean * b.dim(mode);
       N += b.dim(mode);
       count++;
   });

   assert(N>0);

   return sum / N;
}

double MatricesDataCentring::offset_to_mean(const PVec<>& pos) const
{
   const Block &b = find(pos);
   return b.data().offset_to_mean(pos - b.start());
}

//===

void ScarceMatrixDataCentring::center(double global_mean)
{
   IMeanCenteringOld::center(global_mean);

   auto center_cols = [this](Eigen::SparseMatrix<double> &Y, int m)
   {
      for (int k = 0; k < Y.outerSize(); ++k)
      {
         double v = getModeMeanItem(m, k);
         for (Eigen::SparseMatrix<double>::InnerIterator it(Y,k); it; ++it)
         {
               it.valueRef() -= v;
         }
      }
   };

   if (getCenterMode() == CenterModeTypes::CENTER_GLOBAL)
   {
      getYcPtr()->at(0).coeffs() -= global_mean;
      getYcPtr()->at(1).coeffs() -= global_mean;
   }
   else if (getCenterMode() == CenterModeTypes::CENTER_VIEW)
   {
      getYcPtr()->at(0).coeffs() -= getCwiseMean();
      getYcPtr()->at(1).coeffs() -= getCwiseMean();
   }
   else if (getCenterMode() == CenterModeTypes::CENTER_COLS)
   {
      center_cols(getYcPtr()->at(0), 0);
      getYcPtr()->at(1) = getYcPtr()->at(0).transpose();
   }
   else if (getCenterMode() == CenterModeTypes::CENTER_ROWS)
   {
      center_cols(getYcPtr()->at(1), 1);
      getYcPtr()->at(0) = getYcPtr()->at(1).transpose();
   }
   else if (getCenterMode() == CenterModeTypes::CENTER_NONE)
   {
      //do nothing
   }
   else
   {
      throw std::logic_error("Invalid center mode");
   }

   setCentered(true);
}

double ScarceMatrixDataCentring::compute_mode_mean_mn(int mode, int pos)
{
    const auto &col = getYcPtr()->at(mode).col(pos);
    if (col.nonZeros() == 0)
      return getCwiseMean();
    return col.sum() / col.nonZeros();
}

//===

void SparseMatrixData::center(double global_mean)
{
    IMeanCentering::center(global_mean);

    if (getCenterMode() == CenterModeTypes::CENTER_NONE)
    {
       //do nothing
    }
    else
    {
       throw std::logic_error("you cannot center fully know sparse matrix without converting to dense");
    }

    setCentered(true);
}
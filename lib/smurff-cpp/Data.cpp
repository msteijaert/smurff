#include "Data.h"
#include "Noiseless.h"

using namespace smurff;


int IDataDimensions::size() const
{
   return dim().dot();
}

int IDataDimensions::dim(int m) const
{
   return dim().at(m);
}

//===

IMeanCentering::IMeanCentering()
: m_center_mode(CenterModeTypes::CENTER_INVALID)
{
}

void IMeanCentering::center(double upper_mean)
{
   assert(!m_centered);
   m_global_mean = upper_mean;
   m_centered = true;
}

void IMeanCentering::setCenterMode(std::string c)
{
   //-- centering model
   m_center_mode = stringToCenterMode(c);
   if(m_center_mode == CenterModeTypes::CENTER_INVALID)
      throw std::runtime_error("Invalid center mode");
}

double IMeanCentering::mean(int m, int c) const
{
   assert(m_mean_computed);
   return m_mode_mean.at(m)(c);
}

double IMeanCentering::getCwiseMean() const
{
   return m_cwise_mean;
}

double IMeanCentering::getGlobalMean() const
{
   return m_global_mean;
}

void IMeanCentering::setGlobalMean(double value)
{
   m_global_mean = value;
}

double IMeanCentering::getVar() const
{
   return m_var;
}

IMeanCentering::CenterModeTypes IMeanCentering::getCenterMode() const
{
   return m_center_mode;
}

bool IMeanCentering::getMeanComputed() const
{
   return m_mean_computed;
}

const Eigen::VectorXd& IMeanCentering::getModeMean(size_t i) const
{
   return m_mode_mean.at(i);
}

std::string IMeanCentering::getCenterModeName() const
{
   std::string name = centerModeToString(m_center_mode);
   if(name.empty())
      throw std::runtime_error("Invalid center mode");
   return name;
}

std::string IMeanCentering::centerModeToString(IMeanCentering::CenterModeTypes cm)
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

IMeanCentering::CenterModeTypes IMeanCentering::stringToCenterMode(std::string c)
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

//===

IView::IView(const IDataDimensions* data_dim)
:  m_data_dim(data_dim)
{

}

int IView::nview(int mode) const
{
   return 1;
}

int IView::view(int mode, int pos) const
{
   return 0;
}

int IView::view_size(int m, int v) const
{
    return m_data_dim->dim(m);
}

//===

INoisePrecisionMean::INoisePrecisionMean()
{
}

INoiseModel& INoisePrecisionMean::noise() const
{
   assert(noise_ptr);
   return *noise_ptr;
}

void INoisePrecisionMean::setNoiseModel(INoiseModel* nm)
{
   noise_ptr.reset(nm);
}

//===

Data::Data()
: IDataDimensions()
, IDataArithmetic()
, INoisePrecisionMean()
, IMeanCentering()
, IView(this)
{
}

void Data::init_post()
{
   noise().init(this);
}

void Data::init()
{
    init_pre();

    //compute global mean & mode-wise means
    compute_mode_mean();
    center(getCwiseMean());

    init_post();
}

std::ostream& Data::info(std::ostream& os, std::string indent)
{
   os << indent << "Type: " << name << "\n";
   os << indent << "Component-wise mean: " << getCwiseMean() << "\n";
   os << indent << "Component-wise variance: " << var_total() << "\n";
   os << indent << "Center: " << getCenterModeName() << "\n";
   os << indent << "Noise: ";
   noise().info(os, "");
   return os;
}

std::ostream& Data::status(std::ostream& os, std::string indent) const
{
   os << indent << noise().getStatus() << "\n";
   return os;
}

void Data::update(const SubModel& model)
{
   noise().update(model, this);
}

void Data::compute_mode_mean()
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

void Data::init_pre_mean_centering()
{
   m_cwise_mean = sum() / (size() - nna());
}

double Data::predict(const PVec& pos, const SubModel& model) const
{
   return model.dot(pos) + offset_to_mean(pos);
}

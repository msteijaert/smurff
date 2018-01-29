#include "Data.h"
#include "SmurffCpp/IO/MatrixIO.h"

using namespace smurff;

IMeanCentering::IMeanCentering()
: m_center_mode(CenterModeTypes::CENTER_INVALID)
{
}

void IMeanCentering::compute_mode_mean_internal(const Data* data)
{
   assert(!m_mean_computed);
   m_mode_mean.resize(data->nmode());
   for (int m = 0; m < data->nmode(); ++m)
   {
       auto &M = m_mode_mean.at(m);
       M.resize(data->dim(m));
       for (int n = 0; n < data->dim(m); n++)
         M(n) = compute_mode_mean_mn(m, n);

#ifdef DEBUG_MEAN
       std::string filename = ((m == 0) ? "row_mean.csv" : "col_mean.csv");
       std::ofstream out(filename);
       matrix_io::eigen::write_dense_float64_csv(out, M);
#endif
   }
   m_mean_computed = true;
}

void IMeanCentering::init_cwise_mean_internal(const Data* data)
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

void IMeanCentering::setCenterMode(std::string c)
{
   //-- centering model
   m_center_mode = stringToCenterMode(c);
   if(m_center_mode == CenterModeTypes::CENTER_INVALID)
      throw std::runtime_error("Invalid center mode");
}

void IMeanCentering::setCenterMode(CenterModeTypes type)
{
   m_center_mode = type;
}

double IMeanCentering::getCwiseMean() const
{
   assert(m_cwise_mean_initialized);
   return m_cwise_mean;
}

double IMeanCentering::getGlobalMean() const
{
   assert(m_centered);
   return m_global_mean;
}

double IMeanCentering::getVar() const
{
   return m_var;
}

CenterModeTypes IMeanCentering::getCenterMode() const
{
   return m_center_mode;
}

bool IMeanCentering::getMeanComputed() const
{
   return m_mean_computed;
}

const Eigen::VectorXd& IMeanCentering::getModeMean(size_t i) const
{
   assert(m_mean_computed);
   return m_mode_mean.at(i);
}

double IMeanCentering::getModeMeanItem(int m, int c) const
{
   assert(m_mean_computed);
   return m_mode_mean.at(m)(c);
}

std::string IMeanCentering::getCenterModeName() const
{
   return centerModeToString(m_center_mode);
}

//===

Data::Data()
: IMeanCentering()
{
}

void Data::init()
{
    init_pre();

    //compute global mean & mode-wise means
    compute_mode_mean();
    center(getCwiseMean());

    init_post();
}

void Data::init_post()
{
   noise()->init(this);
}

void Data::update(const SubModel& model)
{
   noise()->update(this, model);
}

// #### mean centring functions  ####

void Data::compute_mode_mean()
{
   compute_mode_mean_internal(this);
}

void Data::init_cwise_mean()
{
   init_cwise_mean_internal(this);
}

//#### prediction functions ####

double Data::predict(const PVec<>& pos, const SubModel& model) const
{
    double base = model.dot(pos);
    double off = this->offset_to_mean(pos);
    // std::cout << "prefict at "; pos.info(std::cout); std::cout << " = " << base << " + " << off << std::endl;
    return base + off;
}

//#### dimention functions ####

std::int64_t Data::size() const
{
   return dim().dot();
}

int Data::dim(int m) const
{
   return dim().at(m);
}

//#### view functions ####

int Data::nview(int mode) const
{
   return 1;
}

int Data::view(int mode, int pos) const
{
   return 0;
}

int Data::view_size(int m, int v) const
{
    return this->dim(m);
}

//#### noise, precision, mean functions ####

std::shared_ptr<INoiseModel> Data::noise() const
{
   assert(noise_ptr);
   return noise_ptr;
}

void Data::setNoiseModel(std::shared_ptr<INoiseModel> nm)
{
   noise_ptr = nm;
}

//#### info functions ####

std::ostream& Data::info(std::ostream& os, std::string indent)
{
   os << indent << "Type: " << name << "\n";
   os << indent << "Component-wise mean: " << getCwiseMean() << "\n";
   os << indent << "Component-wise variance: " << var_total() << "\n";
   os << indent << "Center: " << getCenterModeName() << "\n";
   os << indent << "Noise: ";
   noise()->info(os, "");
   return os;
}

std::ostream& Data::status(std::ostream& os, std::string indent) const
{
   os << indent << noise()->getStatus() << "\n";
   return os;
}

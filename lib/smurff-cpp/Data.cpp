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

void IMeanCentering::setCenterMode(std::string c)
{
    //-- centering model
         if (c == "none")   m_center_mode = CenterModeTypes::CENTER_NONE;
    else if (c == "global") m_center_mode = CenterModeTypes::CENTER_GLOBAL;
    else if (c == "view")   m_center_mode = CenterModeTypes::CENTER_VIEW;
    else if (c == "rows")   m_center_mode = CenterModeTypes::CENTER_ROWS;
    else if (c == "cols")   m_center_mode = CenterModeTypes::CENTER_COLS;
    else assert(false);
}

double IMeanCentering::mean(int m, int c) const
{
   assert(m_mean_computed);
   return m_mode_mean.at(m)(c);
}

void IMeanCentering::compute_mode_mean()
{
   assert(!m_mean_computed);
   m_mode_mean.resize(m_dataBase->nmode());
   for (int m = 0; m < m_dataBase->nmode(); ++m)
   {
       auto &M = m_mode_mean.at(m);
       M.resize(m_dataBase->dim(m));
       for (int n = 0; n < m_dataBase->dim(m); n++)
         M(n) = compute_mode_mean(m, n);
   }
   m_mean_computed = true;
}

Data::Data()
   : IDataBase()
   , IMeanCentering(this)
{
    noise_ptr.reset(new Noiseless(this));
}

Data::~Data()
{
}

void Data::init_post()
{
   noise().init();
}

void Data::init()
{
    init_pre();

    //compute global mean & mode-wise means
    compute_mode_mean();
    center(getCwiseMean());

    init_post();
}

INoiseModel& Data::noise() const
{
   assert(noise_ptr);
   return *noise_ptr;
}

void Data::update(const SubModel& model)
{
   noise().update(model);
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

int Data::nview(int mode) const
{
   return 1;
}

int Data::view(int mode, int pos) const
{
   return 0;
}

int Data::view_size(int m,int) const
{
    return dim(m);
}

double Data::predict(const PVec& pos, const SubModel& model) const
{
   return model.dot(pos) + offset_to_mean(pos);
}

#include "Data.h"
#include "Noiseless.h"

using namespace smurff;

Data::Data() : center_mode(Data::CENTER_INVALID)
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
    center(cwise_mean);

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
   os << indent << "Component-wise mean: " << cwise_mean << "\n";
   os << indent << "Component-wise variance: " << var_total() << "\n";
   std::vector<std::string> center_names { "none", "global", "view", "cols", "rows" };
   os << indent << "Center: " << center_names.at(center_mode + 3) << "\n";
   os << indent << "Noise: ";
   noise().info(os, "");
   return os;
}

std::ostream& Data::status(std::ostream& os, std::string indent) const
{
   os << indent << noise().getStatus() << "\n";
   return os;
}

int Data::size() const
{
   return dim().dot();
}

int Data::dim(int m) const
{
   return dim().at(m);
}

// for matrices (nmode() == 2)
int Data::nrow() const
{
   return dim(1);
}

int Data::ncol() const
{
   return dim(0);
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


double Data::mean(int m, int c) const
{
   assert(mean_computed);
   return mode_mean.at(m)(c);
}

void Data::compute_mode_mean()
{
   assert(!mean_computed);
   mode_mean.resize(nmode());
   for(int m=0; m<nmode(); ++m)
   {
       auto &M = mode_mean.at(m);
       M.resize(dim(m));
       for(int n=0; n<dim(m); n++) M(n) = compute_mode_mean(m, n);
   }
   mean_computed = true;
}

void Data::setCenterMode(std::string c)
{
    //-- centering model
         if (c == "none")   center_mode = CENTER_NONE;
    else if (c == "global") center_mode = CENTER_GLOBAL;
    else if (c == "view")   center_mode = CENTER_VIEW;
    else if (c == "rows")   center_mode = CENTER_ROWS;
    else if (c == "cols")   center_mode = CENTER_COLS;
    else assert(false);
}

double Data::predict(const PVec& pos, const SubModel& model) const
{
   return model.dot(pos) + offset_to_mean(pos);
}

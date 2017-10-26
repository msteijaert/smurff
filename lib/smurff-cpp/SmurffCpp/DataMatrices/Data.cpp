#include "Data.h"

using namespace smurff;



Data::Data()
{
}

void Data::init()
{
    init_pre();

    init_post();
}

void Data::init_post()
{
   noise().init(this);
}

void Data::update(const SubModel& model)
{
   noise().update(this, model);
}

//#### prediction functions ####

double Data::predict(const PVec<>& pos, const SubModel& model) const
{
   return model.dot(pos) + this->getCenter()->offset_to_mean(pos);
}

//#### dimention functions ####

int Data::size() const
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

INoiseModel& Data::noise() const
{
   assert(noise_ptr);
   return *noise_ptr;
}

void Data::setNoiseModel(INoiseModel* nm)
{
   noise_ptr.reset(nm);
}

//#### info functions ####

std::ostream& Data::info(std::ostream& os, std::string indent)
{
   os << indent << "Type: " << name << "\n";
   os << indent << "Component-wise mean: " << this->getCenter()->getCwiseMean() << "\n";
   os << indent << "Component-wise variance: " << var_total() << "\n";
   os << indent << "Noise: ";
   noise().info(os, "");
   return os;
}

std::ostream& Data::status(std::ostream& os, std::string indent) const
{
   os << indent << noise().getStatus() << "\n";
   return os;
}
#include "PVec.h"

#include <cassert>

using namespace smurff;

PVec::PVec() : n(0)
{
}

PVec::PVec(size_t n)
   : n(n)
   , v({{0}})
{
   assert(n <= max);
}

PVec::PVec(int a, int b)
   : n(2)
   , v({{a,b}})
{
}

size_t PVec::size() const
{
   return n;
}

const int& PVec::operator[](size_t p) const
{
   return v[p];
}

const int& PVec::at(size_t p) const
{
   assert(p>=0 && p < n); return v[p];
}

int& PVec::operator[](size_t p)
{
   return v[p];
}

int& PVec::at(size_t p)
{
   assert(p>=0 && p < n);
   return v[p];
}

PVec PVec::operator+(const PVec& other) const
{
   assert(n == other.n);
   PVec ret = *this;
   for(size_t i=0; i<n; ++i) { ret[i] += other[i]; }
   return ret;
}

PVec PVec::operator-(const PVec& other) const
{
   assert(n == other.n);
   PVec ret = *this;
   for(size_t i=0; i<n; ++i) { ret[i] -= other[i]; }
   return ret;
}

bool PVec::in(const PVec& start, const PVec& end) const
{
   for(size_t i=0; i<n; ++i)
   {
         if (at(i) < start.at(i))return false;
         if (at(i) >= end.at(i)) return false;
   }
   return true;
}

int PVec::dot() const
{
   int ret = 1;
   for(size_t i=0; i<n; ++i) ret *= at(i);
   return ret;
}

std::ostream& PVec::info(std::ostream& os) const
{
   os << "[ ";
   for(size_t i=0; i<n; ++i) os << at(i) << ((i != n-1) ? " x " : "");
   os << " ]";
   return os;
}
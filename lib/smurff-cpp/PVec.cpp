#include "PVec.h"

#include <cassert>

using namespace smurff;

PVec::PVec()
{
}

PVec::PVec(size_t n)
{
   m_v.resize(n);
}

PVec::PVec(int a, int b)
   : m_v({ a, b })
{
}

PVec::PVec(const std::initializer_list<int>& l)
   : m_v(l)
{
}

size_t PVec::size() const
{
   return m_v.size();
}

const int& PVec::operator[](size_t p) const
{
   return m_v[p];
}

const int& PVec::at(size_t p) const
{
   assert(p >= 0 && p < m_v.size());
   return m_v[p];
}

int& PVec::operator[](size_t p)
{
   return m_v[p];
}

int& PVec::at(size_t p)
{
   assert(p >= 0 && p < m_v.size());
   return m_v[p];
}

PVec PVec::operator+(const PVec& other) const
{
   assert(m_v.size() == other.m_v.size());
   PVec ret = *this;
   for (size_t i = 0; i < m_v.size(); ++i)
      ret[i] += other[i];
   return ret;
}

PVec PVec::operator-(const PVec& other) const
{
   assert(m_v.size() == other.m_v.size());
   PVec ret = *this;
   for (size_t i = 0; i < m_v.size(); ++i)
      ret[i] -= other[i];
   return ret;
}

bool PVec::in(const PVec& start, const PVec& end) const
{
   for (size_t i = 0; i < m_v.size(); ++i)
   {
      if (at(i) < start.at(i))
         return false;
      if (at(i) >= end.at(i))
         return false;
   }
   return true;
}

int PVec::dot() const
{
   int ret = 1;
   for (size_t i = 0; i < m_v.size(); ++i)
      ret *= at(i);
   return ret;
}

std::ostream& PVec::info(std::ostream& os) const
{
   os << "[ ";
   for(size_t i = 0; i < m_v.size(); ++i)
      os << at(i) << ((i != m_v.size()-1) ? " x " : "");
   os << " ]";
   return os;
}
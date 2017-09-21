#include "PVec.h"

#include <sstream>

using namespace smurff;

PVec::PVec(size_t n)
{
   if (n == 0)
      throw std::length_error("Cannot initialize PVec with zero length");
   m_v.resize(n);
   std::fill(m_v.begin(), m_v.end(), 0);
}

PVec::PVec(const std::initializer_list<int>& l)
   : m_v(l)
{
   if (m_v.size() == 0)
      throw std::length_error("Cannot initialize PVec with zero length");
}

size_t PVec::size() const
{
   return m_v.size();
}

const int& PVec::operator[](size_t p) const
{
   if (p >= m_v.size())
   {
      std::stringstream ss;
      ss << "Cannot access m_v[" << p << "]";
      throw std::out_of_range(ss.str());
   }
   return m_v[p];
}

const int& PVec::at(size_t p) const
{
   if (p >= m_v.size())
   {
      std::stringstream ss;
      ss << "Cannot access m_v[" << p << "]";
      throw std::out_of_range(ss.str());
   }
   return m_v[p];
}

int& PVec::operator[](size_t p)
{
   if (p >= m_v.size())
   {
      std::stringstream ss;
      ss << "Cannot access m_v[" << p << "]";
      throw std::out_of_range(ss.str());
   }
   return m_v[p];
}

int& PVec::at(size_t p)
{
   if (p >= m_v.size())
   {
      std::stringstream ss;
      ss << "Cannot access m_v[" << p << "]";
      throw std::out_of_range(ss.str());
   }
   return m_v[p];
}

PVec PVec::operator+(const PVec& other) const
{
   if (m_v.size() != other.m_v.size())
      throw std::length_error("Both PVec intances must have the same size");

   PVec ret = *this;
   for (size_t i = 0; i < m_v.size(); ++i)
      ret[i] += other[i];
   return ret;
}

PVec PVec::operator-(const PVec& other) const
{
   if (m_v.size() != other.m_v.size())
      throw std::length_error("Both PVec intances must have the same size");

   PVec ret = *this;
   for (size_t i = 0; i < m_v.size(); ++i)
      ret[i] -= other[i];
   return ret;
}

bool PVec::in(const PVec& start, const PVec& end) const
{
   if (m_v.size() != start.size())
      throw std::length_error("All PVec intances must have the same size");

   if (m_v.size() != end.size())
      throw std::length_error("All PVec intances must have the same size");

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
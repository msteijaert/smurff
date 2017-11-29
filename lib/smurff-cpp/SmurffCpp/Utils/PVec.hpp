#pragma once

#include <string>
#include <iostream>
#include <sstream>
#include <array>
#include <numeric>
#include <algorithm>

namespace smurff
{
   template<size_t MaxSize = 3>
   class PVec
   {
   private:
      size_t m_size;
      std::array<int, MaxSize> m_v;

   public:
      PVec(size_t size)
         : m_size(size)
      {
         if (m_size == 0)
            throw std::length_error("Cannot initialize PVec with zero length");

         if (m_size > MaxSize)
            throw std::length_error("Cannot initialize PVec with size greater than MaxSize");

         std::fill(m_v.begin(), m_v.end(), 0);
      }

      PVec(const std::initializer_list<int>& l)
      {
         m_size = std::distance(l.begin(), l.end());

         if (m_size == 0)
            throw std::length_error("Cannot initialize PVec with zero length");

         if (m_size > MaxSize)
            throw std::length_error("Cannot initialize PVec with size greater than MaxSize");

         std::copy(l.begin(), l.end(), m_v.begin()); // m_v already has correct size
      }

      // Accept any int container
      template<template<typename, typename ...> class T, typename ... V>
      PVec(const T<int, V...>& v)
      {
         m_size = std::distance(v.begin(), v.end());

         if (m_size == 0)
            throw std::length_error("Cannot initialize PVec with zero length");

         if (m_size > MaxSize)
            throw std::length_error("Initializer size is greater than MaxSize");

         std::copy(v.begin(), v.end(), m_v.begin()); // m_v already has correct size
      }

   public:
      // meta info
      size_t size() const
      {
         return m_size;
      }

      const int& operator[](size_t p) const
      {
         if (p >= m_size)
         {
            std::stringstream ss;
            ss << "Cannot access m_v[" << p << "]";
            throw std::out_of_range(ss.str());
         }
         return m_v[p];
      }

      const int& at(size_t p) const
      {
         if (p >= m_size)
         {
            std::stringstream ss;
            ss << "Cannot access m_v[" << p << "]";
            throw std::out_of_range(ss.str());
         }
         return m_v[p];
      }

      int& operator[](size_t p)
      {
         if (p >= m_size)
         {
            std::stringstream ss;
            ss << "Cannot access m_v[" << p << "]";
            throw std::out_of_range(ss.str());
         }
         return m_v[p];
      }

      int& at(size_t p)
      {
         if (p >= m_size)
         {
            std::stringstream ss;
            ss << "Cannot access m_v[" << p << "]";
            throw std::out_of_range(ss.str());
         }
         return m_v[p];
      }

      PVec operator+(const PVec& other) const
      {
         if (m_size != other.m_size)
            throw std::length_error("Both PVec intances must have the same size");

         PVec ret(*this);
         std::transform(m_v.begin(), m_v.begin() + m_size, other.m_v.begin(), ret.m_v.begin(), std::plus<int>());
         return ret;
      }

      PVec operator-(const PVec& other) const
      {
         if (m_size != other.m_size)
            throw std::length_error("Both PVec intances must have the same size");

         PVec ret(*this);
         std::transform(m_v.begin(), m_v.begin() + m_size, other.m_v.begin(), ret.m_v.begin(), std::minus<int>());
         return ret;
      }

      bool operator==(const PVec& other) const
      {
         if (m_size != other.m_size)
            return false;

         auto p1 = std::mismatch(m_v.begin(), m_v.begin() + m_size, other.m_v.begin());
         return p1.first == m_v.begin() + m_size || p1.second == other.m_v.begin() + other.m_size;
      }

      bool operator!=(const PVec& other) const
      {
         return !(*this == other);
      }

      bool in(const PVec& start, const PVec& end) const
      {
         if (m_size != start.m_size)
            throw std::length_error("All PVec intances must have the same size");

         if (m_size != end.m_size)
            throw std::length_error("All PVec intances must have the same size");

         for (size_t i = 0; i < m_size; ++i)
         {
            if (at(i) < start.at(i))
               return false;
            if (at(i) >= end.at(i))
               return false;
         }
         return true;
      }

      int dot() const
      {
         return std::accumulate(m_v.begin(), m_v.begin() + m_size, 1, std::multiplies<int>());
      }

      std::ostream& info(std::ostream& os) const
      {
         os << "[ ";
         for(size_t i = 0; i < m_size; ++i)
            os << at(i) << ((i != m_size - 1) ? " x " : "");
         os << " ]";
         return os;
      }

      std::ostream& save(std::ostream& os) const
      {
         for(size_t i = 0; i < m_size; ++i)
            os << at(i) << ((i != m_size - 1) ? "," : "");
         return os;
      }
   };
}
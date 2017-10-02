#pragma once

#include <array>
#include <numeric>
#include <algorithm>

namespace smurff
{
   template<size_t MaxSize = 3>
   class SVec
   {
   private:
      size_t m_size;
      std::array<int, MaxSize> m_v;

   public:
      SVec::SVec(size_t size)
         : m_size(size)
      {
         if (m_size == 0)
            throw std::length_error("Cannot initialize SVec with zero length");

         if (m_size > MaxSize)
            throw std::length_error("Cannot initialize SVec with size greater than MaxSize");

         std::fill(m_v.begin(), m_v.end(), 0);
      }

      SVec::SVec(const std::initializer_list<int>& l)
         : m_size(std::count(l.begin(), l.end()))
      {
         if (m_size == 0)
            throw std::length_error("Cannot initialize SVec with zero length");

         if (m_size > MaxSize)
            throw std::length_error("Cannot initialize SVec with size greater than MaxSize");

         std::copy(l.begin(), l.end(), m_v.begin());
      }

      // Accept any int container
      template<template<typename, typename ...> typename T, typename ... V>
      SVec(const T<int, V...>& v)
      {
         size_t vCount = std::count(v.begin(), v.end());

         if (vCount == 0)
            throw std::length_error("Cannot initialize SVec with zero length");

         if (vCount > MaxSize)
            throw std::length_error("Initializer size is greater than MaxSize");

         std::copy(v.begin(), v.end(), m_v.begin());
      }

   public:
      // meta info
      size_t SVec::size() const
      {
         return m_size;
      }

      const int& SVec::operator[](size_t p) const
      {
         if (p >= m_size)
         {
            std::stringstream ss;
            ss << "Cannot access m_v[" << p << "]";
            throw std::out_of_range(ss.str());
         }
         return m_v[p];
      }

      const int& SVec::at(size_t p) const
      {
         if (p >= m_size)
         {
            std::stringstream ss;
            ss << "Cannot access m_v[" << p << "]";
            throw std::out_of_range(ss.str());
         }
         return m_v[p];
      }

      int& SVec::operator[](size_t p)
      {
         if (p >= m_size)
         {
            std::stringstream ss;
            ss << "Cannot access m_v[" << p << "]";
            throw std::out_of_range(ss.str());
         }
         return m_v[p];
      }

      int& SVec::at(size_t p)
      {
         if (p >= m_size)
         {
            std::stringstream ss;
            ss << "Cannot access m_v[" << p << "]";
            throw std::out_of_range(ss.str());
         }
         return m_v[p];
      }

      SVec SVec::operator+(const SVec& other) const
      {
         if (m_size != other.m_size)
            throw std::length_error("Both SVec intances must have the same size");

         SVec ret(*this);
         std::transform(m_v.begin(), m_v.end(), other.m_v.begin(), ret.m_v.begin(), std::plus<int>());
         return ret;
      }

      SVec SVec::operator-(const SVec& other) const
      {
         if (m_size != other.m_size)
            throw std::length_error("Both SVec intances must have the same size");

         SVec ret(*this);
         std::transform(m_v.begin(), m_v.end(), other.m_v.begin(), ret.m_v.begin(), std::minus<int>());
         return ret;
      }

      bool SVec::operator==(const SVec& other) const
      {
         if (m_size != other.m_size)
            return false;

         auto p1 = std::mismatch(m_v.begin(), m_v.end(), other.m_v.begin());
         return p1.first == m_v.end() || p1.second == other.m_v.end();
      }

      bool SVec::operator!=(const SVec& other) const
      {
         return !(*this == other);
      }


      bool SVec::in(const SVec& start, const SVec& end) const
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

      int SVec::dot() const
      {
         return std::accumulate(m_v.begin(), m_v.end(), 1, std::multiplies<int>());
      }

      std::ostream& SVec::info(std::ostream& os) const
      {
         os << "[ ";
         for(size_t i = 0; i < m_size; ++i)
            os << at(i) << ((i != m_size-1) ? " x " : "");
         os << " ]";
         return os;
      }
   };
}
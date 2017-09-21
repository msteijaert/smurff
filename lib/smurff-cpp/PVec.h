#pragma once

#include <stddef.h>
#include <vector>
#include <iostream>
#include <initializer_list>

namespace smurff
{
   class PVec
   {
   private:
      std::vector<int> m_v;

   public:
      PVec(size_t n);
      PVec(const std::initializer_list<int>& l);

      // Accept any int container
      template<template<typename, typename ...> typename T, typename ... V>
      PVec(const T<int, V...>& v)
         : m_v(v.begin(), v.end())
      {
         if (m_v.size() == 0)
            throw std::length_error("Cannot initialize PVec with zero length");
      }

   public:
      // meta info
      size_t size() const;

      // const accessor
      const int& operator[](size_t p) const;
      const int& at(size_t p) const;

      // non-const accessor
      int& operator[](size_t p);
      int& at(size_t p);

      // operators
      PVec operator+(const PVec& other) const;
      PVec operator-(const PVec& other) const;

      bool in(const PVec& start, const PVec& end) const;
      int dot() const;

      std::ostream& info(std::ostream& os) const;
  };
}
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
      PVec();
      PVec(size_t n);
      PVec(int a, int b);
      PVec(const std::initializer_list<int>& l);

   public:
      // meta info
      size_t size() const;

      // const accessor
      const int &operator[](size_t p) const;
      const int &at(size_t p) const;

      // non-const accessor
      int &operator[](size_t p);
      int &at(size_t p);

      // operators
      PVec operator+(const PVec& other) const;
      PVec operator-(const PVec& other) const;

      bool in(const PVec& start, const PVec& end) const;
      int dot() const;

      std::ostream& info(std::ostream& os) const;
  };
}
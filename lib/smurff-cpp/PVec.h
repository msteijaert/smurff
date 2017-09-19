#pragma once

#include <stddef.h>
#include <array>
#include <iostream>

namespace smurff
{
   class PVec
   {
   private:
      static const unsigned int max = 2; // only matrices for the moment
      size_t n;
      std::array<int, max> v;

   public:
      // c'tor
      PVec();
      PVec(size_t n);
      PVec(int a, int b);

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
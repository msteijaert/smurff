#pragma once

#include <iostream>

namespace smurff {

   class ISideInfo
   {
   public:
      virtual ~ISideInfo() {}

      virtual int cols() const = 0;

      virtual int rows() const = 0;

      virtual std::ostream& print(std::ostream &os) const = 0;

      virtual bool is_dense() const = 0;
   };

}

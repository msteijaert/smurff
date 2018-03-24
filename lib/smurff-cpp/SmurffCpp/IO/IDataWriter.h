#pragma once

#include <memory>
#include <string>

namespace smurff
{
   class MatrixConfig;
   class TensorConfig;

   class IDataWriter
   {
   public:
      virtual ~IDataWriter(){}

   public:
      virtual void write(std::shared_ptr<const MatrixConfig> mc) const = 0;
      virtual void write(std::shared_ptr<const TensorConfig> tc) const = 0;
   };
}

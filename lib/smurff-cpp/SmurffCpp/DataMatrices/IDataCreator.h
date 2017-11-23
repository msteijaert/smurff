#pragma once

#include <memory>
#include <string>

namespace smurff
{
   class Data;
   class MatrixConfig;
   class TensorConfig;

   class IDataCreator
   {
   public:
      virtual ~IDataCreator(){}

   public:
      virtual std::shared_ptr<Data> create(std::shared_ptr<const MatrixConfig> mc) const = 0;
      virtual std::shared_ptr<Data> create(std::shared_ptr<const TensorConfig> tc) const = 0;
   };
}
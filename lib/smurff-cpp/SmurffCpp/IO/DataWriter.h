#pragma once

#include <memory>

#include "IDataWriter.h"

namespace smurff
{
   class DataWriter : public IDataWriter
   {
   private:
      std::string m_filename;

   public:
      DataWriter(const std::string& filename)
         : m_filename(filename)
      {

      }

   public:
      void write(std::shared_ptr<const MatrixConfig> mc) const override;
      void write(std::shared_ptr<const TensorConfig> tc) const override;
   };
}

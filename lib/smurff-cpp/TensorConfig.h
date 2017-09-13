#pragma once

#include <vector>
#include <iostream>

#include "NoiseConfig.h"

namespace smurff
{    
   class TensorConfig
   {
   private:
      NoiseConfig m_noiseConfig;
      
   protected:
      bool m_isDense;
      bool m_isBinary;

      int m_nmodes;
      int m_nnz;

      std::vector<int> m_dims;
      std::vector<int> m_columns;
      std::vector<double> m_values;
      
   protected:
      TensorConfig(bool isDense, bool isBinary, int nmodes, int nnz);
      
   public:
      TensorConfig(int* columns, int nmodes, double* values, int nnz, int* dims);
   
   public:
      virtual ~TensorConfig();

   public:
      const NoiseConfig& getNoiseConfig() const;
      void setNoiseConfig(const NoiseConfig& value);

      bool isDense() const;
      bool isBinary() const;

      int getNModes() const;
      int getNNZ() const;
      
      const std::vector<int>& getDims() const;
      const std::vector<int>& getColumns() const;
      const std::vector<double>& getValues() const;
      
   public:
      virtual std::ostream& info(std::ostream& os) const;
   };
}
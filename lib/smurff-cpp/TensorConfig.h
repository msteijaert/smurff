#pragma once

#include <vector>
#include <iostream>
#include <memory>

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

      size_t m_nmodes;
      size_t m_nnz;

      std::shared_ptr<std::vector<size_t> > m_dims;
      std::shared_ptr<std::vector<size_t> > m_columns;
      std::shared_ptr<std::vector<double> > m_values;
      
   protected:
      TensorConfig(bool isDense, bool isBinary, size_t nmodes, size_t nnz, const NoiseConfig& noiseConfig);
      
   public:
      TensorConfig( const std::vector<size_t>& dims
                  , const std::vector<size_t>& columns
                  , const std::vector<double>& values
                  , const NoiseConfig& noiseConfig
                  );

      TensorConfig( std::shared_ptr<std::vector<size_t> > dims
                  , std::shared_ptr<std::vector<size_t> > columns
                  , std::shared_ptr<std::vector<double> > values
                  , const NoiseConfig& noiseConfig
                  );

   public:
      TensorConfig(int* columns, int nmodes, double* values, int nnz, int* dims, const NoiseConfig& noiseConfig);
   
   public:
      virtual ~TensorConfig();

   public:
      const NoiseConfig& getNoiseConfig() const;
      void setNoiseConfig(const NoiseConfig& value);

      bool isDense() const;
      bool isBinary() const;

      size_t getNModes() const;
      size_t getNNZ() const;
      
      const std::vector<size_t>& getDims() const;
      const std::vector<size_t>& getColumns() const;
      const std::vector<double>& getValues() const;

      std::shared_ptr<std::vector<size_t> > getDimsPtr() const;
      std::shared_ptr<std::vector<size_t> > getColumnsPtr() const;
      std::shared_ptr<std::vector<double> > getValuesPtr() const;
      
   public:
      virtual std::ostream& info(std::ostream& os) const;
   };
}
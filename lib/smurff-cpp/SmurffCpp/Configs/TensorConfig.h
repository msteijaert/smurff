#pragma once

#include <vector>
#include <iostream>
#include <memory>
#include <cstdint>

#include <SmurffCpp/Configs/NoiseConfig.h>

#include <SmurffCpp/IO/IDataWriter.h>

namespace smurff
{
   class Data;

   class TensorConfig : public std::enable_shared_from_this<TensorConfig>
   {
   private:
      NoiseConfig m_noiseConfig;

   protected:
      bool m_isDense;
      bool m_isBinary;

      std::uint64_t m_nmodes;
      std::uint64_t m_nnz;

      std::shared_ptr<std::vector<std::uint64_t> > m_dims;
      std::shared_ptr<std::vector<std::uint32_t> > m_columns;
      std::shared_ptr<std::vector<double> > m_values;

   protected:
      TensorConfig(bool isDense, bool isBinary, 
                   std::uint64_t nmodes, std::uint64_t nnz, 
                   const NoiseConfig& noiseConfig);

   //
   // Dense double tensor constructors
   //
   public:
      TensorConfig(const std::vector<std::uint64_t>& dims, const std::vector<double> values, 
                   const NoiseConfig& noiseConfig);

      TensorConfig(std::vector<std::uint64_t>&& dims, std::vector<double>&& values, 
                   const NoiseConfig& noiseConfig);

      TensorConfig(std::shared_ptr<std::vector<std::uint64_t> > dims, std::shared_ptr<std::vector<double> > values, 
                   const NoiseConfig& noiseConfig);

   //
   // Sparse double tensor constructors
   //
   public:
      TensorConfig(const std::vector<std::uint64_t>& dims, const std::vector<std::uint32_t>& columns, const std::vector<double>& values, 
                   const NoiseConfig& noiseConfig);

      TensorConfig(std::vector<std::uint64_t>&& dims, std::vector<std::uint32_t>&& columns, std::vector<double>&& values, 
                   const NoiseConfig& noiseConfig);

      TensorConfig(std::shared_ptr<std::vector<std::uint64_t> > dims, std::shared_ptr<std::vector<std::uint32_t> > columns, std::shared_ptr<std::vector<double> > values, 
                   const NoiseConfig& noiseConfig);

   //
   // Sparse binary tensor constructors
   //
   public:
      TensorConfig(const std::vector<std::uint64_t>& dims, const std::vector<std::uint32_t>& columns, const NoiseConfig& noiseConfig);

      TensorConfig(std::vector<std::uint64_t>&& dims, std::vector<std::uint32_t>&& columns, 
                   const NoiseConfig& noiseConfig);

      TensorConfig(std::shared_ptr<std::vector<std::uint64_t> > dims, std::shared_ptr<std::vector<std::uint32_t> > columns, 
                   const NoiseConfig& noiseConfig);

   public:
      virtual ~TensorConfig();

   public:
      const NoiseConfig& getNoiseConfig() const;
      void setNoiseConfig(const NoiseConfig& value);

      bool isDense() const;
      bool isBinary() const;

      std::uint64_t getNModes() const;
      std::uint64_t getNNZ() const;

      const std::vector<std::uint64_t>& getDims() const;
      const std::vector<std::uint32_t>& getColumns() const;
      const std::vector<double>& getValues() const;

      std::shared_ptr<std::vector<std::uint64_t> > getDimsPtr() const;
      std::shared_ptr<std::vector<std::uint32_t> > getColumnsPtr() const;
      std::shared_ptr<std::vector<double> > getValuesPtr() const;

   public:
      virtual std::ostream& info(std::ostream& os) const;

      virtual std::shared_ptr<Data> toData(bool scarce) const;

      virtual std::shared_ptr<Data> toData(const std::vector<std::shared_ptr<TensorConfig> >& row_features, 
                                           const std::vector<std::shared_ptr<TensorConfig> >& col_features) const;

   public:
      virtual void write(std::shared_ptr<IDataWriter> writer) const;
   };
}
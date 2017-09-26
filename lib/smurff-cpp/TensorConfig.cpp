#include "TensorConfig.h"

using namespace smurff;

TensorConfig::TensorConfig ( bool isDense
                           , bool isBinary
                           , std::uint64_t nmodes
                           , std::uint64_t nnz
                           , const NoiseConfig& noiseConfig
                           )
   : m_noiseConfig(noiseConfig)
   , m_isDense(isDense)
   , m_isBinary(isBinary)
   , m_nmodes(nmodes)
   , m_nnz(nnz)
   , m_dims(std::make_shared<std::vector<std::uint64_t> >())
   , m_columns(std::make_shared<std::vector<std::uint64_t> >())
   , m_values(std::make_shared<std::vector<double> >())
{
}

TensorConfig::TensorConfig( const std::vector<std::uint64_t>& dims
                          , const std::vector<std::uint64_t>& columns
                          , const std::vector<double>& values
                          , const NoiseConfig& noiseConfig
                          )
   : m_noiseConfig(noiseConfig)
   , m_isDense(false)
   , m_isBinary(false)
   , m_nmodes(dims.size())
   , m_nnz(values.size())
{
   if (columns.size() != values.size() * dims.size())
      throw std::runtime_error("Cannot create TensorConfig instance: 'columns' size should be the same as size of 'values' times size of 'dims'");

   m_dims = std::make_shared<std::vector<std::uint64_t> >(dims);
   m_columns = std::make_shared<std::vector<std::uint64_t> >(columns);
   m_values = std::make_shared<std::vector<double> >(values);
}

TensorConfig::TensorConfig( std::vector<std::uint64_t>&& dims
                          , std::vector<std::uint64_t>&& columns
                          , std::vector<double>&& values
                          , const NoiseConfig& noiseConfig
                          )
   : TensorConfig( std::make_shared<std::vector<std::uint64_t> >(std::move(dims))
                 , std::make_shared<std::vector<std::uint64_t> >(std::move(columns))
                 , std::make_shared<std::vector<double> >(std::move(values))
                 , noiseConfig
                 )
{
}

TensorConfig::TensorConfig( std::shared_ptr<std::vector<std::uint64_t> > dims
                          , std::shared_ptr<std::vector<std::uint64_t> > columns
                          , std::shared_ptr<std::vector<double> > values
                          , const NoiseConfig& noiseConfig
                          )
   : m_noiseConfig(noiseConfig)
   , m_isDense(false)
   , m_isBinary(false)
   , m_nmodes(dims->size())
   , m_nnz(values->size())
   , m_dims(dims)
   , m_columns(columns)
   , m_values(values)
{
   if (columns->size() != values->size() * dims->size())
      throw std::runtime_error("Cannot create TensorConfig instance: 'columns' size should be the same as size of 'values' times size of 'dims'");
}

TensorConfig::~TensorConfig()
{
}

bool TensorConfig::isDense() const
{
   return m_isDense;
}

bool TensorConfig::isBinary() const
{
   return m_isBinary;
}

std::uint64_t TensorConfig::getNNZ() const
{
   return m_nnz;
}

std::uint64_t TensorConfig::getNModes() const
{
   return m_nmodes;
}

const std::vector<std::uint64_t>& TensorConfig::getDims() const
{
   return *m_dims;
}

const std::vector<std::uint64_t>& TensorConfig::getColumns() const
{
   return *m_columns;
}

const std::vector<double>& TensorConfig::getValues() const
{
   return *m_values;
}

std::shared_ptr<std::vector<std::uint64_t> > TensorConfig::getDimsPtr() const
{
   return m_dims;
}

std::shared_ptr<std::vector<size_t> > TensorConfig::getColumnsPtr() const
{
   return m_columns;
}

std::shared_ptr<std::vector<double> > TensorConfig::getValuesPtr() const
{
   return m_values;
}

const NoiseConfig& TensorConfig::getNoiseConfig() const
{
   return m_noiseConfig;
}

void TensorConfig::setNoiseConfig(const NoiseConfig& value)
{
   m_noiseConfig = value;
}

std::ostream& TensorConfig::info(std::ostream& os) const
{
   if (!m_dims->size())
   {
      os << "0";
   }
   else
   {
      os << m_dims->operator[](0);
      for (std::size_t i = 1; i < m_dims->size(); i++)
         os << " x " << m_dims->operator[](i);
   }
   return os;
}
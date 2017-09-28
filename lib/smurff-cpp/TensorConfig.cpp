#include "TensorConfig.h"

#include <numeric>

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
   , m_columns(std::make_shared<std::vector<std::uint32_t> >())
   , m_values(std::make_shared<std::vector<double> >())
{
}

TensorConfig::TensorConfig( const std::vector<std::uint64_t>& dims
                          , const std::vector<double> values
                          , const NoiseConfig& noiseConfig
                          )
   : m_noiseConfig(noiseConfig)
   , m_isDense(true)
   , m_isBinary(false)
   , m_nmodes(dims.size())
   , m_nnz(std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<std::uint64_t>()))
{
   if (dims.size() == 0)
      throw std::length_error("Cannot create TensorConfig instance: 'dims' size cannot be zero");

   if (values.size() == 0)
      throw std::length_error("Cannot create TensorConfig instance: 'values' size cannot be zero");

   if (values.size() != m_nnz)
      throw std::length_error("Cannot create TensorConfig instance: 'values' size and 'nnz' must be the same");

   m_dims = std::make_shared<std::vector<std::uint64_t> >(dims);
   m_columns = std::make_shared<std::vector<std::uint32_t> >();
   m_columns->reserve(m_dims->size() * m_nnz);
   m_values = std::make_shared<std::vector<double> >(values);

	for (std::size_t i = 0; i < m_dims->size(); i++)
	{
		std::uint64_t j_max =
			std::accumulate(m_dims->begin(), m_dims->begin() + i, 1, std::multiplies<std::uint64_t>());
		for (std::uint64_t j = 0; j < j_max; j++)
		{
			for (std::uint64_t k = 0; k < m_dims->operator[](i); k++)
			{
				std::uint64_t l_max =
					std::accumulate(m_dims->begin() + i + 1, m_dims->end(), 1, std::multiplies<std::uint64_t>());
				for (std::uint64_t l = 0; l < l_max; l++)
				{
					m_columns->push_back(static_cast<std::uint32_t>(k));
				}
			}
		}
	}
}

TensorConfig::TensorConfig( std::vector<std::uint64_t>&& dims
                          , std::vector<double>&& values
                          , const NoiseConfig& noiseConfig
                          )
   : TensorConfig( std::make_shared<std::vector<std::uint64_t> >(std::move(dims))
                 , std::make_shared<std::vector<double> >(std::move(values))
                 , noiseConfig
                 )
{
}

TensorConfig::TensorConfig( std::shared_ptr<std::vector<std::uint64_t> > dims
                          , std::shared_ptr<std::vector<double> > values
                          , const NoiseConfig& noiseConfig
                          )
   : m_noiseConfig(noiseConfig)
   , m_isDense(true)
   , m_isBinary(false)
   , m_nmodes(dims->size())
   , m_nnz(std::accumulate(dims->begin(), dims->end(), 1, std::multiplies<std::uint64_t>()))
   , m_dims(dims)
   , m_values(values)
{
   if (m_dims->size() == 0)
      throw std::length_error("Cannot create TensorConfig instance: 'dims' size cannot be zero");

   if (m_values->size() == 0)
      throw std::length_error("Cannot create TensorConfig instance: 'values' size cannot be zero");

   if (m_values->size() != m_nnz)
      throw std::length_error("Cannot create TensorConfig instance: 'values' size and 'nnz' must be the same");

   m_columns = std::make_shared<std::vector<std::uint32_t> >();
   m_columns->reserve(m_dims->size() * m_nnz);

	for (std::size_t i = 0; i < m_dims->size(); i++)
	{
		std::uint64_t j_max =
			std::accumulate(m_dims->begin(), m_dims->begin() + i, 1, std::multiplies<std::uint64_t>());
		for (std::uint64_t j = 0; j < j_max; j++)
		{
			for (std::uint64_t k = 0; k < m_dims->operator[](i); k++)
			{
				std::uint64_t l_max =
					std::accumulate(m_dims->begin() + i + 1, m_dims->end(), 1, std::multiplies<std::uint64_t>());
				for (std::uint64_t l = 0; l < l_max; l++)
				{
					m_columns->push_back(static_cast<std::uint32_t>(k));
				}
			}
		}
	}
}

TensorConfig::TensorConfig( const std::vector<std::uint64_t>& dims
                          , const std::vector<std::uint32_t>& columns
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
   m_columns = std::make_shared<std::vector<std::uint32_t> >(columns);
   m_values = std::make_shared<std::vector<double> >(values);
}

TensorConfig::TensorConfig( std::vector<std::uint64_t>&& dims
                          , std::vector<std::uint32_t>&& columns
                          , std::vector<double>&& values
                          , const NoiseConfig& noiseConfig
                          )
   : TensorConfig( std::make_shared<std::vector<std::uint64_t> >(std::move(dims))
                 , std::make_shared<std::vector<std::uint32_t> >(std::move(columns))
                 , std::make_shared<std::vector<double> >(std::move(values))
                 , noiseConfig
                 )
{
}

TensorConfig::TensorConfig( std::shared_ptr<std::vector<std::uint64_t> > dims
                          , std::shared_ptr<std::vector<std::uint32_t> > columns
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

TensorConfig::TensorConfig( const std::vector<std::uint64_t>& dims
                          , const std::vector<std::uint32_t>& columns
                          , const NoiseConfig& noiseConfig
                          )
   : m_noiseConfig(noiseConfig)
   , m_isDense(false)
   , m_isBinary(true)
   , m_nmodes(dims.size())
   , m_nnz(columns.size() / dims.size())
{
   if (dims.size() == 0)
      throw std::length_error("Cannot create TensorConfig instance: 'dims' size cannot be zero");

   if (columns.size() == 0)
      throw std::length_error("Cannot create TensorConfig instance: 'columns' size cannot be zero");

   m_dims = std::make_shared<std::vector<std::uint64_t> >(dims);
   m_columns = std::make_shared<std::vector<std::uint32_t> >(columns);
   m_values = std::make_shared<std::vector<double> >(m_nnz);
   std::fill(m_values->begin(), m_values->end(), 1);
}

TensorConfig::TensorConfig( std::vector<std::uint64_t>&& dims
                          , std::vector<std::uint32_t>&& columns
                          , const NoiseConfig& noiseConfig
                          )
   : TensorConfig( std::make_shared<std::vector<std::uint64_t> >(std::move(dims))
                 , std::make_shared<std::vector<std::uint32_t> >(std::move(columns))
                 , noiseConfig
                 )
{
}

TensorConfig::TensorConfig( std::shared_ptr<std::vector<std::uint64_t> > dims
                          , std::shared_ptr<std::vector<std::uint32_t> > columns
                          , const NoiseConfig& noiseConfig
                          )
   : m_noiseConfig(noiseConfig)
   , m_isDense(false)
   , m_isBinary(true)
   , m_nmodes(dims->size())
   , m_nnz(columns->size() / dims->size())
   , m_dims(dims)
   , m_columns(columns)
   , m_values(std::make_shared<std::vector<double> >())
{
   if (dims->size() == 0)
      throw std::length_error("Cannot create TensorConfig instance: 'dims' size cannot be zero");

   if (columns->size() == 0)
      throw std::length_error("Cannot create TensorConfig instance: 'columns' size cannot be zero");

   m_values->resize(m_nnz);
   std::fill(m_values->begin(), m_values->end(), 1);
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

const std::vector<std::uint32_t>& TensorConfig::getColumns() const
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

std::shared_ptr<std::vector<std::uint32_t> > TensorConfig::getColumnsPtr() const
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
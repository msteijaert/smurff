#include "TensorConfig.h"

#include <numeric>

#include <SmurffCpp/IO/IDataWriter.h>
#include <SmurffCpp/DataMatrices/IDataCreator.h>
#include <SmurffCpp/Utils/Error.h>
#include <SmurffCpp/IO/GenericIO.h>
#include <SmurffCpp/Utils/StringUtils.h>

#define POS_TAG "pos"
#define FILE_TAG "file"
#define DENSE_TAG "dense"
#define SCARCE_TAG "scarce"
#define SPARSE_TAG "sparse"
#define TYPE_TAG "type"

#define NONE_TAG "none"

#define NOISE_MODEL_TAG "noise_model"
#define PRECISION_TAG "precision"
#define SN_INIT_TAG "sn_init"
#define SN_MAX_TAG "sn_max"
#define NOISE_THRESHOLD_TAG "noise_threshold"

using namespace smurff;

TensorConfig::TensorConfig ( bool isDense
                           , bool isBinary
                           , bool isScarce
                           , std::uint64_t nmodes
                           , std::uint64_t nnz
                           , const NoiseConfig& noiseConfig
                           )
   : m_noiseConfig(noiseConfig)
   , m_isDense(isDense)
   , m_isBinary(isBinary)
   , m_isScarce(isScarce)
   , m_nmodes(nmodes)
   , m_nnz(nnz)
   , m_dims(std::make_shared<std::vector<std::uint64_t> >())
   , m_columns(std::make_shared<std::vector<std::uint32_t> >())
   , m_values(std::make_shared<std::vector<double> >())
{
}

//
// Dense double tensor constructors
//

TensorConfig::TensorConfig( const std::vector<std::uint64_t>& dims
                          , const std::vector<double> values
                          , const NoiseConfig& noiseConfig
                          )
   : m_noiseConfig(noiseConfig)
   , m_isDense(true)
   , m_isBinary(false)
   , m_isScarce(false)
   , m_nmodes(dims.size())
   , m_nnz(std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<std::uint64_t>()))
{
   if (dims.size() == 0)
   {
      THROWERROR("Cannot create TensorConfig instance: 'dims' size cannot be zero");
   }

   if (values.size() == 0)
   {
      THROWERROR("Cannot create TensorConfig instance: 'values' size cannot be zero");
   }

   if (values.size() != m_nnz)
   {
      THROWERROR("Cannot create TensorConfig instance: 'values' size and 'nnz' must be the same");
   }

   m_dims = std::make_shared<std::vector<std::uint64_t> >(dims);
   m_columns = std::make_shared<std::vector<std::uint32_t> >();
   m_columns->reserve(m_dims->size() * m_nnz);
   m_values = std::make_shared<std::vector<double> >(values);

   //construct N dimentions

	for (std::vector<uint64_t>::iterator it = m_dims->begin(); it != m_dims->end(); it++)
	{
		std::uint64_t i_max = std::accumulate(it + 1, m_dims->end(), 1, std::multiplies<std::uint64_t>());
		for (std::uint64_t i = 0; i < i_max; i++)
		{
			for (std::uint64_t j = 0; j < *it; j++)
			{
				std::uint64_t k_max = std::accumulate(m_dims->begin(), it, 1, std::multiplies<std::uint64_t>());
				for (std::uint64_t k = 0; k < k_max; k++)
				{
					m_columns->push_back(static_cast<std::uint32_t>(j));
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
   , m_isScarce(false)
   , m_nmodes(dims->size())
   , m_nnz(std::accumulate(dims->begin(), dims->end(), 1, std::multiplies<std::uint64_t>()))
   , m_dims(dims)
   , m_values(values)
{
   if (m_dims->size() == 0)
   {
      THROWERROR("Cannot create TensorConfig instance: 'dims' size cannot be zero");
   }

   if (m_values->size() == 0)
   {
      THROWERROR("Cannot create TensorConfig instance: 'values' size cannot be zero");
   }

   if (m_values->size() != m_nnz)
   {
      THROWERROR("Cannot create TensorConfig instance: 'values' size and 'nnz' must be the same");
   }

   m_columns = std::make_shared<std::vector<std::uint32_t> >();
   m_columns->reserve(m_dims->size() * m_nnz);

   //construct N dimentions

	for (std::vector<uint64_t>::iterator it = m_dims->begin(); it != m_dims->end(); it++)
	{
		std::uint64_t i_max = std::accumulate(it + 1, m_dims->end(), 1, std::multiplies<std::uint64_t>());
		for (std::uint64_t i = 0; i < i_max; i++)
		{
			for (std::uint64_t j = 0; j < *it; j++)
			{
				std::uint64_t k_max = std::accumulate(m_dims->begin(), it, 1, std::multiplies<std::uint64_t>());
				for (std::uint64_t k = 0; k < k_max; k++)
				{
					m_columns->push_back(static_cast<std::uint32_t>(j));
				}
			}
		}
	}
}

//
// Sparse double tensor constructors
//

TensorConfig::TensorConfig( const std::vector<std::uint64_t>& dims
                          , const std::vector<std::uint32_t>& columns
                          , const std::vector<double>& values
                          , const NoiseConfig& noiseConfig
                          , bool isScarce
                          )
   : m_noiseConfig(noiseConfig)
   , m_isDense(false)
   , m_isBinary(false)
   , m_isScarce(isScarce)
   , m_nmodes(dims.size())
   , m_nnz(values.size())
{
   if (columns.size() != values.size() * dims.size())
   {
      THROWERROR("Cannot create TensorConfig instance: 'columns' size should be the same as size of 'values' times size of 'dims'");
   }

   m_dims = std::make_shared<std::vector<std::uint64_t> >(dims);
   m_columns = std::make_shared<std::vector<std::uint32_t> >(columns);
   m_values = std::make_shared<std::vector<double> >(values);
}

TensorConfig::TensorConfig( std::vector<std::uint64_t>&& dims
                          , std::vector<std::uint32_t>&& columns
                          , std::vector<double>&& values
                          , const NoiseConfig& noiseConfig
                          , bool isScarce
                          )
   : TensorConfig( std::make_shared<std::vector<std::uint64_t> >(std::move(dims))
                 , std::make_shared<std::vector<std::uint32_t> >(std::move(columns))
                 , std::make_shared<std::vector<double> >(std::move(values))
                 , noiseConfig, isScarce
                 )
{
}

TensorConfig::TensorConfig( std::shared_ptr<std::vector<std::uint64_t> > dims
                          , std::shared_ptr<std::vector<std::uint32_t> > columns
                          , std::shared_ptr<std::vector<double> > values
                          , const NoiseConfig& noiseConfig
                          , bool isScarce
                          )
   : m_noiseConfig(noiseConfig)
   , m_isDense(false)
   , m_isBinary(false)
   , m_isScarce(isScarce)
   , m_nmodes(dims->size())
   , m_nnz(values->size())
   , m_dims(dims)
   , m_columns(columns)
   , m_values(values)
{
   if (columns->size() != values->size() * dims->size())
   {
      THROWERROR("Cannot create TensorConfig instance: 'columns' size should be the same as size of 'values' times size of 'dims'");
   }
}

//
// Sparse binary tensor constructors
//

TensorConfig::TensorConfig( const std::vector<std::uint64_t>& dims
                          , const std::vector<std::uint32_t>& columns
                          , const NoiseConfig& noiseConfig
                          , bool isScarce
                          )
   : m_noiseConfig(noiseConfig)
   , m_isDense(false)
   , m_isBinary(true)
   , m_isScarce(isScarce)
   , m_nmodes(dims.size())
   , m_nnz(columns.size() / dims.size())
{
   if (dims.size() == 0)
   {
      THROWERROR("Cannot create TensorConfig instance: 'dims' size cannot be zero");
   }

   if (columns.size() == 0)
   {
      THROWERROR("Cannot create TensorConfig instance: 'columns' size cannot be zero");
   }

   m_dims = std::make_shared<std::vector<std::uint64_t> >(dims);
   m_columns = std::make_shared<std::vector<std::uint32_t> >(columns);
   m_values = std::make_shared<std::vector<double> >(m_nnz, 1);
}

TensorConfig::TensorConfig( std::vector<std::uint64_t>&& dims
                          , std::vector<std::uint32_t>&& columns
                          , const NoiseConfig& noiseConfig
                          , bool isScarce
                          )
   : TensorConfig( std::make_shared<std::vector<std::uint64_t> >(std::move(dims))
                 , std::make_shared<std::vector<std::uint32_t> >(std::move(columns))
                 , noiseConfig, isScarce
                 )
{
}

TensorConfig::TensorConfig( std::shared_ptr<std::vector<std::uint64_t> > dims
                          , std::shared_ptr<std::vector<std::uint32_t> > columns
                          , const NoiseConfig& noiseConfig
                          , bool isScarce
                          )
   : m_noiseConfig(noiseConfig)
   , m_isDense(false)
   , m_isBinary(true)
   , m_isScarce(isScarce)
   , m_nmodes(dims->size())
   , m_nnz(columns->size() / dims->size())
   , m_dims(dims)
   , m_columns(columns)
{
   if (dims->size() == 0)
   {
      THROWERROR("Cannot create TensorConfig instance: 'dims' size cannot be zero");
   }

   if (columns->size() == 0)
   {
      THROWERROR("Cannot create TensorConfig instance: 'columns' size cannot be zero");
   }

   m_values = std::make_shared<std::vector<double> >(m_nnz, 1);
}

TensorConfig::~TensorConfig()
{
}

//
// other methods
//

bool TensorConfig::isDense() const
{
   return m_isDense;
}

bool TensorConfig::isBinary() const
{
   return m_isBinary;
}

bool TensorConfig::isScarce() const
{
   return m_isScarce;
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

void TensorConfig::setFilename(const std::string &f)
{
    m_filename = f;
}

const std::string &TensorConfig::getFilename() const
{
    return m_filename;
}

void TensorConfig::setPos(const PVec<>& p)
{
   m_pos = std::make_shared<PVec<>>(p);
}

bool TensorConfig::hasPos() const
{
    return m_pos != nullptr;
}

const PVec<>& TensorConfig::getPos() const
{
    return *m_pos;
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
   if (getFilename().size())
   {
        os << " \"" << getFilename() << "\"";
   }
   if (hasPos())
   {
        os << " @[" << getPos() << "]";
   }
   return os;
}

std::string TensorConfig::info() const
{
    std::stringstream ss;
    info(ss);
    return ss.str();
}

std::ostream& TensorConfig::save_tensor_config(std::ostream& os, const std::string sec_name, int sec_idx, const std::shared_ptr<TensorConfig> &cfg)
{
   //write section name
   os << "[" << sec_name;
   if (sec_idx >= 0)
      os << "_" << sec_idx;
   os << "]" << std::endl;

   //write tensor config and noise config
   if (cfg)
   {
      cfg->save(os);
   }
   else
   {
      os << "file = " << NONE_TAG << std::endl;
   }

   return os;
}

std::ostream& TensorConfig::save(std::ostream& os) const
{
   //write tensor config position
   if (this->hasPos())
      os << POS_TAG << " = " << this->getPos() << std::endl;

   //write tensor config filename
   os << FILE_TAG << " = " << this->getFilename() << std::endl;

   //write tensor config type
   std::string type_str = this->isDense() ? DENSE_TAG : this->isScarce() ? SCARCE_TAG : SPARSE_TAG;
   os << TYPE_TAG << " = " << type_str << std::endl;

   //write noise config
   auto &noise_config = this->getNoiseConfig();
   if (noise_config.getNoiseType() != NoiseTypes::unset)
   {
      os << NOISE_MODEL_TAG << "  = " << smurff::noiseTypeToString(noise_config.getNoiseType()) << std::endl;
      os << PRECISION_TAG << " = " << noise_config.getPrecision() << std::endl;
      os << SN_INIT_TAG << " = " << noise_config.getSnInit() << std::endl;
      os << SN_MAX_TAG << " = " << noise_config.getSnMax() << std::endl;
      os << NOISE_THRESHOLD_TAG << " = " << noise_config.getThreshold() << std::endl;
   }

   return os;
}

std::shared_ptr<TensorConfig> TensorConfig::restore_tensor_config(const INIFile& reader, const std::string sec_name)
{
   //restore filename
   std::string filename = reader.get(sec_name, FILE_TAG, NONE_TAG);
   if (filename == NONE_TAG)
      return std::shared_ptr<TensorConfig>();

   //restore type
   bool is_scarce = reader.get(sec_name, TYPE_TAG, SCARCE_TAG) == SCARCE_TAG;

   //restore data
   auto cfg = generic_io::read_data_config(filename, is_scarce);

   //restore instance
   cfg->restore(reader, sec_name);

   return cfg;
}

bool TensorConfig::restore(const INIFile& reader, const std::string sec_name)
{
   //restore position
   std::string pos_str = reader.get(sec_name, POS_TAG, NONE_TAG);
   if (pos_str != NONE_TAG)
   {
      std::vector<int> tokens;
      smurff::split(pos_str, tokens, ',');

      //assign position
      this->setPos(PVec<>(tokens));
   }

   //restore noise model
   NoiseConfig noise;

   NoiseTypes noiseType = smurff::stringToNoiseType(reader.get(sec_name, NOISE_MODEL_TAG, smurff::noiseTypeToString(NoiseTypes::unset)));
   if (noiseType != NoiseTypes::unset)
   {
      noise.setNoiseType(noiseType);
      noise.setPrecision(reader.getReal(sec_name, PRECISION_TAG, NoiseConfig::PRECISION_DEFAULT_VALUE));
      noise.setSnInit(reader.getReal(sec_name, SN_INIT_TAG, NoiseConfig::ADAPTIVE_SN_INIT_DEFAULT_VALUE));
      noise.setSnMax(reader.getReal(sec_name, SN_MAX_TAG, NoiseConfig::ADAPTIVE_SN_MAX_DEFAULT_VALUE));
      noise.setThreshold(reader.getReal(sec_name, NOISE_THRESHOLD_TAG, NoiseConfig::PROBIT_DEFAULT_VALUE));
   }

   //assign noise model
   this->setNoiseConfig(noise);

   return true;
}

std::shared_ptr<Data> TensorConfig::create(std::shared_ptr<IDataCreator> creator) const
{
   return creator->create(shared_from_this());
}

void TensorConfig::write(std::shared_ptr<IDataWriter> writer) const
{
   writer->write(shared_from_this());
}

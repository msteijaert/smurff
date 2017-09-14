#include "TensorConfig.h"

using namespace smurff;

TensorConfig::TensorConfig(bool isDense, bool isBinary, int nmodes, int nnz, const NoiseConfig& noiseConfig)
   : m_noiseConfig(noiseConfig)
   , m_isDense(isDense)
   , m_isBinary(isBinary)
   , m_nmodes(nmodes)
   , m_nnz(nnz)
{
}

TensorConfig::TensorConfig(int* columns, int nmodes, double* values, int nnz, int* dims, const NoiseConfig& noiseConfig)
   : m_noiseConfig(noiseConfig)
   , m_nmodes(nmodes)
   , m_nnz(nnz)
{
   m_columns.reserve(nmodes * nnz);
   m_values.reserve(nnz);

   for (int i = 0; i < nmodes; i++)
      m_dims.push_back(dims[i]);

   for (int i = 0; i < nmodes * nnz; i++)
      m_columns.push_back(columns[i]);

   for (int i = 0; i < nnz; i++)
      m_values.push_back(values[i]);
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

int TensorConfig::getNNZ() const
{
   return m_nnz;
}

int TensorConfig::getNModes() const
{
   return m_nmodes;
}

const std::vector<int>& TensorConfig::getDims() const
{
   return m_dims;
}

const std::vector<int>& TensorConfig::getColumns() const
{
   return m_columns;
}

const std::vector<double>& TensorConfig::getValues() const
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
   if (!m_dims.size())
   {
      os << "0";
   }
   else
   {
      os << m_dims[0];
      for (size_t i = 1; i < m_dims.size(); i++)
         os << " x " << m_dims[i];
   }
   return os;
}
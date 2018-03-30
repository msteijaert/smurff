#pragma once

#include <algorithm>
#include <memory>
#include <cstdint>

#include <SmurffCpp/Model.h>
#include <SmurffCpp/Utils/PVec.hpp>

namespace smurff {

template<class T>
class VMatrixExprIterator : public std::iterator<
	std::input_iterator_tag,   // iterator_category
	T,                         // value_type
	std::uint32_t,             // difference_type
	T,                        // pointer
	T                         // reference
>
{
private:
   const Model& m_model;
   PVec<> m_off;
   PVec<> m_dims;
	std::uint32_t m_mode;
	std::uint32_t m_num;

public:
	//begin constructor
   VMatrixExprIterator(const Model& model, PVec<> off, PVec<> dims, std::uint32_t mode, std::uint32_t num)
      : m_model(model), m_off(off), m_dims(dims), m_mode(mode), m_num(num == mode ? num + 1 : num)
   {
   }

	//end constructor
   VMatrixExprIterator(std::uint32_t num)
      : m_model(Model::no_model), m_off(1), m_dims(1), m_mode(-1), m_num(num)
   {
   }

public:
	//copy constructors
   VMatrixExprIterator(const VMatrixExprIterator<T>& other)
      : m_model(other.m_model),
        m_off(other.m_off),
        m_dims(other.m_dims),
        m_mode(other.m_mode),
        m_num(other.m_num)
   {
   }

public:
   VMatrixExprIterator<T>& operator++()
   {
      m_num++;
      if (m_num == m_mode)
         m_num++;
      return *this;
   }

   VMatrixExprIterator<T> operator++(int)
   {
      VMatrixExprIterator<T> retval = *this;
      ++(*this);
      return retval;
   }

   bool operator==(VMatrixExprIterator<T> other) const
   {
      return m_num == other.m_num;
   }

   bool operator!=(VMatrixExprIterator<T> other) const
   {
      return !(*this == other);
   }

   T operator*()
   {
      return m_model.U(m_num).block(0, m_off[m_num], m_model.nlatent(), m_dims[m_num]);
   }

   T operator->()
   {
      return this->operator*();
   }
};

}
